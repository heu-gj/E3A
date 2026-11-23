"""F01/F02 regressions on the public adapter, not a replacement test criterion."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from e3a.adapters.mask_validity import (
    conservative_pixel_support,
    linear_assignment,
    sample_centers,
)
from e3a.adapters.videomt import VidEoMTMaskCriterion
from e3a.data.collate import collate_e3a_batch, collate_entity_targets


def clip(*, frames=1, tracks=1, size=(8, 10), complete=False):
    return {
        "labels": torch.zeros(tracks, dtype=torch.int64),
        "ids": torch.arange(10, 10 + tracks)[:, None].expand(-1, frames).clone(),
        "masks": torch.ones(tracks, frames, *size),
        "mask_valid": torch.ones(tracks, frames, dtype=torch.bool),
        "pixel_valid": torch.ones(frames, *size, dtype=torch.bool),
        "entity_supervision_valid": complete,
    }


def predictions(*, frames=1, slots=3, size=(8, 10), batch=1):
    return {
        "mask_logits": torch.zeros(batch, frames, slots, *size, requires_grad=True),
        "objectness_logits": torch.zeros(batch, frames, slots, requires_grad=True),
    }


def adapter(frames=1, points=4096, costs=None):
    return VidEoMTMaskCriterion(
        frames=frames, num_points=points,
        loss_weights={"loss_ce": 1., "loss_mask": 1., "loss_dice": 1.},
        matching_costs=costs,
    )


def execute(target, pred=None, *, points=4096):
    pred = pred or predictions(frames=target["ids"].shape[1], size=target["masks"].shape[-2:])
    module = adapter(target["ids"].shape[1], points)
    losses, matches = module.forward_with_matches(pred, [target])
    return module, pred, losses, matches


def test_f01_all_valid_matches_bce_dice_equations_and_gradients():
    target = clip()
    _, pred, loss, mapping = execute(target)
    assert mapping == [{10: 0}]
    labels = target["masks"][0, 0].flatten()
    expected_bce = F.binary_cross_entropy_with_logits(torch.zeros_like(labels), labels)
    expected_dice = 1 - (labels.sum() + 1) / (0.5 * labels.numel() + labels.sum() + 1)
    torch.testing.assert_close(loss["loss_mask"], expected_bce)
    torch.testing.assert_close(loss["loss_dice"], expected_dice)
    loss["loss_objective"].backward()
    assert pred["mask_logits"].grad[0, 0, 0].abs().sum() > 0
    assert pred["objectness_logits"].grad[0, 0, 0] < 0


@pytest.mark.parametrize("prediction_size", [(8, 10), (4, 5), (5, 7), (12, 15)])
def test_f01_unknown_logit_perturbation_loss_cost_assignment_invariance(prediction_size):
    target = clip()
    target["pixel_valid"][0, 2:4, 4:6] = False
    target["masks"][0, 0, 2:4, 4:6] = 0  # Reader's zero placeholder is not background GT.
    pred = predictions(size=prediction_size)
    mod, _, before, mapping = execute(target, pred)
    coarse_valid, _ = conservative_pixel_support(target["pixel_valid"][0], prediction_size)
    flat, _ = mod._validate_targets([target], batch_size=1, num_frames=1,
                                    num_queries=3, device=torch.device("cpu"))
    outputs = {"pred_logits": torch.stack([pred["objectness_logits"][0, 0],
                                           torch.zeros(3)], -1),
               "pred_masks": pred["mask_logits"][0, 0, :, None]}
    before_cost = mod.matcher.cost_matrix(outputs, flat[0]).detach()
    changed = {k: v.detach().clone().requires_grad_() for k, v in pred.items()}
    with torch.no_grad():
        changed["mask_logits"][:, :, :, ~coarse_valid] = 10000.
    after, after_mapping = mod.forward_with_matches(changed, [target])
    changed_outputs = {**outputs, "pred_masks": changed["mask_logits"][0, 0, :, None]}
    after_cost = mod.matcher.cost_matrix(changed_outputs, flat[0]).detach()
    torch.testing.assert_close(before_cost, after_cost, rtol=0, atol=0)
    assert mapping == after_mapping
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    after["loss_objective"].backward()
    assert torch.count_nonzero(changed["mask_logits"].grad[:, :, :, ~coarse_valid]) == 0
    assert changed["mask_logits"].grad.abs().sum() > 0


def test_f01_spatial_padding_does_not_change_mask_losses():
    base = clip(size=(3, 4))
    _, _, before, _ = execute(base)
    padded = clip(size=(6, 8))
    padded["pixel_valid"][:] = False
    padded["pixel_valid"][:, :3, :4] = True
    padded["masks"][:] = 0
    padded["masks"][:, :, :3, :4] = 1
    pred = predictions(size=(6, 8))
    _, _, after, _ = execute(padded, pred)
    for key in ("loss_mask", "loss_dice"):
        torch.testing.assert_close(before[key], after[key])
    after["loss_objective"].backward()
    assert torch.count_nonzero(pred["mask_logits"].grad[:, :, :, 3:]) == 0
    assert torch.count_nonzero(pred["mask_logits"].grad[:, :, :, :, 4:]) == 0


def test_f01_temporal_padding_has_no_mask_or_classification_gradient():
    target = clip(frames=3, complete=torch.tensor([True, True, False]))
    target["ids"][:, 2] = -1
    target["masks"][:, 2] = 0
    target["mask_valid"][:, 2] = False
    target["pixel_valid"][2] = False
    _, pred, losses, _ = execute(target)
    losses["loss_objective"].backward()
    assert torch.count_nonzero(pred["mask_logits"].grad[:, 2]) == 0
    assert torch.count_nonzero(pred["objectness_logits"].grad[:, 2]) == 0
    assert pred["objectness_logits"].grad[:, :2].abs().sum() > 0


def test_f01_mask_valid_overrides_valid_pixels():
    target = clip(frames=2, complete=True)
    target["ids"][:, 1] = -1
    target["masks"][:, 1] = 0
    target["mask_valid"][:, 1] = False
    _, pred, loss, _ = execute(target)
    loss["loss_objective"].backward()
    assert torch.count_nonzero(pred["mask_logits"].grad[:, 1]) == 0
    assert torch.count_nonzero(pred["objectness_logits"].grad[:, 1]) == 0


def test_missing_mask_valid_does_not_turn_unknown_absence_into_negative_gt():
    target = clip(frames=2, complete=True)
    target.pop("mask_valid")
    target["ids"][:, 1] = -1
    target["masks"][:, 1] = 0
    _, pred, loss, _ = execute(target)
    loss["loss_objective"].backward()
    assert torch.count_nonzero(pred["mask_logits"].grad[:, 1]) == 0
    assert torch.count_nonzero(pred["objectness_logits"].grad[:, 1]) == 0


def test_explicit_track_absence_can_supervise_mask_without_scene_no_object():
    target = clip(frames=2, complete=False)
    target["ids"][:, 1] = -1
    target["masks"][:, 1] = 0
    _, pred, loss, _ = execute(target)
    loss["loss_objective"].backward()
    assert (pred["mask_logits"].grad[0, 1, 0] > 0).all()
    assert torch.count_nonzero(pred["objectness_logits"].grad[:, 1]) == 0


@pytest.mark.parametrize("tracks", [0, 1])
@pytest.mark.parametrize("complete", [False, True])
def test_f01_f02_all_invalid_clip_is_zero_finite_and_unmatched(tracks, complete):
    target = clip(tracks=tracks, complete=complete)
    target["pixel_valid"][:] = False
    _, pred, loss, mapping = execute(target)
    assert mapping == [{}]
    for value in loss.values():
        assert torch.isfinite(value) and value.item() == 0
    loss["loss_objective"].backward()
    for value in pred.values():
        assert torch.count_nonzero(value.grad) == 0


def test_all_invalid_empty_batch_does_not_overflow_ignored_logit_reduction():
    target = clip(tracks=0)
    target["pixel_valid"][:] = False
    pred = predictions()
    with torch.no_grad():
        for value in pred.values():
            value.fill_(1e38)
    _, _, loss, _ = execute(target, pred)
    assert all(torch.isfinite(v) and v.item() == 0 for v in loss.values())
    loss["loss_objective"].backward()
    assert all(torch.count_nonzero(v.grad) == 0 for v in pred.values())


def test_wholly_uncovered_track_rows_are_ignored_without_false_background():
    target = clip(tracks=2, complete=True)
    target["ids"][:] = -1
    target["masks"][:] = 0
    target["mask_valid"][:] = False
    _, pred, loss, mapping = execute(target)
    assert mapping == [{}]
    assert all(v.item() == 0 and torch.isfinite(v) for v in loss.values())
    loss["loss_objective"].backward()
    assert all(torch.count_nonzero(v.grad) == 0 for v in pred.values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mask_loss_dtype_normalization_keeps_invalid_gradients_zero(dtype):
    target = clip()
    target["pixel_valid"][:, 1, 1] = False
    pred = {k: v.detach().to(dtype=dtype).requires_grad_() for k, v in predictions().items()}
    _, _, loss, _ = execute(target, pred)
    loss["loss_objective"].backward()
    assert loss["loss_objective"].dtype == torch.float32
    assert pred["mask_logits"].grad.dtype == dtype
    assert torch.count_nonzero(pred["mask_logits"].grad[:, :, :, 1, 1]) == 0
    assert pred["mask_logits"].grad.abs().sum() > 0


@pytest.mark.parametrize("fraction", [1, 2, 4])
def test_f01_bce_is_normalized_by_valid_pixel_count(fraction):
    target = clip(size=(4, 8))
    target["pixel_valid"][:, 4 // fraction:] = False
    _, _, loss, _ = execute(target)
    torch.testing.assert_close(loss["loss_mask"], torch.tensor(0.69314718))


@pytest.mark.parametrize("target_size,prediction_size", [((5, 7), (3, 4)),
                                                      ((6, 8), (3, 4)),
                                                      ((3, 4), (6, 8))])
def test_f01_pixel_center_bilinear_matches_grid_sample(target_size, prediction_size):
    torch.manual_seed(2)
    values = torch.randn(2, *prediction_size)
    h, w = target_size
    indices = torch.arange(h * w)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    points = torch.stack(((x + .5) / w * 2 - 1, (y + .5) / h * 2 - 1), -1)
    reference = F.grid_sample(values[:, None], points[None].expand(2, -1, -1, -1),
                              align_corners=False, padding_mode="border")[:, 0].flatten(1)
    torch.testing.assert_close(sample_centers(values, indices, target_size), reference,
                               atol=1e-6, rtol=1e-5)


def test_f01_bilinear_support_excludes_valid_centers_next_to_invalid_coarse_cell():
    valid = torch.ones(8, 8, dtype=torch.bool)
    valid[2, 2] = False
    coarse, safe = conservative_pixel_support(valid, (4, 4))
    assert not coarse[1, 1]
    assert valid[3, 3] and not safe[3, 3]
    x = torch.zeros(1, 4, 4, requires_grad=True)
    sample_centers(x, safe.flatten().nonzero().flatten(), (8, 8)).sum().backward()
    assert x.grad[0, 1, 1] == 0


@pytest.mark.parametrize("points", [1, 4, 8])
def test_f01_sparse_uncertainty_sampling_ignores_invalid_logits(points):
    target = clip(size=(12, 12))
    target["pixel_valid"][:, 5:7, 5:7] = False
    pred = predictions(size=(6, 6))
    mod, _, before, mapping = execute(target, pred, points=points)
    coarse, _ = conservative_pixel_support(target["pixel_valid"][0], (6, 6))
    with torch.no_grad():
        pred["mask_logits"][:, :, :, ~coarse] = -10000.
    after, remapped = mod.forward_with_matches(pred, [target])
    assert mapping == remapped
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_f01_all_invalid_mask_cost_has_zero_contribution():
    target = clip()
    target["pixel_valid"][:] = False
    mod = adapter(costs={"cost_class": 0., "cost_mask": 1., "cost_dice": 1.})
    flat, _ = mod._validate_targets([target], batch_size=1, num_frames=1,
                                    num_queries=2, device=torch.device("cpu"))
    cost = mod.matcher.cost_matrix({"pred_logits": torch.randn(2, 2),
                                   "pred_masks": torch.randn(2, 1, 8, 10)}, flat[0])
    assert torch.count_nonzero(cost) == 0


@pytest.mark.parametrize("complete,tracks", [(True, 1), (True, 0), (False, 1), (False, 0)])
def test_f02_complete_incomplete_and_empty_classification_gradients(complete, tracks):
    target = clip(tracks=tracks, complete=complete)
    _, pred, loss, mapping = execute(target)
    loss["loss_ce"].backward()
    grad = pred["objectness_logits"].grad[0, 0]
    if tracks:
        matched = mapping[0][10]
        assert grad[matched] < 0
        other = torch.arange(3) != matched
        assert (grad[other] > 0).all() if complete else torch.count_nonzero(grad[other]) == 0
    else:
        assert (grad > 0).all() if complete else torch.count_nonzero(grad) == 0
    if complete and not tracks:
        torch.testing.assert_close(loss["loss_ce"], torch.tensor(0.69314718))
    assert torch.isfinite(loss["loss_ce"])


def test_f02_more_unmatched_queries_do_not_change_incomplete_scene_loss():
    target = clip(complete=False)
    values = []
    for slots in [1, 4, 10]:
        _, pred, loss, mapping = execute(target, predictions(slots=slots))
        loss["loss_ce"].backward()
        assert mapping == [{10: 0}]
        assert torch.count_nonzero(pred["objectness_logits"].grad[:, :, 1:]) == 0
        values.append(loss["loss_ce"].detach())
    for value in values[1:]:
        torch.testing.assert_close(values[0], value)


def test_f02_partial_complete_frames_gate_no_object_independently():
    target = clip(frames=3, tracks=0, complete=torch.tensor([True, False, True]))
    _, pred, losses, _ = execute(target)
    losses["loss_ce"].backward()
    grad = pred["objectness_logits"].grad
    assert torch.count_nonzero(grad[:, 1]) == 0
    assert (grad[:, [0, 2]] > 0).all()


def test_f02_one_uncovered_track_disables_negatives_but_retains_known_positive():
    target = clip(frames=2, tracks=2, complete=True)
    target["ids"][1, 1] = -1
    target["mask_valid"][1, 1] = False
    target["masks"][1, 1] = 0
    _, pred, loss, mapping = execute(target, predictions(frames=2, slots=3))
    loss["loss_ce"].backward()
    grad = pred["objectness_logits"].grad[0, 1]
    assert grad[mapping[0][10]] < 0
    assert grad[mapping[0][11]] == 0
    assert grad[2] == 0


def test_f02_missing_scene_flag_is_not_complete_coverage():
    target = clip(tracks=0)
    target.pop("entity_supervision_valid")
    _, pred, loss, _ = execute(target)
    loss["loss_ce"].backward()
    assert loss["loss_ce"].item() == 0
    assert torch.count_nonzero(pred["objectness_logits"].grad) == 0


@pytest.mark.parametrize("flag,expected", [
    (True, [[True] * 3] * 2),
    (torch.tensor(True), [[True] * 3] * 2),
    (torch.tensor([True, False]), [[True] * 3, [False] * 3]),
    (torch.tensor([[True], [False]]), [[True] * 3, [False] * 3]),
    (torch.tensor([[True, False, True], [False, True, False]]),
     [[True, False, True], [False, True, False]]),
])
def test_f02_batched_scene_flag_shapes(flag, expected):
    clips = [clip(frames=3, tracks=0), clip(frames=3, tracks=0)]
    for target in clips:
        target.pop("entity_supervision_valid")  # The batch flag is the explicit source here.
    targets = collate_entity_targets(clips)
    pred = predictions(batch=2, frames=3)
    loss = adapter(3)(pred, {"entity_targets": targets, "entity_supervision_valid": flag})
    assert torch.isfinite(loss["loss_objective"])
    loss["loss_objective"].backward()
    nonzero = pred["objectness_logits"].grad.abs().sum(-1) > 0
    assert nonzero.tolist() == expected


@pytest.mark.parametrize("flag", [1, "yes", torch.ones(1), torch.ones(2, 2, dtype=torch.bool)])
def test_f02_invalid_scene_flag_shapes_and_dtypes_are_rejected(flag):
    target = clip()
    target["entity_supervision_valid"] = flag
    with pytest.raises((ValueError, TypeError), match="entity_supervision_valid"):
        execute(target)


def test_joint_collation_preserves_all_three_validity_contracts():
    targets = [clip(frames=2, complete=False), clip(frames=2, tracks=0, complete=True)]
    targets[0]["pixel_valid"][0, 1, 1] = False
    samples = [{"images": torch.zeros(2, 3, 8, 10), "timestamps": torch.tensor([0., 1.]),
                "entity_targets": t, "entity_supervision_valid": t["entity_supervision_valid"]}
               for t in targets]
    batch = collate_e3a_batch(samples)
    assert batch["entity_supervision_valid"].tolist() == [False, True]
    assert not batch["entity_targets"]["pixel_valid"][0, 0, 1, 1]
    assert not batch["entity_targets"]["mask_valid"][1].any()
    pred = predictions(batch=2, frames=2)
    loss = adapter(2)(pred, batch)
    loss["loss_ce"].backward()
    assert pred["objectness_logits"].grad[0, 0, 0] < 0
    assert torch.count_nonzero(pred["objectness_logits"].grad[0, :, 1:]) == 0
    assert (pred["objectness_logits"].grad[1] > 0).all()


def test_direct_target_collation_preserves_explicit_scene_coverage():
    padded = collate_entity_targets([clip(tracks=0, complete=True)])
    assert padded["entity_supervision_valid"].tolist() == [[True]]
    loss = adapter()(predictions(), {"entity_targets": padded})
    torch.testing.assert_close(loss["loss_ce"], torch.tensor(0.69314718))


@pytest.mark.parametrize("padded", [True, False])
def test_outer_scene_flag_cannot_override_explicit_inner_incompleteness(padded):
    target = clip(complete=False)
    targets = collate_entity_targets([target]) if padded else [target]
    pred = predictions()
    loss = adapter()(pred, {"entity_targets": targets, "entity_supervision_valid": True})
    loss["loss_ce"].backward()
    assert pred["objectness_logits"].grad[0, 0, 0] < 0
    assert torch.count_nonzero(pred["objectness_logits"].grad[:, :, 1:]) == 0


def test_joint_auxiliary_frame_coverage_can_only_restrict_scene_flag():
    target = clip(frames=2, complete=True)
    target["entity_classification_valid"] = torch.tensor([False, True])
    _, pred, loss, mapping = execute(target)
    loss["loss_ce"].backward()
    grad = pred["objectness_logits"].grad[0]
    assert (grad[:, mapping[0][10]] < 0).all()
    assert torch.count_nonzero(grad[0, 1:]) == 0
    assert (grad[1, 1:] > 0).all()


def test_joint_auxiliary_flag_cannot_promote_an_incomplete_scene():
    target = clip(complete=False)
    target["entity_classification_valid"] = torch.tensor([True])
    _, pred, loss, _ = execute(target)
    loss["loss_ce"].backward()
    assert pred["objectness_logits"].grad[0, 0, 0] < 0
    assert torch.count_nonzero(pred["objectness_logits"].grad[:, :, 1:]) == 0


def test_joint_partial_unknown_pixels_do_not_erase_known_positives():
    target = clip(complete=True)
    target["pixel_valid"][0, 0, 0] = False
    _, pred, loss, _ = execute(target)
    loss["loss_ce"].backward()
    assert pred["objectness_logits"].grad[0, 0, 0] < 0
    assert torch.count_nonzero(pred["objectness_logits"].grad[:, :, 1:]) == 0


def test_f01_nontrivial_multi_entity_assignment_and_cost_are_invariant():
    target = clip(tracks=2, size=(8, 10))
    target["masks"][0, :, :, 5:] = 0
    target["masks"][1, :, :, :5] = 0
    target["pixel_valid"][:, 3:5, 4:6] = False
    pred = predictions(slots=2)
    with torch.no_grad():
        pred["mask_logits"][0, 0] = target["masks"].flip(0)[:, 0] * 12 - 6
    mod, _, loss, mapping = execute(target, pred)
    assert mapping == [{10: 1, 11: 0}]
    with torch.no_grad():
        pred["mask_logits"][0, 0, 0, ~target["pixel_valid"][0]] = -10000
        pred["mask_logits"][0, 0, 1, ~target["pixel_valid"][0]] = 10000
    after, remapped = mod.forward_with_matches(pred, [target])
    assert remapped == mapping
    for key in loss:
        torch.testing.assert_close(loss[key], after[key], rtol=0, atol=0)


def test_f01_first_reliable_frame_matching_ignores_invalid_prefix():
    target = clip(frames=2)
    target["pixel_valid"][0] = False
    pred = predictions(frames=2, slots=2)
    with torch.no_grad():
        pred["objectness_logits"][0, 0, 0] = 10000
        pred["mask_logits"][0, 1, 1] = 8
    _, _, loss, mapping = execute(target, pred)
    assert mapping == [{10: 1}]
    loss["loss_objective"].backward()
    assert torch.count_nonzero(pred["mask_logits"].grad[:, 0]) == 0
    assert torch.count_nonzero(pred["objectness_logits"].grad[:, 0]) == 0


def test_matcher_keeps_clip_slot_identity_when_later_predictions_swap():
    target = clip(frames=2, tracks=2, size=(4, 6))
    target["masks"][0, :, :, 3:] = 0
    target["masks"][1, :, :, :3] = 0
    pred = predictions(frames=2, slots=2, size=(4, 6))
    with torch.no_grad():
        pred["mask_logits"][0, 0] = target["masks"][:, 0] * 12 - 6
        pred["mask_logits"][0, 1] = pred["mask_logits"][0, 0].flip(0)
    _, _, _, mapping = execute(target, pred)
    assert mapping == [{10: 0, 11: 1}]


@pytest.mark.parametrize("queries,tracks", [(1, 1), (4, 1), (3, 2), (4, 3), (3, 3), (2, 0)])
def test_exact_assignment_matches_exhaustive_optimum(queries, tracks):
    torch.manual_seed(11)
    cost = torch.randn(queries, tracks)
    source, target = linear_assignment(cost)
    assert source.device == cost.device and source.dtype == torch.int64
    expected = min(sum(float(cost[q, t]) for t, q in enumerate(choices))
                   for choices in itertools.permutations(range(queries), tracks))
    assert float(cost[source, target].sum()) == pytest.approx(expected, abs=1e-6)


def test_unsafe_injected_upstream_implementations_are_not_silently_accepted():
    with pytest.raises(TypeError, match="supports_entity_validity"):
        VidEoMTMaskCriterion(frames=1, num_points=4,
                            loss_weights={"loss_ce": 1., "loss_mask": 1., "loss_dice": 1.},
                            matcher=torch.nn.Identity(), criterion=torch.nn.Identity())
