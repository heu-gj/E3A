import pytest
import torch
from torch import nn

from e3a.adapters.videomt import VidEoMTMaskCriterion
from e3a.data.collate import collate_entity_targets


class RecordingMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.outputs = None
        self.targets = None
        self.cost_class = 1.0
        self.cost_mask = 1.0
        self.cost_dice = 1.0
        self.num_points = 32
        self.frames = 2

    def forward(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets
        return [
            (
                torch.tensor([0], dtype=torch.int64, device=outputs["pred_logits"].device),
                torch.tensor([0], dtype=torch.int64, device=outputs["pred_logits"].device),
            )
            if target["labels"].numel()
            else (
                torch.empty(0, dtype=torch.int64, device=outputs["pred_logits"].device),
                torch.empty(0, dtype=torch.int64, device=outputs["pred_logits"].device),
            )
            for target in targets
        ]


class RecordingCriterion(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.losses = ["labels", "masks"]
        self.mask_loss_indices = None
        self.mask_loss_targets = None

    def forward(self, outputs, targets):
        self.matcher(outputs, targets)
        masks = outputs["pred_masks"]
        logits = outputs["pred_logits"]
        return {
            "loss_ce": logits.square().mean(),
            "loss_mask": masks.square().mean(),
            "loss_dice": masks.abs().mean(),
        }

    def loss_labels(self, outputs, targets, indices, num_masks):
        return {"loss_ce": outputs["pred_logits"].square().mean()}

    def loss_masks(self, outputs, targets, indices, num_masks):
        self.mask_loss_indices = indices
        self.mask_loss_targets = targets
        valid_count = sum(int(target_index.numel()) for _, target_index in indices)
        loss = outputs["pred_masks"].sum() * 0.0 + float(valid_count)
        return {"loss_mask": loss, "loss_dice": loss}


def make_criterion(frames=2, matcher=None, criterion=None):
    matcher = matcher or RecordingMatcher()
    if isinstance(matcher, RecordingMatcher):
        matcher.frames = frames
    criterion = criterion or RecordingCriterion(matcher)
    return VidEoMTMaskCriterion(
        frames=frames,
        num_points=32,
        loss_weights={"loss_ce": 1.0, "loss_mask": 2.0, "loss_dice": 3.0},
        matcher=matcher,
        criterion=criterion,
    )


def make_clip_target():
    labels = torch.tensor([0], dtype=torch.int64)
    ids = torch.tensor([[41, 41]], dtype=torch.int64)
    masks = torch.zeros(1, 2, 8, 8)
    masks[0, 0, 1:4, 1:4] = 1
    masks[0, 1, 2:6, 2:6] = 1
    return {
        "labels": labels,
        "ids": ids,
        "masks": masks,
        # This synthetic fixture represents complete source annotation coverage.
        "entity_frame_supervision_valid": torch.ones(2, dtype=torch.bool),
    }


def test_adapter_converts_e3a_outputs_to_official_videomt_frame_contract():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [make_clip_target()],
    )

    assert matcher.outputs["pred_masks"].shape == (2, 3, 1, 8, 8)
    assert matcher.outputs["pred_logits"].shape == (2, 3, 2)
    torch.testing.assert_close(
        matcher.outputs["pred_logits"][0],
        matcher.outputs["pred_logits"][1],
    )
    assert len(matcher.targets) == 2
    assert matcher.targets[0]["ids"].shape == (1, 1)
    assert matcher.targets[1]["ids"].shape == (1, 1)
    assert matcher.targets[0]["masks"].shape == (1, 1, 8, 8)
    assert matcher.targets[1]["masks"].shape == (1, 1, 8, 8)
    assert matcher.targets[0]["mask_valid"].tolist() == [[True]]
    assert matcher.targets[1]["mask_valid"].tolist() == [[True]]
    torch.testing.assert_close(matcher.targets[0]["ids"], torch.tensor([[41]]))
    torch.testing.assert_close(matcher.targets[1]["ids"], torch.tensor([[41]]))

    expected_objective = (
        losses["loss_ce"] + 2.0 * losses["loss_mask"] + 3.0 * losses["loss_dice"]
    )
    torch.testing.assert_close(losses["loss_objective"], expected_objective)
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert objectness_logits.grad is not None


def test_adapter_exposes_the_same_clip_consistent_track_to_slot_assignment():
    adapter = make_criterion()
    losses, track_to_slot = adapter.forward_with_matches(
        {
            "mask_logits": torch.randn(1, 2, 3, 8, 8, requires_grad=True),
            "objectness_logits": torch.randn(1, 2, 3, requires_grad=True),
        },
        [make_clip_target()],
    )
    assert torch.isfinite(losses["loss_objective"])
    assert track_to_slot == [{41: 0}]


def test_adapter_skips_unannotated_mask_frames_but_keeps_explicit_absence():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    target = make_clip_target()
    target["ids"] = torch.tensor([[41, -1]], dtype=torch.int64)
    target["masks"][0, 1].zero_()
    target["mask_valid"] = torch.tensor([[True, False]])
    target["entity_frame_supervision_valid"] = torch.tensor([True, False])

    losses, track_to_slot = adapter.forward_with_matches(
        {
            "mask_logits": torch.randn(1, 2, 3, 8, 8, requires_grad=True),
            "objectness_logits": torch.randn(1, 2, 3, requires_grad=True),
        },
        [target],
    )

    assert track_to_slot == [{41: 0}]
    assert criterion.mask_loss_indices is not None
    assert criterion.mask_loss_targets[0]["mask_valid"].tolist() == [[True]]
    assert criterion.mask_loss_targets[1]["mask_valid"].tolist() == [[False]]
    assert criterion.mask_loss_indices[0][1].tolist() == [0]
    assert criterion.mask_loss_indices[1][1].numel() == 0
    assert torch.isfinite(losses["loss_objective"])


def test_adapter_handles_an_all_empty_batch_without_sampling_empty_masks():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)
    empty_target = {
        "labels": torch.empty(0, dtype=torch.int64),
        "ids": torch.empty(0, 2, dtype=torch.int64),
        "masks": torch.empty(0, 2, 8, 8),
    }

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [empty_target],
    )

    assert losses["loss_ce"].item() == 0
    assert losses["loss_mask"].item() == 0
    assert losses["loss_dice"].item() == 0
    losses["loss_objective"].backward()
    assert objectness_logits.grad is not None
    assert mask_logits.grad is not None
    assert torch.count_nonzero(objectness_logits.grad) == 0
    assert torch.count_nonzero(mask_logits.grad) == 0
    assert matcher.targets is None


def test_adapter_rejects_track_id_changes_within_a_clip():
    adapter = make_criterion()
    target = make_clip_target()
    target["ids"] = torch.tensor([[41, 42]], dtype=torch.int64)

    with pytest.raises(ValueError, match="changes ID across frames"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            [target],
        )


def test_adapter_rejects_more_visible_tracks_than_entity_slots():
    adapter = make_criterion()
    target = {
        "labels": torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        "ids": torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.int64),
        "masks": torch.ones(4, 2, 8, 8),
    }

    with pytest.raises(ValueError, match="only 3 entity slots"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            [target],
        )


def test_adapter_rejects_nonzero_mask_for_a_missing_track_frame():
    adapter = make_criterion()
    target = make_clip_target()
    target["ids"] = torch.tensor([[41, -1]], dtype=torch.int64)
    target["masks"][0, 1, 2:6, 2:6] = 0.5

    with pytest.raises(ValueError, match="zero masks wherever track ID is -1"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            [target],
        )


def test_real_videomt_criterion_contract_when_optional_dependencies_are_installed():
    pytest.importorskip("detectron2")
    pytest.importorskip("scipy")

    adapter = VidEoMTMaskCriterion(
        frames=2,
        num_points=32,
        loss_weights={"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0},
    )
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [make_clip_target()],
    )

    assert {"loss_ce", "loss_mask", "loss_dice", "loss_objective"} <= set(losses)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert objectness_logits.grad is not None


def test_adapter_accepts_the_official_vggt_trainer_batch_signature():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)
    clip_target = make_clip_target()
    batch = {
        "images": torch.rand(1, 2, 3, 8, 8),
        "timestamps": torch.tensor([[0.0, 0.1]]),
        "entity_targets": {
            "labels": torch.tensor([[0]], dtype=torch.int64),
            "ids": clip_target["ids"].unsqueeze(0),
            "masks": clip_target["masks"].unsqueeze(0),
            "pixel_valid": torch.ones(1, 2, 8, 8, dtype=torch.bool),
            "track_counts": torch.tensor([1], dtype=torch.int64),
        },
    }

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        batch,
        schedule_progress=0.25,
    )

    assert torch.isfinite(losses["loss_objective"])
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert objectness_logits.grad is not None


def test_padded_targets_stay_batch_aligned_under_upstream_gradient_accumulation():
    from pathlib import Path
    import sys

    training_root = Path(__file__).resolve().parents[1] / "third_party" / "vggt-omega" / "training"
    if not training_root.is_dir():
        pytest.skip("pinned VGGT-Ω training utilities are not checked out")
    sys.path.insert(0, str(training_root))
    from trainer_utils import get_chunk_from_data

    batch_size, frames = 4, 2
    clip_targets = []
    for batch_index, count in enumerate([1, 0, 2, 1]):
        labels = torch.zeros(count, dtype=torch.int64)
        ids = torch.full((count, frames), -1, dtype=torch.int64)
        masks = torch.zeros(count, frames, 8, 8)
        for track_index in range(count):
            ids[track_index] = 100 + batch_index * 10 + track_index
            masks[track_index, 0, 1:4, 1:4] = 1
            masks[track_index, 1, 2:6, 2:6] = 1
        clip_targets.append({"labels": labels, "ids": ids, "masks": masks})

    batch = {
        "images": torch.rand(batch_size, frames, 3, 8, 8),
        "timestamps": torch.arange(frames).float().expand(batch_size, -1),
        "entity_targets": collate_entity_targets(clip_targets),
        "entity_supervision_valid": torch.tensor([True, False, True, True]),
        "entity_frame_supervision_valid": torch.tensor(
            [
                [True, False],
                [False, False],
                [True, True],
                [False, True],
            ],
            dtype=torch.bool,
        ),
    }
    chunk = get_chunk_from_data(batch, chunk_id=0, num_chunks=2)

    assert chunk["images"].shape[0] == 2
    assert chunk["entity_supervision_valid"].tolist() == [True, False]
    assert chunk["entity_frame_supervision_valid"].tolist() == [
        [True, False],
        [False, False],
    ]
    assert chunk["entity_targets"]["labels"].shape[0] == 2
    assert chunk["entity_targets"]["ids"].shape[0] == 2
    assert chunk["entity_targets"]["masks"].shape[0] == 2
    assert chunk["entity_targets"]["track_counts"].tolist() == [1, 0]

    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    losses = adapter(
        {
            "mask_logits": torch.randn(2, frames, 3, 8, 8, requires_grad=True),
            "objectness_logits": torch.randn(2, frames, 3, requires_grad=True),
        },
        chunk,
        schedule_progress=0.5,
    )
    assert torch.isfinite(losses["loss_objective"])
    assert len(matcher.targets) == frames
    assert [target["labels"].numel() for target in matcher.targets] == [1, 1]


def test_legacy_padded_batch_without_mask_valid_marks_only_real_rows_annotated():
    adapter = make_criterion()
    clips = [make_clip_target(), {
        "labels": torch.empty(0, dtype=torch.int64),
        "ids": torch.empty(0, 2, dtype=torch.int64),
        "masks": torch.empty(0, 2, 8, 8),
    }]
    padded = collate_entity_targets(clips)
    legacy_padded = {key: value for key, value in padded.items() if key != "mask_valid"}

    unpacked = adapter._unpack_batched_targets(
        legacy_padded,
        batch_size=2,
        num_frames=2,
        device=torch.device("cpu"),
    )

    assert unpacked[0]["mask_valid"].tolist() == [[True, True]]
    assert unpacked[1]["mask_valid"].shape == (0, 2)


def test_adapter_rejects_trainer_batch_without_entity_targets():
    adapter = make_criterion()
    with pytest.raises(ValueError, match="entity_targets"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            {"images": torch.rand(1, 2, 3, 8, 8)},
            schedule_progress=0.0,
        )



def _install_detectron2_point_sampling_stubs(monkeypatch):
    """Replace only Detectron2's point-sampling helpers for CPU upstream tests."""
    import sys
    from types import ModuleType

    import torch.nn.functional as F

    detectron2 = ModuleType("detectron2")
    detectron2.__path__ = []
    utils = ModuleType("detectron2.utils")
    utils.__path__ = []
    comm = ModuleType("detectron2.utils.comm")
    comm.get_world_size = lambda: 1

    projects = ModuleType("detectron2.projects")
    projects.__path__ = []
    point_rend = ModuleType("detectron2.projects.point_rend")
    point_rend.__path__ = []
    point_features = ModuleType("detectron2.projects.point_rend.point_features")

    def point_sample(input_tensor, point_coords, align_corners=False):
        grid = point_coords.mul(2.0).sub(1.0).unsqueeze(1)
        return F.grid_sample(
            input_tensor,
            grid,
            align_corners=align_corners,
        ).squeeze(2)

    def get_uncertain_point_coords_with_randomness(
        logits,
        uncertainty_func,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    ):
        batch_size = logits.shape[0]
        sampled_count = max(int(num_points * oversample_ratio), num_points)
        uncertain_count = min(
            int(num_points * importance_sample_ratio),
            num_points,
        )
        random_count = num_points - uncertain_count
        sampled_coords = torch.rand(
            batch_size,
            sampled_count,
            2,
            device=logits.device,
            dtype=logits.dtype,
        )
        sampled_logits = point_sample(
            logits,
            sampled_coords,
            align_corners=False,
        )
        uncertainty = uncertainty_func(sampled_logits).squeeze(1)
        if uncertain_count:
            indices = uncertainty.topk(uncertain_count, dim=1).indices
            uncertain_coords = sampled_coords.gather(
                1,
                indices.unsqueeze(-1).expand(-1, -1, 2),
            )
        else:
            uncertain_coords = sampled_coords[:, :0]
        random_coords = torch.rand(
            batch_size,
            random_count,
            2,
            device=logits.device,
            dtype=logits.dtype,
        )
        return torch.cat((uncertain_coords, random_coords), dim=1)

    point_features.point_sample = point_sample
    point_features.get_uncertain_point_coords_with_randomness = (
        get_uncertain_point_coords_with_randomness
    )

    for name, module in {
        "detectron2": detectron2,
        "detectron2.utils": utils,
        "detectron2.utils.comm": comm,
        "detectron2.projects": projects,
        "detectron2.projects.point_rend": point_rend,
        "detectron2.projects.point_rend.point_features": point_features,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_pinned_videomt_matcher_and_criterion_run_on_cpu_with_sampling_boundary_stubs(
    monkeypatch,
):
    from pathlib import Path
    import sys

    pytest.importorskip("scipy")
    repo_root = Path(__file__).resolve().parents[1]
    videomt_root = repo_root / "third_party" / "videomt"
    if not videomt_root.is_dir():
        pytest.skip("pinned VidEoMT submodule is not checked out")

    try:
        import detectron2  # noqa: F401
    except ImportError:
        _install_detectron2_point_sampling_stubs(monkeypatch)

    adapter = VidEoMTMaskCriterion(
        frames=3,
        num_points=32,
        loss_weights={"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0},
    )
    assert adapter.matcher.__class__.__name__ == "VideoHungarianMatcher_Consistent"
    assert adapter.criterion.__class__.__name__ == "VideoSetCriterion"
    assert "videomt.config" not in sys.modules
    assert "videomt.videomt" not in sys.modules

    target = {
        "labels": torch.tensor([0], dtype=torch.int64),
        "ids": torch.tensor([[71, -1, 71]], dtype=torch.int64),
        "masks": torch.zeros(1, 3, 8, 8),
    }
    target["masks"][0, 0, 1:4, 1:4] = 1
    target["masks"][0, 2, 2:6, 2:6] = 1

    flat_targets, _ = adapter._validate_targets(
        [target],
        batch_size=1,
        num_frames=3,
        num_queries=2,
        device=torch.device("cpu"),
    )
    matcher_outputs = {
        "pred_logits": torch.randn(3, 2, 2),
        "pred_masks": torch.randn(3, 2, 1, 8, 8),
    }
    matches = adapter.matcher(matcher_outputs, flat_targets)
    assert len(matches) == 3
    assert all(torch.equal(matches[0][0], item[0]) for item in matches[1:])
    assert all(torch.equal(matches[0][1], item[1]) for item in matches[1:])
    assert matches[0][1].tolist() == [0]

    mask_logits = torch.randn(1, 3, 2, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 3, 2, requires_grad=True)
    losses, track_to_slot = adapter.forward_with_matches(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [target],
    )

    assert set(track_to_slot[0]) == {71}
    assert 0 <= track_to_slot[0][71] < 2
    assert {"loss_ce", "loss_mask", "loss_dice", "loss_objective"} <= set(losses)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert objectness_logits.grad is not None
    assert torch.isfinite(mask_logits.grad).all()
    assert torch.isfinite(objectness_logits.grad).all()

    empty_target = {
        "labels": torch.empty(0, dtype=torch.int64),
        "ids": torch.empty(0, 3, dtype=torch.int64),
        "masks": torch.empty(0, 3, 8, 8),
    }
    empty_losses = adapter(
        {
            "mask_logits": torch.randn(1, 3, 2, 8, 8, requires_grad=True),
            "objectness_logits": torch.randn(1, 3, 2, requires_grad=True),
        },
        [empty_target],
    )
    assert torch.isfinite(empty_losses["loss_ce"])
    assert empty_losses["loss_mask"].item() == 0
    assert empty_losses["loss_dice"].item() == 0


def _make_real_videomt_adapter(monkeypatch, *, frames):
    from pathlib import Path

    pytest.importorskip("scipy")
    repo_root = Path(__file__).resolve().parents[1]
    videomt_root = repo_root / "third_party" / "videomt"
    if not videomt_root.is_dir():
        pytest.skip("pinned VidEoMT submodule is not checked out")

    try:
        import detectron2  # noqa: F401
    except ImportError:
        _install_detectron2_point_sampling_stubs(monkeypatch)

    return VidEoMTMaskCriterion(
        frames=frames,
        num_points=64,
        loss_weights={"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0},
    )


def _make_partial_pixel_valid_target(*, frames=1):
    labels = torch.tensor([0], dtype=torch.int64)
    ids = torch.full((1, frames), 71, dtype=torch.int64)
    masks = torch.zeros(1, frames, 8, 8)
    pixel_valid = torch.zeros(frames, 8, 8, dtype=torch.bool)
    pixel_valid[:, :4, :] = True
    masks[0, :, 1:3, 1:3] = 1.0
    # Deliberately place poison labels in unknown pixels; they must be ignored.
    masks[0, :, 5:7, 5:7] = 1.0
    return {
        "labels": labels,
        "ids": ids,
        "masks": masks,
        "pixel_valid": pixel_valid,
    }


def _make_pixel_valid_predictions(*, invalid_query0_value, invalid_query1_value):
    mask_logits = torch.full((1, 1, 2, 8, 8), -4.0)
    # Query 0 is correct on every known pixel.
    mask_logits[0, 0, 0, :4, :] = -4.0
    mask_logits[0, 0, 0, 1:3, 1:3] = 4.0
    # Query 1 is the opposite on every known pixel.
    mask_logits[0, 0, 1, :4, :] = 4.0
    mask_logits[0, 0, 1, 1:3, 1:3] = -4.0
    # These values are adversarial but lie entirely in unknown rows.
    mask_logits[0, 0, 0, 4:, :] = invalid_query0_value
    mask_logits[0, 0, 1, 4:, :] = invalid_query1_value
    return mask_logits.requires_grad_()


def test_pixel_valid_excludes_unknown_pixels_from_matching_loss_and_gradients(monkeypatch):
    adapter = _make_real_videomt_adapter(monkeypatch, frames=1)
    target = _make_partial_pixel_valid_target(frames=1)
    objectness = torch.zeros(1, 1, 2, requires_grad=True)

    first_logits = _make_pixel_valid_predictions(
        invalid_query0_value=-80.0,
        invalid_query1_value=80.0,
    )
    torch.manual_seed(19)
    first_losses, first_matches = adapter.forward_with_matches(
        {"mask_logits": first_logits, "objectness_logits": objectness},
        [target],
    )
    assert first_matches == [{71: 0}]
    first_losses["loss_objective"].backward()

    assert first_logits.grad is not None
    assert torch.count_nonzero(first_logits.grad[..., 4:, :]) == 0
    assert torch.count_nonzero(first_logits.grad[..., :4, :]) > 0

    second_logits = _make_pixel_valid_predictions(
        invalid_query0_value=80.0,
        invalid_query1_value=-80.0,
    )
    torch.manual_seed(19)
    second_losses, second_matches = adapter.forward_with_matches(
        {
            "mask_logits": second_logits,
            "objectness_logits": objectness.detach().clone(),
        },
        [target],
    )
    assert second_matches == first_matches
    for key in ("loss_ce", "loss_mask", "loss_dice", "loss_objective"):
        torch.testing.assert_close(first_losses[key], second_losses[key])


def test_pixel_valid_matcher_uses_first_frame_with_known_foreground(monkeypatch):
    adapter = _make_real_videomt_adapter(monkeypatch, frames=2)
    target = _make_partial_pixel_valid_target(frames=2)
    target["pixel_valid"][0].zero_()
    target["masks"][0, 0].fill_(1.0)
    target["masks"][0, 1].zero_()
    target["masks"][0, 1, 1:3, 1:3] = 1.0

    mask_logits = torch.randn(1, 2, 2, 8, 8, requires_grad=True)
    objectness_logits = torch.zeros(1, 2, 2, requires_grad=True)
    losses, matches = adapter.forward_with_matches(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [target],
    )

    assert matches[0].get(71) in {0, 1}
    assert torch.isfinite(losses["loss_mask"])
    assert torch.isfinite(losses["loss_dice"])
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert torch.count_nonzero(mask_logits.grad[:, 0]) == 0


def test_pixel_valid_rejects_tracks_without_any_known_foreground_frame():
    adapter = make_criterion(frames=2)
    target = make_clip_target()
    target["pixel_valid"] = torch.zeros(2, 8, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match="no pixel-valid visible mask frame"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            [target],
        )


def test_batched_pixel_valid_is_preserved_by_adapter():
    adapter = make_criterion()
    clip = make_clip_target()
    pixel_valid = torch.ones(1, 2, 8, 8, dtype=torch.bool)
    pixel_valid[:, 0, 0, 0] = False
    batch = {
        "labels": torch.tensor([[0]], dtype=torch.int64),
        "ids": clip["ids"].unsqueeze(0),
        "masks": clip["masks"].unsqueeze(0),
        "mask_valid": torch.ones(1, 1, 2, dtype=torch.bool),
        "pixel_valid": pixel_valid,
        "track_counts": torch.tensor([1], dtype=torch.int64),
    }

    unpacked = adapter._unpack_batched_targets(
        batch,
        batch_size=1,
        num_frames=2,
        device=torch.device("cpu"),
    )
    assert torch.equal(unpacked[0]["pixel_valid"], pixel_valid[0])


def test_entity_supervision_valid_filters_unannotated_clips_from_all_entity_losses():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    valid_clip = make_clip_target()
    empty_clip = {
        "labels": torch.empty(0, dtype=torch.int64),
        "ids": torch.empty(0, 2, dtype=torch.int64),
        "masks": torch.empty(0, 2, 8, 8),
    }
    padded = collate_entity_targets([valid_clip, empty_clip])
    padded["pixel_valid"] = torch.ones(2, 2, 8, 8, dtype=torch.bool)
    batch = {
        "entity_targets": padded,
        "entity_supervision_valid": torch.tensor([True, False]),
    }

    mask_logits = torch.randn(2, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(2, 2, 3, requires_grad=True)
    with torch.no_grad():
        mask_logits[1].fill_(75.0)
        objectness_logits[1].fill_(75.0)

    losses, track_to_slot = adapter.forward_with_matches(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        batch,
    )
    assert track_to_slot[0] == {41: 0}
    assert track_to_slot[1] == {}
    assert matcher.targets is not None
    assert len(matcher.targets) == 2
    assert matcher.outputs["pred_logits"].shape[0] == 2

    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert objectness_logits.grad is not None
    assert torch.count_nonzero(mask_logits.grad[1]) == 0
    assert torch.count_nonzero(objectness_logits.grad[1]) == 0


def test_explicitly_invalid_entity_clip_is_ignored_even_if_targets_are_present():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    batch = {
        "entity_targets": collate_entity_targets([make_clip_target()]),
        "entity_supervision_valid": torch.tensor([False]),
    }
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)

    losses, track_to_slot = adapter.forward_with_matches(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        batch,
    )
    assert track_to_slot == [{}]
    assert losses["loss_ce"].item() == 0
    assert losses["loss_mask"].item() == 0
    assert losses["loss_dice"].item() == 0
    assert matcher.targets is None
    losses["loss_objective"].backward()
    assert torch.count_nonzero(mask_logits.grad) == 0
    assert torch.count_nonzero(objectness_logits.grad) == 0


def test_entity_supervision_valid_requires_boolean_batch_vector():
    adapter = make_criterion()
    batch = {
        "entity_targets": collate_entity_targets([make_clip_target()]),
        "entity_supervision_valid": torch.tensor([1], dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="must be bool with shape"):
        adapter(
            {
                "mask_logits": torch.randn(1, 2, 3, 8, 8),
                "objectness_logits": torch.randn(1, 2, 3),
            },
            batch,
        )



def test_explicitly_valid_empty_scene_trains_no_object_classification_only():
    matcher = RecordingMatcher()
    criterion = RecordingCriterion(matcher)
    adapter = make_criterion(matcher=matcher, criterion=criterion)
    empty_clip = {
        "labels": torch.empty(0, dtype=torch.int64),
        "ids": torch.empty(0, 2, dtype=torch.int64),
        "masks": torch.empty(0, 2, 8, 8),
    }
    padded = collate_entity_targets([empty_clip])
    padded["pixel_valid"] = torch.ones(1, 2, 8, 8, dtype=torch.bool)
    batch = {
        "entity_targets": padded,
        # This may only be true when the source explicitly verifies an empty scene.
        "entity_supervision_valid": torch.tensor([True]),
    }
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.ones(1, 2, 3, requires_grad=True)

    losses, track_to_slot = adapter.forward_with_matches(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        batch,
    )

    assert track_to_slot == [{}]
    assert matcher.targets is not None
    assert len(matcher.targets) == 2
    assert all(target["labels"].numel() == 0 for target in matcher.targets)
    assert losses["loss_ce"].item() > 0
    assert losses["loss_mask"].item() == 0
    assert losses["loss_dice"].item() == 0

    losses["loss_objective"].backward()
    assert objectness_logits.grad is not None
    assert torch.count_nonzero(objectness_logits.grad) > 0
    assert mask_logits.grad is not None
    assert torch.count_nonzero(mask_logits.grad) == 0



def test_pixel_invalid_frames_are_excluded_from_no_object_classification(monkeypatch):
    adapter = _make_real_videomt_adapter(monkeypatch, frames=2)
    target = _make_partial_pixel_valid_target(frames=2)
    # Frame 0 has no known pixels; frame 1 is fully annotated.
    target["pixel_valid"][0].zero_()
    target["pixel_valid"][1].fill_(True)
    target["entity_frame_supervision_valid"] = torch.ones(2, dtype=torch.bool)
    target["masks"][0, 0].fill_(1.0)  # Poison annotation in the unknown frame.
    target["masks"][0, 1].zero_()
    target["masks"][0, 1, 1:3, 1:3] = 1.0

    class_loss_batches = []
    original_loss_labels = adapter.criterion.loss_labels

    def record_class_loss(outputs, targets, indices, num_masks):
        class_loss_batches.append((outputs["pred_logits"].shape[0], len(targets)))
        return original_loss_labels(outputs, targets, indices, num_masks)

    monkeypatch.setattr(adapter.criterion, "loss_labels", record_class_loss)

    mask_logits = torch.randn(1, 2, 2, 8, 8, requires_grad=True)
    objectness_logits = torch.zeros(1, 2, 2, requires_grad=True)
    losses, matches = adapter.forward_with_matches(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [target],
    )

    assert matches[0].get(71) in {0, 1}
    assert class_loss_batches == [(1, 1)]
    assert torch.isfinite(losses["loss_ce"])
    assert torch.isfinite(losses["loss_mask"])
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert torch.count_nonzero(mask_logits.grad[:, 0]) == 0
    assert torch.count_nonzero(mask_logits.grad[:, 1]) > 0
    # The unknown frame must not leak into the clip-level class-logit pooling.
    assert torch.count_nonzero(objectness_logits.grad[:, 0]) == 0
    assert torch.count_nonzero(objectness_logits.grad[:, 1]) > 0



def test_entity_frame_supervision_valid_overrides_pixel_default_for_class_loss(monkeypatch):
    adapter = _make_real_videomt_adapter(monkeypatch, frames=2)
    target = make_clip_target()
    target["entity_frame_supervision_valid"] = torch.tensor([False, True])

    class_loss_batches = []
    original_loss_labels = adapter.criterion.loss_labels

    def record_class_loss(outputs, targets, indices, num_masks):
        class_loss_batches.append((outputs["pred_logits"].shape[0], len(targets)))
        return original_loss_labels(outputs, targets, indices, num_masks)

    monkeypatch.setattr(adapter.criterion, "loss_labels", record_class_loss)

    mask_logits = torch.randn(1, 2, 2, 8, 8, requires_grad=True)
    objectness_logits = torch.zeros(1, 2, 2, requires_grad=True)
    losses, matches = adapter.forward_with_matches(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [target],
    )

    assert matches[0].get(41) in {0, 1}
    assert class_loss_batches == [(1, 1)]
    assert torch.isfinite(losses["loss_ce"])
    losses["loss_objective"].backward()
    assert objectness_logits.grad is not None
    assert torch.count_nonzero(objectness_logits.grad[:, 0]) == 0
    assert torch.count_nonzero(objectness_logits.grad[:, 1]) > 0



def test_adapter_defaults_no_object_classification_off_without_source_coverage():
    adapter = make_criterion()
    target = make_clip_target()
    target.pop("entity_frame_supervision_valid")
    mask_logits = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    objectness_logits = torch.randn(1, 2, 3, requires_grad=True)

    losses = adapter(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [target],
    )
    assert losses["loss_ce"].item() == 0
    assert losses["loss_mask"].item() > 0
    assert losses["loss_dice"].item() > 0
    losses["loss_objective"].backward()
    assert objectness_logits.grad is not None
    assert torch.count_nonzero(objectness_logits.grad) == 0



def test_mask_invalid_frame_is_excluded_from_consistent_matching_even_when_pixels_are_valid(
    monkeypatch,
):
    adapter = _make_real_videomt_adapter(monkeypatch, frames=2)
    target = _make_partial_pixel_valid_target(frames=2)
    target["pixel_valid"].fill_(True)
    target["ids"][0, 0] = -1
    target["masks"][0, 0].zero_()
    target["mask_valid"] = torch.tensor([[False, True]])

    def forbidden_upstream_matcher(*args, **kwargs):
        raise AssertionError(
            "partially annotated tracks must use the validity-aware matcher"
        )

    monkeypatch.setattr(adapter.matcher, "forward", forbidden_upstream_matcher)
    mask_logits = torch.randn(1, 2, 2, 8, 8, requires_grad=True)
    objectness_logits = torch.zeros(1, 2, 2, requires_grad=True)
    losses, matches = adapter.forward_with_matches(
        {"mask_logits": mask_logits, "objectness_logits": objectness_logits},
        [target],
    )

    assert matches[0].get(71) in {0, 1}
    assert torch.isfinite(losses["loss_mask"])
    assert torch.isfinite(losses["loss_dice"])
    losses["loss_objective"].backward()
    assert mask_logits.grad is not None
    assert torch.count_nonzero(mask_logits.grad[:, 0]) == 0
    assert torch.count_nonzero(mask_logits.grad[:, 1]) > 0
