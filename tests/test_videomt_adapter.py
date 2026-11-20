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
        valid_count = sum(int(target_index.numel()) for _, target_index in indices)
        loss = outputs["pred_masks"].sum() * 0.0 + float(valid_count)
        return {"loss_mask": loss, "loss_dice": loss}


def make_criterion(frames=2, matcher=None, criterion=None):
    matcher = matcher or RecordingMatcher()
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
    return {"labels": labels, "ids": ids, "masks": masks}


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

    losses, track_to_slot = adapter.forward_with_matches(
        {
            "mask_logits": torch.randn(1, 2, 3, 8, 8, requires_grad=True),
            "objectness_logits": torch.randn(1, 2, 3, requires_grad=True),
        },
        [target],
    )

    assert track_to_slot == [{41: 0}]
    assert criterion.mask_loss_indices is not None
    assert matcher.targets[0]["mask_valid"].tolist() == [[True]]
    assert matcher.targets[1]["mask_valid"].tolist() == [[False]]
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

    assert losses["loss_ce"].isfinite()
    assert losses["loss_mask"].item() == 0
    assert losses["loss_dice"].item() == 0
    losses["loss_objective"].backward()
    assert objectness_logits.grad is not None
    assert mask_logits.grad is not None
    assert torch.count_nonzero(mask_logits.grad) == 0


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
    }
    chunk = get_chunk_from_data(batch, chunk_id=0, num_chunks=2)

    assert chunk["images"].shape[0] == 2
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
    assert len(matcher.targets) == 2 * frames
    assert [target["labels"].numel() for target in matcher.targets] == [1, 1, 0, 0]


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

    assert track_to_slot == [{71: 0}]
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
