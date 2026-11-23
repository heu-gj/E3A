"""Validity-aware implementations of VidEoMT's matching/loss interfaces.

The task definitions are unchanged: first-visible-frame consistent Hungarian
matching, foreground/no-object CE, point BCE and Dice. Points are GT pixel
centers with conservative validity on *all* bilinear logit neighbors. This
module needs only PyTorch; it never imports a detector or another backbone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _axis_neighbors(length: int, prediction_length: int, device: torch.device):
    # Integer arithmetic avoids rounding across a bilinear cell boundary.
    numerator = (2 * torch.arange(length, device=device) + 1) * prediction_length - length
    lower = torch.div(numerator, 2 * length, rounding_mode="floor")
    fraction = (numerator - lower * (2 * length)).float() / (2 * length)
    return lower.clamp(0, prediction_length - 1), (lower + 1).clamp(
        0, prediction_length - 1
    ), fraction


def conservative_pixel_support(
    pixel_valid: torch.Tensor, prediction_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid logit cells and safe GT centers, including interpolation.

    Adaptive max pooling of invalidity includes every source pixel overlapping
    each prediction cell (also for nonintegral scales). A GT center is used only
    if its source validity and all four bilinear neighbors are valid. Border
    replication, not zero padding, defines samples at the image boundary.
    """
    if pixel_valid.dtype != torch.bool or pixel_valid.ndim != 2:
        raise ValueError("pixel_valid must be bool [H_gt,W_gt].")
    height, width = pixel_valid.shape
    ph, pw = prediction_size
    if min(height, width, ph, pw) <= 0:
        raise ValueError("Pixel grids must have positive dimensions.")
    if (height, width) == (ph, pw):
        return pixel_valid, pixel_valid
    invalid = F.adaptive_max_pool2d((~pixel_valid)[None, None].float(), (ph, pw))[0, 0]
    prediction_valid = invalid == 0
    y0, y1, _ = _axis_neighbors(height, ph, pixel_valid.device)
    x0, x1, _ = _axis_neighbors(width, pw, pixel_valid.device)
    safe = pixel_valid.clone()
    for ys in (y0, y1):
        for xs in (x0, x1):
            safe &= prediction_valid[ys[:, None], xs[None, :]]
    return prediction_valid, safe


def valid_point_indices(valid: torch.Tensor, limit: int) -> torch.Tensor:
    """Deterministic stratified sampling; use all valid pixels below the cap."""
    if limit <= 0:
        return torch.empty(0, dtype=torch.int64, device=valid.device)
    indices = valid.flatten().nonzero(as_tuple=False).flatten()
    if indices.numel() <= limit:
        return indices
    if limit == 1:
        return indices[indices.numel() // 2:indices.numel() // 2 + 1]
    offsets = torch.arange(limit, device=indices.device) * (indices.numel() - 1) // (limit - 1)
    return indices[offsets]


def sample_centers(
    logits: torch.Tensor, indices: torch.Tensor, target_size: tuple[int, int],
) -> torch.Tensor:
    """Bilinear samples [N,P] at exact GT centers; no invalid neighbor mixing.

    Indices must come from conservative_pixel_support. Explicit integer
    neighbors implement align_corners=False / padding_mode=border without
    floating coordinate rounding causing an unvalidated neighbor to enter.
    """
    height, width = target_size
    ph, pw = logits.shape[-2:]
    if (height, width) == (ph, pw):
        return logits.flatten(1)[:, indices]
    y0, y1, dy = _axis_neighbors(height, ph, logits.device)
    x0, x1, dx = _axis_neighbors(width, pw, logits.device)
    y, x = indices // width, indices % width
    dy, dx = dy[y], dx[x]
    return (
        logits[:, y0[y], x0[x]] * ((1 - dy) * (1 - dx))
        + logits[:, y0[y], x1[x]] * ((1 - dy) * dx)
        + logits[:, y1[y], x0[x]] * (dy * (1 - dx))
        + logits[:, y1[y], x1[x]] * (dy * dx)
    )


def linear_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact rectangular Hungarian assignment [queries,tracks], tracks<=queries.

    Like the pinned SciPy matcher, assignment is a detached CPU operation. The
    transfer is explicit and indices are returned on the original device. The
    shortest-augmenting-path solver avoids an optional SciPy runtime dependency
    for this small query/track matrix; there is no greedy approximation.
    """
    queries, tracks = cost.shape
    if tracks > queries:
        raise ValueError("Cannot assign more tracks than available queries.")
    if not torch.isfinite(cost).all():
        raise ValueError("Assignment costs must be finite.")
    if tracks == 0:
        empty = torch.empty(0, dtype=torch.int64, device=cost.device)
        return empty, empty
    matrix = cost.detach().to(device="cpu", dtype=torch.float64).T.tolist()
    u, v = [0.] * (tracks + 1), [0.] * (queries + 1)
    matched, way = [0] * (queries + 1), [0] * (queries + 1)
    for row in range(1, tracks + 1):
        matched[0] = row
        column = 0
        minimum, used = [float("inf")] * (queries + 1), [False] * (queries + 1)
        while True:
            used[column] = True
            current_row = matched[column]
            delta, next_column = float("inf"), 0
            for j in range(1, queries + 1):
                if not used[j]:
                    reduced = matrix[current_row - 1][j - 1] - u[current_row] - v[j]
                    if reduced < minimum[j]:
                        minimum[j], way[j] = reduced, column
                    if minimum[j] < delta:
                        delta, next_column = minimum[j], j
            for j in range(queries + 1):
                if used[j]:
                    u[matched[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while column:
            previous = way[column]
            matched[column] = matched[previous]
            column = previous
    source = [j - 1 for j in range(1, queries + 1) if matched[j]]
    target = [matched[j] - 1 for j in range(1, queries + 1) if matched[j]]
    return (torch.tensor(source, dtype=torch.int64, device=cost.device),
            torch.tensor(target, dtype=torch.int64, device=cost.device))


def positive_tracks(target: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Known foreground, independent of whether the scene is fully enumerated."""
    masks = target["masks"][:, 0]
    return (
        (target["ids"].flatten() >= 0) & target["mask_valid"].flatten()
        & (masks * target["pixel_valid"]).flatten(1).gt(0).any(1)
    )


def _index_list(indices: torch.Tensor) -> list[int]:
    """Explicit small-index transfer for the discrete matching control flow."""
    return indices.detach().to(device="cpu").tolist()


class ValidityAwareConsistentMatcher(nn.Module):
    """VidEoMT's first-visible-frame policy with validity-aware mask costs."""

    supports_entity_validity = True

    def __init__(self, *, frames: int, num_points: int, cost_class: float = 1.,
                 cost_mask: float = 1., cost_dice: float = 1.) -> None:
        super().__init__()
        self.frames, self.num_points = frames, num_points
        self.cost_class, self.cost_mask, self.cost_dice = cost_class, cost_mask, cost_dice

    def cost_matrix(
        self, outputs: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """[K,N] costs; masks [K,1,h,w], GT [N,1,H,W], pixel_valid [H,W]."""
        masks = outputs["pred_masks"][:, 0].float()
        gt = target["masks"][:, 0].float()
        cost = -outputs["pred_logits"].float().softmax(-1)[:, target["labels"]]
        _, safe = conservative_pixel_support(target["pixel_valid"], masks.shape[-2:])
        indices = valid_point_indices(safe, self.num_points)
        total = self.cost_class * cost
        if indices.numel():
            predicted = sample_centers(masks, indices, gt.shape[-2:])
            labels = gt.flatten(1)[:, indices]
            # Same pairwise BCE/Dice formulas as VidEoMT, restricted to valid points.
            ce = (F.softplus(predicted).mean(1)[:, None]
                  - predicted @ labels.T / indices.numel())
            prob = predicted.sigmoid()
            dice = 1 - (2 * prob @ labels.T + 1) / (
                prob.sum(1)[:, None] + labels.sum(1)[None] + 1)
            total = total + self.cost_mask * ce + self.cost_dice * dice
        return total

    @torch.no_grad()
    def forward(
        self, outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if len(targets) % self.frames:
            raise ValueError("Flattened frames must contain complete clips.")
        results = []
        queries = outputs["pred_logits"].shape[1]
        device = outputs["pred_logits"].device
        for start in range(0, len(targets), self.frames):
            assigned: dict[int, int] = {}
            for f in range(self.frames):
                target = targets[start + f]
                new = [row for row in _index_list(positive_tracks(target).nonzero().flatten())
                       if row not in assigned]
                if not new:
                    continue
                available = [q for q in range(queries) if q not in assigned.values()]
                source = torch.tensor(available, dtype=torch.int64, device=device)
                rows = torch.tensor(new, dtype=torch.int64, device=device)
                part = {**target, "labels": target["labels"][rows],
                        "masks": target["masks"][rows]}
                prediction = {key: value[start + f, source] for key, value in outputs.items()}
                costs = self.cost_matrix(prediction, part)
                qi, ti = linear_assignment(costs)
                for q, t in zip(_index_list(qi), _index_list(ti), strict=True):
                    assigned[new[t]] = available[q]
            ordered = sorted(assigned, key=lambda row: assigned[row])
            match = (torch.tensor([assigned[row] for row in ordered], dtype=torch.int64,
                                  device=device),
                     torch.tensor(ordered, dtype=torch.int64, device=device))
            results.extend([match] * self.frames)
        return results


class ValidityAwareVideoCriterion(nn.Module):
    """VidEoMT-compatible CE/BCE/Dice calls with independent validity gates."""

    supports_entity_validity = True

    def __init__(self, *, matcher: nn.Module, num_points: int, eos_coef: float,
                 oversample_ratio: float, importance_sample_ratio: float) -> None:
        super().__init__()
        self.matcher = matcher
        self.num_points, self.eos_coef = num_points, eos_coef
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.losses = ["labels", "masks"]

    def loss_labels(self, outputs, targets, indices, num_masks: float | None = None):
        logits = outputs["pred_logits"].float()
        numerator, denominator = logits.flatten()[:0].sum(), logits.new_zeros(())
        for f, (target, (source, rows)) in enumerate(zip(targets, indices, strict=True)):
            labels = torch.ones(logits.shape[1], dtype=torch.int64, device=logits.device)
            selected = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            # Unknown pixels also invalidate a claim of *complete* scene coverage.
            complete = bool(target["entity_classification_valid"].item())
            complete = complete and bool(target["pixel_valid"].all().item())
            complete = complete and bool(target["mask_valid"].all().item())
            if complete:
                selected[:] = True
            positive = positive_tracks(target)[rows]
            labels[source[positive]] = 0
            selected[source[positive]] = True
            if selected.any().item():
                ce = F.cross_entropy(logits[f, selected], labels[selected], reduction="none")
                weights = torch.where(labels[selected] == 0, 1., self.eos_coef)
                numerator = numerator + (ce * weights).sum()
                denominator = denominator + weights.sum()
        normalizer = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        return {"loss_ce": numerator / normalizer}

    def loss_masks(self, outputs, targets, indices, num_masks: float | None = None):
        logits = outputs["pred_masks"].float()
        bce, dice = logits.flatten()[:0].sum(), logits.flatten()[:0].sum()
        count = 0
        for f, (target, (source, rows)) in enumerate(zip(targets, indices, strict=True)):
            _, safe = conservative_pixel_support(target["pixel_valid"], logits.shape[-2:])
            candidates = valid_point_indices(
                safe, max(self.num_points, int(self.num_points * self.oversample_ratio))
            )
            if not candidates.numel():
                continue
            for query, row in zip(_index_list(source), _index_list(rows), strict=True):
                if not bool(target["mask_valid"][row].item()):
                    continue
                values = sample_centers(logits[f, query, 0][None], candidates, safe.shape)[0]
                points = candidates
                if candidates.numel() > self.num_points:
                    important = int(self.num_points * self.importance_sample_ratio)
                    uncertain = values.detach().abs().neg().topk(important).indices
                    remaining = torch.ones_like(candidates, dtype=torch.bool)
                    remaining[uncertain] = False
                    other = valid_point_indices(remaining, self.num_points - important)
                    chosen = torch.cat([uncertain, other])
                    values, points = values[chosen], candidates[chosen]
                labels = target["masks"][row, 0].flatten()[points].float()
                bce = bce + F.binary_cross_entropy_with_logits(values, labels, reduction="mean")
                prob = values.sigmoid()
                dice = dice + 1 - (2 * (prob * labels).sum() + 1) / (
                    prob.sum() + labels.sum() + 1)
                count += 1
        return {"loss_mask": bce / max(count, 1), "loss_dice": dice / max(count, 1)}

    def forward(self, outputs, targets, ret_match_result=False):
        indices = self.matcher(outputs, targets)
        losses = {**self.loss_labels(outputs, targets, indices),
                  **self.loss_masks(outputs, targets, indices)}
        return (losses, indices) if ret_match_result else losses
