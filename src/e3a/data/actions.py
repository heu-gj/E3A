"""Dataset-prefixed action vocabulary helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def action_label_key(dataset: str, source_label: str | int) -> str:
    """Create an unambiguous key for a source-native action label."""
    normalized_dataset = dataset.strip().lower()
    normalized_label = str(source_label).strip()
    if not normalized_dataset or not normalized_label:
        raise ValueError("dataset and source_label must be non-empty.")
    if "::" in normalized_dataset:
        raise ValueError("dataset names cannot contain the '::' separator.")
    return f"{normalized_dataset}::{normalized_label}"


def build_dataset_prefixed_action_vocabulary(
    labels_by_dataset: Mapping[str, Sequence[str | int]],
) -> dict[str, int]:
    """Build one deterministic action vocabulary without cross-dataset merging.

    Example:
        {
            "egoexo4d": ["step_unique_id=843", "step_unique_id=844"],
            "hoi4d": ["pour", "pick and place"],
        }

    Identical text labels from different datasets receive different class IDs.
    Pass training-split labels only; validation/test-only labels are intentionally
    absent and should be marked action_valid=False by the target adapter.
    """
    if not isinstance(labels_by_dataset, Mapping):
        raise TypeError("labels_by_dataset must be a mapping.")
    keys: set[str] = set()
    for dataset, labels in labels_by_dataset.items():
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset names must be non-empty strings.")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise TypeError(f"labels for dataset {dataset!r} must be a sequence.")
        for label in labels:
            keys.add(action_label_key(dataset, label))
    return {key: index for index, key in enumerate(sorted(keys))}
