from e3a.data.actions import (
    action_label_key,
    build_dataset_prefixed_action_vocabulary,
)


def test_dataset_prefixed_action_vocabulary_does_not_merge_identical_labels():
    vocabulary = build_dataset_prefixed_action_vocabulary(
        {
            "egoexo4d": ["step_unique_id=843", "step_unique_id=844"],
            "hoi4d": ["pour", "step_unique_id=843"],
        }
    )

    assert vocabulary == {
        "egoexo4d::step_unique_id=843": 0,
        "egoexo4d::step_unique_id=844": 1,
        "hoi4d::pour": 2,
        "hoi4d::step_unique_id=843": 3,
    }
    assert action_label_key("HOI4D", "pour") == "hoi4d::pour"


def test_dataset_prefixed_vocabulary_rejects_empty_labels():
    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        build_dataset_prefixed_action_vocabulary({"hoi4d": [""]})
