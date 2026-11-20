import numpy as np
import pytest

from e3a.data.hoi4d import (
    decode_hoi4d_motion_mask,
    hoi4d_color_palette,
    build_hoi4d_entity_clip_targets,
)


def test_hoi4d_palette_and_category_specific_semantic_instance_mapping():
    palette = hoi4d_color_palette()
    rgb = np.stack(
        [
            [palette[1], palette[2], palette[3]],
            [palette[4], palette[5], palette[6]],
        ]
    )

    semantic, instance = decode_hoi4d_motion_mask(rgb, "C8")

    assert semantic.tolist() == [[9, 47, 9], [9, 9, 47]]
    assert instance.tolist() == [[1, 2, 1], [3, 3, 4]]


def test_hoi4d_clip_targets_merge_object_parts_exclude_hands_and_preserve_coverage():
    palette = hoi4d_color_palette()
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    first[0, 0] = palette[1]  # object instance 1
    first[0, 1] = palette[2]  # right hand, excluded
    first[0, 2] = palette[3]  # another part of object instance 1
    first[1, 0] = palette[4]  # object instance 3
    first[1, 1] = palette[5]  # another part of object instance 3
    first[1, 2] = palette[6]  # left hand, excluded

    third = np.zeros((2, 3, 3), dtype=np.uint8)
    third[1, 2] = palette[4]  # instance 3 only

    targets = build_hoi4d_entity_clip_targets([first, None, third], "C8")

    assert targets["labels"].tolist() == [0, 0]
    assert targets["ids"].tolist() == [[1, -1, -1], [3, -1, 3]]
    assert targets["mask_valid"].tolist() == [
        [True, False, True],
        [True, False, True],
    ]
    assert targets["masks"].shape == (2, 3, 2, 3)
    assert targets["masks"][0, 0].sum().item() == 2
    assert targets["masks"][1, 0].sum().item() == 2
    assert targets["masks"][1, 2].sum().item() == 1


def test_hoi4d_decoder_rejects_unknown_non_background_palette_colors():
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = (123, 45, 67)

    with pytest.raises(ValueError, match="not defined by the official category palette"):
        decode_hoi4d_motion_mask(rgb, "C1")


def test_hoi4d_clip_requires_at_least_one_available_mask_frame():
    with pytest.raises(ValueError, match="At least one HOI4D mask frame"):
        build_hoi4d_entity_clip_targets([None, None], "C1")
