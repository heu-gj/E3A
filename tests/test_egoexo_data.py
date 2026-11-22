import numpy as np
import pytest

from e3a.data.egoexo import (
    EGOEXO_HAND_JOINTS,
    audit_egoexo_relation_coverage,
    build_egoexo_anticipation_target,
    build_egoexo_keystep_vocabulary,
    build_egoexo_relation_clip_targets,
    parse_egoexo_hand_pose_annotations,
    parse_egoexo_keystep_segments,
    resolve_egoexo_video_path,
)


def make_hand_annotation(*, low_view_joint=None):
    joints = {}
    for hand_index, hand in enumerate(("right", "left")):
        for joint_index, joint_name in enumerate(EGOEXO_HAND_JOINTS):
            view_count = 2 if (hand, joint_name) == low_view_joint else 3
            joints[f"{hand}_{joint_name}"] = {
                "x": float(joint_index + 1),
                "y": float(hand_index + 2),
                "z": 3.0,
                "num_views_for_3d": view_count,
            }
    return {"12": [{"annotation3D": joints}]}


def test_egoexo_hand_pose_reorders_joints_and_transforms_world_to_camera():
    annotations = make_hand_annotation(low_view_joint=("right", "wrist"))
    camera_pose = {
        "12": {
            "aria": {
                "camera_extrinsics": [
                    [1.0, 0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0, -2.0],
                    [0.0, 0.0, 1.0, -3.0],
                ]
            }
        }
    }

    parsed = parse_egoexo_hand_pose_annotations(
        annotations,
        camera_pose,
        camera_name="aria",
        min_views_for_3d=3,
        source_to_model_units=2.0,
    )

    frame = parsed[12]
    assert frame["keypoints_camera"].shape == (2, 21, 3)
    assert frame["valid"].shape == (2, 21)
    assert frame["valid"][0, 0].item() is False
    assert frame["valid"][0, 1].item() is True
    # right thumb_1 world=(2,2,3), then world-to-camera=(-?); scale is applied last.
    np.testing.assert_allclose(
        frame["keypoints_camera"][0, 1].numpy(),
        np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(frame["keypoints_camera"][0, 0].numpy(), [0, 0, 0])


def test_egoexo_hand_pose_auto_annotation_can_disable_manual_view_threshold():
    annotations = make_hand_annotation(low_view_joint=("right", "wrist"))
    annotations["12"][0]["annotation3D"]["right_wrist"].pop("num_views_for_3d")
    camera_pose = {
        "12": {
            "aria": {
                "camera_extrinsics": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            }
        }
    }

    parsed = parse_egoexo_hand_pose_annotations(
        annotations,
        camera_pose,
        min_views_for_3d=None,
    )

    assert parsed[12]["valid"][0, 0].item() is True


def test_egoexo_hand_pose_supports_official_camera_first_pose_layout():
    annotations = make_hand_annotation()
    camera_pose = {
        "aria": {
            "camera_intrinsics": np.eye(3).tolist(),
            "camera_extrinsics": {
                "12": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            },
        }
    }

    parsed = parse_egoexo_hand_pose_annotations(
        annotations,
        camera_pose,
        min_views_for_3d=3,
    )

    assert parsed[12]["valid"].all()
    np.testing.assert_allclose(
        parsed[12]["keypoints_camera"][1, 0].numpy(),
        np.asarray([1.0, 3.0, 3.0], dtype=np.float32),
    )


def test_egoexo_hand_pose_fails_closed_when_camera_pose_is_missing():
    with pytest.raises(KeyError, match="camera_extrinsics"):
        parse_egoexo_hand_pose_annotations(
            make_hand_annotation(),
            {},
            camera_name="aria",
            min_views_for_3d=3,
        )





def test_egoexo_relation_coverage_audit_reports_only_complete_camera_frames():
    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [
                        {"track_id": "object_a"},
                        {"track_id": "object_b"},
                    ]
                },
                "object_masks": {
                    "object_a": {
                        "cam04": {"annotated_frames": [0, 1, 2], "annotation": {}}
                    },
                    "object_b": {
                        "cam04": {"annotated_frames": [0, 2], "annotation": {}}
                    },
                },
            }
        }
    }

    audit = audit_egoexo_relation_coverage(relation)

    assert audit["summary"]["take_count"] == 1
    assert audit["summary"]["takes_with_matching_object_enumeration"] == 1
    assert audit["summary"]["eligible_camera_count"] == 1
    camera = audit["takes"][0]["cameras"][0]
    assert camera["camera_name"] == "cam04"
    assert camera["coverage_complete"] is True
    assert camera["eligible_for_no_object"] is True
    assert camera["common_annotated_frames"] == [0, 2]


def test_egoexo_relation_coverage_audit_fails_closed_on_sampled_mask_subset():
    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [
                        {"track_id": "object_a"},
                        {"track_id": "object_b"},
                    ]
                },
                "object_masks": {
                    "object_a": {
                        "cam04": {"annotated_frames": [0, 1], "annotation": {}}
                    }
                },
            }
        }
    }

    audit = audit_egoexo_relation_coverage(relation)

    take = audit["takes"][0]
    assert take["status"] == "OBJECT_ENUMERATION_MISMATCH"
    assert take["enumerated_only_tracks"] == ["object_b"]
    assert take["eligible_camera_count"] == 0
    assert take["cameras"][0]["eligible_for_no_object"] is False


def test_egoexo_relation_coverage_audit_fails_closed_on_missing_camera_track():
    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [
                        {"track_id": "object_a"},
                        {"track_id": "object_b"},
                    ]
                },
                "object_masks": {
                    "object_a": {
                        "cam04": {"annotated_frames": [0, 1], "annotation": {}}
                    },
                    "object_b": {
                        "cam05": {"annotated_frames": [0, 1], "annotation": {}}
                    },
                },
            }
        }
    }

    audit = audit_egoexo_relation_coverage(relation)

    camera = audit["takes"][0]["cameras"][0]
    assert camera["camera_name"] == "cam04"
    assert camera["missing_camera_tracks"] == ["object_b"]
    assert camera["eligible_for_no_object"] is False

def test_egoexo_relation_targets_distinguish_absence_from_unannotated_frames():
    mask0 = np.zeros((2, 3), dtype=np.bool_)
    mask0[0, 1] = True
    mask2 = np.zeros((2, 3), dtype=np.bool_)
    mask2[1, 2] = True

    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [
                        {"track_id": "stainless bowl_0"},
                        {"track_id": "left hand_0"},
                    ]
                },
                "object_masks": {
                    "stainless bowl_0": {
                        "aria01_214-1": {
                            "annotated_frames": [0, 1, 2],
                            "annotation": {
                                "0": {"width": 3, "height": 2, "encodedMask": "m0"},
                                "2": {"width": 3, "height": 2, "encodedMask": "m2"},
                            },
                        }
                    },
                    "left hand_0": {
                        "aria01_214-1": {
                            "annotated_frames": [0, 1, 2],
                            "annotation": {
                                "0": {"width": 3, "height": 2, "encodedMask": "hand"}
                            },
                        }
                    },
                }
            }
        }
    }

    def decoder(record):
        return {"m0": mask0, "m2": mask2, "hand": mask0}[record["encodedMask"]]

    targets = build_egoexo_relation_clip_targets(
        relation,
        "take-1",
        "aria01_214-1",
        [0, 1, 2, 3],
        mask_decoder=decoder,
    )

    assert targets["track_names"] == ["stainless bowl_0"]
    assert targets["ids"].tolist() == [[0, -1, 0, -1]]
    assert targets["mask_valid"].tolist() == [[True, True, True, False]]
    assert targets["entity_frame_supervision_valid"].tolist() == [True, True, True, False]
    assert targets["entity_supervision_valid"] is True
    assert targets["masks"].shape == (1, 4, 2, 3)
    assert targets["masks"][0, 1].sum().item() == 0
    assert targets["masks"][0, 2].sum().item() == 1


def test_egoexo_frame_classification_requires_coverage_for_every_object_track():
    mask = np.ones((2, 2), dtype=np.bool_)
    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [
                        {"track_id": "object_a"},
                        {"track_id": "object_b"},
                    ]
                },
                "object_masks": {
                    "object_a": {
                        "cam04": {
                            "annotated_frames": [0, 1],
                            "annotation": {
                                "0": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    },
                    "object_b": {
                        "cam04": {
                            "annotated_frames": [0],
                            "annotation": {
                                "0": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    },
                }
            }
        }
    }

    targets = build_egoexo_relation_clip_targets(
        relation,
        "take-1",
        "cam04",
        [0, 1],
        mask_decoder=lambda _: mask,
    )

    assert targets["track_names"] == ["object_a", "object_b"]
    assert targets["entity_frame_supervision_valid"].tolist() == [True, False]
    assert targets["entity_supervision_valid"] is True


def test_egoexo_frame_classification_is_disabled_when_object_masks_are_sampled_subset():
    mask = np.ones((2, 2), dtype=np.bool_)
    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [
                        {"track_id": "object_a"},
                        {"track_id": "object_b"},
                    ]
                },
                "object_masks": {
                    "object_a": {
                        "cam04": {
                            "annotated_frames": [0, 1],
                            "annotation": {
                                "0": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    }
                },
            }
        }
    }

    targets = build_egoexo_relation_clip_targets(
        relation,
        "take-1",
        "cam04",
        [0, 1],
        mask_decoder=lambda _: mask,
    )

    assert targets["track_names"] == ["object_a"]
    assert targets["entity_supervision_valid"] is True
    assert targets["entity_frame_supervision_valid"].tolist() == [False, False]


def test_egoexo_frame_classification_fails_closed_when_an_object_has_no_camera_record():
    mask = np.ones((2, 2), dtype=np.bool_)
    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [
                        {"track_id": "object_a"},
                        {"track_id": "object_b"},
                    ]
                },
                "object_masks": {
                    "object_a": {
                        "cam04": {
                            "annotated_frames": [0, 1],
                            "annotation": {
                                "0": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    },
                    "object_b": {
                        "cam05": {
                            "annotated_frames": [0, 1],
                            "annotation": {
                                "0": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    },
                }
            }
        }
    }

    targets = build_egoexo_relation_clip_targets(
        relation,
        "take-1",
        "cam04",
        [0, 1],
        image_size=(2, 2),
        mask_decoder=lambda _: mask,
    )

    assert targets["track_names"] == ["object_a"]
    assert targets["entity_frame_supervision_valid"].tolist() == [False, False]
    assert targets["entity_supervision_valid"] is True


def test_egoexo_hand_filter_keeps_non_hand_objects_containing_the_word_hand():
    mask = np.ones((2, 2), dtype=np.bool_)
    relation = {
        "annotations": {
            "take-1": {
                "object_masks": {
                    "hand blender_0": {
                        "cam04": {
                            "annotated_frames": [0],
                            "annotation": {
                                "0": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    }
                }
            }
        }
    }

    targets = build_egoexo_relation_clip_targets(
        relation,
        "take-1",
        "cam04",
        [0],
        mask_decoder=lambda _: mask,
    )

    assert targets["track_names"] == ["hand blender_0"]
    assert targets["ids"].tolist() == [[0]]


def test_egoexo_anticipation_rejects_invalid_segment_end_time():
    with pytest.raises(ValueError, match="invalid interval or step ID"):
        build_egoexo_anticipation_target(
            [{"start_time": 2.0, "end_time": 1.0, "step_unique_id": 42}],
            observation_end_time=1.0,
            action_vocabulary={42: 0},
        )


def test_egoexo_relation_rejects_mask_frames_outside_annotated_coverage():
    relation = {
        "annotations": {
            "take-1": {
                "object_masks": {
                    "object_0": {
                        "cam04": {
                            "annotated_frames": [0],
                            "annotation": {
                                "1": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    }
                }
            }
        }
    }

    with pytest.raises(ValueError, match="outside annotated_frames"):
        build_egoexo_relation_clip_targets(
            relation,
            "take-1",
            "cam04",
            [0, 1],
            mask_decoder=lambda _: np.ones((2, 2), dtype=np.bool_),
        )


def test_egoexo_video_path_uses_official_aria_stream_alias(tmp_path):
    relative = "aria01/rgb.mp4"
    path = tmp_path / "takes" / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"video")
    takes = [
        {
            "take_uid": "take-1",
            "root_dir": "takes",
            "frame_aligned_videos": {
                "aria01": {"rgb": {"relative_path": relative}}
            },
        }
    ]

    resolved = resolve_egoexo_video_path(
        tmp_path,
        takes,
        "take-1",
        "aria01_214-1",
    )

    assert resolved == path



def test_egoexo_keystep_parser_and_train_only_vocabulary_use_step_unique_ids():
    raw = {
        "annotations": {
            "take-1": {
                "segments": [
                    {
                        "start_time": 3.0,
                        "end_time": 4.0,
                        "step_unique_id": 42,
                        "step_name": "Second action",
                    },
                    {
                        "start_time": 0.0,
                        "end_time": 2.0,
                        "step_unique_id": 7,
                        "step_name": "First action",
                    },
                ]
            }
        }
    }

    segments = parse_egoexo_keystep_segments(raw, "take-1")
    vocabulary = build_egoexo_keystep_vocabulary(raw)

    assert [segment["step_unique_id"] for segment in segments] == [7, 42]
    assert vocabulary == {7: 0, 42: 1}


def test_egoexo_anticipation_uses_first_strictly_future_keystep():
    segments = [
        {"start_time": 0.0, "end_time": 1.5, "step_unique_id": 7},
        {"start_time": 2.0, "end_time": 3.0, "step_unique_id": 42},
        {"start_time": 4.0, "end_time": 5.0, "step_unique_id": 99},
    ]
    target = build_egoexo_anticipation_target(
        segments,
        observation_end_time=1.5,
        action_vocabulary={7: 0, 42: 1},
    )

    assert target == {
        "target_entity_track_id": -1,
        "target_entity_valid": False,
        "action_label": 1,
        "action_valid": True,
        "time_to_event": 0.5,
        "time_valid": True,
    }


def test_egoexo_anticipation_keeps_time_but_masks_unseen_action_vocabulary_ids():
    target = build_egoexo_anticipation_target(
        [{"start_time": 2.0, "end_time": 3.0, "step_unique_id": 99}],
        observation_end_time=1.0,
        action_vocabulary={7: 0},
    )

    assert target["action_label"] == -1
    assert target["action_valid"] is False
    assert target["time_to_event"] == 1.0
    assert target["time_valid"] is True
    assert target["target_entity_valid"] is False


def test_egoexo_anticipation_without_future_event_is_fully_invalid():
    target = build_egoexo_anticipation_target(
        [{"start_time": 0.0, "end_time": 3.0, "step_unique_id": 7}],
        observation_end_time=3.0,
        action_vocabulary={7: 0},
    )

    assert target["action_valid"] is False
    assert target["time_valid"] is False
    assert target["target_entity_valid"] is False



def test_egoexo_anticipation_uses_dataset_prefixed_action_vocabulary():
    target = build_egoexo_anticipation_target(
        [{"start_time": 2.0, "end_time": 3.0, "step_unique_id": 843}],
        observation_end_time=1.0,
        action_vocabulary={"egoexo4d::step_unique_id=843": 4},
    )

    assert target["action_label"] == 4
    assert target["action_valid"] is True



def test_egoexo_frame_coverage_requires_all_camera_object_tracks_to_be_annotated():
    mask = np.ones((2, 2), dtype=np.bool_)
    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [
                        {"track_id": "object_a"},
                        {"track_id": "object_b"},
                    ]
                },
                "object_masks": {
                    "object_a": {
                        "cam04": {
                            "annotated_frames": [0, 1],
                            "annotation": {
                                "0": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    },
                    "object_b": {
                        "cam04": {
                            "annotated_frames": [0],
                            "annotation": {
                                "0": {"width": 2, "height": 2, "encodedMask": "m"}
                            },
                        }
                    },
                }
            }
        }
    }

    targets = build_egoexo_relation_clip_targets(
        relation,
        "take-1",
        "cam04",
        [0, 1],
        mask_decoder=lambda _: mask,
    )

    assert targets["entity_frame_supervision_valid"].tolist() == [True, False]
    assert targets["entity_supervision_valid"] is True


def test_egoexo_fully_annotated_empty_window_is_valid_no_object_supervision():
    relation = {
        "annotations": {
            "take-1": {
                "object_names": {
                    "annotation": [{"track_id": "object_a"}]
                },
                "object_masks": {
                    "object_a": {
                        "cam04": {
                            "annotated_frames": [0, 1],
                            "annotation": {},
                        }
                    }
                }
            }
        }
    }

    targets = build_egoexo_relation_clip_targets(
        relation,
        "take-1",
        "cam04",
        [0, 1],
        image_size=(2, 2),
    )

    assert targets["labels"].numel() == 0
    assert targets["entity_frame_supervision_valid"].tolist() == [True, True]
    assert targets["entity_supervision_valid"] is True
