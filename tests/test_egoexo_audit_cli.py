import csv
import json
import runpy
import sys
from pathlib import Path


def test_egoexo_relation_coverage_cli_writes_parseable_reports(tmp_path, monkeypatch):
    relation = {
        "annotations": {
            "complete_take": {
                "object_names": {
                    "annotation": [
                        {"track_id": "cup_0"},
                        {"track_id": "bottle_0"},
                    ]
                },
                "object_masks": {
                    "cup_0": {
                        "cam04": {"annotated_frames": [0, 1, 2], "annotation": {}}
                    },
                    "bottle_0": {
                        "cam04": {"annotated_frames": [0, 2], "annotation": {}}
                    },
                },
            },
            "missing_camera_take": {
                "object_names": {
                    "annotation": [
                        {"track_id": "cup_1"},
                        {"track_id": "bottle_1"},
                    ]
                },
                "object_masks": {
                    "cup_1": {
                        "cam04": {"annotated_frames": [0, 1], "annotation": {}}
                    },
                    "bottle_1": {
                        "cam05": {"annotated_frames": [0, 1], "annotation": {}}
                    },
                },
            },
            "no_camera_records_take": {
                "object_names": {
                    "annotation": [{"track_id": "cup_2"}]
                },
                "object_masks": {"cup_2": {}},
            },
            "sampled_mask_subset_take": {
                "object_names": {
                    "annotation": [
                        {"track_id": "cup_3"},
                        {"track_id": "bottle_3"},
                    ]
                },
                "object_masks": {
                    "cup_3": {
                        "cam04": {"annotated_frames": [0, 1], "annotation": {}}
                    }
                },
            },
        }
    }

    source_path = tmp_path / "relation.json"
    source_text = json.dumps(relation)
    source_path.write_text(source_text, encoding="utf-8")
    output_dir = tmp_path / "audit-output"
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_egoexo_relation_coverage.py"
    )
    namespace = runpy.run_path(
        str(script_path),
        run_name="e3a_egoexo_relation_coverage_cli_test",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--input",
            str(source_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    namespace["main"]()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    take_rows = list(
        csv.DictReader((output_dir / "takes.csv").open(encoding="utf-8", newline=""))
    )
    camera_rows = list(
        csv.DictReader(
            (output_dir / "camera_coverage.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )

    assert summary["take_count"] == 4
    assert summary["takes_with_matching_object_enumeration"] == 3
    assert summary["object_enumeration_match_rate"] == 0.75
    assert summary["takes_with_sampled_mask_subset"] == 1
    assert summary["takes_with_missing_camera_records"] == 2
    assert summary["takes_without_camera_records"] == 1
    assert summary["eligible_camera_count"] == 1
    assert summary["eligible_camera_frame_pairs"] == 2

    assert len(take_rows) == 4
    complete_take = next(row for row in take_rows if row["take_uid"] == "complete_take")
    assert complete_take["enumerated_nonhand_track_count"] == "2"
    assert complete_take["mask_nonhand_track_count"] == "2"
    assert complete_take["camera_record_count"] == "1"

    assert len(camera_rows) == 4
    complete_camera = next(
        row
        for row in camera_rows
        if row["take_uid"] == "complete_take" and row["camera_name"] == "cam04"
    )
    assert complete_camera["coverage_complete"] == "True"
    assert complete_camera["common_annotated_frame_count"] == "2"

    assert source_path.read_text(encoding="utf-8") == source_text
