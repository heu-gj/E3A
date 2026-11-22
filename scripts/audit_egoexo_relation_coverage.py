"""Read-only metadata audit for Ego-Exo4D Relation no-object supervision coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from e3a.data.egoexo import audit_egoexo_relation_coverage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One Relation JSON file or a directory containing only Relation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/audits/EgoExo4D-Relation-Coverage"),
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Recursive filename pattern when --input is a directory (default: *.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if input_path.is_file():
        annotation_files = [input_path]
    elif input_path.is_dir():
        annotation_files = sorted(input_path.rglob(args.pattern))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not annotation_files:
        raise FileNotFoundError(
            f"No annotation JSON files found under {input_path} "
            f"with pattern {args.pattern!r}."
        )

    take_rows: list[dict[str, Any]] = []
    camera_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    for annotation_path in annotation_files:
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read Relation JSON {annotation_path}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"Relation JSON root must be an object: {annotation_path}")

        audit = audit_egoexo_relation_coverage(payload)
        relative_path = (
            str(annotation_path.relative_to(input_path))
            if input_path.is_dir()
            else annotation_path.name
        )
        source_sha256 = _sha256(annotation_path)
        source_files.append(
            {"annotation_file": relative_path, "sha256": source_sha256}
        )

        for take in audit["takes"]:
            row = {key: value for key, value in take.items() if key != "cameras"}
            row["annotation_file"] = relative_path
            row["source_sha256"] = source_sha256
            take_rows.append(row)
            for camera in take.get("cameras", []):
                camera_row = dict(camera)
                camera_row["take_uid"] = take["take_uid"]
                camera_row["annotation_file"] = relative_path
                camera_row["source_sha256"] = source_sha256
                camera_rows.append(camera_row)

    summary = {
        "annotation_file_count": len(source_files),
        "take_count": len(take_rows),
        "takes_with_valid_object_names": sum(
            bool(row.get("object_names_valid")) for row in take_rows
        ),
        "takes_with_matching_object_enumeration": sum(
            bool(row.get("enumeration_matches")) for row in take_rows
        ),
        "takes_with_any_eligible_camera": sum(
            bool(row.get("eligible_camera_count", 0)) for row in take_rows
        ),
        "camera_count": len(camera_rows),
        "cameras_with_complete_coverage": sum(
            bool(row["coverage_complete"]) for row in camera_rows
        ),
        "eligible_camera_count": sum(
            bool(row["eligible_for_no_object"]) for row in camera_rows
        ),
        "eligible_camera_frame_pairs": sum(
            int(row["common_annotated_frame_count"])
            for row in camera_rows
            if row["eligible_for_no_object"]
        ),
        "source_files": source_files,
        "source_masks_decoded": False,
        "source_files_modified": False,
        "training_or_model_execution": False,
    }

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "takes.csv",
        take_rows,
        [
            "annotation_file",
            "source_sha256",
            "take_uid",
            "status",
            "object_names_valid",
            "object_masks_valid",
            "enumeration_matches",
            "enumerated_nonhand_tracks",
            "mask_nonhand_tracks",
            "enumerated_only_tracks",
            "mask_only_tracks",
            "camera_count",
            "eligible_camera_count",
            "eligible_camera_frame_pairs",
        ],
    )
    _write_csv(
        output_dir / "camera_coverage.csv",
        camera_rows,
        [
            "annotation_file",
            "source_sha256",
            "take_uid",
            "camera_name",
            "enumerated_track_count",
            "camera_track_coverage_count",
            "missing_camera_tracks",
            "invalid_coverage_tracks",
            "coverage_complete",
            "eligible_for_no_object",
            "common_annotated_frames",
            "common_annotated_frame_count",
        ],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
