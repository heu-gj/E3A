"""Read-only full HOI4D hand-pose coverage audit; outputs stay outside the dataset.

No MANO forward, extraction, download or training is performed. Missing files
are unannotated/unknown, never automatically classified as download corruption.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from e3a.data.hoi4d_dataset import (
    _NumpyArraysOnlyUnpickler, _read_mano_record, resolve_hoi4d_sequence_paths,
)

SHAPES = {"poseCoeff": (48,), "beta": (10,), "trans": (3,), "kps2D": (21, 2)}
SIDES = ("right", "left")
HAND_ROOT = Path("extracted/hand_pose/Hand_pose")
FIXED = ("ZY20210800001/H1/C20/N30/S295/s04/T4",
         "ZY20210800001/H1/C20/N18/S286/s01/T1")


def sha_stream(stream):
    h = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        h.update(chunk)
    return h.hexdigest()


def sequence_id(parts):
    """Validate complete release hierarchy; never key sequences by T alone."""
    prefixes = ("ZY", "H", "C", "N", "S", "s", "T")
    if len(parts) != 7 or any(not x.startswith(p) or not x[len(p):].isdigit()
                              for x, p in zip(parts, prefixes)):
        return None
    return "/".join(parts)


def inspect_record(path):
    """Use the production parser; restricted raw read diagnoses its failures."""
    result = {"parser_ok": False, "raw_read_ok": False, "shape_ok": False,
              "nan": 0, "inf": 0, "fields": {}, "error": None}
    try:
        record = _read_mano_record(path)
        result.update(parser_ok=True, raw_read_ok=True, shape_ok=True)
    except Exception as exc:
        result["error"] = type(exc).__name__
        result["error_message"] = str(exc).replace(str(path), '${RECORD}')
        try:
            with path.open("rb") as f:
                record = _NumpyArraysOnlyUnpickler(f, encoding="latin1").load()
            result["raw_read_ok"] = True
        except Exception as raw_exc:
            result["raw_error"] = type(raw_exc).__name__
            result["raw_error_message"] = str(raw_exc).replace(str(path), '${RECORD}')
            result["failure_class"] = (
                'PARSER_UNSUPPORTED_GLOBAL' if 'Disallowed global' in str(raw_exc)
                else 'RESTRICTED_DESERIALIZATION_FAILED')
            return result
    if not isinstance(record, dict):
        result["error"] = "NOT_MAPPING"
        return result
    shape_ok = True
    for name, shape in SHAPES.items():
        if name not in record:
            result["fields"][name] = {"missing": True}
            shape_ok = False
            continue
        try:
            value = np.asarray(record[name])
            numeric = value.astype(np.float32)
            nan, inf = int(np.isnan(numeric).sum()), int(np.isinf(numeric).sum())
            finite = numeric[np.isfinite(numeric)]
            result["fields"][name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                "nan": nan, "inf": inf,
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None}
            result["nan"] += nan
            result["inf"] += inf
            shape_ok &= value.shape == shape
        except (TypeError, ValueError, OverflowError):
            result["fields"][name] = {"numeric_conversion": False}
            shape_ok = False
    result["shape_ok"] = bool(shape_ok)
    return result


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {"status": "BLOCKED", "error": "VIDEO_OPEN_FAILED"}
        n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if n <= 0 or not np.isfinite(fps) or fps <= 0:
            return {"status": "BLOCKED", "error": "INVALID_VIDEO_METADATA"}
        return {"status": "PASS", "frame_count": n, "fps": fps, "duration": n / fps}
    finally:
        cap.release()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def archive_member_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        return None
    for i, part in enumerate(p.parts):
        if part in ("handpose_right_hand", "handpose_left_hand"):
            return (HAND_ROOT / Path(*p.parts[i:])).as_posix()
    return None


def compare_archives(root, db):
    archives = sorted(p for p in root.rglob("*") if p.is_file() and "hand" in p.name.lower()
                      and "pose" in p.name.lower() and
                      (p.name.lower().endswith(".zip") or ".tar" in p.name.lower()))
    results = []
    db.execute("CREATE TABLE members(path TEXT PRIMARY KEY, digest TEXT)")
    for archive in archives:
        before = archive.stat()
        row = {"path": archive.relative_to(root).as_posix(), "size": before.st_size,
               "members": 0, "matched": 0, "missing": [], "mismatch": [],
               "duplicate_members": [], "unsafe_members": [], "errors": []}
        def member(name, stream):
            if not name.lower().endswith(".pickle"):
                return
            dest = archive_member_path(name)
            if dest is None:
                row["unsafe_members"].append(name)
                return
            row["members"] += 1
            digest = sha_stream(stream)  # ZIP stream reaches EOF and verifies CRC.
            prior = db.execute("SELECT digest FROM members WHERE path=?", (dest,)).fetchone()
            if prior:
                row["duplicate_members"].append({"path": dest, "conflicting": prior[0] != digest})
            else:
                db.execute("INSERT INTO members VALUES (?,?)", (dest, digest))
            actual = db.execute("SELECT digest FROM files WHERE path=?", (dest,)).fetchone()
            if actual is None:
                row["missing"].append(dest)
            elif actual[0] != digest:
                row["mismatch"].append(dest)
            else:
                row["matched"] += 1
        try:
            if archive.name.lower().endswith(".zip"):
                with zipfile.ZipFile(archive) as z:
                    for entry in z.infolist():
                        if not entry.is_dir():
                            with z.open(entry) as stream:
                                member(entry.filename, stream)
            else:
                with tarfile.open(archive, "r|*") as t:
                    for entry in t:
                        if entry.isfile():
                            with t.extractfile(entry) as stream:
                                member(entry.name, stream)
        except Exception as exc:
            row["errors"].append(type(exc).__name__)
        after = archive.stat()
        row["source_stat_unchanged"] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        results.append(row)
        print(json.dumps({"archive": row['path'], "members": row['members'], "matched": row['matched']}), flush=True)
    db.commit()
    extra = [r[0] for r in db.execute("SELECT path FROM files WHERE path NOT IN (SELECT path FROM members) ORDER BY path")]
    integrity = "NOT_VERIFIABLE"
    if results:
        integrity = "EXTRACTION_INCOMPLETE" if any(x["missing"] or x["mismatch"] or x["errors"] or x["unsafe_members"] for x in results) else "EXTRACTION_COMPLETE"
    return {"archives": results, "extracted_extra_paths": extra if results else [],
            "extraction_integrity": integrity,
            "extraction_scope": "Byte equivalence to available local archives only; no official release manifest supplied.",
            "download_integrity": "NOT_VERIFIABLE", "official_checksum_manifest": "NOT_AVAILABLE"}


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Audit output must be outside the dataset.")
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    database = output / "inventory.sqlite"
    if database.exists():
        raise FileExistsError("Refusing to overwrite an existing audit database.")
    db = sqlite3.connect(database)
    db.execute("CREATE TABLE files(path TEXT PRIMARY KEY, digest TEXT)")
    side_stats, frames, valid_frames, directories = {}, {}, {}, {}
    file_counts = {side: Counter() for side in SIDES}
    anomalies = []
    record_count = 0
    with (evidence / "file_integrity.jsonl").open("w") as details:
        for side in SIDES:
            hand = root / HAND_ROOT / ("handpose_" + side + "_hand")
            counter = Counter()
            frames[side], valid_frames[side] = defaultdict(set), defaultdict(set)
            directories[side] = set()
            if hand.exists():
                for d in hand.rglob("*"):
                    if d.is_dir():
                        seq = sequence_id(d.relative_to(hand).parts)
                        if seq:
                            directories[side].add(seq)
            for path in sorted(hand.rglob("*.pickle")) if hand.exists() else []:
                counter["files"] += 1
                rel = path.relative_to(root).as_posix()
                seq = sequence_id(path.parent.relative_to(hand).parts)
                if seq:
                    file_counts[side][seq] += 1
                index = int(path.stem) if path.stem.isascii() and path.stem.isdigit() else None
                if seq is None or index is None:
                    anomalies.append({"path": rel, "sequence_recognized": seq is not None,
                                      "numeric_filename": index is not None})
                if seq and index is not None:
                    if index in frames[side][seq]:
                        anomalies.append({"path": rel, "duplicate_numeric_frame_index": index})
                        counter["duplicate_frame_indices"] += 1
                    frames[side][seq].add(index)
                stat = path.stat()
                with path.open("rb") as stream:
                    digest = sha_stream(stream)
                item = inspect_record(path)
                after = path.stat()
                unchanged = (stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                if not unchanged:
                    raise RuntimeError("Input changed during read: " + rel)
                item.update(path=rel, side=side, sequence_id=seq, frame_index=index,
                            size=stat.st_size, sha256=digest, source_stat_unchanged=unchanged)
                db.execute("INSERT INTO files VALUES (?,?)", (rel, digest))
                for flag in ("parser_ok", "raw_read_ok", "shape_ok"):
                    counter[flag] += int(item[flag])
                counter["nan_files"] += item["nan"] > 0
                counter["inf_files"] += item["inf"] > 0
                counter["zero_size"] += stat.st_size == 0
                counter["restricted_read_failures"] += not item["raw_read_ok"]
                counter["non_numeric_filenames"] += index is None
                counter["unknown_hierarchy"] += seq is None
                counter["nan_values"] += item["nan"]
                counter["inf_values"] += item["inf"]
                if item["parser_ok"] and seq and index is not None:
                    valid_frames[side][seq].add(index)
                details.write(json.dumps(item, sort_keys=True) + "\n")
                record_count += 1
                if record_count % 10000 == 0:
                    db.commit()
                    print(json.dumps({"pickles_processed": record_count}), flush=True)
            db.commit()
            side_stats[side] = {**dict(counter), "directory_exists": hand.is_dir(),
                               "sequence_directories": len(directories[side]),
                               "sequences_with_files": len(frames[side]),
                               "unique_take_count_operational": len(frames[side]),
                               "unique_camera_count": len({s.split('/')[0] for s in frames[side]}),
                               "cameras": sorted({s.split('/')[0] for s in frames[side]}),
                               "take_definition": "full seven-component Sequence ID; no independent take_uid in this release"}
    rgb_root = root / "extracted/release/HOI4D_release"
    annotation_root = root / "extracted/annotations/HOI4D_annotations"
    videos = {}
    for p in sorted(rgb_root.glob("*/*/*/*/*/*/*/align_rgb/image.mp4")):
        sid = sequence_id(p.parent.parent.relative_to(rgb_root).parts)
        if sid:
            videos[sid] = p
    annotations = {sid for p in annotation_root.glob("*/*/*/*/*/*/*") if p.is_dir()
                   if (sid := sequence_id(p.relative_to(annotation_root).parts))}
    universe = sorted(set(videos) | annotations | set().union(*directories.values()))
    rows, differences = [], []
    for seq in universe:
        right, left = frames['right'].get(seq, set()), frames['left'].get(seq, set())
        meta = video_metadata(videos[seq]) if seq in videos else {"status": "NOT_RUN"}
        count = meta.get("frame_count")
        common = right & left
        row = {"sequence_id": seq, "rgb": seq in videos, "annotation": seq in annotations,
            "right_directory": seq in directories['right'], "left_directory": seq in directories['left'],
            "right_files": len(right), "left_files": len(left), "intersection": len(common),
            "right_total_pickle_files": file_counts['right'][seq],
            "left_total_pickle_files": file_counts['left'][seq],
            "union": len(right | left), "right_only_frames": len(right - left),
            "left_only_frames": len(left - right), "right_parser_valid": len(valid_frames['right'].get(seq, set())),
            "left_parser_valid": len(valid_frames['left'].get(seq, set())),
            "video_status": meta['status'], "frame_count": count, "fps": meta.get('fps'),
            "duration": meta.get('duration'), "right_out_of_range": sum(i >= count for i in right) if count else None,
            "left_out_of_range": sum(i >= count for i in left) if count else None,
            "right_unannotated_unknown": count - sum(i < count for i in right) if count else None,
            "left_unannotated_unknown": count - sum(i < count for i in left) if count else None,
            "union_unannotated_unknown": count - sum(i < count for i in right | left) if count else None}
        rows.append(row)
        for reason, flag in {
            "RGB_WITHOUT_RIGHT": seq in videos and not right,
            "RGB_WITHOUT_LEFT": seq in videos and not left,
            "ANNOTATION_WITHOUT_BOTH": seq in annotations and not right and not left,
            "HAND_WITHOUT_RGB": bool(right or left) and seq not in videos,
            "HAND_WITHOUT_ANNOTATION": bool(right or left) and seq not in annotations,
        }.items():
            if flag:
                differences.append({"sequence_id": seq, "reason": reason})
    columns = list(rows[0]) if rows else ['sequence_id']
    for name in ['sequence_coverage.csv', 'sequence_frame_coverage.csv']:
        write_csv(evidence / name, rows, columns)
    for side in SIDES:
        write_csv(evidence / (side + '_hand_coverage.csv'),
                  [{"sequence_id": s, "files": file_counts[side][s],
                    "unique_numeric_frames": len(frames[side].get(s, set())),
                    "parser_valid_frames": len(valid_frames[side].get(s, set())),
                    "frame_indices": json.dumps(sorted(frames[side].get(s, set())))} for s in universe],
                  ['sequence_id', 'files', 'unique_numeric_frames', 'parser_valid_frames', 'frame_indices'])
    write_csv(evidence / 'missing_sequence_report.csv', differences, ['sequence_id', 'reason'])
    archive = compare_archives(root, db)
    write_json(evidence / 'archive_extraction_comparison.json', archive)
    candidates = sorted(set(universe) - set(FIXED))
    selected = list(FIXED) + random.Random(20261124).sample(candidates, min(10, len(candidates)))
    byseq = {x['sequence_id']: x for x in rows}
    sampled = []
    for seq in selected:
        detail = {"sequence_id": seq, "coverage": byseq.get(seq), "sides": {}}
        paths = resolve_hoi4d_sequence_paths(root, seq)
        for side in SIDES:
            folder = paths.right_hand_dir if side == 'right' else paths.left_hand_dir
            files = sorted(folder.glob('*.pickle'))
            picks = files[:1] + (files[-1:] if len(files) > 1 else [])
            detail['sides'][side] = {'files': len(files), 'directory': folder.relative_to(root).as_posix(),
                'samples': [{'path': p.relative_to(root).as_posix(), **inspect_record(p)} for p in picks]}
        sampled.append(detail)
    write_json(evidence / 'sampled_sequence_validation.json', sampled)
    vr = [x for x in rows if x['video_status'] == 'PASS']
    video_frames = sum(x['frame_count'] for x in vr)
    rs, ls = set(frames['right']), set(frames['left'])
    summary = {'status': 'PARTIAL', 'sequence_universe_scope': 'All locally extracted RGB and annotation sequences plus both hand directory trees; official full-release completeness not known.',
        'sides': side_stats, 'sequence_universe': len(universe), 'rgb_sequences': len(videos),
        'annotation_sequences': len(annotations), 'both_hand_sequences': len(rs & ls),
        'right_only_sequences': len(rs - ls), 'left_only_sequences': len(ls - rs),
        'both_missing_in_local_universe': len(set(universe) - rs - ls),
        'unique_take_definition': 'Sequence ID itself; independent capture/take identity UNKNOWN',
        'video_metadata_success': len(vr), 'video_metadata_failures': len(videos) - len(vr),
        'video_frame_denominator': video_frames,
        'right_in_range_video_frames': sum(x['right_files'] - x['right_out_of_range'] for x in vr),
        'left_in_range_video_frames': sum(x['left_files'] - x['left_out_of_range'] for x in vr),
        'union_in_range_video_frames': sum(x['frame_count'] - x['union_unannotated_unknown'] for x in vr),
        'out_of_range_indices': sum(x['right_out_of_range'] + x['left_out_of_range'] for x in vr),
        'unknown_missing_hand_frame_slots': sum(x['right_unannotated_unknown'] + x['left_unannotated_unknown'] for x in vr),
        'anomalous_paths': anomalies, 'difference_reasons': dict(Counter(x['reason'] for x in differences)),
        'extraction_integrity': archive['extraction_integrity'], 'download_integrity': 'NOT_VERIFIABLE',
        'numeric_semantics': 'No physical range thresholds asserted; per-file finite minima/maxima recorded for review. Units, frame sensor timing, joints and pose convention UNKNOWN.',
        'parser_scope': 'Production _read_mano_record plus restricted diagnostic unpickler; no arbitrary pickle.load.',
        'manifest_source_checks': 'SHA256 of every pickle at read; local archive member bytes compared where available; unchanged size/mtime during each read.'}
    write_json(evidence / 'handpose_inventory.json', summary)
    write_json(evidence / 'mano_schema_integrity.json', {'sides': side_stats, 'expected_shapes': SHAPES,
        'per_file_details': 'evidence/file_integrity.jsonl', 'coordinate_semantics': 'UNKNOWN',
        'finite_does_not_mean_physical_gt': True})
    with (output / 'source_SHA256SUMS.txt').open('w') as f:
        for path, digest in db.execute('SELECT path,digest FROM files ORDER BY path'):
            f.write(digest + '  ' + path + '\n')
    db.close()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.root, args.output_dir), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
