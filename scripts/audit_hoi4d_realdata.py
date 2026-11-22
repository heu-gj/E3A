"""H2 independent real-source integrity audit; never mutates the dataset.

All commands must be run with the repository .venv Python. No model is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from e3a.data.hoi4d_dataset import HOI4DDataset, resolve_hoi4d_sequence_paths
from e3a.data.hoi4d_dataset import _read_video_frames
from e3a.data.hoi4d import build_hoi4d_anticipation_target, decode_hoi4d_motion_mask

BASE = '155820d65c2f8e47101d5aa204d0fcea79cb6776'
SEQ = 'ZY20210800001/H1/C20/N30/S295/s04/T4'
OUT = Path('reports/audits/H2-HOI4D-RealData-Integrity-Audit')
REPO = Path(__file__).resolve().parents[1]


def save(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def require_venv():
    if Path(sys.prefix).resolve() != (REPO / '.venv').resolve():
        raise RuntimeError('Use the repository .venv; no alternate Python environment allowed.')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sanitize(text, root):
    return text.replace(str(root), '<dataset-root>').replace(str(REPO), '<repository-root>').replace(str(sys.executable), 'python')


def smoke(root):
    command = [sys.executable, '-u', 'scripts/smoke_test_hoi4d_dataset.py',
               '--height', '540', '--width', '960', '--num-workers', '0']
    started = time.monotonic()
    p = subprocess.run(command, text=True, capture_output=True,
                       env={**os.environ, 'HOI4D_ROOT': str(root), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    stdout, stderr = sanitize(p.stdout, root), sanitize(p.stderr, root)
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    (OUT / 'logs/smoke_stdout.txt').write_text(stdout)
    (OUT / 'logs/smoke_stderr.txt').write_text(stderr)
    result = {'status': 'PASS' if p.returncode == 0 else 'BLOCKED', 'exit_code': p.returncode,
              'command': ['python', *command[1:]], 'elapsed_seconds': time.monotonic() - started,
              'execution_timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'summary': json.loads(stdout) if p.returncode == 0 else None,
              'scope': 'Real H1 DataLoader full scan. Not independent reference or physical GT validation.'}
    save('logs/smoke_test.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if p.returncode:
        raise RuntimeError('H1 smoke failed; see sanitized logs')


def environment():
    p = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
    save('logs/environment.json', {'python': platform.python_version(), 'environment': '.venv',
        'packages': {x: importlib.metadata.version(x) for x in ['e3a', 'numpy', 'torch', 'torchvision', 'opencv-python', 'Pillow']},
        'base_commit': BASE, 'initial_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'client_date': '2026-11-22', 'client_timezone': 'Asia/Singapore',
        'host_execution_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'timestamp_note': 'Client date is task calendar date; host UTC clock is recorded separately and may differ.',
        'pip_check_exit_code': p.returncode, 'pip_check_stdout': p.stdout.strip(),
        'pip_cache_warning': bool(p.stderr), 'torch_cuda_build': torch.version.cuda,
        'model_forward_backward': 'NOT_RUN'})


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metrics(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape:
        raise ValueError('Metric shapes differ')
    error = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return {'elements': int(error.size), 'different_elements': int(np.count_nonzero(error)),
            'max_absolute_error': float(error.max(initial=0)), 'mean_absolute_error': float(error.mean())}


def independent_mask(rgb):
    """Fixed C20 reference from pinned official table, not production decoder."""
    sem = np.full(rgb.shape[:2], -1, np.int16)
    inst = sem.copy()
    # palette0 background; 1 chair; 2 right hand; 3 left hand. Index4 stays UNKNOWN.
    for color, semantic, instance in [((0, 0, 0), 0, 0), ((128, 0, 0), 2, 1),
                                      ((0, 128, 0), 47, 2), ((128, 128, 0), 47, 3)]:
        selected = np.all(rgb == color, axis=-1)
        sem[selected], inst[selected] = semantic, instance
    return sem, inst, inst >= 0


def conservative_half_validity(source_valid):
    """Independent exact 2x downsampling rule: all four source pixels valid."""
    h, w = source_valid.shape
    if h % 2 or w % 2:
        raise ValueError('This H2 reference only supports exact 2x resize')
    return source_valid.reshape(h // 2, 2, w // 2, 2).all(axis=(1, 3))


def numeric_pose_index(filename_index, raw):
    return {'file_index': int(filename_index), 'raw_frameId': raw.get('frameId'),
            'isEffective': raw.get('isEffective'), 'dataList_count': len(raw.get('dataList', [])),
            'objects': [{'id': obj.get('id'), 'label': obj.get('label'),
                         'box_field_exists': '2dBox' in obj,
                         'camera1_box_nonempty': bool(obj.get('2dBox', {}).get('camera1', []))}
                        for obj in raw.get('dataList', [])]}


def strict_future(events, cutoff):
    candidates = [e for e in events if float(e['startTime']) > cutoff]
    return min(candidates, key=lambda e: (float(e['startTime']), float(e['endTime']))) if candidates else None


def action_check(events, cutoff, target):
    future = strict_future(events, cutoff)
    expected = float(future['startTime']) - cutoff if future else 0.
    return {'cutoff': cutoff, 'expected_next_event': future['event'] if future else None,
            'expected_onset': float(future['startTime']) if future else None,
            'expected_time_to_event': expected, 'actual_time_to_event': target['time_to_event'],
            'actual_time_valid': target['time_valid'], 'actual_action_valid': target['action_valid'],
            'actual_target_entity_valid': target['target_entity_valid'],
            'pass': (bool(target['time_valid']) == (future is not None)
                     and abs(target['time_to_event'] - expected) < 1e-7
                     and not target['action_valid'] and not target['target_entity_valid'])}


class SourceNumpyUnpickler(pickle.Unpickler):
    """Reference loader permits NumPy reconstruction only, no arbitrary imports."""
    def find_class(self, module, name):
        if (module, name) in {('numpy.core.multiarray', '_reconstruct'),
                              ('numpy._core.multiarray', '_reconstruct')}:
            return np.core.multiarray._reconstruct
        if module == 'numpy' and name in {'ndarray', 'dtype'}:
            return getattr(np, name)
        raise pickle.UnpicklingError('Non-NumPy global in source annotation')


def source_hand(path):
    with path.open('rb') as f:
        return SourceNumpyUnpickler(f, encoding='latin1').load()


def reference(root):
    """Separate sequential passes; persisted resized reference stays local-only."""
    paths = resolve_hoi4d_sequence_paths(root, SEQ)
    local = OUT / 'evidence/local_only'
    local.mkdir(parents=True, exist_ok=True)
    videos = {}
    video_rows = {'rgb': [], 'depth': []}
    for name, path in [('rgb', paths.rgb_video), ('depth', paths.depth_video)]:
        cap = cv2.VideoCapture(str(path))
        if name == 'depth':
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        meta = {'source_path': str(path.relative_to(root)), 'fps': cap.get(5),
                'width': int(cap.get(3)), 'height': int(cap.get(4)), 'container_frames': int(cap.get(7)),
                'source_sha256': sha(path), 'codec_fourcc': int(cap.get(6))}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if name == 'rgb':
                converted = frame[..., ::-1].copy()
                resized = cv2.resize(converted, (960, 540), interpolation=cv2.INTER_AREA)
                row = {'frame': idx, 'source_dtype': str(frame.dtype), 'raw_bgr_sha256': array_hash(frame),
                       'raw_rgb_sha256': array_hash(converted), 'resized_rgb_sha256': array_hash(resized)}
            else:
                if frame.ndim != 2 or frame.dtype != np.uint16:
                    raise ValueError('Depth reference is not single-channel uint16')
                resized = frame[::2, ::2].copy()  # exact half-size nearest, independent of cv2 resize
                row = {'frame': idx, 'source_dtype': str(frame.dtype), 'raw_uint16_sha256': array_hash(frame),
                       'resized_uint16_sha256': array_hash(resized), 'source_min': int(frame.min()),
                       'source_max': int(frame.max()), 'source_invalid_fraction': float(np.mean(frame == 0)),
                       'resized_invalid_fraction': float(np.mean(resized == 0))}
            row.update(pts_seconds=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.,
                       reported_next_frame=cap.get(cv2.CAP_PROP_POS_FRAMES))
            np.savez_compressed(local / f'reference_{name}_{idx:05d}.npz', values=resized)
            video_rows[name].append(row)
            idx += 1
        cap.release()
        meta['decoded_frames'] = idx
        videos[name] = meta
        print('reference video', name, idx, flush=True)
    if any(v['decoded_frames'] != 300 for v in videos.values()):
        raise ValueError('Expected 300 independently decoded frames')
    raw_action = json.loads(paths.action_file.read_text())
    events = raw_action['events']
    boundaries = sorted({max(0, min(299, f)) for e in events for t in [e['startTime'], e['endTime']]
                         for f in [math.floor(float(t) * videos['rgb']['fps']), math.ceil(float(t) * videos['rgb']['fps'])]})
    vis_frames = sorted(set([0, 75, 150, 220, 224, 299] + boundaries))
    masks, poses, hands = [], [], {'right': [], 'left': []}
    for idx in range(300):
        mask_path = paths.mask_dir / f'{idx:05d}.png'
        with Image.open(mask_path) as image:
            mode = image.mode
            mask = np.asarray(image.convert('RGB'))
        sem, inst, valid = independent_mask(mask)
        target_valid = conservative_half_validity(valid)
        entity = (inst[::2, ::2] == 1) & target_valid
        packed = mask[..., 0].astype(np.uint32) * 65536 + mask[..., 1].astype(np.uint32) * 256 + mask[..., 2]
        colors, counts = np.unique(packed, return_counts=True)
        row = {'frame': idx, 'source_path': str(mask_path.relative_to(root)), 'source_sha256': sha(mask_path),
               'source_mode': mode, 'source_shape': list(mask.shape),
               'colors': [{'rgb': [int(c >> 16), int((c >> 8) & 255), int(c & 255)], 'pixels': int(n)} for c, n in zip(colors, counts)],
               'semantic_pixels': {str(k): int(np.sum(sem == k)) for k in np.unique(sem)},
               'instance_pixels': {str(k): int(np.sum(inst == k)) for k in np.unique(inst)},
               'source_unknown_pixels': int((~valid).sum()), 'downsample_invalid_pixels': int((~target_valid).sum()),
               'downsample_known_chair_pixels': int(entity.sum())}
        np.savez_compressed(local / f'reference_mask_{idx:05d}.npz', pixel_valid=target_valid, entity=entity)
        masks.append(row)
        pp = paths.objpose_dir / f'{idx}.json'
        raw = json.loads(pp.read_text())
        poses.append(numeric_pose_index(idx, raw) | {'source_path': str(pp.relative_to(root)), 'source_sha256': sha(pp)})
        for side, folder in [('right', paths.right_hand_dir), ('left', paths.left_hand_dir)]:
            file = folder / f'{idx}.pickle'
            if not file.is_file():
                continue
            hand = source_hand(file)
            hands[side].append({'frame': idx, 'source_path': str(file.relative_to(root)), 'source_sha256': sha(file),
                'fields': {key: {'shape': list(np.asarray(value).shape), 'dtype': str(np.asarray(value).dtype),
                                 'finite': bool(np.isfinite(value).all()), 'array_sha256': array_hash(value)}
                           for key, value in hand.items()}})
        if idx % 50 == 0:
            print('reference masks/pose/hand', idx, flush=True)
    save('evidence/local_only/reference_index.json', {'videos': videos, 'video_frames': video_rows,
         'masks': masks, 'poses': poses, 'hands': hands, 'visualization_frames': vis_frames,
         'raw_action': raw_action, 'source_intrinsics': np.load(paths.intrinsics_file).tolist()})
    print('reference done', flush=True)


def visualization(idx, source_rgb, depth, raw_mask, valid, hand_records, events, pts):
    """Source-res six-panel local diagnostic. No unverified objpose projection."""
    _, inst, _ = independent_mask(raw_mask)
    overlay = source_rgb.copy()
    fg = inst > 0
    overlay[fg] = (source_rgb[fg] * .55 + raw_mask[fg] * .45).astype(np.uint8)
    unknown = source_rgb.copy()
    unknown[~valid] = (255, 255, 0)
    # Enlarge the display indicator only in a separate panel; preserve pixel counts in evidence.
    marker = cv2.dilate((~valid).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    unknown[marker] = (255, 255, 0)
    positive = depth[depth > 0]
    ceiling = np.percentile(positive, 99) if positive.size else 1
    scaled = np.clip(depth.astype(float) * 255 / max(ceiling, 1), 0, 255).astype(np.uint8)
    display_depth = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    joints = source_rgb.copy()
    for side, hand in hand_records.items():
        for joint, xy in enumerate(np.asarray(hand['kps2D'])):
            x, y = map(int, xy)
            cv2.circle(joints, (x, y), 6, (255, 0, 255) if side == 'right' else (0, 255, 255), -1)
            cv2.putText(joints, str(joint), (x, y), 0, .4, (255, 255, 255), 1)
    panels = [('RGB source', source_rgb), ('RGB + provisional instance (diagnostic)', overlay),
              ('UNKNOWN yellow: marker dilated for display', unknown),
              ('Depth display normalized; units UNKNOWN', display_depth),
              ('MANO raw2D: coordinate semantics unverified', joints)]
    canvas = Image.new('RGB', (1440, 820), 'white')
    draw = ImageDraw.Draw(canvas)
    for p, (label, values) in enumerate(panels):
        x, y = p % 3 * 480, p // 3 * 350
        canvas.paste(Image.fromarray(values).resize((480, 270)), (x, y + 35))
        draw.text((x + 5, y + 5), label, fill='black')
    active = [f"{e['event']}\n[{e['startTime']:.6f},{e['endTime']:.6f})" for e in events
              if e['startTime'] <= pts < e['endTime']]
    text = f'Frame {idx}\nVideo PTS {pts:.9f}s\nActive:\n' + ('\n'.join(active) or 'no active interval')
    future = strict_future(events, pts)
    text += '\nNext strict onset: ' + (f"{future['event']} at {future['startTime']:.6f}s" if future else 'NONE')
    text += '\nDiagnostic overlay only.\nNot calibration/contact/physical GT.\nNo objpose projection.'
    draw.multiline_text((965, 390), text, fill='black', spacing=8)
    folder = OUT / 'visualizations/local_only'
    folder.mkdir(parents=True, exist_ok=True)
    canvas.save(folder / f'frame_{idx:05d}_panels.jpg')
    for label, arr in [('rgb_source', source_rgb), ('instance_overlay', overlay), ('unknown_display', unknown),
                       ('depth_display', display_depth), ('mano_raw2d', joints)]:
        Image.fromarray(arr).save(folder / f'frame_{idx:05d}_{label}.jpg')


def audit(root):
    ref = json.loads((OUT / 'evidence/local_only/reference_index.json').read_text())
    paths = resolve_hoi4d_sequence_paths(root, SEQ)
    dataset = HOI4DDataset(root, SEQ, clip_length=8, clip_stride=8, image_size=(540, 960))
    rng = np.random.default_rng(220)
    order = rng.permutation(len(dataset)).tolist()
    rgb_rows, depth_rows, mask_rows, pose_rows, hand_rows, actions, intrinsics, frames = [], [], [], [], [], [], [], []
    seen, rendered = set(), set()
    mismatches = []
    K = np.asarray(ref['source_intrinsics'], dtype=np.float64)
    expected_K = K.copy()
    expected_K[0] *= .5
    expected_K[1] *= .5
    start_time = time.monotonic()
    for count, sample_index in enumerate(order):
        sample = dataset[sample_index]
        indices = sample['frame_indices'].tolist()
        # Actual production raw seek helper is checked against independent sequential hashes.
        raw_rgb = _read_video_frames(paths.rgb_video, indices)
        raw_depth = _read_video_frames(paths.depth_video, indices, depth=True)
        for position, idx in enumerate(indices):
            # End-aligned H1 clip overlaps four frames: compare duplicate observations too.
            with np.load(OUT / f'evidence/local_only/reference_rgb_{idx:05d}.npz') as f:
                resized_rgb = f['values']
            with np.load(OUT / f'evidence/local_only/reference_depth_{idx:05d}.npz') as f:
                resized_depth = f['values']
            with np.load(OUT / f'evidence/local_only/reference_mask_{idx:05d}.npz') as f:
                target_valid, target_entity = f['pixel_valid'], f['entity']
            actual_rgb = sample['images'][position].permute(1, 2, 0).numpy()
            actual_depth = sample['depth'][position, 0].numpy()
            rgb_m = metrics(resized_rgb.astype(np.float32) / np.float32(255), actual_rgb)
            depth_m = metrics(resized_depth, actual_depth)
            rgb_raw_equal = array_hash(raw_rgb[position]) == ref['video_frames']['rgb'][idx]['raw_rgb_sha256']
            depth_raw_equal = array_hash(raw_depth[position]) == ref['video_frames']['depth'][idx]['raw_uint16_sha256']
            entity = sample['entity_targets']
            actual_entity = (entity['masks'][:, position] > 0).any(dim=0).numpy()
            valid_equal = np.array_equal(target_valid, entity['pixel_valid'][position].numpy())
            entity_equal = np.array_equal(target_entity, actual_entity)
            pp = json.loads((paths.objpose_dir / f'{idx}.json').read_text())
            pose_equal = pp == sample['objpose_records'][position]
            frameid_equal = pp['frameId'] == int(sample['objpose_frame_ids'][position])
            frame_m = {'file_index': idx, 'sample_index': sample_index,
                       'actual_timestamp': float(sample['timestamps'][position]),
                       'sequential_pts': ref['video_frames']['rgb'][idx]['pts_seconds'],
                       'index_div_fps': idx / ref['videos']['rgb']['fps'],
                       'raw_frameId': pp['frameId'], 'pose_record_equal': pose_equal}
            frames.append(frame_m)
            rgb_rows.append({'frame': idx, 'sample_index': sample_index, 'resized_normalized_metrics': rgb_m,
                             'raw_seek_matches_sequential_rgb': rgb_raw_equal})
            depth_rows.append({'frame': idx, 'sample_index': sample_index, 'resized_raw_value_metrics': depth_m,
                               'raw_seek_matches_sequential_uint16': depth_raw_equal,
                               'dataset_dtype': str(actual_depth.dtype),
                               'uint16_values_exact_after_float32_cast': np.array_equal(actual_depth, resized_depth.astype(np.float32)),
                               'depth_valid_equal': np.array_equal(sample['depth_valid'][position, 0].numpy(), resized_depth > 0)})
            mask_rows.append({'frame': idx, 'sample_index': sample_index, 'pixel_valid_equal': valid_equal,
                              'entity_mask_equal': entity_equal, 'dataset_invalid_pixels': int((~entity['pixel_valid'][position]).sum()),
                              'unknown_pixels_excluded_from_entity': not bool(actual_entity[~target_valid].any()),
                              'visible_track_ids': entity['ids'][:, position].tolist(),
                              'track_labels': entity['labels'].tolist(),
                              'entity_supervision_valid': sample['entity_supervision_valid']})
            pose_rows.append({'file_index': idx, 'sample_index': sample_index,
                              'raw_json_equal': pose_equal, 'raw_frameId_preserved': frameid_equal,
                              'record_present': bool(sample['objpose_record_present'][position]),
                              'video_index_equals_requested_filename': int(sample['frame_indices'][position]) == idx})
            for side_index, (side, folder) in enumerate([('right', paths.right_hand_dir), ('left', paths.left_hand_dir)]):
                hp = folder / f'{idx}.pickle'
                present = hp.is_file()
                actual_valid = bool(sample['mano_parameters']['valid'][position, side_index])
                checks = {}
                if present:
                    source = source_hand(hp)
                    for src, dst in [('poseCoeff', 'pose_coefficients'), ('beta', 'betas'), ('trans', 'translations'), ('kps2D', 'keypoints_2d')]:
                        expected = np.asarray(source[src], dtype=np.float32).copy()
                        if src == 'kps2D':
                            expected *= .5
                        checks[src] = metrics(expected, sample['mano_parameters'][dst][position, side_index].numpy())
                else:
                    for dst in ['pose_coefficients', 'betas', 'translations', 'keypoints_2d']:
                        checks[dst] = {'placeholder_nonzero_elements': int(torch.count_nonzero(sample['mano_parameters'][dst][position, side_index]))}
                hand_rows.append({'frame': idx, 'side': side, 'sample_index': sample_index,
                                  'source_present': present, 'dataset_valid': actual_valid, 'checks': checks})
            if not (rgb_raw_equal and depth_raw_equal and valid_equal and entity_equal and pose_equal and frameid_equal
                    and rgb_m['different_elements'] == 0 and depth_m['different_elements'] == 0):
                mismatches.append({'sample_index': sample_index, 'frame': idx})
            if idx in ref['visualization_frames'] and idx not in rendered:
                with Image.open(paths.mask_dir / f'{idx:05d}.png') as image:
                    raw_mask = np.asarray(image.convert('RGB'))
                _, _, validity = independent_mask(raw_mask)
                hand = {side: source_hand(folder / f'{idx}.pickle') for side, folder in
                        [('right', paths.right_hand_dir), ('left', paths.left_hand_dir)] if (folder / f'{idx}.pickle').is_file()}
                visualization(idx, raw_rgb[position], raw_depth[position], raw_mask, validity, hand,
                              ref['raw_action']['events'], ref['video_frames']['rgb'][idx]['pts_seconds'])
                rendered.add(idx)
            seen.add(idx)
        cut = float(sample['timestamps'][-1])
        source_intervals = [(e['event'], float(e['startTime']), float(e['endTime'])) for e in ref['raw_action']['events']]
        dataset_intervals = [(e['event'], e['start_time'], e['end_time']) for e in sample['action_segments']]
        actions.append(action_check(ref['raw_action']['events'], cut, sample['anticipation_targets']) |
                       {'sample_index': sample_index, 'source_intervals_preserved': source_intervals == dataset_intervals})
        intrinsics.append({'sample_index': sample_index, 'valid': sample['intrinsics_valid'],
                           'metrics': metrics(expected_K.astype(np.float32), sample['intrinsics'].numpy())})
        print('random Dataset clip', count + 1, '/', len(order), 'source indices', indices[0], indices[-1], flush=True)
    if seen != set(range(300)):
        raise ValueError('Independent Dataset comparison did not cover every frame')
    elapsed = time.monotonic() - start_time
    events = ref['raw_action']['events']
    boundary = []
    segments = [{'event': e['event'], 'start_time': float(e['startTime']), 'end_time': float(e['endTime']), 'category': 'C20'} for e in events]
    for cutoff in sorted({0., 20., *[max(0., float(e['startTime']) + eps) for e in events for eps in [-1e-8, 0., 1e-8]]}):
        actual = build_hoi4d_anticipation_target(segments, cutoff, {})
        boundary.append(action_check(events, cutoff, actual))
    order_spec = {'shuffle_seed': 220, 'clip_sample_order': order, 'observations_including_overlap': len(frames),
                  'unique_frames': len(seen), 'overlap_note': '38 clips x8=304 observations; final end-aligned clip292..299 duplicates292..295',
                  'elapsed_seconds': elapsed}
    save('evidence/rgb_depth_integrity.json', {'status': 'PASS' if not mismatches else 'PARTIAL',
        'scope': 'Digital decoding/value consistency, not physical calibration. Same OpenCV backend for reference/reader.',
        'reference_method': 'Independent sequential VideoCapture read to EOF, no seek. RGB BGR reversal +INTER_AREA; depth uint16 with convertRGB=0, independent stride2 downsample.',
        'randomization': order_spec, 'videos': ref['videos'], 'source_frames': ref['video_frames'],
        'rgb_comparisons': rgb_rows, 'depth_comparisons': depth_rows, 'mismatches': mismatches,
        'depth_dtype_contract': 'Source uint16, Dataset float32 retaining raw integer values exactly; Dataset output is NOT uint16.',
        'sensor_frame_alignment': 'UNKNOWN; equal metadata or decoding is not physical registration'})
    save('evidence/mask_integrity.json', {'status': 'PASS' if all(r['pixel_valid_equal'] and r['entity_mask_equal'] for r in mask_rows) else 'PARTIAL',
        'scope': 'All300 real source images vs independently hardcoded pinned C20 mapping; local instance IDs remain provisional',
        'official_mapping_commit': '788b43552176c77c01c3e98dd5f101fd6a94ee7a',
        'official_source': 'https://github.com/hoi4d/HOI4D_Script/blob/788b43552176c77c01c3e98dd5f101fd6a94ee7a/prepare_4Dseg/utils.py',
        'source_frames': ref['masks'], 'dataset_comparisons': mask_rows, 'frame220': ref['masks'][220],
        'pixel_loss_consumption': 'NOT_RUN: pixel_valid exclusion verified in Dataset only, no claim that an arbitrary downstream loss consumes it',
        'unknown_semantics': 'UNKNOWN, no index4 semantic inference'})
    save('evidence/objpose_integrity.json', {'status': 'PASS' if all(r['raw_json_equal'] and r['raw_frameId_preserved'] for r in pose_rows) else 'PARTIAL',
        'scope': 'Source filename and raw field preservation, not physical GT or sensor-time association',
        'files_read': len(ref['poses']), 'missing_or_corrupt': [], 'records': ref['poses'], 'dataset_comparisons': pose_rows,
        'coordinates': 'UNKNOWN', 'physical_units': 'UNKNOWN', 'transform_direction': 'UNKNOWN',
        'entity_slot_association': 'NOT_RUN; no numeric ID join'})
    mano_ok = all(r['source_present'] == r['dataset_valid'] and
                  all(v.get('different_elements', v.get('placeholder_nonzero_elements', 0)) == 0 for v in r['checks'].values())
                  for r in hand_rows)
    mano_ok = mano_ok and all(v['finite'] for rows in ref['hands'].values() for r in rows for v in r['fields'].values())
    save('evidence/mano_integrity.json', {'status': 'PASS' if mano_ok else 'PARTIAL', 'scope': 'Raw parameter fidelity and resize arithmetic only; no MANO forward or3D joints',
        'hand_order': ['right', 'left'], 'source_hands': ref['hands'], 'coverage': {side: {'valid_frames': len(rows),
            'missing_frame_indices': sorted(set(range(300)) - {r['frame'] for r in rows})} for side, rows in ref['hands'].items()},
        'dataset_comparisons': hand_rows, 'kps2D_resize_scale': [.5, .5],
        'coordinate_semantics': 'UNKNOWN; arithmetic consistency is not physical joint projection validation'})
    save('evidence/action_integrity.json', {'status': 'PASS' if all(r['pass'] for r in actions + boundary) else 'PARTIAL',
        'source_path': str(paths.action_file.relative_to(root)), 'source_sha256': sha(paths.action_file),
        'video_duration': ref['videos']['rgb']['decoded_frames'] / ref['videos']['rgb']['fps'],
        'declared_duration': ref['raw_action']['info']['duration'],
        'events': [{'event': e['event'], 'start_time': e['startTime'], 'end_time': e['endTime']} for e in events],
        'intervals_within_video': all(0 <= e['startTime'] < e['endTime'] <= 20 for e in events),
        'source_order_monotonic_start': all(a['startTime'] <= b['startTime'] for a, b in zip(events, events[1:])),
        'overlaps': [{'previous_event_index': i, 'next_event_index': i + 1, 'seconds': a['endTime'] - b['startTime']}
                     for i, (a, b) in enumerate(zip(events, events[1:])) if a['endTime'] > b['startTime']],
        'clip_checks': actions, 'boundary_checks': boundary, 'time_valid_clips': sum(r['actual_time_valid'] for r in actions),
        'action_valid_clips': sum(r['actual_action_valid'] for r in actions), 'vocabulary': 'NONE; no training vocabulary generated',
        'cutoff_note': 'Dataset uses float32 video timestamps. Boundary tests separately use source double starts ±1e-8.',
        'target_entity_valid': False})
    save('evidence/intrinsics_integrity.json', {'status': 'PASS' if all(r['metrics']['different_elements'] == 0 for r in intrinsics) else 'PARTIAL',
        'source_path': str(paths.intrinsics_file.relative_to(root)), 'source_sha256': sha(paths.intrinsics_file),
        'original_K': K.tolist(), 'scale_x': .5, 'scale_y': .5, 'expected_K': expected_K.tolist(),
        'dataset_comparisons': intrinsics, 'physical_calibration': 'NOT_RUN; scaling only'})
    with (OUT / 'evidence/frame_alignment.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(frames[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(frames)
    save('evidence/audit_execution.json', {'execution_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'randomization': order_spec, 'visualization_frames': sorted(rendered),
        'initial_required_visual_frames': [0, 75, 150, 220, 224, 299],
        'action_boundary_selection': 'floor/ceil(startTime*FPS and endTime*FPS), clamped to0..299',
        'production_code_modified': False, 'new_dependencies': 'pytest only, explicitly approved by user',
        'mismatches': mismatches, 'model_forward_backward': 'NOT_RUN'})
    print('H2 completed; reference mismatches', len(mismatches), 'visual frames', len(rendered), flush=True)


def load(name):
    return json.loads((OUT / name).read_text())


def visualize(root):
    ref = load('evidence/local_only/reference_index.json')
    paths = resolve_hoi4d_sequence_paths(root, SEQ)
    rgb = cv2.VideoCapture(str(paths.rgb_video))
    depth = cv2.VideoCapture(str(paths.depth_video))
    depth.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    positions = set(ref['visualization_frames'])
    for idx in range(300):
        ok_rgb, bgr = rgb.read()
        ok_depth, raw_depth = depth.read()
        if not (ok_rgb and ok_depth):
            raise ValueError('Sequential visualization source ended early')
        if idx not in positions:
            continue
        with Image.open(paths.mask_dir / f'{idx:05d}.png') as image:
            mask = np.asarray(image.convert('RGB'))
        _, _, valid = independent_mask(mask)
        hands = {side: source_hand(folder / f'{idx}.pickle') for side, folder in
                 [('right', paths.right_hand_dir), ('left', paths.left_hand_dir)] if (folder / f'{idx}.pickle').is_file()}
        visualization(idx, bgr[..., ::-1].copy(), raw_depth, mask, valid, hands, ref['raw_action']['events'],
                      ref['video_frames']['rgb'][idx]['pts_seconds'])
    rgb.release()
    depth.release()
    print('Rendered', len(positions), 'source positions; six files per position, local-only', flush=True)


def report(root):
    modalities = {name: load(f'evidence/{name}_integrity.json')
                  for name in ['rgb_depth', 'mask', 'objpose', 'mano', 'action', 'intrinsics']}
    rgb, mask, pose, hand, action, K = [modalities[k] for k in ['rgb_depth', 'mask', 'objpose', 'mano', 'action', 'intrinsics']]
    for row in mask['source_frames']:
        for color in row['colors']:
            value = sum(((channel >> (7 - bit)) & 1) << (3 * bit + c)
                        for bit in range(8) for c, channel in enumerate(color['rgb']))
            color['global_palette_index'] = value if value < 256 else None
            color['defined_in_C20'] = value <= 3
    color_checks = []
    for values in sorted({tuple(c['rgb']) for row in mask['source_frames'] for c in row['colors']}):
        rgb_color = np.asarray(values, dtype=np.uint8).reshape(1, 1, 3)
        expected_sem, expected_inst, expected_valid = independent_mask(rgb_color)
        try:
            sem, inst = decode_hoi4d_motion_mask(rgb_color, 'C20')
            color_checks.append({'rgb': list(values), 'expected_semantic_id': int(expected_sem[0, 0]),
                'expected_instance_id': int(expected_inst[0, 0]), 'decoder_rejected': False,
                'pass': bool(expected_valid[0, 0] and np.array_equal(sem, expected_sem) and np.array_equal(inst, expected_inst))})
        except ValueError:
            color_checks.append({'rgb': list(values), 'expected_semantic_id': -1, 'expected_instance_id': -1,
                                 'decoder_rejected': True, 'pass': not bool(expected_valid[0, 0])})
    mask['all_observed_color_decoder_checks'] = color_checks
    mask['decoder_color_check_scope'] = 'Each distinct RGB actually present across300 source masks vs independent pinned mapping; whole-image entity/validity checked separately.'
    # frame220 was serialized as a separate copy; preserve the enriched version.
    mask['frame220'] = next(row for row in mask['source_frames'] if row['frame'] == 220)
    save('evidence/mask_integrity.json', mask)
    with (OUT / 'evidence/frame_alignment.csv').open() as f:
        frame_rows = list(csv.DictReader(f))
    timestamp_error = max(abs(float(r['actual_timestamp']) - float(r['index_div_fps'])) for r in frame_rows)
    # Separate initialization-level observation; no extra image scan required.
    dataset = HOI4DDataset(root, SEQ, clip_length=8, clip_stride=8, image_size=(540, 960))
    raw = json.loads(resolve_hoi4d_sequence_paths(root, SEQ).action_file.read_text())
    actual_segments = dataset._sequences[0]['segments']
    same_intervals = ([(e['event'], float(e['startTime']), float(e['endTime'])) for e in raw['events']]
                      == [(e['event'], e['start_time'], e['end_time']) for e in actual_segments])
    action['initialization_interval_preservation'] = {'status': 'PASS' if same_intervals else 'PARTIAL',
        'source_intervals_equal': same_intervals,
        'scope': 'Separate actual Dataset initialization; raw event label/start/end versus loaded segments. No vocabulary generated.'}
    save('evidence/action_integrity.json', action)
    checks = {
        'rgb_zero_value_mismatches': all(x['resized_normalized_metrics']['different_elements'] == 0 and x['raw_seek_matches_sequential_rgb'] for x in rgb['rgb_comparisons']),
        'depth_zero_value_mismatches': all(x['resized_raw_value_metrics']['different_elements'] == 0 and x['raw_seek_matches_sequential_uint16'] and x['depth_valid_equal'] for x in rgb['depth_comparisons']),
        'pixel_valid_and_entity_equal': all(x['pixel_valid_equal'] and x['entity_mask_equal'] and x['unknown_pixels_excluded_from_entity'] for x in mask['dataset_comparisons']),
        'semantic_and_instance_color_mapping': all(row['pass'] for row in color_checks),
        'pose_fidelity': all(x['raw_json_equal'] and x['raw_frameId_preserved'] and x['record_present'] for x in pose['dataset_comparisons']),
        'mano_fidelity': hand['status'] == 'PASS',
        'action_time_fidelity': all(x['pass'] for x in action['clip_checks'] + action['boundary_checks']) and same_intervals,
        'intrinsics_fidelity': all(x['metrics']['different_elements'] == 0 for x in K['dataset_comparisons']),
        'timestamps_float32_tolerance': timestamp_error <= 2e-6,
    }
    integration_status = 'PASS' if all(checks.values()) else 'PARTIAL'
    for name, data in modalities.items():
        assert data['status'] in {'PASS', 'PARTIAL', 'BLOCKED', 'NEEDS_REVIEW', 'NOT_RUN'}
    result = {'task_id': 'H2-HOI4D-RealData-Integrity-Audit', 'task_status': 'PARTIAL',
        'digital_integrity_status': integration_status,
        'status_rationale': 'Digital reader consistency measured independently; physical units/geometry/contact/target/persistent identity remain unverified and invalid.',
        'base_commit': BASE, 'branch': 'audit/h2-hoi4d-realdata-integrity', 'sequence_id': SEQ,
        'dataset': {'root': '.', 'release_revision': 'UNKNOWN; same extracted layout as P1'},
        'client_execution_date': '2026-11-22', 'checks': checks,
        'audit_results': {name: {'status': data['status'], 'evidence_level': 'REAL_SOURCE_FULL_SEQUENCE_COMPARISON',
            'evidence_files': [f'evidence/{name}_integrity.json'],
            'scope': data.get('scope', 'Raw field/value/temporal arithmetic only, no physical GT certification')}
            for name, data in modalities.items()},
        'unknowns': {'sensor_timestamps': 'UNKNOWN', 'rgb_depth_physical_registration': 'NEEDS_REVIEW',
            'depth_physical_units': 'UNKNOWN', 'object_pose_coordinates_units_transform_direction': 'UNKNOWN',
            'MANO_3D_reconstruction': 'NOT_RUN', 'MANO_2D_physical_semantics': 'NEEDS_REVIEW',
            'persistent_physical_identity_or_entity_slot_join': 'NEEDS_REVIEW',
            'camera_physical_calibration': 'NOT_RUN', 'pixel_valid_consumption_by_losses': 'NOT_RUN',
            'target_entity_valid': False, 'interaction_valid': False},
        'source_publication': 'Only metadata/statistics; source images/depth/reference arrays/complete annotations stay local-only.',
        'production_changes': 'NONE', 'training': 'NOT_RUN'}
    if (OUT / 'logs/tests.json').exists():
        result['tests'] = load('logs/tests.json')
    save('report.json', result)
    stats = {'rgb_max_error': max(r['resized_normalized_metrics']['max_absolute_error'] for r in rgb['rgb_comparisons']),
             'depth_max_error': max(r['resized_raw_value_metrics']['max_absolute_error'] for r in rgb['depth_comparisons']),
             'depth_source_invalid_mean': float(np.mean([r['source_invalid_fraction'] for r in rgb['source_frames']['depth']])),
             'depth_resized_invalid_mean': float(np.mean([r['resized_invalid_fraction'] for r in rgb['source_frames']['depth']])),
             'unique_frame_count': rgb['randomization']['unique_frames'],
             'read_observations': rgb['randomization']['observations_including_overlap'],
             'unknown_source_pixels': sum(r['source_unknown_pixels'] for r in mask['source_frames']),
             'invalid_target_pixels': sum(r['downsample_invalid_pixels'] for r in mask['source_frames']),
             'MANO_valid_unique': sum(x['valid_frames'] for x in hand['coverage'].values()),
             'boundary_checks': len(action['boundary_checks']),
             'timestamp_float32_max_error_seconds': timestamp_error,
             'all_checks': checks}
    save('evidence/summary_metrics.json', stats)
    print(json.dumps(stats, indent=2), flush=True)


def tests(root):
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_hoi4d_realdata_audit.py',
               'tests/test_hoi4d_dataset.py', 'tests/test_hoi4d_data.py', 'tests/test_e3a_collate.py']
    p = subprocess.run(command, text=True, capture_output=True, env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    output = sanitize(p.stdout + p.stderr, root)
    (OUT / 'logs/pytest.txt').write_text(output)
    save('logs/tests.json', {'exit_code': p.returncode, 'status': 'PASS' if p.returncode == 0 else 'BLOCKED',
        'command': ['python', *command[1:]], 'scope': 'Synthetic helper tests + published real-evidence regressions. Not new physical GT validation.',
        'execution_timestamp_utc': datetime.now(timezone.utc).isoformat(), 'stdout': 'logs/pytest.txt',
        'pytest_version': importlib.metadata.version('pytest'), 'approved_dependency_installation': 'User approved pytest in repository .venv only'})
    print(output, flush=True)
    if p.returncode:
        raise RuntimeError('Tests failed; see logs')


def seal():
    payload, withheld = [], []
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.name in {'manifest.json', 'SHA256SUMS.txt'}:
            continue
        row = {'archive_path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
        (withheld if 'local_only' in p.parts else payload).append(row)
    ref = load('evidence/local_only/reference_index.json')
    source_checksums = [{'dataset_path': v['source_path'], 'sha256': v['source_sha256']} for v in ref['videos'].values()]
    source_checksums += [{'dataset_path': r['source_path'], 'sha256': r['source_sha256']}
                         for rows in [ref['masks'], ref['poses'], ref['hands']['right'], ref['hands']['left']] for r in rows]
    for key in ['action', 'intrinsics']:
        item = load(f'evidence/{key}_integrity.json')
        source_checksums.append({'dataset_path': item['source_path'], 'sha256': item['source_sha256']})
    save('manifest.json', {'repository': 'https://github.com/heu-gj/E3A', 'branch': 'audit/h2-hoi4d-realdata-integrity',
        'base_commit': BASE, 'current_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'commit_semantics': 'current_commit is the parent observed when sealing; a commit cannot embed its own hash. Final delivered HEAD is in PR history.',
        'dataset_version': 'UNKNOWN extracted HOI4D release revision', 'audited_sequence': SEQ,
        'execution_timestamp_utc': datetime.now(timezone.utc).isoformat(), 'client_date': '2026-11-22',
        'path_namespaces': {'dataset': 'all source paths relative to configured dataset root',
                            'archive': 'this directory', 'repository': 'repository-relative code paths'},
        'file_inventory': source_checksums, 'artifact_inventory': payload, 'withheld_local_inventory': withheld,
        'sha256_checksums': {r['archive_path']: r['sha256'] for r in payload},
        'checksum_policy': 'Manifest excludes itself/SHA256SUMS; SHA256SUMS includes manifest but excludes itself.',
        'publication': 'No original data, full annotations, raw images, checkpoints, unauthorized derived images or host paths uploaded.'})
    files = [OUT / r['archive_path'] for r in payload] + [OUT / 'manifest.json']
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n' for p in sorted(files)))
    print('sealed public', len(files) + 1, 'withheld', len(withheld), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['smoke', 'reference', 'audit', 'visualize', 'report', 'tests', 'seal'])
    parser.add_argument('--root', type=Path, default=os.environ.get('HOI4D_ROOT'))
    args = parser.parse_args()
    require_venv()
    if args.root is None:
        parser.error('Provide --root or HOI4D_ROOT')
    if args.phase == 'smoke':
        environment()
        smoke(args.root)
    elif args.phase == 'reference':
        reference(args.root)
    elif args.phase == 'audit':
        audit(args.root)
    elif args.phase == 'report':
        report(args.root)
    elif args.phase == 'visualize':
        visualize(args.root)
    elif args.phase == 'tests':
        tests(args.root)
    else:
        seal()
