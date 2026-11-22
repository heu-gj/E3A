"""H2 metric/helper tests; never replace the real-source audit."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('h2_audit', ROOT / 'scripts/audit_hoi4d_realdata.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_metrics_detect_uint16_difference_without_overflow():
    result = AUDIT.metrics(np.array([0, 65535], np.uint16), np.array([1, 0], np.uint16))
    assert result['different_elements'] == 2
    assert result['max_absolute_error'] == 65535
    assert result['mean_absolute_error'] == 32768
    with pytest.raises(ValueError):
        AUDIT.metrics(np.zeros(2), np.zeros(3))


def test_c20_known_foreground_survives_unknown_index4():
    rgb = np.array([[[128, 0, 0], [0, 0, 128]], [[0, 128, 0], [128, 128, 0]]], np.uint8)
    sem, inst, valid = AUDIT.independent_mask(rgb)
    assert sem.tolist() == [[2, -1], [47, 47]]
    assert inst.tolist() == [[1, -1], [2, 3]]
    assert valid.tolist() == [[True, False], [True, True]]


def test_unknown_pixel_anywhere_in_2x2_invalidates_downsample():
    valid = np.ones((4, 4), bool)
    valid[1, 1] = False  # nearest sampling at0,0 would miss it
    result = AUDIT.conservative_half_validity(valid)
    assert result.tolist() == [[False, True], [True, True]]


def test_frameid_cannot_replace_filename_or_be_assumed_fixed_offset():
    a = AUDIT.numeric_pose_index(150, {'frameId': 16, 'isEffective': 1, 'dataList': []})
    b = AUDIT.numeric_pose_index(151, {'frameId': 151, 'isEffective': 1, 'dataList': []})
    assert a['file_index'] == 150 and a['raw_frameId'] == 16
    assert b['raw_frameId'] == b['file_index']
    assert a['raw_frameId'] - a['file_index'] != b['raw_frameId'] - b['file_index']


def test_raw_depth_uint16_values_exact_in_float32_without_scale():
    raw = np.array([[0, 8000, 65535, 999], [1, 2, 3, 4]], np.uint16)
    expected = raw[::2, ::2]
    assert np.array_equal(expected.astype(np.float32), expected)
    assert AUDIT.metrics(expected, expected.astype(np.float32))['different_elements'] == 0
    assert AUDIT.metrics(expected, expected.astype(np.float32) / 1000)['different_elements'] > 0


def test_missing_mano_is_missing_not_valid_zero_hand(tmp_path):
    missing = tmp_path / '299.pickle'
    assert not missing.is_file()
    with pytest.raises(FileNotFoundError):
        AUDIT.source_hand(missing)


def test_strictly_future_onset_before_at_after_and_overlap():
    events = [{'event': 'grasp', 'startTime': 1., 'endTime': 4.},
              {'event': 'carry', 'startTime': 3., 'endTime': 5.}]
    assert AUDIT.strict_future(events, 1. - 1e-8)['event'] == 'grasp'
    assert AUDIT.strict_future(events, 1.)['event'] == 'carry'
    assert AUDIT.strict_future(events, 1. + 1e-8)['event'] == 'carry'
    assert AUDIT.strict_future(events, 3.) is None
    target = {'time_to_event': 2., 'time_valid': True, 'action_valid': False, 'target_entity_valid': False}
    assert AUDIT.action_check(events, 1., target)['pass']
    assert not AUDIT.action_check(events, 1., target | {'target_entity_valid': True})['pass']


def test_hash_accounts_for_channel_order():
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], np.uint8)
    assert AUDIT.array_hash(bgr) != AUDIT.array_hash(bgr[..., ::-1])


def test_real_evidence_covers_all_frames_and_known_baselines():
    # This is an evidence regression check; it does not execute new physical measurements.
    rgb = json.loads((AUDIT.OUT / 'evidence/rgb_depth_integrity.json').read_text())
    mask = json.loads((AUDIT.OUT / 'evidence/mask_integrity.json').read_text())
    mano = json.loads((AUDIT.OUT / 'evidence/mano_integrity.json').read_text())
    action = json.loads((AUDIT.OUT / 'evidence/action_integrity.json').read_text())
    assert rgb['randomization']['unique_frames'] == 300
    assert len(rgb['source_frames']['depth']) == 300
    assert not rgb['mismatches']
    assert mask['frame220']['source_unknown_pixels'] == 37
    assert mask['frame220']['downsample_invalid_pixels'] == 16
    assert mano['coverage']['right']['valid_frames'] == 234
    assert mano['coverage']['left']['valid_frames'] == 233
    assert action['time_valid_clips'] == 34 and action['action_valid_clips'] == 0
