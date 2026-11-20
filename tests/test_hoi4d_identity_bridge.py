"""Audit helper tests; synthetic cases do not establish physical correspondence."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from e3a.data.hoi4d import decode_hoi4d_motion_mask, hoi4d_color_palette

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('identity_audit', ROOT/'scripts/audit_hoi4d_identity_bridge.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_frameid_anomaly_detected_in_archived_real_observations():
    evidence=json.loads((AUDIT.ARCHIVE/'evidence/objpose_frameid_audit.json').read_text())
    rows={r['file_index']:r for r in evidence['records']}
    assert [rows[i]['raw_frameId'] for i in [0,1,9,10,11,19,20,21,149,150,151,299]] == [1,1,9,2,11,19,3,21,149,16,151,31]
    assert not AUDIT.offset_summary(evidence['records'])['fixed_offset_supported']
    old_stats=json.loads((AUDIT.ARCHIVE/'evidence/objpose_statistics.json').read_text())
    assert all(r['frameId']==rows[r['file_index']]['raw_frameId'] for r in old_stats)


def test_file_index_frameid_and_timestamp_stay_separate():
    row=AUDIT.frame_record(150,{'frameId':16,'filePath':'internal/150.pcd','dataList':[],'isEffective':1})
    assert row['file_index']==150 and row['raw_frameId']==16
    assert row['filePath_pcd_index']==150
    assert row['video_decoded_index'] is None and row['actual_annotation_timestamp'] is None
    assert row['candidate_video_index']==150


def test_c20_ignore_proposal_preserves_known_regions_and_hands():
    palette=hoi4d_color_palette()
    rgb=palette[np.array([[0,1,2,3,4]])]
    sem,ins,ignored=AUDIT.diagnostic_ignore_palette(rgb,'C20')
    assert sem.tolist()==[[0,2,47,47,-1]]
    assert ins.tolist()==[[0,1,2,3,-1]]
    assert ignored.tolist()==[[False,False,False,False,True]]
    with pytest.raises(ValueError):decode_hoi4d_motion_mask(rgb,'C20')


@pytest.mark.parametrize('category,semantics',[('C13',[31,47,46]),('C7',[15,47,46])])
def test_multi_object_category_maps(category,semantics):
    rgb=hoi4d_color_palette()[np.array([[1,2,3]])]
    semantic,instance=decode_hoi4d_motion_mask(rgb,category)
    assert semantic.tolist()==[semantics]
    assert instance.tolist()==[[1,2,3]]


def test_foreground_offset_uses_static_max_not_number_of_ids():
    assert AUDIT.foreground_id([0,2,9],1)==10
    assert AUDIT.foreground_id([0,1,2],3)==5
    with pytest.raises(ValueError):AUDIT.foreground_id([],1)


@pytest.mark.parametrize('mapping,identity,matched',[(None,True,True),(3,False,True),(3,True,False)])
def test_target_stays_invalid_when_any_mapping_bridge_is_missing(mapping,identity,matched):
    assert AUDIT.target_gate(mapping,identity,matched)=={'target_entity_track_id':-1,'target_entity_valid':False}
