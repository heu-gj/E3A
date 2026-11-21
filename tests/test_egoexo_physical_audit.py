"""Audit regressions. Synthetic helper cases do not validate physical geometry."""
import importlib.util
import io
import json
import hashlib
import re
from pathlib import Path

from e3a.data.egoexo import parse_egoexo_hand_pose_annotations


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('egoexo_audit',ROOT/'scripts/audit_egoexo_physical.py')
AUDIT=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_stream_real_schema_with_chunk_boundaries_and_escaped_braces():
    source={'ds':'test-only','annotations':{'take1':{'name':'a"}b','nested':{'a':[1,2]}},'take2':{'object_masks':{}}}}
    assert dict(AUDIT.iter_relation_takes(io.StringIO(json.dumps(source)),chunk=7))==source['annotations']


def test_relation_three_states_are_not_collapsed():
    record={'annotated_frames':[0,30,60],'annotation':{'30':{'encodedMask':'synthetic-only'}}}
    assert AUDIT.classify_frame(record,30)=='POSITIVE_RECORD'
    assert AUDIT.classify_frame(record,0)=='EXPLICIT_ABSENCE'
    assert AUDIT.classify_frame(record,31)=='UNANNOTATED'
    assert AUDIT.coverage(record)['explicit_absence_count']==2


def test_coverage_detects_masks_outside_annotated_frames():
    assert AUDIT.coverage({'annotated_frames':[0],'annotation':{'30':{}}})['mask_outside_coverage']==[30]


def test_hand_filter_does_not_remove_tools_named_hand_blender():
    assert AUDIT.is_hand('right hand_0')
    assert AUDIT.is_hand('left_hand_1')
    assert not AUDIT.is_hand('hand blender_0')


def test_multiple_records_first_empty_current_parser_discards_later_record_risk():
    # A constructed adversarial case, NOT evidence this occurs in the local release.
    annotation={'0':[{'annotation3D':{}},{'annotation3D':{'right_wrist':{'x':1.,'y':2.,'z':3.}}}]}
    camera={'0':{'aria':{'camera_extrinsics':[[1,0,0,0],[0,1,0,0],[0,0,1,0]]}}}
    assert parse_egoexo_hand_pose_annotations(annotation,camera,min_views_for_3d=None)=={}


def test_missing_data_cannot_be_reported_as_zero_reprojection_error():
    p=AUDIT.OUT/'evidence/hand_reprojection.json'
    data=json.loads(p.read_text())
    assert data['status']=='BLOCKED' and data['execution_status']=='NOT_RUN'
    assert data['current_direction']['median_error'] is None
    assert data['inverse_direction']['p90_error'] is None
    ready=json.loads((AUDIT.OUT/'evidence/supervision_readiness.json').read_text())
    assert not ready['target_entity_valid'] and not ready['interaction_valid']


def test_public_archive_checksums_and_no_raw_or_sensitive_payloads():
    manifest=json.loads((AUDIT.OUT/'manifest.json').read_text())
    expected={x['archive_path'] for x in manifest['artifact_inventory']}|{'manifest.json'}
    actual=set()
    for line in (AUDIT.OUT/'SHA256SUMS.txt').read_text().splitlines():
        sha,name=line.split('  ',1);path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts and 'local_only' not in path.parts
        assert path.suffix not in {'.mp4','.png','.jpg','.npy','.pickle','.zip','.pt','.pth','.ckpt'}
        data=(AUDIT.OUT/name).read_bytes()
        assert len(data)<10*1024*1024
        assert hashlib.sha256(data).hexdigest()==sha
        assert not re.search(rb'/(?:home|shares2|tmp)/',data)
        actual.add(name)
    assert actual==expected


def test_real_mask_observations_and_video_sources_remain_scoped():
    decode=json.loads((AUDIT.OUT/'evidence/relation_mask_decode.json').read_text())
    assert decode['decoded_records']==70 and not decode['all_encoded_masks_checked']
    assert all(x['adapter_sdk_equal'] and x['source_shape']==x['decoded_shape'] for x in decode['records'])
    v=json.loads((AUDIT.OUT/'evidence/video_alignment.json').read_text())
    assert len(v['videos'])==2
    assert all(x['fps']==30 and x['resolver']['status']=='BLOCKED' for x in v['videos'])
    assert all(len(x['sampled_frames'])>=5 for x in v['videos'])
