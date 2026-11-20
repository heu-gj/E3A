"""Build metadata and checksums for the existing Ego-Exo4D audit evidence."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import re
from datetime import datetime, timezone

OUT=Path('reports/audits/P1-EGOEXO4D-Physical-Data-Audit')
BASE='bade1b8d68205335ce039490ded919965216fe70'


def load(name):return json.loads((OUT/name).read_text())


def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def report():
    selected=load('evidence/candidate_selection_summary.json')
    inventory=load('evidence/dataset_inventory.json')
    decode=load('evidence/relation_mask_decode.json')
    readiness=load('evidence/supervision_readiness.json')
    validation=load('evidence/video_alignment.json')
    # Additional cadence measurements on the selected real raw record; no GT invention.
    raw=load('evidence/local_only/selected_relation.json')
    cadence=[]
    for obj,cams in raw['object_masks'].items():
        for camera,c in cams.items():
            numbers=sorted({int(x) for x in c['annotated_frames']})
            deltas=sorted({b-a for a,b in zip(numbers,numbers[1:])})
            cadence.append({'object':obj,'camera':camera,'source_frame_deltas':deltas,
                'annotation_fps':c['annotation_fps'],'source_frame_range':[numbers[0],numbers[-1]] if numbers else None})
    save('evidence/relation_frame_cadence.json',{'status':'PASS','scope':'Actual annotated frame-number differences, not independent sensor synchronization',
        'records':cadence,'candidate_source_fps':30,'observed_annotation_fps':1,
        'warning':'Frame numbers progress by30 at1Hz annotations; dividing source indices by annotation_fps would be incorrect.'})
    validation['local_visual_review']={'status':'PARTIAL','reviewed_aria_frames':[0,7770,15510,23280,31020],
        'reviewed_exo_frames':[15510],
        'observations':'Recipe mask at start/end, sliced tomato/bowl/utensils mid-take align coarsely after source-to-video scaling. Dense labels obscure boundaries; no pixel-exact registration or independent clock certification.',
        'source_images_published':False}
    save('evidence/video_alignment.json',validation)
    findings=[
        {'component':'parse_egoexo_hand_pose_annotations','status':'NEEDS_REVIEW','kind':'potential multi-record information loss',
         'evidence':'src/e3a/data/egoexo.py selects frame_records[0]. Synthetic adversarial test with valid later record confirms code discards it; local occurrence UNKNOWN because EgoPose files missing.',
         'scope':'CODE_AND_SYNTHETIC_RISK_ONLY; not a real release bug confirmation'},
        {'component':'_resolve_camera_extrinsics','status':'NEEDS_REVIEW','kind':'geometry unverified',
         'evidence':'Supports frame-first and camera-first finite3x4 matrices; no local source matrix, units or forward/inverse reprojection evidence.'},
        {'component':'decode_egoexo_relation_mask','status':'PASS','kind':'real sampled decoder compatibility',
         'evidence':f'{decode["decoded_records"]} actual records decoded by official ego4d SDK1.7.3 and compared with E3A; shapes and binary values agree.',
         'evidence_file':'evidence/relation_mask_decode.json'},
        {'component':'build_egoexo_relation_clip_targets','status':'PASS','kind':'bounded real clip contract',
         'evidence':'Two actual tracks, three source frames; explicit absence retained as valid with ID-1. This is source-resolution contract verification, not downscaled-image training approval.',
         'evidence_file':'evidence/relation_mask_decode.json'},
        {'component':'resolve_egoexo_video_path','status':'BLOCKED','kind':'confirmed local path incompatibility',
         'evidence':'aria04_214-1 correctly selects aria04/rgb metadata, but require_exists fails: only downscaled/448 file exists. Same issue for cam01.',
         'evidence_file':'evidence/video_alignment.json'},
        {'component':'parse_egoexo_keystep_segments','status':'BLOCKED','kind':'real compatibility NOT_RUN',
         'evidence':'Code uses step_unique_id and seconds; raw Keystep files absent. Duration bounds/taxonomy/task timing require dataset-reader checks.'},
        {'component':'build_egoexo_keystep_vocabulary','status':'BLOCKED','kind':'real training vocabulary NOT_RUN',
         'evidence':'Code builds sorted unique training IDs; no raw training annotations or vocabulary membership validation locally.'},
        {'component':'build_egoexo_anticipation_target','status':'PARTIAL','kind':'code/synthetic verification only',
         'evidence':'Strictly future onset and invalid target behavior have unit tests; no real raw segments to run required boundary sweep.'}]
    save('evidence/adapter_review.json',{'production_code_modified':False,'findings':findings,
        'manual_automatic_policy':'Required min_views_for_3d argument prevents silent default; baseline manual>=3 and automaticNone are code policy, not validated against missing local annotations.',
        'source_units':'Default1.0 preserves values, does not establish meters',
        'empty_clip_risk':'A clip with zero observed tracks must not become a negative scene merely because annotations are unavailable'})
    result={'task_id':'P1-EGOEXO4D-Physical-Data-Audit','task_status':'PARTIAL','base_commit':BASE,
        'branch':'audit/p1-egoexo-physical-data','commit_semantics':'Commit embedded in manifest is the parent observed at sealing. Final delivered HEAD is given by PR/history, avoiding self-reference.',
        'prerequisite':{'PR1':'MERGED','squash_commit':BASE,'initial_worktree':'CLEAN','remote_task_branch_initial_commit':BASE},
        'dataset':{'root':'.','metadata_version':inventory['metadata_version'],'metadata_date':inventory['metadata_date'],
            'annotation_version':'Relations ds20240311_20:16:25; independent release revision UNKNOWN','download_receipt':'NOT_FOUND'},
        'take_uid':selected['selected']['take_uid'],'take_name':selected['selected']['take_name'],
        'sources':{'ego_pose':'https://docs.ego-exo4d-data.org/annotations/ego_pose/',
            'relations':'https://docs.ego-exo4d-data.org/annotations/relations/',
            'keystep':'https://docs.ego-exo4d-data.org/annotations/keystep/'},
        'audit_results':{
            'metadata_version':{'status':'PASS','evidence_level':'ACTUAL_LOCAL_METADATA_FIELD',
                'evidence':'raw/metadata.json explicitly version=v2/date240325; not inferred from folder name',
                'evidence_files':['evidence/dataset_inventory.json']},
            'relation_decode':{'status':'PASS','evidence_level':'REAL_SAMPLE_OFFICIAL_SDK',
                'evidence':f'{decode["decoded_records"]} actual decoded records; no sampled empty mask; not exhaustive',
                'evidence_files':['evidence/relation_mask_decode.json']},
            'relation_coverage':{'status':'PASS','evidence_level':'SELECTED_REAL_METADATA_SET_CHECK',
                'evidence':'25 track/camera records checked; no encoded frame outside annotated_frames; missing entries and unannotated frames kept distinct',
                'evidence_files':['evidence/relation_coverage.json']},
            'frame_alignment':{'status':'PARTIAL','evidence_level':'VIDEO_SEEK_PTS_AND_WITHHELD_VISUAL_INSPECTION',
                'evidence':'Five temporal positions plus exact mask-frame samples per camera; scaling visually supports association, not independent time/calibration certification',
                'evidence_files':['evidence/video_alignment.json','evidence/cross_modal_frames.csv','evidence/resolver_alias_audit.json']},
            'hand':{'status':'BLOCKED','execution_status':'NOT_RUN','evidence_files':['evidence/ego_pose_coverage.json']},
            'camera_geometry':{'status':'BLOCKED','execution_status':'NOT_RUN','evidence_files':['evidence/camera_pose_validation.json','evidence/hand_reprojection.json','evidence/mps_availability.json']},
            'keystep':{'status':'BLOCKED','execution_status':'NOT_RUN','evidence_files':['evidence/keystep_validation.json']},
            'Hand_to_Entity':{'status':'BLOCKED','evidence':'No real EgoPose and no hand tracks in selected relation take; visual hand presence is not3D/contact GT'},
            'Entity_to_Future_Action':{'status':'BLOCKED','evidence':'Raw event annotation absent; no verified Keystep event-to-object relation'}},
        'supervision_readiness':readiness,'adapter_findings':findings,
        'missing_data_to_provide':['EgoPose manual and automatic hand annotation roots','EgoPose camera-pose JSON root',
            'Raw Keystep JSON with annotations/taxonomy/vocabulary (not keystep_benchmark)','MPS calibration/timesync if available',
            'Release/download manifests explaining137 train relation takes absent from takes metadata'],
        'unresolved_issues':['No all-modality take: hand/camera/raw keystep files missing across checked local root',
            'Downscaled video resolver and source-mask scaling need explicit Dataset policy',
            'Source clocks, distortion/crop, coordinates and physical units not verified',
            'EgoPose multiple-record semantics UNKNOWN locally','137 relation_train UID records absent from takes metadata',
            'No validated event-specific tool/patient/container roles, target mapping or physical contact'],
        'safety':{'training':'NOT_RUN','production_code_changes':False,'target_entity_valid':False,'interaction_valid':False,
                  'raw_data_published':False,'unapproved_images_published':False}}
    if (OUT/'logs/tests.json').exists():result['tests']=load('logs/tests.json')
    save('report.json',result)


def seal():
    payload=[];withheld=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.name in {'manifest.json','SHA256SUMS.txt'}:continue
        row={'archive_path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':digest(p)}
        (withheld if 'local_only' in p.parts else payload).append(row)
    r=load('report.json');inv=load('evidence/dataset_inventory.json')
    manifest={'repository':'https://github.com/heu-gj/E3A','branch':r['branch'],'base_commit':BASE,
        'current_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'commit_semantics':'Parent HEAD observed at sealing; final archive commit is identified externally by PR/history',
        'execution_timestamp_utc':datetime.now(timezone.utc).isoformat(),'dataset_version':r['dataset'],
        'audited_take':{'uid':r['take_uid'],'name':r['take_name']},
        'path_namespaces':{'dataset':'. (all source dataset paths relative to configured root)',
            'archive':'this directory','repository':'repository root','external_sdk':'module names plus version/SHA256; no host paths'},
        'source_inventory':inv,'file_inventory':'evidence/take_inventory.csv',
        'artifact_inventory':payload,'withheld_local_inventory':withheld,
        'sha256_checksums':{x['archive_path']:x['sha256'] for x in payload},
        'checksum_policy':'Manifest excludes itself and SHA256SUMS to avoid recursion. SHA256SUMS includes manifest, excludes itself.'}
    save('manifest.json',manifest)
    files=[OUT/x['archive_path'] for x in payload]+[OUT/'manifest.json']
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.relative_to(OUT)}\n' for p in sorted(files)))
    print('public files',len(payload)+2,'withheld local files',len(withheld))


def tests():
    command=[sys.executable,'-m','pytest','tests/test_egoexo_data.py','-v',
        'tests/test_action_vocabulary.py','tests/test_entity_target_collate.py',
        'tests/test_e3a_collate.py','tests/test_egoexo_physical_audit.py']
    process=subprocess.run(command,text=True,capture_output=True)
    text=(process.stdout+process.stderr).replace(str(Path.cwd()),'<repository-root>').replace(sys.executable,'python')
    assert not re.search(r'/(?:home|shares2|tmp)/',text),'Test log still contains local absolute paths'
    (OUT/'logs/pytest.txt').write_text(text)
    save('logs/tests.json',{'execution_timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'command':['python',*command[1:]],'exit_code':process.returncode,
        'status':'PASS' if process.returncode==0 else 'BLOCKED',
        'passed':int(re.search(r'(\d+) passed',text).group(1)) if process.returncode==0 else None,
        'evidence_level':'SYNTHETIC_UNIT_CASES_AND_ARCHIVED_REAL_OBSERVATION_REGRESSIONS; NOT MISSING_MODALITY_PHYSICAL_VALIDATION',
        'log':'logs/pytest.txt','real_source_measurements':'Separate audit script and evidence files'})
    print(text)
    if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['report','seal','tests'])
    phase=parser.parse_args().phase
    if phase=='report':report()
    elif phase=='tests':tests()
    else:seal()
