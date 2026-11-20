"""Read-only follow-up audit. Public outputs are relative-path statistics only."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from e3a.data.hoi4d import decode_hoi4d_motion_mask, hoi4d_color_palette

COMMIT = '788b43552176c77c01c3e98dd5f101fd6a94ee7a'
ARCHIVE = Path('reports/audits/P1-HOI4D-Physical-Data-Audit')
SEQ = 'ZY20210800001/H1/C20/N30/S295/s04/T4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    path = ARCHIVE / 'evidence' / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def frame_record(index, raw):
    internal = raw.get('filePath', '')
    return {'file_index': index, 'raw_frameId': raw.get('frameId'),
        'filePath': 'ANNOTATION_INTERNAL/' + Path(internal).name if internal else '',
        'filePath_sha256': hashlib.sha256(internal.encode()).hexdigest(),
        'filePath_nonempty': bool(internal), 'filePath_pcd_index': int(Path(internal).stem) if internal else None,
        'isEffective': raw.get('isEffective'), 'dataList_count': len(raw.get('dataList', [])),
        'dataList': [{'object_id': o.get('id'), 'object_label': o.get('label'), 'keys': list(o)} for o in raw.get('dataList', [])],
        'video_decoded_index': None, 'actual_annotation_timestamp': None,
        'candidate_video_index': index, 'alignment_status': 'PARTIAL; content checks only, no independent annotation timestamp'}


def offset_summary(records):
    offsets = sorted({r['raw_frameId'] - r['file_index'] for r in records})
    return {'offsets': offsets, 'fixed_offset_supported': len(offsets) == 1,
        'frameId_equals_filename_count': sum(r['raw_frameId'] == r['file_index'] for r in records)}


def diagnostic_ignore_palette(rgb, category):
    """Audit-only proposal; never changes the production decoder."""
    known_counts = {'C20': 3, 'C13': 3, 'C7': 3}
    palette = hoi4d_color_palette()
    known = np.zeros(rgb.shape[:2], dtype=bool)
    for color in palette[:known_counts[category] + 1]:
        known |= np.all(rgb == color, axis=-1)
    temporary = rgb.copy()
    temporary[~known] = 0
    semantic, instance = decode_hoi4d_motion_mask(temporary, category)
    semantic[~known] = -1
    instance[~known] = -1
    return semantic, instance, ~known


def foreground_id(static_ids, local_id):
    if not static_ids or local_id <= 0:
        raise ValueError('Need verified static IDs and positive local foreground ID')
    return int(max(static_ids)) + local_id


def target_gate(event_to_track, identity_verified, observed_and_matched):
    valid = event_to_track is not None and identity_verified and observed_and_matched
    return {'target_entity_track_id': event_to_track if valid else -1, 'target_entity_valid': bool(valid)}


def bbox_iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.


def read_pcd(path):
    with path.open() as stream:
        header = []
        for line in stream:
            header.append(line.strip())
            if line.startswith('DATA'): break
        assert header[-1] == 'DATA ascii'
        return header, np.loadtxt(stream)


def video_pts(path, raw_depth=False):
    cap = cv2.VideoCapture(str(path))
    if raw_depth:cap.set(cv2.CAP_PROP_CONVERT_RGB,0)
    metadata = {'fps': cap.get(5), 'width': cap.get(3), 'height': cap.get(4), 'fourcc': int(cap.get(6))}
    times = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        if not times:
            metadata['first_frame_dtype']=str(frame.dtype)
            metadata['first_frame_shape']=list(frame.shape)
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    cap.release()
    return metadata | {'decoded_frames': len(times), 'pts_seconds': times}


def run(root, official):
    import subprocess
    assert subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True).strip()
    annroot = root / 'extracted/annotations/HOI4D_annotations'
    ann = annroot / SEQ
    pinned = ['README.md','prepare_4Dseg/prepare_4Dseg_dataset.py','prepare_4Dseg/scene2frame.py',
        'prepare_4Dseg/pixel2category.py','prepare_4Dseg/utils.py',
        'definitions/motion segmentation/label.csv','definitions/task/task_definitions.csv',
        'prepare_4Dseg/palette.py','utils/decode.py']
    sources = [{'official_repo_path': f, 'sha256': sha(official/f),
        'url': f'https://github.com/hoi4d/HOI4D_Script/blob/{COMMIT}/'+f.replace(' ','%20')} for f in pinned]
    video = root / 'extracted/release/HOI4D_release' / SEQ / 'align_rgb/image.mp4'
    timeline = video_pts(video)
    records = []
    rawposes = []
    for i in range(300):
        path = ann / 'objpose' / f'{i}.json'
        raw = json.loads(path.read_text())
        rawposes.append(raw)
        row = frame_record(i, raw)
        row.update(source_path=str(path.relative_to(root)), source_sha256=sha(path),
                   candidate_video_pts_seconds=timeline['pts_seconds'][i])
        records.append(row)
    summary = offset_summary(records)
    summary.update(nonempty_filePath_count=sum(r['filePath_nonempty'] for r in records),
        nonempty_filePath_indices=[r['file_index'] for r in records if r['filePath_nonempty']],
        frameId_semantics='UNKNOWN; mixed numbering; ordinal keyframe hypothesis is inference, not official specification',
        filePath_pattern='Nonempty for 0,10,...,290,299; embedded PCD basename equals file index; frameId enumerates 1..31 there',
        nonempty_path_frameId_sequence=[r['raw_frameId'] for r in records if r['filePath_nonempty']],
        timestamp_status='Decoded video PTS measured; annotation capture timestamps UNKNOWN',
        filePath_privacy='Public filePath keeps basename only; original internal directory includes personal metadata. Raw files remain unchanged locally.',
        focus_indices=[0,1,9,10,11,19,20,21,149,150,151,299])
    save('objpose_frameid_audit.json', {'status':'PASS','evidence_level':'RAW_JSON_REREAD_WITH_SHA256',
        'scope':'Observed field values and rejection of fixed offset, not frameId semantic identification',
        'summary':summary,'video_path':str(video.relative_to(root)),'video':timeline,'records':records})
    with (ARCHIVE/'evidence/objpose_frameid_audit.csv').open('w',newline='') as stream:
        fields=['file_index','raw_frameId','filePath','filePath_nonempty','filePath_pcd_index','isEffective',
            'dataList_count','dataList','video_decoded_index','actual_annotation_timestamp','candidate_video_index',
            'candidate_video_pts_seconds','alignment_status','source_path','source_sha256','filePath_sha256']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in records:
            writer.writerow({k:('UNKNOWN' if v is None else v) for k,v in (row | {'dataList':json.dumps(row['dataList'])}).items()})
    print('frameId',summary,flush=True)
    K = np.load(root/'extracted/camera_params/camera_params/ZY20210800001/intrin.npy')
    geometry=[]
    for i, raw in enumerate(rawposes):
        mask=np.asarray(Image.open(ann/'2Dseg/mask'/f'{i:05d}.png').convert('RGB'))
        chair=np.all(mask==hoi4d_color_palette()[1],axis=-1)
        yy,xx=np.where(chair)
        mb=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
        o=raw['dataList'][0]
        boxes=o.get('2dBox',{}).get('camera1',[])
        bb=[*boxes[0],*boxes[1]] if len(boxes)==2 else None
        c=np.array([o['center'][a] for a in 'xyz']);uv=K@c;uv=uv[:2]/uv[2]
        u,v=np.rint(uv).astype(int)
        geometry.append({'file_index':i,'raw_frameId':raw['frameId'],'object_id':o['id'],'object_label':o['label'],
            'mask_bbox':mb,'box_present':bb is not None,'box_mask_bbox_iou':bbox_iou(bb,mb) if bb else None,
            'box_contains_mask_fraction':float(np.mean((xx>=bb[0])&(xx<=bb[2])&(yy>=bb[1])&(yy<=bb[3]))) if bb else None,
            'camera_coordinate_hypothesis_uv':uv.tolist(),
            'projected_center_in_chair':bool(0<=u<1920 and 0<=v<1080 and chair[v,u]),
            'mask_pixels':int(chair.sum())})
        if i%75==0: print('C20 geometry',i,flush=True)
    ious=[r['box_mask_bbox_iou'] for r in geometry if r['box_present']]
    identity={'status':'PARTIAL','evidence_level':'RAW_JSON_AND_REAL_MASK_GEOMETRY', 'sequence_id':SEQ,
        'N30':{'status':'PARTIAL','official_N_semantics':'object instance ID',
            'evidence':'C20/N30 path, all 300 objects labeled chair; nonempty internal filePath contains chair30; single chair palette region',
            'explicit_N_to_pose_id_field':False,'source_CAD_identity_verified':False},
        'pose_ids':sorted({r['object_id'] for r in geometry}), 'pose_labels':sorted({r['object_label'] for r in geometry}),
        'box_frames':len(ious),'missing_box_frames':300-len(ious),'bbox_iou_percentiles':np.percentile(ious,[0,25,50,75,100]).tolist(),
        'center_hypothesis_hits':sum(r['projected_center_in_chair'] for r in geometry),
        'warning':'Projected center assumes camera coordinates; support does not establish physical units or per-frame synchronization. 269 unboxed frames cannot be independently box-verified.',
        'per_frame':geometry,'verified_pose_mask_identity':'PARTIAL; strong category/coarse geometry support, not exact persistent identity certification',
        'E3A_slot':'NOT_RUN; clip-local learned slot requires matcher assignment, never slot=source ID',
        'target_gate':target_gate(None,False,False)}
    save('identity_mapping.json', identity)
    # Execute only the official pixel selector, not the 4D generator.
    sys.path.insert(0,str(official/'prepare_4Dseg'))
    import pixel2category
    import utils as official_utils
    assert Path(official_utils.__file__).resolve()==(official/'prepare_4Dseg/utils.py').resolve()
    path=ann/'2Dseg/mask/00220.png'
    rgb=np.asarray(Image.open(path).convert('RGB'))
    colors,counts=np.unique(rgb.reshape(-1,3),axis=0,return_counts=True)
    colorrows=[]
    palette=hoi4d_color_palette()
    for color,count in zip(colors,counts):
        pi=int(np.where(np.all(palette==color,axis=1))[0][0])
        colorrows.append({'rgb':color.tolist(),'palette_index':pi,'pixels':int(count),
            'official_behavior':'background' if pi==0 else ('foreground mapping' if pi<=3 else 'skip: outside C20 mapping')})
    arrs,sems,ids=pixel2category.f(str(path),'C20')
    covered=np.logical_or.reduce(arrs)
    try:
        decode_hoi4d_motion_mask(rgb,'C20');err=None
    except ValueError as error:err=str(error)
    sem,ins,unknown=diagnostic_ignore_palette(rgb,'C20')
    save('frame220_palette_audit.json',{'status':'PASS','evidence_level':'PINNED_OFFICIAL_SELECTOR_EXECUTED_ON_REAL_MASK',
        'source_path':str(path.relative_to(root)),'sha256':sha(path),'colors':colorrows,
        'official_commit':COMMIT,'official_semantics':sems.tolist(),'official_instances':ids.tolist(),
        'official_ignored_foreground_pixels':int(np.sum((rgb.sum(axis=-1)>0)&~covered)),
        'official_background_behavior':'scene2frame excludes all nonblack mask pixels from background even if pixel2category skips their class',
        'e3a_behavior':'ValueError (fail closed)','e3a_error':err,'ignore_index_proposal':-1,
        'proposal_ignored_pixels':int(unknown.sum()),'production_decoder_changed':False,
        'recommendation':'Keep known masks; ignore unknown pixels in class/mask losses with pixel-level validity. Do not convert unknown to negative background. Current adapter lacks this pixel-level validity contract.'})
    # Read and validate actual legacy log matrices and static labeled point cloud.
    lines=(ann/'3Dseg/output.log').read_text().splitlines()
    rawM=np.array([np.array([list(map(float,row.split())) for row in lines[i+1:i+5]]) for i in range(0,len(lines),5)])
    headers=[list(map(int,lines[i].split())) for i in range(0,len(lines),5)]
    pcdheader,pc=read_pcd(ann/'3Dseg/label.pcd')
    static_ids=np.unique(pc[:,-1]).astype(int).tolist()
    dep={x:importlib.util.find_spec(x) is not None for x in ['open3d','plyfile','pandas','scipy','cv2']}
    camera={'raw_log_records':len(rawM),'log_headers':headers,'max_rotation_orthogonality_error':float(np.max(np.linalg.norm(rawM[:,:3,:3].transpose(0,2,1)@rawM[:,:3,:3]-np.eye(3),axis=(1,2)))),
        'raw_matrix_direction':'Camera-to-scene under the official Open3D LOG reader convention; source-data physical calibration remains PARTIAL',
        'official_extrinsic_direction':'Open3D reader returns inverse(raw log matrix), i.e. scene-to-camera; PointCloud factory inverts extrinsic for camera-to-scene',
        'annotation_video_timestamps':'UNKNOWN; log headers index records, no capture timestamps',
        'units':'UNKNOWN physical unit in release metadata; official depth default uses scale=1000, truncation=3 (library convention)',
        'sources':['https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/io/file_format/FileLOG.cpp',
                   'https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloudFactory.cpp']}
    if dep['open3d']:
        import open3d as o3d
        traj=o3d.io.read_pinhole_camera_trajectory(str(ann/'3Dseg/output.log'))
        E=np.array([p.extrinsic for p in traj.parameters])
        camera.update(open3d_version=o3d.__version__,parsed_records=len(E),
            reader_inverse_raw_max_error=float(np.max(np.abs(E-np.linalg.inv(rawM)))),
            reader_equals_raw_max_error=float(np.max(np.abs(E-rawM))),
            default_log_intrinsics=traj.parameters[0].intrinsic.intrinsic_matrix.tolist(),
            official_Kinect2_intrinsics=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault).intrinsic_matrix.tolist(),
            actual_intrin_npy=K.tolist())
    pipeline={'status':'BLOCKED','evidence_level':'PINNED_CODE_INSPECTION_AND_REAL_INPUT_VALIDATION','official_commit':COMMIT,'sources':sources,
        'inputs':{f:str((ann/f).relative_to(root)) if (ann/f).exists() else None for f in ['3Dseg/output.log','3Dseg/label.pcd','3Dseg/label.txt','refine_2Dseg','2Dseg/mask']},
        'static_pcd':{'header':pcdheader,'semantic_ids':np.unique(pc[:,-2]).tolist(),'instance_ids':static_ids,
                      'points':len(pc),'sha256':sha(ann/'3Dseg/label.pcd')},
        'camera':camera,'dependencies':dep,
        'offset_rule':'foreground_dynamic_ID = max(static_instance_IDs) + category_local_instance_ID',
        'C20_dynamic_IDs_if_inputs_verified':{str(i):foreground_id(static_ids,i) for i in [1,2,3]},
        'generated_labels_status':'NOT_RUN; missing prerequisites; no full or partial point-cloud labels generated',
        'input_mismatch_analysis':{
            'A_code_version_difference':'NOT_ESTABLISHED: the pinned README and generator themselves disagree',
            'B_release_difference':'UNKNOWN: local release revision is not identified',
            'C_preprocessing':'PARTIAL: PCD-to-table and RGB-D frame decoding would be necessary, but no pinned documented conversion/semantic equivalence verified',
            'D_other_package':'UNKNOWN: no refined-mask package or provenance located in selected sequence'},
        'blockers':['label.txt absent; do not alias label.pcd','refine_2Dseg absent; original 2Dseg is not proven refined equivalent',
            'plyfile missing; no dependency installed','source RGB-D are videos; generator expects unpadded i.jpg/i.png, decode.py emits zero-padded names',
            'wrapper iterates data_dir string as directories and later joins it as a string; mask path lookup inconsistent with refine_2Dseg check',
            'default full wrapper spawns 24 processes/300 frames and suppresses runtime exceptions; not run',
            'hardcoded Kinect2 intrinsics differ from released intrin.npy; needs calibration review'],
        'candidate_mask_compatibility':'PARTIAL: real RGB palette matches selector except37 skipped pixels; refinement provenance UNKNOWN',
        'offset_logic_status':'PASS rule evaluated on real static IDs; NOT proof of generated dynamic labels'}
    save('dynamic_4d_pipeline_audit.json',pipeline)
    # Select highest-covered local C13/T4, falling back to C7/T4 only if none complete.
    candidates=json.loads((ARCHIVE/'evidence/candidates.json').read_text())
    selected=[]
    for cat in ['C13','C7']:
        for c in candidates:
            if f'/{cat}/' not in c['sequence'] or not c['sequence'].endswith('/T4'): continue
            paths={k:root/v for k,v in c['paths'].items()}
            counts={k:(sum(f.is_file() for f in p.iterdir()) if p.is_dir() else int(p.is_file())) for k,p in paths.items()}
            c=c | {'actual_counts':counts}
            selected.append(c)
        complete=[c for c in selected if all(c['actual_counts'][k]>0 for k in ['rgb','depth','mask','objpose','action','right'])]
        if complete:break
    complete.sort(key=lambda c:(-min(c['actual_counts'][k] for k in ['mask','objpose','right']),
                                -sum(c['actual_counts'].values()),c['sequence']))
    chosen=complete[0];mp={k:root/v for k,v in chosen['paths'].items()};cat=chosen['sequence'].split('/')[2]
    mrows=[];mposes=[];unknowns=[]
    for i in range(300):
        rgb=np.asarray(Image.open(mp['mask']/f'{i:05d}.png').convert('RGB'))
        sem,ins,unk=diagnostic_ignore_palette(rgb,cat)
        areas={str(k):int(np.sum(ins==k)) for k in [1,2,3]}
        mrows.append({'file_index':i,'mask_path':str((mp['mask']/f'{i:05d}.png').relative_to(root)),
                      'instance_pixel_counts':areas,'unknown_pixels':int(unk.sum())})
        data=json.loads((mp['objpose']/f'{i}.json').read_text())
        mposes.extend((o['id'],o['label']) for o in data['dataList'])
        if i%75==0:print('multi-object',i,flush=True)
    action=json.loads((mp['action']/'color.json').read_text())
    cutoff_events=[{'event_index':i,'label':ev['event'],'start_time':ev['startTime'],'end_time':ev['endTime'],
                   'properties_keys':list(ev.get('properties',{}))} for i,ev in enumerate(action['events'])]
    # Save local-only stills around the cut, never publish source images.
    local=ARCHIVE/'visualizations/local_only';local.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(mp['rgb']))
    for i in [0,150,224,225,250]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
        if ok:
            mask=np.asarray(Image.open(mp['mask']/f'{i:05d}.png').convert('RGB'))
            base=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            fg=np.any(mask!=0,axis=-1);base[fg]=(base[fg]*.55+mask[fg]*.45).astype(np.uint8)
            Image.fromarray(base).resize((960,540)).save(local/f'followup_{cat}_{i:05d}.jpg')
    cap.release()
    multi={'status':'PARTIAL','evidence_level':'REAL_MASKS_POSE_SCHEMA_ACTION_EVENTS_AND_PINNED_TASK_DEFINITION',
        'sequence_id':chosen['sequence'],'category':cat,'selected':chosen,
        'selection':'C13/T4 first; six required modalities present. Descending minimum(mask,objpose,right) then total files then lexical sequence.',
        'candidate_count':len(selected),'ranked_candidates':[{'sequence_id':c['sequence'],'counts':c['actual_counts']} for c in complete],
        'official_task':'Cut apple' if cat=='C13' else 'Take the ball out of the bowl',
        'source_events':cutoff_events,'action_path':str((mp['action']/'color.json').relative_to(root)),
        'event_schema_keys':list(action['events'][0]),'explicit_event_to_instance_mapping':False,
        'pose_id_label_counts':[{ 'id':k[0],'label':k[1],'count':v} for k,v in Counter(mposes).items()],
        'local_instances':{'1':'Knife (candidate tool)' if cat=='C13' else 'Bowl (container)', '2':'Right Hand',
                           '3':'Apple (candidate patient/target)' if cat=='C13' else 'Ball (candidate target)'},
        'instance_3_visible_frames':sum(r['instance_pixel_counts']['3']>0 for r in mrows),
        'per_frame_masks':mrows,'apple_3d_pose':'NOT_FOUND in selected objpose objects',
        'role_conclusion':'Task definition plus cut-phase imagery supports Knife as tool and Apple as patient at cut; earlier Grasp/Pickup may concern knife. This is semantic interpretation, not source-backed event target GT.',
        'persistent_identity':'PARTIAL: distinct palette region consistently numbered3; no independently generated 4D track verification; fragments/occlusion semantics UNKNOWN',
        'target_supervision':target_gate(None,False,False),
        'release_target_policy':'Do not treat N* as target. Need audited event-specific relation and observed-prefix matcher assignment before enabling target loss.',
        'publication':'Images withheld locally; only statistics and small event metadata published'}
    multi['videos']={'rgb':video_pts(mp['rgb']), 'depth':video_pts(mp['depth']/'depth_video.avi',True)}
    save('multi_object_target_audit.json',multi)
    update_existing(records,summary,identity,pipeline,multi,sources)
    print('DONE',flush=True)


def update_existing(records,summary,identity,pipeline,multi,sources):
    path=ARCHIVE/'evidence/observations.json';obs=json.loads(path.read_text())
    obs['objpose']['frameId_minus_filename']=summary['offsets']
    obs['objpose']['frameId_semantics']='UNKNOWN; see objpose_frameid_audit.json'
    obs['objpose']['fixed_frameId_offset_supported']=False
    obs['index_validation']['annotation_timestamp']='UNKNOWN; raw frameId mixed numbering, not a video index'
    obs['sources']={k:v.replace('/blob/main/',f'/blob/{COMMIT}/') if 'HOI4D_Script' in v else v for k,v in obs['sources'].items()}
    obs['official_hoi4d_commit']=COMMIT
    obs['followup_timestamp_utc']=datetime.now(timezone.utc).isoformat()
    save('observations.json',obs)
    path=ARCHIVE/'evidence/cross_modal_frames.csv'
    with path.open() as stream:rows=list(csv.DictReader(stream))
    fields=list(rows[0])+['objpose_file_index','raw_objpose_frameId','verified_pose_video_index','actual_annotation_timestamp',
                         'candidate_video_decoded_index','candidate_video_pts_seconds','pose_video_alignment_status']
    for i,row in enumerate(rows):
        row.update(objpose_frameId=records[i]['raw_frameId'],objpose_file_index=i,raw_objpose_frameId=records[i]['raw_frameId'],
            verified_pose_video_index='UNKNOWN',actual_annotation_timestamp='UNKNOWN',candidate_video_decoded_index=i,
            candidate_video_pts_seconds=records[i]['candidate_video_pts_seconds'],pose_video_alignment_status='PARTIAL')
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    path=ARCHIVE/'report.json';report=json.loads(path.read_text())
    report['official_hoi4d_commit']=COMMIT
    report['followup_base_commit']='dccda559c1cdbaac9f88e1a7717fd5710a0d0fa3'
    report['followup']={'status':'PARTIAL','objpose_frameId':'UNKNOWN mixed numbering; fixed-offset claim retracted',
        'identity_status':identity['status'],'pipeline_status':pipeline['status'],'multi_object_status':multi['status'],
        'multi_object_sequence_id':multi['sequence_id'],'source_commit':COMMIT,
        'generation_status':'NOT_RUN','target_entity_valid':False}
    files=['objpose_frameid_audit.json','objpose_frameid_audit.csv','identity_mapping.json','dynamic_4d_pipeline_audit.json',
           'frame220_palette_audit.json','multi_object_target_audit.json']
    report['evidence_files']+=['evidence/'+f for f in files if 'evidence/'+f not in report['evidence_files']]
    for name,status,filename,description in [
        ('objpose_frameid','PASS','objpose_frameid_audit.json','300 raw files reread; mixed frameId values observed, constant offset disproved. Semantic meaning UNKNOWN.'),
        ('identity_bridge','PARTIAL','identity_mapping.json','31 source boxes compared; all-frame projected center test remains coordinate hypothesis.'),
        ('official_dynamic_pipeline','BLOCKED','dynamic_4d_pipeline_audit.json','Pinned code inspected, log parsed; missing refined masks/label.txt/plyfile and wrapper inconsistencies prevent official generation.'),
        ('frame220_official_behavior','PASS','frame220_palette_audit.json','Pinned official selector executed on real frame220;37 extra pixels skipped. Production decoder unchanged.'),
        ('multi_object_target','PARTIAL','multi_object_target_audit.json','Independent Apple mask observed; no event-specific instance target relation.')]:
        report['audit_results'][name]={'status':status,'evidence':description,'evidence_level':'REAL_DATA_AND_PINNED_CODE; SEE_SCOPED_DETAILS',
                                     'evidence_files':['evidence/'+filename]}
    report['audit_results']['part_to_dynamic_instance']['evidence']='Official offset rule verified on real static IDs, but generation NOT_RUN; source-pose-mask identity only PARTIAL.'
    report['unresolved_issues']+=['raw objpose.frameId meaning UNKNOWN; no constant offset','official generator input packaging and wrapper inconsistencies','Apple per-event target relation and 3D pose missing']
    report['loss_readiness']['entity']['gate']='Known mask regions can be retained with proposed pixel validity; dynamic persistent mapping and unknown/empty policy still unverified'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')


def camera_reprojection(root):
    """Five-frame diagnostic, not 4D label generation or unit certification."""
    ann=root/'extracted/annotations/HOI4D_annotations'/SEQ
    _,pc=read_pcd(ann/'3Dseg/label.pcd')
    xyz=pc[::5,:3]
    lines=(ann/'3Dseg/output.log').read_text().splitlines()
    matrices=np.array([np.array([list(map(float,row.split())) for row in lines[i+1:i+5]]) for i in range(0,len(lines),5)])
    K=np.load(root/'extracted/camera_params/camera_params/ZY20210800001/intrin.npy')
    cap=cv2.VideoCapture(str(root/'extracted/depth/HOI4D_depth_video'/SEQ/'align_depth/depth_video.avi'))
    cap.set(cv2.CAP_PROP_CONVERT_RGB,0)
    rows=[]
    for i in range(300):
        ok,depth=cap.read()
        if not ok:break
        if i not in [0,75,150,224,299]:continue
        mask=np.asarray(Image.open(ann/'2Dseg/mask'/f'{i:05d}.png').convert('RGB'))
        background=np.all(mask==0,axis=-1)
        row={'candidate_frame_index':i,'sampled_static_points':len(xyz),'hypotheses':{}}
        for label,E in [('official_inverse_raw',np.linalg.inv(matrices[i])),('incorrect_direct_raw',matrices[i])]:
            camera=xyz@E[:3,:3].T+E[:3,3]
            front=camera[:,2]>.05;camera=camera[front]
            projected=camera@K.T
            uv=np.rint(projected[:,:2]/projected[:,2:3]).astype(int)
            inbounds=(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1080)
            uv=uv[inbounds];camera=camera[inbounds]
            zbuffer=np.full(1920*1080,np.inf)
            np.minimum.at(zbuffer,uv[:,1]*1920+uv[:,0],camera[:,2])
            zbuffer=zbuffer.reshape(1080,1920)
            use=np.isfinite(zbuffer)&background&(depth>0)
            ratio=depth[use]/zbuffer[use]
            error=np.abs(depth[use]-zbuffer[use]*1000)
            row['hypotheses'][label]={'valid_background_pixels':int(use.sum()),
                'median_depth_raw_per_scene_unit':float(np.median(ratio)),
                'candidate_scale_1000_absolute_error_raw_percentiles':np.percentile(error,[25,50,75,90]).tolist()}
        rows.append(row)
    cap.release()
    path=ARCHIVE/'evidence/dynamic_4d_pipeline_audit.json';data=json.loads(path.read_text())
    data['camera']['real_static_reprojection_diagnostic']={'status':'PARTIAL','rows':rows,
        'method':'Source PCD every5th point, nearest-pixel zbuffer, released intrin.npy, exclude foreground/zero depth. Compare reader inverse(raw) with direct(raw). Scale1000 is a candidate, not certified unit.',
        'limitations':'Sparse sampling, quantization, occlusion, scene changes and unverified time pairing; no calibrated error threshold or GT labels created.'}
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--official-root',type=Path,required=True)
    parser.add_argument('--camera-only',action='store_true')
    args=parser.parse_args()
    if args.camera_only:camera_reprojection(args.root)
    else:run(args.root,args.official_root)
