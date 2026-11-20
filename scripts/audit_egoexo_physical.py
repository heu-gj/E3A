"""Read-only Ego-Exo4D audit. No production adapter or source data writes."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from e3a.data.egoexo import (resolve_egoexo_video_path,decode_egoexo_relation_mask,
    build_egoexo_relation_clip_targets,EGOEXO_HAND_JOINTS,EGOEXO_HAND_ORDER)

BASE='bade1b8d68205335ce039490ded919965216fe70'
TASK='P1-EGOEXO4D-Physical-Data-Audit'
OUT=Path('reports/audits')/TASK


def save(name,data):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


class JSONStream:
    """Bounded per-value stdlib JSON reader; does not load a whole split."""
    def __init__(self,stream,chunk=1024*1024):
        self.stream=stream;self.chunk=chunk;self.buf='';self.eof=False
        self.decoder=json.JSONDecoder()

    def refill(self):
        block=self.stream.read(self.chunk)
        self.buf+=block
        if not block:self.eof=True

    def whitespace(self):
        self.buf=self.buf.lstrip()
        while not self.buf and not self.eof:self.refill();self.buf=self.buf.lstrip()

    def expect(self,char):
        self.whitespace()
        if not self.buf.startswith(char):raise ValueError(f'Expected JSON delimiter {char}')
        self.buf=self.buf[1:]

    def value(self):
        self.whitespace()
        while True:
            try:
                value,end=self.decoder.raw_decode(self.buf)
                self.buf=self.buf[end:]
                return value
            except json.JSONDecodeError:
                if self.eof:raise
                self.refill()


def iter_relation_takes(stream,chunk=1024*1024):
    parser=JSONStream(stream,chunk)
    parser.expect('{')
    while True:
        parser.whitespace()
        if parser.buf.startswith('}'):raise ValueError('Missing annotations mapping')
        key=parser.value();parser.expect(':')
        if key!='annotations':
            parser.value();parser.expect(',');continue
        parser.expect('{')
        while True:
            parser.whitespace()
            if parser.buf.startswith('}'):return
            uid=parser.value();parser.expect(':');value=parser.value()
            yield uid,value
            parser.whitespace()
            if parser.buf.startswith('}'):return
            parser.expect(',')


def is_hand(name):
    return re.fullmatch(r'(?:(?:left|right) )?hand(?: \d+)?',re.sub(r'[_\s]+',' ',name.lower()).strip()) is not None


def coverage(record):
    annotated={int(x) for x in record.get('annotated_frames',[])}
    encoded={int(x) for x in record.get('annotation',{})}
    return {'annotated_count':len(annotated),'encoded_count':len(encoded),
            'explicit_absence_count':len(annotated-encoded),
            'mask_outside_coverage':sorted(encoded-annotated),
            'first_frame':min(annotated) if annotated else None,'last_frame':max(annotated) if annotated else None,
            'annotation_fps':record.get('annotation_fps')}


def classify_frame(record,frame):
    if frame not in {int(x) for x in record['annotated_frames']}:return 'UNANNOTATED'
    return 'POSITIVE_RECORD' if str(frame) in record['annotation'] else 'EXPLICIT_ABSENCE'


def metadata_video_paths(root,take):
    rows=[]
    for cam,streams in take.get('frame_aligned_videos',{}).items():
        for stream,record in streams.items():
            if not (cam.startswith('cam') or stream=='rgb'):continue
            expected=root/take['root_dir']/record['relative_path']
            low=root/take['root_dir']/'frame_aligned_videos/downscaled/448'/Path(record['relative_path']).name
            found=expected if expected.is_file() else (low if low.is_file() else None)
            rows.append({'camera':cam,'stream':stream,'metadata_path':str(expected.relative_to(root)),
                         'actual_path':str(found.relative_to(root)) if found else None,
                         'metadata_path_exists':expected.is_file(),
                         'bytes':found.stat().st_size if found else None,
                         'mapping':'exact metadata' if expected.is_file() else 'explicit local downscaled/448 fallback (audit only)'})
    return rows


def inventory(root):
    takes=json.loads((root/'raw/takes.json').read_text())
    metadata=json.loads((root/'raw/metadata.json').read_text())
    files=[p for p in root.rglob('*') if p.is_file()]
    source_json=[{'dataset_path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in files if p.suffix=='.json' and p.stat().st_size<100_000_000]
    save('evidence/dataset_inventory.json',{'metadata_version':metadata.get('version'),'metadata_date':metadata.get('date'),
        'metadata_take_count':len(takes),'local_file_count':len(files),'suffix_counts':dict(Counter(p.suffix for p in files)),
        'json_files':source_json,'large_annotation_files':[{'dataset_path':str(p.relative_to(root)),'bytes':p.stat().st_size,
            'sha256_status':'NOT_RUN (large source; published artifacts hashed separately)'} for p in files if p.suffix=='.json' and p.stat().st_size>=100_000_000],
        'missing_modalities':['EgoPose manual/automatic 2D and3D JSON','EgoPose camera poses','Raw Keystep annotations/taxonomy/vocabulary','keystep_benchmark','MPS/timesync/calibration'],
        'search_scope':'Only selected dataset root recursively and immediate dataset-parent candidate names; no disk-wide search',
        'annotation_revision':'Relations ds inspected from source header; metadata v2 does not independently version every annotation'})
    rows=[]
    for take in takes:
        videos=metadata_video_paths(root,take)
        rows.append({'take_uid':take['take_uid'],'take_name':take['take_name'],
            'duration_sec':take.get('duration_sec'),'task_start_sec':take.get('task_start_sec'),'task_end_sec':take.get('task_end_sec'),
            'aria_videos':sum(v['actual_path'] is not None and v['stream']=='rgb' for v in videos),
            'exo_videos':sum(v['actual_path'] is not None and v['camera'].startswith('cam') for v in videos),
            'ego_pose':0,'camera_pose':0,'raw_keystep':0,'relation_tracks':0,'relation_aria_masks':0,
            'nonhand_tracks':0,'relation_split':None,'videos':videos})
    byuid={r['take_uid']:r for r in rows}
    splitstats=[]
    for path in sorted((root/'raw/annotations').glob('relations_*.json')):
        count=0;match=0;ds=None
        with path.open() as stream:
            prefix=stream.read(120);ds=re.search(r'"ds"\s*:\s*"([^"]+)"',prefix).group(1);stream.seek(0)
            for uid,relation in iter_relation_takes(stream):
                count+=1
                if uid not in byuid:continue
                match+=1;r=byuid[uid];tracks=relation.get('object_masks',{})
                r['relation_split']=path.name
                r['relation_tracks']=len(tracks)
                r['nonhand_tracks']=sum(not is_hand(name) for name in tracks)
                r['relation_aria_masks']=sum(len(c.get('annotation',{})) for name,cams in tracks.items() if not is_hand(name)
                                            for camera,c in cams.items() if camera.startswith('aria'))
                if count%100==0:print(path.name,count,flush=True)
        splitstats.append({'dataset_path':str(path.relative_to(root)),'ds':ds,'takes':count,'matched_takes_metadata':match})
        print('split done',path.name,count,flush=True)
    # Prefer available Aria+exo and at least2 nonhand tracks; then annotation coverage.
    eligible=[r for r in rows if r['aria_videos'] and r['exo_videos'] and r['nonhand_tracks']>=2]
    eligible.sort(key=lambda r:(-r['relation_aria_masks'],-r['nonhand_tracks'],r['take_uid']))
    assert eligible,'No take with locally available ego/exo and multiple objects'
    selected=eligible[0]
    save('evidence/candidate_selection_summary.json',{'status':'PARTIAL','all_modality_take_count':0,
        'selection_rule':'Available Aria RGB+exo+>=2 nonhand tracks, then descending encoded Aria masks, track count, lexical UID',
        'eligible_count':len(eligible),'splits':splitstats,'selected':selected,
        'top_candidates':[{k:v for k,v in r.items() if k!='videos'} for r in eligible[:20]],
        'missing_modalities_are_not_zero_valid_joints':'Missing files; all hand/geometry/keystep measurements NOT_RUN'})
    auxiliary_metadata(root,takes,selected)
    with (OUT/'evidence/take_inventory.csv').open('w',newline='') as stream:
        fields=[k for k in rows[0] if k!='videos'];writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    # Selected raw annotations stay local-only. Never publish encoded masks or annotator IDs.
    target=root/'raw/annotations'/selected['relation_split']
    with target.open() as stream:
        for uid,record in iter_relation_takes(stream):
            if uid==selected['take_uid']:
                save('evidence/local_only/selected_relation.json',record);break
    print('SELECTED',json.dumps({k:v for k,v in selected.items() if k!='videos'}),flush=True)


def auxiliary_metadata(root,takes,selected):
    take=next(t for t in takes if t['take_uid']==selected['take_uid'])
    save('evidence/mps_availability.json',{'status':'BLOCKED','evidence_level':'REAL_METADATA_AND_SOURCE_PATH_EXISTENCE',
        'take_uid':take['take_uid'],'has_trimmed_trajectory_metadata':take.get('has_trimmed_trajectory'),
        'has_trimmed_vrs_metadata':take.get('has_trimmed_vrs'),
        'actual_paths':[{'dataset_path':str(Path(take['root_dir'])/take[k]),
                        'exists':(root/take['root_dir']/take[k]).exists()}
                       for k in ['vrs_relative_path','trimmed_eye_gaze_relative_path','trimmed_trajectory_relative_path']],
        'camera_record_fields':sorted({k for c in take['capture']['cameras'] for k in c}),
        'camera_records_contain_intrinsics_extrinsics':any('intrinsic' in k or 'extrinsic' in k for c in take['capture']['cameras'] for k in c),
        'reason':'Metadata advertises assets; actual files absent. Camera records contain device/path descriptors, not calibration matrices.'})
    records=[]
    for t in takes:
        candidates=[v for v in metadata_video_paths(root,t) if v['camera']=='aria01' and v['stream']=='rgb' and v['actual_path']]
        if not candidates:continue
        v=candidates[0]
        expected=resolve_egoexo_video_path(root,takes,t['take_uid'],'aria01_214-1',require_exists=False)
        cap=cv2.VideoCapture(str(root/v['actual_path']));ok,frame=cap.read()
        records.append({'take_uid':t['take_uid'],'take_name':t['take_name'],'camera_argument':'aria01_214-1',
            'metadata_lookup':'aria01/rgb','resolved_metadata_path':str(expected.relative_to(root)),
            'actual_path':v['actual_path'],'metadata_file_exists':expected.is_file(),
            'actual_frame0_read_ok':ok,'actual_frame0_shape':list(frame.shape) if ok else None,'fps':cap.get(5)})
        cap.release();break
    save('evidence/resolver_alias_audit.json',{'status':'PARTIAL',
        'evidence_level':'REAL_TAKES_METADATA_AND_ACTUAL_LOW_RES_VIDEO_OPEN','aria01_records':records,
        'selected_take_aria':'aria04_214-1 -> aria04/rgb tested in video_alignment.json',
        'conclusion':'Stream alias selects correct metadata. Actual downscaled path remains unsupported by production require_exists=True.'})


def audit(root,sdk):
    sys.path.append(str(sdk))
    from ego4d.research.util.masks import decode_mask
    import ego4d.research.util.masks as sdk_masks
    import ego4d.research.util.lzstring as sdk_lz
    selected=json.loads((OUT/'evidence/candidate_selection_summary.json').read_text())['selected']
    record=json.loads((OUT/'evidence/local_only/selected_relation.json').read_text())
    takes=json.loads((root/'raw/takes.json').read_text());take=next(t for t in takes if t['take_uid']==selected['take_uid'])
    tracks=record['object_masks'];stats=[]
    for name,cameras in tracks.items():
        for camera,c in cameras.items():
            stats.append({'object_name':name,'is_hand':is_hand(name),'camera':camera,**coverage(c),
                          'dimension_counts':dict(Counter(f"{x['width']}x{x['height']}" for x in c['annotation'].values()))})
    save('evidence/relation_coverage.json',{'status':'PASS','scope':'Track metadata/coverage sets, not all masks decoded',
        'take_uid':selected['take_uid'],'source_path':'raw/annotations/'+selected['relation_split'],
        'take_keys':list(record),'object_tracks':list(tracks),'track_camera_records':stats,
        'outside_annotated_frames_total':sum(len(r['mask_outside_coverage']) for r in stats),
        'cross_camera_identity':'Source track names group cameras, but independent physical ID validation NOT_RUN'})
    videos=[];decoded=[];frames_table=[];camera_aria=next(v['camera']+'_214-1' for v in selected['videos'] if v['stream']=='rgb' and v['actual_path'])
    relevant=[v for v in selected['videos'] if v['actual_path'] and (v['stream']=='rgb' or v['camera']==take.get('best_exo'))]
    if not any(v['camera'].startswith('cam') for v in relevant):relevant.append(next(v for v in selected['videos'] if v['actual_path'] and v['camera'].startswith('cam')))
    for v in relevant:
        path=root/v['actual_path'];name=v['camera']+'_214-1' if v['stream']=='rgb' else v['camera']
        try:
            resolved=resolve_egoexo_video_path(root,takes,selected['take_uid'],name)
            resolver={'status':'PASS','dataset_path':str(resolved.relative_to(root))}
        except FileNotFoundError:
            resolver={'status':'BLOCKED','reason':'Metadata full-resolution path absent; downloaded downscaled path is not resolved by current adapter'}
        cap=cv2.VideoCapture(str(path));n=int(cap.get(7));fps=cap.get(5);w=int(cap.get(3));h=int(cap.get(4))
        sample_ids=sorted({0,round((n-1)*.25),round((n-1)*.5),round((n-1)*.75),n-1})
        mask_camera=name
        available=sorted({int(f) for cams in tracks.values() for c,cr in cams.items() if c==mask_camera for f in cr['annotation']})
        mask_samples=sorted({min(available,key=lambda x:abs(x-i)) for i in sample_ids}) if available else []
        samples=[]
        for idx in sorted(set(sample_ids+mask_samples)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,bgr=cap.read()
            row={'camera':name,'decoded_index_requested':idx,'read_ok':ok,
                'actual_pos_frames_after_read':cap.get(cv2.CAP_PROP_POS_FRAMES),
                'pts_seconds':cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                'role':'whole-video temporal sample' if idx in sample_ids else 'nearest annotated mask frame (decoded at its own index)',
                'ego_pose':'MISSING','camera_pose':'MISSING','keystep':'MISSING','target_entity_valid':False,'interaction_valid':False}
            frames_table.append(row)
            if not ok:continue
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);overlay=rgb.copy();maskstats=[]
            for ti,(obj,cams) in enumerate(tracks.items()):
                if name not in cams:continue
                c=cams[name];mask=c['annotation'].get(str(idx))
                state=classify_frame(c,idx)
                if mask is None:
                    maskstats.append({'object':obj,'state':state});continue
                raw=np.asarray(decode_mask(dict(mask)));via_adapter=decode_egoexo_relation_mask(mask)
                assert np.array_equal(raw.astype(bool),via_adapter)
                yy,xx=np.where(raw>0)
                item={'object':obj,'frame':idx,'camera':name,'state':state,'source_shape':[mask['height'],mask['width']],
                      'decoded_shape':list(raw.shape),'nonzero_pixels':len(xx),'empty':not bool(len(xx)),
                      'adapter_sdk_equal':True,'scale_x':w/mask['width'],'scale_y':h/mask['height'],
                      'bbox_xyxy_source':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None}
                decoded.append(item);maskstats.append(item)
                resized=cv2.resize(raw.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
                color=np.array([(ti*73+80)%255,(ti*113+150)%255,(ti*47+60)%255])
                overlay[resized]=(overlay[resized]*.55+color*.45).astype(np.uint8)
                if resized.any():
                    y,x=np.where(resized);cv2.putText(overlay,obj,(int(x.min()),int(y.min())),0,.35,(255,255,255),1)
            canvas=Image.new('RGB',(w*2,h+30),'white');canvas.paste(Image.fromarray(rgb),(0,30));canvas.paste(Image.fromarray(overlay),(w,30))
            ImageDraw.Draw(canvas).text((4,4),f'{name} decoded frame {idx}: RGB | scaled relation overlay',fill='black')
            vis=OUT/'visualizations/local_only'/f'{name}_{idx:06d}.jpg';vis.parent.mkdir(parents=True,exist_ok=True);canvas.save(vis)
            samples.append(row|{'mask_records':maskstats,'local_only_visualization':str(vis.relative_to(OUT))})
        cap.release()
        videos.append(v|{'camera_annotation_name':name,'fps':fps,'reported_frame_count':n,'duration_container_seconds':n/fps if fps else None,
                         'width':w,'height':h,'resolver':resolver,'sampled_frames':samples,
                         'whole_video_decode':'NOT_RUN; bounded seek samples only'})
        print('video',name,n,fps,w,h,'samples',len(samples),flush=True)
    # Real adapter validity test for two object tracks, bounded3 frame window.
    names=[name for name,cams in tracks.items() if not is_hand(name) and camera_aria in cams][:2]
    maskframes=sorted({int(f) for name in names for f in tracks[name][camera_aria]['annotation']})
    chosen=maskframes[:1]+[maskframes[len(maskframes)//2],maskframes[-1]]
    chosen=sorted(set(chosen));subset={n:tracks[n] for n in names}
    try:
        targets=build_egoexo_relation_clip_targets({'annotations':{selected['take_uid']:{'object_masks':subset}}},
            selected['take_uid'],camera_aria,chosen)
        adapter={'status':'PASS','source_tracks':names,'frames':chosen,'track_names':targets['track_names'],
                 'ids':targets['ids'].tolist(),'mask_valid':targets['mask_valid'].tolist(),'mask_shape':list(targets['masks'].shape)}
    except ValueError as error:adapter={'status':'BLOCKED','error':str(error)}
    save('evidence/relation_mask_decode.json',{'status':'PASS' if decoded else 'BLOCKED',
        'scope':'Bounded sampled encoded records, not full take-mask validation; overlay alignment reviewed separately',
        'sdk_version':importlib.metadata.version('ego4d'),'sdk_source_checksums':{
            'ego4d/research/util/masks.py':hashlib.sha256(Path(sdk_masks.__file__).read_bytes()).hexdigest(),
            'ego4d/research/util/lzstring.py':hashlib.sha256(Path(sdk_lz.__file__).read_bytes()).hexdigest()},
        'dependency_strategy':'Existing pure-Python SDK files with existing audit NumPy/pycocotools; no installation or custom decoder',
        'decoded_records':len(decoded),'empty_sampled_records':sum(x['empty'] for x in decoded),'records':decoded,
        'clip_adapter':adapter,'all_encoded_masks_checked':False})
    save('evidence/video_alignment.json',{'status':'PARTIAL','take_uid':selected['take_uid'],
        'takes_source':'raw/takes.json','metadata_duration_sec':take.get('duration_sec'),
        'task_start_sec':take.get('task_start_sec'),'task_end_sec':take.get('task_end_sec'),
        'task_timing_is_not_keystep':'Do not invent missing segment labels from take timing',
        'camera_aria':camera_aria,'videos':videos,
        'alignment_caveat':'annotation_fps is sampling cadence, not a frame-index divisor. Source frame keys tested as candidate indices; scaling/temporal verification remains partial without full-resolution/timesync/camera poses.',
        'multiview_policy':'Aria is primary temporal stream; exo is a separate camera, never concatenated into T'})
    with (OUT/'evidence/cross_modal_frames.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(frames_table[0]),lineterminator='\n');writer.writeheader();writer.writerows(frames_table)
    missing_reports(selected)


def missing_reports(selected):
    missing={'status':'BLOCKED','execution_status':'NOT_RUN','evidence_level':'DATASET_FILE_INVENTORY',
        'evidence_file':'evidence/dataset_inventory.json','reason':'Required real annotation files absent from checked local dataset root'}
    save('evidence/ego_pose_coverage.json',missing|{'canonical_order':{'hands':EGOEXO_HAND_ORDER,'joints':EGOEXO_HAND_JOINTS},
        'manual_2d':None,'automatic_2d':None,'manual_3d':None,'automatic_3d':None,'per_joint_validity':None,'per_frame_view_counts':None,
        'record_multiplicity':'UNKNOWN; frame_records[0] can only be flagged as potential loss, not a confirmed local multi-record bug'})
    save('evidence/camera_pose_validation.json',missing|{'intrinsics':None,'extrinsics':None,'SO3':None,
        'coordinate_direction':'UNKNOWN on local release','physical_units':'UNKNOWN','source_to_model_units_1':'Preserves source values; not unit certification',
        'MPS':'NOT_FOUND'})
    save('evidence/hand_reprojection.json',missing|{'current_direction':{'median_error':None,'p90_error':None},
        'inverse_direction':{'median_error':None,'p90_error':None},'per_camera':None,'manual_vs_automatic':None,
        'distortion_resize_crop':'Downscaled videos found; no hand/camera records to verify these transforms',
        'forbidden_substitute':'No synthetic points or zero errors used for missing observations'})
    save('evidence/keystep_validation.json',missing|{'raw_keystep':'NOT_FOUND','keystep_benchmark':'NOT_FOUND',
        'parser_compatibility':'NOT_RUN','vocabulary_membership':'NOT_RUN','taxonomy_parent_links':'NOT_RUN',
        'overlaps_gaps_duration':'NOT_RUN','event_boundary_checks':'NOT_RUN'})
    (OUT/'evidence/keystep_timeline.csv').write_text('take_uid,start_time,end_time,step_id,step_unique_id,status\n'+selected['take_uid']+',UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,BLOCKED_MISSING_RAW_KEYSTEP\n')
    readiness={
        'Entity':{'status':'PARTIAL','reason':'Real sampled mask decode/coverage validated; full-resolution path mismatch and downscaled alignment require explicit adapter policy'},
        'Hand':{'status':'BLOCKED','reason':'No real EgoPose files'},
        'Geometry':{'status':'BLOCKED','reason':'No camera poses/intrinsics/MPS or hand reprojection'},
        'Interaction':{'status':'BLOCKED','reason':'No validated contact labels; overlap is not contact'},
        'Target':{'status':'BLOCKED','reason':'No local raw keystep and no verified event-to-track mapping'},
        'Action':{'status':'BLOCKED','reason':'Raw Keystep absent; synthetic parser tests are not real compatibility checks'},
        'Time':{'status':'BLOCKED','reason':'No source future-event onsets; take timing is not event GT'}}
    save('evidence/supervision_readiness.json',{'status':'PARTIAL','tasks':readiness,
        'target_entity_valid':False,'interaction_valid':False,'Hand_to_Entity':'BLOCKED: no 3D hand source',
        'Entity_to_Future_Action':'BLOCKED: no raw event-to-object evidence',
        'role_assignment':'Object names alone do not prove tool/patient/container/background roles; all event-specific roles UNKNOWN'})


def fingerprints(root):
    path=OUT/'evidence/dataset_inventory.json';inv=json.loads(path.read_text())
    def digest_file(p):
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        return h.hexdigest()
    for record in inv['large_annotation_files']:
        record['sha256']=digest_file(root/record['dataset_path'])
        record['sha256_status']='PASS: source byte fingerprint only'
        print('fingerprint',record['dataset_path'],flush=True)
    selected=json.loads((OUT/'evidence/candidate_selection_summary.json').read_text())['selected']
    take_root=root/'takes'/selected['take_name']
    inv['selected_take_files']=[]
    for p in sorted(take_root.rglob('*')):
        if p.is_file():
            inv['selected_take_files'].append({'dataset_path':str(p.relative_to(root)),'bytes':p.stat().st_size,
                'sha256':digest_file(p)})
            print('fingerprint',p.name,flush=True)
    save('evidence/dataset_inventory.json',inv)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['inventory','audit','fingerprints'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--sdk-site',type=Path)
    args=parser.parse_args()
    if args.phase=='inventory':inventory(args.root)
    elif args.phase=='fingerprints':fingerprints(args.root)
    else:
        if args.sdk_site is None:parser.error('--sdk-site is required for existing official decoder')
        audit(args.root,args.sdk_site)
