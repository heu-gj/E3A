"""Read-only HOI4D P1 audit; writes evidence only inside reports/."""
from pathlib import Path
import json
import os
import sys
import pickle
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from e3a.data.hoi4d import (decode_hoi4d_motion_mask, hoi4d_color_palette,
    parse_hoi4d_action_segments, build_hoi4d_anticipation_target)

if not os.environ.get('HOI4D_ROOT'):
    raise SystemExit('Set HOI4D_ROOT to the local dataset root before running the audit.')
ROOT = Path(os.environ['HOI4D_ROOT'])
# Reproduction outputs may contain source data. Never publish this directory.
OUT = Path('reports/audits/P1-HOI4D-Physical-Data-Audit/evidence/local_only/reproduction')
RGB = ROOT / 'extracted/release/HOI4D_release'
ANN = ROOT / 'extracted/annotations/HOI4D_annotations'
DEP = ROOT / 'extracted/depth/HOI4D_depth_video'
HAND = ROOT / 'extracted/hand_pose/Hand_pose'

def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def scan():
    rows = []
    for video in RGB.glob('*/*/*/*/*/*/*/align_rgb/image.mp4'):
        seq = video.parent.parent.relative_to(RGB)
        a = ANN / seq
        paths = {'rgb': video, 'depth': DEP / seq / 'align_depth',
                 'mask': a / '2Dseg/mask', 'instance4d': a / '3Dseg',
                 'objpose': a / 'objpose', 'action': a / 'action',
                 'right': HAND / 'handpose_right_hand' / seq,
                 'left': HAND / 'handpose_left_hand' / seq}
        counts = {}
        for k,p in paths.items():
            counts[k] = (sum(1 for x in p.iterdir() if x.is_file()) if p.is_dir() else int(p.is_file()))
        rows.append({'sequence': str(seq), 'counts': counts,
                     'score': sum(counts[k] > 0 for k in counts),
                     'paths': {k:str(p) for k,p in paths.items()}})
    rows.sort(key=lambda r: (-r['score'], -min(r['counts'][k] for k in ['mask','objpose','right']), r['sequence']))
    dump('candidates.json', rows)
    print('candidates', len(rows), 'top', json.dumps(rows[:3], indent=2))

def audit():
    rows = json.loads((OUT / 'candidates.json').read_text())
    # All present modalities first, then minimum dense annotation coverage,
    # then combined hand coverage, then lexical sequence ID.
    rows.sort(key=lambda r: (-r['score'], -min(r['counts'][k] for k in ['mask','objpose','right','left']),
                             -sum(r['counts'][k] for k in ['right','left']), r['sequence']))
    dump('candidates.json', rows)
    r = rows[0]
    p = {k:Path(v) for k,v in r['paths'].items()}
    print('SELECTED', r['sequence'], flush=True)
    inventory = []
    for modality, path in p.items():
        for f in sorted(path.rglob('*') if path.is_dir() else [path]):
            if f.is_file():
                inventory.append({'modality':modality,'path':str(f),'bytes':f.stat().st_size})
    dump('inventory.json', inventory)
    result = {'sequence':r,'root':str(ROOT),'status':'PARTIAL', 'version':'UNKNOWN (local release directories; no version manifest)',
              'environment':{'python':sys.executable,'numpy':np.__version__,'opencv':cv2.__version__},
              'modalities':{k:{'count':sum(x['modality']==k for x in inventory),
                   'bytes':sum(x['bytes'] for x in inventory if x['modality']==k)} for k in p}}
    sample_ids = [0,75,150,224,299]
    videos = {}
    frames = {}
    for name, f in [('rgb',p['rgb']),('depth',p['depth']/'depth_video.avi')]:
        cap = cv2.VideoCapture(str(f))
        if name == 'depth':
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        meta = {'path':str(f),'opened':cap.isOpened(),'width':cap.get(3),'height':cap.get(4),
                'fps':cap.get(5),'reported_frames':cap.get(7),
                'fourcc':''.join(chr((int(cap.get(6)) >> (8*i)) & 255) for i in range(4))}
        count = 0
        timestamps = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            if count in sample_ids:
                frames[(name,count)] = frame
            count += 1
        cap.release()
        meta.update(decoded_frames=count,timestamps_ms=timestamps)
        videos[name]=meta
        print(name, {k:v for k,v in meta.items() if k!='timestamps_ms'}, flush=True)
    result['videos']=videos
    # Inspect static PCD fields and IDs; these are not frame-wise dynamic labels.
    f = p['instance4d']/'label.pcd'
    with f.open() as stream:
        header=[]
        for line in stream:
            header.append(line.strip())
            if line.startswith('DATA'): break
        points=np.loadtxt(stream)
    result['static_pcd']={'path':str(f),'header':header,'shape':list(points.shape),
                          'labels':np.unique(points[:,-2]).tolist(),'IDs':np.unique(points[:,-1]).tolist()}
    Kpath=ROOT/'extracted/camera_params/camera_params'/Path(r['sequence']).parts[0]/'intrin.npy'
    K=np.load(Kpath)
    result['intrinsics']={'path':str(Kpath),'matrix':K.tolist()}
    poses=[]
    for f in sorted(p['objpose'].glob('*.json'),key=lambda f:int(f.stem)):
        data=json.loads(f.read_text())
        objects=data.get('dataList',data.get('objects',[]))
        for obj in objects:
            rot=Rotation.from_euler('XYZ',[obj['rotation'][a] for a in 'xyz']).as_matrix()
            poses.append({'file_index':int(f.stem),'frameId':data.get('frameId'),'path':str(f),
                'root_keys':list(data),'object':obj,'orthogonality_error':float(np.linalg.norm(rot.T@rot-np.eye(3))),
                'determinant':float(np.linalg.det(rot))})
    dump('objpose_records.json',poses)
    result['objpose']={'count':len(poses),'frameId_minus_filename':sorted(set(x['frameId']-x['file_index'] for x in poses)),
        'ids':sorted(set(x['object']['id'] for x in poses)),
        'max_orthogonality_error':max(x['orthogonality_error'] for x in poses),
        'det_range':[min(x['determinant'] for x in poses),max(x['determinant'] for x in poses)],
        'sample':poses[0], 'coordinate_frame':'UNKNOWN','units':'UNKNOWN','transform_direction':'UNKNOWN',
        'representation':'Euler XYZ, radians per official SciPy reader; not quaternion or supplied 4x4',
        'physical_gt_status':'NEEDS_REVIEW'}
    hands={}
    hand_data={}
    for side in ['right','left']:
        records=[]
        for f in sorted(p[side].glob('*.pickle'),key=lambda f:int(f.stem)):
            # Only NumPy array reconstruction is allowed from the local pickle.
            class ArraysOnly(pickle.Unpickler):
                def find_class(self,module,name):
                    if (module,name) in {('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype')}:
                        return super().find_class(module,name)
                    raise pickle.UnpicklingError(f'Unexpected pickle global {module}.{name}')
            h=ArraysOnly(io.BytesIO(f.read_bytes()),encoding='latin1').load()
            hand_data[(side,int(f.stem))]=h
            records.append({'frame':int(f.stem),'path':str(f),'fields':{
                k:{'shape':list(np.asarray(v).shape),'dtype':str(np.asarray(v).dtype),
                   'finite':bool(np.isfinite(v).all())} for k,v in h.items()},'trans':h['trans'].tolist()})
        hands[side]={'count':len(records),'frames':[x['frame'] for x in records],
                     'missing_frames':sorted(set(range(300))-{x['frame'] for x in records}),
                     'sample':records[0] if records else None,
                     'beta_max_deviation':float(max(np.max(np.abs(hand_data[(side,x['frame'])]['beta']-hand_data[(side,records[0]['frame'])]['beta'])) for x in records)) if records else None}
        dump(f'mano_{side}_records.json',records)
    result['mano']=hands
    raw=json.loads((p['action']/'color.json').read_text())
    segments=parse_hoi4d_action_segments(raw,category=Path(r['sequence']).parts[2])
    checks=[]
    for t in sorted({0.,20.,*[i/15 for i in sample_ids],
                     *[max(0,s['start_time']+delta) for s in segments for delta in [-1e-8,0,1e-8]]}):
        if t>20: continue
        target=build_hoi4d_anticipation_target(segments,t,{})
        future=[s for s in segments if s['start_time']>t]
        expected=min((s['start_time'] for s in future),default=None)
        passed=target['time_valid']==(expected is not None) and (expected is None or abs(target['time_to_event']-(expected-t))<1e-10)
        checks.append({'cutoff':t,'expected_onset':expected,'target':target,'pass':passed})
    overlaps=[{'previous':a,'next':b,'seconds':a['end_time']-b['start_time']} for a,b in zip(segments,segments[1:]) if a['end_time']>b['start_time']]
    result['action']={'path':str(p['action']/'color.json'),'root_keys':list(raw),'raw':raw,
        'segments':segments,'parser_compatible':True,'boundary_checks':checks,'overlaps':overlaps,
        'duration_matches_video':raw['info']['duration']==videos['rgb']['decoded_frames']/videos['rgb']['fps'],
        'target_instance_id':'UNKNOWN; event.id is event ID, properties are empty'}
    stats=[]
    samples=[]
    palette=hoi4d_color_palette()
    for f in sorted(p['mask'].glob('*.png')):
        idx=int(f.stem)
        im=Image.open(f)
        mask=np.asarray(im.convert('RGB'))
        decoder_error=None
        unknown=np.zeros(mask.shape[:2],dtype=bool)
        try:
            sem,inst=decode_hoi4d_motion_mask(mask,Path(r['sequence']).parts[2])
        except ValueError as exc:
            decoder_error=str(exc)
            known=np.any(np.all(mask[:,:,None,:]==palette[None,None,:4,:],axis=-1),axis=-1)
            unknown=~known
            diagnostic_mask=mask.copy();diagnostic_mask[unknown]=0
            sem,inst=decode_hoi4d_motion_mask(diagnostic_mask,Path(r['sequence']).parts[2])
            sem[unknown]=-1;inst[unknown]=-1
        packed=mask[:,:,0].astype(np.uint32)*65536+mask[:,:,1].astype(np.uint32)*256+mask[:,:,2]
        colors=[[int(c>>16),int((c>>8)&255),int(c&255)] for c in np.unique(packed)]
        entry={'frame':idx,'path':str(f),'mode':im.mode,'shape':list(mask.shape),
               'colors':colors,'semantic_ids':np.unique(sem).tolist(),
               'raw_decoder_error':decoder_error,'unknown_pixels':int(unknown.sum()),
               'instances':[]}
        for iid in np.unique(inst):
            if iid<=0:continue
            y,x=np.where(inst==iid)
            entry['instances'].append({'provisional_id':int(iid),'semantic_ids':np.unique(sem[inst==iid]).tolist(),
                'pixels':len(x),'bbox_xyxy':[int(x.min()),int(y.min()),int(x.max()),int(y.max())]})
        stats.append(entry)
        if idx % 25 == 0: print('mask',idx,flush=True)
        if idx not in sample_ids: continue
        rgb=cv2.cvtColor(frames[('rgb',idx)],cv2.COLOR_BGR2RGB)
        depth=frames[('depth',idx)]
        valid=depth[depth>0]
        dstat={'dtype':str(depth.dtype),'shape':list(depth.shape),'min':int(depth.min()),'max':int(depth.max()),
               'zero_fraction':float(np.mean(depth==0)),'nonzero_percentiles':np.percentile(valid,[1,50,99]).tolist()}
        scaled=np.clip(depth.astype(float)/max(1,np.percentile(valid,99))*255,0,255).astype(np.uint8)
        if scaled.ndim==2: scaled=cv2.cvtColor(cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        overlay=rgb.copy()
        fg=inst>0
        overlay[fg]=(rgb[fg]*.5+mask[fg]*.5).astype(np.uint8)
        kpimage=rgb.copy()
        hand_match={}
        for side, iid in [('right',2),('left',3)]:
            h=hand_data.get((side,idx))
            if h is None:
                hand_match[side]=None
                continue
            kp=np.asarray(h['kps2D']).astype(int)
            in_bounds=(kp[:,0]>=0)&(kp[:,0]<inst.shape[1])&(kp[:,1]>=0)&(kp[:,1]<inst.shape[0])
            hit=np.zeros(21,dtype=bool)
            hit[in_bounds]=inst[kp[in_bounds,1],kp[in_bounds,0]]==iid
            hand_match[side]={'kps_in_hand_mask':int(hit.sum()),'total':21}
            for j,(x,y) in enumerate(kp):
                cv2.circle(kpimage,(int(x),int(y)),6,(255,0,255) if side=='right' else (0,255,255),-1)
                cv2.putText(kpimage,str(j),(int(x),int(y)),0,.4,(255,255,255),1)
        instance_rgb=palette[inst.astype(int)]
        instance_rgb[unknown]=(255,255,0)
        panels=[('RGB',rgb),('Depth raw pseudocolor',scaled),('Motion palette',mask),
                ('PROVISIONAL instance IDs',instance_rgb),('RGB + motion',overlay),('MANO kps2D',kpimage)]
        canvas=Image.new('RGB',(960,600),'white')
        draw=ImageDraw.Draw(canvas)
        for n,(label,array) in enumerate(panels):
            tile=Image.fromarray(array).resize((320,180))
            x=(n%3)*320; y=(n//3)*300
            canvas.paste(tile,(x,y+25));draw.text((x+5,y+5),f'{idx}: {label}',fill='black')
        canvas.save(OUT/f'frame_{idx:05d}_audit.jpg')
        for name,array in [('rgb',rgb),('depth_visual',scaled),('motion',mask),('instance_PROVISIONAL',instance_rgb),('overlay',overlay)]:
            Image.fromarray(array).save(OUT/f'frame_{idx:05d}_{name}.png')
        np.save(OUT/f'frame_{idx:05d}_depth_raw.npy',depth)
        samples.append({'frame':idx,'time_seconds':idx/videos['rgb']['fps'],'depth':dstat,'hand_mask_checks':hand_match,
            'mask':entry,'objpose':str(p['objpose']/f'{idx}.json'),
            'right_mano':str(p['right']/f'{idx}.pickle') if (p['right']/f'{idx}.pickle').exists() else None,
            'left_mano':str(p['left']/f'{idx}.pickle') if (p['left']/f'{idx}.pickle').exists() else None,
            'actions_at_time':[s for s in segments if s['start_time']<=idx/15<s['end_time']]})
        print('sample',idx,dstat,flush=True)
    dump('mask_statistics.json',stats)
    result['mask']={'inspected_frames':len(stats),'decoded_frames':sum(x['raw_decoder_error'] is None for x in stats),
                    'raw_decoder_failed_frames':[x['frame'] for x in stats if x['raw_decoder_error']],
                    'unknown_pixels':sum(x['unknown_pixels'] for x in stats),'semantic_ids':sorted({s for x in stats for s in x['semantic_ids']}),
                    'modes':sorted({x['mode'] for x in stats}),'dynamic_instance_gt':'NOT FOUND; 3Dseg is static scene',
                    'part_grouping_status':'NEEDS_REVIEW; C20 has only one object part, cannot test multi-part merging'}
    result['samples']=samples
    dump('audit_evidence.json',result)
    print('DONE',flush=True)

def finalize():
    import csv
    import hashlib
    e=json.loads((OUT/'audit_evidence.json').read_text())
    stats=json.loads((OUT/'mask_statistics.json').read_text())
    poses=json.loads((OUT/'objpose_records.json').read_text())
    p={k:Path(v) for k,v in e['sequence']['paths'].items()}
    seg=e['action']['segments']
    checks=[]
    for t in sorted({0.,20.,*[x['time_seconds'] for x in e['samples']],
                     *[max(0,s['start_time']+delta) for s in seg for delta in [-1e-8,0,1e-8]]}):
        target=build_hoi4d_anticipation_target(seg,t,{})
        expected=min((s['start_time'] for s in seg if s['start_time']>t),default=None)
        checks.append({'cutoff':t,'expected_onset':expected,'target':target,
                       'pass':target['time_valid']==(expected is not None) and
                              (expected is None or abs(target['time_to_event']-(expected-t))<1e-10)})
    e['action']['boundary_checks']=checks
    K=np.asarray(e['intrinsics']['matrix'])
    physical=[]
    for sample in e['samples']:
        i=sample['frame']
        mask=np.asarray(Image.open(OUT/f'frame_{i:05d}_motion.png').convert('RGB'))
        try:
            sem,inst=decode_hoi4d_motion_mask(mask,'C20')
        except ValueError:
            known=np.any(np.all(mask[:,:,None,:]==hoi4d_color_palette()[None,None,:4,:],axis=-1),axis=-1)
            diagnostic_mask=mask.copy();diagnostic_mask[~known]=0
            sem,inst=decode_hoi4d_motion_mask(diagnostic_mask,'C20')
            sem[~known]=-1;inst[~known]=-1
        d=np.load(OUT/f'frame_{i:05d}_depth_raw.npy')
        rgb=np.asarray(Image.open(OUT/f'frame_{i:05d}_rgb.png'))
        valid=(d>0)
        # Depth discontinuities exclude invalid-depth boundaries for this diagnostic.
        gx=np.zeros_like(d,dtype=float);gy=gx.copy()
        gx[:,1:]=np.abs(np.diff(d.astype(float),axis=1))
        gy[1:]=np.abs(np.diff(d.astype(float),axis=0))
        de=(np.maximum(gx,gy)>40)&cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        b=(inst==1).astype(np.uint8)
        boundary=cv2.morphologyEx(b,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
        dist=cv2.distanceTransform((~de).astype(np.uint8),cv2.DIST_L2,3)
        shifts=[]
        ys,xs=np.where(boundary)
        for dy in range(-16,17,4):
            for dx in range(-16,17,4):
                xx=np.clip(xs+dx,0,d.shape[1]-1); yy=np.clip(ys+dy,0,d.shape[0]-1)
                shifts.append({'dx':dx,'dy':dy,'median_distance_px':float(np.median(dist[yy,xx])),
                               'mean_distance_px':float(np.mean(dist[yy,xx]))})
        center=np.array([poses[i]['object']['center'][a] for a in 'xyz'])
        uv=K@center;uv=uv[:2]/uv[2]
        row={'frame':i,'project_center_as_camera_hypothesis_uv':uv.tolist(),
             'object_depth_raw_median':float(np.median(d[(inst==1)&valid])),
             'pose_center_raw':center.tolist(),
             'unshifted_edge_distance':next(x for x in shifts if x['dx']==0 and x['dy']==0),
             'best_grid_shift':min(shifts,key=lambda x:x['mean_distance_px']),
             'warning':'Diagnostic only. Depth threshold 40 raw units; sparse edges/occlusion bias shift; no calibrated transform inferred.'}
        physical.append(row)
        overlay=rgb.copy();overlay[de]=(0,255,255);overlay[boundary]=(255,0,255)
        Image.fromarray(overlay).resize((960,540)).save(OUT/f'frame_{i:05d}_depth_edges.jpg')
    e['physical_diagnostics']=physical
    with (OUT/'cross_modal_frames.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['frame','time_seconds','rgb','depth','mask','provisional_instance_ids','objpose','objpose_frameId','mano_right','mano_left','action','dynamic_instance_id'])
        writer.writeheader()
        for i in range(300):
            writer.writerow({'frame':i,'time_seconds':i/15,'rgb':str(p['rgb'])+f'#decoded_frame={i}',
                'depth':str(p['depth']/'depth_video.avi')+f'#decoded_frame={i}',
                'mask':str(p['mask']/f'{i:05d}.png'),
                'provisional_instance_ids':','.join(str(x['provisional_id']) for x in stats[i]['instances']),
                'objpose':str(p['objpose']/f'{i}.json'),'objpose_frameId':poses[i]['frameId'],
                'mano_right':str(p['right']/f'{i}.pickle') if i in e['mano']['right']['frames'] else 'MISSING',
                'mano_left':str(p['left']/f'{i}.pickle') if i in e['mano']['left']['frames'] else 'MISSING',
                'action':'|'.join(s['event'] for s in seg if s['start_time']<=i/15<s['end_time']),
                'dynamic_instance_id':'UNKNOWN'})
    e['index_validation']={'mask_indices_complete':sorted(x['frame'] for x in stats)==list(range(300)),
        'objpose_indices_complete':sorted(x['file_index'] for x in poses)==list(range(300)),
        'annotation_timestamp':'UNKNOWN; masks and MANO have filenames only; objpose.frameId uses mixed numbering and must not be substituted for filename/video index',
        'video_timestamps_equal':np.allclose(e['videos']['rgb']['timestamps_ms'],e['videos']['depth']['timestamps_ms']),
        'sensor_dropped_frames':'UNKNOWN; encoded continuity does not prove original sensor continuity'}
    e['sources']={
        'instructions':'https://github.com/leolyliu/HOI4D-Instructions',
        'decode':'https://github.com/hoi4d/HOI4D_Script/blob/main/utils/decode.py',
        'palette':'https://github.com/hoi4d/HOI4D_Script/blob/main/utils/color_map.py',
        'parts':'https://github.com/hoi4d/HOI4D_Script/blob/main/definitions/motion%20segmentation/label.csv',
        'instance_map':'https://github.com/hoi4d/HOI4D_Script/blob/main/prepare_4Dseg/utils.py',
        'dynamic_conversion':'https://github.com/hoi4d/HOI4D_Script/blob/main/prepare_4Dseg/scene2frame.py',
        'mano_layer':'https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py',
        'mano_license':'https://mano.is.tue.mpg.de/'}
    e['conclusions']={
        'video_decode':{'status':'PASS','evidence':'videos: both decoded 300, 15 FPS, 1920x1080'},
        'palette':{'status':'PARTIAL','evidence':'mask_statistics.json: real masks contain colors outside C20 built-in mapping; raw decoder fails closed; diagnostics retain unknown pixels as -1, never GT'},
        'alignment':{'status':'PARTIAL','evidence':'5 audit panels + physical_diagnostics; content corresponds, pixel-exact calibration and annotation timestamps UNKNOWN'},
        'part_to_dynamic_instance':{'status':'NEEDS_REVIEW','evidence':'C20 contains only chair part; no dynamic instance files in selected inventory; static PCD IDs are a different namespace'},
        'object_pose':{'status':'NEEDS_REVIEW','evidence':'objpose_records.json: Euler SO(3) numerical checks pass; physical coordinate/unit/direction/mask ID correspondence UNKNOWN'},
        'mano_parameters':{'status':'PASS','evidence':'mano_right_records.json and mano_left_records.json: actual fields and finite values'},
        'mano_21_joint_reconstruction':{'status':'BLOCKED','evidence':'manopth and smplx absent in selected environment; no supplied licensed model path; forward/reprojection not run'},
        'action_time':{'status':'PASS','evidence':'action.raw.info.duration=20; both streams 300/15=20; parser accepts 14 events; all strict-future boundary checks pass. This is timeline consistency, not frame-accurate manual event validation.'},
        'hand_entity_action_chain':{'status':'BLOCKED','evidence':'No source-backed event-to-object target ID or contact label; 3D hand reconstruction/coordinate validation incomplete'}}
    e['dependencies']={'missing':['ffmpeg/ffprobe in PATH','manopth','smplx'],
        'MANO_models':'UNKNOWN availability; no authorized model path supplied, no global disk search or download',
        'tests':'26 passed in 1.72s (HOI4D/actions/entity collate/E3A collate); synthetic tests do not prove physical correctness'}
    e['code_review']={'read':['docs/architecture/overview.md','docs/implementation/roadmap.md',
        'docs/implementation/training.md','src/e3a/data/hoi4d.py','src/e3a/data/actions.py',
        'src/e3a/data/collate.py','tests/test_hoi4d_data.py'],
        'parser_only':['HOI4D palette category mappings (provisional IDs)','action JSON variants','clip target builders','batch collation'],
        'contract':{'entity':'labels[N], ids[N,T], masks[N,T,H,W], mask_valid[N,T]',
                    'hands':'keypoints_camera[T,2,21,3]; hand axis Right, Left; wrist/thumb/index/middle/ring/pinky',
                    'missing':'false validity, not negative targets'},
        'missing':['raw Dataset/DataLoader','MANO forward and verified reprojection','objpose reader with verified coordinate transform',
                   'clip-level entity supervision validity','verified target/contact source labels']}
    e['loss_readiness']={
        'entity':{'status':'PARTIAL','gate':'quarantine unknown-color frame; persistent identity and empty-window policy require validation'},
        'hand':{'status':'BLOCKED','gate':'licensed MANO assets + pinned forward + source reprojection'},
        'geometry':{'status':'NEEDS_REVIEW','gate':'depth physical units and registration; coordinate direction'},
        'interaction':{'status':'BLOCKED','gate':'no source-backed contact annotations'},
        'target_entity':{'status':'BLOCKED','gate':'no action target object ID'},
        'action_time':{'status':'PARTIAL','gate':'timeline parser validated; train-only vocabulary, splits and leak-free sampling pending'}}
    e['recommended_files']={'src/e3a/data/hoi4d.py':'versioned palette schema/unknown policy; verified instance and frame-index joins',
        'src/e3a/data/hoi4d_dataset.py (new)':'separate source roots, video reader, coverage and event windows',
        'src/e3a/data/hoi4d_geometry.py (new)':'licensed MANO and objpose coordinate adapters after validation',
        'src/e3a/data/collate.py':'clip-level entity_supervision_valid with criterion integration',
        'tests/test_hoi4d_data.py':'real-release opt-in fixtures including frame220 unknown color, mixed raw frameId numbering, depth uint16',
        'docs/implementation/roadmap.md':'update only verified acceptance gates',
        'docs/implementation/training.md':'record source-backed supervision availability',
        'src/e3a/data/actions.py':'no parser defect observed; build actual vocabulary from audited training split'}
    e['next_dataset_recommendations']=['freeze release manifest and source fingerprints','quarantine frame220 until unknown semantics reviewed',
        'retain source filename index, frameId and timestamps separately','cache only verified sample metadata; no future frames in model input',
        'false validity for unavailable hand/target/contact labels','audit subject/source-video splits before generating training vocabulary']
    for failed in e['mask']['raw_decoder_failed_frames']:
        mask=np.asarray(Image.open(p['mask']/f'{failed:05d}.png').convert('RGB'))
        unknown=~np.any(np.all(mask[:,:,None,:]==hoi4d_color_palette()[None,None,:4,:],axis=-1),axis=-1)
        yy,xx=np.where(unknown)
        x0=max(0,int(xx.min())-25);x1=min(mask.shape[1],int(xx.max())+26)
        y0=max(0,int(yy.min())-25);y1=min(mask.shape[0],int(yy.max())+26)
        marked=mask.copy();marked[unknown]=(255,255,0)
        Image.fromarray(marked).save(OUT/f'unknown_frame_{failed:05d}.png')
        Image.fromarray(marked[y0:y1,x0:x1]).resize((500,500)).save(OUT/f'unknown_frame_{failed:05d}_crop.png')
        e.setdefault('unknown_color_regions',[]).append({'frame':failed,'pixels':int(unknown.sum()),
            'bbox_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
            'colors':np.unique(mask[unknown],axis=0).tolist(),'semantic':'UNKNOWN','instance':'UNKNOWN'})
    e['source_fingerprints']=[]
    for f in [p['rgb'],p['depth']/'depth_video.avi',p['action']/'color.json',p['objpose']/'0.json',p['mask']/'00000.png',Kpath_from(e)]:
        digest=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        e['source_fingerprints'].append({'path':str(f),'bytes':f.stat().st_size,'sha256':digest.hexdigest()})
    dump('audit_evidence.json',e)
    print('FINALIZED local-only evidence; review and sanitize before publication.')

def Kpath_from(e):
    return Path(e['intrinsics']['path'])

if __name__ == '__main__':
    if '--finalize' in sys.argv: finalize()
    elif '--audit' in sys.argv: audit()
    else: scan()
