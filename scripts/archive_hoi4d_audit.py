"""Consolidate existing P1 results; never opens or changes the source dataset."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone

TASK = 'P1-HOI4D-Physical-Data-Audit'
DEST = Path('reports/audits') / TASK
BASE = '9f96d5813655e0fb9ceebad107b2ff2d1d2d5d07'
BRANCH = 'audit/p1-hoi4d-physical-data'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def prepare():
    old = Path('reports/hoi4d_audit')
    doc = Path('docs/data_audit/hoi4d_single_sequence.json')
    assert doc.read_bytes() == (old / 'audit_evidence.json').read_bytes()
    e = json.loads(doc.read_text())
    root = e['root']
    interpreter = e['environment']['python']
    inventory = []
    for folder in [old, Path('docs/data_audit')]:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.name != 'REVIEW_PROTOCOL.md':
                inventory.append({'source_repo_path':f.as_posix(), 'bytes':f.stat().st_size,
                                  'sha256_before_sanitization':digest(f)})

    def clean_string(value):
        return value.replace(root + '/', '').replace(root, '.').replace(interpreter, 'python')

    def clean(value):
        if isinstance(value, dict):
            return {clean_string(k):clean(v) for k,v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return clean_string(value) if isinstance(value, str) else value

    for folder in ['evidence/local_only','visualizations/local_only','logs']:
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    # Frozen source snapshots are local-only, not maintained final reports.
    for name in ['audit_evidence.json','objpose_records.json','mano_right_records.json','mano_left_records.json']:
        src = old / name
        data = clean(json.loads(src.read_text()))
        write_json(src, data)
        src.rename(DEST / 'evidence/local_only' / name)
    for f in list(old.iterdir()):
        if f.suffix in {'.png','.jpg','.npy'}:
            sub = 'evidence/local_only' if f.suffix == '.npy' else 'visualizations/local_only'
            f.rename(DEST / sub / f.name)
    for name in ['candidates.json','inventory.json','mask_statistics.json']:
        src=old/name
        write_json(src,clean(json.loads(src.read_text())))
    csvpath=old/'cross_modal_frames.csv'
    csvpath.write_text(clean_string(csvpath.read_text()))
    # Publish schema/numerical validation, not a full copy of source GT annotations.
    poses=json.loads((DEST/'evidence/local_only/objpose_records.json').read_text())
    write_json(DEST/'evidence/objpose_statistics.json',[
        {k:r[k] for k in ['file_index','frameId','path','root_keys','orthogonality_error','determinant']}
        | {'object_keys':list(r['object']), 'id':r['object']['id'], 'label':r['object']['label']}
        for r in poses])
    for side in ['right','left']:
        data=json.loads((DEST/f'evidence/local_only/mano_{side}_records.json').read_text())
        write_json(DEST/f'evidence/mano_{side}_statistics.json',[
            {k:v for k,v in r.items() if k!='trans'} for r in data])
    observed=clean(e)
    observed['action'].pop('raw')
    observed['action'].pop('segments')
    observed['action']['overlaps']=[{'seconds':r['seconds'],
        'previous_end':r['previous']['end_time'],'next_start':r['next']['start_time']} for r in e['action']['overlaps']]
    observed['action']['event_count']=len(e['action']['segments'])
    observed['action']['duration_seconds']=e['action']['raw']['info']['duration']
    observed['action']['source_event_keys']=list(e['action']['raw']['events'][0])
    observed['objpose']['sample']={'root_keys':e['objpose']['sample']['root_keys'],
        'object_keys':list(e['objpose']['sample']['object']), 'file_index':0,'frameId':1}
    for side in ['right','left']:
        observed['mano'][side]['sample'].pop('trans',None)
    for sample in observed['samples']:
        sample.pop('actions_at_time',None)
    for name in ['status','version','conclusions','recommended_files','next_dataset_recommendations']:
        observed.pop(name,None)
    write_json(DEST/'evidence/observations.json', observed)
    levels={'video_decode':'REAL_DATA_MEASUREMENT','palette':'REAL_DATA_MEASUREMENT',
        'alignment':'LOCAL_VISUAL_INSPECTION_WITHHELD_AND_EDGE_DIAGNOSTIC',
        'part_to_dynamic_instance':'INCOMPLETE_IDENTITY_EVIDENCE','object_pose':'NUMERICAL_ONLY_PHYSICS_UNVERIFIED',
        'mano_parameters':'REAL_DATA_SCHEMA_AND_FINITE_VALUES',
        'mano_21_joint_reconstruction':'DEPENDENCY_BLOCKED_NOT_RUN',
        'action_time':'REAL_ANNOTATION_TIMELINE_AND_PARSER_CHECKS',
        'hand_entity_action_chain':'MISSING_SOURCE_SUPERVISION'}
    results=clean(e['conclusions'])
    for key,value in results.items():
        value['evidence_level']=levels[key]
        value['evidence_files']=['evidence/observations.json']
    results['video_decode']['evidence_field']='videos'
    results['palette']['evidence_files']+=['evidence/mask_statistics.json']
    results['mano_parameters']['evidence_files']+=['evidence/mano_right_statistics.json','evidence/mano_left_statistics.json']
    results['mano_parameters']['evidence']=results['mano_parameters']['evidence'].replace('_records.json','_statistics.json')
    results['object_pose']['evidence_files']+=['evidence/objpose_statistics.json']
    results['object_pose']['evidence']=results['object_pose']['evidence'].replace('_records.json','_statistics.json')
    results['action_time']['evidence']='observations.action: duration_seconds=20, event_count=14; videos=300/15 seconds; 46 strictly-future checks passed. Not manual frame-accurate boundary validation.'
    results['action_time']['evidence_field']='action.boundary_checks'
    results['mano_21_joint_reconstruction']['execution_status']='NOT_RUN'
    report={'task_id':TASK,'task_status':'PARTIAL','base_commit':BASE,'audit_commit':None,'branch':BRANCH,
        'commit_semantics':'audit_commit pins the payload commit after sealing; final pushed HEAD is reported in the PR.',
        'dataset':{'name':'HOI4D','root':'.','version':e['version'],'source_path_namespace':'dataset-root-relative'},
        'sequence_id':e['sequence']['sequence'],'audit_results':results,
        'evidence_files':['evidence/observations.json','evidence/inventory.json','evidence/candidates.json',
            'evidence/mask_statistics.json','evidence/objpose_statistics.json','evidence/mano_right_statistics.json',
            'evidence/mano_left_statistics.json','evidence/cross_modal_frames.csv'],
        'tests':{'historical_execution':{'status':'PASS','passed':26,'duration_seconds':1.72,
            'level':'SYNTHETIC_PARSER_AND_COLLATION_ONLY','raw_log_retained':False,
            'provenance':'Prior audit tool output; no historical stdout file was captured. Archival rerun is logged separately.'}},
        'unresolved_issues':['UNKNOWN release revision and manifest','frame220: 37 undefined C20 color pixels',
            'unverified dynamic instance namespace/part grouping','UNKNOWN depth units and exact spatial registration',
            'UNKNOWN objpose coordinate frame/units/transform direction','licensed MANO assets and local forward/reprojection unavailable',
            'no source-backed action-target instance or contact annotation','train-only vocabulary/splits/leak-free raw Dataset pending'],
        'loss_readiness':e['loss_readiness'],'recommended_files':e['recommended_files'],
        'next_dataset_recommendations':e['next_dataset_recommendations']}
    write_json(doc,report)
    md=Path('docs/data_audit/hoi4d_single_sequence.md')
    text=clean_string(md.read_text())
    # Replace old artifact links; all image/array evidence remains local-only.
    text=text.replace('../../scripts/','../../../scripts/').replace('(hoi4d_single_sequence.json)','(report.json)')
    text=text.replace('../../reports/hoi4d_audit/','evidence/')
    text=text.replace('[reports/hoi4d_audit](evidence/)', '[归档证据](evidence/)')
    text=text.replace('所有原始文件绝对路径、大小','所有源文件的数据集根相对路径、大小')
    text=text.replace('实际根：`.`','归档中将实际数据集根记为 `.`，不公开本机根路径')
    text=text.replace('objpose_records.json','objpose_statistics.json').replace('mano_right_records.json','mano_right_statistics.json').replace('mano_left_records.json','mano_left_statistics.json').replace('mano_{right,left}_records.json','mano_{right,left}_statistics.json')
    text=text.replace('action.raw/segments','evidence/observations.json 的 action 统计').replace('action.raw','action')
    text=text.replace('reports/hoi4d_audit','reports/audits/'+TASK)
    text=re.sub(r'\[([^\]]+)\]\(evidence/([^)]*\.(?:jpg|png))\)',r'\1（本地保留、未公开：visualizations/local_only/\2）',text)
    text=text.replace('完整区间及源字段见 JSON `evidence/observations.json 的 action 统计`','公开JSON保留事件数量、字段名及边界检查；完整原标注快照仅本地保留')
    text=text.replace('真实原值及全300帧见','公开逐帧索引及数值诊断见')
    text=text.replace('从0编号；与此处视频一致。','从0编号；与此处视频一致。')
    text+='''

## 13. 归档说明与证据等级

本目录是唯一维护的最终结果。源数据路径均相对于数据集根；代码路径相对于仓库；证据链接相对于本归档。旧报告中两份JSON逐字节一致，归档前已核对哈希。原始标注快照和图像不进入Git；保留在本目录的Git忽略区域，并记录在manifest的withheld inventory。

| 结论 | 状态 | 证据等级及位置 |
|---|---|---|
| 文件覆盖率与视频解码 | PASS | 真实文件/解码测量：evidence/inventory.json、observations.json 的 videos |
| mask色彩兼容性 | PARTIAL | 真实300帧检查：mask_statistics.json；frame220失败37像素 |
| 严格跨模态配准 | PARTIAL | 本地目视检查、时间戳与边缘诊断；图像未公开，尚非物理标定 |
| 动态物理身份 | NEEDS_REVIEW | 缺逐帧动态实例文件；不能以静态ID替代 |
| objpose结构/旋转数值 | PASS | objpose_statistics.json；仅数学自洽检查 |
| objpose物理坐标/单位 | NEEDS_REVIEW | 没有经验证变换或单位证据 |
| MANO参数结构/有限性 | PASS | mano_*_statistics.json |
| MANO 3D恢复 | BLOCKED / NOT_RUN | 资产与依赖不足，未执行forward |
| Action时间表与未来事件逻辑 | PASS | observations.json action.boundary_checks，46项 |
| Action帧级人工语义边界 | PARTIAL | 时长一致不等价逐帧人工边界验证 |
| 完整Hand→Entity→Action链路 | BLOCKED | 缺目标、接触与3D对应证据 |

测试结果单独记录在logs；合成测试不用于提升真实数据结论等级。源数据重审未在归档阶段重复执行。归档阶段仅整理先前结果并重新运行已有解析/colllation测试及归档完整性检查。详见README与manifest。
'''
    # Reproduction requires the caller's local Python and HOI4D_ROOT; no host path.
    text=text.replace('python scripts/audit_hoi4d_physical.py','HOI4D_ROOT=<local-dataset-root> python scripts/audit_hoi4d_physical.py')
    md.write_text(text)
    (old/'README.md').write_text('''# P1-HOI4D-Physical-Data-Audit

Final status: **PARTIAL**. Start with [report.md](report.md), [report.json](report.json)
and [manifest.json](manifest.json). Shared policy: [REVIEW_PROTOCOL.md](../../../docs/data_audit/REVIEW_PROTOCOL.md).

Published evidence is metadata/statistics only. All dataset paths are relative to
the dataset root; repository and archive references have separate namespaces.
Original videos, masks, depth arrays, full annotation copies, checkpoints, secrets
and unapproved derived images are not published. Local-only material is retained
and described by hashes in the manifest, but will not exist in a public clone.
See [visualization policy](visualizations/README.md).

The archive does not claim verified physical coordinates, persistent entity IDs,
contact labels, action targets or recovered MANO joints. Every PASS is scoped to
the evidence named in report.json. Parser/collation tests are synthetic; actual
dataset observations are recorded separately. Archival is not a rerun of the data audit.

`base_commit` is the original audited source. `audit_commit` pins the payload
commit. `manifest.current_commit` is the HEAD observed when sealing metadata;
the final metadata commit cannot contain its own hash and is identified by the PR.

`SHA256SUMS.txt` covers every published archive file except itself, including
manifest.json. The manifest hashes published payloads except itself/checksum file
to avoid recursion. Run `sha256sum -c SHA256SUMS.txt` from this directory.

Files: README.md, report.md/json, manifest.json, SHA256SUMS.txt;
evidence/{observations,inventory,candidates,mask_statistics,objpose_statistics,
mano_right_statistics,mano_left_statistics}.json, evidence/cross_modal_frames.csv;
logs/{archival_provenance,tests}.json, logs/pytest.txt; visualizations/README.md;
.gitignore. Code remains in scripts/ and tests/.
''')
    (DEST/'visualizations/README.md').write_text('''# Withheld visualizations

Permission to publish these dataset-derived images was not established. No images
are included in Git. Existing RGB/mask copies, overlays, panels and depth displays
are preserved locally in local_only/. Raw depth NPY samples are preserved under
evidence/local_only/. Metadata and SHA256 values are in manifest.json.

These are local audit evidence, not invented annotations. Source annotation
snapshots are also excluded. Local-only files are not expected in a public clone.
''')
    (DEST/'.gitignore').write_text('evidence/local_only/\nvisualizations/local_only/\n!logs/\n!logs/*.json\n!logs/*.txt\n')
    write_json(DEST/'logs/archival_provenance.json',{
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'base_commit':BASE,
        'initial_branch':'feat/e3a-foundation','initial_head':BASE,'existing_open_prs':[],
        'initial_artifacts_tracked':False,'duplicate_json_bytes_equal':True,
        'original_artifacts':inventory,
        'relocation_policy':'Sanitized public text is staged then git mv. Initially untracked restricted files use local rename, never staged. No evidence deleted.',
        'sanitization':'Dataset-root-relative source paths; interpreter replaced with python; full annotation snapshots withheld; public statistics derived from prior evidence.',
        'original_audit_timestamp_utc':'UNKNOWN; prior report recorded local date 2026-11-21 only',
        'archival_real_data_rerun':'NOT_RUN'})
    print('Prepared sanitized public text and preserved all restricted evidence locally; git mv remains to run.')


def run_tests():
    command=[__import__('sys').executable,'-m','pytest','-q',
        'tests/test_hoi4d_data.py','tests/test_action_vocabulary.py',
        'tests/test_entity_target_collate.py','tests/test_e3a_collate.py','tests/test_hoi4d_archive.py']
    result=subprocess.run(command,capture_output=True,text=True)
    text=result.stdout+result.stderr
    assert not re.search(r'/(?:home|shares2|tmp)/',text), 'Test output requires path sanitization'
    (DEST/'logs/pytest.txt').write_text(text)
    write_json(DEST/'logs/tests.json',{'execution_timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'command':['python',*command[1:]],'exit_code':result.returncode,
        'status':'PASS' if result.returncode==0 else 'BLOCKED',
        'level':'SYNTHETIC_PARSER_COLLATION_AND_ARCHIVE_INTEGRITY; NOT REAL_DATA_PHYSICAL_VALIDATION',
        'passed':int(re.search(r'(\d+) passed',text).group(1)) if result.returncode==0 else None,
        'real_data_reaudit':'NOT_RUN','log':'logs/pytest.txt'})
    if result.returncode: raise SystemExit(result.returncode)
    print(text)


def seal(audit_commit):
    report=json.loads((DEST/'report.json').read_text())
    report['audit_commit']=audit_commit
    report['tests']['archival_rerun']=json.loads((DEST/'logs/tests.json').read_text())
    write_json(DEST/'report.json',report)
    published=[];withheld=[]
    for p in sorted(DEST.rglob('*')):
        if not p.is_file() or p.name in {'manifest.json','SHA256SUMS.txt'}: continue
        rel=p.relative_to(DEST).as_posix()
        row={'archive_path':rel,'bytes':p.stat().st_size,'sha256':digest(p)}
        (withheld if 'local_only' in p.parts else published).append(row)
    manifest={'repository':'https://github.com/heu-gj/E3A','branch':BRANCH,'base_commit':BASE,
        'current_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'audit_commit':audit_commit,'commit_semantics':'current_commit is the archival parent HEAD at sealing; final pushed HEAD is external to this self-contained manifest.',
        'execution_timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'original_audit_timestamp_utc':'UNKNOWN (local date 2026-11-21 recorded)',
        'dataset_version':report['dataset']['version'],'audited_sequence':report['sequence_id'],
        'path_namespaces':{'dataset':'. (source paths relative to dataset root)',
                           'archive':'this directory','repository':'repository root'},
        'file_inventory':json.loads((DEST/'evidence/inventory.json').read_text()),
        'source_checksums':json.loads((DEST/'evidence/observations.json').read_text())['source_fingerprints'],
        'artifact_inventory':published,'withheld_local_inventory':withheld,
        'checksum_policy':'Artifact inventory excludes manifest.json and SHA256SUMS.txt; SHA256SUMS.txt includes manifest.json but excludes itself.',
        'sha256_checksums':{x['archive_path']:x['sha256'] for x in published}}
    source_hashes={x['path']:x['sha256'] for x in manifest['source_checksums']}
    for item in manifest['file_inventory']:
        item['sha256']=source_hashes.get(item['path'])
        item['checksum_status']='PASS' if item['sha256'] else 'NOT_RUN'
    write_json(DEST/'manifest.json',manifest)
    files=[DEST/x['archive_path'] for x in published]+[DEST/'manifest.json']
    (DEST/'SHA256SUMS.txt').write_text(''.join(f'{digest(f)}  {f.relative_to(DEST).as_posix()}\n' for f in sorted(files)))
    print('Sealed',len(published)+2,'public files;',len(withheld),'withheld local files')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','tests','seal'])
    parser.add_argument('--audit-commit')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='tests':run_tests()
    else:seal(args.audit_commit)
