"""Validate the published archive, independently of the local HOI4D installation."""
import hashlib
import json
import re
from pathlib import Path


ARCHIVE = Path(__file__).resolve().parents[1] / 'reports/audits/P1-HOI4D-Physical-Data-Audit'


def test_archive_hashes_inventory_and_publication_boundary():
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
    actual = set()
    for line in (ARCHIVE / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert 'local_only' not in Path(name).parts
        payload = ARCHIVE / name
        assert payload.suffix not in {'.npy', '.png', '.jpg', '.zip', '.mp4', '.avi', '.pickle', '.pt', '.pth'}
        assert payload.stat().st_size < 10 * 1024 * 1024
        content = payload.read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected
        assert not re.search(rb'/(?:home|shares2|Users|tmp)/', content)
        actual.add(name)
    expected = {x['archive_path'] for x in manifest['artifact_inventory']} | {'manifest.json'}
    assert actual == expected
    for item in manifest['artifact_inventory']:
        assert hashlib.sha256((ARCHIVE / item['archive_path']).read_bytes()).hexdigest() == item['sha256']
    assert len(manifest['file_inventory']) == 1073
    assert all(not Path(x['path']).is_absolute() for x in manifest['file_inventory'])


def test_passes_have_scoped_evidence_and_unknowns_remain_unresolved():
    report = json.loads((ARCHIVE / 'report.json').read_text())
    assert report['task_status'] == 'PARTIAL'
    for result in report['audit_results'].values():
        assert result['status'] in {'PASS', 'PARTIAL', 'BLOCKED', 'NEEDS_REVIEW', 'NOT_RUN'}
        if result['status'] == 'PASS':
            assert result['evidence'] and result['evidence_level'] and result['evidence_files']
            assert all((ARCHIVE / name).is_file() for name in result['evidence_files'])
    assert report['audit_results']['mano_21_joint_reconstruction']['execution_status'] == 'NOT_RUN'
    observations = json.loads((ARCHIVE / 'evidence/observations.json').read_text())
    assert observations['mask']['raw_decoder_failed_frames'] == [220]
    assert observations['mask']['unknown_pixels'] == 37
    assert observations['objpose']['coordinate_frame'] == 'UNKNOWN'
    checks = observations['action']['boundary_checks']
    assert len(checks) == 46 and all(x['pass'] for x in checks)
    assert all(not x['target']['target_entity_valid'] for x in checks)
