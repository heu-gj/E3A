"""Synthetic tests validate audit logic, never real-data coverage claims."""
import csv
import importlib.util
import json
import pickle
import subprocess
import sys
import tarfile
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'hand_audit', Path(__file__).parents[1] / 'scripts/audit_hoi4d_handpose_coverage.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
SEQ = 'ZY20210800001/H1/C20/N18/S286/s01/T1'


def put(root, side='left', filename='30.pickle', value=None):
    p = root / m.HAND_ROOT / ('handpose_' + side + '_hand') / SEQ / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        value = {k: np.zeros(shape, dtype=np.float32) for k, shape in m.SHAPES.items()}
    p.write_bytes(pickle.dumps(value))
    return p


def test_complete_sequence_identity_not_task_suffix():
    assert m.sequence_id(SEQ.split('/')) == SEQ
    assert m.sequence_id(['T1']) is None
    assert m.sequence_id(SEQ.replace('/N18/', '/N19/').split('/')) != SEQ


@pytest.mark.parametrize('name,shape', m.SHAPES.items())
def test_production_parser_shapes(tmp_path, name, shape):
    p = put(tmp_path)
    result = m.inspect_record(p)
    assert result['parser_ok'] and result['shape_ok']
    assert result['fields'][name]['shape'] == list(shape)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_rejected_and_diagnosed(tmp_path, bad):
    data = {k: np.zeros(s) for k, s in m.SHAPES.items()}
    data['trans'][0] = bad
    result = m.inspect_record(put(tmp_path, value=data))
    assert not result['parser_ok'] and result['raw_read_ok']
    assert result['nan'] + result['inf'] == 1


def test_missing_field_and_bad_shape(tmp_path):
    data = {'poseCoeff': np.zeros(47)}
    result = m.inspect_record(put(tmp_path, value=data))
    assert not result['shape_ok'] and not result['parser_ok']
    assert result['fields']['beta']['missing']


@pytest.mark.parametrize('content', [b'', b'not a pickle'])
def test_corrupt_and_empty_files(tmp_path, content):
    p = put(tmp_path)
    p.write_bytes(content)
    assert not m.inspect_record(p)['raw_read_ok']


def test_restricted_unpickler_does_not_execute_globals(tmp_path):
    class Unsafe:
        def __reduce__(self):
            return eval, ('1+1',)
    p = put(tmp_path, value=Unsafe())
    result = m.inspect_record(p)
    assert not result['raw_read_ok']
    assert result['raw_error'] == 'UnpicklingError'


def test_full_audit_empty_side_universe_and_out_of_range(tmp_path, monkeypatch):
    root = tmp_path / 'dataset'
    put(root)
    put(root, filename='name.pickle')
    put(root, filename='030.pickle')  # Numeric duplicate, not a second frame.
    video = root / 'extracted/release/HOI4D_release' / SEQ / 'align_rgb/image.mp4'
    video.parent.mkdir(parents=True)
    video.touch()
    ann = root / 'extracted/annotations/HOI4D_annotations' / SEQ
    ann.mkdir(parents=True)
    missing = ann.parent / 'T2'
    missing.mkdir()
    monkeypatch.setattr(m, 'video_metadata', lambda _: dict(status='PASS', frame_count=20, fps=10, duration=2))
    out = tmp_path / 'output'
    summary = m.audit(root, out)
    assert summary['sides']['left']['files'] == 3
    assert summary['sides']['left']['duplicate_frame_indices'] == 1
    assert summary['sides']['left']['non_numeric_filenames'] == 1
    assert summary['out_of_range_indices'] == 1
    assert summary['both_missing_in_local_universe'] == 1
    assert summary['unknown_missing_hand_frame_slots'] == 40
    assert summary['extraction_integrity'] == 'NOT_VERIFIABLE'
    with (out / 'evidence/sequence_coverage.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 2
    assert summary['download_integrity'] == 'NOT_VERIFIABLE'


@pytest.mark.parametrize('missing,mismatch', [(False, False), (True, False), (False, True)])
def test_archive_equivalence_and_missing_members(tmp_path, missing, mismatch):
    root = tmp_path / 'dataset'
    file = put(root)
    archive = root / 'Hand_pose.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('Hand_pose/handpose_left_hand/' + SEQ + '/30.pickle',
                   b'wrong bytes' if mismatch else file.read_bytes())
        if missing:
            z.writestr('Hand_pose/handpose_right_hand/' + SEQ + '/31.pickle', file.read_bytes())
    summary = m.audit(root, tmp_path / 'out')
    expected = ('EXTRACTION_INCOMPLETE' if missing else
                'NOT_VERIFIABLE' if mismatch else 'EXTRACTION_COMPLETE')
    assert summary['extraction_integrity'] == expected
    comparison = json.loads((tmp_path / 'out/evidence/archive_extraction_comparison.json').read_text())
    if mismatch:
        assert comparison['EXTRACTION_EQUIVALENCE'] == 'NEEDS_REVIEW'
    assert summary['download_integrity'] == 'NOT_VERIFIABLE'


def test_unsafe_archive_paths_not_equivalent():
    assert m.archive_member_path('../handpose_left_hand/x.pickle') is None
    assert m.archive_member_path('/handpose_left_hand/x.pickle') is None
    assert m.archive_member_path('Hand_pose/handpose_left_hand/' + SEQ + '/30.pickle') == (
        m.HAND_ROOT / 'handpose_left_hand' / SEQ / '30.pickle').as_posix()


def test_tar_members_compared_without_extraction(tmp_path):
    root = tmp_path / 'dataset'
    p = put(root)
    data = p.read_bytes()
    with tarfile.open(root / 'hand_pose.tar.gz', 'w:gz') as t:
        entry = tarfile.TarInfo('Hand_pose/handpose_left_hand/' + SEQ + '/30.pickle')
        entry.size = len(data)
        t.addfile(entry, io.BytesIO(data))
    before = p.read_bytes()
    result = m.audit(root, tmp_path / 'out')
    assert result['extraction_integrity'] == 'EXTRACTION_COMPLETE'
    assert p.read_bytes() == before


def test_invalid_numeric_dtype_and_negative_filename(tmp_path):
    data = {k: np.zeros(shape) for k, shape in m.SHAPES.items()}
    data['trans'] = ['x', 'y', 'z']
    p = put(tmp_path, filename='-1.pickle', value=data)
    assert not m.inspect_record(p)['parser_ok']
    assert not m.inspect_record(p)['shape_ok']


def test_output_inside_dataset_rejected(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        m.audit(tmp_path, tmp_path / 'out')


def test_cli_help_and_json_csv_serialization(tmp_path):
    root = tmp_path / 'dataset'
    p = put(root)
    before = p.read_bytes()
    output = tmp_path / 'out'
    script = Path(__file__).parents[1] / 'scripts/audit_hoi4d_handpose_coverage.py'
    result = subprocess.run([sys.executable, str(script), '--root', str(root), '--output-dir', str(output)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.splitlines()[-1])['sides']['left']['files'] == 1
    assert p.read_bytes() == before
    for file in (output / 'evidence').glob('*.json'):
        json.loads(file.read_text())
    with (output / 'evidence/left_hand_coverage.csv').open() as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1 and rows[0]['sequence_id'] == SEQ


def archive_fixture(tmp_path, entries, *, local=True, archive_name='hand_pose.zip'):
    root = tmp_path / 'dataset'
    root.mkdir(exist_ok=True)
    file = put(root) if local else None
    archive = tmp_path / archive_name  # Explicitly outside root.
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
        for path, content in entries:
            z.writestr(path, content)
    return root, archive, file


MEMBER = 'wrapper/Hand_pose/handpose_left_hand/' + SEQ + '/30.pickle'


def reconcile(root, archives, **kwargs):
    from e3a.data.handpose_archive import reconcile as run
    return run(root, archives, **kwargs)


def test_external_archive_exact_sha_and_side_separation(tmp_path):
    root = tmp_path / 'dataset'
    file = put(root)
    archive = tmp_path / 'outside.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(MEMBER, file.read_bytes())
    result = reconcile(root, [archive], output=tmp_path / 'report')
    assert result['EXTRACTION_EQUIVALENCE'] == 'PASS'
    assert result['ARCHIVE_CRC_INTEGRITY'] == 'PASS'
    assert result['DOWNLOAD_COMPLETENESS'] == 'NOT_VERIFIABLE'
    assert result['sides']['left']['unique_valid_members'] == 1
    assert result['sides']['right']['unique_valid_members'] == 0
    with (tmp_path / 'report/evidence/member_comparison.csv').open() as f:
        row = list(csv.DictReader(f))[0]
        assert row['seq'] == SEQ and row['raw'] == MEMBER
        assert row['result'] == 'EXACT_MATCH' and len(row['digest']) == 64


@pytest.mark.parametrize('entries', [[], [('readme.txt', b'no hand data')]])
def test_empty_or_no_hand_archive_never_complete(tmp_path, entries):
    root, archive, _ = archive_fixture(tmp_path, entries, local=False)
    result = reconcile(root, [archive])
    assert result['EXTRACTION_EQUIVALENCE'] != 'PASS'
    assert result['extraction_integrity'] != 'EXTRACTION_COMPLETE'


def test_missing_extracted_file_blocked(tmp_path):
    root, archive, _ = archive_fixture(tmp_path, [(MEMBER, b'bytes')], local=False)
    result = reconcile(root, [archive])
    assert result['EXTRACTION_EQUIVALENCE'] == 'BLOCKED'
    assert len(result['missing_paths']) == 1


def test_local_extra_is_not_equivalent(tmp_path):
    root = tmp_path / 'dataset'
    file = put(root)
    put(root, side='right')
    archive = tmp_path / 'hand.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(MEMBER, file.read_bytes())
    result = reconcile(root, [archive])
    assert result['EXTRACTION_EQUIVALENCE'] == 'NEEDS_REVIEW'
    assert result['sides']['right']['extra_files'] == 1
    assert result['extraction_integrity'] != 'EXTRACTION_COMPLETE'


def test_content_mismatch_requires_review(tmp_path):
    root, archive, _ = archive_fixture(tmp_path, [(MEMBER, b'wrong')])
    result = reconcile(root, [archive])
    assert result['EXTRACTION_EQUIVALENCE'] == 'NEEDS_REVIEW'
    assert len(result['mismatch_paths']) == 1


@pytest.mark.parametrize('conflict', [False, True])
def test_duplicate_archive_members_never_silently_override(tmp_path, conflict):
    root, archive, _ = archive_fixture(tmp_path, [(MEMBER, b'a'), (MEMBER, b'b' if conflict else b'a')], local=False)
    result = reconcile(root, [archive])
    assert result['archives'][0]['duplicate_members'] == 1
    assert result['archives'][0]['conflicting_members'] == int(conflict)
    assert result['EXTRACTION_EQUIVALENCE'] != 'PASS'


def test_conflicting_members_across_archives(tmp_path):
    root, a, _ = archive_fixture(tmp_path, [(MEMBER, b'a')], local=False)
    b = tmp_path / 'second.zip'
    with zipfile.ZipFile(b, 'w') as z:
        z.writestr(MEMBER, b'b')
    result = reconcile(root, [a, b])
    assert result['archives'][1]['conflicting_members'] == 1
    assert result['EXTRACTION_EQUIVALENCE'] != 'PASS'


@pytest.mark.parametrize('bad_path', ['/' + MEMBER, '../' + MEMBER,
                                     'C:\\' + MEMBER.replace('/', '\\'),
                                     'root/../' + MEMBER])
def test_unsafe_paths_reported_and_never_extracted(tmp_path, bad_path):
    root, archive, _ = archive_fixture(tmp_path, [(bad_path, b'a')], local=False)
    result = reconcile(root, [archive])
    assert result['archives'][0]['invalid_members'] == 1
    assert result['EXTRACTION_EQUIVALENCE'] != 'PASS'


def test_windows_relative_paths_are_normalized_strictly(tmp_path):
    root = tmp_path / 'dataset'
    file = put(root)
    archive = tmp_path / 'hand.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(MEMBER.replace('/', '\\'), file.read_bytes())
    result = reconcile(root, [archive])
    assert result['EXTRACTION_EQUIVALENCE'] == 'PASS'


def test_crc_error_detected_from_full_member_read(tmp_path):
    root, archive, _ = archive_fixture(tmp_path, [(MEMBER, b'payload-unique-123')], local=False)
    data = bytearray(archive.read_bytes())
    index = data.index(b'payload-unique-123')
    data[index] ^= 0xff
    archive.write_bytes(data)
    result = reconcile(root, [archive])
    assert result['ARCHIVE_CRC_INTEGRITY'] == 'BLOCKED'
    assert result['archives'][0]['crc_errors'] == 1


def test_unreadable_archive_blocks_crc_and_equivalence(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    archive = tmp_path / 'bad.zip'
    archive.write_bytes(b'not zip')
    result = reconcile(root, [archive])
    assert result['ARCHIVE_READABILITY'] == 'BLOCKED'
    assert result['ARCHIVE_CRC_INTEGRITY'] == 'BLOCKED'
    assert result['EXTRACTION_EQUIVALENCE'] == 'BLOCKED'


@pytest.mark.parametrize('which', ['zip', 'local'])
def test_source_changed_after_hash_prevents_pass(tmp_path, monkeypatch, which):
    from e3a.data import handpose_archive as engine
    root = tmp_path / 'dataset'
    file = put(root)
    archive = tmp_path / 'hand.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(MEMBER, file.read_bytes())
    original = engine.stable_hash
    calls = {}
    def changed(path):
        digest, stamp = original(path)
        calls[path] = calls.get(path, 0) + 1
        if path == (archive if which == 'zip' else file) and calls[path] == 2:
            stamp = (*stamp[:-1], stamp[-1] + 1)
        return digest, stamp
    monkeypatch.setattr(engine, 'stable_hash', changed)
    result = engine.reconcile(root, [archive])
    assert result['EXTRACTION_EQUIVALENCE'] == 'BLOCKED'
    if which == 'local':
        assert result['ARCHIVE_CRC_INTEGRITY'] == 'PASS'


def test_zero_size_member_and_wrapper_directory(tmp_path):
    root, archive, _ = archive_fixture(tmp_path, [('wrapper/', b''), (MEMBER, b'')], local=False)
    result = reconcile(root, [archive])
    assert result['archives'][0]['zero_byte_files'] == 1
    assert result['archives'][0]['invalid_members'] == 0


def test_hand01_explicit_external_archive_parameter(tmp_path):
    root = tmp_path / 'dataset'
    file = put(root)
    archive = tmp_path / 'outside.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(MEMBER, file.read_bytes())
    result = m.audit(root, tmp_path / 'out', archive_paths=[archive])
    assert result['extraction_integrity'] == 'EXTRACTION_COMPLETE'


def test_ambiguous_multiple_hand_roots_are_not_guessed(tmp_path):
    path = 'handpose_right_hand/' + MEMBER
    root, archive, _ = archive_fixture(tmp_path, [(path, b'a')], local=False)
    result = reconcile(root, [archive])
    assert result['archives'][0]['invalid_members'] == 1


def test_two_wrapper_paths_same_destination_report_duplicate(tmp_path):
    root = tmp_path / 'dataset'
    file = put(root)
    archive = tmp_path / 'hand.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(MEMBER, file.read_bytes())
        z.writestr('second/' + MEMBER, file.read_bytes())
    result = reconcile(root, [archive])
    assert result['archives'][0]['duplicate_members'] == 1
    assert result['EXTRACTION_EQUIVALENCE'] == 'NEEDS_REVIEW'


def test_same_file_size_is_not_sha_equivalence(tmp_path):
    root = tmp_path / 'dataset'
    file = put(root)
    data = bytearray(file.read_bytes())
    data[-1] ^= 1
    archive = tmp_path / 'hand.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(MEMBER, data)
    result = reconcile(root, [archive])
    assert result['sides']['left']['classification_counts']['CONTENT_MISMATCH'] == 1
    assert result['EXTRACTION_EQUIVALENCE'] == 'NEEDS_REVIEW'


def test_read_failure_one_member_does_not_skip_later_entries(tmp_path):
    root, archive, _ = archive_fixture(tmp_path, [(MEMBER, b'bad-payload'),
        (MEMBER.replace('30.pickle', '31.pickle'), b'valid-payload')], local=False)
    data = bytearray(archive.read_bytes())
    data[data.index(b'bad-payload')] ^= 1
    archive.write_bytes(data)
    result = reconcile(root, [archive], output=tmp_path / 'report')
    assert result['archives'][0]['crc_errors'] == 1
    assert result['archives'][0]['members_streamed'] == 1
    assert len(result['missing_paths']) == 1


def test_local_extra_files_csv_keeps_explicit_classification(tmp_path):
    root, archive, _ = archive_fixture(tmp_path, [], local=True)
    reconcile(root, [archive], output=tmp_path / 'report')
    with (tmp_path / 'report/evidence/extra_files.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['result'] == 'EXTRA_EXTRACTED_FILE'
