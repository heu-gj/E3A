"""Read-only archive reconciliation. No deserialization or extraction occurs."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import stat
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

HAND_ROOT = Path('extracted/hand_pose/Hand_pose')
SIDES = {'handpose_right_hand': 'right', 'handpose_left_hand': 'left'}
PREFIXES = ('ZY', 'H', 'C', 'N', 'S', 's', 'T')


def sha_stream(stream):
    h = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def stable_hash(path):
    before = fingerprint(path)
    with path.open('rb') as f:
        digest = sha_stream(f)
    if fingerprint(path) != before:
        raise RuntimeError('SOURCE_CHANGED_DURING_READ')
    return digest, before


def normalize_member(name, directory=False):
    """Allow explicit wrapper directories; retain exact side/7-level/file suffix.

    Backslashes are separators, never literal filenames. Absolute/drive paths,
    traversal, repeated hand roots and malformed hierarchies are rejected.
    Different wrappers resolving to one path remain duplicate records, not merged.
    """
    cleaned = name.replace('\\', '/')
    if not cleaned or cleaned.startswith('/') or re.match(r'^[A-Za-z]:', cleaned):
        return None, 'ABSOLUTE_OR_EMPTY_PATH'
    parts = cleaned.rstrip('/').split('/')
    if any(p in ('', '.', '..') or ':' in p or '\x00' in p for p in parts):
        return None, 'UNSAFE_PATH_COMPONENT'
    roots = [i for i, p in enumerate(parts) if p in SIDES]
    if len(roots) != 1:
        if directory and not roots:
            return {'path': None, 'side': None, 'sequence_id': None,
                    'frame_index': None}, 'WRAPPER_DIRECTORY'
        return None, 'NO_HAND_ROOT' if not roots else 'AMBIGUOUS_HAND_ROOT'
    index = roots[0]
    suffix = parts[index:]
    if directory:
        return {'path': (HAND_ROOT / Path(*suffix)).as_posix(),
                'side': SIDES[suffix[0]], 'sequence_id': None, 'frame_index': None}, None
    if len(suffix) != 9:
        return None, 'UNEXPECTED_HIERARCHY'
    seq = suffix[1:8]
    if any(not p.startswith(pre) or not p[len(pre):].isascii()
           or not p[len(pre):].isdigit() for p, pre in zip(seq, PREFIXES)):
        return None, 'INVALID_SEQUENCE'
    filename = suffix[8]
    if not re.fullmatch(r'[0-9]+\.pickle', filename):
        return None, 'UNEXPECTED_FILE'
    return {'path': (HAND_ROOT / Path(*suffix)).as_posix(), 'side': SIDES[suffix[0]],
            'sequence_id': '/'.join(seq), 'frame_index': int(filename[:-7])}, None


def _label(path, root, number):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return '${EXTERNAL_ARCHIVE_' + str(number) + '}/' + path.name


def reconcile(root, archives, output=None, local_db=None):
    """Stream all members including unexpected files, verify CRC, hash both sources.

    local_db optionally supplies HAND-01 files(path,digest); those values are
    freshly re-read, not trusted as current content. Return legacy summary keys
    plus distinct integrity verdicts. SQLite holds the full inventory on disk.
    """
    root = Path(root).resolve()
    archives = [Path(p).resolve() for p in archives]
    if len(set(archives)) != len(archives):
        raise ValueError('Duplicate archive input paths.')
    out = None if output is None else Path(output).resolve()
    if out is not None:
        if out == root or root in out.parents:
            raise ValueError('Output must be outside dataset.')
        out.mkdir(parents=True, exist_ok=True)
        dbpath = out / 'reconciliation.sqlite'
        if dbpath.exists():
            raise FileExistsError('Refusing to overwrite inventory.')
        db = sqlite3.connect(dbpath)
    else:
        db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE local(path TEXT PRIMARY KEY,digest TEXT,size INT,side TEXT,seq TEXT,frame INT,stamp TEXT)')
    db.execute('CREATE TABLE members(archive TEXT,ordinal INT,raw TEXT,path TEXT,side TEXT,seq TEXT,frame INT,size INT,compressed INT,crc INT,digest TEXT,is_dir INT,result TEXT,error TEXT)')
    db.execute('CREATE INDEX members_path ON members(path)')
    db.execute('CREATE INDEX members_raw ON members(archive,raw)')
    errors = []
    # Full hand-directory inventory includes unexpected local files as anomalies.
    if local_db is None:
        paths = (p for p in sorted((root / HAND_ROOT).rglob('*')) if p.is_file() or p.is_symlink())
    else:
        paths = (root / row[0] for row in local_db.execute('SELECT path FROM files ORDER BY path'))
    for i, path in enumerate(paths, 1):
        rel = path.relative_to(root).as_posix()
        norm, error = normalize_member(rel)
        if path.is_symlink() or norm is None:
            errors.append({'source': 'LOCAL', 'path': rel, 'error': error or 'SYMLINK'})
            continue
        try:
            digest, stamp = stable_hash(path)
            db.execute('INSERT INTO local VALUES(?,?,?,?,?,?,?)',
                       (rel, digest, stamp[2], norm['side'], norm['sequence_id'], norm['frame_index'], json.dumps(stamp)))
        except (OSError, RuntimeError) as exc:
            errors.append({'source': 'LOCAL', 'path': rel, 'error': type(exc).__name__})
        if i % 100000 == 0:
            db.commit()
            print(json.dumps({'local_files_hashed': i}), flush=True)
    db.commit()
    archive_rows = []
    for number, archive in enumerate(archives):
        label = _label(archive, root, number)
        row = {'path': label, 'central_directory_readable': False, 'members': 0,
               'crc_errors': 0, 'read_errors': 0, 'invalid_members': 0,
               'duplicate_members': 0, 'conflicting_members': 0, 'zero_byte_files': 0,
               'members_streamed': 0, 'source_unchanged': False}
        archive_rows.append(row)
        try:
            digest_before, stamp = stable_hash(archive)
            row.update(sha256=digest_before, size_bytes=stamp[2], mtime_ns=stamp[3])
        except (OSError, RuntimeError) as exc:
            row.update(read_errors=1, error=type(exc).__name__)
            continue

        def consume(ordinal, name, size, compressed, crc, is_dir, stream, unsafe_type=False):
            norm, reason = normalize_member(name, is_dir)
            if unsafe_type:
                norm, reason = None, 'SYMLINK_OR_SPECIAL_MEMBER'
            digest = None
            result = 'DIRECTORY' if is_dir else 'INVALID_ARCHIVE_MEMBER'
            if not is_dir:
                if size == 0:
                    row['zero_byte_files'] += 1
                try:
                    digest = sha_stream(stream)  # Read even unexpected members to EOF.
                    row['members_streamed'] += 1
                except Exception as exc:
                    row['read_errors'] += 1
                    reason = type(exc).__name__
                    if isinstance(exc, zipfile.BadZipFile) and 'CRC' in str(exc):
                        row['crc_errors'] += 1
                    result = 'READ_ERROR'
            dest = norm['path'] if norm else None
            if norm is None:
                row['invalid_members'] += 1
            if is_dir and db.execute('SELECT 1 FROM members WHERE archive=? AND raw=?', (label, name)).fetchone():
                row['duplicate_members'] += 1
                result = 'DUPLICATE_ARCHIVE_MEMBER'
            if norm and not is_dir:
                row['members'] += 1
                prior = db.execute('SELECT digest FROM members WHERE path=? AND is_dir=0', (dest,)).fetchall()
                if prior:
                    row['duplicate_members'] += 1
                    if any(p[0] != digest for p in prior):
                        row['conflicting_members'] += 1
                    if result != 'READ_ERROR':
                        result = 'DUPLICATE_ARCHIVE_MEMBER'
                elif result != 'READ_ERROR':
                    actual = db.execute('SELECT digest,size FROM local WHERE path=?', (dest,)).fetchone()
                    result = ('MISSING_EXTRACTED_FILE' if actual is None else
                              'EXACT_MATCH' if actual == (digest, size) else 'CONTENT_MISMATCH')
            db.execute('INSERT INTO members VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (label, ordinal, name, dest, norm['side'] if norm else None,
                 norm['sequence_id'] if norm else None, norm['frame_index'] if norm else None,
                 size, compressed, crc, digest, int(is_dir), result, reason))
            if ordinal % 100000 == 0:
                db.commit()
                print(json.dumps({'archive': label, 'members_read': ordinal}), flush=True)
        try:
            if archive.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive) as z:
                    infos = z.infolist()
                    row['central_directory_readable'] = True
                    row['total_entries'] = len(infos)
                    for ordinal, entry in enumerate(infos):
                        mode = entry.external_attr >> 16
                        try:
                            with z.open(entry) as stream:
                                consume(ordinal, entry.filename, entry.file_size, entry.compress_size,
                                        entry.CRC, entry.is_dir(), stream, stat.S_ISLNK(mode))
                        except Exception as exc:
                            row['read_errors'] += 1
                            db.execute('INSERT INTO members(archive,ordinal,raw,result,error,is_dir) VALUES(?,?,?,?,?,?)',
                                       (label, ordinal, entry.filename, 'READ_ERROR', type(exc).__name__, int(entry.is_dir())))
            else:
                with tarfile.open(archive, 'r|*') as tar:
                    row['central_directory_readable'] = True
                    for ordinal, entry in enumerate(tar):
                        stream = tar.extractfile(entry) if entry.isfile() else None
                        try:
                            consume(ordinal, entry.name, entry.size, None, None, entry.isdir(), stream,
                                    not entry.isdir() and not entry.isfile())
                        finally:
                            if stream is not None:
                                stream.close()
        except Exception as exc:
            row['read_errors'] += 1
            row['error'] = type(exc).__name__
        try:
            after, after_stamp = stable_hash(archive)
            row['sha256_after'] = after
            row['source_unchanged'] = stamp == after_stamp and after == digest_before
        except (OSError, RuntimeError):
            row['source_unchanged'] = False
        db.commit()
    # Full second hash pass catches changes after a file's first read as well.
    checked = 0
    for path, digest, stamp in db.execute('SELECT path,digest,stamp FROM local ORDER BY path'):
        try:
            new, new_stamp = stable_hash(root / path)
            if new != digest or list(new_stamp) != json.loads(stamp):
                errors.append({'source': 'LOCAL', 'path': path, 'error': 'SOURCE_CHANGED'})
        except (OSError, RuntimeError):
            errors.append({'source': 'LOCAL', 'path': path, 'error': 'SOURCE_CHANGED_OR_READ_ERROR'})
        checked += 1
    extra = [x[0] for x in db.execute('SELECT path FROM local WHERE path NOT IN (SELECT path FROM members WHERE path IS NOT NULL AND is_dir=0) ORDER BY path')]
    missing = [x[0] for x in db.execute("SELECT DISTINCT path FROM members WHERE result='MISSING_EXTRACTED_FILE' ORDER BY path")]
    mismatch = [x[0] for x in db.execute("SELECT DISTINCT path FROM members WHERE result='CONTENT_MISMATCH' ORDER BY path")]
    duplicates = sum(x['duplicate_members'] for x in archive_rows)
    invalid = sum(x['invalid_members'] for x in archive_rows)
    read_errors = sum(x['read_errors'] for x in archive_rows)
    archive_unchanged = all(x['source_unchanged'] for x in archive_rows)
    unchanged = archive_unchanged and not errors
    enough = bool(archives) and all(x['members'] > 0 for x in archive_rows)
    equivalence = ('BLOCKED' if missing or read_errors or not unchanged else
                   'NEEDS_REVIEW' if extra or mismatch or duplicates or invalid else
                   'PASS' if enough else 'NOT_VERIFIABLE')
    readability = 'PASS' if archive_rows and all(x['central_directory_readable'] and not x['read_errors'] for x in archive_rows) else 'BLOCKED' if archives else 'NOT_RUN'
    crc_status = ('BLOCKED' if read_errors or not archive_unchanged else
                  'PASS' if archive_rows and all(Path(p).suffix.lower() == '.zip' for p in archives) else 'NOT_RUN')
    sides = {}
    for side in ('right', 'left'):
        local_count = db.execute('SELECT count(*) FROM local WHERE side=?', (side,)).fetchone()[0]
        counts = dict(db.execute('SELECT result,count(*) FROM members WHERE side=? AND is_dir=0 GROUP BY result', (side,)))
        sides[side] = {'archive_members': sum(counts.values()), 'unique_valid_members': db.execute('SELECT count(DISTINCT path) FROM members WHERE side=? AND is_dir=0', (side,)).fetchone()[0],
                       'local_files': local_count, 'classification_counts': counts,
                       'extra_files': sum('/handpose_' + side + '_hand/' in p for p in extra)}
    result = {'ARCHIVE_READABILITY': readability, 'ARCHIVE_CRC_INTEGRITY': crc_status,
        'EXTRACTION_EQUIVALENCE': equivalence, 'DOWNLOAD_COMPLETENESS': 'NOT_VERIFIABLE',
        'archives': archive_rows, 'sides': sides, 'missing_paths': missing,
        'mismatch_paths': mismatch, 'extracted_extra_paths': extra,
        'source_errors': errors, 'local_files_rehashed': checked,
        'extraction_integrity': 'EXTRACTION_COMPLETE' if equivalence == 'PASS' else
            'EXTRACTION_INCOMPLETE' if missing and not read_errors else 'NOT_VERIFIABLE',
        'download_integrity': 'NOT_VERIFIABLE',
        'status': 'BLOCKED' if equivalence == 'BLOCKED' else 'NEEDS_REVIEW' if equivalence == 'NEEDS_REVIEW' else 'PARTIAL'}
    if out is not None:
        ev = out / 'evidence'
        ev.mkdir(exist_ok=True)
        def dump_csv(name, query):
            cursor = db.execute(query)
            with (ev / name).open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([x[0] for x in cursor.description])
                writer.writerows(cursor)
        dump_csv('member_comparison.csv', 'SELECT * FROM members ORDER BY archive,ordinal')
        dump_csv('local_sha256.csv', 'SELECT path,digest AS sha256,size,side,seq,frame FROM local ORDER BY path')
        for name, classification in [('missing_files.csv', 'MISSING_EXTRACTED_FILE'), ('mismatch_files.csv', 'CONTENT_MISMATCH')]:
            dump_csv(name, "SELECT archive,raw,path,digest AS sha256,result FROM members WHERE result='" + classification + "' ORDER BY path")
        dump_csv('extra_files.csv', "SELECT path,digest AS sha256,size,'EXTRA_EXTRACTED_FILE' AS result FROM local WHERE path NOT IN (SELECT path FROM members WHERE path IS NOT NULL AND is_dir=0) ORDER BY path")
        seqframes = defaultdict(lambda: defaultdict(set))
        for source, query in [('local', 'SELECT side,seq,frame FROM local'), ('zip', 'SELECT side,seq,frame FROM members WHERE is_dir=0 AND seq IS NOT NULL')]:
            for side, seq, frame in db.execute(query):
                seqframes[seq][source + '_' + side].add(frame)
        sequence_rows = []
        for seq, sets in sorted(seqframes.items()):
            row = {'sequence_id': seq}
            for key in ['local_right', 'local_left', 'zip_right', 'zip_left']:
                row[key] = len(sets[key])
            for source in ['zip', 'local']:
                row[source + '_intersection'] = len(sets[source + '_right'] & sets[source + '_left'])
                row[source + '_union'] = len(sets[source + '_right'] | sets[source + '_left'])
            sequence_rows.append(row)
        for name in ['sequence_coverage.csv', 'frame_coverage.csv']:
            with (ev / name).open('w', newline='') as f:
                columns = list(sequence_rows[0]) if sequence_rows else ['sequence_id']
                w = csv.DictWriter(f, fieldnames=columns)
                w.writeheader(); w.writerows(sequence_rows)
        for name, obj in [('archive_integrity.json', {'archives': archive_rows, 'ARCHIVE_READABILITY': readability, 'ARCHIVE_CRC_INTEGRITY': crc_status}),
                          ('archive_inventory.json', {'sides': sides, 'members': 'member_comparison.csv'}),
                          ('extraction_comparison.json', result),
                          ('source_integrity.json', {'status': 'PASS' if unchanged else 'BLOCKED', 'local_files_rehashed': checked, 'errors': errors, 'archives_unchanged': all(x['source_unchanged'] for x in archive_rows)})]:
            (ev / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    db.close()
    return result
