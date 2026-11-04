"""Guarded native TM evidence for an externally approved saved-cohort runner.

No command-line entry point, approval inference, retries, fits or panel exports.
The pinned library owns canonical correspondence and native TM mathematics.
"""
import gzip
import hashlib
import importlib
import io
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import time

from rbp_states import core_feature_indices, native_call_bounds, need, record, save


LIBRARY_MODULES = ('src.analysis.nested_cluster_candidates',
                   'src.analysis.structure_io', 'src.analysis.tm_state_geometry', 'src.analysis.tm_state_sqlite')


def frozen_source_evidence():
    """Read source hashes for inclusion in the immutable outer execution plan."""
    result = {module: record(importlib.import_module(module).__file__) for module in LIBRARY_MODULES}
    result['consumer.rbp_native_tm'] = record(__file__)
    result['consumer.rbp_states'] = record(Path(__file__).with_name('rbp_states.py'))
    return result


def _bindings(value):
    need(isinstance(value, dict) and re.fullmatch('[0-9a-f]{64}', value.get('plan_sha256', ''))
         and all(re.fullmatch('[0-9a-f]{40}', value.get(key, '')) for key in ('consumer_commit', 'library_commit')),
         'Frozen plan and source commit bindings required')
    need(value.get('sources') == frozen_source_evidence(), 'Frozen native source evidence changed')


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _fresh_db(path, schema):
    path = Path(path).absolute()
    need(not path.exists() and '..' not in path.parts
         and not any(p.is_symlink() for p in [path, *path.parents]), 'Fresh non-symlink SQLite output required')
    connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA journal_mode=DELETE'); connection.execute('PRAGMA synchronous=FULL')
    connection.executescript(schema)
    return connection


class CanonicalEvidenceStore:
    """Keep complete native residue mappings once on disk, not all in RAM.

    The underlying CanonicalCAStore validates originals and uses the pinned
    canonical_ca_arrays ordinal mapping. Compact records passed to the native
    scorer bind the full immutable mapping by SHA256. Originals stay untouched.
    """
    def __init__(self, members, core, sequence_sha256, output, check, *, base_factory=None):
        from src.analysis.tm_state_geometry import CanonicalCAStore
        self.members = {row['model']: dict(row) for row in members}
        need(len(self.members) == len(members) and self.members, 'Distinct frozen source members required')
        self.names, self.check, self.output = list(self.members), check, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.database = self.output/'canonical-inputs.sqlite'
        self._lock, self._closed = threading.RLock(), False
        self._db = _fresh_db(self.database, '''
            CREATE TABLE inputs(model TEXT PRIMARY KEY, compact_json TEXT NOT NULL,
                compact_sha256 TEXT NOT NULL, evidence_gzip BLOB NOT NULL, evidence_sha256 TEXT NOT NULL);
            CREATE TRIGGER immutable_input BEFORE UPDATE ON inputs BEGIN SELECT RAISE(ABORT,'Immutable canonical input'); END;
            CREATE TRIGGER no_input_delete BEFORE DELETE ON inputs BEGIN SELECT RAISE(ABORT,'Canonical input cannot be deleted'); END;
        ''')
        self.base = (base_factory or CanonicalCAStore)(members, [i+1 for i in core], sequence_sha256,
                                                      self.output/'canonical-ca')

    def canonical_record(self, model):
        with self._lock:
            self.check(); need(not self._closed and model in self.members, 'Unknown/closed canonical input')
            member = self.members[model]
            need(record(member['source_path'])['sha256'] == member['sha256'], 'Original model hash changed')
            cached = self._db.execute('SELECT compact_json,compact_sha256 FROM inputs WHERE model=?', (model,)).fetchone()
            if cached:
                raw = cached['compact_json'].encode()
                need(hashlib.sha256(raw).hexdigest() == cached['compact_sha256'], 'Canonical evidence hash changed')
                compact = json.loads(raw)
            else:
                try:
                    full = self.base.canonical_record(model)
                    need(full['source_sha256'] == member['sha256'], 'Canonical source binding changed')
                    raw = _json(full); proof_sha = hashlib.sha256(raw).hexdigest()
                    compact = {key: value for key, value in full.items() if key != 'mapping'}
                    compact['mapping_sha256'] = hashlib.sha256(_json(full['mapping'])).hexdigest()
                    compact['canonical_input_evidence_sha256'] = proof_sha
                    encoded = _json(compact)
                    self._db.execute('INSERT INTO inputs VALUES(?,?,?,?,?)',
                        (model, encoded.decode(), hashlib.sha256(encoded).hexdigest(), gzip.compress(raw, mtime=0), proof_sha))
                finally:
                    # The pinned base class is still responsible for constructing
                    # each record; releasing its memo avoids millions of mapping dicts.
                    self.base._records.clear()
            need(record(compact['path'])['sha256'] == compact['sha256'], 'Canonical CA file changed')
            self.check(); return compact

    def prepare(self):
        for name in self.names: self.canonical_record(name)
        return len(self.names)

    def close(self):
        with self._lock:
            if self._closed: return
            try: need(self._db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Canonical SQLite integrity failed')
            finally: self._db.close(); self._closed = True


def read_canonical_evidence(database, model):
    """Read and verify one saved complete native-label mapping; no coordinate work."""
    from urllib.parse import quote
    path = Path(database).absolute()
    need(not any(p.is_symlink() for p in [path, *path.parents]), 'Symlinked canonical evidence forbidden')
    path = path.resolve(strict=True)
    with sqlite3.connect('file:'+quote(str(path))+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute('SELECT * FROM inputs WHERE model=?', (model,)).fetchone()
        need(row is not None, 'Missing canonical input evidence')
        return _decode_canonical_row(row)


def _decode_canonical_row(row):
    with gzip.GzipFile(fileobj=io.BytesIO(row['evidence_gzip'])) as stream:
        raw = stream.read(8*1024*1024+1)
    need(len(raw) <= 8*1024*1024 and hashlib.sha256(raw).hexdigest() == row['evidence_sha256'],
         'Canonical input evidence changed')
    full = json.loads(raw); compact_raw = row['compact_json'].encode()
    need(hashlib.sha256(compact_raw).hexdigest() == row['compact_sha256'], 'Compact canonical input changed')
    compact = json.loads(compact_raw)
    expected = {key: value for key, value in full.items() if key != 'mapping'}
    expected.update(mapping_sha256=hashlib.sha256(_json(full['mapping'])).hexdigest(),
                    canonical_input_evidence_sha256=row['evidence_sha256'])
    need(compact == expected, 'Compact/full native correspondence differs')
    return full


def read_completed_columns(output, *, expected_source_bindings, expected_scope_sha256):
    """Verify saved complete columns read-only, including every native pair proof.

    This does not authorize a resume, reuse pending directions, or open a scorer.
    The recovery controller separately audits historical reservations and freezes
    which verified columns/pairs it may reuse in an explicitly approved plan.
    """
    import numpy as np
    from urllib.parse import quote
    from src.analysis.tm_state_geometry import validate_cached_pair
    from src.analysis.tm_state_sqlite import decode_complete
    output = Path(output).absolute()
    need(not any(p.is_symlink() for p in [output, *output.parents]), 'Symlinked native output forbidden')
    need(record(output/'native-scope.json')['sha256'] == expected_scope_sha256, 'Frozen native scope hash differs')
    scope = json.loads((output/'native-scope.json').read_text())
    need(scope['source_bindings'] == expected_source_bindings, 'Frozen recovery source bindings differ')
    names = [row['model'] for row in scope['members']]; indices = {name: i for i, name in enumerate(names)}
    values, degrees = {}, [0]*len(names)
    def connection(name):
        path = output/name
        need(path.is_file() and not path.is_symlink(), 'Missing native evidence database')
        db = sqlite3.connect('file:'+quote(str(path))+'?mode=ro', uri=True); db.row_factory = sqlite3.Row
        need(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Native evidence SQLite integrity failed')
        return db
    with connection('inputs/canonical-inputs.sqlite') as inputs:
        input_hashes = {}
        expected_members = {row['model']: row for row in scope['members']}
        for row in inputs.execute('SELECT * FROM inputs'):
            full = _decode_canonical_row(row)
            need(full['model'] in expected_members and full['source_sha256'] == expected_members[full['model']]['sha256']
                 and full['sequence_sha256'] == scope['sequence_sha256']
                 and full['canonical_positions'] == scope['canonical_positions'], 'Frozen native input binding differs')
            input_hashes[full['model']] = [full['source_sha256'], full['sha256']]
    with connection('native-directions.sqlite') as queries, connection('native-tm.sqlite') as proofs:
        for row in queries.execute('SELECT * FROM model_pairs'):
            saved = proofs.execute('SELECT * FROM proofs WHERE pair_key=?', (row['result_key'],)).fetchone()
            need(saved is not None and saved['state'] == 'COMPLETE', 'Column has no complete native proof')
            decoded = decode_complete(saved)
            result = validate_cached_pair(decoded['result'], decoded['key'], decoded['intent']['fingerprint'], scope['protocol'])
            left, right = row['first'], row['second']
            need(left in indices and right in indices and left < right and row['score'] == result['score']
                 and decoded['intent']['fingerprint']['canonical_positions'] == scope['canonical_positions']
                 and decoded['intent']['fingerprint']['sequence_sha256'] == scope['sequence_sha256'],
                 'Column pair identity/score differs')
            # Bind original/canonical pair hashes, including aliases of identical files.
            need(left in input_hashes and right in input_hashes and sorted([input_hashes[left], input_hashes[right]]) ==
                 decoded['intent']['fingerprint']['original_and_canonical_sha256'], 'Column native source hashes differ')
            i, j = indices[left], indices[right]
            values[tuple(sorted((i, j)))] = float(row['score']); degrees[i] += 1; degrees[j] += 1
    columns = {}
    for i, count in enumerate(degrees):
        if count == len(names)-1:
            columns[i] = np.asarray([1. if i == j else values[tuple(sorted((i, j)))] for j in range(len(names))])
    return columns


class NativeTMRun:
    """Fresh guarded native scorer compatible with rbp_states.run_states.

    ``runtime`` freezes usalign, usalign_sha256 and optional timeout_seconds<=180.
    ``source_bindings`` freezes plan_sha256, consumer_commit, library_commit, and
    sources=frozen_source_evidence(). ``check`` is the outer pressure/cancellation
    guard. ``prepare`` streams canonical input preparation. Complete unordered
    pair requests are cached in the pinned native SQLite scorer. Every individual
    direction is separately journaled before launch and after its mocked/native
    runner returns. Interrupted intents persist; automatic retry is forbidden.
    """
    def __init__(self, members, retained_core_indices, sequence_sha256, output, *, runtime,
                 source_bindings, check, maximum_directional_calls=None, reserve_calls=lambda count: None,
                 _base_factory=None, _runner=None):
        from src.analysis.tm_state_sqlite import SQLiteNativeTMScorer
        _bindings(source_bindings)
        self.output, self.check = Path(output), check
        core = list(retained_core_indices)
        core_feature_indices(max(core)+1 if core else 0, core)
        cap = native_call_bounds(len(members))['directional_call_ceiling']
        maximum_directional_calls = cap if maximum_directional_calls is None else maximum_directional_calls
        need(type(maximum_directional_calls) is int and 0 <= maximum_directional_calls <= cap, 'Native reservation cap differs')
        need(callable(check), 'Explicit pressure guard required')
        self.check(); need(not self.output.exists(), 'Fresh native run output required')
        self.output.mkdir(parents=True)
        self.source_bindings, self.runtime = source_bindings, dict(runtime)
        self._lock, self._local, self._closed = threading.RLock(), threading.local(), False
        self._reserved, self._direction_count = 0, 0
        self._outer_reserve, self._runner = reserve_calls, _runner or subprocess.run
        self.maximum_directional_calls = maximum_directional_calls
        self.directions_database = self.output/'native-directions.sqlite'
        self._directions = _fresh_db(self.directions_database, '''
            CREATE TABLE directions(id INTEGER PRIMARY KEY,state TEXT NOT NULL,
                command_json TEXT NOT NULL,started_ns INTEGER NOT NULL, result_json TEXT);
            CREATE TRIGGER immutable_direction BEFORE UPDATE ON directions WHEN OLD.state!='PENDING'
                BEGIN SELECT RAISE(ABORT,'Immutable native direction'); END;
            CREATE TRIGGER no_direction_delete BEFORE DELETE ON directions
                BEGIN SELECT RAISE(ABORT,'Native direction cannot be deleted'); END;
            CREATE TABLE model_pairs(first TEXT NOT NULL,second TEXT NOT NULL,score REAL NOT NULL,
                result_key TEXT NOT NULL,PRIMARY KEY(first,second));
            CREATE TRIGGER immutable_model_pair BEFORE UPDATE ON model_pairs
                BEGIN SELECT RAISE(ABORT,'Immutable native model pair'); END;
            CREATE TRIGGER no_model_pair_delete BEFORE DELETE ON model_pairs
                BEGIN SELECT RAISE(ABORT,'Native model pair cannot be deleted'); END;
        ''')
        self.store = CanonicalEvidenceStore(members, core, sequence_sha256, self.output/'inputs', check,
                                             base_factory=_base_factory)
        self.database = self.output/'native-tm.sqlite'
        try:
            self.scorer = SQLiteNativeTMScorer(self.store, runtime['usalign'], self.database, maximum_directional_calls,
                executable_sha256=runtime['usalign_sha256'], timeout=runtime.get('timeout_seconds', 180),
                reserve_calls=self._reserve, on_call=self._begin, check_cancel=check, runner=self._direction)
        except BaseException:
            self.store.close(); self._directions.close(); self._closed = True; raise
        self.protocol, self.failed = self.scorer.protocol, self.scorer.failed
        save(self.output/'native-scope.json', dict(source_bindings=source_bindings, runtime=runtime,
            canonical_positions=[i+1 for i in core], sequence_sha256=sequence_sha256,
            members=members, maximum_directional_calls=maximum_directional_calls,
            protocol=self.protocol, automatic_retries=0))

    @property
    def directional_calls(self): return self.scorer.directional_calls

    @property
    def reserved_calls(self): return self._reserved

    def _reserve(self, count):
        with self._lock:
            self.check(); need(self._reserved+count <= self.maximum_directional_calls, 'Native reservation exceeded')
            self._outer_reserve(count); self._reserved += count

    def _begin(self, command):
        with self._lock:
            self.check(); self._direction_count += 1
            need(self._direction_count <= self._reserved, 'Native direction lacks reservation')
            cursor = self._directions.execute('INSERT INTO directions(state,command_json,started_ns) VALUES(?,?,?)',
                ('PENDING', _json(command).decode(), time.time_ns()))
            self._local.direction_id = cursor.lastrowid

    def _direction(self, command, **kwargs):
        try:
            self.check(); result = self._runner(command, **kwargs)
            evidence = dict(stdout=result.stdout, stderr=result.stderr,
                returncode=getattr(result, 'returncode', 0), matrix_text=Path(command[-1]).read_text())
            state = 'COMPLETE'
        except BaseException as error:
            evidence, state = dict(error_type=type(error).__name__, error=str(error)), 'FAILED'
            with self._lock:
                self._directions.execute('UPDATE directions SET state=?,result_json=? WHERE id=? AND state=?',
                    (state, _json(evidence).decode(), self._local.direction_id, 'PENDING'))
            raise
        with self._lock:
            self._directions.execute('UPDATE directions SET state=?,result_json=? WHERE id=? AND state=?',
                (state, _json(evidence).decode(), self._local.direction_id, 'PENDING'))
        self.check(); return result

    def prepare(self):
        self.check(); return self.store.prepare()

    def score(self, first, second):
        self.check(); result = self.scorer.score(first, second)
        if first != second:
            left, right = sorted((first, second))
            with self._lock:
                row = self._directions.execute('SELECT score,result_key FROM model_pairs WHERE first=? AND second=?', (left, right)).fetchone()
                if row is None:
                    self._directions.execute('INSERT INTO model_pairs VALUES(?,?,?,?)', (left, right, result['score'], result['key']))
                else:
                    need(row['score'] == result['score'] and row['result_key'] == result['key'], 'Native pair cache differs')
        self.check(); return result

    def close(self, *, require_complete=True):
        if self._closed: return
        error = None
        try:
            self.scorer.close(require_complete=require_complete)
            need(self._directions.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Direction SQLite integrity failed')
            if require_complete:
                need(self._directions.execute("SELECT count(*) FROM directions WHERE state!='COMPLETE'").fetchone()[0] == 0,
                     'Unresolved native directions; no automatic retry')
        except BaseException as caught:
            error = caught
        finally:
            for closer in (self.store.close, self._directions.close):
                try: closer()
                except BaseException as caught:
                    if error is None: error = caught
            self._closed = True
            save(self.output/'evidence.json', dict(schema_version=1, protocol=self.protocol,
                native_evidence=dict(format='canonical-TM-lossless-sqlite-proof-v1', file=record(self.database)),
                canonical_inputs=record(self.store.database), native_directions=record(self.directions_database),
                actual_directional_calls=self.directional_calls, reserved_directional_calls=self.reserved_calls,
                maximum_directional_calls=self.maximum_directional_calls, complete=require_complete and error is None,
                error=None if error is None else dict(type=type(error).__name__, message=str(error)),
                source_bindings=self.source_bindings))
        if error is not None: raise error

    def __enter__(self): return self

    def __exit__(self, error_type, error, traceback): self.close(require_complete=error_type is None)
