import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rbp_native_tm import NativeTMRun, frozen_source_evidence, read_canonical_evidence, read_completed_columns
from rbp_states import record

try:
    import numpy
    from src.analysis.tm_state_sqlite import iter_complete_proofs
    NATIVE = True
except ImportError:
    NATIVE = False


@unittest.skipUnless(NATIVE, 'Pinned native evidence imports optional in source CI')
class NativeEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        members = []
        for i in range(50):
            path = root/f'm{i}.source'; path.write_bytes(f'mocked original {i}'.encode())
            members.append(dict(model=f'm{i}', source_path=str(path), sha256=record(path)['sha256']))
        binary = root/'mock-USalign'; binary.write_bytes(b'NEVER EXECUTED')
        runtime = dict(usalign=str(binary), usalign_sha256=record(binary)['sha256'])
        bindings = dict(plan_sha256='c'*64, consumer_commit='d'*40, library_commit='e'*40,
                        sources=frozen_source_evidence())
        class MockCanonicalStore:
            """Fixed fake evidence; neither model parser nor coordinate math runs."""
            def __init__(self, inputs, positions, sequence_sha256, output):
                self.members = {r['model']: r for r in inputs}
                self.output = Path(output); self.output.mkdir()
                self.positions, self.sequence_sha256, self._records = positions, sequence_sha256, {}
            def canonical_record(self, name):
                member = self.members[name]
                path = self.output/(name+'.cif'); path.write_bytes(('MOCK '+name).encode())
                mapping = [dict(canonical_position=ordinal, chain='native-X', native_residue=[' ', n, code])
                           for ordinal, n, code in [(1, 0, ' '), (2, 1, 'A'), (3, 1, 'B')]]
                result = dict(model=name, path=str(path), sha256=record(path)['sha256'],
                    source_sha256=member['sha256'], sequence_sha256=self.sequence_sha256,
                    canonical_positions=self.positions, mapping=mapping, ca_count=3)
                self._records[name] = result
                return result
        calls = []
        def runner(command, **kwargs):
            self.assertEqual(kwargs, dict(capture_output=True, text=True, timeout=180, check=True))
            calls.append(command)
            Path(command[-1]).write_text('0 0 1 0 0\n1 0 0 1 0\n2 0 0 0 1\n')
            value = .81 if Path(command[1]).stem == 'm0' else .83
            return SimpleNamespace(stdout=f'TM-score= {value:.5f} (normalized by average length)\n',
                                   stderr='', returncode=0)
        return members, runtime, bindings, MockCanonicalStore, runner, calls

    def make(self, root, fixture, **kwargs):
        members, runtime, bindings, factory, runner, _ = fixture
        return NativeTMRun(members, [0, 1, 2], 'a'*64, root/'run', runtime=runtime,
            source_bindings=bindings, check=kwargs.pop('check', lambda: None),
            _base_factory=factory, _runner=kwargs.pop('_runner', runner), **kwargs)

    def test_durable_bidirectional_cache_native_mapping_and_complete_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture = self.fixture(root)
            members, runtime, bindings, factory, runner, calls = fixture
            checks = []
            run = self.make(root, fixture, maximum_directional_calls=98, check=lambda: checks.append(1))
            def observed_runner(command, **kwargs):
                with sqlite3.connect(run.database) as db:
                    self.assertGreater(db.execute("SELECT count(*) FROM proofs WHERE state='PENDING'").fetchone()[0], 0)
                with sqlite3.connect(run.directions_database) as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM directions WHERE state='PENDING'").fetchone()[0], 1)
                return runner(command, **kwargs)
            run._runner = observed_runner
            self.assertEqual(run.prepare(), 50)
            self.assertEqual(run.store.base._records, {})
            for i in range(1, 50): self.assertAlmostEqual(run.score('m0', f'm{i}')['score'], .82)
            self.assertAlmostEqual(run.score('m1', 'm0')['score'], .82)
            self.assertEqual(run.directional_calls, 98)
            self.assertEqual(run.reserved_calls, 98)
            self.assertEqual(len(calls), 98)
            native = run.store.canonical_record('m0')
            self.assertNotIn('mapping', native)
            self.assertEqual(len(native['mapping_sha256']), 64)
            run.close()
            self.assertEqual(len(list(iter_complete_proofs(run.database))), 49)
            full = read_canonical_evidence(run.store.database, 'm0')
            self.assertEqual([(r['native_residue'][1], r['native_residue'][2]) for r in full['mapping']],
                             [(0, ' '), (1, 'A'), (1, 'B')])
            columns = read_completed_columns(root/'run', expected_source_bindings=bindings,
                expected_scope_sha256=record(root/'run'/'native-scope.json')['sha256'])
            self.assertEqual(set(columns), {0})
            self.assertEqual(columns[0][0], 1.)
            self.assertTrue(all(abs(value-.82) < 1e-12 for value in columns[0][1:]))
            evidence = json.loads((root/'run'/'evidence.json').read_text())
            self.assertTrue(evidence['complete'])
            self.assertGreater(len(checks), len(calls))
            with sqlite3.connect(run.directions_database) as db:
                with self.assertRaisesRegex(sqlite3.IntegrityError, 'Immutable'):
                    db.execute("UPDATE directions SET result_json='{}' WHERE id=1")

    def test_budget_and_pressure_stop_before_any_mock_native_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture = self.fixture(root)
            run = self.make(root, fixture, maximum_directional_calls=0)
            with self.assertRaisesRegex(ValueError, 'ceiling'): run.score('m0', 'm1')
            self.assertEqual(fixture[-1], [])
            run.close(require_complete=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture = self.fixture(root); stop = [False]
            def guard():
                if stop[0]: raise RuntimeError('mock sustained pressure stop')
            run = self.make(root, fixture, check=guard)
            stop[0] = True
            with self.assertRaisesRegex(RuntimeError, 'pressure'): run.score('m0', 'm1')
            self.assertEqual(fixture[-1], [])
            run.close(require_complete=False)

    def test_failed_second_direction_preserves_evidence_and_forbids_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture = self.fixture(root); runner = fixture[-2]; count = [0]
            def fail_second(command, **kwargs):
                count[0] += 1
                if count[0] == 2: raise subprocess.CalledProcessError(1, command, stderr='mock failure')
                return runner(command, **kwargs)
            run = self.make(root, fixture, _runner=fail_second)
            with self.assertRaises(subprocess.CalledProcessError): run.score('m0', 'm1')
            with self.assertRaisesRegex(RuntimeError, 'prior failure'): run.score('m0', 'm1')
            run.close(require_complete=False)
            with sqlite3.connect(run.database) as db:
                self.assertEqual(db.execute('SELECT state FROM proofs').fetchone()[0], 'PENDING')
            with sqlite3.connect(run.directions_database) as db:
                self.assertEqual([row[0] for row in db.execute('SELECT state FROM directions ORDER BY id')], ['COMPLETE', 'FAILED'])
            self.assertEqual(count[0], 2)
            self.assertFalse(json.loads((root/'run'/'evidence.json').read_text())['complete'])

    def test_frozen_source_and_runtime_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture = self.fixture(root)
            fixture[2]['sources']['consumer.rbp_states']['sha256'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'source evidence changed'): self.make(root, fixture)
            self.assertFalse((root/'run').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture = self.fixture(root)
            fixture[1]['usalign_sha256'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'executable hash'): self.make(root, fixture)
            self.assertEqual(fixture[-1], [])


if __name__ == '__main__': unittest.main()
