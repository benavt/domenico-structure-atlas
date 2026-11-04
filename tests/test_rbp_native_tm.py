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
from unittest.mock import patch

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
    def fixture(self, root, n=50):
        members = []
        for i in range(n):
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
                    rows = db.execute("SELECT intent_json FROM waves WHERE state='PENDING'").fetchall()
                    self.assertEqual(len(rows), 1)
                    intent = json.loads(rows[0][0])
                    self.assertGreater(intent['reserved_calls'],0)
                    self.assertLessEqual(len(intent['pairs']),25)
                    self.assertGreater(intent['cumulative_reserved_after'],intent['cumulative_reserved_before'])
                return runner(command, **kwargs)
            run._runner = observed_runner
            self.assertEqual(run.prepare(), 50)
            self.assertEqual(run.store.base._records, {})
            values = run.score_many([('m0',f'm{i}') for i in range(1,50)],workers=4)
            for value in values: self.assertAlmostEqual(value['score'], .82)
            self.assertAlmostEqual(run.score('m1', 'm0')['score'], .82)
            self.assertEqual(run.directional_calls, 98)
            self.assertEqual(run.reserved_calls, 98)
            self.assertEqual(len(calls), 98)
            native = run.store.canonical_record('m0')
            self.assertNotIn('mapping', native)
            self.assertEqual(len(native['mapping_sha256']), 64)
            run.close()
            with sqlite3.connect(run.database) as db:
                waves = [json.loads(r[0]) for r in db.execute('SELECT intent_json FROM waves ORDER BY id')]
                self.assertEqual([len(r['pairs']) for r in waves[:2]],[25,24])
                self.assertEqual(sum(r['reserved_calls'] for r in waves),98)
                self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0],'ok')
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

    def test_preparation_flushes_last_partial_group_and_reuses_873_saved_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root,n=873);old=self.make(root,fixture)
            old.prepare()
            with sqlite3.connect(old.store.database) as db:
                sizes=[r[0] for r in db.execute('SELECT rows FROM batches ORDER BY id')]
                self.assertEqual(sizes,[50]*17+[23])
            old.close()
            reuse=dict(scope=record(old.output/'native-scope.json'),canonical_inputs=record(old.store.database))
            before={Path(r['path']):Path(r['path']).read_bytes() for r in reuse.values()}
            before.update({p:p.read_bytes() for p in (old.output/'inputs/canonical-ca').iterdir()})
            factory=fixture[3]
            class NoNewCanonicalization(factory):
                def canonical_record(self,name):raise AssertionError('Verified saved input should be reused')
            fresh=NativeTMRun(fixture[0],[0,1,2],'a'*64,root/'successor',runtime=fixture[1],
                source_bindings=fixture[2],check=lambda:None,_base_factory=NoNewCanonicalization,_runner=fixture[4],reuse_inputs=reuse)
            self.assertEqual(fresh.prepare(),873)
            self.assertEqual(fresh.store.reused_count,873)
            self.assertEqual(list((fresh.output/'inputs/canonical-ca').iterdir()),[])
            fresh.close()
            self.assertEqual(json.loads((fresh.output/'evidence.json').read_text())['reused_inputs'],873)
            for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)
            self.assertEqual(fixture[-1],[])

    def test_canonical_checkpoint_observes_durable_batch_and_stops_before_next_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.fixture(root, n=101)
            callbacks = []
            def checkpoint(checkpoint_id):
                self.assertFalse(run.store._db.in_transaction)
                self.assertFalse(run.store._batch)
                with sqlite3.connect(run.store.database) as observer:
                    self.assertEqual(observer.execute('SELECT count(*) FROM inputs').fetchone()[0], 50)
                    self.assertEqual(observer.execute('SELECT rows FROM batches').fetchone()[0], 50)
                callbacks.append(checkpoint_id)
                raise RuntimeError('mock cooperative stop after durable input batch')
            run = self.make(root, fixture, checkpoint=checkpoint)
            with self.assertRaisesRegex(RuntimeError, 'cooperative stop'):
                run.prepare()
            self.assertEqual(callbacks, ['canonical-inputs:50'])
            self.assertTrue(run.failed.is_set())
            self.assertEqual(run.store.prepared_count, 50)
            with self.assertRaisesRegex(ValueError, 'stopped'):
                run.prepare()
            with self.assertRaisesRegex(RuntimeError, 'prior failure'):
                run.score('m0', 'm1')
            self.assertEqual(fixture[-1], [])
            run.close(require_complete=False)

    def test_native_checkpoint_observes_sealed_wave_without_open_transaction_and_stops_next_wave(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.fixture(root)
            callbacks = []
            def checkpoint(checkpoint_id):
                callbacks.append(checkpoint_id)
                if checkpoint_id.startswith('canonical-inputs:'):
                    return
                self.assertFalse(run._db.in_transaction)
                self.assertIsNone(run._wave)
                with sqlite3.connect(run.database) as observer:
                    self.assertEqual(observer.execute("SELECT count(*) FROM waves WHERE state='COMPLETE'").fetchone()[0], 1)
                    self.assertEqual(observer.execute("SELECT count(*) FROM proofs WHERE state='COMPLETE'").fetchone()[0], 25)
                    self.assertEqual(observer.execute("SELECT count(*) FROM directions WHERE state='COMPLETE'").fetchone()[0], 50)
                raise RuntimeError('mock cooperative stop after durable native wave')
            run = self.make(root, fixture, checkpoint=checkpoint)
            with self.assertRaisesRegex(RuntimeError, 'cooperative stop'):
                run.score_many([('m0', f'm{i}') for i in range(1, 50)], workers=4)
            self.assertEqual(callbacks, ['canonical-inputs:26', 'native-wave:1:50'])
            self.assertTrue(run.failed.is_set())
            self.assertEqual(run.directional_calls, 50)
            self.assertEqual(run.reserved_calls, 50)
            with self.assertRaisesRegex(RuntimeError, 'prior failure'):
                run.score('m0', 'm1')
            run.close(require_complete=False)
            evidence = read_completed_columns(run.output, expected_source_bindings=fixture[2],
                expected_scope_sha256=record(run.output/'native-scope.json')['sha256'], return_evidence=True)
            self.assertEqual(evidence['historical_charged_calls'], 50)
            self.assertEqual(len(evidence['waves']), 1)
            self.assertEqual(evidence['waves'][0]['state'], 'COMPLETE')
            self.assertTrue(evidence['waves'][0]['sealed'])

    def test_changed_reuse_source_or_native_math_is_rejected(self):
        for damage in ('db','math','cif'):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);fixture=self.fixture(root);old=self.make(root,fixture);old.prepare();old.close()
                scope=old.output/'native-scope.json'
                if damage=='math':
                    value=json.loads(scope.read_text());value['source_bindings']['sources']['src.analysis.tm_state_geometry']['sha256']='f'*64
                    scope.write_text(json.dumps(value))
                reuse=dict(scope=record(scope),canonical_inputs=record(old.store.database))
                if damage=='db':reuse['canonical_inputs']['sha256']='f'*64
                if damage=='cif':next((old.output/'inputs/canonical-ca').iterdir()).write_bytes(b'changed')
                fresh=None
                try:
                    with self.assertRaises(ValueError):
                        fresh=NativeTMRun(fixture[0],[0,1,2],'a'*64,root/'next',runtime=fixture[1],source_bindings=fixture[2],
                            check=lambda:None,_base_factory=fixture[3],_runner=fixture[4],reuse_inputs=reuse)
                        fresh.prepare()
                finally:
                    if fresh is not None:fresh.close(require_complete=False)
                self.assertEqual(fixture[-1],[])

    def test_uncommitted_wave_retains_exact_durable_reservation_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root);run=self.make(root,fixture,maximum_directional_calls=50)
            with patch.object(run,'_commit_barrier',side_effect=RuntimeError('mock lost completion barrier')):
                with self.assertRaisesRegex(RuntimeError,'completion barrier'):
                    run.score_many([('m0',f'm{i}') for i in range(1,26)],workers=4)
            with self.assertRaisesRegex(RuntimeError,'prior failure'):run.score('m0','m1')
            run.close(require_complete=False)
            with sqlite3.connect(run.database) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM proofs').fetchone()[0],0)
                self.assertEqual(db.execute('SELECT count(*) FROM directions').fetchone()[0],0)
                row=db.execute('SELECT state,intent_json,completion_json FROM waves').fetchone()
                self.assertEqual((row[0],row[2]),('PENDING',None))
                self.assertEqual(json.loads(row[1])['reserved_calls'],50)
            read=read_completed_columns(run.output,expected_source_bindings=fixture[2],
                expected_scope_sha256=record(run.output/'native-scope.json')['sha256'],return_evidence=True)
            self.assertEqual(read['columns'],{})
            self.assertEqual(read['historical_charged_calls'],50)
            self.assertEqual(len(read['pending_waves']),1)
            self.assertFalse(read['pending_waves'][0]['sealed'])

    def test_begin_failure_keeps_durable_intent_and_stops_before_any_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root);run=self.make(root,fixture);original=run._db
            class FailedBegin:
                def __getattr__(self,name):return getattr(original,name)
                def execute(self,statement,*args):
                    value=original.execute(statement,*args)
                    if statement=='BEGIN IMMEDIATE':raise RuntimeError('mock BEGIN failure after reservation')
                    return value
            run._db=FailedBegin()
            with self.assertRaisesRegex(RuntimeError,'BEGIN failure'):run.score('m0','m1')
            self.assertTrue(run.failed.is_set());self.assertIsNone(run._wave)
            self.assertFalse(original.in_transaction);self.assertEqual(run.reserved_calls,2)
            with self.assertRaisesRegex(RuntimeError,'prior failure'):run.score('m0','m1')
            run.close(require_complete=False)
            self.assertEqual(fixture[-1],[])
            with sqlite3.connect(run.database) as db:
                row=db.execute('SELECT state,intent_json,completion_json FROM waves').fetchone()
                self.assertEqual((row[0],row[2]),('PENDING',None))
                self.assertEqual(json.loads(row[1])['reserved_calls'],2)

    def test_complete_prefix_survives_failed_next_wave_without_inflating_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root);runner=fixture[-2];count=[0]
            def fail_52(command,**kwargs):
                count[0]+=1
                if count[0]==52:raise subprocess.CalledProcessError(1,command)
                return runner(command,**kwargs)
            run=self.make(root,fixture,_runner=fail_52,maximum_directional_calls=98)
            with self.assertRaises(subprocess.CalledProcessError):run.score_many([('m0',f'm{i}') for i in range(1,50)])
            run.close(require_complete=False)
            read=read_completed_columns(run.output,expected_source_bindings=fixture[2],
                expected_scope_sha256=record(run.output/'native-scope.json')['sha256'],return_evidence=True)
            self.assertEqual([w['state'] for w in read['waves']],['COMPLETE','PENDING'])
            self.assertEqual(read['waves'][0]['verified_complete_pairs'],25)
            self.assertEqual(read['historical_charged_calls'],98)
            self.assertEqual(count[0],52)
            self.assertEqual(run.reserved_calls,98)

    def test_fingerprint_aliases_share_one_exact_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root);members=fixture[0]
            Path(members[1]['source_path']).write_bytes(Path(members[0]['source_path']).read_bytes())
            members[1]['sha256']=members[0]['sha256'];base=fixture[3]
            class Aliased(base):
                def canonical_record(self,name):
                    row=super().canonical_record(name)
                    if name=='m1':
                        path=self.output/'m0.cif';row.update(path=str(path),sha256=record(path)['sha256'])
                    return row
            run=NativeTMRun(members,[0,1,2],'a'*64,root/'run',runtime=fixture[1],source_bindings=fixture[2],check=lambda:None,
                maximum_directional_calls=2,_base_factory=Aliased,_runner=fixture[4])
            run.prepare();result=run.score_many([('m0','m2'),('m1','m2')],workers=2)
            self.assertEqual(run.reserved_calls,2);self.assertEqual(run.directional_calls,2)
            self.assertEqual(result[0]['score'],result[1]['score']);run.close()
            # Both aliases have identical hash-qualified native evidence, but an
            # unlogged model query must still fail the wave's exact provenance.
            with sqlite3.connect(run.database) as db:
                intent=json.loads(db.execute('SELECT intent_json FROM waves').fetchone()[0]);intent['pairs']=intent['pairs'][:1]
                raw=(json.dumps(intent,sort_keys=True,separators=(',',':'))+'\n').encode()
                db.execute('DROP TRIGGER immutable_wave')
                db.execute('UPDATE waves SET intent_json=?,intent_sha256=?',(raw.decode(),hashlib.sha256(raw).hexdigest()))
            with self.assertRaisesRegex(ValueError,'logged native wave queries'):
                read_completed_columns(run.output,expected_source_bindings=fixture[2],
                    expected_scope_sha256=record(run.output/'native-scope.json')['sha256'])

    def test_complete_wave_reader_requires_exact_actual_and_reserved_call_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=self.fixture(root);run=self.make(root,fixture);run.score('m0','m1');run.close()
            with sqlite3.connect(run.database) as db:
                value=json.loads(db.execute('SELECT completion_json FROM waves').fetchone()[0]);value['actual_calls_after']=1
                raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
                db.execute('DROP TRIGGER immutable_wave')
                db.execute('UPDATE waves SET completion_json=?,completion_sha256=?',(raw.decode(),hashlib.sha256(raw).hexdigest()))
            with self.assertRaisesRegex(ValueError,'exact call accounting'):
                read_completed_columns(run.output,expected_source_bindings=fixture[2],
                    expected_scope_sha256=record(run.output/'native-scope.json')['sha256'])


if __name__ == '__main__': unittest.main()
