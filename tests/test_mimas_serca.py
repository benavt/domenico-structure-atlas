"""Production guards use fake constructors and saved arrays; never estimator fits."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mimas_serca as subject
import mimas_cpu as cpu
import mimas_contract as contract


class PrimaryCurveTests(unittest.TestCase):
    def fixture(self):
        class Fake:
            def __init__(self,n_components=2,min_dist=.1,spread=1.,random_state=7,a=None,b=None):
                self.a,self.b=a,b
        return SimpleNamespace(UMAP=Fake),dict(a=1.5,b=.8,min_dist=.1,spread=1.)
    def test_public_parameters_and_signature_restore_on_success_and_failure(self):
        module,recipe=self.fixture();original=module.UMAP.__init__
        for fail in(False,True):
            try:
                with subject.primary_curve(module,recipe)as calls:
                    value=module.UMAP(random_state=123);self.assertEqual((value.a,value.b),(1.5,.8));self.assertEqual(calls,[123])
                    if fail:raise RuntimeError('owned failure')
            except RuntimeError:pass
            self.assertIs(module.UMAP.__init__,original)
    def test_no_wrong_protocol_or_third_repeat(self):
        module,recipe=self.fixture()
        for args in [dict(n_components=10),dict(min_dist=0.),dict(random_state=42),dict(a=5.)]:
            with self.assertRaises(ValueError),subject.primary_curve(module,recipe):module.UMAP(**args)
        with self.assertRaises(ValueError),subject.primary_curve(module,recipe):
            module.UMAP(random_state=7);module.UMAP(random_state=123);module.UMAP(random_state=123)


class ProductionGuardTests(unittest.TestCase):
    def test_full_curve_provenance_guard_accepts_three_stage_scope_before_reading_proofs(self):
        import reducer_diagnostic as diag
        class ProvenanceReached(Exception):pass
        entry=dict(path='/bound/recipe',bytes=1,sha256='a'*64)
        p=dict(phase='serca-full',diagnostic_profile='native',stages=['umap1','labels','density'],curve_control=dict(policy='public-umap-a-b-from-closed-pax-trace',recipe=entry),trace_sources={'layouts':{}},frozen_files=[entry])
        fake=SimpleNamespace(bound=lambda _:(_ for _ in()).throw(ProvenanceReached()))
        with patch.object(diag,'modules',return_value=(fake,None,None)):
            with self.assertRaises(ProvenanceReached):diag.verify_curve_plan(p)
            p['stages']=['umap1']
            with self.assertRaisesRegex(ValueError,'scope required'):diag.verify_curve_plan(p)
    def test_long_deadline_is_enabled_only_for_explicit_production(self):
        import mimas_package as package
        p=dict(run_root='/run/scoped',runtime_prefix='/runtime',runtime_python='/runtime/bin/python',wheelhouse='/wheels',requirements='/requirements',runner='/controller.py',partition='debug',phase='smoke',resources=dict(maximum_cpus=4,maximum_memory_bytes=64*1024**3,maximum_wall_seconds=8*3600))
        with self.assertRaises(ValueError):package.smoke_batch('/plan.json',p,'/base/python',environment_exists=True)
        p['phase']='serca-full';body=package.smoke_batch('/plan.json',p,'/base/python',environment_exists=True)
        self.assertIn('#SBATCH --time=480',body)
    def test_existing_completed_methods_cannot_consume_new_fits(self):
        for stage in('labels','density'):
            result=dict(phase_complete=True,stage=stage,inputs={'frozen':1},scientific_fits=dict(PCA=1,UMAP=0,TSNE=0,HDBSCAN=0))
            with self.assertRaisesRegex(ValueError,'must never be refitted'):subject.validate_receipt({'inputs':{'frozen':1}},stage,result)
    def test_complete_density_requires_default_hdbscan_and_budget(self):
        r=dict(phase_complete=True,stage='density',inputs={},scientific_fits=dict(PCA=0,UMAP=15,TSNE=0,HDBSCAN=30),full_recursive_HDBSCAN=True,default_cluster_labels='HDBSCAN')
        subject.validate_receipt({'inputs':{}},'density',r)
        for key,value in [('default_cluster_labels','UMAP KNN'),('full_recursive_HDBSCAN',False)]:
            with self.assertRaises(ValueError):subject.validate_receipt({'inputs':{}},'density',dict(r,**{key:value}))
        r['scientific_fits']['UMAP']=16
        with self.assertRaises(ValueError):subject.validate_receipt({'inputs':{}},'density',r)
    def test_exact_gate_rejects_one_bit_or_dtype_change(self):
        class Array:
            shape=(5072,2);dtype=SimpleNamespace(str='<f4')
            def __init__(self,raw):self.raw=raw
            def tobytes(self):return self.raw
        r=dict(models=['frozen'],local_paths=['native'],projection={'path':'new'})
        old=dict(projection={'path':'old'},local_paths=['native'])
        with patch.object(subject,'read',return_value=old),patch.object(subject,'projection',side_effect=[Array(b'a'),Array(b'a')]):
            self.assertEqual(subject.require_full_correspondence({},r,{} )['projection_fits'],0)
        with patch.object(subject,'read',return_value=old),patch.object(subject,'projection',side_effect=[Array(b'a'),Array(b'b')]):
            with self.assertRaisesRegex(ValueError,'stop before labels'):subject.require_full_correspondence({},r,{})
    def test_malformed_projection_is_rejected_before_numpy_import(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'huge.npz'
            def header(dtype,shape):
                import struct
                body=(repr(dict(descr=dtype,fortran_order=False,shape=shape))+'\n').encode()
                return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(body))+body
            with zipfile.ZipFile(path,'w')as z:
                z.writestr('embeddings.npy',header('<f4',(5072,2)))
                z.writestr('filenames.npy',header('<U999999999',(5072,)))
                z.writestr('filepaths.npy',header('<U1',(5072,)))
            with self.assertRaises(ValueError):subject.projection(cpu.record(path),['unused'])
    @unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy optional in source CI')
    def test_saved_projection_roundtrip_preserves_dtype_values_and_paths(self):
        import numpy as np
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'saved.npz';names=[f'm{i}'for i in range(5072)];paths=['/old/'+n+'.pdb'for n in names];xy=np.arange(10144,dtype=np.float32).reshape(5072,2)
            np.savez_compressed(p,embeddings=xy,filenames=np.asarray(names),filepaths=np.asarray(paths))
            value=subject.projection(cpu.record(p),names,paths)
            self.assertEqual(value.dtype,xy.dtype);self.assertEqual(value.tobytes(),xy.tobytes())
            with self.assertRaises(ValueError):subject.projection(cpu.record(p),names,list(reversed(paths)))
    def test_full_scientific_phase_is_bounded_and_fresh_processes_include_every_stage(self):
        self.assertEqual(cpu.STAGES['serca-full'],('umap1','labels','density'))
        self.assertEqual(cpu.FIT_LIMITS['serca-full'],(18,30))
        self.assertEqual(cpu.STAGE_LIMITS['density'],(15,30))
        # Actual sanitation exercises the new full-phase stage list with existing base fixture.
        with tempfile.TemporaryDirectory()as tmp:
            plan=dict(run_root=str(Path(tmp)/'run'),runtime_prefix='/runtime',runtime_python='/runtime/bin/python',
                      resources=dict(maximum_cpus=4,maximum_memory_bytes=64*1024**3,maximum_swap_bytes=0,maximum_pids=64))
            plan.update(phase='serca-full',diagnostic_profile='native')
            for stage in cpu.STAGES['serca-full']:
                plan['active_stage']=stage
                env=contract.sanitized_environment(plan)
                self.assertEqual(env['NUMBA_NUM_THREADS'],'1')
                self.assertNotIn('PYTHONPATH',env)


class MemoryGuardTests(unittest.TestCase):
    def fixture(self,root):
        policy=dict(admission_available_bytes=128*1024**3,runtime_available_bytes=32*1024**3,maximum_job_memory_full_percent=.5,maximum_job_memory_current_bytes=60*1024**3,consecutive_failures=3)
        return subject.MemoryGuard(dict(production_pressure=policy),root)
    def test_admission_and_actual_interval_pressure_stop_only_after_three_samples(self):
        with tempfile.TemporaryDirectory()as tmp:
            guard=self.fixture(Path(tmp))
            with patch.object(guard,'available',return_value=127*1024**3):
                with self.assertRaisesRegex(ValueError,'no fit'):guard.admit()
            with patch.object(guard,'available',return_value=256*1024**3),patch.object(guard,'full_total',side_effect=[0,50000,100000,150000]):
                guard.admit();guard.sample({'memory':{'current_bytes':4*1024**3}},0);guard.sample({'memory':{'current_bytes':4*1024**3}},5);guard.sample({'memory':{'current_bytes':4*1024**3}},10)
                with self.assertRaisesRegex(ValueError,'this job only'):guard.sample({'memory':{'current_bytes':4*1024**3}},15)
            self.assertEqual(guard.failures,3)
    def test_old_pressure_average_is_irrelevant_when_interval_counter_clears(self):
        with tempfile.TemporaryDirectory()as tmp:
            guard=self.fixture(Path(tmp))
            with patch.object(guard,'available',return_value=256*1024**3),patch.object(guard,'full_total',side_effect=[0,50000,50000,50000]):
                for now in(0,5,10,15):guard.sample({'memory':{'current_bytes':4*1024**3}},now)
            self.assertEqual(guard.failures,0)

    def test_missing_psi_is_explicit_and_current_memory_still_stops_the_job(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);guard=self.fixture(root);observation={'memory':{'current_bytes':61*1024**3}}
            with patch.object(guard,'available',return_value=256*1024**3),patch.object(guard,'full_total',return_value=None):
                guard.sample(observation,0);guard.sample(observation,5)
                with self.assertRaisesRegex(ValueError,'this job only'):guard.sample(observation,10)
            rows=[json.loads(x)for x in(root/'memory-pressure.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['job_memory_full_interval_percent']is None and r['job_psi_available']is False for r in rows))
            self.assertEqual(guard.failures,3)
    def test_a_known_psi_counter_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory()as tmp:
            guard=self.fixture(Path(tmp));observation={'memory':{'current_bytes':4*1024**3}}
            with patch.object(guard,'available',return_value=256*1024**3),patch.object(guard,'full_total',side_effect=[0,None]):
                guard.sample(observation,0)
                with self.assertRaisesRegex(ValueError,'PSI disappeared'):guard.sample(observation,5)


if __name__=='__main__':unittest.main()
