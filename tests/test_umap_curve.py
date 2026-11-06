"""Curve provenance and real isolated preflight only; zero scientific fits."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mimas_cpu as cpu
import mimas_contract as contract
import reducer_diagnostic as diag


def fixture(root):
    def put(name,value):
        p=root/name;p.write_text(json.dumps(value));return cpu.record(p)
    recipe=dict(a=1.5,b=.9,spread=1.,min_dist=.1)
    data=dict(model_count=64,construct_residues=993,retained_reference_positions=list(range(57,951)),
        feature_sha256='f'*64,model_order_sha256='e'*64,pair_mask_sha256='d'*64)
    scalar=put('scalar.json',dict(a=recipe['a'],b=recipe['b']))
    trace=put('trace.json',dict(status='complete',counts=dict(umap=1,tsne=0),events=[dict(name='umap1_optimizer_scalars',artifact=scalar)]))
    receipt=put('receipt.json',dict(stage='umap1',phase_complete=True,diagnostic_profile='native',native_library_commit='a'*40,
        inputs=data,trace=trace,parameters=dict(n_neighbors=15,min_dist=.1,n_components=2,verbose=False,random_seed=42)))
    source=put('source-plan.json',dict(execution_host='pax',phase='isolate64',diagnostic_profile='native'))
    closed=put('completion.json',dict(status='diagnostic_pilot_complete',plan_sha256=source['sha256'],stage_receipts=dict(umap1=receipt)))
    spec=dict(policy='public-umap-a-b-from-closed-pax-trace',recipe=put('recipe.json',recipe),source_scalars=scalar,
        source_trace=trace,source_receipt=receipt,source_plan=source,source_completion=closed)
    return dict(phase='curve64',stages=['umap1'],diagnostic_profile='native',curve_control=spec,
        native=dict(library_commit='a'*40),trace_sources={'layouts':{}},inputs=data,
        frozen_files=[v for v in spec.values()if isinstance(v,dict)])


class CurveContractTests(unittest.TestCase):
    def test_closed_recipe_provenance_and_exact_one_fit_budget(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=fixture(Path(tmp));diag.verify_curve_plan(p)
            self.assertEqual(cpu.STAGES['curve64'],('umap1',));self.assertEqual(cpu.FIT_LIMITS['curve64'],(1,0))
            self.assertEqual(diag.stages(p),('umap1',))
            p.update(run_root=str(Path(tmp)/'run'),runtime_python='/runtime/bin/python',runtime_prefix='/runtime',
                resources=dict(maximum_cpus=2,maximum_memory_bytes=8*1024**3,maximum_swap_bytes=0,maximum_pids=64),active_stage='tsne1')
            with self.assertRaises(ValueError):contract.sanitized_environment(p,environ={})

    def test_recipe_drift_wrong_cohort_and_unbound_or_open_source_stop(self):
        for change in ['recipe','mask','binding','closed','profile']:
            with self.subTest(change=change),tempfile.TemporaryDirectory()as tmp:
                p=fixture(Path(tmp))
                if change=='recipe':
                    e=p['curve_control']['recipe'];q=Path(e['path']);d=json.loads(q.read_text());d['a']=2.;q.write_text(json.dumps(d))
                elif change=='mask':p['inputs']=dict(p['inputs'],pair_mask_sha256='0'*64)
                elif change=='binding':p['frozen_files'].pop()
                elif change=='closed':Path(p['curve_control']['source_completion']['path']).write_text('{}')
                else:p['diagnostic_profile']='generic-jit-v1'
                with self.assertRaises(ValueError):diag.verify_curve_plan(p)

    def test_actual_isolated_curve_preflight_works_without_execution_approval(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=fixture(root);source=Path(cpu.__file__).absolute().parent
            stub=root/'stub.json';stub.write_text('{}');entry=cpu.record(stub)
            p.update(operation='mimas-cpu-validation',maximum_projection_fits=1,maximum_hdbscan_extractions=0,
                create_environment=False,run_root=str(root/'run'),runtime_python='/runtime/bin/python',runtime_prefix='/runtime',
                portability_helper=entry,runtime_payload=entry,dependency_sources=[],
                trace_sources={k:entry for k in ['umap','sklearn','layouts']})
            p['frozen_files'] += [entry]+[cpu.record(source/name)for name in ['mimas_cpu.py','mimas_contract.py','mimas_smoke.py','mimas_pilot.py','reducer_diagnostic.py']]
            path=root/'plan.json';path.write_text(json.dumps(p))
            for action in ['preflight','run']:
                r=subprocess.run([sys.executable,'-I','-B',cpu.__file__,action,str(path)],capture_output=True,text=True,timeout=10)
                if action=='preflight':self.assertEqual(r.returncode,0,r.stderr)
                else:self.assertNotEqual(r.returncode,0)
                self.assertFalse((root/'run').exists())


if __name__=='__main__':unittest.main()
