"""Saved tiny arrays only; exact gates reject approximate-only agreement."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import umap_curve_compare as subject
import mimas_pilot as pilot
import test_reducer_trace_compare as fixtures


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy for saved fixtures')
class CurveComparisonTests(unittest.TestCase):
    def fixture(self,coordinate_delta=0.,trace_delta=False):
        f=fixtures.SavedComparisonTests();f.setUp();self.addCleanup(f.tmp.cleanup)
        req=f.fixture();old=json.loads(Path(req['path']).read_text());base=json.loads(Path(old['master']['path']).read_text())
        recipe=dict(a=1.5,b=.9,min_dist=.1,spread=1.);recipe_entry=f.put(recipe)
        plans={};runs={};reference=None;self.traces=[]
        for host in ['pax','mimas']:
            key=host+'-native';oldplan=json.loads(Path(base['host_plans'][key]['path']).read_text())
            closed=json.loads(Path(old['runs'][key]['completion']['path']).read_text())
            if host=='pax':
                native=deepcopy(oldplan);native.update(phase='isolate64',execution_host='pax',diagnostic_profile='native')
                pe=f.put(native);done=deepcopy(closed);done.update(plan_sha256=pe['sha256'],stage_receipts={k:v for k,v in closed['stage_receipts'].items()if k in ['umap1','tsne1']})
                reference=dict(plan=pe,completion=f.put(done),locations=f.put({}))
            plan=deepcopy(oldplan);plan.update(phase='curve64',execution_host=host,curve_control=dict(recipe=recipe_entry))
            pe=f.put(plan);plans[host]=pe
            receipt=json.loads(Path(closed['stage_receipts']['umap1']['path']).read_text())
            path=f.root/(host+'-curve.npz');xy=f.xy.copy()
            if host=='mimas':xy[0,0]+=coordinate_delta
            with path.open('wb')as stream:f.np.savez(stream,embeddings=xy)
            projection=pilot.record(path)
            events=[dict(name='umap_curve_requested',values=dict(a=None,b=None,min_dist=.1,spread=1.)),
                dict(name='umap_curve_effective',values=recipe),dict(name='umap1_optimizer_scalars',values=dict(a=1.5,b=.9)),
                dict(name='umap1_epoch0_scalars',values={}),dict(name='umap1_epoch1_scalars',values={}),
                dict(name='umap1_epoch1_after_head_embedding',raw_sha256='changed'if host=='mimas'and trace_delta else'same')]
            trace=f.put(dict(status='complete',counts=dict(umap=1,tsne=0),sources={},umap_curve=recipe_entry,
                curve_applications=1,optimizer_epoch_trace=True,epoch_calls=500,initialization_replays=0,events=events))
            self.traces.append(trace);receipt.update(projection=projection,trace=trace,outputs=[projection,trace],curve_control=plan['curve_control'])
            done=f.put(dict(status='diagnostic_pilot_complete',plan_sha256=pe['sha256'],stage_receipts=dict(umap1=f.put(receipt))))
            runs[host]=dict(completion=done,locations=f.put({}))
        master=f.put(dict(operation='paired-umap-curve64',maximum_projection_fits=2,maximum_hdbscan_extractions=0,
            host_plans=plans,curve_recipe=recipe_entry,reference_pax=reference,analysis_source=pilot.record(subject.__file__)))
        return f,f.put(dict(operation='closed-umap-curve-comparison',master=master,runs=runs))

    def test_exact_gate_requires_coordinates_trace_and_preserved_pax_reference(self):
        f,req=self.fixture();r=subject.compare(req['path'],f.root/'out')
        self.assertTrue(r['exact_gate_passed']);self.assertTrue(r['pax_native_bitwise_preserved'])
        self.assertEqual(r['new_comparison_fits'],0);self.assertFalse(r['larger_cohort_authorized'])

    def test_approximately_equal_coordinates_do_not_pass_exact_gate(self):
        f,req=self.fixture(coordinate_delta=1e-12);r=subject.compare(req['path'],f.root/'out')
        self.assertTrue(r['geometry']['raw_allclose']);self.assertFalse(r['cross_host_bitwise_identical'])
        self.assertFalse(r['exact_gate_passed'])

    def test_matching_final_coordinates_do_not_override_trace_disagreement(self):
        f,req=self.fixture(trace_delta=True);r=subject.compare(req['path'],f.root/'out')
        self.assertTrue(r['cross_host_bitwise_identical']);self.assertFalse(r['exact_gate_passed'])
        self.assertEqual(r['first_differing_checkpoint'],'umap1_epoch1_after_head_embedding')

    def test_changed_closed_trace_is_rejected(self):
        f,req=self.fixture();Path(self.traces[0]['path']).write_text('{}')
        with self.assertRaises(ValueError):subject.compare(req['path'],f.root/'out')


if __name__=='__main__':unittest.main()
