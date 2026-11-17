"""Tiny saved projections/checkpoint metadata only; no scientific engines."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reducer_trace_compare as subject
import hpc_pilot as pilot
import reducer_diagnostic as diag


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy for saved fixtures')
class SavedComparisonTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np;self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.counter=0
        self.names=['m'+str(i)for i in range(64)];self.xy=np.column_stack((np.arange(64),np.sin(np.arange(64)))).astype(float)
    def put(self,value):
        p=self.root/('json'+str(self.counter)+'.json');self.counter+=1;p.write_text(json.dumps(value));return pilot.record(p)
    def fixture(self):
        plans={};runs={};self.receipts=[]
        adapter=pilot.record(diag.__file__);native_adapter=pilot.record(pilot.__file__)
        source=pilot.record(subject.__file__)
        helper={'path':'/bound/helper','bytes':1,'sha256':'b'*64}
        order=hashlib.sha256('\n'.join(self.names).encode()).hexdigest()
        for key in ['workstation-native','hpc-native','workstation-generic','hpc-generic']:
            profile='native'if key.endswith('-native')else'generic-jit-v1'
            inputs=dict(model_count=64,construct_residues=993,retained_reference_positions=list(range(57,951)),feature_sha256='f'*64,pair_mask_sha256='e'*64,model_order_sha256=order)
            plan=dict(phase='repro64',inputs=inputs,diagnostic_profile=profile,native=dict(library_commit=pilot.LIBRARY_COMMIT),
                trace_sources={},portability_helper=helper,frozen_files=[adapter,native_adapter])
            plans[key]=self.put(plan);closed={}
            for stage in ['umap1','tsne1','umap2','tsne2']:
                method=stage[:-1];p=self.root/(key+stage+'.npz')
                with p.open('wb')as stream:self.np.savez(stream,embeddings=self.xy)
                projection=pilot.record(p)
                trace=self.put(dict(status='complete',counts=dict(umap=int(method=='umap'),tsne=int(method=='tsne')),
                    sources={},events=[dict(name='checkpoint',shape=[64,2],dtype='<f4',raw_sha256='c'*64)]))
                params=(dict(n_neighbors=15,min_dist=.1,n_components=2,verbose=False,random_seed=42)if method=='umap'
                    else dict(perplexity=30.,implementation='python',verbose=False,random_seed=42))
                r=dict(target='toy',models=self.names,model_flags=[{}]*64,inputs=inputs,stage=stage,phase_complete=True,
                    diagnostic_profile=profile,native_library_commit=pilot.LIBRARY_COMMIT,portability_helper=helper,
                    diagnostic_adapter=adapter,module_source=native_adapter,native_sources={n:dict(sha256=h)for n,h in pilot.LIBRARY_SOURCE_HASHES.items()},
                    trace=trace,scientific_fits=dict(PCA=int(method=='tsne'),UMAP=int(method=='umap'),TSNE=int(method=='tsne'),HDBSCAN=0),
                    parameters=params,outputs=[projection,trace],projection=projection)
                closed[stage]=self.put(r);self.receipts.append(closed[stage])
            done=self.put(dict(status='diagnostic_pilot_complete',plan_sha256=plans[key]['sha256'],stage_receipts=closed))
            runs[key]=dict(completion=done,locations=self.put({}))
        master=self.put(dict(maximum_projection_fits=24,maximum_hdbscan_extractions=0,analysis_source=source,host_plans=plans))
        return self.put(dict(operation='closed-reducer-trace-comparison',master=master,runs=runs))
    def test_all_closed_runs_compare_sixteen_pairs_without_fits(self):
        request=self.fixture();result=subject.compare(request['path'],self.root/'out')
        self.assertEqual(len(result['pairs']),16);self.assertEqual(result['new_projection_fits'],0)
        self.assertTrue(all(r['coordinates_within_frozen_tolerance']for r in result['pairs']))
        self.assertTrue(all(r['first_differing_checkpoint']is None for r in result['pairs']))
        self.assertFalse(result['cluster_stability_validated'])
    def test_native_pair_gate_requires_both_complete_native_runs(self):
        request=self.fixture();manifest=json.loads(Path(request['path']).read_text())
        master=json.loads(Path(manifest['master']['path']).read_text())
        names=['workstation-native','hpc-native'];native=json.loads(Path(master['host_plans']['workstation-native']['path']).read_text())
        context=dict(inputs=native['inputs'],native=native['native'],portability_helper=native['portability_helper'],
            native_gate_plan_shas={k:master['host_plans'][k]['sha256']for k in names})
        gate=dict(operation='closed-native-pair-gate',runs={k:dict(manifest['runs'][k],plan=master['host_plans'][k])for k in names})
        diag.verify_native_pair(context,gate)
        gate['runs'].pop('hpc-native')
        with self.assertRaises(ValueError):diag.verify_native_pair(context,gate)

    def test_changed_closed_receipt_is_rejected(self):
        request=self.fixture();p=Path(self.receipts[0]['path']);value=json.loads(p.read_text());value['diagnostic_profile']='wrong';p.write_text(json.dumps(value))
        with self.assertRaises(ValueError):subject.compare(request['path'],self.root/'out')
    def test_coordinate_header_shape_checked_before_array_load(self):
        p=self.root/'bad.npz'
        with p.open('wb')as stream:self.np.savez(stream,embeddings=self.np.ones((3,2)))
        with self.assertRaises(ValueError):subject.projection(pilot.record(p),{})

if __name__=='__main__':unittest.main()
