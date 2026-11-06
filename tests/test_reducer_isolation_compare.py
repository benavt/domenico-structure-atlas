"""Closed saved fixtures only; no UMAP/t-SNE/PCA execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reducer_isolation_compare as subject
import hpc_pilot as pilot
import test_reducer_trace_compare as fixtures


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy for tiny saved fixtures')
class IsolationComparisonTests(unittest.TestCase):
    def fixture(self):
        f=fixtures.SavedComparisonTests();f.setUp();self.addCleanup(f.tmp.cleanup)
        request=f.fixture();old=json.loads(Path(request['path']).read_text());master=json.loads(Path(old['master']['path']).read_text())
        plans={};runs={};references={};self.traces=[]
        shared=f.root/'init.npy';f.np.save(shared,f.np.arange(128,dtype=f.np.float32));init=pilot.record(shared)
        for host in ['workstation','hpc']:
            key=host+'-native';proof=old['runs'][key]
            native=json.loads(Path(master['host_plans'][key]['path']).read_text());native['execution_host']=host
            native_entry=f.put(native)
            closed=json.loads(Path(proof['completion']['path']).read_text());closed['plan_sha256']=native_entry['sha256']
            references[host]=dict(plan=native_entry,completion=f.put(closed),locations=f.put({}))
            plan=deepcopy(native);plan.update(phase='isolate64',isolation=dict(tsne_initialization=init))
            plan_entry=f.put(plan);plans[host]=plan_entry;receipts={}
            for stage in ['umap1','tsne1']:
                receipt=json.loads(Path(closed['stage_receipts'][stage]['path']).read_text())
                trace=json.loads(Path(receipt['trace']['path']).read_text());trace['optimizer_epoch_trace']=True
                trace.update(epoch_calls=500 if stage=='umap1'else 0,
                    tsne_initialization=init if stage=='tsne1'else None,initialization_replays=int(stage=='tsne1'))
                events=([dict(name=n,values={})for n in ['umap1_optimizer_scalars','umap1_epoch0_scalars','umap1_epoch1_scalars',
                    'umap1_epoch0_after_head_embedding','umap1_epoch1_after_head_embedding']]if stage=='umap1'else[
                    dict(name='tsne1_native_initial',raw_sha256=host),
                    dict(name='tsne1_initial',raw_sha256='same',artifact=init),
                    dict(name='tsne1_initialization_source',values=init)])
                trace['events']=events;trace_entry=f.put(trace);self.traces.append(trace_entry)
                receipt['trace']=trace_entry;receipt['isolation']=plan['isolation'];receipt['outputs']=[receipt['projection'],trace_entry]+([init]if stage=='tsne1'else[])
                receipts[stage]=f.put(receipt)
            runs[host]=dict(locations=f.put({}),completion=f.put(dict(status='diagnostic_pilot_complete',plan_sha256=plan_entry['sha256'],stage_receipts=receipts)))
        newmaster=f.put(dict(operation='paired-reducer-isolation',maximum_projection_fits=6,maximum_hdbscan_extractions=0,
            analysis_source=pilot.record(subject.__file__),host_plans=plans,reference_runs=references))
        return f, f.put(dict(operation='closed-reducer-isolation-comparison',master=newmaster,runs=runs))

    def test_comparison_proves_shared_init_and_separates_native_initial_difference(self):
        f,req=self.fixture();result=subject.compare(req['path'],f.root/'out')
        self.assertEqual(len(result['pairs']),6);self.assertEqual(result['new_comparison_fits'],0)
        self.assertEqual(sum(sum(r[k]for k in ['PCA','UMAP','TSNE'])for r in result['actual_fit_accounting'].values()),6)
        tsne=next(p for p in result['pairs']if p['name']=='cross-host-tsne1')
        self.assertEqual(tsne['first_differing_recorded_checkpoint'],'tsne1_native_initial')
        self.assertIsNone(tsne['first_differing_effective_checkpoint']);self.assertTrue(tsne['coordinate_gate_passed'])
        self.assertFalse(result['cluster_stability_validated'])

    def test_changed_trace_stops_before_comparison(self):
        f,req=self.fixture();Path(self.traces[0]['path']).write_text('{}')
        with self.assertRaises(ValueError):subject.compare(req['path'],f.root/'out')

    def test_source_descriptors_cannot_be_silently_relocated_to_other_content(self):
        from reducer_trace_compare import read
        f,req=self.fixture();entry=pilot.record(f.root/'init.npy')
        with self.assertRaises(ValueError):read(entry,{entry['path']:dict(entry,sha256='0'*64)})


if __name__=='__main__':unittest.main()
