"""Saved fixtures, hashes and metadata only; no estimators fitted."""
import importlib.util,json
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import saved_curve_evidence as subject
import test_umap_curve_compare as fixtures


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy fixture creation')
class EvidenceTests(unittest.TestCase):
    def fixture(self,delta=0.):
        maker=fixtures.CurveComparisonTests();f,req=maker.fixture(coordinate_delta=delta);self.addCleanup(maker.doCleanups)
        request=json.loads(Path(req['path']).read_text());master=json.loads(Path(request['master']['path']).read_text())
        runs={};self.traces=[]
        names=['umap1_graph_data','umap1_graph_indices','umap1_graph_indptr','umap1_initial','umap1_rng',
            'umap1_optimizer_scalars','umap1_epoch0_scalars','umap1_epoch1_scalars','umap1_epoch1_after_head_embedding']
        names+=['extra'+str(i)for i in range(35-len(names))]
        for host in ['workstation','hpc']:
            spec=master['reference_workstation']if host=='workstation'else dict(request['runs']['hpc'],plan=master['host_plans']['hpc'])
            done=json.loads(Path(spec['completion']['path']).read_text());r=json.loads(Path(done['stage_receipts']['umap1']['path']).read_text())
            trace=json.loads(Path(r['trace']['path']).read_text());trace['events']=[dict(name=n,values=dict(a=1.5,b=.9)if n=='umap1_optimizer_scalars'else{})for n in names]
            if host=='hpc':trace['events'].append(dict(name='umap_curve_effective',values=dict(a=1.5,b=.9,min_dist=.1,spread=1.)))
            entry=f.put(trace);self.traces.append(entry);r['trace']=entry;r['outputs']=[r['projection'],entry]
            done['stage_receipts']['umap1']=f.put(r);runs[host]=dict(spec,completion=f.put(done))
        return f,f.put(dict(operation='closed-saved-umap64-correspondence',runs=runs,recipe=master['curve_recipe']))
    def test_closed_native_workstation_and_controlled_hpc_can_supply_reuse_gate(self):
        f,proof=self.fixture();result=subject.verify(proof)
        self.assertEqual(result['projection_fits'],0);self.assertEqual(result['common_checkpoints'],35)
        self.assertFalse(result['new_workstation_controlled_fit_completed'])
    def test_tiny_coordinate_difference_is_not_accepted_as_exact(self):
        f,proof=self.fixture(delta=1e-12)
        with self.assertRaisesRegex(ValueError,'gate fails'):subject.verify(proof)
    def test_changed_evidence_cannot_be_reused(self):
        f,proof=self.fixture();Path(self.traces[0]['path']).write_text('{}')
        with self.assertRaises(ValueError):subject.verify(proof)


if __name__=='__main__':unittest.main()
