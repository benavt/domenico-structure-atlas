"""Saved1000-point fixtures; no estimators or large feature arrays."""
from copy import deepcopy
import hashlib,importlib.util,json
from pathlib import Path
import sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import umap1000_compare as subject
import hpc_pilot as pilot
import test_umap_curve_compare as fixtures


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy saved fixtures')
class ComparisonTests(unittest.TestCase):
    def fixture(self,delta=0.):
        maker=fixtures.CurveComparisonTests();f,req=maker.fixture();self.addCleanup(maker.doCleanups)
        r=json.loads(Path(req['path']).read_text());m=json.loads(Path(r['master']['path']).read_text());plans={};runs={}
        ids=['m'+str(i)for i in range(1000)];xy=f.np.column_stack((f.np.arange(1000),f.np.sin(f.np.arange(1000)))).astype(float)
        for host in ['workstation','hpc']:
            plan=json.loads(Path(m['host_plans'][host]['path']).read_text());plan['phase']='curve1000'
            plan['inputs'].update(model_count=1000,model_order_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
            pe=f.put(plan);plans[host]=pe;done=json.loads(Path(r['runs'][host]['completion']['path']).read_text())
            receipt=json.loads(Path(done['stage_receipts']['umap1']['path']).read_text());receipt.update(inputs=plan['inputs'],models=ids,model_flags=[{}]*1000)
            p=f.root/(host+'1000.npz');values=xy.copy()
            if host=='hpc':values[0,0]+=delta
            with p.open('wb')as stream:f.np.savez(stream,embeddings=values)
            oldprojection=receipt['projection'];receipt['projection']=pilot.record(p)
            receipt['outputs']=[receipt['projection']if e==oldprojection else e for e in receipt['outputs']]
            done.update(plan_sha256=pe['sha256'],stage_receipts=dict(umap1=f.put(receipt)))
            runs[host]=dict(completion=f.put(done),locations=f.put({}))
        master=f.put(dict(operation='paired-umap-curve1000',maximum_projection_fits=2,maximum_hdbscan_extractions=0,
            host_plans=plans,curve_recipe=m['curve_recipe'],correspondence64={'path':'/bound/proof'},analysis_source=pilot.record(subject.__file__)))
        return f,f.put(dict(operation='closed-umap1000-comparison',master=master,runs=runs))
    def test_closed1000_points_pass_exact_gate_without_new_fits(self):
        f,req=self.fixture()
        with patch('saved_curve_evidence.verify',return_value={'status':'verified_saved_fixture'}):result=subject.compare(req['path'],f.root/'out')
        self.assertTrue(result['exact_gate_passed']);self.assertEqual(result['projection_fits'],2)
        self.assertEqual(result['new_comparison_fits'],0);self.assertFalse(result['larger_cohort_authorized'])
    def test_near_match_does_not_pass_exact_gate(self):
        f,req=self.fixture(delta=1e-12)
        with patch('saved_curve_evidence.verify',return_value={'status':'verified_saved_fixture'}):result=subject.compare(req['path'],f.root/'out')
        self.assertFalse(result['exact_gate_passed']);self.assertTrue(result['geometry']['raw_allclose'])
    def test_closed_reference1000_gate_checks_guard_exit_and_coordinate_shape(self):
        import saved_curve_evidence as evidence
        f,req=self.fixture();request=json.loads(Path(req['path']).read_text());master=json.loads(Path(request['master']['path']).read_text())
        source=json.loads(Path(master['host_plans']['workstation']['path']).read_text());spec=request['runs']['workstation']
        exit=f.put(dict(status='complete',returncode=0))
        gate=f.put(dict(operation='closed-workstation1000-predecessor',plan=master['host_plans']['workstation'],
            completion=spec['completion'],locations=spec['locations'],pressure_exit=exit))
        plan=dict(workstation1000_gate=gate['path'],workstation1000_plan_sha256=master['host_plans']['workstation']['sha256'],inputs=source['inputs'],curve_control=source['curve_control'])
        self.assertEqual(evidence.verify_workstation1000_predecessor(plan)['projection_fits'],0)
        bad=f.put(dict(status='failed',returncode=1));body=json.loads(Path(gate['path']).read_text());body['pressure_exit']=bad;Path(gate['path']).write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError,'guard did not complete'):evidence.verify_workstation1000_predecessor(plan)

    def test64_row_sidecar_cannot_masquerade_as1000(self):
        maker=fixtures.CurveComparisonTests();f,req=maker.fixture();self.addCleanup(maker.doCleanups)
        m=json.loads(Path(json.loads(Path(req['path']).read_text())['master']['path']).read_text());done=json.loads(Path(m['reference_workstation']['completion']['path']).read_text())
        receipt=json.loads(Path(done['stage_receipts']['umap1']['path']).read_text())
        with self.assertRaises(ValueError):subject.projection(receipt['projection'],{})


if __name__=='__main__':unittest.main()
