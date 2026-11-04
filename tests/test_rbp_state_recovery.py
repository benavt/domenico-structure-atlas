"""Saved-stage recovery contract tests; no scientific/native operations."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rbp_state_recovery import verify_saved_stages
from rbp_parity import record


class RecoveryTests(unittest.TestCase):
    def fixture(self,root):
        views={}
        for view in ['primary','density']:
            views[view]={}
            for artifact in ['bonsai','medoid_tm']:
                path=root/(view+'-'+artifact+'.json');path.write_text(json.dumps([view,artifact]))
                views[view][artifact]=record(path)
        analysis=dict(path='/native/analysis.json',sha256='a'*64,bytes=1)
        saved=dict(density_analysis=analysis,density_artifacts=dict(views['density'],analysis=analysis,
            ensemble_report=dict(path='/native/report.json',sha256='b'*64,bytes=1)),primary_artifacts=views['primary'])
        plan=dict(parent_scientific_plan=dict(sha256='c'*64),manifest=dict(sha256='d'*64))
        density=dict(status='complete',models=['m1','m2'],source_workflow_manifest_sha256='d'*64,
            UMAP_fits=15,HDBSCAN_extractions=30,structural_clusters=2,unassigned_models=0)
        audit=dict(status='partial_completed_stages_independently_verified',plan=dict(sha256='c'*64),completed=dict(
            density=dict(publication_analysis=analysis,models=2,clusters=2,unassigned_models=0),
            representatives=dict(report=saved['density_artifacts']['ensemble_report'],inventory_proof_status='passed',full271_CA_mask=True),
            bonsai={key:dict(file=value['bonsai']) for key,value in views.items()},
            grids={key:dict(file=value['medoid_tm']) for key,value in views.items()}))
        return plan,saved,density,audit,['m1','m2']

    def test_exact_saved_views_are_accepted_without_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));before=deepcopy(args)
            self.assertTrue(verify_saved_stages(*args));self.assertEqual(args,before)

    def test_changed_prior_plan_or_model_registry_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));args[3]['plan']['sha256']='e'*64
            with self.assertRaisesRegex(ValueError,'acceptance'):verify_saved_stages(*args)
            args=self.fixture(Path(tmp));args[2]['models']=['m2','m1']
            with self.assertRaisesRegex(ValueError,'density analysis'):verify_saved_stages(*args)

    def test_wrong_view_tree_or_missing_atom_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));args[1]['primary_artifacts']['bonsai']=args[1]['density_artifacts']['bonsai']
            with self.assertRaisesRegex(ValueError,'Bonsai/TM'):verify_saved_stages(*args)
            args=self.fixture(Path(tmp));args[3]['completed']['representatives']['inventory_proof_status']='unverified'
            with self.assertRaisesRegex(ValueError,'native atom'):verify_saved_stages(*args)

    def test_mutated_saved_artifact_stops_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));Path(args[1]['density_artifacts']['medoid_tm']['path']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'Bonsai/TM'):verify_saved_stages(*args)

if __name__=='__main__':unittest.main()
