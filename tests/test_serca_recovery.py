"""Opaque originals, static PCA values and subprocess mocks; no real fits."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import serca_recovery as recovery


def put(path,value):
    path.write_text(json.dumps(value));return recovery.record(path)


class WorkspaceTests(unittest.TestCase):
    def fixture(self,root):
        plan=dict(run_root=str(root),ssd_scratch=dict(root=str(root/'stage-tmp')))
        p=root/'plan.json';p.write_text('{}')
        return p,plan

    @patch.object(recovery,'stage_processes',side_effect=lambda:{os.getpid()})
    def test_only_completed_stage_workspace_is_retired(self,_processes):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=self.fixture(root)
            durable=root/'saved-result.npz';durable.write_bytes(b'keep')
            recovery.create_ssd(path,plan)
            (root/'stage-tmp/umap/input.npy').write_bytes(b'disposable')
            recovery.retire_stage(path,plan,'umap',{os.getpid()})
            self.assertFalse((root/'stage-tmp/umap').exists())
            self.assertTrue((root/'stage-tmp/tsne').is_dir())
            self.assertEqual(durable.read_bytes(),b'keep')

    @patch.object(recovery,'stage_processes',side_effect=lambda:{os.getpid()})
    def test_foreign_marker_or_links_preserve_workspace_and_external_file(self,_processes):
        for bad in ('marker','symlink','hardlink','path'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path,plan=self.fixture(root)
                external=root/'original.pdb';external.write_bytes(b'preserve')
                recovery.create_ssd(path,plan)
                if bad=='marker':(root/'stage-tmp/.owner.json').write_text('{}')
                if bad=='symlink':(root/'stage-tmp/umap/source').symlink_to(external)
                if bad=='hardlink':os.link(external,root/'stage-tmp/umap/source')
                if bad=='path':plan['ssd_scratch']['root']=str(root/'other')
                with self.assertRaises(ValueError):recovery.retire_stage(path,plan,'umap',{os.getpid()})
                self.assertEqual(external.read_bytes(),b'preserve')
                self.assertTrue((root/'stage-tmp/umap').exists())

    @patch.object(recovery,'stage_processes',side_effect=lambda:{os.getpid()})
    def test_subprocess_owns_ssd_tmpdir_and_failure_never_counts_as_success(self,_processes):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=self.fixture(root)
            context=dict(stage_tmpdirs=dict(umap=str(root/'stage-tmp/umap')))
            child=Mock();child.poll.return_value=1;child.returncode=1
            with patch.object(recovery.subprocess,'Popen',return_value=child) as launch:
                with self.assertRaisesRegex(ValueError,'failed'):
                    recovery.run_child(path,plan,context,'umap',check=lambda:None)
            self.assertEqual(launch.call_args.kwargs['env']['TMPDIR'],context['stage_tmpdirs']['umap'])
            self.assertIn('--stage',launch.call_args.args[0])
            child.terminate.assert_not_called()

    def test_stage_descendants_prevent_workspace_retirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=self.fixture(root);recovery.create_ssd(path,plan)
            with patch.object(recovery,'stage_processes',return_value={os.getpid(),999999}):
                with self.assertRaisesRegex(ValueError,'descendants remain'):
                    recovery.retire_stage(path,plan,'umap',{os.getpid()})
            self.assertTrue((root/'stage-tmp/umap').is_dir())


class ScopeBindingTests(unittest.TestCase):
    def test_rebound_pca_proof_engine_or_settings_are_rejected(self):
        plan=dict(target=dict(feature_proof=dict(path='accepted',sha256='a',bytes=1)),
                  sources=dict(library=dict(commit='accepted-library')))
        pca=dict(feature_proof=plan['target']['feature_proof'],native_library_required='accepted-library',
                 settings=dict(n_components=2))
        recovery.verify_saved_pca(pca,plan)
        for key,bad in (('feature_proof',dict(path='another',sha256='b',bytes=1)),
                        ('native_library_required','different'),('settings',dict(n_components=3))):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'binding differs'):
                recovery.verify_saved_pca(dict(pca,**{key:bad}),plan)

    def test_child_context_cannot_change_approved_input_or_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'plan.json';path.write_text('{}')
            plan=dict(run_root=str(root),target=dict(selected_cache='accepted'),native_density=dict(commit='native'),
                      ssd_scratch=dict(root=str(root/'stage-tmp')))
            context=dict(plan_sha256=recovery.sha256(path),descriptor=plan['target'],native_density=plan['native_density'],
                newprimary_manifest=str(root/'primary/manifest.progress.json'),metrics_db=str(root/'metrics.sqlite'),
                stage_tmpdirs={s:str(root/'stage-tmp'/s) for s in recovery.STAGES})
            recovery.validate_context(path,plan,context)
            for key,bad in (('descriptor',dict(selected_cache='another')),('metrics_db',str(root/'old.sqlite')),
                            ('newprimary_manifest',str(root/'old.json')),('stage_tmpdirs',dict(umap='/dev/shm'))):
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'exact approved'):
                    recovery.validate_context(path,plan,dict(context,**{key:bad}))


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy for static saved-PCA reuse')
class SavedPCATests(unittest.TestCase):
    def test_exact_saved_pca_is_copied_without_fitting_and_originals_stay_unchanged(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();source=old/'models';source.mkdir()
            names=['model-'+str(i) for i in range(5072)];rows=[]
            for name in names:
                p=source/(name+'.pdb');p.write_text('opaque original '+name)
                rows.append(dict(model_id=name,path=str(p),sha256=recovery.sha256(p)))
            ratios=np.asarray([.4,.2]);side=old/'pca_embeddings.npz'
            np.savez_compressed(side,embeddings=np.zeros((5072,2)),filenames=np.asarray(names),
                filepaths=np.asarray([r['path'] for r in rows]),explained_variance_ratio=ratios)
            html=old/'pca-base.html';html.write_text('<html>verified PCA</html>')
            png=old/'pca_model_origin.png';png.write_bytes(b'opaque verified PNG')
            proof=put(old/'feature-proof.json',dict(feature_sha256='a'*64,retained_reference_positions=list(range(57,951))))
            checkpoint=put(old/'pca.json',dict(status='complete',method='pca',base_html=str(html),
                models_sha256='ordered',primary_PCA_fits=1,feature_proof=proof,
                native_library_required='native',settings=dict(n_components=2),
                outputs=[recovery.record(p) for p in (html,side,png)]))
            metrics=old/'metrics.sqlite';metrics.write_bytes(b'closed exact metrics bytes')
            desc=dict(target='SERCA_P04191',
                axes=put(old/'axes.json',dict(names=names,names_sha256='ordered')),
                intake=put(old/'intake.json',dict(target='SERCA_P04191',analysis_dir=str(source),models=rows)),
                terminal_report=put(old/'terminal.json',dict(retained_reference_positions=list(range(57,951)))),
                feature_proof=proof,prior_workflow_manifest=put(old/'prior.json',dict(input_inventory='bound elsewhere')),
                model_type_config=dict(path=str(old/'types.json')),selected_cache={},source_metadata=dict(path=str(old/'metadata.json')))
            run=root/'new';run.mkdir();plan=dict(target=desc,run_root=str(run),
                ssd_scratch=dict(root=str(run/'stage-tmp')),native_density={},sources=dict(library=dict(commit='native')),
                reuse=dict(metrics=recovery.record(metrics),pca_checkpoint=checkpoint,feature_sha256='a'*64,plan={}))
            plan_path=root/'plan.json';plan_path.write_text('{}');events=[]
            context=recovery.prepare_context(plan_path,plan,check=lambda:None,checkpoint=events.append)
            manifest=json.loads(Path(context['newprimary_manifest']).read_text())
            self.assertEqual(manifest['derivation']['new_standalone_PCA_fits'],0)
            self.assertEqual(manifest['clustering']['pca']['explained_variance_ratio'],ratios.tolist())
            self.assertEqual(Path(manifest['clustering']['pca']['html']).read_bytes(),html.read_bytes())
            self.assertEqual(recovery.read(context['method_receipts']['pca'])['primary_PCA_fits'],0)
            self.assertTrue(recovery.read(context['method_receipts']['pca'])['PCA_reused'])
            self.assertEqual((run/'metrics.sqlite').read_bytes(),metrics.read_bytes())
            self.assertEqual(html.read_text(),'<html>verified PCA</html>')
            self.assertEqual(len(context['members']),5072)
            self.assertEqual(events[-1],'serca-inherited-PCA-closed')
            self.assertTrue(all(recovery.sha256(r['path'])==r['sha256'] for r in rows))


class ApprovalTests(unittest.TestCase):
    def test_attempts_come_from_frozen_plan_and_cannot_hide_an_interrupted_fit(self):
        value=dict(previous_attempts_charged=24,maximum_new_projection_fits=20,maximum_cumulative_attempts=44)
        self.assertEqual(recovery.attempt_accounting(dict(fit_accounting=value)),
            dict(previous_attempts_charged=24,maximum_cumulative_attempts=44))
        for key,bad in (('previous_attempts_charged',True),('maximum_new_projection_fits',21),('maximum_cumulative_attempts',43)):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'accounting differs'):
                recovery.attempt_accounting(dict(fit_accounting=dict(value,**{key:bad})))

    def test_no_approval_or_widened_budget_cannot_reach_native_controls(self):
        for bad in ('approval','budget','target'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan=dict(operation='serca-selected-cache-recovery',settings=dict(recovery.SCOPE),
                    target=dict(target='SERCA_P04191',model_count=5072))
                if bad=='budget':plan['settings']['repeated_PCA_fits']=1
                if bad=='target':plan['target']['target']='5NTase_P07024'
                p=root/'plan.json';p.write_text(json.dumps(plan))
                put(root/'approval.json',dict(approved_by='user',approved_plan_sha256='0'*64 if bad=='approval' else recovery.sha256(p)))
                with patch('run_reviewed_plan.verify_controls') as controls:
                    with self.assertRaises(ValueError):recovery.approved(p)
                    controls.assert_not_called()


if __name__=='__main__':unittest.main()
