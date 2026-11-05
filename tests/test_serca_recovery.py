"""Opaque originals, static PCA values and subprocess mocks; no real fits."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import serca_recovery as recovery


def put(path,value):
    path.write_text(json.dumps(value));return recovery.record(path)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(recovery,'mount_type',return_value='ext4'))

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
            root=Path(tmp);path,plan=self.fixture(root);recovery.create_ssd(path,plan)
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
    def test_full_cohort_umap_pilot_has_one_fit_and_no_density_or_later_stages(self):
        plan=dict(execution_stages=['umap'],fit_accounting=dict(previous_attempts_charged=25,maximum_new_projection_fits=1,maximum_cumulative_attempts=26))
        self.assertEqual(recovery.scientific_scope(plan)['maximum_new_projection_fits'],1)
        self.assertEqual(recovery.scientific_scope(plan)['maximum_hdbscan_extractions'],0)
        self.assertEqual(recovery.attempt_accounting(plan)['maximum_cumulative_attempts'],26)
        for stages in ([],['tsne'],['umap','density'],list(recovery.STAGES)+['umap']):
            with self.subTest(stages=stages),self.assertRaises(ValueError):recovery.execution_stages(dict(execution_stages=stages))

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


class TmpfsWorkspaceTests(unittest.TestCase):
    def fixture(self,root):
        output=root/'run';output.mkdir();outer=root/'ram';outer.mkdir()
        plan=dict(run_root=str(output),environment=dict(TMPDIR=str(outer)),
            ssd_scratch=dict(backing='tmpfs',root=str(outer/'native-stages'),minimum_free_bytes_before_stage=200))
        path=root/'plan.json';path.write_text('{}')
        info=outer.stat()
        put(outer/'.owner.json',dict(plan_sha256=recovery.sha256(path),uid=os.getuid(),device=info.st_dev,inode=info.st_ino))
        self.enterContext(patch('analysis_scratch.scratch_root',return_value=outer))
        self.enterContext(patch.object(recovery,'mount_type',return_value='tmpfs'))
        self.enterContext(patch.object(recovery.shutil,'disk_usage',return_value=SimpleNamespace(free=1000)))
        return path,plan,outer

    def test_missing_or_tampered_outer_marker_prevents_inner_allocation(self):
        for bad in ('missing','tampered'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                path,plan,outer=self.fixture(Path(tmp));marker=outer/'.owner.json'
                if bad=='missing':marker.unlink()
                else:marker.write_text('{}')
                with self.assertRaisesRegex(ValueError,'Outer RAM ownership'):
                    recovery.create_ssd(path,plan)
                self.assertFalse((outer/'native-stages').exists())

    def test_tmpfs_lifecycle_removes_only_inner_owned_tree_then_outer_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,outer=self.fixture(Path(tmp));recovery.create_ssd(path,plan)
            workspace=outer/'native-stages';saved=Path(plan['run_root'])/'saved.npz';saved.write_bytes(b'preserve')
            (workspace/'umap/input.npy').write_bytes(b'static disposable fixture')
            with patch.object(recovery,'stage_processes',return_value={os.getpid()}):
                recovery.retire_stage(path,plan,'umap',{os.getpid()})
            self.assertFalse((workspace/'umap').exists());self.assertTrue((workspace/'tsne').is_dir())
            with patch('analysis_scratch.assert_quiescent'),patch.object(recovery,'stage_processes',return_value={os.getpid()}),patch('analysis_scratch.cleanup') as outer_cleanup:
                outer_cleanup.side_effect=lambda *args:self.assertFalse(workspace.exists())
                recovery.cleanup(path,plan);outer_cleanup.assert_called_once_with(path,plan)
            self.assertEqual(saved.read_bytes(),b'preserve')
            proof=json.loads((Path(plan['run_root'])/'native-workspace-cleanup.json').read_text())
            self.assertEqual(proof['backing'],'tmpfs');self.assertEqual(proof['status'],'removed_owned_disposable_tmpfs_workspace')
            self.assertFalse((Path(plan['run_root'])/'ssd-cleanup.json').exists())

    def test_tmpfs_root_cannot_escape_exact_outer_root_or_change_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,outer=self.fixture(Path(tmp));root=Path(plan['run_root'])
            plan.update(target={'selected_cache':'accepted'},native_density={'commit':'native'})
            context=dict(plan_sha256=recovery.sha256(path),descriptor=plan['target'],native_density=plan['native_density'],
                newprimary_manifest=str(root/'primary/manifest.progress.json'),metrics_db=str(root/'metrics.sqlite'),
                stage_tmpdirs={s:str(outer/'native-stages'/s) for s in recovery.STAGES})
            recovery.validate_context(path,plan,context)
            with self.assertRaisesRegex(ValueError,'exact approved'):
                recovery.validate_context(path,plan,dict(context,stage_tmpdirs={s:str(root/'stage-tmp'/s) for s in recovery.STAGES}))
            plan['ssd_scratch']['root']=str(root/'stage-tmp')
            with self.assertRaisesRegex(ValueError,'exact approved RAM'):recovery.ssd_root(plan)

    def test_capacity_and_filesystem_gates_precede_creation_and_child_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,outer=self.fixture(Path(tmp));workspace=outer/'native-stages'
            with patch.object(recovery.shutil,'disk_usage',return_value=SimpleNamespace(free=199)):
                with self.assertRaisesRegex(ValueError,'Insufficient free tmpfs'):recovery.create_ssd(path,plan)
            self.assertFalse(workspace.exists())
            with patch.object(recovery,'mount_type',return_value='ext4'):
                with self.assertRaisesRegex(ValueError,'approved tmpfs'):recovery.create_ssd(path,plan)
            recovery.create_ssd(path,plan)
            context=dict(stage_tmpdirs={s:str(workspace/s) for s in recovery.STAGES})
            with patch.object(recovery.shutil,'disk_usage',return_value=SimpleNamespace(free=199)),patch.object(recovery.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'Insufficient free tmpfs'):
                    recovery.run_child(path,plan,context,'umap',check=lambda:None)
                launch.assert_not_called()
            self.assertTrue((workspace/'umap').is_dir())

    def test_tmpfs_cleanup_refuses_live_descendants_and_foreign_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,outer=self.fixture(Path(tmp));recovery.create_ssd(path,plan);workspace=outer/'native-stages'
            with patch('analysis_scratch.assert_quiescent'),patch.object(recovery,'stage_processes',return_value={os.getpid(),99999}),patch('analysis_scratch.cleanup') as outer_cleanup:
                with self.assertRaisesRegex(ValueError,'including descendants'):recovery.cleanup(path,plan)
                outer_cleanup.assert_not_called()
            (workspace/'.owner.json').write_text('{}')
            with patch('analysis_scratch.assert_quiescent'),patch.object(recovery,'stage_processes',return_value={os.getpid()}),patch('analysis_scratch.cleanup') as outer_cleanup:
                with self.assertRaisesRegex(ValueError,'ownership identity'):recovery.cleanup(path,plan)
                outer_cleanup.assert_not_called()
            self.assertTrue(workspace.is_dir())


class ContinuationScopeTests(unittest.TestCase):
    def plan(self):
        return dict(reuse={key:{} for key in ('umap_checkpoint','umap_plan','umap_completion','umap_audit')},
            execution_stages=['tsne','labels','density'],
            fit_accounting=dict(previous_attempts_charged=26,maximum_new_projection_fits=19,maximum_cumulative_attempts=45))

    def test_closed_umap_inheritance_allows_only_remaining_stages_and_nineteen_fits(self):
        plan=self.plan();self.assertEqual(recovery.execution_stages(plan),['tsne','labels','density'])
        self.assertEqual(recovery.scientific_scope(plan)['maximum_new_projection_fits'],19)
        self.assertEqual(recovery.scientific_scope(plan)['maximum_hdbscan_extractions'],30)
        self.assertEqual(recovery.attempt_accounting(plan)['maximum_cumulative_attempts'],45)
        for stages in (['umap'],list(recovery.STAGES),['labels'],['tsne','density'],[]):
            with self.subTest(stages=stages),self.assertRaisesRegex(ValueError,'remaining stages'):
                recovery.execution_stages(dict(plan,execution_stages=stages))
        for stages,fits in ((['tsne'],2),(['tsne','labels'],4)):
            scope=recovery.scientific_scope(dict(plan,execution_stages=stages))
            self.assertEqual(scope['maximum_new_projection_fits'],fits);self.assertEqual(scope['maximum_hdbscan_extractions'],0)

    def test_incomplete_reuse_evidence_cannot_silently_refit(self):
        plan=self.plan();del plan['reuse']['umap_audit']
        with self.assertRaisesRegex(ValueError,'independent audit'):recovery.execution_stages(plan)

    def test_child_cli_rejects_reused_umap_before_context_or_numerical_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.plan();plan['run_root']=tmp
            with patch.object(recovery,'approved',return_value=plan),patch.object(sys,'argv',['serca_recovery.py',tmp+'/plan.json','--stage','umap']):
                with self.assertRaisesRegex(ValueError,'outside the approved prefix'):recovery.main()
            self.assertFalse((Path(tmp)/'recovery-context.json').exists())


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy for static closed-UMAP fixtures')
class SavedUMAPTests(unittest.TestCase):
    def fixture(self,root,bad=None):
        import numpy as np
        old=root/'pilot';primary=old/'primary';(primary/'clustering/umap').mkdir(parents=True);(primary/'checkpoints').mkdir()
        fresh=root/'new/primary';(fresh/'clustering').mkdir(parents=True);(fresh/'checkpoints').mkdir()
        names=['model-'+str(i) for i in range(5072)];members=[dict(model=n,source_path='/unopened-originals/'+n+'.pdb') for n in names]
        order=hashlib.sha256('\n'.join(names).encode()).hexdigest();core=list(range(57,951))
        feature=dict(feature_sha256='a'*64,retained_reference_positions=core,pair_mask_sha256='b'*64)
        report=dict(retained_reference_positions=core,pair_mask_sha256='b'*64)
        target=dict(feature_proof=put(old/'feature-proof.json',feature),selected_cache=dict(path='/unopened-selected-cache.npz',sha256='c'*64,bytes=4054316678))
        library=dict(commit='f'*40,checkout='/unchanged-native-library')
        plan=dict(target=target,sources=dict(library=library),runtime_python='/existing/python',runtime_fingerprint='/bound/runtime.json',
            fit_accounting=dict(previous_attempts_charged=26,maximum_new_projection_fits=19,maximum_cumulative_attempts=45))
        pilot_plan=put(old/'execution-plan.json',dict(target=target,execution_stages=['umap'],sources=dict(library=library),
            runtime_python=plan['runtime_python'],runtime_fingerprint=plan['runtime_fingerprint'],run_root=str(old)))
        params=dict(n_neighbors=30 if bad=='parameters' else 15,min_dist=.1,n_components=2)
        base=primary/'checkpoints/umap-base.html';base.write_text('Opaque accepted UMAP HTML')
        image=primary/'clustering/umap/model-origin.png';image.write_bytes(b'Opaque accepted image')
        xy=np.zeros((5072,2),np.float32)
        if bad=='finite':xy[0,0]=np.nan
        saved_names=names.copy()
        if bad=='order':saved_names[0]='another-model'
        side=primary/'clustering/umap_embeddings.npz'
        np.savez_compressed(side,embeddings=xy,filenames=np.asarray(saved_names),filepaths=np.asarray([r['source_path'] for r in members]))
        outputs=[recovery.record(p) for p in (base,side,image)]
        checkpoint=put(primary/'checkpoints/umap.json',dict(method='umap',status='complete',parameters=params,base_html=str(base),
            native_output_html=str(primary/'clustering/umap.html'),outputs=outputs,
            source_feature_proof=target['feature_proof'],selected_cache=target['selected_cache'],
            feature_sha256='z'*64 if bad=='features' else feature['feature_sha256'],models_sha256=order,
            UMAP_fits=1,tsne_fits=0,internal_tsne_PCA_fits=0,primary_PCA_fits=0,cache_reused_without_rewrite=True,native_library=library))
        metrics=old/'metrics.sqlite';metrics.write_bytes(b'Opaque closed accepted metrics')
        view=dict(html=str(primary/'clustering/umap.html'),embeddings=str(side),params=params,
            images=[dict(image=str(image),kind='model_type',title='Model origin')])
        manifest=put(primary/'manifest.progress.json',dict(identifier='SERCA_P04191',structure_count=5072,
            terminal_detection=report,defaults=dict(random_seed=42),metrics_db=str(metrics),
            method_receipts=dict(umap=checkpoint),clustering=dict(umap=view)))
        completion=put(old/'serca-recovery-completion.json',dict(status='serca_selected_stages_ready',full_clustering_complete=False,
            completed_recovery_stages=['umap'],new_projection_fits_max=1,HDBSCAN_extractions_max=0,
            maximum_cumulative_attempts=26,plan_sha256=pilot_plan['sha256'],primary_manifest=manifest))
        guard=put(old/'guard-exit.json',dict(status='complete',returncode=0))
        audit=put(old/'independent-audit.json',dict(status='accepted_partial_UMAP_only_RAM_pilot',plan=pilot_plan,
            checkpoint=checkpoint,partial_completion=completion,models=5072,construct_residues=993,retained_residues=894,
            canonical_core=[58,951],guard_completion=guard,raw_feature_sha256=feature['feature_sha256'],
            model_order_sha256=order,pair_mask_sha256=feature['pair_mask_sha256'],umap=dict(artifact=recovery.record(side))))
        plan['reuse']=dict(umap_plan=pilot_plan,umap_checkpoint=checkpoint,umap_completion=completion,umap_audit=audit,
                           feature_sha256=feature['feature_sha256'],metrics=recovery.record(metrics))
        originals=[*outputs,checkpoint,manifest,completion,pilot_plan,audit,recovery.record(metrics)]
        return plan,fresh,names,members,feature,dict(names_sha256=order),report,originals

    def call(self,case):return recovery.reuse_saved_umap(*case[:7])

    def test_closed_UMAP_and_figures_are_copied_exactly_without_original_or_cache_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=self.fixture(Path(tmp));view,receipt=self.call(case)
            value=recovery.read(receipt)
            self.assertTrue(value['UMAP_reused']);self.assertEqual(value['UMAP_fits'],0)
            self.assertEqual(value['prior_UMAP_fits_charged'],1)
            self.assertEqual(value['source_checkpoint'],case[0]['reuse']['umap_checkpoint'])
            self.assertEqual(view['images'][0]['kind'],'model_type');self.assertTrue(view['images'][0]['reused'])
            self.assertEqual(Path(view['html']).read_text(),'Opaque accepted UMAP HTML')
            self.assertTrue(all(recovery.record(r['path'])==r for r in case[-1]))
            parent=recovery.read(case[0]['reuse']['umap_checkpoint'])
            self.assertEqual([(r['bytes'],r['sha256']) for r in value['outputs']],[(r['bytes'],r['sha256']) for r in parent['outputs']])
            self.assertFalse(Path('/unopened-originals').exists())
            self.assertFalse(Path('/unopened-selected-cache.npz').exists())

    def test_mismatched_protocol_features_order_or_nonfinite_saved_values_are_rejected(self):
        for bad,reason in (('parameters','protocol'),('features','features'),('order','population'),('finite','population')):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                case=self.fixture(Path(tmp),bad=bad)
                with self.assertRaisesRegex(ValueError,reason):self.call(case)
                self.assertFalse((case[1]/'checkpoints/umap.json').exists())

    def test_tampered_bound_image_stops_inheritance_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=self.fixture(Path(tmp));saved=recovery.read(case[0]['reuse']['umap_checkpoint'])
            image=next(r for r in saved['outputs'] if r['path'].endswith('.png'))
            Path(image['path']).write_bytes(b'Changed input; do not repair')
            with self.assertRaisesRegex(ValueError,'Bound input changed'):self.call(case)
            self.assertEqual(Path(image['path']).read_bytes(),b'Changed input; do not repair')
            self.assertFalse((case[1]/'checkpoints/umap.json').exists())


if __name__=='__main__':unittest.main()
