"""Static tiny cache fixtures; every fit, coordinate reader and plot is mocked."""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import serca_stage as stage_adapter
import test_target_projections as primary_fixture
from test_terminal_rescreen import bound,write


class Prepared:
    def __init__(self,**kwargs):self.__dict__.update(kwargs)
    def clustering_data_for(self,directory,paths,requested_positions):
        assert directory==self.directory and list(paths)==self.paths
        assert list(requested_positions)==self.requested_reference_positions
        return dict(flattened=self.flattened,filenames=[p.stem for p in self.paths],filepaths=[str(p) for p in self.paths])


class StageTests(unittest.TestCase):
    def fixture(self,root):
        import numpy as np
        helper=primary_fixture.PrimaryProjectionTests()
        desc,terminal,metrics,members,types,clusters,scratch,selected=helper.fixture(root)
        funcs,state=helper.mock_adapters(root,selected);state['members']=members
        desc.update(terminal_report=terminal,model_type_config=types,cluster_config=clusters)
        cache=root/'selected-cache.npz';cache.write_bytes(b'Mocked selected cache: never read by these tests')
        desc['selected_cache']=dict(**bound(cache),bytes=cache.stat().st_size,member='flattened.npy',shape=list(selected.shape))
        desc['feature_proof']=write(root/'accepted-feature-proof.json',dict(feature_sha256=hashlib.sha256(selected.tobytes()).hexdigest()))
        with np.load(desc['cache']['path'],allow_pickle=False) as saved:signature=json.loads(str(saved['source_signature']))
        metadata=json.loads(__import__('gzip').decompress(Path(desc['source_metadata']['path']).read_bytes()))
        report=json.loads(Path(terminal['path']).read_text());names=[r['model'] for r in members]
        matrix=selected.copy(order='C');matrix.setflags(write=False)
        proof=dict(shape=list(matrix.shape),dtype='float16',readonly=True,
            source_signature=signature,metadata=metadata,retained_reference_positions=report['retained_reference_positions'],
            requested_reference_positions=report['reference_positions'],pair_mask_sha256=report['pair_mask_sha256'],
            model_order_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
            feature_sha256=hashlib.sha256(selected.tobytes()).hexdigest())
        primary=root/'primary';(primary/'clustering/pca').mkdir(parents=True);(primary/'checkpoints').mkdir()
        html=primary/'clustering/pca.html';html.write_text('Unlabelled immutable PCA copy')
        base=primary/'checkpoints/pca-base.html';base.write_text(html.read_text())
        xyfile=primary/'clustering/pca_embeddings.npz'
        np.savez_compressed(xyfile,embeddings=np.arange(12,dtype=float).reshape(6,2),filenames=np.asarray(names),
                            filepaths=np.asarray([r['source_path'] for r in members]),explained_variance_ratio=[.6,.2])
        image=primary/'clustering/pca/pca.png';image.write_bytes(b'Mocked saved PCA figure')
        pca=primary/'checkpoints/pca.json';stage_adapter.save(pca,dict(method='pca',status='complete',PCA_reused=True,
            primary_PCA_fits=0,outputs=[stage_adapter.record(p) for p in (base,xyfile,image)]))
        manifest=dict(identifier=desc['target'],output_dir=str(primary),source_dir=str(root),metrics_db=str(metrics),
            clustering=dict(pca=dict(html=str(html),embeddings=str(xyfile),images=[dict(image=str(image))],
                params={'n_components':2},explained_variance_ratio=[.6,.2])),analysis_status='running',status='running',
            defaults=dict(random_seed=42,tsne_implementation='python'),terminal_detection=report,
            derivation=dict(accepted_terminal_report=terminal),ensemble_confidence_preflight={'model_count':6,'analysis_model_count':6},
            model_type_config=str(types['path']),model_type_config_source='explicit',plot_model_type_config=types['path'],
            method_receipts=dict(pca=stage_adapter.record(pca)))
        manifest_path=primary/'manifest.progress.json';stage_adapter.save(manifest_path,manifest)
        native_density={'source':'frozen native primitive'}
        context=dict(descriptor=desc,newprimary_manifest=str(manifest_path),metrics_db=str(metrics),members=members,
            native_density=native_density,method_receipts=manifest['method_receipts'],
            stage_tmpdirs={s:str(root/'ssd'/s) for s in stage_adapter.STAGES})
        for p in context['stage_tmpdirs'].values():Path(p).mkdir(parents=True)
        plan=dict(run_root=str(root),resources={'maximum_cpus':4},native_density=native_density,
                  sources={'library':{'commit':'fe94ba765fd7a9f9894a5e5ba11a57f582a39365'}})
        state.update(mapped=0,unmapped=0,cache_writer=Mock(side_effect=AssertionError('Must not rewrite selected cache')))
        @contextmanager
        def mapped(cache,**kwargs):
            assert cache==desc['selected_cache'] and kwargs['feature_proof']==desc['feature_proof']
            state['mapped']+=1
            try:yield matrix,proof
            finally:state['unmapped']+=1
        class Worker:
            def __init__(self,**kwargs):self.options=kwargs
            def run(self):
                options=self.options;method=options['clustering_method'];state['methods'].append(method)
                state['worker_options'].append(options)
                reuse=options['prepared_distograms']
                data=reuse.clustering_data_for(root,[Path(r[0]) for r in signature['files']],report['reference_positions'])
                np.testing.assert_array_equal(data['flattened'],selected)
                self_cache=reuse.persist_clustering_cache(desc['target'],state['cache_writer'],
                    residue_selection_key='shared_msa',basename_keyword='',metric_filter_column=None,metric_filter_threshold=None)
                assert self_cache==Path(desc['selected_cache']['path'])
                if state.get('fail')==method:options['error_callback']('mock failure');return
                Path(options['output_file']).write_text('Mocked '+method+' HTML')
                options['embedding_callback'](np.arange(12,dtype=float).reshape(6,2),names,[Path(r[0]) for r in signature['files']])
        funcs['prepared']=Prepared;funcs['worker']=Worker
        def knn(manifest,projections,**kwargs):
            state['knn_calls']+=1;np.testing.assert_array_equal(kwargs['features'],selected)
            for method in projections:
                with Path(manifest['clustering'][method]['html']).open('a') as stream:stream.write(' Labelled')
            labels=primary/'clustering/labels.csv'
            with labels.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,['model','source_path','umap_x','umap_y','cluster_label','color']);writer.writeheader()
                for row in members:writer.writerow(dict(model=row['model'],source_path=row['source_path'],umap_x=0,umap_y=1,cluster_label='C1',color='#123456'))
            selection=primary/'clustering/selection.json';selection.write_text('{}')
            return dict(labels_csv=str(labels),report_json=str(selection),feature_input={'sha256':proof['feature_sha256']})
        funcs['knn']=knn
        def density(features,names,labels,output,**kwargs):
            self.assertEqual(features.dtype,np.float16);self.assertFalse(features.flags.writeable)
            self.assertTrue(kwargs['allow_stored_float16']);self.assertNotIn('expected_feature_sha256',kwargs)
            self.assertEqual(kwargs['mode'],'casp17_recursive');Path(output).mkdir()
            return dict(models=names,density=['C1']*5+['C1.unassigned'],UMAP_fits=15,HDBSCAN_extractions=30,
                feature_proof=dict(stored_dtype='float16',fit_dtype='float32',
                    feature_sha256=hashlib.sha256(features.astype(np.float32).tobytes()).hexdigest()))
        def density_labels(result,rows,output,**kwargs):
            self.assertEqual(kwargs['maximum_clusters'],len(names))
            a=Path(output)/'labels.csv';b=Path(output)/'assigned-labels.csv';a.write_text('All6models inclnoise');b.write_text('Assigned5models')
            return a,b
        def primary_rows(manifest,members):
            with Path(manifest['knn_clustering']['labels_csv']).open() as stream:return list(csv.DictReader(stream))
        return plan,context,dict(selected=mapped,native=funcs,density=density,density_labels=density_labels,
                                 primary_rows=primary_rows),state,proof

    def call(self,stage,plan,context,adapters):
        with patch.dict(os.environ,TMPDIR=context['stage_tmpdirs'][stage]),patch.object(stage_adapter,'assert_ssd_tmpdir',return_value=Path(context['stage_tmpdirs'][stage])):
            return stage_adapter.run_stage(stage,plan,context,check=lambda:None,log=lambda _:None,adapters=adapters)

    def test_all_four_stages_preserve_PCA_and_reuse_selected_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,context,adapters,state,proof=self.fixture(root)
            cache=Path(context['descriptor']['selected_cache']['path']);before=cache.read_bytes()
            pca=Path(root/'primary/clustering/pca_embeddings.npz').read_bytes()
            for name in stage_adapter.STAGES:
                result=self.call(name,plan,context,adapters);self.assertEqual(result['status'],'complete')
                self.assertNotIn('metadata',result['selected_cache_proof'])
                self.assertEqual(state['mapped'],state['unmapped'])
                if name!='density':self.assertFalse((root/'primary/manifest.json').exists())
            self.assertEqual(state['methods'],['umap','tsne']);state['cache_writer'].assert_not_called()
            self.assertEqual(cache.read_bytes(),before);self.assertEqual((root/'primary/clustering/pca_embeddings.npz').read_bytes(),pca)
            self.assertTrue(all(not o['enable_pymol'] and o['force_recompute'] and o['tsne_implementation']=='python' for o in state['worker_options']))
            manifest=json.loads((root/'primary/manifest.json').read_text());self.assertEqual(manifest['status'],'complete')
            self.assertEqual(manifest['deliverable_status'],'pending_model_selection_and_exports')
            density=json.loads((root/'density/density-receipt.json').read_text())
            self.assertTrue(density['default_cluster_labels']);self.assertFalse(density['representatives_selected'])
            self.assertNotEqual(density['feature_proof']['feature_sha256'],proof['feature_sha256'])
            for method in ('umap','tsne'):
                receipt=json.loads((root/f'primary/checkpoints/{method}.json').read_text())
                self.assertEqual(receipt['primary_PCA_fits'],0)
                for row in receipt['outputs']:self.assertEqual(stage_adapter.record(row['path']),row)
            snapshot=density['primary_manifest'];self.assertEqual(stage_adapter.record(snapshot['path']),snapshot)

    def test_failed_stage_closes_map_and_never_marks_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,context,adapters,state,_=self.fixture(root);state['fail']='umap'
            with self.assertRaisesRegex(ValueError,'Native stage failed'):self.call('umap',plan,context,adapters)
            self.assertEqual(state['mapped'],state['unmapped']);self.assertFalse((root/'primary/checkpoints/umap.json').exists())
            self.assertFalse((root/'primary/manifest.json').exists());self.assertTrue((root/'primary/checkpoints/pca.json').is_file())

    def test_labels_require_verified_all_methods_and_new_HTML_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,context,adapters,state,_=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'umap receipt'):self.call('labels',plan,context,adapters)
            self.assertEqual(state['knn_calls'],0);self.assertFalse((root/'primary/manifest.json').exists())

    def test_cache_composition_rejects_other_persistence_requests_without_writer(self):
        target=stage_adapter.ExistingPreparedCache(SimpleNamespace(),'SERCA',Path('/frozen'),{'path':'/frozen/cache.npz'})
        writer=Mock()
        for kwargs in (dict(residue_selection_key='custom',basename_keyword=''),
                       dict(residue_selection_key='shared_msa',basename_keyword='x'),
                       dict(residue_selection_key='shared_msa',basename_keyword='',metric_filter_column='score')):
            with self.assertRaisesRegex(ValueError,'persistence'):target.persist_clustering_cache('SERCA',writer,**kwargs)
        writer.assert_not_called();self.assertEqual(target.reuse_calls,0)

    def test_workspace_gate_precedes_native_imports(self):
        with patch.object(stage_adapter,'assert_ssd_tmpdir',side_effect=ValueError('wrong workspace')),patch.object(stage_adapter,'adapters_for_native') as native:
            with self.assertRaisesRegex(ValueError,'wrong workspace'):
                stage_adapter.run_stage('umap',{}, {},check=lambda:None,log=lambda _:None)
            native.assert_not_called()


if __name__=='__main__':unittest.main()
