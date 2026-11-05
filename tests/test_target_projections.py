"""Tiny saved arrays and mocked workers/plots/labels only; no engine fits."""
import hashlib
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import target_projections as adapter
import test_terminal_rescreen as terminal_fixture
from test_terminal_rescreen import bound, write

AVAILABLE = importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(AVAILABLE,'Optional numeric fixture runtime')
class PrimaryProjectionTests(unittest.TestCase):
    def fixture(self,root):
        import numpy as np
        desc,axes,baseline,full,engine=terminal_fixture.SavedTerminalTests().fixture(root,removed=range(3))
        root=Path(root);names=axes['names'];positions=axes['reference_positions']
        metadata=dict(axes['metadata'],version=1,reference_group_id='sg1',
            sequence_groups=[dict(id='sg1',sequence_length=len(positions),reference_to_group_index=positions,
                sequence_to_matrix_index=positions,chain_ids=['A']*len(positions),residue_numbers=positions,
                file_basenames=names)],file_assignments={name:'sg1' for name in names})
        source_metadata=root/'distogram_selection_metadata.json.gz'
        source_metadata.write_bytes(gzip.compress(json.dumps(metadata).encode()))
        desc['source_metadata']=bound(source_metadata)
        rows=[]
        for name,row in zip(names,full):
            path=root/(name+'.bin');path.write_bytes(row.tobytes());status=path.stat()
            rows.append([str(path),status.st_size,status.st_mtime_ns])
        signature=dict(files=rows,metadata_sha256=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest())
        native_manifest=dict(file_format='bin',storage='condensed',dtype='float16',endianness='little',
            variable_length=True,per_file_n_residues={name:len(positions) for name in names})
        desc['distogram_manifest']=write(root/'distogram_manifest.json',native_manifest)
        desc['distogram_manifest']['bytes']=Path(desc['distogram_manifest']['path']).stat().st_size
        source=Path(desc['cache']['path'])
        np.savez(source,flattened=full,reference_positions=np.asarray(axes['reference_positions']),
            requested_reference_positions=np.asarray(axes['reference_positions']),source_signature=json.dumps(signature),
            terminal_detection=json.dumps(baseline),filenames=np.asarray(names,dtype=object))
        desc['cache'].update(bound(source),bytes=source.stat().st_size)
        report=engine[0].detect_terminal_variance.return_value
        keep=np.ones(34,bool);keep[:3]=False;i,j=np.triu_indices(34,1);mask=keep[i]&keep[j]
        report.update(reference_positions=list(range(34)),retained_reference_positions=list(range(3,34)),
                      input_feature_count=len(mask),retained_feature_count=int(mask.sum()),
                      pair_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        report_bound=write(root/'accepted-report.json',report)
        members=[]
        for name in names:
            path=root/(name+'.pdb');path.write_text('Mocked original: '+name)
            members.append(dict(model=name,source_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        old=root/'old-metrics.sqlite';old.write_text('Old immutable seed')
        fresh=root/'fresh-metrics.sqlite';fresh.write_text('Fresh complete copied metrics')
        prior=dict(identifier=desc['target'],structure_count=len(names),metrics_db=str(old),
            ensemble_selection_scope=dict(unscored_cluster_policy='structural_medoid'),
            dataset_scope='All original models and physicality flags',input_inventory=str(root/'inventory.json'),
            ensemble_confidence_preflight=dict(analysis_model_count=6,model_count=6))
        write(root/'inventory.json',dict(models=[dict(model=r['model'],original_source_path=r['source_path'],
            original_sha256=r['sha256']) for r in members]))
        desc['prior_workflow_manifest']=write(root/'prior.json',prior)
        model_types=write(root/'old-label-rules.json',dict(schema_version=1,rules=[{'key':'native','name':'Preserved label'}]))
        clusters=write(root/'old-cluster-settings.json',dict(schema_version=1,settings={'repeat_seeds':[7,123],'query_seed':42,'maximum_subclusters_per_component':8}))
        scratch=root/'scratch';scratch.mkdir()
        return desc,report_bound,fresh,members,model_types,clusters,scratch,full[:,mask].copy(order='C')

    def mock_adapters(self,root,expected,*,failure=None):
        import numpy as np
        state=dict(active=False,methods=[],checkpoints=[],worker_options=[],knn_calls=0)
        cache=root/'processed'/'SERCA_fixture'/'clustering_preprocessed_results'/'cache.npz'
        def cache_path(*args,**kwargs):
            self.assertEqual(args,('SERCA_fixture',root.name))
            self.assertEqual(kwargs,dict(basename_keyword='',residue_selection_key='shared_msa'))
            cache.parent.mkdir(parents=True,exist_ok=True);return cache
        class Worker:
            def __init__(self,**options):self.options=options
            def run(worker):
                options=worker.options;method=options['clustering_method'];state['active']=True
                try:
                    state['methods'].append(method);state['worker_options'].append(options)
                    if failure==method:
                        options['error_callback']('mocked engine stop');return
                    matrix=options['prepared_distograms'].flattened
                    np.testing.assert_array_equal(matrix,expected)
                    assert matrix.dtype==np.float16 and not matrix.flags.writeable
                    if not cache.exists():np.savez(cache,flattened=matrix)
                    Path(options['output_file']).write_text('var workflowClusterAnnotations = null;')
                    names=[Path(p).stem for p in options['prepared_distograms'].paths]
                    xy=np.arange(len(names)*2,dtype=float).reshape(len(names),2)
                    options['embedding_callback'](xy,names,options['prepared_distograms'].paths)
                finally:state['active']=False
        def plots(xy,names,paths,method,schemes,metrics,source,output,**kwargs):
            output=Path(output);output.mkdir(parents=True);image=output/(method+'-origin.png');image.write_bytes(b'Mocked figure')
            return [dict(image=str(image))]
        def knn(manifest,projections,**kwargs):
            state['knn_calls']+=1
            np.testing.assert_array_equal(kwargs['features'],expected)
            self.assertEqual(kwargs['feature_names'],[row['model'] for row in state['members']])
            # The real public operation annotates the current HTML. Base copies
            # in method receipts must remain immutable and hash-verifiable.
            for method in projections:
                with Path(manifest['clustering'][method]['html']).open('a') as stream:stream.write('\nLabel annotation')
            labels=Path(manifest['output_dir'])/'clustering/labels.csv';labels.write_text('model,cluster_label\n')
            report=Path(manifest['output_dir'])/'clustering/selection.json';report.write_text('{}')
            return dict(labels_csv=str(labels),report_json=str(report),feature_input=dict(
                sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
        funcs=dict(prepared=lambda **kwargs:SimpleNamespace(**kwargs),worker=Worker,
            labels=lambda metadata,positions:[str(p) for p in positions],cluster_settings=lambda path:SimpleNamespace(repeat_seeds=(7,123),query_seed=42,maximum_subclusters_per_component=8),
            knn=knn,schemes=lambda db,path:['unchanged labels'],plots=plots,
            ratios=lambda xy,matrix:[.6,.2],cache_path=cache_path,
            confidence=lambda manifest,paths:(None,None,None,manifest['ensemble_confidence_preflight'],None),
            metadata=dict(read=lambda directory:json.loads(gzip.decompress((directory/'distogram_selection_metadata.json.gz').read_bytes())),
                sanitize=lambda metadata:metadata,validate=lambda metadata:[],paths=lambda directory:sorted(directory.glob('*.bin')),
                manifest=lambda directory:json.loads((directory/'distogram_manifest.json').read_text()),
                dimension=lambda path,manifest:manifest['per_file_n_residues'][path.stem],
                resolve=lambda assignments,stem,metadata,count:stem if assignments.get(stem)=='sg1' else None))
        return funcs,state

    def execute(self,root,*,failure=None,edit=None):
        desc,report,metrics,members,types,clusters,scratch,expected=self.fixture(root)
        funcs,state=self.mock_adapters(root,expected,failure=failure);state['members']=members
        if edit:edit(desc,report,members)
        def checkpoint(identifier):
            self.assertFalse(state['active'],'A running estimator cannot be a safe checkpoint')
            state['checkpoints'].append(identifier)
        with patch.dict(os.environ,DATA_PROCESSED=str(root/'processed')),patch.object(adapter,'assert_ram_scratch') as ram:
            result=adapter.run(desc,report,metrics,root/'primary',scratch,members=members,
                model_type_config=types,cluster_config=clusters,check=lambda:None,checkpoint=checkpoint,adapters=funcs)
            ram.assert_called_once_with(scratch)
        return result,state,desc,metrics,expected

    def test_native_public_route_exact_values_whole_registry_and_closed_receipts(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result,state,desc,metrics,expected=self.execute(root)
            self.assertEqual(state['methods'],['pca','umap','tsne']);self.assertEqual(state['knn_calls'],1)
            self.assertTrue(all(not o['enable_pymol'] and o['force_recompute'] and o['tsne_implementation']=='python'
                                and o['metrics_db_path']==str(metrics) for o in state['worker_options']))
            feature=result['retained_features'];self.assertEqual(feature['dtype'],'float16');self.assertTrue(feature['temporary'])
            self.assertEqual(feature['feature_sha256'],hashlib.sha256(expected.tobytes()).hexdigest())
            self.assertEqual(feature['file_sha256'],adapter.record(feature['path'])['sha256'])
            mapped=np.load(feature['path'],mmap_mode='r',allow_pickle=False)
            try:np.testing.assert_array_equal(mapped,expected)
            finally:mapped._mmap.close()
            manifest=json.loads(Path(result['manifest']['path']).read_text())
            self.assertEqual(manifest['analysis_status'],'complete')
            self.assertEqual(manifest['deliverable_status'],'pending_density_model_selection_and_exports')
            self.assertFalse(manifest['derivation']['coordinates_inherited'])
            self.assertIsInstance(manifest['model_type_config'],str)
            self.assertEqual(manifest['model_type_config_source'],'explicit')
            self.assertEqual(manifest['ensemble_selection_scope']['unscored_cluster_policy'],'structural_medoid')
            self.assertEqual(manifest['dataset_scope'],'All original models and physicality flags')
            for method in state['methods']:
                receipt=json.loads((root/'primary/checkpoints'/(method+'.json')).read_text())
                for artifact in receipt['outputs']:self.assertEqual(adapter.record(artifact['path']),artifact)
                self.assertIn('Label annotation',(root/'primary/clustering'/(method+'.html')).read_text())
                self.assertNotIn('Label annotation',Path(receipt['base_html']).read_text())
            self.assertEqual(state['checkpoints'][-1],'SERCA_fixture:primary-complete')
            self.assertEqual(result['projection_fits_max'],6)
            self.assertTrue(Path(result['feature_cache']['path']).is_file())
            self.assertEqual(metrics.read_text(),'Fresh complete copied metrics')

    def test_method_failure_preserves_only_closed_previous_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'umap failed'):self.execute(root,failure='umap')
            self.assertTrue((root/'primary/checkpoints/pca.json').is_file())
            self.assertFalse((root/'primary/checkpoints/umap.json').exists())
            self.assertFalse((root/'primary/manifest.json').exists())
            self.assertTrue((root/'scratch/SERCA_fixture-retained-features.npy').is_file())

    def test_cooperative_stop_during_staging_never_calls_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            desc,report,metrics,members,types,clusters,scratch,expected=self.fixture(root)
            funcs,state=self.mock_adapters(root,expected);state['members']=members
            def checkpoint(identifier):
                if ':features:' in identifier:raise RuntimeError('safe staged-row stop')
            with patch.dict(os.environ,DATA_PROCESSED=str(root/'processed')),patch.object(adapter,'assert_ram_scratch'):
                with self.assertRaisesRegex(RuntimeError,'safe staged-row stop'):
                    adapter.run(desc,report,metrics,root/'primary',scratch,members=members,
                        model_type_config=types,cluster_config=clusters,check=lambda:None,checkpoint=checkpoint,adapters=funcs)
            self.assertEqual(state['methods'],[])
            self.assertFalse((root/'primary/manifest.json').exists())
            self.assertTrue(Path(desc['cache']['path']).is_file())

    def test_scope_and_metadata_rebuild_risks_fail_before_any_worker(self):
        for risk in ('extra-repeat','native-repeat','invalid-metadata','read-mismatch','extra-bin',
                     'changed-bin','missing-assignment','dimension','confidence','inventory'):
            with self.subTest(risk=risk),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                desc,report,metrics,members,types,clusters,scratch,expected=self.fixture(root)
                funcs,state=self.mock_adapters(root,expected);state['members']=members
                if risk=='extra-repeat':
                    value=json.loads(Path(clusters['path']).read_text());value['settings']['repeat_seeds'].append(999)
                    clusters.update(write(Path(clusters['path']),value))
                if risk=='native-repeat':funcs['cluster_settings']=lambda path:SimpleNamespace(repeat_seeds=(7,123,999),query_seed=42,maximum_subclusters_per_component=8)
                if risk=='invalid-metadata':funcs['metadata']['validate']=lambda metadata:['Incomplete mappings']
                if risk=='read-mismatch':funcs['metadata']['read']=lambda directory:{'different_sidecar':True}
                if risk=='extra-bin':(root/'unrecorded.bin').write_bytes(b'Unrecorded')
                if risk=='changed-bin':(root/'model-0.bin').write_bytes(b'Changed distance row')
                if risk=='missing-assignment':funcs['metadata']['resolve']=lambda *args:None
                if risk=='dimension':funcs['metadata']['dimension']=lambda *args,**kwargs:33
                if risk=='confidence':funcs['confidence']=lambda manifest,paths:(None,None,None,{'analysis_model_count':5},None)
                if risk=='inventory':
                    value=json.loads((root/'inventory.json').read_text());value['models'][0]['original_sha256']='0'*64
                    write(root/'inventory.json',value)
                with patch.dict(os.environ,DATA_PROCESSED=str(root/'processed')),patch.object(adapter,'assert_ram_scratch'):
                    with self.assertRaises(ValueError):
                        adapter.run(desc,report,metrics,root/'primary',scratch,members=members,
                            model_type_config=types,cluster_config=clusters,check=lambda:None,checkpoint=lambda _:None,adapters=funcs)
                self.assertEqual(state['methods'],[])
                self.assertFalse((root/'primary/manifest.json').exists())

    @unittest.skipUnless(importlib.util.find_spec('src') is not None,'Optional native public metadata helpers')
    def test_real_public_metadata_helpers_accept_only_complete_static_fixture(self):
        import numpy as np
        from src.analysis.distogram_selection_metadata import (read_distogram_selection_metadata,
            sanitize_shared_reference_positions,validate_distogram_selection_metadata,resolve_file_assignment_key)
        from src.analysis.half_matrix import list_distogram_paths,read_distogram_manifest,peek_distance_matrix_n_residues
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);desc,*_=self.fixture(root)
            metadata=read_distogram_selection_metadata(root)
            with np.load(desc['cache']['path'],allow_pickle=False) as saved:signature=json.loads(str(saved['source_signature']))
            funcs=dict(read=read_distogram_selection_metadata,sanitize=sanitize_shared_reference_positions,
                validate=validate_distogram_selection_metadata,resolve=resolve_file_assignment_key,
                paths=list_distogram_paths,manifest=read_distogram_manifest,dimension=peek_distance_matrix_n_residues)
            proof=adapter.metadata_preflight(metadata,signature,root,list(range(34)),funcs,lambda:None,read_distogram_manifest(root))
            self.assertEqual(proof['files'],6)
            self.assertFalse(proof['coordinate_rebuild_required'])
            self.assertFalse(proof['sidecar_rewrite_required'])

    def test_bad_mask_registry_or_existing_namespace_fail_before_fits(self):
        for case in ('mask','registry','cache','prior-projection'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                def edit(desc,report,members):
                    if case=='mask':
                        value=json.loads(Path(report['path']).read_text());value['pair_mask_sha256']='0'*64
                        report.update(write(Path(report['path']),value))
                    if case=='registry':members.reverse()
                    if case=='prior-projection':
                        path=root/'processed/SERCA_fixture/pca_results/unique-old.npz'
                        path.parent.mkdir(parents=True);path.write_text('Never overwrite a prior projection')
                    if case=='cache':
                        path=root/'processed/SERCA_fixture/clustering_preprocessed_results/cache.npz'
                        path.parent.mkdir(parents=True);path.write_text('Never overwrite old native results')
                with self.assertRaises(ValueError):self.execute(root,edit=edit)
                self.assertFalse((root/'primary/checkpoints/pca.json').exists())


if __name__=='__main__':unittest.main()
