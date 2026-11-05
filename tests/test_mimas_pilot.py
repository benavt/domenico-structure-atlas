"""Tiny static arrays/fake reducers only; no native fit/session/export is run."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mimas_pilot as pilot


def put(path,value):
    path.write_text(json.dumps(value));return pilot.record(path)


def fixture(root, *, array=False):
    spec={};models=[]
    for i in range(8):
        path=root/('model-'+str(i)+'.pdb');path.write_bytes(('Opaque model '+str(i)).encode())
        models.append(dict(model='m'+str(i),local_path=str(path),original_sha256=pilot.record(path)['sha256'],
                           flags={'physicality':'rejected' if i==7 else 'unavailable'}))
    matrixfile=root/'selected.npy'
    if array:
        import numpy as np
        matrix=np.arange(24,dtype=np.float16).reshape(8,3)
        np.save(matrixfile,matrix,allow_pickle=False);raw=hashlib.sha256(memoryview(matrix).cast('B')).hexdigest()
    else:matrixfile.write_bytes(b'Opaque metadata-only placeholder');raw='a'*64
    spec.update(features=pilot.record(matrixfile),members=put(root/'members.json',{'members':models}),
        feature_sha256=raw,model_order_sha256=hashlib.sha256('\n'.join(r['model'] for r in models).encode()).hexdigest(),
        model_count=8,construct_residues=3,retained_reference_positions=[0,1,2],pair_mask_sha256='b'*64)
    config=put(root/'components.json',dict(schema_version=1,settings=pilot.DEFAULT_COMPONENT_SETTINGS))
    plan=dict(target='TOY',inputs=spec,native=dict(library_commit=pilot.LIBRARY_COMMIT,
        density_source=dict(path='/bound/native/hd_density_experiment.py',commit=pilot.DENSITY_COMMIT,sha256=pilot.DENSITY_SHA256),
        cluster_config=config),phase_outputs={},workers=1)
    return plan,models


class StdlibGates(unittest.TestCase):
    def test_invalid_stage_and_existing_output_precede_native_imports(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pilot,'native_adapters') as native:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'supported stage'):pilot.run_stage({},'unknown',root)
            (root/'old.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Fresh empty'):pilot.run_stage({},'pca',root)
            native.assert_not_called();self.assertEqual((root/'old.txt').read_text(),'preserve')

    def test_bad_feature_descriptor_or_original_hash_stops_before_mapping_imports(self):
        for bad in ('feature','original'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp,patch.object(pilot,'native_adapters') as native:
                root=Path(tmp);plan,members=fixture(root)
                if bad=='feature':plan['inputs']['features']['sha256']='0'*64
                else:Path(members[0]['local_path']).write_text('Changed original')
                with self.assertRaisesRegex(ValueError,'Bound file changed|Original model bytes changed'):
                    pilot.run_stage(plan,'pca',root/'out')
                native.assert_not_called()

    def test_changed_model_order_core_or_configuration_is_rejected(self):
        for bad in ('order','core','config'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,_=fixture(root)
                if bad=='order':plan['inputs']['model_order_sha256']='0'*64
                if bad=='core':plan['target']='SERCA_P04191'
                if bad=='config':
                    value=dict(schema_version=1,settings=dict(pilot.DEFAULT_COMPONENT_SETTINGS,repeat_seeds=[7,123,456]))
                    plan['native']['cluster_config']=put(root/'components.json',value)
                with self.assertRaisesRegex(ValueError,'model order|SERCA requires|default component'):
                    pilot.load_inputs(plan,lambda:None)

    def test_cancelled_check_propagates_without_backend_import(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pilot,'native_adapters') as native:
            root=Path(tmp);plan,_=fixture(root)
            def stop():raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError,'cancelled'):pilot.run_stage(plan,'pca',root/'out',check=stop)
            native.assert_not_called()


@dataclass(frozen=True)
class Settings:
    maximum_subclusters_per_component:int=8
    minimum_feature_silhouette:float=.15
    minimum_seed_ari:float=.8
    silhouette_queries_per_component:int=512
    repeat_seeds:tuple=(7,123)
    query_seed:int=42


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy for tiny static pilot fixtures')
class StageTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.plan,self.members=fixture(self.root,array=True)
        self.enterContext(patch.dict(os.environ,DATA_PROCESSED=str(self.root/'native')))
        self.calls=[];self.opened=0;self.closed=0;self.options={};self.fail=None
        self.functions=self.adapters()

    def adapters(self):
        import numpy as np
        @contextmanager
        def mapped(entry):
            self.opened+=1;matrix=np.load(entry['path'],mmap_mode='r',allow_pickle=False)
            try:yield matrix
            finally:matrix._mmap.close();self.closed+=1
        def native_dir(identifier,method):return Path(os.environ['DATA_PROCESSED'])/identifier/(method+'_results')
        def reducer(method):
            def fit(distograms,names,paths,identifier,directory,**kwargs):
                self.calls.append(method);self.options[method]=kwargs
                self.assertIsNone(distograms);self.assertEqual(kwargs['flattened'].dtype,np.float16)
                self.assertFalse(kwargs['flattened'].flags.writeable);self.assertTrue(kwargs['force_recompute'])
                folder=native_dir(identifier,method);folder.mkdir(parents=True);(folder/'native.json').write_text('{}')
                if self.fail=='reorder':names.reverse()
                if self.fail=='original':Path(self.members[0]['local_path']).write_bytes(b'Changed during fake fit')
                values=np.column_stack((np.arange(len(names)),np.arange(len(names))+1)).astype(np.float32)
                return values,names,paths
            return fit
        def labels(xy,features,**kwargs):
            self.calls.append('labels');self.options['labels']=kwargs
            self.assertEqual(kwargs['base_seed'],42);self.assertEqual(kwargs['n_neighbors'],15)
            numeric=np.asarray([0]*4+[1]*4,dtype=np.int32)
            if self.fail=='labels':numeric=numeric[:-1]
            return numeric,dict(point_count=8,repeat_graphs=[{'seed':7},{'seed':123}],
                feature_input={'sha256':self.plan['inputs']['feature_sha256']})
        def density(features,names,primary,output,**kwargs):
            self.calls.append('density0');self.options['density0']=kwargs
            self.assertTrue(kwargs['allow_stored_float16']);self.assertEqual(kwargs['mode'],'global')
            self.assertNotIn('expected_feature_sha256',kwargs);self.assertEqual(features.dtype,np.float16)
            output=Path(output);output.mkdir()
            result=dict(models=names,density=['C1.1']*(len(names)-1)+['C1.unassigned'],mode='global',selected_fits=1,
                UMAP_fits=3,HDBSCAN_extractions=6,stopping={'reason':'single_global_fit'},
                native_source=dict(commit=pilot.DENSITY_COMMIT,sha256=pilot.DENSITY_SHA256),
                feature_proof=dict(stored_dtype='float16',fit_dtype='float32',retained_reference_positions=[0,1,2]))
            pilot.save(output/'analysis.json',result);return result
        return dict(matrix_loader=mapped,all_finite=lambda x:bool(np.isfinite(x).all()),
            feature_sha256=lambda x:hashlib.sha256(memoryview(x).cast('B')).hexdigest(),row_blocks=lambda x:(x[i:i+2] for i in range(0,len(x),2)),
            pca=reducer('pca'),umap=reducer('umap'),tsne=reducer('tsne'),settings=lambda _:Settings(),labels=labels,density=density,
            native_directory=native_dir,source_evidence=lambda:{'mock_native_engines':True},log=lambda _:None)

    def call_stage(self,stage):
        output=self.root/stage;output.mkdir()
        result=pilot.run_stage(self.plan,stage,output,adapters=self.functions)
        self.assertEqual(result,json.loads((output/'receipt.json').read_text()))
        self.plan['phase_outputs'][stage]=pilot.record(output/'receipt.json')
        return result

    def test_all_stages_keep_identity_flags_native_parameters_and_nine_fit_budget(self):
        total={'PCA':0,'UMAP':0,'TSNE':0,'HDBSCAN':0}
        before=pilot.record(self.plan['inputs']['features']['path'])
        receipts=[]
        for stage in pilot.STAGES:
            result=self.call_stage(stage);receipts.append(result)
            self.assertEqual(result['status'],'diagnostic_not_full_clustering');self.assertFalse(result['full_clustering_complete'])
            self.assertEqual(result['models'],[r['model'] for r in self.members])
            self.assertEqual(result['local_paths'],[r['local_path'] for r in self.members])
            self.assertEqual(result['model_flags'],[r['flags'] for r in self.members])
            for key,count in result['scientific_fits'].items():total[key]+=count
        self.assertEqual(total,{'PCA':2,'UMAP':6,'TSNE':1,'HDBSCAN':6})
        self.assertEqual(self.options['pca']['n_components'],2)
        self.assertEqual((self.options['umap']['n_neighbors'],self.options['umap']['min_dist'],self.options['umap']['n_components']),(15,.1,2))
        self.assertEqual(self.options['tsne']['implementation'],'python');self.assertEqual(self.options['tsne']['perplexity'],30.)
        self.assertEqual(self.calls,list(pilot.STAGES));self.assertEqual(self.opened,self.closed)
        self.assertEqual(pilot.record(before['path']),before)
        for result in receipts:
            for entry in result['outputs']:self.assertEqual(pilot.record(entry['path']),entry)
        final=receipts[-1];self.assertFalse(final['full_recursive_HDBSCAN']);self.assertFalse(final['default_cluster_labels'])
        import numpy as np
        with np.load(final['labels']['path'],allow_pickle=False) as saved:
            self.assertEqual(saved['cluster_labels'].tolist()[-1],'C1.unassigned')
            self.assertEqual(json.loads(saved['flags_json'].tolist()[-1]),{'physicality':'rejected'})

    def test_umap_can_run_without_pca_or_other_phase_outputs(self):
        self.call_stage('umap');self.assertEqual(self.calls,['umap'])
        self.assertFalse((self.root/'pca').exists());self.assertFalse((self.root/'labels').exists())

    def test_label_stage_requires_all_closed_projection_receipts(self):
        self.call_stage('pca')
        with self.assertRaisesRegex(ValueError,'Closed umap'):self.call_stage('labels')
        self.assertNotIn('labels',self.calls);self.assertFalse((self.root/'labels/receipt.json').exists())
        self.assertEqual(self.opened,self.closed)

    def test_reducer_model_reorder_or_input_change_never_seals_receipt(self):
        self.fail='reorder'
        with self.assertRaisesRegex(ValueError,'identities'):self.call_stage('pca')
        self.assertFalse((self.root/'pca/receipt.json').exists());self.assertEqual(self.opened,self.closed)
        with self.assertRaisesRegex(ValueError,'Fresh empty'):pilot.run_stage(self.plan,'pca',self.root/'pca',adapters=self.functions)

    def test_changed_original_during_fake_fit_is_detected_afterward(self):
        self.fail='original'
        with self.assertRaisesRegex(ValueError,'input identity changed'):self.call_stage('pca')
        self.assertFalse((self.root/'pca/receipt.json').exists());self.assertEqual(self.opened,self.closed)

    def test_raw_feature_hash_mismatch_closes_map_before_any_fit(self):
        self.plan['inputs']['feature_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Raw feature'):self.call_stage('pca')
        self.assertEqual(self.calls,[]);self.assertEqual(self.opened,self.closed)

    def test_native_configuration_directory_is_checked_without_mutating_it(self):
        self.functions['native_directory']=lambda *args:self.root/'wrong'
        with self.assertRaisesRegex(ValueError,'config was cached differently'):self.call_stage('pca')
        self.assertEqual(self.calls,[]);self.assertEqual(self.opened,self.closed)

    def test_tampered_prior_sidecar_cannot_be_used_for_labels(self):
        for stage in ('pca','umap','tsne'):self.call_stage(stage)
        receipt=json.loads((self.root/'umap/receipt.json').read_text())
        Path(receipt['projection']['path']).write_bytes(b'Changed saved projection')
        with self.assertRaisesRegex(ValueError,'Bound file changed'):self.call_stage('labels')
        self.assertNotIn('labels',self.calls);self.assertEqual(self.opened,self.closed)


if __name__=='__main__':unittest.main()
