import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_ca import write_ca


class PublicationTests(unittest.TestCase):
    def test_native_zero_and_insertion_labels_and_coordinate_tokens_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original = root/'native.pdb'; output = root/'display.cif.gz'
            lines = []
            for i, (number, code) in enumerate([(0, ' '), (1, 'A'), (1, 'B')], 1):
                lines.append(f'ATOM  {i:5d}  CA  ALA A{number:4d}{code}   {i+.125:8.3f}{-2.25:8.3f}{3.5:8.3f}  1.00 85.00           C')
            raw = ('\n'.join(lines)+'\nEND\n').encode(); original.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            descriptor = write_ca(original, output, native_sha256=digest, expected_count=3)
            display = gzip.decompress(output.read_bytes())
            self.assertEqual(descriptor['sha256'], hashlib.sha256(display).hexdigest())
            self.assertEqual(descriptor['native_sha256'], digest)
            rows = [line.split() for line in display.decode().splitlines() if line.startswith('ATOM ')]
            self.assertEqual([(row[15],row[9]) for row in rows], [('0','?'),('1','A'),('1','B')])
            self.assertEqual([row[10] for row in rows], ['1.125','2.125','3.125'])
            self.assertEqual(original.read_bytes(),raw)

    def test_wrong_hash_or_ambiguous_count_does_not_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original = root/'native.pdb'; original.write_text('END\n'); output = root/'display.cif.gz'
            with self.assertRaisesRegex(ValueError,'hash changed'):
                write_ca(original,output,native_sha256='a'*64,expected_count=3)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError,'count'):
                write_ca(original,output,native_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),expected_count=3)
            self.assertFalse(output.exists())

    def test_publication_uses_shared_renderer_and_has_no_custom_template(self):
        source = Path(__file__).resolve().parents[1]
        publisher = (source/'scripts/publish_atlas.py').read_text()
        self.assertIn('from structure_clustering.publication import export',publisher)
        self.assertRegex(publisher,r"'renderer'\s*:\s*'ais-explorer-v1'")
        self.assertIn("for method in ['pca', 'umap', 'tsne']",publisher)
        self.assertIn("manifest['analysis_status'] != 'complete'",publisher)
        self.assertFalse((source/'templates/screening-atlas').exists())

    def test_coordinate_delivery_is_representative_only_unless_explicitly_selected(self):
        from publish_atlas import structure_delivery
        self.assertEqual(structure_delivery({}),'representatives')
        self.assertEqual(structure_delivery({'structure_delivery':'all-point-ca'}),'all-point-ca')
        with self.assertRaisesRegex(ValueError,'Unknown structure delivery'):
            structure_delivery({'structure_delivery':'silent-fallback'})

    def test_partial_release_requires_explicit_pending_cohorts(self):
        from publish_atlas import publication_scope
        study={'targets':{'RBP':{},'5NT':{},'SERCA':{}}}
        self.assertEqual(publication_scope({'targets':{'RBP':{}},'pending_targets':['5NT','SERCA']},study),(['RBP'],['5NT','SERCA']))
        for pending in [[],['SERCA'],['5NT','SERCA','RBP'],['5NT','SERCA','SERCA']]:
            with self.subTest(pending=pending),self.assertRaisesRegex(ValueError,'explicitly account'):
                publication_scope({'targets':{'RBP':{}},'pending_targets':pending},study)

    def test_strict_bonsai_binding_matches_selected_view_and_retained_mask(self):
        from copy import deepcopy
        from publish_atlas import add_supplementary
        from snapshot import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            tree={'schema_version':1,'models':[{'model':name} for name in ['m1','m2']],
                'layouts':[],'method':'Saved fixture','scope':'Two selected fixture models','group_count':1,
                'provenance':{**{key:'a'*64 for key in ['result_sha256','hdf_sha256','tree_sha256',
                    'model_order_sha256','source_features_sha256','pair_mask_sha256']},
                    'retained_reference_positions':[0,1,2],'feature_count':3,'settings':{}}}
            def attempt(value,selected,retained,folder):
                artifact=root/(folder+'.json');artifact.write_text(json.dumps(value))
                destination=source/folder;destination.mkdir()
                data={'representatives':[],'provenance':{'pair_mask_sha256':'a'*64}}
                add_supplementary(data,{'id':'density','bonsai':{'path':str(artifact),'sha256':sha256(artifact)}},
                    destination,source,['m1','m2','m3'],selected_models=selected,
                    retained_reference_positions=retained,strict=True)
                return data
            valid=attempt(tree,['m2','m1'],[0,1,2],'valid')
            self.assertEqual(valid['bonsai']['selected_models'],2)
            for change in ['view','retained','pair_mask']:
                value=deepcopy(tree);selected=['m1','m2'];retained=[0,1,2]
                if change=='view':selected=['m1','m3']
                elif change=='retained':retained=[0,2]
                else:value['provenance']['pair_mask_sha256']='b'*64
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'population or residue mask'):
                    attempt(value,selected,retained,change)

    def test_saved_report_downloads_are_scoped_to_their_actual_label_view(self):
        from publish_atlas import report_downloads
        from snapshot import sha256
        with tempfile.TemporaryDirectory() as temporary:
            artifact=Path(temporary)/'saved-report.txt';artifact.write_text('Predefined report receipt fixture only')
            report={'file':str(artifact),'sha256':sha256(artifact),'models':4,
                'url':'https://example.org/original-umap-report.pptx','label':'Original UMAP preliminary report',
                'view_ids':['primary']}
            self.assertEqual(report_downloads([report],4,'primary',strict=True)[0]['label'],report['label'])
            self.assertEqual(report_downloads([report],4,'density',strict=True),[])
            legacy={key:value for key,value in report.items() if key!='view_ids'}
            self.assertEqual(len(report_downloads([legacy],4,'whole')),1)
            self.assertEqual(len(report_downloads([legacy],4,'density')),1)
            for selection in [None,[],['primary','primary'],['whole'],'primary',['retired']]:
                wrong=dict(report)
                if selection is None:del wrong['view_ids']
                else:wrong['view_ids']=selection
                with self.subTest(selection=selection),self.assertRaisesRegex(ValueError,'explicit valid'):
                    report_downloads([wrong],4,'density',strict=True)

    def density_fixture(self):
        from collections import Counter
        names=[f'opaque_model_{i:04d}' for i in range(60)]
        initial=['C1']*40+['C2']*20
        final=['C1.1']*15+['C1.2']*15+['C1.unassigned']*10+['C2']*20
        settings={'seeds':[42,7,123],'n_neighbors':30,'min_dist':0.,'min_cluster_size':15,'min_samples':11,'workers':2}
        fits=[]
        for seed,leaf_noise,eom_noise in [(42,10,2),(7,8,3),(123,6,4)]:
            for method,noise,clusters in [('leaf',leaf_noise,2),('eom',eom_noise,1)]:
                fits.append({'seed':seed,'method':method,'clusters':clusters,'noise_models':noise,
                    'elapsed_seconds_from_seed_start':100.+seed/1000,'private_path':'/home/private/fit'})
        agreement={}
        for method,records in [('leaf',[(7,28,.6,.95,.85),(123,29,.55,.93,.85)]),
                               ('eom',[(7,36,.8,.9,.925),(123,35,.75,.85,.9)])]:
            agreement[method]={str(seed):{'ari_including_noise':ari,'jointly_assigned_models':joint,
                'jointly_assigned_fraction':joint/40,'ari_jointly_assigned':joint_ari,
                'noise_status_agreement':noise_agreement,'private_path':'/home/private/repeat'}
                for seed,joint,ari,joint_ari,noise_agreement in records}
        directory='/home/private/runtime/fits/selection-00-10D'
        analysis={'mode':'casp17_recursive','structural_clusters':3,'unassigned_models':10,
            'selected_fits':1,'UMAP_fits':3,'HDBSCAN_extractions':6,
            'inference_settings':dict(settings,dimension=10,metric='euclidean',primary_extraction='leaf',primary_seed=42,
                private_path='/home/private/runtime'),
            'stopping':{'round':1,'reason':'two_or_fewer_child_clusters','child_clusters':2,'private_path':'/home/private/stop'},
            'history':[{'round':0,'labels':initial,'populations':dict(Counter(initial)),'structural_clusters':2},
                {'round':1,'labels':final,'populations':dict(Counter(final)),'selected_leaf':'C1','selected_models':40,
                    'child_clusters':2,'noise_models_in_fit':10,'structural_clusters':3,'unassigned_models':10,'fit':directory}],
            'selections':[{'directory':directory,'selected_indices':list(range(40)),'models':names[:40],
                'feature_sha256':'b'*64,'files_sha256':{'result.json':'c'*64,'/home/private/extra':'d'*64},
                'diagnostics':{'dimension':10,'settings':settings,'fits':fits,'seed_agreement':agreement,'private_path':'/home/private/result'}}],
            'native_source':{'path':'/home/private/native.py','sha256':'d'*64,'hierarchical_colors_sha256':'e'*64},
            'private_bindings':{'host':'/home/private/runtime'},'source_model_files':['/home/private/native.pdb']}
        return analysis,{name:{'cluster_label':label} for name,label in zip(names,final)},names

    def test_density_diagnostics_publish_saved_counts_and_agreement_without_private_arrays(self):
        from publish_atlas import density_diagnostics
        analysis,labels,names=self.density_fixture()
        result=density_diagnostics(analysis,labels,names,'a'*64,strict=True)
        self.assertEqual(result['status'],'complete');self.assertEqual(result['models'],60)
        self.assertEqual(result['structural_clusters'],3);self.assertEqual(result['unassigned_models'],10)
        self.assertEqual(result['repeat_seed_leaf_noise_varied_selections'],1)
        self.assertEqual(result['rounds'][1]['noise_models_in_fit'],10)
        self.assertEqual(result['selections'][0]['fits'][0]['elapsed_seconds_from_seed_start'],100.042)
        self.assertEqual(result['selections'][0]['seed_agreement']['leaf']['7']['ari_including_noise'],.6)
        self.assertEqual(result['selections'][0]['seed_agreement']['leaf']['7']['ari_jointly_assigned'],.95)
        self.assertIn('not a stability acceptance gate',result['interpretation'])
        encoded=json.dumps(result)
        for forbidden in ['/home/','opaque_model_','directory','selected_indices','populations','private_','source_model_files']:
            self.assertNotIn(forbidden,encoded)
        self.assertNotIn('labels',result['rounds'][1]);self.assertNotIn('models',result['selections'][0])
        self.assertNotIn('≥0.80',result['interpretation'])

    def test_density_diagnostics_reject_inconsistent_native_evidence_and_make_legacy_absence_explicit(self):
        from publish_atlas import density_diagnostics
        from copy import deepcopy
        analysis,labels,names=self.density_fixture()
        for change in ['final_count','round_count','fit_count','agreement_fraction','agreement_noise','source_hash','time','parent']:
            value=deepcopy(analysis)
            if change=='final_count':value['unassigned_models']=9
            elif change=='round_count':value['history'][1]['populations']['C1.1']=14
            elif change=='fit_count':value['selections'][0]['diagnostics']['fits'][0]['noise_models']=9
            elif change=='agreement_fraction':value['selections'][0]['diagnostics']['seed_agreement']['leaf']['7']['jointly_assigned_fraction']=.5
            elif change=='agreement_noise':value['selections'][0]['diagnostics']['seed_agreement']['leaf']['7']['noise_status_agreement']=.1
            elif change=='source_hash':value['native_source']['sha256']='/home/private/native.py'
            elif change=='time':value['selections'][0]['diagnostics']['fits'][0]['elapsed_seconds_from_seed_start']=float('nan')
            else:value['history'][1]['selected_leaf']='/home/private/leaf'
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Saved density diagnostics'):
                density_diagnostics(value,labels,names,'a'*64,strict=True)
        legacy=density_diagnostics({},labels,names,'a'*64)
        self.assertEqual(legacy['status'],'legacy_not_recorded')
        with self.assertRaisesRegex(ValueError,'complete native evidence'):
            density_diagnostics({},labels,names,'a'*64,strict=True)
        with self.assertRaisesRegex(ValueError,'complete native evidence'):
            density_diagnostics({'mode':'casp17_recursive'},labels,names,'a'*64)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'Native saved-array reader is optional in source CI')
    def test_missing_or_mismatched_real_projection_cohort_is_rejected(self):
        import numpy as np
        from publish_atlas import projections
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest={'clustering':{}}
            for method in ['pca','umap','tsne']:
                path=root/(method+'.npz')
                np.savez(path,filenames=['one','two'],embeddings=np.array([[0.,1.],[2.,3.]]))
                manifest['clustering'][method]={'embeddings':str(path)}
            names, _ = projections(manifest,{'one':{},'two':{}})
            self.assertEqual(names,['one','two'])
            np.savez(root/'tsne.npz',filenames=['old-one','old-two'],embeddings=np.array([[0.,1.],[2.,3.]]))
            with self.assertRaisesRegex(ValueError,'complete accepted cohort'):
                projections(manifest,{'one':{},'two':{}})


if __name__ == '__main__': unittest.main()
