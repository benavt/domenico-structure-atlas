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
