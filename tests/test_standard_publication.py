import gzip
import hashlib
import importlib.util
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
        self.assertIn("'renderer':'ais-explorer-v1'",publisher)
        self.assertIn("for method in ['pca', 'umap', 'tsne']",publisher)
        self.assertIn("manifest['analysis_status'] != 'complete'",publisher)
        self.assertFalse((source/'templates/screening-atlas').exists())

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
