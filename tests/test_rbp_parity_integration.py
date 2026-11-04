"""Integration of saved geometry/provenance only; native fits/renders are mocked."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rbp_parity as adapter

AVAILABLE = importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('src') is not None


@unittest.skipUnless(AVAILABLE, 'Optional NumPy/library imports for mocked integration')
class SavedArtifactIntegrationTests(unittest.TestCase):
    def numerical_fixture(self, root):
        import numpy as np
        names = ['first', 'second', 'third', 'fourth']
        matrix = (np.arange(4*36585, dtype=np.float32).reshape(4, 36585)/100).astype(np.float16)
        terminal = dict(retained_reference_positions=list(range(271)), pair_mask_sha256='f'*64)
        manifest = dict(terminal_detection=terminal, knn_clustering=dict(feature_input=dict(
            sha256=hashlib.sha256(matrix.tobytes()).hexdigest())))
        cache = root/'saved-features.npz'
        np.savez_compressed(cache, flattened=matrix,
            source_signature=json.dumps(dict(files=[['/never-read/'+name+'.bin', 0, 0] for name in names])),
            reference_positions=np.arange(271), requested_reference_positions=np.arange(271),
            terminal_detection=json.dumps(terminal), ignored_object_metadata=np.array([object()], dtype=object))
        entry = dict(path=str(cache), sha256=adapter.sha256(cache), maximum_npy_bytes=matrix.nbytes+1024)
        selected = [dict(model=name, cluster_label=label, confidence=None, rank=rank, state=rank)
                    for name, label, rank in [('fourth', 'C2', 1), ('third', 'C2', 2), ('first', 'C1', 1)]]
        selection = dict(identifier=adapter.TARGET, clusters=[
            dict(cluster_label='C2', selected=selected[:2]), dict(cluster_label='C1', selected=selected[2:])])
        return names, matrix, manifest, entry, selection

    def test_supplementary_reuses_exact_saved_float16_rows_and_selection_order(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); names, matrix, manifest, entry, selection = self.numerical_fixture(root)
            frozen_selection, frozen_manifest = deepcopy(selection), deepcopy(manifest)
            scratch = root/'scratch'; scratch.mkdir(); seen = {}; checks = []
            def grid(plan, executable, **kwargs):
                self.assertEqual(plan, frozen_selection)
                self.assertEqual(executable, '/unused/native-USalign')
                self.assertEqual(kwargs['workers'], 4)
                seen['grid'] = True
                return dict(labels=['C2', 'C1'], matrix=[[1., .7], [.7, 1.]])
            def bonsai(features, selected, output, *, identifier, feature_source, log):
                self.assertEqual(identifier, adapter.TARGET)
                self.assertEqual(features.dtype, np.float16)
                np.testing.assert_array_equal(features, matrix[[3, 2, 0]])
                self.assertEqual([r['model'] for r in selected], ['fourth', 'third', 'first'])
                self.assertEqual(feature_source['kind'], 'frozen_models_by_features')
                self.assertEqual(feature_source['source_cache_sha256'], entry['sha256'])
                self.assertEqual(feature_source['model_order_sha256'], adapter.compact_sha(names))
                self.assertEqual(feature_source['terminal_detection'], manifest['terminal_detection'])
                self.assertFalse((scratch/'rbp-saved-features.npy').exists())
                seen['bonsai'] = True
                return dict(schema_version=1, models=[dict(model=r['model']) for r in selected], scope='Mocked saved features')
            with patch('src.analysis.cluster_medoid_tm.calculate_medoid_tm', side_effect=grid), \
                 patch('src.analysis.bonsai_interactive.run_ensemble_bonsai', side_effect=bonsai), \
                 patch('src.analysis.bonsai_interactive.selected_shared_distance_features',
                       side_effect=AssertionError('Original .bin files must never be reread')), \
                 patch.dict(adapter.os.environ, {'USALIGN_OFFLOAD_MACHINE':'', 'USALIGN_PARALLEL_WORKERS':''}):
                result = adapter.supplementary(manifest, selection, root/'supplementary', '/unused/native-USalign',
                    lambda:checks.append(True), lambda message:None, feature_entry=entry, names=names, scratch=scratch)
            self.assertEqual(seen, dict(grid=True, bonsai=True)); self.assertTrue(checks)
            self.assertEqual(selection, frozen_selection); self.assertEqual(manifest, frozen_manifest)
            self.assertEqual(adapter.sha256(entry['path']), entry['sha256'])
            self.assertEqual(adapter.record(result['bonsai']['path']), result['bonsai'])
            self.assertEqual(adapter.record(result['medoid_tm']['path']), result['medoid_tm'])
            self.assertFalse(list(scratch.iterdir()))

    def test_regions_record_exact_parent_proof_and_preserve_all_other_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flexibility = dict(cores=[dict(indices=[0, 1, 270], key='D1')], residue_keys=[['A', '0'], ['A', '0A']],
                model_count=10926, reason='Retain exact saved finding', evidence=dict(value=.42))
            report = dict(identifier=adapter.TARGET, structural_flexibility=flexibility,
                          clusters=[dict(cluster_label='Cluster 1', untouched=dict(value=[1, 2, 3]))],
                          other_saved_fields=dict(pse='/private/retained.pse', confidence_metric='mean_plddt'))
            parent_report = root/'original-report.json'; parent_report.write_text(json.dumps(report))
            manifest = dict(identifier=adapter.TARGET, cluster_ensembles=dict(report=str(parent_report), status='awaiting_user_view'),
                terminal_detection=dict(retained_reference_positions=list(range(271))),
                clustering=dict(pca=dict(embeddings='/untouched/pca.npz')), extra_saved_field={'keep':True})
            parent_manifest = root/'original-manifest.json'; parent_manifest.write_text(json.dumps(manifest))
            manifest['_source_path'] = str(parent_manifest)
            originals = deepcopy(manifest), deepcopy(report)
            original_bytes = parent_manifest.read_bytes(), parent_report.read_bytes()
            figure = dict(domain_colors={'D1':'#00BFFF'}, chain_breaks=[], color_scale_max_angstrom2=42.)
            def render(saved, cores, destination):
                self.assertIs(saved, manifest)
                self.assertEqual(cores, flexibility)
                self.assertIsNot(cores, flexibility)
                Path(destination).write_bytes(b'Mocked image bytes; no rendering performed')
                return deepcopy(figure)
            with patch('src.analysis.domain_variance_figure.render_saved_domain_variance', side_effect=render) as mocked:
                result = adapter.region_artifacts(manifest, report, root/'regions', lambda:None)
            self.assertEqual(mocked.call_count, 1)
            derived = json.loads(Path(result['ensemble_report']['path']).read_text())
            associated = json.loads(Path(result['manifest']['path']).read_text())
            proof = dict(kind='saved_variance_presentation', parent_workflow_manifest_sha256=adapter.sha256(parent_manifest),
                         parent_ensemble_report_sha256=adapter.sha256(parent_report))
            expected_report = deepcopy(report)
            expected_report['structural_flexibility']['heatmap'] = figure; expected_report['presentation_derivation'] = proof
            self.assertEqual(derived, expected_report)
            expected_manifest = deepcopy(manifest); expected_manifest.pop('_source_path')
            expected_manifest['presentation_derivation'] = proof
            expected_manifest['cluster_ensembles']['report'] = result['ensemble_report']['path']
            self.assertEqual(associated, expected_manifest)
            self.assertEqual((manifest, report), originals)
            self.assertEqual((parent_manifest.read_bytes(), parent_report.read_bytes()), original_bytes)
            self.assertEqual(adapter.record(result['image']['path']), result['image'])


if __name__ == '__main__':unittest.main()
