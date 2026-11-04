"""Density adapter checks use saved tiny arrays and a mocked native fitter only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rbp_density as adapter

AVAILABLE = importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('src') is not None


class ProviderBoundaryTests(unittest.TestCase):
    def test_descriptor_and_provider_hash_are_checked_before_import(self):
        with self.assertRaisesRegex(ValueError, 'descriptor'):
            adapter.load_native({'path':'not-opened', 'sha256':'changed', 'commit':adapter.NATIVE_COMMIT})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'hd_density_experiment.py'
            path.write_text('raise AssertionError("Unverified code must never execute")')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                adapter.load_native(dict(path=str(path), sha256=adapter.NATIVE_SHA256,
                                         commit=adapter.NATIVE_COMMIT))

    def test_label_aliases_are_lossless_and_collisions_rejected(self):
        self.assertEqual(adapter._labels(['Cluster 2', 'Cluster 1', 'Cluster 2'])[0], ['C2', 'C1', 'C2'])
        for labels in (['C1', 'Cluster 1'], ['Cluster 0'], ['C1.unassigned'], ['unknown']):
            with self.assertRaises(ValueError):adapter._labels(labels)


class SavedFixtureBackend:
    """No estimators: write preselected test labels in the native saved-file API."""
    def __init__(self, labels, tamper=None, mutate=False):
        self.labels, self.tamper, self.mutate = labels, tamper, mutate
        self.calls = []

    @staticmethod
    def label_key(label):
        return tuple(int(x) for x in label.removeprefix('C').removesuffix('.unassigned').split('.'))

    def structural_labels(self, labels):
        return sorted({x for x in labels if not x.endswith('.unassigned')}, key=self.label_key)

    def named_children(self, raw, names, parent):
        groups = sorted(set(raw)-{-1}, key=lambda value:(-sum(raw == value),
                        min(name for name, label in zip(names, raw) if label == value)))
        mapping = {label:f'{parent}.{i}' for i, label in enumerate(groups, 1)}
        return [mapping.get(label, parent+'.unassigned') for label in raw], len(groups)

    @staticmethod
    def apply_split(current, indices, children, count):
        result = current.copy()
        if count >= 2:
            for index, child in zip(indices, children):result[int(index)] = child
        return result

    def fit_selection(self, features, names, indices, dimension, settings, directory, log):
        import numpy as np
        block = np.asarray(features[indices], dtype=np.float32)
        self.calls.append(dict(indices=indices.copy(), values=block.copy(), dimension=dimension,
                               settings=settings.copy()))
        if self.mutate:block[:] = -999
        raw = np.asarray(self.labels(len(indices), len(self.calls)), dtype=np.int64)
        directory.mkdir(parents=True)
        fits = []
        for seed in adapter.SEEDS:
            np.save(directory/f'embedding-seed-{seed}.npy', np.zeros((len(indices), dimension), np.float32))
            for method in ('leaf', 'eom'):
                np.savez_compressed(directory/f'{method}-seed-{seed}.npz', labels=raw,
                                    probabilities=np.ones(len(raw), np.float32))
                fits.append(dict(seed=seed, method=method, clusters=len(set(raw)-{-1}),
                                 noise_models=int(np.sum(raw == -1)), elapsed_seconds_from_seed_start=0.))
        result = dict(dimension=dimension, selected_indices=list(map(int, indices)),
                      models=[names[i] for i in indices], settings=settings, fits=fits,
                      seed_agreement={method:{str(seed):{} for seed in adapter.SEEDS[1:]}
                                      for method in ('leaf', 'eom')})
        if self.tamper:self.tamper(result, directory)
        (directory/'result.json').write_text(json.dumps(result))


@unittest.skipUnless(AVAILABLE, 'Requires optional NumPy/library runtime for mock-only adapter integration')
class DensityAdapterTests(unittest.TestCase):
    def inputs(self, count=90):
        import numpy as np
        return np.arange(count*3, dtype=np.float32).reshape(count, 3), [f'model-{i:04d}' for i in range(count)]

    def invoke(self, backend, matrix, names, primary, output, **kwargs):
        with patch.object(adapter, 'load_native', return_value=(backend, dict(sha256=adapter.NATIVE_SHA256))):
            return adapter.run(matrix, names, primary, output, native_source={}, retained_positions=[0, 1, 2], **kwargs)

    def test_recursive_largest_leaf_preserves_lineage_noise_and_model_count(self):
        import numpy as np
        matrix, names = self.inputs()
        backend = SavedFixtureBackend(lambda count, call:[0]*20+[1]*20+[2]*19+[-1] if call == 1 else [0]*15+[1]*15)
        with tempfile.TemporaryDirectory() as tmp:
            result = self.invoke(backend, matrix, names, ['Cluster 1']*60+['Cluster 2']*30, Path(tmp)/'out')
            self.assertEqual(result['models'], names)
            self.assertEqual(result['primary_label_map'], {'Cluster 1':'C1', 'Cluster 2':'C2'})
            self.assertEqual([r.get('selected_leaf') for r in result['history'][1:]], ['C1', 'C2'])
            self.assertEqual(result['density'][:20], ['C1.1']*20)
            self.assertEqual(result['density'][59], 'C1.unassigned')
            self.assertEqual(result['density'][60:], ['C2.1']*15+['C2.2']*15)
            self.assertEqual(result['structural_clusters'], 5)
            self.assertEqual(result['unassigned_models'], 1)
            self.assertEqual((result['selected_fits'], result['UMAP_fits'], result['HDBSCAN_extractions']), (2, 6, 12))
            self.assertEqual(result['stopping']['reason'], 'two_or_fewer_child_clusters')
            self.assertFalse(result['display_coordinates']['refitted'])
            self.assertEqual(backend.calls[0]['settings']['workers'], 4)
            np.testing.assert_array_equal(backend.calls[1]['values'], matrix[60:])
            self.assertTrue(result['selections'][0]['files_sha256']['leaf-seed-42.npz'])
            self.assertEqual(json.loads((Path(tmp)/'out/analysis.json').read_text())['density'], result['density'])

    def test_recursive_unsplit_parent_retains_legacy_noise_policy(self):
        for labels in ([0]*29+[-1], [-1]*30):
            with self.subTest(labels=labels[:1]), tempfile.TemporaryDirectory() as tmp:
                matrix, names = self.inputs(30)
                backend = SavedFixtureBackend(lambda count, call:labels)
                result = self.invoke(backend, matrix, names, ['Cluster 1']*30, Path(tmp)/'out')
                self.assertEqual(result['density'], ['C1']*30)
                self.assertEqual(result['history'][1]['noise_models_in_fit'], labels.count(-1))
                self.assertEqual(result['unassigned_models'], 0)

    def test_global_fits_whole_cohort_once_and_keeps_noise(self):
        matrix, names = self.inputs(30)
        backend = SavedFixtureBackend(lambda count, call:[0]*14+[1]*15+[-1])
        with tempfile.TemporaryDirectory() as tmp:
            result = self.invoke(backend, matrix, names, ['Cluster 2']*20+['Cluster 1']*10,
                                 Path(tmp)/'out', mode='global', workers=2)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(backend.calls[0]['indices'].tolist(), list(range(30)))
            self.assertEqual(result['density'][:14], ['C1.2']*14)
            self.assertEqual(result['density'][-1], 'C1.unassigned')
            self.assertEqual((result['UMAP_fits'], result['HDBSCAN_extractions']), (3, 6))
            self.assertEqual(result['stopping']['reason'], 'single_global_fit')

    def test_five_round_cap_is_not_inferred_from_cluster_sizes(self):
        matrix, names = self.inputs(200)
        backend = SavedFixtureBackend(lambda count, call:[0]*(count-30)+[1]*15+[2]*15)
        with tempfile.TemporaryDirectory() as tmp:
            result = self.invoke(backend, matrix, names, ['C1']*200, Path(tmp)/'out')
            self.assertEqual(len(backend.calls), 5)
            self.assertEqual(result['stopping']['reason'], 'five_refinement_limit')
            self.assertEqual((result['UMAP_fits'], result['HDBSCAN_extractions']), (15, 30))

    def test_small_leaf_stops_without_any_fit(self):
        matrix, names = self.inputs(15)
        backend = SavedFixtureBackend(lambda count, call:[])  # Must never be called.
        with tempfile.TemporaryDirectory() as tmp:
            result = self.invoke(backend, matrix, names, ['C1']*15, Path(tmp)/'out')
            self.assertFalse(backend.calls)
            self.assertEqual(result['stopping']['reason'], 'insufficient_models_for_frozen_settings')

    def test_tampered_feature_hash_and_invalid_shapes_stop_before_output(self):
        import numpy as np
        matrix, names = self.inputs(30)
        invalid = [(matrix, names, ['C1']*30, dict(expected_feature_sha256='0'*64)),
                   (matrix.astype(np.float16), names, ['C1']*30, {}),
                   (matrix[:, :2], names, ['C1']*30, {}),
                   (matrix, names[:-1], ['C1']*30, {}),
                   (matrix, names, ['C1']*29, {}),
                   (matrix, names[:-1]+[names[0]], ['C1']*30, {}),
                   (matrix*float('nan'), names, ['C1']*30, {}),
                   (matrix-1, names, ['C1']*30, {})]
        for features, models, primary, kwargs in invalid:
            with self.subTest(shape=features.shape, kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                backend = SavedFixtureBackend(lambda count, call:[])
                output = Path(tmp)/'out'
                with self.assertRaises(ValueError):self.invoke(backend, features, models, primary, output, **kwargs)
                self.assertFalse(output.exists())
                self.assertFalse(backend.calls)

    def test_native_result_tamper_fails_without_modifying_input(self):
        import numpy as np
        matrix, names = self.inputs(30)
        original = matrix.copy()
        def wrong_shape(result, directory):
            np.savez_compressed(directory/'leaf-seed-42.npz', labels=np.zeros(29, dtype=int), probabilities=np.ones(29))
        for tamper in (lambda result, directory:result['models'].reverse(), wrong_shape,
                       lambda result, directory:result.update(seed_agreement={})):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as tmp:
                backend = SavedFixtureBackend(lambda count, call:[0]*15+[1]*15, tamper=tamper, mutate=True)
                with self.assertRaises(ValueError):self.invoke(backend, matrix, names, ['C1']*30, Path(tmp)/'out')
                np.testing.assert_array_equal(matrix, original)

    def test_prepared_rows_reject_different_selection(self):
        import numpy as np
        prepared = adapter._PreparedRows(np.ones((2, 3), np.float32), np.array([2, 4]))
        with self.assertRaises(ValueError):prepared[np.array([4, 2])]
        self.assertIs(prepared[np.array([2, 4])], prepared.matrix)

    def test_existing_output_and_changed_mode_or_workers_never_fit(self):
        matrix, names = self.inputs(30)
        backend = SavedFixtureBackend(lambda count, call:[])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):self.invoke(backend, matrix, names, ['C1']*30, Path(tmp))
            for kwargs in (dict(mode='changed'), dict(workers=8), dict(workers=True)):
                with self.assertRaises(ValueError):self.invoke(backend, matrix, names, ['C1']*30, Path(tmp)/'out', **kwargs)
            self.assertFalse(backend.calls)


if __name__ == '__main__':unittest.main()
