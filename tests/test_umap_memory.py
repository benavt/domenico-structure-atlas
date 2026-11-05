"""Static arrays and fake native calls only; never fit real UMAP/NNDescent."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import umap_memory


class GateTests(unittest.TestCase):
    def test_cancelled_check_precedes_optional_imports_and_estimator_work(self):
        estimator = Mock()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            umap_memory.fit_transform(estimator, object(), check=Mock(side_effect=RuntimeError('cancelled')))
        estimator.get_params.assert_not_called()
        estimator.fit_transform.assert_not_called()

    def test_explicit_check_and_fresh_evidence_are_required(self):
        with self.assertRaises(TypeError):
            umap_memory.fit_transform(Mock(), object(), check=None)
        with self.assertRaises(ValueError):
            umap_memory.fit_transform(Mock(), object(), check=lambda: None, evidence={'old': 'proof'})


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'Optional NumPy for static array fixtures')
class StaticNativeTests(unittest.TestCase):
    def fixture(self, count=4096):
        import numpy as np
        matrix = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
        matrix.flags.writeable = False
        events = []

        class Search:
            pass

        class Estimator:
            def __init__(self):
                self.parameters = dict(metric='euclidean', unique=False, n_jobs=1, low_memory=True,
                    angular_rp_forest=False, densmap=False, output_dens=False,
                    transform_mode='embedding', force_approximation_algorithm=False,
                    metric_kwds=None, random_state=42, precomputed_knn=(None, None, None),
                    n_neighbors=3, verbose=False)
                self.received = []
                self.fit_parameters = []
                self.fail = False

            def get_params(self, deep=False):
                return self.parameters.copy()

            def set_params(self, **parameters):
                self.parameters.update(parameters)
                events.append(('parameters', parameters))
                return self

            def fit_transform(self, data):
                self.received.append(data)
                self.fit_parameters.append(self.parameters.copy())
                if self.fail:
                    raise RuntimeError('fake fit stopped')
                rng = self.parameters['random_state']
                if isinstance(rng, np.random.RandomState):
                    events.append(('embedding_rng', rng))
                    return rng.randint(0, 100, size=(2, 2))
                return 'ordinary-result'

        estimator = Estimator()
        thread_count = [6]

        def set_threads(count):
            events.append(('threads', count))
            thread_count[0] = count

        def nearest(data, neighbors, metric, metric_kwds, angular, rng, **kwargs):
            self.assertTrue(np.shares_memory(data, matrix))
            self.assertEqual(thread_count[0], 1)
            self.assertFalse(data.flags.writeable)
            self.assertEqual((neighbors, metric, metric_kwds, angular), (3, 'euclidean', {}, False))
            self.assertEqual(kwargs, dict(low_memory=True, use_pynndescent=True, n_jobs=1, verbose=False))
            events.append(('neighbors_rng', rng))
            rng.randint(0, 100, size=7)  # Fake native consumption; not an actual neighbor search.
            shape = (len(data), neighbors)
            return np.zeros(shape, np.int32), np.zeros(shape, np.float32), Search()

        runtime = dict(estimator_class=Estimator, search_class=Search, nearest_neighbors=nearest,
            check_random_state=lambda seed: np.random.RandomState(seed),
            get_num_threads=lambda: thread_count[0], set_num_threads=set_threads,
            umap_version='0.5.11', pynndescent_version='0.5.13', umap_source=dict(sha256='a' * 64))
        return np, matrix, estimator, runtime, events, thread_count

    def test_same_advanced_rng_exact_k_input_identity_and_parameter_thread_restoration(self):
        np, matrix, estimator, runtime, events, threads = self.fixture()
        before = hashlib.sha256(matrix.tobytes()).hexdigest()
        original = estimator.get_params()
        evidence = {}
        with patch.object(umap_memory, '_native_runtime', return_value=runtime):
            result = umap_memory.fit_transform(estimator, matrix, check=lambda: None, evidence=evidence)
        expected = np.random.RandomState(42)
        expected.randint(0, 100, size=7)
        np.testing.assert_array_equal(result, expected.randint(0, 100, size=(2, 2)))
        states = [item[1] for item in events if item[0] in ('neighbors_rng', 'embedding_rng')]
        self.assertIs(states[0], states[1])
        self.assertIs(estimator.received[0], matrix)
        self.assertEqual(estimator.get_params(), original)
        self.assertEqual(threads[0], 6)
        self.assertEqual(hashlib.sha256(matrix.tobytes()).hexdigest(), before)
        self.assertFalse(matrix.flags.writeable)
        self.assertEqual(evidence['route'], 'native-precomputed-neighbors')
        self.assertEqual(evidence['status'], 'complete')
        self.assertFalse(evidence['actual_memory_saving_measured'])
        self.assertFalse(evidence['input_hash_verified_by_helper'])
        self.assertNotIn('path', str(evidence))

    def test_small_exact_boundary_and_unsupported_cases_delegate_without_rebinding(self):
        changes = [dict(count=4095), dict(unique=True), dict(metric='cosine'), dict(n_jobs=2),
                   dict(low_memory=False), dict(random_state=None), dict(random_state=True),
                   dict(metric_kwds={'p': 2}), dict(n_neighbors=4096),
                   dict(precomputed_knn=('caller', 'owned')), dict(force_approximation_algorithm=True)]
        for change in changes:
            with self.subTest(change=change):
                _, matrix, estimator, runtime, events, threads = self.fixture(change.get('count', 4096))
                estimator.parameters.update({key: value for key, value in change.items() if key != 'count'})
                original = estimator.get_params()
                with patch.object(umap_memory, '_native_runtime') as load:
                    self.assertEqual(umap_memory.fit_transform(estimator, matrix, check=lambda: None), 'ordinary-result')
                load.assert_not_called()
                self.assertEqual(estimator.parameters, original)
                self.assertEqual(events, [])
                self.assertEqual(threads[0], 6)
                self.assertIs(estimator.received[0], matrix)

    def test_wrong_dtype_layout_inputs_stay_on_ordinary_route(self):
        np, matrix, estimator, _, _, _ = self.fixture()
        for data in (matrix.astype(np.float64), matrix[:, ::-1], matrix.tolist()):
            with self.subTest(kind=type(data).__name__):
                with patch.object(umap_memory, '_native_runtime') as load:
                    self.assertEqual(umap_memory.fit_transform(estimator, data, check=lambda: None), 'ordinary-result')
                load.assert_not_called()
                self.assertIs(estimator.received[-1], data)

    def test_writable_workspace_uses_zero_copy_readonly_view_without_changing_source(self):
        np, matrix, estimator, runtime, _, _ = self.fixture()
        matrix.flags.writeable = True
        before = hashlib.sha256(matrix.tobytes()).hexdigest()
        evidence = {}
        with patch.object(umap_memory, '_native_runtime', return_value=runtime):
            umap_memory.fit_transform(estimator, matrix, check=lambda: None, evidence=evidence)
        received = estimator.received[0]
        self.assertIsNot(received, matrix)
        self.assertTrue(np.shares_memory(received, matrix))
        self.assertFalse(received.flags.writeable)
        self.assertTrue(matrix.flags.writeable)
        self.assertEqual(hashlib.sha256(matrix.tobytes()).hexdigest(), before)
        self.assertTrue(evidence['readonly_view_created'])

    def test_unaudited_runtime_delegates_without_neighbor_work(self):
        _, matrix, estimator, runtime, events, _ = self.fixture()
        runtime['umap_version'] = 'different-version'
        runtime['nearest_neighbors'] = Mock(side_effect=AssertionError('must not compute'))
        with patch.object(umap_memory, '_native_runtime', return_value=runtime):
            self.assertEqual(umap_memory.fit_transform(estimator, matrix, check=lambda: None), 'ordinary-result')
        runtime['nearest_neighbors'].assert_not_called()
        self.assertEqual(events, [])

    def test_failed_fit_restores_state_and_never_retries_ordinary_path(self):
        _, matrix, estimator, runtime, _, threads = self.fixture()
        original = estimator.get_params()
        estimator.fail = True
        evidence = {}
        with patch.object(umap_memory, '_native_runtime', return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, 'fake fit stopped'):
                umap_memory.fit_transform(estimator, matrix, check=lambda: None, evidence=evidence)
        self.assertEqual(len(estimator.received), 1)
        self.assertEqual(estimator.parameters, original)
        self.assertEqual(threads[0], 6)
        self.assertEqual(evidence['status'], 'failed')

    def test_cancel_after_native_neighbors_preserves_parameters_and_stops_before_fit(self):
        _, matrix, estimator, runtime, _, threads = self.fixture()
        original = estimator.get_params()
        calls = [0]

        def check():
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError('stop after neighbors')

        with patch.object(umap_memory, '_native_runtime', return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, 'stop after neighbors'):
                umap_memory.fit_transform(estimator, matrix, check=check)
        self.assertEqual(estimator.received, [])
        self.assertEqual(estimator.parameters, original)
        self.assertEqual(threads[0], 6)

    def test_invalid_neighbor_shape_fails_closed_instead_of_triggering_native_fallback(self):
        np, matrix, estimator, runtime, _, threads = self.fixture()
        runtime['nearest_neighbors'] = Mock(return_value=(np.zeros((2, 2), np.int32),
                                                         np.zeros((2, 2), np.float32), object()))
        with patch.object(umap_memory, '_native_runtime', return_value=runtime):
            with self.assertRaisesRegex(ValueError, 'refuse fallback'):
                umap_memory.fit_transform(estimator, matrix, check=lambda: None)
        self.assertEqual(estimator.received, [])
        self.assertEqual(threads[0], 6)

    def test_readonly_input_cannot_be_mutated_by_fake_neighbor_backend(self):
        _, matrix, estimator, runtime, _, threads = self.fixture()

        def bad_neighbors(data, *args, **kwargs):
            data[0, 0] = 12

        runtime['nearest_neighbors'] = bad_neighbors
        with patch.object(umap_memory, '_native_runtime', return_value=runtime):
            with self.assertRaises(ValueError):
                umap_memory.fit_transform(estimator, matrix, check=lambda: None)
        self.assertEqual(estimator.received, [])
        self.assertEqual(threads[0], 6)


if __name__ == '__main__':
    unittest.main()
