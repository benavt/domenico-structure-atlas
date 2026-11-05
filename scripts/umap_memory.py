"""Avoid UMAP's retained identity-index copy through its public k-NN API.

This helper changes allocation only for the audited large dense Euclidean path.
It is not a checkpoint, approximation change, or permission to execute a fit.
Callers own exact runtime/input binding and scientific execution approval.
"""
from importlib import metadata
import hashlib
import inspect
import json


AUDITED_UMAP_VERSION = '0.5.11'
AUDITED_PYNNDESCENT_VERSION = '0.5.13'


def random_state_sha256(state):
    name, keys, position, gaussian, cached = state.get_state()
    value = [name, keys.tolist(), int(position), int(gaussian), float(cached)]
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _native_runtime():
    """Load optional numerical dependencies only when a caller requests a fit."""
    import numba
    from sklearn.utils import check_random_state
    from umap import UMAP
    from umap.umap_ import nearest_neighbors
    from pynndescent import NNDescent

    source = inspect.getsourcefile(nearest_neighbors)
    source_evidence = None
    if source is not None:
        with open(source, 'rb') as stream:
            source_evidence = dict(sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
    return dict(
        estimator_class=UMAP,
        search_class=NNDescent,
        nearest_neighbors=nearest_neighbors,
        check_random_state=check_random_state,
        get_num_threads=numba.get_num_threads,
        set_num_threads=numba.set_num_threads,
        umap_version=metadata.version('umap-learn'),
        pynndescent_version=metadata.version('pynndescent'),
        umap_source=source_evidence,
    )


def _unsupported_reason(parameters, data, np):
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        return 'input is not a dense NumPy matrix'
    if data.dtype != np.float32 or not data.flags.c_contiguous:
        return 'input is not prepared C-contiguous float32'
    if data.shape[0] < 4096:
        return 'preserve the native small-data exact branch'
    if data.shape[1] == 0:
        return 'input has no features'
    required = dict(metric='euclidean', unique=False, n_jobs=1, low_memory=True,
                    angular_rp_forest=False, densmap=False, output_dens=False,
                    transform_mode='embedding', force_approximation_algorithm=False)
    for key, expected in required.items():
        value = parameters.get(key)
        if type(value) is not type(expected) or value != expected:
            return 'unsupported parameter: ' + key
    metric_arguments = parameters.get('metric_kwds')
    if metric_arguments is not None and (not isinstance(metric_arguments, dict) or metric_arguments):
        return 'unsupported metric arguments'
    neighbors = parameters.get('n_neighbors')
    if type(neighbors) is not int or not 2 <= neighbors < data.shape[0]:
        return 'unsupported neighbor count'
    seed = parameters.get('random_state')
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        return 'an explicit integer seed is required'
    if not 0 <= seed <= 2**32 - 1:
        return 'seed is outside RandomState integer bounds'
    existing = parameters.get('precomputed_knn')
    if not isinstance(existing, tuple) or len(existing) not in (2, 3) or any(
            item is not None for item in existing):
        return 'preserve caller-supplied precomputed neighbors'
    return None


def fit_transform(estimator, prepared_F32_C_data, *, check, evidence=None):
    """Return ordinary UMAP embeddings, optionally recording allocation-route proof.

    ``evidence`` may be an empty caller-owned dict; it receives route/status,
    versions and the UMAP source digest, never input values or source paths.
    The reviewed route accepts prepared C float32 matrices, N>=4096 and seeded
    single-threaded Euclidean UMAP0.5.11/PyNNDescent0.5.13. All unsupported cases
    call the estimator's ordinary ``fit_transform(data)`` without rebinding it.

    Native neighbors consume the same RandomState subsequently passed to UMAP.
    Its original public parameters and the caller's Numba thread mask are
    restored even on failure. There is no automatic fallback after work starts.
    A writable prepared workspace is exposed through a zero-copy read-only view;
    the caller's original flags and values remain untouched. The fitted estimator
    retains the neighbor data view, so its backing workspace must outlive any
    subsequent estimator use. Callers consuming only embeddings may release it
    after this function returns.

    This removes the retained neighbor-input copy; UMAP's graph/embedding copies
    and RP-tree allocations remain. The helper never copies or hashes the full
    feature matrix: the caller must bind and verify its immutable source bytes.
    """
    if not callable(check):
        raise TypeError('An explicit check callback is required')
    if evidence is not None and (not isinstance(evidence, dict) or evidence):
        raise ValueError('Evidence must be an empty caller-owned dictionary')
    check()
    import numpy as np

    proof = evidence if evidence is not None else {}
    proof.update(schema_version=1, status='started',
                 performance_claim='unmeasured; removes one retained identity-index copy only')
    parameters = estimator.get_params(deep=False)
    reason = _unsupported_reason(parameters, prepared_F32_C_data, np)
    runtime = None
    if reason is None:
        runtime = _native_runtime()
        proof.update(umap_version=runtime['umap_version'],
                     pynndescent_version=runtime['pynndescent_version'],
                     umap_source=runtime['umap_source'])
        if (runtime['umap_version'] != AUDITED_UMAP_VERSION
                or runtime['pynndescent_version'] != AUDITED_PYNNDESCENT_VERSION):
            reason = 'preserve an unaudited native version'
        elif not isinstance(estimator, runtime['estimator_class']):
            reason = 'preserve a non-native estimator'
    if reason is not None:
        proof.update(route='ordinary', reason=reason)
        try:
            result = estimator.fit_transform(prepared_F32_C_data)
            check()
        except BaseException as error:
            proof.update(status='failed', failure_type=type(error).__name__)
            raise
        proof['status'] = 'complete'
        return result

    fit_data = prepared_F32_C_data
    view_created = bool(prepared_F32_C_data.flags.writeable)
    if view_created:
        fit_data = prepared_F32_C_data.view()
        fit_data.flags.writeable = False
    proof.update(route='native-precomputed-neighbors', models=int(prepared_F32_C_data.shape[0]),
                 features=int(prepared_F32_C_data.shape[1]), seed=int(parameters['random_state']),
                 neighbors=parameters['n_neighbors'], input_readonly=True,
                 input_c_contiguous=True, input_dtype='float32',
                 source_input_writable=view_created, readonly_view_created=view_created,
                 avoided_retained_copy_bytes=int(prepared_F32_C_data.nbytes),
                 input_hash_verified_by_helper=False,
                 actual_memory_saving_measured=False)
    original_threads = runtime['get_num_threads']()
    original_parameters = {key: parameters[key] for key in ('random_state', 'precomputed_knn')}
    parameters_attempted = False
    failure = None
    try:
        runtime['set_num_threads'](1)
        check()
        state = runtime['check_random_state'](parameters['random_state'])
        proof['initial_random_state_sha256'] = random_state_sha256(state)
        neighbors = runtime['nearest_neighbors'](
            fit_data, parameters['n_neighbors'], 'euclidean', {}, False,
            state, low_memory=True, use_pynndescent=True, n_jobs=1,
            verbose=parameters.get('verbose', False),
        )
        check()
        if not isinstance(neighbors, tuple) or len(neighbors) != 3:
            raise ValueError('Native neighbors must return the complete search-index triple')
        proof['after_neighbors_random_state_sha256'] = random_state_sha256(state)
        indices, distances, search = neighbors
        expected_shape = (len(prepared_F32_C_data), parameters['n_neighbors'])
        if (not isinstance(indices, np.ndarray) or not isinstance(distances, np.ndarray)
                or indices.shape != expected_shape or distances.shape != expected_shape
                or not np.issubdtype(indices.dtype, np.integer)
                or distances.dtype != np.float32
                or not isinstance(search, runtime['search_class'])):
            raise ValueError('Native neighbor contract differs; refuse fallback recomputation')
        parameters_attempted = True
        estimator.set_params(random_state=state, precomputed_knn=neighbors)
        result = estimator.fit_transform(fit_data)
        proof['final_random_state_sha256'] = random_state_sha256(state)
        check()
        proof['status'] = 'complete'
        return result
    except BaseException as error:
        failure = error
        proof.update(status='failed', failure_type=type(error).__name__)
        raise
    finally:
        cleanup_errors = []
        if parameters_attempted:
            try:
                estimator.set_params(**original_parameters)
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            runtime['set_num_threads'](original_threads)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            proof.update(status='failed', restoration_failed=True)
            if failure is not None:
                for error in cleanup_errors:
                    failure.add_note('UMAP helper restoration failed: ' + repr(error))
            else:
                raise cleanup_errors[0]
