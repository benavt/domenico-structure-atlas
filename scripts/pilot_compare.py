"""Compare two CLOSED, matched 64-model CPU pilots; never fit a model.

A frozen contract binds the cohort, raw F16 features, protocol, source and
conservative numerical tolerances. A later saved-input manifest binds the actual
closed receipts. Small projection geometry diagnostics cannot certify full-data
clustering stability or a hardware capacity/performance limit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile


STAGES = ('pca', 'umap', 'tsne', 'labels', 'density0')
DEFAULT_THRESHOLDS = dict(pca_relative_pairwise_error=1e-5,
    pca_aligned_normalized_residual=1e-5, pca_scale_ratio_error=1e-5,
    coordinate_rtol=1e-5, coordinate_atol=1e-6, neighbor_k=10)
MAXIMUM_BLOCK_BYTES = 16 * 1024**2
MAXIMUM_JSON_BYTES = 16 * 1024**2


def need(value, message):
    if not value:
        raise ValueError(message)


def _pilot():
    source = str(Path(__file__).absolute().parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    import mimas_pilot
    return mimas_pilot


def _content(entry):
    need(isinstance(entry, dict) and type(entry.get('bytes')) is int
         and entry['bytes'] >= 0 and isinstance(entry.get('sha256'), str)
         and len(entry['sha256']) == 64
         and all(c in '0123456789abcdef' for c in entry['sha256']),
         'Exact bytes/SHA-256 descriptor required')
    return entry['bytes'], entry['sha256']


def _bound(entry, *, locations=None, check=lambda: None):
    original = _content(entry)
    replacement = (locations or {}).get(entry.get('path'), entry)
    need(_content(replacement) == original, 'Relocation changed artifact content identity')
    return _pilot().bound(replacement, check)


def _read(entry, *, locations=None, check=lambda: None):
    need(entry['bytes'] <= MAXIMUM_JSON_BYTES, 'Saved JSON exceeds the bounded metadata allowance')
    return json.loads(_bound(entry, locations=locations, check=check).read_text())


def _compact(value):
    return _pilot().compact_sha(value)


def _same(actual, expected, message):
    need(actual == expected, message)


def _protocol(plan, expected, locations, check):
    pilot = _pilot()
    need(plan.get('phase') == 'pilot64' and plan.get('target') == expected['target']
         and plan.get('stages') == list(STAGES)
         and plan.get('maximum_projection_fits') == 9
         and plan.get('maximum_hdbscan_extractions') == 6,
         'A complete matched 64-model pilot plan is required, not a full-cohort run')
    spec = plan['inputs']
    for key in ('model_count', 'construct_residues', 'retained_reference_positions',
                'feature_sha256', 'pair_mask_sha256', 'model_order_sha256'):
        _same(spec.get(key), expected[key], 'Pilot input identity differs: ' + key)
    _same(_content(spec['features']), _content(expected['features']), 'F16 feature-container identity differs')
    native = plan['native']
    _same(native.get('library_commit'), pilot.LIBRARY_COMMIT, 'Native library pin differs')
    provider = native['density_source']
    _same((provider.get('commit'), provider.get('sha256')),
          (pilot.DENSITY_COMMIT, pilot.DENSITY_SHA256), 'Density source pin differs')
    config = _read(native['cluster_config'], locations=locations, check=check)
    _same(config, dict(schema_version=1, settings=pilot.DEFAULT_COMPONENT_SETTINGS),
          'Primary clustering settings differ from the frozen native defaults')
    workers = plan.get('workers', 1)
    need(type(workers) is int and 1 <= workers <= 4, 'Explicit bounded worker count required')
    return dict(library_commit=pilot.LIBRARY_COMMIT,
                density_commit=pilot.DENSITY_COMMIT, density_source_sha256=pilot.DENSITY_SHA256,
                cluster_config=config, workers=workers)


def _members(plan, expected, locations, check):
    registry = _read(plan['inputs']['members'], locations=locations, check=check)
    rows = registry['members']
    need(isinstance(rows, list) and len(rows) == 64, 'Exactly 64 saved member rows required')
    names = [row['model'] for row in rows]
    paths = [row['local_path'] for row in rows]
    need(len(set(names)) == 64 and len(set(paths)) == 64
         and all(isinstance(name, str) and name for name in names)
         and all(isinstance(path, str) and Path(path).is_absolute() for path in paths),
         'Unique complete model/path order required')
    _same(hashlib.sha256('\n'.join(names).encode()).hexdigest(), expected['model_order_sha256'],
          'Saved member order differs')
    for row in rows:
        need(isinstance(row.get('original_sha256'), str) and len(row['original_sha256']) == 64
             and all(c in '0123456789abcdef' for c in row['original_sha256']) and 'flags' in row,
             'Original hash/flags are missing from saved member metadata')
    # Native model files are deliberately NOT opened. Their accepted identities
    # are cross-checked from the already verified, hash-bound member registries.
    identity = [dict(model=row['model'], original_sha256=row['original_sha256'], flags=row['flags'])
                for row in rows]
    return names, paths, identity


def _sidecar(entry, key, names, paths, flags, locations, check):
    import numpy as np
    path = _bound(entry, locations=locations, check=check)
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        need(len(items) == len({item.filename for item in items})
             and sum(item.file_size for item in items) <= MAXIMUM_BLOCK_BYTES,
             'Duplicate or oversized saved sidecar')
    with np.load(path, allow_pickle=False) as saved:
        _same(saved['filenames'].tolist(), names, 'Sidecar model order differs')
        _same(saved['filepaths'].tolist(), paths, 'Sidecar local-path order differs')
        _same([json.loads(value) for value in saved['flags_json'].tolist()], flags,
              'Sidecar model flags differ')
        value = saved[key].copy()
    if key == 'embeddings':
        need(value.shape == (64, 2) and value.dtype.kind == 'f' and np.isfinite(value).all(),
             'A finite saved 64x2 display projection is required')
    else:
        need(value.shape == (64,) and value.dtype.kind in 'US'
             and all(isinstance(item, str) and item for item in value.tolist()),
             'A complete nonempty saved 64-model partition is required')
    return value


def _source_proof(receipt, plan):
    pilot = _pilot()
    _same(receipt.get('native_library_commit'), pilot.LIBRARY_COMMIT, 'Receipt library source differs')
    sources = receipt.get('native_sources', {})
    _same(set(sources), set(pilot.LIBRARY_SOURCE_HASHES), 'Complete native source proof required')
    for name, sha in pilot.LIBRARY_SOURCE_HASHES.items():
        _same(sources[name].get('sha256'), sha, 'Native source bytes differ: ' + name)
    module = receipt['module_source']
    matches = [item for item in plan['frozen_files']
               if Path(item['path']).name == 'mimas_pilot.py' and _content(item) == _content(module)]
    need(len(matches) == 1, 'Pilot adapter source is not bound by its execution plan')
    return module['sha256']


def _load_host(host, expected, check):
    locations = host.get('artifact_locations', {})
    need(isinstance(locations, dict) and all(isinstance(path, str) and Path(path).is_absolute()
         for path in locations), 'Explicit absolute artifact relocation map required')
    plan = _read(host['plan'], locations=locations, check=check)
    protocol = _protocol(plan, expected, locations, check)
    names, paths, identity = _members(plan, expected, locations, check)
    flags = [row['flags'] for row in identity]
    need(set(host['receipts']) == set(STAGES), 'All five closed stage receipts are required')
    arrays, receipts, adapter_sources = {}, {}, set()
    fingerprint = _compact(dict(target=plan['target'], inputs=plan['inputs'],
                                native=plan['native'], workers=plan.get('workers', 1)))
    for stage in STAGES:
        check()
        receipt = _read(host['receipts'][stage], locations=locations, check=check)
        need(receipt.get('schema_version') == 1 and receipt.get('stage') == stage
             and receipt.get('status') == 'diagnostic_not_full_clustering'
             and receipt.get('phase_complete') is True
             and receipt.get('full_clustering_complete') is False,
             'A missing, failed or incomplete pilot stage cannot pass: ' + stage)
        _same(receipt.get('target'), expected['target'], 'Receipt target differs')
        _same(receipt.get('inputs'), plan['inputs'], 'Receipt input descriptors differ from its plan')
        _same(receipt.get('input_fingerprint'), fingerprint, 'Receipt local protocol fingerprint differs')
        _same(receipt.get('models'), names, 'Receipt cohort/order differs')
        _same(receipt.get('local_paths'), paths, 'Receipt local paths differ')
        _same(receipt.get('model_flags'), flags, 'Receipt flags differ')
        _same(receipt.get('input_integrity'), dict(file_sha256_verified_before_map=True,
            raw_float16_sha256_verified=True, readonly_mapping=True, identities_unchanged_after_phase=True),
            'Receipt lacks the completed raw-input integrity proof')
        need(all(receipt.get(key) == 0 for key in ('coordinate_parses', 'scoring_calls', 'molecular_exports')),
             'Pilot receipt contains work outside the matched distance-only protocol')
        adapter_sources.add(_source_proof(receipt, plan))
        outputs = receipt['outputs']
        need(isinstance(outputs, list) and outputs and len({item['path'] for item in outputs}) == len(outputs),
             'Complete unique bound stage output list required')
        for item in outputs:
            _bound(item, locations=locations, check=check)
        fits = receipt['scientific_fits']
        expected_fits = dict(PCA=0, UMAP=0, TSNE=0, HDBSCAN=0)
        if stage in ('pca', 'umap', 'tsne'):
            expected_fits.update(PCA=1 if stage in ('pca', 'tsne') else 0,
                                 UMAP=1 if stage == 'umap' else 0, TSNE=1 if stage == 'tsne' else 0)
            params = (dict(n_components=2, random_seed=42) if stage == 'pca' else
                      dict(n_neighbors=15, min_dist=.1, n_components=2, verbose=False, random_seed=42)
                      if stage == 'umap' else
                      dict(perplexity=30., implementation='python', verbose=False, random_seed=42))
            _same(receipt.get('parameters'), params, 'Saved reducer parameters/seed differ')
            key, entry = 'embeddings', receipt['projection']
        else:
            key, entry = 'cluster_labels', receipt['labels']
            if stage == 'labels':
                _same(receipt.get('repeat_UMAP_fits_maximum'), 2, 'Primary repeat budget differs')
                selections = [item for item in outputs if Path(item['path']).name == 'selection.json']
                need(len(selections) == 1, 'Primary selection proof missing')
                selection = _read(selections[0], locations=locations, check=check)
                repeats = selection.get('repeat_graphs')
                need(isinstance(repeats, list) and (not repeats or
                     len(repeats) == 2 and {row['seed'] for row in repeats} == {7, 123}),
                     'Primary repeat seeds differ')
                need(selection.get('point_count') == 64
                     and selection.get('feature_input', {}).get('sha256') == expected['feature_sha256'],
                     'Primary selection feature/cohort proof differs')
                expected_fits['UMAP'] = len(repeats)
                # Native data-dependent branches may use zero or two repeat
                # fits. That outcome is not a different requested protocol.
                protocol['primary_repeat_seed_budget'] = [7, 123]
            else:
                need(receipt.get('full_recursive_HDBSCAN') is False
                     and receipt.get('default_cluster_labels') is False,
                     'Initial density diagnostics must not claim full/default clustering')
                need(receipt['density_analysis'] in outputs, 'Density analysis is outside bound outputs')
                analysis = _read(receipt['density_analysis'], locations=locations, check=check)
                expected_settings = dict(seeds=[42, 7, 123], n_neighbors=30, min_dist=0.,
                    min_cluster_size=15, min_samples=11, workers=protocol['workers'], dimension=10,
                    metric='euclidean', primary_extraction='leaf', primary_seed=42)
                need(analysis.get('models') == names and analysis.get('mode') == 'global'
                     and analysis.get('selected_fits') == 1 and analysis.get('UMAP_fits') == 3
                     and analysis.get('HDBSCAN_extractions') == 6
                     and analysis.get('stopping', {}).get('reason') == 'single_global_fit',
                     'Initial density cohort/mode/fit proof differs')
                _same(analysis.get('inference_settings'), expected_settings, 'Density settings/seeds differ')
                provider = analysis.get('native_source', {})
                _same((provider.get('commit'), provider.get('sha256')),
                      (protocol['density_commit'], protocol['density_source_sha256']), 'Density native proof differs')
                proof = analysis.get('feature_proof', {})
                need(proof.get('dtype') == 'float32' and proof.get('stored_dtype') == 'float16'
                     and proof.get('fit_dtype') == 'float32'
                     and proof.get('retained_reference_positions') == expected['retained_reference_positions']
                     and proof.get('shape') == [64, 399171], 'Density precision/core/shape proof differs')
                _same(proof.get('model_order_sha256'), _compact(names), 'Density feature model order differs')
                _same(analysis.get('primary_input_labels'), arrays['labels'].tolist(),
                      'Density input partition differs from the saved primary labels')
                _same(proof.get('primary_partition_sha256'), _compact(arrays['labels'].tolist()),
                      'Density primary-partition proof differs')
                protocol['density_feature_sha256'] = proof.get('feature_sha256')
                protocol['density_scaling'] = proof.get('scaling')
                _same(protocol['density_scaling'],
                      'Raw saved Cα distances cast to float32; no standardization or feature PCA.',
                      'Density feature scaling differs')
                selections = analysis.get('selections', [])
                need(len(selections) == 1 and selections[0].get('selected_indices') == list(range(64))
                     and selections[0].get('models') == names
                     and selections[0].get('feature_sha256') == proof.get('feature_sha256'),
                     'Density selection must cover the identical complete 64-model feature matrix')
                diagnostics = selections[0].get('diagnostics', {})
                need(diagnostics.get('dimension') == 10
                     and diagnostics.get('settings') == {key:expected_settings[key] for key in
                        ('seeds', 'n_neighbors', 'min_dist', 'min_cluster_size', 'min_samples', 'workers')},
                     'Density selected-fit parameter proof differs')
                fit_rows = diagnostics.get('fits', [])
                need(len(fit_rows) == 6
                     and {(row.get('seed'), row.get('method')) for row in fit_rows}
                     == {(seed, method) for seed in (42, 7, 123) for method in ('leaf', 'eom')},
                     'All six native density seed/extraction diagnostics are required')
                receipts['density_analysis'] = analysis
                expected_fits.update(UMAP=3, HDBSCAN=6)
        _same(fits, expected_fits, 'Actual stage fit accounting differs: ' + stage)
        need(entry in outputs, 'Saved sidecar is outside bound stage outputs')
        arrays[stage] = _sidecar(entry, key, names, paths, flags, locations, check)
        receipts[stage] = receipt
    need(len(adapter_sources) == 1, 'Pilot adapter changed between stages')
    protocol['adapter_source_sha256'] = next(iter(adapter_sources))
    _same(receipts['density_analysis'].get('density'), arrays['density0'].tolist(),
          'Density analysis and saved labels differ')
    density = arrays['density0'].tolist()
    _same(receipts['density_analysis'].get('structural_clusters'),
          len({label for label in density if not label.endswith('.unassigned')}),
          'Saved density structural count differs')
    _same(receipts['density_analysis'].get('unassigned_models'),
          sum(label.endswith('.unassigned') for label in density), 'Saved density noise count differs')
    runtime = _read(host['runtime'], locations=locations, check=check)
    return dict(plan=plan, protocol=protocol, names=names, identity=identity,
                arrays=arrays, receipts=receipts, runtime=runtime)


def _matrix_distances(matrix, expected_raw_sha256, *, check=lambda: None,
                      block_bytes=MAXIMUM_BLOCK_BYTES):
    import numpy as np
    n = len(matrix)
    need(type(block_bytes) is int and n * 8 <= block_bytes <= MAXIMUM_BLOCK_BYTES,
         'Bounded diagnostic block allowance required')
    raw = hashlib.sha256()
    rows = max(1, block_bytes // (matrix.shape[1] * matrix.dtype.itemsize))
    for start in range(0, n, rows):
        check()
        raw.update(memoryview(matrix[start:start + rows]).cast('B'))
    _same(raw.hexdigest(), expected_raw_sha256, 'Raw F16 feature values/order differ')
    converted = hashlib.sha256()
    with np.nditer(matrix, flags=['external_loop', 'buffered'], op_flags=['readonly'],
                   op_dtypes=[np.float32], casting='safe', order='C',
                   buffersize=max(1, block_bytes // 4)) as iterator:
        for block in iterator:
            check()
            converted.update(memoryview(block).cast('B'))
    result = np.zeros((n, n), dtype=np.float64)
    width = max(1, block_bytes // (n * 8))
    for start in range(0, matrix.shape[1], width):
        check()
        block = np.asarray(matrix[:, start:start + width], dtype=np.float64, order='C')
        need(np.isfinite(block).all() and not np.any(block < 0), 'Invalid saved raw distances')
        norms = np.einsum('ij,ij->i', block, block)
        result += norms[:, None] + norms[None, :] - 2 * (block @ block.T)
    np.maximum(result, 0, out=result)
    np.fill_diagonal(result, 0)
    np.sqrt(result, out=result)
    return result, dict(raw_float16_sha256=raw.hexdigest(), float32_sha256=converted.hexdigest(),
                        maximum_feature_block_bytes=block_bytes,
                        distance_metric='Euclidean over exact F16 values accumulated in bounded float64 blocks')


def feature_distances(entry, expected_raw_sha256, *, check=lambda: None, block_bytes=MAXIMUM_BLOCK_BYTES):
    """Diagnostic Euclidean distances in bounded float64 column blocks.

    Raw F16 bytes are verified first in C row order. This is a saved-data
    diagnostic calculation, not a native reducer or an alternative feature fit.
    """
    import numpy as np
    path = _bound(entry, check=check)
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        need(isinstance(matrix, np.ndarray) and matrix.shape == (64, 399171)
             and matrix.dtype == np.float16 and matrix.flags.c_contiguous
             and not matrix.flags.writeable, 'Exact read-only 64x399171 F16 features required')
        return _matrix_distances(matrix, expected_raw_sha256, check=check, block_bytes=block_bytes)
    finally:
        if getattr(matrix, '_mmap', None) is not None:
            matrix._mmap.close()


def _distances(array):
    import numpy as np
    delta = array[:, None, :] - array[None, :, :]
    return np.sqrt(np.einsum('ijk,ijk->ij', delta, delta))


def geometry(reference, candidate, thresholds):
    import numpy as np
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    centered_a, centered_b = a - a.mean(axis=0), b - b.mean(axis=0)
    norm_a, norm_b = float(np.linalg.norm(centered_a)), float(np.linalg.norm(centered_b))
    raw_close = bool(np.allclose(a, b, rtol=thresholds['coordinate_rtol'], atol=thresholds['coordinate_atol']))
    exact = (reference.dtype == candidate.dtype and reference.tobytes() == candidate.tobytes())
    if norm_a == 0 or norm_b == 0:
        return dict(exact_coordinates=exact, raw_allclose=raw_close, rigid_allclose=None,
                    degenerate=True, scale_ratio=None, aligned_normalized_residual=None,
                    relative_pairwise_error=None, scale_adjusted_residual=None)
    u, _, vt = np.linalg.svd(centered_b.T @ centered_a)
    aligned = centered_b @ (u @ vt)  # Reflection allowed; no scale is fitted away.
    da, db = _distances(a), _distances(b)
    triangles = np.triu_indices(len(a), 1)
    base = float(np.sqrt(np.mean(da[triangles] ** 2)))
    best_scale = float(np.sum(aligned * centered_a) / (norm_b ** 2))
    return dict(exact_coordinates=exact, raw_allclose=raw_close,
        rigid_allclose=bool(np.allclose(centered_a, aligned,
            rtol=thresholds['coordinate_rtol'], atol=thresholds['coordinate_atol'])),
        degenerate=False, scale_ratio=norm_b / norm_a,
        aligned_normalized_residual=float(np.linalg.norm(aligned - centered_a) / norm_a),
        relative_pairwise_error=float(np.sqrt(np.mean((db[triangles] - da[triangles]) ** 2)) / base),
        scale_adjusted_residual=float(np.linalg.norm(best_scale * aligned - centered_a) / norm_a))


def neighborhood(reference_distances, a, b, k):
    import numpy as np
    n = len(a)
    need(type(k) is int and 0 < k < n / 2, 'Trustworthiness needs a frozen k below N/2')
    def order(distances):
        values = distances.copy()
        np.fill_diagonal(values, np.inf)
        return np.argsort(values, axis=1, kind='stable')
    input_order = order(reference_distances)
    ranks = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        ranks[i, input_order[i]] = np.arange(1, n + 1)
    orders = [order(_distances(value))[:, :k] for value in (a, b)]
    def trust(selected):
        penalty = 0
        for i in range(n):
            penalty += sum(max(0, int(ranks[i, j]) - k) for j in selected[i])
        return 1 - 2 * penalty / (n * k * (2 * n - 3 * k - 1))
    overlap = sum(len(set(orders[0][i]) & set(orders[1][i])) for i in range(n)) / (n * k)
    first, second = trust(orders[0]), trust(orders[1])
    ties = any(len(set(row)) < len(row) for row in reference_distances)
    return dict(k=k, reference_trustworthiness=first, candidate_trustworthiness=second,
                trustworthiness_absolute_difference=abs(first - second),
                cross_host_neighbor_overlap=overlap, reference_distance_ties=ties,
                tie_policy='Stable model-order tie break; diagnostics only, no acceptance threshold')


def _ari(first, second):
    need(len(first) == len(second), 'Partition lengths differ')
    count = len(first)
    if count < 2:
        return None
    choose = lambda n: n * (n - 1) // 2
    rows, columns, cells = Counter(first), Counter(second), Counter(zip(first, second))
    observed = sum(choose(value) for value in cells.values())
    left = sum(choose(value) for value in rows.values())
    right = sum(choose(value) for value in columns.values())
    expected = left * right / choose(count)
    denominator = .5 * (left + right) - expected
    return 1.0 if denominator == 0 else (observed - expected) / denominator


def partition(first, second, *, density=False):
    a, b = list(first), list(second)
    noise_a = [density and value.endswith('.unassigned') for value in a]
    noise_b = [density and value.endswith('.unassigned') for value in b]
    structural_a = {value for value, noise in zip(a, noise_a) if not noise}
    structural_b = {value for value, noise in zip(b, noise_b) if not noise}
    common = [i for i in range(len(a)) if not noise_a[i] and not noise_b[i]]
    degenerate = len(structural_a) <= 1 or len(structural_b) <= 1
    ari = _ari(a, b)
    noise_exact = noise_a == noise_b
    equivalent = ari is not None and abs(ari - 1) <= 1e-12 and noise_exact
    return dict(adjusted_rand_index=ari, noise_masks_identical=noise_exact,
        noise_agreement=sum(x == y for x, y in zip(noise_a, noise_b)) / len(a),
        reference_noise_models=sum(noise_a), candidate_noise_models=sum(noise_b),
        reference_structural_groups=len(structural_a), candidate_structural_groups=len(structural_b),
        common_assigned_models=len(common), structural_ari=None if degenerate else
            _ari([a[i] for i in common], [b[i] for i in common]),
        partitions_equivalent=equivalent, degenerate=degenerate,
        structural_validation=equivalent and not degenerate,
        status='degenerate_not_structural_validation' if degenerate else
            ('equivalent_partition' if equivalent else 'review_required'))


def _markdown(result):
    def number(value):
        return 'undefined' if value is None else format(value, '.6g')
    lines = ['# Matched 64-model SERCA CPU pilot comparison', '',
        '**Protocol:** matched saved 64-model inputs, native wrapper/provider source and settings. '
        'Runtime environments are reported separately.',
        '**Numerical verdict:** ' + result['status'].replace('_', ' ') + '.', '',
        'Retained canonical positions: **58–951** (894 residues; 399,171 distance features per model).', '',
        'This diagnostic does not establish full-cohort clustering stability or certify hardware capacity.', '',
        '| Stage | Result | Key diagnostics |', '|---|---|---|']
    for stage in STAGES:
        item = result['stages'][stage]
        if stage in ('pca', 'umap', 'tsne'):
            geometry = item['geometry']
            neighbors = item['neighborhood']
            details = ('Rigid residual ' + number(geometry['aligned_normalized_residual'])
                + '; pair-distance error ' + number(geometry['relative_pairwise_error'])
                + '; scale ratio ' + number(geometry['scale_ratio'])
                + '; trustworthiness Pax/Mimas ' + number(neighbors['reference_trustworthiness'])
                + '/' + number(neighbors['candidate_trustworthiness'])
                + '; neighbor overlap ' + number(neighbors['cross_host_neighbor_overlap']))
        else:
            labels = item['partition']
            details = ('ARI ' + number(labels['adjusted_rand_index'])
                + '; noise Pax/Mimas ' + str(labels['reference_noise_models'])
                + '/' + str(labels['candidate_noise_models'])
                + '; identical noise masks ' + str(labels['noise_masks_identical']).lower())
        lines.append('| ' + stage + ' | ' + item['status'].replace('_', ' ') + ' | ' + details + ' |')
    lines += ['', 'PCA accepts centering and orthogonal/reflection alignment while checking scale. '
        'UMAP/t-SNE require raw or rigid-aligned coordinates within the frozen numerical tolerance. '
        'Trustworthiness and neighborhood overlap are informational; they cannot rescue a failed numerical comparison.', '',
        'Label IDs may be permuted. Noise membership must match exactly. '
        'All-noise or one-structural-group outputs do not validate structural clustering.', '',
        'Runtime records differ: **' + ('no' if result['runtime_records_identical'] else 'yes') + '**. '
        'Inspect comparison.json for the bound runtime records and diagnostic values.', '']
    return '\n'.join(lines)


def compare(input_manifest, output, *, check=lambda: None):
    manifest = json.loads(Path(input_manifest).read_text()) if isinstance(input_manifest, (str, Path)) else input_manifest
    need(manifest.get('schema_version') == 1
         and manifest.get('operation') == 'paired-pilot-saved-comparison', 'Saved comparison manifest required')
    contract = _read(manifest['contract'], check=check)
    need(contract.get('schema_version') == 1 and contract.get('operation') == 'paired-pilot-comparison',
         'Frozen matched-pilot comparison contract required')
    _same(contract.get('thresholds'), DEFAULT_THRESHOLDS, 'Unreviewed numerical threshold contract')
    source = _bound(contract['comparator_source'], check=check)
    need(source == Path(__file__).absolute(), 'Frozen comparator source differs from the running source')
    expected = contract['expected']
    pilot = _pilot()
    need(expected.get('target') == 'SERCA_P04191' and expected.get('model_count') == 64
         and expected.get('construct_residues') == 993
         and expected.get('retained_reference_positions') == list(range(57, 951))
         and expected.get('pair_mask_sha256') == pilot.SERCA_PAIR_MASK,
         'Exactly the accepted SERCA 64-model core58–951 is required')
    need(set(manifest['hosts']) == {'pax', 'mimas'}
         and set(contract['host_plans']) == {'pax', 'mimas'}, 'One Pax and one Mimas pilot required')
    hosts = {}
    for name in ('pax', 'mimas'):
        host = manifest['hosts'][name]
        _same(_content(host['plan']), _content(contract['host_plans'][name]),
              'Host execution plan differs from the frozen comparison contract')
        hosts[name] = _load_host(host, expected, check)
    _same(hosts['pax']['identity'], hosts['mimas']['identity'], 'Original cohort identities/flags differ')
    _same(hosts['pax']['protocol'], hosts['mimas']['protocol'], 'Cross-host source/settings/protocol differ')
    distances, feature_diagnostic = feature_distances(expected['features'], expected['feature_sha256'], check=check)
    _same(hosts['pax']['protocol']['density_feature_sha256'], feature_diagnostic['float32_sha256'],
          'Density float32 feature proof differs from the bound raw F16 values')
    stages = {}
    thresholds = contract['thresholds']
    for stage in ('pca', 'umap', 'tsne'):
        check()
        a, b = hosts['pax']['arrays'][stage], hosts['mimas']['arrays'][stage]
        values = geometry(a, b, thresholds)
        if stage == 'pca':
            passes = (not values['degenerate']
                and values['relative_pairwise_error'] <= thresholds['pca_relative_pairwise_error']
                and values['aligned_normalized_residual'] <= thresholds['pca_aligned_normalized_residual']
                and abs(values['scale_ratio'] - 1) <= thresholds['pca_scale_ratio_error'])
        else:
            passes = not values['degenerate'] and (values['raw_allclose'] or values['rigid_allclose'])
        stages[stage] = dict(status='numerically_equivalent_within_frozen_tolerance' if passes else 'review_required',
                             geometry=values, neighborhood=neighborhood(distances, a, b, thresholds['neighbor_k']))
    for stage in ('labels', 'density0'):
        values = partition(hosts['pax']['arrays'][stage].tolist(), hosts['mimas']['arrays'][stage].tolist(), density=stage == 'density0')
        stages[stage] = dict(status=values['status'], partition=values)
    passed = all(stages[stage]['status'] == 'numerically_equivalent_within_frozen_tolerance'
                 for stage in ('pca', 'umap', 'tsne')) and all(
                     stages[stage]['partition']['structural_validation'] for stage in ('labels', 'density0'))
    result = dict(schema_version=1, status='numerically_equivalent_within_frozen_tolerances' if passed else 'review_required',
        protocol_parity=True, numerical_equivalence_within_frozen_tolerances=passed,
        protocol_parity_scope='Exact ordered 64-model identities, raw F16 values, mask, Fe94 wrapper/db52 provider sources and requested settings/seeds; runtime environment facts are reported separately.',
        full_cohort_results_preserved_claim=False, diagnostic_not_full_clustering=True,
        comparison_contract=manifest['contract'], comparator_source=contract['comparator_source'],
        expected=expected, feature_diagnostic=feature_diagnostic,
        model_identity_sha256=_compact(hosts['pax']['identity']),
        protocol=hosts['pax']['protocol'], thresholds=thresholds, stages=stages,
        runtime_records_identical=hosts['pax']['runtime'] == hosts['mimas']['runtime'],
        runtimes={name:dict(source=manifest['hosts'][name]['runtime'], facts=hosts[name]['runtime']) for name in hosts},
        input_receipts={name:manifest['hosts'][name]['receipts'] for name in hosts},
        observed_fit_accounting={name:{stage:hosts[name]['receipts'][stage]['scientific_fits']
                                      for stage in STAGES} for name in hosts},
        scientific_projection_fits=0, HDBSCAN_extractions=0, coordinate_parses=0)
    destination = Path(output).absolute()
    need('..' not in destination.parts and not any(path.is_symlink() for path in [destination, *destination.parents])
         and not destination.exists(), 'A fresh non-symlink comparison destination is required')
    check()
    destination.mkdir(parents=True)
    _pilot().save(destination / 'comparison.json', result)
    with (destination / 'comparison.md').open('x') as stream:
        stream.write(_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_manifest')
    parser.add_argument('output')
    args = parser.parse_args()
    result = compare(args.input_manifest, args.output)
    print(json.dumps(dict(status=result['status'], protocol_parity=result['protocol_parity'], output=args.output)))


if __name__ == '__main__':
    main()
