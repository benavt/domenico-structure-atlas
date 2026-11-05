"""RBP density inference using the hash-bound CASP17 native primitive.

Importing this adapter performs no scientific work. Its caller owns a frozen
approved scientific plan, runtime/cgroup limits, native source and output scope.
Display PCA/UMAP/t-SNE coordinates are never fitted or changed here. CASP17's
historical matched-child comparisons and landmark validation are deliberately
omitted: RBP has no corresponding frozen child populations or landmark result.
"""
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import types


NATIVE_COMMIT = 'db52d611ff77546ce6839b3b8f65f5dc9c323074'
NATIVE_SHA256 = '2b7fc129474d92d7f4ca5b347c313b25964ff8059fd6d56a4f49efc0afd5e049'
COLORS_SHA256 = '16ebe58a11bca716b867a8d23f29d58bb5aa47a93010774a6599763003bb57c8'
SEEDS = (42, 7, 123)
DIMENSION = 10
MAX_REFINEMENTS = 5
FLOAT32_HASH_BLOCK_BYTES = 16 * 1024**2
_LOADS = itertools.count()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def _save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _regular(path):
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in [path, *path.parents]) or not path.is_file():
        raise ValueError('Native provider must be an exact regular non-symlink file')
    return path


def load_native(source):
    """Load only reviewed bytes; private path/commit bindings belong in the plan."""
    if (not isinstance(source, dict) or source.get('commit') != NATIVE_COMMIT
            or source.get('sha256') != NATIVE_SHA256 or not source.get('path')):
        raise ValueError('Native provider descriptor differs from the reviewed CASP17 primitive')
    path = _regular(source['path'])
    if path.name != 'hd_density_experiment.py':
        raise ValueError('Wrong native primitive filename')
    code = path.read_bytes()
    if hashlib.sha256(code).hexdigest() != NATIVE_SHA256:
        raise ValueError('Native primitive hash changed')
    colors = _regular(path.with_name('hierarchical_colors.py'))
    colors_code = colors.read_bytes()
    if hashlib.sha256(colors_code).hexdigest() != COLORS_SHA256:
        raise ValueError('Native lineage helper hash changed')
    root = path.parents[2]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                     text=True, stderr=subprocess.PIPE).strip()
    if commit != NATIVE_COMMIT:
        raise ValueError('Native source checkout revision changed')
    # Relative imports resolve to the reviewed sibling bytes, not another clone.
    package = '_rbp_density_native_'+NATIVE_SHA256[:12]+'_'+str(next(_LOADS))
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(path.parent)]
    sys.modules[package] = namespace
    loaded = []
    try:
        for suffix, filename, content in [('hierarchical_colors', colors, colors_code),
                                          ('hd_density_experiment', path, code)]:
            name = package+'.'+suffix
            spec = importlib.util.spec_from_file_location(name, filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loaded.append(name)
            exec(compile(content, str(filename), 'exec'), module.__dict__)
        return module, dict(path=str(path), sha256=NATIVE_SHA256, commit=commit,
                            hierarchical_colors_sha256=COLORS_SHA256)
    except BaseException:
        for name in loaded:sys.modules.pop(name, None)
        sys.modules.pop(package, None)
        raise


def _labels(values):
    """Losslessly alias current `Cluster N` names for native lineage ordering."""
    result, aliases = [], {}
    for label in values:
        if not isinstance(label, str) or not label or label.endswith('.unassigned'):
            raise ValueError('Primary labels must be assigned nonempty structural groups')
        match = re.fullmatch(r'Cluster ([1-9]\d*)', label)
        canonical = 'C'+match[1] if match else label
        if not re.fullmatch(r'C[1-9]\d*(?:\.[1-9]\d*)*', canonical):
            raise ValueError('Primary labels require explicit native C lineages or Cluster N names')
        if canonical in aliases.values() and aliases.get(label) != canonical:
            raise ValueError('Primary label aliases collide')
        aliases[label] = canonical
        result.append(canonical)
    return result, aliases


def _stored_float16_sha256(matrix):
    """Hash canonical float32 values with bounded casts, never a cohort copy."""
    import numpy as np
    digest = hashlib.sha256()
    rows = max(1, FLOAT32_HASH_BLOCK_BYTES // (matrix.shape[1]*np.dtype(np.float32).itemsize))
    for start in range(0, len(matrix), rows):
        block = matrix[start:start+rows]
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError('Features must contain finite nonnegative raw Cα distances')
        cast = np.asarray(block, dtype=np.float32, order='C')
        digest.update(memoryview(cast).cast('B'))
    return digest.hexdigest()


def feature_proof(features, names, primary_labels, retained_positions=None, expected_sha256=None,
                  *, allow_stored_float16=False):
    """Legacy dtype/hash describe float32 native fits; stored_dtype records input storage."""
    import numpy as np
    from src.analysis.feature_workspace import row_blocks, feature_sha256
    matrix = np.asarray(features)
    if type(allow_stored_float16) is not bool:
        raise ValueError('Stored float16 input requires an explicit boolean opt-in')
    stored_float16 = matrix.dtype == np.float16 and allow_stored_float16
    if (matrix.ndim != 2 or (matrix.dtype != np.float32 and not stored_float16)
            or matrix.shape[0] != len(names)
            or not len(names) or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names)
            or len(primary_labels) != len(names) or matrix.shape[1] < 1):
        raise ValueError('Expected exact ordered unique model IDs and a raw float32 feature matrix')
    if stored_float16 and (matrix.flags.writeable or not matrix.flags.c_contiguous):
        raise ValueError('Stored float16 features must be read-only and C-contiguous')
    labels, aliases = _labels(primary_labels)
    positions = None if retained_positions is None else list(retained_positions)
    if positions is not None:
        if (not positions or positions != sorted(set(positions))
                or any(type(p) is not int or p < 0 for p in positions)
                or matrix.shape[1] != len(positions)*(len(positions)-1)//2):
            raise ValueError('Feature width differs from the frozen canonical Cα selection')
    if stored_float16:
        digest = _stored_float16_sha256(matrix)
    else:
        for block in row_blocks(matrix):
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError('Features must contain finite nonnegative raw Cα distances')
        digest = feature_sha256(matrix)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Frozen feature values or row order changed')
    return matrix, labels, aliases, dict(shape=list(matrix.shape), dtype='float32',
        stored_dtype=str(matrix.dtype), fit_dtype='float32',
        feature_sha256=digest, model_order_sha256=_json_sha(names),
        primary_partition_sha256=_json_sha(primary_labels),
        retained_reference_positions=positions,
        scaling='Raw saved Cα distances cast to float32; no standardization or feature PCA.')


class _PreparedRows:
    """Provide the native fitter's exact index request without a second full copy."""
    def __init__(self, matrix, indices):
        self.matrix, self.indices = matrix, indices

    def __getitem__(self, indices):
        import numpy as np
        if not np.array_equal(np.asarray(indices), self.indices):
            raise ValueError('Native fitter requested rows outside its frozen selection')
        return self.matrix


def _fit_result(directory, names, indices, settings):
    import numpy as np
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    if (result.get('dimension') != DIMENSION or result.get('selected_indices') != list(map(int, indices))
            or result.get('models') != [names[i] for i in indices] or result.get('settings') != settings):
        raise ValueError('Native fit result differs from its frozen model selection/settings')
    fits = result.get('fits', [])
    expected = {(seed, method) for seed in SEEDS for method in ('leaf', 'eom')}
    if len(fits) != 6 or {(r.get('seed'), r.get('method')) for r in fits} != expected:
        raise ValueError('Expected three UMAP fits and all six native HDBSCAN extractions')
    labels = None
    for seed in SEEDS:
        embedding = np.load(directory/f'embedding-seed-{seed}.npy', mmap_mode='r', allow_pickle=False)
        if embedding.shape != (len(indices), DIMENSION) or not np.isfinite(embedding).all():
            raise ValueError('Native inference embedding has invalid shape or values')
        for method in ('leaf', 'eom'):
            with np.load(directory/f'{method}-seed-{seed}.npz', allow_pickle=False) as stored:
                raw, probabilities = stored['labels'], stored['probabilities']
                if (raw.shape != (len(indices),) or raw.dtype.kind not in 'iu' or np.any(raw < -1)
                        or probabilities.shape != raw.shape or not np.isfinite(probabilities).all()
                        or np.any((probabilities < 0) | (probabilities > 1))):
                    raise ValueError('Native HDBSCAN labels/probabilities are invalid')
                count, noise = len(set(raw.tolist())-{-1}), int(np.sum(raw == -1))
                row = next(r for r in fits if (r['seed'], r['method']) == (seed, method))
                if row.get('clusters') != count or row.get('noise_models') != noise:
                    raise ValueError('Native extraction counts differ from saved labels')
                if seed == 42 and method == 'leaf':labels = raw.copy()
    agreement = result.get('seed_agreement')
    if (not isinstance(agreement, dict) or set(agreement) != {'leaf', 'eom'}
            or any(not isinstance(agreement[method], dict) or set(agreement[method]) != {'7', '123'}
                   for method in ('leaf', 'eom'))):
        raise ValueError('Native seed diagnostics are missing')
    files = {str(p.relative_to(directory)): _sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    return labels, {key: value for key, value in result.items() if key not in ('models', 'selected_indices')}, files


def run(features, names, primary_labels, output, *, native_source,
        mode='casp17_recursive', retained_positions=None, expected_feature_sha256=None,
        workers=4, log=lambda _: None, check=lambda: None, allow_stored_float16=False):
    """Fit density labels only; all new scientific execution needs caller approval.

    CASP17 mode mirrors only its genuine largest-leaf inference loop. There are
    no invented matched-child populations, 20D comparisons or landmark scores.
    Global mode makes one all-cohort 10D selection and retains all native noise.
    The opt-in stored-float16 path hashes float32 values in bounded blocks and
    casts only each selected fit workspace; native inference still sees float32.
    """
    import numpy as np
    from src.analysis.feature_workspace import feature_input_workspace, feature_sha256
    if mode not in ('casp17_recursive', 'global'):
        raise ValueError('Unknown RBP density inference mode')
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Operational HDBSCAN workers must be from one to four')
    names, primary_labels = list(names), list(primary_labels)
    if mode == 'global' and len(names) <= 15:
        raise ValueError('Global frozen density settings need at least sixteen models')
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in [output, *output.parents]):
        raise FileExistsError('Use a fresh immutable output directory; no automatic retry')
    matrix, current, aliases, proof = feature_proof(features, names, primary_labels,
        retained_positions, expected_feature_sha256, allow_stored_float16=allow_stored_float16)
    native, provider = load_native(native_source)
    output.mkdir(parents=True)
    settings = dict(seeds=list(SEEDS), n_neighbors=30, min_dist=0.,
                    min_cluster_size=15, min_samples=11, workers=workers)
    _save(output/'feature-proof.json', proof)
    _save(output/'native-source.json', provider)
    history = [dict(round=0, labels=current.copy(), populations=dict(Counter(current)),
                    structural_clusters=len(set(current)))]
    selections, cache = [], {}

    def fit(indices):
        key = tuple(map(int, indices))
        if key not in cache:
            check()
            directory = output/'fits'/f'selection-{len(cache):02d}-10D'
            # Same native block values, order, dtype and writable C layout. Only
            # allocation differs: bounded staging replaces its advanced-index copy.
            with feature_input_workspace(matrix, indices=indices, dtype=np.float32) as staged:
                selected_hash = feature_sha256(staged)
                native.fit_selection(_PreparedRows(staged, indices), names, indices,
                                     DIMENSION, settings, directory, log)
            raw, diagnostics, files = _fit_result(directory, names, indices, settings)
            record = dict(directory=str(directory), selected_indices=list(key),
                          models=[names[i] for i in key], feature_sha256=selected_hash,
                          diagnostics=diagnostics, files_sha256=files)
            selections.append(record)
            cache[key] = raw, record
            check()
        return cache[key]

    if mode == 'global':
        indices = np.arange(len(names), dtype=np.intp)
        raw, record = fit(indices)
        current, children = native.named_children(raw, names, 'C1')
        history.append(dict(round=1, selected_leaf='global', selected_models=len(indices),
            child_clusters=children, noise_models_in_fit=int(np.sum(raw == -1)), labels=current.copy(),
            populations=dict(Counter(current)), structural_clusters=len(native.structural_labels(current)),
            fit=record['directory']))
        stopping = dict(round=1, reason='single_global_fit')
    else:
        stopping = None
        for number in range(1, MAX_REFINEMENTS+1):
            check()
            counts = Counter(label for label in current if not label.endswith('.unassigned'))
            parent = min(counts, key=lambda label: (-counts[label], native.label_key(label)))
            indices = np.flatnonzero(np.asarray(current) == parent)
            if len(indices) <= 15:
                stopping = dict(round=number-1, reason='insufficient_models_for_frozen_settings')
                break
            raw, record = fit(indices)
            children, count = native.named_children(raw, [names[i] for i in indices], parent)
            current = native.apply_split(current, indices, children, count)
            row = dict(round=number, selected_leaf=parent, selected_models=len(indices),
                child_clusters=count, noise_models_in_fit=int(np.sum(raw == -1)),
                structural_clusters=len(native.structural_labels(current)), labels=current.copy(),
                unassigned_models=sum(label.endswith('.unassigned') for label in current),
                populations=dict(Counter(current)), fit=record['directory'])
            history.append(row)
            _save(output/f'round-{number:02d}.json', row)
            if count <= 2 or number == MAX_REFINEMENTS:
                stopping = dict(round=number, child_clusters=count,
                    reason='two_or_fewer_child_clusters' if count <= 2 else 'five_refinement_limit')
                break
    _save(output/'stopping.json', stopping)
    result = dict(mode=mode, models=names, primary_input_labels=primary_labels,
        primary_label_map=aliases, primary=history[0]['labels'], density=current,
        history=history, stopping=stopping, selections=selections, feature_proof=proof,
        native_source=provider, inference_settings={**settings, 'dimension':DIMENSION,
            'metric':'euclidean', 'primary_extraction':'leaf', 'primary_seed':42},
        selected_fits=len(selections), UMAP_fits=3*len(selections), HDBSCAN_extractions=6*len(selections),
        operational_workers=workers, structural_clusters=len(native.structural_labels(current)),
        unassigned_models=sum(label.endswith('.unassigned') for label in current),
        display_coordinates=dict(refitted=False, requirement='Reuse the exact saved PCA/UMAP/t-SNE coordinates.'),
        unsplit_parent_policy='CASP17 recursive mode retains the original parent, including fitted noise, when fewer than two nonnoise children are found; fitted noise remains explicit in round diagnostics. Global mode retains native noise labels directly.',
        omitted_diagnostics=['No RBP matched historical child populations; no 20D matched-population comparisons.',
                             'No landmark TM matrix or feature/landmark validation is claimed.'],
        interpretation='Density labels and representative-relative states are separate. Equal label text across views does not establish equal membership.')
    _save(output/'analysis.json', result)
    return result
