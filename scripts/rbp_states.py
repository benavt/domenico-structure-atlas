"""Bounded saved-cohort TM states and panel indices; no executable entry point.

The separately approved outer runner owns resource/pressure guards, native runtime,
input hashes, retained core choice, and coordinate exports. This module reuses the
pinned library's geometry and sampling; it does not fit or alter projections.
``retained_core_indices`` are zero-based sequence ordinals, never native labels.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path


SEARCH = dict(cutoff=.8, min_support=50, max_states=3, max_trials=12,
              screen_size=200, max_new_columns=8)
SAMPLING = dict(alpha=.5, k=50, chains=4, steps=10000, burn=1000,
                uniform_mixture=.1)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def native_call_bounds(model_count):
    """Conservative reservation, including rejected columns and selected-pair audit."""
    need(type(model_count) is int and model_count >= 50, 'At least 50 models required')
    pairs = 8 * model_count + 12 * 200 + 3000
    return dict(full_columns=8, screened_trials=12, screen_size=200,
                audit_pairs=3000, unique_pair_ceiling=pairs,
                directional_call_ceiling=2 * pairs)


def core_feature_indices(construct_residues, retained_core_indices):
    """Columns of a complete upper-triangle distance vector for a frozen core."""
    core = list(retained_core_indices)
    need(type(construct_residues) is int and construct_residues >= 2
         and len(core) >= 2 and core == sorted(set(core))
         and all(type(i) is int and 0 <= i < construct_residues for i in core),
         'Core must contain unique increasing zero-based sequence indices')
    return [i * (2 * construct_residues-i-1)//2 + j-i-1
            for offset, i in enumerate(core) for j in core[offset+1:]]


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if hasattr(value, 'tolist'): return value.tolist()
    return value


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def record(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.resolve()), sha256=digest, bytes=path.stat().st_size)


class NativePairOracle:
    """Only this attempt's unordered pairs; scorer supplies durable native evidence.

    ``scorer`` is the pinned NativeTMScorer/SQLiteNativeTMScorer and must already
    bind hash-checked originals through CanonicalCAStore(core+1). The native layer
    reserves each direction before launch. This wrapper independently validates
    protocol, correspondence, both directions, mean, and a conservative call cap.
    """
    def __init__(self, names, scorer, core, sequence_sha256, output, check, workers=1):
        from src.analysis.tm_state_geometry import PROTOCOL_VERSION, TM_ARGUMENTS
        self.names, self.scorer, self.core = list(names), scorer, [i+1 for i in core]
        self.sequence_sha256, self.output, self.check = sequence_sha256, Path(output), check
        self.limit = native_call_bounds(len(names))['directional_call_ceiling']
        self.workers, self.memo, self.events, self.columns = workers, {}, [], set()
        need(type(workers) is int and 1 <= workers <= 8, 'Invalid native worker cap')
        protocol = scorer.protocol
        need(protocol.get('version') == PROTOCOL_VERSION
             and protocol.get('arguments') == list(TM_ARGUMENTS)
             and protocol.get('correspondence') == 'same canonical residue index'
             and protocol.get('normalization') == 'average length'
             and protocol.get('aggregation') == 'mean of two directions'
             and isinstance(protocol.get('executable_sha256'), str)
             and len(protocol['executable_sha256']) == 64, 'Native TM protocol differs')
        need(scorer.directional_calls == 0, 'Fresh scorer required; no implicit historical reuse')
        self.output.mkdir(parents=True, exist_ok=False)

    def _value(self, pair, result):
        i, j = pair
        fingerprint = result.get('fingerprint', {})
        need(result.get('protocol') == self.scorer.protocol
             and fingerprint.get('canonical_positions') == self.core
             and fingerprint.get('sequence_sha256') == self.sequence_sha256,
             'Native result protocol/core/sequence binding differs')
        forward, reverse = result['forward'], result['reverse']
        need((forward.get('source_model'), forward.get('target_model')) == (self.names[i], self.names[j])
             and (reverse.get('source_model'), reverse.get('target_model')) == (self.names[j], self.names[i]),
             'Native bidirectional identities differ')
        values = [forward['score'], reverse['score'], result['score']]
        need(all(not isinstance(v, (bool, str)) and math.isfinite(float(v))
                 and 0 <= float(v) <= 1 for v in values), 'Invalid native TM score')
        need(abs(float(values[2])-(float(values[0])+float(values[1]))/2) <= 1e-12,
             'Native TM value must be the bidirectional mean')
        return float(values[2])

    def __call__(self, pairs):
        import numpy as np
        self.check(); pairs = list(pairs); n = len(self.names)
        need(all(type(i) is int and type(j) is int and 0 <= i < n and 0 <= j < n
                 for i, j in pairs), 'Invalid original model-index pair')
        keys = [tuple(sorted(pair)) for pair in pairs]
        for i, j in keys:
            if i == j: self.memo[i, j] = 1.
        missing = sorted(set(keys)-set(self.memo))
        before = self.scorer.directional_calls
        need(before + 2*len(missing) <= self.limit, 'Native call ceiling exceeded before scoring')
        def one(pair):
            self.check()
            return pair, self._value(pair, self.scorer.score(self.names[pair[0]], self.names[pair[1]]))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for pair, value in pool.map(one, missing): self.memo[pair] = value
        after = self.scorer.directional_calls
        need(after <= self.limit and after-before <= 2*len(missing), 'Native call ceiling exceeded')
        touched = sorted({i for pair in missing for i in pair})
        for i in touched:
            if i not in self.columns and all(j == i or tuple(sorted((i, j))) in self.memo for j in range(n)):
                column = np.asarray([1. if i == j else self.memo[tuple(sorted((i, j)))] for j in range(n)])
                np.savez_compressed(self.output/f'column-{i}.npz', anchor_index=i, scores=column)
                self.columns.add(i)
        self.events.append(dict(requested_pairs=len(pairs), new_unique_pairs=len(missing),
                                directional_calls_before=before, directional_calls_after=after,
                                completed_columns=sorted(self.columns)))
        # Immutable checkpoints survive later geometry/sampling failure. Native SQLite
        # remains the authoritative pair/intent proof, including incomplete columns.
        save(self.output/f'checkpoint-{len(self.events):04d}.json', self.events[-1])
        self.check()
        return [self.memo[key] for key in keys]


def _sampling(cohort, clustering, indices, ordinal, core, prior, tau, output, seed, check, chains_fn, disco_fn):
    import numpy as np
    from src.analysis.tm_state_subset import LAW
    ids = np.asarray(indices, dtype=np.int64)
    need(len(ids) >= 50 and len(set(ids.tolist())) == len(ids), 'State cannot supply 50 unique members')
    energies = np.asarray(cohort['energies'], dtype=float)[ids]
    labels = np.asarray(clustering['hdbscan_labels'])[ids]
    methods = np.asarray(cohort['methods'])[ids]
    logw = np.asarray(prior['log_probabilities'])[ids]
    logw = logw-float(np.max(logw))
    local_seed = seed+ordinal*1000
    initials = [np.argsort(energies, kind='stable')[:50], np.argsort(-energies, kind='stable')[:50],
                np.arange(50), np.arange(50) if len(ids) == 50 else
                np.random.default_rng(local_seed+9).choice(len(ids), 50, replace=False)]
    check()
    runs = chains_fn(logw, k=50, chains=4, steps=10000, burn=1000, seed=local_seed,
        initials=initials, labels=labels, energies=energies, methods=methods,
        noise_labels=clustering['hdbscan_noise_labels'], uniform_mixture=.1)
    check()
    disco = disco_fn(logw, energies, labels, k=50, seed=local_seed+500,
        noise_labels=clustering['hdbscan_noise_labels'], model_ids=[cohort['models'][i] for i in ids])
    need(disco['law'] == 'fine_cluster_min_energy_seeds_then_plackett_luce_control',
         'DISCO must retain the pinned coverage/Plackett-Luce control law')
    need(len(runs['chains']) == 4 and runs['law'] == LAW and runs['production_chain'] == 0
         and runs['production_iteration'] == 10000, 'Fixed production chain/protocol differs')
    selections = {'mcmc50': np.asarray(runs['chains'][0]['final']), 'disco50': np.asarray(disco['indices'])}
    panels = {}
    for method, selected in selections.items():
        need(selected.dtype.kind in 'iu' and selected.shape == (50,)
             and len(set(selected.tolist())) == 50 and ((selected >= 0) & (selected < len(ids))).all(),
             'Panel must contain 50 distinct certified state members')
        original = ids[selected].tolist()
        panels[method] = dict(indices=original, models=[cohort['models'][i] for i in original])
    output.mkdir(parents=True, exist_ok=False)
    arrays = dict(cohort_indices=ids, log_weights=logw,
                  production_indices=panels['mcmc50']['indices'], disco_indices=panels['disco50']['indices'])
    for number, chain in enumerate(runs['chains']):
        arrays.update({f'chain{number}_{key}': value for key, value in chain.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(output/'arrays.npz', **arrays)
    summary = dict(SAMPLING, base_seed=seed, seed=local_seed, ordinal=ordinal, state_members=len(ids),
        original_target_iqr=tau, tau=tau, conditioning_t=.1, energy_scope='full_construct',
        scoring_residues=cohort['construct_residues'], target=cohort['target'], range_id=cohort['range_id'],
        canonical_positions=[i+1 for i in core], target_law=LAW, status='complete',
        prior_scope=f'Original HDBSCAN labels and sizes across the complete {len(cohort["models"])}-model cohort; conditioned on state membership.',
        noise_labels=clustering['hdbscan_noise_labels'], production_chain=0, production_iteration=10000,
        transition_seeds=runs['transition_seeds'], initial_uniform_seed=local_seed+9 if len(ids)>50 else None,
        disco_seed=local_seed+500, diagnostics=runs['diagnostics'],
        chain_metadata=[{key: chain[key] for key in ('seed', 'proposals', 'k', 'degenerate_single_subset', 'group_keys', 'method_keys')}
                        for chain in runs['chains']],
        disco={key: value for key, value in disco.items() if key != 'indices'}, arrays=record(output/'arrays.npz'),
        exact_subset_control_panels=0,
        interpretation='Fixed 50-per-state quotas; MCMC product-subset and DISCO coverage/PL are distinct laws. Completion is not convergence or a population estimate.')
    save(output/'summary.json', summary); check()
    return dict(methods=panels, summary=record(output/'summary.json'), diagnostics=runs['diagnostics'],
                mh_proposals=sum(chain['proposals'] for chain in runs['chains']))


def run_states(cohort, clustering, features, retained_core_indices, output, scorer, *,
               source_bindings, seed, workers=1, check=lambda: None, adapters=None):
    """Produce index-only state catalog inside an approved guarded outer runner.

    ``cohort``: target, range_id, models, methods, energies, sequence_sha256,
    construct_residues, conditioning_t=.1, energy_scope='full_construct'.
    ``clustering``: primary_labels, hdbscan_labels, hdbscan_noise_labels,
    clustering_positions (one-based ordinals), saved_source_views (3 frozen NPZs).
    ``features``: C-contiguous float32 distances for exactly retained_core_indices.
    ``source_bindings``: frozen plan, input/energy/label/projection/source hashes.
    No coordinate export occurs here: an outer reviewed exporter consumes panels'
    original registry indices plus retained core, preserving all native labels.
    Test adapters replace only geometry/samplers/audit; never native protocol checks.
    """
    import numpy as np
    from src.analysis.pa_residual_geometry import run_search
    from src.analysis.ensemble_sampling import energy_distribution
    from src.analysis.residual_state_sampling import proposal_groups, audit_pairs
    from src.analysis.tm_state_geometry import assign_membership
    from src.analysis.tm_state_subset import run_subset_chains, disco_subset
    adapters = {} if adapters is None else dict(adapters)
    names = list(cohort['models']); n = len(names); core = list(retained_core_indices)
    cols = core_feature_indices(cohort['construct_residues'], core)
    need(n >= 50 and len(set(names)) == n and all(isinstance(s, str) and s for s in names), 'Unique model identities required')
    need(type(seed) is int and seed >= 0 and source_bindings, 'Frozen source bindings and seed required')
    need(len(cohort['sequence_sha256']) == 64 and cohort['conditioning_t'] == .1
         and cohort['energy_scope'] == 'full_construct', 'Full-construct ProteinEBM t=0.10 required')
    energies = np.asarray(cohort['energies'], dtype=float)
    need(energies.shape == (n,) and np.isfinite(energies).all(), 'Complete finite saved energies required')
    for label in (cohort['methods'], clustering['primary_labels'], clustering['hdbscan_labels']):
        need(len(label) == n and all(isinstance(s, str) and s for s in label), 'Saved labels must match complete cohort')
    noise = clustering['hdbscan_noise_labels']
    need(isinstance(noise, (list, tuple)) and all(isinstance(s, str) for s in noise), 'Explicit HDBSCAN noise labels required')
    tau = float(np.quantile(energies, .75)-np.quantile(energies, .25))
    need(math.isfinite(tau) and tau > 0, 'Full-cohort energy IQR must be positive')
    # Validate the full-cohort prior before spending any native calls; only later
    # condition it on the independently discovered TM state membership.
    prior = energy_distribution(energies, clustering['hdbscan_labels'], .5, tau, noise)
    need(set(clustering['saved_source_views']) == {'pca', 'umap', 'tsne'}, 'All three saved source projections required')
    need(isinstance(features, np.ndarray) and features.dtype == np.dtype('float32')
         and features.flags.c_contiguous and features.shape == (n, len(cols)),
         'Frozen float32 retained-core distance features differ')
    need(all(np.isfinite(features[start:start+256]).all() for start in range(0, n, 256)),
         'Nonfinite retained-core distance features')
    feature_sha = hashlib.sha256(memoryview(features).cast('B')).hexdigest()
    output = Path(output); need(not output.exists(), 'Fresh immutable state output required')
    check(); output.mkdir(parents=True)
    save(output/'inputs.json', dict(source_bindings=source_bindings, models=names,
        retained_core_indices=core, canonical_positions=[i+1 for i in core],
        feature_sha256=feature_sha, budgets=native_call_bounds(n), seed=seed))
    save(output/'prior-summary.json', {key: value for key, value in prior.items()
                                     if key not in ('log_probabilities', 'probabilities')})
    oracle = NativePairOracle(names, scorer, core, cohort['sequence_sha256'], output/'geometry-checkpoints', check, workers)
    groups = proposal_groups(clustering['primary_labels'], clustering['hdbscan_labels'], cohort['methods'])
    search = adapters.get('run_search', run_search)(names, features, groups, [], {}, oracle,
                                                   settings=dict(SEARCH, feature_count=len(cols)))
    scores = np.asarray(search['scores'], dtype=float); anchors = search['anchors']
    need(len(anchors) <= 3 and len(set(anchors)) == len(anchors)
         and all(type(i) is int and 0 <= i < n for i in anchors)
         and scores.shape == (n, len(anchors)) and np.isfinite(scores).all()
         and ((scores >= 0) & (scores <= 1)).all(), 'Final TM geometry shape differs')
    if anchors:
        assignment = assign_membership(names, [names[i] for i in anchors], scores, cutoff=.8, min_state_size=50)
        need(all(oracle.memo.get(tuple(sorted((i, anchor))), 1. if i == anchor else None) == scores[i, j]
                 for j, anchor in enumerate(anchors) for i in range(n)),
             'Final TM columns must have complete native query evidence')
    else:
        assignment = search['assignment']
        need(assignment['states'] == [] and assignment['unassigned_indices'] == list(range(n)),
             'Empty geometry must leave all models explicitly unassigned')
    need(plain(search['assignment']) == plain(assignment), 'TM representative-star assignment certificate differs')
    budgets = search['budgets']
    need(search.get('cached_anchor_indices') == [] and search.get('baseline', {}).get('anchors') == []
         and budgets['new_full_columns'] <= 8 and budgets['noncached_trials'] <= 12
         and budgets['screen_pair_requests'] <= 2400 and budgets['completion_pair_requests'] <= 8*n,
         'Fresh geometry cache/trial/column bounds differ')
    need(all(state['supported'] and state['count'] >= 50 for state in assignment['states']), 'Unsupported state cannot be sampled')
    pairs = adapters.get('audit_pairs', audit_pairs)(assignment, names)
    need(len(pairs) <= 3000, 'Selected-pair audit ceiling exceeded')
    for row, value in zip(pairs, oracle([(row['i'], row['j']) for row in pairs])): row['tm_score'] = value
    save(output/'pair-audit.json', dict(all_pairs_certified=False, acceptance_gate=False, pairs=pairs))
    np.savez_compressed(output/'scores.npz', anchors=np.asarray([names[i] for i in anchors], dtype='U'), model_to_anchor=scores)
    search = dict(search, kind='bounded-actual-residual-TM080-geometry', target=cohort['target'])
    save(output/'search.json', search); save(output/'membership.json', assignment)
    save(output/'pair-query-ledger.json', dict(events=oracle.events, native_directional_calls=scorer.directional_calls))
    states = []; proposals = 0
    for ordinal, state in enumerate(assignment['states']):
        sample = _sampling(cohort, clustering, state['indices'], ordinal, core, prior, tau,
            output/'sampling'/state['id'], seed, check,
            adapters.get('run_subset_chains', run_subset_chains), adapters.get('disco_subset', disco_subset))
        proposals += sample.pop('mh_proposals'); need(proposals <= 120000, 'MH proposal envelope exceeded')
        states.append(dict(state, minimum_member_tm=float(scores[state['indices'], ordinal].min()), **sample))
    catalog = dict(schema_version=1, kind='saved-clustering-residual-states-and-subsets',
        target=cohort['target'], range_id=cohort['range_id'], models=names, methods=cohort['methods'],
        canonical_positions=[i+1 for i in core], clustering_positions=clustering['clustering_positions'],
        sequence_sha256=cohort['sequence_sha256'], construct_residues=cohort['construct_residues'],
        TM_cutoff=.8, min_state_size=50, proposal_feature_proof=dict(feature_sha256=feature_sha,
        shape=list(features.shape), dtype='float32'), saved_source_views=clustering['saved_source_views'],
        source_bindings=source_bindings, states=states, unassigned_indices=assignment['unassigned_indices'],
        geometry=record(output/'search.json'), membership=record(output/'membership.json'),
        sampling=[state['summary'] for state in states], coordinate_export_status='pending separate reviewed exporter',
        summary=dict(supported_states=len(states), supported_models=n-len(assignment['unassigned_indices']),
        unassigned_models=len(assignment['unassigned_indices']), stop_reason=search['stop_reason'],
        native_directional_calls=scorer.directional_calls, native_call_bounds=native_call_bounds(n),
        geometry_budgets=budgets, pair_audit_pairs=len(pairs), mh_proposals=proposals, panels=2*len(states),
        new_clustering_fits=0, exact_subset_control_panels=0),
        interpretation='Native TM0.80 representative stars; bounded search can miss modes. No all-pairs or biological-state/population claim.')
    save(output/'catalog.json', catalog); check()
    return plain(catalog)
