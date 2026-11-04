"""Verify saved TM states and attach schema-two public files; no scientific work.

Only hash-bound saved catalogs, scores, sampling/alignment evidence and selected
PDB/PSE files are read. Scientific view JSON stays byte-for-byte unchanged.
"""
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

from rbp_panel_exports import regular, record

TARGET, RANGE, SCOPE = 'RBP_P02925', 'all', 'full271'
CORE = list(range(1, 272))
METHODS = ('mcmc50', 'disco50')
COLORS = ('#0080ff', '#ff8000', '#339933')


def need(condition, message):
    if not condition: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode()


def logical_hash(value):
    return digest(json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def read(entry):
    need(isinstance(entry, dict) and isinstance(entry.get('path'), str)
         and re.fullmatch('[0-9a-f]{64}', entry.get('sha256', ''))
         and type(entry.get('bytes')) is int, 'Expected exact path/size/SHA256 input record')
    path = regular(entry['path']); raw = path.read_bytes()
    need(len(raw) == entry['bytes'] and digest(raw) == entry['sha256'], 'Frozen saved artifact hash/size changed')
    return raw


def load_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Repeated JSON key'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite saved JSON')))


def arrays(entry):
    import numpy as np
    raw = read(entry)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        need(all(item.file_size <= 64*1024*1024 for item in archive.infolist()), 'Saved array exceeds public verification bound')
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def indices(value, n, count=None):
    need(isinstance(value, list) and len(value) == len(set(value))
         and all(type(i) is int and 0 <= i < n for i in value)
         and (count is None or len(value) == count), 'Invalid distinct saved registry indices')
    return value


def privacy(raw, name):
    from structure_clustering.privacy import privacy_check
    privacy_check(raw, name)


def _raw_view(root, entry, view, names, methods, sequence_sha):
    logical = entry['file']
    need(isinstance(logical, str) and re.fullmatch('[A-Za-z0-9_./-]+\\.json(?:\\.gz)?', logical)
         and not logical.startswith('/') and all(p not in ('', '.', '..') for p in logical.split('/')),
         'Unsafe scientific JSON route')
    path = root/logical
    if not path.exists() and not logical.endswith('.gz'): path = root/(logical+'.gz')
    frozen = record(path); stored = read(frozen)
    raw = gzip.decompress(stored) if path.suffix == '.gz' else stored
    need(digest(raw) == entry['sha256'], 'Frozen scientific JSON hash differs')
    data = load_json(raw); provenance = data.get('provenance', {})
    sequence = provenance.get('sequence_sha256', provenance.get('original_parent_provenance', {}).get('sequence_sha256'))
    need(data['target'] == TARGET and data['view_id'] == view and data.get('range_id', 'all') == RANGE
         and data['models'] == names and data['methods'] == methods and len(data['clusters']) == len(names)
         and data['residues'] == 271 and sequence == sequence_sha, 'Scientific view target/order/sequence differs')
    need(data.get('canonical_positions', CORE) == CORE, 'Scientific view retained mask differs')
    return data, frozen


def _method_aliases(native_methods, method_labels):
    need(all(isinstance(key, str) and key.strip() for key in native_methods), 'Native method keys must be nonempty strings')
    keys = sorted(set(native_methods))
    if method_labels is None: aliases = {key:key for key in keys}
    else:
        need(isinstance(method_labels, dict) and set(keys) <= set(method_labels)
             and all(isinstance(key, str) and key.strip() and isinstance(label, str) and label.strip()
                     for key, label in method_labels.items())
             and len(set(method_labels.values())) == len(method_labels),
             'Explicit method labels must cover native keys with one-to-one display names')
        aliases = {key:method_labels[key] for key in keys}
    displayed = [aliases[key] for key in native_methods]
    proof = dict(schema_version=1, applied=method_labels is not None, key_to_display_name=aliases,
        native_method_order_sha256=logical_hash(native_methods), display_method_order_sha256=logical_hash(displayed),
        scientific_method_keys_preserved=True, proposal_bucket_names_preserved=True)
    return displayed, proof


def _summary(state, names, target, range_id, tau):
    import numpy as np
    saved = load_json(read(state['summary']))
    expected = dict(status='complete', alpha=.5, k=50, chains=4, steps=10000, burn=1000,
        uniform_mixture=.1, production_chain=0, production_iteration=10000, conditioning_t=.1,
        energy_scope='full_construct', scoring_residues=271, canonical_positions=CORE,
        target=target, range_id=range_id, state_members=len(state['indices']),
        target_law='product_weight_fixed_cardinality_subset')
    need(all(saved.get(key) == value for key, value in expected.items()), 'Saved sampling production policy/core differs')
    iqr = saved.get('original_target_iqr')
    need(type(iqr) in (int, float) and math.isfinite(iqr) and iqr > 0 and saved.get('tau') == iqr
         and (tau is None or tau == iqr), 'Sampling tau must be the shared full-cohort energy IQR')
    need(saved.get('prior_scope') == f'Original HDBSCAN labels and sizes across the complete {len(names)}-model cohort; conditioned on state membership.'
         and saved.get('disco', {}).get('law') == 'fine_cluster_min_energy_seeds_then_plackett_luce_control',
         'Saved shared HDBSCAN prior or DISCO law differs')
    data = arrays(saved['arrays']); ids = state['indices']
    need(data['cohort_indices'].tolist() == ids, 'Sampling state registry differs')
    local = data['chain0_final']
    need(local.dtype.kind in 'iu' and local.shape == (50,) and len(set(local.tolist())) == 50
         and ((local >= 0) & (local < len(ids))).all(), 'Saved production chain0 final subset differs')
    production = [ids[i] for i in local.tolist()]
    need(production == state['methods']['mcmc50']['indices'] == data['production_indices'].tolist()
         and state['methods']['disco50']['indices'] == data['disco_indices'].tolist(),
         'Exported panels differ from saved production MCMC50/DISCO50 indices')
    need(np.asarray(data['log_weights']).shape == (len(ids),) and np.isfinite(data['log_weights']).all(),
         'Saved conditional prior weights differ')
    from src.analysis.atlas_residual_states import _diagnostics
    return iqr, _diagnostics(saved['diagnostics']), saved['arrays']


def _pdb(raw, selected, alignment, sequence_sha):
    """Inspect frames/sequence/native labels; never load or transform coordinates."""
    amino = dict(zip(('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL').split(),
                     'ARNDCQEGHILKMFPSTWYV')); amino.update(MSE='M')
    frames, sources, active, keys, sequence, atom_count = 0, [], False, [], [], 0
    for line in raw.decode('ascii').splitlines():
        if line.startswith('MODEL '):
            need(not active and int(line[6:].strip()) == frames+1, 'Invalid PDB frame order')
            active, keys, sequence, atom_count = True, [], [], 0
        elif line.startswith('REMARK 900 SOURCE_MODEL '):
            need(active, 'PDB source identity outside frame'); sources.append(line[24:])
        elif line.startswith(('ATOM  ', 'HETATM')):
            need(active and len(line) >= 54 and all(math.isfinite(float(line[a:b])) for a, b in ((30,38),(38,46),(46,54))),
                 'Malformed/nonfinite saved PDB atom')
            atom_count += 1
            if line[12:16].strip() == 'CA':
                key = (line[21].strip(), int(line[22:26]), line[26], line.startswith('HETATM'))
                residue = line[17:20].strip(); need(residue in amino, 'Unknown saved protein CA residue')
                if key in keys:
                    need(sequence[keys.index(key)] == amino[residue], 'Ambiguous alternate CA residue')
                else: keys.append(key); sequence.append(amino[residue])
        elif line.startswith('ENDMDL'):
            need(active and frames < 50, 'Invalid PDB frame ending')
            proof = alignment['members'][frames]
            expected = [(row['chain'].strip(), row['native_residue'][1], row['native_residue'][2],
                         row['native_residue'][0] != ' ') for row in proof['canonical_mapping']]
            need(keys == expected and len(keys) == 271 and digest(''.join(sequence).encode()) == sequence_sha
                 and proof['atom_count'] == atom_count, 'Saved full-construct PDB sequence/native mapping/inventory differs')
            active = False; frames += 1
    need(not active and frames == 50 and sources == selected, 'Saved PDB must retain exact50 model identities/order')


def attach(source_root, exported_catalog_entry, scores_entry, raw_views, *, expected_source_bindings,
           expected_model_order, sequence_sha256, plan_sha256, source_commit, method_labels=None):
    """Attach only verified saved panels/payload/catalog to a public source tree.

    Input artifacts use exact path/bytes/sha256 records. Raw views use relative
    file routes and SHA256 of decoded scientific JSON (gzip remains unchanged).
    Returned ``files`` are ready for the caller's explicit publication allowlist;
    ``tm_state_selection_catalog`` and ``target_entry`` contain schema-two routing.
    Optional ``method_labels`` explicitly maps saved native keys to one-to-one
    public display names; it never changes scientific methods or proposal buckets.
    The outer caller owns execution acceptance and exact publication approval.
    """
    import numpy as np
    from src.analysis.tm_state_geometry import assign_membership
    root = Path(source_root).absolute()
    need(root.is_dir() and not any(p.is_symlink() for p in [root, *root.parents]), 'Regular public source root required')
    need(re.fullmatch('[0-9a-f]{64}', sequence_sha256) and re.fullmatch('[0-9a-f]{64}', plan_sha256)
         and re.fullmatch('[0-9a-f]{40}', source_commit), 'Frozen sequence/plan/source hashes required')
    cat = load_json(read(exported_catalog_entry)); names = list(expected_model_order); n = len(names)
    need(n >= 50 and len(set(names)) == n and all(isinstance(name, str) and name for name in names)
         and cat['models'] == names and len(cat['methods']) == n, 'Complete model registry/order differs')
    need(cat.get('schema_version') == 1 and cat.get('kind') == 'saved-clustering-residual-states-and-subsets'
         and cat['target'] in ('RBP', TARGET) and cat['construct_residues'] == 271 and cat['canonical_positions'] == CORE
         and cat['clustering_positions'] == CORE and cat['sequence_sha256'] == sequence_sha256
         and cat['TM_cutoff'] == .8 and cat['min_state_size'] == 50
         and cat['coordinate_export_status'] == 'complete', 'Expected completed full271 TM0.80 panel catalog')
    need(cat['source_bindings'] == expected_source_bindings
         and expected_source_bindings.get('plan_sha256') == plan_sha256
         and expected_source_bindings.get('consumer_commit') == source_commit, 'Frozen source/plan binding differs')
    need(set(raw_views) == {'primary', 'density'}, 'Exactly Primary and HDBSCAN raw views required')
    display_methods, method_proof = _method_aliases(cat['methods'], method_labels)
    views, raw_records = {}, []
    for view, entry in raw_views.items():
        views[view], frozen = _raw_view(root, entry, view, names, display_methods, sequence_sha256); raw_records.append(frozen)
    need(views['primary']['projections'] == views['density']['projections']
         and set(views['primary']['projections']) == {'pca', 'umap', 'tsne'}, 'Both fine views must reuse all saved projections')
    for projection in views['primary']['projections'].values():
        xy = np.asarray(projection['xy'], dtype=float)
        need(xy.shape == (n, 2) and np.isfinite(xy).all(), 'Complete saved projection coordinates required')
    states = cat['states']; need(isinstance(states, list) and len(states) <= 3, 'At most three supported states allowed')
    data = arrays(scores_entry); anchors = data['anchors'].tolist(); scores = np.asarray(data['model_to_anchor'])
    need(anchors == [state['anchor'] for state in states] and len(set(anchors)) == len(anchors)
         and scores.shape == (n, len(states)) and np.isfinite(scores).all()
         and ((scores >= 0) & (scores <= 1)).all()
         and all(anchor in names and scores[names.index(anchor), j] == 1. for j, anchor in enumerate(anchors)),
         'Saved actual TM columns/order differ')
    if states:
        membership = assign_membership(names, anchors, scores, cutoff=.8, min_state_size=50)
        need(all(state['indices'] == actual['indices'] for state, actual in zip(states, membership['states']))
             and cat['unassigned_indices'] == membership['unassigned_indices'], 'Saved representative-star membership differs')
    else: need(cat['unassigned_indices'] == list(range(n)), 'Empty state result must leave complete cohort unassigned')
    bindings = {view: dict(source_data_sha256=raw_views[view]['sha256']) for view in views}
    payload = dict(schema_version=2, kind='tm_state_selection_payload', target=TARGET, range_id=RANGE,
        effective_scope_id=SCOPE, models=names, model_order_sha256=logical_hash(names), sequence_sha256=sequence_sha256,
        run=dict(plan_sha256=plan_sha256, source_commit=source_commit), chain='A', construct_residues=271,
        canonical_positions=CORE, clustering_positions=CORE, mask_sha256=logical_hash(CORE), tm_threshold=.8,
        provenance=dict(method_label_aliases=method_proof),
        ranges={RANGE:dict(effective_scope_id=SCOPE, views=bindings)}, states=[],
        selection_prior=dict(procedure='HDBSCAN', conditioning_t=.1, energy_scope='full_construct', alpha=.5,
            tau_iqr_multiplier=1, scoring_residues=271, production_chain=0, production_iteration=10000,
            subset_law='product-weight', subset_size=50, disco_law='fine_cluster_min_energy_seeds_then_plackett_luce_control'),
        limitations=['Representative-relative neighborhoods; not all-pairs certificates or biological populations.',
            'Bounded residual search can miss modes; remaining models stay unassigned.',
            'Fixed-budget MCMC is exploratory; diagnostics do not establish convergence.',
            'The same full-cohort HDBSCAN prior supplies both views; MCMC and DISCO use distinct subset laws.'])
    blobs, used, tau, input_records = {}, set(), None, [exported_catalog_entry, scores_entry, *raw_records]
    for ordinal, state in enumerate(states):
        sid = 'S'+str(ordinal+1); ids = indices(state['indices'], n)
        need(state['id'] == sid and state['supported'] is True and state['eligible_for_sampling'] is True
             and state['count'] == len(ids) >= 50 and not used.intersection(ids)
             and state['anchor'] == state['representative'] and state['anchor'] in names
             and names.index(state['anchor']) in ids and set(state['methods']) == set(METHODS), 'Unsupported/overlapping saved state')
        used.update(ids); tau, diagnostic, sampling_arrays = _summary(state, names, cat['target'], cat['range_id'], tau)
        input_records.extend([state['summary'], sampling_arrays]); panels = {}
        for method in METHODS:
            panel = state['methods'][method]; selected = indices(panel['indices'], n, 50)
            need(set(selected) <= set(ids) and panel['models'] == [names[i] for i in selected]
                 and panel['full_atom_frames'] == 50 and panel['alignment_canonical_positions'] == CORE,
                 'Saved selected panel identities/core differ')
            alignment = load_json(read(panel['alignment'])); input_records.append(panel['alignment'])
            need(alignment['anchor'] == state['anchor'] and alignment['canonical_positions'] == CORE
                 and len(alignment['members']) == 50 and alignment['verification'] == dict(status='passed', states=50,
                 native_atom_inventory_preserved=True, aligned_coordinates_preserved=True, all_states_visible=True),
                 'Saved native panel roundtrip verification differs')
            need(all(proof['model'] == model and proof['alignment_canonical_positions'] == CORE
                     and [m['canonical_position'] for m in proof['canonical_mapping']] == CORE
                     for proof, model in zip(alignment['members'], panel['models'])), 'Saved alignment member binding differs')
            public = dict(indices=selected, models=panel['models'])
            for extension, key in [('pdb', 'ensemble'), ('pse', 'session')]:
                raw = read(panel[extension]); input_records.append(panel[extension])
                route = f'tm-state-ensembles/{TARGET}/{RANGE}/{sid}/{method}.{extension}'
                need(0 < len(raw) <= 64*1024*1024, 'Saved panel artifact exceeds viewer bound')
                privacy(raw, route)
                if extension == 'pdb': _pdb(raw, panel['models'], alignment, sequence_sha256)
                blobs[route] = raw; public[key] = dict(file=route, bytes=len(raw), sha256=digest(raw))
            panels[method] = public
        choice = dict(prior_procedure='HDBSCAN', diagnostics=diagnostic, **panels)
        payload['states'].append(dict(id=sid, color=COLORS[ordinal],
            representative=dict(index=names.index(state['anchor']), model=state['anchor']),
            members=[dict(index=i, model=names[i], tm_score=float(scores[i, ordinal])) for i in ids],
            selections={RANGE:{view:deepcopy(choice) for view in views}}))
    need(set(indices(cat['unassigned_indices'], n)) == set(range(n))-used, 'Complete saved coverage partition differs')
    payload['coverage'] = dict(assigned_models=len(used), unassigned_models=n-len(used))
    payload['flags'] = dict(short_core=False, incomplete_coverage=len(used)<n)
    route = f'tm-states/{TARGET}/{SCOPE}.json'; raw = encode(payload); privacy(raw, route); blobs[route] = raw
    target_entry = dict(model_count=n, model_order_sha256=payload['model_order_sha256'], sequence_sha256=sequence_sha256,
        ranges={RANGE:dict(effective_scope_id=SCOPE, views={view:dict(file=route, sha256=digest(raw), **binding,
            available=bool(states), state_count=len(states), **payload['flags']) for view, binding in bindings.items()})})
    catalog = dict(schema_version=2, kind='tm_state_selection_catalog', targets={TARGET:target_entry})
    blobs['tm-states.json'] = encode(catalog); privacy(blobs['tm-states.json'], 'tm-states.json')
    # Validate every source again before making the saved public allowlist concrete.
    for entry in input_records: read(entry)
    for name, raw in blobs.items():
        path = root/name
        need(not any(p.is_symlink() for p in [path, *path.parents]) and
             (not path.exists() or path.read_bytes() == raw), 'Conflicting existing public artifact')
    for name, raw in blobs.items():
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as stream: stream.write(raw)
    return dict(files=[dict(path=name, bytes=len(raw), sha256=digest(raw)) for name, raw in sorted(blobs.items())],
                tm_state_selection_catalog=catalog, target_entry=target_entry)
