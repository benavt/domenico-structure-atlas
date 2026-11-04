"""Generic exact-selection full-atom state panels, using pinned native helpers.

No CLI, selection, TM scoring, fitting or PyMOL startup occurs on import. An
approved outer runner supplies the saved state catalog, original registry and
pressure/resource safeguards. The catalog's validated canonical comparison core
defines the display fit. Complete construct correspondence and all native atoms
are preserved; native TM membership is never recomputed.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re


METHODS = ('mcmc50', 'disco50')
COLORS = ('marine', 'orange', 'forest')
SAFE_ID = re.compile(r'[A-Za-z0-9_.-]+\Z')
HEX = re.compile(r'[0-9a-f]{64}\Z')


def need(value, message):
    if not value:raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    need('..' not in path.parts and not any(p.is_symlink() for p in [path, *path.parents])
         and path.is_file(), 'Expected an exact regular non-symlink file')
    return path


def record(path):
    path = regular(path)
    with path.open('rb') as stream:sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def _indices(value, count, *, exact=None):
    need(isinstance(value, list) and len(value) == len(set(value))
         and all(type(i) is int and 0 <= i < count for i in value)
         and (exact is None or len(value) == exact), 'Invalid distinct original registry indices')
    return value


def checked_catalog(catalog, members):
    """Validate the saved index/identity contract without opening unselected models."""
    names = catalog['models']; n = len(names); construct = catalog['construct_residues']
    core = catalog['canonical_positions']
    need(catalog.get('schema_version') == 1
         and catalog.get('kind') == 'saved-clustering-residual-states-and-subsets'
         and catalog.get('TM_cutoff') == .8 and catalog.get('min_state_size') == 50,
         'Expected the saved native TM0.80 state catalog')
    need(n >= 50 and len(set(names)) == n
         and all(isinstance(s, str) and SAFE_ID.fullmatch(s) and s not in ('.', '..') for s in names)
         and [row['model'] for row in members] == names, 'Original registry order/identities differ')
    need(type(construct) is int and construct >= 3
         and isinstance(core, list) and len(core) >= 3
         and all(type(p) is int and 1 <= p <= construct for p in core)
         and core == sorted(set(core))
         and HEX.fullmatch(catalog.get('sequence_sha256', '')), 'Canonical comparison core/construct sequence differs')
    for row in members:
        need(Path(row['source_path']).is_absolute() and HEX.fullmatch(row.get('sha256', '')),
             'Exact original source paths and hashes required')
    states = catalog['states']; seen = set()
    need(isinstance(states, list) and len(states) <= 3
         and len({s['id'] for s in states}) == len(states), 'Invalid state registry')
    selected = set()
    for state in states:
        need(SAFE_ID.fullmatch(state['id']) and state['id'] not in ('.', '..')
             and state['supported'] is True and state['count'] >= 50, 'Unsupported/unsafe state')
        ids = _indices(state['indices'], n)
        need(state['count'] == len(ids) and not seen.intersection(ids)
             and state['anchor'] in names and names.index(state['anchor']) in ids, 'State membership/anchor differs')
        seen.update(ids); selected.add(names.index(state['anchor']))
        need(set(state['methods']) == set(METHODS), 'Both saved MCMC50 and DISCO50 panels required')
        for method in METHODS:
            panel = state['methods'][method]; indices = _indices(panel['indices'], n, exact=50)
            need(set(indices) <= set(ids) and panel['models'] == [names[i] for i in indices],
                 'Panel must retain exactly its 50 distinct certified state members')
            selected.update(indices)
    unassigned = _indices(catalog['unassigned_indices'], n)
    need(not seen.intersection(unassigned) and seen.union(unassigned) == set(range(n)),
         'State/unassigned registry must partition the complete cohort')
    return sorted(selected)


def native_adapters():
    """Generic pinned primitives; no CASP target allowlist and no monkeypatching."""
    from src.analysis.nested_cluster_candidates import canonical_ca_arrays
    from src.analysis.casp17_ensemble_exports import read_frame, aligned_member, atom_signature
    from structure_clustering.privacy import privacy_check
    return dict(canonical_ca_arrays=canonical_ca_arrays, read_frame=read_frame,
                aligned_member=aligned_member, atom_signature=atom_signature, privacy_check=privacy_check)


def _identity(signature):
    return sorted((chain.strip(), str(residue[1])+residue[2].strip(), resname.strip(), name.strip(), alt.strip(),
                   residue[0] != ' ', element.strip())
                  for chain, residue, resname, name, alt, element, occupancy, bfactor in signature)


def _pdb_atoms(text):
    atoms, xyz = [], []
    for line in text.splitlines():
        if line.startswith(('ATOM  ', 'HETATM')):
            atoms.append((line[21:22].strip(), line[22:26].strip()+line[26:27].strip(),
                line[17:20].strip(), line[12:16].strip(), line[16:17].strip(),
                line.startswith('HETATM'), line[76:78].strip()))
            coordinate = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            need(all(math.isfinite(v) for v in coordinate), 'Nonfinite exported atom coordinates')
            xyz.append(coordinate)
    need(atoms, 'Empty full-atom PDB member')
    return atoms, xyz


@contextmanager
def _directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path); yield
    finally:os.chdir(previous)


def pymol_pdb_text_reader(cmd):
    """Check the command API without starting PyMOL or importing coordinates."""
    reader = getattr(cmd, 'read_pdbstr', None)
    need(callable(reader), 'PyMOL must provide callable read_pdbstr for saved PDB text')
    return reader


def _session(pdb_text, expected, coordinates, names, path, color, factory, check, privacy_check):
    """Roundtrip a discrete50 session; load strings/save relative paths for privacy."""
    name = 'saved_state_panel'
    def verify(cmd):
        need(cmd.count_states(name) == 50, 'PSE must retain exactly50 models')
        for frame, (inventory, xyz) in enumerate(zip(expected, coordinates), 1):
            check(); atoms = cmd.get_model(name, state=frame).atom
            actual = [(str(a.chain), str(a.resi), a.resn, a.name, a.alt.strip(),
                       bool(a.hetatm), a.symbol.strip()) for a in atoms]
            need(sorted(actual) == sorted(inventory), 'PSE atom inventory/chain/residue identity differs')
            # Match by full identity rather than relying on PyMOL's atom sorting.
            left = sorted(zip(inventory, xyz)); right = sorted(zip(actual, [list(a.coord) for a in atoms]))
            need(all(all(abs(float(x)-float(y)) <= 1e-5 for x, y in zip(a[1], b[1]))
                     for a, b in zip(left, right)), 'PSE aligned atom coordinates differ')
    with _directory(path.parent):
        with factory() as pm:
            cmd = pm.cmd; reader = pymol_pdb_text_reader(cmd)
            cmd.reinitialize(); cmd.set('max_threads', 1)
            reader(pdb_text, name, discrete=1); verify(cmd)
            for frame, model in enumerate(names, 1):cmd.set_title(name, frame, model)
            cmd.hide('everything', 'all'); cmd.show('cartoon', name); cmd.color(color, name)
            cmd.set('all_states', 1); cmd.set('cartoon_transparency', .45); cmd.bg_color('white')
            cmd.orient(name); cmd.zoom(name, buffer=5); cmd.deselect()
            cmd.save(path.name, format='pse', quiet=1)
        need(path.is_file() and path.stat().st_size > 0, 'Native PyMOL did not save the panel session')
        privacy_check(path.read_bytes(), path.name)
        before = record(path)
        with factory() as pm:
            pm.cmd.load(path.name, format='pse', quiet=1); verify(pm.cmd)
            need(pm.cmd.get_setting_int('all_states') == 1, 'PSE must display all50 models')
        need(record(path) == before, 'PSE verification modified the saved artifact')
    return dict(status='passed', states=50, native_atom_inventory_preserved=True,
                aligned_coordinates_preserved=True, all_states_visible=True)


def export_catalog(catalog_path, members, output, *, expected_catalog_sha256,
                   pymol_factory=None, adapters=None, check=lambda: None, log=lambda _: None,
                   checkpoint=lambda _: None):
    """Export saved50 panels only; this scientific export needs exact plan approval.

    ``members`` is the complete ordered original registry: model/source_path/sha256.
    The frozen catalog is ``rbp_states.run_states`` output. Callbacks are injection
    seams for mock-only source tests; production uses the pinned native helpers.
    Each state/method gets one50-model PDB, one discrete50 PSE and alignment proof.
    Sessions are preparatory views; no camera-dependent figure is rendered here.
    """
    import numpy as np
    source = record(catalog_path)
    need(HEX.fullmatch(expected_catalog_sha256) and source['sha256'] == expected_catalog_sha256,
         'Frozen state catalog hash changed')
    catalog = json.loads(Path(source['path']).read_text()); members = list(members)
    selected = checked_catalog(catalog, members)
    output = Path(output).absolute()
    need(not output.exists() and not any(p.is_symlink() for p in [output, *output.parents]),
         'Fresh immutable panel output required')
    source_records = {i:record(members[i]['source_path']) for i in selected}
    need(all(source_records[i]['sha256'] == members[i]['sha256'] for i in selected), 'Original coordinate hash changed')
    funcs = native_adapters() if adapters is None else dict(adapters)
    need(set(funcs) == {'canonical_ca_arrays', 'read_frame', 'aligned_member', 'atom_signature', 'privacy_check'},
         'Exact generic native adapter interfaces required')
    if pymol_factory is None:
        from pymol2 import PyMOL
        pymol_factory = PyMOL
    check(); output.mkdir(parents=True)
    result = deepcopy(catalog); result['states'] = []
    for ordinal, state in enumerate(catalog['states']):
        anchor_index = catalog['models'].index(state['anchor']); exported_state = deepcopy(state)
        for method in METHODS:
            check(); panel = state['methods'][method]; ids = [anchor_index, *panel['indices']]
            arrays, sequence, mappings = funcs['canonical_ca_arrays'](
                [members[i]['source_path'] for i in ids], retained_positions=None, include_mapping=True)
            construct = catalog['construct_residues']; full_order = list(range(1, construct+1))
            need(np.asarray(arrays).shape == (51, construct, 3) and np.isfinite(arrays).all()
                 and len(sequence) == construct and hashlib.sha256(sequence.encode()).hexdigest() == catalog['sequence_sha256']
                 and len(mappings) == 51, 'Canonical whole-construct correspondence differs')
            for mapping in mappings:
                need([r['canonical_position'] for r in mapping] == full_order
                     and all(type(r['canonical_position']) is int for r in mapping)
                     and len({r['chain'] for r in mapping}) == 1
                     and len({(r['chain'], tuple(r['native_residue'])) for r in mapping}) == construct,
                     'Canonical/native chain and residue mapping differs')
            texts, expected, coordinates, proofs = [], [], [], []
            for number, (index, mapping) in enumerate(zip(panel['indices'], mappings[1:]), 1):
                check(); row = members[index]
                case = dict(canonical_positions=catalog['canonical_positions'], construct_residues=construct,
                            sequence_sha256=catalog['sequence_sha256'], chain=mapping[0]['chain'])
                model, ca, native_mapping = funcs['read_frame'](row, case)
                need(native_mapping == mapping
                     and np.asarray(ca).shape == (construct, 3) and np.isfinite(ca).all()
                     and np.array_equal(ca, arrays[number]), 'Native full-construct frame correspondence changed')
                inventory = _identity(funcs['atom_signature'](model))
                text, metadata, low, high = funcs['aligned_member'](row, case, arrays[0])
                atoms, xyz = _pdb_atoms(text)
                need(sorted(atoms) == inventory and metadata['model'] == row['model']
                     and metadata['source_sha256'] == row['sha256'] and metadata['atom_count'] == len(inventory)
                     and metadata['canonical_mapping'] == mapping
                     and metadata['alignment_canonical_positions'] == catalog['canonical_positions'],
                     'Full-atom PDB inventory or canonical alignment proof differs')
                texts.append(text); expected.append(atoms); coordinates.append(xyz); proofs.append(metadata)
                log(f"{state['id']} {method}: full-atom member {number}/50 verified")
            folder = output/state['id']/method; folder.mkdir(parents=True)
            pdb, pse = folder/'ensemble.pdb', folder/'ensemble.pse'
            blocks = ['REMARK 900 CANONICAL COMPARISON CORE CA LEAST-SQUARES DISPLAY FIT; MEMBERSHIP USES NATIVE TM',
                      'REMARK 900 CANONICAL CORE '+','.join(map(str, catalog['canonical_positions'])),
                      f'REMARK 900 FULL {construct}-RESIDUE CONSTRUCT ATOMS RETAINED']
            for frame, (name, text) in enumerate(zip(panel['models'], texts), 1):
                blocks.extend([f'MODEL     {frame:4d}', 'REMARK 900 SOURCE_MODEL '+name,
                    *[line for line in text.splitlines() if line.startswith(('ATOM  ', 'HETATM', 'TER   '))], 'ENDMDL'])
            pdb_text = '\n'.join([*blocks, 'END', ''])
            with pdb.open('x') as stream:stream.write(pdb_text)
            funcs['privacy_check'](pdb_text.encode(), pdb.name)
            verification = _session(pdb_text, expected, coordinates, panel['models'], pse, COLORS[ordinal],
                                    pymol_factory, check, funcs['privacy_check'])
            proof_path = folder/'alignment.json'
            save(proof_path, dict(anchor=state['anchor'], canonical_positions=catalog['canonical_positions'],
                construct_residues=construct, full_construct_canonical_positions=full_order,
                members=proofs, verification=verification, fit='Proper canonical Cα least squares; no pruning; full-atom construct retained.'))
            exported_state['methods'][method] = dict(panel, pdb=record(pdb), pse=record(pse), alignment=record(proof_path),
                full_atom_frames=50, alignment_canonical_positions=catalog['canonical_positions'],
                camera_status='preparatory session; final camera-dependent rendering requires review')
            checkpoint(f'panel-written:{state["id"]}:{method}')
        result['states'].append(exported_state)
    need(record(source['path']) == source and all(record(members[i]['source_path']) == source_records[i] for i in selected),
         'Frozen state catalog or original coordinates changed during export')
    result.update(source_catalog=source, coordinate_export_status='complete',
                  coordinate_export_summary=dict(panels=2*len(catalog['states']), frames=100*len(catalog['states']),
                    unique_original_models=len(selected), new_clustering_fits=0, new_TM_calls=0))
    save(output/'catalog.json', result); check()
    return result
