"""Read-only all-atom verification of saved native representative ensembles.

This is a private local artifact audit, not a publication/privacy assertion. It
does not fit, export, modify or resave any structures or PyMOL sessions. The
approved caller owns native PyMOL startup and pressure/resource safeguards.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


METADATA_TOLERANCE = .0051  # Half a PDB 0.01 occupancy/B-factor unit plus float error.


def need(value, message):
    if not value:raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    need('..' not in path.parts and not any(p.is_symlink() for p in [path, *path.parents])
         and path.is_file(), 'Exact regular non-symlink artifact required')
    with path.open('rb') as stream:digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def _file_inventories(path):
    """Native BioPDB parsing; unpack every alternate residue and atom."""
    from src.analysis.structure_io import get_structure
    inventories = []
    for model in get_structure(path):
        atoms = []
        for chain in model:
            for residue in chain.get_unpacked_list():
                for atom in residue.get_unpacked_list():
                    atoms.append(dict(chain=str(chain.id).strip(),
                        resi=str(residue.id[1])+residue.id[2].strip(), resn=residue.resname.strip(),
                        name=atom.name.strip(), alt=atom.altloc.strip(), hetero=residue.id[0] != ' ',
                        element=str(atom.element).strip().upper(), occupancy=atom.occupancy, bfactor=atom.bfactor))
        inventories.append(atoms)
    return inventories


def _pymol_inventory(cmd, name, state):
    return [dict(chain=str(a.chain).strip(), resi=str(a.resi), resn=a.resn.strip(),
        name=a.name.strip(), alt=a.alt.strip(), hetero=bool(a.hetatm),
        element=a.symbol.strip().upper(), occupancy=a.q, bfactor=a.b)
        for a in cmd.get_model(name, state=state).atom]


def _key(atom):
    return tuple(atom[k] for k in ('chain', 'resi', 'resn', 'name', 'alt', 'hetero', 'element'))


def _same_number(left, right):
    if left is None or right is None:return left is right
    return (not isinstance(left, bool) and not isinstance(right, bool)
            and math.isfinite(float(left)) and math.isfinite(float(right))
            and abs(float(left)-float(right)) <= METADATA_TOLERANCE)


def compare_inventory(expected, actual, context):
    need(expected and Counter(map(_key, expected)) == Counter(map(_key, actual)),
         'All-atom inventory/chain/residue/alternate identity differs: '+context)
    # Sort metadata within an identity as well, retaining duplicate multiplicity.
    def sort_key(atom):
        return (_key(atom), tuple((v is not None, 0. if v is None else float(v))
                                  for v in (atom['occupancy'], atom['bfactor'])))
    for left, right in zip(sorted(expected, key=sort_key), sorted(actual, key=sort_key)):
        need(all(_same_number(left[k], right[k]) for k in ('occupancy', 'bfactor')),
             'Native occupancy/B-factor metadata differs: '+context)


def _inventory_sha(atoms):
    return hashlib.sha256(json.dumps(sorted(atoms, key=lambda a:json.dumps(a, sort_keys=True)),
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_selected_ensembles(selection, *, pymol_factory=None, check=lambda: None):
    """Compare original full atoms with each saved CIF and private PSE state.

    ``selection`` is the result of pinned native ``build_ensemble_session``.
    Only selected cluster objects are audited; the local session's additional
    medoid-reference objects and private paths remain outside this proof's scope.
    Source hashes come from each row's frozen Cα correspondence (or exact sha256).
    """
    groups = [g for g in selection['clusters'] if g.get('cluster_type') != 'reference_only']
    need(groups and len({g['object_name'] for g in groups}) == len(groups), 'Unique selected cluster objects required')
    pse = record(selection['pse']); frozen_sources = {}; expected_groups = []; proofs = []
    for group in groups:
        check(); rows = group['selected']; name = group['object_name']
        need(1 <= len(rows) <= 5 and all(type(row['state']) is int for row in rows)
             and [row['state'] for row in rows] == list(range(1, len(rows)+1))
             and len({row['model'] for row in rows}) == len(rows), 'Saved representative frame order differs')
        cif = record(group['multimodel_structure']); stored = _file_inventories(cif['path'])
        need(len(stored) == len(rows), 'Saved CIF frame count differs: '+name)
        sources = []; frames = []
        for number, row in enumerate(rows, 1):
            check(); source = record(row['source_path'])
            source_hash = (row.get('ca_correspondence') or {}).get('source_sha256', row.get('sha256'))
            need(source_hash == source['sha256'], 'Frozen representative source hash differs: '+row['model'])
            originals = _file_inventories(source['path'])
            need(len(originals) == 1, 'Representative must be one exact native conformer')
            expected = originals[0]; compare_inventory(expected, stored[number-1], name+' CIF state '+str(number))
            frozen_sources[source['path']] = source; sources.append(expected)
            frames.append(dict(model=row['model'], state=number, source=source, atom_count=len(expected),
                source_inventory_sha256=_inventory_sha(expected), cif_inventory_sha256=_inventory_sha(stored[number-1])))
        expected_groups.append((name, sources)); proofs.append(dict(cluster=group['cluster_label'], object_name=name,
            cif=cif, frames=frames, native_atom_inventory_preserved=True, metadata_preserved_within_tolerance=True))
    if pymol_factory is None:
        from pymol2 import PyMOL
        pymol_factory = PyMOL
    with pymol_factory() as pm:
        pm.cmd.load(pse['path'], format='pse', quiet=1)
        for (name, sources), proof in zip(expected_groups, proofs):
            need(pm.cmd.count_states(name) == len(sources), 'Saved PSE frame count differs: '+name)
            for number, (expected, frame) in enumerate(zip(sources, proof['frames']), 1):
                check(); actual = _pymol_inventory(pm.cmd, name, number)
                compare_inventory(expected, actual, name+' PSE state '+str(number))
                frame['pse_inventory_sha256'] = _inventory_sha(actual)
    need(record(pse['path']) == pse and all(record(p['cif']['path']) == p['cif'] for p in proofs)
         and all(record(path) == frozen for path, frozen in frozen_sources.items()),
         'Native source or saved artifact changed during audit')
    return dict(status='passed', selected_groups=len(proofs), selected_frames=sum(len(p['frames']) for p in proofs),
        groups=proofs, pse=pse, metadata_absolute_tolerance=METADATA_TOLERANCE,
        identity_fields=['chain', 'native residue plus insertion', 'residue name', 'atom name', 'alternate', 'hetero', 'element'],
        scope='Selected group frames only; private local PSE. No global session privacy or public-release claim.',
        new_rigid_fits=0, new_exports=0, new_TM_calls=0)
