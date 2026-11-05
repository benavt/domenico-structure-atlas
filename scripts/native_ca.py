"""Lossless native Cα display tokens for the project's compact publication."""
import gzip
import hashlib
import math
from pathlib import Path


def ca_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.pdb':
        rows = []
        for line in path.read_text().splitlines():
            if line.startswith('ENDMDL'): break
            if line[:6].strip() not in ['ATOM', 'HETATM'] or line[12:16].strip() != 'CA': continue
            element = line[76:78].strip() if len(line) >= 78 else 'C'
            if element and element != 'C': continue
            rows.append([line[17:20].strip(), line[21:22], line[22:26].strip(), line[26:27].strip() or '?',
                         line[16:17].strip() or '.', line[30:38].strip(), line[38:46].strip(), line[46:54].strip(),
                         line[54:60].strip() or '1', line[60:66].strip() or '0'])
        return rows
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    data = MMCIF2Dict(path)
    names = data['_atom_site.label_atom_id']; count = len(names)
    def column(*keys, default=None):
        return next((data['_atom_site.'+key] for key in keys if '_atom_site.'+key in data), [default]*count)
    columns = [column('label_comp_id', 'auth_comp_id'), column('auth_asym_id', 'label_asym_id'),
               column('auth_seq_id', 'label_seq_id'), column('pdbx_PDB_ins_code', default='?'),
               column('label_alt_id', default='.'), column('Cartn_x'), column('Cartn_y'), column('Cartn_z'),
               column('occupancy', default='1'), column('B_iso_or_equiv', default='0')]
    models = column('pdbx_PDB_model_num', default='1'); elements = column('type_symbol', default='C')
    first = models[0]
    return [[str(values[i]) for values in columns] for i, name in enumerate(names)
            if name == 'CA' and elements[i] == 'C' and models[i] == first]


def token(value):
    value = str(value)
    if value and not any(c.isspace() for c in value) and value[0] not in '#_;\'"': return value
    if '"' not in value and '\n' not in value: return '"'+value+'"'
    if "'" not in value and '\n' not in value: return "'"+value+"'"
    raise ValueError('Unsupported native identifier in display CIF')


def write_ca(path, destination, *, native_sha256, expected_count):
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != native_sha256:
            raise ValueError('Native coordinate hash changed before Cα display export')
    rows = ca_rows(path)
    keys = [(r[1], r[2], r[3]) for r in rows]
    if len(rows) != expected_count or len(set(keys)) != len(keys) or any(r[4] not in ['.', '?', 'A'] for r in rows):
        raise ValueError('Ambiguous native Cα count/alternate identities; no display correspondence guessed')
    if not all(math.isfinite(float(v)) for row in rows for v in row[5:8]):
        raise ValueError('Nonfinite native display coordinates')
    fields = ['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id', 'label_comp_id',
              'label_asym_id', 'label_entity_id', 'label_seq_id', 'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y',
              'Cartn_z', 'occupancy', 'B_iso_or_equiv', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id',
              'auth_atom_id', 'pdbx_PDB_model_num']
    lines = ['data_native_ca', '#', 'loop_', *['_atom_site.'+f for f in fields]]
    for i, row in enumerate(rows, 1):
        name, chain, residue, insertion, alt, x, y, z, occupancy, b = row
        values = ['ATOM', i, 'C', 'CA', alt, name, chain, '1', i, insertion, x, y, z, occupancy, b,
                  residue, name, chain, 'CA', '1']
        lines.append(' '.join(map(token, values)))
    raw = ('\n'.join(lines)+'\n#\n').encode()
    with Path(destination).open('xb') as stream:
        stream.write(gzip.compress(raw, compresslevel=6, mtime=0))
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'native_sha256': native_sha256,
            'representation': 'ca_trace', 'format': 'cif', 'count': len(rows)}
