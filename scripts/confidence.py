"""Extract genuine native confidence or seed a fresh, exact-cohort metrics store.

Execution belongs to a frozen approved plan. No predictions or scores are run.
"""
import argparse
import json
import math
from pathlib import Path

from snapshot import sha256

KINDS = {
    'afcluster': 'native_nested_scores', 'speachaf': 'native_nested_scores',
    'alphaflow-md': 'native_nested_scores', 'alphaflow-pdb': 'native_nested_scores',
    'esmflow-pdb': 'native_nested_scores', 'afsample2': 'af2_monomer_scores',
    'afsample3': 'af3_summary_and_atom_confidence', 'boltz2': 'boltz2_native_scores',
    'opendde': 'opendde_native_scores', 'bioemu': 'unscored_no_native_confidence',
    'bioemu2': 'unscored_no_native_confidence',
}


def number(value, maximum, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError('Invalid native confidence value')
    return float(value)


def extract_model(entry, loader):
    kind = entry['kind']
    if KINDS.get(entry['method']) != kind:
        raise ValueError('Confidence producer does not match the native method')
    files = entry['files']
    roles = {f['role'] for f in files}
    expected = (set() if kind == 'unscored_no_native_confidence' else
                {'native_summary', 'native_atom_confidence'} if kind == 'af3_summary_and_atom_confidence' else {'native_scores'})
    if roles != expected or len(files) != len(expected):
        raise ValueError('Missing or duplicate native confidence files')
    data = {f['role']: loader(f) for f in files}
    note = kind
    if kind == 'unscored_no_native_confidence':
        mean, ptm = None, None
    elif kind == 'native_nested_scores':
        source = data['native_scores']
        if (source['status'] != 'validated' or source['method'] != entry['method']
                or source['pdb'] != entry['coordinate_source'] or source['pdb_sha256'] != entry['coordinate_sha256']):
            raise ValueError('Native confidence is bound to different coordinates')
        mean, ptm = source['scores']['mean_plddt'], source['scores'].get('ptm')
    elif kind == 'af2_monomer_scores':
        source = data['native_scores']; mean, ptm = source['ranking_confidence'], source.get('ptm')
        note = 'Native AF2 monomer ranking_confidence is mean pLDDT (0–100); not multimer ranking confidence.'
    elif kind == 'af3_summary_and_atom_confidence':
        source = data['native_summary']; values = data['native_atom_confidence']['atom_plddts']
        if not values or len(source['chain_ptm']) != 1 or source.get('iptm') is not None:
            raise ValueError('Expected native single-chain AF3 confidence')
        mean = math.fsum(number(v, 100) for v in values)/len(values); ptm = source.get('ptm')
        note = 'Native AF3 atom-weighted mean pLDDT; summary pTM. No monomer ipTM.'
    elif kind == 'boltz2_native_scores':
        source = data['native_scores']; mean = 100*number(source['complex_plddt'], 1); ptm = source.get('ptm')
        note = 'Native Boltz2 complex pLDDT converted from 0–1 to 0–100; monomer ipTM placeholder omitted.'
    else:
        source = data['native_scores']; mean, ptm = source['plddt'], source.get('ptm')
        note = 'Native OpenDDE scalar pLDDT (0–100); monomer ipTM placeholder omitted.'
    return {'model_id': entry['model_id'], 'target': entry['target'], 'method': entry['method'],
            'coordinate_sha256': entry['coordinate_sha256'], 'mean_plddt': number(mean, 100, optional=True),
            'ptm': number(ptm, 1, optional=True), 'eligible': mean is not None,
            'interpretation': note, 'native_files': [{k: f[k] for k in ('role', 'sha256', 'bytes')} for f in files]}


def extract(catalog_path, expected, output):
    if sha256(catalog_path) != expected:
        raise ValueError('Native confidence catalog changed')
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog['errors'] or len({r['model_id'] for r in catalog['models']}) != len(catalog['models']):
        raise ValueError('Confidence catalog has errors or duplicate identities')
    def loader(file):
        path = Path(file['path'])
        if path.stat().st_size != file['bytes'] or sha256(path) != file['sha256']:
            raise ValueError('Native confidence sidecar changed')
        return json.loads(path.read_text())
    rows = []
    for index, row in enumerate(catalog['models'], 1):
        rows.append(extract_model(row, loader))
        if index % 250 == 0: print(f'Extracted {index}/{len(catalog["models"])} native confidence records', flush=True)
    result = {'schema_version': 1, 'status': 'validated', 'catalog_sha256': expected,
              'source_manifest_sha256': catalog['source_manifest_sha256'], 'new_scoring_calls': 0, 'models': rows}
    with Path(output).open('x') as stream: json.dump(result, stream, separators=(',', ':'), allow_nan=False)


def seed(intake_path, confidence_path, expected_confidence_sha, native_path, model_types, output):
    if sha256(confidence_path) != expected_confidence_sha:
        raise ValueError('Extracted native confidence changed')
    from src.analysis.metrics_sqlite import open_metrics_db, upsert_model, apply_model_type_rules_config, upsert_color_scheme
    intake = json.loads(Path(intake_path).read_text()); confidence = json.loads(Path(confidence_path).read_text())
    if confidence['status'] != 'validated' or confidence['source_manifest_sha256'] != intake['source_manifest_sha256']:
        raise ValueError('Confidence belongs to another accepted snapshot')
    if sha256(native_path) != intake['source_manifest_sha256']:
        raise ValueError('Native score snapshot changed')
    natives = {r['model_id']: r for r in json.loads(Path(native_path).read_text())['models']}
    records = confidence['models']; scored = {r['model_id']: r for r in records}
    if len(scored) != len(records) or set(scored) != set(natives):
        raise ValueError('Confidence and accepted snapshot identities differ')
    output = Path(output)
    if output.exists(): raise FileExistsError('Metrics stores are fresh and never overwritten')
    eligible = []
    conn = open_metrics_db(output, migrate_csv=False)
    try:
        for row in intake['models']:
            name = row['model_id']; native, score = natives[name], scored[name]
            if (row['sha256'] != native['sha256'] or score['coordinate_sha256'] != native['sha256']
                    or score['method'] != native['method'] or score['target'] != intake['target']):
                raise ValueError('Metrics join differs from native model identity')
            fields = dict(source_path=row['path'], input_sha256=row['sha256'],
                mean_plddt=score['mean_plddt'], ptm=score['ptm'],
                proteinebm_t010=native['energy_t010'], native_physicality=native['native_physicality'],
                native_confidence_provenance=json.dumps(score, separators=(',', ':')),
                proteinebm_provenance=json.dumps({'all_saved_levels': {f't{i:02d}': native[f'energy_t{i:02d}'] for i in range(16)},
                    'native_manifest_sha256': intake['source_manifest_sha256'], 'source_energy': native['source_energy']}, separators=(',', ':')))
            upsert_model(conn, name, fields, refresh_metadata=False)
            if score['eligible']: eligible.append(row['path'])
        conn.commit(); apply_model_type_rules_config(conn, model_types)
        upsert_color_scheme(conn, 'proteinebm_t010', 'ProteinEBM t=0.10 · lower is better', column_name='proteinebm_t010', scheme_type='continuous',
            description='Saved full-construct learned arbitrary units; not confidence or calibrated free energy.')
        upsert_color_scheme(conn, 'native_physicality', 'Native physicality flags', column_name='native_physicality', scheme_type='categorical', user_defined=True,
            description='Every accepted model remains included. Historical raw BioEMU filtering is separate.')
        conn.commit(); conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally: conn.close()
    intake.update(metrics_db=str(output.resolve()), ensemble_selection_scope={'source_paths': eligible,
        'reason': 'Genuine native confidence explicitly bound to accepted coordinates; energy is not confidence.',
        'unscored_cluster_policy': 'structural_medoid'})
    with output.with_suffix('.intake.json').open('x') as stream: json.dump(intake, stream, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); commands = p.add_subparsers(dest='stage', required=True)
    e = commands.add_parser('extract'); e.add_argument('--catalog', required=True); e.add_argument('--catalog-sha256', required=True); e.add_argument('--output', required=True)
    s = commands.add_parser('seed')
    for name in ['intake', 'confidence', 'confidence-sha256', 'native', 'model-types', 'output']: s.add_argument('--'+name, required=True)
    a = p.parse_args()
    if a.stage == 'extract': extract(a.catalog, a.catalog_sha256, a.output)
    else: seed(a.intake, a.confidence, a.confidence_sha256, a.native, a.model_types, a.output)
