"""Verify every saved distance row, then screen it in a new immutable run."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def validate_bindings(intake, inventory, names):
    expected = {(row['model_id'], str(Path(row['path']).resolve()), row['sha256']) for row in intake['models']}
    actual = {(row['model'], str(Path(row['source_path']).resolve()), row['sha256']) for row in inventory['models']}
    if len(expected) != len(intake['models']) or len(actual) != len(inventory['models']) or actual != expected:
        raise ValueError('Saved inventory differs from frozen intake')
    if any(row['source_conformer_count'] != 1 or row['chain_id_map'] or row['source_path'] != row['original_source_path'] for row in inventory['models']):
        raise ValueError('Recovery supports only the verified unmodified single-conformer intake')
    if names != {row['model'] for row in inventory['models']}:
        raise ValueError('Saved distance file coverage differs from intake')


def recover(intake_path, inventory_path, directory, output, terminal_config, distance_catalog):
    import numpy as np
    from src.analysis.structure_io import get_structure
    from src.analysis.sequence import Sequence
    from src.analysis.half_matrix import load_distance_matrix, read_distogram_manifest, list_distogram_paths
    from src.analysis.prepared_distograms import prepare_workflow_distograms
    from src.analysis.terminal_variance import load_terminal_settings
    from src.generation.distogram import calculate_distogram
    from scripts.pdb_to_distogram import get_selected_atoms
    from src.app.shared.workflow_exports import export_distogram_views
    from workflow import load_intake, validate_construct, ROOT
    intake, _ = load_intake(intake_path)
    inventory = json.loads(Path(inventory_path).read_text())
    directory, output = Path(directory).resolve(), Path(output).resolve()
    paths = list_distogram_paths(directory)
    validate_bindings(intake, inventory, {path.stem for path in paths})
    if output == directory or directory in output.parents or output in directory.parents:
        raise ValueError('Recovery output must be separate from preserved distance files')
    output.mkdir(parents=True, exist_ok=False)
    manifest = read_distogram_manifest(directory)
    if manifest['storage'] != 'condensed' or np.dtype(manifest['dtype']) != np.float16:
        raise ValueError('Recovery expects the frozen native float16 condensed distances')
    expected_sequence = json.loads((ROOT/'config/study.json').read_text())['targets'][intake['target']]['sequence']
    by_name = {path.stem:path for path in paths}
    catalog = json.loads(Path(distance_catalog).read_text())
    frozen = {row['model']:row for row in catalog['models']}
    if len(frozen) != len(catalog['models']) or set(frozen) != set(by_name) or catalog['input_inventory_sha256'] != digest(inventory_path):
        raise ValueError('Distance catalog differs from frozen inventory')
    verified = []
    for index, row in enumerate(inventory['models'], 1):
        source = Path(row['source_path'])
        if digest(source) != row['sha256']:
            raise ValueError('Native input changed: '+row['model'])
        models = list(get_structure(source))
        if len(models) != 1 or [str(chain.id) for chain in models[0]] != row['chain_ids']:
            raise ValueError('Native conformer or chain identity changed')
        coordinates = get_selected_atoms(source, model=models[0])
        validate_construct(Sequence.from_model(models[0]), len(coordinates), expected_sequence)
        expected = calculate_distogram(coordinates, condensed=True).astype(np.float16)
        actual = load_distance_matrix(by_name[row['model']], as_full=False, manifest=manifest)
        try:
            stored_hash = hashlib.sha256(memoryview(actual)).hexdigest()
            if stored_hash != frozen[row['model']]['distance_sha256'] or row['sha256'] != frozen[row['model']]['input_sha256']:
                raise ValueError('Frozen distance/input binding changed')
            if actual.shape != expected.shape or not np.isfinite(actual).all() or actual.tobytes() != expected.tobytes():
                raise ValueError('Saved distances do not match native coordinates: '+row['model'])
            verified.append(dict(model=row['model'], input_sha256=row['sha256'],
                                 distance_sha256=stored_hash))
        finally:
            actual._mmap.close()
        if digest(source) != row['sha256']:
            raise ValueError('Input changed during verification')
        if index % 50 == 0 or index == len(inventory['models']):
            print(f'Native distance rows verified: {index}/{len(inventory["models"])}', flush=True)
    receipt = dict(schema_version=1, target=intake['target'], input_inventory=str(Path(inventory_path).resolve()),
                   input_inventory_sha256=digest(inventory_path), intake_sha256=digest(intake_path),
                   distance_catalog_sha256=digest(distance_catalog),
                   distogram_dir=str(directory), distogram_manifest_sha256=digest(directory/'distogram_manifest.json'),
                   method='Every saved float16 row compared byte-for-byte with native float32 coordinate-distance recomputation', models=verified)
    (output/'verified_distograms.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with prepare_workflow_distograms(directory, terminal_settings=load_terminal_settings(terminal_config),
                                    log=lambda text: print(text,flush=True)) as prepared:
        plots = export_distogram_views(directory, output/'diagnostics', prepared_distograms=prepared)
        report = dict(target=intake['target'], input_inventory=receipt['input_inventory'],
                      input_inventory_sha256=receipt['input_inventory_sha256'],
                      terminal_settings_sha256=digest(terminal_config), distogram_dir=str(directory),
                      terminal_detection=prepared.terminal_detection, diagnostics=plots,
                      recovery_validation=str(output/'verified_distograms.json'),
                      recovery_validation_sha256=digest(output/'verified_distograms.json'))
        (output/'screening.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intake',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--distograms',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--terminal-config',type=Path,required=True)
    p.add_argument('--distance-catalog',type=Path,required=True)
    a = p.parse_args()
    recover(a.intake,a.inventory,a.distograms,a.output,a.terminal_config,a.distance_catalog)


if __name__ == '__main__': main()
