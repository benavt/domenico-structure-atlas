"""Project choices passed to the pinned library; execution needs its frozen plan."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_construct(sequence, coordinate_count, expected):
    chains = sequence.subsequences
    if len(chains) != 1 or chains[0].sequence != expected or coordinate_count != len(expected):
        raise ValueError('Native model does not match the exact single-chain target construct')


def load_intake(path):
    from snapshot import sha256, verify
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    source = Path(data['analysis_dir']).resolve()
    study = json.loads((ROOT / 'config/study.json').read_text())
    target = data['target']
    if target not in study['targets'] or len(data['models']) != study['targets'][target]['models']:
        raise ValueError('Intake does not match the selected target cohort')
    if data['source_manifest_sha256'] != study['snapshot']['manifest_sha256']:
        raise ValueError('Intake belongs to another snapshot')
    accepted = verify(data['source_manifest'], data['acceptance'], study)
    expected = {(r['model_id'], r['sha256']) for r in accepted if r['target'] == target}
    if {(r['model_id'], r['sha256']) for r in data['models']} != expected:
        raise ValueError('Intake identities differ from accepted coordinates')
    paths = [Path(row['path']).resolve() for row in data['models']]
    if len(set(paths)) != len(paths) or len({r['model_id'] for r in data['models']}) != len(paths):
        raise ValueError('Duplicate intake identities')
    actual = {p.resolve() for p in source.rglob('*') if p.suffix.lower() in ('.pdb', '.cif', '.mmcif')}
    if actual != set(paths):
        raise ValueError('Input directory differs from frozen intake')
    for row, coordinate in zip(data['models'], paths):
        if not coordinate.is_relative_to(source) or coordinate.is_symlink() or sha256(coordinate) != row['sha256']:
            raise ValueError('Coordinate differs from frozen intake: ' + row['model_id'])
    return data, source


def analyze_options(intake, source, screening):
    from snapshot import sha256
    screening = Path(screening).resolve()
    evidence = json.loads(screening.read_text())
    if evidence['input_inventory_sha256'] != sha256(evidence['input_inventory']):
        raise ValueError('Screening inventory changed')
    inventory = json.loads(Path(evidence['input_inventory']).read_text())
    expected = {(str(Path(r['path']).resolve()), r['sha256']) for r in intake['models']}
    actual = {(str(Path(r['original_source_path']).resolve()), r['original_sha256']) for r in inventory['models']}
    if actual != expected:
        raise ValueError('Screening belongs to another intake')
    if evidence['terminal_settings_sha256'] != sha256(ROOT / 'config/terminal_variance.json'):
        raise ValueError('Terminal policy changed')
    direct_inputs = all(r['source_path'] == r['original_source_path'] for r in inventory['models'])
    return dict(
        pdb_id='', identifier=intake['target'], raw_dir=str(source),
        metrics_db=intake['metrics_db'], gather_metrics=False,
        model_type_config=str(ROOT / 'config/model-types.json'),
        chain_mapping=intake.get('chain_mapping'),
        terminal_config=str(ROOT / 'config/terminal_variance.json'),
        cluster_config=str(ROOT / 'config/umap_clusters.json'),
        frozen_terminal_detection=evidence['terminal_detection'],
        reuse_distogram_dir=evidence['distogram_dir'] if direct_inputs else None,
        export_presentation=True, cluster_ensembles=True,
        manifest_metadata={
            'presentation_target': intake['title'],
            'dataset_scope': intake['dataset_scope'],
            'ensemble_selection_scope': intake['ensemble_selection_scope'],
        },
    )


def screen(intake, source, output):
    from snapshot import sha256
    from src.analysis.workflow_inputs import prepare_workflow_inputs
    from scripts.pdb_to_distogram import get_selected_atoms
    from src.app.workers.distogram_worker import DistogramWorker
    from src.analysis.prepared_distograms import prepare_workflow_distograms
    from src.analysis.terminal_variance import load_terminal_settings
    from src.app.shared.workflow_exports import export_distogram_views
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Screening requires a fresh immutable directory')
    if output == source or source in output.parents:
        raise ValueError('Screening outputs must be outside the input directory')
    output.mkdir(parents=True)
    inventory, parsed = prepare_workflow_inputs(source, output, coordinate_loader=get_selected_atoms,
                                              chain_mapping=intake.get('chain_mapping'))
    study = json.loads((ROOT / 'config/study.json').read_text())
    expected = study['targets'][intake['target']]['sequence']
    for sequence, coordinates in parsed.values():
        validate_construct(sequence, len(coordinates), expected)
    distograms = output / 'distograms'
    errors = []
    worker = DistogramWorker('', str(source), str(distograms), identifier=intake['target'],
                            structure_paths=[r['source_path'] for r in inventory['models']],
                            parsed_structure_data=parsed, reuse_structure_parse=True,
                            alignment_processed_dir=output, error_callback=errors.append,
                            status_callback=lambda text: print(text, flush=True))
    worker.run()
    if errors:
        raise RuntimeError('; '.join(map(str, errors)))
    settings = ROOT / 'config/terminal_variance.json'
    with prepare_workflow_distograms(distograms, terminal_settings=load_terminal_settings(settings),
                                    log=lambda text: print(text, flush=True)) as prepared:
        plots = export_distogram_views(distograms, output / 'diagnostics', prepared_distograms=prepared)
        report = dict(target=intake['target'], input_inventory=str(output / 'input_inventory.json'),
                      input_inventory_sha256=sha256(output / 'input_inventory.json'),
                      terminal_settings_sha256=sha256(settings), distogram_dir=str(distograms),
                      terminal_detection=prepared.terminal_detection, diagnostics=plots)
        (output / 'screening.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inspect', 'screen', 'analyze'])
    p.add_argument('--intake', type=Path, required=True)
    p.add_argument('--output', type=Path, help='New screening directory')
    p.add_argument('--screening', type=Path, help='Reviewed exact screening.json')
    p.add_argument('--processed-root', type=Path, help='Run-local output base for analysis')
    a = p.parse_args()
    intake, source = load_intake(a.intake)
    if a.stage == 'inspect':
        from src.analysis.workflow_resources import inspect_inputs
        print(json.dumps(inspect_inputs(source), indent=2)); return
    if a.stage == 'screen':
        if not a.output: p.error('screen requires --output')
        screen(intake, source, a.output); return
    if not a.screening or not a.processed_root: p.error('analyze requires --screening and --processed-root')
    os.environ['DATA_PROCESSED'] = str(a.processed_root.resolve())
    options = analyze_options(intake, source, a.screening)
    from src.app.shared.structure_workflow import StructureWorkflowWorker
    worker = StructureWorkflowWorker(**options, status_callback=lambda text: print(text, flush=True))
    worker.run()
    if worker.error: raise RuntimeError(worker.error)
    print('Manifest: ' + str(worker.manifest_path), flush=True)


if __name__ == '__main__':
    main()
