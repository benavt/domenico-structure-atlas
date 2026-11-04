"""Run the exact approved RBP CASP17-parity stages inside the existing guard.

No scientific work occurs on import. This adapter preserves saved display arrays,
uses the pinned native engines, and writes only a fresh attempt. Publication is a
separate saved-artifact operation; camera-dependent images remain gated.
"""
from contextlib import contextmanager
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from zipfile import ZipFile

from snapshot import sha256

TARGET = 'RBP_P02925'
METHOD_NOTE = ('CASP17 recursive refinement of the saved primary partition: raw retained Cα-distance '
    'features cast to float32 without standardization or feature PCA; 10-dimensional UMAP '
    '(neighbors30, min_dist0, seeds42/7/123), native HDBSCAN leaf extraction '
    '(min_cluster_size15, min_samples11, Euclidean kd_tree); largest nonnoise leaf refined '
    'for at most five rounds, stopping at two or fewer children. The seed42 labels are used; '
    'repeat seeds and EOM extractions are diagnostics. Existing PCA/UMAP/t-SNE display '
    'coordinates are reused exactly. Native fitted noise and CASP17 unsplit-parent behavior '
    'are retained in the saved round diagnostics.')


def need(value, message):
    if not value: raise ValueError(message)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def record(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def compact_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def read_bound(entry):
    need(sha256(entry['path']) == entry['sha256'], 'Frozen input changed: '+entry['path'])
    return json.loads(Path(entry['path']).read_text())


def validate_approval(path):
    plan = json.loads(Path(path).read_text())
    approval = json.loads((Path(path).parent/'approval.json').read_text())
    need(approval.get('approved_by') == 'user'
         and approval.get('approved_plan_sha256') == sha256(path), 'Exact execution approval required')
    need(plan['target'] == TARGET and plan['settings']['density_mode'] == 'casp17_recursive'
         and plan['settings']['tm_core_indices'] == list(range(271)), 'User-selected RBP scope differs')
    need(os.getpgrp() == os.getpid(), 'Driver requires the guard-owned private process group')
    from run_reviewed_plan import verify_controls, control_group
    verify_controls(control_group(), plan['resources'])
    for name, digest in plan['frozen_files'].items():
        need(sha256(name) == digest, 'Frozen file changed: '+name)
    return plan


@contextmanager
def saved_matrix(entry, manifest, names, scratch):
    """Read only numerical NPY and JSON metadata, with exact frozen row identity."""
    import numpy as np
    from src.analysis.feature_workspace import feature_sha256
    path = Path(entry['path']); need(sha256(path) == entry['sha256'], 'Distance cache changed')
    scratch = Path(scratch); destination = scratch/'rbp-saved-features.npy'
    need(not destination.exists(), 'Feature scratch already exists')
    with np.load(path, allow_pickle=False) as saved:
        signature = json.loads(str(saved['source_signature']))
        order = [Path(row[0]).stem for row in signature['files']]
        need(order == names, 'Saved feature order differs from displayed model order')
        need(saved['reference_positions'].tolist() == list(range(271))
             and saved['requested_reference_positions'].tolist() == list(range(271)), 'Saved feature mask differs')
        detection = json.loads(str(saved['terminal_detection']))
        need(detection['pair_mask_sha256'] == manifest['terminal_detection']['pair_mask_sha256'], 'Saved feature pair mask differs')
    matrix = None
    try:
        with ZipFile(path) as archive:
            need(archive.getinfo('flattened.npy').file_size <= entry['maximum_npy_bytes'], 'Saved array exceeds frozen size')
            with archive.open('flattened.npy') as source, destination.open('xb') as output:
                shutil.copyfileobj(source, output, length=1024*1024)
        matrix = np.load(destination, mmap_mode='r', allow_pickle=False)
        need(matrix.shape == (len(names), 36585) and matrix.dtype == np.float16,
             'Saved feature shape/precision differs')
        need(feature_sha256(matrix) == manifest['knn_clustering']['feature_input']['sha256'], 'Saved feature values differ')
        yield matrix
    finally:
        if matrix is not None: matrix._mmap.close()
        if destination.exists(): destination.unlink()


@contextmanager
def frozen_features(entry, manifest, names, scratch):
    """Cast only the verified numerical cache, without deserializing object metadata."""
    import numpy as np
    from src.analysis.feature_workspace import feature_input_workspace, feature_sha256
    with saved_matrix(entry, manifest, names, scratch) as matrix:
        with feature_input_workspace(matrix, dtype=np.float32) as staged:
            yield staged, feature_sha256(staged)


def registry(plan, manifest, study, check, checkpoint=lambda _: None):
    from publish_atlas import projections
    intake = read_bound(plan['intake']); native = read_bound(plan['native'])
    need(native['status'] == 'validated' and plan['native']['sha256'] == study['snapshot']['manifest_sha256'],
         'Accepted snapshot binding differs')
    inputs = {r['model_id']: r for r in intake['models']}
    rows = {r['model_id']: r for r in native['models'] if r['target'] == TARGET}
    need(len(inputs) == len(intake['models']) == len(rows) == study['targets'][TARGET]['models']
         and set(inputs) == set(rows), 'Snapshot identities differ')
    names, _ = projections(manifest, rows)
    with Path(manifest['knn_clustering']['labels_csv']).open() as stream:
        labels = list(csv.DictReader(stream))
    by_label = {r['model']: r for r in labels}
    need(len(by_label) == len(labels) == len(names) and set(by_label) == set(names), 'Primary partition differs')
    members = []
    for index, name in enumerate(names):
        check(); source = inputs[name]
        need(source['sha256'] == rows[name]['sha256'] == sha256(source['path'])
             and Path(by_label[name]['source_path']).resolve() == Path(source['path']).resolve(),
             'Coordinate/score/primary identity binding differs: '+name)
        members.append(dict(model=name, source_path=source['path'], sha256=source['sha256'],
                            method=rows[name]['method'], energy=rows[name]['energy_t010']))
        if (index+1)%500 == 0:
            print(f'RBP: verified {index+1}/{len(names)} originals', flush=True)
            checkpoint(f'original-hashes:{index+1}')
    return names, members, [by_label[name] for name in names]


def write_density_labels(result, primary, output):
    """Keep every model including explicit noise; only assigned groups get poses."""
    from src.analysis.hierarchical_colors import lineage_palette
    names, labels = result['models'], result['density']
    need(0 < len({label for label in labels if not label.endswith('.unassigned')}) <= 732,
         'Assigned density cluster count exceeds the frozen envelope')
    colors = lineage_palette(labels)
    path = Path(output)/'labels.csv'; assigned = Path(output)/'assigned-labels.csv'
    fields = ['model', 'source_path', 'umap_x', 'umap_y', 'cluster_label', 'color']
    with path.open('x', newline='') as complete, assigned.open('x', newline='') as groups:
        all_writer, group_writer = csv.DictWriter(complete, fields), csv.DictWriter(groups, fields)
        all_writer.writeheader(); group_writer.writeheader()
        for name, label, source in zip(names, labels, primary):
            need(source['model'] == name, 'Density model order differs')
            row = {key:source.get(key, '') for key in fields}
            row.update(cluster_label=label, color=colors[label]); all_writer.writerow(row)
            if not label.endswith('.unassigned'): group_writer.writerow(row)
    return path, assigned


def prepare_density_ensembles(manifest, labels, output, check, log):
    """Native confidence ranking/medoids and native-label-preserving PyMOL export."""
    from src.analysis.cluster_ensembles import select_cluster_ensembles
    from src.pymol_integration.cluster_ensembles import build_ensemble_session
    import pymol2
    successor = copy.deepcopy(manifest)
    successor['knn_clustering'] = dict(manifest['knn_clustering'], labels_csv=str(labels))
    with Path(labels).open() as stream: assigned = list(csv.DictReader(stream))
    paths = {str(Path(row['source_path']).resolve()) for row in assigned}
    need(paths, 'No assigned density groups available for representative selection')
    scope = copy.deepcopy(manifest['ensemble_selection_scope'])
    scope['source_paths'] = [str(Path(p).resolve()) for p in scope['source_paths'] if str(Path(p).resolve()) in paths]
    need(scope['source_paths'], 'No eligible native confidence rows in assigned density groups')
    scope['reason'] += ' Density noise remains plotted; representative selection covers assigned density groups only.'
    successor['ensemble_selection_scope'] = scope
    selected = select_cluster_ensembles(successor, models_per_cluster=5, log=log, check_cancel=check)
    report = Path(output)/'ensemble_report.json'
    selected.update(report=str(report), manifest=str(Path(output)/'derived-manifest.json'),
                    source_workflow_manifest_sha256=sha256(manifest['_source_path']),
                    approved_view=None, camera_dependent_images='pending explicit camera review')
    with pymol2.PyMOL() as pymol:
        selected = build_ensemble_session(pymol.cmd, selected, output, check_cancel=check, log=log)
    from rbp_representative_audit import verify_selected_ensembles
    selected['native_atom_inventory_audit'] = verify_selected_ensembles(selected, check=check)
    save(report, selected)
    # The successor is explicitly an export selection view, not a rerun checkpoint.
    save(Path(output)/'derived-manifest.json', dict(schema_version=1, kind='density-selected-ensembles',
        source_manifest=manifest['_source_path'], labels_csv=str(labels), report=str(report),
        display_projections_refitted=False, analysis_recovery_contract=False))
    return selected


def supplementary(manifest, selection, output, executable, check, log, *, feature_entry, names, scratch):
    from src.analysis.cluster_medoid_tm import calculate_medoid_tm
    from src.analysis.bonsai_interactive import run_ensemble_bonsai
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    need(not os.environ.get('USALIGN_OFFLOAD_MACHINE') and not os.environ.get('USALIGN_PARALLEL_WORKERS'),
         'Unplanned USalign parallel/offload override is forbidden')
    check(); grid = calculate_medoid_tm(selection, executable, workers=4, log=log, check_cancel=check)
    save(output/'medoid-tm.json', grid)
    selected = [row for group in selection['clusters'] for row in group['selected']]
    need(0 < len(selected) <= 3660, 'Selected Bonsai population exceeds frozen all-cluster envelope')
    lookup = {name:i for i, name in enumerate(names)}
    selected_names = [row['model'] for row in selected]
    need(len(set(selected_names)) == len(selected_names) and set(selected_names) <= set(names),
         'Bonsai selected identities differ from the frozen cohort')
    with saved_matrix(feature_entry, manifest, names, scratch) as saved:
        matrix = np.asarray(saved[[lookup[name] for name in selected_names]], dtype=np.float16)
    source = dict(kind='frozen_models_by_features', terminal_detection=manifest['terminal_detection'],
        dtype='float16', source_cache_sha256=feature_entry['sha256'], model_order_sha256=compact_sha(names),
        equivalence='Exact selected rows of the hash-verified saved complete Cα-distance cache; unchanged float16 values, model identities and residue/pair mask.')
    check(); tree = run_ensemble_bonsai(matrix, selected, output/'bonsai', identifier=TARGET, feature_source=source, log=log)
    save(output/'bonsai.json', tree); check()
    return dict(bonsai=record(output/'bonsai.json'), medoid_tm=record(output/'medoid-tm.json'))



def region_artifacts(manifest, selection, output, check):
    """Render the already saved full-cohort variance/cores; no new detection."""
    from src.analysis.domain_variance_figure import render_saved_domain_variance
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    check(); report = copy.deepcopy(selection)
    image = output/'annotated-variance.png'
    report['structural_flexibility']['heatmap'] = render_saved_domain_variance(
        manifest, report['structural_flexibility'], image)
    report_path = output/'ensemble_report.json'
    parent_manifest = manifest['_source_path']
    parent_report = manifest['cluster_ensembles']['report']
    proof = dict(kind='saved_variance_presentation',
        parent_workflow_manifest_sha256=sha256(parent_manifest),
        parent_ensemble_report_sha256=sha256(parent_report))
    report['presentation_derivation'] = proof
    save(report_path, report)
    association = copy.deepcopy(manifest); association.pop('_source_path')
    association['presentation_derivation'] = proof
    association['cluster_ensembles']['report'] = str(report_path)
    associated = output/'presentation-manifest.json'; save(associated, association); check()
    return dict(manifest=record(associated), ensemble_report=record(report_path), image=record(image))

def execute(plan_path):
    plan = validate_approval(plan_path); root = Path(plan['run_root'])
    need(root.is_dir() and not (root/'rbp').exists(), 'Fresh guarded attempt required')
    output = root/'rbp'; output.mkdir()
    started = time.monotonic()
    def check():
        need(time.monotonic()-started < plan['resources']['maximum_wall_seconds'], 'Approved wall-time cap reached')
        need(shutil.disk_usage(root).free >= plan['resources']['preserved_free_space_bytes'], 'Storage reserve reached')
        need(shutil.disk_usage(plan['environment']['TMPDIR']).free >= plan['scratch']['minimum_free_bytes_during_run'], 'RAM scratch reserve reached')
        need(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= plan['resources']['additional_storage_bytes'], 'Output disk envelope reached')
    # Pressure monitoring is the existing run_guarded parent, independent of native calls.
    # This local check is intentionally cheap except the occasional output-size audit.
    last_disk = [0.]
    full_check = check
    def check():
        need(time.monotonic()-started < plan['resources']['maximum_wall_seconds'], 'Approved wall-time cap reached')
        if time.monotonic()-last_disk[0] >= 30:
            full_check(); last_disk[0] = time.monotonic()
    log = lambda text: print(text, flush=True)
    study = read_bound(plan['study']); manifest = read_bound(plan['manifest']); manifest['_source_path'] = plan['manifest']['path']
    need(manifest['analysis_status'] == 'complete' and manifest['identifier'] == TARGET, 'Completed RBP analysis required')
    need(manifest['terminal_detection']['retained_reference_positions'] == list(range(271)), 'Frozen retained residue mask differs')
    names, members, primary = registry(plan, manifest, study, check)
    bindings = dict(plan_sha256=sha256(plan_path), consumer_commit=plan['sources']['consumer']['commit'],
                    library_commit=plan['sources']['library']['commit'], manifest=plan['manifest'],
                    native=plan['native'], feature_cache=plan['feature_cache'])
    from rbp_density import run as density_run
    from rbp_states import run_states
    from rbp_native_tm import NativeTMRun, frozen_source_evidence
    with frozen_features(plan['feature_cache'], manifest, names, plan['environment']['TMPDIR']) as (features, feature_sha):
        density = density_run(features, names, [r['cluster_label'] for r in primary], output/'density',
            native_source=plan['native_density'], mode='casp17_recursive', retained_positions=list(range(271)),
            expected_feature_sha256=feature_sha, workers=4, log=log, check=check)
        labels, assigned = write_density_labels(density, primary, output/'density')
        density.update(schema_version=1, status='complete', target=TARGET, algorithm='HDBSCAN',
            source_workflow_manifest_sha256=plan['manifest']['sha256'],
            model_order_sha256=compact_sha(names), model_count=len(names), method_note=METHOD_NOTE)
        # Preserve the native adapter's original full ordered `models` array.
        save(output/'density'/'publication-analysis.json', density)
    density_selection = prepare_density_ensembles(manifest, assigned, output/'density-ensembles', check, log)
    need(density_selection['total_cluster_count'] <= 732, 'Cluster envelope exceeded')
    primary_selection = json.loads(Path(manifest['cluster_ensembles']['report']).read_text())
    supplemental = {key:supplementary(manifest, selected, output/(key+'-supplementary'),
        plan['native_tm_runtime']['usalign'], check, log, feature_entry=plan['feature_cache'],
        names=names, scratch=plan['environment']['TMPDIR']) for key, selected in
        [('primary', primary_selection), ('density', density_selection)]}
    # Release density staging during wide selected-model Bonsai PCA, then recast
    # the same verified cache for states; no display projection is fitted again.
    with frozen_features(plan['feature_cache'], manifest, names, plan['environment']['TMPDIR']) as (features, feature_sha):
        cohort = dict(target=TARGET, range_id='all', models=names, methods=[r['method'] for r in members],
            energies=[r['energy'] for r in members], sequence_sha256=study['targets'][TARGET]['sequence_sha256'],
            construct_residues=271, conditioning_t=.1, energy_scope='full_construct')
        clustering = dict(primary_labels=[r['cluster_label'] for r in primary], hdbscan_labels=density['density'],
            hdbscan_noise_labels=sorted({s for s in density['density'] if s.endswith('.unassigned')}),
            clustering_positions=list(range(1,272)),
            saved_source_views={key:record(manifest['clustering'][key]['embeddings']) for key in ('pca','umap','tsne')})
        native_bindings = dict(bindings, sources=frozen_source_evidence())
        with NativeTMRun(members, list(range(271)), cohort['sequence_sha256'], output/'native-tm',
                         runtime=plan['native_tm_runtime'], source_bindings=native_bindings, check=check) as scorer:
            scorer.prepare()
            states = run_states(cohort, clustering, features, list(range(271)), output/'states', scorer,
                                source_bindings=native_bindings, seed=4200, workers=4, check=check)
    regions = region_artifacts(manifest, primary_selection, output/'regions', check)
    from rbp_panel_exports import export_catalog
    panels = export_catalog(output/'states'/'catalog.json', members, output/'state-panels',
        expected_catalog_sha256=sha256(output/'states'/'catalog.json'), check=check, log=log)
    density_receipt = dict(labels_csv=record(labels), ensemble_report=record(density_selection['report']),
        analysis=record(output/'density'/'publication-analysis.json'), **supplemental['density'])
    save(output/'artifacts.json', dict(schema_version=1, target=TARGET, density_artifacts=density_receipt,
        primary_artifacts=supplemental['primary'], region_artifacts=regions, state_catalog=record(output/'states'/'catalog.json'),
        state_panels=record(output/'state-panels'/'catalog.json')))
    check(); save(root/'rbp-parity-completion.json', dict(status='rbp_parity_science_and_saved_exports_complete',
        plan_sha256=sha256(plan_path), model_count=len(names), source_bindings=bindings,
        density_clusters=density['structural_clusters'], density_unassigned=density['unassigned_models'],
        UMAP_fits=density['UMAP_fits'], HDBSCAN_extractions=density['HDBSCAN_extractions'],
        Bonsai_PCA_fits=2, states=states['summary'], artifacts=record(output/'artifacts.json'),
        camera_dependent_images='pending explicit camera review', public_deployment=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('plan', type=Path)
    execute(parser.parse_args().plan)
