"""Prepare the canonical shared atlas from completed exact-cohort workflows.

No projection fitting or representative selection occurs in publication. Native
coordinate derivatives/archives require their own approved publication scope.
"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from snapshot import sha256


def checked_json(path, digest):
    if sha256(path) != digest: raise ValueError('Frozen saved artifact changed: '+str(path))
    return json.loads(Path(path).read_text())


def projections(manifest, expected):
    import numpy as np
    if not all(method in manifest.get('clustering', {}) for method in ['pca', 'umap', 'tsne']):
        raise ValueError('All three real saved projections are required before publication')
    names, result = None, {}
    for method in ['pca', 'umap', 'tsne']:
        with np.load(manifest['clustering'][method]['embeddings'], allow_pickle=False) as saved:
            current = saved['filenames'].tolist(); xy = saved['embeddings']
            if (set(current) != set(expected) or len(current) != len(expected)
                    or names is not None and names != current or xy.shape != (len(expected), 2)
                    or not np.isfinite(xy).all()):
                raise ValueError('All three real saved projections must match the complete accepted cohort and order')
            names = current
            result[method] = {'xy': xy.tolist(), 'axes': [method.upper()+' 1', method.upper()+' 2']}
            if method == 'pca' and 'explained_variance_ratio' in saved:
                ratios = saved['explained_variance_ratio'].tolist()
                result[method].update(explained_variance_ratio=ratios,
                    axes=[f'PC {i+1} ({100*v:.1f}% variance)' for i, v in enumerate(ratios[:2])])
    return names, result


def continuous(key, name, values):
    finite = [x for x in values if x is not None and math.isfinite(x)]
    return {'key': key, 'name': name, 'type': 'continuous', 'values': values, 'colorscale': 'Viridis',
            'range': [min(finite), max(finite)] if finite else [0, 1], 'unavailable': not finite}


def publication_scope(plan, study):
    shown = set(plan['targets'])
    raw_pending = plan.get('pending_targets', [])
    pending = set(raw_pending)
    if (not shown or len(raw_pending) != len(pending) or shown & pending
            or shown | pending != set(study['targets'])):
        raise ValueError('Publication must explicitly account for every target as saved or pending')
    return list(plan['targets']), raw_pending


def structure_delivery(plan):
    """Keep the full plotted cohort while limiting published coordinate assets."""
    mode = plan.get('structure_delivery', 'representatives')
    if mode not in {'representatives', 'all-point-ca'}:
        raise ValueError('Unknown structure delivery mode')
    return mode


def public_bonsai(tree):
    """Carry saved geometry and scientific evidence without private runtime paths."""
    result = {key: tree[key] for key in ['schema_version', 'models', 'layouts', 'method', 'scope', 'group_count']}
    proof = tree['provenance']
    evidence = {key: proof[key] for key in ['result_sha256', 'hdf_sha256', 'tree_sha256',
        'model_order_sha256', 'source_features_sha256', 'retained_reference_positions',
        'pair_mask_sha256', 'feature_count']}
    settings = proof['settings']
    evidence['settings'] = {key: settings[key] for key in ['schema_version', 'max_components',
        'rescale_by_var', 'use_knn', 'grouping', 'terminal_settings'] if key in settings}
    runtime = settings.get('native_runtime_compatibility')
    if runtime:
        public_runtime = {key: runtime[key] for key in ['schema_version', 'repair',
            'native_helper_sha256', 'implementation_sha256', 'external_files_modified']}
        public_runtime['native_modules'] = [{key: row[key] for key in ['module',
            'source_sha256', 'runtime_source_sha256', 'tracked_patch_sha256']}
            for row in runtime['native_modules']]
        evidence['settings']['native_runtime_compatibility'] = public_runtime
    result['provenance'] = evidence
    return result


def ordered_models_sha256(names):
    return hashlib.sha256(json.dumps(names, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def memberships(path, names):
    with Path(path).open() as stream: rows = list(csv.DictReader(stream))
    labels = {row['model']: row for row in rows}
    if len(labels) != len(rows) or set(labels) != set(names):
        raise ValueError('Cluster memberships differ from projections')
    colors = {}
    for row in rows:
        label, color = row['cluster_label'], row['color']
        if not label or (label in colors and colors[label] != color):
            raise ValueError('Cluster labels have missing or inconsistent colors')
        colors[label] = color
    return labels, colors


def saved_views(frozen, manifest, target, names):
    primary = {'id': 'primary' if frozen.get('density_artifacts') else 'whole',
               'name': 'UMAP KNN', 'data': 'primary.json',
               'labels_csv': manifest['knn_clustering']['labels_csv'],
               'ensemble_report': manifest['cluster_ensembles']['report'],
               'method_note': 'Default seed42 PCA/UMAP/t-SNE; common component-preserving UMAP partition, feature silhouette ≥0.15 and repeat-seed ARI ≥0.80. All three projections retain the exact same model order and memberships.',
               'bonsai': frozen.get('bonsai'), 'medoid_tm': frozen.get('medoid_tm')}
    views = [primary]
    if frozen.get('density_artifacts'):
        artifacts = frozen['density_artifacts']
        analysis = checked_json(artifacts['analysis']['path'], artifacts['analysis']['sha256'])
        if (analysis.get('schema_version') != 1 or analysis.get('status') != 'complete'
                or analysis.get('target') != target or analysis.get('algorithm') != 'HDBSCAN'
                or analysis.get('source_workflow_manifest_sha256') != frozen['manifest_sha256']
                or analysis.get('model_order_sha256') != ordered_models_sha256(names)
                or analysis.get('model_count') != len(names) or analysis.get('models') != names
                or not isinstance(analysis.get('method_note'), str) or not analysis['method_note'].strip()):
            raise ValueError('Saved HDBSCAN analysis is not bound to this complete workflow and model order')
        for key in ['labels_csv', 'ensemble_report']:
            entry = artifacts[key]
            if sha256(entry['path']) != entry['sha256']:
                raise ValueError('Frozen HDBSCAN artifact changed: '+key)
        labels, _ = memberships(artifacts['labels_csv']['path'], names)
        if analysis.get('density') != [labels[name]['cluster_label'] for name in names]:
            raise ValueError('Saved HDBSCAN labels differ from the completed analysis')
        views.append({'id': 'density', 'name': 'HDBSCAN', 'data': 'density.json',
                      'labels_csv': artifacts['labels_csv']['path'],
                      'ensemble_report': artifacts['ensemble_report']['path'],
                      'method_note': analysis['method_note'], 'analysis_sha256': artifacts['analysis']['sha256'],
                      'bonsai': artifacts.get('bonsai'), 'medoid_tm': artifacts.get('medoid_tm')})
    return views


def copy_representatives(report, labels, expected, method_map, folder, source):
    if report.get('residue_correspondence', {}).get('native_residue_labels_preserved') is not True:
        raise ValueError('Native residue correspondence has not been validated for representatives')
    assigned = {row['cluster_label'] for row in labels.values() if not row['cluster_label'].endswith('.unassigned')}
    groups = [group for group in report['clusters'] if not group['cluster_label'].endswith('.unassigned')]
    if (len({group['cluster_label'] for group in groups}) != len(groups)
            or {group['cluster_label'] for group in groups} != assigned):
        raise ValueError('Saved representative groups differ from assigned cluster labels')
    folder.mkdir(parents=True)
    counts = Counter(row['cluster_label'] for row in labels.values())
    representatives = []
    for group in groups:
        label = group['cluster_label']
        if not group['selected'] or group['cluster_size'] != counts[label]:
            raise ValueError('Saved representative group has wrong membership or no selection')
        for selected in group['selected']:
            name = selected['model']
            if (name not in expected or labels[name]['cluster_label'] != label
                    or sha256(selected['source_path']) != expected[name]['sha256']):
                raise ValueError('Representative changed or belongs to a different cluster')
        selected = group['selected'][0]; name = selected['model']; original = Path(selected['source_path'])
        destination = folder/original.name; shutil.copyfile(original, destination)
        representative = {'cluster': label, 'count': group['cluster_size'], 'model': name,
            'method': method_map[expected[name]['method']]['name'], 'color': group['color'],
            'file': str(destination.relative_to(source)), 'format': 'cif' if original.suffix.lower() in ['.cif', '.mmcif'] else 'pdb',
            'sha256': expected[name]['sha256'], 'structure_confidence': None}
        ensemble = Path(group['multimodel_structure']); ensemble_destination = folder/ensemble.name
        shutil.copyfile(ensemble, ensemble_destination)
        representative.update(ensemble_file=str(ensemble_destination.relative_to(source)),
            ensemble_format='cif', ensemble_structure_confidence=None)
        metadata = {'cluster': label, 'ensemble_sha256': sha256(ensemble),
            'alignment_ca_keys': group.get('alignment_ca_keys', group.get('shared_ca_keys')),
            'selection_method': group['selection_method'], 'confidence_metric': report['confidence_metric'],
            'states': [{'model': row['model'], 'state': row['state'], 'rank': row['rank'], 'confidence': row['confidence'],
                'aligned_ca_rmsd': row['aligned_ca_rmsd'], 'native_sha256': expected[row['model']]['sha256'],
                'ca_correspondence': row.get('ca_correspondence')} for row in group['selected']]}
        metadata_file = ensemble_destination.with_suffix('.json')
        metadata_file.write_text(json.dumps(metadata, indent=2)+'\n')
        representative['ensemble_metadata'] = {'file': str(metadata_file.relative_to(source))}
        representatives.append(representative)
    return representatives


def add_supplementary(data, view, folder, source, names, *, selected_models=None, retained_reference_positions=None, strict=False):
    prefix = '' if view['id'] in {'whole', 'primary'} else view['id']+'-'
    if view.get('medoid_tm'):
        entry = view['medoid_tm']; grid = checked_json(entry['path'], entry['sha256'])
        matrix, grid_labels = grid['matrix'], grid['labels']
        if (len(set(grid_labels)) != len(grid_labels)
                or set(grid_labels) != {row['cluster'] for row in data['representatives']}
                or len(matrix) != len(grid_labels) or any(len(row) != len(grid_labels) for row in matrix)
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for row in matrix for v in row)
                or any(matrix[i][j] != matrix[j][i] for i in range(len(matrix)) for j in range(len(matrix)))):
            raise ValueError('Saved medoid TM grid differs from published assigned cluster identities')
        csv_file = folder/(prefix+'medoid-tm.csv')
        with csv_file.open('x', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['cluster', *grid_labels])
            for label, row in zip(grid_labels, matrix): writer.writerow([label, *row])
        data['medoid_tm'] = {'labels': grid_labels, 'matrix': matrix, 'csv': str(csv_file.relative_to(source)),
            'title': 'TM similarity across selected-ensemble medoids',
            'note': 'Exact saved bidirectional, average-length-normalized US-align scores. Selected-ensemble medoids are distinct from full-cluster medoids.'}
        data['provenance']['medoid_tm_export_sha256'] = entry['sha256']
    if view.get('bonsai'):
        entry = view['bonsai']; tree = checked_json(entry['path'], entry['sha256'])
        tree_names = [row['model'] for row in tree['models']]
        if len(set(tree_names)) != len(tree_names) or not set(tree_names).issubset(names):
            raise ValueError('Saved Bonsai contains different model identities')
        if strict:
            proof = tree['provenance']
            if (not isinstance(selected_models, list) or len(set(selected_models)) != len(selected_models)
                    or len(tree_names) != len(selected_models) or set(tree_names) != set(selected_models)
                    or proof.get('retained_reference_positions') != retained_reference_positions
                    or proof.get('pair_mask_sha256') != data['provenance']['pair_mask_sha256']):
                raise ValueError('Saved Bonsai population or residue mask differs from this selected view')
        tree_file = folder/(prefix+'bonsai.json')
        tree_file.write_text(json.dumps(public_bonsai(tree), separators=(',', ':'), allow_nan=False)+'\n')
        data['bonsai'] = {'file': str(tree_file.relative_to(source)), 'sha256': sha256(tree_file), 'selected_models': len(tree_names)}
        data['provenance']['bonsai_native_export_sha256'] = entry['sha256']
        data['provenance']['bonsai_scope'] = tree['scope']


def report_downloads(artifacts, model_count, view_id, *, strict=False):
    downloads = []
    public_view = 'primary' if view_id == 'whole' else view_id
    for artifact in artifacts:
        selected = artifact.get('view_ids')
        if strict or selected is not None:
            if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                    or not set(selected) <= {'primary', 'density'}):
                raise ValueError('Report downloads require explicit valid Primary/HDBSCAN view_ids')
        if artifact['models'] != model_count or sha256(artifact['file']) != artifact['sha256']:
            raise ValueError('Report artifact does not match the published cohort')
        if selected is not None and public_view not in selected:
            continue
        download = {key: artifact[key] for key in ['label', 'url', 'sha256', 'models']}
        if selected is not None: download['view_ids'] = selected
        downloads.append(download)
    return downloads


def saved_region_scope(frozen, manifest, count, names, sequence_sha256, folder, source):
    """Verify a presentation-only copy of saved full-cohort variance evidence."""
    from src.analysis.atlas_residue_scope import _regions
    entries = frozen.get('region_artifacts')
    if not entries:
        return {}, None
    derived = checked_json(entries['manifest']['path'], entries['manifest']['sha256'])
    report = checked_json(entries['ensemble_report']['path'], entries['ensemble_report']['sha256'])
    original_report_path = Path(manifest['cluster_ensembles']['report'])
    original_report = json.loads(original_report_path.read_text())
    proof = {'kind': 'saved_variance_presentation', 'parent_workflow_manifest_sha256': frozen['manifest_sha256'],
             'parent_ensemble_report_sha256': sha256(original_report_path)}
    expected_manifest = copy.deepcopy(manifest)
    expected_manifest['cluster_ensembles']['report'] = entries['ensemble_report']['path']
    expected_manifest['presentation_derivation'] = proof
    expected_report = copy.deepcopy(original_report)
    try:
        expected_report['structural_flexibility']['heatmap'] = report['structural_flexibility']['heatmap']
    except KeyError:
        raise ValueError('Saved region presentation lacks authoritative variance heatmap metadata') from None
    expected_report['presentation_derivation'] = proof
    if derived != expected_manifest or report != expected_report:
        raise ValueError('Saved region presentation changed frozen scientific results or parent proof')
    retained = manifest['terminal_detection']['retained_reference_positions']
    regions, evidence = _regions(Path(entries['manifest']['path']), derived,
        entries['ensemble_report']['path'], count, retained, sequence_sha256)
    if not regions or evidence['model_count'] != len(names):
        raise ValueError('Saved variance cores must identify the complete published source cohort')
    image = entries['image']
    if sha256(image['path']) != image['sha256']:
        raise ValueError('Saved annotated variance figure changed')
    destination = folder/'figures'/'saved-variance-cores.png'; destination.parent.mkdir(exist_ok=True)
    shutil.copyfile(image['path'], destination)
    evidence.update(presentation_manifest_sha256=entries['manifest']['sha256'],
        parent_workflow_manifest_sha256=frozen['manifest_sha256'],
        parent_ensemble_report_sha256=proof['parent_ensemble_report_sha256'], image_sha256=image['sha256'])
    scope = {'regions': regions, 'regions_verified': True, 'region_provenance': evidence,
        'regions_source_models': evidence['model_count'], 'region_residue_count': evidence['region_residue_count'],
        'regions_note': f"Saved empirical variance cores were inferred from all {len(names):,} published models over {evidence['region_residue_count']} supplied construct residues. These are not biological domain or TM-state annotations; unassigned residues remain gray."}
    return scope, {'name': 'Saved full-cohort distance variance and detected rigid cores',
                   'file': str(destination.relative_to(source))}


def refresh_manifest(prepared):
    """Refresh the allowlist after a caller attaches hash-verified saved panels."""
    source, manifest = Path(prepared['source']), Path(prepared['publication_manifest'])
    publication = json.loads(manifest.read_text())
    paths = sorted(source.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('Publication source must not contain symbolic links')
    publication['files'] = [{'path': str(path.relative_to(source)), 'bytes': path.stat().st_size,
                             'sha256': sha256(path)} for path in paths if path.is_file()]
    manifest.write_text(json.dumps(publication, indent=2)+'\n')
    return manifest


def export_prepared(prepared):
    from structure_clustering.publication import export
    manifest = refresh_manifest(prepared)
    export(manifest, Path(prepared['source']), Path(prepared['output']))


def build(plan_path, *, export_site=True):
    from native_ca import write_ca
    from src.analysis.atlas_presentation import renderer_identity
    plan = json.loads(Path(plan_path).read_text())
    delivery = structure_delivery(plan)
    renderer = renderer_identity()
    if renderer['revision'] != plan['renderer_commit'] or renderer['uncommitted_source'] is not False:
        raise ValueError('Publication renderer differs from its exact reviewed source')
    for path, digest in plan.get('frozen_files', {}).items():
        if sha256(path) != digest: raise ValueError('Selected publication input changed')
    if set(plan['vendor_assets']) != {'plotly.min.js', '3Dmol-min.js'}:
        raise ValueError('Canonical atlas requires the two selected vendor assets')
    study = checked_json(plan['study']['path'], plan['study']['sha256'])
    native = checked_json(plan['native']['path'], plan['native']['sha256'])
    confidence = checked_json(plan['confidence']['path'], plan['confidence']['sha256'])
    if (native['status'] != 'validated' or confidence['status'] != 'validated'
            or plan['native']['sha256'] != study['snapshot']['manifest_sha256']
            or confidence['source_manifest_sha256'] != plan['native']['sha256']):
        raise ValueError('Publication requires the accepted scored snapshot and exact native confidence binding')
    shown, pending = publication_scope(plan, study)
    output, source = Path(plan['output']), Path(plan['source'])
    if output.exists() or source.exists(): raise FileExistsError('Use new immutable publication directories')
    source.mkdir(parents=True); (source/'assets').mkdir()
    for name, entry in plan['vendor_assets'].items():
        if sha256(entry['path']) != entry['sha256']: raise ValueError('Vendor asset changed')
        shutil.copyfile(entry['path'], source/'assets'/name)
    by_native = {row['model_id']: row for row in native['models']}
    by_confidence = {row['model_id']: row for row in confidence['models']}
    if len(by_confidence) != len(confidence['models']) or set(by_confidence) != set(by_native):
        raise ValueError('Confidence identities differ from the accepted cohort')
    rules = json.loads((Path(plan['study']['path']).parent/'model-types.json').read_text())['rules']
    method_map = {row['key']: row for row in rules}
    atlas = {'targets': [], 'default_target': shown[0], 'pending_targets': [
        {'id': name, 'name': study['targets'][name]['title'], 'status': 'Analysis pending; no projections published yet.'}
        for name in pending]}
    scopes = {'schema_version': 1, 'targets': {}}
    ranges = {'schema_version': 1, 'targets': {}}
    bindings = {}
    for target, frozen in plan['targets'].items():
        manifest = checked_json(frozen['manifest'], frozen['manifest_sha256'])
        if manifest['analysis_status'] != 'complete' or manifest['identifier'] != target:
            raise ValueError('Real clustering has not completed for '+target)
        expected = {name: row for name, row in by_native.items() if row['target'] == target}
        if len(expected) != study['targets'][target]['models']: raise ValueError('Wrong target snapshot count')
        names, xy = projections(manifest, expected)
        views = saved_views(frozen, manifest, target, names)
        folder = source/target; folder.mkdir()
        primary_labels, _ = memberships(views[0]['labels_csv'], names)
        model_files, display_hashes, native_hashes = [], [], []
        if delivery == 'all-point-ca': (folder/'models').mkdir()
        for index, name in enumerate(names):
            row, score = expected[name], by_confidence[name]
            if score['coordinate_sha256'] != row['sha256']: raise ValueError('Wrong confidence/coordinate join')
            if delivery == 'all-point-ca':
                destination = folder/'models'/(name+'.cif.gz')
                display = write_ca(primary_labels[name]['source_path'], destination, native_sha256=row['sha256'],
                                   expected_count=study['targets'][target]['residues'])
                model_files.append(str(destination.relative_to(source))); display_hashes.append(display['sha256']); native_hashes.append(row['sha256'])
                if (index+1) % 500 == 0: print(f'{target}: packaged {index+1}/{len(names)} exact native Cα traces', flush=True)
        public_archive = None
        if delivery == 'all-point-ca':
            archive = frozen.get('original_archive')
            if (not archive or archive['models'] != len(names) or not str(archive['url']).startswith('https://')
                    or sha256(archive['file']) != archive['sha256']):
                raise ValueError('A verified complete original archive is required for each compact target view')
            inventory = checked_json(archive['inventory'], archive['inventory_sha256'])
            if (inventory['native_manifest_sha256'] != plan['native']['sha256']
                    or len(inventory['models']) != len(expected)
                    or {row['model']: row['sha256'] for row in inventory['models']} != {name: row['sha256'] for name, row in expected.items()}):
                raise ValueError('Original archive inventory differs from accepted native coordinates')
            public_archive = {key: archive[key] for key in ['url', 'sha256', 'bytes', 'models']}
        detection = manifest['terminal_detection']; count = study['targets'][target]['residues']
        positions = [index+1 for index in detection['retained_reference_positions']]
        scope = {'chain': 'A', 'sequence_sha256': study['targets'][target]['sequence_sha256'], 'expected_ca_count': count,
                 'canonical_positions': positions, 'analysis_uses_full_construct': not detection['excluded_indices'],
                 'note': 'Canonical positions here index the supplied construct, not full-length UniProt; native file labels are preserved.',
                 'regions_verified': False, 'reference_ranges': f'construct positions 1–{count}'}
        region_scope, region_figure = saved_region_scope(frozen, manifest, count, names,
            study['targets'][target]['sequence_sha256'], folder, source)
        scope.update(region_scope)
        scopes['targets'][target] = {'views': {view['id']: scope for view in views}}
        method_names = [method_map[expected[name]['method']]['name'] for name in names]
        flags = [expected[name]['native_physicality'] for name in names]
        flag_colors = {'retained': '#228833', 'rejected': '#cc3311', 'unavailable': '#bbbbbb', 'see original100-frame filtering metadata': '#ddcc77'}
        heatmaps = []
        for image in manifest['visualize_distogram']['images']:
            path = Path(image); destination = folder/'figures'/path.name; destination.parent.mkdir(exist_ok=True); shutil.copyfile(path, destination)
            heatmaps.append({'name': path.stem.replace('_', ' ').capitalize(), 'file': str(destination.relative_to(source))})
        if region_figure: heatmaps.append(region_figure)
        view_links = [{'id': view['id'], 'name': view['name'], 'data': target+'/'+view['data']} for view in views]
        target_bindings = {}
        for view in views:
            labels, colors = memberships(view['labels_csv'], names)
            report = json.loads(Path(view['ensemble_report']).read_text())
            rep_folder = folder/'representatives'/(view['id'] if len(views) > 1 else '')
            representatives = copy_representatives(report, labels, expected, method_map, rep_folder, source)
            cluster_values = [labels[name]['cluster_label'] for name in names]
            schemes = [{'key': 'clusters', 'name': 'Clusters', 'type': 'categorical', 'values': cluster_values, 'colors': colors},
                {'key': 'methods', 'name': 'Model origin', 'type': 'categorical', 'values': method_names, 'colors': {rule['name']: rule['color'] for rule in rules}},
                continuous('proteinebm_t010', 'ProteinEBM energy · t=0.10', [expected[name]['energy_t010'] for name in names]),
                continuous('mean_plddt', 'Native mean pLDDT', [by_confidence[name]['mean_plddt'] for name in names]),
                continuous('ptm', 'Native pTM', [by_confidence[name]['ptm'] for name in names]),
                {'key': 'native_physicality', 'name': 'Native physicality flags', 'type': 'categorical', 'values': flags, 'colors': flag_colors}]
            prefix = '' if view['id'] in {'whole', 'primary'} else view['id']+'-'
            membership = folder/(prefix+'memberships-and-scores.csv')
            with membership.open('x', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['model', 'method', 'cluster', 'native_physicality', 'native_sha256', 'mean_plddt', 'ptm', *[f'energy_t{i:02d}' for i in range(16)]])
                for name in names:
                    row, score = expected[name], by_confidence[name]
                    writer.writerow([name, row['method'], labels[name]['cluster_label'], row['native_physicality'], row['sha256'], score['mean_plddt'], score['ptm'], *[row[f'energy_t{i:02d}'] for i in range(16)]])
            data = {'target': target, 'view_id': view['id'], 'view_title': study['targets'][target]['title']+' · '+view['name'],
                'models': names, 'residues': count, 'methods': method_names, 'clusters': cluster_values, 'projections': xy, 'schemes': schemes,
                'representatives': representatives, 'views': view_links, 'membership_file': str(membership.relative_to(source)), 'heatmaps': heatmaps,
                'coordinate_delivery': delivery, 'representative_kind': 'confidence-ranked representative or unscored structural medoid',
                'representative_policy': ('Plot and Bonsai point selections show the corresponding cluster representative. Only cluster representative and saved state-panel coordinates are published. ' if delivery == 'representatives' else '')+'Highest genuine native confidence among the default selected ensemble; unscored clusters use the recorded structural medoid. Energy is not confidence. Unassigned density points remain plotted and have no cluster representative.',
                'mask_note': f'All {count} supplied construct residues retained; native numbering preserved. Biological disorder is not inferred.',
                'clustering_method_note': view['method_note'],
                'provenance': {'workflow_manifest_sha256': frozen['manifest_sha256'], 'native_manifest_sha256': plan['native']['sha256'],
                    'sequence_sha256': study['targets'][target]['sequence_sha256'], 'pair_mask_sha256': detection['pair_mask_sha256'],
                    'confidence_sha256': plan['confidence']['sha256'], 'scientific_library_commit': study['library_commit'], 'renderer_commit': plan['renderer_commit'],
                    'labels_csv_sha256': sha256(view['labels_csv']), 'ensemble_report_sha256': sha256(view['ensemble_report']),
                    'energy_units': 'learned arbitrary units; lower within target; not confidence or calibrated free energy'}}
            if view.get('analysis_sha256'): data['provenance']['density_analysis_sha256'] = view['analysis_sha256']
            if delivery == 'all-point-ca':
                data.update(model_files=model_files, model_sha256=display_hashes, model_native_sha256=native_hashes,
                    model_representation='ca_trace', model_structure_confidence=[None]*len(names), model_physicality=flags, original_archive=public_archive)
            if frozen.get('reports'):
                data['report_downloads'] = report_downloads(frozen['reports'], len(names), view['id'],
                    strict=plan.get('require_rbp_parity', False))
            selected_models = [row['model'] for group in report['clusters']
                if not group['cluster_label'].endswith('.unassigned') for row in group['selected']]
            add_supplementary(data, view, folder, source, names, selected_models=selected_models,
                retained_reference_positions=detection['retained_reference_positions'],
                strict=plan.get('require_rbp_parity', False))
            data_file = folder/view['data']
            data_file.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False)+'\n')
            target_bindings[view['id']] = {'id': view['id'], 'name': view['name'], 'data': target+'/'+view['data'],
                'asset_root': '', 'source_view_id': view['id'], 'source_data_sha256': sha256(data_file),
                'canonical_positions': positions, 'available': True}
        if len(views) > 1:
            ranges['targets'][target] = {'residue_ranges': [{'id': 'all', 'name': 'All residues', 'data': target+'/primary.json',
                'asset_root': '', 'default_view': 'density', 'available': True, 'views': list(target_bindings.values())}]}
        bindings[target] = {'models': names, 'model_count': len(names), 'model_order_sha256': ordered_models_sha256(names),
            'sequence_sha256': study['targets'][target]['sequence_sha256'], 'clustering_positions': positions,
            'scope': scope, 'coordinate_delivery': delivery, 'views': target_bindings}
        atlas['targets'].append({'id': target, 'name': study['targets'][target]['title'], 'data': target+'/primary.json'})
    context_entry = plan['construct_context']
    context_source = checked_json(context_entry['path'], context_entry['sha256'])
    context = {'schema_version': 1, 'verified_on': context_source['retrieved_at_utc'].split('T')[0], 'targets': {}}
    for target, record in context_source['targets'].items():
        if target not in plan['targets']: continue
        start, end = record['modeled_canonical_range']
        if end-start+1 != study['targets'][target]['residues']:
            raise ValueError('Canonical construct context differs from supplied model length')
        context['targets'][target] = {'title': study['targets'][target]['title']+' · '+record['protein'], 'organism': record['organism'],
            'record_summary': f'Supplied construct: UniProt residues {start}–{end} of {record["canonical_length"]}. Atlas residue positions index the supplied construct; native file labels remain intact.',
            'provenance': {'analysis_sequence_sha256': study['targets'][target]['sequence_sha256']},
            'references': [{'label': 'UniProt reference sequence and annotation', 'url': record['source']}],
            'project_writeup': {'status': 'Accepted model snapshot; exploratory clustering', 'biological_relevance': None,
                'ensemble_interpretation': 'Cluster populations describe generated models, not equilibrium state probabilities. Conservative screening retained every supplied construct residue; this does not establish biological order.',
                'project_citations': [], 'project_record_url': 'https://github.com/benavt/domenico-structure-atlas'}}
    for name, payload in [('target-context.json', context), ('atlas.json', atlas), ('clustering-residues.json', scopes)]:
        (source/name).write_text(json.dumps(payload, indent=2)+'\n')
    if ranges['targets']: (source/'residue-ranges.json').write_text(json.dumps(ranges, indent=2)+'\n')
    shown_text = ', '.join(study['targets'][name]['title']+' ('+format(study['targets'][name]['models'], ',')+' models)' for name in shown)
    pending_text = ', '.join(study['targets'][name]['title'] for name in pending)
    scope_note = 'Published now: '+shown_text+'. '+('Pending analyses, not present in the projection selector: '+pending_text+'. ' if pending else '')
    title = 'Domenico structure atlas'+(' · '+', '.join(study['targets'][name]['title'] for name in shown)+' ready' if pending else '')
    title = plan.get('presentation_title', title)
    coordinate_note = ('All projection points are retained; structure selections show cluster representatives. Coordinate files are limited to representatives and explicitly saved state panels. '
        if delivery == 'representatives' else 'Point views are exact native Cα traces; cluster representatives retain full atoms. ')
    clustering_note = ('Saved recursive HDBSCAN is the default label view where available, alongside the original validated UMAP partition. '
        if ranges['targets'] else 'Default validated common cluster partition. ')
    publication = {'schema_version': 1, 'project': 'domenico-structure-atlas', 'renderer': 'ais-explorer-v1',
        'presentation': {'title': title, 'methods': scope_note+'Accepted fixed scored snapshot. Full-construct Cα distances. '+clustering_note+'Genuine native confidence and separate saved ProteinEBM energies. '+coordinate_note+'Native rejection flags never filter the cohort. Camera-dependent reports are delivered after review.'}, 'files': []}
    manifest = source.parent/'publication-manifest.json'; manifest.write_text(json.dumps(publication, indent=2)+'\n')
    prepared = {'source': str(source), 'output': str(output), 'publication_manifest': str(manifest),
        'coordinate_delivery': delivery, 'target_views': bindings, 'renderer_commit': plan['renderer_commit']}
    refresh_manifest(prepared)
    if export_site:
        export_prepared(prepared)
        print('Canonical shared atlas prepared:', output)
    else:
        print('Canonical shared atlas source prepared:', source)
    return prepared


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('plan', type=Path)
    build(parser.parse_args().plan)
