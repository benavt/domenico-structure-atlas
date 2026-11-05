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
import re
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


def residue_presentation(detection, count):
    """Describe the saved canonical mask without inferring or changing its scope."""
    retained = detection.get('retained_reference_positions')
    if (type(count) is not int or count < 3 or not isinstance(retained, list)
            or len(retained) < 3 or any(type(i) is not int or not 0 <= i < count for i in retained)
            or retained != sorted(set(retained))):
        raise ValueError('Saved retained residue positions must uniquely index the supplied construct')
    excluded = sorted(set(range(count))-set(retained))
    if 'excluded_indices' in detection and detection['excluded_indices'] != excluded:
        raise ValueError('Saved excluded residues disagree with the retained mask')
    positions = [i+1 for i in retained]
    segments = []
    for position in positions:
        if segments and position == segments[-1][1]+1:
            segments[-1][1] = position
        else:
            segments.append([position, position])
    ranges = ', '.join(str(first) if first == last else f'{first}–{last}' for first, last in segments)
    full = len(retained) == count
    return dict(positions=positions, full=full, ranges=ranges,
        reference_ranges='construct positions '+ranges,
        range_name='All residues' if full else f'Clustering core · {ranges} ({len(retained)}/{count} residues)',
        mask_note=(f'All {count} supplied construct residues retained; native numbering preserved. Biological disorder is not inferred.'
            if full else f'{len(retained)} of {count} supplied construct residues retained for clustering (construct positions {ranges}); '
                f'{len(excluded)} residues excluded. Native numbering preserved. Biological disorder is not inferred.'),
        ensemble_interpretation=('Cluster populations describe generated models, not equilibrium state probabilities. '
            + ('Conservative screening retained every supplied construct residue; this does not establish biological order.'
               if full else f'Clustering uses {len(retained)} of {count} supplied construct residues (construct positions {ranges}). '
                    'Retained ranges reflect saved screening evidence; biological order is not inferred.')))


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


def density_diagnostics(analysis, labels, names, source_sha256, *, strict=False):
    """Read compact saved quality evidence without exposing paths or refitting."""
    required = {'mode', 'inference_settings', 'stopping', 'history', 'selections',
                'structural_clusters', 'unassigned_models', 'native_source',
                'selected_fits', 'UMAP_fits', 'HDBSCAN_extractions'}
    present = required.intersection(analysis)
    if not present and not strict:
        return {'status': 'legacy_not_recorded', 'source_analysis_sha256': source_sha256,
                'note': 'This legacy saved density view did not record repeat-seed diagnostics.'}
    def require(value, message):
        if not value: raise ValueError('Saved density diagnostics: '+message)
    def integer(value, limit):
        return type(value) is int and 0 <= value <= limit
    def number(value, low, high):
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    def digest(value):
        return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
    def population(values):
        require(isinstance(values, list) and len(values) == len(names)
                and all(isinstance(label, str) and re.fullmatch('[A-Za-z0-9_.-]+', label) for label in values),
                'invalid full-cohort round labels')
        counts = Counter(values)
        return counts, sum(value for label, value in counts.items() if label.endswith('.unassigned'))
    require(present == required, 'complete native evidence is required')
    n = len(names); final = [labels[name]['cluster_label'] for name in names]
    counts, unassigned = population(final)
    structural = sum(not label.endswith('.unassigned') for label in counts)
    require(analysis['structural_clusters'] == structural and analysis['unassigned_models'] == unassigned,
            'final counts differ from saved membership CSV')
    mode, settings = analysis['mode'], analysis['inference_settings']
    expected = dict(seeds=[42, 7, 123], n_neighbors=30, min_dist=0., min_cluster_size=15,
                    min_samples=11, dimension=10, metric='euclidean', primary_extraction='leaf', primary_seed=42)
    require(mode in {'casp17_recursive', 'global'} and isinstance(settings, dict)
            and all(settings.get(key) == value for key, value in expected.items())
            and type(settings.get('workers')) is int and 1 <= settings['workers'] <= 4,
            'inference recipe differs from the recorded native protocol')
    public_settings = {key: settings[key] for key in [*expected, 'workers']}
    selections, history = analysis['selections'], analysis['history']
    require(isinstance(selections, list) and len(selections) <= 5
            and isinstance(history, list) and 1 <= len(history) <= 6,
            'selection/round evidence exceeds the recorded envelope')
    require(analysis['selected_fits'] == len(selections) and analysis['UMAP_fits'] == 3*len(selections)
            and analysis['HDBSCAN_extractions'] == 6*len(selections), 'native fit counters differ')
    selection_by_path, compact_selections = {}, []
    for ordinal, selection in enumerate(selections):
        require(isinstance(selection, dict) and isinstance(selection.get('directory'), str)
                and selection['directory'] not in selection_by_path, 'duplicate or missing native selection binding')
        indices = selection.get('selected_indices')
        require(isinstance(indices, list) and indices == sorted(set(indices)) and indices
                and all(integer(index, n-1) for index in indices)
                and selection.get('models') == [names[index] for index in indices], 'selection registry differs')
        size = len(indices); evidence = selection.get('diagnostics', {})
        require(evidence.get('dimension') == 10
                and evidence.get('settings') == {key: settings[key] for key in ['seeds','n_neighbors','min_dist','min_cluster_size','min_samples','workers']},
                'selection settings differ from the native recipe')
        rows = evidence.get('fits')
        require(isinstance(rows, list) and len(rows) == 6
                and {(row.get('seed'), row.get('method')) for row in rows} == {(seed, method) for seed in [42,7,123] for method in ['leaf','eom']},
                'all six seed/method extraction records are required')
        fits, by_fit = [], {}
        for row in rows:
            require(integer(row.get('clusters'), size) and integer(row.get('noise_models'), size)
                    and row['clusters'] <= size-row['noise_models']
                    and number(row.get('elapsed_seconds_from_seed_start'), 0, 7*24*3600), 'invalid fit count/noise/time')
            public = {key: row[key] for key in ['seed','method','clusters','noise_models','elapsed_seconds_from_seed_start']}
            fits.append(public); by_fit[(row['seed'], row['method'])] = row
        agreement = evidence.get('seed_agreement')
        require(isinstance(agreement, dict) and set(agreement) == {'leaf','eom'}, 'seed agreement is missing')
        public_agreement = {}
        for method in ['leaf','eom']:
            require(isinstance(agreement[method], dict) and set(agreement[method]) == {'7','123'}, 'both repeat-seed comparisons are required')
            public_agreement[method] = {}
            for seed in ['7','123']:
                row = agreement[method][seed]; joint = row.get('jointly_assigned_models')
                require(integer(joint, size) and number(row.get('jointly_assigned_fraction'), 0, 1)
                        and abs(row['jointly_assigned_fraction']-joint/size) < 1e-12
                        and number(row.get('ari_including_noise'), -1, 1)
                        and (row.get('ari_jointly_assigned') is None if joint < 2 else number(row.get('ari_jointly_assigned'), -1, 1))
                        and number(row.get('noise_status_agreement'), 0, 1), 'invalid saved agreement metrics')
                left, right = by_fit[(42, method)], by_fit[(int(seed), method)]
                both_noise = left['noise_models']+right['noise_models']-size+joint
                require(0 <= both_noise <= min(left['noise_models'], right['noise_models'])
                        and joint <= min(size-left['noise_models'], size-right['noise_models'])
                        and abs(row['noise_status_agreement']-(joint+both_noise)/size) < 1e-12,
                        'agreement coverage differs from fit noise counts')
                public_agreement[method][seed] = {key: row[key] for key in ['ari_including_noise','jointly_assigned_models',
                    'jointly_assigned_fraction','ari_jointly_assigned','noise_status_agreement']}
        result_sha = selection.get('files_sha256', {}).get('result.json')
        require(digest(result_sha) and digest(selection.get('feature_sha256')), 'selection source hashes are missing')
        public = dict(selection=ordinal, selected_models=size, result_sha256=result_sha,
            feature_sha256=selection['feature_sha256'], fits=fits, seed_agreement=public_agreement)
        selection_by_path[selection['directory']] = (ordinal, selection, by_fit)
        compact_selections.append(public)
    compact_rounds, used = [], set()
    for ordinal, row in enumerate(history):
        require(row.get('round') == ordinal, 'round numbering differs')
        current, noise = population(row.get('labels'))
        total = sum(not label.endswith('.unassigned') for label in current)
        require(row.get('structural_clusters') == total and row.get('populations') == dict(current)
                and ('unassigned_models' not in row or row['unassigned_models'] == noise), 'round population totals differ')
        compact = dict(round=ordinal, structural_clusters=total, unassigned_models=noise)
        if ordinal:
            parent, size, children, fitted_noise = (row.get(key) for key in ['selected_leaf','selected_models','child_clusters','noise_models_in_fit'])
            require(isinstance(parent, str) and re.fullmatch('[A-Za-z0-9_.-]+', parent)
                    and integer(size, n) and size > 0 and integer(children, size) and integer(fitted_noise, size)
                    and row.get('fit') in selection_by_path, 'round selection/count binding is invalid')
            selected_ordinal, selected, fits = selection_by_path[row['fit']]; used.add(selected_ordinal)
            previous = history[ordinal-1]['labels']
            expected_indices = list(range(n)) if mode == 'global' else [i for i, label in enumerate(previous) if label == parent]
            require(selected['selected_indices'] == expected_indices and len(expected_indices) == size
                    and fits[(42, 'leaf')]['clusters'] == children and fits[(42, 'leaf')]['noise_models'] == fitted_noise,
                    'round counts differ from the selected native seed42 extraction')
            chosen = set(expected_indices)
            if mode == 'casp17_recursive' and children < 2:
                require(row['labels'] == previous, 'unsplit-parent policy changed the saved labels')
            else:
                child_labels = [row['labels'][index] for index in expected_indices]
                prefix = 'C1.' if mode == 'global' else parent+'.'
                require(all(value.startswith(prefix) for value in child_labels)
                        and sum(value.endswith('.unassigned') for value in child_labels) == fitted_noise
                        and len({value for value in child_labels if not value.endswith('.unassigned')}) == children
                        and all(value == previous[index] for index, value in enumerate(row['labels']) if index not in chosen),
                        'round child/noise labels differ from saved extraction counts')
            compact.update(selected_leaf=parent, selected_models=size, child_clusters=children,
                           noise_models_in_fit=fitted_noise, selection=selected_ordinal)
        compact_rounds.append(compact)
    require(history[-1]['labels'] == final and used == set(range(len(selections))), 'round history differs from final CSV or fit registry')
    stopping = analysis['stopping']
    reasons = {'two_or_fewer_child_clusters','five_refinement_limit','insufficient_models_for_frozen_settings','single_global_fit'}
    require(isinstance(stopping, dict) and stopping.get('round') == len(history)-1
            and stopping.get('reason') in reasons, 'stopping record differs from completed rounds')
    public_stop = {key: stopping[key] for key in ['round','reason']}
    if 'child_clusters' in stopping:
        require(stopping['child_clusters'] == history[-1].get('child_clusters'), 'stopping child count differs')
        public_stop['child_clusters'] = stopping['child_clusters']
    native = analysis['native_source']
    require(isinstance(native, dict) and digest(native.get('sha256')) and digest(native.get('hierarchical_colors_sha256')),
            'native implementation source hashes are missing')
    varied = sum(len({fit['noise_models'] for fit in row['fits'] if fit['method'] == 'leaf'}) > 1 for row in compact_selections)
    note = (f'Exploratory seed42 HDBSCAN labels: {unassigned:,} of {n:,} models remain unassigned. '
            f'Repeat seeds changed leaf noise counts in {varied} of {len(selections)} fitted selections; '
            'inspect the saved count and agreement diagnostics. Repeat diagnostics are not a stability acceptance gate. '
            'Fitted noise can differ from final unassigned membership when an unsplit parent is retained.')
    return dict(schema_version=1, status='complete', source_analysis_sha256=source_sha256,
        native_source_sha256=native['sha256'], hierarchical_colors_sha256=native['hierarchical_colors_sha256'],
        models=n, structural_clusters=structural, unassigned_models=unassigned, mode=mode,
        inference_settings=public_settings, stopping=public_stop, rounds=compact_rounds, selections=compact_selections,
        selected_fits=len(selections), UMAP_fits=analysis['UMAP_fits'], HDBSCAN_extractions=analysis['HDBSCAN_extractions'],
        repeat_seed_leaf_noise_varied_selections=varied, interpretation=note)


def saved_views(frozen, manifest, target, names, *, strict=False):
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
        quality = density_diagnostics(analysis, labels, names, artifacts['analysis']['sha256'], strict=strict)
        method_note = analysis['method_note']+' '+quality.get('interpretation', quality.get('note', ''))
        views.append({'id': 'density', 'name': 'HDBSCAN', 'data': 'density.json',
                      'labels_csv': artifacts['labels_csv']['path'],
                      'ensemble_report': artifacts['ensemble_report']['path'],
                      'method_note': method_note, 'analysis_sha256': artifacts['analysis']['sha256'], 'density_diagnostics': quality,
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
    bindings, residue_notes = {}, {}
    for target, frozen in plan['targets'].items():
        manifest = checked_json(frozen['manifest'], frozen['manifest_sha256'])
        if manifest['analysis_status'] != 'complete' or manifest['identifier'] != target:
            raise ValueError('Real clustering has not completed for '+target)
        expected = {name: row for name, row in by_native.items() if row['target'] == target}
        if len(expected) != study['targets'][target]['models']: raise ValueError('Wrong target snapshot count')
        names, xy = projections(manifest, expected)
        views = saved_views(frozen, manifest, target, names, strict=plan.get('require_rbp_parity', False))
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
        residue_note = residue_presentation(detection, count); residue_notes[target] = residue_note
        positions = residue_note['positions']
        scope = {'chain': 'A', 'sequence_sha256': study['targets'][target]['sequence_sha256'], 'expected_ca_count': count,
                 'canonical_positions': positions, 'analysis_uses_full_construct': residue_note['full'],
                 'note': 'Canonical positions here index the supplied construct, not full-length UniProt; native file labels are preserved.',
                 'regions_verified': False, 'reference_ranges': residue_note['reference_ranges']}
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
                'models': names, 'residues': count, 'construct_residues': count, 'clustering_residue_count': len(positions),
                'clustering_positions': positions, 'range_id': 'all',
                'methods': method_names, 'clusters': cluster_values, 'projections': xy, 'schemes': schemes,
                'representatives': representatives, 'views': view_links, 'membership_file': str(membership.relative_to(source)), 'heatmaps': heatmaps,
                'coordinate_delivery': delivery, 'representative_kind': 'confidence-ranked representative or unscored structural medoid',
                'representative_policy': ('Plot and Bonsai point selections show the corresponding cluster representative. Only cluster representative and saved state-panel coordinates are published. ' if delivery == 'representatives' else '')+'Highest genuine native confidence among the default selected ensemble; unscored clusters use the recorded structural medoid. Energy is not confidence. Unassigned density points remain plotted and have no cluster representative.',
                'canonical_positions': positions, 'mask_note': residue_note['mask_note'],
                'clustering_method_note': view['method_note'],
                'provenance': {'workflow_manifest_sha256': frozen['manifest_sha256'], 'native_manifest_sha256': plan['native']['sha256'],
                    'sequence_sha256': study['targets'][target]['sequence_sha256'], 'pair_mask_sha256': detection['pair_mask_sha256'],
                    'confidence_sha256': plan['confidence']['sha256'], 'scientific_library_commit': study['library_commit'], 'renderer_commit': plan['renderer_commit'],
                    'labels_csv_sha256': sha256(view['labels_csv']), 'ensemble_report_sha256': sha256(view['ensemble_report']),
                    'energy_units': 'learned arbitrary units; lower within target; not confidence or calibrated free energy'}}
            if view.get('analysis_sha256'): data['provenance']['density_analysis_sha256'] = view['analysis_sha256']
            if view.get('density_diagnostics'): data['provenance']['density_diagnostics'] = view['density_diagnostics']
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
        if len(views) > 1 or not residue_note['full']:
            ranges['targets'][target] = {'residue_ranges': [{'id': 'all', 'name': residue_note['range_name'], 'data': target+'/primary.json',
                'asset_root': '', 'default_view': 'density' if 'density' in target_bindings else views[0]['id'],
                'available': True, 'canonical_positions': positions, 'views': list(target_bindings.values())}]}
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
                'ensemble_interpretation': residue_notes[target]['ensemble_interpretation'],
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
    has_density = any('density' in binding['views'] for binding in bindings.values())
    clustering_note = ('Saved recursive HDBSCAN is the default label view where available, alongside the original validated UMAP partition. '
        if has_density else 'Default validated common cluster partition. ')
    feature_note = ('Full-construct Cα distances. ' if all(note['full'] for note in residue_notes.values())
        else 'Cα distances over saved retained residue masks: '+ '; '.join(
            f"{study['targets'][target]['title']} {len(note['positions'])}/{study['targets'][target]['residues']} construct residues ({note['ranges']})"
            for target, note in residue_notes.items())+'. ')
    publication = {'schema_version': 1, 'project': 'domenico-structure-atlas', 'renderer': 'ais-explorer-v1',
        'presentation': {'title': title, 'methods': scope_note+'Accepted fixed scored snapshot. '+feature_note+clustering_note+'Genuine native confidence and separate saved ProteinEBM energies. '+coordinate_note+'Native rejection flags never filter the cohort. Camera-dependent reports are delivered after review.'}, 'files': []}
    if has_density: publication['presentation']['clustering_order'] = 'hdbscan-first'
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
