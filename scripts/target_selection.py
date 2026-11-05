"""Generic saved-clustering selection stages; no science occurs on import.

An exact approved outer plan owns process isolation, TMPDIR, pressure controls,
resource admission and originals. Each stage is a fresh process/output; completed
receipts are evidence for a separately approved recovery, never an automatic
retry. Camera-dependent final figures/reports and public state attachment remain
separate review steps. Native engines, confidence rules and sampling are reused.
"""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from target_continuation import bound, read, record

VIEWS = ('primary', 'density')
STAGES = ('primary-ensembles', 'density-ensembles', 'primary-grid', 'density-grid',
          'primary-bonsai', 'density-bonsai', 'states', 'panels', 'regions')


def need(value, message):
    if not value: raise ValueError(message)


def durable_json(path, value):
    """Publish only closed/fsynced immutable metadata, including call intent."""
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def compact_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def label_rows(path, members):
    with Path(path).open() as stream: rows = list(csv.DictReader(stream))
    need([r['model'] for r in rows] == [r['model'] for r in members]
         and [str(Path(r['source_path']).absolute()) for r in rows]
         == [r['source_path'] for r in members]
         and all(r['cluster_label'] for r in rows), 'Label population/order/source differs')
    return rows


def closed_metrics(entry):
    path = bound(entry)
    need(path.suffix == '.sqlite' and not any(Path(str(path)+suffix).exists()
         for suffix in ('-wal','-shm','-journal')), 'A bound closed metrics database without sidecars is required')
    return path


def inputs(plan):
    """Small, bound metadata only. Feature/coordinate engines load in stages."""
    need(plan['operation'] == 'target-selection-preparation', 'Wrong selection operation')
    desc = plan['target']; axes = read(desc['axes']); terminal = read(desc['terminal_report'])
    registry = read(plan['registry']); manifest = read(plan['primary_manifest'])
    diagnostics = read(plan['diagnostics'])
    visual = diagnostics['visualize_distogram']
    aliases = [visual['html'], *visual['images']]
    declared = {str(Path(entry['path']).absolute()) for entry in diagnostics['files']}
    need(all(isinstance(path,str) and path and str(Path(path).absolute()) in declared for path in aliases),
         'Every diagnostic alias must be covered by the frozen diagnostic files')
    density = read(plan['density_receipt']); members = registry['members']
    source_metrics = closed_metrics(plan['metrics_db'])
    need(str(source_metrics) == str(Path(manifest['metrics_db']).absolute()), 'Frozen source metrics binding differs')
    names = axes['names']; core = terminal['retained_reference_positions']
    construct = len(axes['metadata']['reference']['sequence'])
    need(len(names) == desc['model_count'] == len(members) and len(names) >= 50
         and [r['model'] for r in members] == names and len(set(names)) == len(names),
         'Complete original registry order differs')
    need(registry['target'] == manifest['identifier'] == density['target'] == desc['target']
         and registry['construct_residues'] == construct
         and registry['sequence_sha256'] == axes['sequence_sha256'], 'Construct/target/sequence differs')
    need(len(core) >= 3 and core == sorted(set(core))
         and all(type(i) is int and 0 <= i < construct for i in core)
         and terminal['reference_positions'] == list(range(construct))
         and manifest['terminal_detection'] == terminal, 'Accepted canonical comparison core differs')
    need(manifest['analysis_status'] == 'complete' and density['algorithm'] == 'HDBSCAN'
         and density['default_cluster_labels'] is True and density['models'] == names
         and density['retained_reference_positions'] == core
         and density['terminal_report'] == desc['terminal_report'], 'Completed primary/density analyses required')
    need(all(isinstance(r['method'], str) and r['method'] and isinstance(r['energy'], (int, float))
             and not isinstance(r['energy'], bool) and math.isfinite(r['energy'])
             and Path(r['source_path']).is_absolute() and len(r['sha256']) == 64 for r in members),
         'Original methods/finite ProteinEBM energies/source identities required')
    primary = label_rows(bound(plan['primary_labels']), members)
    need(str(bound(plan['primary_labels'])) == str(Path(manifest['knn_clustering']['labels_csv']).absolute()),
         'Primary manifest label binding differs')
    complete = label_rows(bound(plan['density_labels']), members)
    need([r['cluster_label'] for r in complete] == density['density'], 'Native density labels differ')
    with bound(plan['assigned_density_labels']).open() as stream: assigned = list(csv.DictReader(stream))
    need(assigned == [r for r in complete if not r['cluster_label'].endswith('.unassigned')],
         'Assigned density view must preserve every native nonnoise row exactly')
    for view, rows in [('primary', primary), ('density', assigned)]:
        counts = Counter(r['cluster_label'] for r in rows); limit = plan['bounds'][view]
        need(counts and dict(sorted(counts.items())) == limit['cluster_populations']
             and sum(c*(c-1) for c in counts.values()) == limit['medoid_ordered_feature_pairs']
             and limit['maximum_selected_models'] == sum(min(5, c) for c in counts.values()),
             'Frozen group/medoid/selected-model envelope differs before coordinate work')
    from rbp_states import native_call_bounds
    grid_calls = sum(len(plan['bounds'][v]['cluster_populations'])*(len(plan['bounds'][v]['cluster_populations'])-1) for v in VIEWS)
    need(plan['bounds']['state_directional_calls'] == native_call_bounds(len(names))['directional_call_ceiling']
         and plan['bounds']['grid_directional_calls'] == grid_calls
         and plan['bounds']['maximum_directional_calls'] == grid_calls + plan['bounds']['state_directional_calls']
         and plan['bounds']['Bonsai_PCA_fits'] == 2 and plan['bounds']['Bonsai_layout_runs'] == 6
         and type(plan['workers']) is int and 1 <= plan['workers'] <= 4,
         'Exact native-call/Bonsai/CPU budget differs')
    need(plan['range_id'] and isinstance(plan['range_id'], str), 'Explicit analysis range ID required')
    return dict(descriptor=desc, axes=axes, terminal=terminal, members=members, names=names,
                construct=construct, core=core, manifest=manifest, density=density,
                primary=primary, assigned=assigned, complete=complete, diagnostics=diagnostics)


def native_adapters():
    """Import engines only after the approved stage has passed metadata gates."""
    import pymol2
    from saved_feature_cache import mapped_selected_features
    from src.analysis.cluster_ensembles import select_cluster_ensembles
    from src.pymol_integration.cluster_ensembles import build_ensemble_session
    from rbp_representative_audit import verify_selected_ensembles
    from src.analysis.cluster_medoid_tm import calculate_medoid_tm
    from src.analysis.bonsai_interactive import run_ensemble_bonsai
    from src.analysis.feature_workspace import feature_input_workspace
    from rbp_native_tm import NativeTMRun, frozen_source_evidence
    from rbp_states import run_states
    from rbp_panel_exports import export_catalog
    from src.analysis.domain_variance_figure import render_domain_variance
    def session(selection, output, check, log):
        with pymol2.PyMOL() as pymol:
            return build_ensemble_session(pymol.cmd, selection, output, check_cancel=check, log=log)
    return dict(map=mapped_selected_features, select=select_cluster_ensembles, session=session,
        audit=verify_selected_ensembles, grid=calculate_medoid_tm, bonsai=run_ensemble_bonsai,
        workspace=feature_input_workspace, scorer=NativeTMRun, sources=frozen_source_evidence,
        states=run_states, panels=export_catalog, variance=render_domain_variance)


def mapped(desc, check, adapters):
    return adapters['map'](desc['selected_cache'], axes=desc['axes'], baseline=desc['baseline'],
        terminal_report=desc['terminal_report'], source_metadata=desc['source_metadata'],
        feature_proof=desc['feature_proof'], check=check)


def previous(root, stage, target, plan_sha256):
    receipt = read(record(root/(stage+'.json')))
    need(receipt['stage'] == stage and receipt['target'] == target
         and receipt['plan_sha256'] == plan_sha256 and receipt['status'] == 'complete',
         'Stage provenance/completion differs; no implicit historical reuse')
    for entry in receipt['artifacts'].values(): bound(entry)
    return receipt['artifacts']


def ensemble_stage(view, plan, data, output, adapters, check, checkpoint, log):
    manifest = deepcopy(data['manifest']); labels = plan['primary_labels'] if view == 'primary' else plan['assigned_density_labels']
    manifest['knn_clustering']['labels_csv'] = str(bound(labels))
    if view == 'density' and manifest.get('ensemble_selection_scope'):
        paths = {r['source_path'] for r in data['assigned']}
        scope = manifest['ensemble_selection_scope']
        scope['source_paths'] = [p for p in scope['source_paths'] if p in paths]
        # The pinned common-confidence API needs at least one genuine scored row.
        # Do not manufacture a score or broaden the explicit eligible population.
        need(scope['source_paths'], 'No genuine confidence anchors in assigned density view; native API extension required')
        scope['reason'] += ' Density representative selection covers assigned groups; all noise remains in atlas/state cohort.'
    for row in data['members']:
        check(); need(record(row['source_path'])['sha256'] == row['sha256'], 'Original coordinate changed')
    # Native metrics loading may enable WAL/migrate schema. Confine it to an
    # owned verified copy; never pass the frozen source database to that API.
    from serca_recovery import copy_bound
    closed_metrics(plan['metrics_db'])
    copied_metrics = copy_bound(plan['metrics_db'], output/'metrics.sqlite')
    manifest['metrics_db'] = copied_metrics['path']
    selected = adapters['select'](manifest, models_per_cluster=5, log=log, check_cancel=check)
    closed_metrics(plan['metrics_db'])
    completed_metrics = record(output/'metrics.sqlite')
    closed_metrics(completed_metrics)
    need(bound(plan['metrics_db']) == Path(data['manifest']['metrics_db']).absolute(),
         'Frozen source metrics changed during native selection')
    expected = plan['bounds'][view]['cluster_populations']
    need({r['cluster_label']: r['cluster_size'] for r in selected['clusters']} == expected
         and selected['selected_model_count'] <= plan['bounds'][view]['maximum_selected_models']
         and selected['omitted_cluster_labels'] == [], 'Native ensemble population exceeds accepted envelope')
    selected.update(report=str(output/'ensemble_report.json'), manifest=str(output/'manifest.json'),
        source_workflow_manifest_sha256=plan['primary_manifest']['sha256'], approved_view=None,
        camera_dependent_images='pending explicit camera review')
    durable_json(output/'selection-checkpoint.json', selected)
    checkpoint(view+':selection-closed')
    selected = adapters['session'](selected, output, check, log)
    selected['native_atom_inventory_audit'] = adapters['audit'](selected, check=check)
    durable_json(output/'ensemble_report.json', selected)
    manifest['cluster_ensembles'] = dict(status='awaiting_user_view', pse=selected['pse'],
        report=selected['report'], confidence_metric=selected['confidence_metric'],
        confidence_policy=selected['confidence_policy'], total_cluster_count=selected['total_cluster_count'],
        omitted_cluster_labels=[], display_limit_error=selected.get('display_limit_error'))
    manifest['selection_derivation'] = dict(source_primary=plan['primary_manifest'], view=view,
        metrics_source=plan['metrics_db'], metrics_copy_before_native=copied_metrics,
        metrics_copy_after_native=completed_metrics,
        coordinate_export_scope='selected representatives only; complete native atom inventory',
        camera_dependent_final_exports='pending explicit camera review',
        residue_confidence_figures='pending genuine per-residue confidence evidence; scalar means never substituted')
    diagnostics = data['diagnostics']
    for entry in diagnostics['files']: bound(entry)
    manifest['visualize_distogram'] = diagnostics['visualize_distogram']
    durable_json(output/'manifest.json', manifest)
    artifacts = dict(ensemble_report=record(output/'ensemble_report.json'), manifest=record(output/'manifest.json'),
                     selection_checkpoint=record(output/'selection-checkpoint.json'), pse=record(selected['pse']),
                     metrics_db=completed_metrics)
    for i, group in enumerate(selected['clusters']):
        if group.get('multimodel_structure'):
            artifacts['selected_model_ensemble_'+str(i)] = record(group['multimodel_structure'])
    return artifacts


def grid_stage(view, plan, data, output, selection, adapters, check, log):
    need(not any(v for k,v in os.environ.items() if k.startswith(('USALIGN_OFFLOAD', 'USALIGN_PARALLEL'))),
         'Unplanned USalign offload/parallel override')
    count = len(selection['clusters']); calls = count*(count-1)
    need(count == len(plan['bounds'][view]['cluster_populations']), 'Grid group count differs')
    intent = dict(view=view, reserved_directional_calls=calls,
                  failure_charge='all reserved calls uncertain; no automatic retry', native_per_pair_resume=False)
    durable_json(output/'native-grid-intent.json', intent)
    # The pinned helper has no per-pair ledger: reserve the entire grid first.
    working_selection = deepcopy(selection)
    working_selection['report'] = str(output/'selection-for-grid.json')
    try:
        grid = adapters['grid'](working_selection, bound(plan['native_tm_executable']), workers=plan['workers'], log=log, check_cancel=check)
    except BaseException:
        durable_json(output/'native-grid-failed.json', dict(intent, status='failed_all_reserved_calls_charged_uncertain'))
        raise
    import numpy as np
    matrix = np.asarray(grid['matrix'], float)
    need(set(grid['labels']) == set(plan['bounds'][view]['cluster_populations'])
         and len(grid['labels']) == count and matrix.shape == (count,count)
         and np.isfinite(matrix).all() and ((matrix >= 0) & (matrix <= 1)).all()
         and np.array_equal(matrix, matrix.T) and np.array_equal(np.diag(matrix), np.ones(count))
         and len(grid['comparisons']) == calls//2
         and grid['executable_sha256'] == plan['native_tm_executable']['sha256'], 'Native grid evidence differs')
    seen = set()
    for row in grid['comparisons']:
        pair = row['row'],row['column']; directions = row['directions']
        need(all(type(i) is int for i in pair) and 0 <= pair[0] < pair[1] < count
             and pair not in seen and len(directions) == 2
             and all(math.isfinite(float(d['tm_score'])) and 0 <= float(d['tm_score']) <= 1 for d in directions)
             and row['tm_score'] == matrix[pair]
             and row['tm_score'] == sum(float(d['tm_score']) for d in directions)/2,
             'Complete bidirectional grid query evidence differs')
        seen.add(pair)
    need(len(seen) == calls//2 and grid['tm_metric'] == 'mean_bidirectional_average_length_normalized_usalign'
         and grid['normalization'] == 'average_structure_length', 'Native grid protocol differs')
    durable_json(output/'medoid-tm.json', grid)
    durable_json(output/'native-grid-complete.json', dict(intent, status='complete', actual_directional_calls=calls,
                                                        grid=record(output/'medoid-tm.json')))
    return dict(medoid_tm=record(output/'medoid-tm.json'), charge=record(output/'native-grid-complete.json'))


def bonsai_stage(view, plan, data, output, selection, adapters, check, log):
    import numpy as np
    selected = [r for group in selection['clusters'] for r in group['selected']]
    selected_names = [r['model'] for r in selected]; lookup = {name:i for i,name in enumerate(data['names'])}
    need(3 <= len(selected) <= plan['bounds'][view]['maximum_selected_models']
         and len(set(selected_names)) == len(selected_names) and set(selected_names) <= set(lookup),
         'Native Bonsai needs at least three unique selected models within the frozen envelope')
    # The outer process owns an SSD TMPDIR. Copy only selected float16 rows,
    # close the full map, then let native nonzero/full PCA own its float64 work.
    scratch = Path(plan['stage_tmpdirs'][view+'-bonsai'])/'selected.npy'
    need(not scratch.exists() and scratch.parent == Path(os.environ['TMPDIR']).absolute(), 'Owned fresh Bonsai SSD staging required')
    staged = None
    try:
        with mapped(data['descriptor'], check, adapters) as (matrix, proof):
            staged = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.float16,
                                               shape=(len(selected), matrix.shape[1]))
            for i,name in enumerate(selected_names):
                check(); staged[i] = matrix[lookup[name]]
            staged.flush(); staged._mmap.close(); staged = None
        staged = np.load(scratch, mmap_mode='r', allow_pickle=False)
        source = dict(kind='frozen_models_by_features', terminal_detection=data['terminal'], dtype='float16',
            source_cache_sha256=data['descriptor']['selected_cache']['sha256'],
            source_feature_proof=data['descriptor']['feature_proof'], selected_models=selected_names,
            model_order_sha256=proof['model_order_sha256'],
            equivalence='Exact selected rows of the accepted retained Cα-distance cache; unchanged float16 values, order and pair mask.')
        tree = adapters['bonsai'](staged, selected, output/'bonsai', identifier=plan['target']['target'], feature_source=source, log=log)
        durable_json(output/'bonsai.json', tree)
    finally:
        if staged is not None: staged._mmap.close()
        if scratch.exists(): scratch.unlink()
    return dict(bonsai=record(output/'bonsai.json'))


def states_stage(plan, data, output, adapters, check, checkpoint, plan_sha256):
    cohort = dict(target=plan['target']['target'], range_id=plan['range_id'], models=data['names'],
        methods=[r['method'] for r in data['members']], energies=[r['energy'] for r in data['members']],
        sequence_sha256=data['axes']['sequence_sha256'], construct_residues=data['construct'],
        conditioning_t=.1, energy_scope='full_construct')
    clustering = dict(primary_labels=[r['cluster_label'] for r in data['primary']],
        hdbscan_labels=[r['cluster_label'] for r in data['complete']],
        hdbscan_noise_labels=sorted({r['cluster_label'] for r in data['complete'] if r['cluster_label'].endswith('.unassigned')}),
        clustering_positions=[i+1 for i in data['core']],
        saved_source_views={key:record(data['manifest']['clustering'][key]['embeddings']) for key in ('pca','umap','tsne')})
    bindings = dict(plan['source_bindings'], plan_sha256=plan_sha256, primary_manifest=plan['primary_manifest'], density_receipt=plan['density_receipt'],
                    registry=plan['registry'], accepted_terminal_report=plan['target']['terminal_report'], sources=adapters['sources']())
    with adapters['scorer'](data['members'], data['core'], cohort['sequence_sha256'], output/'native-tm',
        runtime=plan['native_tm_runtime'], source_bindings=bindings, check=check,
        maximum_directional_calls=plan['bounds']['state_directional_calls'], checkpoint=checkpoint) as scorer:
        scorer.prepare()
        import numpy as np
        with mapped(data['descriptor'], check, adapters) as (matrix, _):
            with adapters['workspace'](matrix, dtype=np.float32) as features:
                adapters['states'](cohort, clustering, features, data['core'], output/'states', scorer,
                    source_bindings=bindings, seed=plan['state_seed'], workers=plan['workers'], check=check, checkpoint=checkpoint)
    return dict(catalog=record(output/'states/catalog.json'))


def regions_stage(plan, data, root, output, plan_sha256, adapters, check):
    import numpy as np
    source = previous(root, 'primary-ensembles', plan['target']['target'], plan_sha256)
    manifest = read(source['manifest']); report = read(source['ensemble_report'])
    variance = np.load(bound(plan['variance']), mmap_mode='r', allow_pickle=False)
    try:
        need(variance.dtype == np.float64 and variance.shape == (data['construct'],data['construct'])
             and report['structural_flexibility']['residue_keys'] == data['terminal']['input_residue_keys'],
             'Accepted full-cohort variance/native axes differ')
        check(); heatmap = adapters['variance'](variance, report['structural_flexibility'], output/'annotated-variance.png',
                                               terminal_detection=data['terminal'])
    finally: variance._mmap.close()
    heatmap['saved_variance_provenance'] = dict(variance=plan['variance'], accepted_terminal_report=plan['target']['terminal_report'],
                                              precision='native float64 variance; no new detection')
    report['structural_flexibility']['heatmap'] = heatmap
    proof = dict(kind='saved_variance_presentation', parent_workflow_manifest_sha256=source['manifest']['sha256'],
                 parent_ensemble_report_sha256=source['ensemble_report']['sha256'])
    report['presentation_derivation'] = proof; durable_json(output/'ensemble_report.json', report)
    manifest['presentation_derivation'] = proof; manifest['cluster_ensembles']['report'] = str(output/'ensemble_report.json')
    durable_json(output/'manifest.json', manifest)
    return dict(manifest=record(output/'manifest.json'), ensemble_report=record(output/'ensemble_report.json'),
                image=record(output/'annotated-variance.png'))


def run_stage(stage, plan, *, plan_sha256, check=lambda: None,
              checkpoint=lambda _: None, log=lambda _: None, adapters=None):
    """One fresh process per stage, called ONLY by an exact approved outer runner.

    ``registry`` freezes full ordered member/model/source/hash/method/energy rows.
    ``bounds`` freezes both label-population maps and all call/selection ceilings.
    ``stage_tmpdirs`` names owned SSD slots; all numerical engines remain native.
    This callable does not authorize launching: the outer runner must verify the
    user approval, every frozen source/input, controls and stage TMPDIR first.
    """
    need(stage in STAGES and isinstance(plan_sha256,str) and len(plan_sha256)==64, 'Unknown stage/plan identity')
    data = inputs(plan); root = Path(plan['run_root']).absolute(); output = root/stage
    need(root.is_dir() and '..' not in root.parts and not any(p.is_symlink() for p in [root,*root.parents])
         and not output.exists() and not (root/(stage+'.json')).exists(),
         'Fresh immutable stage required; inspect partial outputs before recovery')
    if stage == 'density-ensembles': previous(root,'primary-ensembles',plan['target']['target'],plan_sha256)
    check(); output.mkdir(); adapters = native_adapters() if adapters is None else adapters
    view = stage.split('-',1)[0]
    if stage.endswith('-ensembles'):
        artifacts = ensemble_stage(view,plan,data,output,adapters,check,checkpoint,log)
        if view == 'density':
            from rbp_parity import METHOD_NOTE
            primary_artifacts = previous(root,'primary-ensembles',plan['target']['target'],plan_sha256)
            publication = dict(data['density'],schema_version=1,status='complete',target=plan['target']['target'],
                algorithm='HDBSCAN',source_workflow_manifest_sha256=primary_artifacts['manifest']['sha256'],
                model_order_sha256=compact_sha(data['names']),model_count=len(data['names']),method_note=METHOD_NOTE,
                accepted_density_receipt=plan['density_receipt'])
            durable_json(output/'publication-analysis.json',publication)
            artifacts.update(analysis=record(output/'publication-analysis.json'),labels_csv=plan['density_labels'])
    elif stage.endswith(('-grid','-bonsai')):
        prior = previous(root,view+'-ensembles',plan['target']['target'],plan_sha256)
        selection = read(prior['ensemble_report'])
        artifacts = (grid_stage if stage.endswith('-grid') else bonsai_stage)(
            view,plan,data,output,selection,adapters,check,log)
    elif stage == 'states': artifacts = states_stage(plan,data,output,adapters,check,checkpoint,plan_sha256)
    elif stage == 'panels':
        catalog = previous(root,'states',plan['target']['target'],plan_sha256)['catalog']
        adapters['panels'](bound(catalog),data['members'],output/'panels',expected_catalog_sha256=catalog['sha256'],
                           check=check,log=log,checkpoint=checkpoint)
        artifacts = dict(catalog=record(output/'panels/catalog.json'))
    else: artifacts = regions_stage(plan,data,root,output,plan_sha256,adapters,check)
    need(read(plan['primary_manifest']) == data['manifest'] and read(plan['target']['terminal_report']) == data['terminal'],
         'Frozen primary/mask changed during selection stage')
    check(); durable_json(root/(stage+'.json'),dict(schema_version=1,status='complete',stage=stage,
        target=plan['target']['target'],plan_sha256=plan_sha256,artifacts=artifacts,
        camera_dependent_final_exports='pending explicit camera review',
        public_state_attachment='pending generic saved-catalog adapter', public_deployment=False))
    checkpoint(stage+':closed')
    return record(root/(stage+'.json'))
