"""Source-only contracts: static arrays and mocked engines; no native fits/exports."""
from contextlib import contextmanager
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import tempfile
import unittest
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import target_selection as subject
from target_continuation import record

def write(path, value):
    path.write_text(json.dumps(value))
    return record(path)

def make_case(tmp_path):
    names = ['m' + str(i) for i in range(50)]
    members = []
    for i, name in enumerate(names):
        source = tmp_path / (name + '.pdb')
        source.write_text('mock original atoms ' + name)
        members.append(dict(model=name, source_path=str(source), sha256=record(source)['sha256'], method='native', energy=float(i)))

    def labels(name, values, selected=None):
        path = tmp_path / name
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, ['model', 'source_path', 'cluster_label', 'color'])
            writer.writeheader()
            for i, label in enumerate(values):
                if selected is None or i in selected:
                    writer.writerow(dict(model=names[i], source_path=members[i]['source_path'], cluster_label=label, color='#123456'))
        return record(path)
    p = ['A'] * 25 + ['B'] * 25
    d = ['D'] * 25 + ['D.unassigned'] * 25
    p_csv = labels('primary.csv', p)
    d_csv = labels('density.csv', d)
    a_csv = labels('assigned.csv', d, set(range(25)))
    keys = [['A', str(i + 1)] for i in range(6)]
    axes = write(tmp_path / 'axes.json', dict(names=names, sequence_sha256='c' * 64, metadata=dict(reference=dict(sequence='AAAAAA'))))
    terminal = dict(reference_positions=list(range(6)), retained_reference_positions=[1, 2, 3, 4], pair_mask_sha256='f' * 64, input_residue_keys=keys)
    terminal_entry = write(tmp_path / 'terminal.json', terminal)
    registry = write(tmp_path / 'registry.json', dict(target='TOY', construct_residues=6, sequence_sha256='c' * 64, members=members))
    embeds = {}
    for key in ('pca', 'umap', 'tsne'):
        saved = tmp_path / (key + '.npz')
        saved.write_text('mock saved projection')
        embeds[key] = dict(embeddings=str(saved))
    metrics = tmp_path / 'metrics.sqlite'
    metrics.write_bytes(b'Mocked closed metrics, no native SQLite call')
    manifest = dict(identifier='TOY', analysis_status='complete', terminal_detection=terminal, knn_clustering=dict(labels_csv=p_csv['path']), clustering=embeds, metrics_db=str(tmp_path / 'metrics.sqlite'), ensemble_selection_scope=dict(source_paths=[r['source_path'] for r in members[:3]], reason='Only genuine confidence sidecars', unscored_cluster_policy='structural_medoid'))
    m_entry = write(tmp_path / 'primary.json', manifest)
    density = write(tmp_path / 'density.json', dict(target='TOY', algorithm='HDBSCAN', default_cluster_labels=True, models=names, density=d, retained_reference_positions=[1, 2, 3, 4], terminal_report=terminal_entry))
    root = tmp_path / 'run'
    root.mkdir()
    scratch = tmp_path / 'ssd'
    scratch.mkdir()
    html = tmp_path / 'saved.html'
    html.write_text('Mock saved diagnostic, no rendering')
    diag = write(tmp_path / 'diagnostics.json', dict(files=[record(html)], visualize_distogram=dict(images=[], html=str(html))))
    variance = tmp_path / 'variance.npy'
    if np is None:
        variance.write_bytes(b'Metadata-only placeholder; array stage skipped')
    else:
        np.save(variance, np.zeros((6, 6), dtype=np.float64))
    exe = tmp_path / 'USalign'
    exe.write_text('mock executable, never run')
    from rbp_states import native_call_bounds
    state_calls = native_call_bounds(50)['directional_call_ceiling']
    plan = dict(operation='target-selection-preparation', metrics_db=record(metrics), target=dict(target='TOY', model_count=50, axes=axes, terminal_report=terminal_entry, selected_cache={'sha256': 'b' * 64}, baseline={}, source_metadata={}, feature_proof={}), registry=registry, primary_manifest=m_entry, density_receipt=density, primary_labels=p_csv, density_labels=d_csv, assigned_density_labels=a_csv, diagnostics=diag, variance=record(variance), run_root=str(root), range_id='all', workers=2, native_tm_executable=record(exe), native_tm_runtime={}, source_bindings=dict(source='frozen'), state_seed=4200, stage_tmpdirs={'primary-bonsai': str(scratch), 'density-bonsai': str(scratch)}, bounds=dict(primary=dict(cluster_populations={'A': 25, 'B': 25}, medoid_ordered_feature_pairs=1200, maximum_selected_models=10), density=dict(cluster_populations={'D': 25}, medoid_ordered_feature_pairs=600, maximum_selected_models=5), state_directional_calls=state_calls, grid_directional_calls=2, maximum_directional_calls=state_calls + 2, Bonsai_PCA_fits=2, Bonsai_layout_runs=6))
    return (plan, members)

def adapters(case):
    plan, members = case
    calls = {}

    def select(manifest, **kwargs):
        with Path(manifest['knn_clustering']['labels_csv']).open() as stream:
            rows = list(csv.DictReader(stream))
        groups = []
        for label in dict.fromkeys((r['cluster_label'] for r in rows)):
            group = [r for r in rows if r['cluster_label'] == label]
            selected = [dict(r, confidence=None, rank=1, state=1) for r in group[:2]]
            groups.append(dict(cluster_label=label, cluster_size=len(group), selected=selected, medoid=selected[0]))
        calls['select_manifest'] = deepcopy(manifest)
        return dict(clusters=groups, selected_model_count=sum((len(g['selected']) for g in groups)), omitted_cluster_labels=[], total_cluster_count=len(groups), confidence_metric='mean_plddt', confidence_policy={}, structural_flexibility=dict(residue_keys=plan['target'] and [['A', str(i + 1)] for i in range(6)]))

    def session(selected, output, check, log):
        session = output / 'selection.pse'
        session.write_text('mock session')
        selected['pse'] = str(session)
        for i, group in enumerate(selected['clusters']):
            path = output / (str(i) + '.cif')
            path.write_text('mock full construct')
            group['multimodel_structure'] = str(path)
        return selected

    @contextmanager
    def mapped(*args, **kwargs):
        calls['map'] = kwargs
        yield (np.arange(300, dtype=np.float16).reshape(50, 6), dict(model_order_sha256='e' * 64))

    def bonsai(matrix, selected, output, **kwargs):
        calls['bonsai'] = (matrix.copy(), deepcopy(selected), kwargs)
        return dict(model_count=len(selected))

    @contextmanager
    def workspace(matrix, **kwargs):
        yield matrix.astype(np.float32)

    class Scorer:

        def __init__(self, *args, **kwargs):
            calls['scorer'] = (args, kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def prepare(self):
            calls['prepared'] = True

    def states(cohort, clustering, features, core, output, scorer, **kwargs):
        calls['states'] = (deepcopy(cohort), deepcopy(clustering), features.copy(), list(core))
        output.mkdir()
        write(output / 'catalog.json', dict(mock='no real selection'))

    def variance(matrix, flexibility, path, **kwargs):
        calls['variance'] = (matrix.copy(), kwargs)
        path.write_text('mock image')
        return dict(mock='saved rendering')
    return (dict(select=select, session=session, audit=lambda *a, **k: dict(mock=True), map=mapped, bonsai=bonsai, workspace=workspace, scorer=Scorer, sources=lambda: dict(mock='pinned source'), states=states, variance=variance), calls)

def stage(name, case, adapter):
    return subject.run_stage(name, case[0], plan_sha256='a' * 64, adapters=adapter)

class SelectionTests(unittest.TestCase):

    def setUp(self):
        self.enterContext(patch.dict(os.environ, dict(os.environ)))
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.case = make_case(Path(directory))

    @unittest.skipUnless(np is not None, 'Optional NumPy static-array fixtures unavailable')
    def test_trimmed_core_keeps_full_construct_and_all_noise(self):
        case = self.case
        adapter, calls = adapters(case)
        stage('states', case, adapter)
        cohort, cluster, matrix, core = calls['states']
        self.assertTrue(cohort['construct_residues'] == 6 and cohort['range_id'] == 'all')
        self.assertTrue(core == [1, 2, 3, 4] and cluster['clustering_positions'] == [2, 3, 4, 5])
        self.assertTrue(len(cohort['models']) == len(cluster['hdbscan_labels']) == 50)
        self.assertTrue(cluster['hdbscan_labels'].count('D.unassigned') == 25)
        self.assertTrue(cluster['hdbscan_noise_labels'] == ['D.unassigned'])
        self.assertTrue(matrix.shape == (50, 6) and matrix.dtype == np.float32)
        self.assertTrue(calls['scorer'][1]['maximum_directional_calls'] == 11600)

    def test_group_bound_checked_before_any_engine(self):
        case = self.case
        adapter, calls = adapters(case)
        case[0]['bounds']['density']['cluster_populations'] = {'D': 24}
        with self.assertRaisesRegex(ValueError, 'envelope'):
            stage('density-ensembles', case, adapter)
        self.assertTrue(calls == {} and (not (Path(case[0]['run_root']) / 'density-ensembles').exists()))

    def test_missing_noise_or_changed_identity_rejected(self):
        case = self.case
        adapter, _ = adapters(case)
        entry = case[0]['density_receipt']
        data = json.loads(Path(entry['path']).read_text())
        data['density'][-1] = 'D'
        case[0]['density_receipt'] = write(Path(entry['path']), data)
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            stage('states', case, adapter)

    def test_native_scope_intersected_without_promoting_unscored(self):
        case = self.case
        adapter, calls = adapters(case)
        stage('primary-ensembles', case, adapter)
        stage('density-ensembles', case, adapter)
        manifest = calls['select_manifest']
        self.assertTrue(len(manifest['ensemble_selection_scope']['source_paths']) == 3)
        output = Path(case[0]['run_root']) / 'density-ensembles'
        result = json.loads((output / 'manifest.json').read_text())
        self.assertTrue(result['cluster_ensembles']['status'] == 'awaiting_user_view')
        self.assertTrue(result['visualize_distogram'] == json.loads(Path(case[0]['diagnostics']['path']).read_text())['visualize_distogram'])
        self.assertTrue('genuine per-residue' in result['selection_derivation']['residue_confidence_figures'])
        receipt = json.loads((output.parent / 'density-ensembles.json').read_text())
        self.assertTrue('pse' in receipt['artifacts'] and 'selected_model_ensemble_0' in receipt['artifacts'])
        analysis = json.loads(Path(receipt['artifacts']['analysis']['path']).read_text())
        self.assertTrue(analysis['schema_version'] == 1 and analysis['status'] == 'complete')
        self.assertTrue(analysis['source_workflow_manifest_sha256'] == record(output.parent / 'primary-ensembles/manifest.json')['sha256'])
        self.assertTrue(analysis['density'].count('D.unassigned') == 25 and analysis['default_cluster_labels'] is True)

    def test_empty_eligible_intersection_fails_closed(self):
        case = self.case
        adapter, calls = adapters(case)
        entry = case[0]['primary_manifest']
        value = json.loads(Path(entry['path']).read_text())
        value['ensemble_selection_scope']['source_paths'] = [r['source_path'] for r in case[1][25:]]
        case[0]['primary_manifest'] = write(Path(entry['path']), value)
        stage('primary-ensembles', case, adapter)
        calls.clear()
        with self.assertRaisesRegex(ValueError, 'No genuine confidence'):
            stage('density-ensembles', case, adapter)
        self.assertTrue('select_manifest' not in calls)

    def test_grid_intent_precedes_mock_call_and_uncertain_failure_charged(self):
        case = self.case
        for key in list(os.environ):
            if key.startswith(('USALIGN_OFFLOAD', 'USALIGN_PARALLEL')):
                os.environ.pop(key, None)
        adapter, _ = adapters(case)
        stage('primary-ensembles', case, adapter)
        output = Path(case[0]['run_root']) / 'primary-grid'

        def fail(selection, executable, **kwargs):
            intent = json.loads((output / 'native-grid-intent.json').read_text())
            self.assertTrue(intent['reserved_directional_calls'] == 2)
            raise RuntimeError('mock native failure; no actual call')
        adapter['grid'] = fail
        with self.assertRaisesRegex(RuntimeError, 'mock native'):
            stage('primary-grid', case, adapter)
        self.assertTrue(json.loads((output / 'native-grid-failed.json').read_text())['status'] == 'failed_all_reserved_calls_charged_uncertain')
        self.assertTrue(not (output.parent / 'primary-grid.json').exists())
        with self.assertRaisesRegex(ValueError, 'Fresh immutable'):
            stage('primary-grid', case, adapter)

    @unittest.skipUnless(np is not None, 'Optional NumPy static-array fixtures unavailable')
    def test_bonsai_receives_only_exact_selected_rows_and_closes_scratch(self):
        case = self.case
        adapter, calls = adapters(case)
        stage('primary-ensembles', case, adapter)
        with patch.dict(os.environ, dict(TMPDIR=case[0]['stage_tmpdirs']['primary-bonsai'])):
            stage('primary-bonsai', case, adapter)
        matrix, selected, kwargs = calls['bonsai']
        self.assertTrue([r['model'] for r in selected] == ['m0', 'm1', 'm25', 'm26'])
        self.assertTrue(np.array_equal(matrix, np.arange(300, dtype=np.float16).reshape(50, 6)[[0, 1, 25, 26]]))
        self.assertTrue(kwargs['feature_source']['terminal_detection']['retained_reference_positions'] == [1, 2, 3, 4])
        self.assertTrue(not (Path(os.environ.get('TMPDIR', '/not-used')) / 'selected.npy').exists())
        self.assertTrue(not (Path(case[0]['stage_tmpdirs']['primary-bonsai']) / 'selected.npy').exists())

    @unittest.skipUnless(np is not None, 'Optional NumPy static-array fixtures unavailable')
    def test_region_renderer_uses_accepted_float64_variance_and_no_new_detection(self):
        case = self.case
        adapter, calls = adapters(case)
        stage('primary-ensembles', case, adapter)
        stage('regions', case, adapter)
        matrix, kwargs = calls['variance']
        self.assertTrue(matrix.dtype == np.float64 and matrix.shape == (6, 6))
        self.assertTrue(kwargs['terminal_detection']['retained_reference_positions'] == [1, 2, 3, 4])
        root = Path(case[0]['run_root'])
        report = json.loads((root / 'regions/ensemble_report.json').read_text())
        self.assertTrue(report['structural_flexibility']['heatmap']['saved_variance_provenance']['variance'] == case[0]['variance'])
        self.assertTrue(report['presentation_derivation']['kind'] == 'saved_variance_presentation')

    def test_completed_stage_tamper_is_not_historical_reuse(self):
        case = self.case
        adapter, _ = adapters(case)
        stage('primary-ensembles', case, adapter)
        path = Path(case[0]['run_root']) / 'primary-ensembles/ensemble_report.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Bound input changed'):
            stage('primary-bonsai', case, adapter)

    @unittest.skipUnless(np is not None, 'Optional NumPy static-array fixtures unavailable')
    def test_complete_grid_requires_both_directions_for_every_pair(self):
        case = self.case
        for key in list(os.environ):
            if key.startswith(('USALIGN_OFFLOAD', 'USALIGN_PARALLEL')):
                os.environ.pop(key, None)
        adapter, _ = adapters(case)
        stage('primary-ensembles', case, adapter)

        def grid(selection, executable, **kwargs):
            self.assertTrue(Path(selection['report']).parent.name == 'primary-grid')
            return dict(labels=['A', 'B'], matrix=[[1, 0.7], [0.7, 1]], comparisons=[dict(row=0, column=1, tm_score=0.7, directions=[dict(tm_score=0.6), dict(tm_score=0.8)])], executable_sha256=case[0]['native_tm_executable']['sha256'], tm_metric='mean_bidirectional_average_length_normalized_usalign', normalization='average_structure_length')
        adapter['grid'] = grid
        stage('primary-grid', case, adapter)
        receipt = json.loads((Path(case[0]['run_root']) / 'primary-grid/native-grid-complete.json').read_text())
        self.assertTrue(receipt['actual_directional_calls'] == 2 and receipt['status'] == 'complete')

    @unittest.skipUnless(np is not None, 'Optional NumPy static-array fixtures unavailable')
    def test_too_few_bonsai_representatives_never_runs_native_engine(self):
        case = self.case
        adapter, calls = adapters(case)
        stage('primary-ensembles', case, adapter)
        stage('density-ensembles', case, adapter)
        with patch.dict(os.environ, dict(TMPDIR=case[0]['stage_tmpdirs']['density-bonsai'])):
            with self.assertRaisesRegex(ValueError, 'at least three'):
                stage('density-bonsai', case, adapter)
        self.assertTrue('bonsai' not in calls)

    def test_diagnostic_aliases_must_be_frozen_before_engine(self):
        entry = self.case[0]['diagnostics']
        diagnostics = json.loads(Path(entry['path']).read_text())
        diagnostics['visualize_distogram']['images'] = ['/tmp/not-bound-diagnostic.png']
        self.case[0]['diagnostics'] = write(Path(entry['path']), diagnostics)
        adapter, calls = adapters(self.case)
        with self.assertRaisesRegex(ValueError, 'diagnostic alias'):
            stage('primary-ensembles', self.case, adapter)
        self.assertEqual(calls, {})
    def test_native_metrics_loader_only_receives_verified_owned_copy(self):
        adapter, calls = adapters(self.case)
        source = record(self.case[0]['metrics_db']['path'])
        select = adapter['select']
        def mutating_native_loader(manifest, **kwargs):
            owned = Path(manifest['metrics_db'])
            self.assertNotEqual(owned, Path(source['path']))
            self.assertEqual(record(owned)['sha256'], source['sha256'])
            owned.write_bytes(owned.read_bytes() + b' Mock native owned schema migration')
            return select(manifest, **kwargs)
        adapter['select'] = mutating_native_loader
        stage('primary-ensembles', self.case, adapter)
        self.assertEqual(record(source['path']), source)
        root = Path(self.case[0]['run_root'])
        receipt = json.loads((root/'primary-ensembles.json').read_text())
        owned = receipt['artifacts']['metrics_db']
        self.assertNotEqual(owned['sha256'], source['sha256'])
        manifest = json.loads((root/'primary-ensembles/manifest.json').read_text())
        self.assertEqual(manifest['metrics_db'], owned['path'])
        self.assertEqual(manifest['selection_derivation']['metrics_source'], source)

    def test_live_source_metrics_sidecars_fail_before_engine_or_copy(self):
        adapter, calls = adapters(self.case)
        source = self.case[0]['metrics_db']['path']
        Path(source+'-wal').write_bytes(b'Mock live sidecar')
        with self.assertRaisesRegex(ValueError,'closed metrics'):
            stage('primary-ensembles', self.case, adapter)
        self.assertEqual(calls,{})
        self.assertFalse((Path(self.case[0]['run_root'])/'primary-ensembles').exists())

if __name__ == '__main__':
    unittest.main()
