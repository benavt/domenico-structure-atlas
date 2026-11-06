"""Tiny saved arrays and fake receipts only; no engines, models or fits."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pilot_compare as subject
import hpc_pilot as pilot


def put(path, value):
    path.write_text(json.dumps(value))
    return pilot.record(path)


class PartitionTests(unittest.TestCase):
    def test_permuted_ids_are_equivalent_but_changed_membership_is_not(self):
        self.assertEqual(subject.partition(['a', 'a', 'b', 'b'], ['y', 'y', 'x', 'x'])['adjusted_rand_index'], 1)
        self.assertTrue(subject.partition(['a', 'a', 'b', 'b'], ['y', 'y', 'x', 'x'])['structural_validation'])
        self.assertFalse(subject.partition(['a', 'a', 'b', 'b'], ['a', 'b', 'a', 'b'])['structural_validation'])

    def test_all_noise_and_one_group_are_not_structural_validation(self):
        for density, labels in ((True, ['C1.unassigned'] * 8), (False, ['one'] * 8)):
            result = subject.partition(labels, labels, density=density)
            self.assertEqual(result['adjusted_rand_index'], 1)
            self.assertTrue(result['partitions_equivalent'])
            self.assertTrue(result['degenerate'])
            self.assertIsNone(result['structural_ari'])
            self.assertFalse(result['structural_validation'])

    def test_equal_partition_geometry_cannot_hide_different_noise_membership(self):
        a = ['C1.1', 'C1.1', 'C1.unassigned', 'C1.unassigned']
        b = ['C1.unassigned', 'C1.unassigned', 'C1.2', 'C1.2']
        result = subject.partition(a, b, density=True)
        self.assertEqual(result['adjusted_rand_index'], 1)
        self.assertFalse(result['noise_masks_identical'])
        self.assertFalse(result['partitions_equivalent'])

    def test_module_has_no_scientific_engine_imports(self):
        imports = []
        for node in ast.walk(ast.parse(Path(subject.__file__).read_text())):
            if isinstance(node, ast.Import):
                imports += [name.name.split('.')[0] for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports += [node.module.split('.')[0]]
        self.assertFalse(set(imports) & {'umap', 'sklearn', 'numba', 'torch', 'scipy'})


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'Optional NumPy for tiny saved-array diagnostics')
class SavedComparisonTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        names = ['model-' + str(index) for index in range(64)]
        self.expected = dict(target='SERCA_P04191', model_count=64, construct_residues=993,
            retained_reference_positions=list(range(57, 951)), feature_sha256='a' * 64,
            pair_mask_sha256=pilot.SERCA_PAIR_MASK,
            model_order_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest())
        feature_file = self.root / 'features.npy'
        feature_file.write_bytes(b'Bound descriptor only; feature calculation is mocked for this fixture.')
        self.expected['features'] = pilot.record(feature_file)
        index = np.arange(64, dtype=np.float64)
        self.xy = np.column_stack((index, np.sin(index)))
        self.raw_distances = subject._distances(np.column_stack((index, np.sin(index), np.cos(index))))
        self.feature_diagnostic = dict(float32_sha256='d' * 64, raw_float16_sha256='a' * 64)
        self.host_plans = {}
        self.host_specs = {}
        for hostname in ('workstation', 'hpc'):
            directory = self.root / hostname
            directory.mkdir()
            members = [dict(model=name, local_path='/native/' + hostname + '/' + name + '.pdb',
                            original_sha256=hashlib.sha256(name.encode()).hexdigest(),
                            flags={'physicality': 'rejected' if index == 63 else 'accepted'})
                       for index, name in enumerate(names)]
            spec = {key: value for key, value in self.expected.items() if key != 'target'}
            spec['members'] = put(directory / 'members.json', dict(members=members))
            config = put(directory / 'cluster-config.json', dict(schema_version=1, settings=pilot.DEFAULT_COMPONENT_SETTINGS))
            module = pilot.record(pilot.__file__)
            plan = dict(operation='hpc-cpu-validation', phase='pilot64', target='SERCA_P04191',
                stages=list(subject.STAGES), maximum_projection_fits=9, maximum_hdbscan_extractions=6,
                inputs=spec, native=dict(library_commit=pilot.LIBRARY_COMMIT,
                    density_source=dict(path='/native/' + hostname + '/hd_density_experiment.py',
                                        commit=pilot.DENSITY_COMMIT, sha256=pilot.DENSITY_SHA256),
                    cluster_config=config), workers=1, frozen_files=[module])
            self.host_plans[hostname] = plan
            receipts = {}
            primary = ['Cluster 1'] * 32 + ['Cluster 2'] * 32
            density = ['C1.1'] * 31 + ['C1.2'] * 31 + ['C1.unassigned'] * 2
            for stage in subject.STAGES:
                output = directory / stage
                output.mkdir()
                sidecar = output / ('projection.npz' if stage in ('pca', 'umap', 'tsne') else 'labels.npz')
                arrays = dict(filenames=np.asarray(names), filepaths=np.asarray([row['local_path'] for row in members]),
                              flags_json=np.asarray([json.dumps(row['flags']) for row in members]))
                if stage in ('pca', 'umap', 'tsne'):
                    arrays['embeddings'] = self.xy.copy()
                else:
                    arrays['cluster_labels'] = np.asarray(primary if stage == 'labels' else density)
                np.savez_compressed(sidecar, **arrays)
                entry = pilot.record(sidecar)
                outputs = [entry]
                special = {}
                fits = dict(PCA=0, UMAP=0, TSNE=0, HDBSCAN=0)
                if stage in ('pca', 'umap', 'tsne'):
                    special['projection'] = entry
                    fits.update(PCA=1 if stage in ('pca', 'tsne') else 0,
                                UMAP=1 if stage == 'umap' else 0, TSNE=1 if stage == 'tsne' else 0)
                    special['parameters'] = (dict(n_components=2, random_seed=42) if stage == 'pca' else
                        dict(n_neighbors=15, min_dist=.1, n_components=2, verbose=False, random_seed=42)
                        if stage == 'umap' else
                        dict(perplexity=30., implementation='python', verbose=False, random_seed=42))
                elif stage == 'labels':
                    special.update(labels=entry, repeat_UMAP_fits_maximum=2)
                    outputs.append(put(output / 'selection.json', dict(point_count=64,
                        repeat_graphs=[dict(seed=7), dict(seed=123)], feature_input=dict(sha256='a' * 64))))
                    fits['UMAP'] = 2
                else:
                    analysis = dict(mode='global', models=names, density=density, primary_input_labels=primary,
                        selected_fits=1, UMAP_fits=3, HDBSCAN_extractions=6,
                        stopping=dict(round=1, reason='single_global_fit'),
                        inference_settings=dict(seeds=[42, 7, 123], n_neighbors=30, min_dist=0.,
                            min_cluster_size=15, min_samples=11, workers=1, dimension=10,
                            metric='euclidean', primary_extraction='leaf', primary_seed=42),
                        native_source=plan['native']['density_source'],
                        structural_clusters=2, unassigned_models=2,
                        selections=[dict(selected_indices=list(range(64)), models=names,
                            feature_sha256='d' * 64, diagnostics=dict(dimension=10,
                                settings=dict(seeds=[42, 7, 123], n_neighbors=30, min_dist=0.,
                                    min_cluster_size=15, min_samples=11, workers=1),
                                fits=[dict(seed=seed, method=method) for seed in (42, 7, 123)
                                      for method in ('leaf', 'eom')]))],
                        feature_proof=dict(dtype='float32', stored_dtype='float16', fit_dtype='float32',
                            retained_reference_positions=list(range(57, 951)), shape=[64, 399171],
                            feature_sha256='d' * 64, model_order_sha256=pilot.compact_sha(names),
                            primary_partition_sha256=pilot.compact_sha(primary),
                            scaling='Raw saved Cα distances cast to float32; no standardization or feature PCA.'))
                    density_entry = put(output / 'analysis.json', analysis)
                    special.update(labels=entry, density_analysis=density_entry,
                                   full_recursive_HDBSCAN=False, default_cluster_labels=False)
                    outputs.append(density_entry)
                    fits.update(UMAP=3, HDBSCAN=6)
                result = dict(schema_version=1, stage=stage, status='diagnostic_not_full_clustering',
                    phase_complete=True, full_clustering_complete=False, target='SERCA_P04191',
                    inputs=spec, input_fingerprint=pilot.compact_sha(dict(target=plan['target'], inputs=spec,
                        native=plan['native'], workers=1)), models=names,
                    local_paths=[row['local_path'] for row in members], model_flags=[row['flags'] for row in members],
                    native_library_commit=pilot.LIBRARY_COMMIT,
                    native_sources={key:dict(path='/installed/' + hostname + '/' + key, bytes=1, sha256=sha)
                                    for key, sha in pilot.LIBRARY_SOURCE_HASHES.items()},
                    module_source=module, coordinate_parses=0, scoring_calls=0, molecular_exports=0,
                    input_integrity=dict(file_sha256_verified_before_map=True, raw_float16_sha256_verified=True,
                                         readonly_mapping=True, identities_unchanged_after_phase=True),
                    outputs=outputs, scientific_fits=fits, **special)
                receipts[stage] = put(output / 'receipt.json', result)
            self.host_specs[hostname] = dict(plan=put(directory / 'execution-plan.json', plan), receipts=receipts,
                runtime=put(directory / 'runtime.json', dict(host=hostname, python='3.11', native_pin=pilot.LIBRARY_COMMIT)))
        contract = dict(schema_version=1, operation='paired-pilot-comparison', expected=self.expected,
                        thresholds=subject.DEFAULT_THRESHOLDS, comparator_source=pilot.record(subject.__file__),
                        host_plans={key: value['plan'] for key, value in self.host_specs.items()})
        self.manifest = dict(schema_version=1, operation='paired-pilot-saved-comparison',
                             contract=put(self.root / 'contract.json', contract), hosts=self.host_specs)

    def compare(self, name='comparison'):
        with patch.object(subject, 'feature_distances', return_value=(self.raw_distances, self.feature_diagnostic)):
            return subject.compare(self.manifest, self.root / name)

    def change_receipt(self, stage, change):
        entry = self.host_specs['hpc']['receipts'][stage]
        path = Path(entry['path'])
        result = json.loads(path.read_text())
        change(result)
        self.host_specs['hpc']['receipts'][stage] = put(path, result)

    def replace_coordinates(self, stage, value):
        def update(receipt):
            old = receipt['projection']
            path = Path(old['path'])
            with self.np.load(path, allow_pickle=False) as stored:
                arrays = {name: stored[name].copy() for name in stored.files}
            arrays['embeddings'] = value
            self.np.savez_compressed(path, **arrays)
            new = pilot.record(path)
            receipt['outputs'] = [new if item == old else item for item in receipt['outputs']]
            receipt['projection'] = new
        self.change_receipt(stage, update)

    def test_matched_saved_runs_pass_without_reading_original_models(self):
        result = self.compare()
        self.assertTrue(result['protocol_parity'])
        self.assertTrue(result['numerical_equivalence_within_frozen_tolerances'])
        self.assertFalse(result['runtime_records_identical'])
        self.assertFalse(result['full_cohort_results_preserved_claim'])
        self.assertEqual(result['scientific_projection_fits'], 0)
        self.assertTrue((self.root / 'comparison/comparison.json').is_file())
        self.assertIn('Trustworthiness and neighborhood overlap are informational',
                      (self.root / 'comparison/comparison.md').read_text())

    def test_pca_rotation_reflection_translation_is_allowed_but_scale_is_checked(self):
        transformed = self.xy[:, ::-1] * self.np.array([-1., 1.]) + 100
        self.replace_coordinates('pca', transformed)
        result = self.compare('rigid')
        self.assertEqual(result['stages']['pca']['status'], 'numerically_equivalent_within_frozen_tolerance')
        self.replace_coordinates('pca', self.xy * 2)
        result = self.compare('scaled')
        self.assertEqual(result['stages']['pca']['status'], 'review_required')
        self.assertAlmostEqual(result['stages']['pca']['geometry']['scale_ratio'], 2)
        self.assertLess(result['stages']['pca']['geometry']['scale_adjusted_residual'], 1e-12)

    def test_nonlinear_differences_require_review_despite_unchanged_neighborhoods(self):
        self.replace_coordinates('umap', self.xy * 1.01)
        result = self.compare()
        self.assertEqual(result['status'], 'review_required')
        self.assertTrue(result['protocol_parity'])
        self.assertEqual(result['stages']['umap']['status'], 'review_required')
        self.assertEqual(result['stages']['umap']['neighborhood']['cross_host_neighbor_overlap'], 1)

    def test_missing_or_failed_stage_never_passes_or_creates_output(self):
        entry = self.host_specs['hpc']['receipts'].pop('density0')
        with self.assertRaisesRegex(ValueError, 'five closed'):
            self.compare()
        self.host_specs['hpc']['receipts']['density0'] = entry
        self.change_receipt('density0', lambda result: result.update(phase_complete=False))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.compare()
        self.assertFalse((self.root / 'comparison').exists())

    def test_tampered_saved_output_is_integrity_failure(self):
        receipt = json.loads(Path(self.host_specs['hpc']['receipts']['umap']['path']).read_text())
        Path(receipt['projection']['path']).write_bytes(b'changed after closure')
        with self.assertRaises(ValueError):
            self.compare()
        self.assertFalse((self.root / 'comparison').exists())

    def test_wrong_cohort_full_cohort_or_core_rejected_before_geometry(self):
        for key, value in [('model_count', 5072), ('model_order_sha256', 'f' * 64),
                           ('retained_reference_positions', list(range(58, 952)))]:
            plan = deepcopy(self.host_plans['hpc'])
            plan['inputs'][key] = value
            self.host_specs['hpc']['plan'] = put(self.root / 'hpc/execution-plan.json', plan)
            contract_path = self.root / 'contract.json'
            contract = json.loads(contract_path.read_text())
            contract['host_plans']['hpc'] = self.host_specs['hpc']['plan']
            self.manifest['contract'] = put(contract_path, contract)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'input identity'):
                self.compare()
        self.assertFalse((self.root / 'comparison').exists())

    def test_native_source_drift_or_fit_count_drift_is_integrity_failure(self):
        self.change_receipt('pca', lambda result: result['native_sources']['scripts.pca'].update(sha256='f' * 64))
        with self.assertRaisesRegex(ValueError, 'Native source bytes'):
            self.compare()
        self.change_receipt('pca', lambda result: result['native_sources']['scripts.pca'].update(
            sha256=pilot.LIBRARY_SOURCE_HASHES['scripts.pca']))
        self.change_receipt('pca', lambda result: result['scientific_fits'].update(PCA=2))
        with self.assertRaisesRegex(ValueError, 'fit accounting'):
            self.compare()

    def test_relocation_changes_only_paths_and_preserves_content_binding(self):
        locations = {}
        def move(receipt):
            old = receipt['projection']
            remote = dict(old, path='/remote/hpc/umap/projection.npz')
            locations[remote['path']] = old
            receipt['projection'] = remote
            receipt['outputs'] = [remote if item == old else item for item in receipt['outputs']]
        self.change_receipt('umap', move)
        self.host_specs['hpc']['artifact_locations'] = locations
        self.assertTrue(self.compare()['protocol_parity'])
        locations['/remote/hpc/umap/projection.npz']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Relocation changed'):
            self.compare('bad-relocation')

    def test_relaxed_thresholds_are_not_accepted(self):
        path = self.root / 'contract.json'
        contract = json.loads(path.read_text())
        contract['thresholds']['pca_relative_pairwise_error'] = .8
        self.manifest['contract'] = put(path, contract)
        with self.assertRaisesRegex(ValueError, 'threshold'):
            self.compare()

    def test_density_cast_hash_must_match_the_actual_raw_features(self):
        self.feature_diagnostic['float32_sha256'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'float32 feature proof'):
            self.compare()

    def test_bounded_distance_and_hash_calculation_matches_tiny_reference(self):
        matrix = self.np.arange(15, dtype=self.np.float16).reshape(5, 3)
        matrix.flags.writeable = False
        raw = hashlib.sha256(matrix.tobytes()).hexdigest()
        check = Mock()
        distances, proof = subject._matrix_distances(matrix, raw, check=check, block_bytes=40)
        self.np.testing.assert_allclose(distances, subject._distances(matrix.astype(self.np.float64)))
        self.assertEqual(proof['float32_sha256'], hashlib.sha256(matrix.astype(self.np.float32).tobytes()).hexdigest())
        self.assertGreater(check.call_count, 3)
        with self.assertRaisesRegex(ValueError, 'Raw F16'):
            subject._matrix_distances(matrix, '0' * 64, block_bytes=40)
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            subject._matrix_distances(matrix, raw, check=Mock(side_effect=RuntimeError('cancelled')), block_bytes=40)

    def test_public_feature_reader_refuses_a_different_shape(self):
        path = self.root / 'tiny.npy'
        self.np.save(path, self.np.zeros((5, 3), dtype=self.np.float16))
        with self.assertRaisesRegex(ValueError, '64x399171'):
            subject.feature_distances(pilot.record(path), 'a' * 64)


if __name__ == '__main__':
    unittest.main()
