"""Saved metadata/opaque byte fixtures only; no NumPy/native engines needed."""
import builtins
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mimas_pilot as pilot
import pilot_compare as comparison
import pilot_saved_manifest as subject


def put(path, value):
    path.write_text(json.dumps(value))
    return pilot.record(path)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        names = ['model-' + str(index) for index in range(64)]
        self.expected = dict(target='SERCA_P04191', model_count=64, construct_residues=993,
            retained_reference_positions=list(range(57, 951)), pair_mask_sha256=pilot.SERCA_PAIR_MASK,
            feature_sha256='a' * 64, model_order_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
            features=dict(path='/must-not-open/features.npy', bytes=51094016, sha256='b' * 64))
        host_plans, hosts = {}, {}
        self.plans = {}
        for hostname in ('pax', 'mimas'):
            root = self.root / hostname
            root.mkdir()
            rows = [dict(model=name, local_path='/must-not-open/' + hostname + '/' + name + '.pdb',
                         original_sha256=hashlib.sha256(name.encode()).hexdigest(), flags={'kept': True})
                    for name in names]
            inputs = {key:value for key, value in self.expected.items() if key != 'target'}
            inputs['members'] = put(root / 'members.json', dict(members=rows))
            module = pilot.record(pilot.__file__)
            plan = dict(execution_host=hostname, phase='pilot64', target='SERCA_P04191',
                stages=list(comparison.STAGES), maximum_projection_fits=9, maximum_hdbscan_extractions=6,
                runtime_python='/runtime/' + hostname + '/bin/python', runtime_prefix='/runtime/' + hostname,
                bootstrap=dict(python='/bootstrap/' + hostname), inputs=inputs, workers=1,
                frozen_files=[module], native=dict(library_commit=pilot.LIBRARY_COMMIT,
                    density_source=dict(path='/source/' + hostname, commit=pilot.DENSITY_COMMIT, sha256=pilot.DENSITY_SHA256),
                    cluster_config=put(root / 'cluster-config.json',
                                       dict(schema_version=1, settings=pilot.DEFAULT_COMPONENT_SETTINGS))))
            self.plans[hostname] = plan
            plan_entry = put(root / 'execution-plan.json', plan)
            host_plans[hostname] = plan_entry
            receipts = {}
            for stage in comparison.STAGES:
                artifact = root / (stage + '.npz')
                artifact.write_bytes(b'Opaque fake closed saved array; must not be numerically parsed.')
                output = pilot.record(artifact)
                counts = dict(PCA=0, UMAP=0, TSNE=0, HDBSCAN=0)
                special = {}
                if stage == 'pca':
                    counts['PCA'] = 1
                elif stage == 'umap':
                    counts['UMAP'] = 1
                elif stage == 'tsne':
                    counts.update(PCA=1, TSNE=1)
                elif stage == 'labels':
                    counts['UMAP'] = 2
                    special['repeat_UMAP_fits_maximum'] = 2
                else:
                    counts.update(UMAP=3, HDBSCAN=6)
                    special.update(full_recursive_HDBSCAN=False, default_cluster_labels=False)
                key = 'projection' if stage in ('pca', 'umap', 'tsne') else 'labels'
                result = dict(schema_version=1, stage=stage, status='diagnostic_not_full_clustering',
                    phase_complete=True, full_clustering_complete=False, target=plan['target'], inputs=inputs,
                    input_fingerprint=pilot.compact_sha(dict(target=plan['target'], inputs=inputs, native=plan['native'], workers=1)),
                    models=names, local_paths=[row['local_path'] for row in rows], model_flags=[row['flags'] for row in rows],
                    native_library_commit=pilot.LIBRARY_COMMIT,
                    native_sources={key:dict(path='/source/' + key, bytes=1, sha256=sha)
                                    for key, sha in pilot.LIBRARY_SOURCE_HASHES.items()}, module_source=module,
                    input_integrity=dict(file_sha256_verified_before_map=True, raw_float16_sha256_verified=True,
                                         readonly_mapping=True, identities_unchanged_after_phase=True),
                    coordinate_parses=0, scoring_calls=0, molecular_exports=0,
                    scientific_fits=counts, outputs=[output], **{key:output}, **special)
                receipts[stage] = put(root / (stage + '-receipt.json'), result)
            completion = dict(status='diagnostic_pilot_complete', phase='pilot64', fit_ready=True,
                plan_sha256=plan_entry['sha256'], stage_receipts=receipts, maximum_projection_fits=9,
                maximum_hdbscan_extractions=6, standard_deliverables_complete=False)
            hosts[hostname] = dict(plan=plan_entry, completion=put(root / 'completion.json', completion),
                runtime=put(root / 'runtime.json', dict(executable=plan['runtime_python'],
                    prefix=plan['runtime_prefix'], packages={'numpy': 'recorded'}, bootstrap=plan['bootstrap'])),
                relocations=put(root / 'relocations.json', {}))
        self.contract_path = self.root / 'comparison-contract.json'
        self.contract = dict(schema_version=1, operation='paired-pilot-comparison', expected=self.expected,
            thresholds=comparison.DEFAULT_THRESHOLDS, comparator_source=pilot.record(comparison.__file__), host_plans=host_plans)
        self.request = dict(schema_version=1, operation='assemble-paired-pilot-saved-manifest',
                            contract=put(self.contract_path, self.contract), hosts=hosts)

    def assemble(self, name='derived-manifest.json'):
        return subject.assemble(self.request, self.root / name)

    def mutate_completion(self, host, mutate):
        entry = self.request['hosts'][host]['completion']
        path = Path(entry['path'])
        value = json.loads(path.read_text())
        mutate(value)
        self.request['hosts'][host]['completion'] = put(path, value)

    def mutate_receipt(self, stage, mutate):
        completion_path = Path(self.request['hosts']['mimas']['completion']['path'])
        completion = json.loads(completion_path.read_text())
        path = Path(completion['stage_receipts'][stage]['path'])
        receipt = json.loads(path.read_text())
        mutate(receipt)
        completion['stage_receipts'][stage] = put(path, receipt)
        self.request['hosts']['mimas']['completion'] = put(completion_path, completion)

    def test_success_is_metadata_only_and_does_not_claim_comparison_passed(self):
        original = builtins.__import__
        def no_science(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'umap', 'sklearn', 'numba', 'scipy'}:
                raise AssertionError('No numerical imports in manifest assembly')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=no_science):
            result = self.assemble()
        self.assertEqual(result['operation'], 'paired-pilot-saved-comparison')
        self.assertEqual(result['assembly']['status'], 'saved_inputs_verified_not_compared')
        self.assertEqual(result['assembly']['numerical_comparisons'], 0)
        self.assertEqual(result['assembly']['feature_matrices_opened'], 0)
        self.assertEqual(set(result['hosts']['pax']['receipts']), set(comparison.STAGES))

    def test_completion_must_be_closed_success_for_exact_plan(self):
        for field, value in [('status', 'failed'), ('plan_sha256', '0' * 64),
                             ('fit_ready', False), ('phase', 'umap-full')]:
            saved = deepcopy(self.request)
            before = Path(self.request['hosts']['mimas']['completion']['path']).read_text()
            self.mutate_completion('mimas', lambda result:result.update(**{field:value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'completion'):
                self.assemble()
            Path(saved['hosts']['mimas']['completion']['path']).write_text(before)
            self.request = saved
        self.assertFalse((self.root / 'derived-manifest.json').exists())

    def test_missing_failed_or_excess_fit_stage_cannot_assemble(self):
        self.mutate_receipt('umap', lambda result:result.update(phase_complete=False))
        with self.assertRaises(ValueError):
            self.assemble()
        self.mutate_receipt('umap', lambda result:result.update(phase_complete=True))
        self.mutate_receipt('umap', lambda result:result['scientific_fits'].update(UMAP=2))
        with self.assertRaisesRegex(ValueError, 'fit accounting'):
            self.assemble()
        self.mutate_completion('mimas', lambda result:result['stage_receipts'].pop('density0'))
        with self.assertRaisesRegex(ValueError, 'five completed'):
            self.assemble()

    def test_changed_artifact_is_rejected_as_bytes_without_parsing(self):
        (self.root / 'mimas/umap.npz').write_bytes(b'Changed after closure')
        with self.assertRaises(ValueError):
            self.assemble()
        self.assertFalse((self.root / 'derived-manifest.json').exists())

    def test_source_pin_and_runtime_identity_are_verified(self):
        runtime_path = Path(self.request['hosts']['mimas']['runtime']['path'])
        runtime = json.loads(runtime_path.read_text())
        runtime['prefix'] = '/other/runtime'
        self.request['hosts']['mimas']['runtime'] = put(runtime_path, runtime)
        with self.assertRaisesRegex(ValueError, 'Runtime facts'):
            self.assemble()
        self.mutate_receipt('pca', lambda result:result.update(native_library_commit='f' * 40))
        with self.assertRaisesRegex(ValueError, 'source differs'):
            self.assemble()

    def test_local_copy_of_bound_remote_plan_may_change_only_path(self):
        entry = self.contract['host_plans']['mimas']
        self.contract['host_plans']['mimas'] = dict(entry, path='/remote/frozen/execution-plan.json')
        self.request['contract'] = put(self.contract_path, self.contract)
        self.assertEqual(self.assemble()['assembly']['status'], 'saved_inputs_verified_not_compared')

    def test_explicit_relocation_map_preserves_remote_output_hashes(self):
        locations = {}
        def relocate(receipt):
            old = receipt['projection']
            remote = dict(old, path='/remote/run/umap.npz')
            locations[remote['path']] = old
            receipt['projection'] = remote
            receipt['outputs'] = [remote]
        self.mutate_receipt('umap', relocate)
        path = Path(self.request['hosts']['mimas']['relocations']['path'])
        self.request['hosts']['mimas']['relocations'] = put(path, locations)
        result = self.assemble()
        self.assertEqual(result['hosts']['mimas']['artifact_locations'], locations)
        locations['/remote/run/umap.npz']['sha256'] = 'f' * 64
        self.request['hosts']['mimas']['relocations'] = put(path, locations)
        with self.assertRaisesRegex(ValueError, 'Relocation changed'):
            self.assemble('bad-map.json')

    def test_existing_manifest_is_preserved_and_cancellation_stops_before_write(self):
        output = self.root / 'already.json'
        output.write_text('preserve')
        with self.assertRaises(ValueError):
            self.assemble('already.json')
        self.assertEqual(output.read_text(), 'preserve')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            subject.assemble(self.request, self.root / 'cancelled.json',
                             check=lambda:(_ for _ in ()).throw(RuntimeError('cancelled')))
        self.assertFalse((self.root / 'cancelled.json').exists())


if __name__ == '__main__':
    unittest.main()
