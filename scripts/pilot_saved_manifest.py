"""Assemble verified CLOSED pilot metadata for a later saved-output comparison.

No numerical library, model file or feature matrix is opened. Bound output files
are checked as bytes only. Producing this manifest is not a numerical comparison
and cannot claim that either host's scientific results are preserved.
"""
import argparse
import json
from pathlib import Path
import sys


def _comparison():
    source = str(Path(__file__).absolute().parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    import pilot_compare
    return pilot_compare


def _fit_counts(stage, receipt, need):
    counts = receipt.get('scientific_fits', {})
    need(set(counts) == {'PCA', 'UMAP', 'TSNE', 'HDBSCAN'}
         and all(type(value) is int and value >= 0 for value in counts.values()),
         'Complete integer fit accounting required')
    expected = dict(PCA=0, UMAP=0, TSNE=0, HDBSCAN=0)
    if stage == 'pca':
        expected['PCA'] = 1
    elif stage == 'umap':
        expected['UMAP'] = 1
    elif stage == 'tsne':
        expected.update(PCA=1, TSNE=1)
    elif stage == 'labels':
        need(counts['UMAP'] in (0, 2), 'Primary repeat count is outside the native seed budget')
        expected['UMAP'] = counts['UMAP']
        need(receipt.get('repeat_UMAP_fits_maximum') == 2, 'Primary repeat ceiling differs')
    else:
        expected.update(UMAP=3, HDBSCAN=6)
        need(receipt.get('full_recursive_HDBSCAN') is False
             and receipt.get('default_cluster_labels') is False,
             'Initial density diagnostics are not a completed recursive/default clustering')
    need(counts == expected, 'Actual stage fit accounting differs: ' + stage)
    return counts


def assemble(request, output, *, check=lambda: None):
    comparison = _comparison()
    need = comparison.need
    if isinstance(request, (str, Path)):
        request = json.loads(Path(request).read_text())
    need(request.get('schema_version') == 1
         and request.get('operation') == 'assemble-paired-pilot-saved-manifest',
         'Explicit saved-manifest assembly request required')
    contract_entry = request['contract']
    contract = comparison._read(contract_entry, check=check)
    need(contract.get('schema_version') == 1 and contract.get('operation') == 'paired-pilot-comparison'
         and contract.get('thresholds') == comparison.DEFAULT_THRESHOLDS,
         'Frozen matched-pilot comparison contract required')
    comparison._bound(contract['comparator_source'], check=check)
    expected = contract['expected']
    pilot = comparison._pilot()
    need(expected.get('target') == 'SERCA_P04191' and expected.get('model_count') == 64
         and expected.get('construct_residues') == 993
         and expected.get('retained_reference_positions') == list(range(57, 951))
         and expected.get('pair_mask_sha256') == pilot.SERCA_PAIR_MASK,
         'Only the frozen SERCA-64 accepted core may be assembled')
    need(set(request['hosts']) == {'workstation', 'hpc'}
         and set(contract['host_plans']) == {'workstation', 'hpc'}, 'Exactly two planned hosts are required')
    hosts, completions, identities, protocols, totals = {}, {}, {}, {}, {}
    for name in ('workstation', 'hpc'):
        check()
        spec = request['hosts'][name]
        # A relocation map is an explicit bound JSON file, including when empty.
        locations = comparison._read(spec['relocations'], check=check)
        need(isinstance(locations, dict) and all(isinstance(key, str) and Path(key).is_absolute()
             for key in locations), 'Explicit absolute relocation-map JSON required')
        for descriptor in locations.values():
            comparison._content(descriptor)
            need(isinstance(descriptor.get('path'), str) and Path(descriptor['path']).is_absolute(),
                 'Relocated artifact destination must be absolute')
        plan_entry = spec.get('plan', contract['host_plans'][name])
        need(comparison._content(plan_entry) == comparison._content(contract['host_plans'][name]),
             'Host plan content differs from the frozen comparison contract')
        plan = comparison._read(plan_entry, locations=locations, check=check)
        protocols[name] = comparison._protocol(plan, expected, locations, check)
        need(plan.get('execution_host', 'hpc') == name, 'Execution plan is for the other host')
        names, paths, identities[name] = comparison._members(plan, expected, locations, check)
        flags = [row['flags'] for row in identities[name]]
        completion = comparison._read(spec['completion'], locations=locations, check=check)
        need(completion.get('status') == 'diagnostic_pilot_complete'
             and completion.get('phase') == 'pilot64' and completion.get('fit_ready') is True
             and completion.get('plan_sha256') == plan_entry['sha256']
             and completion.get('maximum_projection_fits') == 9
             and completion.get('maximum_hdbscan_extractions') == 6
             and completion.get('standard_deliverables_complete') is False,
             'A closed successful matching pilot completion is required')
        receipts = completion.get('stage_receipts', {})
        need(set(receipts) == set(comparison.STAGES), 'All five completed stage receipts are required')
        fingerprint = comparison._compact(dict(target=plan['target'], inputs=plan['inputs'],
            native=plan['native'], workers=plan.get('workers', 1)))
        fit_total = dict(PCA=0, UMAP=0, TSNE=0, HDBSCAN=0)
        adapter_sources = set()
        for stage in comparison.STAGES:
            receipt = comparison._read(receipts[stage], locations=locations, check=check)
            need(receipt.get('schema_version') == 1 and receipt.get('stage') == stage
                 and receipt.get('status') == 'diagnostic_not_full_clustering'
                 and receipt.get('phase_complete') is True
                 and receipt.get('full_clustering_complete') is False
                 and receipt.get('target') == expected['target']
                 and receipt.get('inputs') == plan['inputs']
                 and receipt.get('input_fingerprint') == fingerprint
                 and receipt.get('models') == names and receipt.get('local_paths') == paths
                 and receipt.get('model_flags') == flags,
                 'Stage closure/cohort/protocol differs: ' + stage)
            need(receipt.get('input_integrity') == dict(file_sha256_verified_before_map=True,
                 raw_float16_sha256_verified=True, readonly_mapping=True, identities_unchanged_after_phase=True)
                 and all(receipt.get(key) == 0 for key in ('coordinate_parses', 'scoring_calls', 'molecular_exports')),
                 'Stage raw-input integrity or distance-only scope differs')
            adapter_sources.add(comparison._source_proof(receipt, plan))
            counts = _fit_counts(stage, receipt, need)
            for key in fit_total:
                fit_total[key] += counts[key]
            outputs = receipt.get('outputs', [])
            need(outputs and len({item['path'] for item in outputs}) == len(outputs),
                 'Complete unique stage artifact list required')
            for descriptor in outputs:
                comparison._bound(descriptor, locations=locations, check=check)
            key = 'projection' if stage in ('pca', 'umap', 'tsne') else 'labels'
            need(receipt.get(key) in outputs, 'Stage sidecar is not bound by its receipt')
        need(sum(fit_total[key] for key in ('PCA', 'UMAP', 'TSNE')) <= 9
             and fit_total['HDBSCAN'] == 6, 'Completed pilot exceeded its fit scope')
        need(len(adapter_sources) == 1, 'Pilot adapter changed between completed stages')
        protocols[name]['adapter_source_sha256'] = next(iter(adapter_sources))
        runtime = comparison._read(spec['runtime'], locations=locations, check=check)
        need(runtime.get('executable') == plan['runtime_python']
             and runtime.get('prefix') == plan['runtime_prefix']
             and isinstance(runtime.get('packages'), dict)
             and runtime.get('bootstrap') == plan.get('bootstrap'),
             'Runtime facts do not identify the approved host environment')
        hosts[name] = dict(plan=plan_entry, runtime=spec['runtime'], receipts=receipts,
                           artifact_locations=locations)
        completions[name] = dict(completion=spec['completion'], relocations=spec['relocations'])
        totals[name] = fit_total
    need(identities['workstation'] == identities['hpc'], 'Original cohort identities or physicality flags differ')
    need(protocols['workstation'] == protocols['hpc'], 'Host native settings/source protocol differs')
    value = dict(schema_version=1, operation='paired-pilot-saved-comparison',
        contract=contract_entry, hosts=hosts,
        assembly=dict(status='saved_inputs_verified_not_compared', completions=completions,
                      actual_fit_accounting=totals, scientific_fits=0, numerical_comparisons=0,
                      original_coordinate_files_opened=0, feature_matrices_opened=0))
    destination = Path(output).absolute()
    need('..' not in destination.parts and not destination.exists()
         and not any(path.is_symlink() for path in [destination, *destination.parents]),
         'A fresh non-symlink saved-manifest destination is required')
    check()
    # The plan-preparation controller creates the owned output parent; do not
    # create an arbitrary output tree or any runtime/cache directory here.
    need(destination.parent.is_dir(), 'Owned saved-manifest parent must already exist')
    pilot.save(destination, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request')
    parser.add_argument('output')
    args = parser.parse_args()
    value = assemble(args.request, args.output)
    print(json.dumps(dict(status=value['assembly']['status'], output=args.output)))


if __name__ == '__main__':
    main()
