"""Assemble the shared RBP atlas from hash-bound saved results; no science/deploy.

The caller supplies the exact publication-plan SHA256. Optional TM-state inputs
carry a separate scientific plan/source binding. A complete parity publication
requires the saved HDBSCAN, Bonsai, medoid comparisons, variance regions and
state catalog, while earlier baseline plans remain supported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from publish_atlas import build, export_prepared, structure_delivery
from rbp_panel_exports import record
from rbp_public_states import attach, load_json

TARGET = 'RBP_P02925'
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
HEX40 = re.compile(r'[0-9a-f]{40}\Z')


def need(value, message):
    if not value: raise ValueError(message)


def exact_input(entry):
    """Enrich an approved path/SHA descriptor without weakening a stated size."""
    need(isinstance(entry, dict) and isinstance(entry.get('path'), str)
         and isinstance(entry.get('sha256'), str) and HEX64.fullmatch(entry['sha256']),
         'Expected frozen artifact path and SHA256')
    current = record(entry['path'])
    need(current['sha256'] == entry['sha256']
         and ('bytes' not in entry or type(entry['bytes']) is int and current['bytes'] == entry['bytes']),
         'Frozen artifact hash/size changed')
    return current


def relative_file(root, name):
    need(isinstance(name, str) and name and not Path(name).is_absolute()
         and all(p not in ('', '.', '..') for p in name.split('/')) and '\\' not in name,
         'Expected a safe relative public file')
    root = Path(root).absolute(); path = root/name
    need(path.resolve().is_relative_to(root.resolve()), 'Public file leaves source root')
    return record(path)


def inventory(root):
    root = Path(root).absolute(); rows = []
    need(root.is_dir() and not any(p.is_symlink() for p in [root, *root.parents]),
         'Expected a regular publication directory')
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(), 'Publication directory contains a symbolic link')
        if path.is_file():
            row = relative_file(root, path.relative_to(root).as_posix())
            rows.append(dict(path=path.relative_to(root).as_posix(), bytes=row['bytes'], sha256=row['sha256']))
    return rows


def parity_contract(plan):
    required = plan.get('require_rbp_parity', False)
    need(type(required) is bool, 'require_rbp_parity must be boolean')
    state = plan.get('state_artifacts')
    if state is not None:
        need(TARGET in plan['targets'] and isinstance(state, dict), 'RBP state inputs require the published RBP target')
        need(isinstance(state.get('source_bindings'), dict)
             and isinstance(state.get('plan_sha256'), str) and HEX64.fullmatch(state['plan_sha256'])
             and isinstance(state.get('source_commit'), str) and HEX40.fullmatch(state['source_commit'])
             and state['source_bindings'].get('plan_sha256') == state['plan_sha256']
             and state['source_bindings'].get('consumer_commit') == state['source_commit'],
             'State scientific plan/source binding differs')
    if required:
        target = plan['targets'].get(TARGET, {})
        need(structure_delivery(plan) == 'representatives', 'RBP parity requires representative-only coordinates')
        need(target.get('density_artifacts') and state is not None, 'RBP parity requires saved HDBSCAN and TM-state artifacts')
        need(target.get('bonsai') and target.get('medoid_tm')
             and target['density_artifacts'].get('bonsai') and target['density_artifacts'].get('medoid_tm')
             and target.get('region_artifacts'), 'RBP parity requires both saved Bonsai/TM views and variance-region evidence')
    if state is not None:
        need(plan['targets'][TARGET].get('density_artifacts'), 'State attachment requires saved Primary and HDBSCAN views')
        return dict(catalog=exact_input(state['catalog']), scores=exact_input(state['scores']),
                    source_bindings=state['source_bindings'], plan_sha256=state['plan_sha256'], source_commit=state['source_commit'])
    return None


def method_aliases(plan):
    path = (Path(plan['study']['path']).absolute().parent/'model-types.json')
    frozen = plan.get('frozen_files', {}).get(str(path))
    need(isinstance(frozen, str) and HEX64.fullmatch(frozen), 'State publication requires frozen model-type label rules')
    entry = exact_input(dict(path=str(path), sha256=frozen))
    rules = load_json(path.read_bytes())['rules']
    need(isinstance(rules, list) and rules, 'Expected model-type label rules')
    aliases = {row['key']: row['name'] for row in rules}
    need(len(aliases) == len(rules) and all(isinstance(key, str) and key and isinstance(value, str) and value
         for key, value in aliases.items()), 'Model-type label aliases are duplicated or empty')
    return aliases, entry


def checked_views(prepared):
    target = prepared['target_views'][TARGET]
    need(set(target['views']) == {'primary', 'density'}, 'Exactly Primary and HDBSCAN publication views required')
    views = {}
    for key, binding in target['views'].items():
        need(binding['id'] == key and binding['source_view_id'] == key,
             'Prepared public view identity differs')
        current = relative_file(prepared['source'], binding['data'])
        need(current['sha256'] == binding['source_data_sha256'], 'Prepared scientific JSON changed')
        views[key] = dict(file=binding['data'], sha256=binding['source_data_sha256'])
    return target, views


def verify_attachment(source, result):
    need(isinstance(result, dict) and isinstance(result.get('files'), list), 'State converter did not return a public file allowlist')
    rows = result['files']; names = [row['path'] for row in rows]
    need(len(set(names)) == len(names) and 'tm-states.json' in names, 'State converter catalog is absent or duplicated')
    for row in rows:
        current = relative_file(source, row['path'])
        need(current['sha256'] == row['sha256'] and current['bytes'] == row['bytes'],
             'Attached state artifact differs from returned allowlist')
    catalog = load_json(Path(relative_file(source, 'tm-states.json')['path']).read_bytes())
    need(catalog == result['tm_state_selection_catalog']
         and catalog.get('schema_version') == 2 and catalog.get('kind') == 'tm_state_selection_catalog'
         and catalog.get('targets', {}).get(TARGET) == result['target_entry'],
         'Attached schema-two state catalog differs from converter receipt')
    return rows


def publish(plan_path, *, expected_plan_sha256):
    need(isinstance(expected_plan_sha256, str) and HEX64.fullmatch(expected_plan_sha256),
         'Exact publication-plan SHA256 required')
    plan_entry = exact_input(dict(path=str(plan_path), sha256=expected_plan_sha256))
    plan = load_json(Path(plan_entry['path']).read_bytes())
    state = parity_contract(plan)
    aliases, aliases_entry = method_aliases(plan) if state is not None else ({}, None)
    receipt_path = Path(plan.get('receipt', str(Path(plan['output']).with_name(Path(plan['output']).name+'-receipt.json')))).absolute()
    need(not receipt_path.exists() and not any(p.is_symlink() for p in [receipt_path, *receipt_path.parents])
         and not any(receipt_path.is_relative_to(Path(plan[key]).absolute()) for key in ('source', 'output')),
         'Publication receipt needs a new path outside source and site')
    prepared = build(plan_entry['path'], export_site=False)
    exact_input(plan_entry)
    attachment, views, attached = None, {}, []
    if state is not None:
        target, views = checked_views(prepared)
        attachment = attach(prepared['source'], state['catalog'], state['scores'], views,
            expected_source_bindings=state['source_bindings'], expected_model_order=target['models'],
            sequence_sha256=target['sequence_sha256'], plan_sha256=state['plan_sha256'], source_commit=state['source_commit'],
            method_labels=aliases)
        attached = verify_attachment(prepared['source'], attachment)
        checked_views(prepared)  # conversion must not alter a scientific dataset
        exact_input(state['catalog']); exact_input(state['scores']); exact_input(aliases_entry)
    exact_input(plan_entry)
    export_prepared(prepared)  # the single shared-renderer export; never deploys
    source_files = inventory(prepared['source'])
    manifest = exact_input(record(prepared['publication_manifest']))
    publication = load_json(Path(manifest['path']).read_bytes())
    need(publication['files'] == source_files, 'Final source file allowlist differs')
    final = {row['path']: row for row in publication['files']}
    need(all(final.get(row['path']) == row for row in attached), 'Attached state files missing from final publication manifest')
    site_files = inventory(prepared['output'])
    site = {row['path']: row for row in site_files}
    need(all(site.get(row['path']) == row for row in source_files), 'Exported site differs from saved source artifacts')
    exported = relative_file(prepared['output'], 'export-manifest.json')
    exporter = load_json(Path(exported['path']).read_bytes())
    need(exporter.get('input_manifest_sha256') == manifest['sha256'] and exporter.get('scientific_execution') is False
         and exporter.get('files') == [row for row in site_files if row['path'] != 'export-manifest.json'],
         'Shared export receipt differs from the complete prepared site')
    encoded = json.dumps(site_files, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    receipt = dict(schema_version=1, status='saved_rbp_publication_prepared', scientific_execution=False,
        public_deployment=False, publication_plan=plan_entry, publication_manifest=manifest,
        source=prepared['source'], output=prepared['output'], renderer_commit=prepared['renderer_commit'],
        coordinate_delivery=prepared['coordinate_delivery'], raw_views=views,
        state_source=state, method_label_rules=aliases_entry, attached_files=attached, export_manifest=exported,
        site_files=site_files, site_sha256=hashlib.sha256(encoded).hexdigest())
    if attachment is not None: receipt['state_target'] = attachment['target_entry']
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    return dict(prepared=prepared, state_attachment=attachment, receipt=record(receipt_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path); parser.add_argument('--sha256', required=True)
    arguments = parser.parse_args()
    result = publish(arguments.plan, expected_plan_sha256=arguments.sha256)
    print('Saved RBP atlas prepared:', result['prepared']['output'])
    print('Local hash receipt:', result['receipt']['path'])
