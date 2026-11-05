"""Approved, serialized primary/density clustering from accepted feature caches.

No scientific engines load on import. This stage prepares labels for subsequent
representative/state selection; it never exports models or publishes a site.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path
import shutil
import time

from snapshot import sha256


SCOPE = dict(maximum_targets=2, maximum_primary_projection_fits_per_target=6,
             maximum_density_umap_fits_per_target=15,
             maximum_projection_fits=42, maximum_hdbscan_extractions=60,
             native_TM_calls=0, coordinate_parses=0, scoring_calls=0,
             molecular_exports=0, density_mode='casp17_recursive')


def need(value, message):
    if not value:
        raise ValueError(message)


def bound(entry):
    path = Path(entry['path']).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents])
         and path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'],
         'Bound input changed: '+str(path))
    return path


def read(entry):
    return json.loads(bound(entry).read_text())


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def record(path):
    path = Path(path).absolute()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def registry(descriptor, intake, native, *, check=lambda: None,
             checkpoint=lambda _: None):
    """Bind every coordinate byte identity without parsing its atoms."""
    axes = read(descriptor['axes'])
    names = axes['names']
    need(intake['target'] == descriptor['target'] and native['status'] == 'validated'
         and len(names) == descriptor['model_count'] == len(intake['models'])
         and len(set(names)) == len(names), 'Target/cohort descriptor differs')
    source = {r['model_id']: r for r in intake['models']}
    rows = [r for r in native['models'] if r['target'] == descriptor['target']]
    scores = {r['model_id']: r for r in rows}
    need(len(source) == len(scores) == len(rows) == len(names)
         and set(source) == set(scores) == set(names), 'Accepted cohort identities differ')
    members = []
    for number, name in enumerate(names, 1):
        check()
        original, scored = source[name], scores[name]
        path = Path(original['path']).absolute()
        need(path.is_file() and path.is_relative_to(Path(intake['analysis_dir']).absolute())
             and not any(p.is_symlink() for p in [path, *path.parents])
             and original['sha256'] == scored['sha256'] == sha256(path),
             'Original coordinate identity changed: '+name)
        members.append(dict(model=name, source_path=str(path), sha256=original['sha256']))
        if number % 250 == 0:
            checkpoint(descriptor['target']+':original-hashes:'+str(number))
    checkpoint(descriptor['target']+':original-hashes-closed')
    return members


def primary_rows(manifest, members):
    with Path(manifest['knn_clustering']['labels_csv']).open() as stream:
        rows = list(csv.DictReader(stream))
    need([r['model'] for r in rows] == [r['model'] for r in members]
         and [str(Path(r['source_path']).absolute()) for r in rows]
         == [r['source_path'] for r in members]
         and all(r['cluster_label'] for r in rows), 'Primary label identities/order differ')
    return rows


def run_density(primary, members, descriptor, native_source, output, *,
                workers, check, checkpoint, log, adapters=None):
    import numpy as np
    if adapters is None:
        from src.analysis.feature_workspace import feature_input_workspace, feature_sha256
        from rbp_density import run
        from rbp_parity import write_density_labels
        adapters = dict(workspace=feature_input_workspace, digest=feature_sha256,
                        run=run, labels=write_density_labels)
    manifest = read(primary['manifest'])
    rows = primary_rows(manifest, members)
    entry = primary['retained_features']
    report = read(descriptor['terminal_report'])
    derivation = manifest['derivation']; inherited = derivation['retained_feature_proof']
    need(manifest['terminal_detection']['retained_reference_positions'] == report['retained_reference_positions']
         and manifest['terminal_detection']['pair_mask_sha256'] == report['pair_mask_sha256']
         and derivation['accepted_terminal_report'] == descriptor['terminal_report']
         and all(inherited[key] == entry[key] for key in
                 ('path','shape','dtype','file_sha256','feature_sha256')),
         'Primary-to-density feature/mask provenance differs')
    path = Path(entry['path'])
    need(sha256(path) == entry['file_sha256'], 'Retained feature file changed')
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        positions = report['retained_reference_positions']
        names = [r['model'] for r in members]
        need(matrix.dtype == np.float16 and entry['dtype'] == 'float16'
             and list(matrix.shape) == entry['shape']
             == [len(names), len(positions)*(len(positions)-1)//2]
             and matrix.flags.c_contiguous and not matrix.flags.writeable
             and adapters['digest'](matrix) == entry['feature_sha256'],
             'Retained feature precision, axes, order or values differ')
        check(); checkpoint(descriptor['target']+':density-before')
        with adapters['workspace'](matrix, dtype=np.float32) as features:
            result = adapters['run'](features, names, [r['cluster_label'] for r in rows], output,
                native_source=native_source, mode='casp17_recursive', retained_positions=positions,
                expected_feature_sha256=adapters['digest'](features), workers=workers,
                check=check, log=log)
        need(result['models'] == names and len(result['density']) == len(names)
             and 0 <= result['UMAP_fits'] <= 15
             and 0 <= result['HDBSCAN_extractions'] <= 30,
             'Density result identities or scientific budget differ')
        labels, assigned = adapters['labels'](result, rows, output,
            maximum_clusters=len(names))
        result.update(target=descriptor['target'], algorithm='HDBSCAN',
            default_cluster_labels=True, primary_manifest=primary['manifest'],
            terminal_report=descriptor['terminal_report'], labels_csv=str(labels),
            assigned_labels_csv=str(assigned), retained_reference_positions=positions,
            public_coordinates_exported=False, representatives_selected=False)
        save(Path(output)/'density-receipt.json', result)
        checkpoint(descriptor['target']+':density-closed')
        return record(Path(output)/'density-receipt.json')
    finally:
        matrix._mmap.close()


def run_target(descriptor, plan, root, native, *, check, checkpoint, adapters=None):
    if adapters is None:
        from confidence import seed
        from target_projections import run
        adapters = dict(seed=seed, primary=run, density=run_density)
    target = descriptor['target']
    need(isinstance(target, str) and re.fullmatch(r'[A-Za-z0-9_-]+', target), 'Unsafe target identity')
    parent = Path(root).absolute(); output = parent/target
    need(output.parent == parent and not any(p.is_symlink() for p in [output, *output.parents]),
         'Target output is outside the approved root')
    output.mkdir()
    intake = read(descriptor['intake'])
    need(intake['source_manifest_sha256'] == plan['native']['sha256'],
         'Intake belongs to another accepted snapshot')
    members = registry(descriptor, intake, native, check=check, checkpoint=checkpoint)
    model_types = bound(descriptor['model_type_config'])
    bound(plan['confidence']); bound(plan['native'])
    metrics = output/'metrics.sqlite'
    adapters['seed'](descriptor['intake']['path'], plan['confidence']['path'],
        plan['confidence']['sha256'], plan['native']['path'], model_types, metrics)
    checkpoint(target+':metrics-closed')
    scratch = Path(plan['environment']['TMPDIR'])/target
    scratch.mkdir()
    primary = adapters['primary'](descriptor, descriptor['terminal_report'], metrics,
        output/'primary', scratch, members=members, model_type_config=descriptor['model_type_config'],
        cluster_config=descriptor['cluster_config'], check=check, checkpoint=checkpoint,
        log=lambda text: print(target+': '+text, flush=True))
    retained = Path(primary['retained_features']['path']).absolute()
    need(retained.parent == scratch.absolute() and '..' not in retained.parts
         and retained.is_file() and not any(p.is_symlink() for p in [retained, *retained.parents]),
         'Retained scratch ownership differs')
    def identity():
        stat = retained.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
    original_identity = identity(); failure = None
    try:
        density = adapters['density'](primary, members, descriptor, plan['native_density'], output/'density',
            workers=plan['resources']['maximum_cpus'], check=check, checkpoint=checkpoint,
            log=lambda text: print(target+': '+text, flush=True))
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            # No extra multi-GB hash: the read-only input must retain its exact file identity.
            need(retained.is_file() and not retained.is_symlink() and identity() == original_identity,
                 'Retained scratch identity changed; preserve it for review')
            retained.unlink()
        except Exception as cleanup_error:
            if failure is None:
                raise
            failure.add_note('Scratch retirement did not complete: '+str(cleanup_error))
    primary['retained_features'].update(temporary=True, retired_after_density=True)
    receipt = dict(status='primary_and_density_ready_for_selection', target=target,
        models=len(members), terminal_report=descriptor['terminal_report'],
        primary=primary, density=density,
        maximum_primary_projection_fits=6, native_TM_calls=0,
        coordinate_parses=0, scoring_calls=0, molecular_exports=0,
        model_selection_status='pending separately frozen plan using actual cluster counts',
        publication_status='not_deployed', analysis_recovery_contract=False)
    save(output/'target-stage-receipt.json', receipt)
    checkpoint(target+':target-closed')
    return record(output/'target-stage-receipt.json')


def approved(path):
    path = Path(path).absolute(); plan = json.loads(path.read_text())
    approval = json.loads((path.parent/'approval.json').read_text())
    need(approval.get('approved_by') == 'user'
         and approval.get('approved_plan_sha256') == sha256(path), 'Exact execution approval required')
    need(plan['operation'] == 'trimmed-primary-and-density' and plan['settings'] == SCOPE
         and len(plan['targets']) == 2
         and len({r['target'] for r in plan['targets']}) == 2
         and all(isinstance(r['target'], str) and re.fullmatch(r'[A-Za-z0-9_-]+', r['target'])
                 for r in plan['targets']),
         'Approved stage/targets/budgets differ')
    need(os.getpgrp() == os.getpid(), 'Driver requires the guard-owned private process group')
    from run_reviewed_plan import verify_controls, control_group
    verify_controls(control_group(), plan['resources'])
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path); args = parser.parse_args()
    plan = approved(args.plan)
    from run_reviewed_plan import reviewed_guard
    guard = reviewed_guard(plan)
    policy = plan['guard']['io_recovery']
    client = guard.DurableCheckpoint(policy['control_directory'], policy['run_id'])
    started = time.monotonic(); last_storage = [float('-inf')]
    def check():
        now = time.monotonic()
        need(now-started < plan['resources']['maximum_wall_seconds'], 'Approved wall cap reached')
        if now-last_storage[0] >= 30:
            need(shutil.disk_usage(plan['run_root']).free >= plan['resources']['preserved_free_space_bytes'],
                 'Storage reserve reached')
            need(shutil.disk_usage(plan['environment']['TMPDIR']).free
                 >= plan['scratch']['minimum_free_bytes_during_run'], 'RAM scratch reserve reached')
            need(sum(p.stat().st_size for p in Path(plan['run_root']).rglob('*') if p.is_file())
                 <= plan['resources']['additional_storage_bytes'], 'Output disk envelope reached')
            last_storage[0] = now
    native = read(plan['native'])
    targets = []
    client.checkpoint('clustering-start')
    for descriptor in plan['targets']:
        check()
        targets.append(run_target(descriptor, plan, plan['run_root'], native,
            check=check, checkpoint=client.checkpoint))
    save(Path(plan['run_root'])/'analysis-completion.json', dict(
        status='two_target_primary_and_density_ready_for_selection',
        execution_plan_sha256=sha256(args.plan), targets=targets,
        maximum_projection_fits=42, maximum_hdbscan_extractions=60,
        native_TM_calls=0, coordinate_parses=0, scoring_calls=0, molecular_exports=0,
        public_site_deployed=False))


if __name__ == '__main__':
    main()
