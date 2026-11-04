"""Finish only approved remaining RBP state/panel/region work from saved science.

All display projections, recursive density results, representative selections,
Bonsai trees and medoid grids are hash-verified saved inputs, never recomputed.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from snapshot import sha256
from rbp_parity import (TARGET, validate_approval, read_bound, record, registry,
                        frozen_features, save, region_artifacts, need)



def verify_saved_stages(plan, saved, density, audit, names):
    """Require exact independent acceptance of every stage reused by recovery."""
    need(density['status'] == 'complete' and density['models'] == names
         and density['source_workflow_manifest_sha256'] == plan['manifest']['sha256']
         and density['UMAP_fits'] == 15 and density['HDBSCAN_extractions'] == 30,
         'Completed original density analysis differs')
    need(audit.get('status') == 'partial_completed_stages_independently_verified'
         and audit['plan']['sha256'] == plan['parent_scientific_plan']['sha256']
         and audit['completed']['density']['publication_analysis']['sha256'] == saved['density_analysis']['sha256']
         and saved['density_analysis'] == saved['density_artifacts']['analysis']
         and audit['completed']['representatives']['report']['sha256'] == saved['density_artifacts']['ensemble_report']['sha256']
         and audit['completed']['density']['models'] == len(names)
         and audit['completed']['density']['clusters'] == density['structural_clusters']
         and audit['completed']['density']['unassigned_models'] == density['unassigned_models'],
         'Completed-stage acceptance does not bind the reused density artifacts')
    for role, artifacts in [('primary', saved['primary_artifacts']), ('density', saved['density_artifacts'])]:
        for name, evidence in [('bonsai', audit['completed']['bonsai'][role]['file']),
                               ('medoid_tm', audit['completed']['grids'][role]['file'])]:
            need(evidence['sha256'] == artifacts[name]['sha256'] and record(artifacts[name]['path']) == artifacts[name],
                 'Reused Bonsai/TM artifact differs from independent acceptance')
    need(audit['completed']['representatives']['inventory_proof_status'] == 'passed'
         and audit['completed']['representatives']['full271_CA_mask'] is True,
         'Completed representatives lack full native atom/mask acceptance')
    return True


def recovery_checkpoint(plan):
    """Safe-boundary cooperation; RAM progress never authorizes scientific reuse."""
    policy = plan.get('guard', {}).get('io_recovery')
    if policy is None:
        return lambda _: None
    from run_reviewed_plan import reviewed_guard
    guard = reviewed_guard(plan)
    client = guard.DurableCheckpoint(policy['control_directory'], policy['run_id'])
    return client.checkpoint


def main(plan_path):
    plan = validate_approval(plan_path)
    checkpoint = recovery_checkpoint(plan)
    checkpoint('recovery-start')
    root = Path(plan['run_root']); output = root/'rbp'
    need(root.is_dir() and not output.exists(), 'Fresh guarded recovery output required')
    output.mkdir(); start = time.monotonic(); last_audit = [0.]
    def check():
        need(time.monotonic()-start < plan['resources']['maximum_wall_seconds'], 'Approved recovery wall cap reached')
        if time.monotonic()-last_audit[0] >= 30:
            need(shutil.disk_usage(root).free >= plan['resources']['preserved_free_space_bytes'], 'SSD reserve breached')
            need(shutil.disk_usage(plan['environment']['TMPDIR']).free >= plan['scratch']['minimum_free_bytes_during_run'], 'RAM scratch reserve breached')
            need(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= plan['resources']['additional_storage_bytes'], 'Recovery output disk envelope reached')
            last_audit[0] = time.monotonic()
    study, manifest = read_bound(plan['study']), read_bound(plan['manifest'])
    manifest['_source_path'] = plan['manifest']['path']
    names, members, primary = registry(plan, manifest, study, check, checkpoint=checkpoint)
    saved = plan['saved_stages']; density = read_bound(saved['density_analysis'])
    audit = read_bound(saved['acceptance'])
    verify_saved_stages(plan, saved, density, audit, names)
    checkpoint('saved-stage-acceptance-verified')
    # Annotated variance reuses the original full-cohort saved variance/cores;
    # it is completed before the heavier remaining native-state work.
    original_report = json.loads(Path(manifest['cluster_ensembles']['report']).read_text())
    regions = region_artifacts(manifest, original_report, output/'regions', check)
    checkpoint('saved-variance-presentation-written')
    bindings = dict(plan_sha256=sha256(plan_path), consumer_commit=plan['sources']['consumer']['commit'],
        library_commit=plan['sources']['library']['commit'], manifest=plan['manifest'],
        native=plan['native'], feature_cache=plan['feature_cache'],
        reused_science_plan=plan['parent_scientific_plan'], saved_stage_acceptance=saved['acceptance'],
        runtime_guard=plan.get('runtime_guard'))
    from rbp_states import run_states
    from rbp_native_tm import NativeTMRun, frozen_source_evidence
    cohort = dict(target=TARGET, range_id='all', models=names, methods=[r['method'] for r in members],
        energies=[r['energy'] for r in members], sequence_sha256=study['targets'][TARGET]['sequence_sha256'],
        construct_residues=271, conditioning_t=.1, energy_scope='full_construct')
    clustering = dict(primary_labels=[r['cluster_label'] for r in primary], hdbscan_labels=density['density'],
        hdbscan_noise_labels=sorted({label for label in density['density'] if label.endswith('.unassigned')}),
        clustering_positions=list(range(1,272)),
        saved_source_views={method:record(manifest['clustering'][method]['embeddings']) for method in ('pca','umap','tsne')})
    native_bindings = dict(bindings, sources=frozen_source_evidence())
    with frozen_features(plan['feature_cache'], manifest, names, plan['environment']['TMPDIR']) as (features, feature_sha):
        need(feature_sha == density['feature_proof']['feature_sha256'], 'State proposal features differ from saved density input')
        checkpoint('state-features-verified')
        with NativeTMRun(members, list(range(271)), cohort['sequence_sha256'], output/'native-tm',
            runtime=plan['native_tm_runtime'], source_bindings=native_bindings, check=check,
            reuse_inputs=plan['reuse_native_inputs'], checkpoint=checkpoint) as scorer:
            scorer.prepare()
            states = run_states(cohort, clustering, features, list(range(271)), output/'states', scorer,
                source_bindings=native_bindings, seed=4200, workers=4, check=check, checkpoint=checkpoint)
    checkpoint('state-work-complete-features-released')
    from rbp_panel_exports import export_catalog
    export_catalog(output/'states/catalog.json', members, output/'state-panels',
        expected_catalog_sha256=sha256(output/'states/catalog.json'), check=check,
        log=lambda text:print(text,flush=True), checkpoint=checkpoint)
    artifacts = dict(schema_version=1, target=TARGET,
        density_artifacts=saved['density_artifacts'], primary_artifacts=saved['primary_artifacts'],
        region_artifacts=regions, state_catalog=record(output/'states/catalog.json'),
        state_panels=record(output/'state-panels/catalog.json'))
    save(output/'artifacts.json', artifacts); check()
    save(root/'rbp-parity-completion.json', dict(status='rbp_parity_science_and_saved_exports_complete',
        plan_sha256=sha256(plan_path), model_count=len(names), source_bindings=bindings,
        density_clusters=density['structural_clusters'], density_unassigned=density['unassigned_models'],
        UMAP_fits=0, HDBSCAN_extractions=0, Bonsai_PCA_fits=0,
        reused_UMAP_fits=15, reused_HDBSCAN_extractions=30, reused_Bonsai_PCA_fits=2,
        reused_medoid_TM_directional_calls=11568, states=states['summary'],
        artifacts=record(output/'artifacts.json'), camera_dependent_images='pending explicit camera review',
        public_deployment=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('plan',type=Path)
    main(parser.parse_args().plan)
