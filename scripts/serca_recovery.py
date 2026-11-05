"""Exact-plan SERCA recovery; heavy stages have separate bounded processes."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

from snapshot import sha256
from target_continuation import bound, read, record
from target_projections import save

STAGES = ('umap', 'tsne', 'labels', 'density')
SCOPE = dict(target='SERCA_P04191', models=5072, retained_residues=894,
             maximum_new_projection_fits=20, maximum_hdbscan_extractions=30,
             repeated_PCA_fits=0, native_TM_calls=0, coordinate_parses=0,
             molecular_exports=0, scoring_calls=0, density_mode='casp17_recursive')


def need(value, message):
    if not value:raise ValueError(message)


def approved(path):
    path = Path(path).absolute(); plan = json.loads(path.read_text())
    proof = json.loads((path.parent/'approval.json').read_text())
    need(proof.get('approved_by') == 'user' and proof.get('approved_plan_sha256') == sha256(path),
         'Exact recovery-plan approval required')
    need(plan['operation'] == 'serca-selected-cache-recovery' and plan['settings'] == SCOPE
         and plan['target']['target'] == SCOPE['target'] and plan['target']['model_count'] == 5072,
         'Approved SERCA recovery scope differs')
    from run_reviewed_plan import control_group, verify_controls
    group = control_group(); verify_controls(group,plan['resources'])
    need((group/'memory.high').read_text().strip() == str(plan['resources']['memory_high_bytes'])
         and (group/'memory.oom.group').read_text().strip() == '1', 'Required high/OOM-group controls differ')
    return plan


def ssd_root(plan):
    root = Path(plan['ssd_scratch']['root']).absolute()
    need(root.parent == Path(plan['run_root']).absolute() and root.name == 'stage-tmp'
         and not any(p.is_symlink() for p in [root,*root.parents]), 'SSD workspace scope differs')
    return root


def create_ssd(path, plan):
    root = ssd_root(plan);root.mkdir(mode=0o700)
    info = root.stat()
    need(info.st_dev == root.parent.stat().st_dev and not root.is_relative_to('/dev/shm'),
         'Workspace must use the approved output SSD filesystem')
    save(root/'.owner.json',dict(plan_sha256=sha256(path),uid=os.getuid(),device=info.st_dev,inode=info.st_ino))
    for stage in STAGES:(root/stage).mkdir(mode=0o700)


def check_owned_tree(root):
    info = root.stat()
    for item in [root,*root.rglob('*')]:
        value = item.lstat()
        need(value.st_uid == os.getuid() and value.st_dev == info.st_dev
             and (stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode) and value.st_nlink == 1),
             'Workspace contains a foreign owner, link, special file or mount')


def validate_ssd_owner(path, plan):
    root = ssd_root(plan)
    need(root.is_dir() and not root.is_symlink(), 'SSD workspace root changed')
    marker = root/'.owner.json';need(marker.is_file() and not marker.is_symlink(), 'Missing workspace owner')
    info=root.stat()
    need(json.loads(marker.read_text()) == dict(plan_sha256=sha256(path),uid=os.getuid(),device=info.st_dev,inode=info.st_ino),
         'SSD workspace ownership identity changed')
    return root


def stage_processes():
    from run_reviewed_plan import control_group
    group=control_group()
    return {int(pid) for entry in [group/'cgroup.procs',*group.rglob('cgroup.procs')]
            for pid in entry.read_text().split()}


def retire_stage(path, plan, stage, expected_processes):
    need(stage in STAGES, 'Unknown completed stage')
    need(stage_processes()==expected_processes,'Stage descendants remain; preserve its workspace')
    root=validate_ssd_owner(path,plan);slot=root/stage
    need(slot.is_dir() and not slot.is_symlink(), 'Stage workspace changed')
    check_owned_tree(slot);shutil.rmtree(slot)


def cleanup(path, plan):
    from analysis_scratch import assert_quiescent, cleanup as cleanup_ram
    root=ssd_root(plan)
    if root.exists() or root.is_symlink():
        assert_quiescent();validate_ssd_owner(path,plan);check_owned_tree(root);shutil.rmtree(root)
        save(Path(plan['run_root'])/'ssd-cleanup.json',dict(status='removed_owned_disposable_ssd_workspace',
            plan_sha256=sha256(path),path=str(root),service_result=os.environ.get('SERVICE_RESULT')))
    cleanup_ram(path,plan)


def copy_bound(entry, destination):
    source=bound(entry);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as inp,destination.open('xb') as out:
        shutil.copyfileobj(inp,out,16*1024**2);out.flush();os.fsync(out.fileno())
    need(sha256(destination) == entry['sha256'] and sha256(source) == entry['sha256'], 'Copied saved artifact differs')
    return record(destination)


def verify_saved_pca(pca, plan):
    need(pca['feature_proof']==plan['target']['feature_proof']
         and pca['native_library_required']==plan['sources']['library']['commit']
         and pca['settings']==dict(n_components=2), 'Saved PCA feature, engine or settings binding differs')


def validate_context(path, plan, context):
    root=Path(plan['run_root']).absolute()
    need(context['plan_sha256']==sha256(path) and context['descriptor']==plan['target']
         and context['native_density']==plan['native_density']
         and context['newprimary_manifest']==str(root/'primary/manifest.progress.json')
         and context['metrics_db']==str(root/'metrics.sqlite')
         and context['stage_tmpdirs']=={s:str(ssd_root(plan)/s) for s in STAGES},
         'Child context differs from the exact approved recovery scope')


def prepare_context(path, plan, *, check, checkpoint):
    import numpy as np
    target=plan['target'];root=Path(plan['run_root']);primary=root/'primary'
    primary.mkdir();(primary/'clustering').mkdir();(primary/'checkpoints').mkdir()
    axes=read(target['axes']);intake=read(target['intake']);names=axes['names']
    need(intake['target'] == 'SERCA_P04191' and len(names) == len(set(names)) == 5072,
         'SERCA cohort differs')
    by_name={r['model_id']:r for r in intake['models']}
    need(len(by_name) == len(intake['models']) == len(names) and set(by_name) == set(names), 'Intake order/identities differ')
    members=[]
    for number,name in enumerate(names,1):
        check();row=by_name[name];source=Path(row['path']).absolute()
        need(source.is_file() and source.is_relative_to(Path(intake['analysis_dir']).absolute())
             and not any(p.is_symlink() for p in [source,*source.parents]) and sha256(source) == row['sha256'],
             'Original SERCA model changed: '+name)
        members.append(dict(model=name,source_path=str(source),sha256=row['sha256']))
        if number%250==0:checkpoint('serca-original-hashes:'+str(number))
    checkpoint('serca-original-hashes-closed')
    metrics=Path(root/'metrics.sqlite')
    source_metrics=bound(plan['reuse']['metrics'])
    need(not any(source_metrics.with_name(source_metrics.name+suffix).exists() for suffix in ('-wal','-shm','-journal')),
         'Saved metrics database is not a closed snapshot')
    copy_bound(plan['reuse']['metrics'],metrics)
    pca=read(plan['reuse']['pca_checkpoint']);report=read(target['terminal_report'])
    feature=read(target['feature_proof']);verify_saved_pca(pca,plan)
    need(pca['status']=='complete' and pca['method']=='pca' and pca['models_sha256']==axes['names_sha256']
         and feature['feature_sha256']==plan['reuse']['feature_sha256']
         and feature['retained_reference_positions']==report['retained_reference_positions']==list(range(57,951)),
         'Saved PCA/accepted feature mask differs')
    base=Path(pca['base_html']).absolute();sidecar=None;images=[];outputs=[]
    for entry in pca['outputs']:
        source=Path(entry['path']).absolute()
        if source==base:destination=primary/'checkpoints/pca-base.html'
        elif source.name=='pca_embeddings.npz':destination=primary/'clustering/pca_embeddings.npz';sidecar=destination
        elif source.suffix.lower()=='.png':
            destination=primary/'clustering/pca'/source.name
            images.append(dict(image=str(destination),title=source.stem.replace('_',' '),method='pca',kind='clustering',reused=True))
        else:raise ValueError('Unexpected saved PCA artifact')
        outputs.append(copy_bound(entry,destination))
    need(sidecar is not None and images, 'Saved PCA outputs incomplete')
    html=copy_bound(record(primary/'checkpoints/pca-base.html'),primary/'clustering/pca.html')
    with np.load(sidecar,allow_pickle=False) as saved:
        need(saved['embeddings'].shape==(5072,2) and np.isfinite(saved['embeddings']).all()
             and saved['filenames'].tolist()==names
             and saved['filepaths'].tolist()==[r['source_path'] for r in members], 'Saved PCA population/order differs')
        ratios=saved['explained_variance_ratio'].tolist()
    reused=deepcopy(pca)
    reused.update(outputs=outputs,base_html=str(primary/'checkpoints/pca-base.html'),native_output_html=html['path'],
        source_checkpoint=plan['reuse']['pca_checkpoint'],PCA_reused=True,primary_PCA_fits=0,prior_PCA_fits_charged=1)
    save(primary/'checkpoints/pca.json',reused)
    prior=read(target['prior_workflow_manifest'])
    parents={str(Path(r['source_path']).parent) for r in members};need(len(parents)==1,'Mixed source directories')
    manifest=dict(schema_version=1,kind='serca-selected-cache-recovery',identifier='SERCA_P04191',pdb_id='',
        structure_count=5072,source_dir=next(iter(parents)),output_dir=str(primary),
        distogram_dir=str(Path(target['source_metadata']['path']).parent),metrics_db=str(metrics),
        clustering=dict(pca=dict(html=html['path'],embeddings=str(sidecar),params=dict(n_components=2),
            explained_variance_ratio=ratios,images=images)),status='running',analysis_status='running',
        deliverable_status='pending_clustering_model_selection_and_exports',knn_clusters_enabled=True,cluster_ensembles_enabled=True,
        defaults=dict(atom_type='CA',residues='shared',dimensions=2,random_seed=42,tsne_implementation='python'),
        terminal_detection=report,model_type_config=target['model_type_config']['path'],
        plot_model_type_config=target['model_type_config']['path'],model_type_config_source='explicit',
        derivation=dict(accepted_terminal_report=target['terminal_report'],retained_feature_proof=feature,
            selected_cache=target['selected_cache'],PCA_reused_from=plan['reuse']['pca_checkpoint'],
            source_plan=plan['reuse']['plan'],recovery_plan_sha256=sha256(path),
            native_library_commit=plan['sources']['library']['commit'],
            new_standalone_PCA_fits=0,tsne_initialization_PCA_fits_max=1,coordinate_parses=0,scoring_calls=0),
        method_receipts=dict(pca=record(primary/'checkpoints/pca.json')))
    for key in ('input_inventory','dataset_sha256','input_file_count','dataset_scope','presentation_target',
                'ensemble_selection_scope','ensemble_confidence_preflight','requested_exports'):
        if key in prior:manifest[key]=deepcopy(prior[key])
    save(primary/'terminal_detection.json',report)
    save(primary/'manifest.progress.json',manifest)
    context=dict(descriptor=target,newprimary_manifest=str(primary/'manifest.progress.json'),metrics_db=str(metrics),
        members=members,native_density=plan['native_density'],method_receipts=manifest['method_receipts'],
        stage_tmpdirs={s:str(ssd_root(plan)/s) for s in STAGES},parent_pid=os.getpid(),guard_pgid=os.getpgrp(),
        plan_sha256=sha256(path))
    save(root/'recovery-context.json',context);checkpoint('serca-inherited-PCA-closed')
    return context


def sample_child(process, plan, stage):
    from run_reviewed_plan import control_group
    group=control_group();info={}
    for name in ('memory.current','memory.peak','memory.events','memory.stat','memory.pressure','io.pressure'):
        try:info[name]=(group/name).read_text().strip()
        except FileNotFoundError:info[name]=None
    proc={}
    for name in ('status','io'):
        try:
            text=(Path('/proc')/str(process.pid)/name).read_text()
            proc[name]=text if name=='io' else '\n'.join(s for s in text.splitlines() if s.startswith(('VmRSS:','VmHWM:','RssAnon:','RssFile:','RssShmem:','Threads:')))
        except FileNotFoundError:proc[name]=None
    with (Path(plan['run_root'])/'stage-memory.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(monotonic=time.monotonic(),stage=stage,child_pid=process.pid,
            cgroup=info,process=proc,scientific_progress_boundary=False))+'\n')


def run_child(path, plan, context, stage, *, check):
    slot=Path(context['stage_tmpdirs'][stage]);env=dict(os.environ,TMPDIR=str(slot))
    baseline=stage_processes();need(os.getpid() in baseline,'Stage controller is outside its unit')
    process=subprocess.Popen([sys.executable,str(Path(__file__).absolute()),str(path),'--stage',stage],env=env)
    try:
        while process.poll() is None:
            check();sample_child(process,plan,stage)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        need(process.returncode==0,'SERCA '+stage+' failed; preserve closed stages before any retry')
        check()
        need(stage_processes()==baseline,'Stage descendants remain; preserve its workspace')
        return baseline
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path)
    parser.add_argument('--stage',choices=STAGES);parser.add_argument('--cleanup',action='store_true');args=parser.parse_args()
    path=args.plan.absolute();plan=approved(path)
    if args.cleanup:
        need(args.stage is None,'Cleanup cannot run a scientific stage');cleanup(path,plan);return
    root=Path(plan['run_root']);started=time.monotonic();last_size=[float('-inf')]
    def check():
        now=time.monotonic();need(now-started<plan['resources']['maximum_wall_seconds'],'Approved wall cap reached')
        need(shutil.disk_usage(root).free>=plan['resources']['preserved_free_space_bytes'],'SSD reserve reached')
        if now-last_size[0]>=30:
            need(sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<=plan['resources']['additional_storage_bytes'],
                 'Approved SSD output/workspace envelope reached')
            last_size[0]=now
    if args.stage:
        context=json.loads((root/'recovery-context.json').read_text())
        validate_context(path,plan,context)
        need(os.getppid()==context['parent_pid']
             and os.getpgrp()==context['guard_pgid'], 'Stage is not owned by the approved guard worker')
        from serca_stage import run_stage
        run_stage(args.stage,plan,context,check=check,log=lambda text:print('SERCA '+args.stage+': '+text,flush=True))
        return
    need(os.getpgrp()==os.getpid(),'Recovery controller requires the guard-owned process group')
    from run_reviewed_plan import reviewed_guard
    guard=reviewed_guard(plan);policy=plan['guard']['io_recovery']
    client=guard.DurableCheckpoint(policy['control_directory'],policy['run_id'])
    client.checkpoint('serca-recovery-start');create_ssd(path,plan)
    context=prepare_context(path,plan,check=check,checkpoint=client.checkpoint)
    for stage in STAGES:
        check();client.checkpoint('serca-before-'+stage)
        baseline=run_child(path,plan,context,stage,check=check)
        retire_stage(path,plan,stage,baseline)
        client.checkpoint('serca-closed-'+stage)
    result=dict(status='serca_primary_and_density_recovered',plan_sha256=sha256(path),
        primary_manifest=record(root/'primary/manifest.json'),density=record(root/'density/density-receipt.json'),
        PCA_reused=True,new_projection_fits_max=20,HDBSCAN_extractions_max=30,
        previous_attempts_charged=23,maximum_cumulative_attempts=43,
        native_TM_calls=0,coordinate_parses=0,scoring_calls=0,molecular_exports=0,public_site_changed=False,
        model_selection_status='pending separate plan using actual cluster counts')
    save(root/'serca-recovery-completion.json',result)


if __name__=='__main__':main()
