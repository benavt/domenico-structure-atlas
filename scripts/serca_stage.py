"""One approved SERCA recovery stage per fresh process; no work on import.

The parent owns approval, original verification, cgroups, stage-specific disk or tmpfs
TMPDIR and RAM I/O control. This worker never fabricates estimator checkpoints,
restarts fits, re-exports coordinates, or rebuilds numerical feature caches.
"""
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from terminal_rescreen import read_bound
from target_projections import need, record, save, metadata_preflight

STAGES = ('umap','tsne','labels','density')


def adapters_for_native():
    from saved_feature_cache import mapped_selected_features
    from target_projections import native_adapters
    from rbp_density import run
    from rbp_parity import write_density_labels
    from target_continuation import primary_rows
    return dict(selected=mapped_selected_features,native=native_adapters(),density=run,
                density_labels=write_density_labels,primary_rows=primary_rows)


def assert_ssd_tmpdir(stage,plan,context):
    """Require exact backing/slot before importing any numerical engines."""
    from serca_recovery import validate_workspace_owner,assert_backing,check_ram_capacity
    root=validate_workspace_owner(plan,context['plan_sha256']);path=Path(context['stage_tmpdirs'][stage]).absolute()
    need(path==root/stage and os.environ.get('TMPDIR')==str(path) and path.is_dir()
         and not any(p.is_symlink() for p in [path,*path.parents]),'Exact plan-owned stage TMPDIR required before native imports')
    assert_backing(path,plan);check_ram_capacity(plan)
    need(path.stat().st_dev==root.stat().st_dev,'Stage workspace crosses the owned filesystem')
    need(Path(tempfile.gettempdir()).absolute()==path,'Native tempfile backing differs from approved stage TMPDIR')
    return path


def update_manifest(path,manifest):
    """Only the new run's explicit progress manifest may be replaced."""
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.stage-manifest-',delete=False) as stream:
            temporary=Path(stream.name);json.dump(manifest,stream,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


class ExistingPreparedCache:
    """Compose public native preparation; reuse the validated selected NPZ.

    Native workers ask to persist before applying their terminal-policy result
    key. No writer is called: the exact equivalent selected cache already exists.
    No native private methods or native instances are patched.
    """
    def __init__(self,prepared,target,directory,cache):
        self.prepared,self.target,self.directory,self.cache=prepared,target,directory,cache
        self.reuse_calls=0
    def clustering_data_for(self,directory,paths,requested_positions):
        return self.prepared.clustering_data_for(directory,paths,requested_positions)
    def persist_clustering_cache(self,identifier,writer,*,residue_selection_key,
                                basename_keyword=None,metric_filter_column=None,metric_filter_threshold=None):
        need(identifier==self.target and callable(writer) and residue_selection_key=='shared_msa'
             and basename_keyword=='' and metric_filter_column is None and metric_filter_threshold is None,
             'Unapproved selected-cache persistence request')
        self.reuse_calls+=1
        return Path(self.cache['path'])


def _settings(descriptor,manifest,native,check):
    configuration=read_bound(descriptor['cluster_config'],check)
    settings=native['cluster_settings'](descriptor['cluster_config']['path'])
    need(configuration['settings']['repeat_seeds']==[7,123]
         and configuration['settings']['query_seed']==42
         and configuration['settings']['maximum_subclusters_per_component']==8
         and tuple(settings.repeat_seeds)==(7,123) and settings.query_seed==42
         and settings.maximum_subclusters_per_component==8
         and manifest['defaults']['random_seed']==42
         and manifest['defaults']['tsne_implementation']=='python', 'Reviewed primary engine/seed budget differs')
    read_bound(descriptor['model_type_config'],check)
    return settings


def _verify_method(method,manifest,context,check):
    receipt_entry=manifest.get('method_receipts',{}).get(method) or context['method_receipts'].get(method)
    need(receipt_entry is not None,'Verified completed '+method+' receipt required')
    receipt=read_bound(receipt_entry,check)
    need(receipt['method']==method and receipt['status']=='complete','Method receipt is incomplete')
    for entry in receipt['outputs']:
        need(record(entry['path'],check)==entry,'Completed method artifact changed')
    outputs={r['path']:r for r in receipt['outputs']}
    sidecar=manifest['clustering'][method]['embeddings']
    need(sidecar in outputs and record(sidecar,check)==outputs[sidecar], 'Projection sidecar is not bound by its completed receipt')
    return receipt_entry


def _project(stage,plan,context,manifest,matrix,proof,funcs,check,log):
    import numpy as np
    desc=context['descriptor'];native=funcs['native'];target=desc['target']
    members=context['members'];names=[r['model'] for r in members];paths=[r['source_path'] for r in members]
    root=Path(manifest['output_dir']);directory=Path(desc['source_metadata']['path']).absolute().parent
    html=root/'clustering'/(stage+'.html');sidecar=root/'clustering'/(stage+'_embeddings.npz')
    receipt_path=root/'checkpoints'/(stage+'.json');base=root/'checkpoints'/(stage+'-base.html')
    need(not any(p.exists() for p in (html,sidecar,receipt_path,base)), 'Fresh stage artifacts required; no automatic retry')
    metadata=proof['metadata'];signature=proof['source_signature'];requested=proof['requested_reference_positions']
    expected=read_bound(desc['distogram_manifest'],check)
    metadata_preflight(metadata,signature,directory,requested,native['metadata'],check,expected)
    distance_paths=[Path(r[0]).absolute() for r in signature['files']]
    report=read_bound(desc['terminal_report'],check)
    prepared=native['prepared'](directory=directory,paths=distance_paths,stack=matrix,flattened=matrix,
        labels=native['labels'](metadata,proof['retained_reference_positions']),warnings=[],
        reference_positions=proof['retained_reference_positions'],requested_reference_positions=requested,
        terminal_detection=report,source_signature=signature)
    reuse=ExistingPreparedCache(prepared,target,directory,desc['selected_cache'])
    captured=[];errors=[]
    worker=native['worker'](pdb_id='',identifier=target,pdb_dir=manifest['source_dir'],distogram_dir=str(directory),
        output_file=str(html),clustering_method=stage,tsne_implementation='python',force_recompute=True,
        pca_n_components=2,umap_n_neighbors=15,umap_min_dist=.1,umap_n_components=2,perplexity=30.,
        enable_pymol=False,model_type_config_path=desc['model_type_config']['path'],metrics_db_path=context['metrics_db'],
        plot_source_paths=dict(zip(names,paths)),prepared_distograms=reuse,
        embedding_callback=lambda xy,n,p:captured.append((xy,list(n),list(p))),error_callback=errors.append,status_callback=log)
    worker.run();check()
    for key in ('source_metadata','distogram_manifest'):
        need(record(desc[key]['path'],check)['sha256']==desc[key]['sha256'],'Original metadata was changed by native worker')
    need(not errors and len(captured)==1 and reuse.reuse_calls==1,'Native stage failed or did not reuse the selected cache: '+str(errors))
    xy,returned,returned_paths=captured[0]
    need(returned==names and [Path(p).stem for p in returned_paths]==names
         and np.asarray(xy).shape==(len(names),2) and np.isfinite(xy).all(),'Stage projection identities/coordinates differ')
    np.savez_compressed(sidecar,embeddings=xy,filenames=np.asarray(names,dtype=str),filepaths=np.asarray(paths,dtype=str))
    schemes=native['schemes'](context['metrics_db'],desc['model_type_config']['path'])
    images=native['plots'](xy,names,paths,stage,schemes,context['metrics_db'],manifest['source_dir'],root/'clustering'/stage,
                          check_cancel=check,log=log)
    shutil.copy2(html,base)
    params=dict(n_neighbors=15,min_dist=.1,n_components=2) if stage=='umap' else dict(perplexity=30.)
    manifest['clustering'][stage]=dict(html=str(html),embeddings=str(sidecar),params=params,images=images)
    receipt=dict(method=stage,status='complete',parameters=params,base_html=str(base),native_output_html=str(html),
        outputs=[record(p,check) for p in (base,sidecar,*[Path(r['image']) for r in images])],
        source_feature_proof=desc['feature_proof'],selected_cache=desc['selected_cache'],
        feature_sha256=proof['feature_sha256'],models_sha256=proof['model_order_sha256'],
        UMAP_fits=1 if stage=='umap' else 0,tsne_fits=1 if stage=='tsne' else 0,
        internal_tsne_PCA_fits=1 if stage=='tsne' else 0,primary_PCA_fits=0,
        cache_reused_without_rewrite=True,native_library=plan['sources']['library'])
    save(receipt_path,receipt)
    manifest.setdefault('method_receipts',{})[stage]=record(receipt_path,check)
    return record(receipt_path,check)


def _labels(context,manifest,matrix,proof,funcs,settings,check,log):
    import numpy as np
    members=context['members'];names=[r['model'] for r in members];paths=[r['source_path'] for r in members]
    root=Path(manifest['output_dir']);receipt_path=root/'checkpoints/labels.json'
    need(not receipt_path.exists() and 'knn_clustering' not in manifest,'Fresh primary-label stage required')
    projections={}
    for method in ('pca','umap','tsne'):
        _verify_method(method,manifest,context,check)
        html=Path(manifest['clustering'][method]['html']).absolute()
        need(html.is_relative_to(root),'Label annotation must not touch original/cross-run HTML')
        with np.load(manifest['clustering'][method]['embeddings'],allow_pickle=False) as saved:
            need(saved['filenames'].tolist()==names and saved['filepaths'].tolist()==paths,'Saved projection registry differs')
            xy=saved['embeddings'].copy()
        need(xy.shape==(len(names),2) and np.isfinite(xy).all(),'Saved projection coordinates are invalid')
        projections[method]=(xy,names,paths)
    manifest['knn_clustering']=funcs['native']['knn'](manifest,projections,features=matrix,feature_names=names,
        settings=settings,projection='umap',n_clusters=None,check_cancel=check,log=log)
    need(manifest['knn_clustering']['feature_input']['sha256']==proof['feature_sha256'],'Primary label input feature hash differs')
    receipt=dict(stage='labels',status='complete',repeat_UMAP_fits_max=2,
        selected_cache=context['descriptor']['selected_cache'],source_feature_proof=context['descriptor']['feature_proof'],
        outputs=[record(manifest['knn_clustering'][k],check) for k in ('labels_csv','report_json')],
        labelled_views=[record(row['html'],check) for row in manifest['clustering'].values()])
    save(receipt_path,receipt);manifest.update(status='running',analysis_status='primary_complete',
        stage_scope='primary projections and labels only',deliverable_status='pending_density_model_selection_and_exports')
    manifest['primary_labels_receipt']=record(receipt_path,check)
    return record(receipt_path,check)


def _density(plan,context,manifest,matrix,proof,funcs,check,log):
    desc=context['descriptor'];members=context['members'];names=[r['model'] for r in members]
    need(manifest['analysis_status']=='primary_complete','Verified primary labels required before density')
    entry=manifest['primary_labels_receipt'];receipt=read_bound(entry,check)
    for row in receipt['outputs']:need(record(row['path'],check)==row,'Primary label artifact changed')
    source_snapshot=Path(manifest['output_dir'])/'primary-density-input.json'
    save(source_snapshot,manifest)
    rows=funcs['primary_rows'](manifest,members);output=Path(manifest['output_dir']).parent/'density'
    need(not output.exists(),'Fresh density stage required; no automatic retry')
    result=funcs['density'](matrix,names,[r['cluster_label'] for r in rows],output,
        native_source=context['native_density'],mode='casp17_recursive',retained_positions=proof['retained_reference_positions'],
        allow_stored_float16=True,workers=plan['resources']['maximum_cpus'],check=check,log=log)
    need(result['models']==names and len(result['density'])==len(names)
         and 0<=result['UMAP_fits']<=15 and 0<=result['HDBSCAN_extractions']<=30
         and result['feature_proof']['stored_dtype']=='float16' and result['feature_proof']['fit_dtype']=='float32',
         'Density identities, precision or fit budget differs')
    labels,assigned=funcs['density_labels'](result,rows,output,maximum_clusters=len(names))
    result.update(target=desc['target'],algorithm='HDBSCAN',default_cluster_labels=True,
        primary_manifest=record(source_snapshot,check),selected_cache=desc['selected_cache'],
        source_feature_proof=desc['feature_proof'],terminal_report=desc['terminal_report'],
        labels_csv=str(labels),assigned_labels_csv=str(assigned),retained_reference_positions=proof['retained_reference_positions'],
        representatives_selected=False,public_coordinates_exported=False)
    path=output/'density-receipt.json';save(path,result);manifest['density_clustering']=record(path,check)
    return record(path,check)


def run_stage(stage,plan,context,*,check,log,adapters=None):
    """Called once by the parent after exact approval and child control checks."""
    need(stage in STAGES and callable(check) and callable(log),'Explicit stage/check/log required')
    tmpdir=assert_ssd_tmpdir(stage,plan,context)  # Must precede native imports.
    import numpy as np
    funcs=adapters_for_native() if adapters is None else dict(adapters)
    desc=context['descriptor'];path=Path(context['newprimary_manifest']).absolute();root=Path(plan['run_root']).absolute()
    need(path.is_relative_to(root) and path.is_file() and not any(p.is_symlink() for p in [path,*path.parents]),
         'Only the new approved progress manifest may be updated')
    before=record(path,check);manifest=json.loads(path.read_text())
    need(Path(manifest['output_dir']).absolute()==path.parent and manifest['identifier']==desc['target']
         and manifest['metrics_db']==context['metrics_db'],'New primary manifest/context differs')
    settings=_settings(desc,manifest,funcs['native'],check)
    need(context['native_density']==plan['native_density'],'Native density source binding differs')
    names=[r['model'] for r in context['members']]
    metrics=record(context['metrics_db'],check)
    _,_,_,confidence,_=funcs['native']['confidence'](manifest,[r['source_path'] for r in context['members']])
    need(confidence==manifest['ensemble_confidence_preflight'] and record(context['metrics_db'],check)==metrics,
         'Copied metrics confidence scope/policy differs')
    with funcs['selected'](desc['selected_cache'],axes=desc['axes'],baseline=desc['baseline'],
        terminal_report=desc['terminal_report'],source_metadata=desc['source_metadata'],feature_proof=desc['feature_proof'],check=check) as (matrix,proof):
        need(proof['shape']==[len(names),len(proof['retained_reference_positions'])*(len(proof['retained_reference_positions'])-1)//2]
             and proof['dtype']=='float16' and proof['readonly'] is True
             and matrix.dtype==np.float16 and matrix.flags.c_contiguous and not matrix.flags.writeable
             and proof['model_order_sha256']==hashlib.sha256('\n'.join(names).encode()).hexdigest()
             and proof['pair_mask_sha256']==manifest['terminal_detection']['pair_mask_sha256']
             and proof['retained_reference_positions']==manifest['terminal_detection']['retained_reference_positions']
             and manifest['derivation']['accepted_terminal_report']==desc['terminal_report'], 'Selected cache proof differs')
        if stage in ('umap','tsne'):receipt=_project(stage,plan,context,manifest,matrix,proof,funcs,check,log)
        elif stage=='labels':receipt=_labels(context,manifest,matrix,proof,funcs,settings,check,log)
        else:receipt=_density(plan,context,manifest,matrix,proof,funcs,check,log)
        slim={k:v for k,v in proof.items() if k not in ('metadata','source_signature')}
    need(record(context['metrics_db'],check)==metrics,'Fresh copied metrics changed during stage')
    need(record(path,check)==before,'Progress manifest changed concurrently')
    manifest.setdefault('completed_recovery_stages',{})[stage]=dict(receipt=receipt,selected_cache_proof=slim,
        workspace_backing=('plan-owned tmpfs (charged to job memory cgroup)'
            if plan.get('ssd_scratch',{}).get('backing','ssd')=='tmpfs' else 'plan-owned SSD'),TMPDIR=str(tmpdir),fresh_process_required=True)
    result_manifest=path
    if stage=='density':
        result_manifest=path.parent/'manifest.json'
        need(not result_manifest.exists(),'Fresh final primary manifest required')
        manifest.update(status='complete',analysis_status='complete',stage_scope='primary projections and density labels only',
                        deliverable_status='pending_model_selection_and_exports')
        save(result_manifest,manifest)
    update_manifest(path,manifest)
    return dict(stage=stage,status='complete',receipt=receipt,manifest=record(result_manifest,check),
        selected_cache_proof=slim,coordinate_parses=0,scoring_calls=0,molecular_exports=0)
