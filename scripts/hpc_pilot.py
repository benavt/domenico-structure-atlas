"""Portable CPU pilot stages over bound selected distances; stdlib-only import.

The approved Slurm outer runner owns admission, cgroups, fresh processes and
DATA_PROCESSED/TMPDIR before calling this module. These are diagnostics, not a
complete recursive clustering, model-selection or molecular-report workflow.
No fit, source import, input read or environment mutation occurs on import.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re

LIBRARY_COMMIT = 'fe94ba765fd7a9f9894a5e5ba11a57f582a39365'
DENSITY_COMMIT = 'db52d611ff77546ce6839b3b8f65f5dc9c323074'
DENSITY_SHA256 = '2b7fc129474d92d7f4ca5b347c313b25964ff8059fd6d56a4f49efc0afd5e049'
SERCA_PAIR_MASK = 'bafc72c375d8bfebf7297a4e514f5c8ce0278397e83caccfda414a1b3e84c8bd'
STAGES = ('pca', 'umap', 'tsne', 'labels', 'density0')
HEX = re.compile('[0-9a-f]{64}\\Z')
SAFE = re.compile('[A-Za-z0-9_.-]+\\Z')
LIBRARY_SOURCE_HASHES = {
    'scripts.pca': 'f5fad227648b53e4cdc1eb69ba3bb3d85ecc3229ce69b8fcf0a0048c1f6995d4',
    'scripts.umap_clustering': '09851456cac8c83eefe9759dcb63d3c8bd8d774b0a65ea5bd21af971217c2597',
    'scripts.tsne': 'cb743eea32980f36a9e529e3e290f3a1a5a23a8f17b60c82dda8989802631679',
    'src.analysis.feature_workspace': 'bc515eefa91508f79067d8658fee3981fa7215893f14ee4652d325ff4423bfc3',
    'src.analysis.pca_workspace': '34b61c4836c7b1c24663fd438d8b43b9b432503785e16f96d7ca762b7a75b90b',
    'src.analysis.umap_component_clusters': '3aa0dcf0ce287986e983bbaba6484000cb71d62897f9a29d353fb7a42bf27a82',
    'src.analysis.umap_knn_clusters': 'a550e036cc4b9e741b79aceceb84f138e1fa1847763bc60cadf6a2c0ac68e783',
    'src.analysis.clustering_variants': '2f747bb755e5ade01482d03326f0de60ebcc95ebaa23881ca4918a2dca01be7b',
    'src.data_access.accessor': 'df52be6bfc0d3e454eb448929f34534a1ee4312fe84bceee5b54f98c3357dec9',
}
DEFAULT_COMPONENT_SETTINGS = dict(maximum_subclusters_per_component=8,
    minimum_feature_silhouette=.15, minimum_seed_ari=.8,
    silhouette_queries_per_component=512, repeat_seeds=[7,123], query_seed=42)


def need(value, message):
    if not value: raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    need('..' not in path.parts and path.is_file()
         and not any(p.is_symlink() for p in [path,*path.parents]), 'Exact regular file required: '+str(path))
    return path


def identity(path):
    info = regular(path).stat()
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)


def record(path, check=lambda: None):
    path = regular(path);before = identity(path);digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024**2), b''):
            check();digest.update(block)
    need(identity(path)==before, 'File changed during hashing: '+str(path))
    return dict(path=str(path),bytes=before[2],sha256=digest.hexdigest())


def bound(entry, check=lambda: None):
    need(isinstance(entry,dict) and type(entry.get('bytes')) is int and HEX.fullmatch(entry.get('sha256','')),
         'Exact input path/bytes/SHA descriptor required')
    need(record(entry['path'],check)==entry, 'Bound file changed: '+str(entry.get('path')))
    return regular(entry['path'])


def read(entry, check=lambda: None):
    return json.loads(bound(entry,check).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def compact_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_inputs(plan, check):
    need(plan['native']['library_commit']==LIBRARY_COMMIT, 'Pinned Fe94 native library required')
    provider=plan['native']['density_source']
    need(provider.get('commit')==DENSITY_COMMIT and provider.get('sha256')==DENSITY_SHA256, 'Pinned db52 density provider required')
    target = plan['target'];need(isinstance(target,str) and SAFE.fullmatch(target) and target not in ('.','..'),'Unsafe target identifier')
    spec = plan['inputs'];feature_path = bound(spec['features'],check)
    registry = read(spec['members'],check);members = registry['members'];names = [r['model'] for r in members]
    need(type(spec['model_count']) is int and len(names)==spec['model_count']>=2 and len(set(names))==len(names)
         and all(isinstance(n,str) and SAFE.fullmatch(n) and n not in ('.','..') for n in names), 'Unique complete model registry required')
    need(hashlib.sha256('\n'.join(names).encode()).hexdigest()==spec['model_order_sha256'], 'Frozen model order differs')
    paths = [r['local_path'] for r in members]
    need(len(set(paths))==len(paths) and all(isinstance(p,str) and Path(p).is_absolute() for p in paths), 'Unique actual local model paths required')
    model_identities = {}
    for row in members:
        need(HEX.fullmatch(row.get('original_sha256','')) and 'flags' in row, 'Original model hash/flags required')
        path = regular(row['local_path']);need(record(path,check)['sha256']==row['original_sha256'],'Original model bytes changed: '+row['model'])
        model_identities[str(path)] = identity(path)
    core = spec['retained_reference_positions'];construct = spec['construct_residues']
    need(type(construct) is int and construct>=3 and isinstance(core,list) and len(core)>=3
         and core==sorted(set(core)) and all(type(i) is int and 0<=i<construct for i in core), 'Invalid retained canonical core')
    if target=='SERCA_P04191':
        need(construct==993 and core==list(range(57,951)) and spec['pair_mask_sha256']==SERCA_PAIR_MASK, 'SERCA requires the accepted 894-residue core58–951 of993')
    need(HEX.fullmatch(spec['feature_sha256']) and HEX.fullmatch(spec['pair_mask_sha256']), 'Bound raw feature/mask hashes required')
    config = read(plan['native']['cluster_config'],check)
    need(config==dict(schema_version=1,settings=DEFAULT_COMPONENT_SETTINGS), 'Native default component settings must remain unchanged')
    workers = plan.get('workers',1)
    need(type(workers) is int and 1<=workers<=4, 'Density CPU workers must be from1to4')
    frozen = dict(target=target,inputs=spec,native=plan['native'],workers=workers)
    return dict(names=names,paths=paths,members=members,spec=spec,shape=(len(names),len(core)*(len(core)-1)//2),
        feature_identity=identity(feature_path),member_identity=identity(spec['members']['path']),
        model_identities=model_identities,fingerprint=compact_sha(frozen),workers=workers)


@contextmanager
def readonly_matrix(entry):
    import numpy as np
    matrix = np.load(entry['path'],mmap_mode='r',allow_pickle=False)
    try:yield matrix
    finally:
        if getattr(matrix,'_mmap',None) is not None:matrix._mmap.close()


def native_adapters():
    """Verify installed native source bytes; do not patch config or globals."""
    modules = {name:importlib.import_module(name) for name in LIBRARY_SOURCE_HASHES}
    sources = {}
    for name,module in modules.items():
        path = Path(inspect.getsourcefile(module)).resolve(strict=True);entry = record(path)
        need(entry['sha256']==LIBRARY_SOURCE_HASHES[name], 'Installed Fe94 source changed: '+name);sources[name]=entry
    from rbp_density import run as density
    workspace = modules['src.analysis.feature_workspace'];clusters = modules['src.analysis.umap_component_clusters']
    return dict(matrix_loader=readonly_matrix,all_finite=workspace.all_finite,feature_sha256=workspace.feature_sha256,
        row_blocks=workspace.row_blocks,pca=modules['scripts.pca'].run_pca,umap=modules['scripts.umap_clustering'].run_umap,
        tsne=modules['scripts.tsne'].run_tsne,settings=clusters.load_component_settings,labels=clusters.find_component_clusters,
        density=density,native_directory=modules['src.data_access.accessor'].get_clustering_results_dir,
        source_evidence=lambda:sources)


def prior(plan, stage, data, check):
    import numpy as np
    entry = plan.get('phase_outputs',{}).get(stage);need(entry,'Closed '+stage+' pilot receipt required')
    result = read(entry,check)
    need(result['stage']==stage and result['status']=='diagnostic_not_full_clustering' and result['phase_complete'] is True
         and result['input_fingerprint']==data['fingerprint'] and result['models']==data['names']
         and result['local_paths']==data['paths'], 'Prior phase cohort/protocol identity differs')
    for artifact in result['outputs']:bound(artifact,check)
    key = 'projection' if stage in ('pca','umap','tsne') else 'labels'
    need(result[key] in result['outputs'], 'Prior phase sidecar is not bound by its receipt')
    with np.load(bound(result[key],check),allow_pickle=False) as saved:
        need(saved['filenames'].tolist()==data['names'] and saved['filepaths'].tolist()==data['paths'], 'Saved phase model/path order differs')
        if key=='projection':
            values=saved['embeddings'].copy();need(values.shape==(len(data['names']),2) and np.isfinite(values).all(),'Saved display projection must be finite Nx2')
        else:
            values=saved['cluster_labels'].tolist();need(len(values)==len(data['names']) and all(isinstance(v,str) and v for v in values),'Saved primary labels differ')
    return values,result


def npz(path, **arrays):
    import numpy as np
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())


def owned_files(root, check):
    result=[]
    for path in sorted(Path(root).rglob('*')):
        need(not path.is_symlink(), 'Native writer left a symlink')
        if path.is_file():result.append(record(path,check))
        else:need(path.is_dir(),'Native writer left a special file')
    return result


def run_stage(plan, stage, output, check=lambda: None, adapters=None):
    """One explicitly invoked diagnostic stage inside the caller's approved guard.

    Input descriptors are verified before mapping. All reducers use native Fe94
    flattened-input APIs; only saved displays/labels are serialized here. Each
    phase needs an empty output directory and never resumes or silently retries.
    """
    need(stage in STAGES and callable(check),'Explicit supported stage/check required')
    output=Path(output).absolute()
    need('..' not in output.parts and not any(p.is_symlink() for p in [output,*output.parents])
         and (not output.exists() or output.is_dir() and not any(output.iterdir())), 'Fresh empty phase output required; no automatic retry')
    data=load_inputs(plan,check);check()
    if not output.exists():output.mkdir(parents=True)
    ceilings={'pca':1,'umap':1,'tsne':2,'labels':2,'density0':3}
    save(output/'intent.json',dict(stage=stage,target=plan['target'],input_fingerprint=data['fingerprint'],
        status='reserved_before_native_work',maximum_projection_fits=ceilings[stage],
        maximum_HDBSCAN_extractions=6 if stage=='density0' else 0,
        failed_attempt_requires_review_before_any_new_plan=True))
    import numpy as np
    processed=os.environ.get('DATA_PROCESSED')
    need(processed and Path(processed).is_absolute(),'Caller must set absolute DATA_PROCESSED before native imports')
    funcs=native_adapters() if adapters is None else dict(adapters)
    def log(message):
        check();funcs.get('log',lambda text:print(text,flush=True))(str(message))
    artifacts=[record(output/'intent.json',check)];special={};fits=dict(PCA=0,UMAP=0,TSNE=0,HDBSCAN=0)
    flags=np.asarray([json.dumps(row['flags'],sort_keys=True,separators=(',',':'),allow_nan=False) for row in data['members']],dtype=str)
    with funcs['matrix_loader'](data['spec']['features']) as matrix:
        need(isinstance(matrix,np.ndarray) and matrix.dtype==np.float16 and matrix.shape==data['shape']
             and matrix.flags.c_contiguous and not matrix.flags.writeable,'Read-only C-order float16 selected features required')
        need(funcs['all_finite'](matrix) and not any(np.any(block<0) for block in funcs['row_blocks'](matrix)), 'Finite nonnegative distances required')
        need(funcs['feature_sha256'](matrix)==data['spec']['feature_sha256'], 'Raw feature values/order changed')
        if stage in ('pca','umap','tsne'):
            native_dir=Path(processed).absolute()/plan['target']/(stage+'_results')
            need(not native_dir.exists(),'Fresh native method cache required; no retry/overwrite')
            need(Path(funcs['native_directory'](plan['target'],stage)).absolute()==native_dir,'Native DATA_PROCESSED config was cached differently; use a fresh process')
            params=dict(n_components=2) if stage=='pca' else (dict(n_neighbors=15,min_dist=.1,n_components=2,verbose=False) if stage=='umap' else dict(perplexity=30.,implementation='python',verbose=False))
            xy,names,paths=funcs[stage](None,list(data['names']),list(data['paths']),plan['target'],'portable_selected',
                force_recompute=True,status_callback=log,residue_selection_key='shared_msa__terminal-'+data['spec']['pair_mask_sha256'],
                flattened=matrix,**params)
            xy=np.asarray(xy)
            need(list(names)==data['names'] and [str(p) for p in paths]==data['paths']
                 and xy.shape==(len(names),2) and xy.dtype.kind=='f' and np.isfinite(xy).all(), 'Native projection changed identities/shape/finite coordinates')
            file=output/'projection.npz';npz(file,embeddings=xy,filenames=np.asarray(names,dtype=str),filepaths=np.asarray(paths,dtype=str),flags_json=flags)
            special['projection']=record(file,check);artifacts.append(special['projection'])
            need(native_dir.is_dir(),'Native reducer did not write its owned cache')
            artifacts.extend(owned_files(native_dir,check))
            if plan.get('plot_points',False):
                import matplotlib
                matplotlib.use('Agg');from matplotlib import pyplot as plt
                figure,axis=plt.subplots(figsize=(6,5));axis.scatter(xy[:,0],xy[:,1],s=4)
                axis.set(xlabel=stage.upper()+' 1',ylabel=stage.upper()+' 2',title=plan['target']+' CPU pilot')
                destination=output/'projection.png';figure.savefig(destination,dpi=150);plt.close(figure);artifacts.append(record(destination,check))
            fits.update(PCA=1 if stage in ('pca','tsne') else 0,UMAP=1 if stage=='umap' else 0,TSNE=1 if stage=='tsne' else 0)
            special['parameters']=dict(params,random_seed=42)
        elif stage=='labels':
            displays={name:prior(plan,name,data,check)[0] for name in ('pca','umap','tsne')}
            settings=funcs['settings'](bound(plan['native']['cluster_config'],check))
            need(dict(asdict(settings))==dict(DEFAULT_COMPONENT_SETTINGS,repeat_seeds=(7,123)), 'Loaded native settings differ from defaults')
            numeric,report=funcs['labels'](displays['umap'],matrix,settings=settings,n_neighbors=15,min_dist=.1,base_seed=42,check_cancel=check,log=log)
            numeric=np.asarray(numeric)
            need(numeric.shape==(len(data['names']),) and numeric.dtype.kind in 'iu' and np.all(numeric>=0), 'Native primary labels must preserve all models')
            labels=['Cluster '+str(int(value)+1) for value in numeric]
            repeats=report['repeat_graphs'];need(len(repeats) in (0,2) and {r['seed'] for r in repeats}<= {7,123},'Native repeat-fit budget differs')
            need(report['point_count']==len(labels) and report['feature_input']['sha256']==data['spec']['feature_sha256'],'Primary label feature/population proof differs')
            file=output/'labels.npz';npz(file,numeric_labels=numeric,cluster_labels=np.asarray(labels,dtype=str),filenames=np.asarray(data['names'],dtype=str),filepaths=np.asarray(data['paths'],dtype=str),flags_json=flags)
            special['labels']=record(file,check);artifacts.append(special['labels'])
            save(output/'selection.json',report);artifacts.append(record(output/'selection.json',check));fits['UMAP']=len(repeats)
            special['repeat_UMAP_fits_maximum']=2
        else:
            for name in ('pca','umap','tsne'):prior(plan,name,data,check)
            labels,_=prior(plan,'labels',data,check)
            result=funcs['density'](matrix,data['names'],labels,output/'native-density0',
                native_source=plan['native']['density_source'],mode='global',retained_positions=data['spec']['retained_reference_positions'],
                allow_stored_float16=True,workers=data['workers'],check=check,log=log)
            need(result['models']==data['names'] and len(result['density'])==len(data['names'])
                 and result['mode']=='global' and result['selected_fits']==1 and result['UMAP_fits']==3
                 and result['HDBSCAN_extractions']==6 and result['stopping']['reason']=='single_global_fit'
                 and result['native_source']['commit']==DENSITY_COMMIT and result['native_source']['sha256']==DENSITY_SHA256, 'Initial density diagnostic must be exactly one native selection/3UMAP/6HDBSCAN')
            need(result['feature_proof']['stored_dtype']=='float16' and result['feature_proof']['fit_dtype']=='float32'
                 and result['feature_proof']['retained_reference_positions']==data['spec']['retained_reference_positions'],'Native density precision/core proof differs')
            file=output/'density-labels.npz';npz(file,cluster_labels=np.asarray(result['density'],dtype=str),filenames=np.asarray(data['names'],dtype=str),filepaths=np.asarray(data['paths'],dtype=str),flags_json=flags)
            special.update(labels=record(file,check),density_analysis=record(output/'native-density0/analysis.json',check),
                           full_recursive_HDBSCAN=False,default_cluster_labels=False)
            artifacts.append(special['labels']);artifacts.extend(owned_files(output/'native-density0',check));fits.update(UMAP=3,HDBSCAN=6)
    need(identity(data['spec']['features']['path'])==data['feature_identity']
         and identity(data['spec']['members']['path'])==data['member_identity']
         and all(identity(path)==value for path,value in data['model_identities'].items()), 'Frozen input identity changed during phase')
    check()
    receipt=dict(schema_version=1,status='diagnostic_not_full_clustering',phase_complete=True,stage=stage,target=plan['target'],
        input_fingerprint=data['fingerprint'],models=data['names'],local_paths=data['paths'],inputs=plan['inputs'],
        model_flags=[row['flags'] for row in data['members']],native_library_commit=LIBRARY_COMMIT,
        native_sources=funcs['source_evidence'](),module_source=record(__file__,check),scientific_fits=fits,
        full_clustering_complete=False,coordinate_parses=0,scoring_calls=0,molecular_exports=0,
        input_integrity=dict(file_sha256_verified_before_map=True,raw_float16_sha256_verified=True,
                             readonly_mapping=True,identities_unchanged_after_phase=True),outputs=artifacts,**special)
    save(output/'receipt.json',receipt)
    return receipt
