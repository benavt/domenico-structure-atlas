"""Fresh native primary projections from an approved saved-feature terminal mask.

No work on import or automatic resume. The outer reviewed runner owns approval,
original-file verification, cgroups and estimator-stage deadlines. Safe callbacks
occur at real closed boundaries, never as estimator heartbeats. All models remain
in the cohort; coordinate parsing, molecular exports and scoring are absent.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from terminal_rescreen import read_bound, stored_matrix, validate_axes, selection


def need(condition, message):
    if not condition:raise ValueError(message)


def record(path, check=lambda: None):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]), 'Regular non-symlink artifact required')
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024**2), b''):
            check(); sha.update(block)
    return dict(path=str(path), sha256=sha.hexdigest(), bytes=path.stat().st_size)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def assert_ram_scratch(path):
    from analysis_scratch import assert_tmpfs
    assert_tmpfs()
    path = Path(path).absolute()
    need(path.is_dir() and Path('/dev/shm') in path.parents
         and not any(p.is_symlink() for p in [path, *path.parents]), 'Exact real RAM scratch directory required')


def native_adapters():
    from src.analysis.prepared_distograms import PreparedDistograms
    from src.analysis.distogram_selection_metadata import (reference_positions_to_labels,
        read_distogram_selection_metadata, sanitize_shared_reference_positions,
        validate_distogram_selection_metadata, resolve_file_assignment_key)
    from src.analysis.half_matrix import list_distogram_paths, read_distogram_manifest, peek_distance_matrix_n_residues
    from src.analysis.cluster_ensembles import ensemble_confidence_inputs
    from src.analysis.umap_component_clusters import load_component_settings
    from src.app.workers.tsne_worker import TSNEWorker
    from src.app.shared.workflow_knn import export_workflow_knn
    from src.app.shared.workflow_exports import clustering_color_schemes, export_clustering_views, pca_variance_ratios
    from scripts.clustering_common import preprocessing_cache_file
    return dict(prepared=PreparedDistograms, worker=TSNEWorker, labels=reference_positions_to_labels,
        cluster_settings=load_component_settings, knn=export_workflow_knn, schemes=clustering_color_schemes,
        plots=export_clustering_views, ratios=pca_variance_ratios, cache_path=preprocessing_cache_file,
        confidence=ensemble_confidence_inputs, metadata=dict(read=read_distogram_selection_metadata,
            sanitize=sanitize_shared_reference_positions,validate=validate_distogram_selection_metadata,
            resolve=resolve_file_assignment_key,paths=list_distogram_paths,manifest=read_distogram_manifest,
            dimension=peek_distance_matrix_n_residues))



def metadata_preflight(metadata, signature, directory, positions, funcs, check, expected_manifest):
    """Prove public-worker metadata rebuild is unnecessary on frozen inputs.

    No coordinate reader or private worker API is invoked. Both worker rebuild
    predicates are false when these stronger public metadata/stat checks pass.
    Repeat before each estimator; input immutability remains the outer contract.
    """
    clean = funcs['sanitize'](deepcopy(metadata))
    need(funcs['read'](directory) == metadata and not funcs['validate'](clean)
         and clean.get('reference',{}).get('sequence')
         and clean.get('shared_reference_positions') == positions, 'Metadata would require native rebuilding')
    groups = clean.get('sequence_groups') or []
    need(groups and all(type(g.get('sequence_length')) is int and g['sequence_length'] > 0
         and len(g.get('reference_to_group_index') or []) == len(positions) for g in groups),
         'Incomplete native sequence-group mappings')
    paths = [Path(row[0]).absolute() for row in signature['files']]
    need([Path(p).absolute() for p in funcs['paths'](directory)] == paths, 'Native distogram population changed')
    manifest = funcs['manifest'](directory)
    need(manifest == expected_manifest and manifest.get('file_format') == 'bin' and manifest.get('storage') == 'condensed'
         and manifest.get('dtype') == 'float16' and manifest.get('endianness') == 'little', 'Native BIN manifest differs')
    assignments = clean.get('file_assignments') or {}; per_file = manifest.get('per_file_n_residues') or {}
    for row,path in zip(signature['files'],paths):
        check(); status = path.stat()
        need(path.is_file() and not any(p.is_symlink() for p in [path,*path.parents])
             and path.parent == directory and path.suffix == '.bin'
             and [status.st_size,status.st_mtime_ns] == row[1:3]
             and status.st_size == len(positions)*(len(positions)-1)
             and per_file.get(path.stem) == len(positions)
             and funcs['dimension'](path,manifest=manifest) == len(positions)
             and funcs['resolve'](assignments,path.stem,clean,len(positions)) is not None,
             'Native BIN identity/dimension/assignment changed; refusing rebuild fallback')
    return dict(status='passed',files=len(paths),shared_positions=len(positions),
                coordinate_rebuild_required=False,sidecar_rewrite_required=False)


def run(descriptor, terminal_report, metrics_db, output, scratch, *, members,
        model_type_config, cluster_config, check, checkpoint, log=lambda _: None, adapters=None):
    """Run PCA/UMAP/t-SNE plus component-validated primary labels on one target.

    Bound JSON arguments use {path,sha256}; ``members`` is the outer-verified full
    original registry in exact cache order. Native DB reads use the fresh copy;
    stored model-type and confidence policy are inherited without reinterpretation.
    The returned retained-feature NPY belongs to the caller's RAM scratch. It is
    closed here and retained for density; the caller verifies it before reuse and
    removes it only after closing every map. Failure never triggers a retry.
    """
    import numpy as np
    need(callable(check) and callable(checkpoint), 'Explicit guard and safe-boundary callbacks required')
    target = descriptor['target']
    need(isinstance(target, str) and re.fullmatch(r'[A-Za-z0-9_-]+', target), 'Unsafe target identity')
    axes = read_bound(descriptor['axes'], check); baseline = read_bound(descriptor['baseline'], check)
    baseline = baseline.get('terminal_detection', baseline)
    names, positions, keys = validate_axes(axes, baseline)
    metadata = read_bound(descriptor['source_metadata'], check)
    need(all(metadata['reference'][k] == axes['metadata']['reference'][k]
             for k in ('sequence','chain_ids','residue_numbers'))
         and metadata['shared_reference_positions'] == positions, 'Source axes changed')
    report = read_bound(terminal_report, check); chosen = selection(report, positions, keys)
    need(report['model_count'] == len(names) and report['reference_positions'] == positions
         and report['retained_reference_positions'] == chosen['retained_canonical_indices']
         and report['settings'] == read_bound(descriptor['settings'], check)['settings'], 'Accepted terminal report differs')
    kept = report['retained_indices']; retained = np.zeros(len(positions), bool); retained[kept] = True
    i,j = np.triu_indices(len(positions),1); mask = retained[i] & retained[j]; columns = np.flatnonzero(mask)
    need(len(columns) == len(kept)*(len(kept)-1)//2
         and report['retained_feature_count'] == len(columns)
         and hashlib.sha256(mask.tobytes()).hexdigest() == report['pair_mask_sha256'], 'Accepted pair mask differs')
    members = list(members)
    need([r['model'] for r in members] == names and all(
        Path(r['source_path']).is_absolute() and Path(r['source_path']).stem == r['model']
        and re.fullmatch(r'[0-9a-f]{64}', r['sha256']) for r in members), 'Exact complete original registry required')
    parents = {str(Path(r['source_path']).parent) for r in members}
    need(len(parents) == 1, 'One exact original source directory required')
    original_paths = [r['source_path'] for r in members]
    prior = read_bound(descriptor['prior_workflow_manifest'], check)
    need(prior['identifier'] == target and prior['structure_count'] == len(names), 'Prior target/cohort differs')
    read_bound(model_type_config, check); cluster_record = read_bound(cluster_config, check)
    need(cluster_record.get('schema_version') == 1
         and cluster_record['settings'].get('repeat_seeds') == [7,123]
         and cluster_record['settings'].get('query_seed') == 42
         and cluster_record['settings'].get('maximum_subclusters_per_component') == 8,
         'Primary fit budget requires exact seeds7/123, base/query42 and8subclusters')
    metrics = record(metrics_db, check)
    need(str(Path(metrics_db).absolute()) != str(Path(prior['metrics_db']).absolute()), 'Fresh metrics copy required')
    output = Path(output).absolute(); scratch = Path(scratch).absolute(); assert_ram_scratch(scratch)
    need(not output.exists() and not any(p.is_symlink() for p in [output,*output.parents]), 'Fresh immutable primary output required')
    retained_path = scratch/(target+'-retained-features.npy')
    need(not retained_path.exists() and not retained_path.is_symlink(), 'Retained RAM feature path already exists')
    funcs = native_adapters() if adapters is None else dict(adapters)
    need(set(funcs) == {'prepared','worker','labels','cluster_settings','knn','schemes','plots','ratios','cache_path','metadata','confidence'}, 'Native public projection interfaces differ')
    effective_settings = funcs['cluster_settings'](cluster_config['path'])
    need(tuple(effective_settings.repeat_seeds) == (7,123) and effective_settings.query_seed == 42
         and effective_settings.maximum_subclusters_per_component == 8, 'Native component settings exceed reviewed primary fit budget')
    expected_manifest = read_bound(descriptor['distogram_manifest'],check)
    need(record(descriptor['distogram_manifest']['path'],check)['bytes'] == descriptor['distogram_manifest']['bytes'],
         'Bound distogram manifest size differs')
    directory = Path(descriptor['source_metadata']['path']).absolute().parent
    cache_path = Path(funcs['cache_path'](target, directory.name, basename_keyword='', residue_selection_key='shared_msa'))
    processed = Path(os.environ['DATA_PROCESSED']).absolute()
    need(cache_path.is_relative_to(processed) and not cache_path.exists(), 'Fresh native feature-cache namespace required')
    for method in ('pca','umap','tsne'):
        folder = cache_path.parent.parent/(method+'_results')
        need(not folder.exists() or folder.is_dir() and not any(folder.iterdir()), 'Native projection namespace contains prior artifacts')
    output.mkdir(parents=True); (output/'clustering').mkdir(); (output/'checkpoints').mkdir()
    save(output/'terminal_detection.json', report)
    manifest = dict(schema_version=1,kind='fresh-primary-projections-from-accepted-terminal-mask',
        identifier=target,pdb_id='',structure_count=len(names),source_dir=next(iter(parents)),
        distogram_dir=str(directory),output_dir=str(output),metrics_db=str(Path(metrics_db).absolute()),
        clustering={},knn_clusters_enabled=True,cluster_selector='component_preserving_ward_feature_validated',
        cluster_ensembles_enabled=True,analysis_status='running',status='running',
        deliverable_status='pending_density_model_selection_and_exports',
        defaults=dict(atom_type='CA',residues='shared',dimensions=2,random_seed=42,tsne_implementation='python'),
        terminal_detection=report,terminal_detection_report=str(output/'terminal_detection.json'),
        model_type_config_source=prior.get('model_type_config_source','explicit'),model_type_config=model_type_config['path'],
        derivation=dict(kind='fresh_fits_after_changed_terminal_mask',prior_workflow=descriptor['prior_workflow_manifest'],
            accepted_terminal_report=terminal_report,source_cache=descriptor['cache'],axes=descriptor['axes'],
            source_metadata=descriptor['source_metadata'],distogram_manifest=descriptor['distogram_manifest'],model_type_snapshot=model_type_config,
            cluster_settings=cluster_config,seed_metrics=metrics,model_names_sha256=axes['names_sha256'],
            original_registry_sha256=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            coordinates_inherited=False,coordinate_parses=0,scoring_calls=0,new_molecular_exports=0))
    for key in ('input_inventory','dataset_sha256','input_file_count','dataset_scope','presentation_target',
                'ensemble_selection_scope','ensemble_confidence_preflight','requested_exports'):
        if key in prior:manifest[key] = deepcopy(prior[key])
    manifest['plot_model_type_config'] = model_type_config['path']
    inventory_path = Path(prior['input_inventory'])
    inventory = json.loads(inventory_path.read_text())
    original_inventory = {(r['model'],r['original_source_path'],r['original_sha256']) for r in inventory['models']}
    need(len(inventory['models']) == len(names) and original_inventory == {(r['model'],r['source_path'],r['sha256']) for r in members},
         'Inherited input inventory differs from complete original registry')
    manifest['derivation']['input_inventory'] = record(inventory_path,check)
    _,_,_,confidence,_ = funcs['confidence'](manifest,original_paths)
    need(confidence == prior['ensemble_confidence_preflight'], 'Fresh metrics confidence scope/policy differs')
    need(record(metrics_db,check) == metrics, 'Fresh metrics changed during confidence preflight')
    matrix = None; projections = {}; feature_digest = hashlib.sha256()
    try:
        with stored_matrix(descriptor['cache'],axes,baseline,check=check,source_metadata=metadata) as full:
            with np.load(descriptor['cache']['path'],allow_pickle=False) as saved:
                signature = json.loads(str(saved['source_signature']))
            paths = [Path(row[0]).absolute() for row in signature['files']]
            need([p.stem for p in paths] == names and all(p.parent == directory and p.suffix == '.bin' for p in paths),
                 'Native original distance path/order differs')
            metadata_proof = metadata_preflight(metadata,signature,directory,positions,funcs['metadata'],check,expected_manifest)
            manifest['derivation']['metadata_preflight'] = metadata_proof
            matrix = np.lib.format.open_memmap(retained_path,mode='w+',dtype=np.float16,shape=(len(names),len(columns)))
            rows = max(1,16*1024**2//max(1,len(columns)*2))
            for start in range(0,len(names),rows):
                check(); stop = min(start+rows,len(names))
                # Select bounded rows first; never allocate full[:, columns].
                block = np.asarray(full[start:stop])[:,columns].copy(order='C')
                need(block.dtype == np.float16 and np.isfinite(block).all(), 'Nonfinite saved retained features')
                matrix[start:stop] = block; feature_digest.update(memoryview(block).cast('B')); del block
                checkpoint(f'{target}:features:{stop}/{len(names)}')
            matrix.flush(); matrix.setflags(write=False)
            feature_file = record(retained_path,check)
            feature = dict(path=str(retained_path),shape=list(matrix.shape),dtype='float16',
                           file_sha256=feature_file['sha256'],feature_sha256=feature_digest.hexdigest(),
                           temporary=True,retention='Plan-owned RAMscratch; outer completion receipt records retirement after density.')
            save(output/'feature-proof.json',dict(**feature,pair_mask_sha256=report['pair_mask_sha256'],
                models_sha256=axes['names_sha256'],retained_reference_positions=report['retained_reference_positions'],
                equivalence='Exact original cached float16 pairs; no distance recomputation or coordinate parsing.'))
            signature['terminal_rescreen_report_sha256'] = terminal_report['sha256']
            prepared = funcs['prepared'](directory=directory,paths=paths,stack=full,flattened=matrix,
                labels=funcs['labels'](metadata,positions),warnings=[],reference_positions=report['retained_reference_positions'],
                requested_reference_positions=positions,terminal_detection=report,source_signature=signature)
            schemes = funcs['schemes'](metrics_db,model_type_config['path'])
            for method in ('pca','umap','tsne'):
                check(); checkpoint(target+':before-'+method)
                metadata_preflight(metadata,signature,directory,positions,funcs['metadata'],check,expected_manifest)
                captured = []; errors = []; html = output/'clustering'/(method+'.html')
                worker = funcs['worker'](pdb_id='',identifier=target,pdb_dir=manifest['source_dir'],
                    distogram_dir=str(directory),output_file=str(html),clustering_method=method,
                    tsne_implementation='python',force_recompute=True,pca_n_components=2,
                    umap_n_neighbors=min(15,len(names)-1),umap_min_dist=.1,umap_n_components=2,
                    perplexity=30. if len(names)>30 else max(1.,(len(names)-1)/3),enable_pymol=False,
                    model_type_config_path=model_type_config['path'],metrics_db_path=str(metrics_db),
                    plot_source_paths=dict(zip(names,original_paths)),prepared_distograms=prepared,
                    embedding_callback=lambda xy,n,p:captured.append((xy,list(n),list(p))),
                    error_callback=errors.append,status_callback=log)
                worker.run(); check()
                need(record(descriptor['source_metadata']['path'],check)['sha256'] == descriptor['source_metadata']['sha256'],
                     'Original native metadata sidecar changed during worker')
                need(record(descriptor['distogram_manifest']['path'],check)['sha256'] == descriptor['distogram_manifest']['sha256'],
                     'Original distogram manifest changed during worker')
                need(not errors and len(captured)==1, 'Native '+method+' failed or did not return one projection: '+str(errors))
                xy,order,returned_paths = captured[0]
                need(order == names and [Path(p).stem for p in returned_paths] == names
                     and np.asarray(xy).shape == (len(names),2) and np.isfinite(xy).all(), 'Native projection population/order differs')
                projections[method] = (np.asarray(xy),names,original_paths)
                ratios = funcs['ratios'](xy,matrix) if method=='pca' else None
                sidecar = output/'clustering'/(method+'_embeddings.npz')
                np.savez_compressed(sidecar,embeddings=xy,filenames=np.asarray(names,dtype=str),
                    filepaths=np.asarray(original_paths,dtype=str),**({'explained_variance_ratio':ratios} if ratios is not None else {}))
                images = funcs['plots'](xy,names,original_paths,method,schemes,metrics_db,
                    manifest['source_dir'],output/'clustering'/method,explained_variance_ratio=ratios,
                    check_cancel=check,log=log)
                params = dict(n_components=2) if method=='pca' else (dict(n_neighbors=min(15,len(names)-1),min_dist=.1,n_components=2)
                    if method=='umap' else dict(perplexity=30. if len(names)>30 else max(1.,(len(names)-1)/3)))
                manifest['clustering'][method] = dict(html=str(html),embeddings=str(sidecar),params=params,images=images,
                    **({'explained_variance_ratio':ratios} if ratios is not None else {}))
                base_html = output/'checkpoints'/(method+'-base.html')
                shutil.copy2(html,base_html)
                receipt = dict(method=method,status='complete',native_output_html=str(html),base_html=str(base_html),feature_proof=record(output/'feature-proof.json',check),
                    models_sha256=axes['names_sha256'],settings=params,native_library_required='fe94ba765fd7a9f9894a5e5ba11a57f582a39365',
                    outputs=[record(p,check) for p in [base_html,sidecar,*[row['image'] for row in images]]],
                    primary_PCA_fits=1 if method=='pca' else 0,UMAP_fits=1 if method=='umap' else 0,
                    tsne_fits=1 if method=='tsne' else 0,internal_tsne_PCA_fits=1 if method=='tsne' else 0)
                save(output/'checkpoints'/(method+'.json'),receipt)
                checkpoint(target+':closed-'+method)
            checkpoint(target+':before-primary-labels'); check()
            manifest['knn_clustering'] = funcs['knn'](manifest,projections,features=matrix,feature_names=names,
                settings=effective_settings,projection='umap',n_clusters=None,
                check_cancel=check,log=log)
            need(manifest['knn_clustering']['feature_input']['sha256'] == feature['feature_sha256'], 'Native label feature proof differs')
            save(output/'checkpoints'/'labels.json',dict(status='complete',feature_sha256=feature['feature_sha256'],
                outputs=[record(manifest['knn_clustering'][key],check) for key in ('labels_csv','report_json')],
                labelled_views=[record(v['html'],check) for v in manifest['clustering'].values()],
                repeat_UMAP_fits_max=2,automatic_resume=False))
            checkpoint(target+':closed-primary-labels')
        need(record(metrics_db,check)==metrics, 'Fresh metrics values changed during projection phase')
        need(record(model_type_config['path'],check)['sha256']==model_type_config['sha256']
             and record(cluster_config['path'],check)['sha256']==cluster_config['sha256'], 'Stored configuration changed')
        feature_cache = record(cache_path,check)
        manifest.update(status='complete',analysis_status='complete',stage_scope='primary projections and primary labels only')
        manifest['derivation']['retained_feature_proof'] = feature
        manifest['derivation']['native_feature_cache'] = feature_cache
        save(output/'manifest.json',manifest)
        result = dict(manifest=record(output/'manifest.json',check),feature_cache=feature_cache,retained_features=feature,
                      projection_fits_max=6,coordinate_parses=0,scoring_calls=0,molecular_exports=0)
        save(output/'primary-artifacts.json',result); checkpoint(target+':primary-complete')
        return result
    finally:
        if matrix is not None:matrix._mmap.close()
