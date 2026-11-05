"""Bounded saved-output comparison of four CLOSED diagnostic jobs; no fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def modules():
    source=str(Path(__file__).absolute().parent)
    if source not in sys.path:sys.path.insert(0,source)
    import mimas_pilot
    import pilot_compare
    return mimas_pilot,pilot_compare


def read(entry,locations=None):
    pilot,_=modules();replacement=(locations or {}).get(entry['path'],entry)
    if (replacement['bytes'],replacement['sha256'])!=(entry['bytes'],entry['sha256']):raise ValueError('Relocation changed content')
    return pilot.bound(replacement)


def data(entry,locations=None):
    if entry['bytes']>16*1024**2:raise ValueError('Metadata exceeds bounded allowance')
    return json.loads(read(entry,locations).read_text())


def projection(entry,locations):
    import numpy as np
    path=read(entry,locations)
    with zipfile.ZipFile(path)as archive:
        if sum(i.file_size for i in archive.infolist())>16*1024**2:raise ValueError('Sidecar exceeds allowance')
        with archive.open('embeddings.npy')as stream:
            version=np.lib.format.read_magic(stream)
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=reader(stream)
            if shape!=(64,2)or fortran or dtype.kind!='f' or dtype.itemsize not in (4,8):raise ValueError('Invalid saved projection')
    with np.load(path,allow_pickle=False)as saved:
        result=saved['embeddings'].copy()
    if not np.isfinite(result).all():raise ValueError('Nonfinite saved coordinates')
    return result


def compare(request,output):
    pilot,geometry=modules();request=json.loads(Path(request).read_text())
    if request.get('operation')!='closed-reducer-trace-comparison':raise ValueError('Saved comparison manifest required')
    master=data(request['master'])
    source=read(master['analysis_source'])
    if source!=Path(__file__).absolute():raise ValueError('Running comparator differs from frozen source')
    if master['maximum_projection_fits']!=24 or master['maximum_hdbscan_extractions']!=0:raise ValueError('Wrong scientific scope')
    expected_keys={'pax-native','mimas-native','pax-generic','mimas-generic'}
    if set(master['host_plans'])!=expected_keys or set(request['runs'])!=expected_keys:raise ValueError('All four closed runs required')
    stages={};accounting={};identity=None
    for key,plan_entry in master['host_plans'].items():
        plan=data(plan_entry);spec=request['runs'][key];locations=data(spec['locations'])
        completion=data(spec['completion'],locations)
        if (completion.get('status')!='diagnostic_pilot_complete' or completion['plan_sha256']!=plan_entry['sha256']
                or set(completion['stage_receipts'])!={'umap1','tsne1','umap2','tsne2'}):raise ValueError('Incomplete or different diagnostic run')
        stages[key]={};counts=dict(PCA=0,UMAP=0,TSNE=0,HDBSCAN=0)
        for stage,entry in completion['stage_receipts'].items():
            receipt=data(entry,locations)
            geometry._source_proof(receipt,plan)
            if receipt['diagnostic_adapter'] not in plan['frozen_files']:
                raise ValueError('Unbound diagnostic adapter')
            if (len(receipt['models'])!=64 or len(set(receipt['models']))!=64
                    or hashlib.sha256('\n'.join(receipt['models']).encode()).hexdigest()
                    !=plan['inputs']['model_order_sha256']):raise ValueError('Model order differs')
            semantic=dict(target=receipt['target'],models=receipt['models'],flags=receipt['model_flags'],
                features=receipt['inputs']['feature_sha256'],mask=receipt['inputs']['pair_mask_sha256'])
            if identity is None:identity=semantic
            if (semantic!=identity or receipt['stage']!=stage or not receipt['phase_complete']
                    or receipt['diagnostic_profile']!=plan['diagnostic_profile']
                    or receipt['inputs']!=plan['inputs'] or receipt['native_library_commit']!=plan['native']['library_commit']
                    or receipt['portability_helper']!=plan['portability_helper']):raise ValueError('Diagnostic cohort/source/profile differs')
            trace=data(receipt['trace'],locations)
            method=stage[:-1];expected=dict(PCA=int(method=='tsne'),UMAP=int(method=='umap'),TSNE=int(method=='tsne'),HDBSCAN=0)
            params=(dict(n_neighbors=15,min_dist=.1,n_components=2,verbose=False,random_seed=42)
                    if method=='umap' else dict(perplexity=30.,implementation='python',verbose=False,random_seed=42))
            if receipt.get('parameters')!=params:raise ValueError('Reducer recipe differs')
            if (trace['status']!='complete' or trace['counts']!={'umap':int(method=='umap'),'tsne':int(method=='tsne')}
                    or receipt['scientific_fits']!=expected or trace['sources']!=plan['trace_sources']):raise ValueError('Bad closed trace or fit accounting')
            for artifact in receipt['outputs']:read(artifact,locations)
            for event in trace['events']:
                if 'artifact'in event:read(event['artifact'],locations)
            checkpoints=[{k:v for k,v in event.items()if k!='artifact'}for event in trace['events']]
            stages[key][stage]=dict(checkpoints=checkpoints,coordinates=projection(receipt['projection'],locations),receipt=entry)
            for name,n in expected.items():counts[name]+=n
        if counts!=dict(PCA=2,UMAP=2,TSNE=2,HDBSCAN=0):raise ValueError('Job exceeds exact6fit scope')
        accounting[key]=counts
    pairs=[]
    def pair(name,a,sa,b,sb):
        x,y=stages[a][sa],stages[b][sb];left,right=x['checkpoints'],y['checkpoints']
        names=[e['name']for e in left]
        if names!=[e['name']for e in right]:raise ValueError('Checkpoint interfaces differ')
        differing=[name for name,u,v in zip(names,left,right)if u!=v]
        values=geometry.geometry(x['coordinates'],y['coordinates'],geometry.DEFAULT_THRESHOLDS)
        coordinate_pass=not values['degenerate']and(values['raw_allclose']or values['rigid_allclose'])
        pairs.append(dict(name=name,left=dict(run=a,stage=sa),right=dict(run=b,stage=sb),
            first_differing_checkpoint=differing[0]if differing else None,differing_checkpoints=differing,
            coordinates_within_frozen_tolerance=coordinate_pass,geometry=values,
            verdict='matching_recorded_state_and_coordinates'if not differing and coordinate_pass else'review_required'))
    for key in sorted(expected_keys):
        for method in ('umap','tsne'):pair('same-host-repeat',key,method+'1',key,method+'2')
    for profile in ('native','generic'):
        for method in ('umap','tsne'):
            for repeat in ('1','2'):pair('cross-host-'+profile,'pax-'+profile,method+repeat,'mimas-'+profile,method+repeat)
    output=Path(output)
    if output.exists()or not output.is_absolute()or any(p.is_symlink()for p in [output,*output.parents]):raise ValueError('Fresh comparison destination required')
    output.mkdir(parents=True)
    result=dict(status='saved_diagnostic_comparison_complete',master=request['master'],actual_fit_accounting=accounting,
        pairs=pairs,thresholds=geometry.DEFAULT_THRESHOLDS,new_projection_fits=0,
        cluster_stability_validated=False,full_cohort_capacity_validated=False,
        causal_claim='First differing recorded boundary is evidence; earliest floating operation/root cause can remain unobserved.')
    pilot.save(output/'comparison.json',result)
    lines=['# Saved reducer diagnostic comparison','','No new fits. Cluster stability and full-cohort capacity remain unvalidated.','',
           '| Comparison | Left | Right | First different checkpoint | Coordinate gate |','|---|---|---|---|---|']
    for row in pairs:lines.append('| '+row['name']+' | '+row['left']['run']+'/'+row['left']['stage']+' | '+row['right']['run']+'/'+row['right']['stage']+' | '+str(row['first_differing_checkpoint'])+' | '+str(row['coordinates_within_frozen_tolerance'])+' |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest');parser.add_argument('output');args=parser.parse_args()
    result=compare(args.manifest,args.output);print(json.dumps(dict(status=result['status'],pairs=len(result['pairs']))))

if __name__=='__main__':main()
