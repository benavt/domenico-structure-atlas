"""Reverify accepted Pax/Mimas UMAP64 correspondence without fitting or NumPy."""
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile


def bound(entry,locations=None):
    target=(locations or {}).get(entry['path'],entry);p=Path(target['path'])
    if ((entry['bytes'],entry['sha256'])!=(target['bytes'],target['sha256']) or not p.is_absolute()
            or any(x.is_symlink()for x in [p,*p.parents])or not p.is_file()or p.stat().st_size!=entry['bytes']):
        raise ValueError('Saved evidence path/content differs')
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    if h.hexdigest()!=entry['sha256']:raise ValueError('Saved evidence hash differs')
    return p


def data(entry,locations=None):
    if entry['bytes']>16*1024**2:raise ValueError('Evidence metadata exceeds limit')
    return json.loads(bound(entry,locations).read_text())


def embedding_member(entry,locations,rows=64):
    from pilot_inputs import npy_header
    with zipfile.ZipFile(bound(entry,locations))as z:
        names=z.namelist()
        if len(names)!=len(set(names))or sum(x.file_size for x in z.infolist())>16*1024**2:
            raise ValueError('Unsafe saved sidecar')
        raw=z.read('embeddings.npy')
    import io
    stream=io.BytesIO(raw);header,_=npy_header(stream)
    if rows not in (64,1000)or header['shape']!=(rows,2)or header['descr']not in ('<f4','<f8'):
        raise ValueError('Wrong saved coordinates')
    width=4 if header['descr']=='<f4'else 8;body=stream.read()
    if len(body)!=rows*2*width or not all(math.isfinite(v[0])for v in struct.iter_unpack('<f'if width==4 else'<d',body)):
        raise ValueError('Nonfinite or malformed saved coordinates')
    return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def verify_pax1000_predecessor(plan):
    """A remote fit cannot start before the matching reference job closes."""
    import pilot_compare as geometry
    path=Path(plan['pax1000_gate'])
    if not path.is_absolute()or any(p.is_symlink()for p in [path,*path.parents])or path.stat().st_size>16*1024**2:
        raise ValueError('Exact bounded predecessor gate required')
    proof=json.loads(path.read_text())
    if proof.get('operation')!='closed-pax1000-predecessor'or proof['plan']['sha256']!=plan['pax1000_plan_sha256']:
        raise ValueError('Wrong reference predecessor plan')
    source=data(proof['plan']);loc=data(proof['locations']);done=data(proof['completion'],loc)
    if (source['phase']!='curve1000'or source['execution_host']!='pax'or source['diagnostic_profile']!='native'
            or done['status']!='diagnostic_pilot_complete'or done['plan_sha256']!=proof['plan']['sha256']
            or set(done['stage_receipts'])!={'umap1'}):raise ValueError('Reference1000 job is not closed')
    r=data(done['stage_receipts']['umap1'],loc);geometry._source_proof(r,source)
    keys=('model_count','construct_residues','retained_reference_positions','feature_sha256','model_order_sha256','pair_mask_sha256')
    if (any(source['inputs'][k]!=plan['inputs'][k]for k in keys)or r['inputs']!=source['inputs']
            or not r['phase_complete']or r['stage']!='umap1'or r['diagnostic_profile']!='native'
            or len(r['models'])!=1000 or len(set(r['models']))!=1000
            or hashlib.sha256('\n'.join(r['models']).encode()).hexdigest()!=source['inputs']['model_order_sha256']
            or r['scientific_fits']!=dict(PCA=0,UMAP=1,TSNE=0,HDBSCAN=0)
            or r['parameters']!=dict(n_neighbors=15,min_dist=.1,n_components=2,verbose=False,random_seed=42)):
        raise ValueError('Reference1000 scientific scope differs')
    for e in r['outputs']:bound(e,loc)
    trace=data(r['trace'],loc)
    if (trace['status']!='complete'or trace['counts']!=dict(umap=1,tsne=0)
            or trace.get('curve_applications')!=1
            or any(trace['umap_curve'][k]!=plan['curve_control']['recipe'][k]for k in ('bytes','sha256'))):
        raise ValueError('Reference1000 curve evidence differs')
    exit=data(proof['pressure_exit'],loc)
    if exit.get('status')!='complete'or exit.get('returncode')!=0:raise ValueError('Reference pressure guard did not complete')
    embedding_member(r['projection'],loc,rows=1000)
    return dict(status='closed_pax1000_predecessor_verified',projection_fits=0)


def verify(proof_entry):
    import pilot_compare as geometry
    proof=data(proof_entry)
    rows=1000 if proof.get('operation')=='closed-saved-umap1000-correspondence'else 64
    if proof.get('operation')not in ('closed-saved-umap64-correspondence','closed-saved-umap1000-correspondence') or set(proof['runs'])!={'pax','mimas'}:
        raise ValueError('Exact accepted correspondence proof required')
    recipe=data(proof['recipe']);runs={};identity=None
    for host,spec in proof['runs'].items():
        plan=data(spec['plan']);locations=data(spec['locations']);done=data(spec['completion'],locations)
        if (plan['execution_host']!=host or plan['phase']!=('curve1000'if rows==1000 else'isolate64'if host=='pax'else'curve64')
                or plan['diagnostic_profile']!='native' or done['status']!='diagnostic_pilot_complete'
                or done['plan_sha256']!=spec['plan']['sha256']):raise ValueError('Unclosed saved reference')
        r=data(done['stage_receipts']['umap1'],locations);geometry._source_proof(r,plan)
        if (r['stage']!='umap1'or not r['phase_complete']or r['inputs']!=plan['inputs']
                or r['diagnostic_profile']!='native' or r['diagnostic_adapter']not in plan['frozen_files']
                or r['parameters']!=dict(n_neighbors=15,min_dist=.1,n_components=2,verbose=False,random_seed=42)
                or r['scientific_fits']!=dict(PCA=0,UMAP=1,TSNE=0,HDBSCAN=0)):
            raise ValueError('Changed saved native protocol')
        ids=r['models'];current=dict(target=r['target'],models=ids,flags=r['model_flags'],
            feature_sha=r['inputs']['feature_sha256'],mask_sha=r['inputs']['pair_mask_sha256'])
        if (len(ids)!=rows or len(set(ids))!=rows
                or hashlib.sha256('\n'.join(ids).encode()).hexdigest()!=r['inputs']['model_order_sha256']):
            raise ValueError('Wrong saved cohort/order')
        if identity is None:identity=current
        if current!=identity:raise ValueError('Cross-host saved cohort differs')
        for e in r['outputs']:bound(e,locations)
        t=data(r['trace'],locations)
        if t['status']!='complete'or t['counts']!=dict(umap=1,tsne=0)or t['sources']!=plan['trace_sources']:
            raise ValueError('Saved trace not closed')
        events={}
        for event in t['events']:
            if event['name']in events:raise ValueError('Duplicate saved checkpoint')
            if 'artifact'in event:
                if event['artifact']not in r['outputs']:raise ValueError('Unbound saved checkpoint')
                bound(event['artifact'],locations)
            events[event['name']]={k:v for k,v in event.items()if k!='artifact'}
        scalars=events['umap1_optimizer_scalars']['values']
        if any(scalars[k]!=recipe[k]for k in ['a','b']):raise ValueError('Saved curve differs')
        if host=='mimas'or rows==1000:
            if (t.get('curve_applications')!=1 or t['umap_curve']!=plan['curve_control']['recipe']
                    or any(t['umap_curve'][k]!=proof['recipe'][k]for k in ['bytes','sha256'])
                    or events['umap_curve_effective']['values']!=recipe):raise ValueError('Mimas curve was not applied')
        runs[host]=dict(events=events,coordinates=embedding_member(r['projection'],locations,rows=rows))
    left,right=runs['pax'],runs['mimas'];names=list(left['events'])
    required={'umap1_graph_data','umap1_graph_indices','umap1_graph_indptr','umap1_initial','umap1_rng',
              'umap1_optimizer_scalars','umap1_epoch0_scalars','umap1_epoch1_scalars','umap1_epoch1_after_head_embedding'}
    if (len(names)<35 or not required<=set(names)or not set(names)<=set(right['events'])
            or any(left['events'][k]!=right['events'][k]for k in names)
            or left['coordinates']!=right['coordinates']):raise ValueError('Saved exact correspondence gate fails')
    return dict(status='saved_umap'+str(rows)+'_correspondence_verified',projection_fits=0,common_checkpoints=len(names),
        models=identity['models'],
        coordinates=left['coordinates'],model_order_sha256=hashlib.sha256('\n'.join(identity['models']).encode()).hexdigest(),
        feature_sha256=identity['feature_sha'],mask_sha256=identity['mask_sha'],recipe=proof['recipe'],
        new_pax_controlled_fit_completed=rows==1000,reference_policy='Both controlled1000 fits closed'if rows==1000 else'Reuse accepted closed native Pax result; new Pax64 confirmation was refused before fitting')
