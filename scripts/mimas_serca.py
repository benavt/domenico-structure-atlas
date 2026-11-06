"""Approved full-cohort continuation; preserve closed projections and native engines.

Imports are dependency-light. The outer controller owns approval, a fresh process
per stage, kernel limits, smoke, deadlines and disk monitoring. No submission,
fit, model parsing, installation or publication occurs on import.
"""
from contextlib import contextmanager
from functools import wraps
import inspect
import hashlib
import re
import json
from pathlib import Path
import zipfile

RAW_SHA='e2475c9eebdd484dd355a65f1a6ff07efbe5dbd67c4af9edf67941ecb5c936df'
ORDER_SHA='84ee6a3a1f024e2160d3a392c396cc623e6d26863aec44b666b72a5331d98fe2'


def modules():
    import mimas_cpu as cpu
    import mimas_pilot as pilot
    return cpu,pilot


def need(condition,message):
    if not condition:raise ValueError(message)


def read(entry):
    cpu,_=modules();need(entry['bytes']<=16*1024**2,'Bounded full-cohort metadata required')
    return json.loads(cpu.bound(entry).read_text())


class MemoryGuard:
    """Actual interval PSI, host reserve and hard containment remain separate facts."""
    def __init__(self,plan,root):
        self.plan=plan;self.root=Path(root);self.previous=None;self.failures=0
        need(plan['production_pressure']==dict(admission_available_bytes=128*1024**3,runtime_available_bytes=32*1024**3,
             maximum_job_memory_full_percent=.5,maximum_job_memory_current_bytes=60*1024**3,consecutive_failures=3),'Exact reviewed production pressure policy required')
    @staticmethod
    def available():
        values={row.split(':',1)[0]:int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines()if row.startswith('MemAvailable:')}
        return values['MemAvailable']
    @staticmethod
    def full_total(observation):
        path=Path(observation['memory']['records'][0]['path']).parent/'memory.pressure'
        need(path.is_relative_to('/sys/fs/cgroup'),'Actual memory-controller PSI path required')
        try:raw=path.read_text()
        except FileNotFoundError:return None
        full=next(row for row in raw.splitlines()if row.startswith('full '))
        return int(dict(item.split('=',1)for item in full.split()[1:])['total'])
    def admit(self):
        need(self.available()>=self.plan['production_pressure']['admission_available_bytes'],'Mimas host memory below full-job admission reserve; no fit')
    def sample(self,observation,now):
        total=self.full_total(observation);available=self.available();fraction=None
        current=observation['memory']['current_bytes'];need(type(current)is int and current>=0,'Actual job-memory current required')
        if self.previous is not None:
            elapsed=now-self.previous[0];need(elapsed>0,'Actual monotonic sampling interval required')
            if self.previous[1]is not None:
                need(total is not None and total>=self.previous[1],'Previously available job PSI disappeared or decreased')
                fraction=100*(total-self.previous[1])/(elapsed*1000000)
        bad=(available<self.plan['production_pressure']['runtime_available_bytes']
             or current>self.plan['production_pressure']['maximum_job_memory_current_bytes']
             or fraction is not None and fraction>.5)
        self.failures=self.failures+1 if bad else 0;self.previous=(now,total)
        with(self.root/'memory-pressure.jsonl').open('a')as f:f.write(json.dumps(dict(monotonic=now,job_memory_full_interval_percent=fraction,job_psi_available=total is not None,job_memory_current_bytes=current,host_available_bytes=available,consecutive_failures=self.failures))+'\n')
        need(self.failures<3,'Sustained job memory pressure or exhausted host reserve; stop this job only')


def verify_plan(plan):
    """Read-only scientific identity and closed 1000-model prerequisite."""
    cpu,pilot=modules();spec=plan['inputs']
    need(plan['phase']=='serca-full'and plan['execution_host']=='mimas'
         and plan['target']=='SERCA_P04191'and plan['diagnostic_profile']=='native'
         and plan['maximum_projection_fits']==18 and plan['maximum_hdbscan_extractions']==30
         and plan['stages']==['umap1','labels','density']and spec['model_count']==5072
         and spec['feature_sha256']==RAW_SHA and spec['model_order_sha256']==ORDER_SHA
         and spec['construct_residues']==993 and spec['retained_reference_positions']==list(range(57,951))
         and spec['pair_mask_sha256']==pilot.SERCA_PAIR_MASK,'Exact accepted full SERCA scope required')
    import saved_curve_evidence
    gate=plan['correspondence1000'];need(gate in plan['frozen_files'],'Closed1000 gate must be frozen')
    verified=saved_curve_evidence.verify(gate)
    need(verified['status']=='saved_umap1000_correspondence_verified'
         and verified['mask_sha256']==spec['pair_mask_sha256']
         and all(verified['recipe'][k]==plan['curve_control']['recipe'][k]for k in ('bytes','sha256')),
         'Successful matched-curve1000 evidence required')
    members=read(spec['members'])['members'];names=[r['model']for r in members]
    need(len(names)==5072 and len(set(names))==5072 and hashlib.sha256('\n'.join(names).encode()).hexdigest()==ORDER_SHA and set(verified['models'])<=set(names),
         'Full population must include the verified1000 cohort')
    reuse=plan['reuse_projections'];need(set(reuse)=={'pca','umap','tsne'},'Three closed projections required')
    need(all(plan[k]in plan['frozen_files']for k in ('reuse_feature_proof','reuse_intake')),'Reuse cohort proofs must be frozen')
    feature=read(plan['reuse_feature_proof']);intake=read(plan['reuse_intake'])
    need(feature['feature_sha256']==RAW_SHA and feature['models_sha256']==ORDER_SHA
         and feature['retained_reference_positions']==spec['retained_reference_positions']
         and feature['pair_mask_sha256']==spec['pair_mask_sha256'], 'Saved projection feature proof differs')
    need([r['model_id']for r in intake['models']]==names
         and all(r['sha256']==m['original_sha256']for r,m in zip(intake['models'],members)),
         'Saved and native full original registries differ')
    for method,entries in reuse.items():
        for e in entries.values():
            need(e in plan['frozen_files'],'Every reused projection binding must be frozen');cpu.bound(e)
        checkpoint=read(entries['checkpoint'])
        need(checkpoint['status']=='complete'and checkpoint['method']==method
             and checkpoint['models_sha256']==ORDER_SHA and any(all(e[k]==entries['sidecar'][k]for k in ('bytes','sha256'))for e in checkpoint['outputs']),
             'Reused method is not a closed matching projection')
        if method=='pca':
            need(all(checkpoint['feature_proof'][k]==plan['reuse_feature_proof'][k]for k in ('bytes','sha256')),
                 'PCA selected-feature proof differs')
        else:need(checkpoint['feature_sha256']==RAW_SHA,'Reused nonlinear features differ')
    return verified


def projection(entry,names,paths=None):
    """Bound ZIP/NPY sizes before NumPy can allocate or decode any strings."""
    cpu,_=modules();from pilot_inputs import npy_header
    with zipfile.ZipFile(cpu.bound(entry))as z:
        need(len(z.namelist())==len(set(z.namelist()))and sum(x.file_size for x in z.infolist())<=16*1024**2,
             'Unsafe saved full-cohort projection')
        for name in ('embeddings','filenames','filepaths'):
            with z.open(name+'.npy')as f:h,header_bytes=npy_header(f)
            shape=(5072,2)if name=='embeddings'else(5072,)
            need(h['shape']==shape and (h['descr']in('<f4','<f8')if name=='embeddings'else re.fullmatch(r'<U[1-9][0-9]{0,3}',h['descr'])is not None),
                 'Wrong saved full-cohort array schema')
            width=(4 if h['descr']=='<f4'else 8)if name=='embeddings'else 4*int(h['descr'][2:])
            need(0<width<=16384 and header_bytes+5072*width*(2 if name=='embeddings'else 1)==z.getinfo(name+'.npy').file_size,
                 'Declared array size differs from bounded ZIP member')
    import numpy as np
    with np.load(entry['path'],allow_pickle=False)as z:
        xy=z['embeddings'].copy();actual_names=z['filenames'].tolist();actual_paths=z['filepaths'].tolist()
    need(actual_names==names and (paths is None or actual_paths==paths)and np.isfinite(xy).all(),
         'Saved projection identities, paths or finite coordinates differ')
    return xy


def prepare_reuse(plan,root):
    """Copy only accepted coordinates into owned sidecars; preserve their bytes/dtype."""
    import numpy as np
    cpu,pilot=modules();verify_plan(plan);data=pilot.load_inputs(plan,lambda:None)
    paths=[r['path']for r in read(plan['reuse_intake'])['models']];outputs={}
    for method,entries in plan['reuse_projections'].items():
        xy=projection(entries['sidecar'],data['names'],paths);directory=Path(root)/'reused'/method;directory.mkdir(parents=True)
        extra={}
        if method=='pca':
            from pilot_inputs import npy_header
            with zipfile.ZipFile(entries['sidecar']['path'])as z:
                with z.open('explained_variance_ratio.npy')as f:h,n=npy_header(f)
                need(h['shape']==(2,)and h['descr']in('<f4','<f8')and z.getinfo('explained_variance_ratio.npy').file_size==n+2*(4 if h['descr']=='<f4'else 8),'Bounded saved PCA variance ratios required')
            with np.load(entries['sidecar']['path'],allow_pickle=False)as z:ratio=z['explained_variance_ratio'].copy()
            need(np.isfinite(ratio).all(),'Finite saved PCA variance ratios required');extra['explained_variance_ratio']=ratio
        sidecar=directory/'projection.npz';pilot.npz(sidecar,embeddings=xy,filenames=np.asarray(data['names'],dtype=str),filepaths=np.asarray(data['paths'],dtype=str),**extra)
        receipt=dict(status='verified_saved_projection_reuse',stage=method,phase_complete=True,
            projection=cpu.record(sidecar),outputs=[cpu.record(sidecar)],input_fingerprint=data['fingerprint'],
            models=data['names'],local_paths=data['paths'],inputs=plan['inputs'],source_checkpoint=entries['checkpoint'],
            source_sidecar=entries['sidecar'],scientific_fits=dict(PCA=0,UMAP=0,TSNE=0,HDBSCAN=0),
            native_library_commit=pilot.LIBRARY_COMMIT,coordinates_unchanged=True)
        cpu.save(directory/'receipt.json',receipt);outputs[method]=cpu.record(directory/'receipt.json')
    return outputs


def require_full_correspondence(plan,result,reused):
    """The first new fit exercises the full approximate-neighbor path before labels."""
    _,pilot=modules();baseline=read(reused);names=result['models']
    left=projection(result['projection'],names,result['local_paths'])
    right=projection(baseline['projection'],names,baseline['local_paths'])
    need(left.shape==right.shape and left.dtype.str==right.dtype.str and left.tobytes()==right.tobytes(),
         'Full-cohort UMAP differs from accepted Pax coordinates; stop before labels/density')
    return dict(status='full_approximate_path_coordinates_bitwise_equal',projection_fits=0)


@contextmanager
def primary_curve(module,recipe):
    """Reuse the validated public a/b recipe for at most two primary repeats."""
    cls=module.UMAP;original=cls.__init__;signature=inspect.signature(original);applications=[]
    @wraps(original)
    def initialized(*args,**kwargs):
        call=signature.bind(*args,**kwargs);call.apply_defaults();v=call.arguments
        need(v['n_components']==2 and v['min_dist']==recipe['min_dist']and v['spread']==recipe['spread']
             and v['random_state']in(7,123)and len(applications)<2
             and all(v[k]is None or v[k]==recipe[k]for k in ('a','b')),'Unexpected primary repeat constructor')
        applications.append(v['random_state']);v.update(a=recipe['a'],b=recipe['b'])
        return original(*call.args,**call.kwargs)
    cls.__init__=initialized
    try:yield applications
    finally:
        need(cls.__init__ is initialized,'Primary UMAP constructor changed outside its owned context')
        cls.__init__=original


def run_stage(plan,stage,output):
    """Use the existing native primary and CASP17 recursive HDBSCAN implementations."""
    import numpy as np
    cpu,pilot=modules();need(stage in('labels','density'),'Unsupported production stage')
    if stage=='labels':
        import umap.umap_ as umap_module
        recipe=read(plan['curve_control']['recipe']);adapters=pilot.native_adapters();native=adapters['labels']
        def labels(*args,**kwargs):
            with primary_curve(umap_module,recipe)as calls:result=native(*args,**kwargs)
            need(len(calls)==len(result[1]['repeat_graphs']),'Primary curve application/fit count differs')
            cpu.save(Path(output)/'primary-curve.json',dict(recipe=plan['curve_control']['recipe'],effective=recipe,seeds=calls,applications=len(calls),source=cpu.record(__file__)))
            return result
        adapters['labels']=labels;return pilot.run_stage(plan,stage,output,adapters=adapters)
    data=pilot.load_inputs(plan,lambda:None);funcs=pilot.native_adapters();output=Path(output)
    cpu.save(output/'intent.json',dict(stage=stage,maximum_projection_fits=15,maximum_hdbscan_extractions=30,automatic_retry=False))
    displays={method:pilot.prior(plan,method,data,lambda:None)[0]for method in('pca','umap','tsne')}
    labels,_=pilot.prior(plan,'labels',data,lambda:None)
    with pilot.readonly_matrix(plan['inputs']['features'])as matrix:
        need(matrix.dtype==np.float16 and funcs['feature_sha256'](matrix)==RAW_SHA,'Stored full features changed')
        result=funcs['density'](matrix,data['names'],labels,output/'native-density',native_source=plan['native']['density_source'],
            mode='casp17_recursive',retained_positions=plan['inputs']['retained_reference_positions'],
            allow_stored_float16=True,workers=data['workers'],log=lambda message:print(message,flush=True))
    need(result['models']==data['names']and result['mode']=='casp17_recursive'
         and result['UMAP_fits']<=15 and result['HDBSCAN_extractions']<=30,'Recursive inference scope differs')
    from rbp_parity import write_density_labels
    primary=[dict(model=n,source_path=p,cluster_label=l,umap_x=float(x[0]),umap_y=float(x[1]))for n,p,l,x in zip(data['names'],data['paths'],labels,displays['umap'])]
    complete,assigned=write_density_labels(result,primary,output,maximum_clusters=5072)
    flags=np.asarray([json.dumps(m['flags'],sort_keys=True)for m in data['members']],dtype=str)
    pilot.npz(output/'density-labels.npz',filenames=np.asarray(data['names'],dtype=str),filepaths=np.asarray(data['paths'],dtype=str),cluster_labels=np.asarray(result['density'],dtype=str),flags_json=flags)
    need(pilot.identity(data['spec']['features']['path'])==data['feature_identity']
         and pilot.identity(data['spec']['members']['path'])==data['member_identity']
         and all(pilot.identity(p)==v for p,v in data['model_identities'].items()),'Input identity changed during recursive inference')
    artifacts=pilot.owned_files(output,lambda:None)
    receipt=dict(schema_version=1,status='full_clustering_complete',stage=stage,phase_complete=True,target=plan['target'],
        inputs=plan['inputs'],models=data['names'],local_paths=data['paths'],model_flags=[m['flags']for m in data['members']],
        native_library_commit=pilot.LIBRARY_COMMIT,native_sources=funcs['source_evidence'](),module_source=cpu.record(__file__),
        scientific_fits=dict(PCA=0,UMAP=result['UMAP_fits'],TSNE=0,HDBSCAN=result['HDBSCAN_extractions']),outputs=artifacts,
        density_analysis=cpu.record(output/'native-density/analysis.json'),labels_csv=cpu.record(complete),assigned_labels_csv=cpu.record(assigned),
        default_cluster_labels='HDBSCAN',full_recursive_HDBSCAN=True,full_clustering_complete=True,standard_deliverables_complete=False,
        coordinate_parses=0,scoring_calls=0,molecular_exports=0)
    validate_receipt(plan,stage,receipt);cpu.save(output/'receipt.json',receipt);return receipt


def validate_receipt(plan,stage,result):
    need(result['phase_complete']is True and result['stage']==stage and result['inputs']==plan['inputs'],
         'Production receipt scope differs')
    counts=result['scientific_fits'];need(counts['PCA']==counts['TSNE']==0,'Completed PCA/tSNE must never be refitted')
    if stage=='labels':
        e=result.get('primary_curve');need(e is not None and e in result['outputs'],'Bound primary curve evidence required')
        curve=read(e)
        need(curve['recipe']==plan['curve_control']['recipe']and curve['effective']==read(curve['recipe'])
             and curve['applications']==counts['UMAP']and curve['seeds']in([], [7,123]),'Primary curve recipe or seed evidence differs')
    if stage=='density':need(result.get('full_recursive_HDBSCAN')is True and result.get('default_cluster_labels')=='HDBSCAN'
         and counts['UMAP']<=15 and counts['HDBSCAN']<=30,'Default recursive HDBSCAN receipt required')
    else:need(stage=='labels'and counts['UMAP']<=2 and counts['HDBSCAN']==0,'Primary repeat budget exceeded')
