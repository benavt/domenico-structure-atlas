"""Closed paired UMAP curve comparison with an exact gate; no fitting or submission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def modules():
    sys.path.insert(0,str(Path(__file__).absolute().parent))
    import reducer_trace_compare as saved
    import pilot_compare as geometry
    import hpc_cpu as cpu
    return saved,geometry,cpu


def bitwise(left,right):
    return (left.shape==right.shape and left.dtype.str==right.dtype.str
            and left.tobytes(order='C')==right.tobytes(order='C'))


def compare(request_path,output):
    saved,geometry,cpu=modules();request=json.loads(Path(request_path).read_text())
    if request.get('operation')!='closed-umap-curve-comparison':raise ValueError('Closed curve comparison required')
    master=saved.data(request['master'])
    if (master.get('operation')!='paired-umap-curve64' or master['maximum_projection_fits']!=2
            or master['maximum_hdbscan_extractions']!=0 or set(master['host_plans'])!={'workstation','hpc'}
            or set(request['runs'])!={'workstation','hpc'}
            or saved.read(master['analysis_source'])!=Path(__file__).absolute()):raise ValueError('Wrong curve experiment')
    runs={};identity=None
    for host in ('workstation','hpc'):
        entry=master['host_plans'][host];plan=saved.data(entry);spec=request['runs'][host]
        locations=saved.data(spec['locations']);done=saved.data(spec['completion'],locations)
        if (plan.get('phase')!='curve64' or plan['execution_host']!=host or plan['diagnostic_profile']!='native'
                or done['status']!='diagnostic_pilot_complete' or done['plan_sha256']!=entry['sha256']
                or set(done['stage_receipts'])!={'umap1'}):raise ValueError('Unclosed or changed curve run')
        receipt=saved.data(done['stage_receipts']['umap1'],locations);geometry._source_proof(receipt,plan)
        if (receipt['stage']!='umap1' or not receipt['phase_complete'] or receipt['inputs']!=plan['inputs']
                or receipt['diagnostic_adapter'] not in plan['frozen_files']
                or receipt['portability_helper']!=plan['portability_helper']
                or receipt['diagnostic_profile']!='native' or receipt['curve_control']!=plan['curve_control']
                or receipt['parameters']!=dict(n_neighbors=15,min_dist=.1,n_components=2,verbose=False,random_seed=42)
                or receipt['scientific_fits']!=dict(PCA=0,UMAP=1,TSNE=0,HDBSCAN=0)):
            raise ValueError('Changed curve receipt/recipe/source/counts')
        current=dict(target=receipt['target'],models=receipt['models'],flags=receipt['model_flags'],
            feature_sha=plan['inputs']['feature_sha256'],mask_sha=plan['inputs']['pair_mask_sha256'])
        if identity is None:identity=current
        if (current!=identity or len(set(receipt['models']))!=64 or len(receipt['models'])!=64
                or hashlib.sha256('\n'.join(receipt['models']).encode()).hexdigest()!=plan['inputs']['model_order_sha256']):
            raise ValueError('Cohort/order/flags differ')
        trace=saved.data(receipt['trace'],locations);recipe=saved.data(master['curve_recipe'])
        if any(plan['curve_control']['recipe'][k]!=master['curve_recipe'][k]for k in ('bytes','sha256')):
            raise ValueError('Host curve recipe differs from the master')
        if (trace['status']!='complete' or trace['counts']!=dict(umap=1,tsne=0)
                or trace['sources']!=plan['trace_sources'] or trace['umap_curve']!=plan['curve_control']['recipe']
                or trace.get('curve_applications')!=1 or trace.get('optimizer_epoch_trace')is not True
                or trace.get('epoch_calls',0)<2 or trace.get('initialization_replays')!=0):
            raise ValueError('Unclosed curve/epoch evidence')
        for e in receipt['outputs']:saved.read(e,locations)
        events=[]
        for event in trace['events']:
            if 'artifact'in event:
                if event['artifact']not in receipt['outputs']:raise ValueError('Unbound curve evidence')
                saved.read(event['artifact'],locations)
            events.append({k:v for k,v in event.items()if k!='artifact'})
        names=[e['name']for e in events]
        if (len(names)!=len(set(names)) or not {'umap_curve_requested','umap_curve_effective','umap1_optimizer_scalars',
                'umap1_epoch0_scalars','umap1_epoch1_scalars','umap1_epoch1_after_head_embedding'}<=set(names)
                or next(e['values']for e in events if e['name']=='umap_curve_effective')!=recipe):
            raise ValueError('Effective constructor recipe or extended trace missing')
        optimizer=next(e['values']for e in events if e['name']=='umap1_optimizer_scalars')
        if any(optimizer[k]!=recipe[k]for k in ('a','b')):raise ValueError('Optimizer did not use the bound a/b')
        runs[host]=dict(events=events,coordinates=saved.projection(receipt['projection'],locations),identity=current)
    left,right=runs['workstation'],runs['hpc']
    if [e['name']for e in left['events']]!=[e['name']for e in right['events']]:raise ValueError('Trace interfaces differ')
    differences=[a['name']for a,b in zip(left['events'],right['events'])if a!=b]
    reference=master['reference_workstation'];locations=saved.data(reference['locations']);prior_plan=saved.data(reference['plan'])
    done=saved.data(reference['completion'],locations)
    if (done['status']!='diagnostic_pilot_complete' or done['plan_sha256']!=reference['plan']['sha256']
            or set(done['stage_receipts'])!={'umap1','tsne1'}
            or prior_plan['execution_host']!='workstation' or prior_plan['phase']!='isolate64'
            or prior_plan['diagnostic_profile']!='native'):raise ValueError('Unclosed reference')
    prior=saved.data(done['stage_receipts']['umap1'],locations);geometry._source_proof(prior,prior_plan)
    if (not prior['phase_complete'] or prior['inputs']!=prior_plan['inputs'] or prior['models']!=identity['models']
            or prior['parameters']!=dict(n_neighbors=15,min_dist=.1,n_components=2,verbose=False,random_seed=42)
            or prior['scientific_fits']!=dict(PCA=0,UMAP=1,TSNE=0,HDBSCAN=0)
            or prior['model_flags']!=identity['flags'] or prior['inputs']['feature_sha256']!=identity['feature_sha']
            or prior['inputs']['pair_mask_sha256']!=identity['mask_sha']):raise ValueError('Reference cohort differs')
    for e in prior['outputs']:saved.read(e,locations)
    baseline=saved.projection(prior['projection'],locations)
    cross_exact=bitwise(left['coordinates'],right['coordinates']);workstation_preserved=bitwise(left['coordinates'],baseline)
    result=dict(status='closed_umap_curve_comparison_complete',master=request['master'],projection_fits=2,
        actual_fit_accounting={host:dict(PCA=0,UMAP=1,TSNE=0,HDBSCAN=0)for host in runs},new_comparison_fits=0,
        cross_host_bitwise_identical=cross_exact,workstation_native_bitwise_preserved=workstation_preserved,
        differing_checkpoints=differences,first_differing_checkpoint=differences[0]if differences else None,
        exact_gate_passed=cross_exact and workstation_preserved and not differences,
        geometry=geometry.geometry(left['coordinates'],right['coordinates'],geometry.DEFAULT_THRESHOLDS),
        same_host_workstation_geometry=geometry.geometry(left['coordinates'],baseline,geometry.DEFAULT_THRESHOLDS),
        larger_cohort_authorized=False,cluster_stability_validated=False,full_cohort_capacity_validated=False)
    output=Path(output)
    if not output.is_absolute()or output.exists()or any(p.is_symlink()for p in [output,*output.parents]):raise ValueError('Fresh output required')
    output.mkdir(parents=True);cpu.save(output/'comparison.json',result)
    (output/'comparison.md').write_text('# Matched-curve UMAP comparison\n\nExact gate: '+str(result['exact_gate_passed'])+
        '\nCross-host bitwise coordinates: '+str(cross_exact)+'\nOriginal Workstation preserved: '+str(workstation_preserved)+
        '\nFirst differing checkpoint: '+str(result['first_differing_checkpoint'])+'\n\nNo larger-cohort execution, label or capacity validation.\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('request');p.add_argument('output');a=p.parse_args()
    r=compare(a.request,a.output);print(json.dumps({k:r[k]for k in ('status','exact_gate_passed','cross_host_bitwise_identical','workstation_native_bitwise_preserved')}))


if __name__=='__main__':main()
