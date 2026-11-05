"""Run only the exact approved synthetic UMAP parity/memory plan; no real models."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from snapshot import sha256
from target_continuation import bound, record
from target_projections import save

SEEDS = (42, 7, 123)
PROFILES = {'primary': dict(n_neighbors=15, min_dist=.1, n_components=2),
            'density': dict(n_neighbors=30, min_dist=0., n_components=10)}
SCOPE = dict(operation='synthetic-umap-copy-validation', maximum_umap_fits=14,
             real_models=0, other_projection_fits=0, hdbscan_extractions=0,
             coordinate_parses=0, molecular_exports=0, paid_calls=0,
             narrow_shape=[4096, 64], wide_shape=[4096, 8192], seeds=list(SEEDS))


def need(value, message):
    if not value: raise ValueError(message)


def approved(path):
    plan = json.loads(path.read_text());proof = json.loads((path.parent/'approval.json').read_text())
    need(proof.get('approved_by') == 'user' and proof.get('approved_plan_sha256') == sha256(path), 'Exact synthetic plan approval required')
    need(plan['settings'] == SCOPE and plan['operation'] == SCOPE['operation'], 'Synthetic scope differs')
    from run_reviewed_plan import control_group, verify_controls
    group = control_group();verify_controls(group, plan['resources'])
    need((group/'memory.high').read_text().strip() == str(plan['resources']['memory_high_bytes'])
         and (group/'memory.oom.group').read_text().strip() == '1', 'Synthetic high/OOM group controls differ')
    return plan


def cases():
    return [dict(id=profile+'-'+str(seed), dataset='narrow', profile=profile, seed=seed)
            for profile in PROFILES for seed in SEEDS] + [dict(id='wide-42', dataset='wide', profile='primary', seed=42)]


def dataset(root, name, shape, check):
    import numpy as np
    path = root/(name+'.npy');matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=tuple(shape))
    try:
        rng = np.random.RandomState(20261105)
        rows = max(1, (8*1024**2)//(shape[1]*8))
        for start in range(0, shape[0], rows):
            check();stop = min(start+rows, shape[0])
            block = rng.uniform(0., 1., size=(stop-start, shape[1])).astype(np.float32)
            block += (np.arange(start, stop)//1024).astype(np.float32)[:, None]*4.
            matrix[start:stop] = block
        matrix.flush()
    finally:matrix._mmap.close()
    return record(path)


def one_fit(plan, case, mode, check):
    import numpy as np
    import numba
    import umap
    from umap_memory import fit_transform, random_state_sha256
    root = Path(plan['run_root']);context = json.loads((root/'validation-context.json').read_text())
    need(context['plan_sha256'] == plan['_plan_sha256'] and context['parent_pid'] == os.getppid()
         and context['guard_pgid'] == os.getpgrp(), 'Synthetic child ownership differs')
    entry = context['datasets'][case['dataset']];path = bound(entry)
    output = root/(case['id']+'-'+mode);output.mkdir()
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        need(matrix.dtype == np.float32 and matrix.flags.c_contiguous and not matrix.flags.writeable, 'Exact read-only prepared input required')
        params = dict(PROFILES[case['profile']], n_jobs=1, low_memory=True, verbose=False)
        state = np.random.RandomState(case['seed'])
        estimator = umap.UMAP(**params, random_state=state if mode == 'baseline' else case['seed'])
        evidence = {};numba.set_num_threads(1);start = time.monotonic()
        if mode == 'baseline':
            xy = estimator.fit_transform(matrix)
            final_rng = random_state_sha256(state)
        else:
            xy = fit_transform(estimator, matrix, check=check, evidence=evidence)
            need(evidence['route'] == 'native-precomputed-neighbors' and evidence['status'] == 'complete', 'Candidate must exercise audited native route')
            final_rng = evidence['final_random_state_sha256']
        elapsed = time.monotonic()-start;check()
        graph = estimator.graph_.tocsr(copy=True);graph.sort_indices()
        need(np.asarray(xy).shape == (len(matrix), params['n_components']) and np.isfinite(xy).all(), 'Invalid synthetic embedding')
        search = estimator._knn_search_index
        shares = bool(np.shares_memory(search._raw_data, matrix))
        artifact = output/'arrays.npz'
        np.savez_compressed(artifact, embedding=xy, neighbors=estimator._knn_indices,
            distances=estimator._knn_dists, graph_data=graph.data,
            graph_indices=graph.indices, graph_indptr=graph.indptr)
        outcome = dict(status='complete', case=case, mode=mode, parameters=params, input=entry,
            arrays=record(artifact), final_random_state_sha256=final_rng,
            native_neighbor_input_shares_source=shares, elapsed_fit_seconds=elapsed,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            helper=evidence, fit_attempts=1)
        need(sha256(path) == entry['sha256'], 'Synthetic source bytes changed')
        save(output/'receipt.json', outcome)
        del search,estimator,graph
    finally:matrix._mmap.close()


def compare(root, case):
    import numpy as np
    baseline=json.loads((root/(case['id']+'-baseline')/'receipt.json').read_text())
    candidate=json.loads((root/(case['id']+'-candidate')/'receipt.json').read_text())
    need(baseline['input']==candidate['input'] and baseline['final_random_state_sha256']==candidate['final_random_state_sha256'], 'Source or final RNG stream differs')
    with np.load(bound(baseline['arrays']),allow_pickle=False) as a,np.load(bound(candidate['arrays']),allow_pickle=False) as b:
        need(set(a.files)==set(b.files), 'Validation fields differ')
        exact={key:bool(np.array_equal(a[key],b[key])) for key in a.files}
    need(all(exact.values()), 'Native numerical parity differs: '+str(exact))
    need(not baseline['native_neighbor_input_shares_source'] and candidate['native_neighbor_input_shares_source'], 'Retained native input-copy proof differs')
    return dict(case=case, exact_fields=exact, final_random_state_equal=True,
        baseline_RSS_bytes=baseline['peak_RSS_bytes'],candidate_RSS_bytes=candidate['peak_RSS_bytes'],
        RSS_difference_bytes=baseline['peak_RSS_bytes']-candidate['peak_RSS_bytes'],
        baseline_seconds=baseline['elapsed_fit_seconds'],candidate_seconds=candidate['elapsed_fit_seconds'],
        retained_copy_removed_bytes=candidate['helper']['avoided_retained_copy_bytes'],
        baseline=record(root/(case['id']+'-baseline')/'receipt.json'),
        candidate=record(root/(case['id']+'-candidate')/'receipt.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path)
    parser.add_argument('--case');parser.add_argument('--mode',choices=('baseline','candidate'));args=parser.parse_args()
    path=args.plan.absolute();plan=approved(path);plan['_plan_sha256']=sha256(path)
    root=Path(plan['run_root']);started=time.monotonic()
    def check():
        need(time.monotonic()-started < plan['resources']['maximum_wall_seconds'], 'Synthetic wall cap reached')
        need(shutil.disk_usage(root).free >= plan['resources']['preserved_free_space_bytes'], 'SSD reserve reached')
        need(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= plan['resources']['additional_storage_bytes'], 'Synthetic storage cap reached')
    if args.case:
        need(args.mode is not None,'Explicit synthetic mode required')
        selected=next((c for c in cases() if c['id']==args.case),None);need(selected is not None,'Unknown synthetic case')
        one_fit(plan,selected,args.mode,check);return
    need(args.mode is None and os.getpid()==os.getpgrp(), 'Guard-owned controller required')
    from run_reviewed_plan import reviewed_guard
    from serca_recovery import stage_processes
    client=reviewed_guard(plan).DurableCheckpoint(plan['guard']['io_recovery']['control_directory'],plan['guard']['io_recovery']['run_id'])
    (root/'tmp').mkdir(mode=0o700)
    client.checkpoint('synthetic-before-inputs')
    inputs={name:dataset(root,name,plan['settings'][name+'_shape'],check) for name in ('narrow','wide')}
    save(root/'validation-context.json',dict(datasets=inputs,plan_sha256=sha256(path),parent_pid=os.getpid(),guard_pgid=os.getpgrp()))
    client.checkpoint('synthetic-inputs-closed');results=[];charged=0
    for case in cases():
        for mode in ('baseline','candidate'):
            check();charged+=1;need(charged<=14,'Synthetic fit ceiling reached')
            save(root/(case['id']+'-'+mode+'-intent.json'),dict(charged_fit=charged,case=case,mode=mode,plan_sha256=sha256(path)))
            client.checkpoint('synthetic-before-'+case['id']+'-'+mode)
            baseline=stage_processes()
            process=subprocess.Popen([sys.executable,str(Path(__file__).absolute()),str(path),'--case',case['id'],'--mode',mode],env=dict(os.environ,TMPDIR=str(root/'tmp')))
            try:
                while process.poll() is None:
                    check()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                need(process.returncode==0,'Synthetic child failed; preserve charged fit, no automatic retry')
                need(stage_processes()==baseline,'Synthetic descendants remain')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            client.checkpoint('synthetic-closed-'+case['id']+'-'+mode)
        results.append(compare(root,case));save(root/(case['id']+'-comparison.json'),results[-1])
    save(root/'validation-completion.json',dict(status='synthetic_umap_parity_and_copy_avoidance_validated',
        plan_sha256=sha256(path),fits_charged=charged,results=results,
        actual_peak_RSS_saving_is_reported_not_assumed=True,production_execution_authorized=False,
        real_models=0,molecular_exports=0,paid_calls=0))


if __name__=='__main__':main()
