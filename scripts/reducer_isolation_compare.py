"""Compare closed optimizer-isolation trials and frozen native references; no fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def modules():
    sys.path.insert(0, str(Path(__file__).absolute().parent))
    import reducer_trace_compare as saved
    import pilot_compare as geometry
    import mimas_cpu as cpu
    return saved, geometry, cpu


def semantic_event(event):
    result = {k: v for k, v in event.items() if k != 'artifact'}
    if event['name'].endswith('_initialization_source'):
        result['values'] = {k: result['values'][k] for k in ('bytes', 'sha256')}
    return result


def compare(request_path, output):
    saved, geometry, cpu = modules()
    request = json.loads(Path(request_path).read_text())
    if request.get('operation') != 'closed-reducer-isolation-comparison':
        raise ValueError('Exact closed isolation comparison request required')
    master = saved.data(request['master'])
    if (master.get('operation') != 'paired-reducer-isolation'
            or master['maximum_projection_fits'] != 6 or master['maximum_hdbscan_extractions'] != 0
            or saved.read(master['analysis_source']) != Path(__file__).absolute()
            or set(master['host_plans']) != {'pax', 'mimas'}
            or set(request['runs']) != {'pax', 'mimas'}):
        raise ValueError('Wrong isolation scope or comparator')
    stages = {}; reference = {}; accounting = {}; identity = None
    def load(host, entry, spec, isolation):
        nonlocal identity
        plan = saved.data(entry); locations = saved.data(spec['locations'])
        completion = saved.data(spec['completion'], locations)
        wanted = {'umap1', 'tsne1'} if isolation else {'umap1', 'tsne1', 'umap2', 'tsne2'}
        if (completion.get('status') != 'diagnostic_pilot_complete'
                or completion['plan_sha256'] != entry['sha256']
                or set(completion['stage_receipts']) != wanted
                or plan.get('phase') != ('isolate64' if isolation else 'repro64')
                or plan['execution_host'] != host or plan['diagnostic_profile'] != 'native'):
            raise ValueError('Unclosed or incompatible host trial')
        result = {}; counts = dict(PCA=0, UMAP=0, TSNE=0, HDBSCAN=0)
        for stage in ('umap1', 'tsne1'):
            method = stage[:-1]; receipt_entry = completion['stage_receipts'][stage]
            receipt = saved.data(receipt_entry, locations)
            geometry._source_proof(receipt, plan)
            if (receipt.get('diagnostic_adapter') not in plan['frozen_files']
                    or receipt['portability_helper'] != plan['portability_helper']
                    or receipt['inputs'] != plan['inputs'] or receipt['stage'] != stage
                    or not receipt['phase_complete'] or receipt['diagnostic_profile'] != 'native'):
                raise ValueError('Receipt/source/helper/profile differs')
            names = receipt['models']
            current = dict(target=receipt['target'], models=names, flags=receipt['model_flags'],
                features=receipt['inputs']['feature_sha256'], mask=receipt['inputs']['pair_mask_sha256'])
            if identity is None: identity = current
            if (identity != current or len(names) != 64 or len(set(names)) != 64
                    or hashlib.sha256('\n'.join(names).encode()).hexdigest() != plan['inputs']['model_order_sha256']):
                raise ValueError('Scientific cohort/order/flags differs')
            fits = dict(PCA=int(method == 'tsne'), UMAP=int(method == 'umap'), TSNE=int(method == 'tsne'), HDBSCAN=0)
            params = (dict(n_neighbors=15, min_dist=.1, n_components=2, verbose=False, random_seed=42)
                      if method == 'umap' else dict(perplexity=30., implementation='python', verbose=False, random_seed=42))
            trace = saved.data(receipt['trace'], locations)
            if (receipt['scientific_fits'] != fits or receipt['parameters'] != params
                    or trace['status'] != 'complete' or trace['sources'] != plan['trace_sources']
                    or trace['counts'] != dict(umap=int(method == 'umap'), tsne=int(method == 'tsne'))):
                raise ValueError('Changed recipe, trace or fit count')
            for artifact in receipt['outputs']: saved.read(artifact, locations)
            for event in trace['events']:
                if 'artifact' in event:
                    if event['artifact'] not in receipt['outputs']: raise ValueError('Unbound trace artifact')
                    saved.read(event['artifact'], locations)
            events = [semantic_event(event) for event in trace['events']]
            if isolation:
                if (receipt['isolation'] != plan['isolation'] or trace.get('optimizer_epoch_trace') is not True
                        or trace.get('initialization_replays') != int(method == 'tsne')
                        or trace.get('tsne_initialization') != (plan['isolation']['tsne_initialization'] if method == 'tsne' else None)):
                    raise ValueError('Isolation intervention proof differs')
                nameset = {event['name'] for event in events}
                if method == 'umap':
                    if not {'umap1_optimizer_scalars', 'umap1_epoch0_scalars', 'umap1_epoch1_scalars',
                            'umap1_epoch0_after_head_embedding', 'umap1_epoch1_after_head_embedding'} <= nameset or trace.get('epoch_calls', 0) < 2:
                        raise ValueError('Extended epoch evidence missing')
                else:
                    init = plan['isolation']['tsne_initialization']
                    initial = [event for event in trace['events'] if event['name'] == 'tsne1_initial']
                    if (len(initial) != 1 or initial[0]['artifact']['sha256'] != init['sha256']
                            or initial[0]['artifact']['bytes'] != init['bytes']):
                        raise ValueError('Shared initialization was not actually supplied')
                for name, n in fits.items(): counts[name] += n
            result[stage] = dict(events=events, coordinates=saved.projection(receipt['projection'], locations), receipt=receipt_entry)
        if isolation:
            if counts != dict(PCA=1, UMAP=1, TSNE=1, HDBSCAN=0): raise ValueError('Wrong isolation counts')
            accounting[host] = counts
        return result
    for host in ('pax', 'mimas'):
        stages[host] = load(host, master['host_plans'][host], request['runs'][host], True)
        proof = master['reference_runs'][host]
        reference[host] = load(host, proof['plan'], proof, False)
    rows = []
    def pair(name, left, right):
        le = {event['name']: event for event in left['events']}
        re = {event['name']: event for event in right['events']}
        names = [event['name'] for event in left['events'] if event['name'] in re]
        differences = [n for n in names if le[n] != re[n]]
        effective = [n for n in differences if not n.endswith(('_native_initial', '_initialization_source'))]
        g = geometry.geometry(left['coordinates'], right['coordinates'], geometry.DEFAULT_THRESHOLDS)
        passed = not g['degenerate'] and (g['raw_allclose'] or g['rigid_allclose'])
        rows.append(dict(name=name, first_differing_recorded_checkpoint=differences[0] if differences else None,
            differing_checkpoints=differences, compared_checkpoints=names,
            first_differing_effective_checkpoint=effective[0] if effective else None,
            effective_checkpoint_differences=effective,
            coordinate_gate_passed=passed, geometry=g))
    for stage in ('umap1', 'tsne1'):
        if [e['name'] for e in stages['pax'][stage]['events']] != [e['name'] for e in stages['mimas'][stage]['events']]:
            raise ValueError('Different checkpoint interfaces')
        pair('cross-host-' + stage, stages['pax'][stage], stages['mimas'][stage])
        for host in ('pax', 'mimas'): pair(host + '-new-vs-native-' + stage, stages[host][stage], reference[host][stage])
    output = Path(output)
    if not output.is_absolute() or output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Fresh comparison output required')
    output.mkdir(parents=True)
    result = dict(status='closed_isolation_comparison_complete', master=request['master'],
        actual_fit_accounting=accounting, new_comparison_fits=0, pairs=rows,
        thresholds=geometry.DEFAULT_THRESHOLDS, cluster_stability_validated=False, full_cohort_capacity_validated=False)
    cpu.save(output/'comparison.json', result)
    lines = ['# Saved optimizer isolation comparison', '', 'No new fits; no label or capacity validation.', '',
             '| Pair | First different recorded checkpoint | Coordinate gate |', '|---|---|---|']
    for row in rows:
        lines.append('| ' + row['name'] + ' | ' + str(row['first_differing_recorded_checkpoint']) + ' | ' + str(row['coordinate_gate_passed']) + ' |')
    (output/'comparison.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request'); parser.add_argument('output'); args = parser.parse_args()
    result = compare(args.request, args.output)
    print(json.dumps(dict(status=result['status'], pairs=len(result['pairs']))))


if __name__ == '__main__': main()
