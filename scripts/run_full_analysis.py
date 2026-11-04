"""One protected three-target default run, gated by native fixture validation."""
import argparse
import base64
import hashlib
import json
import os
import shutil
import signal
import threading
import time
from pathlib import Path
import shlex
import subprocess
import sys

from snapshot import sha256


def execute(command, **kwargs):
    print('Running: '+shlex.join(list(map(str, command))), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def validate(plan, root):
    import numpy as np
    library = Path(plan['sources']['library']['checkout'])
    output = root/'native-validation'; output.mkdir()
    results = {}
    for method in ['arrays', 'umap', 'tsne']:
        for mode in ['native', 'workspace']:
            folder = output/(method+'-'+mode)
            command = [sys.executable, library/'scripts/benchmark_projection_workspace.py', '--method', method, '--output', folder]
            if mode == 'workspace': command += ['--workspace']
            execute(command)
            results[(method, mode)] = json.loads((folder/'result.json').read_text())
        baseline, candidate = (results[(method, mode)] for mode in ['native', 'workspace'])
        if baseline['source_sha256'] != candidate['source_sha256']:
            raise ValueError('Synthetic inputs differ')
        if method == 'arrays':
            if any(baseline[k] != candidate[k] for k in ['feature_sha256', 'finite', 'cast_sha256']):
                raise ValueError('Feature workspace numerical parity failed')
        else:
            left = np.load(output/(method+'-native')/'embedding.npy', allow_pickle=False)
            right = np.load(output/(method+'-workspace')/'embedding.npy', allow_pickle=False)
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)
    execute([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/test_residue_correspondence.py::test_real_pymol_native_numbering_survives_corresponded_states',
        '--basetemp', output/'pymol-fixture'], cwd=library)
    receipt = {'status': 'validated', 'outer_projection_fits': 4, 'tsne_initialization_pca_fits': 2,
               'native_pymol_fixtures': 1, 'results': {method+'-'+mode: value for (method, mode), value in results.items()}}
    (output/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def collect_confidence(plan, root):
    """Read/hash native metadata in one bounded remote stdlib process."""
    remote = plan['native_confidence']
    consumer = Path(plan['sources']['consumer']['checkout'])
    catalog = Path(remote['catalog'])
    payload = {'root': remote['remote_output'], 'catalog': base64.b64encode(catalog.read_bytes()).decode(),
        'expected_catalog_sha256': remote['catalog_sha256'],
        'extractor': (consumer/'scripts/confidence.py').read_text(),
        'snapshot': (consumer/'scripts/snapshot.py').read_text()}
    wrapper = '''import base64,contextlib,json,os,resource,runpy,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648))
resource.setrlimit(resource.RLIMIT_CPU,(3600,3600))
resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728))
os.nice(19)
p=json.load(sys.stdin);r=Path(p['root']);r.mkdir(parents=True,exist_ok=False)
(r/'catalog.json').write_bytes(base64.b64decode(p['catalog']))
(r/'confidence.py').write_text(p['extractor']);(r/'snapshot.py').write_text(p['snapshot'])
sys.path.insert(0,str(r));sys.argv=['confidence.py','extract','--catalog',str(r/'catalog.json'),'--catalog-sha256',p['expected_catalog_sha256'],'--output',str(r/'confidence.json')]
with contextlib.redirect_stdout(sys.stderr):runpy.run_path(str(r/'confidence.py'),run_name='__main__')
sys.stdout.buffer.write((r/'confidence.json').read_bytes())
'''
    command = remote['ssh_command'] + ['python3 -c '+shlex.quote(wrapper)]
    try:
        result = execute(command, input=json.dumps(payload).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3660)
    except subprocess.CalledProcessError as error:
        (root/'confidence-extraction.log').write_bytes(error.stderr or b'')
        raise
    (root/'confidence-extraction.log').write_bytes(result.stderr)
    if len(result.stdout) > 134217728:
        raise ValueError('Remote confidence output exceeded the approved cap')
    data = json.loads(result.stdout)
    if data['status'] != 'validated' or data['catalog_sha256'] != remote['catalog_sha256']:
        raise ValueError('Native confidence extraction is incomplete')
    output = root/'native-confidence.json'
    with output.open('xb') as stream: stream.write(result.stdout)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('plan', type=Path)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    approval = json.loads((args.plan.parent/'approval.json').read_text())
    if approval.get('approved_plan_sha256') != sha256(args.plan) or approval.get('approved_by') != 'user':
        raise ValueError('Explicit approval of this exact plan is required')
    root = Path(plan['run_root'])
    if os.getpgrp() != os.getpid():
        raise ValueError('Analysis driver requires the guard-owned private process group')
    def storage_guard():
        floor = plan['resources']['preserved_free_space_bytes']
        while True:
            if shutil.disk_usage(root).free < floor:
                print('Storage reserve breached; stopping only this analysis process group.', flush=True)
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return
            time.sleep(5)
    threading.Thread(target=storage_guard, daemon=True).start()
    validate(plan, root)
    confidence = collect_confidence(plan, root)
    import confidence as native_confidence
    from workflow import load_intake
    completion = {'status': 'running', 'targets': {}, 'native_confidence_sha256': sha256(confidence),
                  'execution_plan_sha256': sha256(args.plan)}
    for target in plan['targets']:
        folder = root/target['identifier']; folder.mkdir()
        db = folder/'seed-metrics.sqlite'
        native_confidence.seed(target['intake'], confidence, sha256(confidence), plan['native_manifest'],
                               plan['model_types'], db)
        intake = db.with_suffix('.intake.json')
        load_intake(intake)
        execute([sys.executable, Path(__file__).with_name('workflow.py'), 'analyze', '--intake', intake,
                 '--screening', target['screening'], '--processed-root', folder/'processed'])
        manifests = list((folder/'processed').glob('**/manifest.json'))
        if len(manifests) != 1:
            raise ValueError('Expected one immutable workflow manifest')
        manifest = json.loads(manifests[0].read_text())
        if manifest['analysis_status'] != 'complete' or not manifest.get('cluster_ensembles'):
            raise ValueError('Default workflow analysis/ensemble preparation is incomplete')
        completion['targets'][target['identifier']] = {'manifest': str(manifests[0]), 'manifest_sha256': sha256(manifests[0]),
            'analysis_status': manifest['analysis_status'], 'ensemble_status': manifest['cluster_ensembles']['status']}
        (root/'progress.json').write_text(json.dumps(completion, indent=2)+'\n')
    completion['status'] = 'three_target_analysis_and_camera_sessions_ready'
    (root/'analysis-completion.json').write_text(json.dumps(completion, indent=2)+'\n')


if __name__ == '__main__': main()
