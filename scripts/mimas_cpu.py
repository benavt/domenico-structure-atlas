"""Approved single-task CPU smoke/pilots; never submits or installs anything.

The batch setup is separately frozen and approved. Every numerical stage runs
once in a fresh child, with exact source and input bindings and no retry.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


STAGES = {'probe': (), 'smoke': (), 'pilot64': ('pca', 'umap', 'tsne', 'labels', 'density0'),
          'umap-full': ('umap',), 'repro64': ('umap1', 'tsne1', 'umap2', 'tsne2'), 'isolate64': ('umap1', 'tsne1')}
FIT_LIMITS = {'probe': (0, 0), 'smoke': (0, 0), 'pilot64': (9, 6), 'umap-full': (1, 0), 'repro64': (6, 0), 'isolate64': (3, 0)}
STAGE_LIMITS = {'smoke': (0, 0), 'pca': (1, 0), 'umap': (1, 0),
                'tsne': (2, 0), 'labels': (2, 0), 'density0': (3, 6),
                'umap1': (1, 0), 'tsne1': (2, 0), 'umap2': (1, 0), 'tsne2': (2, 0)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def bound(entry):
    path = Path(entry['path']).absolute()
    if (not path.is_file() or any(p.is_symlink() for p in [path, *path.parents])
            or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']):
        raise ValueError('Bound artifact missing or changed: ' + str(path))
    return path


def record(path):
    path = Path(path).absolute()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def verify(plan_path, *, require_approval=True):
    path = Path(plan_path).absolute()
    plan = json.loads(path.read_text())
    if require_approval:
        approval = json.loads((path.parent / 'approval.json').read_text())
        if approval.get('approved') is not True or approval.get('plan_sha256') != sha(path):
            raise ValueError('Exact approved Mimas CPU validation plan required')
    if (plan.get('operation') != 'mimas-cpu-validation' or plan.get('phase') not in STAGES):
        raise ValueError('Exact approved Mimas CPU validation plan required')
    phase = plan['phase']
    if (plan.get('stages') != list(STAGES[phase])
            or plan.get('maximum_projection_fits') != FIT_LIMITS[phase][0]
            or plan.get('maximum_hdbscan_extractions') != FIT_LIMITS[phase][1]):
        raise ValueError('Phase/stage/fit ceiling differs from reviewed scope')
    for entry in plan['frozen_files']:
        bound(entry)
    own = Path(__file__).absolute()
    entries = {str(Path(e['path']).absolute()): e for e in plan['frozen_files']}
    if str(own) not in entries:
        raise ValueError('Controller is not bound by this plan')
    source = own.parent
    if phase in ('repro64', 'isolate64'):
        if (str(source / 'reducer_diagnostic.py') not in entries
                or plan.get('diagnostic_profile') not in ('native', 'generic-jit-v1')
                or plan.get('create_environment') is not False
                or plan.get('inputs', {}).get('model_count') != 64):
            raise ValueError('Frozen diagnostic adapter/profile and existing runtime required')
        for entry in [plan['portability_helper'], plan['runtime_payload'],
                      *plan['trace_sources'].values(), *plan['dependency_sources']]:
            bound(entry)
            if entry not in plan['frozen_files']:
                raise ValueError('Diagnostic source is not frozen')
    for name in ('mimas_contract.py', 'mimas_smoke.py', 'mimas_pilot.py'):
        if str(source / name) not in entries:
            raise ValueError('Controller dependency is not frozen: ' + name)
    if phase == 'isolate64':
        _modules()  # -I omits the script directory; add only these bound siblings.
        import reducer_diagnostic
        reducer_diagnostic.verify_isolation_plan(plan)
    if plan.get('containment'):
        scope = plan['containment']
        if (scope.get('mode') != 'user_scope'
                or str(source / 'mimas_scope.py') not in entries
                or type(scope.get('scope_max_seconds')) is not int
                or not 1 <= scope['scope_max_seconds'] <= plan['resources']['maximum_wall_seconds']
                or type(scope.get('scope_memory_bytes')) is not int
                or not 0 < scope['scope_memory_bytes'] <= plan['resources']['maximum_memory_bytes']
                or type(scope.get('scope_tasks')) is not int
                or not 0 < scope['scope_tasks'] <= plan['resources']['maximum_pids']):
            raise ValueError('Unsupported private user-scope containment')
        workload = bound(plan['scoped_workload'])
        if str(workload) not in entries:
            raise ValueError('Scoped workload is not frozen')
    root = Path(plan['run_root'])
    if (not root.is_absolute() or '..' in root.parts or
            any(p.is_symlink() for p in [root, *root.parents])):
        raise ValueError('Exact non-symlink run destination required')
    for name in ('runtime_prefix', 'runtime_python'):
        runtime_path = Path(plan[name])
        symlink_mode = name == 'runtime_python' and plan.get('runtime_interpreter') == 'base_symlink'
        checked = list(runtime_path.parents) if symlink_mode else [runtime_path,*runtime_path.parents]
        if (not runtime_path.is_absolute() or '..' in runtime_path.parts
                or any(p.is_symlink() for p in checked)):
            raise ValueError('Exact non-symlink runtime destination required')
        if symlink_mode and (runtime_path.exists() or runtime_path.is_symlink()):
            target = bound(plan['bootstrap']['binary'])
            if not runtime_path.is_symlink() or runtime_path.resolve(strict=True) != target:
                raise ValueError('Runtime link must resolve to the exact bound base interpreter')
    if plan.get('runtime_interpreter', 'copies') not in ('copies', 'base_symlink'):
        raise ValueError('Unsupported interpreter creation mode')
    if not Path(plan['runtime_python']).is_relative_to(Path(plan['runtime_prefix'])):
        raise ValueError('Interpreter must stay in the new runtime prefix')
    if (phase == 'pilot64' and plan['inputs']['model_count'] != 64
            or phase == 'umap-full' and plan['inputs']['model_count'] != 5072):
        raise ValueError('Unexpected pilot population')
    return plan


def _modules():
    # Isolated Python omits the script directory. Only this hash-bound project
    # directory is added; the scientific library must come from its installed wheel.
    source = str(Path(__file__).absolute().parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    import mimas_contract
    return mimas_contract


def check_disk(plan, *, include_sizes=False):
    limits = plan['resources']
    root = Path(plan['run_root'])
    existing = next(p for p in [root, *root.parents] if p.exists())
    if shutil.disk_usage(existing).free < limits.get('minimum_free_disk_bytes', 0):
        raise ValueError('Approved scratch free-space reserve reached')
    if include_sizes and 'maximum_added_disk_bytes' in limits:
        locations = [root, Path(plan['runtime_prefix']),
                     Path(plan['plan_directory']) / 'setup-tmp']
        total = 0
        for location in locations:
            if not location.exists():
                continue
            for directory, _, names in os.walk(location, followlinks=False):
                for name in names:
                    total += (Path(directory) / name).lstat().st_size
        if total > limits['maximum_added_disk_bytes']:
            raise ValueError('Approved setup/runtime/output disk allowance reached')


def _group_alive(process):
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate(process):
    # A dead leader can leave native children alive in its private process group.
    if _group_alive(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        if _group_alive(process):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)


def setup_check(plan_path):
    """Check the existing bootstrap and allocation before any venv/install."""
    plan = verify(plan_path)
    if plan['phase'] != 'smoke' and not (plan['phase']=='pilot64' and plan.get('create_environment') is True):
        raise ValueError('Environment setup is only approved with the smoke phase')
    bootstrap = plan['bootstrap']
    bound(bootstrap['binary'])
    if sys.executable != bootstrap['python'] or sys.prefix != bootstrap['prefix']:
        raise ValueError('Existing bootstrap identity differs')
    contract = _modules()
    contract.sanitized_environment(plan)  # Validate the original new-runtime paths.
    bootstrap_plan = dict(plan, runtime_python=bootstrap['python'],
                          runtime_prefix=bootstrap['prefix'])
    observation = contract.observe(bootstrap_plan)
    save(Path(plan_path).parent / 'setup-resource-observation.json', observation)
    if observation.get('smoke_containment_verified') is not True:
        raise ValueError('Setup blocked: actual Slurm CPU/RAM enforcement is unverified')
    check_disk(plan)
    return observation


def child(plan_path, stage, context_path, context_sha):
    plan = verify(plan_path)
    if (stage == 'smoke' and plan['phase'] != 'smoke'
            or stage != 'smoke' and stage not in STAGES[plan['phase']]):
        raise ValueError('Unapproved child stage')
    contract = _modules()
    if plan['phase'] in ('repro64', 'isolate64'):
        plan = dict(plan, active_stage=stage)
    observation = contract.observe(plan)
    root = Path(plan['run_root'])
    context = Path(context_path).absolute()
    if (context.parent != root or sha(context) != context_sha
            or context.is_symlink()):
        raise ValueError('Owned stage context changed')
    value = json.loads(context.read_text())
    if value['stage'] != stage or value['plan_sha256'] != sha(plan_path):
        raise ValueError('Stage context scope differs')
    if value.get('controller_pid') != os.getppid():
        raise ValueError('Stage belongs to a different live controller')
    if plan['phase'] in ('repro64', 'isolate64'):
        plan = dict(plan, runtime_payload_state=value['runtime_payload_state'])
    intent = json.loads((root / ('intent-' + stage + '.json')).read_text())
    if (intent.get('plan_sha256') != sha(plan_path) or intent.get('stage') != stage
            or intent.get('status') != 'charged_before_launch'
            or (intent.get('projection_fit_ceiling'), intent.get('hdbscan_extraction_ceiling'))
            != STAGE_LIMITS[stage]):
        raise ValueError('Charged native stage intent differs')
    output = root / 'stages' / stage
    if ((stage == 'smoke' and output.exists()) or (stage != 'smoke' and not output.is_dir())
            or any(p.is_symlink() for p in [output, *output.parents])):
        raise ValueError('Owned stage output required')
    save(root / ('resource-observation-' + stage + '.json'), observation)
    if plan['phase'] in ('repro64', 'isolate64'):
        import reducer_diagnostic
        reducer_diagnostic.pre_import(plan)
    if stage == 'smoke':
        import mimas_smoke
        result = mimas_smoke.run_smoke(plan, output)
        if result.get('smoke_checks_passed') is not True:
            raise ValueError('CPU smoke is blocked; inspect its retained receipt')
    else:
        if stage not in STAGES[plan['phase']]:
            raise ValueError('Unapproved stage')
        contract.require_fit_ready(observation)
        plan = dict(plan, phase_outputs=value['phase_outputs'])
        # Reverify the installed wheel/runtime in the actual child rather than
        # trusting a historical smoke receipt or a source checkout on PYTHONPATH.
        import mimas_smoke
        smoke_output = root / 'stage-smoke' / stage
        result = mimas_smoke.run_smoke(plan, smoke_output)
        if result.get('fit_ready') is not True:
            raise ValueError('Installed runtime smoke is not fit-ready')
        if plan['phase'] in ('repro64', 'isolate64'):
            reducer_diagnostic.run_stage(plan, stage, output)
        else:
            import mimas_pilot
            mimas_pilot.run_stage(plan, stage, output)


def run(plan_path, *, popen=subprocess.Popen, clock=time.monotonic, sleep=time.sleep):
    plan = verify(plan_path)
    contract = _modules()
    if plan['phase'] in ('repro64', 'isolate64'):
        import reducer_diagnostic
        reducer_diagnostic.verify_predecessor(plan)
    env = contract.sanitized_environment(plan)
    observation = contract.observe(plan, environ=env)
    check_disk(plan, include_sizes=True)
    if plan['phase'] not in ('probe', 'smoke'):
        contract.require_fit_ready(observation, require_actual_environment=False)
    root = Path(plan['run_root'])
    root.mkdir(parents=False, exist_ok=False)
    save(root / 'owner.json', {'plan_sha256': sha(plan_path), 'uid': os.getuid()})
    save(root / 'resource-observation.json', observation)
    runtime_state = None
    if plan['phase'] in ('repro64', 'isolate64'):
        runtime_state = reducer_diagnostic.verify_runtime_payload(plan, root)
    save(root / 'runtime.json', dict(python=sys.version, executable=sys.executable,
         prefix=sys.prefix, packages={item.metadata['Name']:item.version
         for item in importlib.metadata.distributions()},
         threads={key:env[key] for key in contract.THREAD_VARIABLES}
         if hasattr(contract, 'THREAD_VARIABLES') else {},
         bootstrap=plan.get('bootstrap'), cpp_closure=plan.get('runtime_cpp_closure', [])))
    if plan['phase'] == 'probe':
        save(root / 'completion.json', {'status': 'resource_probe_complete',
             'plan_sha256': sha(plan_path), 'fit_containment_verified': observation.get('fit_containment_verified', False),
             'native_imports': 0, 'projection_fits': 0, 'environment_created': False})
        return root / 'completion.json'
    (root / 'stages').mkdir()
    (root / 'logs').mkdir()
    directories = ('TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'NUMBA_CACHE_DIR',
                   'DATA_PROCESSED', 'DATA_RAW', 'DATA_DB', 'OUTPUTS_GENERATED',
                   'OUTPUTS_RESULTS', 'OUTPUTS_PYMOL', 'CLUSTERING_RESULTS', 'PDB_FILES', 'WORKING_DIR')
    for key in directories:
        if key in env:
            path = Path(env[key])
            if not path.is_relative_to(root) or any(p.is_symlink() for p in [path, *path.parents]):
                raise ValueError('Cache/writer directory escapes this run')
            path.mkdir(parents=True, exist_ok=True)
    phases = ('smoke',) if plan['phase'] == 'smoke' else STAGES[plan['phase']]
    outputs = {};timings={};started = clock()
    try:
        for stage in phases:
            stage_started=clock()
            output = root / 'stages' / stage
            if stage != 'smoke':
                output.mkdir()
            context = root / ('context-' + stage + '.json')
            save(context, {'plan_sha256': sha(plan_path), 'stage': stage,
                           'controller_pid': os.getpid(), 'phase_outputs': outputs,
                           'runtime_payload_state':runtime_state})
            save(root / ('intent-' + stage + '.json'),
                 {'stage': stage, 'plan_sha256': sha(plan_path), 'status': 'charged_before_launch',
                  'projection_fit_ceiling': STAGE_LIMITS[stage][0],
                  'hdbscan_extraction_ceiling': STAGE_LIMITS[stage][1], 'automatic_retry': False})
            flags = contract.python_flags(plan) if plan['phase'] in ('repro64', 'isolate64') else ['-I']
            command = [plan['runtime_python'], *flags, str(Path(__file__).absolute()),
                       'child', str(Path(plan_path).absolute()), stage, str(context), sha(context)]
            with (root / 'logs' / (stage + '.log')).open('xb') as log:
                child_env = (contract.sanitized_environment(dict(plan, active_stage=stage))
                             if plan['phase'] in ('repro64', 'isolate64') else env)
                process = popen(command, env=child_env, cwd=root, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None:
                        if clock() - started > plan['resources']['maximum_wall_seconds']:
                            raise TimeoutError('Approved job wall ceiling reached')
                        current = contract.observe(plan, environ=env)
                        check_disk(plan)
                        if current.get('smoke_containment_verified') is not True:
                            raise ValueError('Actual Slurm CPU/RAM enforcement changed')
                        if plan['phase'] != 'smoke':
                            contract.require_fit_ready(current, require_actual_environment=False)
                        with (root / 'resource-observations.jsonl').open('a') as stream:
                            stream.write(json.dumps(current, allow_nan=False) + '\n')
                        sleep(5)
                    if process.returncode != 0:
                        raise RuntimeError('Native stage failed: ' + stage + '; no retry')
                    if _group_alive(process):
                        raise RuntimeError('Stage leader exited with surviving descendants: ' + stage)
                finally:
                    _terminate(process)
            receipt = output / ('smoke.json' if stage == 'smoke' else 'receipt.json')
            if not receipt.is_file():
                raise ValueError('Stage has no completion receipt: ' + stage)
            result = json.loads(receipt.read_text())
            if stage == 'smoke' and result.get('smoke_checks_passed') is not True:
                raise ValueError('Installed smoke checks did not pass')
            if stage != 'smoke' and result.get('phase_complete') is not True:
                raise ValueError('Scientific phase receipt is incomplete')
            if stage != 'smoke':
                if (result.get('stage') != stage or result.get('inputs') != plan['inputs']
                        or result.get('native_library_commit') != plan['native']['library_commit']):
                    raise ValueError('Scientific phase receipt scope differs')
                counts = result['scientific_fits']
                if plan['phase'] in ('repro64', 'isolate64'):
                    import reducer_diagnostic
                    reducer_diagnostic.validate_receipt(plan, stage, result)
                if (set(counts) != {'PCA', 'UMAP', 'TSNE', 'HDBSCAN'}
                        or any(type(n) is not int or n < 0 for n in counts.values())
                        or sum(counts[k] for k in ('PCA', 'UMAP', 'TSNE')) > STAGE_LIMITS[stage][0]
                        or counts['HDBSCAN'] > STAGE_LIMITS[stage][1]):
                    raise ValueError('Scientific phase exceeded its fit ceiling')
                for artifact in result['outputs']:
                    artifact_path = bound(artifact)
                    if not artifact_path.is_relative_to(root):
                        raise ValueError('Scientific output escapes the owned run')
            outputs[stage] = record(receipt)
            timings[stage]=clock()-stage_started
            save(root / ('resource-final-' + stage + '.json'), contract.observe(plan, environ=env))
            check_disk(plan, include_sizes=True)
        save(root / 'completion.json', {'status': result['status'] if plan['phase'] == 'smoke' else 'diagnostic_pilot_complete',
             'fit_ready': result.get('fit_ready', False) if plan['phase'] == 'smoke' else True,
             'plan_sha256': sha(plan_path), 'phase': plan['phase'], 'stage_receipts': outputs,
             'stage_elapsed_seconds':timings,
             'maximum_projection_fits': plan['maximum_projection_fits'],
             'maximum_hdbscan_extractions': plan['maximum_hdbscan_extractions'],
             'standard_deliverables_complete': False})
    except BaseException as error:
        save(root / 'failure.json', {'status': 'failed', 'plan_sha256': sha(plan_path),
                                   'type': type(error).__name__, 'message': str(error), 'retry': False})
        raise
    return root / 'completion.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'verify', 'setup-check', 'smoke', 'run', 'child'])
    parser.add_argument('plan')
    parser.add_argument('stage', nargs='?')
    parser.add_argument('context', nargs='?')
    parser.add_argument('context_sha', nargs='?')
    args = parser.parse_args()
    if args.action == 'preflight':
        verify(args.plan, require_approval=False)
        print('Frozen bindings and saved-input provenance verified; no execution approval, fit or install.')
    elif args.action == 'verify':
        verify(args.plan)
        print('Exact plan approval and frozen source bindings verified; no fit or install.')
    elif args.action == 'setup-check':
        setup_check(args.plan)
        print('Actual bootstrap/allocation verified before environment creation.')
    elif args.action == 'child':
        child(args.plan, args.stage, args.context, args.context_sha)
    else:
        plan = verify(args.plan)
        if args.action == 'smoke' and plan['phase'] != 'smoke':
            raise ValueError('Smoke command cannot invoke scientific phases')
        print(run(args.plan))


if __name__ == '__main__':
    main()
