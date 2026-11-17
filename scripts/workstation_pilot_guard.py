"""Guard the approved Workstation pilot/setup body before any environment creation.

The surrounding systemd user unit owns CPU/RAM/swap/PID limits. This small
wrapper verifies them, then uses the separately hash-bound b561 pressure guard.
It never installs, fits, submits a unit or changes a kernel/host control itself.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import time


GUARD_COMMIT = 'b561753af4a06166ea38e9eb9332f3b0a808c3b0'
GUARD_SHA256 = 'be60bd2a2a3fec27635e8449b283176e9dca7f8f3aeebf9f4117a509422bef44'


def wait_for_quiet(plan, guard, admission, *, clock=time.monotonic, sleep=time.sleep):
    """Wait before the first workload launch; never retry a started/refused fit."""
    spec = plan.get('admission_wait')
    if spec is None: return True
    need(plan['phase'] == 'curve1000' and set(spec) == {'directory','maximum_seconds','sample_seconds','quiet_samples'},
         'Only the frozen larger-pilot admission wait is supported')
    need(type(spec['maximum_seconds']) is int and 1 <= spec['maximum_seconds'] <= 1800
         and type(spec['sample_seconds']) is int and 1 <= spec['sample_seconds'] <= 60
         and type(spec['quiet_samples']) is int and 2 <= spec['quiet_samples'] <= 12
         and spec['sample_seconds'] * spec['quiet_samples'] <= spec['maximum_seconds'],
         'Finite admission-wait bounds required')
    cpu, _ = _modules(); root = _path(spec['directory'])
    for other in (plan['run_root'],plan['guard_output'],plan['runtime_prefix']):
        other = _path(other)
        need(not root.is_relative_to(other) and not other.is_relative_to(root), 'Separate wait evidence required')
    need(root.parent == _path(plan['guard_output']).parent, 'Wait and guard evidence must share the owned control parent')
    need(not root.exists(), 'Fresh wait evidence required; no retry')
    root.mkdir(parents=False); start = clock(); quiet = 0; previous = guard.linux_snapshot()
    with (root/'samples.jsonl').open('x') as stream:
        while clock() - start < spec['maximum_seconds']:
            remaining = spec['maximum_seconds']-(clock()-start)
            if remaining <= 0: break
            sleep(min(spec['sample_seconds'],remaining))
            current = guard.linux_snapshot(); reasons = guard.violations(current,previous,admission)
            quiet = quiet + 1 if not reasons else 0
            import json
            stream.write(json.dumps(dict(host=current,reasons=reasons,consecutive_quiet=quiet,projection_fits=0))+'\n');stream.flush()
            previous = current
            if quiet >= spec['quiet_samples']:
                cpu.save(root/'result.json',dict(status='quiet_before_first_launch',elapsed_seconds=clock()-start,projection_fits=0))
                return True
    cpu.save(root/'result.json',dict(status='admission_wait_timeout_no_launch',elapsed_seconds=clock()-start,projection_fits=0))
    return False


def need(value, message):
    if not value:
        raise ValueError(message)


def _modules():
    source = str(Path(__file__).absolute().parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    import hpc_cpu
    import hpc_contract
    return hpc_cpu, hpc_contract


def _path(value):
    path = Path(value)
    need(path.is_absolute() and '..' not in path.parts
         and not any(part.is_symlink() for part in [path, *path.parents]),
         'Exact non-symlink owned path required')
    return path


def _guard_binding(plan, cpu):
    source = plan['sources']['guard']
    entry = plan['runtime_guard']
    need(source.get('commit') == GUARD_COMMIT and entry.get('sha256') == GUARD_SHA256,
         'Exactly the reviewed b561 pressure guard is required')
    expected = _path(source['checkout']) / 'structure_clustering/resource_guard.py'
    need(_path(entry['path']) == expected and entry in plan['frozen_files'],
         'Pressure guard path/content is not frozen by the plan')
    return cpu.bound(entry)


def _load_guard(path):
    # Dataclasses need the private module present during execution. Do not bind
    # or replace the scientific library namespace or reuse an arbitrary module.
    name = '_paired_workstation_pressure_guard_' + GUARD_SHA256
    need(name not in sys.modules, 'Private guard namespace already exists')
    specification = importlib.util.spec_from_file_location(name, path)
    need(specification is not None and specification.loader is not None,
         'Guard module loader unavailable')
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@contextmanager
def _child_environment(plan, contract):
    original = dict(os.environ)
    bootstrap = dict(plan, runtime_python=plan['bootstrap']['python'],
                     runtime_prefix=plan['bootstrap']['prefix'])
    clean = contract.sanitized_environment(bootstrap)
    try:
        os.environ.clear()
        os.environ.update(clean)
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def run(plan_path):
    cpu, contract = _modules()
    plan = cpu.verify(plan_path)
    need(plan.get('execution_host') == 'workstation' and plan.get('phase') in ('smoke', 'pilot64', 'repro64', 'isolate64', 'curve64','curve1000'),
         'Only the approved Workstation setup or paired reducer pilot is supported')
    need(not plan.get('containment'), 'Workstation containment belongs to its systemd unit, not a Slurm scope')
    bootstrap = plan['bootstrap']
    cpu.bound(bootstrap['binary'])
    need(sys.executable == bootstrap['python'] and sys.prefix == bootstrap['prefix'],
         'Exact approved bootstrap interpreter/prefix required')
    own = _path(Path(__file__).absolute())
    frozen = {entry['path']:entry for entry in plan['frozen_files']}
    need(str(own) in frozen, 'Workstation launcher source is not frozen')
    cpu.bound(frozen[str(own)])
    workload = cpu.bound(plan['scoped_workload'])
    need(plan['scoped_workload'] in plan['frozen_files'], 'Workstation bash workload is not frozen')
    root = _path(plan['run_root'])
    runtime = _path(plan['runtime_prefix'])
    output = _path(plan['guard_output'])
    need(not output.exists() and not root.exists(), 'Fresh guard/run destinations required; no retry')
    need(not output.is_relative_to(root) and not root.is_relative_to(output)
         and not output.is_relative_to(runtime) and not runtime.is_relative_to(output),
         'Guard output must be separate from workload/runtime outputs')
    parent = next(path for path in [output.parent, *output.parents] if path.exists())
    need(parent.is_dir() and parent.stat().st_uid == os.getuid(), 'Guard output parent is not owned by this user')
    policy = plan['guard']
    need(set(policy) == {'admission', 'runtime', 'interval_seconds', 'consecutive_failures'},
         'Only the explicit strict pressure policy is supported for this pilot')
    need(type(policy['interval_seconds']) is int and policy['interval_seconds'] >= 1
         and type(policy['consecutive_failures']) is int and policy['consecutive_failures'] >= 1
         and type(plan['resources']['maximum_wall_seconds']) is int
         and plan['resources']['maximum_wall_seconds'] > 0, 'Positive frozen monitoring bounds required')
    bootstrap_plan = dict(plan, runtime_python=bootstrap['python'], runtime_prefix=bootstrap['prefix'])
    observation = contract.observe(bootstrap_plan)
    # Environment paths need not exist yet. Verify the actual job and every
    # kernel resource before importing guard code or executing setup/bash.
    need(observation.get('execution_host') == 'workstation'
         and observation.get('smoke_containment_verified') is True
         and all(observation.get(key, {}).get('verified') is True
                 for key in ('scheduler', 'runtime', 'cpu', 'memory', 'swap', 'pids')),
         'Actual Workstation bootstrap/kernel containment is unverified; no setup')
    guard_path = _guard_binding(plan, cpu)
    guard = _load_guard(guard_path)
    admission = guard.ResourceLimits(**policy['admission'])
    runtime_limits = guard.ResourceLimits(**policy['runtime'])
    cpu.check_disk(plan)
    if not wait_for_quiet(plan,guard,admission): return 75
    cpu.check_disk(plan)
    workload_wall = plan.get('workload_wall_seconds',plan['resources']['maximum_wall_seconds'])
    need(type(workload_wall) is int and 1 <= workload_wall <= plan['resources']['maximum_wall_seconds'],
         'Bounded workload wall time required')
    with _child_environment(plan, contract):
        code = guard.run_guarded(['/bin/bash', str(workload)], output,
            admission=admission, runtime=runtime_limits,
            interval_seconds=policy['interval_seconds'],
            maximum_wall_seconds=workload_wall,
            consecutive_failures=policy['consecutive_failures'])
    need(type(code) is int, 'Pressure guard did not return an actual child exit code')
    cpu.save(output / 'launcher.json', dict(schema_version=1, execution_host='workstation',
        plan_sha256=cpu.sha(plan_path), runtime_guard=plan['runtime_guard'],
        bootstrap_observation=observation, workload=plan['scoped_workload'],
        returncode=code, wrapper_scientific_fits=0))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    args = parser.parse_args()
    code = run(args.plan)
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()
