"""A bounded user scope tied to a live, still-in-Slurm supervisor.

Only this attempt's transient scope is created/stopped. No scheduler, ancestor
control, ownership or host configuration is changed. Importing this module runs
no command and imports no scientific engine.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import sys
import time


SYSTEMD_RUN = '/usr/bin/systemd-run'
SYSTEMCTL = '/usr/bin/systemctl'
CGROUP_ROOT = Path('/sys/fs/cgroup')
PROC_ROOT = Path('/proc')
BINDING_NAME = 'scope-binding.json'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _modules():
    source = str(Path(__file__).absolute().parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    import mimas_cpu
    import mimas_contract
    return mimas_cpu, mimas_contract


def _regular(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts
         and not any(p.is_symlink() for p in [path, *path.parents]),
         'Exact non-symlink path required')
    return path


def _save(path, value):
    # These records are single-writer receipts, never mutable heartbeats.
    with _regular(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _identity(pid, *, proc_root=PROC_ROOT):
    need(type(pid) is int and pid > 1, 'A live process PID is required')
    root = proc_root / str(pid)
    value = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    uid_line = next(line for line in (root / 'status').read_text().splitlines()
                    if line.startswith('Uid:'))
    uids = [int(part) for part in uid_line.split(':', 1)[1].split()]
    need(len(uids) == 4 and len(set(uids)) == 1, 'Unexpected process UID state')
    return dict(pid=pid, ppid=int(value[1]), start_time=value[19], uid=uids[0],
                cgroup=(root / 'cgroup').read_text().strip())


def _scope_settings(plan):
    scope = plan.get('containment', {})
    resources = plan['resources']
    need(plan.get('phase') in ('probe', 'smoke','pilot64','repro64') and scope.get('mode') == 'user_scope',
         'This scope launcher supports only approved probes, smokes or matched64 pilots')
    for name in ('scope_memory_bytes', 'scope_tasks', 'scope_max_seconds'):
        need(type(scope.get(name)) is int and scope[name] > 0,
             'Explicit positive scope budget required: ' + name)
    need(scope['scope_memory_bytes'] == resources['maximum_memory_bytes']
         and scope['scope_tasks'] == resources['maximum_pids']
         and scope['scope_max_seconds'] <= resources['maximum_wall_seconds']
         and resources['maximum_swap_bytes'] == 0,
         'Scope budgets differ from the approved workload limits')
    return scope


def _load(plan_path):
    path = _regular(plan_path)
    cpu, contract = _modules()
    plan = cpu.verify(path)
    _scope_settings(plan)
    cpu.bound(plan['scoped_workload'])
    own = _regular(Path(__file__).absolute())
    frozen = {str(Path(item['path']).absolute()): item for item in plan['frozen_files']}
    need(str(own) in frozen, 'Scope source is not frozen by this plan')
    cpu.bound(frozen[str(own)])
    bootstrap = plan['bootstrap']
    cpu.bound(bootstrap['binary'])
    need(sys.executable == bootstrap['python'] and sys.prefix == bootstrap['prefix'],
         'Scope launcher must use the exact bound bootstrap')
    return path, plan, contract


def _bus_environment(plan, contract):
    bootstrap = dict(plan, runtime_python=plan['bootstrap']['python'],
                     runtime_prefix=plan['bootstrap']['prefix'])
    env = contract.sanitized_environment(bootstrap)
    directory = _regular(Path('/run/user') / str(os.getuid()))
    bus = _regular(directory / 'bus')
    need(directory.is_dir() and directory.stat().st_uid == os.getuid()
         and stat.S_ISSOCK(bus.stat().st_mode) and bus.stat().st_uid == os.getuid(),
         'The current user manager bus is unavailable or foreign')
    env['XDG_RUNTIME_DIR'] = str(directory)
    env['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=' + str(bus)
    return env


def _binding(plan_path, plan, contract):
    text = os.environ.get('SLURM_JOB_ID', '')
    need(text.isdecimal() and int(text) > 0, 'A live Slurm allocation is required')
    job = int(text)
    identity = _identity(os.getpid())
    expected = '/system.slice/slurmstepd.scope/job_' + str(job) + '/step_batch/user/task_0'
    need(identity['uid'] == os.getuid() and identity['cgroup'] == '0::' + expected,
         'Scope supervisor must remain in this exact Slurm batch task')
    bootstrap = dict(plan, runtime_python=plan['bootstrap']['python'],
                     runtime_prefix=plan['bootstrap']['prefix'])
    # This process intentionally remains in Slurm. Scope-aware observation is
    # used only after migration, in the keeper/workload.
    bootstrap.pop('containment', None)
    observation = contract.observe(bootstrap)
    need(all(observation.get(key, {}).get('verified') is True
             for key in ('slurm', 'runtime', 'cpu')),
         'The Slurm allocation, bootstrap or CPU budget is unverified')
    allocation_file = _regular(CGROUP_ROOT / expected.lstrip('/') / 'cpuset.cpus.effective')
    allocation = contract._cpus(allocation_file.read_text().strip())
    need(allocation, 'The original Slurm cpuset is unavailable')
    cpus = sorted(allocation)
    need(set(os.sched_getaffinity(0)) <= allocation,
         'Supervisor CPU affinity escaped the Slurm cpuset')
    topology = contract._topology(cpus, Path('/sys/devices/system/cpu'))
    need(contract._cpuset_ready(cpus, plan['resources'], topology),
         'Inherited Slurm CPU affinity exceeds the physical/logical budget')
    plan_sha = digest(plan_path)
    unit = 'domenico-cpu-' + str(job) + '-' + plan_sha[:8] + '.scope'
    scope = '/user.slice/user-{0}.slice/user@{0}.service/app.slice/{1}'.format(os.getuid(), unit)
    return dict(schema=1, plan_sha256=plan_sha, supervisor_pid=os.getpid(),
                supervisor_start_time=identity['start_time'], slurm_cgroup=expected,
                cpus=cpus, unit=unit, scope_cgroup=scope, uid=os.getuid(), job_id=job)


def _read_binding(plan_path, plan, binding_path, binding_sha):
    path = _regular(binding_path)
    need(path == plan_path.parent / BINDING_NAME and digest(path) == binding_sha,
         'Scope binding path/hash differs')
    binding = json.loads(path.read_text())
    uid = os.getuid()
    job = binding.get('job_id')
    need(type(job) is int and job > 0 and binding.get('schema') == 1
         and binding.get('uid') == uid and binding.get('plan_sha256') == digest(plan_path),
         'Scope binding identity differs')
    unit = 'domenico-cpu-' + str(job) + '-' + binding['plan_sha256'][:8] + '.scope'
    expected = '/user.slice/user-{0}.slice/user@{0}.service/app.slice/{1}'.format(uid, unit)
    cpus = binding.get('cpus')
    need(binding.get('unit') == unit and binding.get('scope_cgroup') == expected
         and binding.get('slurm_cgroup') == '/system.slice/slurmstepd.scope/job_' + str(job) + '/step_batch/user/task_0'
         and isinstance(cpus, list) and cpus and cpus == sorted(set(cpus))
         and all(type(cpu) is int and cpu >= 0 for cpu in cpus),
         'Foreign/shared scope or malformed CPU binding')
    need(str(job) == os.environ.get('SLURM_JOB_ID'), 'Scope scheduler environment differs')
    return binding


def _supervisor(binding, *, require_ancestor=False, proc_root=PROC_ROOT):
    identity = _identity(binding['supervisor_pid'], proc_root=proc_root)
    need(identity['uid'] == binding['uid']
         and identity['start_time'] == binding['supervisor_start_time']
         and identity['cgroup'] == '0::' + binding['slurm_cgroup'],
         'Live Slurm supervisor identity/cgroup changed')
    if require_ancestor:
        current = os.getppid()
        seen = set()
        while current != binding['supervisor_pid']:
            need(current > 1 and current not in seen, 'Keeper is not a descendant of the Slurm supervisor')
            seen.add(current)
            current = _identity(current, proc_root=proc_root)['ppid']
    return identity


def _scope_command(plan_path, plan, binding, binding_path, binding_sha):
    scope = _scope_settings(plan)
    properties = ['MemoryMax=' + str(scope['scope_memory_bytes']), 'MemorySwapMax=0',
                  'TasksMax=' + str(scope['scope_tasks']),
                  'RuntimeMaxSec=' + str(scope['scope_max_seconds']),
                  'TimeoutStopSec=5', 'KillMode=control-group', 'SendSIGKILL=yes']
    command = [SYSTEMD_RUN, '--user', '--scope', '--collect', '--no-ask-password',
               '--slice=app.slice', '--unit=' + binding['unit']]
    for value in properties:
        command.extend(['--property=' + value])
    return command + [plan['bootstrap']['python'], '-I', str(Path(__file__).absolute()),
                      'keeper', str(plan_path), str(binding_path), binding_sha]


def _scope_leaf(binding, *, cgroup_root=CGROUP_ROOT):
    path = _regular(cgroup_root / binding['scope_cgroup'].lstrip('/'))
    need(path.is_dir() and path.stat().st_uid == binding['uid'], 'Owned scope cgroup is missing/foreign')
    return path


def _verify_caps(binding, plan, *, cgroup_root=CGROUP_ROOT, configure_oom=False):
    own = _identity(os.getpid())
    need(own['uid'] == binding['uid'] and own['cgroup'] == '0::' + binding['scope_cgroup'],
         'Keeper is outside the exact private scope')
    leaf = _scope_leaf(binding, cgroup_root=cgroup_root)
    settings = _scope_settings(plan)
    expected = {'memory.max': settings['scope_memory_bytes'], 'memory.swap.max': 0,
                'pids.max': settings['scope_tasks']}
    for name, value in expected.items():
        need(_regular(leaf / name).read_text().strip() == str(value),
             'Kernel scope cap differs: ' + name)
    oom = _regular(leaf / 'memory.oom.group')
    if configure_oom:
        # v252 has no MemoryOOMGroup scope property. This one private control is
        # set only after exact membership, ownership and all cap readbacks pass.
        with oom.open('w') as stream:
            stream.write('1')
    need(oom.read_text().strip() == '1', 'Scope-wide OOM containment is unavailable')
    need(_regular(leaf / 'memory.swap.current').read_text().strip() == '0', 'Scope has swapped')
    need(sorted(os.sched_getaffinity(0)) == binding['cpus'], 'Scope CPU affinity changed')
    return leaf


def _stop_scope(binding, env, *, run=subprocess.run, self_stop=False):
    # Never stop the user manager, a slice, an ancestor, or an arbitrary name.
    expected = '/user.slice/user-{0}.slice/user@{0}.service/app.slice/{1}'.format(binding['uid'], binding['unit'])
    need(binding['scope_cgroup'] == expected and binding['unit'].startswith('domenico-cpu-')
         and binding['unit'].endswith('.scope'), 'Refusing to stop a shared/foreign unit')
    result = run([SYSTEMCTL, '--user', '--no-ask-password', 'show', binding['unit'],
                  '--property=LoadState', '--property=ActiveState', '--property=ControlGroup'], env=env,
                 capture_output=True, text=True, timeout=10, check=False)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if fields.get('LoadState') == 'not-found':
        return
    if (result.returncode == 0 and fields.get('ActiveState') in ('inactive', 'failed')
            and not fields.get('ControlGroup')):
        return
    need(result.returncode == 0 and fields.get('ControlGroup') == expected,
         'Cannot prove the exact unit to clean up')
    result = run([SYSTEMCTL, '--user', '--no-ask-password', 'stop', binding['unit']],
                 env=env, capture_output=True, text=True, timeout=15, check=False)
    # A keeper's systemctl child is itself inside the scope being stopped.
    # SIGTERM/SIGKILL here is expected; the outside supervisor still verifies
    # cleanup independently. Other errors retain their ordinary failure status.
    need(result.returncode == 0 or self_stop and result.returncode in (-signal.SIGTERM, -signal.SIGKILL),
         'Exact private scope did not stop')


def _pidfd_open(pid):
    """Open a kernel PID handle even when this Python omitted os.pidfd_open.

    The final syscall route is deliberately restricted to the reviewed Linux
    x86_64 target (pidfd_open syscall 434); other missing APIs fail closed.
    """
    native = getattr(os, 'pidfd_open', None)
    if callable(native):
        return native(pid, 0)
    library = ctypes.CDLL(None, use_errno=True)
    native = getattr(library, 'pidfd_open', None)
    if native is not None:
        native.argtypes = [ctypes.c_int, ctypes.c_uint]
        native.restype = ctypes.c_int
        handle = native(pid, 0)
    else:
        need(sys.platform == 'linux' and os.uname().machine == 'x86_64',
             'A supported kernel pidfd API is required')
        library.syscall.restype = ctypes.c_long
        handle = library.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if handle < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return handle


def _arm_parent(binding):
    _supervisor(binding, require_ancestor=True)
    immediate = os.getppid()
    handle = _pidfd_open(binding['supervisor_pid'])
    try:
        library = ctypes.CDLL(None, use_errno=True)
        need(library.prctl(1, signal.SIGTERM, 0, 0, 0) == 0,
             'Parent-death signal could not be armed')
        need(os.getppid() == immediate, 'Launcher parent exited while arming cancellation')
        _supervisor(binding, require_ancestor=True)
        poller = select.poll()
        poller.register(handle, select.POLLIN | select.POLLHUP | select.POLLERR)
        need(not poller.poll(0), 'Slurm supervisor already exited')
        return handle, poller
    except BaseException:
        os.close(handle)
        raise


def _monitor(process, *, parent_poll=None, check=lambda: None, clock=time.monotonic,
             sleep=time.sleep, seconds, cancelled=lambda: False):
    start = clock()
    while True:
        if cancelled() or parent_poll is not None and parent_poll.poll(0):
            raise RuntimeError('Slurm supervisor exited or cancellation was requested')
        check()
        value = process.poll()
        if value is not None:
            return value
        if clock() - start >= seconds:
            raise TimeoutError('Approved scope wall time reached')
        sleep(1)


def launch(plan_path):
    path, plan, contract = _load(plan_path)
    binding = _binding(path, plan, contract)
    binding_path = path.parent / BINDING_NAME
    _save(binding_path, binding)
    binding_sha = digest(binding_path)
    env = _bus_environment(plan, contract)
    before = subprocess.run([SYSTEMCTL, '--user', '--no-ask-password', 'show', binding['unit'],
                             '--property=LoadState'], env=env, capture_output=True,
                            text=True, timeout=10, check=False)
    need(before.stdout.strip() == 'LoadState=not-found', 'The exact attempt scope already exists')
    cancelled = []
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[signum] = signal.signal(signum, lambda *_: cancelled.append(True))
    try:
        command = _scope_command(path, plan, binding, binding_path, binding_sha)
        process = subprocess.Popen(command, env=env)
        result = _monitor(process, seconds=plan['containment']['scope_max_seconds'] + 10,
                          check=lambda: _supervisor(binding), cancelled=lambda: bool(cancelled))
        # Scope cleanup may signal the keeper itself. The immutable receipt is
        # the actual bash wait status, not systemd's scope Result field.
        exit_path = path.parent / 'scope-workload-exit.json'
        if not exit_path.exists():
            need(result != 0, 'Scope exited without a workload status receipt')
            return result
        receipt = json.loads(_regular(exit_path).read_text())
        need(receipt.get('plan_sha256') == binding['plan_sha256']
             and receipt.get('binding_sha256') == binding_sha
             and type(receipt.get('workload_returncode')) is int,
             'Scope workload status receipt differs')
        return receipt['workload_returncode']
    finally:
        try:
            _stop_scope(binding, env)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def keeper(plan_path, binding_path, binding_sha):
    path, plan, contract = _load(plan_path)
    binding = _read_binding(path, plan, binding_path, binding_sha)
    env = _bus_environment(plan, contract)
    cancelled = []
    previous = {}
    handle = None
    process = None
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[signum] = signal.signal(signum, lambda *_: cancelled.append(True))
    try:
        handle, poller = _arm_parent(binding)
        os.sched_setaffinity(0, binding['cpus'])
        _verify_caps(binding, plan, configure_oom=True)
        workload = _modules()[0].bound(plan['scoped_workload'])
        env['MIMAS_SCOPE_BINDING'] = str(binding_path)
        env['MIMAS_SCOPE_BINDING_SHA256'] = binding_sha
        _save(path.parent / 'scope-ready.json', dict(plan_sha256=digest(path),
              binding_sha256=binding_sha, keeper_pid=os.getpid(), scope=binding['scope_cgroup'],
              memory_oom_group=1, scientific_fits=0))
        process = subprocess.Popen(['/bin/bash', str(workload)], env=env)
        code = _monitor(process, parent_poll=poller,
                        seconds=plan['containment']['scope_max_seconds'],
                        check=lambda: (_supervisor(binding), _verify_caps(binding, plan)),
                        cancelled=lambda: bool(cancelled))
        _save(path.parent / 'scope-workload-exit.json', dict(plan_sha256=digest(path),
              binding_sha256=binding_sha, workload_returncode=code, scientific_fits=0))
        return code
    finally:
        # A process exit does not make a scope successful/empty. Manager stop
        # kills all remaining members, including native grandchildren. This call
        # may terminate the keeper itself; the still-in-Slurm supervisor also
        # performs the same exact-scope cleanup when systemd-run returns.
        try:
            _stop_scope(binding, env, self_stop=True)
        finally:
            if handle is not None:
                os.close(handle)
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    launch_parser = sub.add_parser('launch')
    launch_parser.add_argument('plan')
    keeper_parser = sub.add_parser('keeper')
    keeper_parser.add_argument('plan')
    keeper_parser.add_argument('binding')
    keeper_parser.add_argument('binding_sha256')
    args = parser.parse_args()
    code = (launch(args.plan) if args.action == 'launch'
            else keeper(args.plan, args.binding, args.binding_sha256))
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()
