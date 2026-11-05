"""Read-only Slurm containment evidence; observations never authorize a fit."""
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'BLIS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS')
WRITER_PATHS = dict(DATA_PROCESSED='data/processed', DATA_RAW='data/raw', DATA_DB='data/db',
    DB_PATH='data/db/metrics.sqlite', OUTPUTS_GENERATED='outputs/generated',
    OUTPUTS_RESULTS='outputs/results', OUTPUTS_PYMOL='outputs/pymol',
    CLUSTERING_RESULTS='outputs/results/clustering', WORKING_DIR='work', PDB_FILES='data/pdb_files',
    TMPDIR='tmp', MPLCONFIGDIR='cache/matplotlib', NUMBA_CACHE_DIR='cache/numba', XDG_CACHE_HOME='cache/xdg')


def _need(value, message):
    if not value:
        raise ValueError(message)


def _absolute(value):
    _need(isinstance(value, (str, Path)) and '\x00' not in str(value), 'An absolute path is required')
    path = Path(value)
    _need(path.is_absolute() and '..' not in path.parts, 'Path traversal or relative path is forbidden')
    return path


def _unlinked(path):
    _need(not any(p.is_symlink() for p in [path, *path.parents]), 'Symlinked owned/controller path is forbidden')
    return path


def _plan(plan):
    root = _unlinked(_absolute(plan['run_root']))
    _need(len(root.parts) >= 3, 'A dedicated run directory is required')
    if root.exists():
        _need(root.is_dir() and root.stat().st_uid == os.getuid(), 'Run directory must be owned by the current user')
    python = _absolute(plan['runtime_python']); prefix = _absolute(plan['runtime_prefix'])
    _need(python.is_relative_to(prefix) and python != prefix, 'Interpreter must be in the approved runtime prefix')
    resources = plan['resources']
    for key in ('maximum_cpus', 'maximum_memory_bytes', 'maximum_pids'):
        _need(type(resources[key]) is int and resources[key] > 0, 'Positive integer resource required: ' + key)
    _need(type(resources['maximum_swap_bytes']) is int and resources['maximum_swap_bytes'] >= 0,
          'Explicit nonnegative swap budget required')
    return root, python, prefix, resources


def sanitized_environment(plan, *, environ=None):
    """Return a minimal child environment; create no directories or processes.

    Unknown startup, Python-path, loader and folding hooks are dropped. HOME is
    preserved, never reassigned. All public native writers/caches use this run.
    """
    root, python, prefix, _ = _plan(plan)
    original = dict(os.environ if environ is None else environ)
    _need(all(isinstance(k, str) and isinstance(v, str) and '\x00' not in k + v
              for k, v in original.items()), 'Environment must contain NUL-free strings')
    keep = {'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'SLURM_CONF'}
    result = {k: v for k, v in original.items() if k in keep or k.startswith('SLURM_')}
    result.update(PATH=str(python.parent) + ':/usr/bin:/bin', PYTHONNOUSERSITE='1',
                  PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
                  QT_QPA_PLATFORM='offscreen', STRUCTURE_CLUSTERING_TSNE_IMPL='python',
                  CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
    result.update({key: '1' for key in THREAD_VARIABLES})
    for key, relative in WRITER_PATHS.items():
        result[key] = str(_unlinked(root / relative))
    return result


def _text(path):
    try:
        _unlinked(path)
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


def _positive(text, *, zero=False):
    if text is None:
        return None
    _need(isinstance(text, str) and text.isdecimal(), 'Invalid nonnegative kernel/scheduler integer')
    value = int(text)
    _need(value >= 0 if zero else value > 0, 'Invalid zero resource value')
    return value


def _group_path(value):
    _need(isinstance(value, str) and value.startswith('/') and '\x00' not in value,
          'Invalid cgroup membership path')
    path = PurePosixPath(value)
    _need('..' not in path.parts, 'Cgroup path traversal is forbidden')
    return path


def _unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), value)


def _locations(process, cgroup_root):
    memberships = {}
    for line in (_text(process / 'cgroup') or '').splitlines():
        parts = line.split(':', 2)
        _need(len(parts) == 3, 'Malformed cgroup membership')
        group = _group_path(parts[2])
        for controller in (parts[1].split(',') if parts[1] else ['unified']):
            _need(controller not in memberships, 'Repeated cgroup controller')
            memberships[controller] = group
    locations = {}
    for line in (_text(process / 'mountinfo') or '').splitlines():
        before, separator, after = line.partition(' - ')
        if not separator:
            continue
        fields, tail = before.split(), after.split()
        if len(fields) < 5 or len(tail) < 3 or tail[0] not in ('cgroup', 'cgroup2'):
            continue
        mount = _unlinked(_absolute(_unescape(fields[4])))
        _need(mount.is_relative_to(cgroup_root), 'Cgroup mount escapes the approved controller root')
        mount_root = _group_path(_unescape(fields[3]))
        controllers = ['unified'] if tail[0] == 'cgroup2' else tail[2].split(',')
        for controller in controllers:
            membership = memberships.get(controller)
            if membership is None:
                continue
            # Do not guess a host path when a namespace/mount root is ambiguous.
            _need(membership.is_relative_to(mount_root), 'Cgroup namespace membership does not match mount root')
            leaf = _unlinked(mount / str(membership.relative_to(mount_root)))
            _need(leaf.is_relative_to(mount), 'Cgroup membership escapes its visible mount')
            candidate = dict(leaf=leaf, mount=mount, version=2 if tail[0] == 'cgroup2' else 1)
            previous = locations.get(controller)
            if previous is None or len(mount.parts) > len(previous['mount'].parts):
                locations[controller] = candidate
    return locations


def _ancestors(location):
    if location is None:
        return []
    current = location['leaf']; stop = location['mount']; result = []
    while True:
        result.append(current)
        if current == stop:
            return result
        _need(current.is_relative_to(stop), 'Invalid visible cgroup ancestor')
        current = current.parent


def _controller(locations, controller, unified_file):
    unified = locations.get('unified')
    if any((path / unified_file).is_file() for path in _ancestors(unified)):
        return unified
    return locations.get(controller)


def _limits(location, filename, *, v1_memory=False):
    records = []
    for number, path in enumerate(_ancestors(location)):
        text = _text(path / filename)
        if text is None:
            continue
        applicable = True
        if v1_memory and number:
            applicable = _text(path / 'memory.use_hierarchy') == '1'
        value = None if text == 'max' else _positive(text, zero=True)
        if location['version'] == 1 and filename.startswith('memory.') and value is not None and value >= 2**60:
            value = None  # Kernel v1 unlimited sentinel, never zero swap.
        records.append(dict(path=str(path / filename), raw=text, finite=value,
                            applies_to_process=applicable))
    finite = [r['finite'] for r in records if r['finite'] is not None and r['applies_to_process']]
    return min(finite) if finite else None, records


def _cpus(value):
    if value is None or value == '':
        return None
    result = set()
    for item in value.split(','):
        _need(re.fullmatch(r'\d+(?:-\d+)?', item), 'Invalid CPU set')
        numbers = list(map(int, item.split('-')))
        low, high = (numbers[0], numbers[-1])
        _need(low <= high and high < 1048576, 'Invalid or excessive CPU range')
        result.update(range(low, high + 1))
    return result


def _cpu_evidence(locations, process, maximum):
    cpu = _controller(locations, 'cpu', 'cpu.max')
    unified = cpu is not None and cpu['version'] == 2
    quotas = []
    for path in _ancestors(cpu):
        if unified:
            text = _text(path / 'cpu.max')
            if text is None:
                continue
            values = text.split(); _need(len(values) == 2, 'Invalid CPU quota')
            quota = None if values[0] == 'max' else _positive(values[0])
            period = _positive(values[1])
        else:
            text = _text(path / 'cpu.cfs_quota_us'); period_text = _text(path / 'cpu.cfs_period_us')
            if text is None or period_text is None:
                continue
            quota = None if text == '-1' else _positive(text)
            period = _positive(period_text)
        quotas.append(dict(path=str(path), quota=quota, period=period,
                           cpus=None if quota is None else quota / period))
    finite = [q['cpus'] for q in quotas if q['cpus'] is not None]
    quota = min(finite) if finite else None
    sets = []
    cpuset = _controller(locations, 'cpuset', 'cpuset.cpus.effective')
    for path in _ancestors(cpuset):
        value = _text(path / ('cpuset.cpus.effective' if cpuset['version'] == 2 else 'cpuset.effective_cpus'))
        if value is None:
            value = _text(path / 'cpuset.cpus')
        parsed = _cpus(value)
        if parsed:
            sets.append(parsed)
    effective = set.intersection(*sets) if sets else None
    status = _text(process / 'status') or ''
    affinity_line = next((line.split(':', 1)[1].strip() for line in status.splitlines()
                          if line.startswith('Cpus_allowed_list:')), None)
    affinity = _cpus(affinity_line)
    verified = (quota is not None and 0 < quota <= maximum
                or effective is not None and 0 < len(effective) <= maximum)
    return dict(verified=bool(verified), effective_quota_cpus=quota, quota_records=quotas,
                effective_cpuset=sorted(effective) if effective is not None else None,
                process_affinity=sorted(affinity) if affinity is not None else None)


def observe(plan, *, environ=None, proc_root='/proc', cgroup_root='/sys/fs/cgroup',
            pid=None, runtime_python=None, runtime_prefix=None):
    """Read current Slurm/kernel evidence; report unknowns without inventing caps."""
    _, python, prefix, resources = _plan(plan)
    env = dict(os.environ if environ is None else environ)
    pid = os.getpid() if pid is None else pid
    _need(type(pid) is int and pid > 0, 'Positive process PID required')
    process = _unlinked(_absolute(proc_root) / str(pid))
    root = _unlinked(_absolute(cgroup_root)); locations = _locations(process, root)
    issues = []
    job_id = env.get('SLURM_JOB_ID')
    _need(job_id is None or isinstance(job_id, str), 'Invalid Slurm job ID')
    nodes = [_positive(env[k]) for k in ('SLURM_NNODES', 'SLURM_JOB_NUM_NODES') if k in env]
    tasks = _positive(env.get('SLURM_NTASKS')); cpus = _positive(env.get('SLURM_CPUS_PER_TASK'))
    slurm_verified = bool(job_id and job_id.isdecimal() and int(job_id) > 0 and nodes
                          and all(n == 1 for n in nodes) and tasks == 1
                          and cpus is not None and cpus <= resources['maximum_cpus'])
    if not slurm_verified:
        issues.append('A matching single-node single-task Slurm CPU allocation is unverified')
    actual_python = str(_absolute(runtime_python or sys.executable))
    actual_prefix = str(_absolute(runtime_prefix or sys.prefix))
    runtime_verified = actual_python == str(python) and actual_prefix == str(prefix)
    if not runtime_verified:
        issues.append('Exact interpreter/runtime prefix differs from the approved plan')
    memory = _controller(locations, 'memory', 'memory.max')
    unified = memory is not None and memory['version'] == 2
    memory_limit, memory_records = _limits(memory, 'memory.max' if unified else 'memory.limit_in_bytes', v1_memory=not bool(unified))
    memory_usage = _positive(_text(memory['leaf'] / ('memory.current' if unified else 'memory.usage_in_bytes')), zero=True) if memory else None
    memory_verified = memory_limit is not None and 0 < memory_limit <= resources['maximum_memory_bytes']
    if not memory_verified:
        issues.append('Finite effective cgroup RAM enforcement within the approved budget is unverified')
    cpu = _cpu_evidence(locations, process, resources['maximum_cpus'])
    if not cpu['verified']:
        issues.append('Effective CPU quota/cpuset within the approved budget is unverified')
    if unified:
        swap_limit, swap_records = _limits(memory, 'memory.swap.max')
        swap_usage = _positive(_text(memory['leaf'] / 'memory.swap.current'), zero=True)
        swap_semantics = 'independent cgroup-v2 swap cap'
    else:
        combined, swap_records = _limits(memory, 'memory.memsw.limit_in_bytes', v1_memory=True)
        swap_limit, swap_usage = None, None
        swap_semantics = 'independent swap cap unknown; v1 memsw when present is combined RAM+swap'
    swap_verified = (swap_limit is not None and swap_limit <= resources['maximum_swap_bytes']
                     and swap_usage is not None and swap_usage <= resources['maximum_swap_bytes'])
    pids_location = _controller(locations, 'pids', 'pids.max')
    pids_limit, pids_records = _limits(pids_location, 'pids.max')
    pids_usage = _positive(_text(pids_location['leaf'] / 'pids.current'), zero=True) if pids_location else None
    pids_verified = (pids_limit is not None and 0 < pids_limit <= resources['maximum_pids']
                     and pids_usage is not None and pids_usage <= pids_limit)
    expected_environment = sanitized_environment(plan, environ=env)
    required_keys = set(WRITER_PATHS).union(THREAD_VARIABLES, {'PYTHONNOUSERSITE'})
    mismatched = sorted(key for key in required_keys if env.get(key) != expected_environment[key])
    unexpected = sorted(set(env).difference(expected_environment, {'PWD', 'SHLVL', '_'}))
    environment = dict(verified=not mismatched and not unexpected,
                       source='process_environment' if environ is None else 'supplied_mapping',
                       mismatched_variables=mismatched, unexpected_variables=unexpected)
    fit_issues = list(issues)
    if not environment['verified']:
        fit_issues.append('Owned native-writer/cache/thread/startup environment is unverified')
    if not swap_verified:
        fit_issues.append('Independent swap containment/usage within the approved budget is unverified')
    if not pids_verified:
        fit_issues.append('Finite effective PID containment/usage within the approved budget is unverified')
    if not sys.flags.no_user_site:
        fit_issues.append('Interpreter startup did not disable the user site')
    return dict(schema_version=1, status='blocked' if issues else 'smoke_observed',
        smoke_containment_verified=not issues, fit_containment_verified=not fit_issues,
        blocking_reasons=issues, fit_blocking_reasons=fit_issues,
        resources=dict(resources), environment=environment, slurm=dict(verified=slurm_verified, job_id=job_id, nodes=nodes, tasks=tasks, cpus_per_task=cpus),
        runtime=dict(verified=runtime_verified, python=actual_python, prefix=actual_prefix,
                     expected_python=str(python), expected_prefix=str(prefix), no_user_site=bool(sys.flags.no_user_site)),
        memory=dict(verified=memory_verified, cgroup_version=memory['version'] if memory else None,
                    limit_bytes=memory_limit, current_bytes=memory_usage, records=memory_records),
        swap=dict(verified=swap_verified, limit_bytes=swap_limit, current_bytes=swap_usage,
                  combined_v1_limit_bytes=combined if not unified else None, semantics=swap_semantics, records=swap_records),
        pids=dict(verified=pids_verified, limit=pids_limit, current=pids_usage, records=pids_records), cpu=cpu,
        scientific_fits=0, execution_authorized=False)


def require_fit_ready(observation, *, require_actual_environment=True):
    """Require verified containment only; scientific approval is still separate."""
    _need(observation.get('schema_version') == 1 and observation.get('status') == 'smoke_observed'
          and observation.get('fit_containment_verified') is True
          and not observation.get('blocking_reasons') and not observation.get('fit_blocking_reasons'),
          'Scientific fits blocked: required Slurm/runtime/cgroup containment is unverified')
    _need(type(require_actual_environment) is bool, 'Explicit boolean environment evidence policy required')
    for key in ('slurm', 'runtime', 'memory', 'swap', 'pids', 'cpu', 'environment'):
        _need(observation.get(key, {}).get('verified') is True, 'Scientific fits blocked: unverified ' + key)
    if require_actual_environment:
        _need(observation['environment'].get('source') == 'process_environment',
              'Scientific fits blocked: verify the actual child environment, not a supplied mapping')
    _need(observation['runtime'].get('no_user_site') is True
          and observation['runtime']['python'] == observation['runtime']['expected_python']
          and observation['runtime']['prefix'] == observation['runtime']['expected_prefix'],
          'Scientific fits blocked: runtime/startup scope differs')
    resources = observation['resources']; slurm = observation['slurm']
    _need(isinstance(slurm['job_id'], str) and slurm['job_id'].isdecimal()
          and int(slurm['job_id']) > 0 and slurm['nodes']
          and all(type(n) is int and n == 1 for n in slurm['nodes'])
          and type(slurm['tasks']) is int and slurm['tasks'] == 1
          and type(slurm['cpus_per_task']) is int
          and 0 < slurm['cpus_per_task'] <= resources['maximum_cpus'],
          'Scientific fits blocked: scheduler scope differs')
    _need(_absolute(observation['runtime']['expected_python']).is_relative_to(
              _absolute(observation['runtime']['expected_prefix'])),
          'Scientific fits blocked: interpreter leaves its runtime prefix')
    for section, limit_key, maximum in (
            ('memory', 'limit_bytes', resources['maximum_memory_bytes']),
            ('pids', 'limit', resources['maximum_pids'])):
        value = observation[section][limit_key]
        _need(type(value) is int and 0 < value <= maximum,
              'Scientific fits blocked: effective ' + section + ' cap differs')
    swap = observation['swap']; pids = observation['pids']
    _need(all(type(swap[k]) is int and 0 <= swap[k] <= resources['maximum_swap_bytes']
              for k in ('limit_bytes', 'current_bytes'))
          and type(pids['current']) is int and 0 <= pids['current'] <= pids['limit'],
          'Scientific fits blocked: swap/PID evidence differs')
    cpu = observation['cpu']; quota = cpu['effective_quota_cpus']; cpuset = cpu['effective_cpuset']
    _need((type(quota) in (int, float) and math.isfinite(quota)
           and 0 < quota <= resources['maximum_cpus'])
          or (isinstance(cpuset, list) and 0 < len(set(cpuset)) <= resources['maximum_cpus']
              and all(type(i) is int and i >= 0 for i in cpuset)),
          'Scientific fits blocked: CPU containment differs')
    return observation
