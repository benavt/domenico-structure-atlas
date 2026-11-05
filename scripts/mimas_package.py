"""Source-only, dependency-light wheel evidence and CPU batch preparation.

This module does not install packages, contact hosts, submit jobs or fit models.
Private paths are supplied by an approved project-local plan.
"""
import email
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import zipfile


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('A regular non-symlink artifact is required')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def wheel_contract(path, commit):
    """Bind every wheel payload byte; do not infer a VCS revision from its name."""
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Exact source commit is required')
    entry = record(path)
    hashes = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel member')
        metadata = []
        for item in members:
            name = item.filename
            parts = PurePosixPath(name).parts
            if (not parts or name.startswith('/') or '..' in parts or '\\' in name
                    or name != str(PurePosixPath(name)) or item.is_dir()
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe wheel member')
            if any(p.endswith('.data') for p in parts):
                raise ValueError('Wheel relocation schemes need explicit support')
            if parts[0] in ('data', 'outputs', 'external', 'tests'):
                raise ValueError('Scientific data and external trees cannot be packaged')
            body = archive.read(item)
            if parts[0].endswith('.dist-info'):
                if name.endswith('/METADATA'):
                    metadata.append(email.message_from_bytes(body))
            else:
                hashes[name] = hashlib.sha256(body).hexdigest()
    if (len(metadata) != 1 or metadata[0]['Name'] != 'structure-clustering'
            or metadata[0]['Version'] != '0.1.0' or not hashes):
        raise ValueError('Unexpected library wheel identity')
    return {'distribution': 'structure-clustering', 'version': '0.1.0',
            'commit': commit, 'wheel': entry, 'filehashes': hashes}


def requirements_from_wheels(directory):
    """Make an offline hash-checked requirements file from a frozen wheelhouse."""
    result = {}
    for path in sorted(Path(directory).glob('*.whl')):
        with zipfile.ZipFile(path) as archive:
            candidates = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(candidates) != 1:
                raise ValueError('Wheel metadata missing or ambiguous')
            value = email.message_from_bytes(archive.read(candidates[0]))
        name, version = value['Name'], value['Version']
        if not name or not version or any(c.isspace() for c in name + version):
            raise ValueError('Invalid package metadata')
        key = re.sub('[-_.]+', '-', name).lower()
        if key in result:
            raise ValueError('One wheel per package is required')
        result[key] = f'{name}=={version} --hash=sha256:{sha256(path)}'
    if not result:
        raise ValueError('Empty wheelhouse')
    return '\n'.join(result[key] for key in sorted(result)) + '\n'


def smoke_batch(plan_path, plan, bootstrap_python, *, environment_exists=False):
    """Prepare one CPU batch body for the supported monitored front door.

    Slurm directives use resolved literal values. Execution approval is checked
    before environment creation as well as in the runner. No submission occurs.
    """
    for field in ('run_root', 'runtime_prefix', 'runtime_python', 'wheelhouse', 'requirements', 'runner'):
        value = plan.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute() or '\n' in value:
            raise ValueError('Resolved absolute batch paths required: ' + field)
    partition = plan.get('partition')
    if not isinstance(partition, str) or not re.fullmatch('[A-Za-z0-9_-]+', partition):
        raise ValueError('Verified literal CPU partition required')
    cpus = plan['resources']['maximum_cpus']
    memory = plan['resources']['maximum_memory_bytes']
    seconds = plan['resources']['maximum_wall_seconds']
    if (type(cpus) is not int or not 1 <= cpus <= 4 or type(memory) is not int
            or memory <= 0 or type(seconds) is not int or not 60 <= seconds <= 3600):
        raise ValueError('Bounded smoke resources required')
    if not Path(bootstrap_python).is_absolute():
        raise ValueError('Absolute existing Python 3.11 bootstrap required')
    if not Path(plan_path).is_absolute() or '\n' in str(plan_path):
        raise ValueError('Absolute plan path required')
    q = shlex.quote
    minutes = (seconds + 59) // 60
    lines = ['#!/bin/bash', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
             '#SBATCH --cpus-per-task=' + str(cpus),
             '#SBATCH --mem=' + str((memory + 1024**2 - 1)//1024**2) + 'M',
             '#SBATCH --time=' + str(minutes), '#SBATCH --partition=' + partition,
             'set -euo pipefail',
             'export PATH=/usr/bin:/bin',
             'unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT FLEET_MASK_RECIPE FLEET_MASK_INDEX',
             'for task_pip_var in "${!PIP_@}"; do unset "$task_pip_var"; done',
             q(bootstrap_python) + ' -I ' + q(plan['runner']) + ' verify ' + q(str(plan_path)),
             'export PYTHONNOUSERSITE=1', 'export PIP_NO_INDEX=1',
             'export PIP_CONFIG_FILE=/dev/null',
             'export PIP_DISABLE_PIP_VERSION_CHECK=1', 'export PIP_NO_CACHE_DIR=1']
    if not environment_exists:
        temporary = str(Path(plan_path).parent / 'setup-tmp')
        lines += [q(bootstrap_python) + ' -I ' + q(plan['runner']) + ' setup-check ' + q(str(plan_path)),
                  '/bin/mkdir -m 700 -- ' + q(temporary),
                  'export TMPDIR=' + q(temporary),
                  'test ! -e ' + q(plan['runtime_prefix']),
                  q(bootstrap_python) + ' -I -m venv --copies ' + q(plan['runtime_prefix']),
                  q(plan['runtime_python']) + ' -I -m pip install --no-index --require-hashes --only-binary=:all: --find-links ' + q(plan['wheelhouse']) + ' -r ' + q(plan['requirements'])]
    lines += ['exec ' + q(plan['runtime_python']) + ' -I ' + q(plan['runner']) + ' smoke ' + q(str(plan_path))]
    return '\n'.join(lines) + '\n'
