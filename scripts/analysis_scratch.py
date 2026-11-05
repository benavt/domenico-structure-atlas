"""Own one plan-specific tmpfs directory; systemd cleans it after job shutdown."""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat

from snapshot import sha256


def scratch_root(plan):
    root = Path(plan['environment']['TMPDIR'])
    if (root.parent != Path('/dev/shm') or not root.name.startswith('domenico-atlas-')
            or root.name != plan['scratch']['directory_name']):
        raise ValueError('Scratch must be the exact plan-owned /dev/shm directory')
    return root


def assert_tmpfs():
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    if not any(row[1:3] == ['/dev/shm', 'tmpfs'] for row in mounts):
        raise ValueError('RAM scratch requires the existing /dev/shm tmpfs mount')


def approval(path):
    proof = json.loads((path.parent/'approval.json').read_text())
    if proof.get('approved_by') != 'user' or proof.get('approved_plan_sha256') != sha256(path):
        raise ValueError('Exact plan approval is required for scratch creation or cleanup')


def create(path, plan):
    approval(path)
    root = scratch_root(plan)
    assert_tmpfs()
    if shutil.disk_usage(root.parent).free < plan['scratch']['minimum_free_bytes_before_start']:
        raise ValueError('Insufficient free RAM scratch capacity')
    root.mkdir(mode=0o700)
    info = root.stat()
    marker = dict(plan_sha256=sha256(path), uid=os.getuid(), device=info.st_dev, inode=info.st_ino)
    with (root/'.owner.json').open('x') as stream:
        json.dump(marker, stream)
        stream.flush(); os.fsync(stream.fileno())


def assert_quiescent():
    entry = next(row for row in Path('/proc/self/cgroup').read_text().splitlines() if row.startswith('0::'))
    group = Path('/sys/fs/cgroup')/entry.split('::', 1)[1].lstrip('/')
    if set((group/'cgroup.procs').read_text().split()) != {str(os.getpid())}:
        raise ValueError('Scratch cleanup requires the stopped unit with only ExecStopPost remaining')


def cleanup(path, plan):
    approval(path)
    root = scratch_root(plan)
    assert_tmpfs()
    if not root.exists() and not root.is_symlink():
        return
    assert_quiescent()
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Refusing changed scratch root')
    owner = root/'.owner.json'
    if owner.is_symlink() or not owner.is_file():
        raise ValueError('Missing scratch ownership marker')
    marker = json.loads(owner.read_text())
    info = root.stat()
    if marker != dict(plan_sha256=sha256(path), uid=os.getuid(), device=info.st_dev, inode=info.st_ino):
        raise ValueError('Scratch ownership identity changed')
    for member in [root, *root.rglob('*')]:
        s = member.lstat()
        if (s.st_uid != os.getuid() or s.st_dev != info.st_dev or
                not (stat.S_ISDIR(s.st_mode) or (stat.S_ISREG(s.st_mode) and s.st_nlink == 1))):
            raise ValueError('Scratch contains a symlink, hard link, special file or foreign owner')
    shutil.rmtree(root)
    output = Path(plan['run_root'])
    if output.is_dir():
        with (output/'scratch-cleanup.json').open('x') as stream:
            json.dump(dict(status='removed_owned_ram_scratch', plan_sha256=sha256(path),
                           path=str(root), service_result=os.environ.get('SERVICE_RESULT')), stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    cleanup(args.plan, json.loads(args.plan.read_text()))
