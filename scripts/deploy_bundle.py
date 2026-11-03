"""Unpack an exact, already reviewed static export; never run analysis."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

LIMIT = 950_000_000


def unpack(archive, expected, output):
    archive, output = Path(archive), Path(output)
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('An exact reviewed bundle SHA-256 is required')
    from snapshot import sha256
    if sha256(archive) != expected:
        raise ValueError('Bundle hash changed')
    if output.exists():
        raise ValueError('Deployment requires a fresh directory')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or sum(m.size for m in members) > LIMIT:
            raise ValueError('Duplicate members or oversized site')
        for member in members:
            name = member.name
            if (not member.isfile() or not name or name.startswith('/') or
                    any(p in ('', '.', '..') for p in name.split('/')) or
                    any(c in name for c in ('\\', ':', '?', '#')) or
                    any(ord(c) < 32 for c in name) or
                    any(p.startswith('.') for p in PurePosixPath(name).parts) and name != '.nojekyll'):
                raise ValueError('Unsafe deployment member: ' + name)
        if not {'index.html', 'export-manifest.json', '.nojekyll'} <= set(names):
            raise ValueError('Not a complete saved static export')
        receipt = json.load(tar.extractfile('export-manifest.json'))
        if (receipt.get('schema_version') != 1 or receipt.get('project') != 'domenico-structure-atlas'
                or receipt.get('scientific_execution') is not False):
            raise ValueError('Wrong project or export contract')
        rows = receipt['files']
        if len({r['path'] for r in rows}) != len(rows) or set(names) != {r['path'] for r in rows} | {'export-manifest.json'}:
            raise ValueError('Bundle differs from export allowlist')
        by_name = {m.name: m for m in members}
        for row in rows:
            raw = tar.extractfile(by_name[row['path']]).read()
            if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('Saved artifact differs: ' + row['path'])
        output.mkdir(parents=True)
        for member in members:
            dest = output / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, dest.open('xb') as target:
                import shutil
                shutil.copyfileobj(source, target)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    unpack(a.archive, a.sha256, a.output)
