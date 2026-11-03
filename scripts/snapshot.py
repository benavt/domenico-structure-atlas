"""Verify an accepted scored cohort without fitting, scoring or copying models."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(manifest, acceptance, study, *, verify_coordinates=False):
    manifest, acceptance = Path(manifest), Path(acceptance)
    if sha256(manifest) != study['snapshot']['manifest_sha256']:
        raise ValueError('Scored snapshot hash changed')
    if sha256(acceptance) != study['snapshot']['acceptance_sha256']:
        raise ValueError('Acceptance receipt hash changed')
    data, accepted = json.loads(manifest.read_text()), json.loads(acceptance.read_text())
    rows = data['models']
    if data.get('status') != 'validated' or accepted.get('status') != 'validated':
        raise ValueError('Native scoring acceptance is required')
    if accepted['canonical_manifest_sha256'] != sha256(manifest):
        raise ValueError('Acceptance belongs to another manifest')
    expected = {k: v['models'] for k, v in study['targets'].items()}
    if dict(Counter(r['target'] for r in rows)) != expected:
        raise ValueError('Target coverage changed')
    if accepted['models'] != len(rows) or accepted['energies'] != 16 * len(rows):
        raise ValueError('Accepted score coverage changed')
    if len({r['model_id'] for r in rows}) != len(rows) or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Ambiguous model identity or coordinate path')
    for key in ('checkpoint', 'config_sha256', 'settings'):
        if data[key] != accepted[key]:
            raise ValueError('Scoring provenance changed: ' + key)
    for target, config in study['targets'].items():
        methods = dict(Counter(r['method'] for r in rows if r['target'] == target))
        if methods != config['methods'] or accepted['targets'][target]['methods'] != methods:
            raise ValueError('Method coverage changed: ' + target)
    for row in rows:
        energies = [row[f'energy_t{i:02d}'] for i in range(16)]
        if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in energies):
            raise ValueError('Missing or nonfinite saved energies')
        if row['energy_t010'] != energies[10]:
            raise ValueError('Primary energy alias differs')
        if verify_coordinates and sha256(row['path']) != row['sha256']:
            raise ValueError('Native coordinate hash changed: ' + row['model_id'])
    return rows
