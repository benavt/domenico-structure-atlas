"""Execute one explicitly approved private plan with source and resource checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from snapshot import sha256


def control_group():
    row = next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    return Path('/sys/fs/cgroup')/row.split('::',1)[1].lstrip('/')


def verify_controls(group, resources):
    if (group/'memory.max').read_text().strip() != str(resources['maximum_memory_bytes']):
        raise ValueError('Required memory cap is absent')
    if (group/'memory.swap.max').read_text().strip() != '0':
        raise ValueError('Required zero-swap control is absent')
    quota, period = (group/'cpu.max').read_text().split()
    if quota == 'max' or int(quota)/int(period) > resources['maximum_cpus']:
        raise ValueError('Required CPU quota is absent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    args = parser.parse_args()
    path = args.plan.resolve(); plan = json.loads(path.read_text())
    approval = json.loads((path.parent/'approval.json').read_text())
    if approval.get('approved_by') != 'user' or approval.get('approved_plan_sha256') != sha256(path):
        raise ValueError('Explicit approval of this exact plan is required')
    for name, expected in plan['frozen_files'].items():
        if sha256(name) != expected: raise ValueError('Frozen file changed: '+name)
    for source in plan['sources'].values():
        if subprocess.check_output(['git','-C',source['checkout'],'rev-parse','HEAD'],text=True).strip() != source['commit']:
            raise ValueError('Source revision changed')
        if subprocess.check_output(['git','-C',source['checkout'],'status','--porcelain'],text=True).strip():
            raise ValueError('Source checkout is dirty')
    verify_controls(control_group(),plan['resources'])
    os.environ.update(plan['environment'])
    sys.path.insert(0,plan['sources']['library']['checkout'])
    from structure_clustering.run_records import runtime_identity
    from structure_clustering.resource_guard import ResourceLimits, run_guarded
    if runtime_identity() != json.loads(Path(plan['runtime_fingerprint']).read_text()):
        raise ValueError('Runtime changed')
    if json.loads(Path(plan['validation_receipt']).read_text()).get('status') != 'validated':
        raise ValueError('Native validation is incomplete')
    run = Path(plan['run_root'])
    if run.exists(): raise FileExistsError('Inspect this attempt before planning recovery')
    subprocess.run([sys.executable,plan['fleet_script'],'storage','preflight',plan['project'],
                    '--required-bytes',str(plan['resources']['additional_storage_bytes'])],check=True)
    run.mkdir(parents=True)
    if plan.get('scratch'):
        from analysis_scratch import create
        create(path, plan)
    else:
        Path(plan['environment']['TMPDIR']).mkdir()
    lease = Path(plan['storage_lease'])
    with lease.open('x') as stream:
        json.dump(dict(project=plan['project'],bytes=plan['resources']['additional_storage_bytes'],plan_sha256=sha256(path)),stream)
    try:
        code = run_guarded(plan['command'],run/'control',admission=ResourceLimits(**plan['guard']['admission']),
                           runtime=ResourceLimits(**plan['guard']['runtime']),interval_seconds=5,
                           maximum_wall_seconds=plan['resources']['maximum_wall_seconds'],consecutive_failures=3)
        if code: raise RuntimeError('Approved stage failed; preserve logs and review before any retry')
        if 'expected_result' in plan:
            output = Path(plan['expected_result'])
            result = json.loads(output.read_text())
            if result.get('status') != plan['expected_status']:
                raise ValueError('Scientific acceptance is incomplete')
            completion = dict(status=plan['expected_status'], plan_sha256=sha256(path),
                              result=str(output), result_sha256=sha256(output))
        else:
            output = Path(plan['expected_screening'])
            if not output.is_file(): raise ValueError('Missing screening receipt')
            completion = dict(status='terminal_diagnostics_ready_for_review',
                plan_sha256=sha256(path),screening=str(output),screening_sha256=sha256(output),
                projections_fitted=False,molecular_exports_completed=False)
        (run/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    finally:
        lease.unlink()


if __name__ == '__main__': main()
