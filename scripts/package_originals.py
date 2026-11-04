"""Package exact accepted originals as separate, hash-inventoried release assets.

Requires a frozen approved export plan. No coordinates or scientific values are
changed. These archives are separate from the size-limited GitHub Pages site.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from snapshot import sha256


def package(plan_path):
    from structure_clustering.publication import inspect_artifact
    plan = json.loads(Path(plan_path).read_text())
    if sha256(plan['native_manifest']) != plan['native_manifest_sha256']:
        raise ValueError('Accepted original snapshot changed')
    native = json.loads(Path(plan['native_manifest']).read_text())
    if native['status'] != 'validated': raise ValueError('Originals require an accepted snapshot')
    expected = {row['model_id']: row for row in native['models']}
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'status': 'running', 'plan_sha256':sha256(plan_path), 'targets': {}}
    for target, entry in plan['intakes'].items():
        if sha256(entry['path']) != entry['sha256']: raise ValueError('Frozen input intake changed')
        intake=json.loads(Path(entry['path']).read_text());rows=intake['models']
        if (intake['target']!=target or {r['model_id'] for r in rows}!={n for n,r in expected.items() if r['target']==target}
                or len({r['model_id'] for r in rows})!=len(rows)):
            raise ValueError('Original archive cohort differs from accepted target')
        archive=output/(target+'-originals.tar.gz'); inventory=[]
        with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=3,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w') as tar:
                for index,row in enumerate(sorted(rows,key=lambda r:r['model_id']),1):
                    path=Path(row['path']); name=row['model_id']+path.suffix
                    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.(pdb|cif|mmcif)',name,re.I) or path.is_symlink():
                        raise ValueError('Unsafe native coordinate identity')
                    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
                    if digest!=row['sha256'] or digest!=expected[row['model_id']]['sha256']:
                        raise ValueError('Original coordinate changed')
                    inspect_artifact(data,name)
                    member=tarfile.TarInfo(target+'/'+name);member.size=len(data);member.mode=0o644;member.mtime=0
                    tar.addfile(member,io.BytesIO(data));inventory.append({'model':row['model_id'],'file':target+'/'+name,'bytes':len(data),'sha256':digest})
                    if index%500==0:print(f'{target}: archived {index}/{len(rows)} exact originals',flush=True)
        if archive.stat().st_size>=2_000_000_000:raise ValueError('Original archive exceeds the selected release-asset budget')
        expected_members={row['file']:row for row in inventory};seen=set()
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected_members or member.name in seen:
                    raise ValueError('Original archive member identity differs')
                recorded=expected_members[member.name]
                with tar.extractfile(member) as stream:
                    digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if member.size!=recorded['bytes'] or digest!=recorded['sha256']:
                    raise ValueError('Original archive round-trip changed native bytes')
                seen.add(member.name)
        if seen!=set(expected_members):raise ValueError('Original archive is incomplete')
        inventory_file=output/(target+'-originals-manifest.json')
        inventory_file.write_text(json.dumps({'target':target,'models':inventory,'native_manifest_sha256':plan['native_manifest_sha256']},indent=2)+'\n')
        result['targets'][target]={'url':plan['release_url'].rstrip('/')+'/'+archive.name,'sha256':sha256(archive),
            'bytes':archive.stat().st_size,'models':len(rows),'file':str(archive),
            'inventory':str(inventory_file),'inventory_sha256':sha256(inventory_file)}
        (output/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
    result['status']='validated'
    (output/'completion.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);package(p.parse_args().plan)
