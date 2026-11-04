"""Package accepted scores and completed screening artifacts; never fit science."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from snapshot import sha256, verify


def build(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    for path, digest in plan['frozen_files'].items():
        if sha256(path) != digest:
            raise ValueError('Changed saved input: ' + path)
    study = json.loads(Path(plan['study']).read_text())
    rows = verify(plan['native_manifest'], plan['acceptance'], study)
    output = Path(plan['output'])
    if output.exists():
        raise FileExistsError('Use a new saved publication directory')
    output.mkdir(parents=True)
    template = Path(__file__).resolve().parents[1] / 'templates/screening-atlas'
    for name in ('index.html', 'style.css', 'app.js'):
        shutil.copyfile(template/name, output/name)
    (output/'assets').mkdir()
    for name, path in plan['assets'].items():
        shutil.copyfile(path, output/'assets'/name)
    catalog = {'schema_version': 1, 'title': 'Domenico structure atlas', 'snapshot_date': '2026-11-03',
               'published_data_date': '2026-11-04', 'stage': 'terminal_screening', 'models': len(rows),
               'source_manifest_sha256': study['snapshot']['manifest_sha256'],
               'acceptance_sha256': study['snapshot']['acceptance_sha256'],
               'energy': study['energy'], 'targets': []}
    rules = json.loads((Path(plan['study']).parent/'model-types.json').read_text())['rules']
    methods = {r['key']: {'name': r['name'], 'color': r['color']} for r in rules}
    for target, saved in plan['screenings'].items():
        config = study['targets'][target]
        screening = json.loads(Path(saved).read_text())
        detection = screening['terminal_detection']
        target_rows = [r for r in rows if r['target'] == target]
        directory = output/'targets'/target
        directory.mkdir(parents=True)
        for path in screening['diagnostics']['images']:
            shutil.copyfile(path, directory/Path(path).name)
        # Two directory labels are the only host bindings in these saved viewers.
        # Keep all distributions/numerical arrays unchanged.
        html = Path(screening['diagnostics']['html']).read_text()
        html = html.replace(screening['distogram_dir'], 'Accepted ' + target + ' snapshot')
        html = html.replace('https://cdn.plot.ly/plotly-2.27.0.min.js', '../../assets/plotly.min.js')
        if any(binding in html for binding in ('/home/tiburon/', '/scratch/benavt/', 'file://')):
            raise ValueError('Private binding remains in public screening viewer')
        (directory/'distance-distributions.html').write_text(html)
        (directory/'terminal-screening.json').write_text(json.dumps(detection, indent=2)+'\n')
        example = next(r for r in target_rows if r['method'] == 'afcluster')
        source = Path(plan['local_inputs'])/target/(example['model_id']+Path(example['path']).suffix)
        if sha256(source) != example['sha256']:
            raise ValueError('Illustrative native model changed')
        content = source.read_text()
        if '/home/' in content or '/scratch/' in content:
            raise ValueError('Private binding in illustrative coordinate file')
        shutil.copyfile(source, directory/('example'+source.suffix))
        public_rows = [{'model': r['model_id'], 'method': r['method'], 'energy': r['energy_t010'],
                        'native_physicality': r['native_physicality'],
                        'native_sha256': r['sha256']} for r in target_rows]
        (directory/'scores.json').write_text(json.dumps(public_rows, separators=(',', ':'))+'\n')
        columns = ['model', 'method', 'native_physicality', 'native_sha256'] + [f'energy_t{i:02d}' for i in range(16)]
        with (directory/'scores.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for native, public in zip(target_rows, public_rows):
                writer.writerow({**{k: v for k, v in public.items() if k != 'energy'},
                                 **{f'energy_t{i:02d}': native[f'energy_t{i:02d}'] for i in range(16)}})
        counts = {}
        for row in target_rows:
            label = row['native_physicality']; counts[label] = counts.get(label, 0)+1
        prefix = 'targets/'+target+'/'
        catalog['targets'].append({'id': target, 'title': config['title'], 'models': len(target_rows),
            'sequence': config['sequence'], 'sequence_sha256': config['sequence_sha256'],
            'residues': config['residues'], 'retained_residues': len(detection['retained_indices']),
            'excluded_residues': len(detection['excluded_indices']),
            'methods': [{**methods[k], 'key': k, 'count': count} for k, count in config['methods'].items()],
            'physicality': counts, 'complete_method_panel': config['complete_eleven_method_panel'],
            'screening_sha256': sha256(saved), 'pair_mask_sha256': detection['pair_mask_sha256'],
            'scores': prefix+'scores.json', 'scores_csv': prefix+'scores.csv',
            'terminal_json': prefix+'terminal-screening.json', 'distance_explorer': prefix+'distance-distributions.html',
            'images': {Path(p).stem: prefix+Path(p).name for p in screening['diagnostics']['images']},
            'example': {'file': prefix+'example'+source.suffix, 'model': example['model_id'], 'sha256': example['sha256'],
                        'format': source.suffix.lstrip('.'), 'selection': 'First AF-Cluster input in accepted manifest; illustrative, not ranked or a cluster representative.'}})
    (output/'atlas.json').write_text(json.dumps(catalog, indent=2)+'\n')
    (output/'.nojekyll').touch()
    files = [{'path': str(p.relative_to(output)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(output.rglob('*')) if p.is_file()]
    receipt = {'schema_version': 1, 'project': 'domenico-structure-atlas', 'scientific_execution': False,
               'stage': 'saved_terminal_screening_and_accepted_energy_snapshot',
               'publication_plan_sha256': sha256(plan_path), 'files': files}
    (output/'export-manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    bundle = output.parent/'site.tar.gz'
    if bundle.exists(): raise FileExistsError(bundle)
    with bundle.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for p in sorted(output.rglob('*')):
                if p.is_file():
                    info = archive.gettarinfo(str(p), arcname=str(p.relative_to(output)))
                    info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                    with p.open('rb') as stream: archive.addfile(info, stream)
    print(json.dumps({'bundle': str(bundle), 'sha256': sha256(bundle), 'bytes': bundle.stat().st_size,
                      'published_bytes': sum(r['bytes'] for r in files), 'files': len(files)+1}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('plan', type=Path)
    build(parser.parse_args().plan)
