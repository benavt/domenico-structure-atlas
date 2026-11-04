"""Prepare the canonical shared atlas from completed exact-cohort workflows.

No projection fitting or representative selection occurs in publication. Native
coordinate derivatives/archives require their own approved publication scope.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil

from snapshot import sha256


def checked_json(path, digest):
    if sha256(path) != digest: raise ValueError('Frozen saved artifact changed: '+str(path))
    return json.loads(Path(path).read_text())


def projections(manifest, expected):
    import numpy as np
    if not all(method in manifest.get('clustering', {}) for method in ['pca', 'umap', 'tsne']):
        raise ValueError('All three real saved projections are required before publication')
    names, result = None, {}
    for method in ['pca', 'umap', 'tsne']:
        with np.load(manifest['clustering'][method]['embeddings'], allow_pickle=False) as saved:
            current = saved['filenames'].tolist(); xy = saved['embeddings']
            if (set(current) != set(expected) or len(current) != len(expected)
                    or names is not None and names != current or xy.shape != (len(expected), 2)
                    or not np.isfinite(xy).all()):
                raise ValueError('All three real saved projections must match the complete accepted cohort and order')
            names = current
            result[method] = {'xy': xy.tolist(), 'axes': [method.upper()+' 1', method.upper()+' 2']}
            if method == 'pca' and 'explained_variance_ratio' in saved:
                ratios = saved['explained_variance_ratio'].tolist()
                result[method].update(explained_variance_ratio=ratios,
                    axes=[f'PC {i+1} ({100*v:.1f}% variance)' for i, v in enumerate(ratios[:2])])
    return names, result


def continuous(key, name, values):
    finite = [x for x in values if x is not None and math.isfinite(x)]
    return {'key': key, 'name': name, 'type': 'continuous', 'values': values, 'colorscale': 'Viridis',
            'range': [min(finite), max(finite)] if finite else [0, 1], 'unavailable': not finite}


def build(plan_path):
    from native_ca import write_ca
    from structure_clustering.publication import export
    plan = json.loads(Path(plan_path).read_text())
    study = checked_json(plan['study']['path'], plan['study']['sha256'])
    native = checked_json(plan['native']['path'], plan['native']['sha256'])
    confidence = checked_json(plan['confidence']['path'], plan['confidence']['sha256'])
    if (native['status'] != 'validated' or confidence['status'] != 'validated'
            or plan['native']['sha256'] != study['snapshot']['manifest_sha256']
            or confidence['source_manifest_sha256'] != plan['native']['sha256']):
        raise ValueError('Publication requires the accepted scored snapshot and exact native confidence binding')
    if set(plan['targets']) != set(study['targets']):
        raise ValueError('Standard atlas requires real saved workflows for all three targets')
    output, source = Path(plan['output']), Path(plan['source'])
    if output.exists() or source.exists(): raise FileExistsError('Use new immutable publication directories')
    source.mkdir(parents=True)
    (source/'assets').mkdir()
    for name, entry in plan['vendor_assets'].items():
        if sha256(entry['path']) != entry['sha256']: raise ValueError('Vendor asset changed')
        shutil.copyfile(entry['path'], source/'assets'/name)
    by_native = {row['model_id']: row for row in native['models']}
    by_confidence = {row['model_id']: row for row in confidence['models']}
    if len(by_confidence) != len(confidence['models']) or set(by_confidence) != set(by_native):
        raise ValueError('Confidence identities differ from the accepted cohort')
    rules = json.loads((Path(plan['study']['path']).parent/'model-types.json').read_text())['rules']
    method_map = {row['key']: row for row in rules}
    atlas = {'targets': [], 'default_target': next(iter(plan['targets']))}
    scopes = {'schema_version': 1, 'targets': {}}
    for target, frozen in plan['targets'].items():
        manifest = checked_json(frozen['manifest'], frozen['manifest_sha256'])
        if manifest['analysis_status'] != 'complete' or manifest['identifier'] != target:
            raise ValueError('Real clustering has not completed for '+target)
        expected = {name: row for name, row in by_native.items() if row['target'] == target}
        if len(expected) != study['targets'][target]['models']: raise ValueError('Wrong target snapshot count')
        names, xy = projections(manifest, expected)
        with Path(manifest['knn_clustering']['labels_csv']).open() as stream: members = list(csv.DictReader(stream))
        labels = {row['model']: row for row in members}
        if len(labels) != len(members) or set(labels) != set(names): raise ValueError('Cluster memberships differ from projections')
        report = json.loads(Path(manifest['cluster_ensembles']['report']).read_text())
        if report.get('residue_correspondence', {}).get('native_residue_labels_preserved') is not True:
            raise ValueError('Native residue correspondence has not been validated for representatives')
        folder = source/target; (folder/'models').mkdir(parents=True); (folder/'representatives').mkdir()
        model_files, display_hashes, native_hashes = [], [], []
        for index, name in enumerate(names):
            row, score = expected[name], by_confidence[name]
            if score['coordinate_sha256'] != row['sha256']: raise ValueError('Wrong confidence/coordinate join')
            destination = folder/'models'/(name+'.cif.gz')
            display = write_ca(labels[name]['source_path'], destination, native_sha256=row['sha256'],
                               expected_count=study['targets'][target]['residues'])
            model_files.append(str(destination.relative_to(source))); display_hashes.append(display['sha256']); native_hashes.append(row['sha256'])
            if (index+1) % 500 == 0: print(f'{target}: packaged {index+1}/{len(names)} exact native Cα traces', flush=True)
        representatives = []
        for group in report['clusters']:
            if not group['selected']: continue
            selected = group['selected'][0]; name = selected['model']; original = Path(selected['source_path'])
            if name not in expected or sha256(original) != expected[name]['sha256']: raise ValueError('Representative changed')
            destination = folder/'representatives'/original.name; shutil.copyfile(original, destination)
            representatives.append({'cluster': group['cluster_label'], 'count': group['cluster_size'], 'model': name,
                'method': method_map[expected[name]['method']]['name'], 'color': group['color'],
                'file': str(destination.relative_to(source)), 'format': 'cif' if original.suffix.lower() in ['.cif','.mmcif'] else 'pdb',
                'sha256': expected[name]['sha256'], 'structure_confidence': None})
        cluster_values = [labels[name]['cluster_label'] for name in names]
        colors = {row['cluster_label']: row['color'] for row in members}
        method_names = [method_map[expected[name]['method']]['name'] for name in names]
        schemes = [{'key':'clusters','name':'Clusters','type':'categorical','values':cluster_values,'colors':colors},
                   {'key':'methods','name':'Model origin','type':'categorical','values':method_names,
                    'colors':{rule['name']:rule['color'] for rule in rules}},
                   continuous('proteinebm_t010','ProteinEBM energy · t=0.10',[expected[name]['energy_t010'] for name in names]),
                   continuous('mean_plddt','Native mean pLDDT',[by_confidence[name]['mean_plddt'] for name in names]),
                   continuous('ptm','Native pTM',[by_confidence[name]['ptm'] for name in names])]
        flags = [expected[name]['native_physicality'] for name in names]
        flag_colors = {'retained':'#228833','rejected':'#cc3311','unavailable':'#bbbbbb','see original100-frame filtering metadata':'#ddcc77'}
        schemes.append({'key':'native_physicality','name':'Native physicality flags','type':'categorical','values':flags,'colors':flag_colors})
        membership = folder/'memberships-and-scores.csv'
        with membership.open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['model','method','cluster','native_physicality','native_sha256','mean_plddt','ptm',*[f'energy_t{i:02d}' for i in range(16)]])
            for name in names:
                row,score=expected[name],by_confidence[name]
                writer.writerow([name,row['method'],labels[name]['cluster_label'],row['native_physicality'],row['sha256'],score['mean_plddt'],score['ptm'],*[row[f'energy_t{i:02d}'] for i in range(16)]])
        detection = manifest['terminal_detection']; count = study['targets'][target]['residues']
        data = {'target':target,'view_id':'whole','view_title':study['targets'][target]['title']+' · validated default clustering',
            'models':names,'residues':count,'methods':method_names,'clusters':cluster_values,'projections':xy,'schemes':schemes,
            'representatives':representatives,'views':[{'id':'whole','name':'Validated UMAP clusters','data':target+'/primary.json'}],
            'membership_file':str(membership.relative_to(source)),'heatmaps':[], 'model_files':model_files,'model_sha256':display_hashes,
            'model_native_sha256':native_hashes,'model_representation':'ca_trace','model_structure_confidence':[None]*len(names),
            'model_physicality':flags,'representative_kind':'confidence-ranked representative or unscored structural medoid',
            'representative_policy':'Highest genuine native confidence among the default selected ensemble; unscored clusters use the recorded structural medoid. Energy is not confidence.',
            'mask_note':f'All {count} supplied construct residues retained; native numbering preserved. Biological disorder is not inferred.',
            'clustering_method_note':'Default seed42 PCA/UMAP/t-SNE; common component-preserving UMAP partition, feature silhouette ≥0.15 and repeat-seed ARI ≥0.80. All three projections retain the exact same model order and memberships.',
            'provenance':{'workflow_manifest_sha256':frozen['manifest_sha256'],'native_manifest_sha256':plan['native']['sha256'],
                'sequence_sha256':study['targets'][target]['sequence_sha256'],'pair_mask_sha256':detection['pair_mask_sha256'],
                'confidence_sha256':plan['confidence']['sha256'],'scientific_library_commit':study['library_commit'],
                'energy_units':'learned arbitrary units; lower within target; not confidence or calibrated free energy'}}
        for image in manifest['visualize_distogram']['images']:
            path=Path(image);destination=folder/'figures'/path.name;destination.parent.mkdir(exist_ok=True);shutil.copyfile(path,destination)
            data['heatmaps'].append({'name':path.stem.replace('_',' ').capitalize(),'file':str(destination.relative_to(source))})
        scope={'chain':'A','sequence_sha256':study['targets'][target]['sequence_sha256'],'expected_ca_count':count,
               'canonical_positions':[index+1 for index in detection['retained_reference_positions']],
               'analysis_uses_full_construct':not detection['excluded_indices'],'regions_verified':False,'reference_ranges':f'construct positions 1–{count}'}
        scopes['targets'][target]={'views':{'whole':scope}}
        (folder/'primary.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
        atlas['targets'].append({'id':target,'name':study['targets'][target]['title'],'data':target+'/primary.json'})
    (source/'atlas.json').write_text(json.dumps(atlas,indent=2)+'\n')
    (source/'clustering-residues.json').write_text(json.dumps(scopes,indent=2)+'\n')
    files=[{'path':str(p.relative_to(source)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(source.rglob('*')) if p.is_file()]
    publication={'schema_version':1,'project':'domenico-structure-atlas','renderer':'ais-explorer-v1',
        'presentation':{'title':'Domenico structure atlas','methods':'Accepted RBP, 5NTase and SERCA snapshot. Full-construct Cα distances, default validated common cluster partition, genuine native confidence and separate saved ProteinEBM energies. Point views are exact native Cα traces; cluster representatives retain full atoms. Native rejection flags never filter the cohort. Camera-dependent reports are delivered after review.'},'files':files}
    manifest=source.parent/'publication-manifest.json';manifest.write_text(json.dumps(publication,indent=2)+'\n')
    export(manifest,source,output)
    print('Canonical shared atlas prepared:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path);build(parser.parse_args().plan)
