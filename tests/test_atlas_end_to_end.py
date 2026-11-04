"""Publication-only contract fixture: predefined points, no fitted science."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('structure_clustering'),
                     'Saved export integration uses the existing optional runtime')
class EndToEndPublication(unittest.TestCase):
    def test_shared_export_has_all_real_saved_views_and_exact_originals(self):
        import numpy as np
        from snapshot import sha256
        from native_ca import write_ca
        from package_originals import package
        from publish_atlas import build
        from src.analysis.atlas_presentation import renderer_identity, ASSETS, TEMPLATES
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def save(name,data):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(data));return path
            def descriptor(path):return {'path':str(path),'sha256':sha256(path)}
            rows=[];conf=[];intakes={};workflows={};targets={}
            for t in ['TOY1','TOY2','TOY3']:
                folder=root/t;folder.mkdir();models=[]
                for i in range(4):
                    name=f'{t}__afcluster__{i:04d}';path=folder/(name+'.pdb')
                    path.write_text(''.join(f'ATOM  {j:5d}  CA  ALA A{j-1:4d}    {float(j):8.3f}{float(j%2):8.3f}{float(j%3):8.3f}  1.00 90.00           C\n' for j in range(1,5))+'END\n')
                    row={'model_id':name,'target':t,'method':'afcluster','path':str(path),'sha256':sha256(path),
                         'native_physicality':'retained','energy_t010':float(i),**{f'energy_t{k:02d}':float(i+k) for k in range(16)}}
                    rows.append(row);models.append({'model_id':name,'path':str(path),'sha256':row['sha256']})
                    conf.append({'model_id':name,'coordinate_sha256':row['sha256'],'mean_plddt':90.,'ptm':.7})
                intakes[t]=descriptor(save(t+'/intake.json',{'target':t,'models':models}))
                names=[r['model_id'] for r in models]
                cluster={}
                for method in ['pca','umap','tsne']:
                    path=folder/(method+'.npz');np.savez(path,filenames=names,embeddings=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]))
                    cluster[method]={'embeddings':str(path)}
                labels=folder/'labels.csv'
                with labels.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['model','source_path','cluster_label','color']);writer.writeheader()
                    writer.writerows({'model':r['model_id'],'source_path':r['path'],'cluster_label':'Cluster 1','color':'#4477AA'} for r in models)
                first=models[0];display=folder/'temporary.cif.gz';write_ca(first['path'],display,native_sha256=first['sha256'],expected_count=4)
                ensemble=folder/'Cluster_1.cif';ensemble.write_bytes(gzip.decompress(display.read_bytes()))
                selected={'model':first['model_id'],'source_path':first['path'],'state':1,'rank':1,'confidence':90.,'aligned_ca_rmsd':0.}
                report=save(t+'/report.json',{'residue_correspondence':{'native_residue_labels_preserved':True},'confidence_metric':'mean_plddt',
                    'clusters':[{'cluster_label':'Cluster 1','cluster_size':4,'color':'#4477AA','selected':[selected],
                    'multimodel_structure':str(ensemble),'selection_method':'confidence_ranked','shared_ca_keys':[['A',str(j)] for j in range(1,5)]}]})
                manifest=save(t+'/workflow.json',{'identifier':t,'analysis_status':'complete','clustering':cluster,
                    'knn_clustering':{'labels_csv':str(labels)},'cluster_ensembles':{'report':str(report)},
                    'terminal_detection':{'pair_mask_sha256':'b'*64,'excluded_indices':[],'retained_reference_positions':list(range(4))},
                    'visualize_distogram':{'images':[]}})
                # Native Bonsai export carries private runtime locations in settings.
                # Publication must preserve its geometry/hashes, with public-byte hashes.
                tree=save(t+'/native-bonsai.json',{'schema_version':1,
                    'models':[{'model':name,'vertex':i,'bonsai_group':'B1'} for i,name in enumerate(names)],
                    'layouts':[{'key':'fixture','title':'Fixture only','nodes':[[0.,0.],[1.,0.],[0.,1.],[1.,1.]],'edges':[[0,1],[1,2],[2,3]]}],
                    'method':'Saved fixture geometry; no fit','scope':'Fixture only','group_count':1,
                    'provenance':{**{key:'a'*64 for key in ['result_sha256','hdf_sha256','tree_sha256','model_order_sha256','source_features_sha256','pair_mask_sha256']},
                        'retained_reference_positions':list(range(4)),'feature_count':6,
                        'settings':{'schema_version':2,'max_components':None,'rescale_by_var':False,'use_knn':-1,
                            'grouping':'native','input_archive':str(root/'private-input'),
                            'native_runtime_compatibility':{'schema_version':1,'repair':'native_guards',
                                'native_helper':str(root/'private-helper.py'),'native_helper_sha256':'b'*64,
                                'implementation':str(root/'private-adapter.py'),'implementation_sha256':'c'*64,'external_files_modified':False,
                                'native_modules':[{'module':'native.module','source_path':str(root/'private-source.py'),
                                    'tracked_patch':str(root/'private-patch.diff'),'source_sha256':'d'*64,
                                    'runtime_source_sha256':'e'*64,'tracked_patch_sha256':'f'*64}]}}}})
                workflows[t]={'manifest':str(manifest),'manifest_sha256':sha256(manifest),'bonsai':descriptor(tree)}
                targets[t]={'title':t,'models':4,'residues':4,'sequence_sha256':hashlib.sha256(b'AAAA').hexdigest()}
            native=save('native.json',{'status':'validated','models':rows})
            confidence=save('confidence.json',{'status':'validated','models':conf,'source_manifest_sha256':sha256(native)})
            study=save('config/study.json',{'library_commit':renderer_identity()['revision'],'targets':targets,'snapshot':{'manifest_sha256':sha256(native)}})
            save('config/model-types.json',{'rules':[{'key':'afcluster','name':'AF-Cluster','color':'#4477AA'}]})
            context=save('context.json',{'retrieved_at_utc':'2026-11-04T00:00:00Z','targets':{t:{'protein':'Fixture protein','organism':'Fixture only',
                'modeled_canonical_range':[1,4],'canonical_length':4,'source':'https://example.org/fixture'} for t in targets}})
            originals_plan=save('originals-plan.json',{'native_manifest':str(native),'native_manifest_sha256':sha256(native),
                'intakes':intakes,'output':str(root/'originals'),'release_url':'https://github.com/example/fixture/releases/download/test'})
            package(originals_plan)
            archives=json.loads((root/'originals/completion.json').read_text())['targets']
            for t in workflows:workflows[t]['original_archive']=archives[t]
            vendor={}
            for name in ['plotly.min.js','3Dmol-min.js']:
                path=root/name;path.write_text('/* contract fixture only */');vendor[name]=descriptor(path)
            plan=save('publication-plan.json',{'renderer_commit':renderer_identity()['revision'],'study':descriptor(study),
                'native':descriptor(native),'confidence':descriptor(confidence),'construct_context':descriptor(context),
                'targets':workflows,'vendor_assets':vendor,'source':str(root/'delivery'),'output':str(root/'site')})
            build(plan)
            for name,source in ASSETS.items():
                if name!='index.html':self.assertEqual((root/'site'/name).read_bytes(),(TEMPLATES/source).read_bytes())
            catalog=json.loads((root/'site/atlas.json').read_text());self.assertEqual(len(catalog['targets']),3)
            for target in catalog['targets']:
                data=json.loads((root/'site'/target['data']).read_text())
                self.assertEqual(len(data['models']),4);self.assertEqual(set(data['projections']),{'pca','umap','tsne'})
                self.assertEqual(data['representatives'][0]['ensemble_format'],'cif')
                self.assertEqual(data['original_archive']['models'],4)
                self.assertNotIn(str(root),json.dumps(data))
                tree_file=root/'site'/data['bonsai']['file']
                self.assertEqual(data['bonsai']['sha256'],sha256(tree_file))
                tree=json.loads(tree_file.read_text());self.assertNotIn(str(root),json.dumps(tree))
                self.assertEqual(len(tree['models']),4)
                runtime=tree['provenance']['settings']['native_runtime_compatibility']
                self.assertEqual(runtime['implementation_sha256'],'c'*64)
                self.assertEqual(runtime['native_modules'][0]['tracked_patch_sha256'],'f'*64)
                self.assertEqual(data['provenance']['bonsai_native_export_sha256'],workflows[target['id']]['bonsai']['sha256'])


if __name__=='__main__':unittest.main()
