"""Publication-only contract fixture: predefined points, no fitted science."""
import csv
import copy
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
    def test_representative_default_preserves_full_cohort_and_legacy_point_opt_in(self):
        import numpy as np
        from snapshot import sha256
        from native_ca import write_ca
        from package_originals import package
        from publish_atlas import build, export_prepared, ordered_models_sha256
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
                    'terminal_detection':{'retained_reference_positions':list(range(4))},
                    'structural_flexibility':{'model_count':4,'method':'Saved fixture variance; no inference',
                        'residue_keys':[['A',str(j)] for j in range(4)],
                        'cores':[{'key':'C1','residue_keys':[['A','0'],['A','1']],'indices':[0,1]}]},
                    'clusters':[{'cluster_label':'Cluster 1','cluster_size':4,'color':'#4477AA','selected':[selected],
                    'multimodel_structure':str(ensemble),'selection_method':'confidence_ranked','shared_ca_keys':[['A',str(j)] for j in range(1,5)]}]})
                manifest=save(t+'/workflow.json',{'identifier':t,'analysis_status':'complete','clustering':cluster,
                    'knn_clustering':{'labels_csv':str(labels)},'cluster_ensembles':{'report':str(report)},
                    'terminal_detection':{'pair_mask_sha256':'b'*64,'excluded_indices':[],
                        'retained_reference_positions':list(range(4)),'input_residue_keys':[['A',str(j)] for j in range(4)]},
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
                grid=save(t+'/medoid-tm.json',{'labels':['Cluster 1'],'matrix':[[1.0]]})
                workflows[t]={'manifest':str(manifest),'manifest_sha256':sha256(manifest),'bonsai':descriptor(tree),'medoid_tm':descriptor(grid)}
                targets[t]={'title':t,'models':4,'residues':4,'sequence_sha256':hashlib.sha256(b'AAAA').hexdigest()}
            native=save('native.json',{'status':'validated','models':rows})
            confidence=save('confidence.json',{'status':'validated','models':conf,'source_manifest_sha256':sha256(native)})
            study=save('config/study.json',{'library_commit':renderer_identity()['revision'],'targets':targets,'snapshot':{'manifest_sha256':sha256(native)}})
            save('config/model-types.json',{'rules':[{'key':'afcluster','name':'AF-Cluster','color':'#4477AA'}]})
            context=save('context.json',{'retrieved_at_utc':'2026-11-04T00:00:00Z','targets':{t:{'protein':'Fixture protein','organism':'Fixture only',
                'modeled_canonical_range':[1,4],'canonical_length':4,'source':'https://example.org/fixture'} for t in targets}})
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
                self.assertEqual(data['coordinate_delivery'],'representatives')
                for field in ['model_files','model_sha256','model_native_sha256','model_structure_confidence','original_archive']:
                    self.assertNotIn(field,data)
                self.assertFalse((root/'site'/target['id']/'models').exists())
                self.assertIn('point selections show the corresponding cluster representative',data['representative_policy'])
                with (root/'site'/data['membership_file']).open() as stream:published=list(csv.DictReader(stream))
                expected={row['model_id']:row for row in rows if row['target']==target['id']}
                self.assertEqual(set(data['models']),set(expected));self.assertEqual(len(published),len(expected))
                for row in published:
                    original=expected[row['model']]
                    self.assertEqual(row['native_sha256'],original['sha256'])
                    self.assertEqual(row['native_physicality'],original['native_physicality'])
                    self.assertEqual(float(row['mean_plddt']),90.);self.assertEqual(float(row['ptm']),.7)
                    for k in range(16):self.assertEqual(float(row[f'energy_t{k:02d}']),original[f'energy_t{k:02d}'])
                native_manifest=json.loads(Path(workflows[target['id']]['manifest']).read_text())
                for method in ['pca','umap','tsne']:
                    with np.load(native_manifest['clustering'][method]['embeddings'],allow_pickle=False) as saved:
                        self.assertEqual(data['models'],saved['filenames'].tolist())
                        np.testing.assert_array_equal(data['projections'][method]['xy'],saved['embeddings'])
                self.assertNotIn(str(root),json.dumps(data))
                tree_file=root/'site'/data['bonsai']['file']
                self.assertEqual(data['bonsai']['sha256'],sha256(tree_file))
                tree=json.loads(tree_file.read_text());self.assertNotIn(str(root),json.dumps(tree))
                self.assertEqual(len(tree['models']),4)
                runtime=tree['provenance']['settings']['native_runtime_compatibility']
                self.assertEqual(runtime['implementation_sha256'],'c'*64)
                self.assertEqual(runtime['native_modules'][0]['tracked_patch_sha256'],'f'*64)
                self.assertEqual(data['provenance']['bonsai_native_export_sha256'],workflows[target['id']]['bonsai']['sha256'])
            # An explicit partial release must retain the same renderer and real views.
            partial=json.loads(plan.read_text());partial.update(targets={'TOY1':workflows['TOY1']},
                pending_targets=['TOY2','TOY3'],presentation_title='Fixture atlas · TOY1 ready · TOY2/TOY3 pending',
                source=str(root/'partial-delivery'),output=str(root/'partial-site'))
            partial_plan=save('partial-plan.json',partial);build(partial_plan)
            catalog=json.loads((root/'partial-site/atlas.json').read_text())
            self.assertEqual([x['id'] for x in catalog['targets']],['TOY1'])
            self.assertEqual([x['id'] for x in catalog['pending_targets']],['TOY2','TOY3'])
            self.assertFalse((root/'partial-site/TOY2').exists())
            self.assertIn('Fixture atlas · TOY1 ready · TOY2/TOY3 pending',(root/'partial-site/index.html').read_text())
            for name in ASSETS:
                if name != 'index.html':self.assertEqual((root/'partial-site'/name).read_bytes(),(root/'site'/name).read_bytes())
            # A saved HDBSCAN recipe adds an honestly distinct, curated default view.
            # The predefined labels exercise multiple groups and retained noise, with no fit.
            t='TOY1';old=json.loads(Path(workflows[t]['manifest']).read_text())
            with Path(old['knn_clustering']['labels_csv']).open() as stream:old_members=list(csv.DictReader(stream))
            density_labels=root/t/'density-labels.csv'
            with density_labels.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['model','source_path','cluster_label','color']);writer.writeheader()
                for i,row in enumerate(old_members):
                    writer.writerow({**row,'cluster_label':['D1','D1','D2','root.unassigned'][i],
                        'color':['#4477AA','#4477AA','#EE6677','#BBBBBB'][i]})
            report=json.loads(Path(old['cluster_ensembles']['report']).read_text());template=report['clusters'][0]
            report['clusters']=[]
            for i,(label,size,color) in enumerate([('D1',2,'#4477AA'),('D2',1,'#EE6677')]):
                selected=dict(template['selected'][0],model=old_members[i*2]['model'],source_path=old_members[i*2]['source_path'])
                ensemble=root/t/(label+'.cif');ensemble.write_bytes(Path(template['multimodel_structure']).read_bytes())
                report['clusters'].append(dict(template,cluster_label=label,cluster_size=size,color=color,
                    selected=[selected],multimodel_structure=str(ensemble)))
            density_report=save(t+'/density-report.json',report)
            ordered=[row['model'] for row in old_members]
            analysis=save(t+'/density-analysis.json',{'schema_version':1,'status':'complete','target':t,'algorithm':'HDBSCAN',
                'source_workflow_manifest_sha256':workflows[t]['manifest_sha256'],'model_count':len(ordered),'models':ordered,
                'model_order_sha256':ordered_models_sha256(ordered),'density':['D1','D1','D2','root.unassigned'],'method_note':'Frozen fixture labels; recursive recipe recorded, no fit in test.'})
            density_grid=save(t+'/density-medoid-tm.json',{'labels':['D1','D2'],'matrix':[[1.,.8],[.8,1.]]})
            density=dict(labels_csv=descriptor(density_labels),ensemble_report=descriptor(density_report),
                analysis=descriptor(analysis),bonsai=workflows[t]['bonsai'],medoid_tm=descriptor(density_grid))
            region_report=copy.deepcopy(json.loads(Path(old['cluster_ensembles']['report']).read_text()))
            region_proof={'kind':'saved_variance_presentation','parent_workflow_manifest_sha256':workflows[t]['manifest_sha256'],
                'parent_ensemble_report_sha256':sha256(old['cluster_ensembles']['report'])}
            region_report['structural_flexibility']['heatmap']={'domain_colors':{'C1':'#00BFFF'}}
            region_report['presentation_derivation']=region_proof
            region_report_file=save(t+'/saved-regions-report.json',region_report)
            region_manifest=copy.deepcopy(old);region_manifest['cluster_ensembles']['report']=str(region_report_file)
            region_manifest['presentation_derivation']=region_proof
            region_manifest_file=save(t+'/saved-regions-manifest.json',region_manifest)
            region_image=root/t/'saved-regions.png'
            # Predefined one-pixel image only: no scientific plotting or camera export.
            import struct, zlib
            def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
            region_image.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))
                +chunk(b'IDAT',zlib.compress(b'\x00\xff\xff\xff'))+chunk(b'IEND',b''))
            region_artifacts={'manifest':descriptor(region_manifest_file),'ensemble_report':descriptor(region_report_file),
                'image':descriptor(region_image)}
            dual=dict(partial,targets={t:dict(workflows[t],density_artifacts=density,region_artifacts=region_artifacts)},
                source=str(root/'dual-delivery'),output=str(root/'dual-site'))
            prepared=build(save('dual-plan.json',dual),export_site=False)
            self.assertFalse((root/'dual-site').exists())
            binding=prepared['target_views'][t]
            self.assertEqual(binding['models'],ordered);self.assertEqual(binding['model_order_sha256'],ordered_models_sha256(ordered))
            self.assertEqual(binding['clustering_positions'],[1,2,3,4])
            for key in ['primary','density']:
                view=binding['views'][key]
                self.assertEqual(view['source_data_sha256'],sha256(root/'dual-delivery'/view['data']))
            ranges=json.loads((root/'dual-delivery/residue-ranges.json').read_text())['targets'][t]['residue_ranges'][0]
            self.assertEqual(ranges['default_view'],'density')
            self.assertEqual({v['id'] for v in ranges['views']},{'primary','density'})
            scopes=json.loads((root/'dual-delivery/clustering-residues.json').read_text())['targets'][t]['views']
            self.assertEqual(set(scopes),{'primary','density'})
            self.assertTrue(scopes['density']['regions_verified'])
            self.assertEqual(scopes['density']['regions'][0],{'id':'C1','label':'Detected rigid core 1','start':1,'end':2,'color':'#00BFFF'})
            self.assertEqual(scopes['primary']['regions_source_models'],4)
            self.assertEqual(scopes['primary']['region_provenance']['parent_workflow_manifest_sha256'],workflows[t]['manifest_sha256'])
            primary=json.loads((root/'dual-delivery'/t/'primary.json').read_text())
            displayed=json.loads((root/'dual-delivery'/t/'density.json').read_text())
            self.assertEqual(primary['view_id'],'primary');self.assertEqual(displayed['view_id'],'density')
            self.assertEqual(displayed['models'],primary['models']);self.assertEqual(displayed['projections'],primary['projections'])
            self.assertEqual(displayed['clusters'],['D1','D1','D2','root.unassigned'])
            self.assertEqual([r['cluster'] for r in displayed['representatives']],['D1','D2'])
            self.assertEqual(displayed['medoid_tm']['labels'],['D1','D2'])
            self.assertNotEqual(primary['representatives'][0]['file'],displayed['representatives'][0]['file'])
            self.assertEqual(displayed['provenance']['density_analysis_sha256'],sha256(analysis))
            self.assertEqual(displayed['provenance']['density_diagnostics']['status'],'legacy_not_recorded')
            self.assertEqual(displayed['provenance']['bonsai_scope'],'Fixture only')
            self.assertNotIn('model_files',displayed);self.assertFalse((root/'dual-delivery'/t/'models').exists())
            with (root/'dual-delivery'/displayed['membership_file']).open() as stream:density_members=list(csv.DictReader(stream))
            self.assertEqual(len(density_members),4);self.assertEqual(density_members[-1]['cluster'],'root.unassigned')
            self.assertNotIn(str(root),json.dumps(displayed))
            self.assertIn('Saved full-cohort distance variance and detected rigid cores',{r['name'] for r in displayed['heatmaps']})
            # The controller can attach separately verified saved-state artifacts before one export.
            panel=root/'dual-delivery'/t/'verified-panel.json';panel.write_text('{"fixture":"saved panel receipt"}\n')
            export_prepared(prepared)
            app_hash=sha256(root/'dual-site/app.js')[:12]
            self.assertIn('app.js?v='+app_hash,(root/'dual-site/index.html').read_text())
            allowlist=json.loads(Path(prepared['publication_manifest']).read_text())['files']
            self.assertIn(t+'/verified-panel.json',{row['path'] for row in allowlist})
            self.assertTrue((root/'dual-site'/t/'verified-panel.json').is_file())
            # A hash-bound analysis with a different order is never relabelled as this cohort.
            changed=json.loads(analysis.read_text());changed['models']=list(reversed(ordered))
            wrong=save(t+'/wrong-order-analysis.json',changed)
            bad=dict(dual,source=str(root/'bad-density-delivery'),output=str(root/'bad-density-site'),
                targets={t:dict(workflows[t],density_artifacts=dict(density,analysis=descriptor(wrong)))})
            with self.assertRaisesRegex(ValueError,'HDBSCAN analysis is not bound'):
                build(save('bad-density-plan.json',bad),export_site=False)
            altered=copy.deepcopy(region_manifest);altered['terminal_detection']['excluded_indices']=[0]
            altered_file=save(t+'/changed-region-manifest.json',altered)
            bad_region=dict(dual,source=str(root/'bad-region-delivery'),output=str(root/'bad-region-site'),
                targets={t:dict(workflows[t],density_artifacts=density,
                    region_artifacts=dict(region_artifacts,manifest=descriptor(altered_file)))})
            with self.assertRaisesRegex(ValueError,'region presentation changed frozen scientific results'):
                build(save('bad-region-plan.json',bad_region),export_site=False)
            # Historical all-point publications remain available only by explicit opt-in.
            originals_plan=save('originals-plan.json',{'native_manifest':str(native),'native_manifest_sha256':sha256(native),
                'intakes':intakes,'output':str(root/'originals'),'release_url':'https://github.com/example/fixture/releases/download/test'})
            package(originals_plan)
            archives=json.loads((root/'originals/completion.json').read_text())['targets']
            legacy=json.loads(plan.read_text())
            legacy.update(structure_delivery='all-point-ca',targets={
                target:{**frozen,'original_archive':archives[target]} for target,frozen in workflows.items()},
                source=str(root/'legacy-delivery'),output=str(root/'legacy-site'))
            build(save('legacy-plan.json',legacy))
            for target in legacy['targets']:
                data=json.loads((root/'legacy-site'/target/'primary.json').read_text())
                self.assertEqual(data['coordinate_delivery'],'all-point-ca')
                self.assertEqual(len(data['model_files']),4)
                self.assertEqual(data['original_archive']['models'],4)
                for i,file in enumerate(data['model_files']):
                    decoded=gzip.decompress((root/'legacy-site'/file).read_bytes())
                    self.assertEqual(hashlib.sha256(decoded).hexdigest(),data['model_sha256'][i])
            missing_archive={**legacy,'targets':workflows,'source':str(root/'missing-archive-delivery'),
                             'output':str(root/'missing-archive-site')}
            with self.assertRaisesRegex(ValueError,'verified complete original archive'):
                build(save('missing-archive-plan.json',missing_archive))



if __name__=='__main__':unittest.main()
