"""Saved-publication orchestration only: mocks never fit or deploy science."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rbp_publish as module
from publish_atlas import refresh_manifest
from rbp_panel_exports import record


class WrapperTests(unittest.TestCase):
    def fixture(self, root, *, states=True):
        config=root/'config';config.mkdir()
        labels=config/'model-types.json';labels.write_text(json.dumps({'rules':[{'key':'native','name':'Human label'}]}))
        study=config/'study.json';study.write_text('{}')
        catalog=root/'catalog.json';catalog.write_text('{}')
        scores=root/'scores.npz';scores.write_bytes(b'saved fixture only')
        descriptor=lambda p:{k:v for k,v in record(p).items() if k!='bytes'}
        saved={'path':'not read by mocked build','sha256':'d'*64}
        target=dict(manifest='not read',manifest_sha256='e'*64,bonsai=saved,medoid_tm=saved,
            density_artifacts=dict(labels_csv=saved,ensemble_report=saved,analysis=saved,bonsai=saved,medoid_tm=saved),
            region_artifacts=dict(manifest=saved,ensemble_report=saved,image=saved))
        plan=dict(targets={module.TARGET:target},source=str(root/'source'),output=str(root/'site'),
            study=descriptor(study),frozen_files={str(labels):record(labels)['sha256']},require_rbp_parity=states)
        if states:
            plan['state_artifacts']=dict(catalog=descriptor(catalog),scores=descriptor(scores),
                source_bindings=dict(plan_sha256='a'*64,consumer_commit='b'*40),plan_sha256='a'*64,source_commit='b'*40)
        path=root/'plan.json'
        def save():path.write_text(json.dumps(plan));return record(path)['sha256']
        digest=save();events=[]
        def build(path, *, export_site):
            self.assertFalse(export_site);events.append('build')
            source=Path(plan['source']);source.mkdir();views={}
            for view in ['primary','density']:
                route=view+'.json';(source/route).write_text(json.dumps(dict(target=module.TARGET,view_id=view,models=['m0','m1'])))
                views[view]=dict(id=view,source_view_id=view,data=route,source_data_sha256=record(source/route)['sha256'])
            manifest=root/'publication-manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,files=[])))
            return dict(source=str(source),output=plan['output'],publication_manifest=str(manifest),
                renderer_commit='c'*40,coordinate_delivery='representatives',
                target_views={module.TARGET:dict(models=['m0','m1'],sequence_sha256='f'*64,views=views)})
        def attach(source, catalog, scores, views, **kwargs):
            events.append('attach');self.assertEqual(kwargs['method_labels'],{'native':'Human label'})
            self.assertEqual(catalog['bytes'],2);self.assertIn('bytes',scores)
            self.assertEqual(set(views),{'primary','density'})
            entry=dict(model_count=2,ranges={'all':dict(views={view:dict(available=False,state_count=0) for view in views})})
            public=dict(schema_version=2,kind='tm_state_selection_catalog',targets={module.TARGET:entry})
            path=Path(source)/'tm-states.json';path.write_text(json.dumps(public))
            row=record(path);row['path']='tm-states.json'
            return dict(files=[row],tm_state_selection_catalog=public,target_entry=entry)
        def export(prepared):
            events.append('export');refresh_manifest(prepared)
            output=Path(prepared['output']);output.mkdir()
            for row in module.inventory(prepared['source']):
                dst=output/row['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(prepared['source'])/row['path'],dst)
            (output/'index.html').write_text('Shared-renderer boundary fixture only')
            (output/'.nojekyll').write_bytes(b'')
            receipt=dict(input_manifest_sha256=record(prepared['publication_manifest'])['sha256'],
                scientific_execution=False,files=module.inventory(output))
            (output/'export-manifest.json').write_text(json.dumps(receipt))
        return dict(plan=plan,path=path,digest=digest,save=save,build=build,attach=attach,export=export,events=events)

    def run_wrapper(self,f,**overrides):
        with patch.object(module,'build',side_effect=overrides.get('build',f['build'])) as build,\
             patch.object(module,'attach',side_effect=overrides.get('attach',f['attach'])) as attach,\
             patch.object(module,'export_prepared',side_effect=overrides.get('export',f['export'])) as export:
            result=module.publish(f['path'],expected_plan_sha256=f['digest'])
            return result,build,attach,export

    def test_prepares_attaches_and_exports_once_with_exact_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(Path(tmp));result,build,attach,export=self.run_wrapper(f)
            self.assertEqual(f['events'],['build','attach','export']);export.assert_called_once()
            build.assert_called_once_with(str(f['path']),export_site=False)
            receipt=json.loads(Path(result['receipt']['path']).read_text())
            self.assertFalse(receipt['scientific_execution']);self.assertFalse(receipt['public_deployment'])
            self.assertEqual(receipt['publication_plan']['sha256'],f['digest'])
            self.assertEqual(receipt['state_source']['plan_sha256'],'a'*64)
            self.assertEqual(receipt['method_label_rules']['sha256'],f['plan']['frozen_files'][str(Path(tmp)/'config/model-types.json')])
            self.assertEqual(receipt['state_target']['ranges']['all']['views']['density']['state_count'],0)
            self.assertFalse(receipt['state_target']['ranges']['all']['views']['density']['available'])
            self.assertEqual(receipt['site_files'],module.inventory(f['plan']['output']))
            encoded=json.dumps(receipt['site_files'],separators=(',',':'),ensure_ascii=False).encode()
            self.assertEqual(receipt['site_sha256'],hashlib.sha256(encoded).hexdigest())
            manifest=json.loads(Path(result['prepared']['publication_manifest']).read_text())
            self.assertTrue(set(row['path'] for row in receipt['attached_files']) <= {r['path'] for r in manifest['files']})

    def test_baseline_publication_remains_supported_without_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(Path(tmp),states=False);result,build,attach,export=self.run_wrapper(f)
            self.assertEqual(f['events'],['build','export']);attach.assert_not_called();export.assert_called_once()
            receipt=json.loads(Path(result['receipt']['path']).read_text());self.assertIsNone(receipt['state_source'])

    def test_changed_plan_is_refused_before_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(Path(tmp));f['path'].write_text('{}')
            with patch.object(module,'build') as build,self.assertRaisesRegex(ValueError,'hash/size'):
                module.publish(f['path'],expected_plan_sha256=f['digest'])
            build.assert_not_called()

    def test_parity_requires_the_actual_saved_components(self):
        for missing in ['state','density','bonsai','medoid','regions','all-points','aliases']:
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as tmp:
                f=self.fixture(Path(tmp));target=f['plan']['targets'][module.TARGET]
                if missing=='state':del f['plan']['state_artifacts']
                elif missing=='density':del target['density_artifacts']
                elif missing=='bonsai':del target['bonsai']
                elif missing=='medoid':del target['density_artifacts']['medoid_tm']
                elif missing=='regions':del target['region_artifacts']
                elif missing=='aliases':f['plan']['frozen_files']={}
                else:f['plan']['structure_delivery']='all-point-ca'
                f['digest']=f['save']()
                with patch.object(module,'build') as build,self.assertRaises(ValueError):
                    module.publish(f['path'],expected_plan_sha256=f['digest'])
                build.assert_not_called()

    def test_converter_cannot_change_a_raw_scientific_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(Path(tmp))
            def changed(*args,**kwargs):
                result=f['attach'](*args,**kwargs);(Path(args[0])/'density.json').write_text('{}');return result
            with patch.object(module,'export_prepared') as export,self.assertRaisesRegex(ValueError,'scientific JSON changed'):
                self.run_wrapper(f,attach=changed)
            export.assert_not_called();self.assertFalse(Path(f['plan']['output']).exists())

    def test_converter_file_receipt_cannot_hide_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(Path(tmp))
            def changed(*args,**kwargs):
                result=f['attach'](*args,**kwargs);result['files'][0]['sha256']='0'*64;return result
            with self.assertRaisesRegex(ValueError,'returned allowlist'):self.run_wrapper(f,attach=changed)
            self.assertNotIn('export',f['events'])

    def test_final_manifest_must_include_every_attached_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(Path(tmp))
            def broken(prepared):
                f['export'](prepared)
                path=Path(prepared['publication_manifest']);manifest=json.loads(path.read_text())
                manifest['files']=[r for r in manifest['files'] if r['path']!='tm-states.json'];path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'source file allowlist'):self.run_wrapper(f,export=broken)
            self.assertFalse((Path(tmp)/'site-receipt.json').exists())

    def test_private_or_existing_receipt_destinations_are_not_overwritten(self):
        for invalid in ['inside','exists']:
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as tmp:
                f=self.fixture(Path(tmp))
                path=Path(f['plan']['source'])/'receipt.json' if invalid=='inside' else Path(tmp)/'existing.json'
                if invalid=='exists':path.write_text('preserve me')
                f['plan']['receipt']=str(path);f['digest']=f['save']()
                with patch.object(module,'build') as build,self.assertRaisesRegex(ValueError,'new path outside'):
                    module.publish(f['path'],expected_plan_sha256=f['digest'])
                build.assert_not_called()
                if invalid=='exists':self.assertEqual(path.read_text(),'preserve me')


if __name__=='__main__':unittest.main()
