"""Static arrays/opaque original bytes and mocked engines only; no target fits."""
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import target_continuation as stage

NUMPY = importlib.util.find_spec('numpy') is not None


def put(path, value):
    path.write_text(json.dumps(value))
    return stage.record(path)


def cohort(root):
    originals = root/'originals'; originals.mkdir()
    names = ['m'+str(i) for i in range(4)]
    rows = []
    for name in names:
        p=originals/(name+'.pdb'); p.write_text('opaque original '+name)
        rows.append(dict(model_id=name, path=str(p), sha256=stage.sha256(p)))
    descriptor = dict(target='toy', model_count=4,
        axes=put(root/'axes.json', dict(names=names)))
    native = dict(status='validated', models=[dict(model_id=r['model_id'],
        target='toy', sha256=r['sha256']) for r in rows])
    intake = dict(target='toy', models=rows, analysis_dir=str(originals))
    return descriptor, intake, native


class RegistryTests(unittest.TestCase):
    def test_original_bytes_and_order_are_verified_without_atom_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); descriptor, intake, native=cohort(root)
            intake['models'].reverse(); native['models'].reverse()
            boundaries=[]
            members=stage.registry(descriptor,intake,native,checkpoint=boundaries.append)
            self.assertEqual([r['model'] for r in members], ['m0','m1','m2','m3'])
            self.assertEqual(boundaries, ['toy:original-hashes-closed'])
            self.assertTrue(all(Path(r['source_path']).read_text().startswith('opaque') for r in members))

    def test_changed_coordinate_duplicate_or_different_cohort_stops(self):
        for corruption in ('bytes','duplicate','target','symlink'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); descriptor,intake,native=cohort(root)
                if corruption=='bytes':Path(intake['models'][0]['path']).write_text('changed')
                if corruption=='duplicate':intake['models'][-1]=intake['models'][0]
                if corruption=='target':native['models'][0]['target']='other'
                if corruption=='symlink':
                    p=Path(intake['models'][0]['path']);p.rename(p.with_suffix('.original'))
                    p.symlink_to(p.with_suffix('.original'))
                with self.assertRaises(ValueError):stage.registry(descriptor,intake,native)


@unittest.skipUnless(NUMPY,'Optional NumPy for static-array adapter checks')
class DensityTests(unittest.TestCase):
    def fixture(self, root, *, dtype='float16', order=None):
        import numpy as np
        descriptor,intake,native=cohort(root)
        members=stage.registry(descriptor,intake,native)
        report=put(root/'terminal.json',dict(retained_reference_positions=[2,3,4],pair_mask_sha256='a'*64))
        descriptor['terminal_report']=report
        labels=root/'labels.csv'
        with labels.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['model','source_path','cluster_label'])
            writer.writeheader()
            for i in order or range(4):writer.writerow(dict(model=members[i]['model'],source_path=members[i]['source_path'],cluster_label='C1'))
        manifest=put(root/'manifest.json',dict(knn_clustering=dict(labels_csv=str(labels))))
        matrix=np.arange(12,dtype=dtype).reshape(4,3)
        p=root/'retained.npy';np.save(p,matrix,allow_pickle=False)
        self.calls=[]
        def digest(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
        @contextmanager
        def workspace(a,**kw):
            self.assertEqual(kw,dict(dtype=np.float32))
            yield a.astype(np.float32)
        def run(a,names,groups,output,**kw):
            self.calls.append(kw)
            self.assertEqual(a.dtype,np.float32)
            self.assertEqual(kw['retained_positions'],[2,3,4])
            self.assertEqual(kw['mode'],'casp17_recursive')
            self.assertEqual(kw['expected_feature_sha256'],digest(a))
            Path(output).mkdir()
            return dict(models=names,density=['C1']*4,UMAP_fits=3,HDBSCAN_extractions=6)
        def labels_fn(result,rows,output,**kw):
            self.assertEqual(kw,dict(maximum_clusters=4))
            a=Path(output)/'labels.csv';b=Path(output)/'assigned-labels.csv'
            a.write_text('saved labels');b.write_text('saved assigned labels')
            return a,b
        primary=dict(manifest=manifest,retained_features=dict(path=str(p),
            shape=[4,3],dtype=dtype,file_sha256=stage.sha256(p),feature_sha256=digest(matrix)))
        primary['manifest']=put(root/'manifest.json',dict(knn_clustering=dict(labels_csv=str(labels)),
            terminal_detection=stage.read(report),derivation=dict(accepted_terminal_report=report,
                retained_feature_proof=dict(primary['retained_features']))))
        adapters=dict(workspace=workspace,digest=digest,run=run,labels=labels_fn)
        return primary,members,descriptor,adapters

    def execute(self, root, data):
        primary,members,descriptor,adapters=data
        return stage.run_density(primary,members,descriptor,{},root/'density',
            workers=4,check=lambda:None,checkpoint=lambda _:None,log=lambda _:None,adapters=adapters)

    def test_trimmed_feature_proof_and_default_labels_preserve_complete_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root)
            receipt=self.execute(root,data)
            result=stage.read(receipt)
            self.assertEqual(result['models'],['m0','m1','m2','m3'])
            self.assertTrue(result['default_cluster_labels'])
            self.assertFalse(result['public_coordinates_exported'])
            self.assertFalse(result['representatives_selected'])
            self.assertEqual(len(self.calls),1)
            self.assertTrue(Path(data[0]['retained_features']['path']).exists())

    def test_tampered_features_precision_labels_or_fit_budget_never_emit_acceptance(self):
        for corruption in ('file','values','precision','order','fits','names','mask','handoff'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data=self.fixture(root,dtype='float32' if corruption=='precision' else 'float16',
                    order=[1,0,2,3] if corruption=='order' else None)
                primary,members,descriptor,adapters=data
                if corruption=='file':Path(primary['retained_features']['path']).write_bytes(b'changed')
                if corruption=='values':primary['retained_features']['feature_sha256']='0'*64
                if corruption=='mask':
                    descriptor['terminal_report']=put(root/'other-mask.json',dict(
                        retained_reference_positions=[3,4,5],pair_mask_sha256='b'*64))
                if corruption=='handoff':
                    current=stage.read(primary['manifest'])
                    current['derivation']['retained_feature_proof']['feature_sha256']='0'*64
                    primary['manifest']=put(root/'manifest.json',current)
                if corruption in ('fits','names'):
                    original=adapters['run']
                    def bad(*args,**kwargs):
                        result=original(*args,**kwargs)
                        if corruption=='fits':result['UMAP_fits']=16
                        if corruption=='names':result['models']=list(reversed(result['models']))
                        return result
                    adapters['run']=bad
                with self.assertRaises(ValueError):self.execute(root,data)
                self.assertFalse((root/'density/density-receipt.json').exists())


class LabelWriterTests(unittest.TestCase):
    def test_noise_all_points_and_inherited_small_groups_are_preserved(self):
        from rbp_parity import write_density_labels
        fake = SimpleNamespace(lineage_palette=lambda labels:{s:'#123456' for s in labels})
        modules = {'src':ModuleType('src'),'src.analysis':ModuleType('src.analysis'),
                   'src.analysis.hierarchical_colors':fake}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,modules):
            root=Path(tmp); names=['a','b','c']; labels=['C1','C1.unassigned','C2']
            rows=[dict(model=s,source_path='/'+s,umap_x='1',umap_y='2') for s in names]
            all_points,assigned=write_density_labels(dict(models=names,density=labels),rows,root,
                maximum_clusters=len(names))
            with all_points.open() as f:complete=list(csv.DictReader(f))
            with assigned.open() as f:selected=list(csv.DictReader(f))
            self.assertEqual([r['model'] for r in complete],names)
            self.assertEqual([r['cluster_label'] for r in complete],labels)
            self.assertEqual([r['model'] for r in selected],['a','c'])
        for invalid in ('labels','rows','order'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,modules):
                root=Path(tmp);rows=[dict(model=s,source_path='/'+s) for s in names]
                current=list(labels)
                if invalid=='labels':current.pop()
                if invalid=='rows':rows.pop()
                if invalid=='order':rows.reverse()
                with self.assertRaisesRegex(ValueError,'population/order'):
                    write_density_labels(dict(models=names,density=current),rows,root)
                self.assertEqual(list(root.iterdir()),[])


class LifecycleTests(unittest.TestCase):
    def test_unsafe_target_never_creates_outputs_or_seeds_metrics(self):
        from unittest.mock import Mock
        for target in ('../escape','/absolute','unsafe/child','..'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);seed=Mock()
                with self.assertRaisesRegex(ValueError,'Unsafe target'):
                    stage.run_target(dict(target=target),{},root,{},check=lambda:None,
                        checkpoint=lambda _:None,adapters=dict(seed=seed))
                seed.assert_not_called()
                self.assertEqual(list(root.iterdir()),[])

    def test_disposable_arrays_retire_before_acceptance_even_if_density_fails(self):
        for fail in (False,True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);descriptor,intake,native=cohort(root)
                native_entry=put(root/'native.json',native)
                intake['source_manifest_sha256']=native_entry['sha256']
                descriptor.update(intake=put(root/'intake.json',intake),
                    model_type_config=put(root/'types.json',{}),cluster_config={},terminal_report={})
                ram=root/'ram';ram.mkdir()
                plan=dict(native=native_entry,confidence=put(root/'confidence.json',{}),
                    environment=dict(TMPDIR=str(ram)),native_density={},resources=dict(maximum_cpus=4))
                old=root/'previous-projections.npz';old.write_bytes(b'preserve previous results')
                def seed(*args):Path(args[-1]).write_bytes(b'fresh metrics')
                def primary(descriptor,report,metrics,output,scratch,**kwargs):
                    output.mkdir()
                    retained=scratch/'retained.npy';retained.write_bytes(b'disposable verified features')
                    durable=output/'feature-cache.npz';durable.write_bytes(b'durable features')
                    return dict(retained_features=dict(path=str(retained)),feature_cache=stage.record(durable),
                        manifest=put(output/'manifest.json',dict(analysis_status='complete')))
                def density(primary,*args,**kwargs):
                    self.assertTrue(Path(primary['retained_features']['path']).exists())
                    if fail:raise RuntimeError('injected fit failure')
                    return put(root/'density-receipt.json',dict(status='complete'))
                kwargs=dict(check=lambda:None,checkpoint=lambda _:None,
                    adapters=dict(seed=seed,primary=primary,density=density))
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        stage.run_target(descriptor,plan,root,native,**kwargs)
                    self.assertFalse((root/'toy/target-stage-receipt.json').exists())
                else:
                    result=stage.read(stage.run_target(descriptor,plan,root,native,**kwargs))
                    self.assertTrue(result['primary']['retained_features']['retired_after_density'])
                    self.assertEqual(result['model_selection_status'],
                        'pending separately frozen plan using actual cluster counts')
                    self.assertEqual(result['molecular_exports'],0)
                self.assertFalse((ram/'toy/retained.npy').exists())
                self.assertTrue((root/'toy/primary/feature-cache.npz').exists())
                self.assertEqual(old.read_bytes(),b'preserve previous results')


class ApprovalTests(unittest.TestCase):
    def test_unapproved_or_changed_stage_cannot_reach_resource_checks(self):
        for invalid in ('approval','scope','target'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                plan=dict(operation='trimmed-primary-and-density',settings=dict(stage.SCOPE),
                    targets=[dict(target='a'),dict(target='b')],resources={})
                if invalid=='scope':plan['settings']['native_TM_calls']=1
                if invalid=='target':plan['targets'][1]['target']='a'
                p=root/'plan.json';p.write_text(json.dumps(plan))
                put(root/'approval.json',dict(approved_by='user',approved_plan_sha256=
                    '0'*64 if invalid=='approval' else stage.sha256(p)))
                with patch('run_reviewed_plan.verify_controls') as controls:
                    with self.assertRaises(ValueError):stage.approved(p)
                    controls.assert_not_called()


if __name__=='__main__':unittest.main()
