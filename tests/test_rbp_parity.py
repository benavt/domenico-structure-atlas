"""Source-only identity, allocation and approval checks; no native fits/exports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import rbp_parity as adapter

AVAILABLE=importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('src') is not None

class ApprovalTests(unittest.TestCase):
    def test_changed_plan_never_reaches_native_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=Path(tmp)/'execution-plan.json'; plan.write_text('{"changed":true}')
            (plan.parent/'approval.json').write_text(json.dumps(dict(approved_by='user',approved_plan_sha256='0'*64)))
            with self.assertRaisesRegex(ValueError,'Exact execution approval'):
                adapter.validate_approval(plan)

@unittest.skipUnless(AVAILABLE,'Optional saved-array/library runtime')
class FrozenFeatureTests(unittest.TestCase):
    def fixture(self,root,order=None):
        import numpy as np
        names=['first','second'];matrix=np.arange(2*36585,dtype=np.float32).reshape(2,36585)/4
        matrix=matrix.astype(np.float16)
        detection=dict(pair_mask_sha256='f'*64)
        manifest=dict(terminal_detection=detection,knn_clustering=dict(feature_input=dict(sha256=hashlib.sha256(matrix.tobytes()).hexdigest())))
        source=Path(root)/'saved.npz'
        np.savez_compressed(source,flattened=matrix,reference_positions=np.arange(271),requested_reference_positions=np.arange(271),
             source_signature=json.dumps(dict(files=[[str(Path(root)/(n+'.bin')),0,0] for n in (order or names)])),
             terminal_detection=json.dumps(detection),filenames=np.array(['must-not-deserialize'],dtype=object))
        entry=dict(path=str(source),sha256=adapter.sha256(source),maximum_npy_bytes=matrix.nbytes+1024)
        return entry,manifest,names,matrix

    def test_uses_numeric_member_and_json_order_without_loading_object_metadata(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            entry,manifest,names,matrix=self.fixture(tmp)
            with adapter.frozen_features(entry,manifest,names,tmp) as (features,digest):
                self.assertEqual(features.dtype,np.float32);np.testing.assert_array_equal(features,matrix.astype(np.float32))
                self.assertEqual(digest,hashlib.sha256(matrix.astype(np.float32).tobytes()).hexdigest())
            self.assertFalse((Path(tmp)/'rbp-saved-features.npy').exists());self.assertTrue(Path(entry['path']).exists())

    def test_row_reordering_is_rejected_before_numeric_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry,manifest,names,_=self.fixture(tmp,['second','first'])
            with self.assertRaisesRegex(ValueError,'feature order'):
                with adapter.frozen_features(entry,manifest,names,tmp): self.fail('Must reject reordered distances')
            self.assertFalse((Path(tmp)/'rbp-saved-features.npy').exists())

    def test_numerical_change_or_failed_consumer_cleans_only_owned_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry,manifest,names,_=self.fixture(tmp)
            manifest['knn_clustering']['feature_input']['sha256']='a'*64
            with self.assertRaisesRegex(ValueError,'feature values'):
                with adapter.frozen_features(entry,manifest,names,tmp): pass
            self.assertFalse((Path(tmp)/'rbp-saved-features.npy').exists())
            entry,manifest,names,_=self.fixture(tmp)
            with self.assertRaisesRegex(RuntimeError,'consumer failure'):
                with adapter.frozen_features(entry,manifest,names,tmp):raise RuntimeError('consumer failure')
            self.assertFalse((Path(tmp)/'rbp-saved-features.npy').exists());self.assertTrue(Path(entry['path']).exists())

    def test_noise_is_retained_in_memberships_without_becoming_a_representative(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            rows=[dict(model=n,source_path='/inputs/'+n+'.pdb',umap_x=1,umap_y=2) for n in ['a','b','c']]
            result=dict(models=['a','b','c'],density=['C1.1','C1.unassigned','C2'])
            full,assigned=adapter.write_density_labels(result,rows,tmp)
            with full.open() as s: complete=list(csv.DictReader(s))
            with assigned.open() as s: selected=list(csv.DictReader(s))
            self.assertEqual([r['model'] for r in complete],['a','b','c'])
            self.assertEqual([r['model'] for r in selected],['a','c'])
            self.assertEqual(complete[1]['color'],'#969ca4')

if __name__=='__main__':unittest.main()
