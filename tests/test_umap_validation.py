"""Static parity artifacts and approval gates; never execute a native fit."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import validate_umap_memory as validation


class ScopeTests(unittest.TestCase):
    def test_fixed_cases_charge_fourteen_fits_for_two_profiles_and_memory_pair(self):
        cases=validation.cases()
        self.assertEqual(len(cases)*2,validation.SCOPE['maximum_umap_fits'])
        self.assertEqual({c['profile'] for c in cases},{'primary','density'})
        self.assertEqual(len({c['id'] for c in cases}),7)
        self.assertEqual([c['seed'] for c in cases if c['dataset']=='wide'],[42])

    def test_wrong_approval_or_scope_fails_before_native_controls(self):
        for bad in ('approval','scope'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan=dict(settings=dict(validation.SCOPE),operation=validation.SCOPE['operation'])
                if bad=='scope':plan['settings']['maximum_umap_fits']=16
                path=root/'plan.json';path.write_text(json.dumps(plan))
                (root/'approval.json').write_text(json.dumps(dict(approved_by='user',approved_plan_sha256='0'*64 if bad=='approval' else validation.sha256(path))))
                with patch('run_reviewed_plan.verify_controls') as controls:
                    with self.assertRaises(ValueError):validation.approved(path)
                    controls.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy for tiny static parity artifacts')
class ArtifactTests(unittest.TestCase):
    def make(self,root,*,bad=None):
        import numpy as np
        case=validation.cases()[0]
        for mode in ('baseline','candidate'):
            folder=root/(case['id']+'-'+mode);folder.mkdir()
            arrays=folder/'arrays.npz';xy=np.zeros((4,2),dtype=np.float32)
            if bad=='embedding' and mode=='candidate':xy[0,0]=1.
            np.savez_compressed(arrays,embedding=xy,neighbors=np.zeros((4,2),np.int32),distances=np.zeros((4,2),np.float32),graph_data=np.ones(4,np.float32),graph_indices=np.arange(4),graph_indptr=np.arange(5))
            value=dict(input=dict(path='same-bound-input',sha256='f'*64,bytes=4),arrays=validation.record(arrays),
                final_random_state_sha256='b'*64 if bad=='rng' and mode=='candidate' else 'a'*64,
                native_neighbor_input_shares_source=mode=='candidate',peak_RSS_bytes=100,elapsed_fit_seconds=1,
                helper=dict(avoided_retained_copy_bytes=64))
            validation.save(folder/'receipt.json',value)
        return case

    def test_identical_native_artifacts_require_source_alias_proof_without_assumed_rss_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=self.make(root);out=validation.compare(root,case)
            self.assertTrue(all(out['exact_fields'].values()))
            self.assertEqual(out['RSS_difference_bytes'],0)
            self.assertEqual(out['retained_copy_removed_bytes'],64)

    def test_changed_embedding_or_rng_stream_fails_acceptance(self):
        for bad in ('embedding','rng'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);case=self.make(root,bad=bad)
                with self.assertRaises(ValueError):validation.compare(root,case)


if __name__=='__main__':unittest.main()
