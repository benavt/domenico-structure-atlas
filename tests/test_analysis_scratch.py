"""Shutdown cleanup safety; only tiny temporary files, no scientific execution."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import analysis_scratch as scratch


class ScratchTests(unittest.TestCase):
    def fixture(self, root):
        plan = dict(run_root=str(root), environment={'TMPDIR':'/dev/shm/domenico-atlas-test'},
                    scratch={'directory_name':'domenico-atlas-test', 'minimum_free_bytes_before_start':1})
        path = root/'plan.json'; path.write_text(json.dumps(plan))
        (root/'approval.json').write_text(json.dumps(dict(approved_by='user',approved_plan_sha256=scratch.sha256(path))))
        return path, plan

    def exercise(self, change=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=self.fixture(root);owned=root/'owned';outside=root/'outside'
            outside.write_text('preserve')
            with patch.object(scratch,'scratch_root',return_value=owned), patch.object(scratch,'assert_tmpfs'), patch.object(scratch,'assert_quiescent'):
                scratch.create(path,plan)
                folder=owned/'arrays';folder.mkdir();(folder/'input.npy').write_bytes(b'disposable')
                if change:change(root,path,plan,owned,outside)
                if change:
                    with self.assertRaises((ValueError,FileNotFoundError)):scratch.cleanup(path,plan)
                    self.assertTrue(owned.exists() or owned.is_symlink())
                else:
                    scratch.cleanup(path,plan);self.assertFalse(owned.exists())
                    self.assertEqual(json.loads((root/'scratch-cleanup.json').read_text())['status'],'removed_owned_ram_scratch')
                self.assertEqual(outside.read_text(),'preserve')

    def test_cleanup_preserves_files_outside_owned_directory(self):self.exercise()
    def test_symlink_holds_cleanup(self):self.exercise(lambda r,p,d,o,x:(o/'link').symlink_to(x))
    def test_hardlink_holds_cleanup(self):self.exercise(lambda r,p,d,o,x:os.link(x,o/'link'))
    def test_changed_approval_holds_cleanup(self):self.exercise(lambda r,p,d,o,x:p.write_text('{}'))
    def test_changed_ownership_marker_holds_cleanup(self):self.exercise(lambda r,p,d,o,x:(o/'.owner.json').write_text('{}'))
    def test_root_symlink_holds_cleanup(self):
        def replace(root,path,plan,owned,outside):
            owned.rename(root/'old-owned');owned.symlink_to(root/'old-owned',target_is_directory=True)
        self.exercise(replace)

    def test_arbitrary_or_parent_path_is_rejected(self):
        for value in ['/tmp/domenico-atlas-test','/dev/shm','/dev/shm/domenico-atlas-test/../other','/dev/shm/unrelated']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                scratch.scratch_root({'environment':{'TMPDIR':value},'scratch':{'directory_name':'domenico-atlas-test'}})

    def test_other_process_in_unit_holds_cleanup(self):
        with patch.object(scratch.Path,'read_text',side_effect=['0::/test','999999 '+str(os.getpid())]):
            with self.assertRaisesRegex(ValueError,'stopped unit'):scratch.assert_quiescent()

    def test_capacity_and_missing_approval_block_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=self.fixture(root);owned=root/'owned'
            with patch.object(scratch,'scratch_root',return_value=owned),patch.object(scratch,'assert_tmpfs'),patch.object(scratch.shutil,'disk_usage',return_value=type('Usage',(),{'free':0})()):
                with self.assertRaisesRegex(ValueError,'capacity'):scratch.create(path,plan)
                self.assertFalse(owned.exists())
            (root/'approval.json').unlink()
            with self.assertRaises(FileNotFoundError):scratch.create(path,plan)
