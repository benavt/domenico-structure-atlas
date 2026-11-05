import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from distance_preload import preload


class PreloadTests(unittest.TestCase):
    def fixture(self,root):
        folder=root/'distances';folder.mkdir()
        (folder/'one.bin').write_bytes(b'fixed-native-bytes')
        intake=root/'intake.json';intake.write_text(json.dumps({'models':[{'model_id':'one'}]}))
        screen=root/'screen.json';screen.write_text(json.dumps({'distogram_dir':str(folder)}))
        return {'intake':str(intake),'screening':str(screen)},folder

    def test_read_only_bounded_buffer_and_no_scientific_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,folder=self.fixture(root);result=preload(target,root/'proof.json')
            self.assertEqual((folder/'one.bin').read_bytes(),b'fixed-native-bytes')
            self.assertEqual(result['bytes_read'],len(b'fixed-native-bytes'))
            self.assertEqual(result['read_buffer_bytes'],1024**2)
            self.assertFalse(result['scientific_checks_bypassed'])

    def test_wrong_count_or_symlink_stops_before_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,folder=self.fixture(root)
            (folder/'extra.bin').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'count'):preload(target,root/'proof.json')
            (folder/'extra.bin').unlink();native=folder/'one.bin';native.rename(root/'original.bin');native.symlink_to(root/'original.bin')
            with self.assertRaisesRegex(ValueError,'regular'):preload(target,root/'proof.json')
            self.assertFalse((root/'proof.json').exists())
