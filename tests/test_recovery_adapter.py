"""Adapter orchestration regression: native distances/plots are mocked, not fitted."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import recover_screen as adapter

AVAILABLE = importlib.util.find_spec('numpy') is not None and importlib.util.find_spec('src') is not None


@unittest.skipUnless(AVAILABLE,'Requires the optional library runtime for mocked adapter integration')
class RecoveryAdapterTests(unittest.TestCase):
    def exercise(self, bad=False):
        import numpy as np
        from contextlib import contextmanager
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir();(root/'distances').mkdir()
            (root/'config/study.json').write_text(json.dumps({'targets':{'T':{'sequence':'AAA'}}}))
            coordinate=root/'model.pdb';coordinate.write_bytes(b'Native-coordinate-placeholder; parser is mocked')
            source_hash=adapter.digest(coordinate)
            inventory=root/'inventory.json'
            inventory.write_text(json.dumps({'models':[dict(model='m',source_path=str(coordinate),original_source_path=str(coordinate),
                sha256=source_hash,source_conformer_count=1,chain_id_map={},chain_ids=['A'])]}))
            intake=root/'intake.json';intake.write_text('{}')
            values=np.array([1,2,3],dtype=np.float16);values.tofile(root/'distances/m.bin')
            (root/'distances/distogram_manifest.json').write_text(json.dumps(dict(file_format='bin',storage='condensed',dtype='float16',n_residues=3)))
            catalog=root/'catalog.json';catalog.write_text(json.dumps(dict(input_inventory_sha256=adapter.digest(inventory),
                models=[dict(model='m',input_sha256=source_hash,distance_sha256=adapter.digest(root/'distances/m.bin'))])))
            settings=root/'terminal.json';settings.write_text('{}')
            fake_intake=dict(target='T',models=[dict(model_id='m',path=str(coordinate),sha256=source_hash)])
            fake_sequence=SimpleNamespace(subsequences=[SimpleNamespace(sequence='AAA')])
            @contextmanager
            def prepared(*args,**kwargs):yield SimpleNamespace(terminal_detection={'retained_indices':[0,1,2]})
            contexts=[
                patch('workflow.ROOT',root),patch('workflow.load_intake',return_value=(fake_intake,root)),
                patch('src.analysis.structure_io.get_structure',return_value=[[SimpleNamespace(id='A')]]),
                patch('src.analysis.sequence.Sequence.from_model',return_value=fake_sequence),
                patch('scripts.pdb_to_distogram.get_selected_atoms',return_value=np.zeros((3,3),dtype=np.float32)),
                patch('src.generation.distogram.calculate_distogram',return_value=np.array([1,2,4 if bad else 3],dtype=np.float32)),
                patch('src.analysis.terminal_variance.load_terminal_settings',return_value=None),
                patch('src.analysis.prepared_distograms.prepare_workflow_distograms',prepared),
                patch('src.app.shared.workflow_exports.export_distogram_views',return_value={'images':[]})]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for context in contexts:stack.enter_context(context)
                if bad:
                    with self.assertRaisesRegex(ValueError,'do not match native coordinates'):
                        adapter.recover(intake,inventory,root/'distances',root/'output',settings,catalog)
                    self.assertFalse((root/'output/screening.json').exists())
                else:
                    adapter.recover(intake,inventory,root/'distances',root/'output',settings,catalog)
                    report=json.loads((root/'output/screening.json').read_text())
                    self.assertEqual(report['recovery_validation_sha256'],adapter.digest(root/'output/verified_distograms.json'))
                self.assertEqual((root/'distances/m.bin').read_bytes(),values.tobytes())

    def test_ndarray_return_and_report_completion(self):self.exercise()
    def test_distance_mismatch_survives_cleanup_and_stops_reporting(self):self.exercise(bad=True)
