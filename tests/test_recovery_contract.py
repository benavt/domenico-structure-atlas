import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recover_screen import validate_bindings


class RecoveryContractTests(unittest.TestCase):
    def test_only_exact_single_conformer_identity_can_be_recovered(self):
        intake={'models':[dict(model_id='T__m__0',path='/inputs/model.pdb',sha256='a'*64)]}
        inventory={'models':[dict(model='T__m__0',source_path='/inputs/model.pdb',sha256='a'*64,
                                 original_source_path='/inputs/model.pdb',source_conformer_count=1,chain_id_map={})]}
        validate_bindings(intake,inventory,{'T__m__0'})
        for key,value in [('sha256','b'*64),('source_conformer_count',2),('chain_id_map',{'B':'A'}),
                          ('original_source_path','/another.pdb')]:
            modified=copy.deepcopy(inventory);modified['models'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_bindings(intake,modified,{'T__m__0'})
        with self.assertRaisesRegex(ValueError,'coverage'):validate_bindings(intake,inventory,set())
        duplicate=copy.deepcopy(inventory);duplicate['models']*=2
        with self.assertRaisesRegex(ValueError,'inventory'):validate_bindings(intake,duplicate,{'T__m__0'})


class ResourceControlTests(unittest.TestCase):
    def test_wrong_or_missing_controls_block_scientific_execution(self):
        import tempfile
        from run_reviewed_plan import verify_controls
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); limits=dict(maximum_memory_bytes=1024,maximum_cpus=2)
            baseline={'memory.max':'1024','memory.swap.max':'0','cpu.max':'200000 100000'}
            for name,value in baseline.items():(root/name).write_text(value)
            verify_controls(root,limits)
            for name,bad in [('memory.max','max'),('memory.swap.max','max'),('cpu.max','max 100000'),('cpu.max','400000 100000')]:
                (root/name).write_text(bad)
                with self.subTest(name=name,bad=bad),self.assertRaises(ValueError):verify_controls(root,limits)
                (root/name).write_text(baseline[name])


class VerificationCheckpointTests(unittest.TestCase):
    def test_checkpoint_requires_exact_header_and_unique_identities(self):
        import tempfile
        from recover_screen import load_checkpoint,save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'proof.json';header={'protocol':1,'input_hash':'a'}
            rows=[{'model':'m','input_sha256':'a','distance_sha256':'b'}]
            save_checkpoint(p,header,rows)
            self.assertEqual(load_checkpoint(p,header),{'m':rows[0]})
            with self.assertRaisesRegex(ValueError,'changed'):load_checkpoint(p,{'protocol':1,'input_hash':'c'})
            save_checkpoint(p,header,rows*2)
            with self.assertRaisesRegex(ValueError,'Duplicate'):load_checkpoint(p,header)
            self.assertFalse(p.with_suffix('.partial').exists())
