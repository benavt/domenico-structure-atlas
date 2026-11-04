"""Tiny static arrays and mocked native decisions only; no scientific fitting."""
from dataclasses import asdict, dataclass
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import terminal_rescreen as adapter

AVAILABLE = importlib.util.find_spec('numpy') is not None


@dataclass(frozen=True)
class FakeSettings:
    minimum_remaining_residues: int = 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    return dict(path=str(path), sha256=digest(path))


def write(path, value):
    path.write_text(json.dumps(value)); return bound(path)


class ApprovalTests(unittest.TestCase):
    def test_no_native_import_or_input_load_before_exact_approval(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(adapter, 'native_engine') as native:
            path = Path(tmp)/'execution-plan.json'; path.write_text('{"operation":"saved-terminal-rescreen"}')
            (path.parent/'approval.json').write_text(json.dumps(dict(approved_by='user',approved_plan_sha256='0'*64)))
            with self.assertRaisesRegex(ValueError, 'Exact execution approval'):
                adapter.validate_approval(path)
            native.assert_not_called()


@unittest.skipUnless(AVAILABLE, 'Optional numeric fixture runtime')
class SavedTerminalTests(unittest.TestCase):
    def fixture(self, root, *, removed=(), order=None, dtype=None):
        import numpy as np
        root = Path(root); count, residues = 6, 34; pairs = residues*(residues-1)//2
        names = [f'model-{i}' for i in range(count)]; positions = list(range(residues))
        keys = [['A', str(i)] for i in positions]  # Zero native number remains zero.
        matrix = (np.arange(count*pairs).reshape(count,pairs)/128).astype(dtype or np.float16)
        baseline = dict(schema_version=1, model_count=count, input_residue_keys=keys,
                        excluded_indices=[], retained_indices=positions, segments=[],
                        pair_mask_sha256=hashlib.sha256(np.ones(pairs, bool).tobytes()).hexdigest(),
                        settings={'historical_conservative':True})
        sequence = 'A'*residues
        axes = dict(names=names, names_sha256=adapter.names_digest(names),
            sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),reference_positions=positions,
            metadata=dict(reference=dict(sequence=sequence,chain_ids=['A']*residues,residue_numbers=positions),
                          shared_reference_positions=positions))
        cache = root/'cache.npz'
        np.savez(cache,flattened=matrix,reference_positions=np.array(positions),
            requested_reference_positions=np.array(positions),source_signature=json.dumps(dict(
                files=[[f'/intake/{name}.bin', 0, 0] for name in (order or names)],
                metadata_sha256=hashlib.sha256(json.dumps(axes['metadata'],sort_keys=True).encode()).hexdigest())),
            terminal_detection=json.dumps(baseline),filenames=np.array(names,dtype=object))
        metadata_path = root/'source-metadata.json.gz'
        metadata_path.write_bytes(gzip.compress(json.dumps(axes['metadata']).encode()))
        descriptor = dict(source_metadata=bound(metadata_path),target='SERCA_fixture',cache=dict(**bound(cache),bytes=cache.stat().st_size,
            member='flattened.npy',shape=[count,pairs]),axes=write(root/'axes.json',axes),
            baseline=write(root/'baseline.json',dict(terminal_detection=baseline)),
            settings=write(root/'settings.json',dict(schema_version=1,settings=asdict(FakeSettings()))))
        removed = list(removed); retained = sorted(set(positions)-set(removed))
        report = dict(model_count=count,input_residue_keys=keys,excluded_indices=removed,retained_indices=retained,
                      segments=[],settings=asdict(FakeSettings()),domain_evidence={'cores':[]},
                      terminal_anchor_domains=[{'chain':'A','cores':[]}])
        if removed:
            report['segments']=[dict(end='N',indices=removed,residue_keys=[keys[i] for i in removed])]
        def squareform(values):
            square=np.zeros((residues,residues));i,j=np.triu_indices(residues,1)
            square[i,j]=values;square[j,i]=values;return square
        detector = Mock(return_value=report)
        native = SimpleNamespace(TerminalSettings=FakeSettings,
            load_terminal_settings=Mock(return_value=FakeSettings()),
            reference_terminal_axes=Mock(return_value=([tuple(k) for k in keys],{('A','N'):True,('A','C'):True})),
            detect_terminal_variance=detector)
        return descriptor,axes,baseline,matrix,(native,squareform)

    def test_exact_native_reduction_safe_callbacks_and_baseline_preservation(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            descriptor,axes,baseline,matrix,engine=self.fixture(tmp,removed=range(3))
            original = {name:Path(descriptor[name]['path']).read_bytes() for name in ['cache','axes','baseline','settings']}
            checkpoints=[];checks=Mock()
            receipt=adapter.run_target(descriptor,Path(tmp)/'out',engine=engine,check=checks,
                                       checkpoint=checkpoints.append)
            expected=engine[1](np.var(np.asarray(matrix,float),axis=0))
            with Path(tmp,'out','variance.npy').open('rb') as stream:actual=np.load(stream,allow_pickle=False)
            np.testing.assert_array_equal(actual,expected)
            detector=engine[0].detect_terminal_variance;detector.assert_called_once()
            np.testing.assert_array_equal(detector.call_args.args[0],expected)
            self.assertEqual(detector.call_args.kwargs['chain_lengths'],{'A':34})
            self.assertEqual(detector.call_args.kwargs['eligible_ends'],{('A','N'):True,('A','C'):True})
            selection=json.loads(Path(tmp,'out','selection.json').read_text())
            self.assertEqual(selection['retained_canonical_indices'],list(range(3,34)))
            self.assertEqual(selection['excluded_terminal_ranges'][0]['native_start'],['A','0'])
            self.assertEqual(selection['canonical_index_base'],0)
            self.assertEqual(selection['excluded_terminal_ranges'][0]['canonical_start_position'],1)
            self.assertEqual(selection['excluded_terminal_ranges'][0]['canonical_start_index'],0)
            self.assertEqual(receipt['hierarchical_core_description_calls'],2)
            self.assertEqual(receipt['model_projection_fits'],0)
            self.assertEqual([p for p in checkpoints if ':variance:' in p],
                             ['SERCA_fixture:variance:512/561','SERCA_fixture:variance:561/561'])
            self.assertEqual(checkpoints[-1],'SERCA_fixture:report-closed')
            self.assertGreater(checks.call_count,2)
            for name,data in original.items():self.assertEqual(Path(descriptor[name]['path']).read_bytes(),data)
            self.assertEqual(json.loads(Path(tmp,'out','baseline.json').read_text()),baseline)
            with adapter.stored_matrix(descriptor['cache'],axes,baseline) as mapped:
                self.assertFalse(mapped.flags.writeable);np.testing.assert_array_equal(mapped,matrix)

    def test_empty_default_selection_retains_all_without_forced_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            descriptor,_,_,_,engine=self.fixture(tmp)
            adapter.run_target(descriptor,Path(tmp)/'out',engine=engine)
            selection=json.loads(Path(tmp,'out','selection.json').read_text())
            self.assertEqual(selection['retained_count'],34)
            self.assertEqual(selection['excluded_terminal_ranges'],[])

    def test_wrong_order_dtype_shape_and_metadata_hash_fail_before_detector(self):
        import numpy as np
        cases=['order','dtype','shape','axes-hash','sequence','keys']
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                kwargs={'order':[f'model-{i}' for i in reversed(range(6))]} if case=='order' else {}
                if case=='dtype':kwargs['dtype']=np.float32
                descriptor,axes,_,_,engine=self.fixture(tmp,**kwargs)
                if case=='shape':descriptor['cache']['shape'][0]=7
                if case=='axes-hash':descriptor['axes']['sha256']='0'*64
                if case=='sequence':
                    axes['sequence_sha256']='0'*64;descriptor['axes']=write(Path(tmp)/'axes.json',axes)
                if case=='keys':
                    axes['metadata']['reference']['residue_numbers'][1]=0
                    descriptor['axes']=write(Path(tmp)/'axes.json',axes)
                with self.assertRaises(ValueError):adapter.run_target(descriptor,Path(tmp)/'out',engine=engine)
                engine[0].detect_terminal_variance.assert_not_called()

    def test_crc_compression_and_unsafe_zip_member_fail_closed(self):
        for case in ['CRC','compressed','unsafe']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                descriptor,axes,baseline,_,engine=self.fixture(tmp);cache=Path(descriptor['cache']['path'])
                if case=='CRC':
                    data=bytearray(cache.read_bytes())
                    with ZipFile(cache) as archive:
                        info=archive.getinfo('flattened.npy');offset=info.header_offset+30+len(info.filename.encode())
                        extra=int.from_bytes(data[info.header_offset+28:info.header_offset+30],'little')
                        data[offset+extra+150]^=1
                    cache.write_bytes(data)
                elif case=='unsafe':
                    with ZipFile(cache,'a') as archive:archive.writestr('../outside.npy',b'no')
                else:
                    with ZipFile(cache) as archive:members=[(i.filename,archive.read(i)) for i in archive.infolist()]
                    with ZipFile(cache,'w',compression=ZIP_DEFLATED) as archive:
                        for name,data in members:archive.writestr(name,data)
                descriptor['cache'].update(bound(cache),bytes=cache.stat().st_size)
                with self.assertRaises(Exception):adapter.run_target(descriptor,Path(tmp)/'out',engine=engine)
                engine[0].detect_terminal_variance.assert_not_called()

    def test_internal_cut_and_over_budget_are_rejected_without_complete_receipt(self):
        for removed in [list(range(1,4)),list(range(9))]:
            with self.subTest(removed=removed),tempfile.TemporaryDirectory() as tmp:
                descriptor,_,_,_,engine=self.fixture(tmp,removed=removed)
                with self.assertRaisesRegex(ValueError,'terminal|budget'):
                    adapter.run_target(descriptor,Path(tmp)/'out',engine=engine)
                self.assertFalse(Path(tmp,'out','receipt.json').exists())

    def test_guard_stop_never_writes_completed_report_or_changes_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            descriptor,_,_,_,engine=self.fixture(tmp);before=Path(descriptor['cache']['path']).read_bytes()
            def checkpoint(boundary):
                if ':variance:' in boundary:raise RuntimeError('guard requested stop')
            with self.assertRaisesRegex(RuntimeError,'guard requested stop'):
                adapter.run_target(descriptor,Path(tmp)/'out',engine=engine,checkpoint=checkpoint)
            self.assertEqual(Path(descriptor['cache']['path']).read_bytes(),before)
            self.assertFalse(Path(tmp,'out','receipt.json').exists())
            engine[0].detect_terminal_variance.assert_not_called()

    def test_missing_core_evidence_and_changed_settings_fail_closed(self):
        for case in ['evidence','settings']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                descriptor,_,_,_,engine=self.fixture(tmp)
                if case=='evidence':engine[0].detect_terminal_variance.return_value.pop('domain_evidence')
                else:descriptor['settings']=write(Path(tmp)/'settings.json',dict(schema_version=1,settings={}))
                with self.assertRaises(ValueError):adapter.run_target(descriptor,Path(tmp)/'out',engine=engine)
                self.assertFalse(Path(tmp,'out','receipt.json').exists())


if __name__=='__main__':
    unittest.main()
