"""Static selected arrays only; never open a target cache or a scientific engine."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, BadZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import saved_feature_cache as saved

try:
    import numpy as np
except ImportError:
    np = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def write(path, value):
    path.write_text(json.dumps(value))
    return bound(path)


def array_bytes(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=True)
    return stream.getvalue()


@unittest.skipIf(np is None, 'NumPy is required for static cache fixtures')
class SelectedCacheTests(unittest.TestCase):
    def fixture(self, root, *, precision=None, fortran=False, compressed=False, object_metadata=False):
        count, residues = 6, 34
        names = [f'toy-{index}' for index in range(count)]
        positions = list(range(residues))
        retained = list(range(3, residues-3))
        keys = [['A', str(index)] for index in positions]
        sequence = 'A'*residues
        metadata = dict(reference=dict(sequence=sequence, chain_ids=['A']*residues,
                                      residue_numbers=positions), shared_reference_positions=positions)
        baseline = dict(model_count=count, input_residue_keys=keys,
            retained_indices=positions, excluded_indices=[], segments=[],
            settings=dict(minimum_remaining_residues=10))
        report = copy.deepcopy(baseline)
        removed = positions[:3]+positions[-3:]
        keep = np.zeros(residues, bool)
        keep[retained] = True
        left, right = np.triu_indices(residues, 1)
        mask_sha = hashlib.sha256((keep[left] & keep[right]).tobytes()).hexdigest()
        report.update(reference_positions=positions, retained_indices=retained,
            retained_reference_positions=retained, excluded_indices=removed,
            pair_mask_sha256=mask_sha, retained_feature_count=len(retained)*(len(retained)-1)//2,
            segments=[dict(chain='A', end=end, indices=indices,
                residue_keys=[keys[i] for i in indices])
                for end, indices in [('N', positions[:3]), ('C', positions[-3:])]])
        entries = dict(axes=write(root/'axes.json', dict(names=names,
                names_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                reference_positions=positions, metadata=metadata)),
            baseline=write(root/'baseline.json', baseline),
            source_metadata=write(root/'metadata.json', metadata),
            terminal_report=write(root/'terminal.json', report))
        pairs = report['retained_feature_count']
        expected = (np.arange(count*pairs).reshape(count, pairs)/64).astype(np.float16)
        proof = dict(shape=list(expected.shape), dtype='float16',
            retained_reference_positions=retained, pair_mask_sha256=mask_sha,
            models_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
            feature_sha256=hashlib.sha256(expected.tobytes(order='C')).hexdigest(),
            path='/expired/owned-scratch/selected.npy', temporary=True)
        entries['feature_proof'] = write(root/'proof.json', proof)
        signature = dict(files=[[f'/unavailable/originals/{name}.bin', 1122, 1] for name in names],
            metadata_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
            terminal_rescreen_report_sha256=entries['terminal_report']['sha256'])
        numeric = expected.astype(precision) if precision else expected
        if fortran:
            numeric = np.asfortranarray(numeric)
        values = dict(flattened=numeric, filenames=np.asarray(names, dtype=object),
            filepaths=np.asarray(['/not/read/'+n for n in names], dtype=object),
            source_signature=np.asarray(json.dumps(signature), dtype=object if object_metadata else None),
            terminal_detection=json.dumps(report), matrix_entity='residue',
            reference_positions=np.asarray(retained, dtype=np.int32),
            requested_reference_positions=np.asarray(positions, dtype=np.int32))
        path = root/'selected.npz'
        (np.savez_compressed if compressed else np.savez)(path, **values)
        cache = dict(**bound(path), member='flattened.npy', shape=list(expected.shape))
        return cache, entries, expected

    def replace_member(self, cache, name, payload):
        path = Path(cache['path'])
        with ZipFile(path) as archive:
            members = [(info.filename, archive.read(info)) for info in archive.infolist()]
        with ZipFile(path, 'w') as archive:
            for old_name, old_payload in members:
                archive.writestr(old_name, payload if old_name == name else old_payload)
        cache.update(bound(path))

    def test_readonly_same_descriptor_mapping_exact_selected_values_and_no_object_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, entries, expected = self.fixture(Path(tmp))
            before = Path(cache['path']).read_bytes()
            original_memmap = np.memmap
            with patch('numpy.load', side_effect=AssertionError('No object array loading')), \
                    patch('numpy.memmap', wraps=original_memmap) as mapper:
                with saved.mapped_selected_features(cache, **entries) as (matrix, proof):
                    self.assertIsInstance(matrix, original_memmap)
                    self.assertFalse(matrix.flags.writeable)
                    np.testing.assert_array_equal(matrix, expected)
                    with self.assertRaises(ValueError):
                        matrix[0, 0] = 999
                    self.assertEqual(mapper.call_args.kwargs['mode'], 'r')
                    self.assertTrue(hasattr(mapper.call_args.args[0], 'fileno'))
                    self.assertEqual(proof['shape'], list(expected.shape))
                    self.assertEqual(proof['retained_reference_positions'], list(range(3, 31)))
                    self.assertEqual(proof['requested_reference_positions'], list(range(34)))
                    self.assertTrue(proof['member_crc_verified'])
                    self.assertEqual(proof['feature_sha256'], hashlib.sha256(expected.tobytes()).hexdigest())
                    self.assertEqual(proof['metadata']['reference']['residue_numbers'], list(range(34)))
                    self.assertEqual(proof['source_signature']['terminal_rescreen_report_sha256'],
                                     entries['terminal_report']['sha256'])
                    self.assertFalse(proof['full_array_copy'])
            self.assertTrue(matrix._mmap.closed)
            self.assertEqual(Path(cache['path']).read_bytes(), before)

    def test_invalid_precision_layout_compression_or_object_metadata_never_maps(self):
        cases = [dict(precision=np.float32), dict(fortran=True), dict(compressed=True), dict(object_metadata=True)]
        for options in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                cache, entries, _ = self.fixture(Path(tmp), **options)
                with patch('numpy.memmap') as mapper, self.assertRaises(ValueError):
                    with saved.mapped_selected_features(cache, **entries):
                        self.fail('Invalid input yielded a mapping')
                mapper.assert_not_called()

    def test_order_metadata_report_and_position_rebinding_rejects_before_mapping(self):
        for corruption in ('order', 'metadata', 'report', 'retained', 'requested', 'entity'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cache, entries, _ = self.fixture(root)
                if corruption in ('order', 'metadata', 'report'):
                    with ZipFile(cache['path']) as archive:
                        with archive.open('source_signature.npy') as member:
                            signature = json.loads(np.load(member, allow_pickle=False).item())
                    if corruption == 'order':
                        signature['files'].reverse()
                    else:
                        signature['metadata_sha256' if corruption == 'metadata'
                                  else 'terminal_rescreen_report_sha256'] = '0'*64
                    self.replace_member(cache, 'source_signature.npy', array_bytes(json.dumps(signature)))
                if corruption == 'retained':
                    self.replace_member(cache, 'reference_positions.npy', array_bytes(np.arange(4, 32)))
                if corruption == 'requested':
                    self.replace_member(cache, 'requested_reference_positions.npy', array_bytes(np.arange(33)))
                if corruption == 'entity':
                    self.replace_member(cache, 'matrix_entity.npy', array_bytes('atom'))
                with patch('numpy.memmap') as mapper, self.assertRaises(ValueError):
                    with saved.mapped_selected_features(cache, **entries):
                        self.fail('Rebound input yielded a mapping')
                mapper.assert_not_called()

    def test_feature_proof_and_raw_value_mismatch_never_maps(self):
        for field in ('feature_sha256', 'pair_mask_sha256', 'models_sha256', 'shape', 'retained_reference_positions'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cache, entries, _ = self.fixture(root)
                value = json.loads((root/'proof.json').read_text())
                value[field] = ([1, 2] if field == 'shape' else [4, 5] if field == 'retained_reference_positions' else '0'*64)
                entries['feature_proof'] = write(root/'proof.json', value)
                with patch('numpy.memmap') as mapper, self.assertRaises(ValueError):
                    with saved.mapped_selected_features(cache, **entries):
                        self.fail('Mismatched proof yielded a mapping')
                mapper.assert_not_called()

    def test_container_sha_and_crc_are_independent_gates(self):
        for update_container_hash in (False, True):
            with self.subTest(update_container_hash=update_container_hash), tempfile.TemporaryDirectory() as tmp:
                cache, entries, _ = self.fixture(Path(tmp))
                with ZipFile(cache['path']) as archive:
                    info = archive.getinfo('flattened.npy')
                # Locate the data through the real local header, including ZIP64 extra.
                import struct
                with Path(cache['path']).open('r+b') as stream:
                    stream.seek(info.header_offset+26)
                    name_size, extra_size = struct.unpack('<HH', stream.read(4))
                    stream.seek(info.header_offset+30+name_size+extra_size+128)
                    byte = stream.read(1)
                    stream.seek(-1, 1)
                    stream.write(bytes([byte[0] ^ 1]))
                if update_container_hash:
                    cache.update(bound(Path(cache['path'])))
                error = BadZipFile if update_container_hash else ValueError
                with patch('numpy.memmap') as mapper, self.assertRaises(error):
                    with saved.mapped_selected_features(cache, **entries):
                        self.fail('Corrupt input yielded a mapping')
                mapper.assert_not_called()

    def test_lying_metadata_header_is_rejected_before_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, entries, _ = self.fixture(Path(tmp))
            stream = io.BytesIO()
            np.lib.format.write_array_header_1_0(stream,
                dict(descr='<U5000000', fortran_order=False, shape=()))
            self.replace_member(cache, 'source_signature.npy', stream.getvalue())
            with patch('numpy.frombuffer', side_effect=AssertionError('Unsafe declared allocation')):
                with self.assertRaisesRegex(ValueError, 'byte bound'):
                    with saved.mapped_selected_features(cache, **entries):
                        self.fail('Invalid metadata yielded a mapping')

    def test_stop_closes_mapping_and_replacement_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, entries, _ = self.fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'injected stop'):
                with saved.mapped_selected_features(cache, **entries) as (matrix, _):
                    raise RuntimeError('injected stop')
            self.assertTrue(matrix._mmap.closed)
            with self.assertRaisesRegex(ValueError, 'changed while mapped'):
                with saved.mapped_selected_features(cache, **entries) as (matrix, _):
                    Path(cache['path']).rename(root/'preserved-old.npz')
                    Path(cache['path']).write_bytes(b'replacement')
            self.assertTrue(matrix._mmap.closed)

    def test_resource_stop_before_mapping_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, entries, _ = self.fixture(Path(tmp))
            def stop():
                raise RuntimeError('resource stop')
            with patch('numpy.memmap') as mapper, self.assertRaisesRegex(RuntimeError, 'resource stop'):
                with saved.mapped_selected_features(cache, **entries, check=stop):
                    self.fail('Stopped input yielded a mapping')
            mapper.assert_not_called()


if __name__ == '__main__':
    unittest.main()
