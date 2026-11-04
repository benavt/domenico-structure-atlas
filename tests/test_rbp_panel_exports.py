"""Source-only panel export checks: native parsing/alignment and PyMOL are mocked."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rbp_panel_exports as adapter


def fixture(root):
    names = [f'model-{i}' for i in range(60)]
    members = []
    for name in names:
        path = root/(name+'.pdb'); path.write_text('Frozen mocked input '+name)
        members.append(dict(model=name, **adapter.record(path)))
        members[-1]['source_path'] = members[-1].pop('path')
    catalog = dict(schema_version=1, kind='saved-clustering-residual-states-and-subsets',
        models=names, target='RBP', range_id='all', construct_residues=3, canonical_positions=[1, 2, 3],
        sequence_sha256=hashlib.sha256(b'AAA').hexdigest(), TM_cutoff=.8, min_state_size=50,
        states=[dict(id='S1', anchor=names[0], supported=True, count=60, indices=list(range(60)),
            methods={method:dict(indices=list(range(start, start+50)), models=names[start:start+50])
                     for method, start in [('mcmc50', 10), ('disco50', 0)]})], unassigned_indices=[])
    return catalog, members


class CatalogTests(unittest.TestCase):
    def test_complete_registry_and_saved_selection_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog, members = fixture(Path(tmp))
            self.assertEqual(adapter.checked_catalog(catalog, members), list(range(60)))
            self.assertEqual(catalog['states'][0]['methods']['mcmc50']['indices'][0], 10)

    def test_wrong_core_order_members_and_panel_are_rejected(self):
        for kind in ('core', 'registry', 'duplicate', 'outside-state', 'unsafe-state', 'partition'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                catalog, members = fixture(Path(tmp)); state = catalog['states'][0]
                if kind == 'core':catalog['canonical_positions'] = [2, 3]
                if kind == 'registry':members.reverse()
                if kind == 'duplicate':state['methods']['mcmc50']['indices'][0] = 11
                if kind == 'outside-state':state['indices'] = list(range(10, 60)); state['count'] = 50
                if kind == 'unsafe-state':state['id'] = '../escape'
                if kind == 'partition':catalog['unassigned_indices'] = [0]
                with self.assertRaises(ValueError):adapter.checked_catalog(catalog, members)


AVAILABLE = importlib.util.find_spec('numpy') is not None


class MockCoordinates:
    mapping = [dict(canonical_position=i+1, chain='B', native_residue=[' ', *residue])
               for i, residue in enumerate([(0, ' '), (0, 'A'), (7, ' ')])]
    signature = [('B', (' ', 0, ' '), 'ALA', 'N', ' ', 'N', 1., 10.),
                 ('B', (' ', 0, ' '), 'ALA', 'CA', ' ', 'C', 1., 10.),
                 ('B', (' ', 0, 'A'), 'ALA', 'CA', ' ', 'C', 1., 10.),
                 ('B', (' ', 7, ' '), 'ALA', 'CA', ' ', 'C', 1., 10.),
                 ('W', ('W', 99, 'B'), 'HOH', 'O', 'A', 'O', 1., 10.)]
    def __init__(self, bad_inventory=False, changed_source=None):
        self.bad_inventory, self.changed_source, self.alignments = bad_inventory, changed_source, []

    def canonical(self, paths, **kwargs):
        import numpy as np
        if kwargs != dict(retained_positions=None, include_mapping=True):raise AssertionError(kwargs)
        return np.asarray([np.arange(9).reshape(3, 3) for path in paths]), 'AAA', [self.mapping for path in paths]

    def read_frame(self, row, case):
        import numpy as np
        return SimpleNamespace(signature=self.signature), np.arange(9).reshape(3, 3), self.mapping

    def aligned(self, row, case, reference):
        self.alignments.append((row['model'], case))
        lines = []
        for number, (chain, residue, resn, name, alt, element, occupancy, bfactor) in enumerate(self.signature, 1):
            group = 'ATOM' if residue[0] == ' ' else 'HETATM'
            number_id = 9 if self.bad_inventory and number == 1 else residue[1]
            lines.append(f'{group:<6}{number:5d} {name:>4}{alt:1}{resn:>3} {chain:1}{number_id:4d}{residue[2]:1}   '
                f'{number:8.3f}{number+1:8.3f}{number+2:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2}  ')
        if self.changed_source and len(self.alignments) == 1:self.changed_source.write_text('Changed during mocked export')
        metadata = dict(model=row['model'], source_sha256=row['sha256'], atom_count=len(self.signature),
            canonical_mapping=self.mapping, alignment_canonical_positions=case['canonical_positions'],
            rotation_row_vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation=[0, 0, 0], alignment_CA_RMSD_angstrom=0.)
        return '\n'.join([*lines, 'END', '']), metadata, [0, 0, 0], [9, 9, 9]

    def adapters(self):
        return dict(canonical_ca_arrays=self.canonical, read_frame=self.read_frame, aligned_member=self.aligned,
                    atom_signature=lambda model:model.signature,
                    privacy_check=lambda raw, name:None)


class MockPyMOL:
    """A saved static JSON roundtrip, with no native engine or coordinate work."""
    def __init__(self, events, break_roundtrip=False):
        self.cmd = self; self.events = events; self.frames = []; self.settings = {}; self.break_roundtrip = break_roundtrip
    def __enter__(self):return self
    def __exit__(self, *args):pass
    def reinitialize(self):self.frames = []
    def set(self, key, value):self.settings[key] = value
    def read_pdb(self, text, name, discrete):
        assert discrete == 1
        for block in text.split('ENDMDL')[:-1]:
            atoms, xyz = adapter._pdb_atoms(block)
            self.frames.append([dict(chain=a[0], resi=a[1], resn=a[2], name=a[3], alt=a[4], hetatm=a[5], symbol=a[6], coord=p)
                                for a, p in zip(atoms, xyz)])
    def count_states(self, name):return len(self.frames)
    def get_model(self, name, state):return SimpleNamespace(atom=[SimpleNamespace(**a) for a in self.frames[state-1]])
    def set_title(self, name, state, title):self.events.append(('title', state, title))
    def hide(self, *args):pass
    def show(self, *args):pass
    def color(self, *args):pass
    def bg_color(self, *args):pass
    def orient(self, *args):pass
    def zoom(self, *args, **kwargs):pass
    def deselect(self):pass
    def save(self, filename, **kwargs):
        assert Path(filename).name == filename  # Private absolute paths never enter saved sessions.
        self.events.append(('save', filename))
        Path(filename).write_text(json.dumps(dict(frames=self.frames, settings=self.settings)))
    def load(self, filename, **kwargs):
        assert Path(filename).name == filename
        saved = json.loads(Path(filename).read_text()); self.frames, self.settings = saved['frames'], saved['settings']
        if self.break_roundtrip:self.frames[0][0]['resi'] = '999'
    def get_setting_int(self, key):return self.settings[key]


@unittest.skipUnless(AVAILABLE, 'Optional NumPy needed for mock-only coordinate adapter checks')
class ExportTests(unittest.TestCase):
    def execute(self, root, *, coordinates=None, bad_session=False, expected_hash=None, checkpoint=lambda _: None):
        catalog, members = fixture(root)
        source = root/'saved-catalog.json'; source.write_text(json.dumps(catalog))
        events = []; coords = coordinates or MockCoordinates()
        result = adapter.export_catalog(source, members, root/'output',
            expected_catalog_sha256=expected_hash or adapter.record(source)['sha256'],
            pymol_factory=lambda:MockPyMOL(events, bad_session), adapters=coords.adapters(), checkpoint=checkpoint)
        return result, coords, events, members, source

    def test_backoff_boundary_follows_verified_files_and_precedes_next_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def stop_at_first_panel(identifier):
                calls.append(identifier)
                folder = root/'output/S1/mcmc50'
                self.assertEqual(identifier,'panel-written:S1:mcmc50')
                self.assertEqual(len(adapter._pdb_atoms((folder/'ensemble.pdb').read_text())[0]),250)
                self.assertEqual(json.loads((folder/'alignment.json').read_text())['verification']['states'],50)
                self.assertTrue((folder/'ensemble.pse').is_file())
                raise RuntimeError('cooperative stop at completed panel')
            with self.assertRaisesRegex(RuntimeError,'cooperative stop'):
                self.execute(root,checkpoint=stop_at_first_panel)
            self.assertEqual(len(calls),1)
            self.assertFalse((root/'output/S1/disco50').exists())
            self.assertFalse((root/'output/catalog.json').exists())

    def test_saved50_full_atoms_native_zero_insertion_and_paths_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original_cwd = Path.cwd()
            result, coords, events, members, source = self.execute(root)
            self.assertEqual(Path.cwd(), original_cwd)
            self.assertEqual(len(coords.alignments), 100)
            self.assertTrue(all(case['canonical_positions'] == [1, 2, 3] and case['chain'] == 'B' for _, case in coords.alignments))
            self.assertEqual(result['coordinate_export_summary']['panels'], 2)
            self.assertEqual(result['coordinate_export_summary']['new_TM_calls'], 0)
            self.assertEqual(len([e for e in events if e[0] == 'save']), 2)
            panel = result['states'][0]['methods']['mcmc50']
            text = Path(panel['pdb']['path']).read_text()
            self.assertEqual(text.count('MODEL '), 100)  # MODEL and SOURCE_MODEL remarks.
            self.assertEqual(text.count('ENDMDL'), 50)
            self.assertEqual(text.count('HETATM'), 50)
            self.assertIn(' B   0A', text)
            self.assertIn(' B   0 ', text)
            self.assertEqual(panel['models'][0], 'model-10')
            self.assertEqual(adapter.record(panel['pse']['path']), panel['pse'])
            proof = json.loads(Path(panel['alignment']['path']).read_text())
            self.assertTrue(proof['verification']['native_atom_inventory_preserved'])
            self.assertEqual(proof['members'][0]['canonical_mapping'][0]['native_residue'], [' ', 0, ' '])
            self.assertNotIn('coordinate_export_status', json.loads(source.read_text()))
            self.assertTrue(all(adapter.record(row['source_path'])['sha256'] == row['sha256'] for row in members))

    def test_catalog_hash_source_tamper_and_existing_outputs_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'catalog hash'):self.execute(root, expected_hash='0'*64)
            self.assertFalse((root/'output').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); catalog, members = fixture(root); source = root/'catalog.json'; source.write_text(json.dumps(catalog))
            Path(members[0]['source_path']).write_text('Changed')
            with self.assertRaisesRegex(ValueError, 'coordinate hash'):
                adapter.export_catalog(source, members, root/'output', expected_catalog_sha256=adapter.record(source)['sha256'])
            self.assertFalse((root/'output').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.execute(root)
            with self.assertRaisesRegex(ValueError, 'immutable'):self.execute(root)

    def test_pdb_or_pse_identity_loss_fails_before_completed_catalog(self):
        for bad_coordinates, bad_session in [(True, False), (False, True)]:
            with self.subTest(pdb=bad_coordinates), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); before = Path.cwd()
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    self.execute(root, coordinates=MockCoordinates(bad_inventory=bad_coordinates), bad_session=bad_session)
                self.assertEqual(Path.cwd(), before)
                self.assertFalse((root/'output/catalog.json').exists())

    def test_output_catalog_is_bound_to_all_input_files_after_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); coords = MockCoordinates(changed_source=root/'model-0.pdb')
            with self.assertRaisesRegex(ValueError, 'changed during export'):self.execute(root, coordinates=coords)
            self.assertFalse((root/'output/catalog.json').exists())


if __name__ == '__main__':unittest.main()
