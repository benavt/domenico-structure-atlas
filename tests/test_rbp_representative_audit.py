"""Native representative audit: parser and saved-session reads are mocked."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rbp_representative_audit as adapter


def inventory():
    return [dict(chain='B', resi='0', resn='ALA', name='CA', alt='', hetero=False, element='C', occupancy=1., bfactor=23.457),
            dict(chain='B', resi='0A', resn='ALA', name='N', alt='B', hetero=False, element='N', occupancy=.6, bfactor=12.),
            dict(chain='W', resi='-1B', resn='HOH', name='O', alt='A', hetero=True, element='O', occupancy=.4, bfactor=2.)]


class MockPyMOL:
    def __init__(self, frames, count=None):self.cmd = self; self.frames = frames; self.count = count
    def __enter__(self):return self
    def __exit__(self, *args):pass
    def load(self, path, **kwargs):pass
    def count_states(self, name):return self.count if self.count is not None else len(self.frames)
    def get_model(self, name, state):
        return SimpleNamespace(atom=[SimpleNamespace(chain=a['chain'], resi=a['resi'], resn=a['resn'],
            name=a['name'], alt=a['alt'], hetatm=a['hetero'], symbol=a['element'], q=a['occupancy'], b=a['bfactor'])
            for a in self.frames[state-1]])


class RepresentativeAuditTests(unittest.TestCase):
    def fixture(self, root):
        files = {}
        for name in ('source-one.pdb', 'source-two.cif', 'cluster.cif', 'private.pse'):
            path = root/name; path.write_text('Static saved-fixture placeholder '+name); files[name] = path
        rows = [dict(model='one', source_path=str(files['source-one.pdb']), state=1,
                     ca_correspondence=dict(source_sha256=adapter.record(files['source-one.pdb'])['sha256'])),
                dict(model='two', source_path=str(files['source-two.cif']), state=2,
                     ca_correspondence=dict(source_sha256=adapter.record(files['source-two.cif'])['sha256']))]
        selection = dict(pse=str(files['private.pse']), clusters=[dict(cluster_label='C1.1', object_name='C1_1',
            selected=rows, multimodel_structure=str(files['cluster.cif']))])
        inventories = {str(path):[inventory()] for name, path in files.items() if name.startswith('source-')}
        inventories[str(files['cluster.cif'])] = [inventory(), inventory()]
        return selection, inventories, files

    def audit(self, selection, inventories, *, frames=None, count=None):
        with patch.object(adapter, '_file_inventories', side_effect=lambda path:deepcopy(inventories[str(path)])):
            return adapter.verify_selected_ensembles(selection,
                pymol_factory=lambda:MockPyMOL(frames if frames is not None else [inventory(), inventory()], count))

    def test_source_cif_pse_all_atoms_native_ids_and_format_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection, inv, files = self.fixture(Path(tmp))
            inv[str(files['cluster.cif'])][0][0]['bfactor'] = 23.46
            frames = [inventory(), inventory()]; frames[0].reverse()
            result = self.audit(selection, inv, frames=frames)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['selected_frames'], 2)
            self.assertEqual(result['groups'][0]['frames'][0]['atom_count'], 3)
            self.assertEqual(result['new_exports'], 0)
            self.assertIn('private local PSE', result['scope'])

    def test_cif_native_zero_insertion_hetero_alt_or_element_loss_stops(self):
        for field, wrong in [('resi', '1'), ('alt', ''), ('hetero', False), ('element', 'C')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                selection, inv, files = self.fixture(Path(tmp))
                inv[str(files['cluster.cif'])][0][2][field] = wrong
                with self.assertRaisesRegex(ValueError, 'inventory'):self.audit(selection, inv)

    def test_pse_atom_loss_or_metadata_change_stops(self):
        for change in ('atom', 'occupancy', 'bfactor'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                selection, inv, files = self.fixture(Path(tmp)); frames = [inventory(), inventory()]
                if change == 'atom':frames[1].pop()
                else:frames[1][0][change] += .02
                with self.assertRaises(ValueError):self.audit(selection, inv, frames=frames)

    def test_source_tamper_and_frame_count_or_state_order_mismatch_stop(self):
        for change in ('source', 'cif-count', 'pse-count', 'order'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                selection, inv, files = self.fixture(Path(tmp)); count = None
                if change == 'source':files['source-one.pdb'].write_text('Changed frozen input')
                if change == 'cif-count':inv[str(files['cluster.cif'])].pop()
                if change == 'pse-count':count = 1
                if change == 'order':selection['clusters'][0]['selected'].reverse()
                with self.assertRaises(ValueError):self.audit(selection, inv, count=count)

    def test_reloaded_session_and_native_files_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection, inv, files = self.fixture(Path(tmp)); before = {p:adapter.record(p) for p in files.values()}
            self.audit(selection, inv)
            self.assertEqual(before, {p:adapter.record(p) for p in files.values()})
            frames = [inventory(), inventory()]
            class TamperingPyMOL(MockPyMOL):
                def load(self, *args, **kwargs):files['private.pse'].write_text('Changed during mocked read')
            with patch.object(adapter, '_file_inventories', side_effect=lambda path:deepcopy(inv[str(path)])):
                with self.assertRaisesRegex(ValueError, 'changed during audit'):
                    adapter.verify_selected_ensembles(selection, pymol_factory=lambda:TamperingPyMOL(frames))


if __name__ == '__main__':unittest.main()
