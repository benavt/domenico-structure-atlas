import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from snapshot import sha256, verify
from deploy_bundle import unpack
from workflow import validate_construct


class ContractTests(unittest.TestCase):
    def test_construct_rejects_missing_residues_and_extra_chains(self):
        sequence = lambda *s: SimpleNamespace(subsequences=[SimpleNamespace(sequence=v) for v in s])
        validate_construct(sequence('ACDE'),4,'ACDE')
        for model,n in [(sequence('ACD'),3),(sequence('ACDE','ACDE'),8),(sequence('ACDE'),3)]:
            with self.assertRaisesRegex(ValueError,'construct'):validate_construct(model,n,'ACDE')

    def fixture(self, root, mutate=None):
        rows = [dict(target='T', method='m', model_id=f'T__m__{i}', path=f'/{i}.pdb',
                     sha256='a'*64, energy_t010=float(i), **{f'energy_t{j:02d}':float(i) for j in range(16)})
                for i in range(2)]
        if mutate: mutate(rows)
        data = dict(status='validated', models=rows, checkpoint={'sha256':'b'*64}, config_sha256='c'*64, settings={})
        manifest = root / 'models.json'; manifest.write_text(json.dumps(data))
        accepted = dict(status='validated', models=2, energies=32, canonical_manifest_sha256=sha256(manifest),
                        targets={'T':{'methods':{'m':2}}}, checkpoint=data['checkpoint'], config_sha256=data['config_sha256'], settings={})
        acceptance = root / 'acceptance.json'; acceptance.write_text(json.dumps(accepted))
        study = dict(snapshot=dict(manifest_sha256=sha256(manifest), acceptance_sha256=sha256(acceptance)),
                     targets={'T':{'models':2,'methods':{'m':2}}})
        return manifest, acceptance, study

    def test_saved_identity_and_primary_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp)); self.assertEqual(len(verify(*args)),2)
            args = self.fixture(Path(tmp), lambda r:r[0].update(energy_t010=999))
            with self.assertRaisesRegex(ValueError,'alias'): verify(*args)

    def test_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), lambda r:r[1].update(model_id=r[0]['model_id']))
            with self.assertRaisesRegex(ValueError,'Ambiguous'): verify(*args)

    def test_snapshot_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp)); args[0].write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash'): verify(*args)

    def test_site_bundle_allowlist_and_traversal(self):
        for extra in (None, '../escape.html', 'unlisted.html'):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); blobs={'index.html':b'<p>Saved result</p>', '.nojekyll':b''}
                receipt={'schema_version':1,'project':'domenico-structure-atlas','scientific_execution':False,
                         'files':[dict(path=k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in blobs.items()]}
                blobs['export-manifest.json']=json.dumps(receipt).encode()
                if extra: blobs[extra]=b'Unexpected'
                archive=root/'site.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    for name,raw in blobs.items():
                        member=tarfile.TarInfo(name);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
                if extra:
                    with self.assertRaises(ValueError):unpack(archive,sha256(archive),root/'site')
                    self.assertFalse((root/'site').exists())
                else:
                    unpack(archive,sha256(archive),root/'site')
                    self.assertEqual((root/'site/index.html').read_bytes(),blobs['index.html'])

    def test_only_manual_publication_and_exact_pins(self):
        study=json.loads((ROOT/'config/study.json').read_text())
        self.assertIn(study['library_commit'],(ROOT/'pyproject.toml').read_text())
        pages=(ROOT/'.github/workflows/pages.yml').read_text()
        self.assertIn('workflow_dispatch:',pages)
        self.assertNotIn('pip install',pages)
        self.assertNotIn('workflow.py',pages)
        self.assertLess(len((ROOT/'AGENTS.md').read_bytes()),1000)


if __name__ == '__main__':
    unittest.main()
