"""Source-only tiny provider archives; no real native dependencies or fits."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rbp_density as native
import hpc_smoke as smoke


CODE = b'''from .hierarchical_colors import TOKEN
def fit_selection(*args, **kwargs): raise AssertionError("No fits in source checks")
def fit_diagnostics(*args, **kwargs): raise AssertionError("No fits in source checks")
def structural_labels(*args, **kwargs): raise AssertionError("No fits in source checks")
'''
COLORS = b"TOKEN = 'reviewed tiny source fixture'\n"
FILES = {'src/analysis/hd_density_experiment.py': CODE,
         'src/analysis/hierarchical_colors.py': COLORS}


def record(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class DensitySnapshotTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source_root = self.root/'staged'
        for name, content in FILES.items():
            path = self.source_root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        self.path = self.source_root/'src/analysis/hd_density_experiment.py'
        self.enterContext(patch.object(native, 'NATIVE_SHA256', hashlib.sha256(CODE).hexdigest()))
        self.enterContext(patch.object(native, 'COLORS_SHA256', hashlib.sha256(COLORS).hexdigest()))
        before = set(sys.modules)
        def clean_modules():
            for name in set(sys.modules)-before:
                if name.startswith('_rbp_density_native_'):sys.modules.pop(name, None)
        self.addCleanup(clean_modules)
        self.archive = self.root/'provider.tar'
        self.write_archive()

    def write_archive(self, *, files=None, comment=None, extra=(), global_comment=True):
        headers = {'comment': native.NATIVE_COMMIT if comment is None else comment} if global_comment else {}
        with tarfile.open(self.archive, 'w', format=tarfile.PAX_FORMAT, pax_headers=headers) as output:
            for name in ('src', 'src/analysis'):
                entry = tarfile.TarInfo(name); entry.type = tarfile.DIRTYPE; output.addfile(entry)
            for name, content in (FILES if files is None else files).items():
                entry = tarfile.TarInfo(name); entry.size = len(content); output.addfile(entry, io.BytesIO(content))
            for entry, content in extra:
                output.addfile(entry, None if content is None else io.BytesIO(content))

    def source(self):
        return dict(record(self.path), commit=native.NATIVE_COMMIT, source_archive=record(self.archive))

    def test_exact_archive_loads_without_git_and_reports_transport_provenance(self):
        source = self.source()
        with patch.dict(os.environ, PATH='/no-git-command-available'):
            module, proof = native.load_native(source)
        self.assertEqual(module.TOKEN, 'reviewed tiny source fixture')
        for name in ('fit_selection', 'fit_diagnostics', 'structural_labels'):
            self.assertTrue(callable(getattr(module, name)))
        self.assertEqual(proof, dict(path=str(self.path), commit=native.NATIVE_COMMIT,
            sha256=native.NATIVE_SHA256, hierarchical_colors_sha256=native.COLORS_SHA256,
            source_archive=source['source_archive']))
        self.assertEqual(record(self.archive), source['source_archive'])

    def test_archive_hash_size_or_descriptor_changes_fail_closed(self):
        for change in ('hash', 'size', 'missing', 'extra'):
            with self.subTest(change=change):
                source = self.source(); entry = source['source_archive']
                if change=='hash':entry['sha256'] = '0'*64
                elif change=='size':entry['bytes'] += 1
                elif change=='missing':del entry['sha256']
                else:entry['checkout'] = str(self.source_root)
                with self.assertRaisesRegex(ValueError, 'archive (hash|size)|archive path/bytes/SHA'):
                    native.load_native(source)

    def test_global_workstation_commit_is_required_and_exact(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                self.write_archive(comment='0'*40, global_comment=not missing)
                with self.assertRaisesRegex(ValueError, 'global WORKSTATION commit'):
                    native.load_native(self.source())

    def test_payload_must_equal_both_staged_pinned_sources(self):
        for name in FILES:
            with self.subTest(name=name):
                changed = dict(FILES); changed[name] = b'#'+changed[name][1:]
                self.write_archive(files=changed)
                with self.assertRaisesRegex(ValueError, 'payload differs'):
                    native.load_native(self.source())

    def test_staged_native_or_color_tamper_stops_before_import(self):
        for name, message in [('hd_density_experiment.py','primitive hash'), ('hierarchical_colors.py','lineage helper hash')]:
            with self.subTest(name=name):
                source = self.source(); path = self.path.with_name(name); before = path.read_bytes(); path.write_bytes(before+b'#changed')
                with self.assertRaisesRegex(ValueError, message):native.load_native(source)
                path.write_bytes(before)

    def test_archive_allows_only_two_regular_paths_and_their_parent_directories(self):
        cases = [('extra.py', tarfile.REGTYPE), ('../escape.py', tarfile.REGTYPE),
                 ('/absolute.py', tarfile.REGTYPE), ('src//analysis/other.py', tarfile.REGTYPE),
                 ('src/analysis/link.py', tarfile.SYMTYPE), ('src/analysis/hard.py', tarfile.LNKTYPE),
                 ('src/analysis/extra', tarfile.DIRTYPE), ('src', tarfile.DIRTYPE),
                 ('src/analysis/hd_density_experiment.py', tarfile.REGTYPE)]
        for name, kind in cases:
            with self.subTest(name=name, kind=kind):
                entry = tarfile.TarInfo(name); entry.type = kind; entry.linkname = 'hd_density_experiment.py'
                content = b'bad' if kind==tarfile.REGTYPE else None
                if content is not None:entry.size = len(content)
                self.write_archive(extra=[(entry, content)])
                with self.assertRaisesRegex(ValueError, 'archive member|archive member/path/type'):
                    native.load_native(self.source())
        self.write_archive(files={next(iter(FILES)): CODE})
        with self.assertRaisesRegex(ValueError, 'complete payload'):native.load_native(self.source())

    def test_members_hidden_after_tar_padding_are_still_rejected(self):
        self.write_archive()
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode='w') as output:
            entry = tarfile.TarInfo('hidden.py'); entry.size = 1; output.addfile(entry, io.BytesIO(b'x'))
        with self.archive.open('ab') as stream:stream.write(payload.getvalue())
        with self.assertRaisesRegex(ValueError, 'archive member/path/type'):native.load_native(self.source())

    def test_archive_and_staged_symlinks_are_rejected(self):
        alias = self.root/'alias.tar'; alias.symlink_to(self.archive)
        source = self.source(); source['source_archive']['path'] = str(alias)
        with self.assertRaisesRegex(ValueError, 'non-symlink'):native.load_native(source)
        alias.unlink(); self.archive.rename(alias); self.archive.symlink_to(alias)
        with self.assertRaisesRegex(ValueError, 'non-symlink'):native.load_native(self.source())
        self.archive.unlink(); alias.rename(self.archive)
        for name in ('hd_density_experiment.py', 'hierarchical_colors.py'):
            with self.subTest(staged=name):
                source = self.source(); path = self.path.with_name(name); saved = path.with_suffix('.saved')
                path.rename(saved); path.symlink_to(saved)
                with self.assertRaisesRegex(ValueError, 'non-symlink'):native.load_native(source)
                path.unlink(); saved.rename(path)

    @unittest.skipUnless(shutil.which('git'), 'Git required for unchanged legacy checkout protocol')
    def test_legacy_mode_still_requires_real_matching_git_head(self):
        source = self.source(); del source['source_archive']
        with self.assertRaises(subprocess.CalledProcessError):native.load_native(source)
        def git(*arguments):
            return subprocess.check_output(['git','-C',str(self.source_root),*arguments], text=True, stderr=subprocess.PIPE).strip()
        git('init','-q'); git('add','src')
        git('-c','user.name=Source fixture','-c','user.email=fixture@example.invalid','commit','-qm','Tiny source fixture')
        commit = git('rev-parse','HEAD')
        with patch.object(native,'NATIVE_COMMIT',commit):
            source['commit'] = commit; module, proof = native.load_native(source)
            self.assertTrue(callable(module.fit_selection)); self.assertNotIn('source_archive',proof)
            git('-c','user.name=Source fixture','-c','user.email=fixture@example.invalid','commit','-qm','Changed revision','--allow-empty')
            with self.assertRaisesRegex(ValueError,'checkout revision changed'):native.load_native(source)

    def smoke_fixture(self):
        root = self.root/'run'; root.mkdir(); prefix = self.root/'runtime'; outside = self.root/'outside'; outside.mkdir()
        plan = dict(run_root=str(root),runtime_prefix=str(prefix),runtime_python=str(prefix/'bin/python'),
            smoke=dict(outside_checkout_roots=[str(self.source_root)],library=dict(version='fixture'),
                density_provider=self.source(),cli=dict(analysis_runtime_check=True)))
        environment = {name:str(root/name) for name in ('TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME')}
        environment.update({name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')})
        commands = []
        def command(arguments, **kwargs):
            commands.append(arguments)
            self.assertNotEqual(arguments[0],'git','Archive smoke must not ask for checkout/Git')
            return SimpleNamespace(returncode=0,stderr='',stdout='fixture\n' if arguments[-1]=='--version'
                else json.dumps(dict(runtime_ready=True,profile='analysis_only')))
        adapters = dict(executable=plan['runtime_python'],prefix=str(prefix),cwd=str(outside),environment=environment,
            observe=lambda _:dict(smoke_containment_verified=True,fit_containment_verified=True),run=command,
            provider=native.load_native)
        return plan, adapters, commands

    def test_smoke_archive_branch_uses_real_loader_proof_without_checkout_or_git(self):
        plan, adapters, commands = self.smoke_fixture()
        with patch.object(smoke,'IMPORTS',()), patch.object(smoke,'_installed_wheel',return_value={'_verified_origins':[]}):
            result = smoke.run_smoke(plan,Path(plan['run_root'])/'smoke',adapters=adapters)
        self.assertTrue(result['smoke_checks_passed']); self.assertEqual(len(commands),2)
        self.assertEqual(result['density_provider']['proof']['source_archive'],plan['smoke']['density_provider']['source_archive'])
        self.assertEqual(result['scientific_fits'],0); self.assertEqual(result['density_provider']['scientific_fits'],0)

    def test_smoke_archive_change_is_blocked_before_provider_import(self):
        plan, adapters, _ = self.smoke_fixture(); plan['smoke']['density_provider']['source_archive']['sha256'] = '0'*64
        with patch.object(smoke,'IMPORTS',()), patch.object(smoke,'_installed_wheel',return_value={'_verified_origins':[]}):
            result = smoke.run_smoke(plan,Path(plan['run_root'])/'smoke',adapters=adapters)
        self.assertEqual(result['status'],'blocked'); self.assertIsNone(result['density_provider'])
        self.assertIn('Bound file SHA differs',json.dumps(result['errors']))


if __name__=='__main__':unittest.main()
