"""Tiny zip/text contracts only; no installation or scientific work."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import mimas_package as package


class PackageTests(unittest.TestCase):
    def fixture(self, root, *, extra=None):
        path = root/'structure_clustering-0.1.0-py3-none-any.whl'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('structure_clustering/__init__.py', b'__version__="0.1.0"\n')
            z.writestr('structure_clustering-0.1.0.dist-info/METADATA', 'Name: structure-clustering\nVersion: 0.1.0\n')
            if extra is not None:
                z.writestr(extra, 'blocked')
        return path

    def test_wheel_every_payload_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            result = package.wheel_contract(path, 'a'*40)
            self.assertEqual(result['commit'], 'a'*40)
            self.assertEqual(set(result['filehashes']), {'structure_clustering/__init__.py'})
            self.assertEqual(result['wheel']['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            lock = package.requirements_from_wheels(Path(tmp))
            self.assertEqual(lock, 'structure-clustering==0.1.0 --hash=sha256:'+result['wheel']['sha256']+'\n')

    def test_reject_unsafe_or_bulk_wheel_payload(self):
        for name in ['../escape.py', '/absolute.py', 'x\\escape.py', 'outputs/science.npy', 'data/raw.bin', 'x.data/purelib/a.py']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    package.wheel_contract(self.fixture(Path(tmp), extra=name), 'a'*40)

    def test_batch_is_bounded_offline_and_approval_first(self):
        plan = {'run_root': '/run/new', 'runtime_prefix': '/runtime/new',
                'runtime_python': '/runtime/new/bin/python', 'wheelhouse': '/payload/wheels',
                'requirements': '/payload/requirements.txt', 'runner': '/source/mimas_cpu.py',
                'partition': 'debug', 'resources': {'maximum_cpus': 1,
                'maximum_memory_bytes': 4*1024**3, 'maximum_wall_seconds': 900}}
        body = package.smoke_batch('/payload/plan.json', plan, '/base/bin/python')
        self.assertIn('#SBATCH --mem=4096M', body)
        self.assertIn('#SBATCH --time=15', body)
        self.assertIn('--require-hashes', body)
        self.assertNotIn('sbatch ', body)
        self.assertNotIn('--gres', body)
        self.assertLess(body.index('mimas_cpu.py verify'), body.index('-m venv'))
        self.assertLess(body.index('test ! -e'), body.index('-m venv'))
        with tempfile.NamedTemporaryFile('w', suffix='.sh') as f:
            f.write(body);f.flush()
            self.assertEqual(subprocess.run(['/bin/bash', '-n', f.name], capture_output=True).returncode, 0)
        existing = package.smoke_batch('/payload/plan.json', plan, '/base/bin/python', environment_exists=True)
        self.assertNotIn('-m venv', existing)
        self.assertNotIn('pip install', existing)

    def test_scoped_batch_enters_scope_before_any_environment_creation(self):
        plan = {'run_root':'/run/new','runtime_prefix':'/runtime/new','runtime_python':'/runtime/new/bin/python',
                'wheelhouse':'/payload/wheels','requirements':'/payload/requirements.txt','runner':'/source/mimas_cpu.py',
                'partition':'debug','resources':{'maximum_cpus':1,'maximum_memory_bytes':8*1024**3,'maximum_wall_seconds':3600},
                'containment':{'mode':'user_scope'}}
        body=package.smoke_batch('/payload/plan.json',plan,'/base/bin/python')
        self.assertIn('mimas_scope.py launch',body);self.assertIn('#SBATCH --no-requeue',body)
        self.assertNotIn('pip install',body);self.assertNotIn('-m venv',body)
        plain=dict(plan);plain.pop('containment')
        inner=package.smoke_batch('/payload/plan.json',plain,'/base/bin/python')
        self.assertNotIn('mimas_scope.py',inner);self.assertIn('-m venv',inner)

    def test_batch_rejects_injection_and_unbounded_requests(self):
        plan = {'run_root': '/run/new', 'runtime_prefix': '/runtime/new',
                'runtime_python': '/runtime/new/bin/python', 'wheelhouse': '/payload/wheels',
                'requirements': '/payload/requirements.txt', 'runner': '/source/mimas_cpu.py',
                'partition': 'debug\n#SBATCH --gres=gpu:1', 'resources': {'maximum_cpus': 1,
                'maximum_memory_bytes': 4*1024**3, 'maximum_wall_seconds': 900}}
        with self.assertRaises(ValueError):package.smoke_batch('/payload/plan.json', plan, '/base/bin/python')
        plan['partition']='debug';plan['resources']['maximum_cpus']=100
        with self.assertRaises(ValueError):package.smoke_batch('/payload/plan.json', plan, '/base/bin/python')


if __name__ == '__main__':
    unittest.main()
