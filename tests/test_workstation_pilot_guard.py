"""Mock guard/controller only; no real unit, environment setup or fit."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import workstation_pilot_guard as subject


class WorkstationGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'execution-plan.json'
        self.path.write_text('{}')
        own = dict(path=str(Path(subject.__file__).absolute()), bytes=1, sha256='s')
        workload = dict(path=str(self.root / 'workload.sbatch'), bytes=1, sha256='w')
        guard = dict(path=str(self.root / 'guard-source/structure_clustering/resource_guard.py'),
                     bytes=1, sha256=subject.GUARD_SHA256)
        self.plan = dict(execution_host='workstation', phase='pilot64', run_root=str(self.root / 'run'),
            runtime_prefix=str(self.root / 'new-env'), guard_output=str(self.root / 'guard-output'),
            bootstrap=dict(python=sys.executable, prefix=sys.prefix, binary=dict(path=sys.executable)),
            sources=dict(guard=dict(checkout=str(self.root / 'guard-source'), commit=subject.GUARD_COMMIT)),
            runtime_guard=guard, scoped_workload=workload, frozen_files=[own, workload, guard],
            resources=dict(maximum_wall_seconds=60),
            guard=dict(admission=dict(minimum_available_memory_bytes=4096),
                       runtime=dict(minimum_available_memory_bytes=2048), interval_seconds=5, consecutive_failures=3))
        self.observation = dict(execution_host='workstation', smoke_containment_verified=True,
            **{key:dict(verified=True) for key in ('scheduler', 'runtime', 'cpu', 'memory', 'swap', 'pids')})
        self.cpu = SimpleNamespace(verify=Mock(return_value=self.plan),
            bound=Mock(side_effect=lambda entry:Path(entry['path'])), check_disk=Mock(),
            sha=Mock(return_value='plan-hash'), save=self.save)
        self.contract = SimpleNamespace(observe=Mock(return_value=self.observation),
            sanitized_environment=Mock(return_value=dict(PATH='/approved/bin', HOME='/keep-home', TMPDIR='/owned/tmp')))
        self.guard = SimpleNamespace(ResourceLimits=Mock(side_effect=lambda **kwargs:kwargs),
                                     run_guarded=Mock(side_effect=self.fake_guard))
        self.events = []

    @staticmethod
    def save(path, value):
        Path(path).write_text(json.dumps(value))

    def fake_guard(self, command, output, **options):
        self.events.append(('guard', command, output, options, dict(os.environ)))
        Path(output).mkdir()
        return 0

    def run_mock(self):
        with patch.object(subject, '_modules', return_value=(self.cpu, self.contract)), \
                patch.object(subject, '_load_guard', return_value=self.guard) as load:
            result = subject.run(self.path)
        return result, load

    def test_approval_failure_precedes_observation_guard_or_setup(self):
        self.cpu.verify.side_effect = ValueError('exact approval')
        with patch.object(subject, '_modules', return_value=(self.cpu, self.contract)), \
                patch.object(subject, '_load_guard') as load, self.assertRaises(ValueError):
            subject.run(self.path)
        self.contract.observe.assert_not_called()
        load.assert_not_called()
        self.assertFalse(Path(self.plan['guard_output']).exists())

    def test_bootstrap_kernel_controls_precede_guard_import(self):
        for key in ('scheduler', 'runtime', 'cpu', 'memory', 'swap', 'pids'):
            self.observation[key]['verified'] = False
            with self.subTest(key=key), patch.object(subject, '_modules', return_value=(self.cpu, self.contract)), \
                    patch.object(subject, '_load_guard') as load, self.assertRaisesRegex(ValueError, 'containment'):
                subject.run(self.path)
            load.assert_not_called()
            self.observation[key]['verified'] = True
        self.assertFalse(Path(self.plan['guard_output']).exists())

    def test_wrong_host_bootstrap_and_full_cohort_phase_are_rejected(self):
        for key, value in [('execution_host', 'hpc'), ('phase', 'umap-full'),
                           ('bootstrap', dict(self.plan['bootstrap'], python='/wrong/python'))]:
            original = self.plan[key]
            self.plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_mock()
            self.plan[key] = original

    def test_hash_bound_exact_b561_guard_required(self):
        for key, value in [('sha256', '0' * 64), ('path', str(self.root / 'foreign.py'))]:
            original = self.plan['runtime_guard'][key]
            self.plan['runtime_guard'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_mock()
            self.plan['runtime_guard'][key] = original
        self.plan['sources']['guard']['commit'] = 'f' * 40
        with self.assertRaises(ValueError):
            self.run_mock()

    def test_guard_destination_must_be_fresh_separate_and_non_symlink(self):
        for destination in (self.plan['run_root'], str(Path(self.plan['run_root']) / 'control'),
                            str(Path(self.plan['runtime_prefix']) / 'control')):
            self.plan['guard_output'] = destination
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.run_mock()
        self.plan['guard_output'] = str(self.root / 'old')
        Path(self.plan['guard_output']).mkdir()
        with self.assertRaises(ValueError):
            self.run_mock()
        self.plan['guard_output'] = str(self.root / 'link')
        Path(self.plan['guard_output']).symlink_to(self.root / 'old')
        with self.assertRaises(ValueError):
            self.run_mock()

    def test_exact_bash_and_pressure_settings_preserve_environment_and_logs(self):
        original = dict(os.environ)
        code, load = self.run_mock()
        self.assertEqual(code, 0)
        self.assertEqual(dict(os.environ), original)
        event = self.events[0]
        self.assertEqual(event[1], ['/bin/bash', self.plan['scoped_workload']['path']])
        self.assertEqual(event[2], Path(self.plan['guard_output']))
        self.assertEqual(event[3], dict(admission=self.plan['guard']['admission'],
            runtime=self.plan['guard']['runtime'], interval_seconds=5, maximum_wall_seconds=60, consecutive_failures=3))
        self.assertEqual(event[4], self.contract.sanitized_environment.return_value)
        actual = self.contract.observe.call_args.args[0]
        self.assertEqual(actual['runtime_python'], sys.executable)
        self.assertEqual(actual['runtime_prefix'], sys.prefix)
        load.assert_called_once_with(Path(self.plan['runtime_guard']['path']))
        receipt = json.loads((Path(self.plan['guard_output']) / 'launcher.json').read_text())
        self.assertEqual(receipt['returncode'], 0)
        self.assertEqual(receipt['wrapper_scientific_fits'], 0)

    def test_guard_failure_is_not_retried_or_relabelled_complete(self):
        self.guard.run_guarded.side_effect = RuntimeError('pressure stop')
        before = dict(os.environ)
        with self.assertRaisesRegex(RuntimeError, 'pressure stop'):
            self.run_mock()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(self.guard.run_guarded.call_count, 1)
        self.assertFalse((Path(self.plan['guard_output']) / 'launcher.json').exists())

    def test_actual_nonzero_child_code_propagates(self):
        def failure(*args, **kwargs):
            Path(args[1]).mkdir()
            return 17
        self.guard.run_guarded.side_effect = failure
        code, _ = self.run_mock()
        self.assertEqual(code, 17)
        receipt = json.loads((Path(self.plan['guard_output']) / 'launcher.json').read_text())
        self.assertEqual(receipt['returncode'], 17)

    def test_unreviewed_cooperative_policy_is_rejected(self):
        self.plan['guard']['io_recovery'] = dict(accounting_mode='worker')
        with self.assertRaisesRegex(ValueError, 'strict pressure'):
            self.run_mock()
        self.contract.observe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
