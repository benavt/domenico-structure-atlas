"""Fake processes/controllers only: no unit creation, control or native imports."""
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import hpc_scope as subject


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'execution-plan.json'
        self.path.write_text('{"static":"source-only fixture"}\n')
        self.uid = os.getuid()
        self.plan = dict(phase='smoke', containment=dict(mode='user_scope',
            scope_memory_bytes=8 * 1024**3, scope_tasks=64, scope_max_seconds=3600),
            resources=dict(maximum_memory_bytes=8 * 1024**3, maximum_pids=64,
                           maximum_wall_seconds=3600, maximum_swap_bytes=0),
            bootstrap=dict(python='/bound/bootstrap/bin/python', prefix='/bound/bootstrap'),
            scoped_workload=dict(path=str(self.root / 'scoped.sbatch')))
        self.unit = 'domenico-cpu-771-' + subject.digest(self.path)[:8] + '.scope'
        scope = '/user.slice/user-{0}.slice/user@{0}.service/app.slice/{1}'.format(self.uid, self.unit)
        self.binding = dict(schema=1, plan_sha256=subject.digest(self.path),
            supervisor_pid=987, supervisor_start_time='123456',
            slurm_cgroup='/system.slice/slurmstepd.scope/job_771/step_batch/user/task_0',
            cpus=[2, 50], unit=self.unit, scope_cgroup=scope, uid=self.uid, job_id=771)
        self.binding_path = self.root / subject.BINDING_NAME

    def write_binding(self):
        subject._save(self.binding_path, self.binding)
        return subject.digest(self.binding_path)

    def test_scope_command_has_exact_private_limits_and_no_admin_request(self):
        value = subject._scope_command(self.path, self.plan, self.binding,
                                       self.binding_path, 'a' * 64)
        self.assertEqual(value[:5], [subject.SYSTEMD_RUN, '--user', '--scope', '--collect', '--no-ask-password'])
        for item in ('MemoryMax=8589934592', 'MemorySwapMax=0', 'TasksMax=64',
                     'RuntimeMaxSec=3600', 'TimeoutStopSec=5',
                     'KillMode=control-group', 'SendSIGKILL=yes'):
            self.assertIn('--property=' + item, value)
        self.assertIn('--slice=app.slice', value)
        self.assertIn('--unit=' + self.unit, value)
        self.assertEqual(value[-4:], ['keeper', str(self.path), str(self.binding_path), 'a' * 64])
        self.assertFalse(any('sudo' in item or 'OOMPolicy' in item or 'MemoryOOMGroup' in item for item in value))

    def test_scope_settings_reject_science_or_expanded_resources(self):
        for field, changed in [('phase', 'umap-full'), ('resources', dict(self.plan['resources'], maximum_swap_bytes=1))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject._scope_settings(dict(self.plan, **{field: changed}))
        for field, changed in [('mode', 'system_scope'), ('scope_tasks', 65), ('scope_max_seconds', 3601)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject._scope_settings(dict(self.plan, containment=dict(self.plan['containment'], **{field: changed})))

    def test_binding_requires_exact_plan_and_private_scope(self):
        sha = self.write_binding()
        with patch.dict(os.environ, SLURM_JOB_ID='771'):
            self.assertEqual(subject._read_binding(self.path, self.plan, self.binding_path, sha), self.binding)
            for field, changed in [('scope_cgroup', '/user.slice'), ('unit', 'user@0.service'),
                                   ('uid', self.uid + 1), ('plan_sha256', 'f' * 64)]:
                self.binding_path.write_text(json.dumps(dict(self.binding, **{field: changed})))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    subject._read_binding(self.path, self.plan, self.binding_path, subject.digest(self.binding_path))

    def test_hash_mismatch_and_symlinked_binding_fail(self):
        self.write_binding()
        with self.assertRaises(ValueError):
            subject._read_binding(self.path, self.plan, self.binding_path, '0' * 64)
        copy = self.root / 'copy.json'
        self.binding_path.rename(copy)
        self.binding_path.symlink_to(copy)
        with self.assertRaises(ValueError):
            subject._read_binding(self.path, self.plan, self.binding_path, subject.digest(copy))

    def test_supervisor_requires_live_uid_starttime_and_slurm_membership(self):
        good = dict(uid=self.uid, start_time='123456', cgroup='0::' + self.binding['slurm_cgroup'])
        with patch.object(subject, '_identity', return_value=good):
            self.assertEqual(subject._supervisor(self.binding), good)
        for field, changed in [('uid', self.uid + 1), ('start_time', 'new'), ('cgroup', '0::/user.slice')]:
            with self.subTest(field=field), patch.object(subject, '_identity', return_value=dict(good, **{field: changed})), self.assertRaises(ValueError):
                subject._supervisor(self.binding)

    def test_supervisor_ancestry_cannot_use_an_unrelated_same_uid_process(self):
        good = dict(uid=self.uid, start_time='123456', cgroup='0::' + self.binding['slurm_cgroup'])
        with patch.object(subject.os, 'getppid', return_value=111), \
                patch.object(subject, '_identity', side_effect=[good, dict(ppid=1)]), \
                self.assertRaisesRegex(ValueError, 'descendant'):
            subject._supervisor(self.binding, require_ancestor=True)

    def test_pidfd_and_parent_death_arm_recheck_before_any_work(self):
        poller = Mock()
        poller.poll.return_value = []
        library = Mock()
        library.prctl.return_value = 0
        with patch.object(subject, '_supervisor') as parent, \
                patch.object(subject.os, 'getppid', return_value=987), \
                patch.object(subject, '_pidfd_open', return_value=77) as pidfd, \
                patch.object(subject.ctypes, 'CDLL', return_value=library), \
                patch.object(subject.select, 'poll', return_value=poller):
            handle, actual = subject._arm_parent(self.binding)
        self.assertEqual((handle, actual), (77, poller))
        pidfd.assert_called_once_with(987)
        self.assertEqual(parent.call_count, 2)
        library.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)

    def test_parent_exit_during_arm_closes_handle_and_fails(self):
        library = Mock()
        library.prctl.return_value = 0
        with patch.object(subject, '_supervisor'), \
                patch.object(subject.os, 'getppid', side_effect=[987, 1]), \
                patch.object(subject, '_pidfd_open', return_value=77), \
                patch.object(subject.os, 'close') as close, \
                patch.object(subject.ctypes, 'CDLL', return_value=library), \
                self.assertRaises(ValueError):
            subject._arm_parent(self.binding)
        close.assert_called_once_with(77)

    def test_pidfd_prefers_python_api_and_supports_missing_python_wrapper(self):
        with patch.object(subject.os, 'pidfd_open', return_value=72, create=True) as native:
            self.assertEqual(subject._pidfd_open(987), 72)
        native.assert_called_once_with(987, 0)
        library = Mock()
        library.pidfd_open = None
        library.syscall.return_value = 73
        with patch.object(subject.os, 'pidfd_open', None, create=True), \
                patch.object(subject.ctypes, 'CDLL', return_value=library), \
                patch.object(subject.os, 'uname', return_value=SimpleNamespace(machine='x86_64')):
            self.assertEqual(subject._pidfd_open(987), 73)
        self.assertEqual(library.syscall.call_args.args[0].value, 434)

    def test_pidfd_missing_or_failed_kernel_api_blocks_before_work(self):
        library = Mock()
        library.pidfd_open = None
        library.syscall.return_value = -1
        with patch.object(subject.os, 'pidfd_open', None, create=True), \
                patch.object(subject.ctypes, 'CDLL', return_value=library), \
                patch.object(subject.os, 'uname', return_value=SimpleNamespace(machine='x86_64')), \
                patch.object(subject.ctypes, 'get_errno', return_value=38), \
                self.assertRaises(OSError):
            subject._pidfd_open(987)
        library.syscall.reset_mock()
        with patch.object(subject.os, 'pidfd_open', None, create=True), \
                patch.object(subject.ctypes, 'CDLL', return_value=library), \
                patch.object(subject.os, 'uname', return_value=SimpleNamespace(machine='unreviewed')), \
                self.assertRaises(ValueError):
            subject._pidfd_open(987)
        library.syscall.assert_not_called()

    def test_parent_death_prevents_polling_or_continuing_workload(self):
        process = Mock()
        poller = Mock()
        poller.poll.return_value = [(77, 1)]
        with self.assertRaisesRegex(RuntimeError, 'supervisor'):
            subject._monitor(process, parent_poll=poller, seconds=10)
        process.poll.assert_not_called()

    def test_deadline_and_cancellation_are_fail_closed(self):
        process = Mock()
        process.poll.return_value = None
        with self.assertRaises(TimeoutError):
            subject._monitor(process, seconds=1, clock=Mock(side_effect=[0, 2]), sleep=Mock())
        with self.assertRaises(RuntimeError):
            subject._monitor(process, seconds=1, cancelled=lambda: True)

    def test_kernel_caps_only_configure_oom_on_exact_owned_fake_scope(self):
        leaf = self.root / self.binding['scope_cgroup'].lstrip('/')
        leaf.mkdir(parents=True)
        for key, value in {'memory.max': 8 * 1024**3, 'memory.swap.max': 0,
                           'pids.max': 64, 'memory.oom.group': 0, 'memory.swap.current': 0}.items():
            (leaf / key).write_text(str(value))
        identity = dict(uid=self.uid, cgroup='0::' + self.binding['scope_cgroup'])
        with patch.object(subject, '_identity', return_value=identity), \
                patch.object(subject.os, 'sched_getaffinity', return_value={2, 50}):
            subject._verify_caps(self.binding, self.plan, cgroup_root=self.root, configure_oom=True)
            self.assertEqual((leaf / 'memory.oom.group').read_text(), '1')
            (leaf / 'pids.max').write_text('max')
            with self.assertRaisesRegex(ValueError, 'pids.max'):
                subject._verify_caps(self.binding, self.plan, cgroup_root=self.root)

    def test_wrong_scope_does_not_write_even_oom_flag(self):
        with patch.object(subject, '_identity', return_value=dict(uid=self.uid, cgroup='0::/user.slice')), \
                patch.object(subject, '_scope_leaf') as leaf, self.assertRaises(ValueError):
            subject._verify_caps(self.binding, self.plan, cgroup_root=self.root, configure_oom=True)
        leaf.assert_not_called()

    def test_cleanup_stops_only_exact_unit_after_controlgroup_readback(self):
        run = Mock(side_effect=[SimpleNamespace(returncode=0, stdout='LoadState=loaded\nControlGroup=' + self.binding['scope_cgroup']),
                                SimpleNamespace(returncode=0, stdout='')])
        subject._stop_scope(self.binding, {}, run=run)
        self.assertEqual(run.call_args_list[-1].args[0],
                         [subject.SYSTEMCTL, '--user', '--no-ask-password', 'stop', self.unit])
        run = Mock(return_value=SimpleNamespace(returncode=0, stdout='LoadState=loaded\nControlGroup=/user.slice'))
        with self.assertRaises(ValueError):
            subject._stop_scope(self.binding, {}, run=run)
        self.assertEqual(run.call_count, 1)

    def test_keeper_self_stop_accepts_only_expected_signal_results(self):
        for code in (-signal.SIGTERM,-signal.SIGKILL):
            def fake():return Mock(side_effect=[SimpleNamespace(returncode=0,stdout='LoadState=loaded\nControlGroup='+self.binding['scope_cgroup']),SimpleNamespace(returncode=code,stdout='')])
            subject._stop_scope(self.binding,{},run=fake(),self_stop=True)
            with self.assertRaises(ValueError):subject._stop_scope(self.binding,{},run=fake())
        run=Mock(side_effect=[SimpleNamespace(returncode=0,stdout='LoadState=loaded\nControlGroup='+self.binding['scope_cgroup']),SimpleNamespace(returncode=1,stdout='')])
        with self.assertRaises(ValueError):subject._stop_scope(self.binding,{},run=run,self_stop=True)

    def test_absent_scope_cleanup_never_stops_parent_user_manager(self):
        run = Mock(return_value=SimpleNamespace(returncode=4, stdout='LoadState=not-found'))
        subject._stop_scope(self.binding, {}, run=run)
        self.assertEqual(run.call_count, 1)
        with self.assertRaises(ValueError):
            subject._stop_scope(dict(self.binding, scope_cgroup='/'), {}, run=run)

    def test_keeper_normal_and_failure_status_are_saved_before_whole_scope_cleanup(self):
        for code in (0, 19):
            with self.subTest(code=code):
                for name in ('scope-ready.json', 'scope-workload-exit.json'):
                    (self.root / name).unlink(missing_ok=True)
                process = Mock()
                contract = Mock()
                cpu = Mock()
                cpu.bound.return_value = self.root / 'scoped.sbatch'
                cleanup = Mock()
                cleanup.side_effect = lambda *args, **kwargs: self.assertEqual(
                    json.loads((self.root / 'scope-workload-exit.json').read_text())['workload_returncode'], code)
                with patch.object(subject, '_load', return_value=(self.path, self.plan, contract)), \
                        patch.object(subject, '_read_binding', return_value=self.binding), \
                        patch.object(subject, '_bus_environment', return_value={}), \
                        patch.object(subject, '_arm_parent', return_value=(77, Mock())), \
                        patch.object(subject.os, 'sched_setaffinity'), \
                        patch.object(subject.os, 'close'), \
                        patch.object(subject, '_verify_caps'), \
                        patch.object(subject, '_modules', return_value=(cpu, contract)), \
                        patch.object(subject.subprocess, 'Popen', return_value=process) as popen, \
                        patch.object(subject, '_monitor', return_value=code), \
                        patch.object(subject, '_stop_scope', cleanup):
                    actual = subject.keeper(self.path, self.binding_path, 'c' * 64)
                self.assertEqual(actual, code)
                self.assertEqual(popen.call_args.args[0], ['/bin/bash', str(self.root / 'scoped.sbatch')])
                env = popen.call_args.kwargs['env']
                self.assertEqual(env['HPC_SCOPE_BINDING_SHA256'], 'c' * 64)
                cleanup.assert_called_once()

    def test_keeper_parent_loss_cleans_scope_without_false_exit_receipt(self):
        with patch.object(subject, '_load', return_value=(self.path, self.plan, Mock())), \
                patch.object(subject, '_read_binding', return_value=self.binding), \
                patch.object(subject, '_bus_environment', return_value={}), \
                patch.object(subject, '_arm_parent', side_effect=RuntimeError('parent gone')), \
                patch.object(subject.subprocess, 'Popen') as popen, \
                patch.object(subject, '_stop_scope') as cleanup, self.assertRaises(RuntimeError):
            subject.keeper(self.path, self.binding_path, 'c' * 64)
        popen.assert_not_called()
        cleanup.assert_called_once()
        self.assertFalse((self.root / 'scope-workload-exit.json').exists())

    def test_supervisor_returns_actual_receipt_status_after_keeper_cleanup_signal(self):
        def completed(*args, **kwargs):
            subject._save(self.root / 'scope-workload-exit.json', dict(
                plan_sha256=subject.digest(self.path), binding_sha256=subject.digest(self.binding_path),
                workload_returncode=0))
            return -signal.SIGTERM
        with patch.object(subject, '_load', return_value=(self.path, self.plan, Mock())), \
                patch.object(subject, '_binding', return_value=self.binding), \
                patch.object(subject, '_bus_environment', return_value={}), \
                patch.object(subject.subprocess, 'run', return_value=SimpleNamespace(stdout='LoadState=not-found')), \
                patch.object(subject.subprocess, 'Popen'), \
                patch.object(subject, '_monitor', side_effect=completed), \
                patch.object(subject, '_stop_scope') as cleanup:
            self.assertEqual(subject.launch(self.path), 0)
        cleanup.assert_called_once()

    def test_existing_scope_is_not_started_or_stopped(self):
        with patch.object(subject, '_load', return_value=(self.path, self.plan, Mock())), \
                patch.object(subject, '_binding', return_value=self.binding), \
                patch.object(subject, '_bus_environment', return_value={}), \
                patch.object(subject.subprocess, 'run', return_value=SimpleNamespace(stdout='LoadState=loaded')), \
                patch.object(subject.subprocess, 'Popen') as popen, \
                patch.object(subject, '_stop_scope') as cleanup, self.assertRaises(ValueError):
            subject.launch(self.path)
        popen.assert_not_called()
        cleanup.assert_not_called()

    def test_module_imports_only_standard_library(self):
        source = Path(subject.__file__).read_text()
        names = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module.split('.')[0])
        self.assertFalse(names & {'numpy', 'umap', 'sklearn', 'numba', 'torch', 'structure_clustering'})


if __name__ == '__main__':
    unittest.main()
