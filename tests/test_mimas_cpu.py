"""Controller checks use static manifests and fake children; no scientific imports."""
from copy import deepcopy
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
import mimas_cpu as subject


class FakeContract:
    def __init__(self, observations=None):
        self.observations = list(observations or [self.observation()])
        self.observed = []
        self.fit_checks = []
        self.environment = None

    @staticmethod
    def observation(*, smoke=True, fit=True):
        return dict(status='smoke_observed' if smoke else 'blocked',
                    smoke_containment_verified=smoke, fit_containment_verified=fit,
                    fit_blocking_reasons=[] if fit else ['Unverified swap/pids'])

    def sanitized_environment(self, plan):
        if self.environment is not None:
            return dict(self.environment)
        root = Path(plan['run_root'])
        return dict(TMPDIR=str(root / 'tmp'), MPLCONFIGDIR=str(root / 'mpl'),
                    NUMBA_CACHE_DIR=str(root / 'numba'), XDG_CACHE_HOME=str(root / 'cache'))

    def observe(self, plan, **kwargs):
        self.observed.append((deepcopy(plan), kwargs))
        result = self.observations[0]
        if len(self.observations) > 1:
            self.observations.pop(0)
        return deepcopy(result)

    def require_fit_ready(self, observation, **kwargs):
        self.fit_checks.append((deepcopy(observation), kwargs))
        if not observation.get('fit_containment_verified'):
            raise ValueError('Fit containment is unverified')


class FakeProcess:
    def __init__(self, returncode=0, polls=None):
        self.pid = 123456789
        self.returncode = returncode
        self.polls = list(polls or [])
        self.wait = Mock(return_value=returncode)

    def poll(self):
        if self.polls:
            result = self.polls.pop(0)
            if result is not None:
                self.returncode = result
            return result
        return self.returncode


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.plan_path = self.directory / 'plan.json'
        self.root = self.directory / 'run'
        self.contract = FakeContract()
        self.commands = []
        self.processes = []
        self.modules_patch = patch.object(subject, '_modules', return_value=self.contract)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)

    def plan(self, phase='smoke', **updates):
        source = Path(subject.__file__).absolute().parent
        value = dict(operation='mimas-cpu-validation', phase=phase,
                     stages=list(subject.STAGES[phase]),
                     maximum_projection_fits=subject.FIT_LIMITS[phase][0],
                     maximum_hdbscan_extractions=subject.FIT_LIMITS[phase][1],
                     frozen_files=[subject.record(source / name) for name in
                                   ('mimas_cpu.py', 'mimas_contract.py', 'mimas_smoke.py', 'mimas_pilot.py')],
                     run_root=str(self.root), runtime_python='/approved/runtime/bin/python',
                     runtime_prefix='/approved/runtime',
                     resources=dict(maximum_cpus=1, maximum_memory_bytes=2 * 1024**3,
                                    maximum_swap_bytes=0, maximum_pids=64, maximum_wall_seconds=30),
                     inputs=dict(model_count=5072 if phase == 'umap-full' else 64))
        value.update(updates)
        self.write_plan(value)
        return value

    def write_plan(self, value, *, approved=True):
        self.plan_path.write_text(json.dumps(value))
        (self.directory / 'approval.json').write_text(json.dumps(
            dict(approved=approved, plan_sha256=subject.sha(self.plan_path))))

    def launch(self, command, **kwargs):
        self.commands.append((command, kwargs))
        stage = command[-3]
        output = self.root / 'stages' / stage
        output.mkdir(exist_ok=True)
        receipt = (dict(status='smoke_complete', smoke_checks_passed=True, fit_ready=True)
                   if stage == 'smoke' else dict(status='complete', phase_complete=True))
        subject.save(output / ('smoke.json' if stage == 'smoke' else 'receipt.json'), receipt)
        process = FakeProcess()
        self.processes.append(process)
        return process

    def run_fake(self, *, launch=None, clock=None):
        with patch.object(subject, '_group_alive', return_value=False), \
                patch.object(subject, '_terminate') as terminate:
            result = subject.run(self.plan_path, popen=launch or self.launch,
                                 clock=clock or (lambda: 0), sleep=lambda seconds: None)
        return json.loads(result.read_text()), terminate

    def test_exact_approval_required_before_outputs_or_children(self):
        for approved in (False, 'yes', 1):
            with self.subTest(approved=approved):
                self.write_plan(self.plan(), approved=approved)
                with self.assertRaisesRegex(ValueError, 'Exact approved'):
                    subject.run(self.plan_path, popen=Mock(side_effect=AssertionError('must not spawn')))
                self.assertFalse(self.root.exists())

    def test_runtime_link_is_allowed_only_to_the_bound_base_binary(self):
        base=self.directory/'base-python';base.write_bytes(b'bound base ELF')
        prefix=self.directory/'runtime';(prefix/'bin').mkdir(parents=True)
        executable=prefix/'bin/python';executable.symlink_to(base)
        value=self.plan(runtime_interpreter='base_symlink',runtime_prefix=str(prefix),runtime_python=str(executable),bootstrap={'binary':subject.record(base)})
        self.assertEqual(subject.verify(self.plan_path)['runtime_interpreter'],'base_symlink')
        other=self.directory/'other-python';other.write_bytes(b'foreign')
        executable.unlink();executable.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'exact bound'):subject.verify(self.plan_path)
        executable.unlink();executable.write_bytes(base.read_bytes())
        with self.assertRaisesRegex(ValueError,'exact bound'):subject.verify(self.plan_path)

    def test_plan_hash_mismatch_rejected(self):
        value = self.plan()
        value['runtime_python'] = '/unapproved/python'
        self.plan_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Exact approved'):
            subject.verify(self.plan_path)
        self.assertFalse(self.root.exists())

    def test_bound_input_hash_mismatch_rejected(self):
        artifact = self.directory / 'tiny-input.json'
        artifact.write_text('{}')
        value = self.plan()
        value['frozen_files'].append(subject.record(artifact))
        self.write_plan(value)
        artifact.write_text('[]')  # Same size, so this specifically exercises SHA verification.
        with self.assertRaisesRegex(ValueError, 'Bound artifact missing or changed'):
            subject.verify(self.plan_path)

    def test_controller_and_each_sibling_must_be_hash_bound(self):
        for name in ('mimas_cpu.py', 'mimas_contract.py', 'mimas_smoke.py', 'mimas_pilot.py'):
            with self.subTest(name=name):
                value = self.plan()
                value['frozen_files'] = [e for e in value['frozen_files'] if Path(e['path']).name != name]
                self.write_plan(value)
                with self.assertRaisesRegex(ValueError, 'not bound|not frozen'):
                    subject.verify(self.plan_path)

    def test_phase_stages_population_and_fit_ceilings_cannot_expand(self):
        for field, changed in [('stages', ['pca', 'umap']), ('maximum_projection_fits', 10),
                               ('maximum_hdbscan_extractions', 7), ('inputs', dict(model_count=65))]:
            with self.subTest(field=field):
                value = self.plan('pilot64')
                value[field] = changed
                self.write_plan(value)
                with self.assertRaises(ValueError):
                    subject.verify(self.plan_path)

    def test_probe_retains_blocked_truth_without_native_child_or_environment_creation(self):
        self.plan('probe')
        self.contract.observations = [FakeContract.observation(smoke=False, fit=False)]
        completion, _ = self.run_fake(launch=Mock(side_effect=AssertionError('probe must not spawn')))
        self.assertEqual(completion['status'], 'resource_probe_complete')
        self.assertIs(completion['fit_containment_verified'], False)
        self.assertEqual(completion['native_imports'], 0)
        self.assertEqual(completion['projection_fits'], 0)
        self.assertIs(completion['environment_created'], False)
        self.assertFalse((self.root / 'tmp').exists())
        self.assertFalse((self.root / 'stages').exists())
        self.assertEqual(self.contract.fit_checks, [])

    def test_existing_output_is_preserved_and_not_reused(self):
        self.plan()
        self.root.mkdir()
        sentinel = self.root / 'existing.txt'
        sentinel.write_text('preserve')
        with self.assertRaises(FileExistsError):
            subject.run(self.plan_path, popen=Mock(side_effect=AssertionError('must not spawn')))
        self.assertEqual(sentinel.read_text(), 'preserve')
        self.assertEqual(list(self.root.iterdir()), [sentinel])

    def test_symlink_output_destination_rejected(self):
        other = self.directory / 'other'
        other.mkdir()
        self.root.symlink_to(other, target_is_directory=True)
        self.plan()
        with self.assertRaisesRegex(ValueError, 'non-symlink'):
            subject.verify(self.plan_path)
        self.assertEqual(list(other.iterdir()), [])

    def test_writer_paths_cannot_escape_owned_run(self):
        self.plan()
        escaped = self.directory / 'escaped'
        self.contract.environment = dict(TMPDIR=str(escaped))
        with self.assertRaisesRegex(ValueError, 'escapes this run'):
            subject.run(self.plan_path, popen=Mock(side_effect=AssertionError('must not spawn')))
        self.assertFalse(escaped.exists())

    def test_smoke_fit_blocked_is_successfully_retained_without_promising_fits(self):
        self.plan()
        self.contract.observations = [FakeContract.observation(fit=False)]
        def launch(command, **kwargs):
            process = self.launch(command, **kwargs)
            path = self.root / 'stages' / 'smoke' / 'smoke.json'
            path.write_text(json.dumps(dict(status='smoke_complete_fit_blocked', smoke_checks_passed=True,
                                            fit_ready=False, fit_blocking_reasons=['pids unknown'])))
            return process
        completion, terminate = self.run_fake(launch=launch)
        self.assertEqual(completion['status'], 'smoke_complete_fit_blocked')
        self.assertIs(completion['fit_ready'], False)
        self.assertIs(completion['standard_deliverables_complete'], False)
        self.assertEqual(completion['maximum_projection_fits'], 0)
        self.assertEqual(completion['maximum_hdbscan_extractions'], 0)
        self.assertEqual(self.contract.fit_checks, [])
        self.assertEqual(len(self.commands), 1)
        command, kwargs = self.commands[0]
        self.assertEqual(command[:2], ['/approved/runtime/bin/python', '-I'])
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['cwd'], self.root)
        self.assertEqual(terminate.call_count, 1)

    def test_failed_child_is_not_retried_and_later_stages_do_not_start(self):
        self.plan('pilot64')
        launch = Mock(return_value=FakeProcess(returncode=7))
        with patch.object(subject, '_group_alive', return_value=False), patch.object(subject, '_terminate') as terminate:
            with self.assertRaisesRegex(RuntimeError, 'pca; no retry'):
                subject.run(self.plan_path, popen=launch, clock=lambda: 0, sleep=lambda _: None)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(terminate.call_count, 1)
        failure = json.loads((self.root / 'failure.json').read_text())
        self.assertIs(failure['retry'], False)
        self.assertFalse((self.root / 'completion.json').exists())
        self.assertFalse((self.root / 'context-umap.json').exists())
        self.assertFalse((self.root / 'stages' / 'umap').exists())

    def test_smoke_watch_allows_unknown_swap_or_pids_when_cpu_ram_proof_remains(self):
        self.plan()
        self.contract.observations = [FakeContract.observation(fit=False)]
        def launch(command, **kwargs):
            process = self.launch(command, **kwargs)
            process.polls = [None, 0]
            path = self.root / 'stages' / 'smoke' / 'smoke.json'
            path.write_text(json.dumps(dict(status='smoke_complete_fit_blocked', smoke_checks_passed=True,
                                            fit_ready=False)))
            return process
        completion, _ = self.run_fake(launch=launch)
        self.assertEqual(completion['status'], 'smoke_complete_fit_blocked')
        samples = (self.root / 'resource-observations.jsonl').read_text().splitlines()
        self.assertEqual(len(samples), 1)
        self.assertFalse(json.loads(samples[0])['fit_containment_verified'])
        self.assertEqual(self.contract.fit_checks, [])

    def test_incomplete_child_receipt_cannot_complete_job(self):
        self.plan('umap-full')
        def launch(command, **kwargs):
            process = self.launch(command, **kwargs)
            (self.root / 'stages' / 'umap' / 'receipt.json').write_text(json.dumps(dict(phase_complete=False)))
            return process
        with self.assertRaisesRegex(ValueError, 'receipt is incomplete'):
            self.run_fake(launch=launch)
        self.assertFalse((self.root / 'completion.json').exists())

    def test_success_exit_requires_saved_stage_receipt(self):
        self.plan()
        with self.assertRaisesRegex(ValueError, 'no completion receipt'):
            self.run_fake(launch=Mock(return_value=FakeProcess()))
        self.assertFalse((self.root / 'completion.json').exists())

    def test_wall_watchdog_terminates_fake_child_once(self):
        self.plan()
        process = FakeProcess(returncode=None)
        launch = Mock(return_value=process)
        with patch.object(subject, '_terminate') as terminate:
            with self.assertRaisesRegex(TimeoutError, 'wall ceiling'):
                subject.run(self.plan_path, popen=launch, clock=Mock(side_effect=[0, 0, 31]), sleep=lambda _: None)
        terminate.assert_called_once_with(process)
        self.assertEqual(launch.call_count, 1)
        failure = json.loads((self.root / 'failure.json').read_text())
        self.assertEqual(failure['type'], 'TimeoutError')
        self.assertIs(failure['retry'], False)

    def test_smoke_watchdog_stops_on_lost_cpu_or_ram_enforcement(self):
        self.plan()
        self.contract.observations = [FakeContract.observation(fit=False),
                                      FakeContract.observation(smoke=False, fit=False)]
        process = FakeProcess(returncode=None)
        with patch.object(subject, '_terminate') as terminate:
            with self.assertRaisesRegex(ValueError, 'CPU/RAM enforcement changed'):
                subject.run(self.plan_path, popen=Mock(return_value=process), clock=lambda: 0, sleep=lambda _: None)
        terminate.assert_called_once_with(process)
        self.assertFalse((self.root / 'completion.json').exists())

    def test_fit_watchdog_stops_on_lost_fit_containment(self):
        self.plan('umap-full')
        self.contract.observations = [FakeContract.observation(), FakeContract.observation(fit=False)]
        process = FakeProcess(returncode=None)
        with patch.object(subject, '_terminate') as terminate:
            with self.assertRaisesRegex(ValueError, 'Fit containment'):
                subject.run(self.plan_path, popen=Mock(return_value=process), clock=lambda: 0, sleep=lambda _: None)
        terminate.assert_called_once_with(process)
        self.assertEqual(len(self.contract.fit_checks), 2)
        self.assertFalse((self.root / 'completion.json').exists())

    def test_successful_leader_with_surviving_descendants_is_failure(self):
        self.plan()
        with patch.object(subject, '_group_alive', return_value=True), patch.object(subject, '_terminate') as terminate:
            with self.assertRaisesRegex(RuntimeError, 'surviving descendants'):
                subject.run(self.plan_path, popen=self.launch, clock=lambda: 0, sleep=lambda _: None)
        terminate.assert_called_once_with(self.processes[0])
        self.assertFalse((self.root / 'completion.json').exists())

    def test_group_probe_is_signal_zero_only(self):
        process = FakeProcess()
        with patch.object(subject.os, 'killpg') as kill:
            self.assertTrue(subject._group_alive(process))
        kill.assert_called_once_with(process.pid, 0)
        with patch.object(subject.os, 'killpg', side_effect=ProcessLookupError):
            self.assertFalse(subject._group_alive(process))

    def test_cleanup_kills_surviving_group_even_when_leader_already_exited(self):
        process = FakeProcess(returncode=0)
        with patch.object(subject, '_group_alive', side_effect=[True, True]), \
                patch.object(subject.os, 'killpg') as kill:
            subject._terminate(process)
        self.assertEqual(kill.call_args_list[0].args, (process.pid, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (process.pid, signal.SIGKILL))
        self.assertEqual(process.wait.call_args_list[0].kwargs, dict(timeout=2))
        self.assertEqual(process.wait.call_args_list[1].kwargs, dict(timeout=5))

    def prepare_child(self):
        self.plan()
        (self.root / 'stages').mkdir(parents=True)
        context = self.root / 'context-smoke.json'
        subject.save(context, dict(stage='smoke', plan_sha256=subject.sha(self.plan_path),
                                   controller_pid=os.getppid(), phase_outputs={}))
        subject.save(self.root / 'intent-smoke.json', dict(
            stage='smoke', plan_sha256=subject.sha(self.plan_path), status='charged_before_launch',
            projection_fit_ceiling=subject.STAGE_LIMITS['smoke'][0],
            hdbscan_extraction_ceiling=subject.STAGE_LIMITS['smoke'][1]))
        return context

    def test_actual_smoke_child_allows_fit_blocked_import_receipt(self):
        context = self.prepare_child()
        self.contract.observations = [FakeContract.observation(fit=False)]
        fake_smoke = SimpleNamespace(run_smoke=Mock(return_value=dict(
            status='smoke_complete_fit_blocked', smoke_checks_passed=True, fit_ready=False)))
        with patch.dict(sys.modules, mimas_smoke=fake_smoke):
            subject.child(self.plan_path, 'smoke', context, subject.sha(context))
        fake_smoke.run_smoke.assert_called_once()
        self.assertEqual(self.contract.fit_checks, [])

    def test_child_context_hash_mismatch_blocks_import_work(self):
        context = self.prepare_child()
        fake_smoke = SimpleNamespace(run_smoke=Mock(side_effect=AssertionError('must not import work')))
        with patch.dict(sys.modules, mimas_smoke=fake_smoke):
            with self.assertRaisesRegex(ValueError, 'context changed'):
                subject.child(self.plan_path, 'smoke', context, '0' * 64)
        fake_smoke.run_smoke.assert_not_called()

    def test_child_smoke_output_must_be_fresh(self):
        context = self.prepare_child()
        (self.root / 'stages' / 'smoke').mkdir()
        fake_smoke = SimpleNamespace(run_smoke=Mock(side_effect=AssertionError('must not export work')))
        with patch.dict(sys.modules, mimas_smoke=fake_smoke):
            with self.assertRaisesRegex(ValueError, 'Owned stage output required'):
                subject.child(self.plan_path, 'smoke', context, subject.sha(context))
        fake_smoke.run_smoke.assert_not_called()

    def test_child_rejects_context_from_previous_controller(self):
        context = self.prepare_child()
        value = json.loads(context.read_text())
        value['controller_pid'] = os.getppid() + 1
        context.write_text(json.dumps(value))
        fake_smoke = SimpleNamespace(run_smoke=Mock(side_effect=AssertionError('must not run')))
        with patch.dict(sys.modules, mimas_smoke=fake_smoke):
            with self.assertRaisesRegex(ValueError, 'different live controller'):
                subject.child(self.plan_path, 'smoke', context, subject.sha(context))
        fake_smoke.run_smoke.assert_not_called()

    def test_child_rejects_missing_or_mismatched_charged_stage_intent(self):
        context = self.prepare_child()
        path = self.root / 'intent-smoke.json'
        original = json.loads(path.read_text())
        fake_smoke = SimpleNamespace(run_smoke=Mock(side_effect=AssertionError('must not run')))
        for key, value in [('plan_sha256', '0' * 64), ('stage', 'umap'),
                           ('status', 'completed'), ('projection_fit_ceiling', 1),
                           ('hdbscan_extraction_ceiling', 1)]:
            with self.subTest(field=key):
                path.write_text(json.dumps(dict(original, **{key: value})))
                with patch.dict(sys.modules, mimas_smoke=fake_smoke):
                    with self.assertRaisesRegex(ValueError, 'Charged native stage intent differs'):
                        subject.child(self.plan_path, 'smoke', context, subject.sha(context))
        path.unlink()
        with patch.dict(sys.modules, mimas_smoke=fake_smoke):
            with self.assertRaises(FileNotFoundError):
                subject.child(self.plan_path, 'smoke', context, subject.sha(context))
        fake_smoke.run_smoke.assert_not_called()

    def test_probe_cannot_be_used_to_launch_smoke_or_scientific_child(self):
        self.plan('probe')
        for stage in ('smoke', 'pca', 'umap'):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(ValueError, 'Unapproved child stage'):
                    subject.child(self.plan_path, stage, self.root / 'missing-context.json', '0' * 64)
        self.assertEqual(self.contract.observed, [])
        self.assertFalse(self.root.exists())

    def bootstrap_plan(self):
        binary = self.directory / 'existing-python'
        binary.write_bytes(b'static interpreter identity fixture')
        return self.plan(bootstrap=dict(python=sys.executable, prefix=sys.prefix, binary=subject.record(binary)))

    def test_setup_checks_actual_bootstrap_before_install_without_requiring_fit_ready(self):
        self.bootstrap_plan()
        self.contract.observations = [FakeContract.observation(fit=False)]
        result = subject.setup_check(self.plan_path)
        self.assertTrue(result['smoke_containment_verified'])
        self.assertFalse(result['fit_containment_verified'])
        self.assertEqual(self.contract.observed[0][0]['runtime_python'], sys.executable)
        self.assertEqual(self.contract.observed[0][0]['runtime_prefix'], sys.prefix)
        self.assertEqual(self.contract.observed[0][1], {})  # Actual process, not supplied env.
        self.assertTrue((self.directory / 'setup-resource-observation.json').is_file())
        self.assertFalse(self.root.exists())
        self.assertEqual(self.contract.fit_checks, [])

    def test_setup_rejects_bootstrap_identity_before_resource_probe(self):
        value = self.bootstrap_plan()
        value['bootstrap']['python'] = '/unapproved/python'
        self.write_plan(value)
        with self.assertRaisesRegex(ValueError, 'bootstrap identity differs'):
            subject.setup_check(self.plan_path)
        self.assertEqual(self.contract.observed, [])
        self.assertFalse(self.root.exists())

    def test_setup_rejects_changed_bootstrap_bytes(self):
        value = self.bootstrap_plan()
        Path(value['bootstrap']['binary']['path']).write_bytes(b'changed interpreter')
        with self.assertRaisesRegex(ValueError, 'Bound artifact missing or changed'):
            subject.setup_check(self.plan_path)
        self.assertEqual(self.contract.observed, [])

    def test_setup_retains_blocked_resource_proof_and_does_not_create_environment(self):
        self.bootstrap_plan()
        self.contract.observations = [FakeContract.observation(smoke=False, fit=False)]
        with self.assertRaisesRegex(ValueError, 'Setup blocked'):
            subject.setup_check(self.plan_path)
        receipt = json.loads((self.directory / 'setup-resource-observation.json').read_text())
        self.assertFalse(receipt['smoke_containment_verified'])
        self.assertFalse(self.root.exists())

    def test_setup_is_not_available_to_scientific_phase(self):
        self.plan('pilot64')
        with self.assertRaisesRegex(ValueError, 'only approved with the smoke phase'):
            subject.setup_check(self.plan_path)
        self.assertEqual(self.contract.observed, [])


if __name__ == '__main__':
    unittest.main()
