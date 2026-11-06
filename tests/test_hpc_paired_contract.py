"""Paired64 containment/routing over fake proc/cgroups; no jobs or science."""
from copy import deepcopy
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import hpc_contract as contract
import hpc_package as package
import hpc_scope as scope

GIB = 1024**3


class PairedContractTests(unittest.TestCase):
    def fixture(self, root, *, host='workstation'):
        uid = os.getuid(); unit = 'domenico-paired64-workstation.service'
        membership = ('/user.slice/user-{0}.slice/user@{0}.service/app.slice/{1}'.format(uid, unit)
                      if host=='workstation' else '/slurm/job771/step0')
        proc = root/'proc'; process = proc/'123'; process.mkdir(parents=True)
        controller = root/'controllers'; mount = controller/'unified'; leaf = mount/membership.lstrip('/')
        leaf.mkdir(parents=True)
        (process/'cgroup').write_text('0::'+membership+'\n')
        (process/'mountinfo').write_text('1 0 0:1 / '+str(mount)+' rw - cgroup2 cgroup rw\n')
        (process/'status').write_text('Name:\tpython\nCpus_allowed_list:\t0-3\n')
        for name, value in {'memory.max':str(8*GIB),'memory.current':'100',
            'memory.swap.max':'0','memory.swap.current':'0','pids.max':'64','pids.current':'3',
            'cpu.max':'100000 100000','cpuset.cpus.effective':'0-3'}.items():
            (leaf/name).write_text(value)
        plan = dict(phase='pilot64',execution_host=host,run_root=str(root/'run'),
            runtime_python='/approved/runtime/bin/python',runtime_prefix='/approved/runtime',
            resources=dict(maximum_cpus=1,maximum_memory_bytes=8*GIB,maximum_swap_bytes=0,
                maximum_pids=64,maximum_wall_seconds=3600))
        if host=='workstation':plan.update(unit_name=unit); plan['resources']['required_uid'] = uid
        original = {} if host=='workstation' else dict(SLURM_JOB_ID='771',SLURM_NNODES='1',SLURM_NTASKS='1',SLURM_CPUS_PER_TASK='1')
        env = contract.sanitized_environment(plan,environ=original)
        return plan, env, proc, controller, leaf

    def observed(self, values, *, isolated=True):
        plan, env, proc, controller, _ = values
        with patch.object(contract.sys,'flags',SimpleNamespace(no_user_site=int(isolated))), patch.dict(contract.os.environ,env,clear=True):
            return contract.observe(plan,proc_root=proc,cgroup_root=controller,pid=123,
                runtime_python=plan['runtime_python'],runtime_prefix=plan['runtime_prefix'])

    def test_workstation_needs_exact_owned_service_and_effective_caps_without_slurm(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = self.fixture(Path(tmp)); result = self.observed(values)
            self.assertEqual(result['execution_host'],'workstation'); self.assertFalse(result['slurm']['verified'])
            self.assertEqual(result['scheduler']['kind'],'systemd-user'); self.assertTrue(result['scheduler']['verified'])
            self.assertTrue(result['fit_containment_verified']); self.assertIs(contract.require_fit_ready(result),result)
            self.assertEqual(result['memory']['limit_bytes'],8*GIB)
            self.assertEqual(result['swap']['limit_bytes'],0); self.assertEqual(result['pids']['limit'],64)
            self.assertEqual(result['cpu']['effective_quota_cpus'],1.)
            self.assertEqual(result['scientific_fits'],0); self.assertFalse(result['execution_authorized'])

    def test_mismatched_workstation_unit_or_owner_cannot_fall_back_to_valid_slurm(self):
        for change in ('unit','membership','owner'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                values = self.fixture(Path(tmp)); plan, env, proc, _, _ = values
                env.update(SLURM_JOB_ID='771',SLURM_NNODES='1',SLURM_NTASKS='1',SLURM_CPUS_PER_TASK='1')
                if change=='unit':plan['unit_name'] = 'different.service'
                elif change=='membership':(proc/'123/cgroup').write_text('0::/foreign.service')
                else:plan['resources']['required_uid'] += 1
                result = self.observed(values)
                self.assertEqual(result['status'],'blocked'); self.assertTrue(result['slurm']['verified'])
                self.assertFalse(result['scheduler']['verified'])
                self.assertIn('Matching owned Workstation systemd service is unverified',result['blocking_reasons'])
                with self.assertRaises(ValueError):contract.require_fit_ready(result)

    def test_kernel_peak_and_io_are_from_owned_leaf_and_remain_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = self.fixture(Path(tmp)); leaf = values[-1]
            first = self.observed(values)
            self.assertIsNone(first['memory']['peak_bytes'])
            self.assertTrue(first['fit_containment_verified'])
            (leaf/'memory.peak').write_text('123456')
            (leaf/'io.stat').write_text('8:0 rbytes=1024 wbytes=2048')
            (leaf.parent/'memory.peak').write_text('999999')
            result = self.observed(values)
            self.assertEqual(result['memory']['current_bytes'],100)
            self.assertEqual(result['memory']['peak_bytes'],123456)
            self.assertEqual(result['kernel_counters']['io.stat'],'8:0 rbytes=1024 wbytes=2048')

    def test_workstation_service_name_must_be_explicit_and_safe(self):
        for name in ('','../wrong.service','other.scope','x.service\nother.service'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                values = self.fixture(Path(tmp)); values[0]['unit_name'] = name
                with self.assertRaisesRegex(ValueError,'Exact local service'):self.observed(values)

    def test_workstation_missing_or_expanded_cpu_ram_swap_pid_evidence_blocks_fits(self):
        for name, value in [('cpu.max','max 100000'),('memory.max','max'),
            ('memory.max',str(9*GIB)),('memory.swap.max','1'),('memory.swap.current','1'),
            ('pids.max','max'),('pids.max','65'),('pids.current','65')]:
            with self.subTest(name=name,value=value),tempfile.TemporaryDirectory() as tmp:
                values = self.fixture(Path(tmp)); (values[-1]/name).write_text(value)
                result = self.observed(values); self.assertFalse(result['fit_containment_verified'])
                with self.assertRaises(ValueError):contract.require_fit_ready(result)

    def test_workstation_fit_packet_rechecks_caps_service_scope_and_isolated_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = self.fixture(Path(tmp)); result = self.observed(values)
            for section, key, value in [('memory','limit_bytes',9*GIB),('pids','limit',65),
                ('swap','current_bytes',1),('scheduler','control_group','/foreign')]:
                changed = deepcopy(result); changed[section][key] = value
                with self.subTest(section=section),self.assertRaises(ValueError):contract.require_fit_ready(changed)
            changed = deepcopy(result); changed['cpu']['effective_quota_cpus'] = 2
            with self.assertRaises(ValueError):contract.require_fit_ready(changed)
            self.assertFalse(self.observed(values,isolated=False)['fit_containment_verified'])

    def test_hpc_still_requires_single_node_task_slurm_even_with_valid_kernel_caps(self):
        for key, value in [('SLURM_JOB_ID',''),('SLURM_NNODES','2'),('SLURM_NTASKS','2'),('SLURM_CPUS_PER_TASK','2')]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                values = self.fixture(Path(tmp),host='hpc'); values[1][key] = value
                result = self.observed(values)
                self.assertEqual(result['execution_host'],'hpc'); self.assertIsNone(result['scheduler'])
                self.assertEqual(result['status'],'blocked'); self.assertFalse(result['slurm']['verified'])
                with self.assertRaises(ValueError):contract.require_fit_ready(result)

    def test_hpc_unchanged_caps_and_actual_environment_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = self.fixture(Path(tmp),host='hpc'); result = self.observed(values)
            self.assertTrue(result['fit_containment_verified']); contract.require_fit_ready(result)
            for name in ('memory.max','memory.swap.max','pids.max'):
                path = values[-1]/name; saved = path.read_text(); path.unlink()
                self.assertFalse(self.observed(values)['fit_containment_verified']); path.write_text(saved)
            values[1]['DATA_PROCESSED'] = '/foreign/results'
            self.assertFalse(self.observed(values)['fit_containment_verified'])


class PairedRoutingTests(unittest.TestCase):
    def plan(self):
        return dict(phase='pilot64',run_root='/run/new',runtime_prefix='/runtime/new',
            runtime_python='/runtime/new/bin/python',wheelhouse='/payload/wheels',
            requirements='/payload/requirements.txt',runner='/source/hpc_cpu.py',partition='debug',
            resources=dict(maximum_cpus=1,maximum_memory_bytes=8*GIB,maximum_wall_seconds=3600,
                maximum_pids=64,maximum_swap_bytes=0))

    def test_pilot64_inner_batch_executes_run_and_smoke_still_executes_smoke(self):
        plan = self.plan()
        for phase, action in [('pilot64','run'),('smoke','smoke')]:
            with self.subTest(phase=phase):
                plan['phase'] = phase
                body = package.smoke_batch('/payload/plan.json',plan,'/base/bin/python',environment_exists=True)
                self.assertEqual(shlex.split(body.splitlines()[-1]),
                    ['exec','/runtime/new/bin/python','-I','/source/hpc_cpu.py',action,'/payload/plan.json'])
                self.assertIn('#SBATCH --no-requeue',body); self.assertNotIn('pip install',body)
                self.assertLess(body.index('hpc_cpu.py verify'),body.index('exec '))
        plan['phase'] = 'full-clustering'
        with self.assertRaises(ValueError):package.smoke_batch('/payload/plan.json',plan,'/base/bin/python')

    def test_scoped_pilot_batch_enters_scope_before_setup_and_inner_routes_to_run(self):
        plan = self.plan(); plan['containment'] = dict(mode='user_scope')
        body = package.smoke_batch('/payload/plan.json',plan,'/base/bin/python')
        self.assertEqual(shlex.split(body.splitlines()[-1]),
            ['exec','/base/bin/python','-I','/source/hpc_scope.py','launch','/payload/plan.json'])
        self.assertNotIn('-m venv',body); self.assertNotIn('pip install',body)
        inner = dict(plan); inner.pop('containment')
        self.assertIn('hpc_cpu.py run /payload/plan.json',package.smoke_batch('/payload/plan.json',inner,'/base/bin/python'))

    def test_scope_supports_pilot64_with_same_explicit_caps_and_zero_swap(self):
        plan = self.plan(); plan['bootstrap'] = dict(python='/base/bin/python')
        plan['containment'] = dict(mode='user_scope',scope_memory_bytes=8*GIB,scope_tasks=64,scope_max_seconds=3600)
        self.assertEqual(scope._scope_settings(plan),plan['containment'])
        command = scope._scope_command(Path('/payload/plan.json'),plan,{'unit':'exact.scope'},Path('/payload/binding.json'),'a'*64)
        for property in ('MemoryMax='+str(8*GIB),'MemorySwapMax=0','TasksMax=64','RuntimeMaxSec=3600','KillMode=control-group'):
            self.assertIn('--property='+property,command)
        self.assertIn('--scope',command); self.assertIn('--no-ask-password',command)
        for changes in [dict(phase='umap-full'),dict(resources=dict(plan['resources'],maximum_swap_bytes=1)),
            dict(containment=dict(plan['containment'],scope_tasks=65)),
            dict(containment=dict(plan['containment'],scope_memory_bytes=9*GIB)),
            dict(containment=dict(plan['containment'],scope_max_seconds=3601))]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):scope._scope_settings(dict(plan,**changes))


if __name__=='__main__':unittest.main()
