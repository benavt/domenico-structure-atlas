"""Fake Slurm/proc/controller trees only: no real jobs, engines or inputs."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mimas_contract as contract

GIB = 1024**3


class ContractTests(unittest.TestCase):
    def fixture(self, root, version=2):
        proc = root / 'proc'; process = proc / '123'; process.mkdir(parents=True)
        cg = root / 'controllers'; cg.mkdir()
        plan = dict(run_root=str(root / 'run'), runtime_python='/approved/runtime/bin/python',
                    runtime_prefix='/approved/runtime', resources=dict(maximum_cpus=1,
                    maximum_memory_bytes=4*GIB, maximum_swap_bytes=0, maximum_pids=64))
        env = dict(SLURM_JOB_ID='771', SLURM_NNODES='1', SLURM_NTASKS='1', SLURM_CPUS_PER_TASK='1')
        (process / 'status').write_text('Name:\tpython\nCpus_allowed_list:\t0-7\n')
        mounts = []; memberships = []
        controllers = ('unified',) if version == 2 else ('memory', 'cpu', 'cpuset', 'pids')
        leaves = {}
        for number, controller in enumerate(controllers, 1):
            mount = cg / controller; leaf = mount / 'slurm/job771/step0'; leaf.mkdir(parents=True)
            leaves[controller] = leaf
            if version == 2:
                mounts.append(f'{number} 0 0:{number} / {mount} rw - cgroup2 cgroup rw')
                memberships.append('0::/slurm/job771/step0')
            else:
                mounts.append(f'{number} 0 0:{number} / {mount} rw - cgroup cgroup rw,{controller}')
                memberships.append(f'{number}:{controller}:/slurm/job771/step0')
        (process / 'mountinfo').write_text('\n'.join(mounts)+'\n')
        (process / 'cgroup').write_text('\n'.join(memberships)+'\n')
        if version == 2:
            leaf = leaves['unified']
            values = {'memory.max':str(3*GIB), 'memory.current':'100', 'memory.swap.max':'0',
                      'memory.swap.current':'0', 'pids.max':'64', 'pids.current':'3',
                      'cpu.max':'100000 100000', 'cpuset.cpus.effective':'0-7'}
        else:
            values = None
            for name, fields in {
                'memory':{'memory.limit_in_bytes':str(3*GIB), 'memory.usage_in_bytes':'100',
                          'memory.memsw.limit_in_bytes':str(3*GIB), 'memory.use_hierarchy':'1'},
                'cpu':{'cpu.cfs_quota_us':'100000', 'cpu.cfs_period_us':'100000'},
                'cpuset':{'cpuset.cpus':'0-7'}, 'pids':{'pids.max':'64','pids.current':'3'}
            }.items():
                for key,value in fields.items():(leaves[name]/key).write_text(value)
        if values:
            for key,value in values.items():(leaf/key).write_text(value)
        env=contract.sanitized_environment(plan,environ=env)
        return plan,env,proc,cg,leaves

    def observe(self, args):
        plan,env,proc,cg,_ = args
        with patch.object(contract.sys, 'flags', SimpleNamespace(no_user_site=1)),patch.dict(contract.os.environ,env,clear=True):
            return contract.observe(plan,proc_root=proc,cgroup_root=cg,pid=123,
                runtime_python=plan['runtime_python'],runtime_prefix=plan['runtime_prefix'])

    def test_v2_effective_ancestor_limits_and_fit_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));leaf=args[-1]['unified']
            (leaf/'memory.max').write_text('max');(leaf.parent/'memory.max').write_text(str(2*GIB))
            (leaf/'pids.max').write_text('max');(leaf.parent/'pids.max').write_text('32')
            result=self.observe(args)
            self.assertEqual(result['status'],'smoke_observed');self.assertTrue(result['smoke_containment_verified'])
            self.assertTrue(result['fit_containment_verified']);self.assertIs(contract.require_fit_ready(result),result)
            self.assertEqual(result['memory']['limit_bytes'],2*GIB);self.assertEqual(result['pids']['limit'],32)
            self.assertEqual(result['cpu']['process_affinity'],list(range(8)))
            self.assertEqual(result['scientific_fits'],0);self.assertFalse(result['execution_authorized'])

    def test_unlimited_or_missing_ram_blocks_even_with_slurm_mem_request(self):
        for mode in ('unlimited','missing','too-large'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp));path=args[-1]['unified']/'memory.max'
                if mode=='missing':path.unlink()
                else:path.write_text('max' if mode=='unlimited' else str(8*GIB))
                args[1]['SLURM_MEM_PER_NODE']='4096'
                result=self.observe(args)
                self.assertEqual(result['status'],'blocked');self.assertFalse(result['smoke_containment_verified'])
                with self.assertRaises(ValueError):contract.require_fit_ready(result)

    def test_swap_and_pids_unknown_allow_only_no_fit_smoke(self):
        for names in (('memory.swap.max','memory.swap.current'),('pids.max','pids.current')):
            with self.subTest(names=names),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp));leaf=args[-1]['unified']
                for name in names:(leaf/name).unlink()
                result=self.observe(args)
                self.assertEqual(result['status'],'smoke_observed');self.assertTrue(result['smoke_containment_verified'])
                self.assertFalse(result['fit_containment_verified'])
                with self.assertRaises(ValueError):contract.require_fit_ready(result)

    def test_nonzero_swap_or_oversized_pid_limit_cannot_certify_fits(self):
        for name,value in (('memory.swap.max','100'),('memory.swap.current','1'),('pids.max','128')):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp));(args[-1]['unified']/name).write_text(value)
                result=self.observe(args)
                self.assertFalse(result['fit_containment_verified'])
                with self.assertRaises(ValueError):contract.require_fit_ready(result)

    def test_v1_memsw_is_never_an_independent_zero_swap_cap(self):
        for value in (str(3*GIB),str(2**63-4096)):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp),version=1);leaf=args[-1]['memory']
                (leaf/'memory.memsw.limit_in_bytes').write_text(value)
                result=self.observe(args)
                self.assertEqual(result['memory']['cgroup_version'],1)
                self.assertEqual(result['status'],'smoke_observed')
                self.assertIsNone(result['swap']['limit_bytes']);self.assertIsNone(result['swap']['current_bytes'])
                self.assertFalse(result['fit_containment_verified'])
                with self.assertRaises(ValueError):contract.require_fit_ready(result)

    def test_v1_ancestor_ram_needs_hierarchical_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),version=1);leaf=args[-1]['memory']
            (leaf/'memory.limit_in_bytes').write_text(str(2**63-4096))
            (leaf.parent/'memory.limit_in_bytes').write_text(str(2*GIB))
            (leaf.parent/'memory.use_hierarchy').write_text('0')
            self.assertEqual(self.observe(args)['status'],'blocked')
            (leaf.parent/'memory.use_hierarchy').write_text('1')
            self.assertEqual(self.observe(args)['memory']['limit_bytes'],2*GIB)

    def test_hybrid_mounts_use_the_controller_that_really_enforces_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),version=1);process=args[2]/'123';mount=args[3]/'unified';mount.mkdir()
            with (process/'cgroup').open('a') as f:f.write('0::/\n')
            with (process/'mountinfo').open('a') as f:f.write(f'10 0 0:10 / {mount} rw - cgroup2 cgroup rw\n')
            result=self.observe(args)
            self.assertEqual(result['memory']['cgroup_version'],1);self.assertEqual(result['status'],'smoke_observed')

    def test_namespace_mount_root_mapping_and_visible_ancestor_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));process=args[2]/'123';mount=args[3]/'unified';leaf=mount/'step0';leaf.mkdir()
            (process/'mountinfo').write_text(f'1 0 0:1 /slurm/job771 {mount} rw - cgroup2 cgroup rw\n')
            for key in ('memory.max','memory.current','memory.swap.max','memory.swap.current','pids.max','pids.current','cpu.max'):
                (leaf/key).write_text((args[-1]['unified']/key).read_text())
            result=self.observe(args);self.assertEqual(result['memory']['limit_bytes'],3*GIB)
            self.assertTrue(all(str(mount) in row['path'] for row in result['memory']['records']))
            (process/'cgroup').write_text('0::/different/namespace\n')
            with self.assertRaisesRegex(ValueError,'namespace'):self.observe(args)

    def test_cgroup_path_traversal_mount_escape_and_symlinks_are_rejected(self):
        for mode in ('traversal','mount','symlink'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp));process=args[2]/'123'
                if mode=='traversal':(process/'cgroup').write_text('0::/../../escape\n')
                elif mode=='mount':(process/'mountinfo').write_text('1 0 0:1 / /tmp rw - cgroup2 cgroup rw\n')
                else:
                    path=args[-1]['unified']/'memory.max';path.unlink();path.symlink_to('/etc/passwd')
                with self.assertRaises(ValueError):self.observe(args)

    def test_cpu_request_or_process_affinity_alone_is_not_cgroup_enforcement(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));leaf=args[-1]['unified'];(leaf/'cpu.max').write_text('max 100000')
            (leaf/'cpuset.cpus.effective').unlink();(args[2]/'123/status').write_text('Cpus_allowed_list:\t0\n')
            self.assertEqual(self.observe(args)['status'],'blocked')
            (leaf/'cpuset.cpus.effective').write_text('0')
            self.assertTrue(self.observe(args)['cpu']['verified'])

    def test_scheduler_runtime_and_fit_packet_tampering_fail_closed(self):
        for key,value in (('SLURM_JOB_ID',''),('SLURM_NNODES','2'),('SLURM_NTASKS','2'),('SLURM_CPUS_PER_TASK','2')):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp));args[1][key]=value
                self.assertEqual(self.observe(args)['status'],'blocked')
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));result=self.observe(args)
            for section,key,value in (('slurm','tasks',2),('memory','limit_bytes',10*GIB),
                                      ('runtime','prefix','/another'),('pids','current',65)):
                changed=deepcopy(result);changed[section][key]=value
                with self.subTest(section=section),self.assertRaises(ValueError):contract.require_fit_ready(changed)
            with patch.object(contract.sys,'flags',SimpleNamespace(no_user_site=1)):
                wrong=contract.observe(args[0],environ=args[1],proc_root=args[2],cgroup_root=args[3],pid=123,
                    runtime_python='/other/python',runtime_prefix='/other')
            self.assertEqual(wrong['status'],'blocked')

    def test_planned_environment_cannot_impersonate_actual_child_fit_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));plan,env,proc,cg,_=args
            with patch.object(contract.sys,'flags',SimpleNamespace(no_user_site=1)):
                planned=contract.observe(plan,environ=env,proc_root=proc,cgroup_root=cg,pid=123,
                    runtime_python=plan['runtime_python'],runtime_prefix=plan['runtime_prefix'])
            self.assertEqual(planned['environment']['source'],'supplied_mapping')
            self.assertTrue(planned['fit_containment_verified'])
            with self.assertRaisesRegex(ValueError,'actual child environment'):contract.require_fit_ready(planned)
            contract.require_fit_ready(planned,require_actual_environment=False)
            changed=list(args);changed[1]=dict(env,DATA_PROCESSED='/foreign/output')
            result=self.observe(changed)
            self.assertEqual(result['status'],'smoke_observed')
            self.assertFalse(result['fit_containment_verified'])
            self.assertIn('DATA_PROCESSED',result['environment']['mismatched_variables'])

    def test_environment_drops_hooks_and_overrides_every_native_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));plan=args[0]
            original=dict(args[1],HOME='/real/home',PATH='/folding/bin',PYTHONPATH='/pax/source',
                PYTHONHOME='/wrong',PYTHONSTARTUP='/hook.py',LD_PRELOAD='/hook.so',
                ALPHAFOLD_NATIVE_HOOK='unsafe',DATA_PROCESSED='/pax/old',OPENBLAS_NUM_THREADS='96')
            result=contract.sanitized_environment(plan,environ=original)
            for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','LD_PRELOAD','ALPHAFOLD_NATIVE_HOOK'):
                self.assertNotIn(key,result)
            self.assertEqual(result['HOME'],'/real/home');self.assertEqual(result['PYTHONNOUSERSITE'],'1')
            self.assertEqual(result['SLURM_JOB_ID'],'771')
            self.assertEqual(result['PATH'],'/approved/runtime/bin:/usr/bin:/bin')
            for key in contract.THREAD_VARIABLES:self.assertEqual(result[key],'1')
            for key in contract.WRITER_PATHS:self.assertTrue(Path(result[key]).is_relative_to(Path(plan['run_root'])))
            self.assertEqual(original['DATA_PROCESSED'],'/pax/old')
            plan['run_root']=str(Path(tmp)/'../escape')
            with self.assertRaises(ValueError):contract.sanitized_environment(plan,environ=original)


if __name__=='__main__':
    unittest.main()
