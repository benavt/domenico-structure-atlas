"""Fresh-process/profile and trace contracts; fake engines only, zero fits."""
import hashlib
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reducer_diagnostic as diag
import mimas_contract as contract
import mimas_cpu as cpu
import mimas_package as package


class ProfileTests(unittest.TestCase):
    def plan(self,root,profile='native'):
        return dict(phase='repro64',diagnostic_profile=profile,active_stage='umap1',
            run_root=str(root/'run'),runtime_python='/runtime/bin/python',runtime_prefix='/runtime',
            resources=dict(maximum_cpus=2,maximum_memory_bytes=8*1024**3,maximum_swap_bytes=0,maximum_pids=64))

    def test_generic_target_survives_sanitizer_but_inherited_overrides_do_not(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=self.plan(Path(tmp),'generic-jit-v1')
            env=contract.sanitized_environment(p,environ=dict(NUMBA_CPU_NAME='bad',NUMBA_CPU_FEATURES='+avx512',PYTHONPATH='/bad'))
            self.assertEqual(env['NUMBA_CPU_NAME'],'generic');self.assertEqual(env['NUMBA_CPU_FEATURES'],'')
            self.assertNotIn('PYTHONPATH',env)
            native=contract.sanitized_environment(dict(p,diagnostic_profile='native'),environ=env)
            self.assertNotIn('NUMBA_CPU_NAME',native);self.assertNotIn('NUMBA_CPU_FEATURES',native)
            self.assertTrue(all(env[k]=='1'for k in contract.THREAD_VARIABLES))

    def test_repeat_caches_are_distinct_and_invalid_stage_or_profile_fails(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=self.plan(Path(tmp));a=contract.sanitized_environment(p,environ={})
            b=contract.sanitized_environment(dict(p,active_stage='umap2'),environ={})
            self.assertNotEqual(a['DATA_PROCESSED'],b['DATA_PROCESSED'])
            self.assertTrue(Path(a['DATA_PROCESSED']).is_relative_to(Path(p['run_root'])))
            for updates in [dict(active_stage='pca'),dict(diagnostic_profile='unbound')]:
                with self.assertRaises(ValueError):contract.sanitized_environment(dict(p,**updates),environ={})

    def test_pre_import_rejects_late_profile_or_writer_changes(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=self.plan(Path(tmp),'generic-jit-v1');p['portability_helper']={'path':'/bound/helper.py'};p['runtime_payload']={'path':'/payload'}
            state=Path(tmp)/'state.json';state.write_text(json.dumps(dict(payload=p['runtime_payload'],files=[])));p['runtime_payload_state']={'path':str(state)}
            tool=SimpleNamespace(assert_before_native_imports=Mock())
            fakecpu=SimpleNamespace(record=Mock(return_value={'verified':True}),bound=Mock(return_value=state))
            with patch.object(diag,'modules',return_value=(fakecpu,contract,None)),patch.object(diag,'helper',return_value=tool), patch.object(sys,'dont_write_bytecode',True), patch.object(sys,'pycache_prefix',str(Path(p['run_root'])/'cache/python')):
                env=contract.sanitized_environment(p,environ={})
                with patch.dict(os.environ,env,clear=True):self.assertEqual(diag.pre_import(p),{'verified':True})
                env.pop('NUMBA_CPU_NAME')
                with patch.dict(os.environ,env,clear=True),self.assertRaisesRegex(ValueError,'CPU target'):diag.pre_import(p)
                tool.assert_before_native_imports.side_effect=ValueError('late import')
                with self.assertRaisesRegex(ValueError,'late import'):diag.pre_import(p)

    def test_effective_target_and_native_pool_limits_are_checked_after_imports(self):
        fake=SimpleNamespace(config=SimpleNamespace(CPU_NAME='generic',CPU_FEATURES=''),get_num_threads=lambda:1)
        llvm=SimpleNamespace(get_host_cpu_name=lambda:'physical-cpu',get_host_cpu_features=lambda:SimpleNamespace(flatten=lambda:'+known'))
        llvmlite=SimpleNamespace(binding=llvm)
        pools=SimpleNamespace(threadpool_info=lambda:[{'num_threads':1}])
        with patch.dict(sys.modules,{'numba':fake,'llvmlite':llvmlite,'llvmlite.binding':llvm,'threadpoolctl':pools}):
            self.assertEqual(diag.effective_runtime({'diagnostic_profile':'generic-jit-v1'})['numba_cpu_name'],'generic')
            fake.config.CPU_NAME=None
            with self.assertRaisesRegex(ValueError,'Numba'):diag.effective_runtime({'diagnostic_profile':'generic-jit-v1'})
            fake.config.CPU_NAME='generic';pools.threadpool_info=lambda:[{'num_threads':2}]
            with self.assertRaisesRegex(ValueError,'pool'):diag.effective_runtime({'diagnostic_profile':'generic-jit-v1'})

    def test_exact_fit_budgets_and_no_environment_install_in_existing_runtime(self):
        self.assertEqual(cpu.FIT_LIMITS['repro64'],(6,0))
        self.assertEqual(sum(cpu.STAGE_LIMITS[s][0]for s in cpu.STAGES['repro64']),6)
        p=dict(phase='repro64',run_root='/run/new',runtime_prefix='/runtime',runtime_python='/runtime/bin/python',
            runner='/source/mimas_cpu.py',wheelhouse='/wheels',requirements='/requirements.txt',partition='debug',
            resources=dict(maximum_cpus=2,maximum_memory_bytes=8*1024**3,maximum_wall_seconds=3600))
        body=package.smoke_batch('/plan.json',p,'/base/bin/python',environment_exists=True)
        self.assertIn('mimas_cpu.py run /plan.json',body);self.assertNotIn('pip install',body)
        self.assertNotIn('-m venv',body)

    def test_isolated_children_use_explicit_bytecode_flags_not_ignored_env(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=self.plan(Path(tmp))
            self.assertEqual(contract.python_flags(p),['-I','-B','-X','pycache_prefix='+str(Path(p['run_root'])/'cache/python')])
            self.assertEqual(contract.python_flags(dict(p,phase='pilot64')),['-I'])

    def test_runtime_payload_hashed_once_and_later_identity_change_is_detected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);prefix=root/'runtime';site=prefix/'lib/python3.11/site-packages';site.mkdir(parents=True)
            source=site/'toy.py';source.write_text('# frozen\n')
            payload=root/'payload.json';payload.write_text(json.dumps(dict(runtime_prefix=str(prefix),files=[cpu.record(source)])))
            p=self.plan(root);p.update(runtime_prefix=str(prefix),runtime_python=str(prefix/'bin/python'),runtime_payload=cpu.record(payload),portability_helper={'path':str(source)})
            run=Path(p['run_root']);run.mkdir();p['runtime_payload_state']=diag.verify_runtime_payload(p,run)
            tool=SimpleNamespace(assert_before_native_imports=lambda:None)
            with patch.object(diag,'helper',return_value=tool),patch.object(sys,'dont_write_bytecode',True),patch.object(sys,'pycache_prefix',str(run/'cache/python')):
                with patch.dict(os.environ,contract.sanitized_environment(p,environ={}),clear=True):diag.pre_import(p)
                source.write_text('# changed\n')
                with patch.dict(os.environ,contract.sanitized_environment(p,environ={}),clear=True),self.assertRaisesRegex(ValueError,'changed since'):diag.pre_import(p)


class ReceiptTests(unittest.TestCase):
    def fixture(self,root):
        trace=root/'trace.json';runtime=root/'runtime.json';native=root/'native.json'
        for p in (runtime,native):p.write_text('{}')
        sources={'umap':{'path':'/bound/umap.py'},'sklearn':{'path':'/bound/tsne.py'}}
        trace.write_text(json.dumps(dict(status='complete',counts=dict(umap=1,tsne=0),sources=sources,
            maximum_array_bytes=16*1024**2,maximum_trace_bytes=64*1024**2)))
        p=dict(diagnostic_profile='native',trace_sources=sources,portability_helper={'path':'/bound/helper'})
        r=dict(diagnostic_profile='native',native_stage='umap',portability_helper=p['portability_helper'],
            diagnostic_adapter=cpu.record(diag.__file__),scientific_fits=dict(PCA=0,UMAP=1,TSNE=0,HDBSCAN=0),
            trace=cpu.record(trace),runtime_effective=cpu.record(runtime),runtime_effective_after=cpu.record(runtime),native_receipt=cpu.record(native))
        r['outputs']=[r[k]for k in ('trace','runtime_effective','runtime_effective_after','native_receipt')]
        return p,r,trace

    def test_profile_trace_completeness_cache_bypass_and_fit_count_fail_closed(self):
        with tempfile.TemporaryDirectory()as tmp:
            p,r,trace=self.fixture(Path(tmp));diag.validate_receipt(p,'umap1',r)
            for key,value in [('diagnostic_profile','generic-jit-v1'),('scientific_fits',dict(PCA=1,UMAP=1,TSNE=0,HDBSCAN=0))]:
                with self.subTest(key=key),self.assertRaises(ValueError):diag.validate_receipt(p,'umap1',dict(r,**{key:value}))
            old=json.loads(trace.read_text());old['counts']['umap']=0;trace.write_text(json.dumps(old));r['trace']=cpu.record(trace)
            with self.assertRaisesRegex(ValueError,'trace'):diag.validate_receipt(p,'umap1',r)

    def test_adapter_runs_unchanged_native_method_and_closes_trace_without_extra_fits(self):
        for stage in ('umap1','tsne2'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);source=root/'module.py';source.write_text('# fake module\n')
                entry=cpu.record(source);profile='native';method=stage[:-1]
                plan=dict(active_stage=stage,diagnostic_profile=profile,trace_sources={'umap':entry,'sklearn':entry},
                    portability_helper={'path':'/bound/helper'})
                output=root/'stage';output.mkdir()
                class FakeTrace:
                    def __init__(self,path):self.path=path
                    @contextmanager
                    def install(self,*args,**kwargs):
                        self.path.mkdir()
                        yield self
                        cpu.save(self.path/'trace.json',dict(status='complete',counts={'umap':int(method=='umap'),'tsne':int(method=='tsne')},
                            events=[],sources=plan['trace_sources'],maximum_array_bytes=16*1024**2,maximum_trace_bytes=64*1024**2))
                def native_run(native_plan,native_method,destination):
                    self.assertIs(native_plan,plan);self.assertEqual(native_method,method)
                    destination.mkdir();artifact=destination/'projection.npz';artifact.write_bytes(b'fake saved output')
                    result=dict(scientific_fits={'PCA':int(method=='tsne'),'UMAP':int(method=='umap'),'TSNE':int(method=='tsne'),'HDBSCAN':0},outputs=[cpu.record(artifact)])
                    cpu.save(destination/'receipt.json',result);return result
                fakepilot=SimpleNamespace(native_adapters=Mock(),run_stage=Mock(side_effect=native_run))
                umod=SimpleNamespace(__file__=str(source));tmod=SimpleNamespace(__file__=str(source))
                upkg=SimpleNamespace(umap_=umod,__version__='0.5.11')
                manifold=SimpleNamespace(_t_sne=tmod);spkg=SimpleNamespace(manifold=manifold,__version__='1.7.1')
                with patch.object(diag,'modules',return_value=(cpu,contract,fakepilot)),patch.object(diag,'helper',return_value=SimpleNamespace(ReducerTrace=FakeTrace)), \
                     patch.object(diag,'effective_runtime',return_value={'profile':profile}),patch.object(diag,'runtime_identity_unchanged'), \
                     patch.dict(sys.modules,{'umap':upkg,'umap.umap_':umod,'sklearn':spkg,'sklearn.manifold':manifold,'sklearn.manifold._t_sne':tmod}):
                    result=diag.run_stage(plan,stage,output)
                self.assertEqual(result['stage'],stage);fakepilot.run_stage.assert_called_once()
                self.assertEqual(result['diagnostic_profile'],profile)
                self.assertTrue((output/'receipt.json').is_file())

    def test_native_needs_no_predecessor_but_generic_requires_closed_matching_plan(self):
        diag.verify_predecessor({'diagnostic_profile':'native'})
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'native-plan.json';path.write_text(json.dumps(dict(phase='repro64',diagnostic_profile='native',
                inputs={'cohort':'other'},native={},sources={},execution_host='pax')))
            entry=cpu.record(path);p=dict(diagnostic_profile='generic-jit-v1',native_predecessor=entry,frozen_files=[entry],
                inputs={'cohort':'expected'},native={},sources={},execution_host='pax')
            # Pair-gate absence must block before inspecting any predecessor.
            p['native_pair_gate']=str(Path(tmp)/'missing-gate.json')
            with self.assertRaises(FileNotFoundError):diag.verify_predecessor(p)
            gate=Path(tmp)/'gate.json';gate.write_text('{}');p['native_pair_gate']=str(gate)
            with self.assertRaisesRegex(ValueError,'closed native'):diag.verify_predecessor(p)
            with patch.object(diag,'verify_native_pair'),self.assertRaisesRegex(ValueError,'scope'):diag.verify_predecessor(p)
            p['inputs']={'cohort':'other'}
            with patch.object(diag,'verify_native_pair'),self.assertRaises(KeyError):diag.verify_predecessor(p) # no fabricated completion/run root


if __name__=='__main__':unittest.main()
