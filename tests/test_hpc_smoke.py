"""Smoke contracts use fake imports/CLI/modules only; no native engines load."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import hpc_smoke as subject


def record(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root/'runtime'; self.site = self.prefix/'site-packages'; self.site.mkdir(parents=True)
        self.run = self.root/'run'; self.run.mkdir()
        self.cwd = self.root/'outside'; self.cwd.mkdir()
        self.checkout = self.root/'checkout'; self.checkout.mkdir()
        self.output = self.run/'smoke'
        self.calls = dict(imports=[], find_specs=[], commands=[], provider=0, backend=[], numeric=[])
        self.pin = 'a'*40
        self.payload = {name:('static frozen '+name).encode() for name in subject.REQUIRED_ASSETS}
        self.payload.update({name:b'# Installed package fixture' for name in ['src/__init__.py', 'config/__init__.py',
            'scripts/__init__.py', 'structure_clustering/__init__.py', 'src/app/workers/tsne_worker.py']})
        allowed = [name.removeprefix('scripts/') for name in self.payload if name.startswith('scripts/') and name!='scripts/__init__.py']
        self.payload['structure_clustering/script_assets.json'] = json.dumps(allowed).encode()
        self.wheel = self.root/'structure_clustering-0.1.0-py3-none-any.whl'
        self.install_payload()
        self.plan = dict(runtime_python=str(self.prefix/'bin/python'), runtime_prefix=str(self.prefix), run_root=str(self.run),
            resources=dict(maximum_cpus=1, maximum_memory_bytes=2*1024**3, maximum_swap_bytes=0, maximum_pids=64),
            sources=dict(library=dict(commit=self.pin)), smoke=dict(outside_checkout_roots=[str(self.checkout)],
                library=dict(distribution='structure-clustering', version='0.1.0', commit=self.pin,
                    wheel=record(self.wheel), filehashes={name:hashlib.sha256(raw).hexdigest() for name,raw in self.payload.items()}),
                packaged_files=list(subject.REQUIRED_ASSETS), cli=dict(analysis_runtime_check=True, timeout_seconds=17),
                numeric_roundtrip=False))
        self.modules = {}
        for name in subject.IMPORTS:
            if name in ('structure_clustering', 'src', 'config', 'scripts'):
                relative = name+'/__init__.py'
            elif name=='src.app.workers.tsne_worker':
                relative = 'src/app/workers/tsne_worker.py'
            elif name=='scripts.clustering_common':
                relative = 'scripts/clustering_common.py'
            else:
                relative = name.replace('.', '/')+'/__init__.py'
            module = SimpleNamespace(__file__=str(self.site/relative), __version__='fixture')
            self.modules[name] = module
        self.modules['structure_clustering'].__version__ = '0.1.0'
        self.modules['matplotlib'].use = lambda backend, **kwargs:self.calls['backend'].append((backend, kwargs))
        self.modules['matplotlib'].get_backend = lambda:'agg'
        def no_instantiation(*args, **kwargs):
            raise AssertionError('Smoke must never instantiate a worker/estimator or invoke a fit')
        self.modules['sklearn.cluster'].HDBSCAN = no_instantiation
        self.modules['matplotlib.backends.backend_agg'].FigureCanvasAgg = no_instantiation
        self.modules['src.app.workers.tsne_worker'].TSNEWorker = no_instantiation
        environment = {key:str(self.run/part) for key,part in [('TMPDIR','tmp'), ('MPLCONFIGDIR','mpl'),
            ('NUMBA_CACHE_DIR','numba-cache'), ('XDG_CACHE_HOME','cache')]}
        environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
            DO_NOT_LOG_SECRET='private fixture value')
        self.observation = dict(status='smoke_observed', smoke_containment_verified=True, fit_containment_verified=True)
        self.adapters = dict(import_module=self.import_module, find_spec=self.find_spec,
            distribution=lambda name:SimpleNamespace(version='0.1.0', locate_file=lambda relative:self.site/relative),
            run=self.command, provider=self.provider, observe=lambda plan:deepcopy(self.observation),
            executable=self.plan['runtime_python'], prefix=str(self.prefix), cwd=str(self.cwd), environment=environment)

    def install_payload(self):
        with zipfile.ZipFile(self.wheel, 'w') as wheel:
            for name,raw in self.payload.items():
                wheel.writestr(name,raw)
                destination=self.site/name; destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(raw)
            wheel.writestr('structure_clustering-0.1.0.dist-info/METADATA', 'Name: structure-clustering\nVersion: 0.1.0\n')

    def import_module(self, name):
        self.calls['imports'].append(name); return self.modules[name]

    def find_spec(self, name):
        self.calls['find_specs'].append(name)
        module=self.modules.get(name)
        return None if module is None else SimpleNamespace(origin=module.__file__)

    def command(self, command, **kwargs):
        self.calls['commands'].append((command, kwargs))
        if command[0]=='git':
            stdout='b'*40+'\n'
        elif command[-1]=='--version':
            stdout='0.1.0\n'
        else:
            stdout=json.dumps(dict(runtime_ready=True, profile='analysis_only'))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr='')

    def provider(self, source):
        self.calls['provider']+=1
        module=SimpleNamespace(fit_selection=lambda:None, fit_diagnostics=lambda:None, structural_labels=lambda:None)
        return module, dict(hierarchical_colors_sha256='c'*64)

    def run_smoke(self, **kwargs):
        return subject.run_smoke(self.plan, self.output, adapters=self.adapters, **kwargs)

    def test_import_has_no_native_dependencies(self):
        path=Path(subject.__file__)
        command=[sys.executable, '-I', '-S', '-c',
            'import importlib.util,sys; s=importlib.util.spec_from_file_location("smoke_only",sys.argv[1]); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'assert not any(v in sys.modules for v in ("numpy","scipy","sklearn","numba","pandas","matplotlib"))', str(path)]
        result=subprocess.run(command, cwd=self.cwd, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_verified_wheel_cpu_imports_and_finite_cli_do_not_fit(self):
        result=self.run_smoke()
        self.assertEqual(result['status'],'smoke_complete'); self.assertTrue(result['fit_ready'])
        self.assertEqual(self.calls['imports'],list(subject.IMPORTS))
        self.assertEqual(result['installed_library']['commit'],self.pin)
        self.assertEqual(self.calls['backend'],[('Agg',dict(force=True))])
        self.assertEqual(len(self.calls['commands']),2)
        for command,kwargs in self.calls['commands']:
            self.assertEqual(command[:4],[self.plan['runtime_python'],'-I','-m','structure_clustering'])
            self.assertEqual(kwargs['timeout'],17); self.assertEqual(kwargs['cwd'],str(self.cwd))
        self.assertEqual(result['scientific_fits'],0); self.assertEqual(result['model_reads'],0)
        self.assertNotIn('private fixture value',json.dumps(result))
        self.assertEqual(json.loads((self.output/'smoke.json').read_text()),result)

    def test_unknown_pids_or_swap_blocks_fits_but_allows_contained_smoke(self):
        self.observation.update(fit_containment_verified=False, fit_blocking_reasons=['pids maximum unknown'])
        result=self.run_smoke()
        self.assertEqual(result['status'],'smoke_complete_fit_blocked')
        self.assertTrue(result['smoke_checks_passed']); self.assertFalse(result['fit_ready'])
        self.assertEqual(self.calls['imports'],list(subject.IMPORTS))

    def test_missing_cpu_or_ram_proof_blocks_native_imports(self):
        self.observation.update(status='blocked', smoke_containment_verified=False, fit_containment_verified=False)
        result=self.run_smoke()
        self.assertEqual(result['status'],'blocked')
        self.assertEqual(self.calls['imports'],[]); self.assertEqual(self.calls['commands'],[])

    def test_checkout_cwd_and_unsanitized_environment_are_refused_before_imports(self):
        self.adapters['cwd']=str(self.checkout)
        with self.assertRaisesRegex(ValueError,'outside all source'):
            self.run_smoke()
        self.adapters['cwd']=str(self.cwd); self.adapters['environment']['PYTHONPATH']=str(self.checkout)
        with self.assertRaisesRegex(ValueError,'Unsanitized'):
            self.run_smoke()
        self.assertEqual(self.calls['imports'],[])

    def test_installed_payload_tamper_is_not_treated_as_the_frozen_wheel(self):
        (self.site/'scripts/clustering_common.py').write_bytes(b'wrong installed code')
        result=self.run_smoke()
        self.assertEqual(result['status'],'blocked'); self.assertEqual(self.calls['imports'],[])
        self.assertIn('Installed payload differs',result['errors'][0]['message'])

    def test_incomplete_payload_manifest_or_resource_declaration_is_blocked(self):
        del self.plan['smoke']['library']['filehashes']['util.py']
        result=self.run_smoke()
        self.assertIn('every non-dist-info',result['errors'][0]['message'])
        self.assertEqual(self.calls['imports'],[])

    def test_missing_dependency_is_reported_and_dependent_imports_stop(self):
        del self.modules['scipy']
        result=self.run_smoke()
        self.assertEqual(result['status'],'blocked')
        self.assertIn('Missing dependency/module: scipy',json.dumps(result['errors']))
        self.assertNotIn('src.app.workers.tsne_worker',self.calls['imports'])

    def test_shadowed_module_is_rejected_before_its_import(self):
        self.modules['numpy'].__file__=str(self.checkout/'numpy.py')
        result=self.run_smoke()
        self.assertEqual(result['status'],'blocked')
        self.assertNotIn('numpy',self.calls['imports'])
        self.assertIn('outside approved prefix',json.dumps(result['errors']))

    def test_missing_hdbscan_is_reported_without_instantiating_estimator(self):
        del self.modules['sklearn.cluster'].HDBSCAN
        result=self.run_smoke()
        self.assertIn('HDBSCAN unavailable',json.dumps(result['errors']))
        self.assertNotIn('src.app.workers.tsne_worker',self.calls['imports'])

    def test_library_overlay_inside_prefix_cannot_replace_verified_wheel_source(self):
        self.modules['src.app.workers.tsne_worker'].__file__=str(self.prefix/'overlay/tsne_worker.py')
        result=self.run_smoke()
        self.assertEqual(result['status'],'blocked')
        self.assertNotIn('src.app.workers.tsne_worker',self.calls['imports'])
        self.assertIn('not from the verified wheel payload',json.dumps(result['errors']))

    def test_native_threads_must_match_sanitized_smoke_environment(self):
        self.adapters['environment']['OPENBLAS_NUM_THREADS']='8'
        with self.assertRaisesRegex(ValueError,'bounded native threads'):
            self.run_smoke()
        self.assertEqual(self.calls['imports'],[])

    def test_optional_provider_requires_exact_file_and_checkout_head(self):
        checkout=self.root/'provider'; checkout.mkdir(); path=checkout/'hd_density_experiment.py'; path.write_bytes(b'provider fixture')
        self.plan['smoke']['density_provider']=dict(record(path), checkout=str(checkout), commit='b'*40)
        result=self.run_smoke()
        self.assertEqual(result['density_provider']['status'],'verified_import_only')
        self.assertEqual(self.calls['provider'],1)
        self.assertEqual(result['density_provider']['scientific_fits'],0)

    def test_tiny_roundtrip_is_optional_and_uses_only_supplied_fake_numpy(self):
        self.plan['smoke']['numeric_roundtrip']=True
        numpy=self.modules['numpy']; numpy.float16='float16'
        numpy.asarray=lambda rows,**kwargs:SimpleNamespace(shape=(2,3),dtype='float16',rows=rows)
        def save(path, values, **kwargs):
            self.calls['numeric'].append(('save',kwargs)); path.write_bytes(b'fake tiny NPY')
        numpy.save=save; numpy.load=lambda *args,**kwargs:numpy.asarray([[1.,2.,3.],[4.,5.,6.]])
        numpy.array_equal=lambda left,right:left.rows==right.rows
        result=self.run_smoke()
        self.assertEqual(result['tiny_npy_roundtrip']['shape'],[2,3])
        self.assertEqual(self.calls['numeric'],[('save',dict(allow_pickle=False))])

    def test_resource_cancellation_propagates_instead_of_becoming_a_dependency_error(self):
        count=[0]
        def check():
            count[0]+=1
            if count[0]==4:raise RuntimeError('bounded resource stop')
        with self.assertRaisesRegex(RuntimeError,'bounded resource stop'):
            self.run_smoke(check=check)
        self.assertFalse((self.output/'smoke.json').exists())


if __name__=='__main__':
    unittest.main()
