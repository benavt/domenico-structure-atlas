"""Inspect an explicitly installed CPU runtime; imports never fit or install.

The project launcher owns exact approval, containment and environment creation.
This module imports only the standard library until run_smoke is called.
"""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


IMPORTS = ('structure_clustering', 'src', 'config', 'scripts', 'numpy', 'scipy',
           'sklearn', 'umap', 'pynndescent', 'numba', 'Bio', 'pandas',
           'matplotlib', 'matplotlib.backends.backend_agg', 'sklearn.cluster',
           'scripts.clustering_common', 'src.app.workers.tsne_worker')
ENV_KEYS = ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'LD_PRELOAD', 'TMPDIR',
            'MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'XDG_CACHE_HOME', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_NTASKS', 'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU')
REQUIRED_ASSETS = ('Sequence.py', 'util.py', 'paths.py', 'expand.py',
    'config/config.yaml', 'config/terminal_variance.json', 'config/umap_clusters.json',
    'src/app/templates/ais_explorer.html', 'structure_clustering/script_assets.json',
    'scripts/clustering_common.py', 'scripts/pca.py', 'scripts/umap_clustering.py',
    'scripts/tsne.py', 'scripts/workflow_presentation.mjs',
    'structure_clustering/runtime_patches/2026-10-13_bonsai-small-tree-memory-cleanup.patch',
    'structure_clustering/runtime_patches/2026-10-09_bonsai-empty-nnn-candidates.patch')


def need(value, message):
    if not value:
        raise ValueError(message)


def _digest(stream, check):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        check(); digest.update(block)
    return digest.hexdigest()


def _bound(entry, check):
    path = Path(entry['path']).absolute()
    need(path.is_file(), 'Missing bound file: '+str(path))
    need(path.stat().st_size == entry['bytes'], 'Bound file size differs: '+str(path))
    with path.open('rb') as stream:
        need(_digest(stream, check) == entry['sha256'], 'Bound file SHA differs: '+str(path))
    return path


def _relative(name):
    need(isinstance(name, str) and name and '\\' not in name, 'Invalid packaged file path')
    parts = PurePosixPath(name)
    need(not parts.is_absolute() and all(p not in ('', '.', '..') for p in name.split('/')),
         'Unsafe packaged file path: '+name)
    need(not any(p.endswith('.data') for p in parts.parts), 'Relocated wheel .data payload requires an explicit installer mapping')
    return name


def _production_adapters():
    # Consumer scripts are intentionally retained source, not an installed package.
    from hpc_contract import observe
    def provider(source):
        from rbp_density import load_native
        return load_native(source)
    return dict(import_module=importlib.import_module, find_spec=importlib.util.find_spec,
        distribution=importlib.metadata.distribution, run=subprocess.run, provider=provider,
        observe=observe, executable=sys.executable, prefix=sys.prefix, cwd=str(Path.cwd()),
        environment=dict(os.environ))


def _installed_wheel(plan, spec, adapters, prefix, check):
    library = spec['library']; pin = plan['sources']['library']['commit']
    need(re.fullmatch('[0-9a-f]{40}', pin) and library['commit'] == pin, 'Wheel manifest/source pin differs')
    wheel = _bound(library['wheel'], check)
    expected = library['filehashes']
    need(isinstance(expected, dict) and expected, 'Complete wheel payload file hashes required')
    for name, digest in expected.items():
        _relative(name); need(re.fullmatch('[0-9a-f]{64}', digest), 'Invalid packaged SHA: '+name)
    distribution = adapters['distribution'](library.get('distribution', 'structure-clustering'))
    need(distribution.version == library['version'], 'Installed distribution version differs')
    with zipfile.ZipFile(wheel) as archive:
        members = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        need(len(members) == len(set(members)), 'Duplicate wheel member')
        payload = {name for name in members if not any(p.endswith('.dist-info') for p in PurePosixPath(name).parts)}
        need(payload == set(expected), 'Manifest must bind every non-dist-info wheel payload file exactly')
        installed = {}
        for name, digest in expected.items():
            check()
            with archive.open(name) as stream:
                need(_digest(stream, check) == digest, 'Wheel payload SHA differs: '+name)
            path = Path(distribution.locate_file(name)).resolve()
            need(path.is_relative_to(prefix) and path.is_file(), 'Installed payload leaves the approved prefix: '+name)
            with path.open('rb') as stream:
                need(_digest(stream, check) == digest, 'Installed payload differs from frozen wheel: '+name)
            installed[name] = str(path)
    required = spec['packaged_files']
    need(isinstance(required, list) and required and len(set(required)) == len(required), 'Explicit packaged resource list required')
    need(set(REQUIRED_ASSETS) <= set(required), 'Required CPU/config/template/script/runtime-patch resources must be declared')
    for name in required:
        need(_relative(name) in installed, 'Required packaged resource missing: '+name)
    allowlist_name = 'structure_clustering/script_assets.json'
    need(allowlist_name in installed, 'Packaged script allowlist is missing')
    allowed = json.loads(Path(installed[allowlist_name]).read_text())
    need(isinstance(allowed, list) and all(isinstance(v, str) for v in allowed), 'Malformed packaged script allowlist')
    for name in installed:
        if name.startswith('scripts/') and name != 'scripts/__init__.py':
            need(name.removeprefix('scripts/') in allowed, 'Requested script is outside the packaged allowlist: '+name)
    return dict(version=distribution.version, commit=pin, wheel=library['wheel'],
                installed_payload_files=len(installed), verified_resources=required,
                _verified_origins=list(installed.values()),
                wheel_commit_binding='Exact manifest source commit and frozen wheel/payload hashes; no installed VCS metadata inferred')


def run_smoke(plan, output, check=lambda: None, *, adapters=None):
    """Run only imports/resource/CLI checks and an optional tiny NPY roundtrip.

    ``plan.smoke`` binds library wheel, complete payload hashes and packaged_files;
    optional density_provider uses the existing reviewed isolated native loader.
    Fake adapters replace imports/distributions/processes in source-only tests.
    No estimator or worker is instantiated and no fit method is invoked.
    """
    adapters = _production_adapters() if adapters is None else dict(adapters)
    spec = plan['smoke']; root = Path(plan['run_root']).resolve(); output = Path(output).absolute()
    prefix = Path(plan['runtime_prefix']).resolve(); cwd = Path(adapters['cwd']).resolve()
    need(Path(adapters['executable']).absolute() == Path(plan['runtime_python']).absolute()
         and Path(adapters['prefix']).resolve() == prefix, 'Smoke must use the exact approved interpreter/prefix')
    forbidden = [Path(p).resolve() for p in spec['outside_checkout_roots']]
    need(forbidden and not any(cwd.is_relative_to(p) for p in forbidden), 'Smoke cwd must be outside all source checkouts')
    need(output.resolve().is_relative_to(root) and not output.exists()
         and not any(p.is_symlink() for p in [output, *output.parents]), 'Use a fresh owned smoke output directory')
    environment = dict(adapters['environment'])
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'LD_PRELOAD'):
        need(not environment.get(key), 'Unsanitized import/startup environment: '+key)
    for key in ('TMPDIR', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'XDG_CACHE_HOME'):
        need(environment.get(key) and Path(environment[key]).resolve().is_relative_to(root), 'Owned runtime cache path required: '+key)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        need(environment.get(key) == '1', 'Smoke imports require bounded native threads: '+key)
    check(); output.mkdir(parents=True)
    events, errors, modules = [], [], {}
    guard_error = [None]
    def guarded():
        try:
            check()
        except BaseException as error:
            guard_error[0] = error
            raise
    def attempt(label, operation):
        guarded()
        try:
            value = operation()
        except Exception as error:
            if error is guard_error[0]:
                raise
            errors.append(dict(component=label, error_type=type(error).__name__, message=str(error)))
            events.append(label+': failed: '+str(error))
            return None
        events.append(label+': passed'); return value
    observation = attempt('resource observation', lambda: adapters['observe'](plan))
    smoke_contained = bool(observation and observation.get('smoke_containment_verified') is True)
    if not smoke_contained:
        errors.append(dict(component='resource admission', error_type='ValueError',
            message='Finite CPU/memory, Slurm and interpreter containment must be verified before native imports'))
    identity = attempt('installed wheel', lambda: _installed_wheel(plan, spec, adapters, prefix, guarded))
    verified_origins = set(identity.pop('_verified_origins')) if identity is not None else set()
    imports, commands, density, roundtrip = {}, [], None, None
    if identity is not None and smoke_contained:
        for name in IMPORTS:
            def load(name=name):
                found = adapters['find_spec'](name)
                need(found is not None, 'Missing dependency/module: '+name)
                need(found.origin and Path(found.origin).resolve().is_relative_to(prefix),
                     'Module would load outside approved prefix: '+name)
                owned = name.split('.')[0] in ('structure_clustering', 'src', 'config', 'scripts')
                if owned:
                    need(str(Path(found.origin).resolve()) in verified_origins,
                         'Library module is not from the verified wheel payload: '+name)
                module = adapters['import_module'](name)
                path = Path(module.__file__).resolve()
                need(path.is_relative_to(prefix) and path == Path(found.origin).resolve(),
                     'Imported module origin differs from approved spec: '+name)
                if owned:
                    need(str(path) in verified_origins, 'Imported library module differs from wheel: '+name)
                if name == 'structure_clustering':
                    need(module.__version__ == spec['library']['version'], 'Imported library version differs')
                if name == 'matplotlib':
                    module.use('Agg', force=True)
                    need(str(module.get_backend()).lower() == 'agg', 'Matplotlib Agg backend required')
                if name == 'matplotlib.backends.backend_agg':
                    need(callable(getattr(module, 'FigureCanvasAgg', None)), 'Agg canvas unavailable')
                if name == 'sklearn.cluster':
                    need(callable(getattr(module, 'HDBSCAN', None)), 'sklearn.cluster.HDBSCAN unavailable')
                if name == 'src.app.workers.tsne_worker':
                    need(callable(getattr(module, 'TSNEWorker', None)), 'TSNEWorker unavailable')
                expected_version = spec.get('dependency_versions', {}).get(name)
                version = getattr(module, '__version__', None)
                if expected_version is not None:
                    need(version == expected_version, 'Dependency version differs: '+name)
                modules[name] = module
                return dict(origin=str(path), version=version)
            value = attempt('import '+name, load)
            if value is not None:
                imports[name] = value
            else:
                # Dependent imports could otherwise load the rejected module
                # indirectly before its origin/version problem is corrected.
                break
        cli = spec.get('cli', {}); timeout = cli.get('timeout_seconds', 30)
        need(type(timeout) is int and 1 <= timeout <= 120, 'Finite CLI timeout required')
        probes = [('--version',)]
        if cli.get('analysis_runtime_check', True):
            probes.append(('check-runtime', '--analysis-only'))
        for arguments in probes:
            flags = ['-I']
            if plan.get('phase') in ('repro64', 'isolate64', 'curve64','curve1000'):
                import hpc_contract
                flags = hpc_contract.python_flags(plan)
            command = [plan['runtime_python'], *flags, '-m', 'structure_clustering', *arguments]
            def run(command=command, arguments=arguments):
                result = adapters['run'](command, cwd=str(cwd), env=environment, capture_output=True, text=True, timeout=timeout, check=False)
                row = dict(command=command, returncode=result.returncode, stdout=result.stdout[:65536], stderr=result.stderr[:65536])
                commands.append(row)
                need(result.returncode == 0, 'CLI probe failed: '+' '.join(arguments)+' '+result.stderr[:4096])
                if arguments == ('--version',):
                    need(result.stdout.strip() == spec['library']['version'], 'CLI library version differs')
                else:
                    value = json.loads(result.stdout)
                    need(value.get('runtime_ready') is True and value.get('profile') == 'analysis_only', 'Analysis-only runtime is unavailable')
                return row
            attempt('CLI '+' '.join(arguments), run)
        if spec.get('numeric_roundtrip', False):
            def numeric():
                numpy = modules['numpy']; values = numpy.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=numpy.float16)
                path = output/'tiny.npy'; numpy.save(path, values, allow_pickle=False)
                saved = numpy.load(path, allow_pickle=False)
                need(saved.shape == (2, 3) and saved.dtype == values.dtype and numpy.array_equal(saved, values), 'Tiny NPY roundtrip differs')
                return dict(shape=[2, 3], dtype='float16', bytes=path.stat().st_size, scientific_fits=0)
            roundtrip = attempt('tiny NPY roundtrip', numeric)
        if spec.get('density_provider') is not None:
            def provider():
                source = spec['density_provider']; path = _bound(source, guarded)
                if 'source_archive' in source:
                    _bound(source['source_archive'], guarded)
                else:
                    checkout = Path(source['checkout']).resolve()
                    need(path.resolve().is_relative_to(checkout), 'Density provider leaves its approved checkout')
                    command = ['git', '-C', str(checkout), 'rev-parse', 'HEAD']
                    result = adapters['run'](command, cwd=str(cwd), env=environment, capture_output=True, text=True, timeout=timeout, check=False)
                    need(result.returncode == 0 and result.stdout.strip() == source['commit'], 'Density source HEAD differs')
                module, proof = adapters['provider'](source)
                if 'source_archive' in source:
                    need(proof.get('source_archive') == source['source_archive'], 'Density provider archive verification proof differs')
                for name in ('fit_selection', 'fit_diagnostics', 'structural_labels'):
                    need(callable(getattr(module, name, None)), 'Density provider callable missing: '+name)
                return dict(status='verified_import_only', source=source, proof=proof, scientific_fits=0)
            density = attempt('density provider', provider)
    passed = not errors
    contained = bool(observation and observation.get('fit_containment_verified') is True)
    result = dict(schema_version=1, status=('smoke_complete' if contained else 'smoke_complete_fit_blocked') if passed else 'blocked',
        smoke_checks_passed=passed, fit_containment_verified=contained, fit_ready=passed and contained,
        smoke_containment_verified=smoke_contained,
        fit_readiness_scope='Installed CPU imports and recorded containment; a separate exact scientific plan is required',
        runtime_python=adapters['executable'], runtime_prefix=str(prefix), working_directory=str(cwd),
        environment={key:environment.get(key) for key in ENV_KEYS}, resource_observation=observation,
        installed_library=identity, imports=imports, commands=commands, density_provider=density,
        density_provider_requested=spec.get('density_provider') is not None, tiny_npy_roundtrip=roundtrip,
        errors=errors, scientific_fits=0, model_reads=0, native_TM_calls=0, molecular_exports=0,
        installation_or_network_calls=0, public_deployment=False)
    guarded()
    (output/'smoke.log').write_text('\n'.join(events)+'\n')
    with (output/'smoke.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return result
