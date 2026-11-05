"""Frozen 64-model reducer trace adapter. No native imports or fits on import."""
import importlib.util
import json
import os
from pathlib import Path
import sys

STAGES = ('umap1', 'tsne1', 'umap2', 'tsne2')
PROFILES = ('native', 'generic-jit-v1')
ISOLATION_STAGES = ('umap1', 'tsne1')


def stages(plan):
    return ISOLATION_STAGES if plan.get('phase') == 'isolate64' else STAGES


def verify_isolation_plan(plan):
    cpu, _, _ = modules()
    spec = plan.get('isolation', {})
    if (plan.get('diagnostic_profile') != 'native'
            or spec.get('optimizer_epochs') != [0, 1]
            or spec.get('initialization_policy') != 'hash-bound-pax-native-gradient-entry'
            or 'layouts' not in plan['trace_sources']
            or plan.get('stages') != list(ISOLATION_STAGES)):
        raise ValueError('Exact native optimizer/initialization isolation scope required')
    for entry in (spec['tsne_initialization'], spec['initialization_receipt'], spec['initialization_trace'],
                  spec['initialization_plan'], spec['initialization_completion']):
        if entry not in plan['frozen_files']:
            raise ValueError('Shared initialization provenance is not frozen')
        cpu.bound(entry)
    receipt = json.loads(cpu.bound(spec['initialization_receipt']).read_text())
    trace = json.loads(cpu.bound(spec['initialization_trace']).read_text())
    source_plan = json.loads(cpu.bound(spec['initialization_plan']).read_text())
    completion = json.loads(cpu.bound(spec['initialization_completion']).read_text())
    # Descriptors can relocate, but scientific content and exact source are preserved.
    entry = spec['tsne_initialization']
    if (source_plan.get('execution_host') != 'pax' or source_plan.get('diagnostic_profile') != 'native'
            or source_plan.get('phase') != 'repro64'
            or completion.get('status') != 'diagnostic_pilot_complete'
            or completion.get('plan_sha256') != spec['initialization_plan']['sha256']
            or completion['stage_receipts']['tsne1']['sha256'] != spec['initialization_receipt']['sha256']
            or receipt.get('diagnostic_profile') != 'native' or receipt.get('stage') != 'tsne1'
            or not receipt.get('phase_complete') or receipt['inputs']['feature_sha256'] != plan['inputs']['feature_sha256']
            or receipt['inputs']['model_order_sha256'] != plan['inputs']['model_order_sha256']
            or receipt['inputs']['pair_mask_sha256'] != plan['inputs']['pair_mask_sha256']
            or receipt['native_library_commit'] != plan['native']['library_commit']
            or receipt['trace']['sha256'] != spec['initialization_trace']['sha256']
            or trace.get('status') != 'complete'
            or trace['counts'] != dict(umap=0, tsne=1)
            or not any(event['name'] == 'tsne1_initial'
                       and event['shape'] == [128] and event['dtype'] == '<f4'
                       and event['artifact']['sha256'] == entry['sha256']
                       and event['artifact']['bytes'] == entry['bytes'] for event in trace['events'])):
        raise ValueError('Shared initialization is not the verified native Pax gradient-entry state')


def modules():
    source = str(Path(__file__).absolute().parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    import mimas_cpu
    import mimas_contract
    import mimas_pilot
    return mimas_cpu, mimas_contract, mimas_pilot


def helper(plan):
    cpu, _, _ = modules()
    source = cpu.bound(plan['portability_helper'])
    name = '_bound_portability_' + plan['portability_helper']['sha256']
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).absolute() != source:
            raise ValueError('Loaded diagnostic helper differs')
        return module
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def pre_import(plan):
    """Runs before the actual-child installed-wheel import smoke."""
    cpu, contract, _ = modules()
    if plan.get('diagnostic_profile') not in PROFILES or plan.get('active_stage') not in stages(plan):
        raise ValueError('Bound diagnostic profile and stage required')
    tool = helper(plan)
    tool.assert_before_native_imports()
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(Path(plan['run_root'])/'cache/python'):
        raise ValueError('Diagnostics require explicit -B and owned -X pycache_prefix before imports')
    runtime_identity_unchanged(plan)
    expected = contract.sanitized_environment(plan)
    for key in (*contract.THREAD_VARIABLES, 'DATA_PROCESSED'):
        if os.environ.get(key) != expected[key]:
            raise ValueError('Pre-import writer/thread profile differs: ' + key)
    for key in ('NUMBA_CPU_NAME', 'NUMBA_CPU_FEATURES'):
        if os.environ.get(key) != expected.get(key):
            raise ValueError('Pre-import CPU target differs: ' + key)
    return cpu.record(plan['portability_helper']['path'])


def runtime_identity_unchanged(plan):
    cpu, _, _ = modules()
    state = json.loads(cpu.bound(plan['runtime_payload_state']).read_text())
    if state['payload'] != plan['runtime_payload']:
        raise ValueError('Runtime identity snapshot belongs to another payload')
    for entry in state['files']:
        path = Path(entry['path']);info = path.stat()
        if (path.is_symlink() or [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns] != entry['identity']):
            raise ValueError('Installed runtime changed since controller verification')


def verify_runtime_payload(plan, root):
    """Hash installed package code/data once per job; children check identities."""
    cpu, _, _ = modules()
    manifest = json.loads(cpu.bound(plan['runtime_payload']).read_text())
    if manifest['runtime_prefix'] != plan['runtime_prefix']:
        raise ValueError('Runtime payload belongs to another environment')
    records = []
    for entry in manifest['files']:
        path = cpu.bound(entry);info = path.stat()
        if not path.is_relative_to(Path(plan['runtime_prefix'])/'lib/python3.11/site-packages'):
            raise ValueError('Runtime payload leaves its exact site-packages')
        records.append(dict(path=str(path),identity=[info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns]))
    destination = Path(root)/'runtime-payload-state.json'
    cpu.save(destination,dict(payload=plan['runtime_payload'],files=records))
    return cpu.record(destination)


def verify_predecessor(plan):
    """Generic trials require this host's fully closed, verified native trials."""
    if plan['diagnostic_profile'] == 'native':
        return
    gate = json.loads(Path(plan['native_pair_gate']).read_text())
    verify_native_pair(plan, gate)
    cpu, _, _ = modules()
    entry = plan['native_predecessor']
    if entry not in plan['frozen_files']:
        raise ValueError('Native predecessor plan is not frozen')
    native = json.loads(cpu.bound(entry).read_text())
    if (native.get('phase') != 'repro64' or native.get('diagnostic_profile') != 'native'
            or native['inputs'] != plan['inputs'] or native['native'] != plan['native']
            or native['sources'] != plan['sources']
            or native['execution_host'] != plan['execution_host']):
        raise ValueError('Native predecessor scope differs')
    root = Path(native['run_root'])
    completion = json.loads((root/'completion.json').read_text())
    if (completion.get('status') != 'diagnostic_pilot_complete'
            or completion.get('plan_sha256') != entry['sha256']
            or set(completion.get('stage_receipts',{})) != set(STAGES)):
        raise ValueError('Native diagnostic predecessor is not complete')
    for stage, descriptor in completion['stage_receipts'].items():
        path = cpu.bound(descriptor)
        if not path.is_relative_to(root):
            raise ValueError('Predecessor receipt escaped its run')
        result = json.loads(path.read_text())
        validate_receipt(native,stage,result)
        for output in result['outputs']:
            if not cpu.bound(output).is_relative_to(root):
                raise ValueError('Predecessor output escaped its run')


def verify_native_pair(plan, gate):
    """Both hosts must finish native diagnostics before either generic job."""
    cpu, _, pilot = modules()
    import reducer_trace_compare as saved
    if (gate.get('operation') != 'closed-native-pair-gate'
            or set(gate.get('runs',{})) != {'pax-native','mimas-native'}):
        raise ValueError('Both closed native diagnostic proofs are required')
    keys=('model_count','construct_residues','retained_reference_positions',
          'feature_sha256','model_order_sha256','pair_mask_sha256')
    expected={key:plan['inputs'][key]for key in keys}
    for name,proof in gate['runs'].items():
        if proof['plan']['sha256'] != plan['native_gate_plan_shas'][name]:
            raise ValueError('Native pair proof uses an unapproved plan')
        native=saved.data(proof['plan']);locations=saved.data(proof['locations'])
        if (native.get('diagnostic_profile')!='native' or native.get('phase')!='repro64'
                or {key:native['inputs'][key]for key in keys}!=expected
                or native['native']['library_commit']!=plan['native']['library_commit']):
            raise ValueError('Native pair cohort/source differs')
        completion=saved.data(proof['completion'],locations)
        if (completion.get('status')!='diagnostic_pilot_complete'
                or completion['plan_sha256']!=proof['plan']['sha256']
                or set(completion['stage_receipts'])!=set(STAGES)):
            raise ValueError('Native pair run is not fully closed')
        for stage,entry in completion['stage_receipts'].items():
            result=saved.data(entry,locations);method=stage[:-1]
            fits=dict(PCA=int(method=='tsne'),UMAP=int(method=='umap'),TSNE=int(method=='tsne'),HDBSCAN=0)
            if (not result.get('phase_complete') or result['stage']!=stage
                    or result['diagnostic_profile']!='native' or result['scientific_fits']!=fits
                    or result['inputs']!=native['inputs']
                    or result['diagnostic_adapter']['sha256']!=cpu.sha(__file__)
                    or result['portability_helper']['sha256']!=plan['portability_helper']['sha256']):
                raise ValueError('Native pair stage proof differs')
            for output in result['outputs']:saved.read(output,locations)
            trace=saved.data(result['trace'],locations)
            if trace.get('status')!='complete' or trace['counts']!={'umap':int(method=='umap'),'tsne':int(method=='tsne')}:
                raise ValueError('Native pair trace is incomplete')


def effective_runtime(plan):
    import numba
    import llvmlite.binding as llvm
    from threadpoolctl import threadpool_info
    generic = plan['diagnostic_profile'] == 'generic-jit-v1'
    if (numba.config.CPU_NAME != ('generic' if generic else None)
            or numba.config.CPU_FEATURES not in (('',) if generic else (None, ''))
            or numba.get_num_threads() != 1):
        raise ValueError('Effective Numba CPU/thread profile differs')
    pools = threadpool_info()
    if any(row.get('num_threads') != 1 for row in pools):
        raise ValueError('An actual native thread pool exceeds one worker')
    return dict(profile=plan['diagnostic_profile'],numba_cpu_name=numba.config.CPU_NAME,
                numba_cpu_features=numba.config.CPU_FEATURES,numba_threads=numba.get_num_threads(),
                host_cpu_name=str(llvm.get_host_cpu_name()),
                host_cpu_features=llvm.get_host_cpu_features().flatten(),native_threadpools=pools,
                python=sys.version,executable=sys.executable,prefix=sys.prefix)


def run_stage(plan, stage, output):
    cpu, _, pilot = modules()
    if stage not in stages(plan) or plan.get('active_stage') != stage:
        raise ValueError('Diagnostic stage differs from its context')
    method = stage[:-1]
    # Preserve the exact Fe94 implementation and its initialization/precision.
    # Only DATA_PROCESSED is isolated per child; imported config is checked by pilot.
    pilot.native_adapters()  # Verify native engine bytes before installing any hooks.
    import umap.umap_ as umap_module
    import sklearn.manifold._t_sne as tsne_module
    import umap
    import sklearn
    versions = dict(umap=umap.__version__,sklearn=sklearn.__version__)
    trace_modules = [('umap',umap_module),('sklearn',tsne_module)]
    options = {}; install_options = {}
    if plan.get('phase') == 'isolate64':
        verify_isolation_plan(plan)
        import umap.layouts as layout_module
        trace_modules.append(('layouts',layout_module))
        options = dict(optimizer_epoch=True,
                       tsne_initialization=plan['isolation']['tsne_initialization'] if method == 'tsne' else None)
        install_options = dict(layout_module=layout_module)
    for name, module in trace_modules:
        if cpu.record(module.__file__) != plan['trace_sources'][name]:
            raise ValueError('Pinned trace module changed: ' + name)
    tool = helper(plan)
    runtime = effective_runtime(plan)
    output = Path(output); native_output = output/'native'
    cpu.save(output/'runtime-effective.json',runtime)
    trace = tool.ReducerTrace(output/'trace', **options)
    with trace.install(umap_module,tsne_module,versions=versions,sources=plan['trace_sources'], **install_options):
        result = pilot.run_stage(plan,method,native_output)
    runtime_identity_unchanged(plan)
    cpu.save(output/'runtime-effective-after.json',effective_runtime(plan))
    traced = json.loads((output/'trace/trace.json').read_text())
    expected = dict(umap=int(method=='umap'),tsne=int(method=='tsne'))
    if traced.get('status') != 'complete' or traced.get('counts') != expected:
        raise ValueError('Trace has missing/extra fits or a cache bypass')
    # Both metadata and arrays are bound; every event must close within the owned trace.
    for event in traced['events']:
        if 'artifact' in event:
            path = cpu.bound(event['artifact'])
            if not path.is_relative_to(output/'trace'):
                raise ValueError('Checkpoint escapes owned trace')
    result = dict(result,stage=stage,native_stage=method,
        diagnostic_profile=plan['diagnostic_profile'],diagnostic_adapter=cpu.record(__file__),
        portability_helper=plan['portability_helper'],trace=cpu.record(output/'trace/trace.json'),
        runtime_effective=cpu.record(output/'runtime-effective.json'),
        runtime_effective_after=cpu.record(output/'runtime-effective-after.json'),
        native_receipt=cpu.record(native_output/'receipt.json'))
    if plan.get('phase') == 'isolate64':
        result['isolation'] = plan['isolation']
    result['outputs'] = result['outputs'] + [result['native_receipt'],result['runtime_effective'],
        result['runtime_effective_after'],result['trace']] + [event['artifact'] for event in traced['events'] if 'artifact' in event]
    validate_receipt(plan,stage,result)
    cpu.save(output/'receipt.json',result)
    return result


def validate_receipt(plan, stage, result):
    cpu, _, _ = modules()
    method = stage[:-1]
    expected_fits = dict(PCA=int(method=='tsne'),UMAP=int(method=='umap'),
                         TSNE=int(method=='tsne'),HDBSCAN=0)
    if (result.get('diagnostic_profile') != plan['diagnostic_profile']
            or result.get('native_stage') != method or result.get('scientific_fits') != expected_fits
            or result.get('portability_helper') != plan['portability_helper']
            or result.get('diagnostic_adapter') != cpu.record(__file__)):
        raise ValueError('Diagnostic stage/profile/fit/source proof differs')
    traced = json.loads(cpu.bound(result['trace']).read_text())
    if (traced.get('status') != 'complete'
            or traced.get('counts') != dict(umap=int(method=='umap'),tsne=int(method=='tsne'))
            or traced.get('maximum_array_bytes') != 16*1024**2
            or traced.get('maximum_trace_bytes') != 64*1024**2
            or traced.get('sources') != plan['trace_sources']):
        raise ValueError('Unclosed, oversized or changed diagnostic trace')
    for item in ('trace','runtime_effective','runtime_effective_after','native_receipt'):
        if result[item] not in result['outputs']:
            raise ValueError('Unbound diagnostic output')
    if plan.get('phase') == 'isolate64':
        names = {e['name'] for e in traced['events']}
        replay = plan['isolation']['tsne_initialization'] if method == 'tsne' else None
        expected_events = ({'umap1_optimizer_scalars', 'umap1_epoch0_scalars', 'umap1_epoch1_scalars',
                            'umap1_epoch0_after_head_embedding', 'umap1_epoch1_after_head_embedding'}
                           if method == 'umap' else {'tsne1_native_initial', 'tsne1_initial', 'tsne1_initialization_source'})
        if (result.get('isolation') != plan['isolation']
                or traced.get('optimizer_epoch_trace') is not True
                or method == 'umap' and traced.get('epoch_calls', 0) < 2
                or traced.get('tsne_initialization') != replay
                or traced.get('initialization_replays') != int(method == 'tsne')
                or not expected_events <= names):
            raise ValueError('Missing extended epoch trace or shared initialization proof')
    return result
