"""Approved saved-float16 terminal rescreening; no scientific work on import.

The reviewed outer runner owns admission, cgroup limits and pressure policy.
Checkpoint callbacks mark safe backoff boundaries, not resume acceptance. The
full read-only input map remains necessary; temporary reduction buffers are
released before callbacks. Old conservative reports and inputs are never changed.
"""
from contextlib import contextmanager
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import shutil
import time
from zipfile import ZipFile, ZIP_STORED

BLOCK_FEATURES = 512


def need(condition, message):
    if not condition:
        raise ValueError(message)


def names_digest(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()


def _path(value):
    path = Path(value).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]),
         'Input must be a regular non-symlink file: '+str(path))
    return path


def _hash(stream, check):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(16*1024**2), b''):
        check(); digest.update(block)
    return digest.hexdigest()


def read_bound(entry, check=lambda: None):
    path = _path(entry['path'])
    with path.open('rb') as stream:
        need(_hash(stream, check) == entry['sha256'], 'Frozen input hash differs: '+str(path))
        stream.seek(0)
        return json.loads(gzip.decompress(stream.read())) if path.suffix == '.gz' else json.load(stream)


def _json_member(archive, name):
    import numpy as np
    info = archive.getinfo(name)
    need(info.file_size <= 16*1024**2, 'Cache metadata member exceeds bound')
    with archive.open(info) as stream:
        value = np.load(stream, allow_pickle=False)
        need(value.ndim == 0 and value.dtype.kind in 'US', 'Metadata must be scalar non-object JSON')
        return json.loads(str(value.item()))


def _positions(archive, name):
    import numpy as np
    info = archive.getinfo(name)
    need(info.file_size <= 1024*1024, 'Cache position member exceeds bound')
    with archive.open(info) as stream:
        value = np.load(stream, allow_pickle=False)
        need(value.ndim == 1 and value.dtype.kind in 'iu', 'Positions must be an integer vector')
        return value.tolist()


def validate_axes(axes, baseline):
    names, positions = axes['names'], axes['reference_positions']
    need(isinstance(names, list) and len(names) >= 3 and len(set(names)) == len(names)
         and all(isinstance(n, str) and n and '\n' not in n and Path(n).name == n for n in names),
         'Exact unique model names required')
    need(names_digest(names) == axes['names_sha256'], 'Model-name order hash differs')
    ref = axes['metadata']['reference']; sequence = ref['sequence']; n = len(sequence)
    need(n >= 16 and positions == list(range(n)), 'Full canonical sequence axes required')
    need(hashlib.sha256(sequence.encode()).hexdigest() == axes['sequence_sha256'], 'Sequence hash differs')
    chains, numbers = ref['chain_ids'], ref['residue_numbers']
    need(len(chains) == len(numbers) == n and len(set(map(str, chains))) == 1,
         'One exact native reference chain is required per target')
    keys = [[str(c), str(r)] for c, r in zip(chains, numbers)]
    need(len({tuple(k) for k in keys}) == n and all(c and r != '-1' for c, r in keys),
         'Reference chain/residue identities are absent or ambiguous')
    need(axes['metadata']['shared_reference_positions'] == positions
         and baseline['input_residue_keys'] == keys and baseline['model_count'] == len(names)
         and baseline['retained_indices'] == positions and baseline['excluded_indices'] == [],
         'Baseline is not the same full-axis model cohort')
    return names, positions, keys


@contextmanager
def stored_matrix(entry, axes, baseline, *, check=lambda: None, source_metadata=None):
    """CRC-check a hash-bound STORED NPY, then map its bytes read-only in place.

    Object arrays in the ordinary NPZ are never loaded. Mapping the same open
    descriptor avoids a filename replacement between verification and use.
    """
    import numpy as np
    path = _path(entry['path']); matrix = None
    names, positions, _ = validate_axes(axes, baseline)
    expected = (len(names), len(positions)*(len(positions)-1)//2)
    need(tuple(entry['shape']) == expected and entry['member'] == 'flattened.npy', 'Cache descriptor shape/member differs')
    with path.open('rb') as raw:
        before = os.fstat(raw.fileno())
        need(before.st_size == entry['bytes'] and _hash(raw, check) == entry['sha256'], 'Saved cache hash/size differs')
        raw.seek(0)
        with ZipFile(raw) as archive:
            infos = archive.infolist(); members = [i.filename for i in infos]
            need(len(set(members)) == len(members) and all(
                not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n
                for n in members), 'Unsafe or duplicate ZIP member')
            info = archive.getinfo(entry['member'])
            need(info.compress_type == ZIP_STORED and not info.flag_bits & 1
                 and info.file_size == info.compress_size, 'Only unencrypted ZIP_STORED numeric inputs are supported')
            signature = _json_member(archive, 'source_signature.npy')
            need([Path(row[0]).stem for row in signature['files']] == names, 'Cache model order differs')
            if source_metadata is not None:
                need(signature['metadata_sha256'] == hashlib.sha256(json.dumps(source_metadata, sort_keys=True).encode()).hexdigest(),
                     'Cache source metadata signature differs')
            need(_positions(archive, 'reference_positions.npy') == positions
                 and _positions(archive, 'requested_reference_positions.npy') == positions, 'Cache canonical axes differ')
            detection = _json_member(archive, 'terminal_detection.npy')
            need(detection['input_residue_keys'] == baseline['input_residue_keys']
                 and detection['model_count'] == len(names)
                 and detection['retained_indices'] == positions and detection['excluded_indices'] == []
                 and detection['pair_mask_sha256'] == baseline['pair_mask_sha256'], 'Cache baseline mask differs')
            # Reading the complete member triggers zipfile's CRC verification.
            with archive.open(info) as numeric:
                for block in iter(lambda: numeric.read(16*1024**2), b''):
                    check()
            raw.seek(info.header_offset)
            header = raw.read(30)
            need(len(header) == 30 and header[:4] == b'PK\x03\x04'
                 and struct.unpack_from('<H', header, 8)[0] == ZIP_STORED, 'Invalid local ZIP header')
            filename_length, extra_length = struct.unpack_from('<HH', header, 26)
            local_name = raw.read(filename_length)
            need(local_name.decode('utf-8' if info.flag_bits & 0x800 else 'cp437') == info.filename,
                 'Local ZIP name differs')
            raw.seek(extra_length, 1); member_offset = raw.tell()
            version = np.lib.format.read_magic(raw)
            need(version in ((1, 0), (2, 0)), 'Unsupported numeric NPY version')
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0(raw) if version == (1, 0)
                                      else np.lib.format.read_array_header_2_0(raw))
            offset = raw.tell()
            need(shape == expected and not fortran and dtype == np.dtype('<f2'),
                 'Numeric cache must be exact little-endian float16 C layout')
            need(offset + expected[0]*expected[1]*2 == member_offset+info.file_size
                 and member_offset+info.file_size <= before.st_size, 'Numeric NPY member byte bound differs')
            matrix = np.memmap(raw, dtype='<f2', mode='r', offset=offset, shape=expected, order='C')
            try:
                yield matrix
                after = os.fstat(raw.fileno())
                need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                     == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Cache changed during screening')
            finally:
                matrix._mmap.close()


def native_engine():
    from src.analysis import terminal_variance as native
    from scipy.spatial.distance import squareform
    return native, squareform


def selection(report, positions, keys):
    """Reject any inferred internal exclusion or a bypass of native trim budgets."""
    excluded, retained = report['excluded_indices'], report['retained_indices']
    n = len(positions); all_indices = set(range(n)); removed = set(excluded)
    need(excluded == sorted(removed) and retained == sorted(all_indices-removed)
         and removed <= all_indices and len(removed) <= n//4
         and len(retained) >= report['settings']['minimum_remaining_residues'], 'Invalid retained mask or exclusion budget')
    ranges = []; covered = set()
    for segment in report['segments']:
        indices = segment['indices']; end = segment['end']
        need(end in ('N', 'C') and indices and indices == sorted(set(indices)), 'Only terminal cuts are supported')
        need(indices == (list(range(len(indices))) if end == 'N' else list(range(n-len(indices), n))),
             'An inferred exclusion is not a contiguous actual terminal')
        need(segment['residue_keys'] == [keys[i] for i in indices], 'Inferred native residue identities differ')
        covered.update(indices)
        ranges.append(dict(end=end, canonical_start_index=positions[indices[0]],
                           canonical_end_index=positions[indices[-1]],
                           canonical_start_position=positions[indices[0]]+1,
                           canonical_end_position=positions[indices[-1]]+1,
                           native_start=keys[indices[0]], native_end=keys[indices[-1]], count=len(indices)))
    need(covered == removed and report['input_residue_keys'] == keys, 'Terminal report mask/keys differ')
    return dict(canonical_index_base=0, canonical_position_base=1,
                native_keys_scope='reference_model_only; other models preserve their deposited native IDs',
                retained_canonical_indices=[positions[i] for i in retained],
                retained_native_keys=[keys[i] for i in retained], excluded_terminal_ranges=ranges,
                excluded_count=len(excluded), retained_count=len(retained),
                interpretation='Threshold-based terminal flexibility selection; not a biological disorder annotation.')


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def run_target(descriptor, output, *, check=lambda: None, checkpoint=lambda _: None, engine=None):
    """Compute only saved-float16 variance and native default terminal decisions.

    Callable from an approved guarded runner. ``engine`` is a mock-only seam;
    production uses public functions from the hash-bound native library.
    """
    import numpy as np
    target = descriptor['target']
    need(isinstance(target, str) and re.fullmatch(r'[A-Za-z0-9_-]+', target), 'Unsafe target name')
    axes = read_bound(descriptor['axes'], check); baseline = read_bound(descriptor['baseline'], check)
    baseline = baseline.get('terminal_detection', baseline)
    names, positions, keys = validate_axes(axes, baseline)
    source_metadata = read_bound(descriptor['source_metadata'], check)
    need(all(source_metadata['reference'][k] == axes['metadata']['reference'][k]
             for k in ('sequence', 'chain_ids', 'residue_numbers'))
         and source_metadata['shared_reference_positions'] == positions, 'Frozen axes differ from source metadata')
    settings_record = read_bound(descriptor['settings'], check)
    native, squareform = engine if engine is not None else native_engine()
    settings = native.load_terminal_settings(descriptor['settings']['path'])
    need(settings_record == {'schema_version': 1, 'settings': __import__('dataclasses').asdict(settings)}
         and settings == native.TerminalSettings(), 'Exact reviewed native default settings required')
    output = Path(output).absolute()
    need(not output.exists() and not any(p.is_symlink() for p in [output, *output.parents]), 'Fresh immutable target output required')
    output.mkdir(parents=True)
    with stored_matrix(descriptor['cache'], axes, baseline, check=check, source_metadata=source_metadata) as matrix:
        checkpoint(target+':cache-verified')
        variance = np.empty(matrix.shape[1], dtype=float)
        for start in range(0, matrix.shape[1], BLOCK_FEATURES):
            check(); stop = min(start+BLOCK_FEATURES, matrix.shape[1])
            # Exact native screen_condensed_stack reduction, including dtype,
            # ddof=0 and column block width. No coordinate/summary reconstruction.
            variance[start:stop] = np.var(np.asarray(matrix[:, start:stop], float), axis=0)
            check(); checkpoint(f'{target}:variance:{stop}/{matrix.shape[1]}')
        need(np.isfinite(variance).all() and (variance >= 0).all(), 'Nonfinite or negative distance variance')
        variance = squareform(variance)
        with (output/'variance.npy').open('xb') as stream:
            np.save(stream, variance, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        checkpoint(target+':variance-saved')
        metadata = axes['metadata']; ref = metadata['reference']
        native_keys, ends = native.reference_terminal_axes(metadata, positions)
        need([list(k) for k in native_keys] == keys, 'Native helper reference axes differ')
        chain_lengths = {chain:len({str(number) for c,number in zip(ref['chain_ids'],ref['residue_numbers'])
                                   if str(c) == chain and number != -1}) for chain in dict.fromkeys(map(str,ref['chain_ids']))}
        check()
        report = native.detect_terminal_variance(variance, native_keys, model_count=len(names),
            settings=settings, eligible_ends=ends, chain_lengths=chain_lengths)
        check()
    chosen = selection(report, positions, keys)
    keep = np.zeros(len(positions), bool); keep[report['retained_indices']] = True
    i,j = np.triu_indices(len(positions), 1); mask = keep[i] & keep[j]
    report.update(input_feature_count=len(mask), retained_feature_count=int(mask.sum()),
                  pair_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                  reference_positions=positions, retained_reference_positions=chosen['retained_canonical_indices'])
    need('domain_evidence' in report and len(report.get('terminal_anchor_domains', [])) == 1,
         'Native default empirical-core evidence is incomplete')
    with (output/'pair_mask.npy').open('xb') as stream:
        np.save(stream, mask, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    _write(output/'report.json', report); _write(output/'selection.json', chosen)
    _write(output/'baseline.json', baseline)
    receipt = dict(schema_version=1, target=target, models=len(names), sequence_sha256=axes['sequence_sha256'],
        names_sha256=axes['names_sha256'], inputs=descriptor,
        reduction='Native screen_condensed_stack: storedfloat16 -> np.asarray(...,float64) -> np.var(axis=0,ddof=0),512features/block',
        baseline_recomputed=False, baseline_excluded_indices=baseline['excluded_indices'],
        default_excluded_indices=report['excluded_indices'], hierarchical_core_description_calls=2,
        model_projection_fits=0, native_TM_calls=0, coordinate_parses=0, scoring_calls=0,
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        native_sources=({'mock_injected_for_source_tests':True} if engine is not None else {
            name:dict(path=str(Path(native.__file__).with_name(name)),
                      sha256=hashlib.sha256(Path(native.__file__).with_name(name).read_bytes()).hexdigest())
            for name in ('terminal_variance.py','structural_flexibility.py')}),
        status='complete', checkpoint_semantics='Safe backoff only; RAM progress cannot authorize recovery or scientific reuse.')
    _write(output/'receipt.json', receipt); checkpoint(target+':report-closed')
    return receipt


def validate_approval(path):
    path = _path(path); plan = json.loads(path.read_text())
    approval = json.loads((path.parent/'approval.json').read_text())
    need(approval.get('approved_by') == 'user' and approval.get('approved_plan_sha256') == hashlib.sha256(path.read_bytes()).hexdigest(),
         'Exact execution approval required')
    need(plan.get('operation') == 'saved-terminal-rescreen' and plan['settings'] == dict(
        block_features=512, maximum_targets=2, maximum_hierarchical_core_calculations=4), 'Approved terminal operation/settings differ')
    need(len(plan['targets']) == 2 and len({d['target'] for d in plan['targets']}) == 2, 'Exactly two distinct target descriptors required')
    need(os.getpgrp() == os.getpid(), 'Driver requires the guard-owned private process group')
    from run_reviewed_plan import verify_controls, control_group
    verify_controls(control_group(), plan['resources'])
    for name, digest in plan['frozen_files'].items():
        with _path(name).open('rb') as stream:
            need(_hash(stream, lambda: None) == digest, 'Frozen file changed: '+name)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('plan', type=Path)
    args = parser.parse_args(); plan = validate_approval(args.plan)
    from run_reviewed_plan import reviewed_guard
    guard = reviewed_guard(plan)
    recovery = plan['guard']['io_recovery']
    client = guard.DurableCheckpoint(recovery['control_directory'], recovery['run_id'])
    # Poll/check and safe-boundary callbacks stay separate; never fabricate a
    # boundary from a heartbeat while the native core inference is running.
    started = time.monotonic()
    def check():
        need(time.monotonic()-started < plan['resources']['maximum_wall_seconds'], 'Approved wall cap reached')
        need(shutil.disk_usage(plan['run_root']).free >= plan['resources']['preserved_free_space_bytes'], 'Storage reserve reached')
    client.checkpoint('terminal-rescreen-start')
    receipts = []
    for descriptor in plan['targets']:
        check()
        receipts.append(run_target(descriptor, Path(plan['run_root'])/descriptor['target'],
                                   check=check, checkpoint=client.checkpoint))
    _write(Path(plan['run_root'])/'terminal-rescreen-receipt.json', dict(status='complete',targets=receipts,
        execution_plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),sources=plan['sources'],
        hierarchical_core_description_calls=sum(r['hierarchical_core_description_calls'] for r in receipts),
        model_projection_fits=0,native_TM_calls=0,coordinate_parses=0,scoring_calls=0))


if __name__ == '__main__':
    main()
