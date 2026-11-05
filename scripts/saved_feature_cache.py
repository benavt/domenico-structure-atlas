"""Verify and map an already-selected native feature cache without rebuilding it.

Call only inside an approved, protected execution plan. Importing this module
performs no numerical work. Matrix bytes stay in the original ZIP_STORED member;
only bounded hashing buffers and small non-object metadata are read into RAM.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from zipfile import ZipFile, ZIP_STORED

from terminal_rescreen import read_bound, validate_axes, selection


BLOCK_BYTES = 16 * 1024**2
METADATA_BYTES = 16 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _regular(path):
    path = Path(path).absolute()
    need('..' not in path.parts and path.is_file()
         and not any(p.is_symlink() for p in [path, *path.parents]),
         'Selected cache must be a regular non-symlink file')
    return path


def _header(stream):
    import numpy as np
    version = np.lib.format.read_magic(stream)
    need(version in ((1, 0), (2, 0)), 'Unsupported numeric NPY version')
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = reader(stream)
    return shape, fortran, dtype, stream.tell()


def _metadata_array(archive, name, *, scalar):
    """Check declared allocation before reading any metadata payload."""
    import numpy as np
    info = archive.getinfo(name)
    need(info.file_size <= METADATA_BYTES and not info.flag_bits & 1,
         'Cache metadata member exceeds its bound or is encrypted')
    with archive.open(info) as stream:
        shape, fortran, dtype, offset = _header(stream)
        need(not fortran and not dtype.hasobject
             and (shape == () and dtype.kind in 'US' if scalar else
                  len(shape) == 1 and dtype.kind in 'iu'),
             'Metadata must be non-object scalar text or an integer vector')
        count = 1 if scalar else shape[0]
        size = count * dtype.itemsize
        need(0 < dtype.itemsize <= METADATA_BYTES
             and 0 <= size <= METADATA_BYTES and offset + size == info.file_size,
             'Metadata NPY declaration exceeds its actual byte bound')
        payload = stream.read(size)
        need(len(payload) == size and not stream.read(1), 'Truncated metadata NPY')
    return np.frombuffer(payload, dtype=dtype).reshape(shape)


def _text(archive, name):
    value = _metadata_array(archive, name, scalar=True).item()
    return value.decode('utf-8') if isinstance(value, bytes) else value


def _pair_mask_digest(length, retained):
    """Hash the declared mask without allocating a full pair-index workspace."""
    import numpy as np
    keep = np.zeros(length, dtype=bool)
    keep[retained] = True
    digest = hashlib.sha256()
    for index in range(length-1):
        digest.update((keep[index] & keep[index+1:]).tobytes())
    return digest.hexdigest()


def _inputs(cache, axes, baseline, terminal_report, source_metadata, feature_proof, check):
    axes_value = read_bound(axes, check)
    baseline_value = read_bound(baseline, check)
    baseline_value = baseline_value.get('terminal_detection', baseline_value)
    names, requested, keys = validate_axes(axes_value, baseline_value)
    metadata = read_bound(source_metadata, check)
    need(all(metadata['reference'][key] == axes_value['metadata']['reference'][key]
             for key in ('sequence', 'chain_ids', 'residue_numbers'))
         and metadata['shared_reference_positions'] == requested,
         'Selected-cache reference metadata differs')
    report = read_bound(terminal_report, check)
    chosen = selection(report, requested, keys)
    retained = chosen['retained_canonical_indices']
    mask_sha = _pair_mask_digest(len(requested), report['retained_indices'])
    expected = [len(names), len(retained)*(len(retained)-1)//2]
    proof = read_bound(feature_proof, check)
    need(report['model_count'] == len(names) and report['reference_positions'] == requested
         and report['retained_reference_positions'] == retained
         and report['input_residue_keys'] == keys
         and report['pair_mask_sha256'] == mask_sha
         and report['retained_feature_count'] == expected[1],
         'Accepted selected mask or feature count differs')
    need(cache['member'] == 'flattened.npy' and cache['shape'] == expected
         and proof['shape'] == expected and proof['dtype'] == 'float16'
         and proof['retained_reference_positions'] == retained
         and proof['models_sha256'] == axes_value['names_sha256']
         and proof['pair_mask_sha256'] == mask_sha
         and re.fullmatch(r'[0-9a-f]{64}', proof['feature_sha256']),
         'Selected feature proof/order/precision differs')
    return axes_value, names, requested, retained, metadata, report, proof, tuple(expected)


@contextmanager
def mapped_selected_features(cache, *, axes, baseline, terminal_report,
                             source_metadata, feature_proof, check=lambda: None):
    """Yield ``(readonly_float16_matrix, verification_proof)`` from one open FD.

    Arguments are hash-bound {path,sha256} records. ``cache`` additionally needs
    bytes, member='flattened.npy' and the exact selected [model,pair] shape. The
    full-reference axes/baseline authenticate the already accepted subset; they
    never cause retrimming, distance recomputation or original-coordinate reads.
    ``check`` is a cancellation/resource poll, not a scientific checkpoint.
    """
    import numpy as np
    need(callable(check), 'A resource/cancellation check is required')
    axes_value, names, requested, retained, metadata, report, prior, expected = _inputs(
        cache, axes, baseline, terminal_report, source_metadata, feature_proof, check)
    path = _regular(cache['path'])
    before_path = path.stat()
    with path.open('rb') as raw:
        before = os.fstat(raw.fileno())
        need(stat.S_ISREG(before.st_mode) and _identity(before) == _identity(before_path)
             and before.st_size == cache['bytes'], 'Selected cache identity or size differs')
        archive_digest = hashlib.sha256()
        while True:
            block = raw.read(BLOCK_BYTES)
            if not block:
                break
            archive_digest.update(block)
            del block
            check()
        need(archive_digest.hexdigest() == cache['sha256'], 'Selected cache SHA differs')
        raw.seek(0)
        with ZipFile(raw) as archive:
            members = [info.filename for info in archive.infolist()]
            need(len(members) == len(set(members)) and all(
                not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
                and '\\' not in name for name in members), 'Unsafe or duplicate ZIP member')
            signature = json.loads(_text(archive, 'source_signature.npy'))
            metadata_sha = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
            need([Path(row[0]).stem for row in signature['files']] == names
                 and signature['metadata_sha256'] == metadata_sha
                 and signature['terminal_rescreen_report_sha256'] == terminal_report['sha256'],
                 'Selected cache cohort/source/report signature differs')
            need(_metadata_array(archive, 'reference_positions.npy', scalar=False).tolist() == retained
                 and _metadata_array(archive, 'requested_reference_positions.npy', scalar=False).tolist() == requested
                 and _text(archive, 'matrix_entity.npy') == 'residue'
                 and json.loads(_text(archive, 'terminal_detection.npy')) == report,
                 'Selected cache axes/entity/terminal report differs')
            info = archive.getinfo(cache['member'])
            need(info.compress_type == ZIP_STORED and not info.flag_bits & 1
                 and info.file_size == info.compress_size,
                 'Selected numeric member must be unencrypted ZIP_STORED')
            with archive.open(info) as numeric:
                shape, fortran, dtype, header_bytes = _header(numeric)
                need(shape == expected and not fortran and dtype == np.dtype('<f2')
                     and info.file_size == header_bytes + expected[0]*expected[1]*2,
                     'Selected numeric member must have exact float16 C layout/shape')
                feature_digest = hashlib.sha256()
                data_bytes = 0
                while True:
                    block = numeric.read(BLOCK_BYTES)
                    if not block:
                        break
                    data_bytes += len(block)
                    feature_digest.update(block)
                    del block
                    check()
            # Complete ZipExtFile consumption above also verifies its stored CRC.
            need(feature_digest.hexdigest() == prior['feature_sha256'], 'Selected feature SHA differs')
            raw.seek(info.header_offset)
            local = raw.read(30)
            need(len(local) == 30 and local[:4] == b'PK\x03\x04'
                 and struct.unpack_from('<H', local, 8)[0] == ZIP_STORED
                 and struct.unpack_from('<H', local, 6)[0] == info.flag_bits,
                 'Selected member local ZIP header differs')
            filename_size, extra_size = struct.unpack_from('<HH', local, 26)
            local_name = raw.read(filename_size)
            need(local_name.decode('utf-8' if info.flag_bits & 0x800 else 'cp437') == info.filename,
                 'Selected member local ZIP name differs')
            raw.seek(extra_size, 1)
            member_start = raw.tell()
            mapped_shape, mapped_fortran, mapped_dtype, offset = _header(raw)
            need(mapped_shape == shape and mapped_fortran == fortran and mapped_dtype == dtype
                 and offset-member_start == header_bytes
                 and offset+data_bytes == member_start+info.file_size <= before.st_size,
                 'Selected numeric byte range differs')
            need(_identity(os.fstat(raw.fileno())) == _identity(before)
                 and _identity(path.stat()) == _identity(before), 'Selected cache changed during verification')
            matrix = np.memmap(raw, dtype='<f2', mode='r', offset=offset, shape=expected, order='C')
            verification = dict(schema_version=1, kind='verified_readonly_selected_native_cache',
                cache=cache, shape=list(expected), dtype='float16', order='C', readonly=True,
                member_crc_verified=True, member_data_offset=offset, member_data_bytes=data_bytes,
                feature_sha256=feature_digest.hexdigest(), feature_proof=feature_proof,
                model_order_sha256=axes_value['names_sha256'], sequence_sha256=axes_value['sequence_sha256'],
                retained_reference_positions=retained, requested_reference_positions=requested,
                pair_mask_sha256=report['pair_mask_sha256'], terminal_report=terminal_report,
                source_metadata=source_metadata, source_metadata_sha256=metadata_sha,
                source_signature=signature, metadata=metadata,
                distance_recomputations=0, original_coordinate_reads=0, full_array_copy=False)
            failure = None
            try:
                yield matrix, verification
            except BaseException as error:
                failure = error
                raise
            finally:
                matrix._mmap.close()
                try:
                    need(_identity(os.fstat(raw.fileno())) == _identity(before)
                         and _identity(path.stat()) == _identity(before),
                         'Selected cache changed while mapped')
                except Exception as integrity_error:
                    if failure is None:
                        raise
                    failure.add_note('Selected-cache integrity check failed: '+str(integrity_error))
