"""Fail if files expose private machine names, internal hosts or private paths.

Modes:
  check_privacy.py [REPO]          scan git-tracked files of REPO (default '.')
  check_privacy.py --dir DIR       scan every file under DIR (e.g. a built site)

Private host names are matched case-insensitively anywhere, including embedded
forms.  Detector literals are assembled from fragments so this file and its
tests are not themselves matches.  Containers are recognised by content
(gzip, bzip2, xz, zip/Office/npz, tar, PDF text and PDF attachments) and are
scanned recursively in memory or in a private temporary directory; nothing is
extracted into the tree, links are never followed, and any unreadable,
corrupt, too-deep or too-large container is reported as a finding (fail
closed).
"""
import bz2, gzip, io, lzma, os, re, subprocess, sys, tarfile, tempfile, zipfile
from pathlib import Path

_HOSTS = ('pa' + 'x', 'mi' + 'mas')
HOST_RE = re.compile('|'.join(_HOSTS), re.IGNORECASE)
NETWORK_RE = re.compile(r'\.ts' r'\.net\b|tail' r'scale|github\.' r'rpi\.edu', re.IGNORECASE)
PATH_RE = re.compile(r'/(?:ho' r'me|scr' r'atch|gp' r'fs|Us' r'ers)/[A-Za-z0-9_][\w.-]*')

# Exact Python tarfile API identifiers that merely share letters with a host
# name, allowed only in these files and only as whole lexical tokens.
_T = 'pa' + 'x'
TOKEN_ALLOWS = {
    'scripts/rbp_density.py': (r'(?<=\.)' + _T + r'_headers(?![\w])',),
    'tests/test_hpc_density_snapshot.py': (r'(?<=tarfile\.)' + _T.upper() + r'_FORMAT(?![\w])',
                                           r'(?<=[\s,(])' + _T + r'_headers(?==)',
                                           r'(?<=\.)' + _T + r'_headers(?![\w])'),
    # Vendored plotly.js 3.1.0 (bundled into the built site): camelCase
    # "Opp"+"Axis" identifiers and the Maya-calendar month list.
    'assets/plotly.min.js': (r'(?<=[a-z0-9])Opp' + r'Axis(?=[A-Z(=])',
                             r'(?<="Muan",)"Pa' + r'x"(?=,"Kayab")'),
}
# Synthetic placeholder home/scratch paths used to test report redaction.
# Only the private-path rule is relaxed; host and network rules still apply.
PATH_RULE_EXEMPT = frozenset({'tests/test_public_report.py', 'tests/test_rbp_public_states.py',
                              'tests/test_standard_publication.py'})

MAX_DEPTH = 6
MAX_BYTES = 512 * 1024 * 1024


class ScanError(Exception):
    pass


def scan_text(text, rel_path=None, path_exempt_ok=True):
    allows = TOKEN_ALLOWS.get(rel_path, ())
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        checked = line
        for pattern in allows:
            checked = re.sub(pattern, ' ', checked)
        if HOST_RE.search(checked):
            hits.append((number, 'private host name'))
        elif NETWORK_RE.search(line):
            hits.append((number, 'internal host'))
        elif not (path_exempt_ok and rel_path in PATH_RULE_EXEMPT) and PATH_RE.search(line):
            hits.append((number, 'private path'))
    return hits


def _kind(data):
    if data[:2] == b'\x1f\x8b':
        return 'gzip'
    if data[:3] == b'BZh':
        return 'bzip2'
    if data[:6] == b'\xfd7zXZ\x00':
        return 'xz'
    if data[:4] in (b'PK\x03\x04', b'PK\x05\x06'):
        return 'zip'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if data[:5] == b'%PDF-':
        return 'pdf'
    if len(data) > 262 and data[257:262] == b'ustar':
        return 'tar'
    return None


def _decompress(kind, data):
    module = {'gzip': gzip, 'bzip2': bz2, 'xz': lzma}[kind]
    stream = module.open(io.BytesIO(data)) if kind != 'gzip' else gzip.GzipFile(fileobj=io.BytesIO(data))
    out = stream.read(MAX_BYTES + 1)
    if len(out) > MAX_BYTES:
        raise ScanError('decompressed size limit exceeded')
    return out


def _pdf(label, data, depth, budget):
    with tempfile.TemporaryDirectory(prefix='privacy-pdf-') as tmp:
        os.chmod(tmp, 0o700)
        pdf = os.path.join(tmp, 'input.pdf')
        with open(pdf, 'wb') as handle:
            handle.write(data)
        try:
            text = subprocess.run(['pdftotext', '-q', pdf, '-'], capture_output=True, check=True, timeout=300).stdout
            listing = subprocess.run(['pdfdetach', '-list', pdf], capture_output=True, check=True, timeout=300).stdout
            out = os.path.join(tmp, 'attachments')
            os.mkdir(out, 0o700)
            subprocess.run(['pdfdetach', '-saveall', '-o', out, pdf], capture_output=True, check=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as error:
            raise ScanError('PDF text or attachments could not be read (needs poppler-utils)') from error
        yield label, text.decode('utf-8', 'ignore')
        yield label + '!attachments', listing.decode('utf-8', 'ignore')
        for entry in sorted(os.scandir(out), key=lambda e: e.name):
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ScanError('unexpected PDF attachment entry')
            with open(entry.path, 'rb') as handle:
                yield from _walk(label + '!' + entry.name, handle.read(), depth + 1, budget)


def _png(label, data):
    """Scan PNG text metadata; compressed image data is not text."""
    import struct, zlib
    pos = 8
    while pos < len(data):
        if pos + 8 > len(data):
            raise ScanError('truncated PNG')
        size, ctype = struct.unpack('>I4s', data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + size]
        if len(body) != size:
            raise ScanError('truncated PNG')
        if ctype == b'tEXt':
            yield label + '#tEXt', body.decode('latin-1')
        elif ctype == b'zTXt':
            key, _, rest = body.partition(b'\0')
            yield label + '#zTXt', key.decode('latin-1') + ' ' + zlib.decompress(rest[1:]).decode('latin-1')
        elif ctype == b'iTXt':
            key, _, rest = body.partition(b'\0')
            compressed = rest[:1] == b'\x01'
            fields = rest[2:].split(b'\0', 2)
            text = fields[2] if len(fields) == 3 else b''
            yield label + '#iTXt', b' '.join([key] + fields[:2]).decode('utf-8', 'ignore') + ' ' + (
                zlib.decompress(text) if compressed else text).decode('utf-8', 'ignore')
        elif ctype == b'eXIf':
            yield label + '#eXIf', body.decode('latin-1')
        pos += 12 + size
        if ctype == b'IEND':
            return
    raise ScanError('PNG without IEND')


def _walk(label, data, depth, budget):
    """Yield (label, text) pairs for data and anything nested inside it."""
    budget[0] += len(data)
    if depth > MAX_DEPTH:
        raise ScanError('container nesting limit exceeded')
    if budget[0] > MAX_BYTES:
        raise ScanError('total scanned size limit exceeded')
    kind = _kind(data)
    try:
        if kind in ('gzip', 'bzip2', 'xz'):
            inner = _decompress(kind, data)
            yield from _walk(label + '!' + kind, inner, depth + 1, budget)
            return
        if kind == 'zip':
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    yield label + '!' + info.filename, info.filename
                    if info.is_dir():
                        continue
                    if info.file_size > MAX_BYTES:
                        raise ScanError('archive member size limit exceeded')
                    yield from _walk(label + '!' + info.filename, archive.read(info), depth + 1, budget)
            return
        if kind == 'tar':
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
                for member in archive:
                    yield label + '!' + member.name, member.name + '\n' + (member.linkname or '')
                    for key, value in (getattr(member, _T + "_headers") or {}).items():
                        yield label + '!' + member.name + '#header', f'{key} {value}'
                    if member.isfile():
                        if member.size > MAX_BYTES:
                            raise ScanError('archive member size limit exceeded')
                        yield from _walk(label + '!' + member.name, archive.extractfile(member).read(), depth + 1, budget)
                for key, value in (getattr(archive, _T + "_headers") or {}).items():
                    yield label + '#global-header', f'{key} {value}'
            return
        if kind == 'png':
            yield from _png(label, data)
            return
        if kind == 'pdf':
            yield from _pdf(label, data, depth, budget)
            return
    except ScanError:
        raise
    except (OSError, EOFError, ValueError, __import__('zlib').error, zipfile.BadZipFile, tarfile.TarError, lzma.LZMAError,
            zipfile.LargeZipFile, RuntimeError) as error:
        raise ScanError(f'corrupt or unreadable {kind} container') from error
    yield label, data.decode('utf-8', 'ignore')


def scan_bytes(rel_path, data, scoped=True):
    findings = [(rel_path, 0, 'private name in path')] if scan_text(rel_path) else []
    try:
        for label, text in _walk(rel_path, data, 0, [0]):
            findings += [(label, n, kind) for n, kind in scan_text(text, rel_path, scoped)]
    except ScanError as error:
        findings.append((rel_path, 0, str(error)))
    return findings


def _scan_files(root, paths, scoped=True):
    findings = []
    for rel in paths:
        full = Path(root, rel)
        if full.is_symlink():
            findings += [(rel, 0, kind) for _, kind in scan_text(rel + '\n' + os.readlink(full))]
            continue
        if not full.is_file():
            continue
        findings += scan_bytes(rel, full.read_bytes(), scoped)
    return findings


def scan_repo(root='.'):
    files = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True, check=True).stdout
    return _scan_files(root, [p for p in files.decode().split('\0') if p])


def scan_dir(root):
    paths = []
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            full = os.path.join(base, name)
            if name in files or os.path.islink(full):
                paths.append(os.path.relpath(full, root))
    # Directory mode keeps exact-token allows but no private-path exemptions.
    return _scan_files(root, sorted(paths), scoped=False)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ['--dir']:
        if len(args) != 2 or not os.path.isdir(args[1]):
            print('usage: check_privacy.py --dir DIR', file=sys.stderr)
            return 2
        findings = scan_dir(args[1])
    else:
        findings = scan_repo(args[0] if args else '.')
    for path, line, kind in findings:
        print(f'{path}:{line}: {kind}')
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
