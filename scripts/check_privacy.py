"""Fail if tracked files expose private machine names or private home paths.

Known private host names are matched case-insensitively anywhere, including
embedded forms (for example inside identifiers or file names).  Literals are
assembled from fragments so this file is not itself a match.  Tarfile API names
that merely share letters with a host name are allowed by exact token.
"""
import gzip, io, re, subprocess, sys, zipfile
from pathlib import Path

_HOSTS = ('pa' + 'x', 'mi' + 'mas')
HOST_RE = re.compile('|'.join(_HOSTS), re.IGNORECASE)
OTHER_RE = re.compile(r'\.ts' r'\.net\b|tail' r'scale|github\.' r'rpi\.edu|/(?:home|scratch|gpfs)/(?!private/|annotation\b)[A-Za-z][\w.-]*',
                      re.IGNORECASE)
# Python tarfile API identifiers, not host names.
ALLOWED_TOKENS = ('PA' + 'X_FORMAT', 'pa' + 'x_headers')
ALLOWED_RE = re.compile('|'.join(map(re.escape, ALLOWED_TOKENS)))
# Explicit paths whose synthetic placeholder home/scratch paths test redaction.
# Host-name matching still applies to these files.
PATH_FIXTURES = frozenset({'tests/test_public_report.py'})


def scan_text(text, paths_ok=False):
    text = ALLOWED_RE.sub('', text)
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        if HOST_RE.search(line):
            hits.append((number, 'private host name'))
        elif not paths_ok and OTHER_RE.search(line):
            hits.append((number, 'internal host or private path'))
    return hits


def _texts(path, data):
    name = path.lower()
    if name.endswith('.gz'):
        try:
            data = gzip.decompress(data)
        except OSError:
            return
    if name.endswith(('.zip', '.docx', '.xlsx', '.pptx', '.odt', '.npz', '.whl')):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.namelist():
                    yield path + '!' + member, member
                    yield path + '!' + member, archive.read(member).decode('utf-8', 'ignore')
        except zipfile.BadZipFile:
            pass
        return
    if name.endswith('.pdf'):
        try:
            data = subprocess.run(['pdftotext', '-q', '-', '-'], input=data, capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            yield path, None
            return
    yield path, data.decode('utf-8', 'ignore')


def scan_repo(root='.'):
    files = subprocess.run(['git', '-C', root, 'ls-files', '-z'], capture_output=True, check=True).stdout.decode().split('\0')
    findings = []
    for path in filter(None, files):
        if scan_text(path):
            findings.append((path, 0, 'private name in path'))
        full = Path(root, path)
        if not full.is_file():
            continue
        for label, text in _texts(path, full.read_bytes()):
            if text is None:
                findings.append((label, 0, 'unscannable PDF (install poppler-utils)'))
                continue
            findings += [(label, n, kind) for n, kind in scan_text(text, path in PATH_FIXTURES)]
    return findings


def main(argv=None):
    findings = scan_repo((argv or sys.argv[1:] or ['.'])[0])
    for path, line, kind in findings:
        print(f'{path}:{line}: {kind}')
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
