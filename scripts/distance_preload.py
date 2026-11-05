"""Bounded read-only cache warming overlapped with native input parsing."""
import json
from pathlib import Path
import time


def preload(target, destination):
    """No matrix interpretation or scientific validation; ordinary checks still run."""
    start=time.monotonic()
    intake=json.loads(Path(target['intake']).read_text())
    screening=json.loads(Path(target['screening']).read_text())
    directory=Path(screening['distogram_dir'])
    files=sorted(directory.glob('*.bin'))
    if directory.is_symlink() or len(files)!=len(intake['models']):
        raise ValueError('Distance preload does not match the frozen target count')
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('Distance preload requires regular native cache files')
    expected=sum(p.stat().st_size for p in files)
    if expected>6*1024**3:
        raise ValueError('Distance preload exceeds the approved cache budget')
    buffer=bytearray(1024**2);total=0
    for path in files:
        with path.open('rb',buffering=0) as stream:
            while True:
                count=stream.readinto(buffer)
                if not count:break
                total+=count
                if total>expected:raise ValueError('Distance files grew beyond the frozen preload budget')
    if total!=expected:raise ValueError('Distance files changed during preload')
    result=dict(status='read_only_preload_complete',models=len(files),bytes_read=total,
                read_buffer_bytes=len(buffer),elapsed_seconds=time.monotonic()-start,
                scientific_checks_bypassed=False)
    with Path(destination).open('x') as stream:json.dump(result,stream,indent=2)
    return result
