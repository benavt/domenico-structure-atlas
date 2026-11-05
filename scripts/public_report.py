"""Create a public PPTX copy without changing slides, figures or scientific IDs."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from snapshot import sha256


PRIVATE_PATH = re.compile(r'(?<![\w:/])(?:file://)?/(?:home|Users|scratch|tmp|dev/shm)/[^\s"<>]+')


def prepare(source, output, *, source_sha256):
    source, output = Path(source), Path(output)
    if sha256(source) != source_sha256:
        raise ValueError('Frozen report changed')
    changed, parts = [], []
    with zipfile.ZipFile(source) as archive:
        if sum(info.file_size for info in archive.infolist()) > 512*1024**2:
            raise ValueError('Report exceeds the public-copy memory budget')
        for info in archive.infolist():
            if Path(info.filename).is_absolute() or '..' in Path(info.filename).parts:
                raise ValueError('Unsafe PPTX member')
            data = archive.read(info)
            if info.filename.startswith('ppt/notesSlides/notesSlide') and info.filename.endswith('.xml'):
                notes = ET.fromstring(data); modified = False
                for node in notes.iter():
                    if node.text and PRIVATE_PATH.search(node.text):
                        node.text = PRIVATE_PATH.sub(lambda m: Path(m.group(0)).name, node.text)
                        modified = True
                if modified:
                    data = ET.tostring(notes, encoding='utf-8', xml_declaration=True)
                    changed.append(info.filename)
            if info.filename.endswith(('.xml','.rels','.svg','.json','.txt')) and PRIVATE_PATH.search(data.decode('utf-8')):
                raise ValueError('Private path outside removable speaker-note metadata: '+info.filename)
            parts.append((info,data))
    with output.open('xb') as stream, zipfile.ZipFile(stream,'w') as archive:
        for info,data in parts:archive.writestr(info,data)
    return dict(source_sha256=source_sha256,sha256=sha256(output),bytes=output.stat().st_size,
                modified_note_parts=changed,slides_and_figures_unchanged=True,
                transformation='Only local filesystem paths in speaker notes become basenames; scientific identities, numbers, hashes and citations are retained.')
