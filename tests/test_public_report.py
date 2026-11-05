import pathlib
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from public_report import prepare
from snapshot import sha256


class PublicReportTests(unittest.TestCase):
    def test_only_note_paths_change_and_scientific_metadata_and_citations_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);source=root/'native.pptx';output=root/'public.pptx'
            notes=b'<notes><text>Model m01; confidence 98.5; source /home/researcher/private/model.pdb; image /scratch/run/plot.png; https://example.org/home/annotation</text></notes>'
            with zipfile.ZipFile(source,'w') as z:
                z.writestr('ppt/slides/slide1.xml',b'<slide>Unchanged plot</slide>')
                z.writestr('ppt/media/plot.png',b'Original plot bytes')
                z.writestr('ppt/notesSlides/notesSlide1.xml',notes)
            original=source.read_bytes();result=prepare(source,output,source_sha256=sha256(source))
            with zipfile.ZipFile(output) as z:
                self.assertEqual(z.read('ppt/slides/slide1.xml'),b'<slide>Unchanged plot</slide>')
                self.assertEqual(z.read('ppt/media/plot.png'),b'Original plot bytes')
                note=z.read('ppt/notesSlides/notesSlide1.xml').decode()
                self.assertNotIn('/home/researcher',note);self.assertNotIn('/scratch/run',note)
                self.assertIn('model.pdb',note);self.assertIn('confidence 98.5',note)
                self.assertIn('https://example.org/home/annotation',note)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(result['sha256'],sha256(output))
            with self.assertRaises(FileExistsError):prepare(source,output,source_sha256=sha256(source))

    def test_unreviewed_path_on_slide_or_changed_source_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);source=root/'native.pptx';output=root/'public.pptx'
            with zipfile.ZipFile(source,'w') as z:z.writestr('ppt/slides/slide1.xml',b'<slide>/home/researcher/private.pdb</slide>')
            with self.assertRaisesRegex(ValueError,'outside removable'):prepare(source,output,source_sha256=sha256(source))
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError,'changed'):prepare(source,output,source_sha256='a'*64)
            self.assertFalse(output.exists())
