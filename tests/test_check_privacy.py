import gzip, io, os, shutil, subprocess, sys, tarfile, tempfile, unittest, zipfile
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import check_privacy as cp

A, B = 'pa' + 'x', 'mi' + 'mas'
NET = ('box.tail1.ts' + '.net', 'Tail' + 'scale', 'https://github.' + 'rpi.edu/x')
PATHS = ('/ho' + 'me/someone/data', '/scr' + 'atch/someone', '/gp' + 'fs/u/x', '/Us' + 'ers/someone/x')
HAVE_POPPLER = bool(shutil.which('pdftotext') and shutil.which('pdfdetach'))


def make_pdf(page_text, attachment_name=None, attachment=b''):
    """Hand-built PDF: one text page, optional embedded-file attachment."""
    stream = b'BT /F1 12 Tf 72 720 Td (' + page_text.encode() + b') Tj ET'
    objs = [b'<< /Type /Catalog /Pages 2 0 R' + (b' /Names << /EmbeddedFiles << /Names [(' +
            attachment_name.encode() + b') 6 0 R] >> >>' if attachment_name else b'') + b' >>',
            b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
            b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R'
            b' /Resources << /Font << /F1 5 0 R >> >> >>',
            b'<< /Length %d >>\nstream\n' % len(stream) + stream + b'\nendstream',
            b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    if attachment_name:
        objs += [b'<< /Type /Filespec /F (' + attachment_name.encode() + b') /UF (' + attachment_name.encode() +
                 b') /EF << /F 7 0 R >> >>',
                 b'<< /Type /EmbeddedFile /Length %d >>\nstream\n' % len(attachment) + attachment + b'\nendstream']
    out = io.BytesIO(); out.write(b'%PDF-1.4\n'); offsets = []
    for number, body in enumerate(objs, 1):
        offsets.append(out.tell()); out.write(b'%d 0 obj\n' % number + body + b'\nendobj\n')
    xref = out.tell()
    out.write(b'xref\n0 %d\n0000000000 65535 f \n' % (len(objs) + 1))
    for offset in offsets:
        out.write(b'%010d 00000 n \n' % offset)
    out.write(b'trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (len(objs) + 1, xref))
    return out.getvalue()


def zipped(name, data):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(name, data)
    return buffer.getvalue()


def tarred(name, data, headers=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=getattr(tarfile, A.upper() + '_FORMAT'), **{A + '_headers': headers or {}}) as archive:
        info = tarfile.TarInfo(name); info.size = len(data); archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class TextRulesTest(unittest.TestCase):
    def test_hosts_any_case_and_embedded(self):
        for text in (A, B.upper(), 'run_' + A.title() + 'Pilot', 'members-' + A + '.json', 'x' + B + '1000'):
            self.assertTrue(cp.scan_text(text), text)

    def test_network_and_private_paths(self):
        for text in NET + PATHS:
            self.assertTrue(cp.scan_text(text), text)

    def test_path_fixture_relaxes_only_path_rule(self):
        fixture = next(iter(cp.PATH_RULE_EXEMPT))
        for text in PATHS:
            self.assertFalse(cp.scan_text(text, fixture), text)
        for text in NET + (B, A.upper()):
            self.assertTrue(cp.scan_text(text, fixture), text)
        self.assertTrue(cp.scan_text(PATHS[0]), 'exemption must not apply elsewhere')

    def test_tarfile_tokens_exact_and_path_scoped(self):
        scoped = 'tests/test_hpc_density_snapshot.py'
        line = 'tarfile.open(x, format=tarfile.' + A.upper() + '_FORMAT, ' + A + '_headers=h)'
        self.assertFalse(cp.scan_text(line, scoped))
        self.assertTrue(cp.scan_text(line), 'not allowed outside listed file')
        self.assertTrue(cp.scan_text('archive.' + A + '_headersX', scoped))
        self.assertTrue(cp.scan_text('x' + A + '_headers=h', scoped))
        self.assertTrue(cp.scan_text('tarfile.' + A.upper() + '_FORMAT ' + B, scoped))


def png(chunks):
    import struct, zlib
    out = b'\x89PNG\r\n\x1a\n'
    for ctype, body in chunks + [(b'IEND', b'')]:
        out += struct.pack('>I', len(body)) + ctype + body + struct.pack('>I', zlib.crc32(ctype + body))
    return out


class ContainerTest(unittest.TestCase):
    def test_png_text_chunks_scanned_image_bytes_not(self):
        import zlib
        self.assertFalse(self.kinds(png([(b'IDAT', b'\x00' + A.encode() + b'\x07')])))
        self.assertTrue(self.kinds(png([(b'tEXt', b'Source\x00' + PATHS[0].encode())])))
        self.assertTrue(self.kinds(png([(b'zTXt', b'Comment\x00\x00' + zlib.compress(B.upper().encode()))])))
        self.assertTrue(self.kinds(png([(b'tEXt', b'a\x00b')])[:-6]))


    def kinds(self, data, name='blob'):
        return [kind for _, _, kind in cp.scan_bytes(name, data)]

    def test_gzip_office_xml(self):
        docx = zipped('word/document.xml', '<w:t>' + B.title() + '</w:t>')
        self.assertIn('private host name', self.kinds(gzip.compress(docx)))
        self.assertFalse(self.kinds(gzip.compress(zipped('word/document.xml', '<w:t>clean</w:t>'))))

    def test_tar_member_names_bytes_and_headers(self):
        self.assertIn('private path', self.kinds(gzip.compress(tarred('ok.txt', PATHS[3].encode()))))
        self.assertTrue(self.kinds(tarred('site/' + A.upper() + '.html', b'clean')))
        self.assertTrue(self.kinds(tarred('ok.txt', b'clean', {'comment': B})))
        self.assertFalse(self.kinds(tarred('ok.txt', b'clean', {'comment': 'abc123'})))

    def test_malformed_containers_fail_closed(self):
        self.assertTrue(self.kinds(b'\x1f\x8b\x08\x00garbage'))
        self.assertTrue(self.kinds(b'PK\x03\x04garbage-not-a-zip'))
        self.assertTrue(self.kinds(gzip.compress(b'PK\x03\x04broken')))

    def test_nesting_limit_fails_closed(self):
        data = b'clean'
        for _ in range(cp.MAX_DEPTH + 2):
            data = gzip.compress(data)
        self.assertTrue(any('limit' in k for k in self.kinds(data)))

    def test_pdf_failure_fails_closed(self):
        with mock.patch.object(cp.subprocess, 'run', side_effect=OSError('missing')):
            self.assertTrue(any('PDF' in k for k in self.kinds(make_pdf('clean'))))

    @unittest.skipUnless(HAVE_POPPLER, 'poppler-utils not installed')
    def test_pdf_clean_page_with_dirty_attachment(self):
        self.assertFalse(self.kinds(make_pdf('clean page')))
        self.assertTrue(self.kinds(make_pdf('clean page', 'notes.txt', ('run on ' + B.upper() + '-node').encode())))
        self.assertTrue(self.kinds(make_pdf('clean page', 'x' + A.title() + '.txt', b'clean')))
        nested = zipped('word/document.xml', '<w:t>' + NET[0] + '</w:t>')
        self.assertTrue(self.kinds(make_pdf('clean page', 'r.docx', nested)))
        self.assertTrue(self.kinds(gzip.compress(make_pdf('clean', 'a.txt', PATHS[0].encode())), 'r.pdf.gz'))

    @unittest.skipUnless(HAVE_POPPLER, 'poppler-utils not installed')
    def test_pdf_page_text(self):
        self.assertTrue(self.kinds(make_pdf('host ' + A.upper() + '1')))


class VendoredTokenTest(unittest.TestCase):
    def test_plotly_exact_tokens_only(self):
        path = 'assets/plotly.min.js'
        self.assertFalse(cp.scan_text('il.maskMinOpp' + 'AxisClassName,u.d2pOpp' + 'Axis=1', path))
        self.assertFalse(cp.scan_text('"Muan","Pa' + 'x","Kayab"', path))
        self.assertTrue(cp.scan_text('"Pa' + 'x"', path))
        self.assertTrue(cp.scan_text('host=' + A + '-1', path))
        self.assertTrue(cp.scan_text('maskOpp' + 'Axis', 'other.js'))


class ModesTest(unittest.TestCase):
    def test_directory_mode_scans_untracked_and_has_no_exemptions(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = Path(root, next(iter(cp.PATH_RULE_EXEMPT)))
            fixture.parent.mkdir(parents=True)
            fixture.write_text(PATHS[0])
            Path(root, 'index.html').write_text('clean')
            self.assertEqual([k for _, _, k in cp.scan_dir(root)], ['private path'])
            os.symlink('/nonexistent/' + B, os.path.join(root, 'link'))
            self.assertIn('private host name', [k for _, _, k in cp.scan_dir(root)])
            with mock.patch('sys.stdout', io.StringIO()):
                self.assertEqual(cp.main(['--dir', root]), 1)

    def test_repository_is_clean(self):
        self.assertEqual(cp.scan_repo(str(Path(__file__).resolve().parents[1])), [])


if __name__ == '__main__':
    unittest.main()
