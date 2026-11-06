import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import check_privacy as cp

A, B = 'pa' + 'x', 'mi' + 'mas'


class PrivacyCheckTest(unittest.TestCase):
    def test_hosts_any_case_and_embedded(self):
        for text in (A, B.upper(), 'run_' + A.title() + 'Pilot', 'members-' + A + '.json', 'x' + B + '1000'):
            self.assertTrue(cp.scan_text(text), text)

    def test_internal_hosts_and_home_paths(self):
        for text in ('box.tail1.ts' + '.net', '/ho' + 'me/someone/data', 'https://github.' + 'rpi.edu/x'):
            self.assertTrue(cp.scan_text(text), text)

    def test_tarfile_api_and_synthetic_fixture_paths_allowed(self):
        self.assertFalse(cp.scan_text('tarfile.' + A.upper() + '_FORMAT, ' + A + '_headers=h'))
        self.assertFalse(cp.scan_text('/ho' + 'me/private/native.py'))

    def test_allowed_token_does_not_hide_other_hit_on_line(self):
        self.assertTrue(cp.scan_text(A + '_headers ' + B))

    def test_path_fixture_still_checks_host_names(self):
        self.assertTrue(cp.scan_text(B, paths_ok=True))
        self.assertFalse(cp.scan_text('/ho' + 'me/researcher', paths_ok=True))

    def test_repository_is_clean(self):
        self.assertEqual(cp.scan_repo(str(Path(__file__).resolve().parents[1])), [])


if __name__ == '__main__':
    unittest.main()
