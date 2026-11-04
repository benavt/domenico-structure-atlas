import importlib.util
from pathlib import Path
import sys

import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import confidence


def entry(method, kind):
    return dict(model_id='T__'+method+'__0', target='T', method=method, kind=kind,
                coordinate_sha256='a'*64, coordinate_source='/native/model.pdb', files=[])


def score_entry(method, kind, payload):
    row = entry(method, kind)
    row['files'] = [dict(role='native_scores', bytes=1, sha256='b'*64)]
    return confidence.extract_model(row, lambda _: payload)


def check_af2_monomer_ranking_is_plddt_and_not_multimer_ranking():
    result = score_entry('afsample2', 'af2_monomer_scores', {'ranking_confidence': 82.5, 'ptm': .7})
    assert result['mean_plddt'] == 82.5 and result['ptm'] == .7
    assert 'iptm' not in result and 'ranking_confidence' not in result


def check_boltz_rescaling_and_opendde_omit_monomer_placeholder_iptm():
    result = score_entry('boltz2', 'boltz2_native_scores', {'complex_plddt': .92, 'ptm': .8, 'iptm': 0})
    assert result['mean_plddt'] == 92
    assert 'iptm' not in result
    result = score_entry('opendde', 'opendde_native_scores', {'plddt': 91, 'ptm': .75, 'iptm': 0})
    assert result['mean_plddt'] == 91 and 'iptm' not in result


def check_unscored_models_remain_unscored():
    for method in ('bioemu', 'bioemu2'):
        result = confidence.extract_model(entry(method, 'unscored_no_native_confidence'), lambda _: unittest.TestCase().fail('No score read'))
        assert result['mean_plddt'] is None and result['ptm'] is None and not result['eligible']


def check_nested_scores_require_exact_native_coordinate_binding():
    payload = dict(status='validated', method='afcluster', pdb='/native/model.pdb', pdb_sha256='a'*64,
                   scores={'mean_plddt': 90, 'ptm': .8})
    assert score_entry('afcluster', 'native_nested_scores', payload)['mean_plddt'] == 90
    payload['pdb_sha256'] = 'c'*64
    with unittest.TestCase().assertRaisesRegex(ValueError, 'different coordinates'):
        score_entry('afcluster', 'native_nested_scores', payload)


def check_af3_averages_native_atoms_and_requires_single_chain():
    row = entry('afsample3', 'af3_summary_and_atom_confidence')
    row['files'] = [dict(role=role, bytes=1, sha256='b'*64) for role in ['native_summary', 'native_atom_confidence']]
    data = dict(native_summary={'chain_ptm': [.8], 'ptm': .8, 'iptm': None}, native_atom_confidence={'atom_plddts': [80, 90, 100]})
    result = confidence.extract_model(row, lambda f: data[f['role']])
    assert result['mean_plddt'] == 90
    data['native_summary']['chain_ptm'] = [.8, .9]
    with unittest.TestCase().assertRaisesRegex(ValueError, 'single-chain'):
        confidence.extract_model(row, lambda f: data[f['role']])


def check_invalid_confidence_never_becomes_fabricated_score(value):
    with unittest.TestCase().assertRaises(ValueError):
        score_entry('afsample2', 'af2_monomer_scores', {'ranking_confidence': value, 'ptm': .7})


class ConfidenceTests(unittest.TestCase):
    def test_af2(self): check_af2_monomer_ranking_is_plddt_and_not_multimer_ranking()
    def test_boltz_opendde(self): check_boltz_rescaling_and_opendde_omit_monomer_placeholder_iptm()
    def test_unscored(self): check_unscored_models_remain_unscored()
    def test_exact_binding(self): check_nested_scores_require_exact_native_coordinate_binding()
    def test_af3(self): check_af3_averages_native_atoms_and_requires_single_chain()
    def test_invalid_scores(self):
        for value in [-1, 101, True, float('nan'), float('inf'), '90']:
            with self.subTest(value=value): check_invalid_confidence_never_becomes_fabricated_score(value)
