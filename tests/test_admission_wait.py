"""Bounded readiness wait uses fake time and unchanged pressure decisions; no jobs."""
import json
from pathlib import Path
import sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pax_pilot_guard as subject
import mimas_cpu as cpu


class WaitTests(unittest.TestCase):
    def fixture(self,root):
        return dict(phase='curve1000',run_root=str(root/'run'),guard_output=str(root/'controls/guard'),runtime_prefix=str(root/'runtime'),
            admission_wait=dict(directory=str(root/'controls/wait'),maximum_seconds=30,sample_seconds=5,quiet_samples=2))
    def test_waits_for_consecutive_quiet_samples_without_relaxing_limits(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'controls').mkdir();p=self.fixture(root);now=[0];reasons=iter([['busy'],[],[],[]]);seen=[];limit=object()
            def violation(a,b,actual):seen.append(actual);return next(reasons)
            guard=SimpleNamespace(linux_snapshot=lambda:dict(time=now[0]),violations=violation)
            with patch.object(subject,'_modules',return_value=(cpu,None)):
                r=subject.wait_for_quiet(p,guard,limit,clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n))
            self.assertTrue(r);self.assertEqual(now[0],15);self.assertTrue(all(x is limit for x in seen))
            self.assertFalse((root/'run').exists());self.assertFalse((root/'controls/guard').exists())
            d=json.loads((root/'controls/wait/result.json').read_text());self.assertEqual(d['projection_fits'],0)
    def test_timeout_is_bounded_and_does_not_launch(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'controls').mkdir();p=self.fixture(root);now=[0]
            guard=SimpleNamespace(linux_snapshot=lambda:{},violations=lambda *a:['busy'])
            with patch.object(subject,'_modules',return_value=(cpu,None)):
                r=subject.wait_for_quiet(p,guard,None,clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n))
            self.assertFalse(r);self.assertEqual(now[0],30);self.assertFalse((root/'run').exists())
            self.assertEqual(json.loads((root/'controls/wait/result.json').read_text())['status'],'admission_wait_timeout_no_launch')
    def test_destination_needs_a_closed_reference_before_any_fit(self):
        import reducer_diagnostic as diag
        with tempfile.TemporaryDirectory()as tmp:
            plan=dict(phase='curve1000',execution_host='mimas',diagnostic_profile='native',pax1000_gate=str(Path(tmp)/'missing.json'))
            with self.assertRaises(FileNotFoundError):diag.verify_predecessor(plan)

    def test_bad_scope_or_unbounded_wait_fails_before_evidence(self):
        for phase,seconds in [('curve64',30),('curve1000',1801)]:
            with tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);p=self.fixture(root);p['phase']=phase;p['admission_wait']['maximum_seconds']=seconds
                with self.assertRaises(ValueError):subject.wait_for_quiet(p,None,None)
                self.assertFalse((root/'controls').exists())


if __name__=='__main__':unittest.main()
