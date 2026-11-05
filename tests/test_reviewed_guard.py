"""Source-only guard loading; no subprocess, cgroup mutation or scientific work."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_reviewed_plan import reviewed_guard, verify_controls
from rbp_parity import record


class ReviewedGuardTests(unittest.TestCase):
    def fixture(self, root):
        module = root/'structure_clustering'/'resource_guard.py'
        module.parent.mkdir()
        module.write_text('from dataclasses import dataclass\n@dataclass\nclass Marker:\n    value: int = 7\n')
        entry = record(module)
        plan = dict(runtime_guard=entry, sources=dict(guard=dict(checkout=str(root))),
                    frozen_files={str(module):entry['sha256']})
        return plan, module

    def test_existing_plans_keep_their_installed_guard(self):
        marker = object()
        with patch('run_reviewed_plan.importlib.import_module',return_value=marker) as load:
            self.assertIs(reviewed_guard({}),marker)
            load.assert_called_once_with('structure_clustering.resource_guard')

    def test_explicit_guard_loads_without_rebinding_scientific_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, path = self.fixture(Path(tmp))
            before = {name:value for name,value in sys.modules.items()
                      if name.startswith(('src.','structure_clustering'))}
            loaded = reviewed_guard(plan)
            self.assertEqual(loaded.Marker().value,7)
            self.assertIs(reviewed_guard(plan),loaded)
            after = {name:value for name,value in sys.modules.items()
                     if name.startswith(('src.','structure_clustering'))}
            self.assertEqual(before,after)

    def test_changed_unfrozen_or_wrong_checkout_guard_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, path = self.fixture(Path(tmp))
            wrong = deepcopy(plan); wrong['sources']['guard']['checkout']=str(path.parent)
            with self.assertRaisesRegex(ValueError,'binding'):reviewed_guard(wrong)
            wrong = deepcopy(plan);wrong['frozen_files']={}
            with self.assertRaisesRegex(ValueError,'binding'):reviewed_guard(wrong)
            path.write_text('raise RuntimeError("must never execute")')
            with self.assertRaisesRegex(ValueError,'binding'):reviewed_guard(plan)

    def test_opt_in_mode_requires_its_reviewed_source(self):
        with self.assertRaisesRegex(ValueError,'separately reviewed'):
            reviewed_guard(dict(guard=dict(io_recovery=dict(run_id='attempt'))))

    def test_system_manager_job_still_requires_approved_unprivileged_uid(self):
        with tempfile.TemporaryDirectory() as tmp:
            group=Path(tmp)
            for name,value in [('memory.max','100'),('memory.swap.max','0'),('cpu.max','100000 100000')]:
                (group/name).write_text(value)
            resources=dict(maximum_memory_bytes=100,maximum_cpus=1,required_uid=1000)
            with patch('run_reviewed_plan.os.getuid',return_value=1000):verify_controls(group,resources)
            with patch('run_reviewed_plan.os.getuid',return_value=0):
                with self.assertRaisesRegex(ValueError,'unprivileged'):verify_controls(group,resources)
            resources['required_uid']=0
            with patch('run_reviewed_plan.os.getuid',return_value=0):
                with self.assertRaisesRegex(ValueError,'unprivileged'):verify_controls(group,resources)

if __name__=='__main__':unittest.main()
