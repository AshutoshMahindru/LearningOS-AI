from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "platform"))

import dev  # noqa: E402
import state_guard  # noqa: E402


class DataHomeTests(unittest.TestCase):
    def test_rejects_repository_and_descendants(self) -> None:
        self.assertFalse(dev.validate_data_home(REPO_ROOT, REPO_ROOT).passed)
        self.assertFalse(dev.validate_data_home(REPO_ROOT / ".learningos", REPO_ROOT).passed)

    def test_accepts_writable_external_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            external = Path(temporary_directory) / "learningos-home"
            result = dev.validate_data_home(external, REPO_ROOT)
        self.assertTrue(result.passed, result.detail)

    def test_extracts_major_tool_version(self) -> None:
        self.assertEqual(dev._major_version("v24.19.0"), 24)
        self.assertEqual(dev._major_version("Python 3.12.2"), 3)
        self.assertIsNone(dev._major_version("not installed"))


class LearnerStateGuardTests(unittest.TestCase):
    def test_recognizes_mutable_state_paths(self) -> None:
        self.assertTrue(state_guard.is_learner_state_path(Path("learningos.db")))
        self.assertTrue(state_guard.is_learner_state_path(Path("platform/backend/.learningos/config.json")))
        self.assertTrue(state_guard.is_learner_state_path(Path("platform/backend/artifacts/sha256/ab/file")))
        self.assertFalse(state_guard.is_learner_state_path(Path("tracking/learner_state.json")))
        self.assertFalse(state_guard.is_learner_state_path(Path("platform/backend/app/core/artifact_store.py")))

    def test_detects_untracked_database_in_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "platform").mkdir()
            database = root / "platform" / "learningos.db"
            database.touch()
            violations = state_guard.filesystem_violations(root)
        self.assertEqual(violations, [Path("platform/learningos.db")])


class SupervisorTests(unittest.TestCase):
    def test_default_specs_use_frozen_endpoints_and_canonical_worker(self) -> None:
        args = dev.parse_args([])
        specs = {spec.name: spec for spec in dev.service_specs(args, sys.executable)}
        self.assertEqual(set(specs), {"worker", "backend", "frontend"})
        self.assertIn("platform/worker/daemon.py", specs["worker"].command[-1])
        self.assertIn("8765", specs["backend"].command)
        self.assertIn("5173", specs["frontend"].command)

    @unittest.skipIf(os.name == "nt", "POSIX process-group assertion")
    def test_shutdown_reaps_owned_process_group(self) -> None:
        supervisor = dev.Supervisor(environment=os.environ.copy(), grace_seconds=1)
        managed = supervisor.start(
            dev.ServiceSpec(
                "test-child",
                (sys.executable, "-c", "import time; time.sleep(60)"),
                REPO_ROOT,
            )
        )
        time.sleep(0.1)
        self.assertIsNone(managed.process.poll())
        supervisor.shutdown()
        self.assertIsNotNone(managed.process.poll())

    def test_command_override_does_not_use_a_shell(self) -> None:
        variable = "LEARNINGOS_TEST_COMMAND"
        previous = os.environ.get(variable)
        os.environ[variable] = f'{sys.executable} -c "print(123)"'
        try:
            command = dev.command_override(variable, ("unused",))
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout.strip(), "123")


if __name__ == "__main__":
    unittest.main()
