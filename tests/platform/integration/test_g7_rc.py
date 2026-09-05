"""G7 release-candidate scenario: one-click, offline core, backup, V2, isolation."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
DESKTOP_TOOLS = REPO_ROOT / "tools" / "desktop"
DAEMON_PATH = REPO_ROOT / "platform" / "worker" / "daemon.py"
STATE_GUARD = REPO_ROOT / "tools" / "platform" / "state_guard.py"
FIXTURES = REPO_ROOT / "platform" / "fixtures"
REQUIREMENTS = BACKEND_ROOT / "requirements.txt"
FRONTEND_SRC = REPO_ROOT / "platform" / "frontend" / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.desktop import launch  # noqa: E402
from tools.platform import install  # noqa: E402
from tools.platform import rollback as rollback_cli  # noqa: E402
from tools.platform import update as update_cli  # noqa: E402
from tools.platform import v2_migrate  # noqa: E402

MISSION_IDS = tuple(f"M{i:02d}" for i in range(1, 43))
V2_FREEZE_SHA = "35293455ff769855014a49fa56315b22829e74b1"
EXPERIMENT_TYPE = "experiment"
WP137_STATUSES = {"SUCCESS", "FAILED", "TIMEOUT", "CRASHED"}
PREDICT_BODY = {
    "hypothesis": "generic G7 RC execute succeeds on the frozen runtime",
    "expected_values": {"ok": True},
}
EXECUTE_BODY = {
    "code": (
        "payload = {'ok': True}\n"
        "print(payload)\n"
        "{'type': 'metric', 'title': 'ok', 'payload': payload}"
    ),
    "parameters": {},
}


class RecordingRunner:
    """In-process stand-in for venv/pip/npm so RC bootstrap stays offline."""

    def __init__(
        self,
        *,
        python_version: str = "Python 3.12.2",
        node_version: str = "v20.11.1",
        npm_version: str = "10.8.2",
    ) -> None:
        self.python_version = python_version
        self.node_version = node_version
        self.npm_version = npm_version
        self.calls: list[list[str]] = []
        self.pip_installs = 0
        self.npm_ci = 0
        self.venv_creates = 0

    def run(self, command: Sequence[str], **kwargs) -> subprocess.CompletedProcess[str]:
        argv = [str(part) for part in command]
        self.calls.append(argv)
        name = Path(argv[0]).name.lower()

        if argv[-1] == "--version" and "-m" not in argv:
            if "python" in name:
                return subprocess.CompletedProcess(argv, 0, self.python_version + "\n", "")
            if name == "node":
                return subprocess.CompletedProcess(argv, 0, self.node_version + "\n", "")
            if name.startswith("npm"):
                return subprocess.CompletedProcess(argv, 0, self.npm_version + "\n", "")

        if "-m" in argv and "venv" in argv:
            dest = Path(argv[argv.index("venv") + 1])
            bindir = dest / "bin"
            bindir.mkdir(parents=True, exist_ok=True)
            python = bindir / "python"
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
            (dest / "pyvenv.cfg").write_text(
                "home = /tmp\ninclude-system-site-packages = false\n", encoding="utf-8"
            )
            self.venv_creates += 1
            return subprocess.CompletedProcess(argv, 0, "", "")

        if "-m" in argv and "pip" in argv:
            if "--version" in argv:
                return subprocess.CompletedProcess(argv, 0, "pip 24.0\n", "")
            if "install" in argv:
                self.pip_installs += 1
                return subprocess.CompletedProcess(argv, 0, "", "")

        if "-m" in argv and "ensurepip" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")

        if "-c" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")

        if name.startswith("npm") and "ci" in argv:
            prefix = Path(argv[argv.index("--prefix") + 1]) if "--prefix" in argv else Path(kwargs["cwd"])
            vite = prefix / "node_modules" / ".bin" / "vite"
            vite.parent.mkdir(parents=True, exist_ok=True)
            vite.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            vite.chmod(0o755)
            self.npm_ci += 1
            return subprocess.CompletedProcess(argv, 0, "", "")

        return subprocess.CompletedProcess(argv, 0, "", "")


def _write_fake_repo(root: Path) -> None:
    backend = root / "platform" / "backend"
    frontend = root / "platform" / "frontend"
    worker = root / "platform" / "worker"
    tools_platform = root / "tools" / "platform"
    (backend / "app").mkdir(parents=True)
    frontend.mkdir(parents=True)
    worker.mkdir(parents=True)
    tools_platform.mkdir(parents=True)
    (root / "start.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (root / "start.sh").chmod(0o755)
    (backend / "requirements.txt").write_text("fastapi>=0.100.0\nuvicorn>=0.23.2\n", encoding="utf-8")
    (frontend / "package.json").write_text('{"name":"frontend","private":true}\n', encoding="utf-8")
    (frontend / "package-lock.json").write_text('{"name":"frontend","lockfileVersion":3}\n', encoding="utf-8")
    (backend / "app" / "main.py").write_text("# fake\n", encoding="utf-8")
    (worker / "daemon.py").write_text("# fake\n", encoding="utf-8")
    (tools_platform / "dev.py").write_text("# fake\n", encoding="utf-8")


def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _start_worker(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(DAEMON_PATH)],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_worker(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def _outside_repo(path: Path) -> None:
    with pytest.raises(ValueError):
        path.resolve().relative_to(REPO_ROOT.resolve())


def _skip_if_missing_commit(sha: str, label: str) -> None:
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"{label} {sha} is not in this clone (shallow checkout)")


def _bootstrap(client):
    boot = client.post("/api/v1/auth/bootstrap")
    assert boot.status_code == 200, boot.text
    token = boot.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _package_dir(mission_id: str) -> Path:
    return FIXTURES / mission_id


def _artifacts_for_stage(spec: dict, stage: dict) -> list[dict]:
    stage_id = stage["id"]
    contract = spec.get("gate_contract") if isinstance(spec.get("gate_contract"), dict) else {}
    required = contract.get("required_evidence") if isinstance(contract.get("required_evidence"), list) else []
    for item in required:
        if isinstance(item, dict) and item.get("stage_id") == stage_id and item.get("artifact_type"):
            return [{"artifact_type": item["artifact_type"]}]
    rubric = stage.get("validation_rubric") if isinstance(stage.get("validation_rubric"), dict) else {}
    required_type = rubric.get("required_evidence_type") or rubric.get("artifact_type")
    if isinstance(required_type, str) and required_type:
        return [{"artifact_type": required_type}]
    return [{"artifact_type": "markdown"}]


def _learner_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {str(row[0]) for row in conn.execute("SELECT id FROM learners")}
    finally:
        conn.close()


def _assert_state_guard_clean() -> None:
    completed = subprocess.run(
        [sys.executable, str(STATE_GUARD), "--repo", str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASSED" in completed.stdout
    assert not (REPO_ROOT / ".learningos").exists()
    assert not (REPO_ROOT / "learningos.db").exists()


@pytest.fixture
def g7_home(monkeypatch: pytest.MonkeyPatch):
    home = Path(tempfile.mkdtemp(prefix="los-g7-rc-home-", dir="/tmp"))
    sock = Path(f"/tmp/los-g7-rc-{uuid.uuid4().hex}.sock")
    sock.unlink(missing_ok=True)
    monkeypatch.setenv("LEARNINGOS_HOME", str(home))
    monkeypatch.setenv("LEARNINGOS_WORKER_SOCKET", str(sock))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-never-leak")
    monkeypatch.setenv("LEARNINGOS_USE_KEYCHAIN", "0")
    monkeypatch.delenv("LEARNINGOS_TUTOR_PROVIDER", raising=False)
    env = os.environ.copy()
    env["LEARNINGOS_HOME"] = str(home)
    env["LEARNINGOS_WORKER_SOCKET"] = str(sock)
    env["PYTHONUNBUFFERED"] = "1"
    env["LEARNINGOS_USE_KEYCHAIN"] = "0"
    env.pop("LEARNINGOS_TUTOR_PROVIDER", None)
    try:
        yield {"home": home, "sock": sock, "env": env, "monkeypatch": monkeypatch}
    finally:
        sock.unlink(missing_ok=True)
        shutil.rmtree(home, ignore_errors=True)


def test_g7_one_click_managed_runtime_and_second_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repo = Path(tempfile.mkdtemp(prefix="learningos-g7-rc-repo-", dir="/tmp"))
    data_root = Path(tempfile.mkdtemp(prefix="learningos-g7-rc-runtime-", dir="/tmp"))
    data_home = data_root / "home"
    data_home.mkdir()
    try:
        _write_fake_repo(fake_repo)
        first = install.bootstrap(repo_root=fake_repo, data_home=data_home, runner=RecordingRunner())
        runtime = data_home / "runtime"
        assert runtime.is_dir()
        assert first.python.is_file()
        assert first.python.is_relative_to(data_home.resolve())
        assert not first.python.is_relative_to(fake_repo.resolve())
        assert (runtime / "python" / "pyvenv.cfg").is_file()
        assert (runtime / "frontend" / "node_modules" / ".bin" / "vite").is_file()
        assert (runtime / "bootstrap.json").is_file()
        assert not (fake_repo / ".venv").exists()
        assert not (fake_repo / "learningos.db").exists()
        _outside_repo(runtime)

        second_runner = RecordingRunner()
        plan = install.prepare_launch(
            repo_root=fake_repo,
            data_home=data_home,
            start_args=("--smoke",),
            runner=second_runner,
        )
        assert plan.result.skipped_backend
        assert plan.result.skipped_frontend
        assert second_runner.pip_installs == 0
        assert second_runner.npm_ci == 0
        assert second_runner.venv_creates == 0
        assert Path(plan.command[1]).resolve() == (fake_repo / "start.sh").resolve()
        assert plan.command[-1] == "--smoke"
        assert plan.env["LEARNINGOS_HOME"] == str(data_home.resolve())
        assert Path(plan.env["LEARNINGOS_PYTHON"]).is_relative_to(data_home.resolve())

        recorded: dict[str, object] = {}

        def fake_exec(program: str, command, env) -> None:
            recorded["program"] = program
            recorded["command"] = list(command)
            recorded["env"] = dict(env)

        monkeypatch.setenv("LEARNINGOS_REPO_ROOT", str(fake_repo))
        monkeypatch.setenv("LEARNINGOS_HOME", str(data_home))
        code = launch.main(["--", "--smoke"], runner=RecordingRunner(), executor=fake_exec)
        assert code == 0
        assert str(recorded["command"][1]).endswith("start.sh")
        assert recorded["command"][-1] == "--smoke"
        launch_env = recorded["env"]
        assert isinstance(launch_env, dict)
        assert launch_env["LEARNINGOS_HOME"] == str(data_home.resolve())
        assert Path(launch_env["LEARNINGOS_PYTHON"]).is_relative_to(data_home.resolve())

        help_text = subprocess.run(
            [sys.executable, str(DESKTOP_TOOLS / "launch.py"), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert help_text.returncode == 0, help_text.stderr
        assert "python3 tools/desktop/launch.py" in help_text.stdout or "start.sh" in help_text.stdout
    finally:
        shutil.rmtree(fake_repo, ignore_errors=True)
        shutil.rmtree(data_root, ignore_errors=True)


def test_g7_m01_m42_packages_load_execute_gate_and_tutor_501(g7_home) -> None:
    from fastapi.testclient import TestClient

    from app.core.mission_loader import load_package
    from app.core.secrets import verify_package_checksums
    from app.main import app

    home: Path = g7_home["home"]
    sock: Path = g7_home["sock"]
    env: dict[str, str] = g7_home["env"]
    _outside_repo(home)

    for mission_id in MISSION_IDS:
        package_dir = _package_dir(mission_id)
        assert package_dir.is_dir(), mission_id
        assert (package_dir / "SHA256SUMS").is_file(), mission_id
        listed = verify_package_checksums(package_dir, required=True)
        assert listed
        loaded = load_package(package_dir)
        assert loaded.id
        assert loaded.digest
        assert any(spec.get("id") == mission_id for spec in loaded.missions)

    worker = _start_worker(env)
    try:
        assert _wait_until(lambda: sock.exists()), f"worker socket was not created at {sock}"
        _outside_repo(sock)
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            headers = _bootstrap(client)
            loaded_ids: list[str] = []
            for mission_id in MISSION_IDS:
                response = client.post(
                    "/api/v1/curriculum/packages/load",
                    json={"package_dir": str(_package_dir(mission_id))},
                    headers=headers,
                )
                assert response.status_code == 200, f"{mission_id}: {response.text}"
                loaded_ids.append(response.json()["id"])
            assert len(loaded_ids) == 42

            missions = client.get("/api/v1/missions", headers=headers)
            assert missions.status_code == 200, missions.text
            listed = [item.get("id") for item in missions.json().get("missions", [])]
            assert set(MISSION_IDS) <= set(listed)

            missing_tutor = client.post(
                "/api/v1/tutor/chat",
                json={
                    "session_id": "sess-missing",
                    "stage_id": "stage-1",
                    "role": "SOCRATIC",
                    "prompt": "help",
                },
                headers=headers,
            )
            assert missing_tutor.status_code == 501, missing_tutor.text
            assert missing_tutor.json()["error"]["code"] == "TUTOR_NOT_AVAILABLE"
            assert "openai" not in sys.modules
            assert "sk-test-should-never-leak" not in missing_tutor.text

            config = client.get("/api/v1/system/config")
            assert config.status_code == 200, config.text
            body = config.json()
            assert "OPENAI_API_KEY" not in config.text
            assert "sk-test-should-never-leak" not in config.text
            for key in ("data_home", "database_path", "worker_socket", "bind_host", "api_prefix"):
                assert key in body

            learner = client.post(
                "/api/v1/learners",
                json={"username": "g7-rc", "display_name": "G7 RC"},
                headers=headers,
            )
            assert learner.status_code == 200, learner.text
            learner_id = learner.json()["learner_id"]
            created = client.post(
                "/api/v1/sessions",
                json={"mission_id": "M01", "learner_id": learner_id},
                headers=headers,
            )
            assert created.status_code == 200, created.text
            session_id = created.json()["session_id"]

            spec_resp = client.get("/api/v1/missions/M01", headers=headers)
            assert spec_resp.status_code == 200, spec_resp.text
            spec = spec_resp.json()
            executed = False
            for stage in spec["stages"]:
                stage_id = stage["id"]
                entered = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{stage_id}/enter",
                    headers=headers,
                )
                assert entered.status_code == 200, f"{stage_id}: {entered.text}"
                if stage.get("type") == EXPERIMENT_TYPE:
                    predicted = client.post(
                        f"/api/v1/sessions/{session_id}/stages/{stage_id}/predict",
                        json=PREDICT_BODY,
                        headers=headers,
                    )
                    assert predicted.status_code == 200, predicted.text
                    executed_resp = client.post(
                        f"/api/v1/sessions/{session_id}/stages/{stage_id}/execute",
                        json=EXECUTE_BODY,
                        headers=headers,
                    )
                    assert executed_resp.status_code == 200, executed_resp.text
                    exec_body = executed_resp.json()
                    assert exec_body["status"] == "SUCCESS", exec_body
                    structured = exec_body.get("structured_result") or {}
                    assert structured.get("status") in WP137_STATUSES
                    assert structured.get("status") == "SUCCESS"
                    executed = True
                submitted = client.post(
                    f"/api/v1/sessions/{session_id}/stages/{stage_id}/submit",
                    json={
                        "explanation": f"completed {stage_id} on generic G7 runtime",
                        "artifacts": _artifacts_for_stage(spec, stage),
                    },
                    headers=headers,
                )
                assert submitted.status_code == 200, f"{stage_id}: {submitted.text}"

            assert executed
            gate = client.post(f"/api/v1/sessions/{session_id}/gates/evaluate", headers=headers)
            assert gate.status_code == 200, gate.text
            assert gate.json()["status"] == "PASSED", gate.json()
            final = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert final.status_code == 200, final.text
            assert final.json()["status"] == "COMPLETED"
            assert "openai" not in REQUIREMENTS.read_text(encoding="utf-8").lower()
    finally:
        _stop_worker(worker)


def test_g7_backup_rollback_into_clean_dest_home(g7_home) -> None:
    from app.core.artifact_store import ArtifactStore
    from app.db.database import get_connection, init_db
    from app.db.ledger import EventLedger

    home: Path = g7_home["home"]
    dest = Path(tempfile.mkdtemp(prefix="los-g7-rc-dest-", dir="/tmp"))
    try:
        init_db()
        (home / "config.json").write_text(json.dumps({"theme": "pre-update"}), encoding="utf-8")
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
                ("learner_pre", "preuser", "Pre User"),
            )
            conn.commit()
            EventLedger(conn).append("learner_pre", "note", {"text": "before-update"})
        finally:
            conn.close()
        digest = ArtifactStore().put(b"pre-update-blob", media_type="text/plain")

        update = update_cli.apply_update(home=home, simulate=True)
        assert update.backup.is_file()
        assert update.backup.name.startswith("backup_pre_update_")
        assert update_cli.POST_UPDATE_LEARNER_ID in _learner_ids(home / "learningos.db")

        with pytest.raises(rollback_cli.RollbackError, match="live data home"):
            rollback_cli.rollback_to_dest(home, live_home=home)

        result = rollback_cli.rollback_to_dest(dest, live_home=home)
        assert (dest / "learningos.db").is_file()
        restored = json.loads((dest / "config.json").read_text(encoding="utf-8"))
        assert restored["theme"] == "pre-update"
        assert _learner_ids(dest / "learningos.db") == {"learner_pre"}
        assert (dest / "artifacts" / "sha256" / digest[:2] / digest[2:]).is_file()
        assert "export LEARNINGOS_HOME=" in result.instructions
        assert "./start.sh" in result.instructions
        _outside_repo(dest)
        _assert_state_guard_clean()
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def test_g7_v2_dry_run_additive_import_and_tag_peel(g7_home, tmp_path: Path) -> None:
    populated = REPO_ROOT / "tests" / "platform" / "g7" / "v2_migration" / "fixtures" / "populated"
    dest = tmp_path / "v2-dest"
    dest.mkdir()
    marker = dest / "keep-me.txt"
    marker.write_text("untouched\n", encoding="utf-8")
    before = sorted((path.relative_to(dest).as_posix(), path.stat().st_size) for path in dest.rglob("*"))

    code = v2_migrate.main(["--dry-run", "--source", str(populated), "--home", str(dest)])
    assert code == 0
    assert not (dest / "learningos.db").exists()
    after = sorted((path.relative_to(dest).as_posix(), path.stat().st_size) for path in dest.rglob("*"))
    assert after == before
    assert marker.read_text(encoding="utf-8") == "untouched\n"

    home: Path = g7_home["home"]
    from app.db.database import get_connection, init_db

    init_db()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO learners (id, username, display_name) VALUES (?, ?, ?)",
            ("learner_v3", "alice", "Alice Original"),
        )
        conn.commit()
    finally:
        conn.close()

    result = v2_migrate.import_source(populated, home, force=False, repo_root=REPO_ROOT)
    assert result.id_map["default"] == "v2:default"
    conn = sqlite3.connect(str(home / "learningos.db"))
    conn.row_factory = sqlite3.Row
    try:
        rows = {row["id"]: row for row in conn.execute("SELECT id, username, display_name FROM learners")}
        assert rows["learner_v3"]["display_name"] == "Alice Original"
        assert "v2:default" in rows
    finally:
        conn.close()
    _outside_repo(home)
    _assert_state_guard_clean()

    _skip_if_missing_commit(V2_FREEZE_SHA, "v2 freeze")
    peeled = subprocess.run(
        ["git", "rev-parse", f"{v2_migrate.V2_FREEZE_TAG}^{{commit}}"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if peeled.returncode != 0:
        pytest.skip("v2-prototype-freeze tag is not in this clone (shallow checkout)")
    assert peeled.stdout.strip() == V2_FREEZE_SHA
    freeze_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", V2_FREEZE_SHA, "HEAD"],
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert freeze_is_ancestor.returncode != 0


def test_g7_worker_socket_outside_worktree_and_execute(g7_home) -> None:
    from app.core.config import resolve_worker_socket
    from app.core.worker_client import WorkerClient

    home: Path = g7_home["home"]
    sock: Path = g7_home["sock"]
    env: dict[str, str] = g7_home["env"]
    resolved = resolve_worker_socket(home)
    assert resolved == sock.resolve()
    _outside_repo(resolved)

    proc = _start_worker(env)
    client = WorkerClient(sock)
    try:
        assert _wait_until(client.health), "daemon did not become healthy"
        result = client.execute(
            "print('g7-rc')",
            {"limits": {"timeout_sec": 5, "memory_mb": 256}},
        )
        assert result.get("status") == "SUCCESS", result
        assert "g7-rc" in (result.get("stdout") or "")
        client.shutdown()
        assert _wait_until(lambda: proc.poll() is not None)
    finally:
        _stop_worker(proc)


def test_g7_docs_name_real_one_click_and_omit_stale_paths() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    learner = (REPO_ROOT / "docs" / "LEARNER.md").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "docs" / "RUNTIME_QUICKSTART.md").read_text(encoding="utf-8")
    docs_index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    combined = "\n".join([readme, learner, quickstart, docs_index])
    assert "python3 tools/desktop/launch.py" in readme
    assert "python3 tools/desktop/launch.py" in learner
    assert "tools/platform/install.py" in readme
    assert "tools/platform/install.py" in combined
    assert "worker_daemon" not in combined
    assert "platform/worker/daemon.py" in readme
    assert re.search(r"jupyter is not required", combined, re.IGNORECASE)
    blocks = re.findall(r"```(?:bash)?\n(.*?)```", readme, re.DOTALL)
    assert "tools/desktop/launch.py" in blocks[0]
    assert "pip " not in blocks[0]
    assert "npm " not in blocks[0]
    openai = "_".join(("OPENAI", "API", "KEY"))
    vite = "VITE" + "_"
    hits: list[str] = []
    skip_dirs = {"node_modules", "dist", "__pycache__"}
    for path in FRONTEND_SRC.rglob("*"):
        if not path.is_file() or any(part in skip_dirs for part in path.parts):
            continue
        if path.name.endswith((".test.ts", ".test.tsx")):
            continue
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        if openai in text or vite in text:
            hits.append(path.relative_to(REPO_ROOT).as_posix())
    assert hits == []
    _assert_state_guard_clean()
