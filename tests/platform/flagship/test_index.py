from __future__ import annotations

import hashlib
import json
import re

from tests.platform.flagship.conftest import FLAGSHIP_FIXTURE

VERSION_IDS = [f"V{i:02d}" for i in range(0, 13)]
MISSION_IDS = [f"M{i:02d}" for i in range(1, 43)]


def _index() -> dict:
    return json.loads((FLAGSHIP_FIXTURE / "index.json").read_text(encoding="utf-8"))


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_layout_and_digest() -> None:
    assert (FLAGSHIP_FIXTURE / "manifest.json").is_file()
    assert (FLAGSHIP_FIXTURE / "index.json").is_file()
    assert (FLAGSHIP_FIXTURE / "SHA256SUMS").is_file()
    manifest = json.loads((FLAGSHIP_FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "g6.flagship.index"
    assert manifest["version"] == "6.0.0"
    assert manifest["schema"] == "learningos.flagship.v1"
    assert manifest["index"] == "index.json"
    assert manifest["digest"] == _sha256(FLAGSHIP_FIXTURE / "index.json")
    sums = {}
    for line in (FLAGSHIP_FIXTURE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(None, 1)
        sums[rel.strip()] = digest.strip()
    assert sums["index.json"] == manifest["digest"]
    assert sums["manifest.json"] == _sha256(FLAGSHIP_FIXTURE / "manifest.json")


def test_index_covers_v00_to_v12_and_m01_to_m42() -> None:
    index = _index()
    assert index["schema"] == "learningos.flagship.v1"
    assert index["system"] == "Operations Intelligence System"
    versions = index["versions"]
    assert [item["id"] for item in versions] == VERSION_IDS
    assigned: list[str] = []
    for position, version in enumerate(versions):
        assert re.fullmatch(r"^V[0-9]{2}$", version["id"])
        assert version["name"]
        assert version["architecture_step"]
        assert version["release_tag"]
        assert version["missions"]
        expected_prereq = [] if position == 0 else [VERSION_IDS[position - 1]]
        assert version["prerequisites"] == expected_prereq
        assigned.extend(version["missions"])
    assert assigned == MISSION_IDS
    assert set(index["missions"]) == set(MISSION_IDS)
    assert index["missions"]["M01"]["flagship_version"] == "V00"
    assert index["missions"]["M07"]["flagship_version"] == "V01"
    assert index["missions"]["M42"]["flagship_version"] == "V12"


def test_prerequisites_are_acyclic_and_earlier_on_spine() -> None:
    index = _index()
    missions = index["missions"]
    for mission_id, record in missions.items():
        order = int(record["order_index"])
        blocking = record["prerequisites"]["blocking"]
        helpful = record["prerequisites"]["helpful"]
        assert mission_id not in blocking
        for dep in blocking:
            assert dep in missions
            assert int(missions[dep]["order_index"]) < order
        for dep in helpful:
            assert dep in missions
            assert dep != mission_id
    assert missions["M01"]["prerequisites"]["blocking"] == []
    assert missions["M07"]["prerequisites"]["blocking"] == ["M04", "M05", "M06"]
    assert missions["M15"]["prerequisites"]["blocking"] == ["M05"]
    assert missions["M42"]["prerequisites"]["blocking"] == ["M41"]


def test_artifact_types_are_generic_adr_and_git() -> None:
    index = _index()
    kinds = {item["id"] for item in index["learner_artifact_types"]}
    assert kinds == {"adr", "git"}
    assert "M42" not in json.dumps(index["learner_artifact_types"])
