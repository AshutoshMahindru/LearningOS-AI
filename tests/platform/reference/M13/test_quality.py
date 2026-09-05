from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
M13_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M13"


def _experiment():
    path = M13_PACKAGE / "g5" / "reference" / "M13" / "experiment.py"
    spec = importlib.util.spec_from_file_location("m13_fixture_experiment_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_csv_matches_curriculum_dataset() -> None:
    canonical = REPO_ROOT / "datasets" / "M13" / "knn_scale_cases.csv"
    experiment = _experiment()
    assert experiment.CASES_CSV.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")


def test_knn_query_returns_neighbor_identities() -> None:
    experiment = _experiment()
    blocks = experiment.run_knn_query()
    types = [block["type"] for block in blocks]
    assert types == ["metric", "table"]
    payload = blocks[0]["payload"]
    assert payload["k"] == 3
    assert len(payload["neighbor_ids"]) == 3
    assert len(payload["distances"]) == 3
    assert payload["predicted_class"] in {"guided", "independent"}
    params = experiment.default_parameters()
    params["k"] = 1
    k1 = experiment.run_knn_query(params)
    assert k1[0]["payload"]["k"] == 1
    assert len(k1[0]["payload"]["neighbor_ids"]) == 1
