from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
M12_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M12"


def _experiment():
    path = M12_PACKAGE / "g5" / "reference" / "M12" / "experiment.py"
    spec = importlib.util.spec_from_file_location("m12_fixture_experiment_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_csv_matches_curriculum_dataset() -> None:
    canonical = REPO_ROOT / "datasets" / "M12" / "ensemble_fixture.csv"
    experiment = _experiment()
    assert experiment.ENSEMBLE_CSV.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")


def test_ensemble_comparison_emits_metric_and_votes() -> None:
    blocks = _experiment().run_ensemble_comparison()
    types = [block["type"] for block in blocks]
    assert types == ["metric", "table"]
    payload = blocks[0]["payload"]
    assert payload["n_bootstraps"] == 4
    assert payload["more_trees_repair_corrupted_labels"] is False
    assert len(payload["bag_votes"]) == 4
    assert payload["disagreement"] >= 0
