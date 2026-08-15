from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M10"
DATA = ROOT / "datasets" / "M10"
NOTEBOOK = ROOT / "labs" / "M10_metrics_consequences.ipynb"


def read_score_rows() -> list[dict[str, object]]:
    with (DATA / "asset_alert_scores.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return [
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "score": float(row["risk_score"]),
                "actual": int(row["failure_within_30d"]),
            }
            for row in csv.DictReader(handle)
        ]


def confusion(rows: list[dict[str, object]], threshold: float) -> dict[str, int]:
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for row in rows:
        predicted = int(float(row["score"]) >= threshold)
        actual = int(row["actual"])
        outcome = (
            "TP"
            if predicted and actual
            else "FP"
            if predicted
            else "FN"
            if actual
            else "TN"
        )
        counts[outcome] += 1
    return counts


def metric_row(
    rows: list[dict[str, object]], threshold: float, fp_cost: int = 2, fn_cost: int = 18
) -> dict[str, float | int]:
    counts = confusion(rows, threshold)
    tp, fp, tn, fn = (counts[key] for key in ("TP", "FP", "TN", "FN"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        **counts,
        "accuracy": (tp + tn) / len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": (tp + fp) / len(rows),
        "expected_cost": fp_cost * fp + fn_cost * fn,
    }


class M10MissionPackageTests(unittest.TestCase):
    def test_required_standard_and_review_artifacts_exist(self) -> None:
        mission_files = [
            "manifest.yaml",
            "README.md",
            "content.yaml",
            "experiments.yaml",
            "code_reading.md",
            "no_ai_gate.md",
            "controlled_failure.md",
            "assessment.yaml",
            "evidence_contract.yaml",
            "flagship_integration.md",
            "status.yaml",
            "review_brief.md",
            "adr_prompt.md",
        ]
        data_files = [
            "README.md",
            "asset_alert_scores.csv",
            "consequence_matrix.csv",
            "unseen_threshold_evidence.csv",
        ]
        for name in mission_files:
            self.assertTrue((MISSION / name).is_file(), name)
        for name in data_files:
            self.assertTrue((DATA / name).is_file(), name)
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue((ROOT / "requirements" / "m10.txt").is_file())

    def test_manifest_declares_consequence_first_reviewed_contract(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M10",
            "phase: P2",
            "flagship: V02",
            "pedagogy: consequence-first",
            "formal_engineering_review: true",
            "adr_required: true",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "- M09",
        ]:
            self.assertIn(expected, manifest)

    def test_score_fixture_preserves_validation_test_boundary_and_imbalance(self) -> None:
        rows = read_score_rows()
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({row["case_id"] for row in rows}), 60)
        self.assertTrue(all(0.0 <= float(row["score"]) <= 1.0 for row in rows))
        self.assertTrue(all(row["actual"] in {0, 1} for row in rows))

        by_split = {
            split: [row for row in rows if row["split"] == split]
            for split in ("validation", "test")
        }
        self.assertEqual(set(row["split"] for row in rows), set(by_split))
        for split_rows in by_split.values():
            self.assertEqual(len(split_rows), 30)
            self.assertEqual(sum(int(row["actual"]) for row in split_rows), 6)

    def test_consequence_matrix_is_complete_and_asymmetric(self) -> None:
        with (DATA / "consequence_matrix.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["outcome"] for row in rows], ["TP", "FP", "TN", "FN"])
        costs = {row["outcome"]: int(row["cost_units"]) for row in rows}
        self.assertEqual(costs, {"TP": 0, "FP": 2, "TN": 0, "FN": 18})
        self.assertGreater(costs["FN"], costs["FP"])
        self.assertTrue(all(row["operational_meaning"].strip() for row in rows))

    def test_confusion_matrix_and_metric_invariants_at_half(self) -> None:
        validation = [
            row for row in read_score_rows() if row["split"] == "validation"
        ]
        row = metric_row(validation, 0.50)
        self.assertEqual(
            {key: row[key] for key in ("TP", "FP", "TN", "FN")},
            {"TP": 3, "FP": 3, "TN": 21, "FN": 3},
        )
        self.assertAlmostEqual(float(row["accuracy"]), 0.8)
        self.assertAlmostEqual(float(row["precision"]), 0.5)
        self.assertAlmostEqual(float(row["recall"]), 0.5)
        self.assertAlmostEqual(float(row["f1"]), 0.5)
        self.assertEqual(row["expected_cost"], 60)

    def test_controlled_failure_proves_accuracy_harms_decision_utility(self) -> None:
        validation = [
            row for row in read_score_rows() if row["split"] == "validation"
        ]
        thresholds = sorted(
            {float(row["score"]) for row in validation} | {0.0, 1.01},
            reverse=True,
        )
        table = [metric_row(validation, threshold) for threshold in thresholds]
        accuracy_choice = max(
            table, key=lambda row: (float(row["accuracy"]), float(row["threshold"]))
        )
        feasible = [row for row in table if float(row["alert_rate"]) <= 0.50]
        utility_choice = min(
            feasible,
            key=lambda row: (
                int(row["expected_cost"]),
                -float(row["recall"]),
                float(row["threshold"]),
            ),
        )
        f1_choice = max(
            table, key=lambda row: (float(row["f1"]), float(row["threshold"]))
        )

        self.assertEqual(accuracy_choice["threshold"], 0.92)
        self.assertEqual(accuracy_choice["expected_cost"], 90)
        self.assertEqual(utility_choice["threshold"], 0.18)
        self.assertEqual(utility_choice["expected_cost"], 18)
        self.assertGreater(accuracy_choice["accuracy"], utility_choice["accuracy"])
        self.assertGreater(accuracy_choice["expected_cost"], utility_choice["expected_cost"])
        self.assertEqual(f1_choice["threshold"], 0.31)
        self.assertEqual(f1_choice["expected_cost"], 30)

    def test_locked_threshold_has_expected_untuned_test_confusion(self) -> None:
        test_rows = [row for row in read_score_rows() if row["split"] == "test"]
        self.assertEqual(
            confusion(test_rows, 0.18),
            {"TP": 5, "FP": 9, "TN": 15, "FN": 1},
        )

    def test_unseen_gate_requires_capacity_constrained_cost_reasoning(self) -> None:
        with (DATA / "unseen_threshold_evidence.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            source = list(csv.DictReader(handle))
        self.assertEqual(
            set(source[0]), {"threshold", "tp", "fp", "tn", "fn"}
        )
        self.assertNotIn("recommended_threshold", source[0])
        self.assertNotIn("selected", source[0])

        candidates = []
        for row in source:
            counts = {key: int(row[key]) for key in ("tp", "fp", "tn", "fn")}
            self.assertEqual(sum(counts.values()), 100)
            candidates.append(
                {
                    "threshold": float(row["threshold"]),
                    "alerts": counts["tp"] + counts["fp"],
                    "cost": 4 * counts["fp"] + 60 * counts["fn"],
                }
            )
        feasible = [candidate for candidate in candidates if candidate["alerts"] <= 20]
        self.assertEqual(
            min(feasible, key=lambda candidate: candidate["cost"])["threshold"],
            0.50,
        )

    def test_notebook_is_substantial_stable_clean_and_network_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 24)

        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        for index, cell in enumerate(code_cells):
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs", []), [])
            compile("".join(cell.get("source", [])), f"M10-cell-{index}", "exec")

        code = "\n".join("".join(cell["source"]) for cell in code_cells)
        for symbol in [
            "load_consequence_costs",
            "confusion_counts",
            "metrics_from_counts",
            "threshold_table",
            "roc_auc_pairwise",
            "average_precision",
            "accuracy_choice",
            "f1_choice",
            "utility_choice",
            "LOCKED_THRESHOLD",
        ]:
            self.assertIn(symbol, code)
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
        ]:
            self.assertNotIn(forbidden, code)

        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 8)
        for phrase in [
            "Begin with the decision",
            "Controlled failure",
            "ROC and precision-recall",
            "No-AI Gate",
            "formal engineering review",
        ]:
            self.assertIn(phrase, markdown)

    def test_review_and_adr_require_governed_decision_reasoning(self) -> None:
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8")
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        for phrase in [
            "formal engineering review",
            "controlled failure",
            "Accept / Reject / Defer",
            "Required uncertainty",
            "Test evidence was not used to tune policy",
        ]:
            self.assertIn(phrase.lower(), review.lower())
        for heading in [
            "## Decision",
            "## Context",
            "## Alternatives considered",
            "## Evidence",
            "## Trade-offs and consequences",
            "## Monitoring",
            "## Revisit conditions",
            "## Status",
        ]:
            self.assertIn(heading, adr)

    def test_no_ai_and_evidence_contract_do_not_prefill_learner_work(self) -> None:
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8").lower()
        self.assertIn("without an ai assistant", no_ai)
        self.assertIn("fresh scenario", no_ai)
        self.assertIn("unseen consequences", no_ai)
        self.assertIn("at most 20 alerts", no_ai)

        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))


if __name__ == "__main__":
    unittest.main()
