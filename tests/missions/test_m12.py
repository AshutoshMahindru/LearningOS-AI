from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M12"
NOTEBOOK = ROOT / "labs" / "M12_tree_ensembles.ipynb"
DATASET = ROOT / "datasets" / "M12" / "ensemble_fixture.csv"


class M12MissionPackageTests(unittest.TestCase):
    def test_manifest_declares_complete_mission_contract(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M12",
            "phase: P2",
            "flagship: V03",
            "objective: Compare bagging and boosting through controlled experiments.",
            "  - ensembles",
            "  - bias variance",
            "formal_engineering_review: true",
            "adr_required: true",
            "    - M11",
            "  cpu_only: true",
            "  requires_secrets: false",
            "  requires_paid_api: false",
            "  network_required: false",
        ]:
            self.assertIn(expected, manifest)

        declared = [
            MISSION / name
            for name in [
                "manifest.yaml", "README.md", "content.yaml", "experiments.yaml",
                "code_reading.md", "no_ai_gate.md", "controlled_failure.md",
                "assessment.yaml", "evidence_contract.yaml", "flagship_integration.md",
                "status.yaml", "review_brief.md", "adr_prompt.md",
            ]
        ] + [
            NOTEBOOK,
            DATASET,
            ROOT / "datasets" / "M12" / "README.md",
            ROOT / "datasets" / "M12" / "generate_fixture.py",
            ROOT / "requirements" / "m12.txt",
            ROOT / "tests" / "missions" / "test_m12.py",
        ]
        missing = sorted(str(path.relative_to(ROOT)) for path in declared if not path.is_file())
        self.assertEqual(missing, [])

    def test_dataset_is_balanced_deterministic_and_nontrivial(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 480)
        self.assertEqual(
            list(rows[0]),
            ["x1", "x2", "linear_mix", "periodic", "noise1", "noise2", "target"],
        )
        targets = [int(row["target"]) for row in rows]
        self.assertEqual(set(targets), {0, 1})
        self.assertEqual(targets.count(0), targets.count(1))

        for feature in ["x1", "x2", "linear_mix", "periodic", "noise1", "noise2"]:
            values = [float(row[feature]) for row in rows]
            self.assertGreater(max(values) - min(values), 1.0, feature)

        self.assertEqual(rows[0]["x1"], "-0.54848518")
        self.assertEqual(rows[-1]["target"], "0")
        canonical_rows = "\n".join(
            ",".join(row[column] for column in rows[0]) for row in rows
        )
        digest = hashlib.sha256(canonical_rows.encode("utf-8")).hexdigest()
        self.assertEqual(digest, "558ae8e59a401c71fcb5afcbdaea3ab522a24c3cfe24a3a25d9941dc5de38c82")

    def test_notebook_has_stable_ids_and_no_prefilled_outputs(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 25)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_notebook_implements_required_ensemble_experiments(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

        for token in [
            "DecisionTreeClassifier(max_depth=2",
            "fit_resampled_trees",
            "bootstrap_predictions",
            "BaggingClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "staged_predict",
            "run_size_sweep",
            "ESTIMATOR_COUNTS",
            "run_depth_sweep",
            "CORRUPTION_RATE = 0.28",
            "balanced_accuracy_score",
        ]:
            self.assertIn(token, code)

        for phrase in [
            "Prediction before action",
            "Bootstrap variation",
            "Sequential correction",
            "Bias/variance intuition",
            "Controlled failure",
            "more trees always fixes it",
            "No-AI gate",
            "Formal engineering review",
            "ADR",
        ]:
            self.assertIn(phrase, markdown)

    def test_notebook_code_is_valid_and_offline(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

        for index, cell in enumerate(code_cells):
            compile("".join(cell.get("source", [])), f"M12-cell-{index}", "exec")

        code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
            "read_csv(\"http",
            "read_csv('http",
        ]:
            self.assertNotIn(forbidden, code)

    def test_experiment_contract_controls_comparisons_and_failure(self) -> None:
        experiments = (MISSION / "experiments.yaml").read_text(encoding="utf-8")

        self.assertIn("split_random_state: 1200", experiments)
        self.assertIn("primary_metric: balanced_accuracy", experiments)
        ids = re.findall(r"(?m)^  - id: (E\d+)$", experiments)
        self.assertEqual(ids, ["E1", "E2", "E3", "E4", "E5", "E6", "E7"])
        self.assertEqual(experiments.count("    prediction_prompt:"), 7)
        self.assertEqual(experiments.count("    success_condition:"), 7)
        self.assertGreaterEqual(experiments.count("      - "), 21)

        failure = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        self.assertIn("28 percent", failure)
        self.assertIn("corrupted", failure)
        self.assertIn("more trees always fixes it", failure)
        self.assertIn("test set", failure)

    def test_adr_and_review_require_consequential_tradeoffs(self) -> None:
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8").lower()
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8").lower()

        for term in [
            "latency budget",
            "model-size budget",
            "interpretability",
            "label-quality",
            "revisit conditions",
            "cross-validation",
        ]:
            self.assertIn(term, adr)

        for term in [
            "experimental validity",
            "operational consequences",
            "failure analysis",
            "blocker",
            "verdict",
            "unresolved uncertainty",
        ]:
            self.assertIn(term, review)

    def test_evidence_contract_does_not_prefill_learner_work(self) -> None:
        text = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("mission_id: M12", text)
        ids = set(re.findall(r"(?m)^  - id: ([a-z_]+)$", text))
        self.assertEqual(
            ids,
            {
                "prediction_log",
                "experimental_record",
                "controlled_failure_diagnosis",
                "ensemble_choice_adr",
                "formal_engineering_review",
                "no_ai_gate",
            },
        )
        self.assertIsNone(re.search(r"(?m)^\s*learner_(response|answer|score|evidence)\s*:", text))


if __name__ == "__main__":
    unittest.main()
