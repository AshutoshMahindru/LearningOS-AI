from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M17"
NOTEBOOK = ROOT / "labs" / "M17_probability_uncertainty.ipynb"
DATASET = ROOT / "datasets" / "M17" / "model_predictions.csv"


class M17MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            cell
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.namespace: dict[str, object] = {"__name__": "m17_notebook_test"}
        with redirect_stdout(io.StringIO()):
            for cell in cls.code_cells:
                source = "".join(cell.get("source", []))
                exec(
                    compile(source, f"M17-{cell['id']}", "exec"),
                    cls.namespace,
                )

    def test_standard_package_and_manifest_contract(self) -> None:
        required = {
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
            "adr_prompt.md",
            "status.yaml",
        }
        self.assertEqual(
            required,
            {path.name for path in MISSION.iterdir() if path.is_file()},
        )
        self.assertTrue(DATASET.is_file())
        self.assertTrue((ROOT / "datasets" / "M17" / "README.md").is_file())
        self.assertTrue((ROOT / "requirements" / "m17.txt").is_file())

        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M17",
            "phase: P3",
            "flagship: V04",
            "pedagogy: simulation-and-model-outputs-before-formalism",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "type: base_rate_neglect",
            "adr_prompt_required: true",
            "shared_registry_edits_required: false",
        ]:
            self.assertIn(expected, manifest)

    def test_dataset_has_designed_calibration_and_shift(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))

        self.assertEqual(len(records), 40)
        self.assertEqual(len({row["case_id"] for row in records}), 40)
        self.assertEqual({row["cohort"] for row in records}, {"stable", "deployment_shift"})
        self.assertTrue(all(float(row["model_probability"]) in {0.1, 0.4, 0.8, 0.9} for row in records))
        self.assertTrue(all(int(row["outcome"]) in {0, 1} for row in records))

        grouped: dict[float, list[int]] = {}
        for row in records:
            grouped.setdefault(float(row["model_probability"]), []).append(int(row["outcome"]))

        observed = {
            probability: sum(outcomes) / len(outcomes)
            for probability, outcomes in grouped.items()
        }
        for calibrated_probability in (0.1, 0.4, 0.8):
            self.assertAlmostEqual(observed[calibrated_probability], calibrated_probability)
        self.assertAlmostEqual(observed[0.9], 0.5)

    def test_notebook_has_stable_ids_empty_outputs_and_mission_metadata(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(len(self.notebook["cells"]), 30)
        ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(self.notebook["metadata"]["mission"]["id"], "M17")
        self.assertTrue(self.notebook["metadata"]["mission"]["cpu_only"])
        self.assertFalse(self.notebook["metadata"]["mission"]["network_required"])
        for cell in self.code_cells:
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs", []), [])

    def test_prediction_checkpoints_precede_every_key_experiment(self) -> None:
        cells = self.notebook["cells"]
        key_action_ids = {
            "m17-03-model-summary",
            "m17-05-calibration-table",
            "m17-08-frequency-simulation",
            "m17-12-screening-counts",
            "m17-16-base-rate-failure",
            "m17-18-dependence-check",
            "m17-21-expected-loss",
            "m17-24-thresholds",
            "m17-28-reliability-report",
            "m17-30-base-rate-sweep",
        }
        for index, cell in enumerate(cells):
            if cell.get("id") in key_action_ids:
                previous = "".join(cells[index - 1].get("source", []))
                self.assertEqual(cells[index - 1]["cell_type"], "markdown")
                self.assertIn("Prediction checkpoint", previous)

        ids = {cell["id"]: index for index, cell in enumerate(cells)}
        self.assertLess(ids["m17-08-frequency-simulation"], ids["m17-09-frequency-formalism"])
        self.assertLess(ids["m17-12-screening-counts"], ids["m17-13-conditional-formalism"])

    def test_notebook_code_compiles_and_is_network_secret_free(self) -> None:
        all_code = []
        for cell in self.code_cells:
            source = "".join(cell.get("source", []))
            compile(source, f"M17-{cell['id']}", "exec")
            all_code.append(source)
        code = "\n".join(all_code)
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
            "urlopen",
        ]:
            self.assertNotIn(forbidden, code)

    def test_frequency_simulation_is_deterministic_and_converges(self) -> None:
        simulate = self.namespace["simulate_event_frequency"]
        first = simulate(0.30, 20_000, seed=1704)
        second = simulate(0.30, 20_000, seed=1704)
        self.assertEqual(first, second)
        self.assertEqual(first["events"] + (first["trials"] - first["events"]), 20_000)
        self.assertLess(abs(first["frequency"] - 0.30), 0.02)
        with self.assertRaises(ValueError):
            simulate(1.1, 20)

    def test_conditional_probability_retains_base_rate(self) -> None:
        screening_counts = self.namespace["screening_counts"]
        conditional_probability = self.namespace["conditional_probability"]
        counts = screening_counts(10_000, 0.01, 0.90, 0.91)

        self.assertEqual(counts["affected"], 100)
        self.assertEqual(counts["true_positive"], 90)
        self.assertEqual(counts["false_positive"], 891)
        self.assertEqual(
            sum(counts[key] for key in ("true_positive", "false_negative", "false_positive", "true_negative")),
            counts["population"],
        )
        self.assertAlmostEqual(
            conditional_probability(counts["true_positive"], counts["affected"]),
            0.90,
        )
        posterior = conditional_probability(counts["true_positive"], counts["all_positive"])
        self.assertAlmostEqual(posterior, 90 / 981)
        self.assertLess(posterior, 0.10)

    def test_base_rate_sweep_repairs_the_controlled_failure(self) -> None:
        posterior_from_rates = self.namespace["posterior_from_rates"]
        posteriors = [
            posterior_from_rates(base_rate, 0.90, 0.91)
            for base_rate in (0.01, 0.10, 0.50)
        ]
        self.assertEqual(posteriors, sorted(posteriors))
        self.assertAlmostEqual(posteriors[0], 90 / 981)
        self.assertNotAlmostEqual(posteriors[0], 0.90)

        failure = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        for term in [
            "one seeded root cause",
            "base-rate neglect",
            "denominator",
            "false positives",
            "counterfactual",
            "operational harm",
        ]:
            self.assertIn(term, failure)

    def test_dependence_is_checked_against_product_of_marginals(self) -> None:
        independence_report = self.namespace["independence_report"]
        dependent = independence_report(100, 20, 20, 14)
        independent = independence_report(100, 20, 20, 4)
        self.assertAlmostEqual(dependent["independent_reference"], 0.04)
        self.assertAlmostEqual(dependent["observed_P(A_and_B)"], 0.14)
        self.assertEqual(dependent["relationship"], "dependent")
        self.assertEqual(independent["relationship"], "independent")

    def test_expected_loss_and_thresholds_change_actions(self) -> None:
        choose_action = self.namespace["choose_action"]
        confusion_counts = self.namespace["confusion_counts"]
        threshold_operational_loss = self.namespace["threshold_operational_loss"]
        rows = self.namespace["rows"]

        self.assertEqual(choose_action(0.10, 12, 100)["action"], "wait")
        self.assertEqual(choose_action(0.40, 12, 100)["action"], "act")
        self.assertEqual(choose_action(0.40, 50, 100)["action"], "wait")

        low = confusion_counts(rows, 0.50)
        high = confusion_counts(rows, 0.95)
        self.assertGreater(low["true_positive"], high["true_positive"])
        self.assertLess(low["false_negative"], high["false_negative"])
        self.assertLess(threshold_operational_loss(low), threshold_operational_loss(high))

    def test_calibration_report_detects_deployment_shift(self) -> None:
        reliability_report = self.namespace["reliability_report"]
        report = reliability_report(self.namespace["rows"])
        by_probability = {row["group"]: row for row in report}
        for probability in (0.1, 0.4, 0.8):
            self.assertAlmostEqual(by_probability[probability]["calibration_gap"], 0.0)
        self.assertEqual(by_probability[0.9]["count"], 10)
        self.assertAlmostEqual(by_probability[0.9]["observed_frequency"], 0.5)
        self.assertAlmostEqual(by_probability[0.9]["calibration_gap"], -0.4)

    def test_transfer_evidence_and_v04_decision_contract_are_unfilled(self) -> None:
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        assessment = (MISSION / "assessment.yaml").read_text(encoding="utf-8")
        flagship = (MISSION / "flagship_integration.md").read_text(encoding="utf-8")
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")

        self.assertIn("required_evidence:", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))
        self.assertIn("transfer_required: true", assessment)
        self.assertIn("V04", flagship)
        self.assertIn("Mathematical Instrumentation Layer", flagship)
        for heading in [
            "## Decision",
            "## Context",
            "## Alternatives considered",
            "## Evidence",
            "## Trade-offs",
            "## Revisit conditions",
            "## Status",
        ]:
            self.assertIn(heading, adr)
        self.assertIn("intentionally leaves the decision unanswered", adr)


if __name__ == "__main__":
    unittest.main()
