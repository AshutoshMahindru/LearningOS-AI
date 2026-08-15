from __future__ import annotations

import json
import math
import re
from pathlib import Path
import statistics
import unittest

from simulations.M18.statistical_inference import (
    aggregate_counts,
    bootstrap_differences,
    bootstrap_interval,
    cherry_pick_null_comparisons,
    describe,
    difference_in_means,
    effect_sizes,
    expand_binary_outcomes,
    load_confounding_fixture,
    load_daily_experiment,
    normal_confidence_interval,
    pearson_correlation,
    permutation_test,
    simulate_familywise_false_positive_rate,
    simulate_sample_means,
    z_test_difference,
)


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M18"
NOTEBOOK = ROOT / "labs" / "M18_statistical_inference.ipynb"
EXPERIMENT = ROOT / "datasets" / "M18" / "checkout_experiment_daily.csv"
CONFOUNDING = ROOT / "datasets" / "M18" / "seasonal_correlation.csv"


class M18MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_daily_experiment(EXPERIMENT)
        cls.outcomes_a = expand_binary_outcomes(cls.rows, "A")
        cls.outcomes_b = expand_binary_outcomes(cls.rows, "B")

    def test_required_standard_and_review_artifacts_are_files(self) -> None:
        required = [
            "README.md",
            "manifest.yaml",
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
        for name in required:
            with self.subTest(name=name):
                self.assertTrue((MISSION / name).is_file(), name)

        for path in [
            NOTEBOOK,
            EXPERIMENT,
            CONFOUNDING,
            ROOT / "datasets" / "M18" / "README.md",
            ROOT / "simulations" / "M18" / "statistical_inference.py",
            ROOT / "requirements" / "m18.txt",
        ]:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), str(path))

    def test_manifest_declares_exact_mission_runtime_and_inference_contract(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        expected_fragments = [
            "id: M18",
            "phase: P3",
            "flagship: V04",
            "formal_engineering_review: true",
            "- M17",
            "primary_metric: checkout_conversion",
            "estimand: variant_B_rate_minus_variant_A_rate",
            "planned_comparisons: 1",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "source_notebook_outputs: empty",
            "learner_evidence_prepopulated: false",
            "shared_registry_edits_required: false",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, manifest)

    def test_checkout_fixture_has_expected_balanced_counts_and_effect(self) -> None:
        totals = aggregate_counts(self.rows)
        self.assertEqual(
            totals,
            {
                "A": {"sessions": 600, "conversions": 72, "rate": 0.12},
                "B": {"sessions": 600, "conversions": 84, "rate": 0.14},
            },
        )
        self.assertEqual(len(self.rows), 24)
        self.assertEqual(len(self.outcomes_a), 600)
        self.assertEqual(len(self.outcomes_b), 600)
        self.assertTrue(all(value in {0, 1} for value in self.outcomes_a))
        self.assertTrue(all(value in {0, 1} for value in self.outcomes_b))
        self.assertAlmostEqual(
            difference_in_means(self.outcomes_a, self.outcomes_b), 0.02
        )

    def test_descriptive_statistics_preserve_scale_and_sample_variance(self) -> None:
        summary_a = describe(self.outcomes_a)
        summary_b = describe(self.outcomes_b)
        self.assertEqual(summary_a["n"], 600)
        self.assertEqual(summary_b["n"], 600)
        self.assertAlmostEqual(float(summary_a["mean"]), 0.12)
        self.assertAlmostEqual(float(summary_b["mean"]), 0.14)
        self.assertEqual(summary_a["median"], 0.0)
        self.assertEqual(summary_b["median"], 0.0)
        self.assertAlmostEqual(
            float(summary_a["variance"]), statistics.variance(self.outcomes_a)
        )
        self.assertGreater(float(summary_b["variance"]), 0)

    def test_sampling_variation_shrinks_at_inverse_sqrt_rate(self) -> None:
        small = simulate_sample_means(
            0.13, 25, repetitions=4_000, seed=1801
        )
        large = simulate_sample_means(
            0.13, 400, repetitions=4_000, seed=1802
        )
        small_spread = statistics.stdev(small)
        large_spread = statistics.stdev(large)
        theoretical_small = math.sqrt(0.13 * 0.87 / 25)
        theoretical_large = math.sqrt(0.13 * 0.87 / 400)

        self.assertAlmostEqual(statistics.fmean(small), 0.13, delta=0.006)
        self.assertAlmostEqual(statistics.fmean(large), 0.13, delta=0.002)
        self.assertAlmostEqual(small_spread, theoretical_small, delta=0.004)
        self.assertAlmostEqual(large_spread, theoretical_large, delta=0.001)
        self.assertAlmostEqual(small_spread / large_spread, 4.0, delta=0.35)

    def test_analytic_and_bootstrap_intervals_are_uncertain_and_deterministic(self) -> None:
        analytic = normal_confidence_interval(self.outcomes_a, self.outcomes_b)
        bootstrap_one = bootstrap_interval(
            self.outcomes_a,
            self.outcomes_b,
            resamples=2_000,
            seed=1818,
        )
        bootstrap_two = bootstrap_interval(
            self.outcomes_a,
            self.outcomes_b,
            resamples=2_000,
            seed=1818,
        )
        draws_one = bootstrap_differences(
            self.outcomes_a,
            self.outcomes_b,
            resamples=100,
            seed=99,
        )
        draws_two = bootstrap_differences(
            self.outcomes_a,
            self.outcomes_b,
            resamples=100,
            seed=99,
        )

        self.assertLess(analytic[0], 0)
        self.assertGreater(analytic[1], 0)
        self.assertEqual(bootstrap_one, bootstrap_two)
        self.assertLess(bootstrap_one[0], 0)
        self.assertGreater(bootstrap_one[1], 0)
        self.assertEqual(draws_one, draws_two)
        self.assertAlmostEqual(statistics.fmean(draws_one), 0.02, delta=0.006)

    def test_effect_sizes_and_hypothesis_tests_answer_distinct_questions(self) -> None:
        effects = effect_sizes(self.outcomes_a, self.outcomes_b)
        z_result = z_test_difference(self.outcomes_a, self.outcomes_b)
        permutation = permutation_test(
            self.outcomes_a,
            self.outcomes_b,
            permutations=2_000,
            seed=1819,
        )

        self.assertAlmostEqual(effects["risk_difference"], 0.02)
        self.assertAlmostEqual(effects["relative_risk"], 7 / 6)
        self.assertAlmostEqual(effects["relative_lift"], 1 / 6)
        self.assertAlmostEqual(effects["number_needed_to_treat"], 50)
        self.assertGreater(z_result["p_value"], 0.25)
        self.assertLess(z_result["p_value"], 0.40)
        self.assertGreater(permutation["p_value"], 0.25)
        self.assertLess(permutation["p_value"], 0.45)

    def test_correlation_fixture_exposes_a_named_common_cause(self) -> None:
        rows = load_confounding_fixture(CONFOUNDING)
        temperature = [float(row["temperature_c"]) for row in rows]
        sales = [float(row["ice_cream_sales"]) for row in rows]
        drownings = [float(row["drownings"]) for row in rows]

        self.assertGreater(pearson_correlation(sales, drownings), 0.9)
        self.assertGreater(pearson_correlation(temperature, sales), 0.99)
        self.assertGreater(pearson_correlation(temperature, drownings), 0.9)

        content = (MISSION / "content.yaml").read_text(encoding="utf-8").lower()
        notebook = NOTEBOOK.read_text(encoding="utf-8").lower()
        for term in ["correlation", "causation", "confounding"]:
            self.assertIn(term, content)
        self.assertIn("common seasonal cause", notebook)
        self.assertIn("identification strategy", notebook)

    def test_controlled_failure_exposes_and_repairs_multiplicity(self) -> None:
        failure = cherry_pick_null_comparisons(
            comparisons=20, seed=1800, alpha=0.05
        )
        results = failure["comparisons"]
        selected = failure["selected"]
        adjusted_threshold = float(failure["bonferroni_threshold"])

        self.assertEqual(len(results), 20)
        self.assertEqual(len(selected), 2)
        self.assertLess(float(failure["minimum_p_value"]), 0.05)
        self.assertAlmostEqual(adjusted_threshold, 0.0025)
        self.assertTrue(all(float(item["p_value"]) < 0.05 for item in selected))
        self.assertFalse(
            any(float(item["p_value"]) < adjusted_threshold for item in results)
        )

        familywise = simulate_familywise_false_positive_rate(
            comparisons=20, families=4_000, seed=1820, alpha=0.05
        )
        theoretical = 1 - 0.95**20
        self.assertAlmostEqual(familywise, theoretical, delta=0.03)

        diagnosis = (MISSION / "controlled_failure.md").read_text(
            encoding="utf-8"
        ).lower()
        for term in [
            "cherry-pick",
            "comparison family",
            "selective reporting",
            "bonferroni",
            "benjamini–hochberg",
            "fresh data",
        ]:
            self.assertIn(term, diagnosis)

    def test_notebook_has_stable_ids_empty_outputs_and_executable_code(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 35)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        for index, cell in enumerate(code_cells):
            self.assertIsNone(cell.get("execution_count"), cell["id"])
            self.assertEqual(cell.get("outputs"), [], cell["id"])
            source = "".join(cell.get("source", []))
            compile(source, f"M18-cell-{index}-{cell['id']}", "exec")

        all_code = "\n".join(
            "".join(cell.get("source", [])) for cell in code_cells
        )
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
            self.assertNotIn(forbidden, all_code)

    def test_notebook_covers_the_full_inference_and_failure_contract(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        for phrase in [
            "Predict before running",
            "Sampling variation",
            "standard-error intuition",
            "Estimate first; attach uncertainty",
            "Bootstrap/resampling",
            "Effect size before threshold",
            "useful hypothesis test",
            "Correlation is not a causal",
            "Controlled failure",
            "multiple",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase.lower(), markdown.lower())

        for symbol in [
            "simulate_sample_means",
            "normal_confidence_interval",
            "bootstrap_interval",
            "effect_sizes",
            "z_test_difference",
            "permutation_test",
            "pearson_correlation",
            "cherry_pick_null_comparisons",
            "simulate_familywise_false_positive_rate",
        ]:
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, code)

    def test_evidence_review_and_adr_keep_decisions_auditable(self) -> None:
        evidence = (MISSION / "evidence_contract.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("required_evidence:", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))
        self.assertIn("prefilled_learner_evidence: prohibited", evidence)

        review = (MISSION / "review_brief.md").read_text(encoding="utf-8")
        for heading in [
            "## Review decision requested",
            "## Material assumptions",
            "## Risks and controls",
            "## Required reviewer challenges",
            "## Acceptance criteria",
        ]:
            self.assertIn(heading, review)

        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        for requirement in [
            "minimum practically important effect",
            "comparison-family definition",
            "stopping rule",
            "Alternatives that must be compared",
            "Revisit conditions",
        ]:
            self.assertIn(requirement, adr)


if __name__ == "__main__":
    unittest.main()
