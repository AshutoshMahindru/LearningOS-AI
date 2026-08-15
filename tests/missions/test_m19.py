from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M19"
NOTEBOOK = ROOT / "labs" / "M19_gradients.ipynb"
DATASET = ROOT / "datasets" / "M19" / "tiny_linear.csv"


def load_core():
    path = MISSION / "gradient_core.py"
    spec = importlib.util.spec_from_file_location("m19_gradient_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M19 gradient core")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during module load.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()


def load_fixture() -> tuple[list[float], list[float], list[float]]:
    with DATASET.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [float(row["x"]) for row in rows],
        [float(row["y_one_parameter"]) for row in rows],
        [float(row["y_with_bias"]) for row in rows],
    )


class M19GradientMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.xs, cls.ys, cls.ys_with_bias = load_fixture()

    def test_manual_weight_sweep_has_expected_loss_curve(self) -> None:
        weights = [0.0, 1.0, 2.0, 3.0, 4.0]
        losses = [CORE.one_parameter_loss(self.xs, self.ys, w) for w in weights]
        self.assertEqual(losses, [18.0, 8.0, 2.0, 0.0, 2.0])
        self.assertEqual(weights[losses.index(min(losses))], 3.0)

    def test_finite_difference_matches_analytic_derivative(self) -> None:
        weight = 1.5
        estimated = CORE.finite_difference_gradient(
            self.xs, self.ys, weight, epsilon=1.0e-5
        )
        analytic = CORE.analytic_weight_gradient(self.xs, self.ys, weight)
        self.assertAlmostEqual(analytic, -6.0, places=12)
        self.assertAlmostEqual(estimated, analytic, places=8)

    def test_correct_update_lowers_loss_and_preserves_trace(self) -> None:
        record = CORE.one_parameter_step(
            self.xs, self.ys, weight=1.0, learning_rate=0.2
        )
        self.assertEqual(record.parameters_before, (1.0,))
        self.assertEqual(record.gradient, (-8.0,))
        self.assertAlmostEqual(record.parameters_after[0], 2.6)
        self.assertLess(record.loss_after, record.loss_before)
        self.assertEqual(len(record.predictions), len(self.xs))

    def test_repeated_scalar_steps_decrease_loss_and_recover_weight(self) -> None:
        trace = CORE.run_one_parameter_descent(
            self.xs, self.ys, initial_weight=1.0, learning_rate=0.2, steps=8
        )
        losses = [trace[0].loss_before] + [record.loss_after for record in trace]
        self.assertTrue(all(after < before for before, after in zip(losses, losses[1:])))
        self.assertAlmostEqual(
            trace[-1].parameters_after[0], 3.0, delta=1.0e-5
        )

    def test_wrong_gradient_sign_is_controlled_rising_loss_failure(self) -> None:
        start_weight = 1.0
        learning_rate = 0.2
        gradient = CORE.analytic_weight_gradient(self.xs, self.ys, start_weight)
        faulty_weight = start_weight + learning_rate * gradient
        self.assertAlmostEqual(faulty_weight, -0.6)
        self.assertGreater(
            CORE.one_parameter_loss(self.xs, self.ys, faulty_weight),
            CORE.one_parameter_loss(self.xs, self.ys, start_weight),
        )

    def test_two_parameter_gradient_recovers_weight_and_bias(self) -> None:
        trace = CORE.run_linear_descent(
            self.xs,
            self.ys_with_bias,
            initial_weight=0.0,
            initial_bias=0.0,
            learning_rate=0.2,
            steps=20,
        )
        weight, bias = trace[-1].parameters_after
        self.assertAlmostEqual(weight, 3.0, places=12)
        self.assertAlmostEqual(bias, 2.0, delta=1.0e-4)
        self.assertLess(trace[-1].loss_after, 1.0e-8)

    def test_invalid_mechanics_inputs_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            CORE.one_parameter_loss([], [], 1.0)
        with self.assertRaises(ValueError):
            CORE.one_parameter_loss([1.0], [1.0, 2.0], 1.0)
        with self.assertRaises(ValueError):
            CORE.finite_difference_gradient([1.0], [2.0], 1.0, epsilon=0.0)
        with self.assertRaises(ValueError):
            CORE.update_parameter(1.0, -1.0, learning_rate=-0.1)


class M19MissionContractTests(unittest.TestCase):
    def test_required_mission_artifacts_are_declared_and_present(self) -> None:
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
            "review_brief.md",
            "adr_prompt.md",
            "status.yaml",
            "gradient_core.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M19/{name}", manifest)
        for path in [
            NOTEBOOK,
            DATASET,
            ROOT / "requirements" / "m19.txt",
            ROOT / "tests" / "test_m19.py",
        ]:
            self.assertTrue(path.is_file(), path)

    def test_manifest_encodes_runtime_sequence_and_parallel_safety(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M19",
            "phase: P3",
            "flagship: V04",
            "- calculus",
            "- gradients",
            "- change one parameter manually",
            "- extend the mechanism to weight and bias",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "modifies_lab_status: false",
            "modifies_tracking: false",
            "modifies_root_readme: false",
        ]:
            self.assertIn(expected, manifest)

    def test_notebook_is_clean_stable_substantial_and_offline(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 28)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        code = []
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])
                source = "".join(cell.get("source", []))
                compile(source, f"M19-{cell['id']}", "exec")
                code.append(source)
        all_code = "\n".join(code).lower()
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "urlopen",
            "socket.",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_notebook_enforces_required_pedagogical_order(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        positions = {cell["id"]: index for index, cell in enumerate(notebook["cells"])}
        ordered = [
            "predict-manual-sweep",
            "run-manual-sweep",
            "plot-loss-curve",
            "predict-finite-difference",
            "run-finite-difference",
            "derive-analytic-gradient",
            "predict-one-update",
            "run-one-update",
            "run-many-updates",
            "predict-wrong-sign",
            "run-wrong-sign",
            "introduce-multiple-parameters",
            "run-multiple-parameters",
        ]
        self.assertTrue(
            all(positions[before] < positions[after] for before, after in zip(ordered, ordered[1:]))
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        for phrase in [
            "Predict before running",
            "Controlled failure",
            "parameter → prediction → loss → gradient → update",
            "No-AI gate",
        ]:
            self.assertIn(phrase, markdown)

    def test_failure_gate_review_and_adr_contracts_are_consequential(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8").lower()
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8").lower()
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        for term in ["wrong gradient sign", "one controlled cause", "smallest", "verify"]:
            self.assertIn(term, controlled)
        for term in ["without ai-generated code", "calculate", "smallest repair", "loss falls"]:
            self.assertIn(term, no_ai)
        self.assertIn("wrong-sign controlled failure raises loss", review)
        self.assertIn("templates/ADR.md", adr)
        self.assertIn("parameter - learning_rate * gradient", adr)

    def test_evidence_contract_does_not_prefill_learner_work(self) -> None:
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_(evidence|response)\s*:", evidence))

    def test_fixture_obeys_both_declared_linear_rules(self) -> None:
        xs, one_parameter, with_bias = load_fixture()
        self.assertEqual(len(xs), 5)
        self.assertEqual(sum(xs), 0.0)
        for x, first_target, second_target in zip(xs, one_parameter, with_bias):
            self.assertEqual(first_target, 3.0 * x)
            self.assertEqual(second_target, 3.0 * x + 2.0)
            self.assertTrue(all(math.isfinite(v) for v in (x, first_target, second_target)))


if __name__ == "__main__":
    unittest.main()
