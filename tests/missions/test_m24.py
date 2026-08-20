from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M24"
NOTEBOOK = ROOT / "labs" / "M24_backpropagation.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m24.txt"


def load_core():
    spec = importlib.util.spec_from_file_location("m24_backprop_core", MISSION / "backprop_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M24 backprop core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
RUNTIME_DEPS = importlib.util.find_spec("numpy") is not None


def cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(str(part) for part in source)


def notebook_cells() -> list[dict[str, object]]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def first_code_cell() -> dict[str, object]:
    for cell in notebook_cells():
        if cell.get("cell_type") == "code":
            return cell
    raise AssertionError("notebook has no code cells")


class M24StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "backprop_core.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m24.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(path.is_file())

    def test_required_mission_artifacts_are_declared_and_present(self):
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
            "backprop_core.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M24/{name}", manifest)

    def test_notebook_is_valid_clean_unique_substantial_and_offline(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        cells = notebook["cells"]
        self.assertGreaterEqual(len(cells), 40)
        ids = [cell.get("id") for cell in cells]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 12)
        markdown_chars = sum(
            len(cell_source(cell)) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown_chars, 5000)

        for cell in code_cells:
            with self.subTest(cell=cell.get("id")):
                self.assertIsNone(cell.get("execution_count"), cell.get("id"))
                self.assertEqual(cell.get("outputs"), [], cell.get("id"))
                ast.parse(cell_source(cell))

        all_code = "\n".join(cell_source(cell) for cell in code_cells).lower()
        for forbidden in (
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "urlopen",
            "import torch",
            "from torch",
            "tensorflow",
            "cuda",
            "http://",
            "https://",
            "sklearn",
            "nn.module",
            "dataloader",
            "torch.autograd",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, all_code)

    def test_first_code_cell_bootstraps_repository_root(self):
        cell = first_code_cell()
        self.assertEqual(cell.get("id"), "setup")
        source = cell_source(cell)
        self.assertIn("Path.cwd()", source)
        self.assertIn("parents", source)
        self.assertIn("missions", source)
        self.assertIn("M24", source)
        self.assertIn("backprop_core.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M24.backprop_core import", source)
        self.assertIn("from missions.M23.forward_core import", source)

    def test_future_mission_boundary_stays_closed(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "import torch",
            "from torch",
            "autograd",
            "nn.Module",
            "DataLoader",
            "optimizer",
            "zero_grad",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertNotIn("for epoch", source.lower())

        core = (MISSION / "backprop_core.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", core)
        self.assertNotIn("from torch", core)
        self.assertNotIn("torch.autograd", core)
        self.assertIn("deferred to M25", core)
        self.assertIn("import missions.M23.forward_core", core)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-m19-invariant", "run-m19-invariant"),
            ("predict-forward-graph", "run-forward-graph"),
            ("predict-gradient-sign", "run-gradient-sign"),
            ("predict-scalar-chain", "run-scalar-chain"),
            ("predict-branch", "run-branch"),
            ("predict-activation-derivative", "run-activation-derivative"),
            ("predict-dense-backward", "run-dense-backward"),
            ("predict-finite-difference", "run-finite-difference"),
            ("predict-one-step", "run-one-step"),
            ("predict-failure", "run-failure"),
            ("predict-wrong-relu", "run-wrong-relu"),
            ("predict-failure-repair", "run-failure-repair"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                self.assertIn("Predict before running", cell_source(cells[positions[prediction]]))

        markdown = "\n".join(
            cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 11)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M19 → M23 → M24 → M25",
            "chain rule",
            "finite difference",
            "branch",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_reverse_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "restated_m19_invariant",
            "two_layer_forward",
            "GRAPH_NODES",
            "scalar_chain",
            "reverse_accumulate",
            "branch_gradients",
            "activation_boundary_pair",
            "two_layer_backward",
            "central_finite_difference",
            "finite_difference_sweep",
            "one_step_update",
            'defect="omitted_branch"',
            'defect="wrong_relu_derivative"',
            "check_parameter_gradients",
            "softmax_nll",
            "relu_local_derivative",
            "hidden_branch_contributions",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

    def test_requirements_are_bounded_and_cover_runtime(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        for package in ("numpy", "matplotlib", "nbclient", "nbformat", "pytest"):
            with self.subTest(package=package):
                matching = [line for line in requirements if line.startswith(package)]
                self.assertEqual(len(matching), 1)
                self.assertIn(">=", matching[0])
                self.assertIn("<", matching[0])

    def test_backprop_core_defers_optional_imports_until_runtime(self):
        source = (MISSION / "backprop_core.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        top_level_imports = [
            alias.name
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            node.module or ""
            for node in module.body
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertNotIn("numpy", top_level_imports)
        self.assertFalse(any(name.startswith("sklearn") for name in top_level_imports))
        self.assertNotIn("torch", top_level_imports)
        self.assertFalse(any("numpy" in name for name in top_level_imports))

    def test_learner_facing_contracts_remain_unfilled(self):
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("[UNFILLED BY LEARNER]", adr)
        self.assertIn("Leave all learner responses unfilled", no_ai)
        self.assertIn("intentionally_unpopulated", status)
        notebook_markdown = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "markdown"
        )
        self.assertIn("[UNFILLED BY LEARNER]", notebook_markdown)
        self.assertNotIn("[FILLED", notebook_markdown)

    def test_mission_package_declares_m19_m23_handoff_and_m25_deferral(self):
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        self.assertIn("M19", manifest)
        self.assertIn("M23", manifest)
        self.assertIn("pytorch_autograd_deferred_to_m25", manifest)
        self.assertIn("deferred_to_m25", content)
        self.assertIn("chain rule", content)


class M24ScalarMechanicsTests(unittest.TestCase):
    def test_m19_invariant_matches_analytic_and_finite_difference(self):
        report = CORE.restated_m19_invariant()
        self.assertAlmostEqual(float(report["loss"]), 10.0)
        self.assertAlmostEqual(float(report["analytic_gradient"]), -10.0)
        self.assertAlmostEqual(
            float(report["finite_difference"]),
            float(report["analytic_gradient"]),
            places=8,
        )
        self.assertEqual(report["update_convention"], "parameter - learning_rate * gradient")
        self.assertAlmostEqual(float(report["updated_weight"]), 2.0)

    def test_scalar_chain_hand_values_and_finite_difference(self):
        trace = CORE.scalar_chain()
        self.assertAlmostEqual(trace.z, 1.0)
        self.assertAlmostEqual(trace.h, 1.0)
        self.assertAlmostEqual(trace.y, -1.5)
        self.assertAlmostEqual(trace.loss, 2.0)
        self.assertAlmostEqual(trace.grads["y"], -2.0)
        self.assertAlmostEqual(trace.grads["v"], -2.0)
        self.assertAlmostEqual(trace.grads["h"], 3.0)
        self.assertAlmostEqual(trace.grads["z"], 3.0)
        self.assertAlmostEqual(trace.grads["w"], 6.0)
        self.assertAlmostEqual(trace.grads["b"], 3.0)
        estimated = CORE.central_finite_difference(lambda weight: CORE.scalar_chain_loss(w=weight), CORE.SCALAR_W)
        self.assertAlmostEqual(estimated, trace.grads["w"], places=8)

    def test_reverse_accumulate_adds_at_parents_and_doubles_without_reset(self):
        nodes = CORE.build_scalar_chain_tape()
        first = CORE.reverse_accumulate(nodes, reset=True)
        once = dict(first)
        doubled = CORE.reverse_accumulate(nodes, grads=first, reset=False)
        self.assertAlmostEqual(once["w"], 6.0)
        self.assertAlmostEqual(doubled["w"], 12.0)
        tape = CORE.GradientTape(nodes)
        tape.reset()
        reset = CORE.reverse_accumulate(nodes, grads=tape.grads, reset=True)
        self.assertAlmostEqual(reset["w"], 6.0)

    def test_branch_gradient_equals_summed_contributions(self):
        trace = CORE.branch_gradients()
        self.assertAlmostEqual(trace.loss, 4.0)
        self.assertAlmostEqual(trace.contribution_path1, 2.0)
        self.assertAlmostEqual(trace.contribution_path2, 1.0)
        self.assertAlmostEqual(trace.d_h, 3.0)
        estimated = CORE.central_finite_difference(lambda hidden: CORE.branch_loss(h=hidden), CORE.BRANCH_H)
        self.assertAlmostEqual(estimated, trace.d_h, places=8)
        omitted = CORE.branch_gradients(defect="omitted_branch")
        self.assertAlmostEqual(omitted.d_h, 2.0)
        self.assertFalse(CORE.finite_difference_agrees(omitted.d_h, estimated))

    def test_activation_boundary_blocks_then_releases_upstream_gradient(self):
        dead, live = CORE.activation_boundary_pair()
        self.assertLess(dead.z, 0.0)
        self.assertGreater(live.z, 0.0)
        self.assertAlmostEqual(dead.relu_prime, 0.0)
        self.assertAlmostEqual(live.relu_prime, 1.0)
        self.assertAlmostEqual(dead.grads["w"], 0.0)
        self.assertNotAlmostEqual(live.grads["w"], 0.0)
        dead_fd = CORE.central_finite_difference(
            lambda weight: CORE.scalar_chain_loss(w=weight), CORE.ACTIVATION_W_NEGATIVE
        )
        self.assertAlmostEqual(dead_fd, 0.0, places=8)
        wrong = CORE.scalar_chain(w=CORE.ACTIVATION_W_NEGATIVE, defect="wrong_relu_derivative")
        self.assertGreater(abs(wrong.grads["w"]), 1e-6)
        self.assertFalse(CORE.finite_difference_agrees(wrong.grads["w"], dead_fd))

    def test_scalar_one_step_lowers_loss_without_a_training_loop(self):
        step = CORE.scalar_one_step()
        self.assertLess(float(step["loss_after"]), float(step["loss_before"]))
        self.assertEqual(step["parameter"], "w")
        self.assertEqual(step["update_convention"], "parameter - learning_rate * gradient")

    def test_invalid_mechanics_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            CORE.central_finite_difference(lambda value: value, 1.0, epsilon=0.0)
        with self.assertRaises(ValueError):
            CORE.scalar_one_step(learning_rate=-0.1)
        with self.assertRaises(ValueError):
            CORE.reverse_accumulate(CORE.build_scalar_chain_tape(), output="missing")
        with self.assertRaises(ValueError):
            CORE._normalize_defect("explode_the_graph")


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m24.txt to run NumPy-dependent M24 tests")
class M24NetworkRuntimeTests(unittest.TestCase):
    def test_trusted_forward_graph_is_m23(self):
        forward = CORE.two_layer_forward(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        self.assertTrue(CORE.arrays_close(forward.logits, CORE.REFERENCE_LOGITS, atol=1e-12, rtol=0.0))
        self.assertEqual(CORE.M23_GRAPH_NODES[-1], "probabilities")
        self.assertEqual(CORE.GRAPH_NODES[-1], "loss")
        self.assertEqual(CORE.GRAPH_NODES[:5], CORE.M23_GRAPH_NODES)

    def test_softmax_nll_logit_gradient_is_probability_minus_one_hot(self):
        backward = CORE.reference_backward()
        probabilities = CORE._as_2d(backward.probabilities)
        expected = probabilities.copy()
        expected[0, 0] -= 1.0
        expected[1, 0] -= 1.0
        expected /= 2.0
        self.assertTrue(CORE.arrays_close(backward.d_logits, expected, atol=1e-12, rtol=0.0))
        self.assertAlmostEqual(
            backward.loss,
            CORE.softmax_nll(CORE.REFERENCE_LOGITS, CORE.TEACHING_TARGETS),
        )

    def test_hidden_unit_gradient_sums_class_contributions(self):
        backward = CORE.reference_backward()
        contributions = CORE.hidden_branch_contributions(backward.d_logits[1], CORE.REFERENCE_W2[0])
        self.assertAlmostEqual(sum(contributions), float(backward.d_hidden_activation[1, 0]))
        self.assertEqual(len(contributions), 3)

    def test_smooth_example_matches_central_finite_differences(self):
        reports = CORE.check_parameter_gradients(
            CORE.REFERENCE_X[CORE.TEACHING_ROW],
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            targets=(0,),
        )
        self.assertGreaterEqual(len(reports), 17)
        mismatch = CORE.first_mismatch(reports)
        self.assertIsNone(mismatch, mismatch)

    def test_full_batch_matches_after_skipping_relu_kinks(self):
        kinks = CORE.relu_kink_parameter_entries(
            CORE.REFERENCE_X, CORE.REFERENCE_HIDDEN_PREACTIVATION
        )
        self.assertIn(("b1", (0,)), kinks)
        reports = CORE.check_parameter_gradients(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            skip_relu_kinks=True,
        )
        self.assertIsNone(CORE.first_mismatch(reports))
        hinge = CORE.network_finite_difference(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            name="b1",
            index=(0,),
        )
        analytic = float(CORE.reference_backward().d_b1[0])
        self.assertFalse(CORE.finite_difference_agrees(analytic, hinge))

    def test_gradient_sign_from_loss_movement_matches_backprop(self):
        movement = CORE.perturb_parameter_loss(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        analytic = float(CORE.reference_backward().d_W2[CORE.SIGN_INDEX])
        self.assertLess(movement["loss_plus"], movement["loss_minus"])
        self.assertLess(movement["predicted_sign"], 0.0)
        self.assertLess(analytic, 0.0)
        self.assertEqual(math.copysign(1.0, analytic), movement["predicted_sign"])

    def test_omitted_branch_mismatches_hidden_parameters_not_w2(self):
        correct = CORE.check_parameter_gradients(
            CORE.REFERENCE_X[CORE.TEACHING_ROW],
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            targets=(0,),
            defect="none",
        )
        broken = CORE.check_parameter_gradients(
            CORE.REFERENCE_X[CORE.TEACHING_ROW],
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            targets=(0,),
            defect="omitted_branch",
        )
        self.assertIsNone(CORE.first_mismatch(correct))
        mismatch = CORE.first_mismatch(broken)
        self.assertIsNotNone(mismatch)
        self.assertTrue(mismatch.name.startswith("W1") or mismatch.name.startswith("b1"))
        w2_ok = [report for report in broken if report.name.startswith("W2") or report.name.startswith("b2")]
        self.assertTrue(w2_ok)
        self.assertTrue(all(report.agrees for report in w2_ok))

    def test_wrong_relu_derivative_is_isolated_on_a_dead_unit(self):
        reports = CORE.check_parameter_gradients(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="wrong_relu_derivative",
            skip_relu_kinks=True,
        )
        failed = [report.name for report in reports if not report.agrees]
        self.assertIn("b1(1,)", failed)
        w2_ok = [report for report in reports if report.name.startswith("W2")]
        self.assertTrue(all(report.agrees for report in w2_ok))

    def test_one_network_step_lowers_loss_and_does_not_loop(self):
        step = CORE.one_step_update(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        self.assertLess(float(step["loss_after"]), float(step["loss_before"]))
        self.assertEqual(step["parameter"], "W2")
        self.assertEqual(step["index"], CORE.ONE_STEP_INDEX)
        source = (MISSION / "backprop_core.py").read_text(encoding="utf-8")
        self.assertIn("Not a training loop", source)
        self.assertIn("one declared", source.lower())

    def test_smallest_repair_restores_finite_difference_agreement(self):
        repaired = CORE.check_parameter_gradients(
            CORE.REFERENCE_X[CORE.TEACHING_ROW],
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            targets=(0,),
            defect="none",
        )
        broken = CORE.two_layer_backward(
            CORE.REFERENCE_X[CORE.TEACHING_ROW],
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            targets=(0,),
            defect="omitted_branch",
        )
        correct = CORE.two_layer_backward(
            CORE.REFERENCE_X[CORE.TEACHING_ROW],
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            targets=(0,),
            defect="none",
        )
        self.assertFalse(CORE.arrays_close(broken.d_hidden_activation, correct.d_hidden_activation))
        self.assertTrue(CORE.arrays_close(broken.d_W2, correct.d_W2, atol=1e-12, rtol=0.0))
        self.assertIsNone(CORE.first_mismatch(repaired))

    def test_identity_hidden_map_has_unit_local_derivative(self):
        reports = CORE.check_parameter_gradients(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            hidden_activation="identity",
        )
        self.assertIsNone(CORE.first_mismatch(reports))

    def test_network_rejects_bad_targets_and_learning_rates(self):
        with self.assertRaises(ValueError):
            CORE.softmax_nll(CORE.REFERENCE_LOGITS, (9, 0))
        with self.assertRaises(ValueError):
            CORE.one_step_update(
                CORE.REFERENCE_X,
                CORE.REFERENCE_W1,
                CORE.REFERENCE_B1,
                CORE.REFERENCE_W2,
                CORE.REFERENCE_B2,
                learning_rate=0.0,
            )


if __name__ == "__main__":
    unittest.main()
