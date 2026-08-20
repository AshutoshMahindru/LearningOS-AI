from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M25"
NOTEBOOK = ROOT / "labs" / "M25_pytorch_training.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m25.txt"


def load_core():
    spec = importlib.util.spec_from_file_location("m25_pytorch_training", MISSION / "pytorch_training.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M25 pytorch training core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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


class M25StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "pytorch_training.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m25.py",
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
            "pytorch_training.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M25/{name}", manifest)

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
            "tensorflow",
            "http://",
            "https://",
            "torchvision",
            "fashionmnist",
            "cifar",
            "download=true",
            "cuda",
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
        self.assertIn("M25", source)
        self.assertIn("pytorch_training.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M25.pytorch_training import", source)
        self.assertIn("from missions.M24.backprop_core import", source)

    def test_future_mission_boundary_stays_closed(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "data_corruption",
            "learning_rate_failure",
            "gradient_flow",
            "chaos_day",
            "shuffle_labels",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        core = (MISSION / "pytorch_training.py").read_text(encoding="utf-8")
        self.assertIn("deferred to M26", core)
        self.assertIn("import missions.M24.backprop_core", core)
        self.assertNotIn("data_corruption", core)
        self.assertNotIn("chaos_day", core)

        markdown = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "markdown"
        )
        self.assertIn("M26", markdown)
        self.assertNotIn("Chaos Day", markdown)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-module", "run-module"),
            ("predict-forward-parity", "run-forward-parity"),
            ("predict-autograd-parity", "run-autograd-parity"),
            ("predict-one-step", "run-one-step"),
            ("predict-splits", "run-splits"),
            ("predict-batch", "run-batch"),
            ("predict-train-loop", "run-train-loop"),
            ("predict-eval", "run-eval"),
            ("predict-checkpoint", "run-checkpoint"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-failure", "run-failure"),
            ("predict-mode-failure", "run-mode-failure"),
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
        self.assertGreaterEqual(markdown.count("Predict before running"), 13)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M24 → M25 → M26",
            "zero_grad",
            "held-out",
            "checkpoint",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_loop_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "teaching_module",
            "parameter_ownership",
            "forward_parity_report",
            "autograd_parity_report",
            "canonical_training_step",
            "make_classification_fixture",
            "assert_protected_splits",
            "make_loader",
            "train_model",
            "evaluate",
            "held_out_eval",
            "batch_size_report",
            "checkpoint_roundtrip",
            "gradient_reset_experiment",
            "train_mode_eval_experiment",
            'defect="stale_grad"',
            'defect="train_mode_eval"',
            "inspect.getsource",
            "LOOP_ORDER",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

    def test_requirements_are_bounded_and_cover_runtime(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        for package in ("numpy", "matplotlib", "nbclient", "nbformat", "pytest", "torch"):
            with self.subTest(package=package):
                matching = [line for line in requirements if line.startswith(package)]
                self.assertEqual(len(matching), 1)
                self.assertIn(">=", matching[0])
                self.assertIn("<", matching[0])
        joined = "\n".join(requirements)
        self.assertIn("cpu", joined)
        self.assertIn("cuda", joined)

    def test_training_core_defers_optional_imports_until_runtime(self):
        source = (MISSION / "pytorch_training.py").read_text(encoding="utf-8")
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
        self.assertNotIn("torch", top_level_imports)
        self.assertFalse(any(name.startswith("torch") for name in top_level_imports))
        self.assertNotIn("numpy", top_level_imports)
        self.assertFalse(any(name.startswith("torchvision") for name in top_level_imports))

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

    def test_mission_package_declares_m24_handoff_and_m26_deferral(self):
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        self.assertIn("M24", manifest)
        self.assertIn("M26", manifest)
        self.assertIn("multi_cause_debugging_deferred_to_m26", manifest)
        self.assertIn("deferred_to_m26", content)
        self.assertIn("zero_grad", content)

    def test_unknown_and_future_defects_are_rejected(self):
        self.assertEqual(CORE._normalize_defect("missing_zero_grad"), "stale_grad")
        self.assertEqual(CORE._normalize_defect("eval_in_train_mode"), "train_mode_eval")
        self.assertEqual(CORE._normalize_defect("correct"), "none")
        with self.assertRaises(ValueError):
            CORE._normalize_defect("data_corruption")
        with self.assertRaises(ValueError):
            CORE._normalize_defect("learning_rate_failure")


@unittest.skipUnless(TORCH_AVAILABLE, "install requirements/m25.txt to run PyTorch-dependent M25 tests")
class M25TorchRuntimeTests(unittest.TestCase):
    def test_teaching_module_matches_m24_forward(self):
        report = CORE.forward_parity_report()
        self.assertTrue(report["agrees"], report)
        self.assertTrue(report["logits_agree"])
        self.assertEqual(report["logits"][0], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(report["logits"][1][0], 1.0)
        self.assertAlmostEqual(report["logits"][1][1], 1.5)
        self.assertAlmostEqual(report["logits"][1][2], -0.25)

    def test_autograd_matches_m24_selected_gradients(self):
        report = CORE.autograd_parity_report()
        self.assertTrue(report["agrees"], report)
        self.assertTrue(report["loss_agrees"])
        self.assertAlmostEqual(report["selected_autograd"], report["selected_m24"], places=12)
        self.assertAlmostEqual(report["selected_autograd"], -0.3296553694702, places=10)
        names = {row.name: row.agrees for row in report["rows"]}
        for name in ("W1", "b1", "W2", "b2"):
            self.assertTrue(names[name], name)

    def test_parameter_ownership_registers_linear_leaves(self):
        model = CORE.teaching_module()
        rows = {row["name"]: row for row in CORE.parameter_ownership(model)}
        self.assertEqual(rows["fc1.weight"]["shape"], (2, 3))
        self.assertEqual(rows["fc2.weight"]["shape"], (3, 2))
        self.assertTrue(rows["fc1.weight"]["requires_grad"])
        self.assertTrue(rows["fc1.weight"]["is_leaf"])
        self.assertEqual(rows["fc1.weight"]["device"], "cpu")
        self.assertEqual(rows["fc1.weight"]["dtype"], "float64")

    def test_canonical_step_lowers_loss_and_uses_loop_order(self):
        step = CORE.canonical_training_step()
        self.assertLess(step["loss_after"], step["loss_before"])
        self.assertTrue(step["parameters_moved"])
        self.assertEqual(step["order"], CORE.LOOP_ORDER)
        self.assertTrue(step["trace"].zero_grad_called)
        self.assertTrue(step["trace"].parameters_updated)
        self.assertEqual(step["momentum"], 0.0)

    def test_backward_without_step_does_not_update_parameters(self):
        model = CORE.teaching_module()
        x, y = CORE.teaching_batch()
        before = CORE.snapshot_parameters(model)
        model.train()
        loss = CORE.mean_softmax_nll()(model(x), y)
        loss.backward()
        self.assertIsNotNone(model.fc2.weight.grad)
        self.assertTrue(CORE.parameters_equal(before, CORE.snapshot_parameters(model)))

    def test_protected_splits_are_disjoint_and_complete(self):
        bundle = CORE.make_classification_fixture()
        self.assertTrue(CORE.assert_protected_splits(bundle))
        train, val, held = set(bundle.train_idx), set(bundle.val_idx), set(bundle.held_out_idx)
        self.assertEqual(len(train), 24)
        self.assertEqual(len(val), 6)
        self.assertEqual(len(held), 6)
        self.assertFalse(train & val)
        self.assertFalse(train & held)
        self.assertFalse(val & held)
        self.assertEqual(train | val | held, set(range(36)))

    def test_evaluation_does_not_update_parameters(self):
        model = CORE.teaching_module()
        x, y = CORE.teaching_batch()
        before = CORE.snapshot_parameters(model)
        trace = CORE.evaluate(model, x, y, split="val")
        self.assertFalse(trace.parameters_updated)
        self.assertFalse(trace.model_training)
        self.assertFalse(trace.grad_enabled)
        self.assertTrue(CORE.parameters_equal(before, CORE.snapshot_parameters(model)))

    def test_train_mode_eval_changes_outputs_not_parameters(self):
        report = CORE.train_mode_eval_experiment()
        self.assertFalse(report["correct_training_flag"])
        self.assertTrue(report["wrong_training_flag"])
        self.assertTrue(report["logits_differ"])
        self.assertFalse(report["parameters_updated_correct"])
        self.assertFalse(report["parameters_updated_wrong"])
        self.assertFalse(report["correct_grad_enabled"])
        self.assertTrue(report["wrong_grad_enabled"])

    def test_stale_gradients_accumulate_and_diverge_updates(self):
        report = CORE.gradient_reset_experiment()
        self.assertTrue(report["reset_keeps_second_equal_first"])
        self.assertTrue(report["stale_second_is_sum"])
        self.assertAlmostEqual(report["backward_twice_without_reset"], 2 * report["backward_once"], places=10)
        self.assertTrue(report["updates_diverge"])
        self.assertNotAlmostEqual(report["correct_param_after"], report["stale_param_after"])

    def test_train_loop_lowers_loss_without_touching_held_out(self):
        run = CORE.train_model()
        self.assertEqual(run.device, "cpu")
        self.assertEqual(run.n_hidden, CORE.TRAIN_HIDDEN)
        self.assertLess(run.epoch_traces[-1].train_loss, run.epoch_traces[0].train_loss)
        self.assertGreater(run.epoch_traces[-1].val_accuracy, run.epoch_traces[0].val_accuracy)
        held = CORE.held_out_eval(run)
        self.assertEqual(held.split, "held_out")
        self.assertFalse(held.parameters_updated)
        self.assertFalse(held.model_training)
        train_idx = set(run.train_idx)
        self.assertTrue(set(run.held_out_idx).isdisjoint(train_idx))
        self.assertTrue(set(run.held_out_idx).isdisjoint(set(run.val_idx)))
        for step in run.epoch_traces[0].steps:
            self.assertEqual(step.split, "train")
            self.assertTrue(step.zero_grad_called)

    def test_batch_size_changes_steps_not_universal_invariance(self):
        rows = CORE.batch_size_report(batch_sizes=(4, 12), epochs=3)
        self.assertEqual(rows[0]["steps_per_epoch"], 6)
        self.assertEqual(rows[1]["steps_per_epoch"], 2)
        self.assertEqual(rows[0]["batch_size"], 4)
        self.assertEqual(rows[1]["batch_size"], 12)

    def test_checkpoint_roundtrip_replays_held_out(self):
        run = CORE.train_model(epochs=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m25.pt"
            report = CORE.checkpoint_roundtrip(run, path)
            self.assertTrue(report["held_out_agrees"])
            self.assertTrue(report["replay_agrees"])
            self.assertFalse(report["parameters_updated"])
            self.assertFalse(report["model_training"])
            for key in CORE.CHECKPOINT_KEYS:
                self.assertIn(key, set(report["payload_keys"]) | {
                    "model_state_dict",
                    "optimizer_state_dict",
                    "torch_rng_state",
                })
            self.assertIn("optimizer_state_dict", report["payload_keys"])
            self.assertTrue(report["optimizer_state_loaded"])
            self.assertGreater(len(run.optimizer.state_dict()["state"]), 0)

    def test_dataloader_yields_cpu_float64_batches(self):
        bundle = CORE.make_classification_fixture()
        x, y = bundle.train()
        loader = CORE.make_loader(x, y, batch_size=8, shuffle=True, seed=CORE.DEFAULT_SEED)
        batch_x, batch_y = next(iter(loader))
        self.assertEqual(str(batch_x.device), "cpu")
        self.assertEqual(batch_x.dtype, CORE._require_torch().float64)
        self.assertEqual(tuple(batch_x.shape), (8, 3))

    def test_smallest_repair_restores_reset_and_eval_mode(self):
        broken = CORE.gradient_reset_experiment()
        self.assertTrue(broken["stale_second_is_sum"])
        self.assertTrue(broken["updates_diverge"])
        stale_g1, stale_g2 = broken["stale_step_grads"]
        self.assertNotAlmostEqual(stale_g1, stale_g2)

        x, targets = CORE.teaching_batch()
        model = CORE.teaching_module()
        model.train()
        criterion = CORE.mean_softmax_nll()
        model.zero_grad(set_to_none=True)
        criterion(model(x), targets).backward()
        first = float(model.fc2.weight.grad.detach()[0, 0])
        criterion(model(x), targets).backward()
        stale = float(model.fc2.weight.grad.detach()[0, 0])
        self.assertAlmostEqual(stale, 2 * first, places=10)
        model.zero_grad(set_to_none=False)
        criterion(model(x), targets).backward()
        repaired = float(model.fc2.weight.grad.detach()[0, 0])
        self.assertAlmostEqual(repaired, first, places=10)

        drop = CORE.teaching_module(dropout_p=CORE.MODE_DROPOUT_P)
        wrong = CORE.evaluate(drop, x, targets, split="held_out", defect="train_mode_eval")
        repaired_mode = CORE.evaluate(drop, x, targets, split="held_out", defect="none")
        self.assertTrue(wrong.model_training)
        self.assertFalse(repaired_mode.model_training)
        self.assertFalse(repaired_mode.parameters_updated)
        self.assertFalse(CORE.arrays_close(wrong.logits, repaired_mode.logits, atol=1e-12, rtol=0.0))

    def test_invalid_loop_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            CORE.make_sgd(CORE.teaching_module(), learning_rate=0.0)
        with self.assertRaises(ValueError):
            CORE.train_model(epochs=0)
        with self.assertRaises(ValueError):
            CORE.train_model(batch_size=0)
        with self.assertRaises(ValueError):
            CORE.training_step(
                CORE.teaching_module(),
                *CORE.teaching_batch(),
                CORE.make_sgd(CORE.teaching_module(), learning_rate=0.1),
                defect="train_mode_eval",
            )
        with self.assertRaises(ValueError):
            CORE.evaluate(CORE.teaching_module(), *CORE.teaching_batch(), defect="stale_grad")
        with self.assertRaises(ValueError):
            CORE.evaluate(CORE.teaching_module(), *CORE.teaching_batch(), split="test")


if __name__ == "__main__":
    unittest.main()
