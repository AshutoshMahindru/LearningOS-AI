from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M26"
NOTEBOOK = ROOT / "labs" / "M26_deep_learning_failure.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m26.txt"


def load_core():
    spec = importlib.util.spec_from_file_location("m26_dl_failure_lab", MISSION / "dl_failure_lab.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M26 dl failure lab")
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


class M26StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "dl_failure_lab.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m26.py",
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
            "dl_failure_lab.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M26/{name}", manifest)

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
            "tokenizer",
            "tiktoken",
            "huggingface",
            "from_pretrained",
            "sentence_transformers",
            "transformer",
            "attention",
            "embedding",
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
        self.assertIn("M26", source)
        self.assertIn("dl_failure_lab.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M26.dl_failure_lab import", source)
        self.assertIn("from missions.M25.pytorch_training import", source)

    def test_future_mission_boundary_stays_closed(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        ).lower()
        for forbidden in (
            "tokenizer",
            "tiktoken",
            "embedding",
            "attention",
            "transformer",
            "from_pretrained",
            "huggingface",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        core = (MISSION / "dl_failure_lab.py").read_text(encoding="utf-8").lower()
        self.assertIn("_load_m25", core)
        self.assertIn("train_model", core)
        for forbidden in ("tokenizer", "tiktoken", "embedding", "attention", "transformer"):
            with self.subTest(core=forbidden):
                self.assertNotIn(forbidden, core)

        markdown = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "markdown"
        )
        self.assertIn("P5", markdown)
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        self.assertIn("deferred_to_p5", content)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-healthy", "run-healthy"),
            ("predict-hidden", "run-hidden"),
            ("predict-hidden-disc", "run-hidden-disc"),
            ("predict-data", "run-data"),
            ("predict-scale", "run-scale"),
            ("predict-lr", "run-lr"),
            ("predict-gradflow", "run-gradflow"),
            ("predict-capacity", "run-capacity"),
            ("predict-reg", "run-reg"),
            ("predict-eval", "run-eval"),
            ("predict-hidden-repair", "run-hidden-repair"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-chaos", "run-chaos"),
            ("predict-chaos-disc", "run-chaos-disc"),
            ("predict-chaos-repair", "run-chaos-repair"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                self.assertIn("Predict before running", cell_source(cells[positions[prediction]]))

        markdown = "\n".join(
            cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 15)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M25 → M26 → P5",
            "zero_grad",
            "held-out",
            "smallest repair",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

        hidden_predict = cell_source(cells[positions["predict-hidden"]]) + cell_source(
            cells[positions["predict-chaos"]]
        )
        for spoiler in ("frozen_layer", "tiny_hidden", "label_shuffle", "fc1 is frozen", "width is 1"):
            with self.subTest(spoiler=spoiler):
                self.assertNotIn(spoiler, hidden_predict)

    def test_notebook_prints_required_diagnosis_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "known_good_m25_trace",
            "prepare_fault",
            "run_prepared",
            "public_symptoms",
            "diagnostic_battery",
            "rank_hypotheses",
            "repair_prepared",
            "repair_and_verify",
            "chaos_day",
            "cluster_label_agreement",
            "feature_column_scales",
            "tiny_subset_overfit_check",
            'prepare_fault("label_shuffle"',
            'prepare_fault("feature_scale"',
            'prepare_fault("lr_high"',
            'prepare_fault("lr_low"',
            'prepare_fault("frozen_layer"',
            'prepare_fault("tiny_hidden"',
            'prepare_fault("crushing_dropout"',
            'prepare_fault("train_mode_eval"',
            'prepare_fault("val_leakage"',
            "inspect.getsource",
            "LOOP_ORDER",
            "reveal=False",
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
        source = (MISSION / "dl_failure_lab.py").read_text(encoding="utf-8")
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

    def test_m25_loop_is_imported_not_rewritten(self):
        source = (MISSION / "dl_failure_lab.py").read_text(encoding="utf-8")
        self.assertIn("_load_m25", source)
        self.assertIn("train_model", source)
        tree = ast.parse(source)
        defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("training_step", defined)
        self.assertNotIn("train_model", defined)
        self.assertNotIn("evaluate", defined)
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        self.assertNotIn("TwoLayerNet", classes)

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
        self.assertIn(CORE.UNFILLED, CORE.empty_diagnosis_record().values())
        self.assertFalse(CORE.diagnosis_record_complete(CORE.empty_diagnosis_record()))

    def test_mission_package_declares_m25_handoff_and_p5_deferral(self):
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        self.assertIn("M25", manifest)
        self.assertIn("P5", manifest)
        self.assertIn("language_model_mechanisms_deferred_to_p5", manifest)
        self.assertIn("m25_loop_not_rewritten", manifest)
        self.assertIn("deferred_to_p5", content)
        self.assertIn("repair_from_broken_objects", manifest)

    def test_unknown_and_future_defects_are_rejected(self):
        self.assertEqual(CORE._normalize_defect("data_corruption"), "label_shuffle")
        self.assertEqual(CORE._normalize_defect("learning_rate_failure"), "lr_high")
        self.assertEqual(CORE._normalize_defect("eval_in_train_mode"), "train_mode_eval")
        self.assertEqual(CORE._normalize_defect("correct"), "none")
        with self.assertRaises(ValueError):
            CORE._normalize_defect("tokenization")
        with self.assertRaises(ValueError):
            CORE._normalize_defect("attention_dropout")

    def test_chaos_mapping_is_seeded_and_hidden_repr_redacts_category(self):
        self.assertEqual(CORE.defect_for_seed(CORE.HIDDEN_PRACTICE_SEED), "frozen_layer")
        self.assertEqual(CORE.defect_for_seed(CORE.CHAOS_SEED), "tiny_hidden")
        self.assertNotEqual(
            CORE.category_for_defect(CORE.defect_for_seed(CORE.HIDDEN_PRACTICE_SEED)),
            CORE.category_for_defect(CORE.defect_for_seed(CORE.CHAOS_SEED)),
        )
        families = CORE.chaos_families()
        self.assertEqual(len(families), 5)
        categories = {CORE.category_for_defect(name) for name in families}
        self.assertEqual(categories, {"data", "optimization", "gradient_flow", "architecture", "evaluation"})
        hidden = CORE.PreparedFault(
            defect="tiny_hidden",
            category="architecture",
            seed=CORE.CHAOS_SEED,
            split_seed=CORE.DEFAULT_SPLIT_SEED,
            learning_rate=CORE.DEFAULT_LEARNING_RATE,
            momentum=CORE.DEFAULT_MOMENTUM,
            dropout_p=0.0,
            n_hidden=1,
            epochs=8,
            batch_size=8,
            splits=object(),
            clean_labels=object(),
            clean_features=object(),
            hidden=True,
            original_defect="tiny_hidden",
        )
        shown = repr(hidden)
        self.assertNotIn("tiny_hidden", shown)
        self.assertNotIn("architecture", shown)
        self.assertIn("hidden=True", shown)

    def test_rank_hypotheses_uses_discriminating_signals(self):
        labels = CORE.rank_hypotheses(
            {
                "labels_disagree_with_clusters": True,
                "features_misfcaled": False,
                "loss_oscillates": False,
                "fc1_blocked": False,
                "fc1_moved": True,
                "fc2_moved": True,
                "tiny_restored_overfit": True,
                "tiny_current_overfit": False,
                "train_loss_flat": False,
                "claimed_model_training": False,
                "claimed_n_is_train_sized": False,
            }
        )
        self.assertEqual(labels[0], "labels_corrupted")
        blocked = CORE.rank_hypotheses(
            {
                "fc1_blocked": True,
                "fc1_moved": False,
                "fc2_moved": True,
                "labels_disagree_with_clusters": False,
                "features_misfcaled": False,
                "loss_oscillates": False,
                "tiny_restored_overfit": True,
            }
        )
        self.assertEqual(blocked[0], "blocked_gradient_path")
        leaked = CORE.rank_hypotheses({"claimed_n_is_train_sized": True, "claimed_val_better_than_held_out": True})
        self.assertEqual(leaked[0], "eval_split_leakage")


@unittest.skipUnless(TORCH_AVAILABLE, "install requirements/m26.txt to run PyTorch-dependent M26 tests")
class M26TorchRuntimeTests(unittest.TestCase):
    def test_known_good_m25_trace_holds_healthy_invariants(self):
        diag = CORE.known_good_m25_trace()
        self.assertTrue(CORE.healthy_invariants_hold(diag), CORE.invariant_failures(diag))
        self.assertLess(diag.train_losses[-1], 0.5 * diag.train_losses[0])
        self.assertFalse(diag.held_out.model_training)
        self.assertEqual(diag.train_run.device, "cpu")
        self.assertEqual(diag.fault.defect, "none")
        self.assertTrue(all(diag.layer_moved.values()))

    def test_each_named_defect_has_its_signature(self):
        for defect in (
            "label_shuffle",
            "feature_scale",
            "lr_high",
            "lr_low",
            "frozen_layer",
            "tiny_hidden",
            "crushing_dropout",
            "train_mode_eval",
            "val_leakage",
        ):
            with self.subTest(defect=defect):
                fault = CORE.prepare_fault(defect, seed=CORE.DEFAULT_SEED)
                diag = CORE.run_prepared(fault)
                failures = set(CORE.invariant_failures(diag))
                expected = set(CORE.signature_for_defect(defect))
                self.assertTrue(expected <= failures, (defect, expected, failures))
                self.assertNotEqual(failures, set())

    def test_label_shuffle_is_train_only_and_repair_restores_same_tensor(self):
        fault = CORE.prepare_fault("label_shuffle", seed=CORE.DEFAULT_SEED)
        labels_obj = fault.splits.labels
        broken_agree = CORE.cluster_label_agreement(fault.splits)
        self.assertLess(broken_agree["train_agreement"], 0.7)
        self.assertGreaterEqual(broken_agree["val_agreement"], 0.99)
        self.assertGreaterEqual(broken_agree["held_out_agreement"], 0.99)
        diag = CORE.run_prepared(fault)
        self.assertIn("labels_match_clusters", CORE.invariant_failures(diag))
        repaired = CORE.repair_and_verify(diag)
        self.assertTrue(repaired["retrained"])
        self.assertTrue(repaired["same_splits"])
        self.assertIs(fault.splits.labels, labels_obj)
        self.assertEqual(repaired["failures"], ())
        self.assertGreaterEqual(CORE.cluster_label_agreement(fault.splits)["train_agreement"], 0.99)

    def test_feature_scale_repair_divides_the_same_column(self):
        fault = CORE.prepare_fault("feature_scale", seed=CORE.DEFAULT_SEED)
        features_obj = fault.splits.features
        self.assertGreaterEqual(CORE.feature_scale_ratio(fault.splits), 10.0)
        diag = CORE.run_prepared(fault)
        self.assertIn("features_comparable_scale", CORE.invariant_failures(diag))
        repaired = CORE.repair_and_verify(diag)
        self.assertIs(fault.splits.features, features_obj)
        self.assertLess(CORE.feature_scale_ratio(fault.splits), 10.0)
        self.assertEqual(repaired["failures"], ())

    def test_learning_rate_repairs_the_same_config_object(self):
        high = CORE.prepare_fault("lr_high", seed=CORE.DEFAULT_SEED)
        self.assertEqual(high.learning_rate, CORE.HIGH_LEARNING_RATE)
        high_diag = CORE.run_prepared(high)
        self.assertTrue(CORE._loss_oscillates(high_diag.train_losses))
        high_repair = CORE.repair_and_verify(high_diag)
        self.assertIs(high_repair["diag"].fault, high)
        self.assertEqual(high.learning_rate, CORE.DEFAULT_LEARNING_RATE)
        self.assertEqual(high_repair["failures"], ())

        low = CORE.prepare_fault("lr_low", seed=CORE.DEFAULT_SEED)
        low_diag = CORE.run_prepared(low)
        self.assertLess(abs(low_diag.train_losses[-1] - low_diag.train_losses[0]), 0.05)
        low_repair = CORE.repair_and_verify(low_diag)
        self.assertIs(low_repair["diag"].fault, low)
        self.assertEqual(low.learning_rate, CORE.DEFAULT_LEARNING_RATE)
        self.assertEqual(low_repair["failures"], ())

    def test_frozen_layer_localizes_fc1_and_unfreezes_the_same_hook(self):
        fault = CORE.prepare_fault("frozen_layer", seed=CORE.DEFAULT_SEED)
        diag = CORE.run_prepared(fault)
        self.assertFalse(diag.layer_moved["fc1.weight"])
        self.assertTrue(diag.layer_moved["fc2.weight"])
        self.assertFalse(diag.grad_report["fc1_requires_grad"])
        self.assertTrue(fault.freeze_fc1)
        repaired = CORE.repair_and_verify(diag)
        self.assertFalse(fault.freeze_fc1)
        self.assertTrue(repaired["diag"].grad_report["fc1_requires_grad"])
        self.assertTrue(repaired["diag"].layer_moved["fc1.weight"])
        self.assertEqual(repaired["failures"], ())

    def test_tiny_width_underfits_until_the_same_config_is_widened(self):
        fault = CORE.prepare_fault("tiny_hidden", seed=CORE.DEFAULT_SEED)
        self.assertEqual(fault.n_hidden, CORE.TINY_HIDDEN)
        diag = CORE.run_prepared(fault)
        self.assertIn("fits_training_data", CORE.invariant_failures(diag))
        self.assertGreater(diag.train_losses[-1], 0.25)
        repaired = CORE.repair_and_verify(diag)
        self.assertEqual(fault.n_hidden, CORE.TRAIN_HIDDEN)
        self.assertEqual(repaired["failures"], ())
        self.assertLess(repaired["diag"].train_losses[-1], 0.2)

    def test_evaluation_repairs_keep_the_checkpoint(self):
        mode = CORE.prepare_fault("train_mode_eval", seed=CORE.DEFAULT_SEED)
        mode_diag = CORE.run_prepared(mode)
        self.assertTrue(mode_diag.claimed_val.model_training)
        self.assertFalse(mode_diag.honest_val.model_training)
        mode_repair = CORE.repair_and_verify(mode_diag)
        self.assertFalse(mode_repair["retrained"])
        self.assertTrue(mode_repair["parameters_unchanged"])
        self.assertFalse(mode_repair["claimed"].model_training)
        self.assertEqual(mode_repair["failures"], ())

        leak = CORE.prepare_fault("val_leakage", seed=CORE.DEFAULT_SEED)
        leak_diag = CORE.run_prepared(leak)
        self.assertEqual(leak_diag.claimed_val.n, 24)
        self.assertEqual(leak_diag.honest_val.n, 6)
        leak_repair = CORE.repair_and_verify(leak_diag)
        self.assertFalse(leak_repair["retrained"])
        self.assertTrue(leak_repair["parameters_unchanged"])
        self.assertEqual(leak_repair["claimed"].n, 6)
        self.assertEqual(leak_repair["failures"], ())

    def test_chaos_day_hides_category_and_repairs_from_the_payload(self):
        payload = CORE.chaos_day(CORE.CHAOS_SEED, reveal=False)
        self.assertNotIn("defect", payload["symptoms"])
        self.assertNotIn("category", payload)
        self.assertNotIn("n_hidden", payload["symptoms"])
        self.assertNotIn("learning_rate", payload["symptoms"])
        self.assertIn("train_losses", payload["symptoms"])
        self.assertTrue(payload["fault"].hidden)
        self.assertIn("tiny_hidden", repr(payload["fault"].original_defect))
        self.assertNotIn("tiny_hidden", repr(payload["fault"]))
        failures = CORE.invariant_failures(payload["diag"])
        self.assertTrue(set(CORE.signature_for_defect("tiny_hidden")) <= set(failures))
        repaired = CORE.repair_and_verify(payload["diag"])
        self.assertEqual(repaired["failures"], ())
        revealed = CORE.chaos_day(CORE.CHAOS_SEED, reveal=True)
        self.assertEqual(revealed["defect"], "tiny_hidden")

    def test_tiny_subset_overfit_distinguishes_low_lr_from_healthy_data(self):
        healthy = CORE.prepare_fault("none")
        ok = CORE.tiny_subset_overfit_check(healthy.splits, learning_rate=CORE.DEFAULT_LEARNING_RATE)
        self.assertTrue(ok["overfit"], ok)
        stuck = CORE.tiny_subset_overfit_check(healthy.splits, learning_rate=CORE.LOW_LEARNING_RATE)
        self.assertFalse(stuck["overfit"], stuck)

    def test_regression_flags_catch_recurrence_without_a_healthy_twin(self):
        frozen = CORE.run_prepared(CORE.prepare_fault("frozen_layer"))
        self.assertIn("fc1_receiving_gradients", CORE.invariant_failures(frozen))
        labels = CORE.run_prepared(CORE.prepare_fault("label_shuffle"))
        self.assertIn("labels_match_clusters", CORE.invariant_failures(labels))
        self.assertFalse(CORE.healthy_invariants_hold(frozen))
        self.assertFalse(CORE.healthy_invariants_hold(labels))

    def test_invalid_loop_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            CORE.prepare_fault("not_a_defect")
        with self.assertRaises(ValueError):
            CORE.cheapest_discriminator("not_a_hypothesis")


if __name__ == "__main__":
    unittest.main()
