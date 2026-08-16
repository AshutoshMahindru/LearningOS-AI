from __future__ import annotations

import ast
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import unittest

try:
    import yaml
except ModuleNotFoundError:  # The bare repository runtime has no third-party dependencies.
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M01"
NOTEBOOK = ROOT / "labs" / "M01_ai_ml_landscape.ipynb"


def load_manifest() -> dict[str, object]:
    """Load the manifest with PyYAML, with a narrow fallback for bare CI discovery."""
    text = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)

    def scalar(value: str) -> object:
        """Convert the boolean scalars needed by the manifest contract."""
        if value == "true":
            return True
        if value == "false":
            return False
        return value

    parsed: dict[str, object] = {}
    section: dict[str, object] | None = None
    for line in text.splitlines():
        if line and not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            if value.strip():
                parsed[key] = scalar(value.strip())
                section = None
            else:
                section = {}
                parsed[key] = section
        elif section is not None and line.startswith("  ") and not line.lstrip().startswith("-"):
            key, value = line.strip().split(":", 1)
            section[key] = scalar(value.strip())
    return parsed


class M01MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Load and execute every source code cell once for behavioral assertions."""
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.runtime = {"__name__": "__m01_notebook__"}
        with redirect_stdout(StringIO()):
            for cell in cls.notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile(cell["source"], f"{NOTEBOOK.name}:{cell['id']}", "exec"), cls.runtime)

    def test_required_artifacts_exist(self) -> None:
        """Require every mission-contract artifact at its designated path."""
        required = [
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
        ]
        for name in required:
            self.assertTrue((MISSION / name).is_file(), name)
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue((ROOT / "datasets" / "M01" / "README.md").is_file())
        self.assertTrue((ROOT / "requirements" / "m01.txt").is_file())

    def test_manifest_declares_identity_and_offline_execution_boundary(self) -> None:
        """Validate structured identity and offline execution declarations."""
        manifest = load_manifest()
        self.assertEqual(manifest["id"], "M01")
        self.assertEqual(manifest["phase"], "P0")
        self.assertEqual(manifest["flagship"], "V00")
        self.assertEqual(
            manifest["execution"],
            {
                "cpu_only": True,
                "deterministic_where_practical": True,
                "requires_secrets": False,
                "requires_paid_api": False,
                "network_required": False,
            },
        )

    def test_evidence_contract_requires_complete_system_map_without_prefilled_evidence(self) -> None:
        """Require whole-system evidence while forbidding fabricated learner work."""
        text = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertNotIn("learner_evidence:", text)
        self.assertNotIn("learner_response:", text)
        for layer in [
            "data",
            "features_or_representations",
            "training",
            "model_state",
            "inference",
            "predictions",
            "retrieval",
            "tools",
            "memory",
            "evaluation",
            "observability",
            "compute_infrastructure",
        ]:
            self.assertRegex(text, rf"(?m)^\s+- {re.escape(layer)}$")

    def test_source_notebook_has_stable_ids_no_outputs_and_no_network_code(self) -> None:
        """Enforce stable, output-free, network-free notebook source."""
        notebook = self.notebook
        self.assertEqual(notebook["nbformat"], 4)
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        self.assertTrue(all(cell_ids))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"], cell["id"])
                self.assertEqual(cell["outputs"], [], cell["id"])
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        tree = ast.parse(code)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        forbidden_roots = {"aiohttp", "http", "httpx", "openai", "requests", "socket", "urllib"}
        self.assertFalse(imported_roots & forbidden_roots, imported_roots & forbidden_roots)
        self.assertNotIn("api_key", code.lower())
        self.assertNotIn("os.environ", code)
        for required_symbol in [
            "train_classifier",
            "predict",
            "embed",
            "retrieve",
            "run_application",
            "evaluate",
        ]:
            self.assertIn(required_symbol, code)

    def test_every_experiment_places_prediction_before_action(self) -> None:
        """Ensure all declared experiments require a written prediction first."""
        experiments = (MISSION / "experiments.yaml").read_text(encoding="utf-8")
        declared = re.findall(r"(?m)^  - id: (E\d+)$", experiments)
        self.assertEqual(declared, ["E1", "E2", "E3", "E4", "E5"])
        tagged = [
            (cell.get("metadata", {}).get("experiment_id"), cell.get("metadata", {}).get("role"))
            for cell in self.notebook["cells"]
            if cell.get("metadata", {}).get("experiment_id")
        ]
        for experiment_id in declared:
            prediction = tagged.index((experiment_id, "prediction"))
            action = tagged.index((experiment_id, "action"))
            self.assertLess(prediction, action, experiment_id)
        prediction_cells = [
            cell for cell in self.notebook["cells"]
            if cell.get("metadata", {}).get("role") == "prediction"
        ]
        self.assertTrue(all("write" in cell["source"].lower() for cell in prediction_cells))

    def test_inference_produces_predictions_without_changing_model_state(self) -> None:
        """Prove inference returns scores while leaving learned state unchanged."""
        train_classifier = self.runtime["train_classifier"]
        predict = self.runtime["predict"]
        digest = self.runtime["digest"]
        model = train_classifier(deepcopy(self.runtime["TRAIN"]))
        before = digest(model)
        label, scores = predict(model, "duplicate invoice charge")
        self.assertEqual(label, "billing")
        self.assertEqual(set(scores), set(model["labels"]))
        self.assertEqual(digest(model), before)

    def test_retraining_creates_new_model_state(self) -> None:
        """Prove a labelled-data change plus retraining creates new state."""
        train_classifier = self.runtime["train_classifier"]
        digest = self.runtime["digest"]
        original = train_classifier(deepcopy(self.runtime["TRAIN"]))
        changed_rows = deepcopy(self.runtime["TRAIN"])
        changed_rows.append({"id": 99, "priority": "normal", "text": "refund invoice", "label": "billing"})
        retrained = train_classifier(changed_rows)
        self.assertNotEqual(digest(original), digest(retrained))
        self.assertEqual(original["docs"]["billing"] + 1, retrained["docs"]["billing"])

    def test_retrieval_changes_context_without_changing_model_state(self) -> None:
        """Prove retrieval selects external context without retraining."""
        digest = self.runtime["digest"]
        model = self.runtime["MODEL"]
        before = digest(model)
        expected = {
            "duplicate card charge invoice": "K2",
            "password account recovery": "K1",
            "upload file error": "K3",
        }
        observed = {query: self.runtime["retrieve"](query)[0]["id"] for query in expected}
        self.assertEqual(observed, expected)
        self.assertEqual(digest(model), before)

    def test_controller_owns_branch_tool_executes_and_memory_is_not_weights(self) -> None:
        """Separate controller policy, tool execution, memory, and weights."""
        run_application = self.runtime["run_application"]
        model = deepcopy(self.runtime["MODEL"])
        digest = self.runtime["digest"]
        before = digest(model)
        memory: dict[str, object] = {}
        normal = run_application({"id": 301, "priority": "normal", "text": "reset account password"}, model, memory)
        urgent = run_application({"id": 302, "priority": "urgent", "text": "duplicate invoice charge"}, model, memory)
        self.assertIsNone(normal["tool"])
        self.assertEqual(urgent["tool"]["tool"], "escalation")
        self.assertIn("tool", [event[0] for event in urgent["trace"]])
        self.assertEqual(memory["runs"], 2)
        self.assertEqual(memory["last_ticket"], 302)
        self.assertEqual(digest(model), before)

    def test_system_map_separates_evaluation_from_observability(self) -> None:
        """Require all system layers and distinct evaluation/observability rows."""
        layers = {row[1] for row in self.runtime["SYSTEM_MAP"]}
        expected = {
            "data",
            "features/representations",
            "training",
            "model",
            "inference",
            "predictions",
            "retrieval",
            "tool",
            "agent/controller",
            "memory/state",
            "evaluation",
            "observability",
            "compute/infrastructure",
        }
        self.assertTrue(expected.issubset(layers), expected - layers)
        evaluation = self.runtime["evaluate"](self.runtime["MODEL"], self.runtime["TEST"])
        self.assertEqual(evaluation["accuracy"], 1.0)
        self.assertEqual(evaluation["correct"], [True, True, True])

    def test_controlled_failure_and_assessment_require_boundary_reasoning(self) -> None:
        """Require diagnosis and transfer reasoning instead of vocabulary recall."""
        failure = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        for distinction in ["model weights", "retrieval", "tool", "control flow", "observability", "objective"]:
            self.assertIn(distinction, failure)
        assessment = (MISSION / "assessment.yaml").read_text(encoding="utf-8").lower()
        self.assertIn("assessment_type: transfer", assessment)
        self.assertIn("without a system map", assessment)
        self.assertIn("state whether the scenario describes model training", assessment)
        self.assertIn("refuses to invent a training step", assessment)


if __name__ == "__main__":
    unittest.main()
