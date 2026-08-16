from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M20"
NOTEBOOK = ROOT / "labs" / "M20_optimization_experimentally.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m20.txt"


def load_core():
    path = MISSION / "optimization_core.py"
    spec = importlib.util.spec_from_file_location("m20_optimization_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M20 optimization core")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during module load.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()


def cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(str(part) for part in source)


def manifest_artifacts(text: str) -> set[str]:
    lines = text.splitlines()
    start = lines.index("mission_artifacts:") + 1
    artifacts: set[str] = set()
    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        match = re.fullmatch(r"  - (.+)", line)
        if match:
            artifacts.add(match.group(1))
    return artifacts


class M20OptimizationMechanicsTests(unittest.TestCase):
    def test_objective_and_gradient_are_transparent_and_exact(self) -> None:
        self.assertEqual(CORE.DEFAULT_CURVATURES, (1.0, 10.0))
        self.assertEqual(CORE.DEFAULT_INITIALIZATION, (4.0, 4.0))
        self.assertEqual(CORE.quadratic_loss((4.0, 4.0)), 88.0)
        self.assertEqual(CORE.quadratic_gradient((4.0, 4.0)), (4.0, 40.0))
        self.assertEqual(CORE.quadratic_loss((0.0, 0.0)), 0.0)

        point = (1.25, -0.75)
        epsilon = 1.0e-6
        analytic = CORE.quadratic_gradient(point)
        for index in range(2):
            plus = list(point)
            minus = list(point)
            plus[index] += epsilon
            minus[index] -= epsilon
            estimate = (
                CORE.quadratic_loss(plus) - CORE.quadratic_loss(minus)
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(estimate, analytic[index], places=7)

    def test_learning_rate_regimes_are_controlled_and_observable(self) -> None:
        rates = (1.0e-5, 0.05, 0.19, 0.21)
        traces = CORE.run_learning_rate_sweep(rates, steps=40)
        self.assertEqual(set(traces), set(rates))

        expected = {
            1.0e-5: "stagnating",
            0.05: "converging",
            0.19: "oscillatory_convergence",
            0.21: "diverging",
        }
        self.assertEqual(
            {rate: CORE.diagnose_dynamics(trace) for rate, trace in traces.items()},
            expected,
        )
        for trace in traces.values():
            self.assertEqual(trace[0].parameters_before, CORE.DEFAULT_INITIALIZATION)
            self.assertEqual(trace[0].loss_before, 88.0)
            self.assertEqual(trace[0].gradient, (4.0, 40.0))
            self.assertTrue(all(record.optimizer == "gd" for record in trace))

    def test_stability_boundary_separates_decaying_and_expanding_oscillation(self) -> None:
        below = CORE.run_optimizer("gd", learning_rate=0.19, steps=20)
        above = CORE.run_optimizer("gd", learning_rate=0.21, steps=20)

        below_values = (below[0].parameters_before[1],) + tuple(
            record.parameters_after[1] for record in below
        )
        above_values = (above[0].parameters_before[1],) + tuple(
            record.parameters_after[1] for record in above
        )
        self.assertEqual(CORE.coordinate_sign_changes(below, 1), 20)
        self.assertEqual(CORE.coordinate_sign_changes(above, 1), 20)
        self.assertTrue(
            all(abs(after) < abs(before) for before, after in zip(below_values, below_values[1:]))
        )
        self.assertTrue(
            all(abs(after) > abs(before) for before, after in zip(above_values, above_values[1:]))
        )
        self.assertLess(below[-1].loss_after, below[0].loss_before)
        self.assertGreater(above[-1].loss_after, above[0].loss_before)

    def test_too_small_rate_has_nonzero_gradient_but_stagnates_over_budget(self) -> None:
        trace = CORE.run_optimizer("gd", learning_rate=1.0e-5, steps=40)
        losses = CORE.loss_history(trace)
        relative_improvement = (losses[0] - losses[-1]) / losses[0]
        self.assertGreater(sum(abs(value) for value in trace[0].gradient), 0)
        self.assertEqual(trace[0].applied_update, (4.0e-5, 4.0e-4))
        self.assertLess(relative_improvement, 0.01)
        self.assertEqual(CORE.diagnose_dynamics(trace), "stagnating")

    def test_seeded_sgd_preserves_mean_gradient_and_replays_exactly(self) -> None:
        self.assertEqual(CORE.component_noise_mean(), (0.0, 0.0))
        point = (2.5, -1.5)
        exact = CORE.quadratic_gradient(point)
        component_gradients = [
            tuple(value + noise for value, noise in zip(exact, row))
            for row in CORE.COMPONENT_GRADIENT_NOISE
        ]
        mean_gradient = tuple(
            sum(row[index] for row in component_gradients) / len(component_gradients)
            for index in range(2)
        )
        self.assertEqual(mean_gradient, exact)

        first = CORE.run_optimizer("sgd", learning_rate=0.05, steps=24, seed=2020)
        replay = CORE.run_optimizer("sgd", learning_rate=0.05, steps=24, seed=2020)
        other_seed = CORE.run_optimizer("sgd", learning_rate=0.05, steps=24, seed=2021)
        self.assertEqual(first, replay)
        self.assertNotEqual(first, other_seed)
        self.assertTrue(
            all(record.gradient_source.startswith("component:") for record in first)
        )
        for start in range(0, 24, 4):
            epoch_sources = {record.gradient_source for record in first[start : start + 4]}
            self.assertEqual(epoch_sources, {"component:0", "component:1", "component:2", "component:3"})

    def test_momentum_state_is_visible_and_can_overshoot(self) -> None:
        trace = CORE.run_optimizer(
            "momentum", learning_rate=0.05, momentum=0.9, steps=40
        )
        self.assertEqual(trace[0].gradient, (4.0, 40.0))
        self.assertEqual(trace[0].optimizer_direction, (4.0, 40.0))
        self.assertEqual(trace[0].parameters_after, (3.8, 2.0))
        self.assertEqual(trace[1].gradient, (3.8, 20.0))
        self.assertEqual(trace[1].optimizer_direction, (7.4, 56.0))
        self.assertAlmostEqual(trace[1].parameters_after[0], 3.43)
        self.assertAlmostEqual(trace[1].parameters_after[1], -0.8)

        losses = CORE.loss_history(trace)
        self.assertTrue(any(after > before for before, after in zip(losses, losses[1:])))
        self.assertGreater(CORE.coordinate_sign_changes(trace, 1), 0)

    def test_adam_bias_correction_exposes_scale_adaptation(self) -> None:
        trace = CORE.run_optimizer("adam", learning_rate=0.05, steps=2)
        self.assertEqual(trace[0].gradient, (4.0, 40.0))
        self.assertAlmostEqual(trace[0].optimizer_direction[0], 1.0, places=7)
        self.assertAlmostEqual(trace[0].optimizer_direction[1], 1.0, places=7)
        self.assertAlmostEqual(trace[0].applied_update[0], 0.05, places=8)
        self.assertAlmostEqual(trace[0].applied_update[1], 0.05, places=8)
        self.assertAlmostEqual(trace[0].parameters_after[0], 3.95, places=8)
        self.assertAlmostEqual(trace[0].parameters_after[1], 3.95, places=8)

    def test_optimizer_ranking_changes_with_horizon_and_tuning(self) -> None:
        gd_5 = CORE.run_optimizer("gd", learning_rate=0.05, steps=5)
        momentum_5 = CORE.run_optimizer("momentum", learning_rate=0.05, steps=5)
        gd_40 = CORE.run_optimizer("gd", learning_rate=0.05, steps=40)
        momentum_40 = CORE.run_optimizer("momentum", learning_rate=0.05, steps=40)
        adam_same_rate = CORE.run_optimizer("adam", learning_rate=0.05, steps=40)
        adam_tuned_fixture = CORE.run_optimizer("adam", learning_rate=0.30, steps=40)

        self.assertLess(gd_5[-1].loss_after, momentum_5[-1].loss_after)
        self.assertLess(momentum_40[-1].loss_after, gd_40[-1].loss_after)
        self.assertGreater(adam_same_rate[-1].loss_after, gd_40[-1].loss_after)
        self.assertLess(adam_tuned_fixture[-1].loss_after, gd_40[-1].loss_after)

    def test_trace_preserves_gradient_state_update_parameter_and_loss_chain(self) -> None:
        for optimizer in ("gd", "sgd", "momentum", "adam"):
            with self.subTest(optimizer=optimizer):
                trace = CORE.run_optimizer(
                    optimizer, learning_rate=0.05, steps=8, seed=2020
                )
                self.assertEqual(len(trace), 8)
                for index, record in enumerate(trace):
                    self.assertEqual(record.step, index)
                    expected_update = tuple(
                        0.05 * value for value in record.optimizer_direction
                    )
                    for observed, expected in zip(record.applied_update, expected_update):
                        self.assertAlmostEqual(observed, expected)
                    expected_after = tuple(
                        value - update
                        for value, update in zip(
                            record.parameters_before, record.applied_update
                        )
                    )
                    for observed, expected in zip(record.parameters_after, expected_after):
                        self.assertAlmostEqual(observed, expected)
                    self.assertAlmostEqual(
                        record.loss_after,
                        CORE.quadratic_loss(record.parameters_after),
                    )
                    if index:
                        self.assertEqual(
                            record.parameters_before, trace[index - 1].parameters_after
                        )
                        self.assertEqual(record.loss_before, trace[index - 1].loss_after)

    def test_invalid_inputs_fail_loudly_instead_of_silently_changing_scope(self) -> None:
        invalid_calls = [
            lambda: CORE.quadratic_loss((), ()),
            lambda: CORE.quadratic_loss((1.0,), (1.0, 2.0)),
            lambda: CORE.quadratic_loss((1.0,), (0.0,)),
            lambda: CORE.run_optimizer("unknown", learning_rate=0.1, steps=1),
            lambda: CORE.run_optimizer("gd", learning_rate=0.0, steps=1),
            lambda: CORE.run_optimizer("gd", learning_rate=0.1, steps=0),
            lambda: CORE.run_optimizer("momentum", learning_rate=0.1, steps=1, momentum=1.0),
            lambda: CORE.run_optimizer("adam", learning_rate=0.1, steps=1, beta1=1.0),
            lambda: CORE.run_optimizer("adam", learning_rate=0.1, steps=1, epsilon=0.0),
            lambda: CORE.run_optimizer(
                "sgd",
                learning_rate=0.1,
                steps=1,
                stochastic_noise=((1.0, 0.0),),
            ),
            lambda: CORE.run_learning_rate_sweep((0.1, 0.1), steps=2),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()
        with self.assertRaises(IndexError):
            CORE.coordinate_sign_changes(
                CORE.run_optimizer("gd", learning_rate=0.05, steps=2), 5
            )


class M20MissionContractTests(unittest.TestCase):
    REQUIRED_MISSION_FILES = {
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
        "review_brief.md",
        "adr_prompt.md",
        "status.yaml",
        "optimization_core.py",
    }

    REQUIRED_OWNED_PATHS = {
        *(f"missions/M20/{name}" for name in REQUIRED_MISSION_FILES),
        "labs/M20_optimization_experimentally.ipynb",
        "requirements/m20.txt",
        "tests/missions/test_m20.py",
        "tests/test_m20.py",
    }

    def test_complete_mission_package_is_declared_present_and_substantial(self) -> None:
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(self.REQUIRED_MISSION_FILES <= actual)
        for name in self.REQUIRED_MISSION_FILES - {"optimization_core.py"}:
            text = (MISSION / name).read_text(encoding="utf-8")
            self.assertGreater(len(text), 250, name)

        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        self.assertEqual(manifest_artifacts(manifest), self.REQUIRED_OWNED_PATHS)
        for relative in self.REQUIRED_OWNED_PATHS:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_manifest_encodes_mission_identity_runtime_controls_and_scope(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        expected = [
            "id: M20",
            "title: Understand Optimization Experimentally",
            "phase: P3",
            "flagship: V04",
            "closes_flagship: V04",
            "formal_engineering_review: true",
            "- M19",
            "provides: M21",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "source_notebook_outputs: empty",
            "learner_evidence_prepopulated: false",
            "shared_registry_edits_required: false",
            "modifies_lab_status: false",
            "modifies_tracking: false",
            "modifies_root_readme: false",
            "modifies_workflows: false",
            "modifies_shared_validators: false",
            "modifies_shared_registries: false",
        ]
        for fragment in expected:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, manifest)

    def test_experiment_contract_preserves_controls_and_prediction_order(self) -> None:
        experiments = (MISSION / "experiments.yaml").read_text(encoding="utf-8")
        for experiment_id in [
            "learning-rate-regimes",
            "stability-boundary",
            "gd-versus-sgd",
            "momentum-trade-off",
            "adam-trade-off",
            "too-large-rate-failure",
        ]:
            self.assertIn(f"id: {experiment_id}", experiments)
        self.assertEqual(experiments.count("prediction_required: true"), 6)
        for control in [
            "objective: fixed anisotropic quadratic",
            "initialization: [4.0, 4.0]",
            "prediction_required_before_action: true",
            "Same aggregate objective, initialization, learning rate, and update budget.",
        ]:
            self.assertIn(control, experiments)

    def test_notebook_is_valid_clean_unique_substantial_and_offline(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 40)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        self.assertGreaterEqual(len(code_cells), 15)
        for cell in code_cells:
            self.assertIsNone(cell.get("execution_count"), cell["id"])
            self.assertEqual(cell.get("outputs"), [], cell["id"])
            compile(cell_source(cell), f"M20-{cell['id']}", "exec")

        all_code = "\n".join(cell_source(cell) for cell in code_cells).lower()
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "urlopen",
            "socket.",
            "subprocess",
            "import torch",
            "tensorflow",
            "cuda",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_notebook_enforces_prediction_before_every_action(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        positions = {
            str(cell["id"]): index for index, cell in enumerate(notebook["cells"])
        }
        ordered_pairs = [
            ("predict-learning-rate-regimes", "run-learning-rate-regimes"),
            ("predict-stability-boundary", "run-stability-boundary"),
            ("predict-too-small", "run-too-small"),
            ("predict-too-large", "run-too-large"),
            ("predict-gd-sgd", "run-gd-sgd"),
            ("predict-momentum", "run-momentum"),
            ("predict-adam", "run-adam"),
        ]
        for prediction, action in ordered_pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])

        markdown = "\n".join(
            cell_source(cell)
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.lower().count("predict before running"), 7)
        for phrase in [
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure A",
            "Controlled failure B",
            "Reject universal superiority",
            "UNFILLED BY LEARNER",
            "M19 → M20 → M21",
        ]:
            self.assertIn(phrase, markdown)

    def test_notebook_executes_all_required_optimizer_and_failure_symbols(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            cell_source(cell)
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        for symbol in [
            "run_learning_rate_sweep",
            "diagnose_dynamics",
            "coordinate_sign_changes",
            'run_optimizer("gd"',
            'run_optimizer("sgd"',
            'run_optimizer("momentum"',
            'run_optimizer("adam"',
            "learning_rate=1.0e-5",
            "learning_rate=0.21",
            "seed=2020",
            "seed=2021",
        ]:
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, code)

    def test_controlled_failures_are_isolated_repairable_and_not_superficial(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(
            encoding="utf-8"
        ).lower()
        controlled = " ".join(controlled.split())
        required_terms = [
            "prediction before action",
            "too-small learning rate",
            "stagnation",
            "too-large learning rate",
            "stability boundary",
            "smallest repair",
            "one controlled",
            "gradient → optimizer state → update → parameters → next loss",
            "rejected if it changes curvature",
        ]
        for term in required_terms:
            self.assertIn(term, controlled)

    def test_evidence_assessment_and_status_reject_prefilled_learner_work(self) -> None:
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        assessment = (MISSION / "assessment.yaml").read_text(encoding="utf-8")
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        for identifier in [
            "prediction_log",
            "learning_rate_sweep",
            "optimizer_comparison",
            "controlled_failure_diagnosis",
            "code_reading_trace",
            "no_ai_transfer",
            "optimizer_policy_adr",
        ]:
            self.assertIn(f"id: {identifier}", evidence)
        self.assertIn("prefilled_learner_evidence: prohibited", evidence)
        self.assertIsNone(
            re.search(r"(?m)^\s*learner_(evidence|response|prediction|score)\s*:", evidence)
        )
        self.assertNotRegex(assessment, r"(?m)^\s*(score|answer|passed)\s*:")
        self.assertIn("learner_evidence_status: intentionally_unpopulated", status)
        self.assertIn("learner_closure: pending_evidence_and_review", status)

    def test_no_ai_gate_and_adr_remain_explicitly_unfilled(self) -> None:
        gate = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        normalized_gate = " ".join(gate.split())
        for phrase in [
            "without AI-generated code, calculations, prose, or diagrams",
            "prediction before",
            "full intermediate state",
            "rollback trigger",
            "Leave all responses unfilled",
        ]:
            self.assertIn(phrase, normalized_gate)
        self.assertNotIn("## Answers", gate)
        self.assertNotIn("learner_response:", gate)

        required_fields = [
            "Status",
            "Date",
            "Owner",
            "Decision",
            "Chosen optimizer and learning-rate regime",
            "Evidence",
            "Alternatives considered",
            "Trade-offs",
            "Monitoring",
            "Rollback triggers",
            "Revisit triggers",
        ]
        for field in required_fields:
            self.assertIn(f"**{field}:** [UNFILLED BY LEARNER]", adr)
        self.assertEqual(adr.count("[UNFILLED BY LEARNER]"), len(required_fields))
        self.assertNotRegex(
            adr,
            r"(?mi)^- \*\*(status|decision):\*\*\s+(accepted|approved|proposed|complete)",
        )
        for heading in [
            "## Alternatives that must be compared",
            "## Evidence required",
            "## Monitoring, rollback, and revisit conditions",
        ]:
            self.assertIn(heading, adr)

    def test_formal_review_brief_has_decision_risks_challenges_and_acceptance(self) -> None:
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8")
        for heading in [
            "## Review decision requested",
            "## System under review",
            "## Evidence available to reviewers",
            "## Material assumptions",
            "## Risks and controls",
            "## Required reviewer challenges",
            "## Acceptance criteria",
            "## Open decision",
        ]:
            self.assertIn(heading, review)
        for requirement in [
            "same objective and initialization",
            "prediction cell before its action cell",
            "source notebook cells have stable unique IDs",
            "Learner predictions, scores, no-AI responses, evidence, sign-off, and ADR decisions",
        ]:
            self.assertIn(requirement, review)

    def test_m19_to_m20_to_m21_handoff_is_explicit_and_preserves_boundaries(self) -> None:
        m19_integration = (ROOT / "missions" / "M19" / "flagship_integration.md").read_text(
            encoding="utf-8"
        )
        integration = (MISSION / "flagship_integration.md").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        dependencies = json.loads(
            (ROOT / "data" / "mission_dependencies.json").read_text(encoding="utf-8")
        )["dependencies"]

        self.assertIn("Learning-rate dynamics and broader optimizer comparisons belong to M20", m19_integration)
        self.assertIn("## M19 → M20 boundary", integration)
        self.assertIn("## V04 closure contract", integration)
        self.assertIn("## M20 → M21 handoff", integration)
        self.assertIn("M19 explains", content)
        self.assertIn("M20 manipulates", content)
        self.assertIn("M21", content)
        self.assertEqual(dependencies["M20"]["blocking"], ["M19"])
        self.assertEqual(dependencies["M21"]["blocking"], ["M20"])

    def test_requirements_are_bounded_and_bare_unittest_imports_stay_standard_library(self) -> None:
        requirements = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            requirements,
            [
                "jupyter>=1.1,<2",
                "nbclient>=0.10,<1",
                "nbformat>=5.10,<6",
                "matplotlib>=3.8,<4",
                "pytest>=8,<10",
            ],
        )
        source = Path(__file__).read_text(encoding="utf-8")
        imported_roots: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertTrue(
            {"numpy", "matplotlib", "nbformat", "pytest", "yaml"}.isdisjoint(imported_roots)
        )


if __name__ == "__main__":
    unittest.main()
