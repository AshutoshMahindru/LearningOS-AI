from __future__ import annotations

from contextlib import chdir, redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M09"
NOTEBOOK = ROOT / "labs" / "M09_binary_classification.ipynb"
DATASET_DIR = ROOT / "datasets" / "M09"
DATASET = DATASET_DIR / "learner_disengagement.csv"
GENERATOR = DATASET_DIR / "generate_dataset.py"
ADR_PROMPT = MISSION / "adr_prompt.md"
ADR_HEADINGS = [
    "Decision",
    "Context",
    "Alternatives considered",
    "Evidence",
    "Trade-offs",
    "Revisit conditions",
    "Status",
]
ADR_RESPONSE_BLOCK = re.compile(
    r"<!-- BEGIN LEARNER RESPONSE: ([A-Z_]+) -->(.*?)"
    r"<!-- END LEARNER RESPONSE: \1 -->",
    re.DOTALL,
)


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def load_generator_module():
    spec = importlib.util.spec_from_file_location("m09_dataset_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M09 dataset generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_notebook_code() -> dict[str, object]:
    notebook = load_notebook()
    namespace: dict[str, object] = {"__name__": "__m09_test__"}
    output = io.StringIO()

    with chdir(ROOT), redirect_stdout(output):
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            cell_id = cell.get("id", "unknown")
            exec(compile(source, f"M09-notebook-{cell_id}", "exec"), namespace)

    namespace["_captured_stdout"] = output.getvalue()
    return namespace


def adr_response_blocks_are_empty(text: str) -> bool:
    """Return whether every expected prompt block exists exactly once and is empty."""
    blocks = ADR_RESPONSE_BLOCK.findall(text)
    expected_labels = {
        "DECISION",
        "CONTEXT",
        "ALTERNATIVES",
        "EVIDENCE",
        "TRADE_OFFS",
        "REVISIT_CONDITIONS",
        "STATUS",
    }
    labels = [label for label, _ in blocks]
    return (
        len(labels) == len(expected_labels)
        and set(labels) == expected_labels
        and len(labels) == len(set(labels))
        and all(not response.strip() for _, response in blocks)
    )


def adr_submission_issues(text: str | None) -> list[str]:
    """Apply a small, deterministic quality gate to a learner-authored ADR."""
    if text is None or not text.strip():
        return ["missing submission"]
    if "BEGIN LEARNER RESPONSE" in text or "END LEARNER RESPONSE" in text:
        return ["source prompt is not a learner-authored submission"]

    sections: dict[str, str] = {}
    for heading in ADR_HEADINGS:
        match = re.search(
            rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
            text,
        )
        if match:
            sections[heading] = match.group(1).strip()

    issues = [f"missing section: {heading}" for heading in ADR_HEADINGS if heading not in sections]
    if issues:
        return issues

    minimum_words = {
        "Decision": 18,
        "Context": 28,
        "Alternatives considered": 30,
        "Evidence": 30,
        "Trade-offs": 24,
        "Revisit conditions": 24,
        "Status": 8,
    }
    for heading, minimum in minimum_words.items():
        words = re.findall(r"\b[\w.-]+\b", sections[heading])
        if len(words) < minimum:
            issues.append(f"superficial section: {heading}")

    combined = "\n".join(sections.values()).lower()
    required_terms = {
        "threshold",
        "false positive",
        "false negative",
        "cost",
        "capacity",
        "alternative",
        "trade-off",
        "revisit",
        "owner",
    }
    for term in sorted(required_terms):
        if term not in combined:
            issues.append(f"missing decision concept: {term}")

    evidence = sections["Evidence"].lower()
    for token in ["tp", "tn", "fp", "fn", "precision", "recall", "holdout"]:
        if re.search(rf"\b{token}\b", evidence) is None:
            issues.append(f"missing evidence token: {token}")

    decision = sections["Decision"].lower()
    if re.search(r"\b0\.\d+\b", decision) is None:
        issues.append("decision lacks a numeric threshold")
    if ">=" not in decision and "comparison rule" not in decision:
        issues.append("decision lacks an exact comparison rule")

    alternatives = sections["Alternatives considered"].lower()
    if alternatives.count("alternative") < 3:
        issues.append("fewer than three explicit alternatives")

    revisit = sections["Revisit conditions"].lower()
    if not any(term in revisit for term in ["drift", "prevalence", "calibration"]):
        issues.append("revisit conditions lack model or population change trigger")
    if re.search(r"\b\d+(?:\.\d+)?%?\b", revisit) is None:
        issues.append("revisit conditions lack a quantitative trigger")

    status = sections["Status"].lower()
    if not any(value in status for value in ["proposed", "accepted", "superseded", "rejected"]):
        issues.append("status is not governed")
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", status) is None:
        issues.append("status lacks an ISO date")
    if any(token in combined for token in ["tbd", "todo", "fill this", "lorem ipsum"]):
        issues.append("placeholder content remains")

    return issues


class M09MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = execute_notebook_code()

    def test_required_standard_package_exists(self) -> None:
        required_mission_files = [
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
            "adr_prompt.md",
        ]
        missing = [name for name in required_mission_files if not (MISSION / name).is_file()]
        self.assertEqual(missing, [])
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(DATASET.is_file())
        self.assertTrue(GENERATOR.is_file())
        self.assertTrue((DATASET_DIR / "README.md").is_file())
        self.assertTrue((ROOT / "requirements" / "m09.txt").is_file())

    def test_manifest_declares_m09_execution_contract(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        expected_fragments = [
            "id: M09",
            "title: Make Binary Decisions",
            "phase: P2",
            "flagship: V02",
            "pedagogy: whole-first",
            "adr_required: true",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "- classification",
            "- probability",
            "datasets/M09/learner_disengagement.csv",
            "document_threshold_policy_in_learner_authored_adr",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, manifest)

    def test_whole_first_sequence_is_exact_and_ordered(self) -> None:
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        sequence = [
            "- id: baseline",
            "- id: split",
            "- id: classifier",
            "- id: predicted_probabilities",
            "- id: default_classification",
            "- id: confusion_matrix",
            "- id: threshold_changes",
            "- id: consequences",
        ]
        positions = [content.index(item) for item in sequence]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(positions), len(set(positions)))

    def test_dataset_is_deterministic_binary_and_imbalanced(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames

        self.assertEqual(
            fieldnames,
            [
                "learner_id",
                "account_age_days",
                "weekly_sessions",
                "overdue_tasks",
                "assessment_score",
                "help_requests",
                "disengaged_next_30_days",
            ],
        )
        self.assertEqual(len(rows), 180)
        self.assertEqual(len({row["learner_id"] for row in rows}), 180)

        targets = [int(row["disengaged_next_30_days"]) for row in rows]
        self.assertEqual(set(targets), {0, 1})
        self.assertEqual(sum(targets), 44)
        self.assertGreater(sum(targets) / len(targets), 0.20)
        self.assertLess(sum(targets) / len(targets), 0.30)

        generator = load_generator_module()
        generated = [
            {key: str(value) for key, value in row.items()}
            for row in generator.generate_rows()
        ]
        self.assertEqual(generated, rows)

    def test_dataset_generator_reproduces_committed_bytes(self) -> None:
        generator = load_generator_module()
        with tempfile.TemporaryDirectory() as directory:
            regenerated = Path(directory) / "regenerated.csv"
            generator.write_dataset(regenerated)
            self.assertEqual(regenerated.read_bytes(), DATASET.read_bytes())

    def test_notebook_has_stable_ids_no_outputs_and_whole_first_route(self) -> None:
        notebook = load_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 34)

        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"), cell.get("id"))
                self.assertEqual(cell.get("outputs", []), [], cell.get("id"))

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        route = (
            "baseline → split → classifier → predicted probabilities → "
            "default classification → confusion matrix → threshold changes → consequences"
        )
        self.assertIn(route, markdown_text)
        self.assertGreaterEqual(markdown_text.count("Predict before running"), 7)
        self.assertIn("## 11. Govern the threshold decision with an ADR", markdown_text)
        self.assertIn("missions/M09/adr_prompt.md", markdown_text)
        self.assertIn("separate learner-authored ADR", markdown_text)

    def test_notebook_code_is_syntactically_valid_offline_and_secret_free(self) -> None:
        notebook = load_notebook()
        code_texts = []
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"M09-cell-{cell.get('id')}", "exec")
            code_texts.append(source)

        all_code = "\n".join(code_texts)
        forbidden = [
            "import requests",
            "import httpx",
            "urllib",
            "socket.",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "subprocess",
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, all_code)

        required_symbols = [
            "stratified_split",
            "fit_logistic_classifier",
            "predict_probabilities",
            "classify",
            "confusion_counts",
            "metric_summary",
            "calibration_bins",
            "brier_score",
        ]
        for symbol in required_symbols:
            with self.subTest(symbol=symbol):
                self.assertIn(f"def {symbol}", all_code)

    def test_executed_notebook_preserves_split_and_probability_invariants(self) -> None:
        namespace = self.namespace
        train_rows = namespace["train_rows"]
        test_rows = namespace["test_rows"]
        probabilities = namespace["test_probabilities"]

        self.assertEqual((len(train_rows), len(test_rows)), (135, 45))
        train_ids = {row["learner_id"] for row in train_rows}
        test_ids = {row["learner_id"] for row in test_rows}
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(sum(row["disengaged_next_30_days"] for row in test_rows), 11)
        self.assertEqual(len(probabilities), 45)
        self.assertTrue(all(0.0 < probability < 1.0 for probability in probabilities))
        self.assertLess(min(probabilities), 0.01)
        self.assertGreater(max(probabilities), 0.80)

    def test_confusion_matrix_orientation_and_metric_denominators(self) -> None:
        confusion_counts = self.namespace["confusion_counts"]
        metric_summary = self.namespace["metric_summary"]

        counts = confusion_counts([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(counts, {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(
            metric_summary(counts),
            {"accuracy": 0.5, "precision": 0.5, "recall": 0.5},
        )

        asymmetric = {"tn": 7, "fp": 3, "fn": 1, "tp": 9}
        metrics = metric_summary(asymmetric)
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 0.9)

    def test_default_threshold_and_imbalance_failure_are_observable(self) -> None:
        self.assertEqual(
            self.namespace["baseline_counts"],
            {"tn": 34, "fp": 0, "fn": 11, "tp": 0},
        )
        self.assertEqual(
            self.namespace["default_counts"],
            {"tn": 33, "fp": 1, "fn": 8, "tp": 3},
        )
        baseline_metrics = self.namespace["baseline_metrics"]
        default_metrics = self.namespace["default_metrics"]
        self.assertGreater(baseline_metrics["accuracy"], 0.75)
        self.assertEqual(baseline_metrics["recall"], 0.0)
        self.assertGreater(default_metrics["accuracy"], baseline_metrics["accuracy"])
        self.assertLess(default_metrics["recall"], 0.30)

    def test_threshold_changes_policy_not_model_and_exposes_tradeoffs(self) -> None:
        results = {
            row["threshold"]: row for row in self.namespace["threshold_results"]
        }
        self.assertEqual(set(results), {0.20, 0.30, 0.50, 0.70})
        self.assertEqual(
            [results[value]["predicted_positive"] for value in [0.20, 0.30, 0.50, 0.70]],
            [16, 10, 4, 2],
        )
        self.assertGreater(results[0.20]["recall"], results[0.50]["recall"])
        self.assertLess(results[0.20]["precision"], results[0.50]["precision"])
        self.assertEqual(results[0.30]["accuracy"], results[0.50]["accuracy"])
        self.assertNotEqual(results[0.30]["fn"], results[0.50]["fn"])
        self.assertEqual(
            tuple(self.namespace["model_weights"]),
            self.namespace["weights_snapshot"],
        )
        self.assertEqual(
            tuple(self.namespace["test_probabilities"]),
            self.namespace["probabilities_snapshot"],
        )

    def test_consequence_policy_and_calibration_checks_are_substantive(self) -> None:
        selected = self.namespace["selected_policy"]
        results = {
            row["threshold"]: row for row in self.namespace["threshold_results"]
        }
        self.assertEqual(selected["threshold"], 0.20)
        self.assertEqual(results[0.20]["consequence_cost"], 23)
        self.assertEqual(results[0.50]["consequence_cost"], 41)
        self.assertLess(results[0.20]["consequence_cost"], results[0.50]["consequence_cost"])

        calibration = self.namespace["calibration_table"]
        self.assertEqual(sum(row["count"] for row in calibration), 45)
        self.assertTrue(all(0.0 <= row["observed_rate"] <= 1.0 for row in calibration))
        self.assertLess(self.namespace["model_brier"], self.namespace["constant_brier"])

    def test_controlled_failure_and_no_ai_gate_require_fresh_reasoning(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8").lower()

        for phrase in [
            "threshold `0.50`",
            "majority baseline",
            "false negatives",
            "accuracy, precision and recall",
            "false-positive cost",
            "false-negative cost",
            "not “always use a lower threshold.”",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, controlled)

        for phrase in [
            "without ai-generated analysis or code",
            "predicted defect probability",
            "at most four batches",
            "commit before outcomes",
            "2 × fp + 9 × fn",
            "threshold `0.50`",
            "calibration",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, no_ai)

    def test_evidence_contract_does_not_prefill_learner_evidence(self) -> None:
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", evidence)
        self.assertIn("prefilled_learner_evidence: prohibited", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))

    def test_adr_prompt_is_template_based_threshold_specific_and_unfilled(self) -> None:
        template = (ROOT / "templates" / "ADR.md").read_text(encoding="utf-8")
        adr = ADR_PROMPT.read_text(encoding="utf-8")

        template_headings = re.findall(r"(?m)^## (.+)$", template)
        self.assertEqual(template_headings, ADR_HEADINGS)
        for heading in template_headings:
            self.assertIn(f"## {heading}", adr)

        normalized_adr = " ".join(adr.lower().split())
        for phrase in [
            "selected operating threshold",
            "false-positive and false-negative costs",
            "operating capacity",
            "comparison rule",
            "Alternatives considered",
            "accepted FP/FN balance",
            "measurable triggers",
            "rollback",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(" ".join(phrase.lower().split()), normalized_adr)

        self.assertTrue(adr_response_blocks_are_empty(adr))
        self.assertNotIn("learner_evidence:", adr)
        self.assertNotIn("learner_response:", adr)

        prefilled = adr.replace(
            "<!-- BEGIN LEARNER RESPONSE: DECISION -->\n",
            "<!-- BEGIN LEARNER RESPONSE: DECISION -->\nUse threshold 0.20.\n",
            1,
        )
        self.assertFalse(adr_response_blocks_are_empty(prefilled))

    def test_adr_quality_gate_rejects_bad_artifacts_and_accepts_generic_fixture(self) -> None:
        superficial = "\n\n".join(
            f"## {heading}\nBrief statement." for heading in ADR_HEADINGS
        )
        placeholder = "\n\n".join(
            f"## {heading}\n[Fill this section]" for heading in ADR_HEADINGS
        )
        prompt = ADR_PROMPT.read_text(encoding="utf-8")
        prefilled_prompt = prompt.replace(
            "<!-- BEGIN LEARNER RESPONSE: DECISION -->\n",
            "<!-- BEGIN LEARNER RESPONSE: DECISION -->\nRepository-injected answer.\n",
            1,
        )
        completed_prompt = prompt
        for label in [
            "DECISION",
            "CONTEXT",
            "ALTERNATIVES",
            "EVIDENCE",
            "TRADE_OFFS",
            "REVISIT_CONDITIONS",
            "STATUS",
        ]:
            completed_prompt = completed_prompt.replace(
                f"<!-- BEGIN LEARNER RESPONSE: {label} -->\n",
                f"<!-- BEGIN LEARNER RESPONSE: {label} -->\n"
                "Repository-injected completion.\n",
                1,
            )

        # Parser-only fixture: unrelated to M09 learners, data, thresholds, or costs.
        generic_non_m09_fixture = """# Generic orbital-greenhouse fan inspection ADR

## Decision
The generic controller uses threshold 0.61 with comparison rule fan-risk score >= 0.61 to request a mechanical inspection. Fixture Operations owns the policy, whose expected action volume fits the twelve-inspection weekly capacity.

## Context
This parser fixture concerns synthetic orbital-greenhouse ventilation fans, not learners. A false positive spends two generic maintenance credits on a functioning fan, while a false negative risks twenty credits of crop-temperature loss. The cost units and operating capacity are illustrative test data only.

## Alternatives considered
Alternative one used 0.75 and reduced inspections but missed more failing fans. Alternative two used 0.61 and balanced modeled cost with capacity. Alternative three used 0.45 and exceeded the inspection queue. A fourth alternative disabled automatic requests and was rejected because every inspection would depend on manual discovery.

## Evidence
In a generic 80-fan holdout, the structural fixture records TP 7, TN 60, FP 5 and FN 8, with accuracy 0.838, precision 0.583 and recall 0.467. The illustrative cost is 170 credits. These arbitrary values exist only to exercise required ADR fields and do not reproduce mission evidence.

## Trade-offs
The generic trade-off accepts some false-positive inspections and capacity load to reduce costly false negatives. Fixture Operations owns rollback if the queue exceeds twelve, maintenance credits rise, or the controller causes unexpected greenhouse downtime. Score calibration and fan populations may change independently.

## Revisit conditions
The owner will revisit monthly if capacity falls below twelve inspections, the generic cost ratio changes by 25%, prevalence drifts by 6 percentage points, calibration error exceeds 0.10, or recall falls below 0.40 across 120 resolved fan cases.

## Status
Status: Proposed. Owner: Fixture Operations. Date: 2030-01-15. This generic parser record has no operational or mission-evidence standing.
"""

        rejected = {
            "missing": None,
            "superficial": superficial,
            "placeholder": placeholder,
            "prompt_marker": prompt,
            "prefilled_repository_prompt": prefilled_prompt,
            "completed_repository_prompt": completed_prompt,
        }
        for label, candidate in rejected.items():
            with self.subTest(label=label):
                self.assertTrue(adr_submission_issues(candidate))

        self.assertFalse(adr_response_blocks_are_empty(prefilled_prompt))
        self.assertFalse(adr_response_blocks_are_empty(completed_prompt))
        self.assertEqual(adr_submission_issues(generic_non_m09_fixture), [])

    def test_adr_requirement_is_wired_through_assessment_evidence_and_v02(self) -> None:
        assessment = (MISSION / "assessment.yaml").read_text(encoding="utf-8")
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        readme = (MISSION / "README.md").read_text(encoding="utf-8")
        flagship = (MISSION / "flagship_integration.md").read_text(encoding="utf-8")

        for phrase in [
            "adr: required",
            "defend_threshold_policy_in_learner_authored_adr",
            "adr_governance:",
            "threshold_adr_missing_superficial_or_prefilled",
        ]:
            self.assertIn(phrase, assessment)

        for phrase in [
            "- id: threshold_adr",
            "artifact_type: learner_authored_adr",
            "false_positive_and_false_negative_costs",
            "operating_capacity",
            "monitoring_rollback_and_revisit_triggers",
            "require_unfilled_repository_adr_prompt: true",
        ]:
            self.assertIn(phrase, evidence)

        for document in [readme, flagship]:
            self.assertIn("learner-authored ADR", document)
            self.assertIn("capacity", document)
            self.assertIn("trade-offs", document)
            self.assertIn("revisit", document)

    def test_existing_authoritative_sources_are_reused_without_registry_mutation(self) -> None:
        registry = json.loads(
            (ROOT / "data" / "source_registry.json").read_text(encoding="utf-8")
        )
        source_ids = {source["id"] for source in registry["sources"]}
        self.assertTrue({"sklearn-guide", "stanford-cs229"}.issubset(source_ids))

        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        self.assertIn("global_content_registry_modified: false", content)
        self.assertIn("- sklearn-guide", content)
        self.assertIn("- stanford-cs229", content)


if __name__ == "__main__":
    unittest.main()
