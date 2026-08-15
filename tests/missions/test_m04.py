from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

import pandas as pd

from missions.M04.cleaning import (
    ALLOWED_CATEGORIES,
    ALLOWED_REGIONS,
    ALLOWED_STATUSES,
    MAX_ORDER_DATE,
    MIN_ORDER_DATE,
    assert_analysis_ready,
    clean_orders,
    load_raw,
    raw_vs_clean_comparison,
)


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M04"
DATASET = ROOT / "datasets" / "M04" / "customer_orders_dirty.csv"
NOTEBOOK = ROOT / "labs" / "M04_messy_csv.ipynb"


class M04MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_raw(DATASET)
        cls.result = clean_orders(cls.raw)

    def test_standard_package_and_manifest_contract(self) -> None:
        required = {
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
            "cleaning.py",
        }
        self.assertEqual(required, {path.name for path in MISSION.iterdir()} - {"__init__.py", "__pycache__"})
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(DATASET.is_file())
        self.assertTrue((ROOT / "requirements" / "m04.txt").is_file())

        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M04",
            "phase: P1",
            "flagship: V01",
            "formal_engineering_review: true",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "restart_run_all_required: true",
            "- pandas-start",
        ]:
            self.assertIn(expected, manifest)

    def test_fixture_contains_all_required_interacting_defects(self) -> None:
        raw = self.raw
        self.assertEqual(raw.shape, (36, 11))
        self.assertEqual(int(raw.duplicated(keep="first").sum()), 1)

        normalized_ids = raw["order_id"].str.strip().str.upper()
        repeated_groups = set(normalized_ids.value_counts().loc[lambda s: s > 1].index)
        self.assertEqual(repeated_groups, {"ORD-1003", "ORD-1005", "ORD-1012"})

        csv_text = DATASET.read_text(encoding="utf-8")
        for evidence in [
            "  Vikram  Sen ",
            "VIKRAM@EXAMPLE.COM",
            "home/kitchen",
            "consumer electronics",
            "₹1,250.00",
            "₹1,2O0",
            ",two,",
            "Jan 7 2026",
            "31/02/2026",
            "not a date",
            "2099-01-01",
            "approved wholesale order",
            "9,999,999",
        ]:
            self.assertIn(evidence, csv_text)
        self.assertGreater(int(raw.eq("").sum().sum()), 0)
        self.assertIn("-2", set(raw["units"]))
        self.assertIn("-75", set(raw["unit_price"]))

    def test_duplicate_policy_removes_only_exact_rows_and_preserves_conflicts(self) -> None:
        result = self.result
        self.assertEqual(result.summary["raw_rows"], 36)
        self.assertEqual(result.summary["exact_duplicate_rows_removed"], 1)
        self.assertEqual(result.summary["rows_after_exact_dedupe"], 35)
        self.assertEqual(len(result.duplicate_log), 1)
        self.assertEqual(result.duplicate_log.iloc[0]["source_row"], 5)
        self.assertEqual(result.duplicate_log.iloc[0]["duplicate_of_source_row"], 4)

        conflicts = result.cleaned.loc[
            result.cleaned["observed_defects"].str.contains("conflicting_order_id")
        ]
        self.assertEqual(len(conflicts), 4)
        self.assertEqual(set(conflicts["order_id"]), {"ORD-1005", "ORD-1012"})
        self.assertTrue(
            conflicts["decision_log"].str.contains(
                "preserve_all_conflicting_records_for_review"
            ).all()
        )
        self.assertFalse(conflicts["analysis_ready"].any())
        self.assertEqual(
            len(result.raw), len(result.cleaned) + len(result.duplicate_log)
        )

    def test_normalization_numeric_and_date_parsing_are_evidence_preserving(self) -> None:
        cleaned = self.result.cleaned

        vikram = cleaned.loc[cleaned["source_row"].eq(3)].iloc[0]
        self.assertEqual(vikram["raw_order_id"], " ord-1002 ")
        self.assertEqual(vikram["order_id"], "ORD-1002")
        self.assertEqual(vikram["customer_name"], "Vikram Sen")
        self.assertEqual(vikram["region"], "north")
        self.assertEqual(vikram["category"], "electronics")
        self.assertEqual(vikram["email"], "vikram@example.com")
        self.assertEqual(vikram["unit_price"], 1099)
        self.assertEqual(vikram["order_total"], 3297)
        self.assertEqual(vikram["order_date"], pd.Timestamp("2026-01-05"))
        self.assertIn("ambiguous_date_interpreted_day_first", vikram["observed_defects"])

        currency = cleaned.loc[cleaned["order_id"].eq("ORD-1024")].iloc[0]
        self.assertEqual(currency["raw_unit_price"], "₹ 250")
        self.assertEqual(currency["unit_price"], 250)
        self.assertEqual(currency["order_total"], 1750)
        self.assertEqual(currency["category"], "office")

        malformed = cleaned.loc[cleaned["order_id"].eq("ORD-1011")].iloc[0]
        self.assertEqual(malformed["raw_unit_price"], "₹1,2O0")
        self.assertTrue(pd.isna(malformed["unit_price"]))
        self.assertIn("malformed_unit_price", malformed["observed_defects"])
        self.assertIn("raw columns preserved", malformed["evidence"])

    def test_missingness_decisions_distinguish_missing_from_malformed(self) -> None:
        cleaned = self.result.cleaned
        derived = cleaned.loc[cleaned["order_id"].eq("ORD-1013")].iloc[0]
        malformed = cleaned.loc[cleaned["order_id"].eq("ORD-1019")].iloc[0]

        self.assertEqual(derived["raw_order_total"], "")
        self.assertEqual(derived["order_total"], 1600)
        self.assertIn(
            "derive_missing_total_from_units_times_unit_price",
            derived["decision_log"],
        )
        self.assertTrue(derived["analysis_ready"])

        self.assertEqual(malformed["raw_order_total"], "not recorded")
        self.assertTrue(pd.isna(malformed["order_total"]))
        self.assertIn("malformed_order_total", malformed["observed_defects"])
        self.assertNotIn("derive_missing_total", malformed["decision_log"])
        self.assertFalse(malformed["analysis_ready"])

    def test_constraints_create_review_evidence_without_silent_loss(self) -> None:
        cleaned = self.result.cleaned
        expected_issues = {
            "missing_order_id",
            "invalid_order_id",
            "conflicting_order_id",
            "missing_customer_name",
            "missing_region",
            "unmapped_region",
            "unmapped_category",
            "malformed_units",
            "malformed_unit_price",
            "malformed_order_total",
            "non_positive_units",
            "non_positive_unit_price",
            "non_positive_order_total",
            "malformed_order_date",
            "order_date_out_of_range",
            "invalid_email",
            "missing_status",
            "unmapped_status",
            "order_total_mismatch",
        }
        observed = set(self.result.issue_summary["issue"])
        self.assertTrue(expected_issues.issubset(observed))
        self.assertEqual(len(cleaned), 35)
        self.assertEqual(self.result.summary["rows_needing_review"], 25)
        self.assertTrue(
            cleaned.loc[~cleaned["analysis_ready"], "raw_order_id"].notna().all()
        )
        self.assertTrue(
            cleaned["information_loss"].eq(
                "none_in_cleaned_table_raw_columns_preserved"
            ).all()
        )

    def test_outliers_are_flagged_judged_and_retained(self) -> None:
        cleaned = self.result.cleaned
        outliers = cleaned.loc[cleaned["outlier_flag"]]
        self.assertEqual(len(outliers), self.result.summary["outliers_retained"])
        self.assertGreaterEqual(len(outliers), 2)

        wholesale = cleaned.loc[cleaned["order_id"].eq("ORD-1028")].iloc[0]
        suspicious = cleaned.loc[cleaned["order_id"].eq("ORD-1029")].iloc[0]
        self.assertTrue(wholesale["outlier_flag"])
        self.assertEqual(
            wholesale["outlier_judgment"], "retain_business_exception"
        )
        self.assertTrue(wholesale["analysis_ready"])
        self.assertIn("ORD-1028", set(self.result.analysis_ready["order_id"]))

        self.assertTrue(suspicious["outlier_flag"])
        self.assertEqual(suspicious["outlier_judgment"], "review_required")
        self.assertIn("order_total_mismatch", suspicious["observed_defects"])
        self.assertFalse(suspicious["analysis_ready"])

    def test_analysis_ready_invariants_and_reproducibility(self) -> None:
        ready = self.result.analysis_ready
        self.assertEqual(len(ready), 10)
        self.assertTrue(assert_analysis_ready(ready))
        self.assertTrue(ready["order_id"].is_unique)
        self.assertTrue(ready["units"].gt(0).all())
        self.assertTrue(ready["units"].mod(1).eq(0).all())
        self.assertTrue(set(ready["region"]).issubset(ALLOWED_REGIONS))
        self.assertTrue(set(ready["category"]).issubset(ALLOWED_CATEGORIES))
        self.assertTrue(set(ready["status"]).issubset(ALLOWED_STATUSES))
        self.assertTrue(ready["order_date"].between(MIN_ORDER_DATE, MAX_ORDER_DATE).all())
        expected_total = ready["units"] * ready["unit_price"]
        self.assertTrue((ready["order_total"] - expected_total).abs().le(0.01).all())

        rerun = clean_orders(load_raw(DATASET))
        pd.testing.assert_frame_equal(self.result.cleaned, rerun.cleaned)
        pd.testing.assert_frame_equal(self.result.duplicate_log, rerun.duplicate_log)
        pd.testing.assert_frame_equal(self.result.issue_summary, rerun.issue_summary)
        self.assertEqual(self.result.summary, rerun.summary)

        comparison = raw_vs_clean_comparison(self.result).set_index("metric")["row_count"]
        self.assertEqual(comparison["raw_rows"], 36)
        self.assertEqual(comparison["rows_after_exact_dedupe"], 35)
        self.assertEqual(comparison["analysis_ready_rows"], 10)
        self.assertEqual(comparison["rows_needing_review"], 25)

    def test_notebook_is_substantial_ordered_stable_and_output_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 40)

        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])
                compile("".join(cell.get("source", [])), cell["id"], "exec")

        workflow = "".join(
            next(cell["source"] for cell in notebook["cells"] if cell["id"] == "workflow")
        ).casefold()
        stages = [
            "raw",
            "inspect",
            "predict defects",
            "declare quality expectations",
            "duplicate analysis",
            "normalization",
            "numeric/date parsing",
            "missingness decisions",
            "validate constraints",
            "investigate outliers",
            "reproducible cleaning pipeline",
            "raw-vs-clean comparison",
            "analysis-ready table",
            "invariant assertions",
        ]
        positions = [workflow.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))

    def test_notebook_requires_prediction_and_diagnoses_all_unsafe_shortcuts(self) -> None:
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

        self.assertGreaterEqual(markdown.count("Predict before running"), 10)
        for phrase in [
            "Observed defect",
            "Hypothesis",
            "Decision",
            "Evidence",
            "Lost information / uncertainty",
            "Controlled failure",
            "Formal engineering review",
            "No-AI Gate",
            "V01 integration",
        ]:
            self.assertIn(phrase.casefold(), markdown.casefold())

        for token in [
            ".dropna()",
            "drop_duplicates(subset=\"order_id\"",
            "errors=\"coerce\"",
            "unsafe_without_outliers",
            "pd.testing.assert_frame_equal",
            "assert_analysis_ready",
        ]:
            self.assertIn(token, code)

        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
        ]:
            self.assertNotIn(forbidden, code)

    def test_review_and_evidence_contract_cover_formal_review_dimensions(self) -> None:
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8").casefold()
        for heading in [
            "raw state",
            "data contract",
            "architecture",
            "evidence",
            "controlled failure",
            "validation",
            "uncertainty and information loss",
            "v01 integration",
        ]:
            self.assertIn(heading, review)

        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", evidence)
        self.assertIn("prefilled_learner_evidence: prohibited", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))

        controlled = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").casefold()
        for shortcut in [
            "blanket `dropna`",
            "aggressive `drop_duplicates",
            "silent numeric coercion",
            "automatic deletion",
        ]:
            self.assertIn(shortcut, controlled)

    def test_authoritative_pandas_source_is_registered(self) -> None:
        registry = json.loads(
            (ROOT / "data" / "source_registry.json").read_text(encoding="utf-8")
        )
        pandas_source = next(
            source for source in registry["sources"] if source["id"] == "pandas-start"
        )
        self.assertEqual(pandas_source["publisher"], "pandas")
        self.assertEqual(pandas_source["kind"], "official-docs")
        self.assertIn("pandas", pandas_source["topics"])


if __name__ == "__main__":
    unittest.main()
