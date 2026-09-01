from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")

from g5.reference.M04.cleaning import (  # noqa: E402
    ALLOWED_CATEGORIES,
    ALLOWED_REGIONS,
    ALLOWED_STATUSES,
    assert_analysis_ready,
    clean_orders,
    load_raw,
    raw_vs_clean_comparison,
)
from g5.reference.M04.experiment import (  # noqa: E402
    INVENTORY_CSV,
    ORDERS_CSV,
    measure_unsafe_cleaning_loss,
    run_controlled_failure,
    run_quality_pipeline,
    run_transfer_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_fixture_orders_csv_matches_curriculum_dataset() -> None:
    canonical = REPO_ROOT / "datasets" / "M04" / "customer_orders_dirty.csv"
    assert ORDERS_CSV.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")


def test_quality_pipeline_emits_row_accounting_table_and_metrics() -> None:
    raw = load_raw(ORDERS_CSV)
    result = clean_orders(raw)
    assert result.summary["raw_rows"] == 36
    assert result.summary["exact_duplicate_rows_removed"] == 1
    assert result.summary["rows_after_exact_dedupe"] == 35
    assert result.summary["analysis_ready_rows"] == 10
    assert result.summary["rows_needing_review"] == 25
    assert result.summary["outliers_retained"] >= 2
    assert len(result.raw) == len(result.cleaned) + len(result.duplicate_log)

    conflicts = result.cleaned.loc[
        result.cleaned["observed_defects"].str.contains("conflicting_order_id")
    ]
    assert len(conflicts) == 4
    assert not conflicts["analysis_ready"].any()

    ready = result.analysis_ready
    assert assert_analysis_ready(ready)
    assert ready["order_id"].is_unique
    assert set(ready["region"]).issubset(ALLOWED_REGIONS)
    assert set(ready["category"]).issubset(ALLOWED_CATEGORIES)
    assert set(ready["status"]).issubset(ALLOWED_STATUSES)

    blocks = run_quality_pipeline()
    types = [block["type"] for block in blocks]
    assert types == ["table", "metric"]
    table = blocks[0]["payload"]
    metric = blocks[1]["payload"]
    accounting = {row[0]: row[1] for row in table["rows"]}
    assert accounting["raw_rows"] == 36
    assert accounting["exact_duplicates_logged"] == 1
    assert accounting["analysis_ready_rows"] == 10
    assert metric["raw_rows"] == 36
    assert metric["exact_duplicate_rows_removed"] == 1
    comparison = raw_vs_clean_comparison(result).set_index("metric")["row_count"]
    assert int(comparison["raw_rows"]) == 36


def test_unsafe_shortcuts_quantify_information_loss() -> None:
    payload = measure_unsafe_cleaning_loss()
    assert payload["raw_rows"] == 36
    assert payload["dropna_rows_lost"] > 0
    assert payload["aggressive_id_dedupe_rows_lost"] > 0
    assert payload["silent_numeric_nulls"] > 0
    assert payload["first_unsafe_operation"] == "dropna"
    blocks = run_controlled_failure()
    assert blocks[0]["type"] == "metric"
    assert blocks[0]["payload"]["dropna_rows_lost"] == payload["dropna_rows_lost"]


def test_transfer_inventory_preserves_conflicts_and_bulk_receipt() -> None:
    assert INVENTORY_CSV.is_file()
    blocks = run_transfer_pipeline()
    metric = blocks[1]["payload"]
    assert metric["raw_rows"] == 10
    assert metric["exact_duplicate_rows_removed"] == 1
    assert metric["rows_after_exact_dedupe"] == 9
    assert metric["conflicting_sku_rows"] == 2
    assert metric["outliers_retained"] == 1
    assert metric["analysis_ready_rows"] >= 1
    assert metric["raw_rows"] == (
        metric["rows_after_exact_dedupe"] + metric["exact_duplicate_rows_removed"]
    )
    table = {row[0]: row[1] for row in blocks[0]["payload"]["rows"]}
    assert table["outlier_rows_retained"] == 1
