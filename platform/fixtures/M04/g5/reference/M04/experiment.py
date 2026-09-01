"""Mission-local M04 quality runner. Not a platform API route.

`cleaning.py` is the pandas lab source. Worker execute uses
QUALITY_EXECUTE_SOURCE, which is stdlib-only and reads CSV text from
parameters so the frozen sandbox never imports pandas or pathlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
ORDERS_CSV = PACKAGE_ROOT / "data" / "customer_orders_dirty.csv"
INVENTORY_CSV = PACKAGE_ROOT / "data" / "inventory_transfer.csv"
INVENTORY_COLUMNS = [
    "sku",
    "item_name",
    "warehouse",
    "quantity",
    "unit_cost",
    "received_date",
]

# Sandbox-safe snippet for generic-runtime execute. No imports.
QUALITY_EXECUTE_SOURCE = """
csv_text = parameters["csv_text"]
normalized = csv_text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
lines = [line for line in normalized.split("\\n") if line != ""]
header = lines[0]
body = lines[1:]
raw_rows = len(body)
seen = set()
exact_duplicate_rows = 0
for line in body:
    if line in seen:
        exact_duplicate_rows += 1
    else:
        seen.add(line)
rows_after_exact_dedupe = raw_rows - exact_duplicate_rows

def parse_line(line):
    fields = []
    buf = []
    quoted = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quoted:
            if ch == '"':
                if i + 1 < n and line[i + 1] == '"':
                    buf.append('"')
                    i += 2
                    continue
                quoted = False
            else:
                buf.append(ch)
        elif ch == '"':
            quoted = True
        elif ch == ",":
            fields.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    fields.append("".join(buf))
    return fields

blank_cells = 0
for line in body:
    for value in parse_line(line):
        if value.strip() == "":
            blank_cells += 1

summary = {
    "raw_rows": raw_rows,
    "exact_duplicate_rows": exact_duplicate_rows,
    "rows_after_exact_dedupe": rows_after_exact_dedupe,
    "blank_cells": blank_cells,
}
print(summary)
[
    {
        "type": "table",
        "title": "quality_accounting",
        "payload": {
            "columns": ["metric", "row_count"],
            "rows": [
                ["raw_rows", raw_rows],
                ["exact_duplicate_rows", exact_duplicate_rows],
                ["rows_after_exact_dedupe", rows_after_exact_dedupe],
                ["blank_cells", blank_cells],
            ],
        },
    },
    {"type": "metric", "title": "quality_summary", "payload": summary},
]
"""


def orders_csv_text() -> str:
    return ORDERS_CSV.read_text(encoding="utf-8")


def inventory_csv_text() -> str:
    return INVENTORY_CSV.read_text(encoding="utf-8")


def _jsonable_number(value: Any) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return _jsonable_number(value.item())
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _summary_payload(summary: dict[str, Any]) -> dict[str, int | float]:
    return {str(key): _jsonable_number(item) for key, item in summary.items()}


def run_quality_pipeline(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Run the auditable orders pipeline and return WP-137 table/metric blocks."""

    from .cleaning import clean_orders, load_raw, raw_vs_clean_comparison

    path = Path(csv_path) if csv_path is not None else ORDERS_CSV
    result = clean_orders(load_raw(path))
    comparison = raw_vs_clean_comparison(result)
    rows = []
    for record in comparison.itertuples(index=False):
        rows.append([str(record.metric), int(record.row_count)])
    summary = _summary_payload(dict(result.summary))
    return [
        {
            "type": "table",
            "title": "raw_vs_clean",
            "payload": {"columns": ["metric", "row_count"], "rows": rows},
        },
        {"type": "metric", "title": "quality_summary", "payload": summary},
    ]


def measure_unsafe_cleaning_loss(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Quantify information loss from the four unsafe shortcuts in the lab."""

    import pandas as pd

    from .cleaning import load_raw

    path = Path(csv_path) if csv_path is not None else ORDERS_CSV
    raw = load_raw(path)
    with_na = raw.replace("", pd.NA)
    dropped = with_na.dropna()
    aggressive = raw.drop_duplicates(subset=["order_id"], keep="first")
    coerced = raw.copy()
    for column in ("units", "unit_price", "order_total"):
        stripped = coerced[column].str.replace(r"[₹$£€,\s]", "", regex=True)
        coerced[column] = pd.to_numeric(stripped, errors="coerce")
    silent_numeric_nulls = int(coerced[["units", "unit_price", "order_total"]].isna().sum().sum())
    numeric_units = pd.to_numeric(
        raw["units"].str.replace(r"[₹$£€,\s]", "", regex=True),
        errors="coerce",
    )
    auto_outliers = raw.loc[numeric_units.gt(100)].copy()
    payload = {
        "raw_rows": int(len(raw)),
        "dropna_rows_lost": int(len(raw) - len(dropped)),
        "aggressive_id_dedupe_rows_lost": int(len(raw) - len(aggressive)),
        "silent_numeric_nulls": silent_numeric_nulls,
        "automatic_outlier_deletions": int(len(auto_outliers)),
        "first_unsafe_operation": "dropna",
        "safe_repair": "log exact duplicates only; preserve conflicts, parse failures and outliers",
    }
    return payload


def run_controlled_failure(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = measure_unsafe_cleaning_loss(csv_path)
    return [{"type": "metric", "title": "unsafe_cleaning_loss", "payload": payload}]


def run_transfer_pipeline(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Apply M04 quality discipline to the fresh inventory transfer CSV."""

    import pandas as pd

    path = Path(csv_path) if csv_path is not None else INVENTORY_CSV
    raw = pd.read_csv(path, dtype="string", keep_default_na=False, na_filter=False)
    missing = [column for column in INVENTORY_COLUMNS if column not in raw.columns]
    extra = [column for column in raw.columns if column not in INVENTORY_COLUMNS]
    if missing or extra:
        raise ValueError(f"inventory schema mismatch; missing={missing}, extra={extra}")
    source = raw.loc[:, INVENTORY_COLUMNS].copy()
    fingerprints = source.astype("string").agg("\x1f".join, axis=1)
    duplicate_mask = fingerprints.duplicated(keep="first")
    cleaned = source.loc[~duplicate_mask].copy().reset_index(drop=True)
    cleaned["source_row"] = cleaned.index + 2
    cleaned["sku_canonical"] = cleaned["sku"].map(lambda value: " ".join(str(value).strip().split())).str.upper()
    sku_counts = cleaned.loc[cleaned["sku_canonical"].ne(""), "sku_canonical"].value_counts()
    conflicts = set(sku_counts[sku_counts > 1].index)
    cleaned["conflicting_sku"] = cleaned["sku_canonical"].isin(conflicts)
    qty = pd.to_numeric(cleaned["quantity"], errors="coerce")
    cleaned["quantity_parsed"] = qty
    cleaned["malformed_quantity"] = cleaned["quantity"].str.strip().ne("") & qty.isna()
    cost_text = cleaned["unit_cost"].str.replace(r"[₹$£€,\s]", "", regex=True)
    cleaned["unit_cost_parsed"] = pd.to_numeric(cost_text, errors="coerce")
    cleaned["outlier_flag"] = qty.gt(100).fillna(False)
    approved = cleaned["item_name"].str.contains("approved bulk", case=False, na=False)
    cleaned["outlier_judgment"] = "not_flagged"
    cleaned.loc[cleaned["outlier_flag"], "outlier_judgment"] = "review_required"
    cleaned.loc[cleaned["outlier_flag"] & approved, "outlier_judgment"] = "retain_business_exception"
    blocking = (
        cleaned["sku_canonical"].eq("")
        | cleaned["conflicting_sku"]
        | cleaned["malformed_quantity"]
        | cleaned["warehouse"].map(lambda value: str(value).strip() == "")
        | cleaned["unit_cost_parsed"].isna()
    )
    cleaned["analysis_ready"] = ~blocking
    summary = {
        "raw_rows": int(len(source)),
        "exact_duplicate_rows_removed": int(duplicate_mask.sum()),
        "rows_after_exact_dedupe": int(len(cleaned)),
        "conflicting_sku_rows": int(cleaned["conflicting_sku"].sum()),
        "analysis_ready_rows": int(cleaned["analysis_ready"].sum()),
        "rows_needing_review": int((~cleaned["analysis_ready"]).sum()),
        "outliers_retained": int(cleaned["outlier_flag"].sum()),
    }
    comparison_rows = [
        ["raw_rows", summary["raw_rows"]],
        ["exact_duplicates_logged", summary["exact_duplicate_rows_removed"]],
        ["rows_after_exact_dedupe", summary["rows_after_exact_dedupe"]],
        ["analysis_ready_rows", summary["analysis_ready_rows"]],
        ["rows_needing_review", summary["rows_needing_review"]],
        ["outlier_rows_retained", summary["outliers_retained"]],
    ]
    return [
        {
            "type": "table",
            "title": "inventory_raw_vs_clean",
            "payload": {"columns": ["metric", "row_count"], "rows": comparison_rows},
        },
        {"type": "metric", "title": "inventory_quality_summary", "payload": summary},
    ]
