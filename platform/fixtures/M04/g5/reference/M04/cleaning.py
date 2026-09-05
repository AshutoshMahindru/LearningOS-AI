"""Auditable, deterministic cleaning pipeline for the M04 dirty orders fixture.

The pipeline intentionally keeps raw values beside canonical values. It removes
only exact duplicate rows, records every coercion failure as an issue, preserves
conflicting identifiers for review, and never deletes an outlier automatically.

Pandas is imported inside functions so fixture package collection does not
require it. Official platform CI does not install pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Iterable

pd: Any = None


def _require_pandas() -> Any:
    """Load pandas on first use; collection and package import stay pandas-free."""

    global pd
    if pd is None:
        import pandas as pandas_mod

        pd = pandas_mod
    return pd


RAW_COLUMNS = [
    "order_id",
    "customer_name",
    "region",
    "category",
    "units",
    "unit_price",
    "order_total",
    "order_date",
    "email",
    "status",
    "notes",
]

ALLOWED_REGIONS = {"north", "south", "east", "west", "central"}
ALLOWED_CATEGORIES = {"electronics", "grocery", "home", "apparel", "office"}
ALLOWED_STATUSES = {"complete", "pending", "cancelled", "returned"}
MIN_ORDER_DATE = datetime(2025, 1, 1)
MAX_ORDER_DATE = datetime(2026, 12, 31)

REGION_ALIASES = {
    "north": "north",
    "n": "north",
    "north zone": "north",
    "south": "south",
    "s": "south",
    "south zone": "south",
    "east": "east",
    "e": "east",
    "east zone": "east",
    "west": "west",
    "w": "west",
    "west zone": "west",
    "central": "central",
    "c": "central",
    "central zone": "central",
}

CATEGORY_ALIASES = {
    "electronics": "electronics",
    "electronic": "electronics",
    "consumer electronics": "electronics",
    "grocery": "grocery",
    "groceries": "grocery",
    "home": "home",
    "home & kitchen": "home",
    "home/kitchen": "home",
    "home and kitchen": "home",
    "apparel": "apparel",
    "clothing": "apparel",
    "office": "office",
    "office supplies": "office",
}

STATUS_ALIASES = {
    "complete": "complete",
    "completed": "complete",
    "done": "complete",
    "pending": "pending",
    "in progress": "pending",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "returned": "returned",
}

BLOCKING_ISSUES = {
    "missing_order_id",
    "invalid_order_id",
    "conflicting_order_id",
    "missing_customer_name",
    "missing_region",
    "unmapped_region",
    "missing_category",
    "unmapped_category",
    "missing_units",
    "malformed_units",
    "non_integer_units",
    "non_positive_units",
    "missing_unit_price",
    "malformed_unit_price",
    "non_positive_unit_price",
    "missing_order_total_unresolved",
    "malformed_order_total",
    "non_positive_order_total",
    "order_total_mismatch",
    "missing_order_date",
    "malformed_order_date",
    "order_date_out_of_range",
    "missing_email",
    "invalid_email",
    "missing_status",
    "unmapped_status",
}


@dataclass(frozen=True)
class CleaningResult:
    """All auditable products of a single cleaning run."""

    raw: pd.DataFrame
    cleaned: pd.DataFrame
    analysis_ready: pd.DataFrame
    duplicate_log: pd.DataFrame
    issue_summary: pd.DataFrame
    summary: dict[str, int | float]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the CSV losslessly as strings; blank fields remain blank strings."""

    pd = _require_pandas()
    frame = pd.read_csv(
        Path(path),
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in RAW_COLUMNS]
    if missing or extra:
        raise ValueError(f"schema mismatch; missing={missing}, extra={extra}")
    return frame.loc[:, RAW_COLUMNS].copy()


def _collapse_whitespace(value: object) -> str:
    return " ".join(str(value).strip().split())


def _canonical_text(value: object, aliases: dict[str, str]) -> object:
    pd = _require_pandas()
    collapsed = _collapse_whitespace(value)
    if not collapsed:
        return pd.NA
    return aliases.get(collapsed.casefold(), pd.NA)


def _canonical_name(value: object) -> object:
    pd = _require_pandas()
    collapsed = _collapse_whitespace(value)
    return collapsed.title() if collapsed else pd.NA


def _parse_number(value: object) -> tuple[object, str]:
    pd = _require_pandas()
    raw = str(value).strip()
    if not raw:
        return pd.NA, "missing"
    normalized = re.sub(r"[₹$£€,\s]", "", raw)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", normalized):
        return pd.NA, "malformed"
    return float(normalized), "parsed"


DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%b %d %Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
)


def _parse_date(value: object) -> tuple[object, str]:
    pd = _require_pandas()
    raw = _collapse_whitespace(value)
    if not raw:
        return pd.NaT, "missing"
    for date_format in DATE_FORMATS:
        try:
            parsed = pd.Timestamp(datetime.strptime(raw, date_format))
        except ValueError:
            continue
        if date_format == "%d/%m/%Y":
            day, month, _ = raw.split("/")
            if int(day) <= 12 and int(month) <= 12:
                return parsed, "parsed_ambiguous_day_first"
        return parsed, "parsed"
    return pd.NaT, "malformed"


def _add_issue(frame: pd.DataFrame, mask: pd.Series, issue: str) -> None:
    for index in frame.index[mask.fillna(False)]:
        frame.at[index, "_issues"].append(issue)


def _append_decision(frame: pd.DataFrame, mask: pd.Series, decision: str) -> None:
    for index in frame.index[mask.fillna(False)]:
        frame.at[index, "_decisions"].append(decision)


def _exact_duplicate_log(raw: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    pd = _require_pandas()
    fingerprints = raw.astype("string").agg("\x1f".join, axis=1)
    duplicate_mask = fingerprints.duplicated(keep="first")
    first_source_row: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    for index, fingerprint in fingerprints.items():
        source_row = int(index) + 2
        if fingerprint not in first_source_row:
            first_source_row[fingerprint] = source_row
            continue
        rows.append(
            {
                "source_row": source_row,
                "duplicate_of_source_row": first_source_row[fingerprint],
                "order_id": raw.at[index, "order_id"],
                "observed_defect": "exact_duplicate",
                "decision": "remove_after_full_raw_row_equality_check",
                "evidence": "all raw columns are identical",
                "information_preserved": "duplicate row retained in this audit log",
            }
        )

    columns = [
        "source_row",
        "duplicate_of_source_row",
        "order_id",
        "observed_defect",
        "decision",
        "evidence",
        "information_preserved",
    ]
    return duplicate_mask, pd.DataFrame(rows, columns=columns)


def _iqr_upper_bound(series: pd.Series) -> float:
    values = series.dropna().astype(float)
    if values.empty:
        return float("inf")
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return q3 + 3.0 * (q3 - q1)


def _contains_blocking_issue(issue_text: str) -> bool:
    return bool(set(filter(None, issue_text.split("|"))) & BLOCKING_ISSUES)


def clean_orders(raw: pd.DataFrame) -> CleaningResult:
    """Return an audit-preserving cleaned result without silent row deletion."""

    pd = _require_pandas()
    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"required columns missing: {missing}")

    source = raw.loc[:, RAW_COLUMNS].copy().reset_index(drop=True)
    duplicate_mask, duplicate_log = _exact_duplicate_log(source)
    frame = source.loc[~duplicate_mask].copy().reset_index(drop=False)
    frame["source_row"] = frame.pop("index") + 2

    for column in RAW_COLUMNS:
        frame[f"raw_{column}"] = frame[column].astype("string")

    frame["_issues"] = [[] for _ in range(len(frame))]
    frame["_decisions"] = [[] for _ in range(len(frame))]

    frame["order_id"] = frame["raw_order_id"].map(_collapse_whitespace).str.upper()
    frame["customer_name"] = pd.Series(
        [_canonical_name(value) for value in frame["raw_customer_name"]],
        dtype="string",
    )
    frame["region"] = pd.Series(
        [_canonical_text(value, REGION_ALIASES) for value in frame["raw_region"]],
        dtype="string",
    )
    frame["category"] = pd.Series(
        [_canonical_text(value, CATEGORY_ALIASES) for value in frame["raw_category"]],
        dtype="string",
    )
    frame["email"] = (
        frame["raw_email"].map(_collapse_whitespace).str.casefold().astype("string")
    )
    frame["status"] = pd.Series(
        [_canonical_text(value, STATUS_ALIASES) for value in frame["raw_status"]],
        dtype="string",
    )
    frame["notes"] = frame["raw_notes"].map(_collapse_whitespace).astype("string")

    unit_values = frame["raw_units"].map(_parse_number)
    price_values = frame["raw_unit_price"].map(_parse_number)
    total_values = frame["raw_order_total"].map(_parse_number)
    frame["units"] = pd.Series([item[0] for item in unit_values], dtype="Float64")
    frame["unit_price"] = pd.Series([item[0] for item in price_values], dtype="Float64")
    frame["order_total"] = pd.Series([item[0] for item in total_values], dtype="Float64")
    unit_state = pd.Series([item[1] for item in unit_values], dtype="string")
    price_state = pd.Series([item[1] for item in price_values], dtype="string")
    total_state = pd.Series([item[1] for item in total_values], dtype="string")

    date_values = frame["raw_order_date"].map(_parse_date)
    frame["order_date"] = pd.to_datetime([item[0] for item in date_values])
    date_state = pd.Series([item[1] for item in date_values], dtype="string")

    missing_id = frame["order_id"].eq("")
    _add_issue(frame, missing_id, "missing_order_id")
    _add_issue(
        frame,
        ~missing_id & ~frame["order_id"].str.fullmatch(r"ORD-\d{4}"),
        "invalid_order_id",
    )

    id_counts = frame.loc[~missing_id, "order_id"].value_counts()
    conflicting_ids = set(id_counts[id_counts > 1].index)
    conflict_mask = frame["order_id"].isin(conflicting_ids)
    _add_issue(frame, conflict_mask, "conflicting_order_id")
    _append_decision(frame, conflict_mask, "preserve_all_conflicting_records_for_review")

    _add_issue(frame, frame["customer_name"].isna(), "missing_customer_name")

    raw_region_blank = frame["raw_region"].map(_collapse_whitespace).eq("")
    _add_issue(frame, raw_region_blank, "missing_region")
    _add_issue(frame, ~raw_region_blank & frame["region"].isna(), "unmapped_region")

    raw_category_blank = frame["raw_category"].map(_collapse_whitespace).eq("")
    _add_issue(frame, raw_category_blank, "missing_category")
    _add_issue(
        frame,
        ~raw_category_blank & frame["category"].isna(),
        "unmapped_category",
    )

    _add_issue(frame, unit_state.eq("missing"), "missing_units")
    _add_issue(frame, unit_state.eq("malformed"), "malformed_units")
    _add_issue(
        frame,
        frame["units"].notna() & ~frame["units"].mod(1).eq(0),
        "non_integer_units",
    )
    _add_issue(frame, frame["units"].notna() & frame["units"].le(0), "non_positive_units")

    _add_issue(frame, price_state.eq("missing"), "missing_unit_price")
    _add_issue(frame, price_state.eq("malformed"), "malformed_unit_price")
    _add_issue(
        frame,
        frame["unit_price"].notna() & frame["unit_price"].le(0),
        "non_positive_unit_price",
    )

    can_derive_total = (
        total_state.eq("missing")
        & frame["units"].notna()
        & frame["unit_price"].notna()
    )
    frame.loc[can_derive_total, "order_total"] = (
        frame.loc[can_derive_total, "units"]
        * frame.loc[can_derive_total, "unit_price"]
    )
    _append_decision(
        frame,
        can_derive_total,
        "derive_missing_total_from_units_times_unit_price",
    )
    _add_issue(
        frame,
        total_state.eq("missing") & ~can_derive_total,
        "missing_order_total_unresolved",
    )
    _add_issue(frame, total_state.eq("malformed"), "malformed_order_total")
    _add_issue(
        frame,
        frame["order_total"].notna() & frame["order_total"].le(0),
        "non_positive_order_total",
    )

    comparable_total = (
        frame["units"].gt(0)
        & frame["unit_price"].gt(0)
        & frame["order_total"].gt(0)
    ).fillna(False)
    expected_total = frame["units"] * frame["unit_price"]
    mismatch = comparable_total & (frame["order_total"] - expected_total).abs().gt(0.01)
    _add_issue(frame, mismatch, "order_total_mismatch")

    _add_issue(frame, date_state.eq("missing"), "missing_order_date")
    _add_issue(frame, date_state.eq("malformed"), "malformed_order_date")
    _add_issue(
        frame,
        frame["order_date"].notna()
        & ~frame["order_date"].between(MIN_ORDER_DATE, MAX_ORDER_DATE),
        "order_date_out_of_range",
    )
    ambiguous_date = date_state.eq("parsed_ambiguous_day_first")
    _add_issue(frame, ambiguous_date, "ambiguous_date_interpreted_day_first")
    _append_decision(frame, ambiguous_date, "apply_declared_day_first_date_policy")

    raw_email_blank = frame["email"].eq("")
    _add_issue(frame, raw_email_blank, "missing_email")
    email_pattern = r"[^\s@]+@[^\s@]+\.[^\s@]+"
    _add_issue(
        frame,
        ~raw_email_blank & ~frame["email"].str.fullmatch(email_pattern),
        "invalid_email",
    )

    raw_status_blank = frame["raw_status"].map(_collapse_whitespace).eq("")
    _add_issue(frame, raw_status_blank, "missing_status")
    _add_issue(frame, ~raw_status_blank & frame["status"].isna(), "unmapped_status")

    outlier_reasons: list[str] = []
    thresholds: dict[str, float] = {}
    for column in ("units", "unit_price", "order_total"):
        positive = frame.loc[frame[column].gt(0).fillna(False), column]
        threshold = _iqr_upper_bound(positive)
        thresholds[column] = threshold
        frame[f"_{column}_outlier"] = frame[column].gt(threshold).fillna(False)

    for _, row in frame.iterrows():
        reasons = [
            f"{column}_above_3iqr"
            for column in ("units", "unit_price", "order_total")
            if bool(row[f"_{column}_outlier"])
        ]
        outlier_reasons.append("|".join(reasons))

    frame["outlier_reason"] = pd.Series(outlier_reasons, dtype="string")
    frame["outlier_flag"] = frame["outlier_reason"].ne("")
    approved_exception = frame["notes"].str.contains(
        "approved wholesale order", case=False, na=False
    )
    frame["outlier_judgment"] = "not_flagged"
    frame.loc[frame["outlier_flag"], "outlier_judgment"] = "review_required"
    frame.loc[
        frame["outlier_flag"] & approved_exception,
        "outlier_judgment",
    ] = "retain_business_exception"
    _append_decision(frame, frame["outlier_flag"], "retain_outlier_pending_judgment")

    frame["observed_defects"] = frame["_issues"].map(lambda items: "|".join(items))
    frame["decision_log"] = frame["_decisions"].map(lambda items: "|".join(items))
    frame["uncertainty"] = frame.apply(
        lambda row: "review_required"
        if (
            "conflicting_order_id" in row["_issues"]
            or "ambiguous_date_interpreted_day_first" in row["_issues"]
            or bool(row["outlier_flag"])
        )
        else "none_recorded",
        axis=1,
    )
    frame["evidence"] = frame.apply(
        lambda row: (
            f"source_row={int(row['source_row'])}; raw columns preserved; "
            f"rules={row['observed_defects'] or 'none'}"
        ),
        axis=1,
    )
    frame["information_loss"] = "none_in_cleaned_table_raw_columns_preserved"
    frame["analysis_ready"] = ~frame["observed_defects"].map(_contains_blocking_issue)

    frame = frame.drop(
        columns=[
            "_issues",
            "_decisions",
            "_units_outlier",
            "_unit_price_outlier",
            "_order_total_outlier",
        ]
    )
    front = ["source_row", *RAW_COLUMNS]
    audit = [
        "observed_defects",
        "decision_log",
        "evidence",
        "uncertainty",
        "information_loss",
        "outlier_flag",
        "outlier_reason",
        "outlier_judgment",
        "analysis_ready",
    ]
    raw_columns = [f"raw_{column}" for column in RAW_COLUMNS]
    frame = frame.loc[:, [*front, *audit, *raw_columns]]

    analysis_ready = frame.loc[frame["analysis_ready"]].copy().reset_index(drop=True)
    assert_analysis_ready(analysis_ready)

    exploded = frame["observed_defects"].str.split("|").explode()
    exploded = exploded[exploded.ne("")]
    issue_summary = (
        exploded.value_counts()
        .rename_axis("issue")
        .reset_index(name="count")
        .sort_values(["count", "issue"], ascending=[False, True])
        .reset_index(drop=True)
    )

    summary: dict[str, int | float] = {
        "raw_rows": len(source),
        "exact_duplicate_rows_removed": int(duplicate_mask.sum()),
        "rows_after_exact_dedupe": len(frame),
        "conflicting_id_rows": int(conflict_mask.sum()),
        "analysis_ready_rows": len(analysis_ready),
        "rows_needing_review": int((~frame["analysis_ready"]).sum()),
        "outliers_retained": int(frame["outlier_flag"].sum()),
        "units_outlier_threshold": round(thresholds["units"], 2),
        "unit_price_outlier_threshold": round(thresholds["unit_price"], 2),
        "order_total_outlier_threshold": round(thresholds["order_total"], 2),
    }

    return CleaningResult(
        raw=source,
        cleaned=frame.reset_index(drop=True),
        analysis_ready=analysis_ready,
        duplicate_log=duplicate_log,
        issue_summary=issue_summary,
        summary=summary,
    )


def assert_analysis_ready(frame: pd.DataFrame) -> bool:
    """Assert the analysis-ready contract; return True for notebook display."""

    _require_pandas()
    required = [
        "order_id",
        "customer_name",
        "region",
        "category",
        "units",
        "unit_price",
        "order_total",
        "order_date",
        "email",
        "status",
    ]
    if frame.empty:
        raise AssertionError("analysis-ready table must contain at least one row")
    if frame[required].isna().any().any():
        raise AssertionError("analysis-ready required fields contain missing values")
    if not frame["order_id"].str.fullmatch(r"ORD-\d{4}").all():
        raise AssertionError("analysis-ready IDs violate the declared pattern")
    if not frame["order_id"].is_unique:
        raise AssertionError("analysis-ready IDs must be unique")
    if not set(frame["region"]).issubset(ALLOWED_REGIONS):
        raise AssertionError("analysis-ready regions contain unmapped values")
    if not set(frame["category"]).issubset(ALLOWED_CATEGORIES):
        raise AssertionError("analysis-ready categories contain unmapped values")
    if not set(frame["status"]).issubset(ALLOWED_STATUSES):
        raise AssertionError("analysis-ready statuses contain unmapped values")
    if not (frame["units"] > 0).all() or not frame["units"].mod(1).eq(0).all():
        raise AssertionError("analysis-ready units must be positive whole numbers")
    if not (frame["unit_price"] > 0).all() or not (frame["order_total"] > 0).all():
        raise AssertionError("analysis-ready monetary values must be positive")
    if not frame["order_date"].between(MIN_ORDER_DATE, MAX_ORDER_DATE).all():
        raise AssertionError("analysis-ready dates fall outside the contract")
    expected = frame["units"] * frame["unit_price"]
    if not (frame["order_total"] - expected).abs().le(0.01).all():
        raise AssertionError("analysis-ready totals do not reconcile")
    if frame["observed_defects"].map(_contains_blocking_issue).any():
        raise AssertionError("analysis-ready rows contain blocking issues")
    return True


def raw_vs_clean_comparison(result: CleaningResult) -> pd.DataFrame:
    """Return the row-accounting table used by the notebook and review."""

    pd = _require_pandas()
    metrics: Iterable[tuple[str, int]] = (
        ("raw_rows", len(result.raw)),
        ("exact_duplicates_logged", len(result.duplicate_log)),
        ("rows_after_exact_dedupe", len(result.cleaned)),
        ("analysis_ready_rows", len(result.analysis_ready)),
        ("rows_needing_review", int((~result.cleaned["analysis_ready"]).sum())),
        ("outlier_rows_retained", int(result.cleaned["outlier_flag"].sum())),
    )
    return pd.DataFrame(metrics, columns=["metric", "row_count"])
