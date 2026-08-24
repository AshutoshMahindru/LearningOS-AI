#!/usr/bin/env python3
"""Freeze M40 evaluation-harness properties (offline, deterministic).

Run from the repository root:

    python datasets/M40/freeze_expected.py

Canonical tests load the frozen JSON. They do not call a paid API.
Wall-clock latency is not frozen; cost/step proxies are.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missions.M40.evaluation_harness import (  # noqa: E402
    AGGREGATE_ONLY_POLICY,
    CANONICAL_POLICY,
    EVAL_VERSION,
    HARNESS_VERSION,
    HIDDEN_CRITICAL_CASE,
    calibrate_rubric,
    decide_release_gate,
    inject_regression,
    load_eval_pack,
    load_rubric_labels,
    pack_fingerprint,
    pipeline_with_defect,
    repair_run,
    run_suite,
)


def _row(case_report) -> dict:
    payload = case_report.as_dict()
    payload.pop("cost_proxy", None)
    return payload


def main() -> None:
    pack = load_eval_pack(require_canonical=True)
    baseline = run_suite(pack)
    baseline_gate = decide_release_gate(baseline, CANONICAL_POLICY)
    regression = inject_regression(pack, defect="unsupported_citation")
    regression_gate = decide_release_gate(regression, CANONICAL_POLICY)
    traces = {row.case_id: row.trace for row in baseline.rows}
    rubric = calibrate_rubric(load_rubric_labels(), traces_by_source=traces)
    hidden = pipeline_with_defect(defect="hidden_critical")
    hidden_repaired = repair_run(hidden)
    contaminated = pipeline_with_defect(defect="contaminated_pack")
    contaminated_repaired = repair_run(contaminated)
    payload = {
        "schema_version": 1,
        "note": "Fixture harness properties, not learner evidence.",
        "eval_version": EVAL_VERSION,
        "harness_version": HARNESS_VERSION,
        "pack_hash": pack_fingerprint(pack),
        "n": baseline.n,
        "case_ids": list(baseline.case_ids),
        "holdout_ids": [row.case_id for row in baseline.rows if row.split == "holdout"],
        "task_success_rate": baseline.task_success_rate,
        "n_task_success": baseline.n_task_success,
        "invariant_pass_rate": baseline.invariant_pass_rate,
        "n_critical_fail": baseline.n_critical_fail,
        "critical_fail_rate": baseline.critical_fail_rate,
        "slice_fail_rates": dict(baseline.slice_fail_rates),
        "family_success": dict(baseline.family_success),
        "mean_step_count": baseline.mean_step_count,
        "baseline_gate_passed": baseline_gate.passed,
        "baseline_fail_reasons": list(baseline_gate.fail_reasons),
        "hidden_critical_case": HIDDEN_CRITICAL_CASE,
        "regression": {
            "defect": "unsupported_citation",
            "n_critical_fail": regression.n_critical_fail,
            "critical_fail_rate": regression.critical_fail_rate,
            "task_success_rate": regression.task_success_rate,
            "slice_citation": regression.slice_fail_rates.get("citation_support"),
            "canonical_gate_passed": regression_gate.passed,
            "canonical_fail_reasons": list(regression_gate.fail_reasons),
            "eval_version": regression.eval_version,
            "case_ids": list(regression.case_ids),
        },
        "rubric": {
            "n": rubric["n"],
            "n_disagree": rubric["n_disagree"],
            "disagreement_rate": rubric["disagreement_rate"],
            "deterministic_required_for_invariants": rubric["deterministic_required_for_invariants"],
        },
        "hidden_critical": {
            "aggregate_passed": hidden.decision.passed,
            "repaired_passed": hidden_repaired.decision.passed,
            "repaired_reasons": list(hidden_repaired.decision.fail_reasons),
            "task_success_rate": hidden.report.task_success_rate,
            "critical_fail_rate": hidden.report.critical_fail_rate,
        },
        "contaminated": {
            "pack_version": contaminated.pack_version,
            "aggregate_passed": contaminated.decision.passed,
            "repaired_version": contaminated_repaired.pack_version,
            "repaired_passed": contaminated_repaired.decision.passed,
            "repaired_contaminated": contaminated_repaired.report.pack_contaminated,
            "n_clean": contaminated_repaired.report.n,
        },
        "rows": {row.case_id: _row(row) for row in baseline.rows},
        "downloaded": False,
        "network_required": False,
    }
    target = HERE / "expected.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print("wrote", target)
    print("baseline n", baseline.n, "task_success", baseline.n_task_success, "/", baseline.n)
    print("invariant", baseline.n_invariant_pass, "critical", baseline.n_critical_fail)
    print("baseline gate", baseline_gate.passed, baseline_gate.fail_reasons)
    print("regression gate", regression_gate.passed, regression_gate.fail_reasons)
    print("hidden aggregate", hidden.decision.passed, "repaired", hidden_repaired.decision.passed)
    print("contaminated version", contaminated.pack_version, "clean", contaminated_repaired.pack_version)
    print("rubric disagree", rubric["n_disagree"], "/", rubric["n"])


if __name__ == "__main__":
    main()
