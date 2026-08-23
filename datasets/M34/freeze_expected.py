#!/usr/bin/env python3
"""Freeze M34 RAG evaluation properties (offline, deterministic).

Run from the repository root:

    python datasets/M34/freeze_expected.py

Canonical tests load the frozen JSON. They do not call a paid API.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missions.M34.rag_pipeline import (  # noqa: E402
    CORPUS_VERSION,
    DEFAULT_BUDGET_CHARS,
    DEFAULT_TOP_K,
    EVAL_VERSION,
    INDEX_ID,
    PIPELINE_ID,
    POLICY_GATED,
    POLICY_NAIVE,
    SYNTHESIZER_ID,
    answer_labeled,
    evaluate_set,
    load_canonical_index,
    load_labeled_queries,
    load_questions_payload,
)


def _row(trace) -> dict:
    return {
        "query_id": trace.query_id,
        "retrieval_ids": list(trace.retrieval_ids),
        "packed_ids": list(trace.pack.chunk_ids()),
        "dropped_ids": list(trace.pack.dropped_ids()),
        "status": trace.answer.status,
        "answer": trace.answer.text,
        "citations": list(trace.answer.citation_ids()),
        "support_ok": trace.answer.support.ok,
        "eval_pass": bool(trace.evaluation.get("eval_pass")),
        "primary": trace.evaluation.get("primary"),
        "retrieval_hit": trace.evaluation.get("retrieval_hit"),
        "packed_hit": trace.evaluation.get("packed_hit"),
        "top_score": None if not trace.retrieval_scores else float(trace.retrieval_scores[0]),
        "index_id": trace.index_id,
        "source_hash": trace.source_hash,
        "model": trace.model,
        "version": trace.version,
    }


def main() -> None:
    index = load_canonical_index()
    questions = load_questions_payload()
    default_rows = {}
    for labeled in load_labeled_queries():
        trace = answer_labeled(labeled.query_id, index=index)
        default_rows[labeled.query_id] = _row(trace)

    password_k1 = answer_labeled("rag-password-procedure", top_k=1, index=index)
    password_k3 = answer_labeled("rag-password-procedure", top_k=3, index=index)
    password_budget = answer_labeled(
        "rag-password-procedure",
        top_k=3,
        budget_chars=80,
        index=index,
    )
    ticket_k1 = answer_labeled("rag-ticket-4412", top_k=1, index=index)
    ticket_naive = answer_labeled(
        "rag-ticket-4412",
        top_k=1,
        policy=POLICY_NAIVE,
        index=index,
    )
    ceo = answer_labeled("rag-ceo", index=index)
    ceo_naive = answer_labeled("rag-ceo", policy=POLICY_NAIVE, index=index)
    closed = answer_labeled("rag-legal-forbid", retrieval_enabled=False, index=index)
    unsupported = answer_labeled(
        "rag-reset-login",
        top_k=3,
        defect="unsupported_citation",
        index=index,
    )
    invented = answer_labeled(
        "rag-ticket-4412",
        top_k=1,
        defect="invented_support",
        index=index,
    )
    holdout = evaluate_set(split="holdout", index=index)
    payload = {
        "schema_version": 1,
        "note": "Fixture pipeline properties, not learner evidence.",
        "eval_version": EVAL_VERSION,
        "pipeline_id": PIPELINE_ID,
        "synthesizer_id": SYNTHESIZER_ID,
        "index_id": INDEX_ID,
        "corpus_version": CORPUS_VERSION,
        "source_hash": index.metadata.source_hash,
        "model": index.metadata.embedding.model,
        "version": index.metadata.embedding.version,
        "metric": index.metadata.embedding.metric,
        "normalization": index.metadata.embedding.normalization,
        "default_top_k": DEFAULT_TOP_K,
        "default_budget_chars": DEFAULT_BUDGET_CHARS,
        "policy": POLICY_GATED,
        "question_ids": [row["id"] for row in questions["questions"]],
        "holdout_ids": [row["id"] for row in questions["questions"] if row["split"] == "holdout"],
        "dev_ids": [row["id"] for row in questions["questions"] if row["split"] == "dev"],
        "default": default_rows,
        "password_k1": _row(password_k1),
        "password_k3": _row(password_k3),
        "password_budget80": _row(password_budget),
        "ticket_k1": _row(ticket_k1),
        "ticket_naive_k1": _row(ticket_naive),
        "ceo": _row(ceo),
        "ceo_naive": _row(ceo_naive),
        "legal_closed_book": _row(closed),
        "unsupported_reset": _row(unsupported),
        "invented_ticket": _row(invented),
        "holdout": {
            "n": holdout["n"],
            "n_pass": holdout["n_pass"],
            "pass_rate": holdout["pass_rate"],
            "n_abstain": holdout["n_abstain"],
            "n_unsupported_citation": holdout["n_unsupported_citation"],
        },
        "downloaded": False,
        "network_required": False,
    }
    target = HERE / "expected.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print("wrote", target)
    print("holdout pass", holdout["n_pass"], "/", holdout["n"])
    print("ceo status", ceo.answer.status, "top", ceo.retrieval_ids[:1])
    print("password k1", password_k1.answer.status, password_k1.pack.chunk_ids())
    print("password k3", password_k3.answer.status, password_k3.answer.citation_ids())
    print("ticket k1", ticket_k1.answer.status, ticket_k1.retrieval_ids)
    print("unsupported cites", unsupported.answer.citation_ids(), "ok", unsupported.answer.support.ok)


if __name__ == "__main__":
    main()
