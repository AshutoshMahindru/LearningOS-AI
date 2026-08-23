#!/usr/bin/env python3
"""Freeze M35 retrieval-eval properties (offline, deterministic).

Run from the repository root:

    python datasets/M35/freeze_expected.py

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

from missions.M35.retrieval_eval import (  # noqa: E402
    CANONICAL_CORPUS_VERSION,
    DEFAULT_CANDIDATE_K,
    EVAL_VERSION,
    PIPELINE_ID,
    RERANKER_IDENTITY,
    RERANKER_LEX,
    ExperimentConfig,
    baseline_config,
    evaluate_set,
    generate_candidates,
    label_hash,
    leak_eval_phrasing,
    load_canonical_corpus,
    load_canonical_index,
    load_frozen_queries,
    load_transfer_payload,
    questions_sha256,
    relabel_after_results,
    repair_eval_boundary,
    rescore_with_labels,
    round_metric,
    source_hash,
    worst_queries,
)


def _compact_row(row) -> dict:
    return {
        "query_id": row.query_id,
        "split": row.split,
        "answerable": row.answerable,
        "candidate_ids": list(row.candidate_ids),
        "ranked_ids": list(row.ranked_ids),
        "cosine_ids": list(row.cosine_ids),
        "relevant_ids": list(row.relevant_ids),
        "support_ids": list(row.support_ids),
        "recall_at_k": round_metric(row.recall_at_k),
        "mrr": round_metric(row.mrr),
        "ndcg_at_k": round_metric(row.ndcg_at_k),
        "candidate_recall": round_metric(row.candidate_recall),
        "candidate_support_hit": row.candidate_support_hit,
        "first_relevant_rank": row.first_relevant_rank,
        "first_support_rank": row.first_support_rank,
        "trap_at_1": row.trap_at_1,
        "failure_mode": row.failure_mode,
        "scored_candidates": row.scored_candidates,
        "rerank_cost": row.rerank_cost,
        "proxy_cost": row.proxy_cost,
        "corpus_version": row.corpus_version,
        "index_id": row.index_id,
        "reranker_id": row.reranker_id,
        "eval_version": row.eval_version,
    }


def _compact_set(report) -> dict:
    return {
        "experiment_id": report.config.experiment_id,
        "config_identity": report.config_identity,
        "corpus_version": report.corpus_version,
        "index_id": report.index_id,
        "source_hash": report.source_hash,
        "n": report.n,
        "n_answerable": report.n_answerable,
        "mean_recall_at_k": round_metric(report.mean_recall_at_k),
        "mean_mrr": round_metric(report.mean_mrr),
        "mean_ndcg_at_k": round_metric(report.mean_ndcg_at_k),
        "mean_candidate_recall": round_metric(report.mean_candidate_recall),
        "trap_at_1_rate": round_metric(report.trap_at_1_rate),
        "scored_candidates": report.scored_candidates,
        "mean_proxy_cost": round_metric(report.mean_proxy_cost),
        "leaked": report.leaked,
        "relabeled": report.relabeled,
        "label_hash": report.label_hash,
        "slices": {
            name: {
                "n": values["n"],
                "n_answerable": values["n_answerable"],
                "mean_recall_at_k": round_metric(values["mean_recall_at_k"]),
                "mean_mrr": round_metric(values["mean_mrr"]),
                "mean_ndcg_at_k": round_metric(values["mean_ndcg_at_k"]),
                "mean_candidate_recall": round_metric(values["mean_candidate_recall"]),
                "trap_at_1_rate": round_metric(values["trap_at_1_rate"]),
            }
            for name, values in report.slices.items()
        },
    }


def main() -> None:
    queries = load_frozen_queries()
    source = load_canonical_corpus()
    index = load_canonical_index()
    baseline = evaluate_set(config=baseline_config(), queries=queries, source_corpus=source)
    rerank_cfg = ExperimentConfig(
        experiment_id="rerank-lex",
        corpus_version=CANONICAL_CORPUS_VERSION,
        chunk_mode="canonical",
        chunk_size=None,
        chunk_overlap=0,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker_id=RERANKER_LEX,
    )
    reranked = evaluate_set(config=rerank_cfg, queries=queries, source_corpus=source)
    k1_cfg = ExperimentConfig(
        experiment_id="candidate-k1",
        corpus_version=CANONICAL_CORPUS_VERSION,
        chunk_mode="canonical",
        chunk_size=None,
        chunk_overlap=0,
        candidate_k=1,
        reranker_id=RERANKER_IDENTITY,
    )
    k5_cfg = ExperimentConfig(
        experiment_id="candidate-k5",
        corpus_version=CANONICAL_CORPUS_VERSION,
        chunk_mode="canonical",
        chunk_size=None,
        chunk_overlap=0,
        candidate_k=5,
        reranker_id=RERANKER_IDENTITY,
    )
    k1 = evaluate_set(config=k1_cfg, queries=queries, source_corpus=source)
    k5 = evaluate_set(config=k5_cfg, queries=queries, source_corpus=source)
    merged_cfg = ExperimentConfig(
        experiment_id="chunk-merged",
        corpus_version="m35.corpus.merged.v1",
        chunk_mode="merged",
        chunk_size=None,
        chunk_overlap=0,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker_id=RERANKER_IDENTITY,
    )
    win32_cfg = ExperimentConfig(
        experiment_id="chunk-win32",
        corpus_version="m35.corpus.win32.v1",
        chunk_mode="windows",
        chunk_size=32,
        chunk_overlap=0,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker_id=RERANKER_IDENTITY,
    )
    win48_cfg = ExperimentConfig(
        experiment_id="chunk-win48o16",
        corpus_version="m35.corpus.win48o16.v1",
        chunk_mode="windows",
        chunk_size=48,
        chunk_overlap=16,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker_id=RERANKER_IDENTITY,
    )
    merged = evaluate_set(config=merged_cfg, queries=queries, source_corpus=source)
    win32 = evaluate_set(config=win32_cfg, queries=queries, source_corpus=source)
    win48 = evaluate_set(config=win48_cfg, queries=queries, source_corpus=source)
    hard_cfg = ExperimentConfig(
        experiment_id="hard-negatives",
        corpus_version=CANONICAL_CORPUS_VERSION,
        chunk_mode="canonical",
        chunk_size=None,
        chunk_overlap=0,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker_id=RERANKER_IDENTITY,
        hard_negatives=True,
    )
    hard = evaluate_set(config=hard_cfg, queries=queries, source_corpus=source)
    leak_cfg = ExperimentConfig(
        experiment_id="leak-eval-phrasing",
        corpus_version=CANONICAL_CORPUS_VERSION,
        chunk_mode="canonical",
        chunk_size=None,
        chunk_overlap=0,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker_id=RERANKER_IDENTITY,
        leaked=True,
    )
    leaked = evaluate_set(config=leak_cfg, queries=queries, source_corpus=source)
    leaked_corpus = leak_eval_phrasing(source, queries)
    repaired_corpus, repaired_labels = repair_eval_boundary(
        broken_corpus=leaked_corpus,
        source_corpus=source,
        frozen_labels=queries,
    )
    repaired = evaluate_set(config=baseline_config(), queries=repaired_labels, source_corpus=repaired_corpus)
    gamed_labels = relabel_after_results(queries, baseline.rows)
    gamed = rescore_with_labels(
        baseline.rows,
        gamed_labels,
        config=baseline_config(),
        corpus=source,
        index=index,
        source_corpus=source,
    )
    ticket_query = next(query for query in queries if query.query_id == "rag-ticket-4412")
    ticket = generate_candidates(
        ticket_query.text,
        query_id=ticket_query.query_id,
        candidate_k=3,
        index=index,
        corpus=source,
    )
    worst = [_compact_row(row) for row in worst_queries(baseline.rows, n=4)]
    interesting = (
        "rag-reset-login",
        "rag-password-procedure",
        "rag-ticket-4412",
        "rag-h-invoice",
        "rag-ceo",
        "rag-fifty",
        "rag-refund-deny",
    )
    transfer = load_transfer_payload()
    payload = {
        "schema_version": 1,
        "note": "Fixture ranking properties, not learner evidence.",
        "eval_version": EVAL_VERSION,
        "pipeline_id": PIPELINE_ID,
        "index_id": index.metadata.index_id,
        "corpus_version": CANONICAL_CORPUS_VERSION,
        "source_hash": source_hash(source),
        "model": index.metadata.embedding.model,
        "version": index.metadata.embedding.version,
        "metric": index.metadata.embedding.metric,
        "normalization": index.metadata.embedding.normalization,
        "default_candidate_k": DEFAULT_CANDIDATE_K,
        "questions_sha256": questions_sha256(),
        "label_hash": label_hash(queries),
        "query_ids": [query.query_id for query in queries],
        "holdout_ids": [query.query_id for query in queries if query.split == "holdout"],
        "dev_ids": [query.query_id for query in queries if query.split == "dev"],
        "baseline": _compact_set(baseline),
        "baseline_rows": {row.query_id: _compact_row(row) for row in baseline.rows if row.query_id in interesting},
        "rerank_lex": _compact_set(reranked),
        "rerank_rows": {row.query_id: _compact_row(row) for row in reranked.rows if row.query_id in interesting},
        "candidate_k1": _compact_set(k1),
        "candidate_k5": _compact_set(k5),
        "chunk_merged": _compact_set(merged),
        "chunk_win32": _compact_set(win32),
        "chunk_win48o16": _compact_set(win48),
        "hard_negatives": _compact_set(hard),
        "hard_rows": {row.query_id: _compact_row(row) for row in hard.rows if row.query_id in interesting},
        "leaked": _compact_set(leaked),
        "repaired": _compact_set(repaired),
        "relabeled": _compact_set(gamed),
        "worst": worst,
        "ticket_k3_cosine_ids": list(ticket.ids()),
        "leaked_source_hash": source_hash(leaked_corpus),
        "repaired_source_hash": source_hash(repaired_corpus),
        "gamed_label_hash": label_hash(gamed_labels),
        "transfer_k": transfer["k"],
        "downloaded": False,
        "network_required": False,
    }
    target = HERE / "expected.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print("wrote", target)
    print("baseline ndcg", baseline.mean_ndcg_at_k, "mrr", baseline.mean_mrr)
    print("rerank ndcg", reranked.mean_ndcg_at_k, "mrr", reranked.mean_mrr)
    print("ticket cosine", ticket.ids())
    print("ticket rerank", reranked.row_map()["rag-ticket-4412"].ranked_ids)
    print("password cosine", baseline.row_map()["rag-password-procedure"].ranked_ids)
    print("password rerank", reranked.row_map()["rag-password-procedure"].ranked_ids)
    print("leaked ndcg", leaked.mean_ndcg_at_k, "repaired", repaired.mean_ndcg_at_k)
    print("gamed ndcg", gamed.mean_ndcg_at_k, "label hash equal", label_hash(gamed_labels) == label_hash(queries))
    print("worst", [row.query_id for row in worst_queries(baseline.rows, n=4)])


if __name__ == "__main__":
    main()
