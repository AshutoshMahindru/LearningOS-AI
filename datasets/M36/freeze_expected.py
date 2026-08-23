#!/usr/bin/env python3
"""Freeze M36 hybrid-retrieval properties (offline, deterministic).

Run from the repository root:

    python datasets/M36/freeze_expected.py

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

from missions.M36.hybrid_retrieval import (  # noqa: E402
    BACKEND_EXACT,
    BACKEND_TEACHING_GRAPH,
    DEFAULT_CANDIDATE_K,
    DEFAULT_EF,
    DEFAULT_TOP_K,
    FILTER_DEMO_FILTERS,
    FILTER_DEMO_K,
    FILTER_DEMO_QUERY,
    FILTER_DEMO_QUERY_ID,
    FILTER_DEMO_RELEVANT,
    LOW_EF,
    PIPELINE_ID,
    STORE_ID,
    InfraConfig,
    approximate_search,
    compare_to_exact,
    exact_search,
    filter_placement_trace,
    fuse_channels,
    hybrid_search,
    label_hash,
    load_frozen_queries,
    load_query_map,
    load_transfer_payload,
    m35_baseline_report,
    memory_proxy,
    mix_raw_scores,
    open_teaching_store,
    questions_sha256,
    reciprocal_rank_fusion,
    repair_fusion,
    round_metric,
    slice_channel_rows,
    sparse_search,
)


def _ids(result) -> list[str]:
    return list(result.ids())


def main() -> None:
    store = open_teaching_store()
    queries = load_frozen_queries()
    query_map = load_query_map()
    baseline = m35_baseline_report(queries=queries)
    ticket = query_map["rag-ticket-4412"]
    invoice = query_map["rag-h-invoice"]
    password = query_map["rag-password-procedure"]
    ceo = query_map["rag-ceo"]
    legal = query_map["rag-legal-forbid"]

    dense_ticket = exact_search(store, ticket.text, query_id=ticket.query_id, top_k=DEFAULT_TOP_K)
    sparse_ticket = sparse_search(store, ticket.text, query_id=ticket.query_id, top_k=DEFAULT_TOP_K)
    hybrid_ticket = hybrid_search(
        store,
        ticket.text,
        query_id=ticket.query_id,
        top_k=DEFAULT_TOP_K,
        candidate_k=DEFAULT_CANDIDATE_K,
    )
    dense_invoice = exact_search(store, invoice.text, query_id=invoice.query_id, top_k=DEFAULT_TOP_K)
    sparse_invoice = sparse_search(store, invoice.text, query_id=invoice.query_id, top_k=DEFAULT_TOP_K)
    hybrid_invoice = hybrid_search(
        store,
        invoice.text,
        query_id=invoice.query_id,
        top_k=DEFAULT_TOP_K,
        candidate_k=DEFAULT_CANDIDATE_K,
    )
    dense_pw = exact_search(store, password.text, query_id=password.query_id, top_k=5)
    sparse_pw = sparse_search(store, password.text, query_id=password.query_id, top_k=5)
    mixed_pw = mix_raw_scores(dense_pw, sparse_pw, top_k=5, store=store)
    repaired_pw = repair_fusion(
        broken=mixed_pw,
        dense=dense_pw,
        sparse=sparse_pw,
        store=store,
        top_k=5,
    )
    ceo_low = compare_to_exact(store, ceo.text, query_id=ceo.query_id, top_k=3, ef=LOW_EF)
    ceo_high = compare_to_exact(store, ceo.text, query_id=ceo.query_id, top_k=3, ef=4)
    legal_low = compare_to_exact(store, legal.text, query_id=legal.query_id, top_k=3, ef=LOW_EF)
    trace = filter_placement_trace(
        store,
        FILTER_DEMO_QUERY,
        query_id=FILTER_DEMO_QUERY_ID,
        filters=FILTER_DEMO_FILTERS,
        relevant_ids=FILTER_DEMO_RELEVANT,
        top_k=FILTER_DEMO_K,
    )
    mem = memory_proxy(store)
    slices = slice_channel_rows(store, queries, top_k=DEFAULT_TOP_K, candidate_k=DEFAULT_CANDIDATE_K)
    transfer = load_transfer_payload()
    fused_transfer = reciprocal_rank_fusion(
        (transfer["rrf_example"]["dense_ids"], transfer["rrf_example"]["sparse_ids"]),
        rrf_k=int(transfer["rrf_k"]),
    )
    baseline_cfg = InfraConfig(experiment_id="m35-oracle", backend=BACKEND_EXACT)
    effort_cfg = InfraConfig(
        experiment_id="search-effort",
        backend=BACKEND_TEACHING_GRAPH,
        ef=4,
        fusion=None,
    )
    hybrid_cfg = InfraConfig(
        experiment_id="hybrid-rrf",
        backend=BACKEND_EXACT,
        fusion="rrf",
        candidate_k=DEFAULT_CANDIDATE_K,
    )

    payload = {
        "schema_version": 1,
        "note": "Fixture infrastructure properties, not learner evidence.",
        "eval_version": "m34.eval.v1",
        "pipeline_id": PIPELINE_ID,
        "store_id": STORE_ID,
        "exact_index_id": store.exact.metadata.index_id,
        "corpus_version": store.metadata.corpus_version,
        "source_hash": store.metadata.source_hash,
        "model": store.metadata.embedding_model,
        "version": store.metadata.embedding_version,
        "entry_id": store.adjacency.entry_id,
        "graph_m": store.adjacency.degree_m,
        "graph_long_range": store.adjacency.long_range,
        "chunk_count": store.metadata.chunk_count,
        "questions_sha256": questions_sha256(),
        "label_hash": label_hash(queries),
        "query_ids": [query.query_id for query in queries],
        "baseline": {
            "mean_recall_at_k": round_metric(baseline.mean_recall_at_k),
            "mean_mrr": round_metric(baseline.mean_mrr),
            "mean_ndcg_at_k": round_metric(baseline.mean_ndcg_at_k),
            "scored_candidates": baseline.scored_candidates,
            "index_id": baseline.index_id,
            "ticket_ranked_ids": list(baseline.row_map()["rag-ticket-4412"].ranked_ids),
        },
        "configs": {
            "m35-oracle": baseline_cfg.identity(),
            "search-effort": effort_cfg.identity(),
            "hybrid-rrf": hybrid_cfg.identity(),
        },
        "ann": {
            "rag-ceo": {
                "ef_1": ceo_low.as_dict(),
                "ef_4": ceo_high.as_dict(),
            },
            "rag-legal-forbid": {
                "ef_1": legal_low.as_dict(),
            },
        },
        "filter_demo": trace.as_dict(),
        "channels": {
            "rag-ticket-4412": {
                "dense_ids": _ids(dense_ticket),
                "sparse_ids": _ids(sparse_ticket),
                "hybrid_ids": _ids(hybrid_ticket),
            },
            "rag-h-invoice": {
                "dense_ids": _ids(dense_invoice),
                "sparse_ids": _ids(sparse_invoice),
                "hybrid_ids": _ids(hybrid_invoice),
            },
            "rag-password-procedure": {
                "dense_ids": _ids(dense_pw),
                "sparse_ids": _ids(sparse_pw),
                "mix_ids": _ids(mixed_pw),
                "repaired_rrf_ids": _ids(repaired_pw),
            },
        },
        "memory_proxy": mem.as_dict(),
        "slices": [
            {
                "query_id": row["query_id"],
                "support_id": row["support_id"],
                "dense_top": row["dense_top"],
                "sparse_top": row["sparse_top"],
                "hybrid_top": row["hybrid_top"],
                "dense_support_rank": row["dense_support_rank"],
                "sparse_support_rank": row["sparse_support_rank"],
                "hybrid_support_rank": row["hybrid_support_rank"],
            }
            for row in slices
        ],
        "transfer_rrf_order": [chunk_id for chunk_id, _score in fused_transfer],
        "default_top_k": DEFAULT_TOP_K,
        "default_candidate_k": DEFAULT_CANDIDATE_K,
        "default_ef": DEFAULT_EF,
        "downloaded": False,
        "network_required": False,
    }
    target = HERE / "expected.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
