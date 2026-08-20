#!/usr/bin/env python3
"""Freeze M33 search fixtures (offline, deterministic, no downloads).

Run from the repository root:

    python datasets/M33/generate_index.py

Canonical tests load the frozen JSON; they do not download an encoder
or talk to Qdrant. Chunk vectors are copies of M28 bundled embeddings
with explicit provenance.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missions.M33.semantic_search import (  # noqa: E402
    CORPUS_VERSION,
    FILTER_SCHEMA,
    INDEX_BACKEND,
    INDEX_ID,
    STALE_POLICY,
    TIE_BREAK,
    VECTOR_DIGITS,
    build_index,
    evaluate_labeled,
    load_encoder,
    search,
    source_hash,
    vector_hash,
)

M28 = ROOT / "datasets" / "M28"

DOCUMENT_SPECS = (
    {
        "document_id": "doc-account-access",
        "title": "Account access",
        "metadata": {"topic": "account", "source": "kb-support", "locale": "en"},
        "m28_ids": ("d-password-forgot", "d-login-reset", "d-cannot-signin"),
    },
    {
        "document_id": "doc-device-printer",
        "title": "Printer hardware",
        "metadata": {"topic": "device", "source": "kb-hardware", "locale": "en"},
        "m28_ids": ("d-printer-reset", "d-printer-queue"),
    },
    {
        "document_id": "doc-refund-policy",
        "title": "Refund decisions",
        "metadata": {"topic": "refund", "source": "kb-policy", "locale": "en"},
        "m28_ids": ("d-approve-refund", "d-deny-refund"),
    },
    {
        "document_id": "doc-payments",
        "title": "Payments and invoices",
        "metadata": {"topic": "payment", "source": "kb-billing", "locale": "en"},
        "m28_ids": ("d-pay-fifty", "d-pay-thousand", "d-invoice"),
    },
    {
        "document_id": "doc-tickets",
        "title": "Inspection tickets",
        "metadata": {"topic": "ticket", "source": "ops-log", "locale": "en"},
        "m28_ids": ("d-ticket-4412", "d-ticket-4413"),
    },
    {
        "document_id": "doc-weather",
        "title": "Valley weather",
        "metadata": {"topic": "weather", "source": "kb-misc", "locale": "en"},
        "m28_ids": ("d-rain",),
    },
    {
        "document_id": "doc-legal",
        "title": "Service agreement",
        "metadata": {"topic": "legal", "source": "kb-policy", "locale": "en"},
        "m28_ids": ("d-legal",),
    },
)


def _load(name: str) -> dict:
    return json.loads((M28 / name).read_text(encoding="utf-8"))


def _round_vector(values: list[float]) -> list[float]:
    return [round(float(value), VECTOR_DIGITS) for value in values]


def _chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}::c{index}"


def build_corpus_payload(catalog: dict) -> tuple[dict, dict[str, str]]:
    texts = {row["id"]: row["text"] for row in catalog["corpus"]}
    m28_to_chunk: dict[str, str] = {}
    documents = []
    for spec in DOCUMENT_SPECS:
        chunks = []
        for index, m28_id in enumerate(spec["m28_ids"]):
            chunk_id = _chunk_id(spec["document_id"], index)
            m28_to_chunk[m28_id] = chunk_id
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "local_id": f"c{index}",
                    "text": texts[m28_id],
                    "m28_id": m28_id,
                    "metadata": spec["metadata"],
                }
            )
        documents.append(
            {
                "document_id": spec["document_id"],
                "title": spec["title"],
                "metadata": spec["metadata"],
                "chunks": chunks,
            }
        )
    payload = {
        "schema_version": 1,
        "version": CORPUS_VERSION,
        "authored_for": "M33",
        "source_mission": "M28",
        "kind": "synthetic-teaching-fixture",
        "downloaded": False,
        "network_required": False,
        "personal_data": False,
        "separator": "\\n\\n",
        "filter_schema": list(FILTER_SCHEMA),
        "documents": documents,
        "provenance": {
            "authored_for": "M33",
            "copied_from": "datasets/M28/catalog.json",
            "kind": "synthetic-teaching-fixture",
            "downloaded": False,
            "network_required": False,
            "personal_data": False,
        },
    }
    return payload, m28_to_chunk


def build_vector_payload(embeddings: dict, m28_to_chunk: dict[str, str], *, authored_for: str = "M33") -> dict:
    provenance = dict(embeddings["provenance"])
    provenance["authored_for"] = authored_for
    provenance["copied_from"] = "datasets/M28/embeddings.json"
    provenance["source_mission"] = "M28"
    by_id = {row["id"]: row["vector"] for row in embeddings["items"] if row.get("role") == "document"}
    items = []
    for m28_id, chunk_id in m28_to_chunk.items():
        items.append(
            {
                "chunk_id": chunk_id,
                "m28_id": m28_id,
                "vector": _round_vector(by_id[m28_id]),
            }
        )
    return {"schema_version": 1, "provenance": provenance, "items": items}


def _map_ids(ids: list[str], m28_to_chunk: dict[str, str]) -> list[str]:
    return [m28_to_chunk[item] for item in ids]


def build_queries_payload(catalog: dict, m28_to_chunk: dict[str, str]) -> dict:
    queries = []
    for row in catalog["queries"]:
        mapped = {
            "id": row["id"],
            "text": row["text"],
            "experiment": row.get("experiment"),
            "relevant": _map_ids(list(row.get("relevant", [])), m28_to_chunk),
            "traps": _map_ids(list(row.get("traps", [])), m28_to_chunk),
        }
        if row.get("hard_neighbor"):
            mapped["hard_neighbor"] = m28_to_chunk[row["hard_neighbor"]]
        queries.append(mapped)
    return {
        "schema_version": 1,
        "authored_for": "M33",
        "source_mission": "M28",
        "downloaded": False,
        "network_required": False,
        "note": "Labels are relevance judgments, not cosine ranks. Do not relabel after seeing scores.",
        "queries": queries,
    }


def build_transfer_payload() -> dict:
    provenance = {
        "family": "v08-teaching-embed",
        "model": "v08-teaching-hand",
        "version": "v08.1-transfer",
        "dimensions": 3,
        "dimension_names": ["outage", "facility", "baking"],
        "metric": "cosine",
        "normalization": "l2",
        "pooling": "hand",
        "downloaded": False,
        "network_required": False,
        "not_sentence_transformers": True,
        "not_model_hub": True,
    }
    corpus = {
        "version": "m33.transfer.v1",
        "authored_for": "M33",
        "documents": [
            {
                "document_id": "doc-outage",
                "title": "Outage notes",
                "metadata": {"topic": "infra", "source": "ops", "locale": "en"},
                "chunks": [
                    {
                        "chunk_id": "t-offline",
                        "local_id": "c0",
                        "text": "the server is unreachable",
                        "m28_id": "",
                        "metadata": {"topic": "infra", "source": "ops", "locale": "en"},
                    },
                    {
                        "chunk_id": "t-host",
                        "local_id": "c1",
                        "text": "the host stopped responding",
                        "m28_id": "",
                        "metadata": {"topic": "infra", "source": "ops", "locale": "en"},
                    },
                ],
            },
            {
                "document_id": "doc-facility",
                "title": "Facility notes",
                "metadata": {"topic": "facility", "source": "ops", "locale": "en"},
                "chunks": [
                    {
                        "chunk_id": "t-not-down",
                        "local_id": "c0",
                        "text": "the server is not down",
                        "m28_id": "",
                        "metadata": {"topic": "facility", "source": "ops", "locale": "en"},
                    },
                    {
                        "chunk_id": "t-paint",
                        "local_id": "c1",
                        "text": "the server is painted down the hall",
                        "m28_id": "",
                        "metadata": {"topic": "facility", "source": "ops", "locale": "en"},
                    },
                ],
            },
        ],
    }
    vectors = [
        {"chunk_id": "t-offline", "vector": [1.0, 0.0, 0.0]},
        {"chunk_id": "t-host", "vector": [0.8, 0.6, 0.0]},
        {"chunk_id": "t-not-down", "vector": [0.96, 0.0, 0.28]},
        {"chunk_id": "t-paint", "vector": [0.0, 1.0, 0.0]},
    ]
    queries = [
        {
            "id": "t-q-down",
            "text": "the server is down",
            "experiment": "transfer",
            "relevant": ["t-offline", "t-host"],
            "traps": ["t-not-down", "t-paint"],
        }
    ]
    stale = {
        "chunk_id": "t-offline",
        "indexed_text": "the server is unreachable",
        "live_text": "the server was patched and is healthy",
        "note": "Change live text without rebuilding. Unchecked search still returns indexed_text.",
    }
    query_vector = [1.0, 0.0, 0.0]
    return {
        "schema_version": 1,
        "purpose": "Fresh no-AI exact top-k, labeled success, and stale-index diagnosis. Learner answers are not stored.",
        "provenance": provenance,
        "corpus": corpus,
        "vectors": vectors,
        "queries": queries,
        "query": {"id": "t-q-down", "text": "the server is down", "vector": query_vector},
        "stale_probe": stale,
        "expected": {
            "ranking": ["t-offline", "t-not-down", "t-host", "t-paint"],
            "scores": [1.0, 0.96, 0.8, 0.0],
            "recall_at_2": 0.5,
            "hit_at_1": True,
            "high_score_not_relevant": "t-not-down",
        },
        "downloaded": False,
        "network_required": False,
    }


def _write(name: str, payload: dict) -> None:
    path = HERE / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def main() -> None:
    catalog = _load("catalog.json")
    embeddings = _load("embeddings.json")
    mismatch = _load("mismatch.json")
    corpus_payload, m28_to_chunk = build_corpus_payload(catalog)
    vectors_payload = build_vector_payload(embeddings, m28_to_chunk)
    incompatible = build_vector_payload(mismatch, m28_to_chunk)
    incompatible["provenance"]["copied_from"] = "datasets/M28/mismatch.json"
    incompatible["provenance"]["authored_for"] = "M33"
    queries_payload = build_queries_payload(catalog, m28_to_chunk)

    _write("corpus.json", corpus_payload)
    _write("vectors.json", vectors_payload)
    _write("incompatible_vectors.json", incompatible)
    _write("queries.json", queries_payload)
    _write("transfer.json", build_transfer_payload())

    from missions.M33 import semantic_search as core

    core.load_corpus_payload.cache_clear()
    core.load_vectors_payload.cache_clear()
    core.load_queries_payload.cache_clear()
    core.load_canonical_corpus.cache_clear()
    core.load_canonical_vectors.cache_clear()
    core.load_canonical_index.cache_clear()

    corpus = core.load_canonical_corpus()
    index = core.load_canonical_index()
    encoder = load_encoder()
    assert index.metadata.source_hash == source_hash(corpus)
    assert index.metadata.vector_hash == vector_hash(index.records)
    for chunk in corpus.chunks():
        rebuilt = encoder.encode(chunk.text)
        stored = core.load_canonical_vectors()[chunk.chunk_id]
        if any(abs(a - b) > 1e-8 for a, b in zip(rebuilt, stored, strict=True)):
            raise SystemExit(f"encoder/vector drift on {chunk.chunk_id}")

    expected_rankings = {}
    labeled_eval = {}
    for labeled in core.load_labeled_queries():
        response = search(
            index,
            labeled.text,
            top_k=8,
            query_id=labeled.query_id,
            live_corpus=corpus,
        )
        expected_rankings[labeled.query_id] = list(response.ids())
        labeled_eval[labeled.query_id] = {
            "top3": list(response.ids()[:3]),
            "eval_k3": evaluate_labeled(response, labeled, k=3),
        }
        labeled_eval[labeled.query_id]["eval_k3"] = {
            key: (
                list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in labeled_eval[labeled.query_id]["eval_k3"].items()
        }

    filter_device = search(
        index,
        core.load_query_map()["q-password"].text,
        top_k=5,
        filters={"topic": "device"},
        query_id="q-password",
        live_corpus=corpus,
    )
    filter_account = search(
        index,
        core.load_query_map()["q-password"].text,
        top_k=5,
        filters={"topic": "account"},
        query_id="q-password",
        live_corpus=corpus,
    )

    mixed = core.search_unchecked(
        core.load_incompatible_index(),
        core.encode_query(core.load_query_map()["q-password"].text, query_id="q-password"),
        top_k=5,
        enforce_provenance=False,
    )

    expected = {
        "schema_version": 1,
        "note": "Fixture ranking properties, not learner evidence.",
        "index_id": INDEX_ID,
        "backend": INDEX_BACKEND,
        "corpus_version": CORPUS_VERSION,
        "source_hash": index.metadata.source_hash,
        "vector_hash": index.metadata.vector_hash,
        "filter_schema": list(FILTER_SCHEMA),
        "tie_break": list(TIE_BREAK),
        "stale_policy": STALE_POLICY,
        "m28_to_chunk": m28_to_chunk,
        "rankings": expected_rankings,
        "q-password_top3": expected_rankings["q-password"][:3],
        "q-paraphrase_top3": expected_rankings["q-paraphrase"][:3],
        "q-printer_top2": expected_rankings["q-printer"][:2],
        "q-negation_top2": expected_rankings["q-negation"][:2],
        "q-numeric_top2": expected_rankings["q-numeric"][:2],
        "q-entity_top2": expected_rankings["q-entity"][:2],
        "q-domain_top1": expected_rankings["q-domain"][0],
        "q-low-overlap_top3": expected_rankings["q-low-overlap"][:3],
        "filter_topic_device_top2": list(filter_device.ids()[:2]),
        "filter_topic_account_top3": list(filter_account.ids()[:3]),
        "mixed_password_top3": list(mixed.ids()[:3]),
        "labeled": labeled_eval,
        "stale_mutation": {
            "chunk_id": "doc-account-access::c0",
            "indexed_text": corpus.get_chunk("doc-account-access::c0").text,
            "live_text": "Please reset the printer firmware.",
        },
    }
    _write("expected.json", expected)
    print("chunks", index.metadata.chunk_count, "documents", index.metadata.document_count)
    print("source_hash", index.metadata.source_hash)
    print("q-password top3", expected["q-password_top3"])
    print("mixed top3", expected["mixed_password_top3"])


if __name__ == "__main__":
    main()
