#!/usr/bin/env python3
"""Freeze M28 teaching embeddings (offline, deterministic, no downloads).

Run from the repository root:

    python datasets/M28/generate_embeddings.py

Canonical tests load the frozen JSON; they do not call a model hub.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missions.M28.embedding_core import (  # noqa: E402
    DIMENSION_NAMES,
    TeachingEncoder,
    cosine_similarity,
    inner_product,
    lexical_overlap,
    load_encoder,
    rank_neighbors,
    retrieve_unchecked,
    swap_account_print,
)

ACC, CRED, PRINT, REFUND, NEG, PAY, MAG, TICKET, WEATHER, LEGAL, GLUE = range(11)


def sem(**weights: float) -> list[float]:
    names = {
        "account": ACC,
        "credentials": CRED,
        "print": PRINT,
        "refund": REFUND,
        "negation": NEG,
        "payment": PAY,
        "magnitude": MAG,
        "ticket": TICKET,
        "weather": WEATHER,
        "legal": LEGAL,
        "glue": GLUE,
    }
    values = [0.0] * 11
    for key, weight in weights.items():
        values[names[key]] = float(weight)
    return values


TOKEN_SEMANTICS = {
    "password": sem(account=1.0, credentials=0.85),
    "login": sem(account=0.9, credentials=0.95),
    "credentials": sem(account=0.8, credentials=1.0),
    "sign": sem(account=0.9, credentials=0.65),
    "account": sem(account=1.0, credentials=0.55),
    "forgot": sem(account=0.85, credentials=0.4),
    "cannot": sem(account=0.7, credentials=0.25, negation=0.15),
    "access": sem(account=0.85, credentials=0.35),
    "reset": sem(account=0.22, print=0.22, glue=0.35),
    "printer": sem(print=1.0),
    "queue": sem(print=0.85),
    "stuck": sem(print=0.7),
    "needs": sem(print=0.25, glue=0.2),
    "please": sem(glue=0.45),
    "approve": sem(refund=0.9),
    "refund": sem(refund=1.0, payment=0.2),
    "customer": sem(refund=0.25, payment=0.35),
    "not": sem(negation=1.0),
    "never": sem(negation=1.0),
    "paid": sem(payment=1.0),
    "pay": sem(payment=0.95),
    "dollars": sem(payment=0.85, magnitude=0.1),
    "fifty": sem(payment=0.25, magnitude=0.2),
    "five": sem(payment=0.15, magnitude=0.15),
    "thousand": sem(payment=0.2, magnitude=1.0),
    "ticket": sem(ticket=1.0),
    "waiting": sem(ticket=0.55, glue=0.1),
    "inspection": sem(ticket=0.7),
    "inspect": sem(ticket=0.65),
    "process": sem(ticket=0.25, payment=0.2, glue=0.2),
    "invoice": sem(payment=0.8, ticket=0.25),
    "rain": sem(weather=1.0),
    "tomorrow": sem(weather=0.8),
    "valley": sem(weather=0.55),
    "expected": sem(weather=0.45, glue=0.1),
    "forecast": sem(weather=0.95),
    "weather": sem(weather=1.0),
    "agreement": sem(legal=1.0),
    "forbids": sem(legal=0.85, negation=0.2),
    "assignment": sem(legal=0.75),
    "service": sem(legal=0.45),
    "then": sem(glue=0.1),
}

STOPWORDS = sorted(
    {
        "a",
        "an",
        "and",
        "are",
        "because",
        "for",
        "i",
        "in",
        "is",
        "my",
        "now",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "do",
    }
)

CANONICAL_PROVENANCE = {
    "family": "v06-teaching-embed",
    "model": "v06-teaching-meanpool",
    "version": "v06.1",
    "dimensions": 12,
    "dimension_names": list(DIMENSION_NAMES),
    "metric": "cosine",
    "normalization": "l2",
    "pooling": "mean",
    "downloaded": False,
    "network_required": False,
    "not_sentence_transformers": True,
    "not_model_hub": True,
    "authored_for": "M28",
}

MISMATCH_PROVENANCE = {
    **CANONICAL_PROVENANCE,
    "model": "v06-teaching-meanpool-alt",
    "version": "v06.2",
    "normalization": "none",
    "pooling": "mean",
}

CORPUS = [
    {
        "id": "d-password-forgot",
        "text": "I forgot my password and cannot sign in.",
        "tags": ["account", "paraphrase-set"],
    },
    {
        "id": "d-login-reset",
        "text": "Please reset the login credentials.",
        "tags": ["account", "paraphrase-set", "lexical-trap-source"],
    },
    {
        "id": "d-cannot-signin",
        "text": "I cannot sign in to my account.",
        "tags": ["account", "paraphrase-set", "low-overlap"],
    },
    {
        "id": "d-printer-reset",
        "text": "Please reset the printer.",
        "tags": ["device", "lexical-trap"],
    },
    {
        "id": "d-printer-queue",
        "text": "The printer needs a reset because the queue is stuck.",
        "tags": ["device", "long-doc"],
    },
    {
        "id": "d-approve-refund",
        "text": "Approve the customer refund.",
        "tags": ["refund", "negation-pair"],
    },
    {
        "id": "d-deny-refund",
        "text": "Do not approve the customer refund.",
        "tags": ["refund", "negation-pair"],
    },
    {
        "id": "d-pay-fifty",
        "text": "The customer paid fifty dollars.",
        "tags": ["payment", "numeric-pair"],
    },
    {
        "id": "d-pay-thousand",
        "text": "The customer paid five thousand dollars.",
        "tags": ["payment", "numeric-pair"],
    },
    {
        "id": "d-ticket-4412",
        "text": "Ticket 4412 is waiting for inspection.",
        "tags": ["ticket", "entity-pair"],
    },
    {
        "id": "d-ticket-4413",
        "text": "Ticket 4413 is waiting for inspection.",
        "tags": ["ticket", "entity-pair"],
    },
    {
        "id": "d-rain",
        "text": "Rain is expected tomorrow in the valley.",
        "tags": ["weather", "domain"],
    },
    {
        "id": "d-legal",
        "text": "The service agreement forbids assignment.",
        "tags": ["legal", "domain"],
    },
    {
        "id": "d-invoice",
        "text": "Please process invoice 99281 now.",
        "tags": ["payment", "ticket"],
    },
]

QUERIES = [
    {
        "id": "q-password",
        "text": "I forgot my password and cannot sign in.",
        "experiment": "useful-whole",
        "relevant": ["d-password-forgot", "d-login-reset", "d-cannot-signin"],
        "traps": ["d-printer-reset", "d-printer-queue"],
    },
    {
        "id": "q-paraphrase",
        "text": "Please reset the login credentials.",
        "experiment": "paraphrase",
        "relevant": ["d-login-reset", "d-password-forgot", "d-cannot-signin"],
        "traps": ["d-printer-reset", "d-printer-queue"],
    },
    {
        "id": "q-printer",
        "text": "Please reset the printer.",
        "experiment": "lexical-semantic",
        "relevant": ["d-printer-reset", "d-printer-queue"],
        "traps": ["d-login-reset"],
    },
    {
        "id": "q-low-overlap",
        "text": "I cannot access my account.",
        "experiment": "lexical-semantic",
        "relevant": ["d-cannot-signin", "d-password-forgot", "d-login-reset"],
        "traps": ["d-printer-reset"],
    },
    {
        "id": "q-negation",
        "text": "Do not approve the customer refund.",
        "experiment": "negation",
        "relevant": ["d-deny-refund"],
        "hard_neighbor": "d-approve-refund",
    },
    {
        "id": "q-numeric",
        "text": "The customer paid fifty dollars.",
        "experiment": "numeric",
        "relevant": ["d-pay-fifty"],
        "hard_neighbor": "d-pay-thousand",
    },
    {
        "id": "q-entity",
        "text": "ticket 4412",
        "experiment": "entity",
        "relevant": ["d-ticket-4412"],
        "hard_neighbor": "d-ticket-4413",
    },
    {
        "id": "q-domain",
        "text": "weather forecast for tomorrow",
        "experiment": "domain",
        "relevant": ["d-rain"],
        "traps": ["d-legal", "d-invoice"],
    },
]


def round_vector(values) -> list[float]:
    return [round(float(value), 10) for value in values]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def ranking_ids(result) -> list[str]:
    return [item.id for item in result.results]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"fixture invariant failed: {message}")


def main() -> None:
    token_spec = {
        "schema_version": 1,
        "residual_scale": 0.08,
        "stopwords": STOPWORDS,
        "dimension_names": list(DIMENSION_NAMES),
        "semantic_width": 11,
        "provenance": CANONICAL_PROVENANCE,
        "token_semantics": TOKEN_SEMANTICS,
        "notes": "Frozen teaching table. Not a Sentence Transformers or model-hub checkpoint.",
    }
    write_json(HERE / "token_table.json", token_spec)
    load_encoder.cache_clear()
    encoder = TeachingEncoder(token_spec)

    embedding_items = []
    catalog_vectors = {}
    for row in CORPUS:
        vector = encoder.encode(row["text"])
        catalog_vectors[row["id"]] = vector
        embedding_items.append(
            {"id": row["id"], "role": "document", "vector": round_vector(vector)}
        )
    for row in QUERIES:
        vector = encoder.encode(row["text"])
        catalog_vectors[row["id"]] = vector
        embedding_items.append(
            {"id": row["id"], "role": "query", "vector": round_vector(vector)}
        )

    embeddings_payload = {
        "schema_version": 1,
        "provenance": CANONICAL_PROVENANCE,
        "items": embedding_items,
    }
    write_json(HERE / "embeddings.json", embeddings_payload)

    mismatch_items = []
    for row in CORPUS:
        raw = encoder.encode(row["text"], pooling="mean", normalization="none")
        swapped = swap_account_print(raw)
        mismatch_items.append(
            {"id": row["id"], "role": "document", "vector": round_vector(swapped)}
        )
    mismatch_payload = {
        "schema_version": 1,
        "provenance": MISMATCH_PROVENANCE,
        "defect": "account and print axes swapped; stored without L2; different model/version",
        "items": mismatch_items,
    }
    write_json(HERE / "mismatch.json", mismatch_payload)

    from missions.M28.embedding_core import EmbeddedItem, Provenance, VectorSpace, provenance_from_mapping

    canon_prov = provenance_from_mapping(CANONICAL_PROVENANCE)
    mismatch_prov = provenance_from_mapping(MISMATCH_PROVENANCE)
    documents = [
        EmbeddedItem(id=row["id"], text=row["text"], vector=tuple(catalog_vectors[row["id"]]), role="document")
        for row in CORPUS
    ]
    space = VectorSpace(provenance=canon_prov, items=tuple(documents))
    mismatch_docs = []
    for row, payload in zip(CORPUS, mismatch_items):
        mismatch_docs.append(
            EmbeddedItem(id=row["id"], text=row["text"], vector=tuple(payload["vector"]), role="document")
        )
    mismatch_space = VectorSpace(provenance=mismatch_prov, items=tuple(mismatch_docs))

    def retrieve(query_id: str, top_k: int | None = None):
        query = next(row for row in QUERIES if row["id"] == query_id)
        return rank_neighbors(
            catalog_vectors[query_id],
            space,
            query_text=query["text"],
            query_id=query_id,
            query_provenance=canon_prov,
            top_k=top_k,
        )

    password = retrieve("q-password")
    paraphrase = retrieve("q-paraphrase")
    printer = retrieve("q-printer")
    low_overlap = retrieve("q-low-overlap")
    negation = retrieve("q-negation")
    numeric = retrieve("q-numeric")
    entity = retrieve("q-entity")
    domain = retrieve("q-domain")

    account_ids = {"d-password-forgot", "d-login-reset", "d-cannot-signin"}
    printer_ids = {"d-printer-reset", "d-printer-queue"}

    require(set(password.ids()[:3]) <= account_ids, f"password top3={password.ids()[:3]}")
    require(not set(password.ids()[:3]) & printer_ids, f"password mixed printers {password.ids()[:3]}")
    require(set(paraphrase.ids()[:3]) <= account_ids, f"paraphrase top3={paraphrase.ids()[:3]}")
    require(paraphrase.top_id in account_ids, f"paraphrase top={paraphrase.top_id}")
    require(printer.top_id in printer_ids, f"printer top={printer.top_id}")
    require(printer.ids()[1] in printer_ids, f"printer second={printer.ids()[1]}")
    require("d-login-reset" not in printer.ids()[:2], f"printer ranked login {printer.ids()[:3]}")

    login_text = "Please reset the login credentials."
    printer_query = "Please reset the printer."
    login_lex = lexical_overlap(printer_query, login_text)
    printer_lex = lexical_overlap(printer_query, "Please reset the printer.")
    require(login_lex >= 0.5, f"login lexical {login_lex}")
    require(printer_lex == 1.0, f"exact printer lexical {printer_lex}")
    login_cos = cosine_similarity(catalog_vectors["q-printer"], catalog_vectors["d-login-reset"])
    printer_cos = cosine_similarity(catalog_vectors["q-printer"], catalog_vectors["d-printer-reset"])
    require(printer_cos > login_cos, f"printer cosine {printer_cos} vs login {login_cos}")

    low_lex = lexical_overlap("I cannot access my account.", login_text)
    low_cos = cosine_similarity(catalog_vectors["q-low-overlap"], catalog_vectors["d-login-reset"])
    require(low_lex < 0.25, f"low overlap lexical {low_lex}")
    require(low_cos > 0.7, f"low overlap cosine {low_cos}")
    require(set(low_overlap.ids()[:3]) <= account_ids, f"low-overlap top3={low_overlap.ids()[:3]}")

    require(negation.top_id == "d-deny-refund", f"negation top={negation.top_id}")
    require(negation.ids()[1] == "d-approve-refund", f"negation second={negation.ids()[1]}")
    require(negation.results[1].score > 0.85, f"approve still high {negation.results[1].score}")

    require(numeric.top_id == "d-pay-fifty", f"numeric top={numeric.top_id}")
    require(numeric.ids()[1] == "d-pay-thousand", f"numeric second={numeric.ids()[1]}")
    require(numeric.results[1].score > 0.85, f"thousand still high {numeric.results[1].score}")

    require(entity.top_id == "d-ticket-4412", f"entity top={entity.top_id}")
    require(entity.ids()[1] == "d-ticket-4413", f"entity second={entity.ids()[1]}")
    require(entity.margin() is not None and entity.margin() < 0.12, f"entity margin {entity.margin()}")

    require(domain.top_id == "d-rain", f"domain top={domain.top_id}")
    require("d-legal" not in domain.ids()[:2], f"domain legal {domain.ids()[:3]}")

    mixed = retrieve_unchecked(
        catalog_vectors["q-password"],
        mismatch_space,
        query_text=QUERIES[0]["text"],
        query_id="q-password",
        query_provenance=canon_prov,
        top_k=5,
        score_fn="dot",
    )
    require(
        mixed.ids()[0] in printer_ids or mixed.ids()[1] in printer_ids,
        f"mixed password should surface a printer doc, got {mixed.ids()[:3]}",
    )
    require(password.ids()[0] not in printer_ids, "canonical password ranked a printer first")

    sum_queue = encoder.encode(
        "The printer needs a reset because the queue is stuck.",
        pooling="sum",
        normalization="none",
    )
    sum_short = encoder.encode("Please reset the printer.", pooling="sum", normalization="none")
    query_l2 = catalog_vectors["q-printer"]
    require(
        inner_product(query_l2, sum_queue) > inner_product(query_l2, sum_short),
        "sum-pooled inner product should prefer the long printer document",
    )
    require(
        cosine_similarity(query_l2, catalog_vectors["d-printer-reset"])
        >= cosine_similarity(query_l2, catalog_vectors["d-printer-queue"]),
        "cosine on L2 should not prefer the long doc over the exact short match",
    )

    transfer = {
        "schema_version": 1,
        "purpose": "Fresh no-AI ranking. Learner answers are not stored.",
        "provenance": {
            "family": "v06-teaching-embed",
            "model": "v06-teaching-hand",
            "version": "v06.1-transfer",
            "dimensions": 4,
            "dimension_names": ["outage", "facility", "baking", "negation"],
            "metric": "cosine",
            "normalization": "l2",
            "pooling": "hand",
            "downloaded": False,
            "network_required": False,
            "not_sentence_transformers": True,
            "not_model_hub": True,
        },
        "query": {
            "id": "t-query",
            "text": "the server is down",
            "vector": [1.0, 0.0, 0.0, 0.0],
        },
        "corpus": [
            {
                "id": "t-host",
                "text": "the host stopped responding",
                "vector": [0.96, 0.28, 0.0, 0.0],
            },
            {
                "id": "t-paint",
                "text": "the server is painted down the hall",
                "vector": [0.2, 0.9797958971, 0.0, 0.0],
            },
            {
                "id": "t-bread",
                "text": "bake the bread longer",
                "vector": [0.0, 0.0, 1.0, 0.0],
            },
            {
                "id": "t-not-down",
                "text": "the server is not down",
                "vector": [0.8, 0.0, 0.0, 0.6],
            },
        ],
        "mismatch_probe": {
            "id": "t-query-alt",
            "text": "the server is down",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "provenance": {
                "family": "v06-teaching-embed",
                "model": "other-encoder",
                "version": "v09.0",
                "dimensions": 4,
                "metric": "dot",
                "normalization": "none",
                "pooling": "hand",
                "downloaded": False,
                "network_required": False,
            },
        },
    }
    write_json(HERE / "transfer.json", transfer)

    expected = {
        "q-password_top3": list(password.ids()[:3]),
        "q-paraphrase_top3": list(paraphrase.ids()[:3]),
        "q-printer_top2": list(printer.ids()[:2]),
        "q-negation_top2": list(negation.ids()[:2]),
        "q-numeric_top2": list(numeric.ids()[:2]),
        "q-entity_top2": list(entity.ids()[:2]),
        "q-domain_top1": domain.top_id,
        "q-low-overlap_top3": list(low_overlap.ids()[:3]),
        "mixed_password_top3": list(mixed.ids()[:3]),
        "negation_approve_score": round(float(negation.results[1].score), 6),
        "numeric_thousand_score": round(float(numeric.results[1].score), 6),
        "entity_margin": round(float(entity.margin() or 0.0), 6),
        "printer_vs_login_cosine": {
            "printer": round(float(printer_cos), 6),
            "login": round(float(login_cos), 6),
        },
    }
    catalog = {
        "schema_version": 1,
        "provenance": {
            "authored_for": "M28",
            "kind": "synthetic-teaching-fixture",
            "downloaded": False,
            "network_required": False,
            "personal_data": False,
        },
        "corpus": CORPUS,
        "queries": QUERIES,
        "expected": expected,
        "pairwise_subset": [
            "d-password-forgot",
            "d-login-reset",
            "d-printer-reset",
            "d-approve-refund",
            "d-deny-refund",
            "d-rain",
        ],
    }
    write_json(HERE / "catalog.json", catalog)

    print("wrote M28 fixtures")
    for label, result in (
        ("password", password),
        ("paraphrase", paraphrase),
        ("printer", printer),
        ("low-overlap", low_overlap),
        ("negation", negation),
        ("numeric", numeric),
        ("entity", entity),
        ("domain", domain),
        ("mixed", mixed),
    ):
        print(label, ranking_ids(result)[:4], [round(item.score, 3) for item in result.results[:4]])


if __name__ == "__main__":
    main()
