#!/usr/bin/env python3
"""Freeze the M27 teaching tokenizer (offline, deterministic, no downloads).

Run from the repository root:

    python datasets/M27/generate_tokenizer.py

The output JSON is the canonical vocabulary. Tests load the frozen files;
they do not retrain a tokenizer or fetch encodings from the network.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_PREFIX = "\u2581"
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")
N_MERGES = 67

BPE_TRAIN = (
    "the cat sat on the mat",
    "the cat sat",
    "the cat sat on the mat then stop",
    "please inspect ticket 4412",
    "please inspect the invoice",
    "please process the invoice now",
    "please process ticket 4412 now",
    "pay the invoice",
    "customer 88 paid the invoice",
    "renew the token budget",
    "the token budget is sixteen tokens",
    "inspect ticket then stop",
    "see the cat on the mat",
    "process ticket 4412 then stop",
    "please inspect ticket then process the invoice",
    "please inspect ticket 4412 then process the invoice now",
    "stop",
    "the cat sat.",
    "pay now",
    "renew now",
)

ALPHABET = (
    (WORD_PREFIX,)
    + tuple("abcdefghijklmnopqrstuvwxyz")
    + tuple("0123456789")
    + tuple(".,!?:/_-")
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def pretokens(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def merge_in_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    a, b = pair
    out: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def word_freqs(corpus: tuple[str, ...]) -> dict[tuple[str, ...], int]:
    freqs: Counter[tuple[str, ...]] = Counter()
    for line in corpus:
        for token in pretokens(line):
            if re.fullmatch(r"\w+", token):
                freqs[(WORD_PREFIX, *tuple(token))] += 1
            else:
                freqs[(token,)] += 1
    return dict(freqs)


def pair_stats(freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    stats: Counter[tuple[str, str]] = Counter()
    for word, freq in freqs.items():
        for left, right in zip(word, word[1:]):
            stats[left, right] += freq
    return dict(stats)


def learn_bpe(corpus: tuple[str, ...], n_merges: int) -> list[list[str]]:
    freqs = word_freqs(corpus)
    merges: list[list[str]] = []
    for _ in range(n_merges):
        stats = pair_stats(freqs)
        if not stats:
            break
        pair = min(stats.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[0]
        if stats[pair] < 2:
            break
        merges.append([pair[0], pair[1]])
        freqs = {merge_in_word(word, pair): freq for word, freq in freqs.items()}
    return merges


def apply_bpe(token: str, merges: list[list[str]]) -> list[str]:
    if not re.fullmatch(r"\w+", token):
        return [token]
    symbols = (WORD_PREFIX, *tuple(token))
    for pair in merges:
        symbols = merge_in_word(symbols, (pair[0], pair[1]))
    return list(symbols)


def main() -> None:
    merges = learn_bpe(BPE_TRAIN, N_MERGES)
    word_pieces = sorted({token for line in BPE_TRAIN for token in pretokens(line)})
    word_vocab = list(SPECIAL_TOKENS) + word_pieces
    bpe_vocab = list(SPECIAL_TOKENS) + [symbol for symbol in ALPHABET if symbol not in SPECIAL_TOKENS]
    for left, right in merges:
        piece = left + right
        if piece not in bpe_vocab:
            bpe_vocab.append(piece)

    tokenizer = {
        "schema_version": 1,
        "family": "v06-teaching-tokenizer",
        "version": "v06.1",
        "provenance": {
            "authored_for": "M27",
            "kind": "synthetic-teaching-fixture",
            "downloaded": False,
            "network_required": False,
            "not_a_huggingface_model": True,
            "not_tiktoken": True,
            "generator": "datasets/M27/generate_tokenizer.py",
            "description": (
                "Tiny deterministic word vocabulary and BPE merge list learned "
                "once from the bundled bpe_train corpus. Not a production encoding."
            ),
        },
        "special_tokens": list(SPECIAL_TOKENS),
        "word_prefix": WORD_PREFIX,
        "normalization": {
            "lowercase": True,
            "strip": True,
            "collapse_whitespace": True,
        },
        "truncation_side": "right",
        "padding_side": "right",
        "word": {
            "name": "v06-teaching-word",
            "scheme": "word",
            "vocab": word_vocab,
        },
        "bpe": {
            "name": "v06-teaching-bpe",
            "scheme": "bpe",
            "alphabet": list(ALPHABET),
            "n_merges": len(merges),
            "merges": merges,
            "vocab": bpe_vocab,
        },
    }

    canonical = "the cat sat on the mat"
    failure = "please inspect ticket 4412 then approve_refund"
    texts = {
        "schema_version": 1,
        "provenance": {
            "authored_for": "M27",
            "kind": "synthetic-teaching-fixture",
            "downloaded": False,
            "network_required": False,
            "personal_data": False,
        },
        "canonical_sentence": canonical,
        "surface_variants": {
            "base": "The cat sat on the mat.",
            "whitespace": "The  cat sat on the mat.",
            "casing": "THE CAT SAT ON THE MAT.",
            "punctuation": "The cat sat on the mat!",
        },
        "rare_strings": {
            "identifier": "ticket xgztq9 is open",
            "url": "see https://learn.os/t/42 now",
            "number": "invoice 99281 is due",
        },
        "padding_batch": [
            "the cat sat",
            "the cat sat on the mat",
            "please inspect ticket 4412 then stop",
        ],
        "truncation_text": "please inspect ticket 4412 then process the invoice now",
        "comparison_corpus": [
            "the cat sat on the mat",
            "please inspect ticket 4412 then stop",
            "ticket xgztq9 is open",
            "invoice 99281 is due",
            "see https://learn.os/t/42 now",
        ],
        "controlled_failure": {
            "text": failure,
            "critical_suffix": "approve_refund",
            "max_tokens": 12,
            "defective_unit": "words",
            "notes": (
                "Six whitespace words look like they fit a 12-token window if "
                "one word equals one token. BPE fragments the suffix."
            ),
        },
        "bpe_train": list(BPE_TRAIN),
        "expected": {
            "canonical_word_tokens": ["[BOS]", *pretokens(canonical), "[EOS]"],
            "canonical_bpe_tokens": ["[BOS]", *sum((apply_bpe(tok, merges) for tok in pretokens(canonical)), []), "[EOS]"],
            "failure_bpe_content": sum((apply_bpe(tok, merges) for tok in pretokens(failure)), []),
        },
    }

    (HERE / "teaching_tokenizer.json").write_text(
        json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (HERE / "texts.json").write_text(
        json.dumps(texts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'teaching_tokenizer.json'}")
    print(f"wrote {HERE / 'texts.json'}")
    print(f"word vocab {len(word_vocab)}  bpe vocab {len(bpe_vocab)}  merges {len(merges)}")
    print("canonical bpe", texts["expected"]["canonical_bpe_tokens"])
    print("failure bpe content", texts["expected"]["failure_bpe_content"])


if __name__ == "__main__":
    main()
