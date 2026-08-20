"""Transparent teaching tokenizers for Mission M27.

M27 turns text into token pieces and integer IDs using two bundled local
schemes (word lookup vs tiny deterministic BPE). Semantic embeddings,
attention, and transformer blocks remain deferred to M28-M30.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import re

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_PREFIX = "\u2581"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
CHAR_PER_TOKEN_HEURISTIC = 4
DEFAULT_SCHEME = "bpe"


class TokenBudgetError(ValueError):
    """Raised when a text does not fit a declared token budget."""

    def __init__(self, *, needed: int, max_tokens: int, budget_unit: str, text: str):
        self.needed = needed
        self.max_tokens = max_tokens
        self.budget_unit = budget_unit
        self.text = text
        super().__init__(
            f"{budget_unit} budget overflow: needed {needed}, max_tokens {max_tokens}"
        )


def default_tokenizer_path() -> Path:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "datasets" / "M27" / "teaching_tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing bundled tokenizer fixture: {path}")
    return path


def default_texts_path() -> Path:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "datasets" / "M27" / "texts.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing bundled text fixture: {path}")
    return path


def load_spec(path: Path | None = None) -> dict:
    target = Path(path) if path is not None else default_tokenizer_path()
    return json.loads(target.read_text(encoding="utf-8"))


def load_texts(path: Path | None = None) -> dict:
    target = Path(path) if path is not None else default_texts_path()
    return json.loads(target.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"\s+", " ", text.strip().lower())


def pretokens(text: str) -> tuple[str, ...]:
    """Split normalized text into words and single punctuation marks."""

    return tuple(TOKEN_RE.findall(normalize_text(text)))


def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    left, right = pair
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
            out.append(left + right)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def apply_bpe(token: str, merges: tuple[tuple[str, str], ...], word_prefix: str = WORD_PREFIX) -> tuple[str, ...]:
    """Apply frozen BPE merges inside one pretoken. Punctuation is left intact."""

    if not re.fullmatch(r"\w+", token):
        return (token,)
    symbols = (word_prefix, *tuple(token))
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return symbols


@dataclass(frozen=True)
class TokenizerIdentity:
    family: str
    name: str
    version: str
    scheme: str
    vocab_size: int
    special_tokens: tuple[str, ...]
    truncation_side: str
    padding_side: str
    downloaded: bool
    network_required: bool


@dataclass(frozen=True)
class Encoding:
    text: str
    normalized: str
    tokens: tuple[str, ...]
    ids: tuple[int, ...]
    padding_mask: tuple[int, ...]
    truncated: bool
    dropped_tokens: tuple[str, ...]
    scheme: str
    tokenizer_name: str
    tokenizer_version: str
    special_tokens_added: bool

    @property
    def length(self) -> int:
        return len(self.ids)

    @property
    def content_length(self) -> int:
        return sum(self.padding_mask) - (2 if self.special_tokens_added else 0)


@dataclass(frozen=True)
class BatchEncoding:
    encodings: tuple[Encoding, ...]
    input_ids: tuple[tuple[int, ...], ...]
    padding_mask: tuple[tuple[int, ...], ...]
    max_length: int
    scheme: str
    tokenizer_name: str
    tokenizer_version: str


@dataclass(frozen=True)
class PackedContext:
    text: str
    kept_text: str
    encoding: Encoding
    budget_unit: str
    max_tokens: int
    original_word_count: int
    original_char_count: int
    original_token_count: int
    heuristic_fit: bool
    truncated: bool
    silent: bool
    dropped_text: str
    dropped_tokens: tuple[str, ...]

    def decode(self, tokenizer: "TeachingTokenizer") -> str:
        return tokenizer.decode(self.encoding.ids)

    def contains(self, needle: str, tokenizer: "TeachingTokenizer") -> bool:
        decoded = tokenizer.decode(self.encoding.ids)
        compact = decoded.replace(" ", "")
        target = needle.lower()
        return target in decoded.lower() or target in compact.lower()


class TeachingTokenizer:
    """One frozen teaching scheme: word lookup or tiny BPE."""

    def __init__(self, spec: dict, scheme: str):
        key = _scheme_key(scheme)
        family = str(spec["family"])
        version = str(spec["version"])
        special = tuple(spec["special_tokens"])
        if special != SPECIAL_TOKENS:
            raise ValueError(f"unsupported special-token set: {special}")
        self.word_prefix = str(spec.get("word_prefix", WORD_PREFIX))
        self.normalization = dict(spec.get("normalization") or {})
        self.truncation_side = str(spec.get("truncation_side", "right"))
        self.padding_side = str(spec.get("padding_side", "right"))
        if self.truncation_side != "right" or self.padding_side != "right":
            raise ValueError("teaching tokenizer only ships right-truncation and right-padding")
        table = spec[key]
        vocab = tuple(table["vocab"])
        self._id_of = {token: index for index, token in enumerate(vocab)}
        self._token_of = {index: token for token, index in self._id_of.items()}
        if any(self._id_of[token] != index for index, token in enumerate(SPECIAL_TOKENS)):
            raise ValueError("special tokens must occupy ids 0..3")
        merges = tuple((a, b) for a, b in table.get("merges", ()))
        self.merges = merges
        self.scheme = key
        self.identity = TokenizerIdentity(
            family=family,
            name=str(table["name"]),
            version=version,
            scheme=key,
            vocab_size=len(vocab),
            special_tokens=special,
            truncation_side=self.truncation_side,
            padding_side=self.padding_side,
            downloaded=bool(spec.get("provenance", {}).get("downloaded", False)),
            network_required=bool(spec.get("provenance", {}).get("network_required", False)),
        )
        self.vocab = vocab
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def version(self) -> str:
        return self.identity.version

    def token_to_id(self, token: str) -> int:
        return self._id_of.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        return self._token_of.get(int(token_id), UNK_TOKEN)

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Return content pieces without special tokens."""

        pieces: list[str] = []
        for token in pretokens(text):
            if self.scheme == "word":
                pieces.append(token if token in self._id_of else UNK_TOKEN)
            else:
                pieces.extend(apply_bpe(token, self.merges, self.word_prefix))
        return tuple(pieces)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
        padding: bool = False,
        pad_to_length: int | None = None,
    ) -> Encoding:
        if max_length is not None and int(max_length) < 0:
            raise ValueError("max_length must be non-negative")
        if add_special_tokens and max_length is not None and int(max_length) < 2:
            raise ValueError("max_length must be at least 2 when special tokens are added")

        normalized = normalize_text(text)
        content = self.tokenize(text)
        dropped: tuple[str, ...] = ()
        truncated = False
        room = None
        if max_length is not None and truncation:
            room = int(max_length) - (2 if add_special_tokens else 0)
            if room < 0:
                raise ValueError("max_length too small for special tokens")
            if len(content) > room:
                dropped = content[room:]
                content = content[:room]
                truncated = True

        if add_special_tokens:
            tokens = (BOS_TOKEN, *content, EOS_TOKEN)
        else:
            tokens = content
        ids = tuple(self.token_to_id(token) for token in tokens)
        mask = tuple(1 for _ in ids)

        target = pad_to_length if pad_to_length is not None else (max_length if padding else None)
        if padding:
            if target is None:
                raise ValueError("padding requires pad_to_length or max_length")
            pad_n = int(target) - len(ids)
            if pad_n < 0:
                raise ValueError("sequence longer than pad target; enable truncation")
            tokens = tokens + (PAD_TOKEN,) * pad_n
            ids = ids + (self.pad_id,) * pad_n
            mask = mask + (0,) * pad_n

        return Encoding(
            text=text,
            normalized=normalized,
            tokens=tokens,
            ids=ids,
            padding_mask=mask,
            truncated=truncated,
            dropped_tokens=dropped,
            scheme=self.scheme,
            tokenizer_name=self.name,
            tokenizer_version=self.version,
            special_tokens_added=add_special_tokens,
        )

    def encode_batch(
        self,
        texts: list[str] | tuple[str, ...],
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = True,
        padding: bool = True,
    ) -> BatchEncoding:
        encoded = [
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding=False,
            )
            for text in texts
        ]
        if padding:
            target = int(max_length) if max_length is not None else max((len(item.ids) for item in encoded), default=0)
            encoded = [
                self.encode(
                    item.text,
                    add_special_tokens=add_special_tokens,
                    max_length=target,
                    truncation=truncation,
                    padding=True,
                    pad_to_length=target,
                )
                for item in encoded
            ]
        else:
            target = max((len(item.ids) for item in encoded), default=0)
        return BatchEncoding(
            encodings=tuple(encoded),
            input_ids=tuple(item.ids for item in encoded),
            padding_mask=tuple(item.padding_mask for item in encoded),
            max_length=target,
            scheme=self.scheme,
            tokenizer_name=self.name,
            tokenizer_version=self.version,
        )

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special_tokens: bool = True) -> str:
        tokens = tuple(self.id_to_token(token_id) for token_id in ids)
        return self.decode_pieces(tokens, skip_special_tokens=skip_special_tokens)

    def decode_pieces(self, tokens: list[str] | tuple[str, ...], *, skip_special_tokens: bool = True) -> str:
        kept: list[str] = []
        skip = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN} if skip_special_tokens else set()
        for token in tokens:
            if token in skip:
                continue
            kept.append(token)
        if self.scheme == "bpe":
            raw = "".join(token for token in kept if token != UNK_TOKEN)
            raw = raw.replace(self.word_prefix, " ")
            return re.sub(r"\s+", " ", raw).strip()
        words: list[str] = []
        for token in kept:
            if token == UNK_TOKEN:
                words.append(UNK_TOKEN)
            elif len(token) == 1 and not re.fullmatch(r"\w+", token) and words:
                words[-1] = words[-1] + token
            else:
                words.append(token)
        return " ".join(words)

    def token_count(self, text: str, *, add_special_tokens: bool = True) -> int:
        n_content = len(self.tokenize(text))
        return n_content + (2 if add_special_tokens else 0)

    def trace_bpe_word(self, word: str) -> tuple[tuple[str, ...], ...]:
        """Return merge snapshots for one word-like pretoken."""

        if self.scheme != "bpe":
            raise ValueError("BPE traces are only defined for the bpe scheme")
        token = normalize_text(word)
        if not token or any(ch.isspace() for ch in token):
            raise ValueError("trace one pretoken at a time")
        if not re.fullmatch(r"\w+", token):
            return ((token,),)
        symbols = (self.word_prefix, *tuple(token))
        snapshots = [symbols]
        for pair in self.merges:
            merged = merge_pair(symbols, pair)
            if merged != symbols:
                symbols = merged
                snapshots.append(symbols)
        return tuple(snapshots)


def _scheme_key(scheme: str) -> str:
    key = str(scheme).lower().strip()
    aliases = {
        "word": "word",
        "words": "word",
        "whitespace": "word",
        "whitespace/word": "word",
        "bpe": "bpe",
        "subword": "bpe",
        "byte": "bpe",
        "byte-like": "bpe",
    }
    if key not in aliases:
        raise ValueError(f"unknown scheme {scheme!r}; use 'word' or 'bpe'")
    return aliases[key]


@lru_cache(maxsize=4)
def load_tokenizer(scheme: str = DEFAULT_SCHEME, path: str | None = None) -> TeachingTokenizer:
    spec = load_spec(Path(path) if path else None)
    tokenizer = TeachingTokenizer(spec, scheme)
    if tokenizer.identity.downloaded or tokenizer.identity.network_required:
        raise RuntimeError("M27 required path must use the bundled offline tokenizer")
    return tokenizer


def compare_schemes(text: str) -> dict[str, object]:
    word = load_tokenizer("word")
    bpe = load_tokenizer("bpe")
    word_enc = word.encode(text)
    bpe_enc = bpe.encode(text)
    return {
        "text": text,
        "normalized": normalize_text(text),
        "word_tokens": word_enc.tokens,
        "word_ids": word_enc.ids,
        "word_length": word_enc.length,
        "bpe_tokens": bpe_enc.tokens,
        "bpe_ids": bpe_enc.ids,
        "bpe_length": bpe_enc.length,
        "length_delta": bpe_enc.length - word_enc.length,
        "word_name": word.name,
        "bpe_name": bpe.name,
        "version": word.version,
    }


def pack_for_context(
    text: str,
    tokenizer: TeachingTokenizer,
    *,
    max_tokens: int,
    budget_unit: str = "tokens",
    on_overflow: str = "truncate",
    add_special_tokens: bool = True,
) -> PackedContext:
    """Fit text into a context window using a named counting unit.

    `budget_unit="words"` or `"characters"` is the defective heuristic: a
    cheap count can claim the text fits, after which tokenizer truncation
    still drops a suffix. `budget_unit="tokens"` is the repair.
    """

    unit = str(budget_unit).lower()
    overflow = str(on_overflow).lower()
    if overflow not in {"truncate", "raise"}:
        raise ValueError("on_overflow must be 'truncate' or 'raise'")
    if int(max_tokens) < (2 if add_special_tokens else 0):
        raise ValueError("max_tokens is smaller than the special-token overhead")

    original_words = len(text.split())
    original_chars = len(text)
    original_tokens = tokenizer.token_count(text, add_special_tokens=add_special_tokens)

    if unit == "tokens":
        needed = original_tokens
        heuristic_fit = needed <= int(max_tokens)
        kept_text = text
    elif unit == "words":
        needed = original_words
        heuristic_fit = needed <= int(max_tokens)
        if heuristic_fit:
            kept_text = text
        else:
            kept_text = " ".join(text.split()[: int(max_tokens)])
    elif unit in {"characters", "chars"}:
        needed = (original_chars + CHAR_PER_TOKEN_HEURISTIC - 1) // CHAR_PER_TOKEN_HEURISTIC
        heuristic_fit = original_chars <= int(max_tokens) * CHAR_PER_TOKEN_HEURISTIC
        if heuristic_fit:
            kept_text = text
        else:
            kept_text = text[: int(max_tokens) * CHAR_PER_TOKEN_HEURISTIC]
    else:
        raise ValueError("budget_unit must be 'tokens', 'words', or 'characters'")

    if overflow == "raise" and not heuristic_fit:
        raise TokenBudgetError(needed=needed, max_tokens=int(max_tokens), budget_unit=unit, text=text)

    encoding = tokenizer.encode(
        kept_text,
        add_special_tokens=add_special_tokens,
        max_length=int(max_tokens),
        truncation=True,
        padding=False,
    )
    dropped_tokens = encoding.dropped_tokens
    dropped_text = tokenizer.decode_pieces(dropped_tokens, skip_special_tokens=True)
    silent = bool(heuristic_fit and encoding.truncated)
    return PackedContext(
        text=text,
        kept_text=kept_text,
        encoding=encoding,
        budget_unit=unit,
        max_tokens=int(max_tokens),
        original_word_count=original_words,
        original_char_count=original_chars,
        original_token_count=original_tokens,
        heuristic_fit=heuristic_fit,
        truncated=encoding.truncated or (kept_text != text),
        silent=silent,
        dropped_text=dropped_text,
        dropped_tokens=dropped_tokens,
    )


def encoding_report(encoding: Encoding) -> dict[str, object]:
    """Compact observable evidence for one encoded string."""

    return {
        "scheme": encoding.scheme,
        "tokenizer": encoding.tokenizer_name,
        "version": encoding.tokenizer_version,
        "normalized": encoding.normalized,
        "tokens": encoding.tokens,
        "ids": encoding.ids,
        "padding_mask": encoding.padding_mask,
        "length": encoding.length,
        "truncated": encoding.truncated,
        "dropped_tokens": encoding.dropped_tokens,
    }
