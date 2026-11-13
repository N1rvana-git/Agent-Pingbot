"""Text primitives shared by the retrieval chain.

The project has to run offline (no model download, no network) and still rank
Chinese engineering standards well, so tokenisation uses character n-grams
instead of a word segmenter:

* every CJK character is a token;
* every adjacent CJK pair is a token (recovering most two-character words);
* ASCII runs are normalised with NFKC, then split into word/number tokens so
  that ``TB 10621`` and ``TB10621`` match each other.

Hashing vectors and BM25 are built from the same token stream, which keeps the
two retrieval legs comparable during ablation.
"""
from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections import Counter
from typing import Iterable, Sequence

HASH_VECTOR_DIMS = 512

_CJK_RE = re.compile(r"[\u4e00-\u9fa5]")
_ASCII_RE = re.compile(r"[A-Z0-9]+(?:[./\-][A-Z0-9]+)*")
_ASCII_PART_RE = re.compile(r"[A-Z]+|\d+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalise full-width characters and collapse whitespace.

    Args:
        text: Raw text.

    Returns:
        str: NFKC-normalised text with single spaces.
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def tokenize(text: str) -> list[str]:
    """Tokenise Chinese/English text without an external segmenter.

    Args:
        text: Raw text.

    Returns:
        list[str]: ASCII word/number tokens followed by CJK unigrams and bigrams.
    """
    upper = normalize_text(text).upper()
    cjk_chars = _CJK_RE.findall(upper)
    bigrams = [cjk_chars[index] + cjk_chars[index + 1] for index in range(len(cjk_chars) - 1)]

    ascii_tokens: list[str] = []
    for chunk in _ASCII_RE.findall(upper):
        ascii_tokens.append(chunk)
        # "TB10621" -> ("TB", "10621") so the spaced form is also matched;
        # pure alphabetic or numeric chunks would duplicate themselves.
        parts = _ASCII_PART_RE.findall(chunk)
        if len(parts) > 1:
            ascii_tokens.extend(parts)
    return ascii_tokens + cjk_chars + bigrams


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return the unit-length version of ``vector`` (zero vector stays zero).

    Args:
        vector: Input vector.

    Returns:
        list[float]: L2-normalised vector.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def hash_vector(tokens: Iterable[str], dims: int = HASH_VECTOR_DIMS) -> list[float]:
    """Build a deterministic sublinear-TF hashing vector.

    ``zlib.crc32`` is used instead of ``hash()`` because Python randomises
    string hashing per process, which would make the index unreproducible.

    Args:
        tokens: Token stream.
        dims: Vector dimension.

    Returns:
        list[float]: L2-normalised hashing vector.
    """
    if dims <= 0:
        raise ValueError("dims must be positive")
    counts = Counter(zlib.crc32(token.encode("utf-8")) % dims for token in tokens)
    vector = [0.0] * dims
    for index, count in counts.items():
        vector[index] = 1.0 + math.log(count)
    return l2_normalize(vector)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        float: Similarity in ``[-1, 1]``; ``0.0`` for empty input.
    """
    if not a or not b:
        return 0.0
    return float(sum(left * right for left, right in zip(a, b)))


def query_terms(text: str, *, min_length: int = 2) -> list[str]:
    """Return de-duplicated query terms used for coverage scoring.

    Args:
        text: Query text.
        min_length: Minimum token length kept.

    Returns:
        list[str]: Query terms in first-seen order.
    """
    seen: dict[str, None] = {}
    for token in tokenize(text):
        if len(token) >= min_length and token not in seen:
            seen[token] = None
    return list(seen)


def coverage(terms: Sequence[str], haystack: str) -> float:
    """Fraction of query terms that appear in ``haystack``.

    Args:
        terms: Query terms.
        haystack: Text to search (case-insensitive).

    Returns:
        float: Coverage in ``[0, 1]``.
    """
    if not terms:
        return 0.0
    upper = normalize_text(haystack).upper()
    hits = sum(1 for term in terms if term in upper)
    return hits / len(terms)
