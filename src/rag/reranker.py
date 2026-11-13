"""Heuristic reranking for fused retrieval results.

RRF only reads ranks, so two chunks with the same fused score are ordered by
whichever leg saw them first. The reranker reads the actual text and the
section path, which matters for standards: a query mentioning ``TB 10621`` or
"最小曲线半径" should prefer the section whose heading carries that clause,
even when the dense leg ranked a generic chapter first.

The reranker is deterministic and dependency-free on purpose: it runs in CI and
in the offline demo, and its weights can be reviewed like any other policy.
"""
from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

from .text import coverage, normalize_text, query_terms

_STANDARD_ID_RE = re.compile(
    r"(?:GB/T|GB|TB/T|TB|ISO|IEC|UIC|EN)\s*/?\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)
_LIMIT_INTENT_RE = re.compile(r"限值|安全|要求|规定|允许|最大|最小|阈值|超标|合规|冲突|风险")


class Reranker(Protocol):
    """Protocol implemented by rerankers."""

    name: str

    def rerank(
        self, query: str, candidates: Sequence[dict[str, Any]], *, top_k: int
    ) -> list[dict[str, Any]]:
        """Reorder candidates for a query."""
        ...


def _standard_ids(text: str) -> set[str]:
    """Extract normalised standard ids (``TB 10621`` and ``TB10621`` match).

    Args:
        text: Query or document text.

    Returns:
        set[str]: Compact ids without spaces or slashes.
    """
    return {
        re.sub(r"[\s/]+", "", match.upper())
        for match in _STANDARD_ID_RE.findall(normalize_text(text))
    }


class HeuristicReranker:
    """Deterministic content-aware reranker.

    Args:
        base_weight: Weight of the fused retrieval score.
        coverage_weight: Weight of query-term coverage in the chunk.
        path_weight: Bonus when a query term appears in the section path.
        standard_id_weight: Bonus when a standard id mentioned in the query
            appears in the chunk or its path.
        limit_intent_weight: Bonus for limit/safety sections when the query
            asks about a requirement or a limit.
        length_penalty: Penalty applied to very short or very long chunks.
    """

    name = "heuristic"

    def __init__(
        self,
        *,
        base_weight: float = 1.0,
        coverage_weight: float = 0.8,
        path_weight: float = 0.25,
        standard_id_weight: float = 0.45,
        limit_intent_weight: float = 0.15,
        length_penalty: float = 0.05,
    ) -> None:
        self.base_weight = base_weight
        self.coverage_weight = coverage_weight
        self.path_weight = path_weight
        self.standard_id_weight = standard_id_weight
        self.limit_intent_weight = limit_intent_weight
        self.length_penalty = length_penalty

    def rerank(
        self, query: str, candidates: Sequence[dict[str, Any]], *, top_k: int
    ) -> list[dict[str, Any]]:
        """Rescore candidates and return the best ``top_k``.

        Args:
            query: User query.
            candidates: Fused candidates carrying ``content``, ``path`` and ``score``.
            top_k: Number of results to return.

        Returns:
            list[dict[str, Any]]: Candidates ordered by ``rerank_score``.
        """
        if not candidates:
            return []

        terms = query_terms(query)
        query_ids = _standard_ids(query)
        limit_intent = bool(_LIMIT_INTENT_RE.search(query))
        rescored: list[dict[str, Any]] = []

        for item in candidates:
            content = str(item.get("content") or "")
            path = str(item.get("path") or item.get("title") or "")
            haystack = f"{path}\n{content}"
            base = float(item.get("score") or 0.0)

            term_coverage = coverage(terms, haystack)
            path_bonus = self.path_weight if path and coverage(terms, path) > 0 else 0.0
            document_ids = _standard_ids(haystack)
            id_bonus = (
                self.standard_id_weight
                if query_ids and document_ids.intersection(query_ids)
                else 0.0
            )
            intent_bonus = (
                self.limit_intent_weight
                if limit_intent and _LIMIT_INTENT_RE.search(haystack)
                else 0.0
            )
            length = len(content)
            penalty = self.length_penalty if length < 120 or length > 900 else 0.0

            score = (
                self.base_weight * base
                + self.coverage_weight * term_coverage
                + path_bonus
                + id_bonus
                + intent_bonus
                - penalty
            )
            enriched = dict(item)
            enriched["rerank_score"] = round(score, 4)
            enriched["query_coverage"] = round(term_coverage, 3)
            enriched["reranker"] = self.name
            rescored.append(enriched)

        rescored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return rescored[: max(1, top_k)]
