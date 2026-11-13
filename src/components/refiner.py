"""Knowledge refinement (decompose-filter-recompose).

A retrieved chunk often mixes several clauses; only a few sentences actually
answer the query. The refiner splits the chunk into sentence-level strips,
keeps the strips that share vocabulary with the query, and asks the evaluator to
score only those. The lexical pre-filter matters: scoring every sentence with
an LLM would multiply one retrieval into dozens of calls.
"""
from __future__ import annotations

import re
from typing import List

from ..rag.text import tokenize
from .evaluator import RetrievalEvaluator

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_STRIP_MIN_CHARS = 6
_STRIP_MAX_CHARS = 240
_MAX_STRIPS = 6
_MAX_OFFLINE_STRIPS = 4
_MAX_CONTEXT_CHARS = 1200


class KnowledgeRefiner:
    """Refine a document into high-relevance strips.

    Args:
        evaluator: Retrieval evaluator used to score strips.
        max_strips: Maximum number of strips scored per document.
    """

    def __init__(self, evaluator: RetrievalEvaluator, max_strips: int = _MAX_STRIPS) -> None:
        self._evaluator = evaluator
        self._max_strips = max(1, max_strips)

    def refine(self, query: str, document: str) -> str:
        """Keep the strips of ``document`` that answer ``query``.

        Args:
            query: User query.
            document: Retrieved document text.

        Returns:
            str: Joined relevant strips (empty when nothing survives).
        """
        strips = self._split_into_strips(document)
        if not strips:
            return ""

        query_terms = set(tokenize(query))
        relevant = [
            strip for strip in strips if query_terms.intersection(tokenize(strip))
        ]
        candidates = (relevant or strips)[: self._max_strips]

        scores = self._evaluator.score_documents(query, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1].score, reverse=True)
        if self._evaluator.uses_llm:
            # The LLM judge has calibrated scores; negative means "does not answer".
            kept = [strip for strip, result in ranked if result.score > 0]
        else:
            # Coverage scoring maps "some overlap" to negative values, so a > 0
            # cut-off would drop correct sentences. The lexical pre-filter has
            # already limited candidates to strips sharing query terms; keep
            # the best few so the context stays focused.
            kept = [strip for strip, _result in ranked[:_MAX_OFFLINE_STRIPS]]

        selected: List[str] = []
        total = 0
        for strip in kept:
            if selected and total + len(strip) > _MAX_CONTEXT_CHARS:
                break
            selected.append(strip)
            total += len(strip)
        return "\n".join(selected).strip()

    def _split_into_strips(self, document: str) -> List[str]:
        """Split a document into sentence-sized strips.

        Args:
            document: Raw document text.

        Returns:
            List[str]: Non-empty strips, capped at ``_STRIP_MAX_CHARS``.
        """
        parts = _SENTENCE_BOUNDARY_RE.split(document or "")
        strips: List[str] = []
        for part in parts:
            text = (part or "").strip()
            if len(text) < _STRIP_MIN_CHARS:
                continue
            if len(text) > _STRIP_MAX_CHARS:
                text = text[:_STRIP_MAX_CHARS].rstrip()
            strips.append(text)
        return strips
