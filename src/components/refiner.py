"""Knowledge refinement (Decompose-then-Recompose)."""
from __future__ import annotations

import re
from typing import List

from .evaluator import RetrievalEvaluator


class KnowledgeRefiner:
    """Refine knowledge by decomposing, filtering, and recomposing."""

    def __init__(self, evaluator: RetrievalEvaluator) -> None:
        """Initialize refiner.

        Args:
            evaluator: Retrieval evaluator instance.
        """
        self._evaluator = evaluator

    def refine(self, query: str, document: str) -> str:
        """Refine a document into high-relevance strips.

        Args:
            query: User query.
            document: Raw retrieved document.

        Returns:
            str: Refined context string.
        """
        strips = self._split_into_strips(document)
        scores = self._evaluator.score_documents(query, strips)
        kept = [s for s, r in zip(strips, scores) if r.score > 0]
        return "\n".join(kept).strip()

    def _split_into_strips(self, document: str) -> List[str]:
        """Split document into strips (placeholder by sentence).

        Args:
            document: Raw document text.

        Returns:
            List[str]: List of strips.
        """
        strips = re.split(r"(?<=[.!?。！？])\s+", document)
        return [s.strip() for s in strips if s.strip()]
