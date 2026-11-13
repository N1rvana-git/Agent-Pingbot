"""Retrieval evaluator for the CRAG loop.

The evaluator scores each retrieved chunk in ``[-1, 1]`` and the graph turns the
scores into a Correct / Incorrect / Ambiguous decision. With an API key the
score comes from a strict JSON-parsed LLM call; without one, lexical overlap
keeps the pipeline (and CI) fully offline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import Settings
from ..rag.text import tokenize


@dataclass
class EvaluationResult:
    """Evaluation result for a document.

    Args:
        score: Relevance score in ``[-1, 1]``.
        rationale: Explanation or the name of the fallback heuristic.
    """

    score: float
    rationale: str


class EvaluationSchema(BaseModel):
    """Strict schema for evaluator output.

    Args:
        relevance_score: Score in ``[-1, 1]``.
        reasoning: Brief reason for the score.
    """

    relevance_score: float = Field(..., ge=-1.0, le=1.0)
    reasoning: str


class RetrievalEvaluator:
    """Score how well retrieved documents answer the query.

    Args:
        settings: Runtime settings.
        logger: Optional logger.
    """

    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        self._llm = (
            ChatOpenAI(model=settings.eval_model, temperature=0, api_key=settings.openai_api_key)
            if settings.openai_api_key
            else None
        )
        self._parser = PydanticOutputParser(pydantic_object=EvaluationSchema)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a strict retrieval evaluator. Assess if the document answers the query."),
                ("user", "Query: {query}\nDocument: {document}\n\n{format_instructions}"),
            ]
        )

    @property
    def uses_llm(self) -> bool:
        """Whether the evaluator scores documents with an LLM."""
        return self._llm is not None

    def score_documents(self, query: str, documents: List[str]) -> List[EvaluationResult]:
        """Score each document for relevance to the query.

        Args:
            query: User query.
            documents: Retrieved document texts.

        Returns:
            List[EvaluationResult]: One score per document.
        """
        if not documents:
            return []
        if not self._llm:
            return self._fallback_scores(query, documents)

        results: List[EvaluationResult] = []
        for document in documents:
            score, rationale = self._score_with_llm(query, document)
            results.append(EvaluationResult(score=score, rationale=rationale))
        return results

    def _score_with_llm(self, query: str, document: str) -> tuple[float, str]:
        """Score one document with the LLM.

        Args:
            query: User query.
            document: Document text.

        Returns:
            tuple[float, str]: Score and rationale; ``0.0`` on parse failure.
        """
        try:
            chain = self._prompt | self._llm | self._parser
            result: EvaluationSchema = chain.invoke(
                {
                    "query": query,
                    "document": document,
                    "format_instructions": self._parser.get_format_instructions(),
                }
            )
            return result.relevance_score, result.reasoning
        except Exception as exc:  # noqa: BLE001 - a failed judge must not break the graph
            self._logger.warning("Evaluator parse error: %s", exc)
            return 0.0, "parse_error"

    def _fallback_scores(self, query: str, documents: List[str]) -> List[EvaluationResult]:
        """Score documents by query-term coverage (works for Chinese).

        Jaccard would punish long chunks merely for being long, which is
        backwards here: a section that contains every query term is relevant
        even when it also contains a table. Coverage of the *query* terms
        answers "does this chunk talk about what was asked?" directly.

        Args:
            query: User query.
            documents: Document texts.

        Returns:
            List[EvaluationResult]: Spread scores in ``[-1, 1]``.
        """
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return [EvaluationResult(score=0.0, rationale="empty_query") for _ in documents]

        results: List[EvaluationResult] = []
        for document in documents:
            document_tokens = set(tokenize(document))
            similarity = len(query_tokens & document_tokens) / len(query_tokens)
            # Map coverage [0, 1] onto [-1, 1]: no overlap -> incorrect,
            # half the query terms -> ambiguous boundary, most terms -> correct.
            scaled = max(-1.0, min(1.0, 2 * similarity - 1))
            results.append(EvaluationResult(score=scaled, rationale="query_coverage"))
        return results


def determine_crag_action(scores: List[float], upper: float = 0.5, lower: float = -0.5) -> str:
    """Determine the CRAG action from relevance scores.

    Args:
        scores: Relevance scores.
        upper: Threshold above which the retrieval is Correct.
        lower: Threshold below which every result is Incorrect.

    Returns:
        str: ``correct``, ``incorrect`` or ``ambiguous``.
    """
    if not scores:
        return "ambiguous"
    if any(score > upper for score in scores):
        return "correct"
    if all(score < lower for score in scores):
        return "incorrect"
    return "ambiguous"
