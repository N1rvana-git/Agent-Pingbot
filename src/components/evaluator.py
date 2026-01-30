"""Retrieval evaluator component."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import Settings


@dataclass
class EvaluationResult:
    """Evaluation result for a document.

    Args:
        score: Relevance score in [-1, 1].
        rationale: Optional explanation text.
    """

    score: float
    rationale: str


class EvaluationSchema(BaseModel):
    """Strict schema for evaluator output.

    Args:
        relevance_score: Score in [-1, 1].
        reasoning: Brief reason for score.
    """

    relevance_score: float = Field(..., ge=-1.0, le=1.0)
    reasoning: str


class RetrievalEvaluator:
    """Evaluate relevance between query and documents."""

    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        """Initialize evaluator.

        Args:
            settings: Project settings.
            logger: Optional logger.
        """
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
                (
                    "user",
                    "Query: {query}\nDocument: {document}\n\n{format_instructions}",
                ),
            ]
        )

    def score_documents(self, query: str, documents: List[str]) -> List[EvaluationResult]:
        """Score each document for relevance to the query.

        Args:
            query: User query.
            documents: List of document texts.

        Returns:
            List[EvaluationResult]: Scores per document.
        """
        if not documents:
            return []

        if not self._llm:
            return self._fallback_scores(query, documents)

        results: List[EvaluationResult] = []
        for doc in documents:
            score, rationale = self._score_with_llm(query, doc)
            results.append(EvaluationResult(score=score, rationale=rationale))
        return results

    def _score_with_llm(self, query: str, document: str) -> tuple[float, str]:
        """Score a document using LLM with strict parsing.

        Args:
            query: User query.
            document: Document text.

        Returns:
            tuple[float, str]: (score, rationale)
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
        except Exception as exc:
            self._logger.exception("Evaluator parse error: %s", exc)
            return 0.0, "parse_error"

    def _fallback_scores(self, query: str, documents: List[str]) -> List[EvaluationResult]:
        """Fallback lexical scoring using Jaccard similarity.

        Args:
            query: User query.
            documents: List of document texts.

        Returns:
            List[EvaluationResult]: Scores per document.
        """
        query_tokens = self._tokenize(query)
        results: List[EvaluationResult] = []
        for doc in documents:
            doc_tokens = self._tokenize(doc)
            score = self._jaccard_score(query_tokens, doc_tokens)
            scaled = max(-1.0, min(1.0, 2 * score - 1))
            results.append(EvaluationResult(score=scaled, rationale="token_overlap"))
        return results

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into a set of normalized tokens.

        Args:
            text: Input text.

        Returns:
            set[str]: Token set.
        """
        return {token for token in text.lower().split() if token.strip()}

    def _jaccard_score(self, a: set[str], b: set[str]) -> float:
        """Compute Jaccard similarity.

        Args:
            a: Token set A.
            b: Token set B.

        Returns:
            float: Jaccard similarity in [0, 1].
        """
        if not a or not b:
            return 0.0
        intersection = a.intersection(b)
        union = a.union(b)
        return len(intersection) / max(1, len(union))


def determine_crag_action(scores: List[float], upper: float = 0.5, lower: float = -0.5) -> str:
    """Determine CRAG action based on score thresholds.

    Args:
        scores: List of relevance scores.
        upper: Upper threshold for Correct.
        lower: Lower threshold for Incorrect.

    Returns:
        str: One of "correct", "incorrect", or "ambiguous".
    """
    if not scores:
        return "ambiguous"
    if any(score > upper for score in scores):
        return "correct"
    if all(score < lower for score in scores):
        return "incorrect"
    return "ambiguous"
