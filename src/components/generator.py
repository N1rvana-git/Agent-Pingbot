"""Answer generator using LLM."""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from ..config import Settings


class AnswerGenerator:
    """Generate final answers from context and question."""

    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        """Initialize generator.

        Args:
            settings: Project settings.
            logger: Optional logger.
        """
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        self._client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def generate(self, question: str, context: str) -> str:
        """Generate answer based on context.

        Args:
            question: User question.
            context: Retrieved/refined context.

        Returns:
            str: Generated answer.
        """
        if not self._client:
            self._logger.warning("OPENAI_API_KEY not set; returning context-only answer")
            return context

        prompt = (
            "Answer the question based strictly on the provided context. "
            "If the context is insufficient, state that you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self._settings.gen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as exc:
            self._logger.exception("Generator LLM error: %s", exc)
            return ""
