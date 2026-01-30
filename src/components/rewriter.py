"""Query rewriting component for web search."""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from ..config import Settings


class QueryRewriter:
    """Rewrite natural language questions into search keywords."""

    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        """Initialize query rewriter.

        Args:
            settings: Project settings.
            logger: Optional logger.
        """
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        self._client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def rewrite(self, question: str) -> str:
        """Rewrite the question into keyword-style search query.

        Args:
            question: User question.

        Returns:
            str: Rewritten search query.
        """
        if not self._client:
            self._logger.warning("OPENAI_API_KEY not set; using original query")
            return question

        system_prompt = (
            "You are a search query optimizer. "
            "Extract at most three keywords separated by comma from the question as queries for web search. "
            "Include topic background and main intent.\n\n"
            "Examples:\n"
            "Q: What is Henry Feilden's occupation?\n"
            "A: Henry Feilden, occupation\n"
            "Q: In what city was Billy Carlson born?\n"
            "A: city, Billy Carlson, born\n"
            "Q: What is the religion of John Gwynn?\n"
            "A: religion of John Gwynn\n"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._settings.rewrite_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Q: {question}\nA:"},
                ],
                temperature=0,
                max_tokens=60,
            )
            text = response.choices[0].message.content.strip()
            if "A:" in text:
                text = text.split("A:")[-1].strip()
            return text or question
        except Exception as exc:
            self._logger.exception("Rewrite error: %s", exc)
            return question
