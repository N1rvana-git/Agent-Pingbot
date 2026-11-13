"""Query rewriting for the web-search correction step."""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from ..config import Settings

_SYSTEM_PROMPT = (
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


class QueryRewriter:
    """Turn a natural language question into search keywords.

    Args:
        settings: Runtime settings.
        logger: Optional logger.
    """

    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        self._client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def rewrite(self, question: str) -> str:
        """Rewrite the question into a keyword-style search query.

        Args:
            question: User question.

        Returns:
            str: Rewritten query, or the original question when offline.
        """
        if not self._client:
            self._logger.info("No chat model configured; using the original query")
            return question

        try:
            response = self._client.chat.completions.create(
                model=self._settings.rewrite_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Q: {question}\nA:"},
                ],
                temperature=0,
                max_tokens=60,
            )
            content = (response.choices[0].message.content or "").strip() if response.choices else ""
            if "A:" in content:
                content = content.split("A:")[-1].strip()
            return content or question
        except Exception as exc:  # noqa: BLE001 - search is best-effort
            self._logger.warning("Query rewrite failed: %s", exc)
            return question
