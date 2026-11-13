"""Answer generation grounded in retrieved context."""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from ..config import Settings

_SYSTEM_PROMPT = (
    "You are a railway standards assistant. Answer strictly from the provided "
    "context. If the context is insufficient, say so explicitly instead of "
    "guessing. When a section path is available, cite it in square brackets. "
    "Reply in the language of the question."
)


class AnswerGenerator:
    """Generate the final answer from retrieved context.

    Args:
        settings: Runtime settings.
        logger: Optional logger.
    """

    def __init__(self, settings: Settings, logger: Optional[logging.Logger] = None) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)
        self._client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def generate(self, question: str, context: str) -> str:
        """Generate an answer grounded in ``context``.

        Args:
            question: User question.
            context: Refined and/or searched context.

        Returns:
            str: Generated answer; the context itself when offline.
        """
        if not context.strip():
            return "未在知识库中检索到相关内容，请补充标准号或问题细节。"
        if not self._client:
            self._logger.info("No chat model configured; returning retrieved context")
            return context

        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        try:
            response = self._client.chat.completions.create(
                model=self._settings.gen_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            if not response.choices:
                return context
            return (response.choices[0].message.content or "").strip() or context
        except Exception as exc:  # noqa: BLE001 - degrade to context instead of failing
            self._logger.warning("Generator LLM error, returning context: %s", exc)
            return context
