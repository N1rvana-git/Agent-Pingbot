"""Web search component."""
from __future__ import annotations

import logging
from typing import List, Optional

from tavily import TavilyClient


class WebSearcher:
    """Web search wrapper using Tavily."""

    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None) -> None:
        """Initialize web searcher.

        Args:
            api_key: Tavily API key.
            logger: Optional logger.
        """
        self._api_key = api_key
        self._logger = logger or logging.getLogger(__name__)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search the web and return top-k snippets.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List[str]: List of result snippets.
        """
        if not self._api_key:
            self._logger.warning("TAVILY_API_KEY not set; skipping web search")
            return []

        try:
            client = TavilyClient(api_key=self._api_key)
            response = client.search(query=query, max_results=top_k)
            results = response.get("results", [])
            snippets = [r.get("content", "") or r.get("snippet", "") for r in results]
            return [s for s in snippets if s]
        except Exception as exc:
            self._logger.exception("Web search failed: %s", exc)
            return []
