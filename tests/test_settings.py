"""Tests for environment-driven settings."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from src.config import SettingsError, load_settings

_BASE_ENV = {
    "OPENAI_API_KEY": "",
    "TAVILY_API_KEY": "",
    "RETRIEVAL_MODE": "hybrid",
    "RETRIEVER_K": "5",
    "RAG_RERANK": "1",
    "CRAG_UPPER_THRESHOLD": "0.5",
    "CRAG_LOWER_THRESHOLD": "-0.5",
    "EMBEDDING_BACKEND": "auto",
    "COLLECTION_NAME": "rail_crag",
}


def _env(**overrides: str) -> dict[str, str]:
    values = dict(_BASE_ENV)
    values.update(overrides)
    return values


class SettingsTests(unittest.TestCase):
    """Parsing, validation and defaults."""

    def test_defaults(self) -> None:
        with mock.patch.dict(os.environ, _env(), clear=False):
            settings = load_settings()
        self.assertEqual(settings.retrieval_mode, "hybrid")
        self.assertEqual(settings.retriever_k, 5)
        self.assertTrue(settings.rerank)
        self.assertEqual(settings.collection_name, "rail_crag")

    def test_values_are_read_from_the_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            _env(RETRIEVAL_MODE="bm25", RETRIEVER_K="3", RAG_RERANK="0", COLLECTION_NAME="kb"),
            clear=False,
        ):
            settings = load_settings()
        self.assertEqual(settings.retrieval_mode, "bm25")
        self.assertEqual(settings.retriever_k, 3)
        self.assertFalse(settings.rerank)
        self.assertEqual(settings.collection_name, "kb")

    def test_invalid_mode_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, _env(RETRIEVAL_MODE="fuzzy"), clear=False):
            with self.assertRaises(SettingsError):
                load_settings()

    def test_invalid_boolean_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, _env(RAG_RERANK="sometimes"), clear=False):
            with self.assertRaises(SettingsError):
                load_settings()

    def test_invalid_integer_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, _env(RETRIEVER_K="five"), clear=False):
            with self.assertRaises(SettingsError):
                load_settings()

    def test_thresholds_must_be_ordered(self) -> None:
        with mock.patch.dict(
            os.environ,
            _env(CRAG_UPPER_THRESHOLD="-0.9", CRAG_LOWER_THRESHOLD="0.9"),
            clear=False,
        ):
            with self.assertRaises(SettingsError):
                load_settings()

    def test_required_keys_are_enforced_when_requested(self) -> None:
        with mock.patch.dict(os.environ, _env(), clear=False):
            with self.assertRaises(SettingsError):
                load_settings(require_keys=True)

    def test_embedding_key_falls_back_to_the_chat_key(self) -> None:
        with mock.patch.dict(
            os.environ, _env(OPENAI_API_KEY="sk-test", EMBEDDING_API_KEY=""), clear=False
        ):
            settings = load_settings()
        self.assertEqual(settings.embedding_api_key, "sk-test")


if __name__ == "__main__":
    unittest.main()
