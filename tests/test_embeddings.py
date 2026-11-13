"""Tests for embedding backends."""
from __future__ import annotations

import unittest

from src.rag.embeddings import (
    HashingEmbeddingFunction,
    build_embedding_function,
    describe_embedding,
)
from tests.support import make_settings


class HashingEmbeddingTests(unittest.TestCase):
    """Chroma calls both ``__call__`` and ``embed_query``; both must return batches."""

    def test_call_returns_one_vector_per_document(self) -> None:
        function = HashingEmbeddingFunction(dims=32, quiet=True)
        vectors = function(["铁路 标准", "隧道 限界"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 32)

    def test_embed_query_returns_a_batch(self) -> None:
        function = HashingEmbeddingFunction(dims=32, quiet=True)
        vectors = function.embed_query(["最小曲线半径"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 32)

    def test_same_text_produces_same_vector(self) -> None:
        function = HashingEmbeddingFunction(dims=32, quiet=True)
        self.assertEqual(function(["TB 10621"]), function(["TB 10621"]))

    def test_metadata_is_exposed(self) -> None:
        function = HashingEmbeddingFunction(dims=64, quiet=True)
        self.assertEqual(function.name(), "hashing-64")
        self.assertEqual(function.get_config(), {"dims": 64})
        self.assertFalse(function.is_semantic)

    def test_invalid_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HashingEmbeddingFunction(dims=0, quiet=True)


class BuildEmbeddingTests(unittest.TestCase):
    """Backend selection and visible degradation."""

    def test_auto_without_endpoint_falls_back_to_hashing(self) -> None:
        settings = make_settings(embedding_backend="auto", embedding_base_url="")
        function = build_embedding_function(settings, quiet=True)
        self.assertEqual(function.name(), "hashing-512")

    def test_explicit_hashing_backend(self) -> None:
        settings = make_settings(embedding_backend="hashing")
        function = build_embedding_function(settings, quiet=True)
        self.assertFalse(describe_embedding(function)["semantic"])

    def test_openai_backend_without_endpoint_is_an_error(self) -> None:
        settings = make_settings(embedding_backend="openai", embedding_base_url="")
        with self.assertRaises(ValueError):
            build_embedding_function(settings, quiet=True)

    def test_describe_embedding_reports_backend(self) -> None:
        settings = make_settings(embedding_backend="hashing")
        metadata = describe_embedding(build_embedding_function(settings, quiet=True))
        self.assertEqual(metadata["backend"], "hashing-512")
        self.assertEqual(metadata["dim"], 512)


if __name__ == "__main__":
    unittest.main()
