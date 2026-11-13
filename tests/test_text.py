"""Tests for tokenisation and hashing primitives."""
from __future__ import annotations

import unittest

from src.rag.text import (
    cosine,
    coverage,
    hash_vector,
    l2_normalize,
    normalize_text,
    query_terms,
    tokenize,
)


class TokenizeTests(unittest.TestCase):
    """Tokeniser behaviour that BM25 and the reranker depend on."""

    def test_ascii_standard_ids_are_split_into_parts(self) -> None:
        tokens = tokenize("TB 10621")
        self.assertIn("10621", tokens)
        self.assertIn("TB", tokens)

    def test_compact_and_spaced_standard_ids_share_tokens(self) -> None:
        self.assertIn("10621", tokenize("TB10621"))
        self.assertIn("TB", tokenize("TB10621"))

    def test_chinese_unigrams_and_bigrams(self) -> None:
        tokens = tokenize("曲线半径")
        self.assertIn("曲", tokens)
        self.assertIn("曲线", tokens)
        self.assertIn("半径", tokens)

    def test_full_width_characters_are_normalised(self) -> None:
        self.assertEqual(normalize_text("ＴＢ　１０６２１"), "TB 10621")

    def test_empty_text_yields_no_tokens(self) -> None:
        self.assertEqual(tokenize(""), [])


class VectorTests(unittest.TestCase):
    """Hashing vector properties."""

    def test_hash_vector_is_deterministic_across_calls(self) -> None:
        self.assertEqual(hash_vector(tokenize("铁路 标准")), hash_vector(tokenize("铁路 标准")))

    def test_hash_vector_is_unit_length(self) -> None:
        vector = hash_vector(tokenize("高速铁路设计规范"), dims=64)
        norm = sum(value * value for value in vector) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_empty_tokens_yield_zero_vector(self) -> None:
        self.assertEqual(hash_vector([], dims=8), [0.0] * 8)

    def test_l2_normalize_handles_zero_vector(self) -> None:
        self.assertEqual(l2_normalize([0.0, 0.0]), [0.0, 0.0])

    def test_cosine_of_identical_vectors_is_one(self) -> None:
        vector = hash_vector(tokenize("轨距 1435"), dims=32)
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=6)

    def test_cosine_of_orthogonal_vectors_is_zero(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)


class CoverageTests(unittest.TestCase):
    """Query coverage drives the heuristic reranker."""

    def test_coverage_counts_query_terms_in_document(self) -> None:
        terms = query_terms("最小曲线半径")
        self.assertGreater(coverage(terms, "TB 10621 最小曲线半径一般值 7000 米"), 0.5)

    def test_coverage_of_unrelated_document_is_zero(self) -> None:
        self.assertEqual(coverage(query_terms("最小曲线半径"), "隧道微气压波限值"), 0.0)

    def test_query_terms_are_unique(self) -> None:
        terms = query_terms("曲线曲线")
        self.assertEqual(len(terms), len(set(terms)))


if __name__ == "__main__":
    unittest.main()
