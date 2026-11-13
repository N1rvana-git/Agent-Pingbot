"""Tests for the CRAG evaluator and refiner."""
from __future__ import annotations

import unittest

from src.components.evaluator import RetrievalEvaluator, determine_crag_action
from src.components.refiner import KnowledgeRefiner
from tests.support import make_settings


class EvaluatorTests(unittest.TestCase):
    """Offline scoring must work for Chinese text."""

    def _evaluator(self) -> RetrievalEvaluator:
        return RetrievalEvaluator(make_settings())

    def test_relevant_document_scores_higher(self) -> None:
        results = self._evaluator().score_documents(
            "最小曲线半径是多少",
            ["最小曲线半径一般值为7000米。", "隧道微气压波限值20Pa。"],
        )
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(results[0].rationale, "query_coverage")

    def test_empty_documents(self) -> None:
        self.assertEqual(self._evaluator().score_documents("问题", []), [])

    def test_scores_stay_in_range(self) -> None:
        results = self._evaluator().score_documents("轨距 1435", ["轨距 1435", "无关内容"])
        for result in results:
            self.assertGreaterEqual(result.score, -1.0)
            self.assertLessEqual(result.score, 1.0)


class ActionTests(unittest.TestCase):
    """The CRAG decision thresholds."""

    def test_no_scores_is_ambiguous(self) -> None:
        self.assertEqual(determine_crag_action([]), "ambiguous")

    def test_high_score_is_correct(self) -> None:
        self.assertEqual(determine_crag_action([0.9, -0.2]), "correct")

    def test_all_low_is_incorrect(self) -> None:
        self.assertEqual(determine_crag_action([-0.8, -0.9]), "incorrect")

    def test_mixed_is_ambiguous(self) -> None:
        self.assertEqual(determine_crag_action([0.1, -0.2]), "ambiguous")


class RefinerTests(unittest.TestCase):
    """Refinement keeps query-relevant strips only."""

    def test_irrelevant_sentences_are_dropped(self) -> None:
        refiner = KnowledgeRefiner(RetrievalEvaluator(make_settings()))
        refined = refiner.refine(
            "最小曲线半径",
            "隧道出口微气压波限值为20Pa。最小曲线半径一般值为7000米。接触网跨距需要校核。",
        )
        self.assertIn("7000", refined)
        self.assertNotIn("微气压波", refined)

    def test_empty_document(self) -> None:
        refiner = KnowledgeRefiner(RetrievalEvaluator(make_settings()))
        self.assertEqual(refiner.refine("问题", ""), "")


if __name__ == "__main__":
    unittest.main()
