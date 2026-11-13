"""Tests for retrieval metrics."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.rag.evaluation import compare_modes, evaluate, load_eval_cases


class _FakeRetriever:
    """Returns a fixed ranking per query."""

    def __init__(self) -> None:
        self._rankings = {
            "hit-first": ["a", "b"],
            "hit-second": ["x", "a", "y"],
            "miss": ["x", "y"],
        }

    def retrieve(self, query: str, *, top_k: int = 5, mode: str = "hybrid", rerank: object = None):
        """Return the configured ranking as result dictionaries."""
        return [
            {"id": chunk_id, "content": "命中关键词" if chunk_id == "a" else "无关", "path": ""}
            for chunk_id in self._rankings[query][:top_k]
        ]


class EvaluateTests(unittest.TestCase):
    """Metric definitions must match the documented formulas."""

    def test_metrics_for_mixed_rankings(self) -> None:
        cases = [
            {"id": "1", "query": "hit-first", "expected_keywords": ["命中"]},
            {"id": "2", "query": "hit-second", "expected_keywords": ["命中"]},
            {"id": "3", "query": "miss", "expected_keywords": ["命中"]},
        ]
        report = evaluate(_FakeRetriever(), cases, name="test", top_k=5)
        self.assertAlmostEqual(report.hit_rate, 2 / 3)
        self.assertAlmostEqual(report.mrr, (1.0 + 0.5) / 3)
        expected_ndcg = (1.0 + 1.0 / math.log2(3)) / 3
        self.assertAlmostEqual(report.ndcg, expected_ndcg)
        self.assertEqual(report.failures, ["3"])

    def test_empty_case_list_does_not_divide_by_zero(self) -> None:
        report = evaluate(_FakeRetriever(), [], name="empty")
        self.assertEqual(report.queries, 0)
        self.assertEqual(report.hit_rate, 0.0)

    def test_report_serialisation_rounds_values(self) -> None:
        cases = [{"id": "1", "query": "hit-first", "expected_keywords": ["命中"]}]
        payload = evaluate(_FakeRetriever(), cases).as_dict()
        self.assertEqual(payload["hit_rate"], 1.0)
        self.assertEqual(payload["mrr"], 1.0)
        self.assertEqual(json.loads(json.dumps(payload))["top_k"], 5)


class CompareModesTests(unittest.TestCase):
    """The ablation always reports the four configurations."""

    def test_four_reports(self) -> None:
        cases = [{"id": "1", "query": "hit-first", "expected_keywords": ["命中"]}]
        reports = compare_modes(_FakeRetriever(), cases, top_k=3)
        self.assertEqual(
            [report.name for report in reports],
            ["BM25", "Vector", "Hybrid (RRF)", "Hybrid + Rerank"],
        )


class LoadCasesTests(unittest.TestCase):
    """Bundled cases are valid; malformed cases fail loudly."""

    def test_bundled_cases_are_valid(self) -> None:
        cases = load_eval_cases()
        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

    def test_missing_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps([{"id": "x", "query": "q"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_eval_cases(path)


if __name__ == "__main__":
    unittest.main()
