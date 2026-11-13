"""Tests for hybrid retrieval and reranking."""
from __future__ import annotations

import unittest

from src.rag.reranker import HeuristicReranker
from src.rag.retriever import HybridRetriever, bm25_rank, build_bm25_index
from src.rag.store import ChunkRecord
from tests.support import FakeStore


def _corpus() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id="curve",
            content="高速铁路设计规范规定最小曲线半径一般值为7000米，困难值为5500米。",
            metadata={"path": "2.1 线路空间几何与限界约束机制", "source": "rail.md"},
        ),
        ChunkRecord(
            chunk_id="tunnel",
            content="隧道出口的微气压波幅值应满足噪声控制要求，通常不超过20Pa。",
            metadata={"path": "2.3 桥梁与隧道工程的结构健康逻辑", "source": "rail.md"},
        ),
        ChunkRecord(
            chunk_id="slab",
            content="CRTS III型轨道板平面度误差不超过0.3mm，是预制场质检的核心指标。",
            metadata={"path": "2.2 轨道系统与精密工程控制", "source": "rail.md"},
        ),
    ]


def _retriever(store: FakeStore, **overrides: object) -> HybridRetriever:
    defaults = {
        "rrf_k": 60,
        "candidate_pool": 10,
        "default_mode": "hybrid",
        "default_rerank": False,
    }
    defaults.update(overrides)
    return HybridRetriever(store, **defaults)  # type: ignore[arg-type]


class Bm25Tests(unittest.TestCase):
    """BM25 ranks exact standard vocabulary first."""

    def test_exact_term_wins(self) -> None:
        corpus = _corpus()
        index = build_bm25_index(corpus)
        ranked = bm25_rank("微气压波", corpus, index)
        self.assertEqual(ranked[0][1].chunk_id, "tunnel")

    def test_no_match_returns_empty(self) -> None:
        corpus = _corpus()
        index = build_bm25_index(corpus)
        self.assertEqual(bm25_rank("火星基地", corpus, index), [])

    def test_path_tokens_are_indexed(self) -> None:
        corpus = _corpus()
        index = build_bm25_index(corpus)
        ranked = bm25_rank("结构健康逻辑", corpus, index)
        self.assertEqual(ranked[0][1].chunk_id, "tunnel")


class HybridRetrieverTests(unittest.TestCase):
    """Mode selection, fusion and failure behaviour."""

    def test_bm25_mode_uses_only_the_lexical_leg(self) -> None:
        retriever = _retriever(FakeStore(_corpus(), vector_order=["tunnel"]))
        results = retriever.retrieve("最小曲线半径", top_k=2, mode="bm25")
        self.assertEqual(results[0]["id"], "curve")
        self.assertIsNone(results[0]["vector_rank"])

    def test_vector_mode_uses_only_the_dense_leg(self) -> None:
        retriever = _retriever(FakeStore(_corpus(), vector_order=["slab", "curve"]))
        results = retriever.retrieve("任何查询", top_k=2, mode="vector")
        self.assertEqual([item["id"] for item in results], ["slab", "curve"])

    def test_hybrid_merges_both_legs(self) -> None:
        retriever = _retriever(FakeStore(_corpus(), vector_order=["tunnel", "curve"]))
        results = retriever.retrieve("最小曲线半径", top_k=3, mode="hybrid")
        ids = [item["id"] for item in results]
        self.assertIn("curve", ids)
        self.assertIn("tunnel", ids)

    def test_hybrid_rerank_returns_scores(self) -> None:
        retriever = _retriever(
            FakeStore(_corpus(), vector_order=["slab", "curve"]), default_rerank=True
        )
        results = retriever.retrieve("最小曲线半径", top_k=3, mode="hybrid")
        self.assertEqual(results[0]["reranker"], "heuristic")
        self.assertIn("rerank_score", results[0])

    def test_dense_failure_degrades_to_bm25(self) -> None:
        retriever = _retriever(FakeStore(_corpus(), vector_error="gateway down"))
        results = retriever.retrieve("微气压波", top_k=2, mode="hybrid")
        self.assertEqual(results[0]["id"], "tunnel")
        self.assertEqual(results[0]["vector_error"], "gateway down")

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _retriever(FakeStore(_corpus())).retrieve("查询", mode="fuzzy")

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(_retriever(FakeStore(_corpus())).retrieve("   "), [])

    def test_empty_corpus_returns_nothing(self) -> None:
        self.assertEqual(_retriever(FakeStore([])).retrieve("查询"), [])

    def test_corpus_is_cached_between_queries(self) -> None:
        store = FakeStore(_corpus(), vector_order=["curve"])
        retriever = _retriever(store)
        store.count = lambda: 3  # type: ignore[method-assign]
        retriever.retrieve("微气压波")
        first = retriever._chunks
        retriever.retrieve("微气压波")
        self.assertIs(retriever._chunks, first)

    def test_trace_returns_four_configurations(self) -> None:
        trace = _retriever(FakeStore(_corpus(), vector_order=["curve"])).retrieve_with_trace(
            "最小曲线半径", top_k=2
        )
        self.assertEqual(
            set(trace["runs"]), {"hybrid_rerank", "hybrid", "bm25", "vector"}
        )
        self.assertIn("latency_ms", trace["runs"]["hybrid"])
        self.assertEqual(trace["corpus"]["collection"], "fake")


class RerankerTests(unittest.TestCase):
    """The reranker prefers the section that names the queried standard."""

    def test_standard_id_bonus_changes_the_order(self) -> None:
        candidates = [
            {
                "id": "generic",
                "content": "最小曲线半径的一般规定。",
                "path": "1 总则",
                "score": 0.9,
            },
            {
                "id": "specific",
                "content": "最小曲线半径的一般规定。",
                "path": "A.2 勘察与设计标准 TB 10621",
                "score": 0.9,
            },
        ]
        ranked = HeuristicReranker().rerank("TB 10621 最小曲线半径", candidates, top_k=2)
        self.assertEqual(ranked[0]["id"], "specific")
        self.assertGreater(ranked[0]["rerank_score"], ranked[1]["rerank_score"])

    def test_empty_candidates(self) -> None:
        self.assertEqual(HeuristicReranker().rerank("查询", [], top_k=3), [])


if __name__ == "__main__":
    unittest.main()
