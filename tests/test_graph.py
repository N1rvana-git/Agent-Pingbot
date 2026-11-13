"""End-to-end test of the offline CRAG graph."""
from __future__ import annotations

import unittest

from src.graph.builder import build_crag_graph
from src.graph.nodes import CRAGNodes
from src.ingestion.pipeline import Ingestor
from src.rag.store import ChromaStore
from tests.support import TempDirMixin, make_settings

_CORPUS = """# 高速铁路设计规范

## 2.1 线路空间几何

高速铁路设计规范规定，时速350公里路段的最小曲线半径一般值为7000米，困难值为5500米。

## 2.3 隧道工程

隧道出口的微气压波幅值通常不超过20Pa，以控制对周边居民区的噪声影响。
"""


class GraphTests(TempDirMixin, unittest.TestCase):
    """The graph must answer offline, from a real temporary index."""

    def setUp(self) -> None:
        self.settings = make_settings(chroma_persist_dir=self.temp_dir())
        store = ChromaStore(self.settings)
        outcome = Ingestor(store).ingest_text(_CORPUS, "rail.md")
        self.assertEqual(outcome.chunks, 2)

    def test_graph_runs_end_to_end(self) -> None:
        graph = build_crag_graph(CRAGNodes(self.settings))
        result = graph.invoke({"question": "最小曲线半径一般值是多少？"})

        self.assertTrue(result.get("final_answer"))
        self.assertIn("7000", result.get("final_answer", ""))
        self.assertIn(result.get("confidence"), {"correct", "ambiguous", "incorrect"})
        self.assertEqual(result["retrieval_trace"]["mode"], "hybrid")
        self.assertGreaterEqual(len(result.get("retrieved_documents", [])), 1)

    def test_retrieval_trace_contains_provenance(self) -> None:
        graph = build_crag_graph(CRAGNodes(self.settings))
        result = graph.invoke({"question": "隧道微气压波限值"})
        hit = result["retrieval_trace"]["hits"][0]
        self.assertEqual(hit["path"], "高速铁路设计规范 > 2.3 隧道工程")
        self.assertEqual(hit["source"], "rail.md")
        self.assertIsNotNone(hit["score"])

    def test_bm25_only_graph_still_answers(self) -> None:
        settings = make_settings(chroma_persist_dir=self.settings.chroma_persist_dir)
        import dataclasses

        settings = dataclasses.replace(settings, retrieval_mode="bm25", rerank=False)
        graph = build_crag_graph(CRAGNodes(settings))
        result = graph.invoke({"question": "微气压波"})
        self.assertIn("20Pa", result.get("final_answer", ""))
        self.assertEqual(result["retrieval_trace"]["mode"], "bm25")


if __name__ == "__main__":
    unittest.main()
