"""Tests for the Chroma persistence layer."""
from __future__ import annotations

import unittest

from src.ingestion.mineru_parser import Chunk
from src.rag.embeddings import HashingEmbeddingFunction
from src.rag.store import ChromaStore, IndexMismatchError
from tests.support import TempDirMixin, make_settings


def _chunks() -> list[Chunk]:
    return [
        Chunk("铁路曲线半径一般值为7000米，困难值为5500米。", {"path": "2.1 线路"}),
        Chunk("隧道出口微气压波限值为20Pa。", {"path": "2.3 隧道"}),
        Chunk("CRTS III型轨道板平面度误差不超过0.3mm。", {"path": "2.2 轨道"}),
    ]


class ChromaStoreTests(TempDirMixin, unittest.TestCase):
    """Write, read, query and reset a real (temporary) Chroma collection."""

    def _store(self, **overrides: object) -> ChromaStore:
        settings = make_settings(chroma_persist_dir=self.temp_dir(), **overrides)
        return ChromaStore(settings)

    def test_add_and_count_chunks(self) -> None:
        store = self._store()
        self.assertEqual(store.add_chunks(_chunks(), "rail.md"), 3)
        self.assertEqual(store.count(), 3)

    def test_reingesting_a_source_replaces_its_chunks(self) -> None:
        store = self._store()
        store.add_chunks(_chunks(), "rail.md")
        store.add_chunks(_chunks()[:1], "rail.md")
        self.assertEqual(store.count(), 1)
        self.assertEqual(len(store.all_chunks()), 1)

    def test_chunk_ids_are_deterministic(self) -> None:
        first = self._store()
        first.add_chunks(_chunks(), "rail.md")
        ids_first = [chunk.chunk_id for chunk in first.all_chunks()]

        second_settings = make_settings(chroma_persist_dir=self.temp_dir())
        second = ChromaStore(second_settings)
        second.add_chunks(_chunks(), "rail.md")
        self.assertEqual(ids_first, [chunk.chunk_id for chunk in second.all_chunks()])

    def test_metadata_is_preserved(self) -> None:
        store = self._store()
        store.add_chunks(_chunks(), "rail.md")
        record = next(chunk for chunk in store.all_chunks() if "隧道" in chunk.content)
        self.assertEqual(record.metadata["path"], "2.3 隧道")
        self.assertEqual(record.metadata["source"], "rail.md")

    def test_vector_search_ranks_the_matching_chunk_first(self) -> None:
        store = self._store()
        store.add_chunks(_chunks(), "rail.md")
        hits = store.vector_search("微气压波限值", 3)
        self.assertTrue(hits)
        tunnel = next(chunk for chunk in store.all_chunks() if "微气压波" in chunk.content)
        self.assertEqual(hits[0].chunk_id, tunnel.chunk_id)

    def test_empty_collection_returns_no_hits(self) -> None:
        self.assertEqual(self._store().vector_search("任何问题", 3), [])

    def test_reset_clears_the_collection(self) -> None:
        store = self._store()
        store.add_chunks(_chunks(), "rail.md")
        store.reset()
        self.assertEqual(store.count(), 0)

    def test_backend_change_is_reported(self) -> None:
        settings = make_settings(chroma_persist_dir=self.temp_dir())
        ChromaStore(settings).add_chunks(_chunks(), "rail.md")
        with self.assertRaises(IndexMismatchError):
            ChromaStore(settings, embedding_function=HashingEmbeddingFunction(dims=64, quiet=True))

    def test_large_batch_is_split_into_upserts(self) -> None:
        store = self._store()
        chunks = [Chunk(f"第{index}条标准正文内容。", {"path": f"{index}"}) for index in range(300)]
        self.assertEqual(store.add_chunks(chunks, "bulk.md"), 300)
        self.assertEqual(store.count(), 300)


if __name__ == "__main__":
    unittest.main()
