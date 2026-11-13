"""Tests for the ingestion pipeline."""
from __future__ import annotations

import unittest
from pathlib import Path

from src.ingestion.pipeline import Ingestor, find_source_files, read_text_file
from src.rag.store import ChromaStore
from tests.support import TempDirMixin, make_settings


class ReadTextTests(TempDirMixin, unittest.TestCase):
    """Chinese corpora are not always UTF-8."""

    def test_reads_utf8(self) -> None:
        path = Path(self.temp_dir()) / "a.md"
        path.write_text("# 标准\n正文", encoding="utf-8")
        self.assertIn("标准", read_text_file(str(path)))

    def test_reads_gb18030(self) -> None:
        path = Path(self.temp_dir()) / "b.md"
        path.write_bytes("# 标准\n正文".encode("gb18030"))
        self.assertIn("标准", read_text_file(str(path)))

    def test_strips_utf8_bom(self) -> None:
        path = Path(self.temp_dir()) / "c.md"
        path.write_bytes(b"\xef\xbb\xbf" + "# 标准\n".encode("utf-8"))
        self.assertTrue(read_text_file(str(path)).startswith("# 标准"))


class IngestorTests(TempDirMixin, unittest.TestCase):
    """File and text ingestion share one code path."""

    def setUp(self) -> None:
        self.settings = make_settings(chroma_persist_dir=self.temp_dir())
        self.store = ChromaStore(self.settings)
        self.ingestor = Ingestor(self.store)

    def test_ingest_text(self) -> None:
        outcome = self.ingestor.ingest_text(
            "# 1 总则\n标准正文内容足够长，能够被正常索引。", "inline.md"
        )
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.chunks, 1)
        self.assertEqual(self.store.count(), 1)

    def test_ingest_missing_file(self) -> None:
        outcome = self.ingestor.ingest_file(str(Path(self.temp_dir()) / "missing.md"))
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error, "file not found")

    def test_unsupported_suffix(self) -> None:
        path = Path(self.temp_dir()) / "notes.txt"
        path.write_text("text", encoding="utf-8")
        outcome = self.ingestor.ingest_file(str(path))
        self.assertFalse(outcome.ok)
        self.assertIn("unsupported", outcome.error or "")

    def test_empty_document_is_reported(self) -> None:
        outcome = self.ingestor.ingest_text("", "empty.md")
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.chunks, 0)

    def test_ingest_many_continues_after_failure(self) -> None:
        directory = self.temp_dir()
        good = Path(directory) / "good.md"
        good.write_text("# 1 总则\n标准正文内容足够长，能够被正常索引。", encoding="utf-8")
        outcomes = self.ingestor.ingest_many(
            [str(good), str(Path(directory) / "missing.md")]
        )
        self.assertEqual([outcome.ok for outcome in outcomes], [True, False])


class FindSourcesTests(TempDirMixin, unittest.TestCase):
    """Glob expansion is recursive and filtered."""

    def test_finds_markdown_recursively(self) -> None:
        root = Path(self.temp_dir())
        (root / "nested").mkdir()
        (root / "a.md").write_text("# A", encoding="utf-8")
        (root / "nested" / "b.markdown").write_text("# B", encoding="utf-8")
        (root / "c.txt").write_text("noise", encoding="utf-8")
        found = find_source_files(str(root / "**" / "*"))
        self.assertEqual(len(found), 2)


if __name__ == "__main__":
    unittest.main()
