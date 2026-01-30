"""Tests for MarkdownHierarchySplitter."""
from __future__ import annotations

from src.ingestion.mineru_parser import MarkdownHierarchySplitter


def test_markdown_hierarchy_splitter_basic() -> None:
    """Ensure headings produce hierarchical paths."""
    md = "# 1 总则\n正文A\n## 1.1 范围\n正文B"
    splitter = MarkdownHierarchySplitter()
    chunks = splitter.parse(md)
    assert len(chunks) == 2
    assert chunks[0].metadata["path"] == "1 总则"
    assert chunks[1].metadata["path"] == "1 总则 > 1.1 范围"
