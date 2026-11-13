"""Tests for Markdown cleaning and hierarchy-aware chunking."""
from __future__ import annotations

import unittest

from src.ingestion.mineru_parser import (
    MarkdownHierarchySplitter,
    clean_heading,
    clean_markdown,
    pack_paragraphs,
)


class CleanMarkdownTests(unittest.TestCase):
    """Noise removal must keep prose and drop figures."""

    def test_drops_base64_figure_definitions(self) -> None:
        text = "正文\n\n[image1]: <data:image/png;base64,AAAA>\n\n结论"
        cleaned = clean_markdown(text)
        self.assertIn("正文", cleaned)
        self.assertIn("结论", cleaned)
        self.assertNotIn("base64", cleaned)

    def test_drops_html_and_inline_images(self) -> None:
        cleaned = clean_markdown("<p>标准</p> ![图](a.png) <img src='x'>")
        self.assertEqual(cleaned, "标准")

    def test_unescapes_markdown_punctuation(self) -> None:
        self.assertIn("2.1", clean_markdown(r"\*\*2\.1 线路\*\*"))

    def test_keeps_table_rows_but_drops_separators(self) -> None:
        table = "| 标准 | 参数 |\n| :---- | :---- |\n| GB 146.1 | 限界 |"
        cleaned = clean_markdown(table)
        self.assertIn("GB 146.1", cleaned)
        self.assertNotIn(":----", cleaned)

    def test_clean_heading_strips_decoration(self) -> None:
        self.assertEqual(clean_heading("**2\\. 基础设施**"), "2. 基础设施")

    def test_decorative_heading_becomes_empty(self) -> None:
        self.assertEqual(clean_heading("---"), "")


class PackParagraphTests(unittest.TestCase):
    """Packing must respect the size budget."""

    def test_short_body_is_a_single_chunk(self) -> None:
        self.assertEqual(pack_paragraphs("短文本", 100), ["短文本"])

    def test_paragraphs_are_packed_up_to_the_limit(self) -> None:
        body = "\n\n".join(["甲" * 60, "乙" * 60, "丙" * 60])
        parts = pack_paragraphs(body, 150)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(len(part) <= 150 for part in parts))

    def test_oversized_paragraph_is_split_on_sentences(self) -> None:
        body = "。".join(["这是一个很长的句子" * 5] * 6) + "。"
        parts = pack_paragraphs(body, 120)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 120 for part in parts))


class SplitterTests(unittest.TestCase):
    """Hierarchy, metadata and size bounds."""

    def test_headings_produce_hierarchical_paths(self) -> None:
        markdown = "# 1 总则\n正文A内容足够长\n## 1.1 范围\n正文B内容足够长"
        chunks = MarkdownHierarchySplitter(min_chars=1).parse(markdown)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["path"], "1 总则")
        self.assertEqual(chunks[1].metadata["path"], "1 总则 > 1.1 范围")

    def test_level_pops_back_to_parent(self) -> None:
        markdown = "# 1\nA 内容足够长\n## 1.1\nB 内容足够长\n# 2\nC 内容足够长"
        chunks = MarkdownHierarchySplitter(min_chars=1).parse(markdown)
        self.assertEqual(chunks[2].metadata["path"], "2")

    def test_long_section_is_split_into_numbered_parts(self) -> None:
        markdown = "# 1\n" + "\n\n".join(["段落" * 80] * 3)
        chunks = MarkdownHierarchySplitter(max_chars=300, min_chars=10).parse(markdown)
        self.assertEqual(len(chunks), 3)
        self.assertEqual([chunk.metadata["part"] for chunk in chunks], [1, 2, 3])
        self.assertTrue(all(chunk.metadata["parts"] == 3 for chunk in chunks))

    def test_short_sections_are_skipped(self) -> None:
        chunks = MarkdownHierarchySplitter(min_chars=50).parse("# 1\n太短")
        self.assertEqual(chunks, [])

    def test_figure_only_section_is_dropped(self) -> None:
        markdown = "# 1\n[image1]: <data:image/png;base64,AAAA>"
        self.assertEqual(MarkdownHierarchySplitter().parse(markdown), [])

    def test_empty_document(self) -> None:
        self.assertEqual(MarkdownHierarchySplitter().parse(""), [])


if __name__ == "__main__":
    unittest.main()
