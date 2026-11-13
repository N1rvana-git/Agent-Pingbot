"""Markdown parsing for MinerU output and curated standards.

MinerU's Markdown is rich but noisy: base64-embedded figures, HTML fragments,
escaped punctuation, bold headings and very long appendix sections. Feeding
that straight into a vector store produces a few giant chunks (one appendix
chunk can contain hundreds of kilobytes of base64 and drown every query), so
parsing and chunking are treated as part of the retrieval quality problem:

1. clean each section (drop figure definitions, link targets, HTML, emphasis);
2. keep the heading hierarchy as ``path`` metadata for citation and ranking;
3. pack paragraphs up to ``max_chars`` and split on paragraph boundaries so a
   clause is never cut mid-sentence;
4. skip sections that carry no retrievable text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_IMAGE_INLINE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\[[^\]]*\]")
_MD_IMAGE_DEF_RE = re.compile(r"(?m)^\s*\[[^\]]+\]:\s*<data:[^\n]*>\s*$")
_HTML_IMG_RE = re.compile(r"<img[^>]*>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?(?:br|p|div|span|table|tr|td|th|sup|sub)\b[^>]*>", re.IGNORECASE)
_MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>~|])")
_TABLE_SEPARATOR_RE = re.compile(r"(?m)^\s*\|?[\s:|-]{3,}\|?\s*$")
_EMPHASIS_RE = re.compile(r"\*\*|__")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SPACES_RE = re.compile(r"[ \t]{2,}")
_PUNCT_ONLY_RE = re.compile(r"^[\s\-=_*#.。·]+$")


@dataclass(frozen=True)
class Chunk:
    """A parsed content chunk.

    Args:
        content: Cleaned body text (without the heading).
        metadata: Flat metadata including the heading ``path``.
    """

    content: str
    metadata: Dict[str, Any]


def clean_markdown(text: str) -> str:
    """Remove Markdown/HTML noise that hurts retrieval.

    Args:
        text: Raw Markdown section body.

    Returns:
        str: Cleaned text.
    """
    cleaned = _MD_IMAGE_DEF_RE.sub("", text or "")
    cleaned = _MD_IMAGE_INLINE_RE.sub("", cleaned)
    cleaned = _MD_IMAGE_REF_RE.sub("", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _HTML_IMG_RE.sub("", cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = _TABLE_SEPARATOR_RE.sub("", cleaned)
    cleaned = _MD_ESCAPE_RE.sub(r"\1", cleaned)
    cleaned = _EMPHASIS_RE.sub("", cleaned)
    cleaned = "\n".join(
        line for line in cleaned.splitlines() if line.strip() not in {"---", "***", "___"}
    )
    cleaned = _SPACES_RE.sub(" ", cleaned)
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_heading(title: str) -> str:
    """Normalise a heading for use in the section path.

    Args:
        title: Raw heading text.

    Returns:
        str: Cleaned heading (empty for decorative separators).
    """
    cleaned = _MD_ESCAPE_RE.sub(r"\1", title or "")
    cleaned = _EMPHASIS_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" #")
    if _PUNCT_ONLY_RE.match(cleaned):
        return ""
    return cleaned


def _hard_split(paragraph: str, max_chars: int) -> List[str]:
    """Split an oversized paragraph on sentence boundaries, then hard-cut.

    Args:
        paragraph: Paragraph text.
        max_chars: Maximum chunk size.

    Returns:
        List[str]: Paragraph parts.
    """
    sentences = re.split(r"(?<=[。！？!?；;])", paragraph)
    parts: List[str] = []
    buffer = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(buffer) + len(sentence) <= max_chars:
            buffer += sentence
            continue
        if buffer:
            parts.append(buffer)
        while len(sentence) > max_chars:
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        buffer = sentence
    if buffer:
        parts.append(buffer)
    return parts


def pack_paragraphs(body: str, max_chars: int) -> List[str]:
    """Pack paragraphs into chunks without cutting a paragraph mid-way.

    Args:
        body: Cleaned section body.
        max_chars: Maximum chunk size.

    Returns:
        List[str]: Chunk bodies.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    body = body.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    chunks: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_hard_split(paragraph, max_chars))
            continue
        if not buffer:
            buffer = paragraph
        elif len(buffer) + len(paragraph) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{paragraph}"
        else:
            chunks.append(buffer)
            buffer = paragraph
    if buffer:
        chunks.append(buffer)
    return chunks


class MarkdownHierarchySplitter:
    """Split Markdown into hierarchy-aware, size-bounded chunks.

    Args:
        max_chars: Maximum chunk length; longer sections are packed into parts.
        min_chars: Sections shorter than this carry no retrievable content.
    """

    def __init__(self, max_chars: int = 800, min_chars: int = 12) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if min_chars < 0:
            raise ValueError("min_chars must be >= 0")
        self.max_chars = max_chars
        self.min_chars = min_chars

    def parse(self, markdown_text: str) -> List[Chunk]:
        """Parse Markdown into chunks.

        Args:
            markdown_text: Markdown source.

        Returns:
            List[Chunk]: Chunks with ``path``, ``section``, ``level`` and part metadata.
        """
        lines = (markdown_text or "").splitlines()
        heading_stack: List[Tuple[int, str]] = []
        buffer: List[str] = []
        chunks: List[Chunk] = []

        for line in lines:
            match = _HEADING_RE.match(line)
            if match:
                self._flush(buffer, heading_stack, chunks)
                buffer = []
                level = len(match.group(1))
                title = clean_heading(match.group(2))
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                if title:
                    heading_stack.append((level, title))
                continue
            buffer.append(line)

        self._flush(buffer, heading_stack, chunks)
        return chunks

    def _flush(
        self,
        buffer: List[str],
        heading_stack: List[Tuple[int, str]],
        chunks: List[Chunk],
    ) -> None:
        """Clean, split and emit the current section.

        Args:
            buffer: Raw section lines.
            heading_stack: Current heading chain.
            chunks: Output accumulator.
        """
        if not buffer:
            return
        body = clean_markdown("\n".join(buffer))
        if len(body) < self.min_chars:
            return

        path = " > ".join(title for _level, title in heading_stack)
        section = heading_stack[-1][1] if heading_stack else ""
        level = heading_stack[-1][0] if heading_stack else 0
        parts = pack_paragraphs(body, self.max_chars)
        total = len(parts)
        for index, part in enumerate(parts, start=1):
            if len(part) < self.min_chars:
                continue
            chunks.append(
                Chunk(
                    content=part,
                    metadata={
                        "path": path,
                        "section": section,
                        "level": level,
                        "part": index,
                        "parts": total,
                    },
                )
            )
