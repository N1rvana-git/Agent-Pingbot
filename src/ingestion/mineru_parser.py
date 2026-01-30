"""MinerU markdown parser for hierarchical splitting."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Chunk:
    """A parsed content chunk.

    Args:
        content: The text content of the chunk.
        metadata: Metadata including hierarchical path.
    """

    content: str
    metadata: Dict[str, str]


class MarkdownHierarchySplitter:
    """Split MinerU markdown into hierarchical chunks.

    The parser uses heading markers (#, ##, ###) to build a path stack.
    """

    header_pattern = re.compile(r"^(#+)\s+(.*)$")

    def parse(self, markdown_text: str) -> List[Chunk]:
        """Parse markdown content into chunks.

        Args:
            markdown_text: The markdown text from MinerU.

        Returns:
            List[Chunk]: A list of chunks with metadata paths.
        """
        lines = markdown_text.splitlines()
        header_stack: List[Tuple[int, str]] = []
        current_content: List[str] = []
        chunks: List[Chunk] = []

        for line in lines:
            match = self.header_pattern.match(line)
            if match:
                if current_content:
                    path = " > ".join([h[1] for h in header_stack])
                    chunks.append(Chunk(content="\n".join(current_content).strip(), metadata={"path": path}))
                    current_content = []

                level = len(match.group(1))
                title = match.group(2).strip()

                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, title))
            else:
                if line.strip():
                    current_content.append(line)

        if current_content:
            path = " > ".join([h[1] for h in header_stack])
            chunks.append(Chunk(content="\n".join(current_content).strip(), metadata={"path": path}))

        return chunks
