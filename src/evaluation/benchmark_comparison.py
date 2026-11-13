"""Benchmark Standard RAG against the CRAG pipeline.

Standard RAG = dense retrieval + one generation call.
CRAG = hybrid retrieval + evaluator + refinement/web-search + generation.

Run with ``python -m src.evaluation.benchmark_comparison`` after ingesting the
sample corpus. With no API key the generator returns the retrieved context, so
the comparison still exercises retrieval and routing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..components.generator import AnswerGenerator
from ..config import Settings, load_settings
from ..graph.builder import build_crag_graph
from ..graph.nodes import CRAGNodes
from ..rag.retriever import HybridRetriever
from ..rag.store import get_store
from ..utils.logging_utils import configure_stdio, setup_logging


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark row.

    Args:
        question: Input question.
        method: Method name.
        confidence: CRAG decision (``n/a`` for standard RAG).
        context_source: Where the context came from.
        answer: Generated answer.
        latency: Wall-clock seconds.
    """

    question: str
    method: str
    confidence: str
    context_source: str
    answer: str
    latency: float


class BenchmarkRunner:
    """Run Standard RAG and CRAG over the same questions.

    Args:
        settings: Optional settings override.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._logger = setup_logging(name="rail-crag.benchmark")
        self._settings = settings or load_settings(require_keys=False)
        store = get_store(self._settings)
        self._retriever = HybridRetriever(
            store,
            rrf_k=self._settings.rrf_k,
            candidate_pool=self._settings.candidate_pool,
            default_mode=self._settings.retrieval_mode,
            default_rerank=self._settings.rerank,
            logger=self._logger,
        )
        self._generator = AnswerGenerator(self._settings, logger=self._logger)
        self._crag_app = build_crag_graph(
            CRAGNodes(self._settings, logger=self._logger)
        )

    def run_standard_rag(self, query: str, k: int = 3) -> BenchmarkResult:
        """Dense retrieval followed by one generation call.

        Args:
            query: Input question.
            k: Number of chunks.

        Returns:
            BenchmarkResult: Standard RAG row.
        """
        started = time.perf_counter()
        docs = self._retriever.retrieve(query, top_k=k, mode="vector", rerank=False)
        context = "\n\n".join(doc.get("content", "") for doc in docs)
        answer = self._generator.generate(query, context)
        return BenchmarkResult(
            question=query,
            method="Standard RAG",
            confidence="n/a",
            context_source="dense top-k",
            answer=answer,
            latency=time.perf_counter() - started,
        )

    def run_crag(self, query: str) -> BenchmarkResult:
        """Run the full CRAG graph.

        Args:
            query: Input question.

        Returns:
            BenchmarkResult: CRAG row.
        """
        started = time.perf_counter()
        result = self._crag_app.invoke({"question": query})
        confidence = str(result.get("confidence", "unknown"))
        source = {
            "correct": "hybrid + refinement",
            "ambiguous": "hybrid + web search",
            "incorrect": "web search",
        }.get(confidence, "hybrid")
        return BenchmarkResult(
            question=query,
            method="CRAG",
            confidence=confidence,
            context_source=source,
            answer=result.get("final_answer", ""),
            latency=time.perf_counter() - started,
        )

    def compare(self, questions: List[str]) -> pd.DataFrame:
        """Benchmark every question with both methods.

        Args:
            questions: Input questions.

        Returns:
            pd.DataFrame: One row per method and question.
        """
        results: List[BenchmarkResult] = []
        for question in questions:
            self._logger.info("Benchmarking: %s", question)
            results.append(self.run_standard_rag(question))
            results.append(self.run_crag(question))
        return pd.DataFrame([row.__dict__ for row in results])


def main() -> None:
    """CLI entry for the benchmark comparison."""
    configure_stdio()
    runner = BenchmarkRunner()
    questions = [
        "高速铁路时速350公里路段的最小曲线半径是多少？",
        "CRTS III型轨道板的平面度误差限值是多少？",
        "CTCS-3级列控系统叠加了什么通信方式？",
        "列车在火星上的建设标准是什么？",
    ]
    frame = runner.compare(questions)
    columns = ["question", "method", "confidence", "context_source", "answer", "latency"]
    preview = frame[columns].copy()
    preview["answer"] = preview["answer"].astype(str).str.slice(0, 60) + "…"
    preview["latency"] = preview["latency"].round(3)
    print(preview.to_markdown(index=False))


if __name__ == "__main__":
    main()
