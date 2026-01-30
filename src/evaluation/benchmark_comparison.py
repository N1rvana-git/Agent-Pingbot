"""Benchmark CRAG vs Standard RAG."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..components.generator import AnswerGenerator
from ..components.vector_store import VectorStore
from ..config import load_settings
from ..graph.builder import build_crag_graph
from ..utils.logging_utils import setup_logging


@dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark result record.

    Args:
        question: Input question.
        method: Method name.
        action_log: Action trace.
        context_source: Context source summary.
        answer: Generated answer.
        latency: Execution time in seconds.
    """

    question: str
    method: str
    action_log: str
    context_source: str
    answer: str
    latency: float


class BenchmarkRunner:
    """Benchmark runner for CRAG and Standard RAG."""

    def __init__(self) -> None:
        """Initialize runner with dependencies."""
        self._logger = setup_logging(name="rail-crag.benchmark")
        self._settings = load_settings(require_keys=False)
        self._vector_store = VectorStore(self._settings, logger=self._logger)
        self._generator = AnswerGenerator(self._settings, logger=self._logger)
        self._crag_app = build_crag_graph()

    def run_standard_rag(self, query: str, k: int = 3) -> BenchmarkResult:
        """Run Standard RAG: retrieve then generate.

        Args:
            query: Input question.
            k: Top-k documents.

        Returns:
            BenchmarkResult: Result for standard RAG.
        """
        start = time.time()
        docs = self._vector_store.search(query, k)
        context = "\n\n".join([doc.content for doc in docs])
        answer = self._generator.generate(query, context)
        return BenchmarkResult(
            question=query,
            method="Standard RAG",
            action_log="N/A",
            context_source="Internal DB Only",
            answer=answer,
            latency=time.time() - start,
        )

    def run_crag(self, query: str) -> BenchmarkResult:
        """Run full CRAG pipeline.

        Args:
            query: Input question.

        Returns:
            BenchmarkResult: Result for CRAG.
        """
        start = time.time()
        final_answer = ""
        action_log: List[str] = []
        context_source = "Internal"

        inputs = {"question": query}
        for output in self._crag_app.stream(inputs):
            for _, state_update in output.items():
                if isinstance(state_update, dict) and "confidence" in state_update:
                    action = state_update.get("confidence", "")
                    if action:
                        action_log.append(f"Action: {action.upper()}")
                    if action == "incorrect":
                        context_source = "Web Search"
                    elif action == "ambiguous":
                        context_source = "Hybrid"
                if isinstance(state_update, dict) and "final_answer" in state_update:
                    final_answer = state_update.get("final_answer", "")

        return BenchmarkResult(
            question=query,
            method="CRAG",
            action_log=" -> ".join(action_log),
            context_source=context_source,
            answer=final_answer,
            latency=time.time() - start,
        )

    def compare(self, questions: List[str]) -> pd.DataFrame:
        """Run benchmark for a list of questions.

        Args:
            questions: List of questions.

        Returns:
            pd.DataFrame: Results dataframe.
        """
        results: List[BenchmarkResult] = []
        for q in questions:
            self._logger.info("Benchmarking: %s", q)
            results.append(self.run_standard_rag(q))
            results.append(self.run_crag(q))

        df = pd.DataFrame([r.__dict__ for r in results])
        return df


def main() -> None:
    """CLI entry for benchmark comparison."""
    runner = BenchmarkRunner()
    questions = [
        "What is the standard gauge width for railways?",
        "What are the construction specifications for the Hyperloop on Mars?",
        "Tell me about the track requirements defined in section 1.1.",
    ]
    df = runner.compare(questions)
    cols = ["question", "method", "action_log", "context_source", "answer", "latency"]
    print(df[cols].to_markdown(index=False))


if __name__ == "__main__":
    main()
