"""ModelScope environment verification utilities."""
from __future__ import annotations

from typing import Any, Dict

from .logging_utils import setup_logging


def test_modelscope_env() -> bool:
    """Verify ModelScope installation with a simple NLP pipeline.

    Returns:
        bool: True if the pipeline runs successfully, False otherwise.
    """
    logger = setup_logging(name="rail-crag.modelscope")
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except Exception as exc:
        logger.exception("ModelScope import failed: %s", exc)
        return False

    try:
        word_segmentation = pipeline(
            Tasks.word_segmentation,
            "damo/nlp_structbert_word-segmentation_chinese-base",
        )
        text = "铁路标准知识库构建指南"
        result: Dict[str, Any] = word_segmentation(text)
        logger.info("ModelScope environment verification succeeded: %s", result)
        return True
    except Exception as exc:
        logger.exception("ModelScope pipeline failed: %s", exc)
        return False


if __name__ == "__main__":
    test_modelscope_env()
