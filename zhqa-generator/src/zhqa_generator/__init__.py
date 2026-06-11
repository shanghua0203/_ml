"""
zhqa_generator
~~~~~~~~~~~~~~
將中文純文字切割並透過 Ollama LLM 自動生成 QA 問答對。

快速開始：
    >>> from zhqa_generator import chunk_text, generate_qa_pairs
    >>> chunks = chunk_text(open("article.txt").read())
    >>> pairs = generate_qa_pairs(chunks, model="gemma3:4b")
"""

from .core import (
    DEFAULT_SYSTEM_PROMPT,
    chunk_text,
    generate_qa_for_chunk,
    generate_qa_pairs,
    split_sentences,
)

__version__ = "0.1.0"
__all__ = [
    "chunk_text",
    "split_sentences",
    "generate_qa_for_chunk",
    "generate_qa_pairs",
    "DEFAULT_SYSTEM_PROMPT",
]
