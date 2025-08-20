# deepsearch/bm25_index.py
# -*- coding: utf-8 -*-
"""BM25 lexical index management with disk persistence."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from loguru import logger
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Simple alphanum + CJK tokenization (lower-cased)."""
    return [t for t in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", text.lower()) if t]


@dataclass
class BM25Record:
    """Saved BM25 corpus & documents."""
    corpus_tokens: List[List[str]]
    corpus_docs: List[Document]


class BM25Store:
    """BM25 store with build/load/search."""

    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.store_dir / "bm25.pkl"
        self._bm25: BM25Okapi | None = None
        self._docs: List[Document] = []
        self._tokens: List[List[str]] = []

    def build(self, docs: List[Document]) -> None:
        """Build BM25 index from documents and save to disk."""
        if not docs:
            raise ValueError("构建 BM25 失败：文档列表为空")
        logger.info("开始构建 BM25 索引")
        self._docs = docs
        self._tokens = [_tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(self._tokens)
        self.save()
        logger.info(f"BM25 索引已保存至: {self.path}")

    def save(self) -> None:
        """Persist index to disk."""
        if self._bm25 is None:
            raise RuntimeError("BM25 尚未构建，无法保存")
        record = BM25Record(corpus_tokens=self._tokens, corpus_docs=self._docs)
        with open(self.path, "wb") as f:
            pickle.dump(record, f)

    def load(self) -> None:
        """Load index from disk."""
        if not self.path.exists():
            raise FileNotFoundError(f"BM25 索引不存在: {self.path}，请先构建。")
        with open(self.path, "rb") as f:
            record: BM25Record = pickle.load(f)
        self._tokens = record.corpus_tokens
        self._docs = record.corpus_docs
        self._bm25 = BM25Okapi(self._tokens)
        logger.info(f"BM25 索引已加载: {self.path}")

    def lexical_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """BM25 search; score: higher is better."""
        if self._bm25 is None:
            raise RuntimeError("BM25 尚未加载，无法检索")
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in idxs]
