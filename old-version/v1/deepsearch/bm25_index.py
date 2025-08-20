# deepsearch/bm25_index.py
# -*- coding: utf-8 -*-
"""BM25 索引管理：构建、保存（pickle）、加载与检索。

使用 rank_bm25 进行倒排检索，支持磁盘持久化以适配生产环境。
"""

from __future__ import annotations

from typing import List, Tuple
from dataclasses import dataclass
from langchain.schema import Document
from loguru import logger
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import re


def _tokenize(text: str) -> List[str]:
    """简易分词器：按非字母数字分隔，小写化。"""
    return [t for t in re.split(r"[^0-9A-Za-z]+", text.lower()) if t]


@dataclass
class BM25Record:
    """BM25 存储记录结构。"""
    corpus_tokens: List[List[str]]
    corpus_docs: List[Document]


class BM25Store:
    """BM25 存储与检索封装。"""

    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.store_dir / "bm25.pkl"
        self._bm25: BM25Okapi | None = None
        self._docs: List[Document] = []
        self._tokens: List[List[str]] = []

    def build(self, docs: List[Document]) -> None:
        """构建 BM25 索引并保存。"""
        if not docs:
            raise ValueError("构建 BM25 失败：文档列表为空")
        logger.info("开始构建 BM25 索引")
        self._docs = docs
        self._tokens = [_tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(self._tokens)
        self.save()
        logger.info(f"BM25 索引已保存至: {self.path}")

    def save(self) -> None:
        """持久化 BM25 到磁盘。"""
        if self._bm25 is None:
            raise RuntimeError("BM25 尚未构建，无法保存")
        record = BM25Record(corpus_tokens=self._tokens, corpus_docs=self._docs)
        with open(self.path, "wb") as f:
            pickle.dump(record, f)

    def load(self) -> None:
        """从磁盘加载 BM25。"""
        if not self.path.exists():
            raise FileNotFoundError(f"BM25 索引不存在: {self.path}，请先构建。")
        with open(self.path, "rb") as f:
            record: BM25Record = pickle.load(f)
        self._tokens = record.corpus_tokens
        self._docs = record.corpus_docs
        self._bm25 = BM25Okapi(self._tokens)
        logger.info(f"BM25 索引已加载: {self.path}")

    def lexical_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """BM25 检索。

        Returns:
            List[(Document, score)]: 得分越大越相关。
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 尚未加载，无法检索")
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        # 取 top-k
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in idxs]
