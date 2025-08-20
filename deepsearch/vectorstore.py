# deepsearch/vectorstore.py
# -*- coding: utf-8 -*-
"""FAISS vector store management with disk persistence."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from loguru import logger


class FaissStore:
    """FAISS-based dense vector index with persistence."""

    def __init__(self, store_dir: str, embeddings: Embeddings):
        self.store_dir = store_dir
        self.embeddings = embeddings
        self.vs: FAISS | None = None

    def build(self, docs: List[Document]) -> None:
        """Build index from documents and persist to disk."""
        if not docs:
            raise ValueError("构建 FAISS 向量库失败：文档列表为空")
        logger.info("开始构建 FAISS 向量库")
        self.vs = FAISS.from_documents(docs, self.embeddings)
        Path(self.store_dir).mkdir(parents=True, exist_ok=True)
        self.vs.save_local(self.store_dir)
        logger.info(f"FAISS 向量库已保存至: {self.store_dir}")

    def load(self) -> None:
        """Load index from disk."""
        p = Path(self.store_dir)
        if not p.exists():
            raise FileNotFoundError(f"FAISS 路径不存在: {self.store_dir}，请先构建。")
        self.vs = FAISS.load_local(self.store_dir, self.embeddings, allow_dangerous_deserialization=True)
        logger.info(f"FAISS 向量库已加载: {self.store_dir}")

    def dense_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Dense similarity search with score (higher is better)."""
        if self.vs is None:
            raise RuntimeError("FAISS 尚未加载，无法检索")
        results = self.vs.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in results]
