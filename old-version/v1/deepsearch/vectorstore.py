# deepsearch/vectorstore.py
# -*- coding: utf-8 -*-
"""FAISS 向量库封装：构建、保存、加载与检索。"""

from __future__ import annotations

from typing import List, Tuple
from pathlib import Path
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings


class FaissStore:
    """FAISS 向量库管理器（落盘路径 + Embeddings 注入）。"""

    def __init__(self, store_dir: str, embeddings: Embeddings):
        """初始化。

        Args:
            store_dir: FAISS 持久化路径。
            embeddings: Embeddings 实例（m3e）。
        """
        self.store_dir = store_dir
        self.embeddings = embeddings
        self.vs: FAISS | None = None

    def build(self, docs: List[Document]) -> None:
        """根据文档构建向量库并保存。"""
        if not docs:
            raise ValueError("构建 FAISS 向量库失败：文档列表为空")
        logger.info("开始构建 FAISS 向量库")
        self.vs = FAISS.from_documents(docs, self.embeddings)
        Path(self.store_dir).mkdir(parents=True, exist_ok=True)
        self.vs.save_local(self.store_dir)
        logger.info(f"FAISS 向量库已保存至: {self.store_dir}")

    def load(self) -> None:
        """从本地加载 FAISS 向量库。"""
        p = Path(self.store_dir)
        if not p.exists():
            raise FileNotFoundError(f"FAISS 路径不存在: {self.store_dir}，请先运行构建脚本。")
        self.vs = FAISS.load_local(self.store_dir, self.embeddings, allow_dangerous_deserialization=True)
        logger.info(f"FAISS 向量库已加载: {self.store_dir}")

    def dense_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """向量检索，返回文档与相似度分数（score 越大越相似）。"""
        if self.vs is None:
            raise RuntimeError("FAISS 尚未加载，无法检索")
        results = self.vs.similarity_search_with_score(query, k=k)
        # 统一输出格式：[(Document, score_float)]
        return [(doc, float(score)) for doc, score in results]
