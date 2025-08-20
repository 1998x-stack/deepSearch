# deepsearch/memory_store.py
# -*- coding: utf-8 -*-
"""Persistent memory vector store (FAISS-based).

This store keeps small, durable memory snippets extracted from searches or chats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from loguru import logger


class MemoryStore:
    """FAISS-based memory store with lazy initialization."""

    def __init__(self, store_dir: str, embeddings: Embeddings):
        self.store_dir = Path(store_dir)
        self.embeddings = embeddings
        self.vs: FAISS | None = None
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load FAISS index if exists; else keep None until first add."""
        idx = self.store_dir / "index.faiss"
        if idx.exists():
            self.vs = FAISS.load_local(str(self.store_dir), self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Memory 向量库已加载: {self.store_dir}")
        else:
            logger.info(f"Memory 向量库尚未初始化: {self.store_dir}")

    def _ensure_init(self, text: str, metadata: Dict) -> None:
        """Ensure FAISS is initialized with first memory text."""
        if self.vs is None:
            logger.info("MemoryStore: 初始化向量库索引")
            self.vs = FAISS.from_texts([text], self.embeddings, metadatas=[metadata])
            self.vs.save_local(str(self.store_dir))

    def add_memory(self, text: str, metadata: Dict) -> None:
        """Add a single memory snippet."""
        if not text or not text.strip():
            return
        if self.vs is None:
            self._ensure_init(text, metadata)
        else:
            self.vs.add_texts([text], metadatas=[metadata])
            self.vs.save_local(str(self.store_dir))

    def add_memories(self, texts: List[str], metadatas: List[Dict]) -> None:
        """Add multiple memory snippets."""
        if not texts:
            return
        if self.vs is None:
            # 初始化用第一条
            self._ensure_init(texts[0], metadatas[0])
            if len(texts) > 1:
                self.vs.add_texts(texts[1:], metadatas=metadatas[1:])
                self.vs.save_local(str(self.store_dir))
        else:
            self.vs.add_texts(texts, metadatas=metadatas)
            self.vs.save_local(str(self.store_dir))

    def search_memory(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        """Retrieve related memory items (higher score = closer)."""
        if self.vs is None:
            return []
        res = self.vs.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in res]
