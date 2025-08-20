# deepsearch/embeddings.py
# -*- coding: utf-8 -*-
"""Sentence-Transformers based local m3e embeddings adapter."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.embeddings.base import Embeddings
from loguru import logger
from sentence_transformers import SentenceTransformer


class M3EEmbeddings(Embeddings):
    """Local m3e embeddings via SentenceTransformer."""

    def __init__(self, model_dir: str, normalize_embeddings: bool = True):
        p = Path(model_dir)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"本地 m3e 模型目录不存在: {model_dir}")
        logger.info(f"加载本地 m3e 模型: {model_dir}")
        self.model = SentenceTransformer(str(p))
        self.normalize = normalize_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        if not texts:
            return []
        vecs = self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        vec = self.model.encode([text], normalize_embeddings=self.normalize, show_progress_bar=False)[0]
        return vec.tolist()
