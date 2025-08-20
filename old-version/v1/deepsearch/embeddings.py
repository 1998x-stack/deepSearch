# deepsearch/embeddings.py
# -*- coding: utf-8 -*-
"""m3e 本地 Embeddings 适配器。

将 sentence-transformers 的本地 m3e 模型包装为 LangChain Embeddings 接口。
"""

from __future__ import annotations

from typing import Iterable, List
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from pathlib import Path
from loguru import logger


class M3EEmbeddings(Embeddings):
    """基于 sentence-transformers 的 m3e 向量化实现。

    注意：要求 `model_dir` 下存在可用的 m3e 模型（如 moka-ai/m3e-base 的本地拷贝）。
    """

    def __init__(self, model_dir: str, normalize_embeddings: bool = True):
        """初始化 M3EEmbeddings。

            Args:
                model_dir: 本地 m3e 模型目录。
                normalize_embeddings: 是否在输出时进行向量归一化，利于余弦相似度。
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"本地 m3e 模型目录不存在: {model_dir}")
        logger.info(f"加载本地 m3e 模型: {model_dir}")
        self.model = SentenceTransformer(str(model_path))
        self.normalize = normalize_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档。"""
        if not texts:
            return []
        vecs = self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化。"""
        vec = self.model.encode([text], normalize_embeddings=self.normalize, show_progress_bar=False)[0]
        return vec.tolist()
