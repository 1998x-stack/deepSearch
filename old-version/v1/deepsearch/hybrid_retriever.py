# deepsearch/hybrid_retriever.py
# -*- coding: utf-8 -*-
"""混合检索组件：BM25 + Dense（FAISS），并使用 RRF 融合。"""

from __future__ import annotations

from typing import List, Tuple
from dataclasses import dataclass
from langchain.schema import Document
from deepsearch.vectorstore import FaissStore
from deepsearch.bm25_index import BM25Store
from deepsearch.utils import reciprocal_rank_fusion
from loguru import logger


@dataclass
class RankedSource:
    """融合后用于返回的来源信息。"""
    source: str
    score: float
    snippet: str


class HybridRetriever:
    """混合检索器，封装 BM25 + Dense 并进行 RRF 融合。"""

    def __init__(
        self,
        faiss_store: FaissStore,
        bm25_store: BM25Store,
        k_bm25: int = 12,
        k_dense: int = 12,
        rrf_k: int = 8,
    ):
        self.faiss = faiss_store
        self.bm25 = bm25_store
        self.k_bm25 = k_bm25
        self.k_dense = k_dense
        self.rrf_k = rrf_k

    def search(self, query: str) -> Tuple[List[Document], List[RankedSource]]:
        """执行混合检索并返回融合后的文档与来源信息。

        Args:
            query: 原始或重写后的查询。

        Returns:
            docs: 去重后的 Document 列表（按融合分数排序）。
            sources: 可返回给前端的来源信息（source/score/snippet）。
        """
        # 1) BM25
        lex_results = self.bm25.lexical_search(query, k=self.k_bm25)
        # 2) Dense
        dense_results = self.faiss.dense_search(query, k=self.k_dense)

        # 将两个结果各自转为 (doc_id, score) 排名列表（按输入顺序视为排名）
        # 这里采用 simple id = enumerate over combined unique docs
        # 为 RRF，需要对齐 doc_id；我们先构建全量文档池（保持顺序稳定）。
        pool_docs: List[Document] = []
        doc_index: dict[str, int] = {}

        def add_doc(d: Document):
            key = (d.metadata or {}).get("source", "") + "::" + d.page_content[:64]
            if key not in doc_index:
                doc_index[key] = len(pool_docs)
                pool_docs.append(d)
            return doc_index[key]

        rank_bm25 = [(add_doc(d), s) for d, s in lex_results]
        rank_dense = [(add_doc(d), s) for d, s in dense_results]

        fused = reciprocal_rank_fusion([rank_bm25, rank_dense], k=self.rrf_k)

        # 按融合结果输出
        out_docs: List[Document] = []
        out_sources: List[RankedSource] = []
        for doc_id, rrf_score in fused:
            d = pool_docs[doc_id]
            snippet = d.page_content[:180].replace("\n", " ")
            src = (d.metadata or {}).get("source", "unknown")
            out_docs.append(d)
            out_sources.append(RankedSource(source=src, score=rrf_score, snippet=snippet))

        logger.info(f"Hybrid 检索完成，返回文档: {len(out_docs)}")
        return out_docs, out_sources
