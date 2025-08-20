# deepsearch/hybrid_retriever.py
# -*- coding: utf-8 -*-
"""Hybrid retriever: BM25 + Dense via RRF fusion and safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from langchain.schema import Document
from loguru import logger

from deepsearch.bm25_index import BM25Store
from deepsearch.utils import document_fingerprint, reciprocal_rank_fusion
from deepsearch.vectorstore import FaissStore


@dataclass
class RankedSource:
    """Source info for client consumption."""
    source: str
    score: float
    snippet: str


class HybridRetriever:
    """Hybrid retriever with configurable k and RRF constants."""

    def __init__(
        self,
        faiss_store: FaissStore,
        bm25_store: BM25Store,
        k_bm25: int = 12,
        k_dense: int = 12,
        rrf_k: int = 8,
        rrf_constant: float = 60.0,
    ):
        if k_bm25 < 1 or k_dense < 1 or rrf_k < 1:
            raise ValueError("k_bm25/k_dense/rrf_k 必须为正整数")
        self.faiss = faiss_store
        self.bm25 = bm25_store
        self.k_bm25 = k_bm25
        self.k_dense = k_dense
        self.rrf_k = rrf_k
        self.rrf_constant = rrf_constant

    def search(self, query: str) -> Tuple[List[Document], List[RankedSource]]:
        """Run BM25 + Dense, fuse, deduplicate, and return."""
        # (1) Individual channels
        lex_results = self.bm25.lexical_search(query, k=self.k_bm25)  # [(doc, score)]
        dense_results = self.faiss.dense_search(query, k=self.k_dense)  # [(doc, score)]

        # (2) Build a stable pool (doc_id is based on fingerprint)
        pool_docs: List[Document] = []
        doc_index: Dict[str, int] = {}

        def add_doc(d: Document) -> int:
            src = (d.metadata or {}).get("source", "unknown")
            key = document_fingerprint(d.page_content, src)
            if key not in doc_index:
                doc_index[key] = len(pool_docs)
                pool_docs.append(d)
            return doc_index[key]

        rank_bm25 = [(add_doc(d), s) for d, s in lex_results]
        rank_dense = [(add_doc(d), s) for d, s in dense_results]

        # (3) RRF fusion
        fused = reciprocal_rank_fusion(
            rankings=[rank_bm25, rank_dense],
            k=self.rrf_k,
            constant=self.rrf_constant,
        )

        # (4) Format output
        out_docs: List[Document] = []
        out_sources: List[RankedSource] = []
        for doc_id, rrf_score in fused:
            d = pool_docs[doc_id]
            snippet = d.page_content[:200].replace("\n", " ")
            src = (d.metadata or {}).get("source", "unknown")
            out_docs.append(d)
            out_sources.append(RankedSource(source=src, score=rrf_score, snippet=snippet))

        logger.info(f"Hybrid 检索完成: BM25={len(lex_results)}, Dense={len(dense_results)}, Fused={len(out_docs)}")
        return out_docs, out_sources
