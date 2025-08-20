# app/deps.py
# -*- coding: utf-8 -*-
"""依赖注入模块。

集中化初始化/单例资源，便于在 FastAPI 端点复用，提高性能与可维护性。
"""

from __future__ import annotations

from functools import lru_cache
from loguru import logger
from langchain_openai import ChatOpenAI
from deepsearch.embeddings import M3EEmbeddings
from deepsearch.vectorstore import FaissStore
from deepsearch.bm25_index import BM25Store
from deepsearch.hybrid_retriever import HybridRetriever
from app.config import settings
from deepsearch.rag_chain import RAGAnswerer
from deepsearch.tools.query_rewrite import QueryRewriteTool
from deepsearch.tools.web_search import WebSearchTool
from deepsearch.tools.confidence import ConfidenceTool


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """提供 ChatOpenAI 单例。

    Returns:
        ChatOpenAI: 用于问答/重写等的 LLM 客户端。
    """
    logger.info("初始化 ChatOpenAI（gpt-4o）")
    return ChatOpenAI(model="gpt-4o", temperature=0.2)


@lru_cache(maxsize=1)
def get_embeddings() -> M3EEmbeddings:
    """提供 m3e 本地 Embeddings 单例。"""
    logger.info("初始化本地 m3e Embeddings")
    return M3EEmbeddings(model_dir=settings.model_dir)


@lru_cache(maxsize=1)
def get_faiss_store() -> FaissStore:
    """提供 FAISS 向量库单例（从磁盘加载）。"""
    logger.info("加载 FAISS 向量库")
    store = FaissStore(store_dir=settings.faiss_store_dir, embeddings=get_embeddings())
    store.load()
    return store


@lru_cache(maxsize=1)
def get_bm25_store() -> BM25Store:
    """提供 BM25 索引单例（从磁盘加载）。"""
    logger.info("加载 BM25 索引库")
    bm25 = BM25Store(store_dir=settings.bm25_store_dir)
    bm25.load()
    return bm25


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    """提供混合检索器单例。"""
    logger.info("初始化 HybridRetriever")
    return HybridRetriever(
        faiss_store=get_faiss_store(),
        bm25_store=get_bm25_store(),
        k_bm25=settings.hybrid_k_bm25,
        k_dense=settings.hybrid_k_dense,
        rrf_k=settings.rrf_k,
    )


@lru_cache(maxsize=1)
def get_rag_answerer() -> RAGAnswerer:
    """提供 RAG 答案生成器单例。"""
    return RAGAnswerer(llm=get_llm())


@lru_cache(maxsize=1)
def get_tools() -> dict:
    """提供工具集合（便于 Agent 组装）。"""
    return {
        "query_rewrite": QueryRewriteTool(llm=get_llm()),
        "web_search": WebSearchTool(),
        "confidence": ConfidenceTool(llm=get_llm()),
    }
