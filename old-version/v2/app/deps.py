# app/deps.py
# -*- coding: utf-8 -*-
"""Dependency injection for reusable singletons in FastAPI."""

from __future__ import annotations

from functools import lru_cache

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI

from app.config import settings
from deepsearch.embeddings import M3EEmbeddings
from deepsearch.vectorstore import FaissStore
from deepsearch.bm25_index import BM25Store
from deepsearch.hybrid_retriever import HybridRetriever
from deepsearch.rag_chain import RAGAnswerer
from deepsearch.tools.query_rewrite import QueryRewriteTool
from deepsearch.tools.web_search import WebSearchTool
from deepsearch.tools.confidence import ConfidenceTool


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Provide a singleton ChatOpenAI."""
    logger.info(f"初始化 ChatOpenAI(model={settings.llm_model}, temperature={settings.llm_temperature})")
    # 中文：设置请求超时，增强鲁棒性
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> M3EEmbeddings:
    """Provide singleton local m3e embeddings."""
    logger.info("初始化本地 m3e Embeddings")
    return M3EEmbeddings(model_dir=settings.model_dir)


@lru_cache(maxsize=1)
def get_faiss_store() -> FaissStore:
    """Load singleton FAISS store from disk."""
    logger.info("加载 FAISS 向量库")
    store = FaissStore(store_dir=settings.faiss_store_dir, embeddings=get_embeddings())
    store.load()
    return store


@lru_cache(maxsize=1)
def get_bm25_store() -> BM25Store:
    """Load singleton BM25 store from disk."""
    logger.info("加载 BM25 索引库")
    bm25 = BM25Store(store_dir=settings.bm25_store_dir)
    bm25.load()
    return bm25


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    """Provide singleton hybrid retriever."""
    logger.info("初始化 HybridRetriever")
    return HybridRetriever(
        faiss_store=get_faiss_store(),
        bm25_store=get_bm25_store(),
        k_bm25=settings.hybrid_k_bm25,
        k_dense=settings.hybrid_k_dense,
        rrf_k=settings.rrf_k,
        rrf_constant=settings.rrf_constant,
    )


@lru_cache(maxsize=1)
def get_rag_answerer() -> RAGAnswerer:
    """Provide singleton RAGAnswerer with token budgeting."""
    return RAGAnswerer(llm=get_llm(), token_budget=settings.rag_token_budget)


@lru_cache(maxsize=1)
def get_tools() -> dict:
    """Provide tools as a dictionary for agent assembly & orchestrator direct use."""
    return {
        "query_rewrite": QueryRewriteTool(llm=get_llm()),
        "web_search": WebSearchTool(),
        "confidence": ConfidenceTool(llm=get_llm()),
    }
