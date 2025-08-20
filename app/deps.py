# app/deps.py
# -*- coding: utf-8 -*-
"""Dependency injection for reusable singletons in FastAPI."""

from __future__ import annotations

from functools import lru_cache

from loguru import logger
from langchain_openai import ChatOpenAI

from app.config import settings
from deepsearch.embeddings import M3EEmbeddings
from deepsearch.vectorstore import FaissStore
from deepsearch.bm25_index import BM25Store
from deepsearch.hybrid_retriever import HybridRetriever
from deepsearch.rag_chain import RAGAnswerer
from deepsearch.memory_store import MemoryStore                    # ⭐
from deepsearch.tools.query_rewrite import QueryRewriteTool
from deepsearch.tools.web_search import WebSearchTool
from deepsearch.tools.confidence import ConfidenceTool
from deepsearch.tools.memory import RelatedMemoryTool              # ⭐


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Provide a singleton ChatOpenAI."""
    logger.info(f"初始化 ChatOpenAI(model={settings.llm_model}, temperature={settings.llm_temperature})")
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
def get_memory_store() -> MemoryStore:
    """Load singleton MemoryStore (FAISS-based)."""
    logger.info("加载 Memory 向量库")
    ms = MemoryStore(store_dir=settings.memory_store_dir, embeddings=get_embeddings())
    ms.load()  # 若不存在则懒初始化（add时会构建）
    return ms


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
    return RAGAnswerer(
        llm=get_llm(),
        token_budget=settings.rag_token_budget,
        memory_budget=settings.memory_token_budget,   # ⭐
    )


@lru_cache(maxsize=1)
def get_tools() -> dict:
    """Provide tools as a dictionary for agent assembly & orchestrator direct use."""
    return {
        "query_rewrite": QueryRewriteTool(llm=get_llm()),
        "web_search": WebSearchTool(),
        "confidence": ConfidenceTool(llm=get_llm()),
        "memory": RelatedMemoryTool(                     # ⭐
            llm=get_llm(),
            memory_store=get_memory_store(),
            memory_budget=settings.memory_token_budget,
        ),
    }
