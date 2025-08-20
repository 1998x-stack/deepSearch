# app/main.py
# -*- coding: utf-8 -*-
"""FastAPI entry points."""

from __future__ import annotations

import uuid

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from loguru import logger

from app.config import settings
from app.schemas import DeepSearchRequest, DeepSearchResponse, SourceDocument
from app import deps
from deepsearch.agent import DeepSearchAgentOrchestrator


app = FastAPI(
    title="DeepSearch: LangChain + FastAPI + Hybrid RAG + ReAct",
    version="1.1.0",
    description="Industrial-grade deep search pipeline with LangSmith tracing and callbacks.",
)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "tracing": settings.tracing_enabled,
        "model": settings.llm_model,
        "confidence_threshold": settings.confidence_threshold,
    }


@app.post("/deepsearch", response_model=DeepSearchResponse)
def deepsearch(req: DeepSearchRequest, x_request_id: str | None = Header(default=None)) -> DeepSearchResponse:
    """Main deepsearch endpoint.

    End-to-end flow:
    1) query rewrite → 3 queries
    2) hybrid search (BM25 + Embeddings with RRF)
    3) RAG answer
    4) confidence scoring
    5) low-confidence loop with rewrite &/or web search (ReAct agent suggestions included)
    """
    request_id = x_request_id or str(uuid.uuid4())
    logger.bind(request_id=request_id).info(f"收到查询: {req.query}")

    # Inject dependencies
    hybrid = deps.get_hybrid_retriever()
    rag = deps.get_rag_answerer()
    tools = deps.get_tools()
    llm = deps.get_llm()

    orchestrator = DeepSearchAgentOrchestrator(
        llm=llm,
        hybrid_retriever=hybrid,
        rag_answerer=rag,
        query_rewrite_tool=tools["query_rewrite"],
        web_search_tool=tools["web_search"],
        confidence_tool=tools["confidence"],
        max_iterations=req.max_iterations or settings.max_iterations,
        confidence_threshold=req.confidence_threshold or settings.confidence_threshold,
    )

    try:
        result = orchestrator.run(query=req.query, request_id=request_id)
    except FileNotFoundError as fe:
        logger.exception("DeepSearch 执行失败（资源缺失）")
        raise HTTPException(status_code=500, detail=f"资源缺失: {fe}")
    except Exception as e:
        logger.exception("DeepSearch 执行失败")
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        SourceDocument(source=s.source, score=s.score, snippet=s.snippet)
        for s in result.sources
    ]

    return DeepSearchResponse(
        query=req.query,
        final_answer=result.final_answer,
        confidence=result.confidence,
        iterations=result.iterations,
        used_tools=result.used_tools,
        sources=sources,
        trace_url=result.trace_url,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
