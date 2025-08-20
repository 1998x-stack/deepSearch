# app/main.py
# -*- coding: utf-8 -*-
"""FastAPI 入口。

提供 /health 与 /deepsearch 两个端点：
- /deepsearch 实现 ReAct + 混合检索 + RAG + 置信度循环。
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from app.config import settings
from app.schemas import DeepSearchRequest, DeepSearchResponse, SourceDocument
from app import deps
from deepsearch.agent import DeepSearchAgentOrchestrator


app = FastAPI(
    title="DeepSearch with LangChain + FastAPI",
    version="1.0.0",
    description="Hybrid (BM25 + Dense) RAG with ReAct Agent, LangSmith tracing & callbacks.",
)


@app.get("/health")
def health() -> dict:
    """健康检查端点。"""
    return {"status": "ok", "tracing": settings.tracing_enabled}


@app.post("/deepsearch", response_model=DeepSearchResponse)
def deepsearch(req: DeepSearchRequest) -> DeepSearchResponse:
    """主检索/问答端点。

    流程：
    1) 查询重写 -> 3 个候选查询
    2) 混合检索（BM25 + 向量，RRF 融合）
    3) RAG 生成答案
    4) 置信度评估，若不足则触发二次流程（重写/网页搜索或两者）
    5) 循环直至置信度达标或迭代用尽

    Args:
        req: DeepSearchRequest 对象，包含原始 query 等。

    Returns:
        DeepSearchResponse: 包含最终答案、置信度、使用的工具、来源文档等。
    """
    logger.info(f"收到查询: {req.query}")

    # 注入资源
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
        result = orchestrator.run(query=req.query)
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
