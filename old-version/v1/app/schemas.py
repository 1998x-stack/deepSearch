# app/schemas.py
# -*- coding: utf-8 -*-
"""Pydantic 数据模型定义。

定义 FastAPI 的请求/响应模型，保证接口契约清晰稳健。
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class DeepSearchRequest(BaseModel):
    """DeepSearch API 请求体。

    Attributes:
        query: 用户原始查询。
        max_iterations: 可选，覆盖默认最大迭代次数。
        confidence_threshold: 可选，覆盖默认置信度阈值。
    """

    query: str = Field(..., min_length=1)
    max_iterations: Optional[int] = None
    confidence_threshold: Optional[float] = None


class SourceDocument(BaseModel):
    """检索到的来源文档信息。"""

    source: str
    score: float
    snippet: str


class DeepSearchResponse(BaseModel):
    """DeepSearch API 响应体。

    Attributes:
        query: 原始查询。
        final_answer: 最终回答文本。
        confidence: 回答置信度（0~1）。
        iterations: 实际迭代次数。
        used_tools: 实际使用到的工具序列。
        sources: 相关文档源列表（去重后的Top-K）。
        trace_url: LangSmith 追踪页面链接（如果可用）。
    """

    query: str
    final_answer: str
    confidence: float
    iterations: int
    used_tools: List[str]
    sources: List[SourceDocument]
    trace_url: Optional[str] = None
