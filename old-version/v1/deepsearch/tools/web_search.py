# deepsearch/tools/web_search.py
# -*- coding: utf-8 -*-
"""Web 搜索 Tool（Mock 版）。

可替换为真实实现（如：SerpAPI/DuckDuckGo/SearchAPI），
此处以占位模拟为主，便于在 ReAct 决策里调用。
"""

from __future__ import annotations

from typing import List, Dict
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from loguru import logger


class WebSearchInput(BaseModel):
    """Web 搜索输入参数。"""
    query: str = Field(..., min_length=1, description="搜索关键词")
    top_k: int = Field(5, ge=1, le=10, description="返回条数")


class WebSearchOutput(BaseModel):
    """Web 搜索返回结构。"""
    results: List[Dict]


class WebSearchTool(StructuredTool):
    """可并入 Agent 的 Web 搜索工具（Mock）。"""

    name: str = "web_search"
    description: str = (
        "Web 搜索（mock），输入{'query': '...', 'top_k': 5}，"
        "返回模拟的搜索结果列表（title/url/snippet）。"
    )

    def __init__(self):
        def _fn(query: str, top_k: int = 5) -> Dict:
            return self._search(query, top_k)

        super().__init__(
            name=self.name,
            description=self.description,
            func=_fn,
            args_schema=WebSearchInput,
            return_direct=False,
        )

    def _search(self, query: str, top_k: int = 5) -> Dict:
        """Mock 实现：返回固定模板的结果。"""
        logger.info(f"[Mock] Web 搜索: {query}, top_k={top_k}")
        results = [
            {
                "title": f"Mock Result {i+1} for {query}",
                "url": f"https://example.com/search?q={query}&rank={i+1}",
                "snippet": f"This is a mock snippet #{i+1} for query: {query}.",
            }
            for i in range(top_k)
        ]
        return WebSearchOutput(results=results).model_dump()
