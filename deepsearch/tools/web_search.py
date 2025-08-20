# deepsearch/tools/web_search.py
# -*- coding: utf-8 -*-
"""Mock Web search tool."""

from __future__ import annotations

import json
from typing import Dict, List

from langchain.tools import Tool
from loguru import logger


class WebSearchTool:
    """Web search mock tool."""

    name: str = "web_search"
    description: str = (
        "Web 搜索（mock）；输入为查询字符串，返回JSON字符串，包含若干条(title/url/snippet)。"
    )

    def __init__(self):
        def _call(query: str) -> str:
            results = self.search(query=query, top_k=5)
            return json.dumps({"results": results}, ensure_ascii=False)

        self._tool = Tool(
            name=self.name,
            description=self.description,
            func=_call,
        )

    def as_tool(self) -> Tool:
        """Return Tool for LangChain Agent consumption."""
        return self._tool

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Mock implementation returning deterministic items."""
        top_k = min(max(1, int(top_k)), 10)
        logger.info(f"[Mock] Web 搜索: {query}, top_k={top_k}")
        results = [
            {
                "title": f"Mock Result {i+1} for {query}",
                "url": f"https://example.com/search?q={query}&rank={i+1}",
                "snippet": f"This is a mock snippet #{i+1} for query: {query}.",
            }
            for i in range(top_k)
        ]
        return results
