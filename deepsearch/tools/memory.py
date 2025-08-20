# deepsearch/tools/memory.py
# -*- coding: utf-8 -*-
"""Related memory extraction, retrieval, and compression tool.

功能：
1) 从检索结果中提取“可长期记忆”的关键信息，并写入 MemoryStore；
2) 从 MemoryStore 中检索与当前 query 相关的记忆；
3) 使用 LLM 将检索到的记忆进行压缩（生成紧凑上下文），用于 RAG。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from langchain.tools import Tool
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from loguru import logger

from deepsearch.memory_store import MemoryStore
from deepsearch.utils import limit_by_token_budget, try_parse_json

PROMPT_EXTRACT = Path("prompts/memory_extract.md")
PROMPT_COMPRESS = Path("prompts/memory_compress.md")


class RelatedMemoryTool:
    """Memory management tool with LangChain Tool adapter."""

    name: str = "related_memory"
    description: str = (
        "从搜索结果提取可存储记忆、检索相关记忆并压缩为紧凑上下文。"
        "输入 JSON，例如："
        '{"action":"extract_and_store","query":"...","sources":[{"source":"...","snippet":"..."}]} 或 '
        '{"action":"retrieve","query":"...","top_k":8} 或 '
        '{"action":"compress","items":["m1","m2",...]}'
    )

    def __init__(self, llm: ChatOpenAI, memory_store: MemoryStore, memory_budget: int = 800):
        if not PROMPT_EXTRACT.exists():
            raise FileNotFoundError(f"缺少提示词文件: {PROMPT_EXTRACT}")
        if not PROMPT_COMPRESS.exists():
            raise FileNotFoundError(f"缺少提示词文件: {PROMPT_COMPRESS}")
        self.llm = llm
        self.ms = memory_store
        self.tok_budget = max(200, int(memory_budget))
        self.tpl_extract = PROMPT_EXTRACT.read_text(encoding="utf-8")
        self.tpl_compress = PROMPT_COMPRESS.read_text(encoding="utf-8")

        def _call(json_payload: str) -> str:
            try:
                payload = json.loads(json_payload)
            except Exception:
                payload = {}
            action = payload.get("action", "retrieve")
            if action == "extract_and_store":
                query = payload.get("query", "")
                sources = payload.get("sources", [])
                stored = self.extract_and_store(query, sources)
                return json.dumps({"stored": stored}, ensure_ascii=False)
            elif action == "retrieve":
                query = payload.get("query", "")
                top_k = int(payload.get("top_k", 8))
                items = self.retrieve_related(query=query, top_k=top_k)
                return json.dumps({"items": items}, ensure_ascii=False)
            elif action == "compress":
                items = payload.get("items", [])
                summary = self.compress_items(items)
                return json.dumps({"summary": summary}, ensure_ascii=False)
            else:
                return json.dumps({"error": "unknown action"}, ensure_ascii=False)

        self._tool = Tool(
            name=self.name,
            description=self.description,
            func=_call,
        )

    def as_tool(self) -> Tool:
        """Return Tool for LangChain Agent consumption."""
        return self._tool

    # --------- Public APIs for orchestrator ---------

    def extract_and_store(self, query: str, sources: List[Dict]) -> List[str]:
        """Extract salient memory from latest search sources and store them."""
        if not sources:
            return []
        # 将 sources 的 snippet 拼接，并做 token 裁剪，控制成本
        snippets = [s.get("snippet", "") for s in sources if s.get("snippet")]
        packed = "\n".join(limit_by_token_budget(snippets, token_budget=self.tok_budget))

        prompt = self.tpl_extract.replace("{{query}}", query).replace("{{snippets}}", packed)
        res = self.llm.invoke(prompt)
        content = res.content if hasattr(res, "content") else str(res)

        # 期待 JSON 数组，元素为“可记忆的句子/事实”
        parsed = try_parse_json(content)
        candidates: List[str] = []
        if isinstance(parsed, list):
            candidates = [str(x).strip() for x in parsed if str(x).strip()]
        else:
            # 回退：行级解析
            for line in content.splitlines():
                t = line.strip("-* \t")
                if t:
                    candidates.append(t)

        # 将候选写入 MemoryStore
        now = datetime.utcnow().isoformat()
        metadatas = [{"type": "memory", "created": now, "source": "search_memory", "query": query}] * len(candidates)
        if candidates:
            self.ms.add_memories(candidates, metadatas)
        logger.info(f"Memory 提取并写入: {len(candidates)} 条")
        return candidates

    def retrieve_related(self, query: str, top_k: int = 8) -> List[str]:
        """Retrieve top-k related memory items for the query."""
        top_k = min(max(1, int(top_k)), 20)
        results = self.ms.search_memory(query=query, k=top_k)
        items = []
        for d, _score in results:
            items.append(d.page_content)
        return items

    def compress_items(self, items: List[str]) -> str:
        """Compress a list of memory items to a compact context string."""
        if not items:
            return ""
        # 控制 token 预算
        budgeted = limit_by_token_budget(items, token_budget=self.tok_budget)
        merged = "\n".join(budgeted)
        prompt = self.tpl_compress.replace("{{items}}", merged)
        res = self.llm.invoke(prompt)
        content = res.content if hasattr(res, "content") else str(res)
        # 直接返回压缩后的段落/要点
        return content.strip()
