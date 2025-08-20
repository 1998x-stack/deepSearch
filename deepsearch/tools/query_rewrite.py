# deepsearch/tools/query_rewrite.py
# -*- coding: utf-8 -*-
"""Query rewrite tool (GPT-4o) with robust JSON parsing and deduplication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from loguru import logger

from deepsearch.utils import try_parse_json

PROMPT_PATH = Path("prompts/query_rewrite.md")


class QueryRewriteTool:
    """Encapsulates rewrite logic and exposes a LangChain Tool adapter."""

    name: str = "query_rewrite"
    description: str = (
        "将原始用户查询改写为3条多样化候选检索式，用于扩大召回与覆盖面。"
    )

    def __init__(self, llm: ChatOpenAI):
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"缺少重写提示词文件: {PROMPT_PATH}")
        self.llm = llm
        self.template = PROMPT_PATH.read_text(encoding="utf-8")

        def _call(query: str) -> str:
            rewrites = self.rewrite(query)
            return json.dumps({"rewrites": rewrites}, ensure_ascii=False)

        self._tool = Tool(
            name=self.name,
            description=self.description,
            func=_call,
        )

    def as_tool(self) -> Tool:
        """Return Tool for LangChain Agent consumption."""
        return self._tool

    def rewrite(self, query: str) -> List[str]:
        """Return 3 diverse rewrites with fallback & dedup."""
        if not query or not query.strip():
            return [query][:1]
        prompt = self.template.replace("{{query}}", query.strip())
        res = self.llm.invoke(prompt)
        content = res.content if hasattr(res, "content") else str(res)

        parsed = try_parse_json(content)
        candidates: List[str] = []
        if isinstance(parsed, list):
            candidates = [str(x).strip() for x in parsed if str(x).strip()]
        else:
            for line in content.splitlines():
                line = line.strip("-* \t")
                if len(line) > 0:
                    candidates.append(line)

        uniq = []
        seen = set()
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
            if len(uniq) == 3:
                break
        if len(uniq) < 3:
            uniq.extend([query] * (3 - len(uniq)))

        logger.info(f"重写结果: {uniq}")
        return uniq[:3]
