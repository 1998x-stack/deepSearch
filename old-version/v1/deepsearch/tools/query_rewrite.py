# deepsearch/tools/query_rewrite.py
# -*- coding: utf-8 -*-
"""查询重写工具（GPT-4o）。

将原始 query 改写为 3 个多样化的候选检索式，用于提升召回与稳健性。
"""

from __future__ import annotations

from typing import Dict, List
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pathlib import Path
from loguru import logger

PROMPT_PATH = Path("prompts/query_rewrite.md")


class QueryRewriteInput(BaseModel):
    """输入：单个原始查询。"""
    query: str = Field(..., min_length=1, description="原始用户查询")


class QueryRewriteOutput(BaseModel):
    """输出：三个候选查询。"""
    rewrites: List[str]


class QueryRewriteTool(StructuredTool):
    """LangChain Tool：查询重写（返回3条备选）。"""

    name: str = "query_rewrite"
    description: str = (
        "将原始用户查询改写为3条多样化候选检索式，"
        "以提升召回率与覆盖面。输入为{'query': '...'}。"
    )

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"缺少重写提示词文件: {PROMPT_PATH}")
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")

        def _fn(query: str) -> Dict:
            return self._rewrite(query)

        super().__init__(
            name=self.name,
            description=self.description,
            func=_fn,  # 按 LC v0.2 兼容：func 接收一个 str (json-like) 或直接 query
            args_schema=QueryRewriteInput,
            return_direct=False,
        )

    def _rewrite(self, query: str) -> Dict:
        """内部实现：调用 LLM 生成3条改写。"""
        text = self.prompt.replace("{{query}}", query)
        res = self.llm.invoke(text)
        content = res.content if hasattr(res, "content") else str(res)

        # 期待模型按提示返回 JSON 数组；这里做鲁棒解析
        rewrites: List[str] = []
        for line in content.splitlines():
            line = line.strip("-* \t")
            if len(line) > 2:
                rewrites.append(line)
        rewrites = rewrites[:3] if len(rewrites) >= 3 else (rewrites + [query])[:3]

        logger.info(f"重写结果: {rewrites}")
        return QueryRewriteOutput(rewrites=rewrites).model_dump()
