# deepsearch/rag_chain.py
# -*- coding: utf-8 -*-
"""RAG answerer with token budgeting and memory fusion."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from loguru import logger

from deepsearch.callbacks import cost_track, get_default_callbacks
from deepsearch.utils import limit_by_token_budget

PROMPT_PATH = Path("prompts/rag_answer.md")


class RAGAnswerer:
    """RAG answer generator using a Stuff-like prompt with optional memory."""

    def __init__(self, llm: ChatOpenAI, token_budget: int = 2500, memory_budget: int = 800):
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"缺少 RAG 提示词文件: {PROMPT_PATH}")
        self.llm = llm
        self.prompt = PromptTemplate.from_template(PROMPT_PATH.read_text(encoding="utf-8"))
        self.token_budget = max(400, token_budget)
        self.memory_budget = max(200, memory_budget)

    def answer(self, question: str, docs: List[Document], memory: Optional[str] = None) -> str:
        """Generate answer from question + contextual docs + optional memory."""
        if not question.strip():
            return "问题为空，请提供有效问题。"

        # 文档上下文裁剪（token预算）
        if docs:
            text_chunks = [d.page_content for d in docs]
            packed_context = "\n\n---\n\n".join(
                limit_by_token_budget(text_chunks, token_budget=self.token_budget)
            )
        else:
            packed_context = ""

        # 记忆上下文裁剪（token预算）
        mem_context = memory or ""
        if mem_context:
            mem_context = "\n".join(
                limit_by_token_budget([mem_context], token_budget=self.memory_budget)
            )

        callbacks = get_default_callbacks()
        content = self.prompt.format(context=packed_context, memory=mem_context, question=question)

        with cost_track() as cb:
            res = self.llm.invoke(content, callbacks=callbacks)
            logger.info(
                f"RAG LLM 完成: prompt_tokens={cb.prompt_tokens}, "
                f"completion_tokens={cb.completion_tokens}, cost=${cb.total_cost:.6f}"
            )

        return res.content if hasattr(res, "content") else str(res)
