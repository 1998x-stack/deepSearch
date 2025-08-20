# deepsearch/rag_chain.py
# -*- coding: utf-8 -*-
"""RAG answerer with token budgeting and tracing."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from loguru import logger

from deepsearch.callbacks import cost_track, get_default_callbacks
from deepsearch.utils import limit_by_token_budget

PROMPT_PATH = Path("prompts/rag_answer.md")


class RAGAnswerer:
    """RAG answer generator using a Stuff-like prompt."""

    def __init__(self, llm: ChatOpenAI, token_budget: int = 2500):
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"缺少 RAG 提示词文件: {PROMPT_PATH}")
        self.llm = llm
        self.prompt = PromptTemplate.from_template(PROMPT_PATH.read_text(encoding="utf-8"))
        self.token_budget = max(400, token_budget)  # 中文：最低保底预算，避免过短导致无上下文

    def answer(self, question: str, docs: List[Document]) -> str:
        """Generate answer from question + contextual docs."""
        if not question.strip():
            return "问题为空，请提供有效问题。"
        if not docs:
            logger.warning("RAGAnswerer: 无可用上下文，将尝试直接回答")
            context = ""
        else:
            # 中文：按 token 预算裁剪上下文，避免超窗口
            text_chunks = [d.page_content for d in docs]
            packed = limit_by_token_budget(text_chunks, token_budget=self.token_budget)
            context = "\n\n---\n\n".join(packed)

        callbacks = get_default_callbacks()
        content = self.prompt.format(context=context, question=question)

        with cost_track() as cb:
            res = self.llm.invoke(content, callbacks=callbacks)
            logger.info(
                f"RAG LLM 完成: prompt_tokens={cb.prompt_tokens}, "
                f"completion_tokens={cb.completion_tokens}, cost=${cb.total_cost:.6f}"
            )

        return res.content if hasattr(res, "content") else str(res)
