# deepsearch/rag_chain.py
# -*- coding: utf-8 -*-
"""RAG 回答链：Stuff 模式，将检索到的上下文拼接后交给 LLM 生成答案。"""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from deepsearch.callbacks import get_default_callbacks, cost_track


PROMPT_PATH = Path("prompts/rag_answer.md")


class RAGAnswerer:
    """RAG 答案生成器。"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"缺少 RAG 提示词文件: {PROMPT_PATH}")
        self.prompt = PromptTemplate.from_file(str(PROMPT_PATH))

    def answer(self, question: str, docs: List[Document]) -> str:
        """根据问题与上下文文档生成回答文本。

        Args:
            question: 用户问题。
            docs: 检索到的文档列表。

        Returns:
            str: 生成的答案文本。
        """
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        callbacks = get_default_callbacks()

        with cost_track() as cb:
            res = self.llm.invoke(
                self.prompt.format(context=context, question=question),
                callbacks=callbacks,
            )
            logger.info(
                f"RAG LLM 完成: prompt_tokens={cb.prompt_tokens}, "
                f"completion_tokens={cb.completion_tokens}, cost=${cb.total_cost:.6f}"
            )

        # ChatOpenAI 返回 AIMessage
        return res.content if hasattr(res, "content") else str(res)
