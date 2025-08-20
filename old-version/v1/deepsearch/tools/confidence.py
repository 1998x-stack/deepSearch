# deepsearch/tools/confidence.py
# -*- coding: utf-8 -*-
"""置信度评分 Tool。

结合启发式指标（覆盖度、来源多样性、回答长度等）与 LLM 自评，输出 [0,1] 置信度与原因。
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from pathlib import Path
from loguru import logger
import math

PROMPT_PATH = Path("prompts/confidence.md")


class ConfidenceInput(BaseModel):
    """置信度评估输入。"""
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    sources: List[Dict] = Field(default_factory=list, description="[{source, score, snippet}, ...]")


class ConfidenceOutput(BaseModel):
    """置信度评估输出。"""
    confidence: float
    reasons: str


class ConfidenceTool(StructuredTool):
    """结合启发式 + LLM 自评的置信度评分工具。"""

    name: str = "confidence_score"
    description: str = (
        "评估当前回答的置信度，输入为{'question', 'answer', 'sources': [...]}"
        "，输出[0,1]置信度与原因。"
    )

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"缺少置信度提示词文件: {PROMPT_PATH}")
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")

        def _fn(question: str, answer: str, sources: List[Dict]) -> Dict:
            return self._score(question, answer, sources)

        super().__init__(
            name=self.name,
            description=self.description,
            func=_fn,
            args_schema=ConfidenceInput,
            return_direct=False,
        )

    @staticmethod
    def _heuristic(question: str, answer: str, sources: List[Dict]) -> float:
        """启发式置信度：来源多样性、RRF 分数、回答长度简单合成。"""
        if not sources:
            return 0.2
        uniq_sources = len({s["source"] for s in sources})
        avg_score = sum(s["score"] for s in sources) / len(sources)
        length_penalty = 1.0 if 40 <= len(answer) <= 1200 else 0.8
        diversity = min(1.0, uniq_sources / 4.0)
        base = 0.5 * diversity + 0.4 * min(1.0, avg_score / 0.05) + 0.1 * length_penalty
        return max(0.0, min(1.0, base))

    def _score(self, question: str, answer: str, sources: List[Dict]) -> Dict:
        """综合启发式 + LLM 评估，返回最终置信度与原因。"""
        h = self._heuristic(question, answer, sources)
        text = self.prompt.replace("{{question}}", question).replace("{{answer}}", answer)
        res = self.llm.invoke(text)
        llm_comment = res.content if hasattr(res, "content") else str(res)

        # 将 LLM 字面打分解析（0~1），失败则给出中性值
        llm_score = 0.6
        for tok in llm_comment.split():
            try:
                val = float(tok)
                if 0.0 <= val <= 1.0:
                    llm_score = val
                    break
            except Exception:
                continue

        # 融合：加权平均
        final = 0.5 * h + 0.5 * llm_score
        final = max(0.0, min(1.0, final))
        logger.info(f"置信度评分：heuristic={h:.3f}, llm={llm_score:.3f}, final={final:.3f}")
        reasons = f"Heuristic={h:.2f}; LLM={llm_score:.2f}. LLM点评: {llm_comment[:400]}"
        return ConfidenceOutput(confidence=final, reasons=reasons).model_dump()
