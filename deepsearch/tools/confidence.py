# deepsearch/tools/confidence.py
# -*- coding: utf-8 -*-
"""Confidence scoring tool (hybrid heuristic + LLM JSON scoring)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from loguru import logger

from deepsearch.utils import try_parse_json

PROMPT_PATH = Path("prompts/confidence.md")


class ConfidenceTool:
    """Estimate answer confidence between 0 and 1 with reasons."""

    name: str = "confidence_score"
    description: str = "评估回答置信度（0~1）与原因。"

    def __init__(self, llm: ChatOpenAI):
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"缺少置信度提示词文件: {PROMPT_PATH}")
        self.llm = llm
        self.template = PROMPT_PATH.read_text(encoding="utf-8")

        def _call(json_payload: str) -> str:
            try:
                payload = json.loads(json_payload)
            except Exception:
                payload = {}
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            sources = payload.get("sources", [])
            result = self.score(question=question, answer=answer, sources=sources)
            return json.dumps(result, ensure_ascii=False)

        self._tool = Tool(
            name=self.name,
            description=self.description,
            func=_call,
        )

    def as_tool(self) -> Tool:
        """Return Tool for LangChain Agent consumption."""
        return self._tool

    @staticmethod
    def _heuristic(question: str, answer: str, sources: List[Dict]) -> float:
        """Simple heuristic: diversity + avg score + answer length."""
        if not answer.strip():
            return 0.0
        if not sources:
            return 0.2
        uniq_sources = len({s.get("source", "") for s in sources if s.get("source")})
        avg_score = sum(float(s.get("score", 0.0)) for s in sources) / max(1, len(sources))
        length = len(answer.strip())
        length_penalty = 1.0 if 40 <= length <= 1500 else 0.85
        diversity = min(1.0, uniq_sources / 4.0)
        base = 0.5 * diversity + 0.35 * min(1.0, avg_score / 0.05) + 0.15 * length_penalty
        return max(0.0, min(1.0, base))

    def score(self, question: str, answer: str, sources: List[Dict]) -> Dict:
        """Return blended confidence and reasons."""
        h = self._heuristic(question, answer, sources)
        prompt = self.template.replace("{{question}}", question).replace("{{answer}}", answer)
        res = self.llm.invoke(prompt)
        content = res.content if hasattr(res, "content") else str(res)

        parsed = try_parse_json(content)
        llm_score = 0.6
        comment = "N/A"
        if isinstance(parsed, dict):
            try:
                s = float(parsed.get("score", 0.6))
                if 0.0 <= s <= 1.0:
                    llm_score = s
                comment = str(parsed.get("comment", ""))
            except Exception:
                pass
        else:
            for tok in content.replace(",", " ").split():
                try:
                    val = float(tok)
                    if 0.0 <= val <= 1.0:
                        llm_score = val
                        break
                except Exception:
                    continue
            comment = content[:300]

        final = max(0.0, min(1.0, 0.5 * h + 0.5 * llm_score))
        logger.info(f"置信度评分：heuristic={h:.3f}, llm={llm_score:.3f}, final={final:.3f}")

        return {
            "confidence": final,
            "reasons": f"Heuristic={h:.2f}; LLM={llm_score:.2f}; Comment={comment}",
        }
