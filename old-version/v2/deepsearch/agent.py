# deepsearch/agent.py
# -*- coding: utf-8 -*-
"""ReAct-powered orchestration loop for DeepSearch.

Implements:
- First pass: rewrite → hybrid search → RAG → confidence
- Low-confidence loop (max N): heuristic improvements and ReAct Agent hinting
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from loguru import logger

from deepsearch.hybrid_retriever import HybridRetriever, RankedSource
from deepsearch.rag_chain import RAGAnswerer
from deepsearch.tools.confidence import ConfidenceTool
from deepsearch.tools.query_rewrite import QueryRewriteTool
from deepsearch.tools.web_search import WebSearchTool
from deepsearch.utils import document_fingerprint


@dataclass
class OrchestratorResult:
    """Final result container."""
    final_answer: str
    confidence: float
    iterations: int
    used_tools: List[str]
    sources: List[RankedSource]
    trace_url: Optional[str] = None


class DeepSearchAgentOrchestrator:
    """Main orchestrator for DeepSearch loop."""

    def __init__(
        self,
        llm: ChatOpenAI,
        hybrid_retriever: HybridRetriever,
        rag_answerer: RAGAnswerer,
        query_rewrite_tool: QueryRewriteTool,
        web_search_tool: WebSearchTool,
        confidence_tool: ConfidenceTool,
        max_iterations: int,
        confidence_threshold: float,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be within [0,1]")

        self.llm = llm
        self.hybrid = hybrid_retriever
        self.rag = rag_answerer
        self.t_rewrite = query_rewrite_tool
        self.t_web = web_search_tool
        self.t_conf = confidence_tool
        self.max_iterations = max_iterations
        self.conf_threshold = confidence_threshold

        # Assemble a ReAct Agent with our tools for suggestions during low-confidence loops.
        self.agent = initialize_agent(
            tools=[self.t_rewrite.as_tool(), self.t_web.as_tool(), self.t_conf.as_tool()],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    @staticmethod
    def _dedup_docs(docs: List, limit: int = 24) -> List:
        """Deduplicate by fingerprint and cap maximum list size."""
        out: List = []
        seen = set()
        for d in docs:
            src = (d.metadata or {}).get("source", "unknown")
            fp = document_fingerprint(d.page_content, src)
            if fp not in seen:
                out.append(d)
                seen.add(fp)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _collect_sources(all_sources: List[RankedSource], limit: int = 8) -> List[RankedSource]:
        """Deduplicate by `source` and cap list size for client return."""
        out: List[RankedSource] = []
        seen = set()
        for s in all_sources:
            if s.source not in seen:
                seen.add(s.source)
                out.append(s)
            if len(out) >= limit:
                break
        return out

    def _rewrite_and_retrieve(self, query: str) -> Tuple[List, List[RankedSource]]:
        """Rewrite → run hybrid search on each → merge and dedup docs & sources."""
        rewrites = self.t_rewrite.rewrite(query)
        all_docs, all_sources = [], []
        for rq in rewrites:
            docs, srcs = self.hybrid.search(rq)
            all_docs.extend(docs)
            all_sources.extend(srcs)
        uniq_docs = self._dedup_docs(all_docs, limit=30)
        return uniq_docs, all_sources

    def _score(self, question: str, answer: str, sources: List[RankedSource]) -> Dict:
        """Score with confidence tool."""
        payload = {
            "question": question,
            "answer": answer,
            "sources": [s.__dict__ for s in sources],
        }
        return self.t_conf.score(**payload)

    def _agent_hint(self, question: str, last_conf: float, reason_note: str) -> Dict:
        """Ask ReAct agent for the next step suggestion in free-form text, parse keywords."""
        hint_prompt = (
            "You are a ReAct agent helping an iterative RAG system.\n"
            f"Question: {question}\n"
            f"Last confidence: {last_conf:.2f}\n"
            f"Notes: {reason_note}\n"
            "Decide next step: choose one of [rewrite, web_search, both]. "
            "Reply with a single JSON object: {\"action\": \"rewrite|web_search|both\", \"notes\": \"...\"}."
        )
        try:
            raw = self.agent.run(hint_prompt)
        except Exception as e:
            logger.warning(f"Agent hint failed: {e}")
            return {"action": "rewrite", "notes": "fallback: agent unavailable"}
        parsed = {"action": "rewrite", "notes": "fallback"}
        try:
            obj = json.loads(raw) if raw.strip().startswith("{") else json.loads(raw[raw.find("{"): raw.rfind("}")+1])
            act = obj.get("action", "rewrite")
            if act not in {"rewrite", "web_search", "both"}:
                act = "rewrite"
            parsed = {"action": act, "notes": obj.get("notes", "")}
        except Exception:
            # fallback by keywords
            lower = raw.lower()
            if "web" in lower:
                parsed = {"action": "web_search", "notes": raw[:200]}
        return parsed

    def run(self, query: str, request_id: str | None = None) -> OrchestratorResult:
        """Run the orchestrated deepsearch loop."""
        used_tools: List[str] = []
        best_answer = ""
        best_conf = 0.0
        best_sources: List[RankedSource] = []
        iterations = 0

        # === First pass ===
        docs, srcs = self._rewrite_and_retrieve(query)
        used_tools.append(self.t_rewrite.name)

        answer = self.rag.answer(question=query, docs=docs[:12])
        score_res = self._score(query, answer, srcs)
        used_tools.append(self.t_conf.name)
        conf = float(score_res["confidence"])
        reason_note = score_res["reasons"]

        best_answer, best_conf = answer, conf
        best_sources = self._collect_sources(srcs)

        iterations = 1
        logger.info(f"[{request_id}] 初次置信度: {conf:.3f}, 阈值: {self.conf_threshold:.2f}")
        if conf >= self.conf_threshold:
            return OrchestratorResult(
                final_answer=answer, confidence=conf, iterations=iterations,
                used_tools=used_tools, sources=best_sources, trace_url=None
            )

        # === Improvement loop ===
        while iterations < self.max_iterations:
            iterations += 1
            # Ask agent for a suggestion only after first failure.
            suggestion = self._agent_hint(query, conf, reason_note)
            action = suggestion.get("action", "rewrite")
            logger.info(f"[{request_id}] 第{iterations}轮 Agent 建议: {action} / {suggestion.get('notes','')}")

            # Heuristic: always try rewrite; optionally append web search
            use_rewrite = action in {"rewrite", "both"}
            use_web = action in {"web_search", "both"}

            docs, srcs = self._rewrite_and_retrieve(query) if use_rewrite else ([], [])
            if use_rewrite:
                used_tools.append(self.t_rewrite.name)

            if use_web:
                results = self.t_web.search(query=query, top_k=3)
                used_tools.append(self.t_web.name)
                # Convert web results into pseudo-Documents
                from langchain.schema import Document
                for r in results:
                    docs.append(Document(page_content=r["snippet"], metadata={"source": r["url"]}))
                    srcs.append(RankedSource(source=r["url"], score=0.02, snippet=r["snippet"]))

            # Dedup and answer again
            docs = self._dedup_docs(docs, limit=30)
            answer = self.rag.answer(question=query, docs=docs[:12])
            score_res = self._score(query, answer, srcs)
            used_tools.append(self.t_conf.name)
            conf = float(score_res["confidence"])
            reason_note = score_res["reasons"]

            if conf > best_conf:
                best_conf = conf
                best_answer = answer
                best_sources = self._collect_sources(srcs)

            logger.info(f"[{request_id}] 第{iterations}轮 置信度: {conf:.3f}")
            if conf >= self.conf_threshold:
                break

        return OrchestratorResult(
            final_answer=best_answer,
            confidence=best_conf,
            iterations=iterations,
            used_tools=used_tools,
            sources=best_sources,
            trace_url=None,
        )
