# deepsearch/agent.py
# -*- coding: utf-8 -*-
"""ReAct Agent 编排与深检索控制循环。

流程：
1) 调用 QueryRewriteTool 生成 3 个查询
2) 使用 HybridRetriever 检索（BM25 + FAISS + RRF）
3) 调用 RAGAnswerer 生成回答
4) 调用 ConfidenceTool 评估置信度
5) 若低于阈值：分析，选择再次重写/调用 Web 搜索，或两者
6) 最多循环 max_iterations 次，直到置信度达标或用尽
"""

from __future__ import annotations

from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from deepsearch.hybrid_retriever import HybridRetriever, RankedSource
from deepsearch.rag_chain import RAGAnswerer
from deepsearch.tools.query_rewrite import QueryRewriteTool
from deepsearch.tools.web_search import WebSearchTool
from deepsearch.tools.confidence import ConfidenceTool


@dataclass
class OrchestratorResult:
    """编排结果返回结构。"""
    final_answer: str
    confidence: float
    iterations: int
    used_tools: List[str]
    sources: List[RankedSource]
    trace_url: Optional[str] = None


class DeepSearchAgentOrchestrator:
    """围绕 ReAct Agent 的深检索编排器。"""

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
        self.llm = llm
        self.hybrid = hybrid_retriever
        self.rag = rag_answerer
        self.t_rewrite = query_rewrite_tool
        self.t_web = web_search_tool
        self.t_conf = confidence_tool
        self.max_iterations = max_iterations
        self.conf_threshold = confidence_threshold

        # 组装到 Agent（ReAct），供策略性调用（这里主要用于二次阶段决策）
        self.agent = initialize_agent(
            tools=[self.t_rewrite, self.t_web, self.t_conf],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def _collect_sources(self, all_sources: List[RankedSource], limit: int = 6) -> List[RankedSource]:
        """按来源去重并截断，适合返回给前端展示。"""
        out: List[RankedSource] = []
        seen = set()
        for s in all_sources:
            if s.source not in seen:
                seen.add(s.source)
                out.append(s)
            if len(out) >= limit:
                break
        return out

    def run(self, query: str) -> OrchestratorResult:
        """执行完整深检索循环。"""
        used_tools: List[str] = []
        best_answer = ""
        best_conf = 0.0
        best_sources: List[RankedSource] = []

        # 第一次：强制先做重写->混合检索->RAG->置信度
        rewrites = self.t_rewrite.run({"query": query})["rewrites"]
        used_tools.append(self.t_rewrite.name)

        # 对三个重写分别检索，合并来源，提升覆盖度
        merged_docs = []
        merged_sources: List[RankedSource] = []
        for rq in rewrites:
            docs, srcs = self.hybrid.search(rq)
            merged_docs.extend(docs)
            merged_sources.extend(srcs)

        # 简单去重（按 page_content 片段）
        seen_snippet = set()
        uniq_docs = []
        for d in merged_docs:
            key = d.page_content[:160]
            if key not in seen_snippet:
                uniq_docs.append(d)
                seen_snippet.add(key)

        # 生成回答
        answer = self.rag.answer(question=query, docs=uniq_docs[:12])
        # 初始置信度
        conf_res = self.t_conf.run({"question": query, "answer": answer, "sources": [s.__dict__ for s in merged_sources]})
        used_tools.append(self.t_conf.name)
        conf = float(conf_res["confidence"])

        best_answer, best_conf = answer, conf
        best_sources = self._collect_sources(merged_sources)

        logger.info(f"初次回答置信度: {conf:.3f} (阈值 {self.conf_threshold:.2f})")
        if conf >= self.conf_threshold:
            return OrchestratorResult(
                final_answer=answer,
                confidence=conf,
                iterations=1,
                used_tools=used_tools,
                sources=best_sources,
                trace_url=None,
            )

        # 若不足，则进入循环：结合 Agent 决策（重写 or web 搜索 or both）
        for it in range(2, self.max_iterations + 1):
            logger.info(f"进入第 {it} 轮改进流程（低置信度）")

            # 简单决策策略（也可交由 self.agent.run 编排更细粒度步骤）：
            # 1) 优先尝试：再次重写 + 检索
            rewrites = self.t_rewrite.run({"query": query})["rewrites"]
            used_tools.append(self.t_rewrite.name)
            merged_docs = []
            merged_sources = []
            for rq in rewrites:
                docs, srcs = self.hybrid.search(rq)
                merged_docs.extend(docs)
                merged_sources.extend(srcs)

            # 2) 若来源仍稀疏或重叠严重，补充 web 搜索（mock）
            if len({s.source for s in merged_sources}) < 3:
                web = self.t_web.run({"query": query, "top_k": 3})
                used_tools.append(self.t_web.name)
                # 将 web 结果作为补充上下文片段（模拟）
                for r in web["results"]:
                    from langchain.schema import Document
                    merged_docs.append(
                        Document(page_content=r["snippet"], metadata={"source": r["url"]})
                    )
                    merged_sources.append(
                        RankedSource(source=r["url"], score=0.02, snippet=r["snippet"])
                    )

            # 去重 + 回答
            seen_snippet.clear()
            uniq_docs = []
            for d in merged_docs:
                key = d.page_content[:160]
                if key not in seen_snippet:
                    uniq_docs.append(d)
                    seen_snippet.add(key)

            answer = self.rag.answer(question=query, docs=uniq_docs[:12])
            conf_res = self.t_conf.run({"question": query, "answer": answer, "sources": [s.__dict__ for s in merged_sources]})
            used_tools.append(self.t_conf.name)
            conf = float(conf_res["confidence"])

            # 保留更优结果
            if conf > best_conf:
                best_conf = conf
                best_answer = answer
                best_sources = self._collect_sources(merged_sources)

            logger.info(f"第 {it} 轮置信度: {conf:.3f}")
            if conf >= self.conf_threshold:
                break

        return OrchestratorResult(
            final_answer=best_answer,
            confidence=best_conf,
            iterations=it if best_conf >= self.conf_threshold else self.max_iterations,
            used_tools=used_tools,
            sources=best_sources,
            trace_url=None,
        )
