# app/config.py
# -*- coding: utf-8 -*-
"""应用配置模块。

该模块集中管理环境变量、路径、搜索超参等配置项。
遵循 PEP 257 文档字符串规范，并使用类型注解以确保可读性与可维护性。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from loguru import logger
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """全局应用配置（来自环境变量或 .env）。

    Attributes:
        openai_api_key: OpenAI API Key（用于 gpt-4o）。
        langsmith_api_key: LangSmith API Key（用于追踪与可视化）。
        langsmith_project: LangSmith 项目名。
        tracing_enabled: 是否启用 LangSmith 追踪。
        data_pdf_dir: PDF 数据目录。
        model_dir: 本地 m3e 模型目录路径。
        faiss_store_dir: FAISS 向量库持久化目录。
        bm25_store_dir: BM25 索引持久化目录。
        top_k: 基础 Top-K 检索数。
        hybrid_k_bm25: BM25 初次检索数。
        hybrid_k_dense: 向量初次检索数。
        rrf_k: RRF 融合保留的Top-K文档数。
        confidence_threshold: 置信度阈值（低于则触发二次流程）。
        max_iterations: 低置信度时的最大重试轮数。
        host: FastAPI 绑定地址。
        port: FastAPI 端口。
    """

    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="deepsearch", env="LANGSMITH_PROJECT")
    tracing_enabled: bool = Field(default=True, env="LANGSMITH_TRACING")

    data_pdf_dir: str = Field(default="data/pdf")
    model_dir: str = Field(default="model")
    faiss_store_dir: str = Field(default="stores/faiss")
    bm25_store_dir: str = Field(default="stores/bm25")

    top_k: int = 6
    hybrid_k_bm25: int = 12
    hybrid_k_dense: int = 12
    rrf_k: int = 8

    confidence_threshold: float = 0.72
    max_iterations: int = 3

    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# 确保目录存在（如果不存在则创建）
for _p in [settings.data_pdf_dir, settings.model_dir, settings.faiss_store_dir, settings.bm25_store_dir]:
    Path(_p).mkdir(parents=True, exist_ok=True)
    logger.info(f"确保目录存在: {_p}")

# 设置 LangSmith 相关环境变量，启用追踪（可在 .env 控制）
if settings.tracing_enabled:
    os.environ["LANGSMITH_TRACING"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

# 设置 OpenAI API Key（LangChain-OpenAI使用）
if settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
