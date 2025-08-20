# app/config.py
# -*- coding: utf-8 -*-
"""Application settings and environment configuration.

This module validates environment variables, prepares directories,
and exposes a typed settings object for the whole system.

中文说明：
- 集中管理配置（路径、阈值、模型、端口等），并进行基础校验与目录创建。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Global application settings."""

    # === API keys / tracing ===
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="deepsearch", env="LANGSMITH_PROJECT")
    tracing_enabled: bool = Field(default=True, env="LANGSMITH_TRACING")

    # === Paths ===
    data_pdf_dir: str = Field(default="data/pdf")
    model_dir: str = Field(default="model")
    faiss_store_dir: str = Field(default="stores/faiss")
    bm25_store_dir: str = Field(default="stores/bm25")

    # === Retrieval params ===
    top_k: int = 6
    hybrid_k_bm25: int = 12
    hybrid_k_dense: int = 12
    rrf_k: int = 8
    rrf_constant: float = 60.0

    # === Confidence & loop ===
    confidence_threshold: float = 0.75
    max_iterations: int = 3

    # === LLM & budgets ===
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 40
    # RAG context预算（approx tokens）
    rag_token_budget: int = 2500

    # === Server ===
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("confidence_threshold")
    def _check_conf_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence_threshold must be within [0,1].")
        return v

    @validator("max_iterations")
    def _check_iters(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_iterations must be >= 1.")
        return v


settings = Settings()

# Ensure directories exist.
for _p in [settings.data_pdf_dir, settings.model_dir, settings.faiss_store_dir, settings.bm25_store_dir]:
    Path(_p).mkdir(parents=True, exist_ok=True)
    logger.info(f"确保目录存在: {_p}")

# LangSmith tracing env.
if settings.tracing_enabled:
    os.environ["LANGSMITH_TRACING"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

# OpenAI key env.
if settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key