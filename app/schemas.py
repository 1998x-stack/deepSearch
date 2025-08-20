# app/schemas.py
# -*- coding: utf-8 -*-
"""Pydantic request/response models for FastAPI endpoints."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DeepSearchRequest(BaseModel):
    """DeepSearch API request body."""

    query: str = Field(..., min_length=1)
    max_iterations: Optional[int] = None
    confidence_threshold: Optional[float] = None


class SourceDocument(BaseModel):
    """Single source item returned to client."""

    source: str
    score: float
    snippet: str


class DeepSearchResponse(BaseModel):
    """DeepSearch API response body."""

    query: str
    final_answer: str
    confidence: float
    iterations: int
    used_tools: List[str]
    sources: List[SourceDocument]
    trace_url: Optional[str] = None
