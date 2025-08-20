# deepsearch/utils.py
# -*- coding: utf-8 -*-
"""General-purpose utility functions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Sequence, Tuple

from loguru import logger
import tiktoken


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]],
    k: int = 8,
    constant: float = 60.0,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion (RRF) of multiple rankings."""
    scores: Dict[int, float] = {}
    for rank_list in rankings:
        for rank, (doc_id, _score) in enumerate(rank_list):
            rr = 1.0 / (constant + rank + 1.0)
            scores[doc_id] = scores.get(doc_id, 0.0) + rr
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return fused


def document_fingerprint(text: str, source: str | None = None) -> str:
    """Create a deterministic fingerprint for deduplication."""
    h = hashlib.sha256()
    h.update(text[:2048].encode("utf-8", errors="ignore"))
    if source:
        h.update(source.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def limit_by_token_budget(
    docs: Sequence[str],
    token_budget: int,
    model: str = "gpt-4o",
) -> List[str]:
    """Greedy pack of text snippets into token budget (approx)."""
    if token_budget <= 0:
        return []
    enc = tiktoken.get_encoding("cl100k_base")
    packed: List[str] = []
    used = 0
    for d in docs:
        tokens = len(enc.encode(d, disallowed_special=()))
        if tokens > token_budget:
            truncated = enc.decode(enc.encode(d)[: max(0, token_budget - used - 4)])
            if truncated:
                packed.append(truncated)
                used = token_budget
            break
        if used + tokens <= token_budget:
            packed.append(d)
            used += tokens
        else:
            break
    return packed


def try_parse_json(text: str) -> dict | list | None:
    """Robust JSON parsing with cleanup."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None
