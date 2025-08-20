# deepsearch/utils.py
# -*- coding: utf-8 -*-
"""通用工具函数库。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
from loguru import logger
import math


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]],
    k: int = 8,
    constant: float = 60.0,
) -> List[Tuple[int, float]]:
    """对多个检索排名使用 Reciprocal Rank Fusion (RRF) 融合。

    Args:
        rankings: 每个元素为 [(doc_id, score), ...] 的列表，顺序即排名（0为第一）。
        k: 截断后要返回的 top-K。
        constant: RRF 超参，越大则衰减越缓。

    Returns:
        List[Tuple[int, float]]: 融合后的 (doc_id, rrf_score) 列表，按分数降序。
    """
    # 中文说明：RRF 将不同检索器的名次进行调和，避免单一信号失真。
    scores: Dict[int, float] = {}
    for rank_list in rankings:
        for rank, (doc_id, _score) in enumerate(rank_list):
            rr = 1.0 / (constant + rank + 1.0)
            scores[doc_id] = scores.get(doc_id, 0.0) + rr

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    logger.debug(f"RRF 融合后前{k}个: {fused}")
    return fused
