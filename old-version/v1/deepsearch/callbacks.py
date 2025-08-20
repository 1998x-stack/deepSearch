# deepsearch/callbacks.py
# -*- coding: utf-8 -*-
"""回调与追踪工具。

封装常用回调（控制台打印、OpenAI 用量统计），结合 LangSmith 追踪。
"""

from __future__ import annotations

from typing import List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from contextlib import contextmanager
from langchain.callbacks import get_openai_callback


def get_default_callbacks() -> List[BaseCallbackHandler]:
    """提供一组默认回调处理器。

    Returns:
        List[BaseCallbackHandler]: 回调列表。
    """
    return [StdOutCallbackHandler()]


@contextmanager
def cost_track():
    """OpenAI token/费用统计上下文管理器。

    Yields:
        cb: OpenAI 回调对象，包含 token/cost 统计字段。
    """
    with get_openai_callback() as cb:
        yield cb
