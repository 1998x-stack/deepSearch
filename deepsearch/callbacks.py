# deepsearch/callbacks.py
# -*- coding: utf-8 -*-
"""Callback and tracing utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import List

from langchain.callbacks import StdOutCallbackHandler, get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler


def get_default_callbacks() -> List[BaseCallbackHandler]:
    """Return a default set of callbacks (stdout)."""
    return [StdOutCallbackHandler()]


@contextmanager
def cost_track():
    """Context manager for OpenAI token/cost accounting."""
    with get_openai_callback() as cb:
        yield cb
