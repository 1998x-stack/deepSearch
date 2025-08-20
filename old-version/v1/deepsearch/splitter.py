# deepsearch/splitter.py
# -*- coding: utf-8 -*-
"""文本切分模块（RecursiveCharacterTextSplitter）。

兼顾语义完整性与上下文重叠，适配 RAG 工作流。
"""

from __future__ import annotations

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger


def split_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """将原始 Document 列表切分为 chunk 化的 Document 列表。

    Args:
        docs: 原始 Document 列表。
        chunk_size: 每块最大字符数。
        chunk_overlap: 块间重叠字符数。

    Returns:
        List[Document]: 切分后的文档列表。
    """
    if not docs:
        logger.warning("split_documents 收到空文档列表")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"切分完成，chunk 数量: {len(chunks)} (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
