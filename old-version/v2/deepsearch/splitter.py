# deepsearch/splitter.py
# -*- coding: utf-8 -*-
"""Recursive text splitter with overlap for RAG."""

from __future__ import annotations

from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


def split_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Split documents into overlapping chunks.

    Args:
        docs: Source documents.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap size.

    Returns:
        Chunked Documents.
    """
    if not docs:
        logger.warning("split_documents: 输入文档为空，返回空列表")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"切分完成，chunk 数量: {len(chunks)} (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
