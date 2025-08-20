# deepsearch/loaders.py
# -*- coding: utf-8 -*-
"""Document loaders for PDF directories."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from loguru import logger


def load_pdfs(pdf_dir: str) -> List[Document]:
    """Load PDFs into page-level Documents."""
    p = Path(pdf_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"PDF 目录不存在或不可用: {pdf_dir}")
    logger.info(f"开始加载 PDF 目录: {pdf_dir}")
    loader = DirectoryLoader(
        path=pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        logger.warning("没有从 PDF 目录中加载到任何文档")
    logger.info(f"PDF 加载完成，页级文档数量: {len(docs)}")
    return docs
