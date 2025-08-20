# scripts/build_indices.py
# -*- coding: utf-8 -*-
"""Offline index building script: load → split → FAISS + BM25 persist.

Usage:
    python scripts/build_indices.py \
        --chunk-size 800 --chunk-overlap 120
"""

from __future__ import annotations

import argparse

from loguru import logger

from app.config import settings
from deepsearch.bm25_index import BM25Store
from deepsearch.embeddings import M3EEmbeddings
from deepsearch.loaders import load_pdfs
from deepsearch.splitter import split_documents
from deepsearch.vectorstore import FaissStore


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for override."""
    parser = argparse.ArgumentParser(description="Build FAISS & BM25 indices.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for splitter.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap for splitter.")
    return parser.parse_args()


def main() -> None:
    """Main entry for index building."""
    args = parse_args()
    logger.info("开始离线构建索引流程")
    docs = load_pdfs(settings.data_pdf_dir)
    if not docs:
        logger.warning("未加载到任何 PDF 文档，流程结束")
        return

    chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if not chunks:
        logger.warning("切分后没有得到任何 chunk，流程结束")
        return

    # 1) FAISS
    embeddings = M3EEmbeddings(model_dir=settings.model_dir)
    faiss_store = FaissStore(store_dir=settings.faiss_store_dir, embeddings=embeddings)
    faiss_store.build(chunks)

    # 2) BM25
    bm25_store = BM25Store(store_dir=settings.bm25_store_dir)
    bm25_store.build(chunks)

    logger.info("索引构建完成 ✅")


if __name__ == "__main__":
    main()