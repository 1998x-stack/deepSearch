# scripts/build_indices.py
# -*- coding: utf-8 -*-
"""离线构建索引脚本：加载->切分->FAISS & BM25 落盘。

运行：
    python scripts/build_indices.py
"""

from __future__ import annotations

from loguru import logger
from app.config import settings
from deepsearch.loaders import load_pdfs
from deepsearch.splitter import split_documents
from deepsearch.embeddings import M3EEmbeddings
from deepsearch.vectorstore import FaissStore
from deepsearch.bm25_index import BM25Store


def main() -> None:
    """主入口。"""
    logger.info("开始离线构建索引流程")
    docs = load_pdfs(settings.data_pdf_dir)
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=120)

    # 1) FAISS
    embeddings = M3EEmbeddings(model_dir=settings.model_dir)
    faiss_store = FaissStore(store_dir=settings.faiss_store_dir, embeddings=embeddings)
    faiss_store.build(chunks)

    # 2) BM25（用 chunks 级别，便于更精准定位）
    bm25_store = BM25Store(store_dir=settings.bm25_store_dir)
    bm25_store.build(chunks)

    logger.info("索引构建完成 ✅")


if __name__ == "__main__":
    main()
