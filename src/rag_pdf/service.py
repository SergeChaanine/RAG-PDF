"""Orchestration functions shared by the UI and tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rag_pdf.chunking import chunk_document
from rag_pdf.config import ChunkConfig
from rag_pdf.embeddings import LocalEmbeddingModel
from rag_pdf.llm import GroqLanguageModel
from rag_pdf.models import Answer, ChatTurn, ExtractedDocument, IndexSummary
from rag_pdf.vector_store import ChromaVectorStore


def build_index_id(
    document_id: str,
    embedding_model: str,
    chunk_config: ChunkConfig,
) -> str:
    """Include retrieval configuration so stale indexes are never silently reused."""

    identity = (
        f"{document_id}|{embedding_model}|{chunk_config.size_tokens}|"
        f"{chunk_config.overlap_percent}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def index_document(
    document: ExtractedDocument,
    chunk_config: ChunkConfig,
    embedder: LocalEmbeddingModel,
    store: ChromaVectorStore,
) -> IndexSummary:
    index_id = build_index_id(document.document_id, embedder.model_name, chunk_config)
    chunks = chunk_document(document, embedder.tokenizer, chunk_config)
    reused = store.has_index(index_id)
    if not reused:
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        store.add(
            index_id,
            document,
            chunks,
            embeddings,
            embedding_model=embedder.model_name,
            chunk_size=chunk_config.size_tokens,
            overlap_percent=chunk_config.overlap_percent,
        )
    return IndexSummary(
        index_id=index_id,
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        chunk_count=len(chunks),
        character_count=document.character_count,
        language=document.language,
        reused_existing_index=reused,
    )


def answer_question(
    *,
    index_id: str,
    question: str,
    history: Sequence[ChatTurn],
    top_k: int,
    embedder: LocalEmbeddingModel,
    store: ChromaVectorStore,
    language_model: GroqLanguageModel,
) -> Answer:
    standalone = language_model.rewrite_question(question, history)
    query_embedding = embedder.embed_query(standalone)
    sources = tuple(store.search(index_id, query_embedding, top_k))
    answer = language_model.answer(standalone, sources)
    return Answer(text=answer, standalone_question=standalone, sources=sources)
