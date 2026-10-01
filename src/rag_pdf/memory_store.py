"""Non-persistent vector store for isolated experiments and quick tests."""

from __future__ import annotations

from collections.abc import Sequence

from rag_pdf.errors import IndexNotFoundError
from rag_pdf.models import ExtractedDocument, SearchResult, TextChunk


class InMemoryVectorStore:
    """Keep indexed chunks and vectors in process memory; data vanishes on restart."""

    def __init__(self) -> None:
        self._indexes: dict[str, dict[str, tuple[TextChunk, list[float]]]] = {}

    def has_index(self, index_id: str) -> bool:
        return bool(self._indexes.get(index_id))

    def count(self, index_id: str) -> int:
        return len(self._indexes.get(index_id, {}))

    def delete(self, index_id: str) -> None:
        self._indexes.pop(index_id, None)

    def add(
        self,
        index_id: str,
        document: ExtractedDocument,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embedding_model: str,
        chunk_size: int,
        overlap_percent: int,
    ) -> None:
        del document, embedding_model, chunk_size, overlap_percent
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        index = self._indexes.setdefault(index_id, {})
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            index[chunk.chunk_id] = (chunk, list(embedding))

    def search(
        self,
        index_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[SearchResult]:
        index = self._indexes.get(index_id)
        if index is None:
            raise IndexNotFoundError("Process the selected documents before asking questions.")
        query = list(query_embedding)
        scored = [
            (sum(a * b for a, b in zip(query, vector, strict=True)), chunk)
            for chunk, vector in index.values()
        ]
        return [
            SearchResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=score,
            )
            for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        ]
