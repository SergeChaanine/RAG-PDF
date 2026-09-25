"""Persistent Chroma storage isolated by document and index configuration."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import chromadb

from rag_pdf.errors import IndexNotFoundError
from rag_pdf.models import ExtractedDocument, SearchResult, TextChunk


class ChromaVectorStore:
    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))

    @staticmethod
    def collection_name(index_id: str) -> str:
        return f"pdf-{index_id[:40]}"

    def has_index(self, index_id: str) -> bool:
        try:
            return self._client.get_collection(self.collection_name(index_id)).count() > 0
        except Exception:
            return False

    def count(self, index_id: str) -> int:
        try:
            return self._client.get_collection(self.collection_name(index_id)).count()
        except Exception:
            return 0

    def delete(self, index_id: str) -> None:
        """Remove an incomplete index before rebuilding it."""

        with suppress(Exception):
            self._client.delete_collection(self.collection_name(index_id))

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
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        collection = self._client.get_or_create_collection(
            name=self.collection_name(index_id),
            metadata={
                "hnsw:space": "cosine",
                "document_id": document.document_id,
                "filename": document.filename,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "overlap_percent": overlap_percent,
            },
        )
        embedding_payload = (
            embeddings
            if isinstance(embeddings, list)
            else [list(vector) for vector in embeddings]
        )
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embedding_payload,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "page_chunk_index": chunk.page_chunk_index,
                    "token_count": chunk.token_count,
                    "start_index": chunk.start_index,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        index_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[SearchResult]:
        try:
            collection = self._client.get_collection(self.collection_name(index_id))
        except Exception as exc:
            raise IndexNotFoundError("Process the selected PDF before asking questions.") from exc

        count = collection.count()
        if count == 0:
            return []

        response = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        return [
            SearchResult(
                chunk_id=chunk_id,
                text=text,
                page_number=int(metadata["page_number"]),
                chunk_index=int(metadata["chunk_index"]),
                score=1.0 - float(distance),
            )
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
