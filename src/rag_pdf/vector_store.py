"""Persistent vector stores sharing a cosine-similarity search contract."""

from __future__ import annotations

import json
import uuid
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from rag_pdf.models import SearchResult


class VectorStore(Protocol):
    path: Path

    def count(self, index_id: str) -> int: ...
    def delete(self, index_id: str) -> None: ...
    def add(self, index_id, document, chunks, embeddings, **metadata) -> None: ...
    def search(self, index_id, query_embedding, top_k) -> list[SearchResult]: ...


def result(row, score):
    return SearchResult(
        row["chunk_id"],
        row["text"],
        int(row["page_number"]),
        int(row["chunk_index"]),
        float(score),
        row["document_id"],
        row["filename"],
        row["location"],
        row["table_id"],
    )


class ChromaVectorStore:
    def __init__(self, path: Path):
        import chromadb

        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))

    @staticmethod
    def collection_name(index_id):
        return f"rag-{index_id[:40]}"

    def count(self, index_id):
        from chromadb.errors import NotFoundError

        try:
            return self._client.get_collection(self.collection_name(index_id)).count()
        except NotFoundError:
            return 0

    def delete(self, index_id):
        from chromadb.errors import NotFoundError

        with suppress(NotFoundError):
            self._client.delete_collection(self.collection_name(index_id))

    def add(self, index_id, document, chunks, embeddings, **metadata):
        collection = self._client.get_or_create_collection(
            self.collection_name(index_id),
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[asdict(c) for c in chunks],
        )

    def search(self, index_id, query_embedding, top_k):
        collection = self._client.get_collection(
            self.collection_name(index_id), embedding_function=None
        )
        count = collection.count()
        if not count:
            return []
        response = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, count),
            include=["metadatas", "distances"],
        )
        return [
            result(row, 1 - distance)
            for row, distance in zip(
                response["metadatas"][0], response["distances"][0], strict=True
            )
        ]


class QdrantVectorStore:
    def __init__(self, path: Path):
        from qdrant_client import QdrantClient

        self.path = path
        self._client = QdrantClient(path=str(path))

    def count(self, index_id):
        if not self._client.collection_exists(index_id):
            return 0
        return self._client.count(index_id, exact=True).count

    def delete(self, index_id):
        if self._client.collection_exists(index_id):
            self._client.delete_collection(index_id)

    def add(self, index_id, document, chunks, embeddings, **metadata):
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not self._client.collection_exists(index_id):
            self._client.create_collection(
                index_id,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
            )
        self._client.upsert(
            index_id,
            points=[
                PointStruct(id=str(uuid.UUID(c.chunk_id)), vector=list(vector), payload=asdict(c))
                for c, vector in zip(chunks, embeddings, strict=True)
            ],
            wait=True,
        )

    def search(self, index_id, query_embedding, top_k):
        response = self._client.query_points(
            index_id, query=list(query_embedding), limit=top_k, with_payload=True
        )
        return [result(point.payload, point.score) for point in response.points]

    def close(self):
        self._client.close()


class LanceVectorStore:
    def __init__(self, path: Path):
        import lancedb

        self.path = path
        self._client = lancedb.connect(str(path))

    def _table(self, index_id):
        try:
            return self._client.open_table(f"rag_{index_id}")
        except ValueError as exc:
            if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                return None
            raise

    def count(self, index_id):
        table = self._table(index_id)
        return table.count_rows() if table is not None else 0

    def delete(self, index_id):
        if self._table(index_id) is not None:
            self._client.drop_table(f"rag_{index_id}")

    def add(self, index_id, document, chunks, embeddings, **metadata):
        rows = [
            {"vector": list(v), "payload": json.dumps(asdict(c))}
            for c, v in zip(chunks, embeddings, strict=True)
        ]
        table = self._table(index_id)
        if table is None:
            self._client.create_table(f"rag_{index_id}", data=rows)
        else:
            table.add(rows)

    def search(self, index_id, query_embedding, top_k):
        rows = (
            self._table(index_id)
            .search(list(query_embedding))
            .distance_type("cosine")
            .limit(top_k)
            .to_list()
        )
        return [result(json.loads(row["payload"]), 1 - row["_distance"]) for row in rows]


def create_store(backend: str, root: Path) -> VectorStore:
    classes = {
        "Chroma": ChromaVectorStore,
        "Qdrant": QdrantVectorStore,
        "LanceDB": LanceVectorStore,
    }
    if backend not in classes:
        raise ValueError(f"Unknown database: {backend}")
    try:
        return classes[backend](root / backend.lower())
    except ImportError as exc:
        raise RuntimeError("Install the updated requirements.txt to use this database.") from exc
