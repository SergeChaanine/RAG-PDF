"""Persistent SQLite vector store for exact, small-scale experiments."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from rag_pdf.errors import IndexNotFoundError
from rag_pdf.models import ExtractedDocument, SearchResult, TextChunk


class SQLiteVectorStore:
    """Store chunks and embeddings in SQLite and rank by exact cosine similarity."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                index_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                PRIMARY KEY (index_id, chunk_id)
            )"""
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS chunks_by_index ON chunks(index_id)"
        )
        self._connection.commit()

    def has_index(self, index_id: str) -> bool:
        return self.count(index_id) > 0

    def count(self, index_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE index_id = ?", (index_id,)
        ).fetchone()
        return int(row[0])

    def delete(self, index_id: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM chunks WHERE index_id = ?", (index_id,))

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
        rows = [
            (
                index_id,
                chunk.chunk_id,
                chunk.document_id,
                chunk.filename,
                chunk.text,
                chunk.page_number,
                chunk.chunk_index,
                json.dumps(list(embedding)),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self._connection:
            self._connection.executemany(
                """INSERT OR REPLACE INTO chunks
                   (index_id, chunk_id, document_id, filename, text,
                    page_number, chunk_index, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def search(
        self,
        index_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[SearchResult]:
        rows = self._connection.execute(
            """SELECT chunk_id, filename, text, page_number, chunk_index, embedding
               FROM chunks WHERE index_id = ?""",
            (index_id,),
        ).fetchall()
        if not rows:
            raise IndexNotFoundError("Process the selected documents before asking questions.")

        query = list(query_embedding)
        query_norm = math.sqrt(sum(value * value for value in query))
        if query_norm == 0:
            raise ValueError("The query embedding must not be a zero vector.")

        scored: list[tuple[float, tuple[object, ...]]] = []
        for row in rows:
            vector = json.loads(row[5])
            if len(vector) != len(query):
                raise ValueError("The query and stored embeddings have different dimensions.")
            dot = sum(left * right for left, right in zip(query, vector, strict=True))
            norm = math.sqrt(sum(value * value for value in vector))
            score = dot / (query_norm * norm) if norm else 0.0
            scored.append((score, row))

        return [
            SearchResult(
                chunk_id=str(row[0]),
                text=str(row[2]),
                filename=str(row[1]),
                page_number=int(row[3]),
                chunk_index=int(row[4]),
                score=score,
            )
            for score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        ]

    def close(self) -> None:
        self._connection.close()
