"""Durable text embeddings so an interrupted preparation can reuse completed work."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from pathlib import Path


class EmbeddingCache:
    def __init__(self, path, model_name):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.db = sqlite3.connect(path)
        self.db.executescript(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; "
            "CREATE TABLE IF NOT EXISTS vectors (model TEXT, text_hash TEXT, "
            "dimensions INTEGER, vector BLOB, PRIMARY KEY(model,text_hash));"
        )

    @staticmethod
    def _key(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, texts):
        result = []
        for text in texts:
            row = self.db.execute(
                "SELECT dimensions,vector FROM vectors WHERE model=? AND text_hash=?",
                (self.model_name, self._key(text)),
            ).fetchone()
            if row and row[0] > 0 and len(row[1]) == 4 * row[0]:
                result.append(list(struct.unpack(f"<{row[0]}f", row[1])))
            else:
                result.append(None)
        return result

    def put_many(self, texts, vectors):
        rows = []
        for text, vector in zip(texts, vectors, strict=True):
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding cache requires finite, nonempty vectors.")
            rows.append(
                (
                    self.model_name,
                    self._key(text),
                    len(vector),
                    struct.pack(f"<{len(vector)}f", *vector),
                )
            )
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)", rows)

    def close(self):
        self.db.close()
