"""Persistent keyword retrieval over the same chunks as the vector indexes."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import asdict, replace

from rag_pdf.vector_store import result

_STOP_WORDS = (
    "a an the and or of to in on at for from with is are was were be been being "
    "do does did have has had i me my we our you your it its this that these those "
    "what which who when where why how can could would should please tell show give "
    "about some any all by as than then into also "
    "ما ماذا من في على عن الى كيف هل هذا هذه ذلك التي الذي و او ان "
    "خلال عام سنة سنه لعام لسنة لسنه حقق حققت حققه حققته تم تمت كان كانت"
)
STOP_WORDS = frozenset(_STOP_WORDS.split())


def normalize(text):
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.translate(str.maketrans({"ى": "ي", "ـ": ""}))


class KeywordIndex:
    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(path)) as db:
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
                "body, filename, location, index_id UNINDEXED, payload UNINDEXED, "
                "tokenize='porter unicode61')"
            )

    def ensure_index(self, index_id, chunks):
        # Index IDs include the pipeline, document, filename, and chunk settings.
        # A transaction keeps an interrupted keyword build from leaving partial rows.
        with closing(sqlite3.connect(self.path)) as db, db:
            count = db.execute(
                "SELECT count(*) FROM chunks WHERE index_id = ?", (index_id,)
            ).fetchone()[0]
            if count == len(chunks):
                return
            db.execute("DELETE FROM chunks WHERE index_id = ?", (index_id,))
            db.executemany(
                "INSERT INTO chunks(body, filename, location, index_id, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        normalize(chunk.text),
                        normalize(chunk.filename),
                        normalize(chunk.location),
                        index_id,
                        json.dumps(asdict(chunk), ensure_ascii=False),
                    )
                    for chunk in chunks
                ),
            )

    def search(self, index_ids, question, limit, *, document_ids=None):
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", normalize(question))))
        terms = [term for term in terms if term not in STOP_WORDS][:64]
        if not terms or not index_ids:
            return []
        # Only letters/digits reach MATCH; user punctuation never becomes FTS syntax.
        query = " OR ".join(f'"{term}"' for term in terms)
        placeholders = ",".join("?" for _ in index_ids)
        scope = ""
        scope_values = ()
        if document_ids:
            scope = (
                " AND json_extract(payload, '$.document_id') IN ("
                + ",".join("?" for _ in document_ids)
                + ")"
            )
            scope_values = tuple(document_ids)
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                "SELECT payload, -bm25(chunks, 1.0, 3.0, 2.0) FROM chunks WHERE chunks MATCH ? "
                f"AND index_id IN ({placeholders}) {scope} "
                "ORDER BY bm25(chunks, 1.0, 3.0, 2.0), index_id, rowid LIMIT ?",
                (query, *index_ids, *scope_values, limit),
            ).fetchall()
        return [
            replace(
                result(json.loads(row[0]), 0.0),
                retrieval_method="keyword",
                keyword_score=row[1],
            )
            for row in rows
        ]
