"""Index lifecycle, retrieval across the library, and conversation-aware answers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import portalocker

from rag_pdf.catalog import PIPELINE_VERSION
from rag_pdf.models import Answer, IndexSummary


def build_index_id(
    document_id, embedding_model, chunk_config, *, filename="", tokenizer_id="model"
):
    identity = (
        PIPELINE_VERSION,
        document_id,
        filename,
        embedding_model,
        tokenizer_id,
        chunk_config.size_tokens,
        chunk_config.overlap_percent,
    )
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def library_id(documents):
    return hashlib.sha256(
        json.dumps(sorted((d.document_id, d.filename) for d in documents)).encode()
    ).hexdigest()


def index_document(
    document,
    chunk_config,
    embedder,
    store,
    *,
    tokenizer=None,
    tokenizer_id="model",
    vector_cache=None,
):
    from rag_pdf.chunking import chunk_document

    index_id = build_index_id(
        document.document_id,
        embedder.model_name,
        chunk_config,
        filename=document.filename,
        tokenizer_id=tokenizer_id,
    )
    chunks = chunk_document(document, tokenizer or embedder.tokenizer, chunk_config)
    if not chunks:
        raise ValueError("The document produced no chunks.")
    manifest_dir = store.path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{index_id}.json"
    with portalocker.Lock(str(manifest) + ".lock", timeout=120):
        expected = [c.chunk_id for c in chunks]
        reused = (
            manifest.exists()
            and json.loads(manifest.read_text()) == expected
            and store.count(index_id) == len(chunks)
        )
        if not reused:
            manifest.unlink(missing_ok=True)
            store.delete(index_id)
            for start in range(0, len(chunks), embedder.batch_size):
                batch = chunks[start : start + embedder.batch_size]
                key = (index_id, start)
                vectors = vector_cache.get(key) if vector_cache is not None else None
                if vectors is None:
                    vectors = embedder.embed_documents([c.text for c in batch])
                    if vector_cache is not None:
                        vector_cache[key] = vectors
                store.add(index_id, document, batch, vectors)
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(expected), encoding="utf-8")
            temporary.replace(manifest)
    return IndexSummary(
        index_id,
        document.document_id,
        document.filename,
        document.page_count,
        len(chunks),
        document.character_count,
        document.language,
        reused,
    )


def search_library(index_ids, query_embedding, top_k, store):
    if top_k < 1:
        raise ValueError("Top K must be positive.")
    results = [
        hit
        for index_id in dict.fromkeys(index_ids)
        for hit in store.search(index_id, query_embedding, top_k)
    ]
    return sorted(results, key=lambda hit: (-hit.score, hit.document_id, hit.chunk_id))[:top_k]


def expand_tables(sources, documents, max_characters=40_000):
    tables = {(d.document_id, t.table_id): t for d in documents for t in d.tables}
    expanded, seen, used = [], set(), 0
    for source in sources:
        key = (source.document_id, source.table_id)
        if source.table_id and key in seen:
            continue
        table = tables.get(key)
        if table:
            complete = table.render()
            if len(complete) <= max_characters - used:
                source = replace(source, text=complete, location=table.location, table=table)
                seen.add(key)
            else:
                source = replace(
                    source,
                    text=source.text + "\nPARTIAL TABLE: whole-table aggregates are unavailable.",
                )
        if used + len(source.text) > max_characters:
            break
        expanded.append(source)
        used += len(source.text)
    return tuple(expanded)


def answer_question(
    *, index_ids, documents, question, history, top_k, embedder, store, language_model
):
    standalone = language_model.rewrite_question(question, history)
    if standalone.startswith("CLARIFY:"):
        return Answer(standalone.removeprefix("CLARIFY:").strip(), standalone)
    sources = search_library(index_ids, embedder.embed_query(standalone), top_k, store)
    sources = expand_tables(sources, documents)
    return Answer(language_model.answer(standalone, sources), standalone, sources)


class ChatRepository:
    """Local chat persistence scoped by document set, independent of retrieval settings."""

    def __init__(self, root: Path):
        import sqlite3

        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "conversations.sqlite3"
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, messages TEXT)")

    def load(self, key):
        import sqlite3

        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT messages FROM chats WHERE id = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else []

    def save(self, key, messages):
        import sqlite3

        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO chats VALUES (?, ?)", (key, json.dumps(messages)))
