"""Index lifecycle, retrieval across the library, and conversation-aware answers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import portalocker

from rag_pdf.catalog import PIPELINE_VERSION
from rag_pdf.lexical import KeywordIndex
from rag_pdf.models import Answer, IndexSummary

RETRIEVAL_VERSION = "bilingual-v1"


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
        KeywordIndex(store.path / "keywords.sqlite3").ensure_index(index_id, chunks)
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


def search_library(
    index_ids, query_embedding, top_k, store, *, question="", query_variants=()
):
    """Fuse semantic/keyword candidates and reserve coverage across active documents."""
    if top_k < 1:
        raise ValueError("Top K must be positive.")
    index_ids = list(dict.fromkeys(index_ids))
    candidate_limit = max(20, top_k * 4)
    queries = [(question, query_embedding), *query_variants]
    results = [
        hit
        for _, embedding in queries
        for index_id in index_ids
        for hit in store.search(index_id, embedding, candidate_limit)
    ]
    semantic = sorted(results, key=lambda hit: (-hit.score, hit.document_id, hit.chunk_id))
    keyword = []
    keyword_path = store.path / "keywords.sqlite3"
    if question and keyword_path.exists():
        # Search each index separately so another document cannot hide keyword candidates.
        keywords = KeywordIndex(keyword_path)
        for query, _ in queries:
            for index_id in index_ids:
                keyword.extend(keywords.search([index_id], query, candidate_limit))

    candidates = {}
    for hit in semantic:
        candidates.setdefault((hit.document_id, hit.chunk_id), hit)
    for hit in keyword:
        key = (hit.document_id, hit.chunk_id)
        if key in candidates and candidates[key].retrieval_method != "keyword":
            candidates[key] = replace(
                candidates[key],
                retrieval_method="semantic + keyword",
                keyword_score=max(hit.keyword_score, candidates[key].keyword_score),
            )
        elif key not in candidates or hit.keyword_score > candidates[key].keyword_score:
            candidates[key] = hit
    # Keep match strength, not just rank: a weak match to "worksheet" must not
    # beat a strong match to a sheet name and column that dense search missed.
    strongest_keyword = max((hit.keyword_score for hit in keyword), default=0.0)
    relevance = {
        key: (
            0.4 * max(0.0, min(1.0, hit.score))
            + (0.6 * hit.keyword_score / strongest_keyword if strongest_keyword else 0.0)
        )
        for key, hit in candidates.items()
    }

    ranked = sorted(candidates, key=lambda key: (-relevance[key], *key))
    selected, represented = [], set()
    # Keep the best evidence from each document when Top K has enough slots.
    for key in ranked:
        hit = candidates[key]
        if hit.document_id not in represented:
            selected.append(key)
            represented.add(hit.document_id)
            if len(selected) == top_k:
                break
    # Prefer distinct tables; complete tables are expanded below, so their other
    # row chunks would otherwise consume slots without adding evidence.
    tables = {
        (candidates[key].document_id, candidates[key].table_id)
        for key in selected
        if candidates[key].table_id
    }
    # In bilingual libraries, generic words can score higher in the question's
    # language. Reserve a second distinct piece of evidence per document when
    # there is room, before one language consumes all remaining slots.
    if len(represented) > 1 and top_k >= 2 * len(represented):
        counts = dict.fromkeys(represented, 1)
        for key in ranked:
            hit = candidates[key]
            table_key = (hit.document_id, hit.table_id)
            if counts[hit.document_id] >= 2 or key in selected:
                continue
            if hit.table_id and table_key in tables:
                continue
            selected.append(key)
            counts[hit.document_id] += 1
            if hit.table_id:
                tables.add(table_key)
    for key in ranked:
        if len(selected) >= top_k:
            break
        hit = candidates[key]
        table_key = (hit.document_id, hit.table_id)
        if key in selected or (hit.table_id and table_key in tables):
            continue
        selected.append(key)
        if hit.table_id:
            tables.add(table_key)
    return [candidates[key] for key in sorted(selected, key=lambda key: (-relevance[key], *key))]


def expand_tables(sources, documents, max_characters=40_000):
    tables = {(d.document_id, t.table_id): t for d in documents for t in d.tables}
    worksheets = {
        (document.document_id, table.location.rsplit(", rows ", 1)[0]): table
        for document in documents
        if document.file_type == "xlsx"
        for table in document.tables
    }
    expanded, seen, used = [], set(), {}
    document_count = len({source.document_id for source in sources})
    document_budget = max_characters // max(document_count, 1)
    for source in sources:
        remaining = document_budget - used.get(source.document_id, 0)
        # Excel has a separate worksheet-title chunk. A match to that title
        # must lead to the data table, rather than just the sheet's name.
        if not source.table_id and source.text.startswith("Excel worksheet: "):
            worksheet = worksheets.get((source.document_id, source.location))
            if worksheet:
                source = replace(
                    source,
                    table_id=worksheet.table_id,
                    text=replace(worksheet, rows=worksheet.rows[:1]).render(),
                )
        key = (source.document_id, source.table_id)
        if source.table_id and key in seen:
            continue
        table = tables.get(key)
        if table:
            complete = table.render()
            if len(complete) <= remaining:
                source = replace(source, text=complete, location=table.location, table=table)
                seen.add(key)
            else:
                source = replace(
                    source,
                    text=source.text + "\nPARTIAL TABLE: whole-table aggregates are unavailable.",
                )
        if len(source.text) > remaining:
            continue
        expanded.append(source)
        used[source.document_id] = used.get(source.document_id, 0) + len(source.text)
    return tuple(expanded)


def retrieve_sources(
    *, index_ids, documents, question, top_k, embedder, store, language_model=None, queries=None
):
    if queries is None:
        queries = (
            language_model.retrieval_queries(
                question, [document.filename for document in documents]
            )
            if language_model else [question]
        )
    queries = list(dict.fromkeys([question, *queries]))
    embeddings = list(zip(queries, embedder.embed_queries(queries), strict=True))
    sources = search_library(
        index_ids, embeddings[0][1], top_k, store,
        question=question, query_variants=embeddings[1:],
    )
    return sources


def answer_question(
    *, index_ids, documents, question, history, top_k, embedder, store, language_model,
    visualization="auto",
):
    standalone, queries = language_model.prepare_search(
        question, history, [document.filename for document in documents]
    )
    if standalone.startswith("CLARIFY:"):
        return Answer(standalone.removeprefix("CLARIFY:").strip(), "")
    sources = retrieve_sources(
        index_ids=index_ids, documents=documents, question=standalone, top_k=top_k,
        embedder=embedder, store=store, queries=queries,
    )
    sources = expand_tables(sources, documents)
    text = language_model.answer(
        standalone, sources, original_question=question, visualization=visualization
    )
    return Answer(
        text,
        standalone,
        sources,
        tuple(getattr(language_model, "last_charts", ())),
    )


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
