"""Prepare a durable website search package before a live demonstration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import portalocker

from rag_pdf.catalog import PIPELINE_VERSION
from rag_pdf.config import ChunkConfig
from rag_pdf.embedding_cache import EmbeddingCache
from rag_pdf.models import DocumentTable, ExtractedDocument, PageText, TextChunk
from rag_pdf.vector_store import create_store
from rag_pdf.website import (
    CrawlDataset,
    WebsiteChunkPlan,
    WebsiteIndex,
    index_website,
    load_crawl_dataset,
    plan_website_chunks,
)

FORMAT_VERSION = 1


@dataclass(frozen=True)
class PreparedWebsite:
    path: Path
    dataset: CrawlDataset
    index: WebsiteIndex
    manifest: dict
    store: object


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_records(path, records):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    temporary.replace(path)
    return file_hash(path)


def _read_records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _documents(path):
    documents = []
    for record in _read_records(path):
        record["pages"] = tuple(PageText(**page) for page in record["pages"])
        record["tables"] = tuple(
            DocumentTable(
                **{
                    **table,
                    "headers": tuple(table["headers"]),
                    "rows": tuple(tuple(row) for row in table["rows"]),
                }
            )
            for table in record["tables"]
        )
        record["warnings"] = tuple(record["warnings"])
        documents.append(ExtractedDocument(**record))
    return tuple(documents)


def _dataset(root, state):
    metadata = state["dataset"]
    return CrawlDataset(
        Path(metadata["path"]),
        metadata["snapshot_sha256"],
        _documents(root / "documents.jsonl"),
        tuple(metadata["issues"]),
        metadata["input_records"],
        metadata["crawl_report"],
    )


def active_presentation(data_dir):
    """Return only a completed package selected by the preparation command."""
    pointer = Path(data_dir) / "presentations" / "active.json"
    try:
        path = Path(json.loads(pointer.read_text("utf-8"))["path"])
        if (path / "manifest.json").is_file():
            return path
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def load_prepared_website(path):
    """Load frozen documents and indexes; never extract, chunk, or embed sources."""
    root = Path(path).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        if (
            manifest.get("state") != "ready"
            or manifest.get("format_version") != FORMAT_VERSION
            or manifest.get("pipeline_version") != PIPELINE_VERSION
        ):
            raise ValueError("Presentation package has an incompatible format or pipeline.")
        for name in ("documents.jsonl", "chunks.jsonl"):
            if file_hash(root / name) != manifest["files"][name]:
                raise ValueError(f"Presentation file failed its integrity check: {name}")
        dataset = _dataset(root, manifest)
        if len(dataset.documents) != manifest["document_count"]:
            raise ValueError("Presentation source count is inconsistent.")
        backend_path = root / "indexes" / manifest["backend"].lower()
        if not backend_path.is_dir() or not any(backend_path.iterdir()):
            raise ValueError("The saved vector database is missing.")
        completion = backend_path / "manifests" / f"{manifest['index_id']}.json"
        if file_hash(completion) != manifest["completion_sha256"]:
            raise ValueError("The vector completion record has changed.")
        keywords = backend_path / "keywords.sqlite3"
        with closing(sqlite3.connect(keywords.as_uri() + "?mode=ro", uri=True)) as db:
            count = db.execute(
                "SELECT count(*) FROM chunks WHERE index_id = ?", (manifest["index_id"],)
            ).fetchone()[0]
        if count != manifest["chunk_count"]:
            raise ValueError("The saved keyword index is incomplete.")
        store = create_store(manifest["backend"], root / "indexes")
        if store.count(manifest["index_id"]) != manifest["chunk_count"]:
            if hasattr(store, "close"):
                store.close()
            raise ValueError("The saved vector index is incomplete.")
    except (OSError, KeyError, json.JSONDecodeError, sqlite3.Error) as exc:
        raise ValueError(f"Presentation package is missing or damaged: {exc}") from exc
    return PreparedWebsite(
        root,
        dataset,
        WebsiteIndex(
            manifest["index_id"],
            manifest["document_count"],
            manifest["chunk_count"],
            True,
            tuple(manifest["warnings"]),
        ),
        manifest,
        store,
    )


def prepare_website(
    crawl_path,
    output,
    embedder,
    *,
    chunk_config=None,
    backend="Chroma",
    limit=0,
    reextract_html=True,
    data_dir=None,
    progress=None,
    write_batch_size=256,
):
    """Resume checkpoints and publish the active pointer only after every index is ready."""
    config = chunk_config or ChunkConfig()
    source = Path(crawl_path).resolve()
    if source.is_dir():
        source /= "documents.jsonl"
    root = Path(output).resolve()
    if source.is_relative_to(root):
        raise ValueError("Use a separate output directory outside the original crawl.")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_notification = 0.0

    def report(stage, **counters):
        nonlocal last_notification
        now = time.perf_counter()
        if now - last_notification < 5 and counters.get("done") != counters.get("total"):
            return
        value = {
            "stage": stage,
            "updated_at": _now(),
            "elapsed_seconds": round(now - started, 1),
            **counters,
        }
        _json(root / "progress.json", value)
        if progress:
            progress(value)
        last_notification = now

    parameters = {
        "format_version": FORMAT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source": str(source),
        "source_sha256": file_hash(source),
        "embedding_model": embedder.model_name,
        "backend": backend,
        "chunk_config": asdict(config),
        "limit": limit,
        "reextract_html": reextract_html,
    }
    with portalocker.Lock(str(root / "prepare.lock"), timeout=1):
        state_path = root / "build_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text("utf-8"))
            if state["parameters"] != parameters:
                raise ValueError("Build settings or source changed. Use a new output directory.")
        else:
            state = {"parameters": parameters, "files": {}}
            _json(state_path, state)
        if (root / "manifest.json").exists():
            prepared = load_prepared_website(root)
            if data_dir:
                pointer = Path(data_dir) / "presentations" / "active.json"
                pointer.parent.mkdir(parents=True, exist_ok=True)
                _json(pointer, {"path": str(root), "updated_at": _now()})
            report("ready", done=prepared.index.chunk_count, total=prepared.index.chunk_count)
            return prepared
        if "documents.jsonl" in state["files"]:
            if file_hash(root / "documents.jsonl") != state["files"]["documents.jsonl"]:
                raise ValueError(
                    "Saved document checkpoint is damaged. Use a new output directory."
                )
            report("loading_saved_documents")
            dataset = _dataset(root, state)
        else:
            report("cleaning_sources", done=0, total=None)
            dataset = load_crawl_dataset(
                source,
                limit=limit,
                reextract_html=reextract_html,
                progress=lambda scanned, usable: report(
                    "cleaning_sources", done=scanned, total=None, usable_sources=usable
                ),
            )
            if dataset.fingerprint != parameters["source_sha256"]:
                raise ValueError("Crawl changed during preparation. Wait for the crawl to finish.")
            state["dataset"] = dataset.summary()
            state["files"]["documents.jsonl"] = _write_records(
                root / "documents.jsonl", dataset.documents
            )
            _json(state_path, state)
        if "chunks.jsonl" in state["files"]:
            if file_hash(root / "chunks.jsonl") != state["files"]["chunks.jsonl"]:
                raise ValueError("Saved chunk checkpoint is damaged. Use a new output directory.")
            report("loading_saved_chunks")
            plan = WebsiteChunkPlan(
                tuple(TextChunk(**record) for record in _read_records(root / "chunks.jsonl")),
                tuple(state["warnings"]),
            )
        else:
            report("chunking", done=0, total=len(dataset.documents))
            plan = plan_website_chunks(
                dataset,
                config,
                embedder.tokenizer,
                progress=lambda done, total, count: report(
                    "chunking", done=done, total=total, chunks=count
                ),
            )
            state["files"]["chunks.jsonl"] = _write_records(root / "chunks.jsonl", plan.chunks)
            state["warnings"] = list(plan.warnings)
            _json(state_path, state)
        store = create_store(backend, root / "indexes")
        report(
            "embedding",
            done=0,
            total=len(plan.chunks),
            device=getattr(embedder, "device_label", "test"),
        )
        cache = EmbeddingCache(root / "embedding_cache.sqlite3", embedder.model_name)
        try:
            index = index_website(
                dataset,
                config,
                embedder,
                store,
                chunk_plan=plan,
                vector_cache=cache,
                write_batch_size=write_batch_size,
                progress=lambda done, total: report("embedding", done=done, total=total),
            )
        finally:
            cache.close()
        report("warming_up")
        warm_started = time.perf_counter()
        vector = embedder.embed_queries(["How do I apply for admission and financial aid?"])[0]
        hits = store.search(index.index_id, vector, 3)
        if not hits or not all(hit.source_url for hit in hits):
            raise ValueError("Prepared index failed its retrieval warm-up.")
        manifest = {
            "state": "ready",
            "format_version": FORMAT_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "created_at": _now(),
            "parameters": parameters,
            "embedding_model": embedder.model_name,
            "backend": backend,
            "chunk_config": asdict(config),
            "index_id": index.index_id,
            "document_count": index.document_count,
            "chunk_count": index.chunk_count,
            "warnings": list(index.warnings),
            "dataset": state["dataset"],
            "files": state["files"],
            "completion_sha256": file_hash(store.path / "manifests" / f"{index.index_id}.json"),
            "device": getattr(embedder, "device_label", "test"),
            "warmup_seconds": round(time.perf_counter() - warm_started, 3),
            "preparation_seconds_this_run": round(time.perf_counter() - started, 1),
        }
        # This manifest is the commit point: incomplete packages cannot be opened live.
        _json(root / "manifest.json", manifest)
        if data_dir:
            pointer = Path(data_dir) / "presentations" / "active.json"
            pointer.parent.mkdir(parents=True, exist_ok=True)
            _json(pointer, {"path": str(root), "updated_at": _now()})
        report("ready", done=index.chunk_count, total=index.chunk_count)
        return PreparedWebsite(root, dataset, index, manifest, store)
