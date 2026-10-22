"""Read frozen crawl exports and index university sources in one collection."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

import portalocker

from rag_pdf.chunking import chunk_document
from rag_pdf.document_processing import make_table
from rag_pdf.errors import ValidationError
from rag_pdf.lexical import KeywordIndex
from rag_pdf.models import ExtractedDocument, PageText, TextChunk
from rag_pdf.service import build_index_id, library_id


@dataclass(frozen=True)
class CrawlDataset:
    path: Path
    fingerprint: str
    documents: tuple[ExtractedDocument, ...]
    issues: tuple[str, ...]
    record_count: int
    crawl_report: dict

    def summary(self):
        return {
            "path": str(self.path),
            "snapshot_sha256": self.fingerprint,
            "input_records": self.record_count,
            "searchable_documents": len(self.documents),
            "source_types": dict(Counter(d.file_type for d in self.documents)),
            "characters": sum(d.character_count for d in self.documents),
            "issues": list(self.issues),
            "documents_with_warnings": sum(bool(d.warnings) for d in self.documents),
            "crawl_report": self.crawl_report,
        }


@dataclass(frozen=True)
class WebsiteIndex:
    index_id: str
    document_count: int
    chunk_count: int
    reused: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class WebsiteChunkPlan:
    chunks: tuple[TextChunk, ...]
    warnings: tuple[str, ...] = ()


def official_source_url(value):
    """Citation URLs must be public USEK HTTP(S) addresses without credentials."""
    if not isinstance(value, str) or any(char.isspace() for char in value):
        raise ValueError("Source URL must be an HTTP(S) USEK URL.")
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    if (
        parts.scheme not in {"https", "http"}
        or parts.username
        or parts.password
        or not (host == "usek.edu.lb" or host.endswith(".usek.edu.lb"))
    ):
        raise ValueError("Source URL must be an HTTP(S) USEK URL.")
    return value


def _html_sections(text):
    sections, title, lines = [], "Webpage", []
    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            if "\n".join(lines).strip():
                sections.append(PageText(0, "\n".join(lines).strip(), title))
            title, lines = heading.group(1), [line]
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append(PageText(0, "\n".join(lines).strip(), title))
    return tuple(sections)


def _record_document(record, root, reextract_html):
    if not isinstance(record, dict):
        raise ValueError("Record must be a JSON object.")
    url = official_source_url(record.get("final_url") or record.get("source_url", ""))
    source_type = record.get("source_type")
    if source_type not in {"html", "pdf"}:
        raise ValueError("Only HTML and PDF crawl records are supported.")
    if record.get("text_available") is False:
        raise ValueError("No native text was extracted; OCR or manual review is needed.")
    warnings = [str(value) for value in record.get("warnings", [])]
    text = record.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Extracted text must be a string.")
    if source_type == "html" and reextract_html and record.get("raw_path"):
        relative = Path(record["raw_path"])
        snapshot = (root / relative).resolve()
        if relative.is_absolute() or not snapshot.is_relative_to(root.resolve()):
            raise ValueError("HTML snapshot path escapes the crawl directory.")
        if snapshot.is_file():
            try:
                from rag_pdf.crawl_core import extract_html

                # Crawler snapshots are UTF-8, independently of the original page charset.
                parsed, _ = extract_html(snapshot.read_text("utf-8"), url)
                if parsed.get("restricted"):
                    raise ValueError("Login page cannot be used as public university evidence.")
                text = parsed["text"]
                warnings = parsed["warnings"]
            except ImportError:
                warnings.append("Install requirements-crawl.txt to clean the saved HTML again.")
        else:
            warnings.append("Saved HTML snapshot is missing; using the existing extracted text.")
    tables = []
    if source_type == "html":
        pages = _html_sections(text)
        page_count = 0
    else:
        pages, seen = [], set()
        for page in record.get("pages", []):
            number, body = page.get("page_number"), page.get("text", "")
            if (
                not isinstance(number, int)
                or isinstance(number, bool)
                or number < 1
                or number in seen
            ):
                raise ValueError("PDF page numbers must be distinct positive integers.")
            if not isinstance(body, str):
                raise ValueError("PDF page text must be a string.")
            seen.add(number)
            if body.strip():
                pages.append(PageText(number, body, f"p. {number}"))
        if not pages:
            raise ValueError("PDF has no native page text with verifiable page numbers.")
        pages = tuple(pages)
        page_count = max(record.get("page_count", 0), max(seen))
        for i, table in enumerate(record.get("tables", []), 1):
            number = table.get("page_number")
            if number not in seen:
                warnings.append(f"Table {i}: page number is unavailable; table omitted.")
                continue
            parsed = make_table(
                f"web-p{number}-t{i}",
                table.get("rows", []),
                f"p. {number}, Table {i}",
                page_number=number,
            )
            if parsed:
                tables.append(parsed)
    if not pages:
        raise ValueError("No usable text remains after webpage cleanup.")
    title = str(record.get("title") or url).strip()
    filename = title
    if source_type == "pdf":
        filename += f" ({Path(urlsplit(url).path).name or 'document.pdf'})"
    fetched_at = str(record.get("fetched_at") or "")
    identity = json.dumps(
        [
            url,
            filename,
            fetched_at,
            [(p.page_number, p.location, p.text) for p in pages],
            [(t.table_id, t.headers, t.rows) for t in tables],
        ],
        ensure_ascii=False,
    )
    return ExtractedDocument(
        hashlib.sha256(identity.encode()).hexdigest(),
        filename,
        page_count,
        tuple(pages),
        str(record.get("language") or "und"),
        0.0,
        sum(len(page.text) for page in pages) + sum(len(t.render()) for t in tables),
        tuple(tables),
        tuple(warnings),
        source_type,
        url,
        fetched_at,
    )


def load_crawl_dataset(path, *, limit=0, reextract_html=True, progress=None):
    """Read one atomic JSONL export without touching the running crawler's database."""
    path = Path(path).expanduser()
    if path.is_dir() or path.suffix.lower() != ".jsonl":
        path /= "documents.jsonl"
    if limit < 0:
        raise ValueError("Document limit cannot be negative.")
    if not path.is_file():
        raise ValueError(f"No documents.jsonl export found at {path}. Wait for a crawl checkpoint.")
    fingerprint = hashlib.sha256()
    documents, issues, seen, count = [], [], set(), 0
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            fingerprint.update(raw_line)
            line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
            if not line.strip():
                continue
            count += 1
            if limit and len(documents) >= limit:
                continue
            try:
                document = _record_document(json.loads(line), path.parent, reextract_html)
                if document.document_id not in seen:
                    documents.append(document)
                    seen.add(document.document_id)
            except Exception as exc:
                # A malformed individual extraction is visible without losing the corpus.
                issues.append(f"Line {line_number}: {exc}")
            if progress and count % 50 == 0:
                progress(count, len(documents))
    if progress:
        progress(count, len(documents))
    report_path = path.parent / "report.json"
    report = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text("utf-8"))
        except (ValueError, OSError):
            issues.append(
                "Crawl report could not be read; content export was loaded independently."
            )
    return CrawlDataset(
        path.resolve(),
        fingerprint.hexdigest(),
        tuple(documents),
        tuple(issues),
        count,
        report,
    )


def plan_website_chunks(dataset, chunk_config, tokenizer, *, progress=None):
    """Compute and retain the exact chunk plan separately from vector generation."""
    documents = dataset.documents
    if not documents:
        raise ValueError("The crawl export contains no searchable documents.")
    chunks, warnings = [], []
    for number, document in enumerate(documents, 1):
        try:
            document_chunks = chunk_document(document, tokenizer, chunk_config)
        except ValidationError:
            if not document.tables:
                raise
            # Native PDF page text still contains the original cell values. Avoid
            # discarding an entire guide just because one table row is too wide.
            document_chunks = chunk_document(replace(document, tables=()), tokenizer, chunk_config)
            warnings.append(
                f"{document.filename}: wide table indexed as native page text; "
                "table calculations are unavailable for that source."
            )
        chunks.extend(document_chunks)
        if progress and (number % 50 == 0 or number == len(documents)):
            progress(number, len(documents), len(chunks))
    if not chunks:
        raise ValueError("No searchable chunks were produced.")
    return WebsiteChunkPlan(tuple(chunks), tuple(warnings))


def index_website(
    dataset,
    chunk_config,
    embedder,
    store,
    *,
    progress=None,
    chunk_plan=None,
    vector_cache=None,
    write_batch_size=None,
):
    """Build one collection; a persistent cache can resume expensive embeddings."""
    if vector_cache is not None and vector_cache.model_name != embedder.model_name:
        raise ValueError("The embedding cache belongs to a different model.")
    batch_size = write_batch_size or embedder.batch_size
    if batch_size < 1:
        raise ValueError("Index write batch size must be positive.")
    documents = dataset.documents
    plan = chunk_plan or plan_website_chunks(dataset, chunk_config, embedder.tokenizer)
    chunks, warnings = plan.chunks, plan.warnings
    index_id = build_index_id(
        library_id(documents),
        embedder.model_name,
        chunk_config,
        filename="USEK shared website collection",
        tokenizer_id="model",
    )
    directory = store.path / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / f"{index_id}.json"
    expected = [chunk.chunk_id for chunk in chunks]
    with portalocker.Lock(str(manifest) + ".lock", timeout=120):
        try:
            reused = json.loads(manifest.read_text("utf-8")) == expected
        except (ValueError, FileNotFoundError):
            reused = False
        reused = reused and store.count(index_id) == len(chunks)
        if not reused:
            manifest.unlink(missing_ok=True)
            store.delete(index_id)
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                texts = [chunk.text for chunk in batch]
                if vector_cache is None:
                    vectors = embedder.embed_documents(texts)
                else:
                    vectors = vector_cache.get_many(texts)
                    missing = list(
                        dict.fromkeys(
                            text
                            for text, vector in zip(texts, vectors, strict=True)
                            if vector is None
                        )
                    )
                    if missing:
                        fresh = embedder.embed_documents(missing)
                        vector_cache.put_many(missing, fresh)
                        recovered = dict(zip(missing, fresh, strict=True))
                        vectors = [
                            recovered[text] if vector is None else vector
                            for text, vector in zip(texts, vectors, strict=True)
                        ]
                store.add(index_id, documents[0], batch, vectors)
                if progress:
                    progress(min(start + len(batch), len(chunks)), len(chunks))
        if store.count(index_id) != len(chunks):
            raise ValueError("Vector index is incomplete; rerun preparation to resume.")
        KeywordIndex(store.path / "keywords.sqlite3").ensure_index(index_id, chunks)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected), encoding="utf-8")
        temporary.replace(manifest)
    return WebsiteIndex(index_id, len(documents), len(chunks), reused, tuple(warnings))
