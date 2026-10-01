"""Small immutable domain objects passed between RAG pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    location: str = ""


@dataclass(frozen=True)
class DocumentTable:
    table_id: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    location: str
    page_number: int = 0
    caption: str = ""

    def render(self) -> str:
        lines = [self.caption, " | ".join(self.headers)]
        lines.extend(f"Row {i}: " + " | ".join(row) for i, row in enumerate(self.rows, 1))
        return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    filename: str
    page_count: int
    pages: tuple[PageText, ...]
    language: str
    language_confidence: float
    character_count: int
    tables: tuple[DocumentTable, ...] = ()
    warnings: tuple[str, ...] = ()
    file_type: str = "pdf"


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: int
    chunk_index: int
    page_chunk_index: int
    token_count: int
    start_index: int
    location: str = ""
    table_id: str = ""
    row_start: int = 0
    row_end: int = 0
    overlap_tokens: int = 0


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    score: float
    document_id: str = ""
    filename: str = ""
    location: str = ""
    table_id: str = ""
    table: DocumentTable | None = None


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True)
class Answer:
    text: str
    standalone_question: str
    sources: tuple[SearchResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IndexSummary:
    index_id: str
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    character_count: int
    language: str
    reused_existing_index: bool
