"""Small immutable domain objects passed between RAG pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    filename: str
    page_count: int
    pages: tuple[PageText, ...]
    language: str
    language_confidence: float
    character_count: int


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


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    score: float


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
