"""Token-aware document chunking with dynamic percentage overlap."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pdf.config import ChunkConfig
from rag_pdf.models import ExtractedDocument, TextChunk


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def chunk_document(
    document: ExtractedDocument,
    tokenizer: Tokenizer,
    config: ChunkConfig,
) -> list[TextChunk]:
    """Split each page while preserving accurate page-level citations."""

    token_length: Callable[[str], int] = lambda text: len(  # noqa: E731
        tokenizer.encode(text, add_special_tokens=False)
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.size_tokens,
        chunk_overlap=config.overlap_tokens,
        length_function=token_length,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        add_start_index=True,
        strip_whitespace=True,
    )
    source_documents = [
        Document(
            page_content=page.text,
            metadata={
                "document_id": document.document_id,
                "filename": document.filename,
                "page_number": page.page_number,
            },
        )
        for page in document.pages
    ]
    split_documents = splitter.split_documents(source_documents)

    chunks: list[TextChunk] = []
    page_counts: dict[int, int] = {}
    for chunk_index, item in enumerate(split_documents):
        page_number = int(item.metadata["page_number"])
        page_chunk_index = page_counts.get(page_number, 0)
        page_counts[page_number] = page_chunk_index + 1
        start_index = int(item.metadata.get("start_index", -1))
        chunk_id = hashlib.sha256(
            (
                f"{document.document_id}:{page_number}:{page_chunk_index}:"
                f"{start_index}:{item.page_content}"
            ).encode()
        ).hexdigest()[:32]
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                filename=document.filename,
                text=item.page_content,
                page_number=page_number,
                chunk_index=chunk_index,
                page_chunk_index=page_chunk_index,
                token_count=token_length(item.page_content),
                start_index=start_index,
            )
        )
    return chunks
