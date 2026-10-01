"""Token-bounded prose chunks and table row groups with repeated headers."""

from __future__ import annotations

import hashlib
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pdf.config import ChunkConfig
from rag_pdf.errors import ValidationError
from rag_pdf.models import ExtractedDocument, TextChunk


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def chunk_document(
    document: ExtractedDocument, tokenizer: Tokenizer, config: ChunkConfig
) -> list[TextChunk]:
    def length(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.size_tokens,
        chunk_overlap=config.overlap_tokens,
        length_function=length,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        add_start_index=True,
    )
    chunks = []

    def append(text, page, location, start=-1, table_id="", row_start=0, row_end=0, overlap=0):
        index = len(chunks)
        identity = f"{document.document_id}:{index}:{location}:{text}"
        chunks.append(
            TextChunk(
                hashlib.sha256(identity.encode()).hexdigest()[:32],
                document.document_id,
                document.filename,
                text,
                page,
                index,
                index,
                length(text),
                start,
                location,
                table_id,
                row_start,
                row_end,
                overlap,
            )
        )

    groups = []
    for page in document.pages:
        section = page.location.split(", paragraph ")[0]
        if document.file_type == "docx" and groups and groups[-1][3] == section:
            groups[-1][0] += "\n\n" + page.text
            groups[-1][2] = groups[-1][4] + " through " + page.location
        else:
            groups.append(
                [
                    page.text,
                    page.page_number,
                    page.location or f"p. {page.page_number}",
                    section,
                    page.location,
                ]
            )
    for text, page, location, _, _first in groups:
        previous_end = 0
        for item in splitter.create_documents([text]):
            start = item.metadata.get("start_index", 0)
            overlap = length(text[start:previous_end]) if start < previous_end else 0
            append(item.page_content, page, location, start, overlap=overlap)
            previous_end = start + len(item.page_content)

    for table in document.tables:
        prefix = table.caption[:160] + "\n" if table.caption else ""
        prefix += " | ".join(table.headers) + "\n"
        rows = [f"Row {i}: " + " | ".join(row) for i, row in enumerate(table.rows, 1)]
        for row in rows:
            if length(prefix + row) > config.size_tokens:
                raise ValidationError(
                    f"{document.filename}, {table.location}: one table row and its headers "
                    f"exceed {config.size_tokens} tokens. Increase the chunk size."
                )
        start, previous_end = 0, 0
        while start < len(rows):
            end = start + 1
            while (
                end < len(rows)
                and length(prefix + "\n".join(rows[start : end + 1])) <= config.size_tokens
            ):
                end += 1
            overlap = length("\n".join(rows[start:previous_end])) if start < previous_end else 0
            append(
                prefix + "\n".join(rows[start:end]),
                table.page_number,
                f"{table.location}, rows {start + 1}-{end}",
                table_id=table.table_id,
                row_start=start + 1,
                row_end=end,
                overlap=overlap,
            )
            if end == len(rows):
                break
            next_start = end
            while (
                next_start > start + 1
                and length("\n".join(rows[next_start - 1 : end])) <= config.overlap_tokens
            ):
                next_start -= 1
            previous_end, start = end, next_start
    return chunks
