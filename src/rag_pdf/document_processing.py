"""Native PDF and DOCX extraction, retaining tables and honest source locations."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from rag_pdf.errors import ValidationError
from rag_pdf.models import DocumentTable, ExtractedDocument, PageText
from rag_pdf.pdf_processing import document_hash

MAX_BYTES = 20 * 1024 * 1024


def make_table(table_id, data, location, page_number=0, caption="", headers=None):
    rows = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in data]
    rows = [row for row in rows if any(row)]
    if not rows:
        return None
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if headers is None:
        headers, rows = rows[0], rows[1:]
    headers = [str(h or f"Column {i + 1}") for i, h in enumerate(headers)]
    if not rows:
        return None
    return DocumentTable(
        table_id, tuple(headers), tuple(map(tuple, rows)), location, page_number, caption
    )


def extract_document(data: bytes, filename: str) -> ExtractedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise ValidationError("Upload a PDF or Word .docx document.")
    if not data or len(data) > MAX_BYTES:
        raise ValidationError("Documents must be nonempty and at most 20 MB.")
    try:
        pages, tables, warnings, page_count = _pdf(data) if suffix == ".pdf" else _docx(data)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Could not read {Path(filename).name}: {exc}") from exc
    count = sum(len(p.text) for p in pages) + sum(len(t.render()) for t in tables)
    if not count:
        raise ValidationError("No native text or tables found. Scans are not supported.")
    # Numeric tables and short documents must not fail an English-language heuristic.
    return ExtractedDocument(
        document_hash(data),
        Path(filename).name,
        page_count,
        tuple(pages),
        "und",
        0.0,
        count,
        tuple(tables),
        tuple(warnings),
        suffix[1:],
    )


def _pdf(data):
    import pymupdf

    if b"%PDF-" not in data[:1024]:
        raise ValidationError("This file is not a PDF.")
    pages, tables, warnings = [], [], []
    with pymupdf.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise ValidationError("Password-protected PDFs are not supported.")
        if document.page_count > 200:
            raise ValidationError("PDFs are limited to 200 pages.")
        for number, page in enumerate(document, 1):
            bounds = []
            try:
                found = page.find_tables()
                for table in found.tables:
                    table_id = f"p{number}-t{len(tables) + 1}"
                    surrounding = page.get_text(
                        clip=pymupdf.Rect(
                            0, max(0, table.bbox[1] - 45), page.rect.width, table.bbox[1]
                        )
                    ).strip()
                    external = table.header.external
                    parsed = make_table(
                        table_id,
                        table.extract(),
                        f"p. {number}, Table {len(tables) + 1}",
                        number,
                        surrounding,
                        headers=table.header.names if external else None,
                    )
                    if parsed:
                        tables.append(parsed)
                        bounds.append(pymupdf.Rect(table.bbox))
            except Exception as exc:
                warnings.append(
                    f"Page {number}: table detection failed ({type(exc).__name__}); "
                    "check the text preview."
                )
            # Exclude detected cells from prose so table values are not indexed twice.
            lines = []
            for block in page.get_text("dict", sort=True)["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(
                        span["text"]
                        for span in line["spans"]
                        if not any(
                            box.contains(
                                pymupdf.Rect(span["bbox"]).tl
                                + (pymupdf.Rect(span["bbox"]).br - pymupdf.Rect(span["bbox"]).tl)
                                / 2
                            )
                            for box in bounds
                        )
                    ).strip()
                    if text:
                        lines.append(text)
            text = "\n".join(lines)
            if text:
                pages.append(PageText(number, text, f"p. {number}"))
            elif not bounds:
                warnings.append(f"Page {number}: no native text; image content was not read.")
        page_count = document.page_count
    return pages, tables, warnings, page_count


def _docx(data):
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 100 * 1024 * 1024:
            raise ValidationError("The expanded Word document exceeds 100 MB.")
    document = Document(io.BytesIO(data))
    pages, tables, warnings = [], [], []
    heading, previous, paragraph_number = "Document", "", 0

    def visit(container):
        nonlocal heading, previous, paragraph_number
        for item in container.iter_inner_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                paragraph_number += 1
                if item.style and item.style.name.startswith("Heading"):
                    heading = text
                pages.append(PageText(0, text, f"{heading}, paragraph {paragraph_number}"))
                previous = text
            elif isinstance(item, Table):
                number = len(tables) + 1
                parsed = make_table(
                    f"t{number}",
                    [[cell.text for cell in row.cells] for row in item.rows],
                    f"{heading}, Table {number}",
                    caption=previous,
                )
                if parsed:
                    tables.append(parsed)
                seen = set()
                for row in item.rows:
                    for cell in row.cells:
                        if cell._tc not in seen:
                            seen.add(cell._tc)
                            for nested in cell.tables:
                                number = len(tables) + 1
                                parsed = make_table(
                                    f"t{number}",
                                    [[c.text for c in r.cells] for r in nested.rows],
                                    f"{heading}, nested Table {number}",
                                )
                                if parsed:
                                    tables.append(parsed)

    visit(document)
    if document.inline_shapes:
        warnings.append("Embedded images/charts were not read; native text and tables were read.")
    if re.search(rb"<w:(ins|del)\b", document.part.blob):
        warnings.append("Tracked changes exist; accept/reject revisions in Word before indexing.")
    return pages, tables, warnings, 0
