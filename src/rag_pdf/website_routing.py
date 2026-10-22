"""Route explicit catalogue years and responsible offices within the saved corpus."""

import re
from urllib.parse import urlsplit

from rag_pdf.models import SearchResult


def catalogue_scope(documents, queries):
    if not any(doc.source_url for doc in documents):
        return None
    question = " ".join(queries).casefold()
    year = re.search(r"\b(20\d{2})\s*[-–—/]\s*(20\d{2})\b", question)
    if not year or not re.search(r"catalog(?:ue)?", question):
        return None
    scoped = []
    for doc in documents:
        if doc.file_type != "pdf" or not doc.source_url:
            continue
        cover = "\n".join(page.text for page in doc.pages[:2])
        if "catalog" not in (doc.filename + " " + cover[:500]).casefold():
            continue
        # Native PDF text sometimes substitutes a replacement glyph for the dash.
        # The year comes from the cover text, never a download date or URL timestamp.
        if re.search(rf"\b{year[1]}\D{{1,8}}{year[2]}\b", cover):
            scoped.append(doc.document_id)
    return tuple(scoped)


def financial_aid_office_source(documents, queries):
    if "financial aid" not in " ".join(queries).casefold():
        return None
    for doc in documents:
        if (
            doc.file_type == "html"
            and doc.pages
            and urlsplit(doc.source_url).path.rstrip("/").casefold()
            == "/student-service-offices/financial-aid-office"
        ):
            page = doc.pages[0]
            return SearchResult(
                "office-" + doc.document_id,
                page.text[:1500],
                0,
                0,
                1.0,
                document_id=doc.document_id,
                filename=doc.filename,
                location=page.location,
                retrieval_method="responsible office page",
                source_url=doc.source_url,
                fetched_at=doc.fetched_at,
            )
    return None
