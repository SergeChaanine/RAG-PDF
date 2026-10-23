"""Route catalogue years, offices and related degree requirements in saved sources."""

import re
from urllib.parse import urlsplit

from rag_pdf.lexical import normalize
from rag_pdf.models import SearchResult


def _program_identity(document):
    if document.file_type != "html" or not document.pages or not document.source_url:
        return None
    url = urlsplit(document.source_url)
    host = (url.hostname or "").casefold()
    if (
        url.scheme not in {"http", "https"}
        or url.username
        or url.password
        or not (host == "usek.edu.lb" or host.endswith(".usek.edu.lb"))
    ):
        return None
    slug = re.sub(r"-\d+$", "", url.path.rstrip("/").rsplit("/", 1)[-1].casefold())
    level = re.match(r"^(bachelor|master|doctor(?:ate)?|minor|(?:engineering-)?diploma)-", slug)
    if not level:
        return None
    subject = slug.rsplit("-in-", 1)[-1] if "-in-" in slug else slug[level.end() :]
    terms = set(re.findall(r"[^\W_]+", normalize(subject))) - {"of", "and", "the"}
    if not terms:
        return None
    return slug, level[1], terms


def _credit_excerpt(document, max_characters=1600):
    """Keep the source's own numbers and nearby entry-level conditions verbatim."""
    candidates = []
    for page in document.pages:
        for match in re.finditer(
            r"\b(?:\d{1,3}\s*(?:credits?|ECTS)|(?:credits?|ECTS)\s*[:=]?\s*\d{1,3})\b",
            page.text,
            re.I,
        ):
            nearby = page.text[max(0, match.start() - 160) : match.end() + 200]
            requirement = bool(re.search(r"\bprogram(?:me)?\s+requires?\b|\btotal\b", nearby, re.I))
            start = max(0, match.start() - 250)
            end = min(len(page.text), start + max_characters)
            if start:
                # Start at a word boundary without inventing a heading or summary.
                boundary = page.text.find(" ", start, match.start())
                if boundary >= 0:
                    start = boundary + 1
            candidates.append((requirement, page, page.text[start:end]))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def program_credit_sources(documents, queries, *, limit=8):
    """Include related degree-page variants even when dense/reranked search misses one."""
    original = " ".join(queries)
    question = normalize(original)
    if not re.search(r"\b(?:credits?|ects)\b", question):
        return []
    # A named course has its own credits, which must not become a degree total.
    if re.search(r"\b(?:[a-zA-Z]{3}-?\d{3}|[A-Z]{3}\s+\d{3})\b", original):
        return []
    terms = set(re.findall(r"[^\W_]+", question))
    levels = {
        name
        for name in ("bachelor", "master", "doctor", "doctorate", "minor", "diploma")
        if re.search(rf"\b{name}s?\b", question)
    }
    if not levels and ("major" in terms or "undergraduate" in terms):
        levels = {"bachelor"}
    candidates = []
    for document in documents:
        identity = _program_identity(document)
        if not identity:
            continue
        family, level, subject = identity
        if level == "minor" and "minor" not in terms:
            continue
        if levels and level not in levels:
            continue
        if not subject <= terms:
            continue
        excerpt = _credit_excerpt(document)
        if excerpt:
            detailed, page, text = excerpt
            candidates.append((len(subject), detailed, family, document, page, text))
    if not candidates:
        return []
    specificity = max(item[0] for item in candidates)
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].source_url))
    sources, seen = [], set()
    for specificity_score, _, _, document, page, text in candidates:
        if specificity_score < specificity:
            continue
        url = urlsplit(document.source_url)
        canonical = (url.hostname.casefold(), url.path.rstrip("/").casefold())
        if canonical in seen:
            continue
        seen.add(canonical)
        sources.append(
            SearchResult(
                "program-" + document.document_id,
                text,
                page.page_number,
                0,
                1.0,
                document_id=document.document_id,
                filename=document.filename,
                location=page.location,
                retrieval_method="program requirement comparison",
                source_url=document.source_url,
                fetched_at=document.fetched_at,
            )
        )
        if len(sources) >= limit:
            break
    return sources


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
