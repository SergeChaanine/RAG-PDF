"""Reranking and source context checks without downloading any model."""

from dataclasses import replace

import pytest

from rag_pdf.models import ExtractedDocument, PageText, SearchResult
from rag_pdf.service import expand_website_sections, search_library


@pytest.mark.parametrize(
    "question,variants",
    [
        ("How do I register?", ()),
        ("كيف أسجل؟", (("How do I register?", [1.0, 0.0]),)),
    ],
)
def test_reranker_selects_procedure_over_high_scoring_calendar(tmp_path, question, variants):
    calendar = SearchResult(
        "calendar",
        "Registration dates 2025-2026",
        1,
        0,
        0.95,
        document_id="calendar",
        filename="Catalogue",
    )
    procedure = SearchResult(
        "procedure",
        "Register through Banner after paying the installment.",
        57,
        0,
        0.7,
        document_id="guide",
        filename="Catalogue",
    )

    class Store:
        path = tmp_path

        def search(self, *args):
            return [calendar, procedure]

    class Reranker:
        english_only = True

        def rank(self, question, candidates):
            assert question == "How do I register?"
            assert {item.chunk_id for item in candidates} == {"calendar", "procedure"}
            return [(procedure, 4.0), (calendar, -2.0)]

    results = search_library(
        ["index"],
        [1.0, 0.0],
        1,
        Store(),
        question=question,
        query_variants=variants,
        document_diversity=False,
        reranker=Reranker(),
    )
    assert results[0].chunk_id == "procedure"
    assert "reranked" in results[0].retrieval_method


def test_office_context_retains_complete_steps_and_deduplicates_sections():
    text = "Step 1: Request the certificate.\nStep 2: Fill the form.\nStep 3: Book an appointment."
    document = ExtractedDocument(
        "fao",
        "Financial Aid Office",
        0,
        (PageText(0, text, "Procedure"),),
        "en",
        1.0,
        len(text),
        file_type="html",
        source_url="https://www.usek.edu.lb/test-fixture",
    )
    hit = SearchResult(
        "step2",
        "Step 2: Fill the form.",
        0,
        1,
        0.8,
        document_id="fao",
        location="Procedure",
        source_url=document.source_url,
    )
    next_hit = replace(hit, chunk_id="step3", text="Step 3: Book an appointment.")
    sources = expand_website_sections([hit, next_hit], [document])
    assert len(sources) == 1
    assert sources[0].text == text and sources[0].source_url == document.source_url
    assert expand_website_sections([hit], [document], max_characters=10)[0].text == hit.text


def test_pdf_page_expansion_keeps_month_heading_with_registration_dates():
    text = "January 2026\n26-30: Add/Drop and late registration."
    doc = ExtractedDocument(
        "calendar",
        "Academic Calendar",
        38,
        (PageText(38, text, "p. 38"),),
        "en",
        1.0,
        len(text),
        source_url="https://www.usek.edu.lb/test-fixture.pdf",
    )
    hit = SearchResult(
        "dates",
        "26-30: Add/Drop and late registration.",
        38,
        1,
        0.8,
        document_id="calendar",
        location="p. 38",
        source_url=doc.source_url,
    )
    expanded = expand_website_sections([hit], [doc])
    assert expanded[0].text.startswith("January 2026")
    assert expanded[0].page_number == 38 and expanded[0].location == "p. 38"
