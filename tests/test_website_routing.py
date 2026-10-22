"""Year/office routing uses saved source evidence without another crawl."""

from dataclasses import replace

import pytest
from test_pipeline import FakeEmbedder
from test_website import html_record

from rag_pdf.config import ChunkConfig
from rag_pdf.lexical import KeywordIndex
from rag_pdf.models import ExtractedDocument, PageText
from rag_pdf.service import retrieve_sources
from rag_pdf.vector_store import create_store
from rag_pdf.website import CrawlDataset, index_website
from rag_pdf.website_routing import catalogue_scope


def test_catalogue_routing_does_not_scope_uploaded_documents():
    uploaded = ExtractedDocument(
        "upload",
        "Catalogue.pdf",
        1,
        (PageText(1, "University Catalog 2025-2026", "p. 1"),),
        "en",
        1.0,
        30,
    )
    assert catalogue_scope([uploaded], ["2025-2026 catalogue registration"]) is None


@pytest.mark.parametrize("backend", ["Chroma", "Qdrant", "LanceDB"])
def test_catalogue_year_filters_both_dense_and_keyword_candidates(tmp_path, backend):
    old = ExtractedDocument(
        "old",
        "University Catalogue",
        1,
        (
            PageText(
                1, "University Catalog 2020 - 2021\nRegistration internship procedure.", "p. 1"
            ),
        ),
        "en",
        1.0,
        100,
        source_url="https://www.usek.edu.lb/test-fixture-old.pdf",
    )
    current = replace(
        old,
        document_id="current",
        source_url="https://www.usek.edu.lb/fixture.pdf",
        pages=(
            PageText(1, "University Catalog 2025 � 2026\nRegistration internship steps.", "p. 1"),
        ),
    )
    dataset = CrawlDataset(tmp_path, "test", (old, current), (), 2, {})
    store = create_store(backend, tmp_path / "indexes")
    summary = index_website(dataset, ChunkConfig(), FakeEmbedder(), store)
    hits = retrieve_sources(
        index_ids=[summary.index_id],
        documents=dataset.documents,
        question="According to the 2025-2026 catalogue, how does registration work?",
        top_k=3,
        embedder=FakeEmbedder(),
        store=store,
        document_diversity=False,
    )
    assert hits and all(hit.document_id == "current" for hit in hits)
    keyword = KeywordIndex(store.path / "keywords.sqlite3").search(
        [summary.index_id], "registration", 10, document_ids=["current"]
    )
    assert keyword and all(hit.document_id == "current" for hit in keyword)
    assert catalogue_scope((old, current), ["2027-2028 catalogue registration"]) == ()
    if hasattr(store, "close"):
        store.close()


def test_financial_aid_question_includes_complete_responsible_office_page(tmp_path):
    from test_website import export

    from rag_pdf.website import load_crawl_dataset

    record = {
        **html_record(),
        "final_url": "https://www.usek.edu.lb/student-service-offices/financial-aid-office",
        "text": "Step 1: Request the certificate.\nStep 2: Complete the aid form.",
    }
    dataset = load_crawl_dataset(export(tmp_path / "crawl", [record]))
    store = create_store("Chroma", tmp_path / "indexes")
    summary = index_website(dataset, ChunkConfig(), FakeEmbedder(), store)
    hits = retrieve_sources(
        index_ids=[summary.index_id],
        documents=dataset.documents,
        question="How do I apply for financial aid?",
        top_k=3,
        embedder=FakeEmbedder(),
        store=store,
        document_diversity=False,
    )
    assert hits[0].retrieval_method == "responsible office page"
    assert "Step 1" in hits[0].text and "Step 2" in hits[0].text
