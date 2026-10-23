"""Conflicting degree requirements must survive retrieval before the model answers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_llm import model_with_client, response
from test_pipeline import FakeEmbedder

from rag_pdf.models import ExtractedDocument, PageText, SearchResult
from rag_pdf.service import answer_question, retrieve_sources
from rag_pdf.website_routing import program_credit_sources


def program(slug, total, *, variant="2", detailed=True):
    name = slug.replace("-", " ").title()
    content = (
        "Learning outcomes.\n\n" * 100
        + f"This program requires {total} credits for students entering at the sophomore level."
        if detailed
        else f"Bachelor of Engineering in {name}\nMultilingual\n{total}\ncredits\n"
        "For students entering at the Sophomore level."
    )
    return ExtractedDocument(
        slug + variant,
        "Bachelor of Engineering in " + name,
        0,
        (PageText(0, content, "Webpage"),),
        "en",
        1.0,
        len(content),
        file_type="html",
        source_url=f"https://www.usek.edu.lb/en/engineering/"
        f"bachelor-of-engineering-in-{slug}-{variant}",
        fetched_at="2026-10-21" if detailed else "2026-10-22",
    )


def stored_hit(doc):
    return SearchResult(
        "legacy-hit",
        doc.pages[0].text,
        0,
        0,
        1.0,
        document_id=doc.document_id,
        filename=doc.filename,
        location="Webpage",
        source_url=doc.source_url,
    )


@pytest.mark.parametrize(
    "subject,current_total,other_total",
    [("computer-engineering", 146, 183), ("mechanical-engineering", 131, 160)],
)
def test_degree_variants_survive_even_when_search_returns_only_one(
    tmp_path,
    subject,
    current_total,
    other_total,
):
    detailed = program(subject, current_total)
    other = program(subject, other_total, variant="1", detailed=False)
    store = SimpleNamespace(path=tmp_path, search=lambda *args, **kwargs: [stored_hit(other)])
    hits = retrieve_sources(
        index_ids=["saved-index"],
        documents=(other, detailed),
        question=f"How many total credits for the major {subject.replace('-', ' ')}?",
        top_k=2,
        embedder=FakeEmbedder(),
        store=store,
        document_diversity=False,
    )
    assert [hit.document_id for hit in hits] == [detailed.document_id, other.document_id]
    assert str(current_total) in hits[0].text and str(other_total) in hits[1].text
    assert "sophomore" in hits[0].text.lower() and "Multilingual" in hits[1].text
    assert hits[0].text in detailed.pages[0].text
    assert all(hit.source_url for hit in hits)


def test_program_matching_ignores_other_subjects_degrees_and_tracking_duplicates():
    detailed = program("computer-engineering", 146)
    other = program("computer-engineering", 183, variant="1", detailed=False)
    duplicate = replace(detailed, document_id="duplicate", source_url=detailed.source_url + "?t=1")
    master = replace(
        detailed,
        document_id="master",
        source_url="https://www.usek.edu.lb/en/master-of-science-in-computer-engineering",
    )
    minor = replace(
        detailed,
        document_id="minor",
        source_url="https://www.usek.edu.lb/minors/minor-in-computer-engineering",
    )
    unrelated = program("electrical-engineering", 146)
    external = replace(
        detailed,
        document_id="external",
        source_url="https://example.com/bachelor-of-engineering-in-computer-engineering",
    )
    uploaded = replace(detailed, document_id="uploaded", source_url="")
    hits = program_credit_sources(
        (other, detailed, duplicate, master, minor, unrelated, external, uploaded),
        ["Does the Computer Engineering major require 146 or 183 credits?"],
    )
    assert len(hits) == 2
    assert {hit.document_id for hit in hits} == {detailed.document_id, other.document_id}
    master_hits = program_credit_sources(
        (detailed, master), ["Computer engineering master credits"]
    )
    assert [hit.document_id for hit in master_hits] == ["master"]
    minor_hits = program_credit_sources((detailed, minor), ["Computer engineering minor credits"])
    assert [hit.document_id for hit in minor_hits] == ["minor"]


@pytest.mark.parametrize(
    "question",
    [
        "How many credits is GIN515 in Computer Engineering?",
        "How many credits is GIN 515 in Computer Engineering?",
        "How many credits is gin515 in Computer Engineering?",
        "Where is the Computer Engineering department?",
        "How many credits are required?",
    ],
)
def test_no_degree_total_injection_for_courses_or_unidentified_programs(question):
    assert program_credit_sources((program("computer-engineering", 146),), [question]) == []


def test_explicit_catalogue_year_keeps_its_scope_instead_of_program_page_variants(tmp_path):
    webpage = program("computer-engineering", 146)
    catalogue = replace(
        webpage,
        document_id="catalogue",
        filename="University Catalogue",
        file_type="pdf",
        pages=(
            PageText(1, "University Catalogue 2025-2026\nComputer Engineering 146 credits", "p. 1"),
        ),
        source_url="https://www.usek.edu.lb/catalogue.pdf",
    )

    def search(*args, document_ids):
        assert document_ids == ("catalogue",)
        return [stored_hit(catalogue)]

    hits = retrieve_sources(
        index_ids=["index"],
        documents=(webpage, catalogue),
        question="According to the 2025-2026 catalogue, how many credits in Computer Engineering?",
        top_k=2,
        embedder=FakeEmbedder(),
        store=SimpleNamespace(path=tmp_path, search=search),
    )
    assert [hit.document_id for hit in hits] == ["catalogue"]


def test_both_requirements_reach_answer_context_and_conflict_rule_without_reembedding(tmp_path):
    detailed = program("computer-engineering", 146)
    other = program("computer-engineering", 183, variant="1", detailed=False)
    model = model_with_client()
    model.prepare_search = lambda question, *args: (question, [question])
    model._client.chat.completions.create.return_value = response(
        "The degree requirements list 146 credits [Source 1]. "
        "The multilingual page lists 183 [Source 2]. Which curriculum applies?"
    )
    result = answer_question(
        index_ids=["existing-index"],
        documents=(other, detailed),
        question="How many total credits for the major computer engineering?",
        history=[],
        top_k=2,
        embedder=FakeEmbedder(),
        store=SimpleNamespace(path=tmp_path, search=lambda *args, **kwargs: [stored_hit(other)]),
        language_model=model,
        visualization="off",
        document_diversity=False,
    )
    request = model._client.chat.completions.create.call_args.kwargs["messages"]
    assert "146 credits" in request[1]["content"] and "183" in request[1]["content"]
    assert (
        detailed.source_url in request[1]["content"] and other.source_url in request[1]["content"]
    )
    assert "ask which curriculum or admission year applies" in request[0]["content"]
    assert "URL suffix or file name is not proof" in request[0]["content"]
    assert len(result.sources) == 2 and "146" in result.text and "183" in result.text


def test_updated_website_policy_replaces_a_cached_answer_model(monkeypatch):
    from rag_pdf import website_ui

    created = []

    def make_model(*args):
        model = object()
        created.append(model)
        return model

    website_ui.get_website_llm.clear()
    monkeypatch.setattr(website_ui, "GroqLanguageModel", make_model)
    try:
        first = website_ui.get_website_llm("test-key", "test-model", "original rules")
        assert website_ui.get_website_llm("test-key", "test-model", "original rules") is first
        updated = website_ui.get_website_llm("test-key", "test-model", "updated conflict rules")
        assert updated is not first
        assert len(created) == 2
    finally:
        website_ui.get_website_llm.clear()
