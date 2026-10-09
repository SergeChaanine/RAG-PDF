"""Answers must follow the original question, regardless of evidence or query language."""

import pytest
from test_llm import model_with_client, response
from test_pipeline import FakeEmbedder, document
from test_retrieval import CrowdedStore

from rag_pdf.config import ChunkConfig
from rag_pdf.errors import ProviderError
from rag_pdf.llm import NOT_FOUND, NOT_FOUND_AR
from rag_pdf.models import ChatTurn, SearchResult
from rag_pdf.service import answer_question, index_document

EN_QUESTION = "On what date was the expansion plan approved?"
AR_QUESTION = "في أي تاريخ تمت الموافقة على خطة التوسعة؟"
EN_ANSWER = "The expansion plan was approved on 18 March 2025. [Source 1]"
AR_ANSWER = "تم اعتماد خطة التوسعة في 18 مارس 2025. [Source 1]"


@pytest.mark.parametrize(
    "original,resolved,wrong,correct,language",
    [
        (EN_QUESTION, EN_QUESTION, AR_ANSWER, EN_ANSWER, "English"),
        (AR_QUESTION, AR_QUESTION, EN_ANSWER, AR_ANSWER, "Arabic"),
        (EN_QUESTION, AR_QUESTION, AR_ANSWER, EN_ANSWER, "English"),
        (AR_QUESTION, EN_QUESTION, EN_ANSWER, AR_ANSWER, "Arabic"),
    ],
)
def test_wrong_language_is_repaired_from_same_evidence(
    original, resolved, wrong, correct, language
):
    model = model_with_client()
    source = SearchResult("a", "تم اعتماد خطة التوسعة في 18 مارس 2025.", 2, 0, 1)
    create = model._client.chat.completions.create
    create.side_effect = [response(wrong), response(correct)]
    assert model.answer(resolved, [source], original_question=original) == correct
    assert create.call_count == 2
    first = create.call_args_list[0].kwargs["messages"]
    assert f"Required response language: {language}" in first[0]["content"]
    assert original in first[1]["content"]
    assert source.text in create.call_args.kwargs["messages"][1]["content"]
    assert f"Write the answer in {language}" in create.call_args.kwargs["messages"][-1]["content"]


@pytest.mark.parametrize(
    "question,wrong,language",
    [(EN_QUESTION, AR_ANSWER, "English"), (AR_QUESTION, EN_ANSWER, "Arabic")],
)
def test_persistent_language_failure_stops_after_one_retry(question, wrong, language):
    model = model_with_client()
    source = SearchResult("a", "18 March 2025", 2, 0, 1)
    model._client.chat.completions.create.return_value = response(wrong)
    with pytest.raises(ProviderError, match=f"answer in {language}"):
        model.answer(question, [source])
    assert model._client.chat.completions.create.call_count == 2


@pytest.mark.parametrize(
    "original,resolved,expected",
    [(EN_QUESTION, AR_QUESTION, NOT_FOUND), (AR_QUESTION, EN_QUESTION, NOT_FOUND_AR)],
)
def test_missing_evidence_uses_original_question_language(original, resolved, expected):
    model = model_with_client()
    assert model.answer(resolved, [], original_question=original) == expected
    model._client.chat.completions.create.assert_not_called()


def test_english_question_can_include_a_quoted_arabic_project_name():
    model = model_with_client()
    source = SearchResult("a", "The project is واحة النور.", 1, 0, 1)
    expected = 'The project is "واحة النور". [Source 1]'
    model._client.chat.completions.create.return_value = response(expected)
    assert model.answer('What is the project "واحة النور"?', [source]) == expected
    assert model._client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("question", [EN_QUESTION, AR_QUESTION])
def test_numeric_date_is_language_neutral(question):
    model = model_with_client()
    source = SearchResult("a", "2025-03-18", 2, 0, 1)
    expected = "2025-03-18. [Source 1]"
    model._client.chat.completions.create.return_value = response(expected)
    assert model.answer(question, [source]) == expected


def test_arabic_table_can_preserve_english_names_as_source_values():
    model = model_with_client()
    source = SearchResult("a", "Names and counts", 1, 0, 1)
    expected = "| الاسم | الطلبات | المصدر |\n| --- | --- | --- |\n" + "\n".join(
        f"| Customer Name {i} | {i} | [Source 1] |" for i in range(30)
    )
    model._client.chat.completions.create.return_value = response(expected)
    assert model.answer("ما عدد الطلبات لكل عميل؟", [source]) == expected
    assert model._client.chat.completions.create.call_count == 1


def test_original_language_survives_rewriting_and_arabic_chat_history(tmp_path):
    model = model_with_client()
    model._client.chat.completions.create.side_effect = [
        response(AR_QUESTION),
        response(EN_ANSWER),
    ]
    doc = document("report.pdf", "تم اعتماد خطة التوسعة في 18 مارس 2025.")
    store, embedder = CrowdedStore(tmp_path), FakeEmbedder()
    summary = index_document(doc, ChunkConfig(), embedder, store)
    answer = answer_question(
        index_ids=[summary.index_id], documents=[doc], question=EN_QUESTION,
        history=[ChatTurn("assistant", "معلومات عن المشروع باللغة العربية.")],
        top_k=5, embedder=embedder, store=store, language_model=model,
    )
    assert embedder.last_query == AR_QUESTION
    assert answer.standalone_question == EN_QUESTION
    assert answer.text == EN_ANSWER
    assert model._client.chat.completions.create.call_count == 2


def test_arabic_question_about_english_evidence_returns_arabic():
    model = model_with_client()
    source = SearchResult("a", "The expansion was approved on 18 March 2025.", 2, 0, 1)
    model._client.chat.completions.create.return_value = response(AR_ANSWER)
    assert model.answer(AR_QUESTION, [source]) == AR_ANSWER
