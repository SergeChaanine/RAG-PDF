import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_llm import model_with_client, response
from test_pipeline import FakeEmbedder, document
from test_retrieval import CrowdedStore

from rag_pdf.config import ChunkConfig
from rag_pdf.embeddings import LocalEmbeddingModel
from rag_pdf.errors import ProviderError
from rag_pdf.models import ChatTurn
from rag_pdf.service import answer_question, index_document, retrieve_sources


@pytest.mark.parametrize(
    "original,translated",
    [
        ("ما إجمالي الربح خلال 2025؟", "What is the total profit during 2025?"),
        ("When was the expansion approved?", "متى تم اعتماد التوسعة؟"),
    ],
)
def test_retrieval_translation_keeps_original_and_opposite_language(original, translated):
    model = model_with_client()
    model._client.chat.completions.create.return_value = response(translated)
    assert model.retrieval_queries(original, ["report.pdf"]) == [original, translated]
    prompt = model._client.chat.completions.create.call_args.kwargs["messages"]
    assert "profit is not revenue" in prompt[0]["content"]
    assert "report.pdf" in prompt[1]["content"]


def test_wrong_language_translation_does_not_silently_duplicate_original():
    model = model_with_client()
    model._client.chat.completions.create.return_value = response("ما إجمالي الربح؟")
    with pytest.raises(ProviderError, match="retrieval translation was not in English"):
        model.retrieval_queries("ما إجمالي الربح؟")


def test_language_neutral_search_does_not_need_translation():
    model = model_with_client()
    assert model.retrieval_queries("2025") == ["2025"]
    model._client.chat.completions.create.assert_not_called()


def test_follow_up_rewrite_and_translation_use_one_request():
    model = model_with_client()
    create = model._client.chat.completions.create
    create.return_value = response(json.dumps({
        "question": "What was the bookshop's gross profit in 2025?",
        "translation": "ما إجمالي ربح المكتبة خلال عام 2025؟",
    }))
    history = [ChatTurn("user", "Tell me about the bookshop in 2025.")]

    standalone, queries = model.prepare_search("What was its profit?", history, ["books.xlsx"])

    assert "2025" in standalone
    assert queries == (standalone, "ما إجمالي ربح المكتبة خلال عام 2025؟")
    assert create.call_count == 1
    assert create.call_args.kwargs["max_completion_tokens"] == 384
    prompt = create.call_args.kwargs["messages"][1]["content"]
    assert "Tell me about the bookshop in 2025." in prompt


def test_search_cache_is_scoped_by_library_and_history():
    model = model_with_client()
    create = model._client.chat.completions.create
    create.return_value = response(json.dumps({
        "question": "When was the expansion plan approved?",
        "translation": "متى تم اعتماد خطة التوسعة؟",
    }))
    history = [ChatTurn("user", "Tell me about the solar expansion.")]
    first = model.prepare_search("When was it approved?", history, ["a.pdf", "b.pdf"])
    assert model.prepare_search("When was it approved?", history, ["b.pdf", "a.pdf"]) == first
    assert create.call_count == 1
    model.prepare_search("When was it approved?", history, ["other.pdf"])
    model.prepare_search(
        "When was it approved?", [ChatTurn("user", "Tell me about the library expansion.")],
        ["a.pdf", "b.pdf"],
    )
    assert create.call_count == 3


def test_cached_translation_reuses_request_but_invalidates_for_different_files():
    model = model_with_client()
    create = model._client.chat.completions.create
    create.return_value = response("What is the total profit in 2025?")
    question = "ما إجمالي الربح خلال 2025؟"
    first = model.prepare_search(question, [], ["books.xlsx"])
    assert model.prepare_search(question, [], ["books.xlsx"]) == first
    assert model.retrieval_queries(question, ["books.xlsx"]) == list(first[1])
    assert create.call_count == 1
    assert create.call_args.kwargs["max_completion_tokens"] == 256
    model.prepare_search(question, [], ["other.xlsx"])
    assert create.call_count == 2


def test_combined_planner_clarification_still_avoids_search():
    model = model_with_client()
    model._client.chat.completions.create.return_value = response(json.dumps({
        "clarification": "Do you mean the bookshop or the solar project?",
    }))
    answer = answer_question(
        index_ids=[], documents=[], question="When did it expand?",
        history=[ChatTurn("user", "Compare the bookshop and solar project.")],
        top_k=5, embedder=None, store=None, language_model=model,
    )
    assert answer.text == "Do you mean the bookshop or the solar project?"
    assert not answer.sources
    assert answer.standalone_question == ""
    assert model._client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("question,translation,expected", [
    (
        "ا أسماء المنشآت الأربع التي يشملها مشروع واحة النور؟",
        "What are the four facilities included in the واحة النور project?",
        "المنشآت هي الحرم الشمالي ومركز الأبحاث والمركز الصحي ومكتبة المجتمع. [Source 1]",
    ),
    (
        "كم عدد المنشآت المشاركة في مشروع واحة النور للطاقة الشمسية؟",
        "How many facilities participate in the واحة النور solar energy project?",
        "| المشروع | عدد المنشآت | المصدر |\n| --- | --- | --- |\n"
        "| واحة النور | 4 | [Source 1] |",
    ),
    (
        "Who is the project manager of Oasis of Light?",
        "من هي مديرة مشروع واحة النور؟",
        "The project manager is engineer Leila Haddad. [Source 1]",
    ),
    (
        "What are the four facilities that the Oasis of Light project includes?",
        "ما أسماء المنشآت الأربع التي يشملها مشروع واحة النور؟",
        "The facilities are North Campus, Research Center, Health Center, "
        "and Community Library. [Source 1]",
    ),
])
def test_named_project_is_searched_despite_previous_false_clarifications(
    tmp_path, question, translation, expected
):
    doc = document(
        "Arabic_RAG_Solar_Energy_Report.pdf",
        "يشمل مشروع واحة النور أربع منشآت: الحرم الشمالي ومركز الأبحاث "
        "والمركز الصحي ومكتبة المجتمع. مديرة المشروع المهندسة ليلى حداد.",
    )
    store, embedder = CrowdedStore(tmp_path), FakeEmbedder()
    summary = index_document(doc, ChunkConfig(), embedder, store)
    model = model_with_client()
    create = model._client.chat.completions.create
    create.side_effect = [response(translation), response(expected)]
    history = [
        ChatTurn("user", "ما أسماء المنشآت الأربع التي يشملها مشروع واحة النور؟"),
        ChatTurn("assistant", "حدد ما تقصده بمشروع واحة النور، فلا توجد معلومات كافية."),
    ]

    answer = answer_question(
        index_ids=[summary.index_id], documents=[doc], question=question, history=history,
        top_k=5, embedder=embedder, store=store, language_model=model,
    )

    assert answer.text == expected
    assert answer.standalone_question == question
    assert answer.sources[0].filename == doc.filename
    assert "ليلى حداد" in answer.sources[0].text
    assert create.call_count == 2  # Translation and answer, no ambiguity decision.
    first_prompt = create.call_args_list[0].kwargs["messages"]
    assert "Translate the search question" in first_prompt[0]["content"]
    assert "history" not in first_prompt[1]["content"]


def test_unknown_named_project_is_searched_then_reports_missing_evidence(tmp_path):
    model = model_with_client()
    create = model._client.chat.completions.create
    create.return_value = response("Who manages the Unlisted Project?")
    store, embedder = CrowdedStore(tmp_path), FakeEmbedder()
    answer = answer_question(
        index_ids=[], documents=[], question="من مدير مشروع غير مسجل؟",
        history=[ChatTurn("assistant", "لا أعرف هذا المشروع.")], top_k=5,
        embedder=embedder, store=store, language_model=model,
    )
    assert answer.text == "لم أتمكن من العثور على هذه المعلومة في المستندات المرفوعة."
    assert not answer.sources
    assert not answer.standalone_question.startswith("CLARIFY:")
    assert create.call_count == 1


@pytest.mark.parametrize("bad", [
    "not JSON", '{"question":"When?"}',
    '{"question":"When?","translation":"When?"}',
    '{"question":"","translation":"متى؟"}',
])
def test_invalid_combined_query_does_not_get_cached(bad):
    model = model_with_client()
    create = model._client.chat.completions.create
    create.return_value = response(bad)
    for _ in range(2):
        with pytest.raises(ProviderError, match="invalid bilingual search queries"):
            model.prepare_search("When?", [ChatTurn("user", "The expansion.")])
    assert create.call_count == 2


def test_query_batch_keeps_query_instructions_and_normalization():
    model = LocalEmbeddingModel.__new__(LocalEmbeddingModel)
    model.model_name, model.batch_size = "BAAI/bge-small-en-v1.5", 16
    model._model = Mock()
    model._model.max_seq_length = 512
    model._model.tokenizer.encode.side_effect = lambda text, **kwargs: text.split()
    model._model.encode.return_value = SimpleNamespace(tolist=lambda: [[1, 0], [0, 1]])

    vectors = model.embed_queries(["profit 2025", "الربح 2025"])

    assert vectors == [[1, 0], [0, 1]]
    model._model.encode.assert_called_once()
    request = model._model.encode.call_args
    assert request.args[0] == [
        model.BGE_QUERY_INSTRUCTION + "profit 2025",
        model.BGE_QUERY_INSTRUCTION + "الربح 2025",
    ]
    assert request.kwargs["normalize_embeddings"] is True


def test_retrieval_embeds_both_queries_in_one_batch(tmp_path):
    embedder = Mock()
    embedder.embed_queries.return_value = [[1, 0], [0, 1]]
    store = Mock(path=tmp_path)
    store.search.return_value = []
    retrieve_sources(
        index_ids=["one"], documents=[], question="profit", queries=["profit", "الربح"],
        top_k=5, embedder=embedder, store=store,
    )
    embedder.embed_queries.assert_called_once_with(["profit", "الربح"])
    embedder.embed_query.assert_not_called()
    assert [call.args[1] for call in store.search.call_args_list] == [[1, 0], [0, 1]]
