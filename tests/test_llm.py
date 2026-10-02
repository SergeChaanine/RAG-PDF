from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_pdf.errors import ProviderError
from rag_pdf.llm import GroqLanguageModel
from rag_pdf.models import DocumentTable, SearchResult


def response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=[]))],
        usage=None,
    )


def model_with_client():
    model = GroqLanguageModel("test", "openai/gpt-oss-20b")
    model._client = Mock()
    return model


def test_qualitative_question_does_not_offer_table_calculator():
    model = model_with_client()
    model._client.chat.completions.create.return_value = response("Voluntary survey. [Source 1]")
    source = SearchResult(
        "a", "Voluntary survey", 1, 0, 1, table=DocumentTable("t", ("Count",), (("48",),), "p. 1")
    )
    model.answer("Why might the survey suffer from bias?", [source])
    assert "tools" not in model._client.chat.completions.create.call_args.kwargs


def test_invalid_tool_request_gets_one_corrective_retry():
    model = model_with_client()
    create = model._client.chat.completions.create
    create.side_effect = [RuntimeError("tool_use_failed: missing cells"), response("150")]
    assert model._request([], tools=True).choices[0].message.content == "150"
    assert create.call_count == 2
    assert "requires source_number" in create.call_args.kwargs["messages"][-1]["content"]
    create.reset_mock()
    create.side_effect = RuntimeError("tool_use_failed: missing cells")
    with pytest.raises(ProviderError):
        model._request([], tools=True)
    assert create.call_count == 2


def test_quota_wait_is_opt_in_and_obeys_retry_window(monkeypatch):
    model = model_with_client()
    error = RuntimeError("Please try again in 1m2.5s.")
    error.status_code = 429
    create = model._client.chat.completions.create
    create.side_effect = error
    with pytest.raises(ProviderError):
        model._request([])
    assert create.call_count == 1
    clock = [0.0]
    monkeypatch.setattr("rag_pdf.llm.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("rag_pdf.llm.time.sleep", lambda n: clock.__setitem__(0, clock[0] + n))
    model.rate_limit_wait_seconds = 100
    model.on_rate_limit_wait = Mock()
    create.side_effect = [error, response("Recovered")]
    assert model._request([]).choices[0].message.content == "Recovered"
    assert clock[0] == 64.5
    model.on_rate_limit_wait.assert_called_once_with(64.5)


def test_internal_tool_syntax_is_repaired_with_same_evidence_and_bounded_retry():
    model = model_with_client()
    source = SearchResult("a", "The internship lasts eight weeks.", 1, 0, 1)
    create = model._client.chat.completions.create
    create.side_effect = [response("Eight weeks. <|message|>"), response("Eight weeks. [Source 1]")]
    assert model.answer("How long is it?", [source]) == "Eight weeks. [Source 1]"
    assert create.call_count == 2
    assert "The internship lasts eight weeks" in create.call_args.kwargs["messages"][1]["content"]
    create.reset_mock()
    create.side_effect = lambda **kwargs: response("Eight weeks without a citation.")
    with pytest.raises(ProviderError, match="omitted source"):
        model.answer("How long is it?", [source])
    assert create.call_count == 2


def test_extended_source_labels_are_normalized_and_invalid_numbers_rejected():
    model = model_with_client()
    source = SearchResult("a", "Eight weeks.", 1, 0, 1)
    result = model._answer_content(
        response("Eight weeks. \u3010Source\u202f1 | a.pdf | p. 1\u3011"), [source]
    )
    assert result == "Eight weeks. [Source 1]"
    with pytest.raises(ProviderError, match="invalid source"):
        model._answer_content(response("Eight weeks. [Source 2]"), [source])
