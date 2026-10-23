"""Mode switching keeps uploaded files and USEK evidence in their own workflows."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from test_pipeline import FakeEmbedder, document
from test_website import export, html_record

from rag_pdf.config import Settings
from rag_pdf.models import Answer, SearchResult
from rag_pdf.presentation import prepare_website
from rag_pdf.service import index_document

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "entry,default_usek", [("app.py", False), ("pages/1_USEK_Website.py", True)]
)
def test_both_entry_points_switch_between_usek_and_file_upload(
    tmp_path, monkeypatch, entry, default_usek
):
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda cls: Settings(groq_api_key="test", data_dir=tmp_path)),
    )
    app = AppTest.from_file(str(ROOT / entry)).run()
    assert not app.exception
    assert app.toggle[0].label == "USEK Chatbot"
    assert app.toggle[0].value is default_usek
    for usek in (True, False, True, False):
        app.toggle[0].set_value(usek).run()
        assert not app.exception
        assert app.title[0].value == ("USEK Chatbot" if usek else "Document RAG")
        assert bool(app.get("file_uploader")) is (not usek)
        assert bool([b for b in app.button if b.label == "Load crawl snapshot"]) is usek


def test_processed_upload_and_settings_survive_mode_switch_without_reembedding(
    tmp_path, monkeypatch
):
    crawl = export(tmp_path / "crawl", [html_record()])
    package = prepare_website(crawl, tmp_path / "package", FakeEmbedder(), data_dir=tmp_path)
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda cls: Settings(groq_api_key="test", data_dir=tmp_path)),
    )
    uploads = [SimpleNamespace(name="program.pdf", getvalue=lambda: b"test PDF")]
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: uploads)

    class Embedder(FakeEmbedder):
        device_label = "Synthetic test device"

    monkeypatch.setattr("rag_pdf.document_ui.get_embedder", lambda *args: Embedder())
    monkeypatch.setattr("rag_pdf.document_ui.parse_document", lambda *args: document())
    indexed = []

    def record_index(*args):
        indexed.append(args[0].filename)
        return index_document(*args)

    monkeypatch.setattr("rag_pdf.document_ui.index_document", record_index)
    monkeypatch.setattr(
        "rag_pdf.website_ui.get_presentation_runtime",
        lambda *args: (package, Embedder(), object()),
    )
    monkeypatch.setattr("rag_pdf.document_ui.get_language_model", lambda *args: object())
    monkeypatch.setattr("rag_pdf.website_ui.get_website_llm", lambda *args: object())
    searched = []

    def answer(**kwargs):
        docs = kwargs["documents"]
        searched.append(tuple(doc.source_url for doc in docs))
        text = "USEK evidence" if docs[0].source_url else "Uploaded evidence"
        page = docs[0].pages[0]
        source = SearchResult(
            "source",
            page.text,
            page.page_number,
            0,
            1,
            filename=docs[0].filename,
            location=page.location,
            source_url=docs[0].source_url,
        )
        return Answer(text + ". [Source 1]", kwargs["question"], (source,))

    monkeypatch.setattr("rag_pdf.document_ui.answer_question", answer)
    monkeypatch.setattr("rag_pdf.website_ui.answer_question", answer)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    app.slider(key="document_chunk_size").set_value(384).run()
    app.selectbox(key="document_overlap").set_value(30).run()
    app.slider(key="document_top_k").set_value(7).run()
    next(b for b in app.button if b.label == "Process all documents").click().run()
    assert not app.exception
    assert indexed == ["program.pdf"]
    app.chat_input[0].set_value("What does my file say?").run()
    assert not app.exception
    assert searched[-1] == ("",)

    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert any("Ready" in item.value for item in app.success)
    assert not any("Uploaded evidence" in item.value for item in app.markdown)
    app.chat_input[0].set_value("How do I register?").run()
    assert not app.exception
    assert all(url.startswith("https://www.usek.edu.lb/") for url in searched[-1])

    # Streamlit discards hidden uploader state; the application must retain its own copy.
    uploads.clear()
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert app.slider(key="document_chunk_size").value == 384
    assert app.selectbox(key="document_overlap").value == 30
    assert app.slider(key="document_top_k").value == 7
    assert not app.chat_input[0].disabled
    assert indexed == ["program.pdf"]
    assert any("Uploaded evidence" in item.value for item in app.markdown)
    assert not any("USEK evidence" in item.value for item in app.markdown)
    app.chat_input[0].set_value("Tell me more about my file.").run()
    assert not app.exception
    assert searched[-1] == ("",)
    assert indexed == ["program.pdf"]

    next(b for b in app.button if b.label == "Clear saved uploads").click().run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert not app.session_state.get("document_uploads", ())
