"""Failed uploads must be visible where the user asks questions."""

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest
from test_pipeline import document

from rag_pdf.catalog import PIPELINE_VERSION
from rag_pdf.config import Settings
from rag_pdf.models import IndexSummary
from rag_pdf.pdf_processing import document_hash


def test_chat_shows_failed_excel_even_when_pdf_is_searchable(tmp_path, monkeypatch):
    uploads = [
        SimpleNamespace(name="solar.pdf", getvalue=lambda: b"pdf"),
        SimpleNamespace(name="bookshop.xlsx", getvalue=lambda: b"excel"),
    ]
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: uploads)
    monkeypatch.setattr(
        Settings, "from_env",
        classmethod(lambda cls: Settings(groq_api_key="test", data_dir=tmp_path)),
    )
    pdf = document("solar.pdf")
    signature = (
        tuple(sorted((document_hash(file.getvalue()), file.name) for file in uploads)),
        "Chroma", "BAAI/bge-m3", 256, 15, PIPELINE_VERSION,
    )
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state.prepared = {
        "signature": signature,
        "documents": [pdf],
        "summaries": [IndexSummary("index", pdf.document_id, pdf.filename, 1, 1, 100, "en", False)],
        "failures": ["bookshop.xlsx: Cannot read worksheet"],
        "device": "CPU test",
    }
    app.run()
    assert not app.exception
    chat = app.tabs[0]
    assert any("Only 1 of 2" in warning.value for warning in chat.warning)
    assert any("bookshop.xlsx" in error.value for error in chat.error)
    assert any("Searchable files: solar.pdf" in caption.value for caption in chat.caption)
