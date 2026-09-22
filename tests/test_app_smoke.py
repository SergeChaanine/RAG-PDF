from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_upload_state_without_exceptions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "chroma"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "📄 PDF RAG Chatbot"
    assert "Upload a PDF" in app.info[0].value
