"""Presentation preparation persists the expensive work and refuses partial packages."""

import json
import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_pipeline import FakeEmbedder
from test_website import export, html_record

from rag_pdf.config import Settings
from rag_pdf.embedding_cache import EmbeddingCache
from rag_pdf.presentation import active_presentation, load_prepared_website, prepare_website
from rag_pdf.service import retrieve_sources


def test_package_loads_without_raw_crawl_chunking_or_document_embeddings(tmp_path, monkeypatch):
    crawl = export(tmp_path / "crawl", [html_record(), html_record("aid", "Aid assessment.")])
    prepared = prepare_website(crawl, tmp_path / "package", FakeEmbedder(), data_dir=tmp_path)
    assert active_presentation(tmp_path) == prepared.path
    (crawl / "documents.jsonl").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Live loading must not repeat preparation")

    monkeypatch.setattr("rag_pdf.presentation.load_crawl_dataset", forbidden)
    monkeypatch.setattr("rag_pdf.presentation.plan_website_chunks", forbidden)
    monkeypatch.setattr(FakeEmbedder, "embed_documents", forbidden)
    loaded = load_prepared_website(prepared.path)
    assert loaded.dataset.documents == prepared.dataset.documents
    hits = retrieve_sources(
        index_ids=[loaded.index.index_id],
        documents=loaded.dataset.documents,
        question="internship",
        top_k=2,
        embedder=FakeEmbedder(),
        store=loaded.store,
        document_diversity=False,
    )
    assert hits and hits[0].source_url.endswith("test-fixture-registration")


def test_resume_reuses_completed_embeddings_and_does_not_publish_partial_package(tmp_path):
    crawl = export(
        tmp_path / "crawl", [html_record(str(i), f"internship number {i}") for i in range(5)]
    )

    class Interrupted(FakeEmbedder):
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("power interrupted")
            return super().embed_documents(texts)

    with pytest.raises(RuntimeError, match="power interrupted"):
        prepare_website(
            crawl, tmp_path / "package", Interrupted(), data_dir=tmp_path, write_batch_size=2
        )
    assert active_presentation(tmp_path) is None
    assert not (tmp_path / "package" / "manifest.json").exists()

    class Resumed(FakeEmbedder):
        texts = []

        def embed_documents(self, texts):
            self.texts.extend(texts)
            return super().embed_documents(texts)

    model = Resumed()
    prepared = prepare_website(crawl, tmp_path / "package", model, data_dir=tmp_path)
    assert prepared.index.chunk_count == 5
    assert len(model.texts) == 3  # The two completed vectors survived the interrupted build.
    assert active_presentation(tmp_path) == prepared.path


@pytest.mark.parametrize("damage", ["documents", "chunks", "keywords", "vectors", "version"])
def test_loader_refuses_damaged_or_incomplete_packages(tmp_path, damage):
    crawl = export(tmp_path / "crawl", [html_record()])
    ready = prepare_website(crawl, tmp_path / "package", FakeEmbedder())
    if damage in {"documents", "chunks"}:
        (ready.path / f"{damage}.jsonl").write_text("tampered", encoding="utf-8")
    elif damage == "keywords":
        with sqlite3.connect(ready.store.path / "keywords.sqlite3") as db:
            db.execute("DELETE FROM chunks")
    elif damage == "vectors":
        ready.store.delete(ready.index.index_id)
    else:
        path = ready.path / "manifest.json"
        manifest = json.loads(path.read_text("utf-8"))
        manifest["pipeline_version"] = "other"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prepared_website(ready.path)


def test_changed_settings_do_not_replace_a_ready_presentation(tmp_path):
    crawl = export(tmp_path / "crawl", [html_record()])
    prepared = prepare_website(crawl, tmp_path / "package", FakeEmbedder(), data_dir=tmp_path)
    with pytest.raises(ValueError, match="settings or source changed"):
        prepare_website(crawl, tmp_path / "package", FakeEmbedder(), limit=1, data_dir=tmp_path)
    assert active_presentation(tmp_path) == prepared.path
    assert load_prepared_website(prepared.path).index.index_id == prepared.index.index_id


def test_vector_cache_is_persistent_and_separates_models(tmp_path):
    path = tmp_path / "cache.sqlite3"
    cache = EmbeddingCache(path, "first-model")
    cache.put_many(["same text"], [[0.5, 1.0]])
    cache.close()
    cache = EmbeddingCache(path, "first-model")
    assert cache.get_many(["same text", "missing"]) == [[0.5, 1.0], None]
    cache.close()
    cache = EmbeddingCache(path, "other-model")
    assert cache.get_many(["same text"]) == [None]
    with pytest.raises(ValueError):
        cache.put_many(["invalid"], [[float("nan")]])
    cache.close()


def test_presentation_page_opens_ready_and_answers_without_build_buttons(tmp_path, monkeypatch):
    crawl = export(tmp_path / "crawl", [html_record()])
    prepared = prepare_website(crawl, tmp_path / "package", FakeEmbedder(), data_dir=tmp_path)

    class ReadyEmbedder(FakeEmbedder):
        device_label = "Synthetic test device"

    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda cls: Settings(groq_api_key="test", data_dir=tmp_path)),
    )
    monkeypatch.setattr(
        "rag_pdf.website_ui.get_presentation_runtime",
        lambda *args: (prepared, ReadyEmbedder(), object()),
    )
    from rag_pdf.models import Answer, SearchResult

    doc = prepared.dataset.documents[0]
    source = SearchResult(
        "test",
        doc.pages[0].text,
        0,
        0,
        1,
        filename=doc.filename,
        source_url=doc.source_url,
        fetched_at=doc.fetched_at,
    )
    monkeypatch.setattr("rag_pdf.website_ui.get_website_llm", lambda *args: object())
    monkeypatch.setattr(
        "rag_pdf.website_ui.answer_question",
        lambda **kwargs: Answer(
            "Eight weeks. [Source 1]", "How long is the internship?", (source,)
        ),
    )
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "pages" / "1_USEK_Website.py")
    )
    app.run()
    assert not app.exception
    assert any("Ready" in message.value for message in app.success)
    assert not any("Build" in button.label or "Load crawl" in button.label for button in app.button)
    app.chat_input[0].set_value("How long is the internship?").run()
    assert not app.exception
    assert any("Eight weeks" in message.value for message in app.markdown)
