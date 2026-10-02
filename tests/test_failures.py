from dataclasses import replace

import pymupdf
import pytest
from test_pipeline import FakeEmbedder, document

from rag_pdf.config import ChunkConfig
from rag_pdf.document_processing import extract_document
from rag_pdf.errors import ValidationError
from rag_pdf.models import ChatTurn
from rag_pdf.service import answer_question, index_document
from rag_pdf.vector_store import create_store


@pytest.mark.parametrize("backend", ["Chroma", "Qdrant", "LanceDB"])
def test_interrupted_index_is_rebuilt_without_duplicate_rows(tmp_path, backend):
    store = create_store(backend, tmp_path)
    doc = document(text=" ".join(f"word{i}" for i in range(100)))

    class FailingEmbedder(FakeEmbedder):
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interrupted")
            return super().embed_documents(texts)

    with pytest.raises(RuntimeError, match="interrupted"):
        index_document(doc, ChunkConfig(16, 15), FailingEmbedder(), store)
    summary = index_document(doc, ChunkConfig(16, 15), FakeEmbedder(), store)
    assert not summary.reused_existing_index
    assert store.count(summary.index_id) == summary.chunk_count
    if hasattr(store, "close"):
        store.close()


def test_ambiguous_follow_up_asks_without_searching():
    class AmbiguousLLM:
        def rewrite_question(self, question, history):
            return "CLARIFY: Do you mean the internship or the workshop?"

    result = answer_question(
        index_ids=[],
        documents=[],
        question="How long is it?",
        history=[ChatTurn("user", "Compare the internship and workshop.")],
        top_k=5,
        embedder=None,
        store=None,
        language_model=AmbiguousLLM(),
    )
    assert result.text == "Do you mean the internship or the workshop?"
    assert not result.sources


def test_pdf_extractor_keeps_negative_statistics_and_page_locations():
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Net change was -12.5 percent in 2025.")
    parsed = extract_document(pdf.tobytes(), "statistics.pdf")
    assert "-12.5" in parsed.pages[0].text
    assert parsed.pages[0].location == "p. 1"
    pdf.close()


def test_invalid_files_and_empty_scans_are_rejected():
    with pytest.raises(ValidationError):
        extract_document(b"not a PDF", "test.pdf")
    with pytest.raises(ValidationError):
        extract_document(b"not a ZIP", "test.docx")
    pdf = pymupdf.open()
    pdf.new_page()
    with pytest.raises(ValidationError, match="No native text"):
        extract_document(pdf.tobytes(), "scan.pdf")
    pdf.close()


def test_word_prose_can_span_adjacent_paragraphs():
    from test_pipeline import WordTokenizer

    from rag_pdf.chunking import chunk_document
    from rag_pdf.models import PageText

    doc = replace(
        document(),
        file_type="docx",
        pages=(
            PageText(0, "The supervised internship", "Program, paragraph 1"),
            PageText(0, "lasts eight weeks.", "Program, paragraph 2"),
        ),
    )
    chunks = chunk_document(doc, WordTokenizer(), ChunkConfig())
    assert len(chunks) == 1
    assert "lasts eight weeks" in chunks[0].text
    assert "paragraph 1 through Program, paragraph 2" in chunks[0].location
