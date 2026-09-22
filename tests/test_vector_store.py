from pathlib import Path

from rag_pdf.models import ExtractedDocument, PageText, TextChunk
from rag_pdf.vector_store import ChromaVectorStore


def test_chroma_round_trip_returns_most_similar_chunk(tmp_path: Path) -> None:
    document = ExtractedDocument(
        document_id="document-hash",
        filename="paper.pdf",
        page_count=10,
        pages=(PageText(1, "alpha"), PageText(7, "beta")),
        language="en",
        language_confidence=0.99,
        character_count=9,
    )
    chunks = [
        TextChunk("chunk-a", "document-hash", "paper.pdf", "alpha", 1, 0, 0, 1, 0),
        TextChunk("chunk-b", "document-hash", "paper.pdf", "beta", 7, 1, 0, 1, 0),
    ]
    store = ChromaVectorStore(tmp_path / "chroma")
    store.add(
        "index-id",
        document,
        chunks,
        [[1.0, 0.0], [0.0, 1.0]],
        embedding_model="test-model",
        chunk_size=400,
        overlap_percent=15,
    )

    results = store.search("index-id", [0.95, 0.05], top_k=2)

    assert store.has_index("index-id")
    assert results[0].chunk_id == "chunk-a"
    assert results[0].page_number == 1
    assert results[0].score > results[1].score
