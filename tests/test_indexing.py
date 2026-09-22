from rag_pdf.config import ChunkConfig
from rag_pdf.models import ExtractedDocument, PageText
from rag_pdf.service import index_document


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


class FakeEmbedder:
    model_name = "fake-model"
    tokenizer = FakeTokenizer()
    batch_size = 2

    def __init__(self) -> None:
        self.batch_lengths: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_lengths.append(len(texts))
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.added_chunks = 0

    def count(self, index_id: str) -> int:
        return 0

    def delete(self, index_id: str) -> None:
        raise AssertionError("A missing index should not be deleted.")

    def add(self, index_id, document, chunks, embeddings, **kwargs) -> None:
        assert len(chunks) == len(embeddings)
        self.added_chunks += len(chunks)


def test_indexing_embeds_and_writes_in_bounded_batches() -> None:
    text = " ".join(f"word{number}" for number in range(100))
    document = ExtractedDocument(
        document_id="document",
        filename="paper.pdf",
        page_count=10,
        pages=(PageText(1, text),),
        language="en",
        language_confidence=1.0,
        character_count=len(text),
    )
    embedder = FakeEmbedder()
    store = FakeStore()

    summary = index_document(document, ChunkConfig(16, 0), embedder, store)

    assert len(embedder.batch_lengths) > 1
    assert max(embedder.batch_lengths) <= embedder.batch_size
    assert store.added_chunks == summary.chunk_count
