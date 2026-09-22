from rag_pdf.chunking import chunk_document
from rag_pdf.config import ChunkConfig
from rag_pdf.models import ExtractedDocument, PageText


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def test_chunking_preserves_page_metadata_and_token_limit() -> None:
    page_text = " ".join(f"word{index}" for index in range(350))
    document = ExtractedDocument(
        document_id="abc",
        filename="paper.pdf",
        page_count=10,
        pages=(PageText(page_number=4, text=page_text),),
        language="en",
        language_confidence=0.99,
        character_count=len(page_text),
    )

    chunks = chunk_document(
        document,
        WhitespaceTokenizer(),
        ChunkConfig(size_tokens=100, overlap_percent=25),
    )

    assert len(chunks) > 1
    assert all(chunk.page_number == 4 for chunk in chunks)
    assert all(chunk.token_count <= 100 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
