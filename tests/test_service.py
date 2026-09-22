from rag_pdf.config import ChunkConfig
from rag_pdf.service import build_index_id


def test_index_identity_changes_when_retrieval_configuration_changes() -> None:
    original = build_index_id("document", "embedding-model", ChunkConfig(400, 15))
    different_chunks = build_index_id("document", "embedding-model", ChunkConfig(350, 15))
    different_overlap = build_index_id("document", "embedding-model", ChunkConfig(400, 20))

    assert original != different_chunks
    assert original != different_overlap
