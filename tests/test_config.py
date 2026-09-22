import pytest

from rag_pdf.config import ChunkConfig


def test_overlap_is_calculated_from_percentage() -> None:
    config = ChunkConfig(size_tokens=400, overlap_percent=15)

    assert config.overlap_tokens == 60


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(99, 15), (501, 15), (400, -1), (400, 41)],
)
def test_invalid_chunk_settings_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkConfig(size_tokens=size, overlap_percent=overlap)
