import pytest

from rag_pdf.config import ChunkConfig, Settings


def test_overlap_is_calculated_from_percentage() -> None:
    config = ChunkConfig(size_tokens=400, overlap_percent=15)

    assert config.overlap_tokens == 60


def test_default_chunk_size_is_32_tokens() -> None:
    assert ChunkConfig().size_tokens == 32


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(15, 15), (501, 15), (400, -1), (400, 41)],
)
def test_invalid_chunk_settings_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkConfig(size_tokens=size, overlap_percent=overlap)


def test_default_data_dir_uses_writable_user_app_data(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    settings = Settings.from_env()

    assert settings.data_dir == tmp_path / "RAG-PDF" / "chroma"
