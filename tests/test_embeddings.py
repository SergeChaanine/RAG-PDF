import pytest

from rag_pdf.embeddings import resolve_embedding_device


def test_auto_device_prefers_cuda_when_available() -> None:
    assert resolve_embedding_device("auto", cuda_available=True) == "cuda"
    assert resolve_embedding_device("auto", cuda_available=False) == "cpu"


def test_gpu_alias_is_supported() -> None:
    assert resolve_embedding_device("gpu", cuda_available=True) == "cuda"


def test_explicit_cuda_fails_clearly_when_unavailable() -> None:
    with pytest.raises(RuntimeError, match="cannot access CUDA"):
        resolve_embedding_device("cuda", cuda_available=False)
