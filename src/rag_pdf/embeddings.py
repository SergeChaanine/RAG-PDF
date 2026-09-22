"""Local Hugging Face embedding model adapter."""

from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class LocalEmbeddingModel:
    """Create normalized document and query embeddings locally."""

    BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def tokenizer(self):  # Return type differs between supported transformer versions.
        return self._model.tokenizer

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        text = query
        if "bge-" in self.model_name.casefold():
            text = self.BGE_QUERY_INSTRUCTION + query
        embedding = self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
