"""Local Hugging Face embedding model adapter."""

from __future__ import annotations

from collections.abc import Sequence

from rag_pdf.errors import ValidationError


def resolve_embedding_device(requested: str, cuda_available: bool) -> str:
    """Resolve friendly device settings to a Sentence Transformers device."""

    normalized = requested.strip().casefold()
    if normalized in {"auto", "gpu"}:
        return "cuda" if cuda_available else "cpu"
    if normalized == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "EMBEDDING_DEVICE is set to 'cuda', but the installed PyTorch build "
                "cannot access CUDA. Install the CUDA PyTorch build or use 'auto'."
            )
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise ValueError("EMBEDDING_DEVICE must be 'auto', 'cuda', or 'cpu'.")


class LocalEmbeddingModel:
    """Create normalized document and query embeddings locally."""

    BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive.")
        self.model_name = model_name
        self.device = resolve_embedding_device(device, torch.cuda.is_available())
        self.batch_size = batch_size
        if model_name in {"BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B"}:
            self.batch_size = min(batch_size, 16)
        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
        try:
            # Avoid slow Hugging Face network checks once the model is cached.
            self._model = SentenceTransformer(
                model_name,
                device=self.device,
                local_files_only=True,
            )
        except (OSError, ValueError):
            # On the first run, allow Sentence Transformers to download the model.
            self._model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self._model.half()

    @property
    def device_label(self) -> str:
        if self.device != "cuda":
            return "CPU"
        import torch

        return f"{torch.cuda.get_device_name(0)} (CUDA, FP16)"

    @property
    def tokenizer(self):  # Return type differs between supported transformer versions.
        return self._model.tokenizer

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = [self.format_text(text, query=False) for text in texts]
        self.validate_lengths(texts)
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        text = self.format_text(query, query=True)
        self.validate_lengths([text])
        embedding = self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def format_text(self, text: str, *, query: bool) -> str:
        if self.model_name == "intfloat/e5-base-v2":
            return ("query: " if query else "passage: ") + text
        if query and self.model_name == "BAAI/bge-small-en-v1.5":
            return self.BGE_QUERY_INSTRUCTION + text
        if query and self.model_name == "Qwen/Qwen3-Embedding-0.6B":
            return "Instruct: Retrieve document passages that answer the question\nQuery: " + text
        return text

    def validate_lengths(self, texts: Sequence[str]) -> None:
        limit = self._model.max_seq_length
        if any(len(self.tokenizer.encode(text, add_special_tokens=True)) > limit for text in texts):
            raise ValidationError(
                f"An input exceeds {self.model_name}'s {limit}-token limit. "
                "Reduce the chunk size or shorten the question; text was not truncated."
            )
