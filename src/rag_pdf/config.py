"""Runtime configuration and chunking parameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_data_dir() -> Path:
    """Choose a user-writable, platform-appropriate persistence directory."""

    if local_app_data := os.getenv("LOCALAPPDATA"):
        return Path(local_app_data) / "RAG-PDF" / "chroma"
    if xdg_data_home := os.getenv("XDG_DATA_HOME"):
        return Path(xdg_data_home) / "rag-pdf" / "chroma"
    return Path.home() / ".local" / "share" / "rag-pdf" / "chroma"


@dataclass(frozen=True)
class ChunkConfig:
    """Token-aware chunking configuration."""

    size_tokens: int = 32
    overlap_percent: int = 15

    def __post_init__(self) -> None:
        if not 16 <= self.size_tokens <= 500:
            raise ValueError("Chunk size must be between 16 and 500 tokens.")
        if not 0 <= self.overlap_percent <= 40:
            raise ValueError("Chunk overlap must be between 0% and 40%.")
        if self.overlap_tokens >= self.size_tokens:
            raise ValueError("Chunk overlap must be smaller than the chunk size.")

    @property
    def overlap_tokens(self) -> int:
        """Calculate overlap dynamically from the configured percentage."""

        return round(self.size_tokens * self.overlap_percent / 100)


@dataclass(frozen=True)
class PDFRules:
    """Validation rules matching the assignment constraints."""

    min_pages: int = 10
    max_pages: int = 20
    max_file_bytes: int = 20 * 1024 * 1024
    min_total_characters: int = 1_000
    min_characters_per_text_page: int = 80
    min_text_page_ratio: float = 0.60
    min_english_probability: float = 0.70


@dataclass(frozen=True)
class Settings:
    """Environment-backed application settings."""

    groq_api_key: str
    groq_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "auto"
    embedding_batch_size: int = 64
    data_dir: Path = field(default_factory=default_data_dir)

    @classmethod
    def from_env(cls) -> Settings:
        configured_data_dir = os.getenv("RAG_DATA_DIR", "").strip()
        data_dir = (
            Path(os.path.expandvars(configured_data_dir)).expanduser()
            if configured_data_dir
            else default_data_dir()
        )
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            ).strip(),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "auto").strip(),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
            data_dir=data_dir,
        )
