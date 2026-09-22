"""Application-specific exceptions with messages suitable for the UI."""


class RAGError(Exception):
    """Base exception for expected application failures."""


class ValidationError(RAGError):
    """Raised when an uploaded document does not meet project constraints."""


class IndexNotFoundError(RAGError):
    """Raised when a question targets a document that has not been indexed."""


class ProviderError(RAGError):
    """Raised when the configured language-model provider fails."""
