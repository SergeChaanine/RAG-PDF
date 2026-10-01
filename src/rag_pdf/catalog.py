"""Curated choices; model-specific instructions are explicit, not name heuristics."""

EMBEDDING_MODELS = {
    "BAAI/bge-small-en-v1.5": "BGE small · English · lightweight",
    "intfloat/e5-base-v2": "E5 base · English",
    "BAAI/bge-m3": "BGE M3 · multilingual",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3 0.6B · multilingual",
}
LLM_MODELS = {
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "qwen/qwen3.8-27b": "Qwen 3.8 27B (preview)",
}
DATABASES = ("Chroma", "Qdrant", "LanceDB")
PIPELINE_VERSION = "structured-v2.1"
