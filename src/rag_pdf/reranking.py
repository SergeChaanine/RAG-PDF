"""Local passage relevance scoring; bilingual search supplies an English query."""

from __future__ import annotations

from rag_pdf.embeddings import resolve_embedding_device

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


class LocalReranker:
    def __init__(self, name=DEFAULT_RERANKER, device="auto", *, local_files_only=False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = name
        self.english_only = name.startswith("cross-encoder/ms-marco-")
        self.device = resolve_embedding_device(device, torch.cuda.is_available())
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                name,
                local_files_only=local_files_only,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    def rank(self, question, sources):
        import torch

        scores = []
        with torch.inference_mode():
            for start in range(0, len(sources), 8):
                batch = sources[start : start + 8]
                pairs = [(question, f"{s.filename}\n{s.location}\n{s.text}") for s in batch]
                inputs = self.tokenizer(
                    pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                scores.extend(self.model(**inputs).logits.view(-1).float().cpu().tolist())
        return sorted(zip(sources, scores, strict=True), key=lambda row: -row[1])
