import json
from dataclasses import replace

from test_pipeline import FakeEmbedder, WordTokenizer, document

from rag_pdf import benchmark
from rag_pdf.config import Settings
from rag_pdf.errors import ProviderError


def test_resume_preserves_successes_and_retries_only_failed_answers(tmp_path, monkeypatch):
    calls = []
    fail_once = {"openai/gpt-oss-120b"}

    class Embedder(FakeEmbedder):
        device_label = "CPU test"

        def __init__(self, model_name, device, batch_size):
            self.model_name = model_name

    class LLM:
        def __init__(self, key, model):
            self.model = model
            self.last_usage = {}
            self.last_calculations = []

        def rewrite_question(self, question, history):
            return question

        def answer(self, question, sources):
            calls.append(self.model)
            if self.model in fail_once:
                fail_once.remove(self.model)
                raise ProviderError("temporary quota exhaustion")
            return "Eight weeks. [Source 1]"

    monkeypatch.setattr(benchmark, "LocalEmbeddingModel", Embedder)
    monkeypatch.setattr(benchmark, "GroqLanguageModel", LLM)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: WordTokenizer()
    )
    questions = [
        {
            "id": "duration",
            "question": "How long is the internship?",
            "expected": "Eight weeks",
            "answer_patterns": ["Eight weeks"],
            "evidence": ["eight weeks"],
        }
    ]
    configs = [
        benchmark.Experiment(),
        replace(benchmark.Experiment(), name="other_llm", llm="openai/gpt-oss-120b"),
    ]
    settings = Settings(groq_api_key="test", data_dir=tmp_path, embedding_device="cpu")
    first = benchmark.run_study([document()], questions, configs, settings, tmp_path)
    saved_success = dict(first[0])
    assert first[1]["error"]
    resumed = benchmark.run_study([document()], questions, configs, settings, tmp_path)
    assert resumed[0] == saved_success
    assert len(resumed) == 2 and all(not row["error"] for row in resumed)
    assert calls == ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-120b"]
    assert resumed[1]["evidence_reused"]
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_rechecking_saved_answers_normalizes_format_without_changing_answers(tmp_path):
    answer = "Begins on **6\u202fJuly\u202f2026**. 【Source\u202f1】"
    question = {
        "id": "start",
        "question": "When does it start?",
        "expected": "6 July 2026",
        "answer_patterns": ["6 July 2026"],
        "evidence": ["6 July 2026"],
    }
    source = {
        "chunk_id": "a",
        "text": "6 July 2026",
        "page_number": 1,
        "chunk_index": 0,
        "score": 1,
    }
    row = {
        "configuration": "reference",
        "question_id": "start",
        "answer": answer,
        "sources": [source],
        "raw_sources": [source],
        "error": "",
        "answer_check_pass": False,
        "citation_ids_valid": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [question]}))
    benchmark.export_results([row], tmp_path)
    saved = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert saved["answer"] == answer
    assert saved["answer_check_pass"] and saved["citation_ids_valid"]


def test_failed_answer_retains_cached_retrieval_for_independent_comparison(tmp_path):
    import hashlib

    question = {
        "id": "duration",
        "question": "How long?",
        "expected": "Eight weeks",
        "evidence": ["eight weeks"],
    }
    experiment = benchmark.Experiment()
    row = {
        "configuration": experiment.name,
        "question_id": question["id"],
        "error": "provider quota",
        "sources": [],
        "answer": "",
        "database": experiment.database,
        "embedding": experiment.embedding,
        "chunk_size": experiment.chunk_size,
        "overlap": experiment.overlap,
        "top_k": experiment.top_k,
    }
    source = {
        "chunk_id": "a",
        "text": "eight weeks",
        "page_number": 1,
        "chunk_index": 0,
        "score": 1,
    }
    key = hashlib.sha256(
        json.dumps(
            [
                experiment.database,
                experiment.embedding,
                experiment.chunk_size,
                experiment.overlap,
                experiment.top_k,
                question,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / f"{key}.json").write_text(
        json.dumps(
            {
                "sources": [source],
                "raw_sources": [source],
                "standalone": "How long?",
                "retrieval_seconds": 0.01,
                "rewrite_seconds": 0,
            }
        )
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [question]}))
    benchmark.export_results([row], tmp_path)
    assert row["error"] == "provider quota" and row["retrieval_evidence_recovered"]
    assert row["evidence_hit"] and row["top1_evidence_hit"]
    assert row["answer_check_pass"] is None and row["citation_ids_valid"] is None
    result = benchmark.summarize([row])[0]
    assert result["successful"] == 0 and result["evidence_checks"] == "1/1"
    assert result["retrieval_ms"] == 10
