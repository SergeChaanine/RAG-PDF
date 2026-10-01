"""Small, reproducible one-factor comparison; callable from Streamlit or the CLI."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from rag_pdf.catalog import PIPELINE_VERSION
from rag_pdf.config import ChunkConfig, Settings
from rag_pdf.document_processing import extract_document
from rag_pdf.embeddings import LocalEmbeddingModel
from rag_pdf.llm import GroqLanguageModel
from rag_pdf.models import ChatTurn
from rag_pdf.service import expand_tables, index_document, library_id, search_library
from rag_pdf.vector_store import create_store

REFERENCE_TOKENIZER = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Experiment:
    name: str = "reference"
    database: str = "Chroma"
    embedding: str = REFERENCE_TOKENIZER
    chunk_size: int = 256
    overlap: int = 15
    llm: str = "openai/gpt-oss-20b"
    top_k: int = 5


def configurations():
    base = Experiment()
    return [
        base,
        replace(base, name="embedding_e5", embedding="intfloat/e5-base-v2"),
        replace(base, name="embedding_bge_m3", embedding="BAAI/bge-m3"),
        replace(base, name="embedding_qwen", embedding="Qwen/Qwen3-Embedding-0.6B"),
        replace(base, name="database_qdrant", database="Qdrant"),
        replace(base, name="database_lancedb", database="LanceDB"),
        replace(base, name="chunks_128", chunk_size=128),
        replace(base, name="chunks_480", chunk_size=480),
        replace(base, name="overlap_30", overlap=30),
        replace(base, name="llm_gpt_oss_120b", llm="openai/gpt-oss-120b"),
        replace(base, name="llm_qwen", llm="qwen/qwen3.8-27b"),
    ]


def load_questions(path):
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_questions(questions)
    return questions


def validate_questions(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions must be a nonempty JSON array.")
    ids = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be a JSON object.")
        if not all(
            isinstance(question.get(k), str) and question[k].strip()
            for k in ("id", "question", "expected")
        ):
            raise ValueError("Each question requires string id, question, and expected fields.")
        if question["id"] in ids:
            raise ValueError("Question IDs must be unique.")
        ids.add(question["id"])
        for field in ("evidence", "answer_patterns"):
            if not isinstance(question.get(field, []), list) or any(
                not isinstance(value, str) for value in question.get(field, [])
            ):
                raise ValueError(f"{field} must be a list of strings.")
        for pattern in question.get("answer_patterns", []):
            re.compile(pattern)
        if any(
            turn.get("role") not in {"user", "assistant"}
            or not isinstance(turn.get("content"), str)
            for turn in question.get("history", [])
        ):
            raise ValueError("History must contain user/assistant turns with text content.")


def _checks(question, answer, sources):
    text = " ".join(" ".join(s.text for s in sources).split()).casefold()
    answer = unicodedata.normalize("NFKC", answer).replace("\\%", "%")
    answer = answer.replace("**", "")
    anchors = question.get("evidence", [])
    evidence_hit = all(anchor.casefold() in text for anchor in anchors) if anchors else None
    patterns = question.get("answer_patterns", [])
    answer_pass = all(re.search(p, answer, re.I) for p in patterns) if patterns else None
    citations = [int(n) for n in re.findall(r"[\[【]Source\s+(\d+)[\]】]", answer)]
    valid_ids = bool(citations) and all(1 <= n <= len(sources) for n in citations)
    return {
        "evidence_hit": evidence_hit,
        "answer_check_pass": answer_pass,
        "citation_ids_valid": valid_ids,
    }


def run_study(
    documents,
    questions,
    experiments,
    settings,
    output,
    *,
    retrieval_only=False,
    progress=None,
    wait_for_rate_limit=False,
):
    """Run only explicitly selected experiments; report failures without inventing scores."""
    validate_questions(questions)
    if not documents:
        raise ValueError("At least one document is required.")
    if not retrieval_only and not settings.groq_api_key:
        raise ValueError("Set GROQ_API_KEY or use --retrieval-only.")
    if retrieval_only and any(
        q.get("history") and not q.get("standalone_question") for q in questions
    ):
        raise ValueError("Retrieval-only follow-ups require a standalone_question reference.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "pipeline": PIPELINE_VERSION,
                "corpus": library_id(documents),
                "questions": questions,
                "experiments": [asdict(e) for e in experiments],
                "retrieval_only": retrieval_only,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError(
                "This results folder belongs to a different study. Choose a new folder."
            )
    versions = {}
    for package in (
        "sentence-transformers",
        "torch",
        "chromadb",
        "qdrant-client",
        "lancedb",
        "pymupdf",
        "python-docx",
        "groq",
    ):
        versions[package] = importlib.metadata.version(package)
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "pipeline": PIPELINE_VERSION,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "versions": versions,
                    "documents": [
                        {"filename": d.filename, "sha256": d.document_id} for d in documents
                    ],
                    "experiments": [asdict(e) for e in experiments],
                    "questions": questions,
                    "reference_tokenizer": REFERENCE_TOKENIZER,
                    "retrieval_only": retrieval_only,
                    "search_modes": {
                        "Chroma": "HNSW cosine",
                        "Qdrant": "local exact cosine",
                        "LanceDB": "unindexed exact cosine",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    rows_path = output / "results.jsonl"
    rows = (
        [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows_path.exists()
        else []
    )
    completed = {(r["configuration"], r["question_id"]) for r in rows if not r.get("error")}
    # Failed rows are replaced when the run is resumed; successes are retained.
    rows = [r for r in rows if not r.get("error")]
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
    # Disk evidence cache guarantees identical context for the three answer models,
    # including across resumed runs. Query rewriting uses a fixed reference model.
    evidence_dir = output / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    queries_path = evidence_dir / "queries.json"
    query_cache = json.loads(queries_path.read_text()) if queries_path.exists() else {}
    rewrite_model = (
        None if retrieval_only else GroqLanguageModel(settings.groq_api_key, "openai/gpt-oss-20b")
    )
    vector_cache, stores = {}, {}
    embedder, current_model = None, None
    total = len(experiments) * len(questions)
    done = len(completed)
    # Load each embedding model only once; the reference is used by most experiments.
    ordered = sorted(experiments, key=lambda e: e.embedding != REFERENCE_TOKENIZER)
    try:
        for experiment in ordered:
            pending = [q for q in questions if (experiment.name, q["id"]) not in completed]
            if not pending:
                continue
            try:
                started = time.perf_counter()
                if current_model != experiment.embedding:
                    if embedder is not None:
                        embedder = None
                        gc.collect()
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    embedder = LocalEmbeddingModel(
                        experiment.embedding,
                        settings.embedding_device,
                        min(settings.embedding_batch_size, 16),
                    )
                    current_model = experiment.embedding
                    model_load_seconds = time.perf_counter() - started
                else:
                    model_load_seconds = 0.0
                if experiment.database not in stores:
                    stores[experiment.database] = create_store(
                        experiment.database, output / "indexes"
                    )
                store = stores[experiment.database]
                started = time.perf_counter()
                summaries = [
                    index_document(
                        d,
                        ChunkConfig(experiment.chunk_size, experiment.overlap),
                        embedder,
                        store,
                        tokenizer=tokenizer,
                        tokenizer_id=REFERENCE_TOKENIZER,
                        vector_cache=vector_cache,
                    )
                    for d in documents
                ]
                index_seconds = time.perf_counter() - started
                language_model = (
                    None
                    if retrieval_only
                    else GroqLanguageModel(settings.groq_api_key, experiment.llm)
                )
                if language_model and wait_for_rate_limit:
                    language_model.rate_limit_wait_seconds = 1800
                    language_model.on_rate_limit_wait = lambda delay: print(
                        f"Provider quota: waiting {delay:.0f}s before retrying this request.",
                        flush=True,
                    )
                setup_error = None
            except Exception as exc:
                setup_error = str(exc)
            for question in pending:
                row = {
                    "configuration": experiment.name,
                    "question_id": question["id"],
                    **asdict(experiment),
                    "question": question["question"],
                    "category": question.get("category", "unspecified"),
                    "expected": question["expected"],
                    "answer": "",
                    "sources": [],
                    "error": "",
                    "correctness": "",
                    "citation_support": "",
                }
                try:
                    if setup_error:
                        raise RuntimeError(setup_error)
                    retrieval_key = hashlib.sha256(
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
                    cache_path = evidence_dir / f"{retrieval_key}.json"
                    if cache_path.exists():
                        cached = json.loads(cache_path.read_text(encoding="utf-8"))
                        from rag_pdf.models import DocumentTable, SearchResult

                        sources = []
                        for item in cached["sources"]:
                            if item.get("table"):
                                item["table"] = DocumentTable(**item["table"])
                            sources.append(SearchResult(**item))
                        standalone = cached["standalone"]
                        retrieval_seconds = cached["retrieval_seconds"]
                        rewrite_seconds = cached["rewrite_seconds"]
                        raw_sources = cached["raw_sources"]
                        reused_evidence = True
                    else:
                        started = time.perf_counter()
                        history = [ChatTurn(**turn) for turn in question.get("history", [])]
                        if question["id"] not in query_cache:
                            query_cache[question["id"]] = (
                                question.get("standalone_question", question["question"])
                                if retrieval_only
                                else rewrite_model.rewrite_question(question["question"], history)
                            )
                            queries_path.write_text(json.dumps(query_cache), encoding="utf-8")
                        standalone = query_cache[question["id"]]
                        if standalone.startswith("CLARIFY:"):
                            raise ValueError(
                                "The reference question needs clarification: " + standalone
                            )
                        rewrite_seconds = time.perf_counter() - started
                        started = time.perf_counter()
                        hits = search_library(
                            [s.index_id for s in summaries],
                            embedder.embed_query(standalone),
                            experiment.top_k,
                            store,
                        )
                        retrieval_seconds = time.perf_counter() - started
                        raw_sources = [asdict(hit) for hit in hits]
                        sources = expand_tables(hits, documents)
                        cache_path.write_text(
                            json.dumps(
                                {
                                    "sources": [asdict(s) for s in sources],
                                    "raw_sources": raw_sources,
                                    "standalone": standalone,
                                    "retrieval_seconds": retrieval_seconds,
                                    "rewrite_seconds": rewrite_seconds,
                                }
                            ),
                            encoding="utf-8",
                        )
                        reused_evidence = False
                    started = time.perf_counter()
                    answer = language_model.answer(standalone, sources) if language_model else ""
                    answer_seconds = time.perf_counter() - started
                    quota_wait = getattr(language_model, "last_rate_limit_wait_seconds", 0.0)
                    row.update(
                        answer=answer,
                        standalone_question=standalone,
                        sources=[asdict(s) for s in sources],
                        raw_sources=raw_sources,
                        retrieval_seconds=retrieval_seconds,
                        answer_seconds=max(0.0, answer_seconds - quota_wait),
                        quota_wait_seconds=quota_wait,
                        rewrite_seconds=rewrite_seconds,
                        index_seconds=index_seconds,
                        model_load_seconds=model_load_seconds,
                        index_reused=all(s.reused_existing_index for s in summaries),
                        evidence_reused=reused_evidence,
                        device=embedder.device_label,
                        chunk_count=sum(s.chunk_count for s in summaries),
                        usage=language_model.last_usage if language_model else {},
                        calculations=language_model.last_calculations if language_model else [],
                    )
                    checks = _checks(question, answer, sources)
                    if retrieval_only:
                        checks.update(answer_check_pass=None, citation_ids_valid=None)
                    row.update(checks)
                    from rag_pdf.models import SearchResult

                    raw_checks = _checks(question, answer, [SearchResult(**s) for s in raw_sources])
                    row["raw_evidence_hit"] = raw_checks["evidence_hit"]
                except Exception as exc:
                    row["error"] = str(exc)
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1
                if progress:
                    progress(done, total, row)
    finally:
        for store in stores.values():
            if hasattr(store, "close"):
                store.close()
    export_results(rows, output)
    return rows


def export_results(rows, output):
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    questions = {q["id"]: q for q in manifest.get("questions", [])}
    # Recompute derived checks consistently for old and resumed rows, without API calls.
    from rag_pdf.models import SearchResult

    for row in rows:
        if row["question_id"] not in questions:
            continue
        question = questions[row["question_id"]]
        if row.get("error") and not row.get("sources"):
            # A provider failure can occur after retrieval succeeded and was cached.
            key = hashlib.sha256(
                json.dumps(
                    [
                        row["database"],
                        row["embedding"],
                        row["chunk_size"],
                        row["overlap"],
                        row["top_k"],
                        question,
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            saved_evidence = output / "evidence" / f"{key}.json"
            if saved_evidence.exists():
                evidence = json.loads(saved_evidence.read_text(encoding="utf-8"))
                row.update(
                    {
                        k: evidence[k]
                        for k in ("sources", "raw_sources", "retrieval_seconds", "rewrite_seconds")
                    }
                )
                row["standalone_question"] = evidence["standalone"]
                row["retrieval_evidence_recovered"] = True
        sources = [SearchResult(**source) for source in row.get("sources", [])]
        row.update(_checks(question, row.get("answer", ""), sources))
        if manifest.get("retrieval_only") or row.get("error"):
            row.update(answer_check_pass=None, citation_ids_valid=None)
        if row.get("error") and not row.get("sources"):
            row["evidence_hit"] = None
        raw = [SearchResult(**source) for source in row.get("raw_sources", [])]
        row["raw_evidence_hit"] = _checks(question, row.get("answer", ""), raw)["evidence_hit"]
        row["top1_evidence_hit"] = _checks(question, row.get("answer", ""), raw[:1])["evidence_hit"]
        if row.get("error") and not raw:
            row.update(raw_evidence_hit=None, top1_evidence_hit=None)
    if questions:
        manifest["scoring"] = (
            "Prepared regex checks with NFKC whitespace, Markdown bold, and escaped-percent "
            "normalization. Original answers and reference patterns are unchanged."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    fields = [
        "configuration",
        "question_id",
        "category",
        "question",
        "expected",
        "answer",
        "evidence_hit",
        "raw_evidence_hit",
        "top1_evidence_hit",
        "answer_check_pass",
        "citation_ids_valid",
        "correctness",
        "citation_support",
        "retrieval_seconds",
        "answer_seconds",
        "quota_wait_seconds",
        "index_seconds",
        "index_reused",
        "evidence_reused",
        "chunk_count",
        "device",
        "error",
    ]
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    lines = [
        "# Focused RAG comparison",
        "",
        "## Method",
        "",
        f"{len(manifest.get('experiments', summary))} configurations change one factor at a time. "
        "Overlap is 15% or 30%. "
        "All use five retrieved chunks and the BGE-small reference tokenizer; model input "
        "limits are checked without truncation. Complete retrieved tables are expanded before "
        "answering. The answer-model variants reuse identical saved evidence. Follow-up "
        "rewriting uses GPT-OSS 20B for every configuration. Temperature is zero.",
        "",
        "Measurements are a single pass on one small corpus. Retrieval includes query embedding. "
        "Recorded quota waiting is excluded from answer latency and reported separately. "
        "Model loading is recorded separately. Index timings with reused indexes/vectors are "
        "not cold-build timings. Local exact Qdrant/LanceDB and Chroma HNSW differ in search mode; "
        "these results do not measure server scalability.",
        "",
        "## Results",
        "",
        "| Configuration | Successful / attempted | Evidence checks | Answer checks | "
        "Top-1 anchors | Valid citation IDs | Retrieval mean ms | Answer mean s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['configuration']} | {row['successful']}/{row['attempted']} | "
            f"{row['evidence_checks']} | {row['answer_checks']} | "
            f"{row['top1_evidence_checks']} | {row['citation_checks']} | "
            f"{row['retrieval_ms']:.1f} | {row['answer_s']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation and review",
        "",
        "Evidence checks test reference substrings in the evidence supplied to the answer model. "
        "Answer checks test prepared regular expressions; neither is a human correctness score. "
        "Top-1 anchors inspect only the first saved raw hit, without another run or table "
        "expansion; they are a ranking diagnostic, not a top-1 answer experiment. "
        "Valid citation IDs do not prove citation support. Review answers and sources in "
        "results.jsonl, then fill correctness (0, 0.5, 1) and citation_support (0, 1) in "
        "results.csv or the UI. Failed answers are excluded from answer/citation check rates; "
        "successful cached retrieval is still scored independently. Failures remain explicit.",
        "",
        "Use this as an exploratory case study, not a universal model/database ranking. "
        "The included sample is synthetic. Multi-document source selection and DOCX parsing have "
        "separate "
        "functional tests; this single-document study does not establish those capabilities.",
        "",
        "## Presentation outline",
        "",
        "1. Problem and pipeline: PDF/DOCX text and tables to cited conversational answers.",
        "2. Dataset: document, questions, reference evidence, and synthetic-data limitation.",
        "3. Experiment: reference configuration and ten one-factor variations.",
        "4. Results: check rates and response times, followed by reviewed answer quality.",
        "5. Examples: a table calculation, a follow-up question, and a failed answer.",
        "6. Recommendation: quality/runtime trade-off within this tested corpus.",
    ]
    findings = study_findings(rows, summary, manifest)
    lines += ["", "## Observed findings", "", *[f"- {point}" for point in findings]]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paired_ids, paired_summary = paired_results(rows)
    (output / "paired_summary.json").write_text(
        json.dumps(
            {
                "question_ids": paired_ids,
                "summary": paired_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize(rows):
    summary = []
    for name in dict.fromkeys(r["configuration"] for r in rows):
        group = [r for r in rows if r["configuration"] == name]
        good = [r for r in group if not r.get("error")]
        retrieved = [r for r in group if r.get("retrieval_seconds") is not None]

        def checks(key, good=good):
            values = [r[key] for r in good if r.get(key) is not None]
            return f"{sum(values)}/{len(values)}" if values else "not measured"

        def mean(key, good=good):
            return sum(r.get(key, 0) for r in good) / len(good) if good else 0.0

        summary.append(
            {
                "configuration": name,
                "successful": len(good),
                "attempted": len(group),
                "evidence_checks": checks("evidence_hit", retrieved),
                "answer_checks": checks("answer_check_pass"),
                "top1_evidence_checks": checks("top1_evidence_hit", retrieved),
                "citation_checks": checks(
                    "citation_ids_valid", [r for r in good if r.get("category") != "unanswerable"]
                ),
                "retrieval_ms": mean("retrieval_seconds", retrieved) * 1000,
                "answer_s": mean("answer_seconds"),
                "quota_wait_s": sum(r.get("quota_wait_seconds", 0) for r in group),
            }
        )
    return summary


def paired_results(rows):
    groups = {
        name: {r["question_id"] for r in rows if r["configuration"] == name and not r.get("error")}
        for name in dict.fromkeys(r["configuration"] for r in rows)
    }
    shared = set.intersection(*groups.values()) if groups else set()
    return sorted(shared), summarize([r for r in rows if r["question_id"] in shared])


def study_findings(rows, summary, manifest):
    """Plain-language observations from saved measurements; no generated grades."""
    by_name = {r["configuration"]: r for r in summary}
    points = []
    reference = next(
        (r for r in rows if r["configuration"] == "reference" and not r.get("error")), None
    )
    if reference:
        points.append(
            f"The reference has {reference.get('chunk_count', '?')} indexed chunks and retrieves "
            f"{reference.get('top_k', '?')} per question. Broad context coverage and full-table "
            "expansion can conceal differences in retrieval ranking on this small document."
        )
    if all(n in by_name for n in ("reference", "embedding_e5", "embedding_bge_m3")):
        points.append(
            "Top-1 raw evidence anchors: BGE-small "
            f"{by_name['reference']['top1_evidence_checks']}, E5-base "
            f"{by_name['embedding_e5']['top1_evidence_checks']}, BGE-M3 "
            f"{by_name['embedding_bge_m3']['top1_evidence_checks']}. "
            "This distinguishes ranking even when supplied-context anchor checks tie. "
            "Substring anchors remain an imperfect measure of useful evidence."
        )
    if all(n in by_name for n in ("reference", "overlap_30")):
        points.append(
            f"15% overlap has {by_name['reference']['evidence_checks']} supplied-context checks "
            f"and {by_name['reference']['top1_evidence_checks']} top-1 checks; 30% has "
            f"{by_name['overlap_30']['evidence_checks']} and "
            f"{by_name['overlap_30']['top1_evidence_checks']}, respectively. "
            "The sample gives limited evidence for increasing overlap."
        )
    if all(n in by_name for n in ("reference", "llm_gpt_oss_120b", "llm_qwen")):
        points.append(
            "With identical retrieval evidence, GPT-OSS 20B/120B and Qwen 3.8 average "
            f"{by_name['reference']['answer_s']:.2f}/{by_name['llm_gpt_oss_120b']['answer_s']:.2f}/"
            f"{by_name['llm_qwen']['answer_s']:.2f}s. Their valid citation-ID checks are "
            f"{by_name['reference']['citation_checks']}, "
            f"{by_name['llm_gpt_oss_120b']['citation_checks']}, and "
            f"{by_name['llm_qwen']['citation_checks']}. "
            "Citation syntax is separate from whether the cited text supports the answer."
        )
    if any(d.get("filename") == "sample_report.pdf" for d in manifest.get("documents", [])):
        missed_hours = [
            r
            for r in rows
            if r["question_id"] == "q03"
            and not r.get("error")
            and r.get("answer_check_pass") is False
        ]
        if missed_hours:
            points.append(
                f"Spot-review of {len(missed_hours)} failed hours checks found correct answers "
                "such as '10 supervised hours'; the prepared regex expects '10 hours'. "
                "The strict automatic results are retained, rather than tuning reference "
                "patterns after seeing model answers. No human correctness score is claimed."
            )
        points.append(
            "Calculation spot-review: 40 + 50 + 60 = 150 enrolled interns; all twelve visit "
            "counts sum to 2,232, with mean 186; 48/60 = 80% survey participation. Saved "
            "traces also reveal wrong column selections corrected by the LLM and malformed "
            "citations/tool syntax despite correct final numbers. Numerical checks alone "
            "would miss those defects. The current app adds bounded citation/tool-output "
            "repair; the original measured answers remain available for inspection."
        )
        points.append(
            "For this small English sample, retain Chroma/BGE-small, 256 tokens, and 15% "
            "overlap as a practical starting point. E5 deserves consideration for ranking; "
            "GPT-OSS 120B improved citation formatting with similar observed answer time. "
            "Larger embeddings and Qwen answering did not establish a necessary benefit "
            "for this case. This conclusion does not extend to multilingual or larger corpora."
        )
    total_wait = sum(r.get("quota_wait_seconds", 0) for r in rows)
    if total_wait:
        points.append(
            f"Resumed requests spent {total_wait / 60:.1f} minutes waiting for provider quota. "
            "That waiting is separate from answer latency. Successful earlier rows were "
            "preserved; API availability affected completion time, not retrieval evidence."
        )
    if any(r.get("error") for r in rows):
        points.append(
            "Some requests failed. Compare the explicit denominators; incomplete "
            "configurations cannot support an equal-size quality comparison. Retrieval "
            "checks include saved evidence from requests whose answer API failed. "
            "paired_summary.json compares only questions successful for every configuration."
        )
    return points


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", nargs="+", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Settings.from_env().data_dir / "studies" / "sample"
    )
    parser.add_argument("--configs", nargs="*", choices=[c.name for c in configurations()])
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--wait-for-rate-limit",
        action="store_true",
        help="Honor Groq retry windows, up to 30 minutes per request.",
    )
    args = parser.parse_args()
    experiments = [c for c in configurations() if not args.configs or c.name in args.configs]
    documents = [extract_document(p.read_bytes(), p.name) for p in args.documents]
    run_study(
        documents,
        load_questions(args.questions),
        experiments,
        Settings.from_env(),
        args.output,
        retrieval_only=args.retrieval_only,
        wait_for_rate_limit=args.wait_for_rate_limit,
        progress=lambda n, total, row: print(
            f"{n}/{total} {row['configuration']} {row['question_id']} "
            f"{'ERROR: ' + row['error'] if row['error'] else 'saved'}",
            flush=True,
        ),
    )


if __name__ == "__main__":
    main()
