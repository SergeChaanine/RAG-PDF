# Document RAG chatbot

Ask grounded, conversational questions across a library of PDFs and Word `.docx`
documents. Native text and tables are indexed locally; Groq generates answers with
numbered citations identifying the filename and source location.

## Start

```powershell
python -m pip install -r requirements.txt
# Only for a new setup; preserve your existing .env and API key.
Copy-Item .env.example .env
python -m streamlit run app.py
```

Set `GROQ_API_KEY` in `.env`. Existing installations only need the dependency update
and a Streamlit restart. For development, install `requirements-dev.txt` too.

## Document library and conversation

1. Upload one or more PDF or DOCX files. Identical uploads are deduplicated.
2. Choose the database, embedding model, chunk size, and overlap.
3. Select **Process all documents**. All successfully processed uploads are active;
   failed files are shown in **Library and extraction** and excluded from retrieval.
4. Inspect the extracted text and tables in that tab before trusting a new layout.
5. Ask a question in **Chat**. Search runs over every active document and globally
   ranks the matching chunks. An answer may cite several documents.
6. Ask a follow-up such as “How long is it?”. A rewrite step uses recent conversation
   turns to recover the subject before searching. Ambiguous references can trigger
   a clarification question instead of a search.

Conversations and their source excerpts are saved locally in SQLite, keyed by the
document set. Re-upload and process the same files after a restart to reopen the
conversation. Changing retrieval settings or the answer model does not erase it.
Changing the document set starts a separate conversation. **Clear conversation**
clears the saved history for that set. The last 20 turns are supplied to rewriting;
all turns remain visible in the saved conversation.

This is a local single-user application. There is no multi-user authentication or
per-user storage isolation; do not deploy it as a shared service unchanged.

## Native text and tables

- PDF extraction uses PyMuPDF to identify tables and preserve page locations. Table
  cells are excluded from prose to avoid indexing the same values twice.
- Word extraction uses `python-docx` for paragraphs, headings, and tables. Citations
  use heading/paragraph/table locations; Word page numbers are not fabricated.
- Table chunks contain whole rows with repeated column headers. Row overlap is
  bounded by the requested overlap budget, so actual overlap may be lower.
- Complete retrieved tables are expanded into answer context when they fit the
  context budget. A constrained calculator supports sums, means, extrema,
  differences, and percentage changes using validated source cell coordinates.
- Large tables that cannot be expanded are marked partial. Whole-table calculations
  are unavailable for those excerpts. Very wide rows that exceed the chunk size
  cause a clear processing error rather than silent truncation.

Supported inputs: native `.pdf` and `.docx`, up to 20 MB each; PDFs up to 200 pages.
Scans, image-only charts, and legacy `.doc` files are outside the initial scope.
Numeric tables and short documents are accepted without an English-only language
filter. English-only embeddings remain best suited to English documents.
Unusual merged headers, borderless PDF tables, and tables continuing across pages
need extraction-preview review; automatically detected PDF tables are individual
page tables, not guaranteed reconstructions of multi-page tables.

## Choices

| Component | Options |
|---|---|
| Vector storage | Chroma, Qdrant local mode, LanceDB embedded |
| Embeddings | BGE-small English, E5-base-v2, BGE-M3, Qwen3-Embedding-0.6B |
| Answer LLM | GPT-OSS 20B, GPT-OSS 120B, Qwen 3.8 27B |
| Default chunking | 256 tokens with 15% overlap |
| Overlap choices | 15% and 30% |
| Default retrieval | 5 chunks across the library |

The answer models use the existing Groq API key. Qwen 3.8 is a preview model and
availability depends on the account/provider. Llama 3.3 was not available to the
account used for this study, so the third model is Qwen.

E5 passages and queries receive their required prefixes; BGE-small receives its
query instruction; BGE-M3 receives none; Qwen queries receive a retrieval instruction.
Embeddings are normalized and all stores use cosine similarity. Inputs exceeding
the selected embedding model's token limit are rejected, not silently truncated.

Changing embeddings/chunking requires processing again. Compatible complete indexes
are reused. A completion manifest prevents partially written indexes from being
mistaken for finished indexes. Different backends live in separate folders. Old
PDF-only collections are left in place and are not reused by the new pipeline.

## Focused comparison study

Run comparisons with the command-line runner described below. It supports the
synthetic sample or your own PDF/DOCX library with a reference-question JSON file.
The Streamlit interface contains Chat and Library and extraction tabs.

| Configuration | Change from reference |
|---|---|
| reference | Chroma, BGE-small, 256 tokens, 15%, GPT-OSS 20B |
| embedding_e5 | E5-base-v2 |
| embedding_bge_m3 | BGE-M3 |
| embedding_qwen | Qwen3-Embedding-0.6B |
| database_qdrant | Qdrant local |
| database_lancedb | LanceDB |
| chunks_128 | 128 tokens |
| chunks_480 | 480 tokens |
| overlap_30 | 30% overlap |
| llm_gpt_oss_120b | GPT-OSS 120B |
| llm_qwen | Qwen 3.8 27B |

The synthetic Aurora report has three pages, two native tables, and 22 reference
questions covering text, table lookups, calculations, a follow-up, and missing
information. The sample is deliberately fictional. Multi-document retrieval and
DOCX parsing are checked separately in the functional tests.

All study configurations use the same BGE-small tokenizer for chunk boundaries.
The selected model still checks the real encoded input length. Vectors are reused
across databases within a run, and the three answer-model configurations reuse
identical saved evidence. The follow-up rewrite is frozen across configurations.
Successful question runs are saved immediately; the CLI can resume a matching run.
Failures remain explicit and are retried on resume. Use a new output folder if
documents, questions, or configurations change.

To run outside Streamlit:

```powershell
python scripts/create_sample.py
$env:PYTHONPATH = 'src'
python -m rag_pdf.benchmark --documents "$env:LOCALAPPDATA/RAG-PDF/samples/sample_report.pdf" --questions "$env:LOCALAPPDATA/RAG-PDF/samples/sample_questions.json"
python scripts/study_report.py "$env:LOCALAPPDATA/RAG-PDF/studies/sample"
```

Paths above assume the default Windows application-data directory. For custom
documents, supply their paths with `--documents`, a matching JSON with `--questions`,
and a new `--output` directory. `--configs reference database_qdrant` runs a subset;
`--retrieval-only` avoids answer API calls. Model downloads can take time on first use.
Use `--wait-for-rate-limit` when resuming a run after Groq quota errors: it honors
the provider's retry window, with a bounded wait of 30 minutes per request. Quota
waiting is recorded separately and excluded from answer latency. Cached embedding
models can be used without downloads by setting `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` after the first successful download.

Outputs: `results.jsonl` (full answers and sources), `results.csv` (review sheet),
`manifest.json` (versions/settings/input hashes), `summary.json`, and `report.md`.
The report script additionally writes `comparison.png` and a standalone `report.html`.
`paired_summary.json` compares only questions that succeeded for every configuration.
Retrieval checks remain available from cached evidence when answer generation fails.

The measured snapshot is included under `study_results/sample`. Open `report.html`
locally for charts, findings, comparison tables, and cases to inspect. It contains
232 successful answers across 11 configurations; ten Qwen-embedding answer requests
hit Groq's daily token quota. All configurations have complete cached retrieval
measurements. The report also compares the 12 questions answered successfully by
every configuration. Open the saved report to inspect these results.

Automatic reference checks are **not human correctness scores**. Inspect the answer
and its sources, then fill correctness (0, 0.5, 1) and citation support (0, 1) in the
review CSV. A correct citation number alone does not prove
support. A sample question schema is provided in `benchmarks/sample_questions.json`.

These are single-pass, exploratory measurements on a small synthetic corpus.
Qdrant local and unindexed LanceDB perform exact search; Chroma uses HNSW. These are
local application measurements, not server-scale database benchmarks. Retrieval
time includes query embedding. Reused index/vector timings are not cold builds.

## Configuration and storage

| Environment variable | Default |
|---|---|
| `GROQ_API_KEY` | Required for answers |
| `GROQ_MODEL` | `openai/gpt-oss-20b` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DEVICE` | `auto` (CUDA if available, otherwise CPU) |
| `EMBEDDING_BATCH_SIZE` | `64` (capped at 16 for BGE-M3 and Qwen) |
| `RAG_DATA_DIR` | `%LOCALAPPDATA%/RAG-PDF` on Windows; user data directory elsewhere |

Generated indexes, conversations, samples, and UI studies are kept in the writable
application-data directory, outside the repository. Embedding models use the Hugging
Face cache. For a supported NVIDIA/PyTorch installation the sidebar reports CUDA;
otherwise embedding runs on CPU. Keep `.env` out of version control.

## Verification

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check . --no-cache
python -m pytest -q -p no:cacheprovider
```

Tests use in-memory documents, temporary local databases, and deterministic fake
embeddings/LLMs. They do not download models or call Groq. Live study results are
separate from these functional tests.
