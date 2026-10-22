# Document RAG chatbot

Ask grounded, conversational questions across PDFs, Word `.docx`, Excel `.xlsx`, and
PowerPoint `.pptx` documents. Native text, tables, worksheet values, and PowerPoint chart
series are indexed locally; Groq generates answers with numbered citations identifying
the filename and source location.

## Start

```powershell
python -m pip install -r requirements.txt
# Only for a new setup; preserve your existing .env and API key.
Copy-Item .env.example .env
python -m streamlit run app.py
python -m streamlit run app.py --server.fileWatcherType none
```

Set `GROQ_API_KEY` in `.env`. Existing installations only need the dependency update
and a Streamlit restart. For development, install `requirements-dev.txt` too.

## USEK website crawl and chatbot

Version 0.2 adds **USEK Website** in the Streamlit sidebar. It reads your saved
`documents.jsonl` crawl export, re-cleans saved HTML, indexes webpages and PDFs together,
and answers with official source links and original PDF page numbers. The running
standalone crawler can continue unchanged; its existing exports are compatible.

```powershell
python -m pip install -r requirements.txt -r requirements-crawl.txt
python scripts/check_usek.py --crawl-dir "$env:LOCALAPPDATA/RAG-PDF/usek/full"
python scripts/prepare_usek.py --device cuda
python -m streamlit run pages/1_USEK_Website.py
```

The preparation command saves all usable sources, chunks, embeddings, and search indexes
before the presentation, using CUDA on a compatible GPU. It resumes interrupted builds.
The USEK page opens the completed package automatically in **Presentation mode**;
only query embeddings and answers are computed during chat. Disable that mode for
interactive loading/indexing. The check and indexing do not call Groq; answers require
internet and your existing `.env` key. Webpage sources activate additional system rules
for prompt injection, unsupported facts, deadlines, personal data, and account actions.
Downloaded content and indexes stay outside Git. See [the complete USEK workflow](docs/USEK_WEBSITE.md)
for a subset smoke test, the repository crawler, limitations, and presentation validation.

## Document library and conversation

1. Upload one or more PDF, DOCX, XLSX, or PPTX files. Identical uploads are deduplicated.
2. Choose the database, embedding model, chunk size, and overlap.
3. Select **Process all documents**. All successfully processed uploads are active;
   failed files are shown in **Chat** and **Library and extraction**, with an explicit
   searchable-file count, and excluded from retrieval.
4. Inspect the extracted text and tables in that tab before trusting a new layout.
5. Ask a question in **Chat**. Search combines semantic similarity with keyword
   matches in text, filenames, worksheet names, and column headers across every active
   document. Top K reserves one result per document when enough slots are available,
   reserves a second distinct result per document when there are at least twice as many
   slots as documents, then fills remaining slots by combined relevance, avoiding
   duplicate table chunks.
   An answer may cite several documents. If there are more documents than slots,
   the documents with the best combined matches are selected.
6. Ask a follow-up such as “How long is it?”. A rewrite step uses recent conversation
   turns to recover the subject before searching. Ambiguous references can trigger
   a clarification question instead of a search.

Conversations and their source excerpts are saved locally in SQLite, keyed by the
document set. Re-upload and process the same files after a restart to reopen the
conversation. Changing retrieval settings or the answer model does not erase it.
Changing the document set starts a separate conversation. **Clear conversation**
clears the saved history for that set. The last 20 turns are supplied to rewriting;
all turns remain visible in the saved conversation.

Answers follow the language of the original question: English questions receive English
answers, and Arabic questions receive Arabic answers, regardless of the source language or
the rewritten retrieval query. The model translates source evidence without changing the
facts or citations. Answers in the wrong language receive one corrective retry.
For retrieval, the resolved question is also translated between Arabic and English. Both
queries search every active document through semantic and keyword search, and their results
are merged before selecting Top K. This lets Arabic questions match English worksheet headers
and English questions match Arabic PDF text. In chat, follow-up rewriting and translation
share one short model request. Translation and search preparation use bounded in-memory
caches scoped by question, document names, and conversation context. Query embeddings are
batched, and the app reuses model clients and loaded embedding models. The first model load
is slower than subsequent questions. Chat shows elapsed answer time. These optimizations
do not change the answer language. Retrieval-only benchmarks skip model translation.
Standalone questions bypass conversation rewriting, including questions naming a project
that never appeared in history or filenames. The planner resolves references, not whether
document facts exist; missing evidence is determined after searching. Genuine ambiguous
follow-ups can still request clarification, without displaying a fictitious search query.

After updating from a version with semantic-only retrieval, select **Process all documents**
once to build the local keyword indexes. Existing vector indexes can still be reused.

This is a local single-user application. There is no multi-user authentication or
per-user storage isolation; do not deploy it as a shared service unchanged.

## Native text, tables, and Arabic

- PDF extraction uses PyMuPDF to identify tables and preserve page locations. Table
  cells are excluded from prose to avoid indexing the same values twice.
- Word extraction uses `python-docx` for paragraphs, headings, and tables. Citations
  use heading/paragraph/table locations; Word page numbers are not fabricated.
- Excel extraction reads cell values and cached formula results by worksheet. Each
  worksheet is represented as a table using its first row with multiple fields as the
  header; preceding report titles and notes are retained as prose and table context.
  Single-column sheets keep their first populated row as the header. Workbooks without
  worksheet dimension metadata are supported, with size limits checked during reading.
  Formulas without a saved result are retained as formula text and reported as a warning.
- PowerPoint extraction reads slide text, native tables, speaker notes, and data series
  from native charts. Slide numbers are retained in source locations.
- Table chunks contain whole rows with repeated column headers. Row overlap is
  bounded by the requested overlap budget, so actual overlap may be lower.
- Complete retrieved tables are expanded into answer context when they fit the
  context budget. A constrained calculator supports sums, means, extrema,
  differences, and percentage changes using validated source cell coordinates.
- Large tables that cannot be expanded are marked partial. Whole-table calculations
  are unavailable for those excerpts. Very wide rows that exceed the chunk size
  cause a clear processing error rather than silent truncation.
- The answer context budget is shared evenly across retrieved documents, so a large
  table in one file cannot crowd out the evidence from another file.

Supported inputs: native-text `.pdf`, `.docx`, `.xlsx`, and `.pptx`, up to 20 MB each;
PDFs up to 200 pages, Excel workbooks up to 50 sheets and 500,000 cells per sheet.
Arabic and mixed Arabic/English content is preserved as Unicode in extraction. Select
BGE-M3 or Qwen3 multilingual embeddings for Arabic retrieval; English embedding models
are not recommended for Arabic. The default embedding is BGE-M3. Image-only/scanned
PDFs, text embedded only in pictures, legacy `.doc`/`.xls`/`.ppt`, and OCR are not
supported. Excel skips leading report titles when identifying table headers. Unusual merged
headers, borderless PDF tables, and tables continuing across pages need extraction-preview
review; detected PDF tables are individual page tables, not guaranteed multi-page tables.

Answers involving statistics are instructed to use Markdown tables automatically, including
counts, percentages, averages, comparisons, trends, and statistical concepts. Tables use
descriptive headers, preserve units and time periods, and cite sources in each factual row.
Statistics may come from document prose or extracted tables; missing evidence still produces
the normal not-found response.

When the answer model identifies a useful visualization in a complete retrieved table,
it requests a chart using one-based source column numbers. The app validates those columns
and plots only values parsed from the cited table. Invalid or nonnumeric requests are
rejected. Charts appear in the conversation with the table's source location. The model
may decide that a table does not support a useful chart and answer without one. Charts
supplement the statistical answer table.
Normal document answers do not include plotting code or chart configuration JSON. Charts
are rendered only through the validated chart tool; when it cannot be used, the answer
keeps its table and explanation. Code is included only when the question requests it.

The **Statistical figures** setting offers **Automatic**, **Histogram**, **Box plot**,
**Summary cards**, **Scatter plot**, **Line chart**, **Bar chart**, and **Tables only**.
Automatic chooses a suitable format for the statistical question. A selected format is
used when the retrieved data supports it; figures supplement the answer table. Tables only
disables figures while retaining table calculations. Histograms show frequencies, box plots
show quartiles and outliers using 1.5-IQR whiskers, and summary cards report the observation
count, mean, median, minimum, maximum, and sample standard deviation.
These three formats use one numeric column with at least two observations from a complete
source table. Totals and blank values are excluded, selected row numbers can restrict the
period, and invalid text values are rejected. The source, observation count, and exclusions
are shown with the figure. Statistical figures persist with the conversation. PDF prose
without an extracted numeric table still receives a text/table answer.

## Choices

| Component | Options |
|---|---|
| Vector storage | Chroma, Qdrant local mode, LanceDB embedded |
| Embeddings | BGE-M3 multilingual (default), Qwen3-Embedding-0.6B multilingual, BGE-small English, E5-base-v2 English |
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
synthetic sample or your own supported document library with a reference-question JSON file.
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
| `EMBEDDING_MODEL` | `BAAI/bge-m3` |
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
