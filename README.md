# Document RAG Chatbot

A retrieval-augmented generation application for asking grounded questions about selectable-text PDFs and modern Word (.docx) documents. It extracts page-aware PDF text and Word paragraphs and tables, creates overlapping token chunks, embeds them locally, persists vectors in Chroma or uses an in-memory test store, and asks a Groq-hosted LLM to answer using retrieved excerpts.

## Architecture

```text
PDF upload
   → validation and page-aware extraction (PyMuPDF)
   → recursive token chunking (LangChain)
   → local BGE embeddings (Sentence Transformers)
   → persistent cosine index (Chroma)

Question + document-specific chat history
   → standalone-question rewriting (Groq)
   → embedding and top-k retrieval
   → grounded answer with page citations (Groq)
```

Indexes and conversations are isolated by the uploaded file hash. The index identity also contains
the embedding model, chunk size, and overlap, so changing an experiment setting never
silently reuses incompatible vectors.

## Requirements

- Python 3.10 or newer
- A free [Groq API key](https://console.groq.com/keys)
- A 10–20 page PDF containing selectable English text (not a scanned image-only PDF)

The first run downloads `BAAI/bge-small-en-v1.5` from Hugging Face. Later runs use the
local model cache and the persistent Chroma index.

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env`, replace `replace_me` with the Groq API key, then start the application:

```powershell
python -m streamlit run app.py
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python -m streamlit run app.py
remember to turn off smart app control
```

Do not commit `.env`; it is ignored by Git.

## Using the app

1. Upload one or more PDFs or Word documents in the sidebar.
2. Select one or more active documents; retrieval searches all selected indexes.
3. Choose chunk size and overlap, or retain the configured 32 tokens and 15%.
4. Select **Process selected documents**. The first embedding-model load can take a while.
5. Ask questions in the chat box.
6. Open **Retrieved document excerpts** below an answer to inspect its evidence, source files, and scores.
7. Every file uses its own vector index. Chat history is kept for each active document set.

Word tables are converted into labeled row text so values remain associated with their column headings during retrieval. Word files are cited at document level; PDFs retain page citations. The app refuses documents without enough extractable English text. It also instructs
the LLM to say when the retrieved evidence does not contain an answer.

## Configuration

### Runtime experiments

Use the sidebar dropdowns to choose a Groq chat model and a local embedding model without
editing `.env`. Groq choices include GPT-OSS 20B (`openai/gpt-oss-20b`), GPT-OSS 120B
(`openai/gpt-oss-120b`), and Qwen 3.8 27B (`qwen/qwen3.8-27b`, preview). Embedding choices
include BGE Small English (`BAAI/bge-small-en-v1.5`), MiniLM L6 v2
(`sentence-transformers/all-MiniLM-L6-v2`), and BGE Base English
(`BAAI/bge-base-en-v1.5`). Changing the embedding model creates a separate index
automatically. Choose **Persistent Chroma** for approximate nearest-neighbor retrieval,
**Persistent SQLite** for an independent persistent database with exact cosine search, or
**In-memory test database** for temporary indexes. Each persistent backend has default and
test data profiles; in-memory data is cleared when the app process restarts.

After selecting documents, open **Compare models and vector databases**. Enter one test
question at a time and choose whether the expected behavior is to answer from the documents
or to abstain because the answer is absent. Answerable cases need a reference answer and are
scored with token F1; unknown-answer cases count explicit abstentions separately from
unsupported answers. Set the F1 threshold used for answerable-case pass/fail. Optionally name
the expected source file, then select chat models and vector databases. Missing indexes are
created automatically, with preparation time reported separately from query retrieval time.
The report compares retrieval and generation latency, token use, reference token F1, outcome
pass rate, and expected-source retrieval for each database/chat-model/embedding-model
combination. Its model selectors use the same named options as the sidebar dropdowns, and
results show both those names and their underlying model IDs. One Groq request is made per
combination, and a blank model response may be retried. Token F1 measures word overlap, not
factual correctness; inspect generated answers and sources. Repeat questions to build a small
evaluation set. Selecting another embedding model creates a separate index, allowing its
retrieval performance to be compared too.

| Environment variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | required | Authenticates requests to Groq |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Generation and question-rewriting model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local Hugging Face embedding model |
| `EMBEDDING_DEVICE` | `auto` | Uses CUDA when available and otherwise falls back to CPU |
| `EMBEDDING_BATCH_SIZE` | `64` | Limits peak RAM/VRAM while indexing |
| `RAG_DATA_DIR` | OS user app-data directory | Base directory for persistent Chroma and SQLite databases |

### NVIDIA GPU acceleration on Windows

The normal dependency installation can select a CPU-only PyTorch wheel. To use a
CUDA-capable NVIDIA GPU, install the project CUDA wheel after the base requirements:

```powershell
python -m pip install -r requirements-cuda.txt
```

Keep `EMBEDDING_DEVICE=auto` in `.env`. The app will use CUDA when PyTorch can access it
and otherwise fall back to CPU. After installation, restart Streamlit. The sidebar shows
the resolved embedding device after a PDF is processed.

After the first successful model download, later starts load it directly from the local
Hugging Face cache instead of waiting for online update checks.

Verify CUDA independently with:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Embeddings are generated and written to Chroma in batches. `EMBEDDING_BATCH_SIZE=64`
limits peak memory; reduce it to `32` if VRAM is constrained.

## Development checks

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest
```

The automated tests do not download the embedding model or call Groq.

## Current retrieval defaults

- Chunk size: 32 tokens (minimum 16)
- Dynamic overlap: 15%, calculated as 5 tokens at the default size
- Retrieved chunks: 5
- Similarity: cosine
- Embeddings: normalized BGE vectors
- Citations: original one-based PDF page numbers
