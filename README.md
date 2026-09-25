# PDF RAG Chatbot

A retrieval-augmented generation application for asking grounded questions about a
10–20 page, text-based English PDF. It extracts page-aware text, creates overlapping
token chunks, embeds them locally, persists them in Chroma, and asks a Groq-hosted LLM
to answer using only retrieved excerpts.

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

Indexes and conversations are isolated by the PDF hash. The index identity also contains
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
```

Do not commit `.env`; it is ignored by Git.

## Using the app

1. Upload one or more qualifying PDFs in the sidebar.
2. Select the active document.
3. Choose chunk size and overlap, or retain the configured 32 tokens and 15%.
4. Select **Process selected PDF**. The first embedding-model load can take a while.
5. Ask questions in the chat box.
6. Open **Retrieved PDF excerpts** below an answer to inspect its evidence and scores.
7. Switch the active PDF to use its independent vector index and chat history.

The app refuses documents outside the page/language/text requirements. It also instructs
the LLM to say when the retrieved evidence does not contain an answer.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | required | Authenticates requests to Groq |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Generation and question-rewriting model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local Hugging Face embedding model |
| `EMBEDDING_DEVICE` | `auto` | Uses CUDA when available and otherwise falls back to CPU |
| `EMBEDDING_BATCH_SIZE` | `64` | Limits peak RAM/VRAM while indexing |
| `RAG_DATA_DIR` | OS user app-data directory | Optional persistent Chroma database location |

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
