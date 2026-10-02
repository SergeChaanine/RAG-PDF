"""Streamlit document library, grounded chat, and focused comparison study."""

from __future__ import annotations

import sys
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

import streamlit as st
from dotenv import dotenv_values, load_dotenv

from rag_pdf.catalog import DATABASES, EMBEDDING_MODELS, LLM_MODELS
from rag_pdf.config import ChunkConfig, Settings
from rag_pdf.document_processing import extract_document
from rag_pdf.embeddings import LocalEmbeddingModel
from rag_pdf.llm import GroqLanguageModel
from rag_pdf.models import ChatTurn
from rag_pdf.pdf_processing import document_hash
from rag_pdf.service import ChatRepository, answer_question, index_document, library_id
from rag_pdf.vector_store import create_store

ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_FILE)
st.set_page_config(page_title="Document RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False, max_entries=1)
def get_embedder(model_name, device, batch_size):
    return LocalEmbeddingModel(model_name, device, batch_size)


@st.cache_resource(show_spinner=False)
def get_store(backend, path):
    return create_store(backend, Path(path))


@st.cache_data(show_spinner=False, max_entries=32)
def parse_document(data, filename):
    return extract_document(data, filename)


def render_sources(sources):
    if sources:
        with st.expander("Sources and retrieved evidence"):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**[Source {i}] {source['filename']} · {source['location']}**")
                st.caption(f"Cosine similarity: {source['score']:.3f}")
                st.text(source["text"])


settings = Settings.from_env()
groq_key = settings.groq_api_key
if not groq_key:
    with suppress(FileNotFoundError, KeyError):
        groq_key = str(st.secrets.get("GROQ_API_KEY", ""))

st.title("Document RAG")
st.caption("Ask across your PDFs and Word documents, including native tables and statistics.")

with st.sidebar:
    st.header("Document library")
    uploaded = st.file_uploader(
        "Upload PDF or Word documents", type=["pdf", "docx"], accept_multiple_files=True
    )
    backend = st.selectbox("Vector database", DATABASES)
    choices = list(EMBEDDING_MODELS)
    embedding_model = st.selectbox(
        "Embedding model",
        choices,
        index=choices.index(settings.embedding_model) if settings.embedding_model in choices else 0,
        format_func=lambda name: EMBEDDING_MODELS[name],
    )
    # Reread this default: load_dotenv retains old process values across Streamlit reruns.
    default_answer_model = (
        dotenv_values(ENV_FILE).get("GROQ_MODEL") or settings.groq_model
    ).strip()
    llm_choices = list(dict.fromkeys([*LLM_MODELS, default_answer_model]))
    answer_model = st.selectbox(
        "Answer model",
        llm_choices,
        index=llm_choices.index(default_answer_model),
        format_func=lambda name: LLM_MODELS.get(name, name),
    )
    chunk_size = st.slider("Chunk size (tokens)", 64, 480, 256, step=8)
    overlap = st.selectbox("Chunk overlap", [15, 30], format_func=lambda x: f"{x}%")
    top_k = st.slider("Top K", 2, 10, 5)
    config = ChunkConfig(chunk_size, overlap)
    st.caption(f"Requested overlap: {config.overlap_tokens} tokens. Tables overlap by whole rows.")
    if not groq_key:
        st.info(
            "Set GROQ_API_KEY in .env to generate answers. Document processing works without it."
        )

    files = {}
    for file in uploaded or []:
        files.setdefault(document_hash(file.getvalue()), file)
    signature = (
        tuple(sorted((key, f.name) for key, f in files.items())),
        backend,
        embedding_model,
        chunk_size,
        overlap,
    )
    if st.button("Process all documents", type="primary", disabled=not files, width="stretch"):
        documents, summaries, failures = [], [], []
        try:
            with st.spinner("Loading embeddings and processing the library..."):
                store = get_store(backend, str(settings.data_dir))
                embedder = get_embedder(
                    embedding_model, settings.embedding_device, settings.embedding_batch_size
                )
                progress = st.progress(0)
                for i, file in enumerate(files.values()):
                    try:
                        document = parse_document(file.getvalue(), file.name)
                        summary = index_document(document, config, embedder, store)
                        documents.append(document)
                        summaries.append(summary)
                    except Exception as exc:
                        failures.append(f"{file.name}: {exc}")
                    progress.progress((i + 1) / len(files))
                st.session_state.prepared = {
                    "signature": signature,
                    "documents": documents,
                    "summaries": summaries,
                    "failures": failures,
                    "device": embedder.device_label,
                }
        except Exception as exc:
            st.error(f"Could not prepare the library: {exc}")

prepared = st.session_state.get("prepared", {})
matches = prepared.get("signature") == signature
documents = prepared.get("documents", []) if matches else []
summaries = prepared.get("summaries", []) if matches else []
ready = bool(documents)

chat_tab, library_tab = st.tabs(["Chat", "Library and extraction"])

with library_tab:
    if not ready:
        st.info(
            "Upload documents and select Process all documents to view their extracted content."
        )
    for failure in prepared.get("failures", []) if matches else []:
        st.error(failure)
    for document, summary in zip(documents, summaries, strict=True):
        with st.expander(
            f"{document.filename} · {summary.chunk_count} chunks · {len(document.tables)} tables"
        ):
            for warning in document.warnings:
                st.warning(warning)
            for page in document.pages:
                st.caption(page.location)
                st.text(page.text)
            for table in document.tables:
                st.markdown(f"**{table.location}**")
                st.caption(table.caption)
                import pandas as pd

                st.dataframe(
                    pd.DataFrame(
                        table.rows,
                        columns=[f"{i}. {header}" for i, header in enumerate(table.headers, 1)],
                    ),
                    width="stretch",
                )

with chat_tab:
    if ready:
        st.success(
            f"Searching all {len(documents)} processed documents · "
            f"{sum(s.chunk_count for s in summaries)} chunks · {prepared['device']}"
        )
        repository = ChatRepository(settings.data_dir / "chat")
        chat_key = library_id(documents)
        messages = repository.load(chat_key)
        if st.button("Clear conversation", disabled=not messages):
            repository.save(chat_key, [])
            st.rerun()
        st.caption(
            "History is saved locally for this document set, including after a restart. "
            "Upload the same files and process them to reopen it."
        )
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                render_sources(message.get("sources", []))
                if message.get("standalone_question"):
                    st.caption(f"Searched for: {message['standalone_question']}")
    else:
        messages = []
        st.info(
            "Process your document library to begin. Reprocess after changing uploads, "
            "database, embedding model, or chunk settings."
        )

    # Reserve space above the inline input for the turn generated on this run.
    new_turn = st.container()
    question = st.chat_input(
        "Ask about your documents, or follow up on an earlier answer",
        disabled=not (ready and groq_key),
    )
    if question:
        history = [ChatTurn(m["role"], m["content"]) for m in messages]
        messages.append({"role": "user", "content": question})
        repository.save(chat_key, messages)
        with new_turn, st.chat_message("user"):
            st.markdown(question)
        try:
            with new_turn, st.chat_message("assistant"), st.spinner("Searching your library..."):
                language_model = GroqLanguageModel(groq_key, answer_model)
                response = answer_question(
                    index_ids=[s.index_id for s in summaries],
                    documents=documents,
                    question=question,
                    history=history,
                    top_k=top_k,
                    embedder=get_embedder(
                        embedding_model, settings.embedding_device, settings.embedding_batch_size
                    ),
                    store=get_store(backend, str(settings.data_dir)),
                    language_model=language_model,
                )
                sources = [asdict(source) for source in response.sources]
                st.markdown(response.text)
                render_sources(sources)
                st.caption(f"Searched for: {response.standalone_question}")
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.text,
                        "sources": sources,
                        "standalone_question": response.standalone_question,
                        "model": answer_model,
                    }
                )
                repository.save(chat_key, messages)
        except Exception as exc:
            with new_turn:
                st.error(f"Could not answer: {exc}")
