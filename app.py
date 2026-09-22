"""Streamlit entry point for the PDF RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep the app directly runnable without requiring an editable package install.
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

import streamlit as st
from dotenv import load_dotenv

from rag_pdf.config import ChunkConfig, Settings
from rag_pdf.embeddings import LocalEmbeddingModel
from rag_pdf.errors import RAGError
from rag_pdf.llm import GroqLanguageModel
from rag_pdf.models import ChatTurn, IndexSummary
from rag_pdf.pdf_processing import document_hash, extract_pdf
from rag_pdf.service import answer_question, build_index_id, index_document
from rag_pdf.vector_store import ChromaVectorStore

load_dotenv(Path(__file__).resolve().parent / ".env")
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str, device: str, batch_size: int) -> LocalEmbeddingModel:
    return LocalEmbeddingModel(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )


@st.cache_resource(show_spinner=False)
def get_store(path: str) -> ChromaVectorStore:
    return ChromaVectorStore(Path(path))


@st.cache_data(show_spinner=False, max_entries=4)
def parse_pdf(data: bytes, filename: str):
    return extract_pdf(data, filename)


def get_groq_key(settings: Settings) -> str:
    if settings.groq_api_key:
        return settings.groq_api_key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Retrieved PDF excerpts"):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**{number}. Page {source['page_number']} · "
                f"similarity {source['score']:.3f}**"
            )
            st.caption(source["text"])


settings = Settings.from_env()
groq_key = get_groq_key(settings)
store = get_store(str(settings.data_dir))

st.title("📄 PDF RAG Chatbot")
st.caption("Ask grounded questions about a 10–20 page, text-based English PDF.")

if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "chats" not in st.session_state:
    st.session_state.chats = {}

with st.sidebar:
    st.header("Document and retrieval")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Each PDF must contain 10–20 pages of extractable English text.",
    )
    chunk_size = st.slider("Chunk size (tokens)", 16, 480, 32, step=8)
    overlap_percent = st.slider("Chunk overlap", 5, 30, 15, step=5, format="%d%%")
    top_k = st.slider("Retrieved chunks", 2, 10, 5)
    chunk_config = ChunkConfig(chunk_size, overlap_percent)
    st.caption(f"Calculated overlap: {chunk_config.overlap_tokens} tokens")

    file_options = {}
    for uploaded in uploaded_files or []:
        data = uploaded.getvalue()
        label = f"{uploaded.name} · {document_hash(data)[:8]}"
        file_options[label] = uploaded

    selected_label = st.selectbox(
        "Active PDF",
        options=list(file_options),
        disabled=not file_options,
        placeholder="Upload a PDF first",
    )

selected_file = file_options.get(selected_label) if selected_label else None
active_index_id = ""
summary: IndexSummary | None = None
is_indexed = False

if selected_file is not None:
    selected_data = selected_file.getvalue()
    active_index_id = build_index_id(
        document_hash(selected_data), settings.embedding_model, chunk_config
    )
    summary = st.session_state.summaries.get(active_index_id)
    is_indexed = store.has_index(active_index_id)

    with st.sidebar:
        process_label = "Verify index" if is_indexed else "Process selected PDF"
        if st.button(process_label, type="primary", use_container_width=True):
            try:
                with st.spinner("Validating and extracting the PDF..."):
                    document = parse_pdf(selected_data, selected_file.name)
                with st.spinner("Loading the embedding model and building the index..."):
                    embedder = get_embedder(
                        settings.embedding_model,
                        settings.embedding_device,
                        settings.embedding_batch_size,
                    )
                    st.session_state.embedding_device = embedder.device_label
                    summary = index_document(document, chunk_config, embedder, store)
                    st.session_state.summaries[summary.index_id] = summary
                    is_indexed = True
                if summary.reused_existing_index:
                    st.success("The existing index was verified and reused.")
                else:
                    st.success("PDF processed successfully.")
            except (RAGError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The PDF could not be processed: {exc}")

        if is_indexed:
            st.success(f"Ready · {store.count(active_index_id)} chunks")
        else:
            st.info("Process this PDF before asking questions.")

        device_label = st.session_state.get("embedding_device")
        st.caption(
            f"Embedding compute: {device_label or settings.embedding_device.upper()}"
        )

        if not groq_key:
            st.warning("Add GROQ_API_KEY to `.env` to enable answers.")

if summary:
    first, second, third, fourth = st.columns(4)
    first.metric("Pages", summary.page_count)
    second.metric("Chunks", summary.chunk_count)
    third.metric("Characters", f"{summary.character_count:,}")
    fourth.metric("Language", summary.language.upper())

if not selected_file:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

chat_key = active_index_id
messages = st.session_state.chats.setdefault(chat_key, [])

left, right = st.columns([5, 1])
with left:
    st.subheader(selected_file.name)
with right:
    if st.button("Clear chat", disabled=not messages, use_container_width=True):
        st.session_state.chats[chat_key] = []
        st.rerun()

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

ready = is_indexed and bool(groq_key)
question = st.chat_input(
    "Ask a question about the active PDF",
    disabled=not ready,
)
if question:
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        ChatTurn(role=message["role"], content=message["content"])
        for message in messages[:-1]
    ]
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF..."):
                embedder = get_embedder(
                    settings.embedding_model,
                    settings.embedding_device,
                    settings.embedding_batch_size,
                )
                language_model = GroqLanguageModel(groq_key, settings.groq_model)
                response = answer_question(
                    index_id=active_index_id,
                    question=question,
                    history=history,
                    top_k=top_k,
                    embedder=embedder,
                    store=store,
                    language_model=language_model,
                )
            st.markdown(response.text)
            serialized_sources = [
                {
                    "page_number": source.page_number,
                    "score": source.score,
                    "text": source.text,
                }
                for source in response.sources
            ]
            render_sources(serialized_sources)
        messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "sources": serialized_sources,
                "standalone_question": response.standalone_question,
            }
        )
    except RAGError as exc:
        error_message = f"I could not complete the request: {exc}"
        with st.chat_message("assistant"):
            st.error(error_message)
        messages.append({"role": "assistant", "content": error_message, "sources": []})
