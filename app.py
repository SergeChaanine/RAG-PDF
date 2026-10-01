"""Streamlit entry point for the PDF RAG chatbot."""

from __future__ import annotations

import sys
import time
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
from rag_pdf.evaluation import score_answer
from rag_pdf.llm import GroqLanguageModel
from rag_pdf.memory_store import InMemoryVectorStore
from rag_pdf.models import ChatTurn
from rag_pdf.pdf_processing import document_hash, extract_document
from rag_pdf.service import answer_question, build_index_id, index_document
from rag_pdf.sqlite_store import SQLiteVectorStore
from rag_pdf.vector_store import ChromaVectorStore

load_dotenv(Path(__file__).resolve().parent / ".env")
st.set_page_config(page_title="Document RAG Chatbot", page_icon="📄", layout="wide")


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


@st.cache_resource(show_spinner=False)
def get_memory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@st.cache_resource(show_spinner=False)
def get_sqlite_store(path: str) -> SQLiteVectorStore:
    return SQLiteVectorStore(Path(path))


DATABASE_BACKENDS = [
    "Persistent Chroma (default dataset)",
    "Persistent Chroma (test dataset)",
    "Persistent SQLite (default dataset)",
    "Persistent SQLite (test dataset)",
    "In-memory test database",
]

DATABASE_IDENTITIES = {
    "Persistent Chroma (default dataset)": "chroma-default",
    "Persistent Chroma (test dataset)": "chroma-test",
    "Persistent SQLite (default dataset)": "sqlite-default",
    "Persistent SQLite (test dataset)": "sqlite-test",
    "In-memory test database": "memory",
}


def get_backend_store(database_backend: str, data_dir: Path):
    identity = DATABASE_IDENTITIES[database_backend]
    if identity == "memory":
        return get_memory_store(), identity
    if identity.startswith("sqlite"):
        sqlite_name = "sqlite" if identity == "sqlite-default" else "sqlite-test"
        return (
            get_sqlite_store(str(data_dir.parent / sqlite_name / "rag_vectors.sqlite3")),
            identity,
        )
    chroma_path = data_dir if identity == "chroma-default" else data_dir / "test"
    return get_store(str(chroma_path)), identity


@st.cache_data(show_spinner=False, max_entries=4)
def parse_document(data: bytes, filename: str):
    return extract_document(data, filename)


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
    with st.expander("Retrieved document excerpts"):
        for number, source in enumerate(sources, start=1):
            filename = source.get("filename", "document")
            location = (
                filename
                if filename.lower().endswith(".docx")
                else f"{filename} - page {source['page_number']}"
            )
            st.markdown(f"**{number}. {location} - similarity {source['score']:.3f}**")
            st.caption(source["text"])


settings = Settings.from_env()
groq_key = get_groq_key(settings)

st.title("Document RAG Chatbot")
st.caption("Ask grounded questions across selected PDFs and Word documents.")

if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "model_comparison_results" not in st.session_state:
    st.session_state.model_comparison_results = []

with st.sidebar:
    st.header("Documents and retrieval")
    groq_models = {
        "GPT-OSS 20B (fast, default)": "openai/gpt-oss-20b",
        "GPT-OSS 120B (higher capacity)": "openai/gpt-oss-120b",
        "Qwen 3.8 27B (preview)": "qwen/qwen3.8-27b",
    }
    if settings.groq_model not in groq_models.values():
        groq_models[f"Configured model ({settings.groq_model})"] = settings.groq_model
    groq_labels = list(groq_models)
    default_groq_label = next(
        label for label, model_id in groq_models.items() if model_id == settings.groq_model
    )
    selected_groq_label = st.selectbox(
        "Groq chat model",
        groq_labels,
        index=groq_labels.index(default_groq_label),
        key="groq_model_choice",
        help="Qwen 3.8 is a preview model and may be changed or removed by Groq.",
    )
    groq_model = groq_models[selected_groq_label]

    embedding_models = {
        "BGE Small English (default, smaller)": "BAAI/bge-small-en-v1.5",
        "MiniLM L6 v2 (compact alternative)": "sentence-transformers/all-MiniLM-L6-v2",
        "BGE Base English (larger)": "BAAI/bge-base-en-v1.5",
    }
    if settings.embedding_model not in embedding_models.values():
        embedding_models[f"Configured model ({settings.embedding_model})"] = (
            settings.embedding_model
        )
    embedding_labels = list(embedding_models)
    default_embedding_label = next(
        label
        for label, model_id in embedding_models.items()
        if model_id == settings.embedding_model
    )
    selected_embedding_label = st.selectbox(
        "Embedding model",
        embedding_labels,
        index=embedding_labels.index(default_embedding_label),
        key="embedding_model_choice",
        help="Changing this model requires reprocessing documents to create compatible indexes.",
    )
    embedding_model = embedding_models[selected_embedding_label]
    database_backend = st.selectbox(
        "Vector database",
        DATABASE_BACKENDS,
        help=(
            "SQLite uses exact cosine search and suits small experiments. Test datasets "
            "are separate from the default data. In-memory data is cleared on restart."
        ),
    )
    uploaded_files = st.file_uploader(
        "Upload PDFs or Word documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Supports text-based PDFs and modern Word (.docx) files, including tables.",
    )
    chunk_size = st.slider("Chunk size (tokens)", 16, 480, 32, step=8)
    overlap_percent = st.slider("Chunk overlap", 5, 30, 15, step=5, format="%d%%")
    top_k = st.slider("Top K", 2, 10, 5)
    chunk_config = ChunkConfig(chunk_size, overlap_percent)
    st.caption(f"Calculated overlap: {chunk_config.overlap_tokens} tokens")

    file_options = {}
    for uploaded in uploaded_files or []:
        data = uploaded.getvalue()
        label = f"{uploaded.name} | {document_hash(data)[:8]}"
        file_options[label] = uploaded

    selected_labels = st.multiselect(
        "Active documents",
        options=list(file_options),
        placeholder="Upload and select one or more documents",
    )

store, database_identity = get_backend_store(database_backend, settings.data_dir)
if database_identity == "memory":
    with st.sidebar:
        st.info("Test database selected. Its vectors are temporary and disappear on restart.")

selected_files = [file_options[label] for label in selected_labels]
active_index_ids = [
    build_index_id(document_hash(file.getvalue()), embedding_model, chunk_config)
    for file in selected_files
]
indexed_ids = [index_id for index_id in active_index_ids if store.has_index(index_id)]
all_indexed = bool(active_index_ids) and len(indexed_ids) == len(active_index_ids)

embedding_config = (
    embedding_model,
    settings.embedding_device,
    settings.embedding_batch_size,
)
embedding_ready = st.session_state.get("embedding_config") == embedding_config

if selected_files:
    with st.sidebar:
        process_label = (
            "Prepare selected documents"
            if all_indexed and not embedding_ready
            else "Process selected documents"
        )
        if st.button(process_label, type="primary", use_container_width=True):
            try:
                embedder = get_embedder(
                    embedding_model,
                    settings.embedding_device,
                    settings.embedding_batch_size,
                )
                st.session_state.embedding_device = embedder.device_label
                st.session_state.embedding_config = embedding_config
                embedding_ready = True
                for selected_file in selected_files:
                    try:
                        with st.spinner(f"Processing {selected_file.name}..."):
                            document = parse_document(selected_file.getvalue(), selected_file.name)
                            summary = index_document(document, chunk_config, embedder, store)
                            st.session_state.summaries[summary.index_id] = summary
                        st.success(f"{selected_file.name}: ready | {summary.chunk_count} chunks")
                    except (RAGError, ValueError) as exc:
                        st.error(f"{selected_file.name}: {exc}")
                indexed_ids = [i for i in active_index_ids if store.has_index(i)]
                all_indexed = len(indexed_ids) == len(active_index_ids)
            except Exception as exc:
                st.error(f"The embedding model could not be prepared: {exc}")

        if all_indexed and embedding_ready:
            st.success(
                f"Ready | {len(active_index_ids)} documents, "
                f"{sum(store.count(i) for i in active_index_ids)} chunks"
            )
        elif all_indexed:
            st.info("Saved indexes found. Prepare the selected documents before chatting.")
        else:
            st.info(
                "Process all selected documents before asking questions "
                f"({len(indexed_ids)}/{len(active_index_ids)} indexed)."
            )

        device_label = st.session_state.get("embedding_device")
        st.caption(f"Embedding compute: {device_label or settings.embedding_device.upper()}")
        if not groq_key:
            st.warning("Add GROQ_API_KEY to `.env` to enable answers.")

if selected_files and all_indexed:
    summaries = [st.session_state.summaries.get(index_id) for index_id in active_index_ids]
    summaries = [summary for summary in summaries if summary]
    if summaries:
        first, second, third = st.columns(3)
        first.metric("Active documents", len(selected_files))
        second.metric("Chunks", sum(summary.chunk_count for summary in summaries))
        third.metric("Characters", f"{sum(summary.character_count for summary in summaries):,}")

if not selected_files:
    st.info("Upload and select one or more documents in the sidebar to begin.")
    st.stop()

chat_key = f"{database_identity}|" + "|".join(sorted(active_index_ids))
messages = st.session_state.chats.setdefault(chat_key, [])

left, right = st.columns([5, 1])
with left:
    st.subheader(", ".join(file.name for file in selected_files))
with right:
    if st.button("Clear chat", disabled=not messages, use_container_width=True):
        st.session_state.chats[chat_key] = []
        st.rerun()

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

ready = all_indexed and embedding_ready and bool(groq_key)
question = st.chat_input(
    "Ask a question across the active documents",
    disabled=not ready,
)
if question:
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        ChatTurn(role=message["role"], content=message["content"]) for message in messages[:-1]
    ]
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the active documents..."):
                embedder = get_embedder(
                    embedding_model,
                    settings.embedding_device,
                    settings.embedding_batch_size,
                )
                language_model = GroqLanguageModel(groq_key, groq_model)
                response = answer_question(
                    index_id=active_index_ids,
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
                    "filename": source.filename,
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


with st.expander("Compare models and vector databases"):
    st.caption(
        "Run the same test case across selected chat models and vector databases. "
        "Choose whether the answer should be in the documents or should be unknown. "
        "Known-answer cases report lexical token F1 against your reference; this is a "
        "word-overlap proxy, not a factuality judge. Unknown-answer cases count correct "
        "abstentions separately from unsupported answers. Retrieval and generation times "
        "are reported separately. First-time database setup/indexing is reported separately."
    )
    with st.form("chat_model_comparison"):
        comparison_question = st.text_input("Test question")
        expected_behavior = st.radio(
            "Expected behavior",
            ["Answer is in the documents", "Should say it cannot find the answer"],
            horizontal=True,
        )
        should_abstain = expected_behavior == "Should say it cannot find the answer"
        reference_answer = ""
        if not should_abstain:
            reference_answer = st.text_area("Expected answer")
        f1_threshold = st.slider(
            "Minimum token F1 to count an answer as passing",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            disabled=should_abstain,
        )
        expected_source = st.text_input(
            "Expected source filename (optional)",
            help="Checks whether retrieval returned this exact filename.",
        )
        comparison_labels = st.multiselect(
            "Chat models available in the sidebar",
            options=list(groq_models),
            default=[selected_groq_label],
        )
        comparison_embedding_labels = st.multiselect(
            "Embedding models available in the sidebar",
            options=list(embedding_models),
            default=[selected_embedding_label],
            help="Each embedding model gets its own compatible index and retrieval run.",
        )
        comparison_databases = st.multiselect(
            "Vector databases to compare",
            options=DATABASE_BACKENDS,
            default=[database_backend],
            help=(
                "Missing indexes are created automatically for this document set and "
                "embedding model. In-memory indexes are recreated after app restarts."
            ),
        )
        run_comparison = st.form_submit_button("Run comparison", disabled=not ready)

    comparison_context = "|".join(sorted(document_hash(file.getvalue()) for file in selected_files))
    if run_comparison:
        if not comparison_question.strip():
            st.warning("Enter a test question.")
        elif not should_abstain and not reference_answer.strip():
            st.warning("Enter the expected answer for an answerable question.")
        elif not comparison_labels or not comparison_embedding_labels or not comparison_databases:
            st.warning("Select chat models, embedding models, and at least one vector database.")
        else:
            preparation_seconds = {}
            prepared_stores = {}
            preparation_errors = {}
            results = []
            for embedding_label in comparison_embedding_labels:
                embedding_id = embedding_models[embedding_label]
                try:
                    embedder = get_embedder(
                        embedding_id,
                        settings.embedding_device,
                        settings.embedding_batch_size,
                    )
                    for backend in comparison_databases:
                        candidate_store, candidate_identity = get_backend_store(
                            backend, settings.data_dir
                        )
                        key = (candidate_identity, embedding_id)
                        prep_start = time.perf_counter()
                        candidate_index_ids = []
                        for selected_file in selected_files:
                            index_id = build_index_id(
                                document_hash(selected_file.getvalue()),
                                embedding_id,
                                chunk_config,
                            )
                            if not candidate_store.has_index(index_id):
                                document = parse_document(
                                    selected_file.getvalue(), selected_file.name
                                )
                                index_document(document, chunk_config, embedder, candidate_store)
                            candidate_index_ids.append(index_id)
                        preparation_seconds[key] = round(time.perf_counter() - prep_start, 3)
                        prepared_stores[key] = (candidate_store, candidate_index_ids, embedder)
                except Exception as exc:
                    for backend in comparison_databases:
                        preparation_errors[(DATABASE_IDENTITIES[backend], embedding_id)] = str(exc)

            for embedding_label in comparison_embedding_labels:
                embedding_id = embedding_models[embedding_label]
                try:
                    embedder = get_embedder(
                        embedding_id,
                        settings.embedding_device,
                        settings.embedding_batch_size,
                    )
                    query_embedding = embedder.embed_query(comparison_question.strip())
                except Exception as exc:
                    st.error(f"Could not load or query embedding model {embedding_label}: {exc}")
                    continue

                for backend in comparison_databases:
                    _, candidate_identity = get_backend_store(backend, settings.data_dir)
                    key = (candidate_identity, embedding_id)
                    if key not in prepared_stores:
                        for model_label in comparison_labels:
                            results.append(
                                {
                                    "Database": backend,
                                    "Chat model": model_label,
                                    "Chat model ID": groq_models[model_label],
                                    "Embedding model": embedding_label,
                                    "Embedding model ID": embedding_id,
                                    "Question": comparison_question.strip(),
                                    "Expected behavior": expected_behavior,
                                    "Context": comparison_context,
                                    "Outcome": "Database preparation failed",
                                    "Correct outcome": False,
                                    "Abstained": None,
                                    "Reference token F1": None,
                                    "Retrieval seconds": None,
                                    "Index preparation seconds": None,
                                    "Generation seconds": None,
                                    "Total tokens": None,
                                    "Expected source retrieved": "not scored",
                                    "Answer": preparation_errors.get(key, "Index unavailable"),
                                    "Sources": "",
                                }
                            )
                        continue

                    candidate_store, index_ids, _ = prepared_stores[key]
                    retrieval_start = time.perf_counter()
                    retrieved_sources = [
                        source
                        for index_id in index_ids
                        for source in candidate_store.search(index_id, query_embedding, top_k)
                    ]
                    retrieved_sources.sort(key=lambda source: source.score, reverse=True)
                    retrieved_sources = retrieved_sources[:top_k]
                    retrieval_seconds = time.perf_counter() - retrieval_start
                    sources_text = ", ".join(
                        dict.fromkeys(source.filename for source in retrieved_sources)
                    )
                    expected_found = (
                        any(
                            source.filename.casefold() == expected_source.strip().casefold()
                            for source in retrieved_sources
                        )
                        if expected_source.strip()
                        else "not scored"
                    )
                    for model_label in comparison_labels:
                        model_id = groq_models[model_label]
                        language_model = GroqLanguageModel(groq_key, model_id)
                        generation_start = time.perf_counter()
                        try:
                            answer_text = language_model.answer(
                                comparison_question.strip(), retrieved_sources
                            )
                            generation_seconds = time.perf_counter() - generation_start
                            score = score_answer(
                                answer_text,
                                expected_answer=reference_answer,
                                should_abstain=should_abstain,
                                f1_threshold=f1_threshold,
                            )
                            results.append(
                                {
                                    "Database": backend,
                                    "Chat model": model_label,
                                    "Chat model ID": model_id,
                                    "Embedding model": embedding_label,
                                    "Embedding model ID": embedding_id,
                                    "Question": comparison_question.strip(),
                                    "Expected behavior": expected_behavior,
                                    "Context": comparison_context,
                                    "Outcome": score["outcome_label"],
                                    "Correct outcome": score["outcome_correct"],
                                    "Abstained": score["abstained"],
                                    "Reference token F1": (
                                        round(score["token_f1"], 3)
                                        if score["token_f1"] is not None
                                        else None
                                    ),
                                    "Retrieval seconds": round(retrieval_seconds, 3),
                                    "Index preparation seconds": preparation_seconds[key],
                                    "Generation seconds": round(generation_seconds, 2),
                                    "Total tokens": language_model.last_usage["total_tokens"],
                                    "Expected source retrieved": expected_found,
                                    "Answer": answer_text,
                                    "Sources": sources_text,
                                }
                            )
                        except RAGError as exc:
                            results.append(
                                {
                                    "Database": backend,
                                    "Chat model": model_label,
                                    "Chat model ID": model_id,
                                    "Embedding model": embedding_label,
                                    "Embedding model ID": embedding_id,
                                    "Question": comparison_question.strip(),
                                    "Expected behavior": expected_behavior,
                                    "Context": comparison_context,
                                    "Outcome": "Request failed",
                                    "Correct outcome": False,
                                    "Abstained": None,
                                    "Reference token F1": None,
                                    "Retrieval seconds": round(retrieval_seconds, 3),
                                    "Index preparation seconds": preparation_seconds[key],
                                    "Generation seconds": round(
                                        time.perf_counter() - generation_start, 2
                                    ),
                                    "Total tokens": 0,
                                    "Expected source retrieved": expected_found,
                                    "Answer": str(exc),
                                    "Sources": sources_text,
                                }
                            )
            st.session_state.model_comparison_results.extend(results)

    comparison_results = [
        result
        for result in st.session_state.model_comparison_results
        if result.get("Context") == comparison_context
    ]
    if comparison_results:
        aggregate_results = []
        configs = dict.fromkeys(
            (result["Database"], result["Embedding model"], result["Chat model"])
            for result in comparison_results
        )
        for backend, embedding_label, model_label in configs:
            subset = [
                result
                for result in comparison_results
                if (result["Database"], result["Embedding model"], result["Chat model"])
                == (backend, embedding_label, model_label)
            ]

            def numeric(rows: list[dict], key: str) -> list[float]:
                return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

            f1_values = numeric(subset, "Reference token F1")
            retrieval_values = numeric(subset, "Retrieval seconds")
            generation_values = numeric(subset, "Generation seconds")
            token_values = numeric(subset, "Total tokens")
            passed = sum(bool(r.get("Correct outcome")) for r in subset)
            aggregate_results.append(
                {
                    "Vector database": backend,
                    "Chat model": model_label,
                    "Chat model ID": subset[0]["Chat model ID"],
                    "Embedding model": embedding_label,
                    "Embedding model ID": subset[0]["Embedding model ID"],
                    "Cases": len(subset),
                    "Correct outcomes": f"{passed}/{len(subset)}",
                    "Outcome pass rate": round(passed / len(subset), 3),
                    "Avg. retrieval seconds": (
                        round(sum(retrieval_values) / len(retrieval_values), 3)
                        if retrieval_values
                        else None
                    ),
                    "Avg. generation seconds": (
                        round(sum(generation_values) / len(generation_values), 2)
                        if generation_values
                        else None
                    ),
                    "Avg. total tokens": (
                        round(sum(token_values) / len(token_values), 1) if token_values else None
                    ),
                    "Avg. reference token F1": (
                        round(sum(f1_values) / len(f1_values), 3) if f1_values else None
                    ),
                }
            )
        st.subheader("Results by database and model")
        st.dataframe(aggregate_results, hide_index=True, width="stretch")
        st.caption(
            "Outcome pass rate follows the selected rule: explicit abstention for unknown "
            "cases, or non-abstention plus the selected token F1 threshold for known answers."
        )
        st.subheader("Per-question results")
        hidden = {"Answer", "Sources", "Context"}
        st.dataframe(
            [
                {key: value for key, value in result.items() if key not in hidden}
                for result in comparison_results
            ],
            hide_index=True,
            width="stretch",
        )
        if st.button("Clear comparison results"):
            st.session_state.model_comparison_results = [
                result
                for result in st.session_state.model_comparison_results
                if result.get("Context") != comparison_context
            ]
            st.rerun()
        for result in comparison_results:
            with st.expander(
                f"{result['Database']} · {result['Chat model']} · "
                f"{result['Embedding model']} answer and sources"
            ):
                st.markdown(result["Answer"])
                st.caption(f"Retrieved files: {result['Sources'] or 'No sources found'}")
