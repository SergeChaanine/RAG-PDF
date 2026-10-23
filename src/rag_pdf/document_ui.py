"""Streamlit document library, grounded chat, and focused comparison study."""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

import streamlit as st

from rag_pdf.catalog import DATABASES, EMBEDDING_MODELS, LLM_MODELS, PIPELINE_VERSION
from rag_pdf.config import ChunkConfig, Settings
from rag_pdf.document_processing import extract_document
from rag_pdf.embeddings import LocalEmbeddingModel
from rag_pdf.llm import GroqLanguageModel
from rag_pdf.models import ChatTurn
from rag_pdf.pdf_processing import document_hash
from rag_pdf.service import ChatRepository, answer_question, index_document, library_id
from rag_pdf.vector_store import create_store
from rag_pdf.visuals import VISUALIZATION_MODES, distribution_spec, summary_metrics


@st.cache_resource(show_spinner=False, max_entries=1)
def get_embedder(model_name, device, batch_size):
    return LocalEmbeddingModel(model_name, device, batch_size)


@st.cache_resource(show_spinner=False)
def get_store(backend, path):
    return create_store(backend, Path(path))


@st.cache_resource(show_spinner=False, max_entries=4)
def get_language_model(api_key, model):
    return GroqLanguageModel(api_key, model)


@st.cache_data(show_spinner=False, max_entries=32)
def parse_document(data, filename, pipeline_version):
    return extract_document(data, filename)


def render_sources(sources):
    if sources:
        with st.expander("Sources and retrieved evidence"):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**[Source {i}] {source['filename']} · {source['location']}**")
                method = source.get("retrieval_method", "semantic")
                if method == "keyword":
                    st.caption("Retrieved by keyword search")
                else:
                    st.caption(f"Retrieval: {method} · Cosine similarity: {source['score']:.3f}")
                st.text(source["text"])


def render_charts(charts):
    for chart in charts or []:
        st.markdown(f"**{chart['title']}**")
        if chart["chart_type"] in {"histogram", "box", "summary"}:
            st.caption(chart["value_label"])
            if chart["chart_type"] == "summary":
                columns = st.columns(3)
                for i, (label, value) in enumerate(summary_metrics(chart)):
                    columns[i % 3].metric(label, value)
            else:
                st.vega_lite_chart(distribution_spec(chart), width="stretch")
            arabic = chart.get("language") == "Arabic"
            count = len(chart["values"])
            st.caption(f"{count} مشاهدات" if arabic else f"{count} observations")
            if chart.get("skipped_rows"):
                st.caption(
                    "تم استبعاد صفوف الإجمالي والقيم الفارغة."
                    if arabic
                    else "Total rows and blank values were excluded."
                )
            st.caption(f"Source {chart['source_number']} · {chart['source_location']}")
            continue
        import pandas as pd

        frame = pd.DataFrame({"__x": chart["labels"], **chart["series"]})
        y_columns = chart["y_labels"]
        if chart["chart_type"] == "line":
            st.line_chart(frame, x="__x", y=y_columns, x_label=chart["x_label"])
        elif chart["chart_type"] == "bar":
            st.bar_chart(frame, x="__x", y=y_columns, x_label=chart["x_label"])
        else:
            st.scatter_chart(frame, x="__x", y=y_columns, x_label=chart["x_label"])
        st.caption(f"Source {chart['source_number']} · {chart['source_location']}")


@dataclass(frozen=True)
class _SavedUpload:
    name: str
    data: bytes

    def getvalue(self):
        return self.data


def _remember_uploads():
    st.session_state.document_uploads = tuple(
        _SavedUpload(file.name, file.getvalue())
        for file in st.session_state.get("document_upload_widget", [])
    )


def render_documents():
    settings = Settings.from_env()
    groq_key = settings.groq_api_key
    if not groq_key:
        with suppress(FileNotFoundError, KeyError):
            groq_key = str(st.secrets.get("GROQ_API_KEY", ""))

    st.title("Document RAG")
    st.caption(
        "Ask across PDFs, Word, Excel, and PowerPoint files, including native tables "
        "and statistics."
    )

    options = st.session_state.get("document_options", {})

    with st.sidebar:
        st.header("Document library")
        uploaded = st.file_uploader(
            "Upload PDF, Word, Excel, or PowerPoint files",
            type=["pdf", "docx", "xlsx", "pptx"],
            accept_multiple_files=True,
            key="document_upload_widget",
            on_change=_remember_uploads,
        )
        if uploaded:
            st.session_state.document_uploads = tuple(
                _SavedUpload(file.name, file.getvalue()) for file in uploaded
            )
        saved_uploads = st.session_state.get("document_uploads", ())
        if saved_uploads and not uploaded:
            st.caption(f"Using {len(saved_uploads)} documents retained in this session.")
            if st.button("Clear saved uploads"):
                st.session_state.pop("document_uploads", None)
                st.session_state.pop("prepared", None)
                st.rerun()
        backend = st.selectbox(
            "Vector database",
            DATABASES,
            index=DATABASES.index(options.get("backend", DATABASES[0])),
            key="document_backend",
        )
        choices = list(EMBEDDING_MODELS)
        embedding_model = st.selectbox(
            "Embedding model",
            choices,
            index=choices.index(options.get("embedding_model", settings.embedding_model))
            if options.get("embedding_model", settings.embedding_model) in choices
            else 0,
            key="document_embedding_model",
            format_func=lambda name: EMBEDDING_MODELS[name],
        )
        st.caption(
            "For Arabic and mixed Arabic/English files, select a multilingual embedding model "
            "(BGE-M3 or Qwen3). Reprocess after changing it. Scanned/image-only PDFs need OCR "
            "and are not currently read."
        )
        llm_choices = list(dict.fromkeys([*LLM_MODELS, settings.groq_model]))
        answer_model = st.selectbox(
            "Answer model",
            llm_choices,
            index=llm_choices.index(options.get("answer_model", settings.groq_model)),
            key="document_answer_model",
            format_func=lambda name: LLM_MODELS.get(name, name),
        )
        visualization = st.selectbox(
            "Statistical figures",
            list(VISUALIZATION_MODES),
            index=list(VISUALIZATION_MODES).index(options.get("visualization", "auto")),
            key="document_visualization",
            format_func=VISUALIZATION_MODES.get,
            help="Automatic chooses a suitable figure for statistical answers. Select a preferred "
            "format or Tables only. Figures require suitable numeric data in a retrieved table.",
        )
        chunk_size = st.slider(
            "Chunk size (tokens)",
            64,
            480,
            options.get("chunk_size", 256),
            step=8,
            key="document_chunk_size",
        )
        overlap = st.selectbox(
            "Chunk overlap",
            [15, 30],
            index=[15, 30].index(options.get("overlap", 15)),
            format_func=lambda x: f"{x}%",
            key="document_overlap",
        )
        top_k = st.slider("Top K", 2, 10, options.get("top_k", 5), key="document_top_k")
        st.session_state.document_options = {
            "backend": backend,
            "embedding_model": embedding_model,
            "answer_model": answer_model,
            "visualization": visualization,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
        }
        st.caption(
            "Search uses Arabic and English queries, meaning, keywords, and worksheet names. "
            "Top K reserves one result per document, or two when enough slots are available."
        )
        config = ChunkConfig(chunk_size, overlap)
        st.caption(
            f"Requested overlap: {config.overlap_tokens} tokens. Tables overlap by whole rows."
        )
        if not groq_key:
            st.info(
                "Set GROQ_API_KEY in .env to generate answers. "
                "Document processing works without it."
            )

        files = {}
        for file in saved_uploads:
            files.setdefault(document_hash(file.getvalue()), file)
        signature = (
            tuple(sorted((key, f.name) for key, f in files.items())),
            backend,
            embedding_model,
            chunk_size,
            overlap,
            PIPELINE_VERSION,
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
                            document = parse_document(file.getvalue(), file.name, PIPELINE_VERSION)
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
                f"{document.filename} · {summary.chunk_count} chunks · "
                f"{len(document.tables)} tables"
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
        active_failures = prepared.get("failures", []) if matches else []
        if active_failures:
            st.warning(
                f"Only {len(documents)} of {len(files)} uploaded documents are searchable. "
                "The following files failed processing and are excluded from answers."
            )
            for failure in active_failures:
                st.error(failure)
        if ready:
            st.success(
                f"Searching all {len(documents)} processed documents · "
                f"{sum(s.chunk_count for s in summaries)} chunks · {prepared['device']}"
            )
            st.caption(
                "Searchable files: " + ", ".join(document.filename for document in documents)
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
                    render_charts(message.get("charts", []))
                    searched_question = message.get("standalone_question", "")
                    if searched_question and not searched_question.startswith("CLARIFY:"):
                        st.caption(f"Searched for: {message['standalone_question']}")
                    if "elapsed_seconds" in message:
                        st.caption(f"Answered in {message['elapsed_seconds']:.1f} seconds")
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
                with (
                    new_turn,
                    st.chat_message("assistant"),
                    st.spinner("Searching your library..."),
                ):
                    started = time.perf_counter()
                    language_model = get_language_model(groq_key, answer_model)
                    response = answer_question(
                        index_ids=[s.index_id for s in summaries],
                        documents=documents,
                        question=question,
                        history=history,
                        top_k=top_k,
                        embedder=get_embedder(
                            embedding_model,
                            settings.embedding_device,
                            settings.embedding_batch_size,
                        ),
                        store=get_store(backend, str(settings.data_dir)),
                        language_model=language_model,
                        visualization=visualization,
                    )
                    sources = [asdict(source) for source in response.sources]
                    elapsed = time.perf_counter() - started
                    st.markdown(response.text)
                    render_sources(sources)
                    render_charts(response.charts)
                    if response.standalone_question:
                        st.caption(f"Searched for: {response.standalone_question}")
                    st.caption(f"Answered in {elapsed:.1f} seconds")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.text,
                            "sources": sources,
                            "charts": list(response.charts),
                            "standalone_question": response.standalone_question,
                            "model": answer_model,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    repository.save(chat_key, messages)
            except Exception as exc:
                with new_turn:
                    st.error(f"Could not answer: {exc}")
