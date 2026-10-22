"""Streamlit workflow for testing a saved USEK crawl without fetching the website."""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from rag_pdf.catalog import DATABASES, EMBEDDING_MODELS, LLM_MODELS
from rag_pdf.config import ChunkConfig, Settings
from rag_pdf.embeddings import LocalEmbeddingModel
from rag_pdf.llm import GroqLanguageModel
from rag_pdf.models import ChatTurn
from rag_pdf.presentation import active_presentation, load_prepared_website
from rag_pdf.reranking import LocalReranker
from rag_pdf.service import ChatRepository, answer_question, library_id
from rag_pdf.vector_store import create_store
from rag_pdf.website import index_website, load_crawl_dataset, official_source_url


@st.cache_resource(show_spinner=False, max_entries=1)
def get_website_embedder(name, device, batch_size):
    return LocalEmbeddingModel(name, device, batch_size)


@st.cache_resource(show_spinner=False)
def get_website_store(backend, root):
    return create_store(backend, Path(root))


@st.cache_resource(show_spinner=False, max_entries=4)
def get_website_llm(api_key, name):
    return GroqLanguageModel(api_key, name)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_presentation_runtime(path, manifest_stamp, device, batch_size):
    prepared = load_prepared_website(path)
    embedder = LocalEmbeddingModel(
        prepared.manifest["embedding_model"], device, batch_size, local_files_only=True
    )
    # Pay model/CUDA startup and index loading once, before accepting the first question.
    vector = embedder.embed_queries(["How do I register at USEK?"])[0]
    hits = prepared.store.search(prepared.index.index_id, vector, 1)
    reranker = LocalReranker(device=device, local_files_only=True)
    reranker.rank("How do I register at USEK?", hits)
    return prepared, embedder, reranker


def render_website_sources(sources):
    if not sources:
        return
    with st.expander("Sources and retrieved evidence"):
        for number, source in enumerate(sources, 1):
            st.write(f"[Source {number}] {source['filename']} · {source['location']}")
            try:
                url = official_source_url(source.get("source_url", ""))
            except ValueError:
                url = ""
            if url:
                st.link_button(f"Open official source {number}", url)
            if source.get("fetched_at"):
                st.caption(f"Downloaded: {source['fetched_at']} (not a publication date)")
            st.caption(f"Retrieval: {source.get('retrieval_method', 'semantic')}")
            st.text(source["text"])


def render_presentation(settings, key, active):
    with st.sidebar:
        package_path = st.text_input(
            "Prepared presentation package",
            str(active or settings.data_dir / "presentations" / "usek-full"),
        )
        names = list(dict.fromkeys([*LLM_MODELS, settings.groq_model]))
        answer_name = st.selectbox(
            "Answer model",
            names,
            index=names.index(settings.groq_model),
            format_func=lambda name: LLM_MODELS.get(name, name),
            key="prepared_answer_model",
        )
        top_k = st.slider("Top K", 2, 12, 8, key="prepared_top_k")
        st.caption("The saved search index is ready. Answers use Groq and require internet.")
    try:
        path = Path(package_path).expanduser().resolve()
        stamp = (path / "manifest.json").stat().st_mtime_ns
        with st.spinner("Opening saved indexes and warming up the query model..."):
            prepared, embedder, reranker = get_presentation_runtime(
                str(path), stamp, settings.embedding_device, settings.embedding_batch_size
            )
    except Exception as exc:
        st.error(f"Could not open the prepared presentation: {exc}")
        st.info("Run scripts/prepare_usek.py first, or select a completed package.")
        return
    dataset, index = prepared.dataset, prepared.index
    counts = dataset.summary()["source_types"]
    columns = st.columns(3)
    columns[0].metric("Searchable sources", index.document_count)
    columns[1].metric("Webpages", counts.get("html", 0))
    columns[2].metric("PDF documents", counts.get("pdf", 0))
    st.success(f"Ready · {index.chunk_count:,} saved chunks · {embedder.device_label}")
    with st.expander("Prepared dataset details"):
        st.write(f"Prepared: {prepared.manifest['created_at']}")
        st.write(f"Embedding model: {prepared.manifest['embedding_model']}")
        st.write(f"Package: {prepared.path}")
        st.caption(
            f"{len(dataset.issues)} records excluded during preparation. "
            "Answers cover the usable saved sources; crawl failures and PDFs "
            "without native text are not included."
        )
        for issue in dataset.issues[:100]:
            st.caption(issue)
        for warning in index.warnings[:100]:
            st.caption(warning)
    with st.expander("Preview a saved source"):
        selected = st.selectbox(
            "Saved source",
            range(len(dataset.documents)),
            format_func=lambda i: dataset.documents[i].filename,
            key="prepared_source",
        )
        document = dataset.documents[selected]
        st.link_button("Open official source", document.source_url)
        page = st.selectbox(
            "Page or section",
            range(len(document.pages)),
            format_func=lambda i: document.pages[i].location,
            key="prepared_page",
        )
        st.text(document.pages[page].text)
    repository = ChatRepository(settings.data_dir / "website_chat")
    chat_key = "usek:" + library_id(dataset.documents)
    messages = repository.load(chat_key)
    if st.button("Clear website conversation", disabled=not messages):
        repository.save(chat_key, [])
        st.rerun()
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_website_sources(message.get("sources", []))
    if not key:
        st.info("Set GROQ_API_KEY in .env to generate answers from the prepared sources.")
    question = st.chat_input(
        "Ask about USEK, or follow up on an earlier answer",
        disabled=not key,
        key="prepared_question",
    )
    if question:
        history = [ChatTurn(message["role"], message["content"]) for message in messages]
        messages.append({"role": "user", "content": question})
        repository.save(chat_key, messages)
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant"), st.spinner("Searching saved USEK sources..."):
                started = time.perf_counter()
                result = answer_question(
                    index_ids=[index.index_id],
                    documents=dataset.documents,
                    question=question,
                    history=history,
                    top_k=top_k,
                    embedder=embedder,
                    store=prepared.store,
                    language_model=get_website_llm(key, answer_name),
                    visualization="off",
                    document_diversity=False,
                    planning_filenames=["USEK university website and linked public PDFs"],
                    reranker=reranker,
                )
                sources = [asdict(source) for source in result.sources]
                st.markdown(result.text)
                render_website_sources(sources)
                st.caption(f"Answered in {time.perf_counter() - started:.1f} seconds")
                messages.append(
                    {
                        "role": "assistant",
                        "content": result.text,
                        "sources": sources,
                        "standalone_question": result.standalone_question,
                    }
                )
                repository.save(chat_key, messages)
        except Exception as exc:
            st.error(f"Could not answer: {exc}")


def render_website():
    st.set_page_config(page_title="USEK Website RAG", page_icon="🌐", layout="wide")
    settings = Settings.from_env()
    key = settings.groq_api_key
    if not key:
        with suppress(FileNotFoundError, KeyError):
            key = str(st.secrets.get("GROQ_API_KEY", ""))
    st.title("USEK Website")
    st.caption("Ask about registration, admissions, financial aid, and other saved USEK sources.")
    active = active_presentation(settings.data_dir)
    if st.sidebar.toggle("Presentation mode", value=active is not None):
        render_presentation(settings, key, active)
        return
    with st.sidebar:
        crawl_path = st.text_input(
            "Crawl directory or documents.jsonl", str(settings.data_dir / "usek" / "full")
        )
        limit = st.number_input("Sources to load (0 = all)", min_value=0, value=0, step=20)
        clean_html = st.checkbox("Clean saved webpage HTML again", value=True)
        input_signature = (crawl_path, limit, clean_html)
        if st.button("Load crawl snapshot", width="stretch"):
            st.session_state.pop("website_prepared", None)
            st.session_state.pop("website_dataset", None)
            try:
                with st.spinner("Reading the saved crawl and cleaning webpages..."):
                    dataset = load_crawl_dataset(
                        crawl_path, limit=int(limit), reextract_html=clean_html
                    )
                    st.session_state.website_dataset = (input_signature, dataset)
            except Exception as exc:
                st.error(f"Could not load the crawl: {exc}")
        backend = st.selectbox("Vector database", DATABASES, key="website_backend")
        embedding_names = list(EMBEDDING_MODELS)
        embedding_name = st.selectbox(
            "Embedding model",
            embedding_names,
            index=embedding_names.index(settings.embedding_model)
            if settings.embedding_model in embedding_names
            else 0,
            format_func=EMBEDDING_MODELS.get,
            key="website_embedding",
        )
        answer_names = list(dict.fromkeys([*LLM_MODELS, settings.groq_model]))
        answer_name = st.selectbox(
            "Answer model",
            answer_names,
            index=answer_names.index(settings.groq_model),
            format_func=lambda name: LLM_MODELS.get(name, name),
            key="website_answer_model",
        )
        size = st.slider("Chunk size (tokens)", 64, 480, 256, step=8, key="website_chunks")
        overlap = st.selectbox(
            "Chunk overlap", [15, 30], format_func=lambda n: f"{n}%", key="website_overlap"
        )
        top_k = st.slider("Top K", 2, 12, 8, key="website_top_k")
        st.caption(
            "BGE-M3 and Qwen3 support multilingual retrieval. Indexing is local; "
            "a Groq key is needed only for answers."
        )
    saved = st.session_state.get("website_dataset")
    if not saved or saved[0] != input_signature:
        st.info(
            "Select your crawl directory and click Load crawl snapshot. "
            "You can test a completed checkpoint while the crawl continues."
        )
        return
    dataset = saved[1]
    counts = dataset.summary()["source_types"]
    columns = st.columns(3)
    columns[0].metric("Searchable sources", len(dataset.documents))
    columns[1].metric("Webpages", counts.get("html", 0))
    columns[2].metric("PDF documents", counts.get("pdf", 0))
    if limit:
        st.info(
            f"Testing a subset: up to {int(limit)} usable sources from "
            f"{dataset.record_count} export records. Set Sources to load to 0 for all."
        )
    states = dataset.crawl_report.get("states", {})
    if states:
        st.caption(
            "Last saved crawl report: "
            + ", ".join(f"{state}: {count}" for state, count in states.items())
        )
        if any(states.get(state, 0) for state in ("queued", "fetching", "failed", "blocked")):
            st.warning(
                "This saved crawl has pending or unsuccessful URLs. "
                "Answers cover the loaded sources; university-wide completeness is unverified."
            )
    with st.expander("Snapshot details and extraction issues"):
        st.write(f"Export: {dataset.path}")
        st.caption(f"Snapshot SHA-256: {dataset.fingerprint}")
        for issue in dataset.issues[:100]:
            st.warning(issue)
        warnings = [(d.filename, warning) for d in dataset.documents for warning in d.warnings]
        for filename, warning in warnings[:100]:
            st.caption(f"{filename}: {warning}")
        st.caption(f"{len(dataset.issues)} rejected records; {len(warnings)} extraction warnings.")
    if not dataset.documents:
        st.error("No searchable content was found in this export.")
        return
    config = ChunkConfig(size, overlap)
    signature = (input_signature, dataset.fingerprint, backend, embedding_name, size, overlap)
    index_root = str(settings.data_dir / "website_indexes")
    if st.button("Build or reuse website index", type="primary"):
        st.session_state.pop("website_prepared", None)
        try:
            with st.spinner("Loading embeddings and building the website search index..."):
                embedder = get_website_embedder(
                    embedding_name, settings.embedding_device, settings.embedding_batch_size
                )
                store = get_website_store(backend, index_root)
                progress = st.progress(0.0)
                summary = index_website(
                    dataset,
                    config,
                    embedder,
                    store,
                    progress=lambda done, total: progress.progress(done / total),
                )
                progress.progress(1.0)
                st.session_state.website_prepared = (signature, summary)
        except Exception as exc:
            st.error(f"Could not prepare the website index: {exc}")
    prepared = st.session_state.get("website_prepared")
    ready = bool(prepared and prepared[0] == signature)
    chat_tab, preview_tab = st.tabs(["Chat", "Source preview"])
    with preview_tab:
        selected = st.selectbox(
            "Saved source",
            range(len(dataset.documents)),
            format_func=lambda i: dataset.documents[i].filename,
        )
        document = dataset.documents[selected]
        st.link_button("Open official source", document.source_url)
        if document.fetched_at:
            st.caption(f"Downloaded: {document.fetched_at}")
        page = st.selectbox(
            "Page or section",
            range(len(document.pages)),
            format_func=lambda i: document.pages[i].location,
        )
        st.text(document.pages[page].text)
    with chat_tab:
        messages, repository, chat_key = [], None, ""
        if ready:
            summary = prepared[1]
            st.success(
                f"Searching {summary.document_count} sources · {summary.chunk_count} chunks"
                + (" · Reused index" if summary.reused else "")
            )
            for warning in summary.warnings:
                st.warning(warning)
            repository = ChatRepository(settings.data_dir / "website_chat")
            chat_key = "usek:" + library_id(dataset.documents)
            messages = repository.load(chat_key)
            if st.button("Clear website conversation", disabled=not messages):
                repository.save(chat_key, [])
                st.rerun()
            for message in messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    render_website_sources(message.get("sources", []))
        else:
            st.info(
                "Build the website index to begin. Rebuild after changing the loaded sources "
                "or indexing settings."
            )
        if not key:
            st.info(
                "Set GROQ_API_KEY in .env to generate answers. "
                "Loading and indexing work without it."
            )
        question = st.chat_input(
            "Ask about USEK, or follow up on an earlier answer",
            disabled=not (ready and key),
            key="website_question",
        )
        if question:
            history = [ChatTurn(m["role"], m["content"]) for m in messages]
            messages.append({"role": "user", "content": question})
            repository.save(chat_key, messages)
            with st.chat_message("user"):
                st.markdown(question)
            try:
                with st.chat_message("assistant"), st.spinner("Searching saved USEK sources..."):
                    started = time.perf_counter()
                    result = answer_question(
                        index_ids=[prepared[1].index_id],
                        documents=dataset.documents,
                        question=question,
                        history=history,
                        top_k=top_k,
                        embedder=get_website_embedder(
                            embedding_name, settings.embedding_device, settings.embedding_batch_size
                        ),
                        store=get_website_store(backend, index_root),
                        language_model=get_website_llm(key, answer_name),
                        visualization="off",
                        document_diversity=False,
                        planning_filenames=["USEK university website and linked public PDFs"],
                    )
                    sources = [asdict(source) for source in result.sources]
                    st.markdown(result.text)
                    render_website_sources(sources)
                    st.caption(f"Answered in {time.perf_counter() - started:.1f} seconds")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": result.text,
                            "sources": sources,
                            "standalone_question": result.standalone_question,
                        }
                    )
                    repository.save(chat_key, messages)
            except Exception as exc:
                st.error(f"Could not answer: {exc}")
