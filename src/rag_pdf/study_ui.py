"""A small experiment runner and review table inside Streamlit."""

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_pdf.benchmark import configurations, run_study, summarize, validate_questions
from rag_pdf.document_processing import extract_document


def render_study(settings, groq_key, documents):
    st.subheader("Focused comparison")
    st.write(
        "Eleven configurations, changing one setting at a time. "
        "The overlap comparison is 15% versus 30%."
    )
    experiments = configurations()
    st.dataframe(pd.DataFrame([asdict(e) for e in experiments]), hide_index=True, width="stretch")
    mode = st.radio(
        "Study documents", ["Included synthetic sample", "Processed library"], horizontal=True
    )
    questions_file = st.file_uploader(
        "Reference questions (JSON; optional for the sample)", type=["json"], key="study_questions"
    )
    selected = st.multiselect(
        "Configurations to run",
        [e.name for e in experiments],
        default=[e.name for e in experiments],
    )
    retrieval_only = st.checkbox("Retrieval only (no answer API calls)")
    st.caption(
        "First use downloads embedding models. Results include answers, evidence, timing, "
        "and simple reference checks. Human review remains separate from automatic checks."
    )
    if st.button(
        "Run selected comparisons", disabled=not selected or (not retrieval_only and not groq_key)
    ):
        try:
            sample_dir = settings.data_dir / "samples"
            if mode == "Included synthetic sample":
                sample = sample_dir / "sample_report.pdf"
                if not sample.exists():
                    from rag_pdf.sample import create_sample

                    create_sample(sample_dir)
                study_documents = [extract_document(sample.read_bytes(), sample.name)]
            else:
                study_documents = documents
            if mode == "Processed library" and not questions_file:
                raise ValueError("Upload reference questions for your own library.")
            questions = (
                json.loads(questions_file.getvalue())
                if questions_file
                else json.loads((sample_dir / "sample_questions.json").read_text(encoding="utf-8"))
            )
            validate_questions(questions)
            output = settings.data_dir / "studies" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            progress, status = st.progress(0), st.empty()

            def update(done, total, row):
                progress.progress(min(done / total, 1.0))
                status.text(
                    f"{done}/{total}: {row['configuration']} / {row['question_id']}"
                    + (f" — {row['error']}" if row["error"] else "")
                )

            with st.spinner("Running the selected comparisons..."):
                rows = run_study(
                    study_documents,
                    questions,
                    [e for e in experiments if e.name in selected],
                    replace(settings, groq_api_key=groq_key),
                    output,
                    retrieval_only=retrieval_only,
                    progress=update,
                )
            st.session_state.study_rows = rows
            st.session_state.study_output = str(output)
        except Exception as exc:
            st.error(f"Study could not finish: {exc}")
    # Display the bundled completed run on first visit when present.
    bundled = settings.data_dir / "studies" / "sample"
    saved_results = bundled / "results.jsonl"
    if not saved_results.exists():
        bundled = Path(__file__).resolve().parents[2] / "study_results" / "sample"
        saved_results = bundled / "results.jsonl"
    load_saved = st.button("Load latest saved sample results", disabled=not saved_results.exists())
    if ("study_rows" not in st.session_state or load_saved) and saved_results.exists():
        st.session_state.study_rows = [
            json.loads(line)
            for line in (bundled / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        st.session_state.study_output = str(bundled)
    rows = st.session_state.get("study_rows", [])
    if rows:
        output = Path(st.session_state.study_output)
        summary = pd.DataFrame(summarize(rows))
        st.dataframe(summary, hide_index=True, width="stretch")
        st.bar_chart(summary.set_index("configuration")[["answer_s"]])
        st.caption("Answer latency in seconds. Failed runs are listed in the review table.")
        for name in (
            "report.html",
            "comparison.png",
            "results.csv",
            "report.md",
            "results.jsonl",
            "manifest.json",
        ):
            if (output / name).exists():
                st.download_button(
                    f"Download {name}", (output / name).read_bytes(), name, key=f"download_{name}"
                )
        st.subheader("Review answers")
        st.caption(
            "Correctness: 0 = incorrect, 0.5 = partial, 1 = correct. "
            "Citation support: 0 = unsupported, 1 = supported. Leave unreviewed cells blank."
        )
        frame = pd.DataFrame(rows)
        columns = [
            "configuration",
            "question_id",
            "question",
            "expected",
            "answer",
            "answer_check_pass",
            "correctness",
            "citation_support",
            "error",
        ]
        for column in columns:
            if column not in frame:
                frame[column] = ""
        for column in ("correctness", "citation_support"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        reviewed = st.data_editor(
            frame[columns],
            hide_index=True,
            width="stretch",
            disabled=[c for c in columns if c not in {"correctness", "citation_support"}],
            column_config={
                "correctness": st.column_config.NumberColumn(min_value=0, max_value=1, step=0.5),
                "citation_support": st.column_config.NumberColumn(min_value=0, max_value=1, step=1),
            },
        )
        st.download_button(
            "Download reviewed scores",
            reviewed.to_csv(index=False).encode(),
            "reviewed_results.csv",
        )
        choice = st.selectbox(
            "Inspect evidence for a result",
            range(len(rows)),
            format_func=lambda i: f"{rows[i]['configuration']} / {rows[i]['question_id']}",
        )
        st.json(rows[choice])
