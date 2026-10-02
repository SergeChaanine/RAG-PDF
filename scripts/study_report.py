"""Export an existing study to a standalone HTML report and scientific plots."""

import argparse
import base64
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pdf.benchmark import (  # noqa: E402
    export_results,
    paired_results,
    study_findings,
    summarize,
)


def build_report(output):
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    export_results(rows, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize(rows)
    names = [
        r["configuration"]
        + (
            f" ({r['successful']}/{r['attempted']} answers)"
            if r["successful"] < r["attempted"]
            else ""
        )
        for r in summary
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
    axes[0].barh(names, [r["answer_s"] for r in summary], color="#356F97")
    axes[0].set_xlabel("Mean answer time (seconds)")
    axes[0].set_title("Answer latency")
    checks = []
    for row in summary:
        value = row["answer_checks"]
        checks.append(
            100 * int(value.split("/")[0]) / int(value.split("/")[1]) if "/" in value else 0
        )
    axes[1].barh(names, checks, color="#4D8B72")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Passed automatic reference checks (%)")
    axes[1].set_title("Reference checks, not human correctness")
    top1 = []
    for row in summary:
        value = row["top1_evidence_checks"]
        top1.append(
            100 * int(value.split("/")[0]) / int(value.split("/")[1]) if "/" in value else 0
        )
    axes[2].barh(names, top1, color="#997236")
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("First raw hit containing all anchors (%)")
    axes[2].set_title("Top-1 ranking diagnostic")
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Focused RAG comparison", fontsize=14)
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    encoded = base64.b64encode((output / "comparison.png").read_bytes()).decode()
    sections = []
    for row in summary:
        sections.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row[k]))}</td>"
                for k in (
                    "configuration",
                    "successful",
                    "attempted",
                    "evidence_checks",
                    "answer_checks",
                    "top1_evidence_checks",
                    "citation_checks",
                )
            )
            + f"<td>{row['retrieval_ms']:.1f}</td><td>{row['answer_s']:.2f}</td></tr>"
        )
    failures = [
        r
        for r in rows
        if r.get("error")
        or r.get("answer_check_pass") is False
        or (r.get("category") != "unanswerable" and r.get("citation_ids_valid") is False)
    ]
    examples = "".join(
        f"<details><summary>{html.escape(r['configuration'] + ' / ' + r['question_id'])}</summary>"
        f"<p><b>Question:</b> {html.escape(r['question'])}</p>"
        f"<p><b>Expected:</b> {html.escape(r['expected'])}</p>"
        f"<pre>{html.escape(r.get('error') or r['answer'])}</pre></details>"
        for r in failures
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Focused RAG comparison</title><style>
body{font:17px/1.55 system-ui,sans-serif;color:#172c3a;
max-width:1320px;margin:40px auto;padding:0 24px}
h1,h2{line-height:1.2}h1{font-size:36px}h2{margin-top:36px}img{width:100%}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:9px;border-bottom:1px solid #d6e0e6}
td{overflow-wrap:anywhere}.table-scroll{overflow-x:auto}
th{text-align:left;background:#edf3f7}pre{white-space:pre-wrap;font:inherit;background:#f5f7f9;padding:16px}
details{margin:12px 0;padding:12px;border:1px solid #d6e0e6}summary{cursor:pointer}
@media print{body{margin:0;max-width:none}details{break-inside:avoid}
img{max-height:450px;object-fit:contain}}
</style><body><h1>Focused RAG comparison</h1>
<h2>Method</h2><p>The reference uses Chroma, BGE-small, 256-token chunks, 15% overlap,
five retrieved chunks, and GPT-OSS 20B. Each variation changes one factor. The overlap comparison
is 15% versus 30%. Answer-model variants use identical saved evidence. Complete tables are expanded
after retrieval; calculations use validated table cells. Temperature is zero.</p>
<p>All database comparisons run locally. Qdrant and LanceDB use exact search here, while Chroma
uses HNSW. Query embedding is included in retrieval time.
Cached index timings are not cold-build measurements. Recorded provider-quota waits are excluded
from answer latency. Top-1 checks inspect saved first hits; no extra API calls were needed.</p>
<h2>Measured results</h2>"""
    introduction = (
        f"<p>{len(manifest.get('experiments', []))} configurations and "
        f"{len(manifest.get('questions', []))} prepared questions. Documents: "
        + html.escape(", ".join(d["filename"] for d in manifest.get("documents", [])))
        + ".</p>"
    )
    devices = sorted({r["device"] for r in rows if r.get("device")})
    if devices:
        introduction += "<p>Embedding device: " + html.escape("; ".join(devices)) + ".</p>"
    if any(d["filename"] == "sample_report.pdf" for d in manifest.get("documents", [])):
        introduction += (
            "<p>The included Aurora sample is fictional: three pages of selectable text, "
            "two native tables, statistics, a follow-up, and an unanswerable question.</p>"
        )
    page = page.replace("<h2>Method</h2>", introduction + "<h2>Method</h2>")
    page += (
        '<img alt="Answer latency and reference check rates" '
        f'src="data:image/png;base64,{encoded}">'
    )
    page += (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join(
            f"<th>{h}</th>"
            for h in [
                "Configuration",
                "Successful",
                "Attempted",
                "Evidence checks",
                "Answer checks",
                "Top-1 anchors",
                "Valid citation IDs",
                "Retrieval ms",
                "Answer s",
            ]
        )
        + "</tr></thead><tbody>"
        + "".join(sections)
        + "</tbody></table></div>"
    )
    points = study_findings(rows, summary, manifest)
    page += (
        "<h2>Observed findings</h2><ul>"
        + "".join(f"<li>{html.escape(point)}</li>" for point in points)
        + "</ul>"
    )
    page += '<h2>Configurations</h2><div class="table-scroll"><table><thead><tr>'
    keys = ["name", "database", "embedding", "chunk_size", "overlap", "llm", "top_k"]
    page += "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr></thead><tbody>"
    page += "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(e[k]))}</td>" for k in keys) + "</tr>"
        for e in manifest.get("experiments", [])
    )
    page += "</tbody></table></div>"
    shared, paired = paired_results(rows)
    if len(shared) < len(manifest.get("questions", [])):
        page += (
            f"<h2>Equal-size answer comparison</h2><p>{len(shared)} questions succeeded for "
            "every configuration: " + html.escape(", ".join(shared)) + ". "
            "This paired subset avoids comparing answer-check percentages with unequal "
            "denominators. Retrieval diagnostics above include cached evidence even when "
            "the answer API failed. The subset may not cover every question category.</p>"
        )
        page += "<table><thead><tr><th>Configuration</th><th>Answer checks</th>"
        page += "<th>Valid citation IDs</th><th>Mean answer s</th></tr></thead><tbody>"
        page += (
            "".join(
                f"<tr><td>{html.escape(r['configuration'])}</td><td>{r['answer_checks']}</td>"
                f"<td>{r['citation_checks']}</td><td>{r['answer_s']:.2f}</td></tr>"
                for r in paired
            )
            + "</tbody></table>"
        )
    page += """<h2>How to interpret these results</h2><p>Automatic answer checks match prepared
patterns. They can miss correct paraphrases and cannot prove factual correctness. Evidence checks
look for reference strings in the supplied context.
Valid citation numbers do not establish citation support.
Manual scores are kept separately in the review table.</p><p>This small synthetic corpus can be easy
enough for several configurations to tie. A tie suggests a more expensive setup may not
be necessary for this case; it does not establish equivalence on larger or real-world documents.
These single-pass timings should not be treated as statistically significant speed rankings.</p>
<p>The measured study was resumed after API quota failures and a malformed calculator call.
Successful earlier answers were retained. The implementation subsequently added citation/tool-output
repair based on observed defects; that repair does not overwrite the recorded study answers.</p>
<h2>Cases to inspect</h2>""" + (examples or "<p>No failed automatic checks were recorded.</p>")
    page += (
        "<h2>Reproducibility</h2><p>See manifest.json for document hashes, question definitions, "
    )
    page += "package versions, and settings. results.jsonl retains every answer and its sources. "
    page += (
        "Use the app's review table to record correctness and citation support.</p></body></html>"
    )
    (output / "report.html").write_text(page, encoding="utf-8")
    print(output / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build_report(parser.parse_args().output)
