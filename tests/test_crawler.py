"""Offline checks against real local HTTP responses, including crawl resumption."""

import contextlib
import json
import sqlite3
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pymupdf
import pytest

from rag_pdf.crawl_core import CrawlStore, extract_html, extract_pdf, normalize_url


def test_url_scope_and_content_parameters():
    base = "https://www.usek.edu.lb/en/home"
    assert normalize_url("../admission?id=7&utm_source=test#top", base) == (
        "https://www.usek.edu.lb/admission?id=7"
    )
    assert normalize_url("/Content/Assets/Guide.PDF?download=1", base)
    for href in [
        "https://usek.edu.lb.evil.test/x",
        "mailto:test@usek.edu.lb",
        "https://user:password@www.usek.edu.lb/x",
        "/photo.PNG",
        "javascript:alert(1)",
    ]:
        assert normalize_url(href, base) is None
    assert normalize_url("https://library.usek.edu.lb/", base) is None
    assert normalize_url("https://library.usek.edu.lb/", base, include_subdomains=True)


def test_frontier_survives_interruption_and_depth_changes(tmp_path):
    store = CrawlStore(tmp_path)
    store.enqueue("https://www.usek.edu.lb/home", depth=0)
    store.enqueue("https://www.usek.edu.lb/deep", depth=2)
    first = store.claim(1)
    assert first["url"].endswith("home")
    store.close()
    store = CrawlStore(tmp_path)
    store.recover()
    assert store.claim(1)["url"] == first["url"]
    store.finish(first["url"], "done")
    assert store.claim(1) is None
    assert store.claim(2)["url"].endswith("deep")
    store.close()


def test_html_keeps_tables_and_links_but_removes_images():
    content = b"""<html lang="en"><head><title>Admissions</title></head><body>
    <main><h1>Application requirements</h1><p>Submit the required documents online.
    Applications are reviewed by the admissions office. Read all conditions carefully.</p>
    <table><tr><th>Document</th><th>Required</th></tr>
    <tr><td>Transcript</td><td>Yes</td></tr></table>
    <a href="/Content/Assets/guide.pdf">Admissions guide</a>
    <img src="data:image/png;base64,AAAA"><script>alert('no');</script></main></body></html>"""
    record, snapshot = extract_html(content, "https://www.usek.edu.lb/en/home")
    assert "Transcript" in record["text"]
    assert "https://www.usek.edu.lb/Content/Assets/guide.pdf" in record["links"]
    assert record["language"] == "en"
    assert b"<img" not in snapshot and b"base64" not in snapshot and b"<script" not in snapshot


def make_pdf():
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Applicants must submit their transcript and identification.")
        return document.tobytes()


def test_pdf_retains_original_pages_and_flags_empty_scans():
    parsed = extract_pdf(make_pdf(), "https://www.usek.edu.lb/guide.pdf")
    assert parsed["pages"][0]["page_number"] == 1
    assert "transcript" in parsed["text"]
    with pymupdf.open() as document:
        document.new_page()
        empty = extract_pdf(document.tobytes(), "https://www.usek.edu.lb/scan.pdf")
    assert not empty["text_available"]
    assert "OCR" in empty["warnings"][0]


def test_export_deduplicates_content_but_retains_source_urls(tmp_path):
    store = CrawlStore(tmp_path)
    for url in ["https://www.usek.edu.lb/a", "https://www.usek.edu.lb/b"]:
        store.enqueue(url)
        task = store.claim()
        store.save_document(task, url, b"same", "html", {"text": "Identical policy text"})
    report = store.export()
    records = [json.loads(line) for line in (tmp_path / "documents.jsonl").read_text().splitlines()]
    assert report["unique_documents"] == len(records) == 1
    assert len(records[0]["source_urls"]) == 2
    store.close()


@pytest.fixture
def fixture_site():
    requests = []
    pdf = make_pdf()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            content_type = "text/html; charset=utf-8"
            if self.path == "/home":
                body = b"""<html lang="en"><title>University home</title><main>
                <h1>Student information</h1><p>This is a university information page.
                It describes student services, admissions procedures, and registration.
                Students should consult the detailed requirements and application guide.</p>
                <a href="/section">Section</a><a href="/guide.pdf">Guide</a>
                <a href="/photo.png">Photo</a><a href="/media">Media</a>
                <a href="/login">Login</a><a href="/redirect">Redirect</a>
                </main></html>"""
            elif self.path == "/section":
                body = b"""<html><title>Admissions</title><main><h1>Admissions</h1>
                <p>Applicants must complete the admissions process before registration.
                Submit your transcript and identification documents for review.
                More information is available in the detailed requirements.</p>
                <a href="/deep">Detailed requirements</a></main></html>"""
            elif self.path == "/guide.pdf":
                body, content_type = pdf, "application/pdf"
            elif self.path == "/media":
                body, content_type = b"THIS SHOULD NOT BE STORED", "image/png"
            elif self.path == "/login":
                body = b'<html><form><input type="password"></form></html>'
            elif self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/outside")
                self.end_headers()
                return
            elif self.path == "/sitemap.xml":
                body = (
                    f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    f"<url><loc>http://127.0.0.1:{self.server.server_port}/orphan</loc></url>"
                    "</urlset>"
                ).encode()
                content_type = "application/xml"
            elif self.path in {"/deep", "/orphan"}:
                body = (
                    f"<html><title>{self.path}</title><main><h1>Requirements</h1>"
                    f"<p>Details at {self.path}: applicants must submit the listed documents. "
                    "Follow the university procedure and consult admissions if additional "
                    "information is needed. This is a distinct information page.</p>"
                    "</main></html>"
                ).encode()
            elif self.path == "/limited":
                self.send_response(429)
                self.send_header("Retry-After", "120")
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_crawl(output, base, *arguments):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "crawl_usek.py"),
        "--output",
        str(output),
        "--allowed-host",
        "127.0.0.1",
        "--start-url",
        base + "/home",
        "--delay",
        "0",
        *arguments,
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((output / "report.json").read_text())


def test_recursive_http_depth_resume_pdf_and_media_exclusion(tmp_path, fixture_site):
    base, requests = fixture_site
    report = run_crawl(tmp_path, base, "--depth", "1", "--max-pages", "0")
    assert "/photo.png" not in requests and "/outside" not in requests and "/deep" not in requests
    assert report["downloaded_by_type"] == {"html": 2, "pdf": 1}
    assert report["states"]["blocked"] == 1
    assert report["states"]["skipped"] == 1
    assert report["states"]["failed"] == 1  # Offsite redirect.
    assert report["queued_beyond_depth"] == 1
    assert not list(tmp_path.rglob("*.png"))
    run_crawl(tmp_path, base, "--depth", "2", "--max-pages", "0")
    assert requests.count("/home") == requests.count("/guide.pdf") == 1
    assert "/deep" in requests
    records = [json.loads(line) for line in (tmp_path / "documents.jsonl").read_text().splitlines()]
    assert any(record["source_type"] == "pdf" and record["pages"] for record in records)


def test_sitemap_adds_unlinked_content_and_page_limit_is_exact(tmp_path, fixture_site):
    base, requests = fixture_site
    report = run_crawl(
        tmp_path, base, "--depth", "1", "--max-pages", "1", "--sitemap-url", base + "/sitemap.xml"
    )
    assert report["processed_this_run"] == 1
    assert "/sitemap.xml" in requests
    assert "/orphan" not in requests  # Still queued after the page limit.
    run_crawl(
        tmp_path, base, "--depth", "1", "--max-pages", "0", "--sitemap-url", base + "/sitemap.xml"
    )
    assert "/orphan" in requests
    with sqlite3.connect(tmp_path / "crawl.sqlite3") as connection:
        assert connection.execute(
            "SELECT discovery,depth FROM urls WHERE url=?", (base + "/orphan",)
        ).fetchone() == ("sitemap", 0)


def test_rate_limit_is_saved_and_immediate_resume_is_prevented(tmp_path, fixture_site):
    base, requests = fixture_site
    run_crawl(tmp_path, base, "--start-url", base + "/limited", "--depth", "1")
    assert (tmp_path / "resume_after.json").exists()
    before = len(requests)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "crawl_usek.py"),
            "--output",
            str(tmp_path),
            "--allowed-host",
            "127.0.0.1",
            "--start-url",
            base + "/home",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1 and "pause" in result.stdout
    assert len(requests) == before
