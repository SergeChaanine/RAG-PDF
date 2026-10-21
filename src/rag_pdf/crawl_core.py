"""SQLite crawl queue, URL filtering, extraction, and presentation exports."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_START = "https://www.usek.edu.lb/en/home"
DEFAULT_HOSTS = ("www.usek.edu.lb", "usek.edu.lb")
SKIP_SUFFIXES = frozenset(
    [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".bmp",
        ".tif",
        ".tiff",
        ".avif",
        ".mp4",
        ".mp3",
        ".wav",
        ".ogg",
        ".mov",
        ".avi",
        ".webm",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".css",
        ".js",
        ".map",
        ".zip",
        ".rar",
        ".7z",
        ".exe",
        ".dmg",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
    ]
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def normalize_url(href, base, hosts=DEFAULT_HOSTS, include_subdomains=False):
    """Keep meaningful query parameters/path case; reject media and offsite URLs."""
    try:
        parts = urlsplit(urljoin(base, href.strip()))
        host = (parts.hostname or "").lower()
        if parts.scheme.lower() not in {"http", "https"} or parts.username or parts.password:
            return None
        if not any(
            host == allowed or (include_subdomains and host.endswith("." + allowed))
            for allowed in hosts
        ):
            return None
        port = parts.port
        netloc = host
        if port and (parts.scheme.lower(), port) not in {("https", 443), ("http", 80)}:
            netloc = f"{host}:{port}"
        if Path(parts.path).suffix.lower() in SKIP_SUFFIXES:
            return None
        query = urlencode(
            [
                (key, value)
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
                if not key.lower().startswith("utm_")
                and key.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}
            ]
        )
        result = urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", query, ""))
        return result if len(result) <= 4096 else None
    except (ValueError, UnicodeError):
        return None


def extract_html(content, url):
    """Discover links before cleaning; do not save image files or embedded media."""
    from lxml import html
    from trafilatura import extract

    tree = html.fromstring(content, base_url=url)
    bases = tree.xpath("//base/@href")
    base = urljoin(url, bases[0]) if bases else url
    links = tree.xpath("//a/@href | //iframe/@src | //embed/@src | //object/@data")
    links = list(dict.fromkeys(urljoin(base, link) for link in links))
    title = " ".join(tree.xpath("//title/text()")).strip() or url
    language = tree.get("lang", "") or "und"
    headings = [
        {"level": int(node.tag[1]), "text": " ".join(node.text_content().split())}
        for node in tree.xpath("//h1 | //h2 | //h3 | //h4 | //h5 | //h6")
        if node.text_content().strip()
    ]
    if tree.xpath("//input[translate(@type, 'PASSWORD', 'password')='password']"):
        return {"restricted": True, "title": title, "links": [], "warnings": []}, b""
    for node in tree.xpath(
        "//script | //style | //img | //picture | //svg | //video | //audio | //canvas"
        " | //iframe | //embed | //object | //source | //link"
    ):
        node.drop_tree()
    for node in tree.iter():
        for key, value in list(node.attrib.items()):
            if key.lower() == "style" or "data:image/" in value.lower():
                del node.attrib[key]
    snapshot = html.tostring(tree, encoding="utf-8", method="html")
    # USEK's WebForms body contains a repeated global navigation menu.
    content_nodes = tree.xpath(
        "//*[@id='PlaceHolderBody_divMainContentHolder'] | //main | //article"
    )
    container = content_nodes[0] if content_nodes else tree
    for node in container.xpath(
        ".//header | .//footer | .//nav | .//*[@role='navigation'] | .//input | .//button"
        " | .//select | .//textarea | .//*[contains(concat(' ', normalize-space(@class), ' '),"
        " ' mainMenuCont ') or contains(concat(' ', normalize-space(@class), ' '),"
        " ' topMenuCont ') or contains(concat(' ', normalize-space(@class), ' '),"
        " ' footerContainer ') or contains(concat(' ', normalize-space(@class), ' '),"
        " ' headerCont ')]"
    ):
        node.drop_tree()
    clean_html = html.tostring(container, encoding="unicode", method="html")
    if container.tag != "html":
        # Trafilatura expects a document root; a bare <main>/<div> can lose
        # its headings and small tables during extraction.
        clean_html = f"<html><body>{clean_html}</body></html>"
    text = (
        extract(
            clean_html,
            url=url,
            output_format="markdown",
            include_tables=True,
            include_comments=False,
            include_images=False,
            include_links=True,
        )
        or ""
    )
    warnings = []
    if not text.strip():
        text = "\n".join(part.strip() for part in container.itertext() if part.strip())
        warnings.append("Main-content extraction failed; fallback text needs manual review.")
    if len(text.strip()) < 150:
        warnings.append("Very little text; check for a JavaScript page or empty content.")
    return {
        "title": title,
        "language": language,
        "text": text,
        "headings": headings,
        "links": links,
        "warnings": warnings,
    }, snapshot


def extract_pdf(content, url):
    """Retain original page numbers and detected tables, without extracting images."""
    import pymupdf

    if b"%PDF-" not in content[:1024]:
        raise ValueError("Response is not a valid PDF.")
    pages, tables, links, warnings = [], [], [], []
    with pymupdf.open(stream=content, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("PDF requires a password.")
        title = document.metadata.get("title") or Path(urlsplit(url).path).name
        for number, page in enumerate(document, 1):
            text = page.get_text("text", sort=True).strip()
            pages.append({"page_number": number, "text": text})
            links.extend(link["uri"] for link in page.get_links() if link.get("uri"))
            try:
                for table in page.find_tables().tables:
                    tables.append({"page_number": number, "rows": table.extract()})
            except Exception as exc:
                warnings.append(f"Page {number}: table extraction failed ({type(exc).__name__}).")
        page_count = document.page_count
    empty = [page["page_number"] for page in pages if not page["text"]]
    if empty:
        warnings.append(f"Pages without native text (OCR may be needed): {empty}")
    return {
        "title": title,
        "language": "und",
        "text": "\n\n".join(f"[Page {p['page_number']}]\n{p['text']}" for p in pages),
        "text_available": any(page["text"] for page in pages),
        "page_count": page_count,
        "pages": pages,
        "tables": tables,
        "links": list(dict.fromkeys(links)),
        "warnings": warnings,
    }


class CrawlStore:
    """SQLite is the authoritative queue, including work remaining after interruption."""

    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.output / "crawl.sqlite3")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS urls (
                url TEXT PRIMARY KEY, parent_url TEXT, seed_url TEXT, depth INTEGER,
                discovery TEXT, kind TEXT, state TEXT DEFAULT 'queued',
                final_url TEXT, http_status INTEGER, fetched_at TEXT,
                source_type TEXT, content_hash TEXT, record_path TEXT, error TEXT
            );
            CREATE TABLE IF NOT EXISTS links (
                parent_url TEXT, url TEXT, discovery TEXT,
                PRIMARY KEY (parent_url, url, discovery)
            );
            CREATE INDEX IF NOT EXISTS frontier ON urls(state, depth);
        """)

    def recover(self, retry_failures=False):
        with self.connection:
            self.connection.execute("UPDATE urls SET state='queued' WHERE state='fetching'")
            if retry_failures:
                self.connection.execute(
                    "UPDATE urls SET state='queued' WHERE state IN ('failed', 'blocked')"
                )

    def enqueue(self, url, *, parent="", seed=None, depth=0, discovery="link", kind="page"):
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO links VALUES (?, ?, ?)", (parent, url, discovery)
            )
            row = self.connection.execute("SELECT * FROM urls WHERE url=?", (url,)).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO urls(url,parent_url,seed_url,depth,discovery,kind) "
                    "VALUES (?,?,?,?,?,?)",
                    (url, parent, seed or url, depth, discovery, kind),
                )
            elif depth < row["depth"]:
                self.connection.execute(
                    "UPDATE urls SET depth=?,parent_url=?,seed_url=?,discovery=?,state='queued' "
                    "WHERE url=?",
                    (depth, parent, seed or url, discovery, url),
                )

    def claim(self, max_depth=0, *, sitemaps_only=False):
        where, parameters = "state='queued'", []
        if max_depth:
            where += " AND (depth<=? OR kind='sitemap')"
            parameters.append(max_depth)
        if sitemaps_only:
            where += " AND kind='sitemap'"
        with self.connection:
            row = self.connection.execute(
                f"SELECT * FROM urls WHERE {where} "
                "ORDER BY (kind='sitemap') DESC, depth, rowid LIMIT 1",
                parameters,
            ).fetchone()
            if row:
                self.connection.execute(
                    "UPDATE urls SET state='fetching' WHERE url=?", (row["url"],)
                )
        return dict(row) if row else None

    def finish(self, url, state, **values):
        allowed = {
            "final_url",
            "http_status",
            "source_type",
            "content_hash",
            "record_path",
            "error",
        }
        if not values.keys() <= allowed:
            raise ValueError("Unexpected crawl metadata.")
        fields = {"state": state, "fetched_at": utc_now(), **values}
        with self.connection:
            self.connection.execute(
                "UPDATE urls SET " + ",".join(f"{key}=?" for key in fields) + " WHERE url=?",
                [*fields.values(), url],
            )

    def save_document(self, task, response_url, content, source_type, parsed, snapshot=None):
        content_hash, raw_hash = digest(parsed.get("text", "")), digest(content)
        suffix = "html" if source_type == "html" else "pdf"
        relative_raw = f"{source_type}/{raw_hash}.{suffix}"
        raw_path = self.output / relative_raw
        raw_path.parent.mkdir(exist_ok=True)
        if not raw_path.exists():
            temporary = raw_path.with_suffix(".tmp")
            temporary.write_bytes(snapshot if snapshot is not None else content)
            temporary.replace(raw_path)
        record = {
            **parsed,
            "source_url": task["url"],
            "final_url": response_url,
            "source_type": source_type,
            "parent_url": task["parent_url"],
            "seed_url": task["seed_url"],
            "depth_from_seed": task["depth"],
            "discovery": task["discovery"],
            "fetched_at": utc_now(),
            "content_hash": content_hash,
            "download_sha256": raw_hash,
            "raw_sha256": digest(snapshot if snapshot is not None else content),
            "raw_path": relative_raw,
            "text_available": parsed.get("text_available", bool(parsed.get("text", "").strip())),
        }
        relative_record = f"records/{digest(task['url'])}.json"
        write_json(self.output / relative_record, record)
        self.finish(
            task["url"],
            "done" if record["text_available"] else "no_text",
            final_url=response_url,
            http_status=200,
            source_type=source_type,
            content_hash=content_hash,
            record_path=relative_record,
            error=None,
        )
        return record

    def export(self, *, max_depth=0, reason="snapshot", processed=0):
        rows = [dict(row) for row in self.connection.execute("SELECT * FROM urls ORDER BY rowid")]
        groups = {}
        for row in rows:
            if row["state"] in {"done", "no_text"} and row["record_path"]:
                identity = row["content_hash"] if row["state"] == "done" else row["url"]
                groups.setdefault((row["source_type"], identity), []).append(row)
        temporary = self.output / "documents.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for group in groups.values():
                record = json.loads((self.output / group[0]["record_path"]).read_text("utf-8"))
                record["source_urls"] = sorted(
                    {row[key] for row in group for key in ("url", "final_url") if row[key]}
                )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(self.output / "documents.jsonl")
        failures = [
            {key: row[key] for key in ("url", "state", "http_status", "error")}
            for row in rows
            if row["state"] in {"failed", "blocked", "skipped", "no_text"}
        ]
        write_json(self.output / "failures.json", failures)
        states = dict(Counter(row["state"] for row in rows))
        report = {
            "generated_at": utc_now(),
            "finish_reason": reason,
            "processed_this_run": processed,
            "discovered_urls": len(rows),
            "states": states,
            "unique_documents": len(groups),
            "downloaded_by_type": dict(
                Counter(row["source_type"] for row in rows if row["state"] in {"done", "no_text"})
            ),
            "queued_beyond_depth": sum(
                row["state"] == "queued"
                and bool(max_depth)
                and row["depth"] > max_depth
                and row["kind"] != "sitemap"
                for row in rows
            ),
            "depths_from_seeds": dict(Counter(str(row["depth"]) for row in rows)),
            "coverage_note": "Reachable public HTML/PDF content only; failures and limits are "
            "explicit. Sitemap entries are separate seeds; depth is measured from each seed.",
        }
        write_json(self.output / "report.json", report)
        lines = [
            "# USEK crawl report",
            "",
            f"Generated: {report['generated_at']}",
            "",
            f"Finish reason: {reason}",
            f"Discovered URLs: {len(rows)}",
            f"Unique exported documents: {len(groups)}",
            "",
            "| State | URLs |",
            "|---|---:|",
            *[f"| {key} | {value} |" for key, value in sorted(states.items())],
            "",
            report["coverage_note"],
            "",
            "See failures.json for blocked pages, extraction failures, and missing text.",
        ]
        (self.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def close(self):
        self.connection.close()
