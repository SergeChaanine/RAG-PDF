"""Crawl approved USEK public HTML and PDFs, with a persistent recursive queue."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from rag_pdf.crawl_core import (
    DEFAULT_HOSTS,
    DEFAULT_START,
    CrawlStore,
    extract_html,
    extract_pdf,
    normalize_url,
    utc_now,
    write_json,
)


def default_output():
    configured = os.getenv("RAG_DATA_DIR", "").strip()
    if configured:
        root = Path(os.path.expandvars(configured)).expanduser()
    elif os.getenv("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "RAG-PDF"
    else:
        root = Path.home() / ".local" / "share" / "rag-pdf"
    return root / "usek" / "crawl"


def retry_delay(header):
    if not header:
        return 60
    try:
        return max(0, int(header))
    except ValueError:
        try:
            target = parsedate_to_datetime(header.decode("ascii"))
            return max(0, int((target - datetime.now(timezone.utc)).total_seconds()))
        except (ValueError, TypeError, UnicodeError):
            return 60


def make_spider():
    import scrapy
    from lxml import etree
    from scrapy import signals
    from scrapy.exceptions import IgnoreRequest, StopDownload

    class ScopeMiddleware:
        """Validate redirects as well as initial requests, before downloading."""

        @classmethod
        def from_crawler(cls, crawler):
            middleware = cls()
            middleware.crawler = crawler
            return middleware

        def process_request(self, request):
            if not self.crawler.spider.clean_url(request.url):
                raise IgnoreRequest("Request/redirect is outside the configured HTML/PDF scope.")

    class UsekSpider(scrapy.Spider):
        name = "usek_public"

        def __init__(self, store, options, **kwargs):
            super().__init__(**kwargs)
            self.store, self.options = store, options
            self.processed = 0
            self.finish_reason = "frontier_exhausted"
            self.allowed_domains = list(options.allowed_hosts)
            self.halted = False

        @classmethod
        def from_crawler(cls, crawler, *args, **kwargs):
            spider = super().from_crawler(crawler, *args, **kwargs)
            crawler.signals.connect(spider.headers_received, signal=signals.headers_received)
            crawler.signals.connect(spider.spider_error, signal=signals.spider_error)
            return spider

        def clean_url(self, href, base=DEFAULT_START):
            return normalize_url(
                href, base, self.options.allowed_hosts, self.options.include_subdomains
            )

        def headers_received(self, headers, request):
            content_type = headers.get(b"Content-Type", b"").lower().split(b";", 1)[0].strip()
            if content_type and not (
                content_type.startswith(b"text/")
                or content_type
                in {
                    b"application/pdf",
                    b"application/xhtml+xml",
                    b"application/xml",
                    b"application/gzip",
                    b"application/x-gzip",
                    b"application/octet-stream",
                }
            ):
                request.meta["skipped_content_type"] = content_type.decode("ascii", "replace")
                raise StopDownload(fail=False)

        async def start(self):
            if request := self.next_request():
                yield request

        def next_request(self):
            if self.halted:
                return None
            limited = bool(self.options.max_pages and self.processed >= self.options.max_pages)
            task = self.store.claim(self.options.depth, sitemaps_only=limited)
            if task is None:
                queued = self.store.connection.execute(
                    "SELECT COUNT(*) FROM urls WHERE state='queued'"
                ).fetchone()[0]
                if limited:
                    self.finish_reason = "page_limit"
                elif queued:
                    self.finish_reason = "depth_limit"
                return None
            if task["kind"] != "sitemap":
                self.processed += 1
            return scrapy.Request(
                task["url"],
                callback=self.parse,
                errback=self.failed,
                dont_filter=True,
                meta={"task": task, "handle_httpstatus_list": list(range(400, 600))},
            )

        def enqueue_links(self, links, task, *, sitemap=False):
            for href in links:
                url = self.clean_url(href, task["url"])
                if url:
                    self.store.enqueue(
                        url,
                        parent=task["url"],
                        seed=url if sitemap else task["seed_url"],
                        depth=0 if sitemap else task["depth"] + 1,
                        discovery="sitemap" if sitemap else "link",
                    )

        def parse_sitemap(self, response, task):
            body = response.body
            if body.startswith(b"\x1f\x8b"):
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                    body = stream.read(self.options.max_file_mb * 1024 * 1024 + 1)
                if len(body) > self.options.max_file_mb * 1024 * 1024:
                    raise ValueError("Expanded sitemap exceeds the file-size limit.")
            root = etree.fromstring(
                body, parser=etree.XMLParser(resolve_entities=False, no_network=True)
            )
            root_name = etree.QName(root).localname
            if root_name not in {"sitemapindex", "urlset"}:
                raise ValueError("Response is not a sitemap index or URL set.")
            links = root.xpath(
                "./*[local-name()='sitemap' or local-name()='url']/*[local-name()='loc']/text()"
            )
            if root_name == "sitemapindex":
                for href in links:
                    if url := self.clean_url(href, response.url):
                        self.store.enqueue(
                            url, parent=task["url"], kind="sitemap", discovery="sitemap_index"
                        )
            else:
                self.enqueue_links(links, task, sitemap=True)
            self.store.finish(task["url"], "sitemap", final_url=response.url, http_status=200)
            print(f"Sitemap: discovered {len(links)} entries", flush=True)

        def parse_content(self, response, task):
            content_type = response.headers.get(b"Content-Type", b"").lower()
            if b"%PDF-" in response.body[:1024] or b"application/pdf" in content_type:
                source_type = "pdf"
                parsed, snapshot = extract_pdf(response.body, response.url), None
            elif b"html" in content_type or response.body.lstrip().lower().startswith(
                (b"<!doctype html", b"<html")
            ):
                source_type = "html"
                parsed, snapshot = extract_html(response.text, response.url)
                if parsed.get("restricted"):
                    self.store.finish(
                        task["url"],
                        "blocked",
                        final_url=response.url,
                        http_status=response.status,
                        error="Login page (password form).",
                    )
                    return
            else:
                self.store.finish(
                    task["url"],
                    "skipped",
                    final_url=response.url,
                    http_status=response.status,
                    error="Not an HTML page or PDF.",
                )
                return
            self.store.save_document(
                task, response.url, response.body, source_type, parsed, snapshot=snapshot
            )
            self.enqueue_links(parsed.get("links", []), task)
            print(
                f"[{self.processed}] {source_type.upper()} depth={task['depth']} "
                f"chars={len(parsed.get('text', '')):,} {response.url}",
                flush=True,
            )

        def parse(self, response):
            task = response.meta["task"]
            try:
                if response.status in {429, 503} and (
                    response.status == 429 or response.headers.get(b"Retry-After")
                ):
                    seconds = retry_delay(response.headers.get(b"Retry-After"))
                    self.store.finish(
                        task["url"],
                        "queued",
                        http_status=response.status,
                        error=f"Rate limited; wait at least {seconds} seconds.",
                    )
                    self.finish_reason = f"rate_limited_wait_{seconds}_seconds"
                    self.halted = True
                    write_json(
                        self.store.output / "resume_after.json",
                        {
                            "not_before_epoch": datetime.now(timezone.utc).timestamp() + seconds,
                            "url": response.url,
                        },
                    )
                    print(f"Server requested a pause. Resume after {seconds} seconds.", flush=True)
                elif response.status >= 400:
                    self.store.finish(
                        task["url"],
                        "blocked" if response.status in {401, 403} else "failed",
                        final_url=response.url,
                        http_status=response.status,
                        error=f"HTTP {response.status}",
                    )
                elif response.meta.get("skipped_content_type"):
                    self.store.finish(
                        task["url"],
                        "skipped",
                        final_url=response.url,
                        http_status=response.status,
                        error="Excluded media/content type: "
                        + response.meta["skipped_content_type"],
                    )
                elif task["kind"] == "sitemap":
                    self.parse_sitemap(response, task)
                else:
                    self.parse_content(response, task)
            except Exception as exc:
                self.logger.exception("Extraction failed for %s", response.url)
                self.store.finish(
                    task["url"],
                    "failed",
                    final_url=response.url,
                    http_status=response.status,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if self.processed and self.processed % 50 == 0:
                self.store.export(max_depth=self.options.depth, processed=self.processed)
            if request := self.next_request():
                yield request

        def failed(self, failure):
            task = failure.request.meta["task"]
            self.store.finish(task["url"], "failed", error=failure.getErrorMessage())
            print(f"Failed: {task['url']}: {failure.getErrorMessage()}", flush=True)
            if request := self.next_request():
                yield request

        def spider_error(self, failure, response):
            self.finish_reason = "spider_error"
            if response and "task" in response.meta:
                self.store.finish(
                    response.meta["task"]["url"], "failed", error=failure.getErrorMessage()
                )

        def closed(self, reason):
            if reason != "finished":
                self.finish_reason = reason
            report = self.store.export(
                max_depth=self.options.depth, reason=self.finish_reason, processed=self.processed
            )
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    return UsekSpider, ScopeMiddleware


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-url",
        action="append",
        dest="start_urls",
        help="Repeat for more seeds. Default: USEK English homepage.",
    )
    parser.add_argument("--output", type=Path, help="Dataset directory; reuse to resume.")
    parser.add_argument(
        "--depth", type=int, default=3, help="Link depth from each seed; 0=unlimited."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Content URLs attempted this invocation; 0=unlimited.",
    )
    parser.add_argument("--delay", type=float, default=2, help="Minimum seconds between requests.")
    parser.add_argument("--max-file-mb", type=int, default=100, help="Response size limit in MiB.")
    parser.add_argument(
        "--include-sitemap",
        action="store_true",
        help="Add USEK sitemap entries as independent depth-zero seeds.",
    )
    parser.add_argument(
        "--sitemap-url", action="append", default=[], help="Additional sitemap URL."
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        dest="allowed_hosts",
        help="Exact allowlist; repeat. Default: usek.edu.lb and www.usek.edu.lb.",
    )
    parser.add_argument(
        "--include-subdomains",
        action="store_true",
        help="Also follow public subdomains of allowed hosts.",
    )
    parser.add_argument(
        "--retry-failures", action="store_true", help="Requeue failed/blocked URLs."
    )
    parser.add_argument(
        "--obey-robots",
        action="store_true",
        help="Enable robots rules instead of the USEK-approved override.",
    )
    parser.add_argument("--status", action="store_true", help="Export reports without crawling.")
    return parser


def main(argv=None):
    try:
        import portalocker
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit(
            "Install crawler dependencies: python -m pip install -r requirements-crawl.txt"
        ) from exc
    load_dotenv(Path.cwd() / ".env")
    parser = argument_parser()
    options = parser.parse_args(argv)
    if min(options.depth, options.max_pages, options.delay) < 0 or options.max_file_mb < 1:
        parser.error(
            "Depth, page limit, and delay must be nonnegative; file size must be positive."
        )
    options.start_urls = options.start_urls or [DEFAULT_START]
    options.allowed_hosts = tuple(host.lower() for host in (options.allowed_hosts or DEFAULT_HOSTS))
    options.output = (options.output or default_output()).resolve()
    for url in [*options.start_urls, *options.sitemap_url]:
        if not normalize_url(url, url, options.allowed_hosts, options.include_subdomains):
            parser.error(f"Seed is outside the configured HTML/PDF scope: {url}")
    options.output.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(str(options.output / ".crawl.lock"), mode="a", timeout=1):
            return run(options)
    except portalocker.exceptions.LockException:
        parser.error("Another crawler is using this dataset. Stop it or choose another --output.")


def run(options):
    store = CrawlStore(options.output)
    try:
        if options.status:
            print(json.dumps(store.export(max_depth=options.depth), indent=2))
            return 0
        pause_path = options.output / "resume_after.json"
        if pause_path.exists():
            pause = json.loads(pause_path.read_text("utf-8"))
            remaining = pause["not_before_epoch"] - datetime.now(timezone.utc).timestamp()
            if remaining > 0:
                print(f"Server requested a pause. Try again in {int(remaining) + 1} seconds.")
                return 1
        try:
            from scrapy.crawler import CrawlerProcess

            spider_class, scope_middleware = make_spider()
        except ImportError as exc:
            raise SystemExit(
                "Install crawler dependencies: python -m pip install -r requirements-crawl.txt"
            ) from exc
        manifest_path = options.output / "manifest.json"
        scope = {
            "allowed_hosts": list(options.allowed_hosts),
            "include_subdomains": options.include_subdomains,
            "obey_robots": options.obey_robots,
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text("utf-8"))
            if previous["scope"] != scope:
                raise SystemExit(
                    "Dataset has different host/robots scope. Reuse those flags "
                    "or choose a new --output directory."
                )
        else:
            previous = {"created_at": utc_now()}
        write_json(
            manifest_path,
            {
                **previous,
                "last_started_at": utc_now(),
                "scope": scope,
                "depth_limit": options.depth,
                "max_pages_this_run": options.max_pages,
                "delay_seconds": options.delay,
                "max_file_mb": options.max_file_mb,
                "start_urls": options.start_urls,
                "include_sitemap": options.include_sitemap,
                "sitemap_urls": options.sitemap_url,
            },
        )
        store.recover(options.retry_failures)
        for url in options.start_urls:
            normalized = normalize_url(url, url, options.allowed_hosts, options.include_subdomains)
            store.enqueue(normalized, discovery="homepage")
        sitemap_urls = list(options.sitemap_url)
        if options.include_sitemap:
            sitemap_urls.append("https://www.usek.edu.lb/sitemap.xml")
        for url in sitemap_urls:
            normalized = normalize_url(url, url, options.allowed_hosts, options.include_subdomains)
            if not normalized:
                raise SystemExit(
                    "The USEK sitemap is outside your allowed hosts. "
                    "Use --sitemap-url for another site."
                )
            store.enqueue(normalized, kind="sitemap", discovery="sitemap_index")
        print(
            f"Output: {options.output}\nPress Ctrl+C once to stop; repeat the command to resume.",
            flush=True,
        )
        settings = {
            "ROBOTSTXT_OBEY": options.obey_robots,
            "USER_AGENT": "USEK-RAG-StudentProject/0.1",
            "CONCURRENT_REQUESTS": 1,
            "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
            "DOWNLOAD_DELAY": options.delay,
            "DOWNLOAD_DELAY_JITTER": 0,
            "DOWNLOAD_TIMEOUT": 60,
            "DOWNLOAD_MAXSIZE": options.max_file_mb * 1024 * 1024,
            "DOWNLOAD_WARNSIZE": 0,
            "RETRY_TIMES": 2,
            "RETRY_HTTP_CODES": [408, 500, 502, 504, 522, 524],
            "COOKIES_ENABLED": False,
            "TELNETCONSOLE_ENABLED": False,
            "REMOTE_CONTROL_ENABLED": False,
            "LOG_LEVEL": "INFO",
            "LOG_FILE": str(options.output / "crawl.log"),
            "DOWNLOADER_MIDDLEWARES": {scope_middleware: 50},
            # The queue's depth is graph distance, not the order callbacks run in.
            "SPIDER_MIDDLEWARES": {"scrapy.spidermiddlewares.depth.DepthMiddleware": None},
        }
        process = CrawlerProcess(settings)
        crawler = process.create_crawler(spider_class)
        outcome = []
        process.crawl(crawler, store=store, options=options).addErrback(outcome.append)
        process.start()
        if outcome:
            raise RuntimeError(f"Crawler startup failed: {outcome[0].getErrorMessage()}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
