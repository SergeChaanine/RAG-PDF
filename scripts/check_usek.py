"""Validate a completed crawl export without models, network requests, or API keys."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from rag_pdf.config import Settings  # noqa: E402
from rag_pdf.website import load_crawl_dataset  # noqa: E402


def main(argv=None):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crawl-dir", type=Path, default=Settings.from_env().data_dir / "usek" / "full"
    )
    parser.add_argument("--limit", type=int, default=0, help="Usable sources to check; 0=all.")
    parser.add_argument(
        "--no-reextract",
        action="store_true",
        help="Use existing JSONL text instead of cleaning saved HTML again.",
    )
    options = parser.parse_args(argv)
    try:
        dataset = load_crawl_dataset(
            options.crawl_dir, limit=options.limit, reextract_html=not options.no_reextract
        )
    except (ValueError, OSError) as exc:
        print(f"Crawl is not ready: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(dataset.summary(), ensure_ascii=False, indent=2))
    return 0 if dataset.documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
