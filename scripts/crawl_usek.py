"""Run the repository crawler; existing standalone exports stay compatible."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from rag_pdf.crawl import main  # noqa: E402

if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    raise SystemExit(main())
