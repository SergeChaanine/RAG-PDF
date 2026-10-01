"""Generate the synthetic PDF and reference questions in the application data folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from rag_pdf.sample import create_sample  # noqa: E402

if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    for path in create_sample():
        print(path)
