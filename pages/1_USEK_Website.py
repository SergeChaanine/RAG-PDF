"""Start the USEK chatbot with a toggle back to file-upload RAG."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from rag_pdf.app_ui import render_application  # noqa: E402

load_dotenv(ROOT / ".env")
render_application(default_usek=True)
