"""Start the file-upload RAG app with an optional USEK chatbot."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from rag_pdf.app_ui import render_application

load_dotenv(ROOT / ".env")
render_application()
