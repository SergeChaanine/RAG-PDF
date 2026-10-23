"""Switch between the uploaded-document library and saved USEK sources."""

import streamlit as st

from rag_pdf.document_ui import render_documents
from rag_pdf.website_ui import render_website


def render_application(*, default_usek=False):
    st.set_page_config(page_title="USEK Chatbot and Document RAG", page_icon="📚", layout="wide")
    usek = st.sidebar.toggle(
        "USEK Chatbot",
        value=default_usek,
        key="usek_chatbot_enabled",
        help="Turn on to search saved USEK sources. Turn off to upload your own documents.",
    )
    if usek:
        render_website()
    else:
        render_documents()
