import pymupdf
import pytest

from rag_pdf.errors import ValidationError
from rag_pdf.pdf_processing import extract_pdf


def make_pdf(page_count: int, text: str) -> bytes:
    document = pymupdf.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page()
        page.insert_text((72, 50), "Repeated Course Header")
        page.insert_textbox(
            pymupdf.Rect(72, 90, 520, 700),
            f"Page topic {page_number}. {text}",
            fontsize=11,
        )
        page.insert_text((72, 760), f"Page {page_number} of {page_count}")
    data = document.tobytes()
    document.close()
    return data


def test_valid_english_pdf_is_extracted_with_page_metadata() -> None:
    prose = (
        "This research document explains a neural retrieval system and evaluates its "
        "accuracy using a carefully designed experiment. "
    ) * 12
    extracted = extract_pdf(make_pdf(10, prose), "research.pdf")

    assert extracted.page_count == 10
    assert len(extracted.pages) == 10
    assert extracted.language == "en"
    assert extracted.pages[0].page_number == 1
    assert "Repeated Course Header" not in extracted.pages[0].text
    assert "Page 1 of 10" not in extracted.pages[0].text


def test_pdf_outside_page_limit_is_rejected() -> None:
    with pytest.raises(ValidationError, match="10–20"):
        extract_pdf(make_pdf(2, "English text " * 100), "short.pdf")


def test_non_pdf_content_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not a valid PDF"):
        extract_pdf(b"plain text", "fake.pdf")
