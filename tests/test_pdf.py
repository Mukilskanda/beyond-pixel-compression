import shutil

import fitz
import pytest

from textpath.pdf import extract_pdf_pages

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed (see README setup)"
)


def _make_text_pdf(path, pages_text: list[str]) -> None:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _make_image_only_pdf(path, sample_image_path) -> None:
    """A page with no embedded text layer at all — just an inserted image,
    the scanned-document case that must fall back to OCR."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, filename=str(sample_image_path))
    doc.save(str(path))
    doc.close()


def test_native_text_pdf_extracts_without_ocr(tmp_path):
    pdf_path = tmp_path / "native.pdf"
    _make_text_pdf(pdf_path, ["Invoice #12345\nTotal: $500.00"])

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].used_ocr is False
    assert "Invoice #12345" in pages[0].text
    assert "500.00" in pages[0].text


def test_multi_page_native_text_pdf_preserves_page_order(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    _make_text_pdf(pdf_path, ["First page content, well over ten characters", "Second page content, also well over ten characters"])

    pages = extract_pdf_pages(pdf_path)

    assert [p.page_number for p in pages] == [0, 1]
    assert "First page content" in pages[0].text
    assert "Second page content" in pages[1].text
    assert all(p.used_ocr is False for p in pages)


@requires_tesseract
def test_image_only_pdf_falls_back_to_ocr(tmp_path, sample_image):
    pdf_path = tmp_path / "scanned.pdf"
    _make_image_only_pdf(pdf_path, sample_image)

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].used_ocr is True
    # sample_image fixture contains "Invoice #12345" rendered as text-in-image
    assert "Invoice" in pages[0].text or "12345" in pages[0].text


@requires_tesseract
def test_blank_page_falls_back_to_ocr_without_crashing(tmp_path):
    """A page with no embedded text and no image at all (not the same as a
    zero-page document, which PyMuPDF refuses to save) must still resolve
    via the OCR fallback rather than raising."""
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].used_ocr is True
