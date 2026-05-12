from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


_PDF_MAGIC = b"%PDF"
_MIN_CHARS_PER_PAGE = 100
_MAX_PAGES_TO_SAMPLE = 5


def is_pdf(file_bytes: bytes) -> bool:
    """Return True if the bytes look like a PDF (by magic header)."""
    return file_bytes[:4] == _PDF_MAGIC


def scan_pdf(file_bytes: bytes) -> bool:
    """Return True if the PDF appears to be a scanned (image-based) document.

    Opens the PDF with pypdfium2, pulls extractable text from up to the first
    few pages, and treats the document as scanned when the average text per
    sampled page is below a small threshold. Native PDFs typically yield
    hundreds of characters per page; scans yield zero or a few stray glyphs.

    Non-PDF input or any failure to open returns False — let Kreuzberg try.
    """
    if not is_pdf(file_bytes):
        return False

    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 not installed; cannot detect scanned PDFs")
        return False

    try:
        pdf = pdfium.PdfDocument(file_bytes)
    except Exception as exc:  # pdfium raises PdfiumError, but be defensive
        logger.warning("Failed to open PDF for scan detection: %s", exc)
        return False

    try:
        page_count = len(pdf)
        if page_count == 0:
            logger.info("scan_pdf: empty PDF (0 pages) — treating as native")
            return False

        sample = min(page_count, _MAX_PAGES_TO_SAMPLE)
        total_chars = 0
        for i in range(sample):
            page = pdf[i]
            text_page = page.get_textpage()
            try:
                text = text_page.get_text_range()
            finally:
                text_page.close()
                page.close()
            total_chars += len(text.strip())

        avg_chars = total_chars / sample
        is_scanned = avg_chars < _MIN_CHARS_PER_PAGE
        logger.info(
            "scan_pdf: pages=%d sampled=%d total_chars=%d avg=%.1f "
            "threshold=%d scanned=%s",
            page_count, sample, total_chars, avg_chars,
            _MIN_CHARS_PER_PAGE, is_scanned,
        )
        return is_scanned
    finally:
        pdf.close()
