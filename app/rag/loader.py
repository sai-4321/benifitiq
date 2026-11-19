"""Read PDFs page by page and keep where each page came from (file + page number).

Keeping the page number is what makes citations possible later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class PageDoc:
    text: str
    source: str  # file name, e.g. "10050-Medicare-and-You.pdf"
    page: int  # 1-based page number as shown in a PDF viewer


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, min_chars: int = 40) -> list[PageDoc]:
    reader = PdfReader(str(path))
    pages: list[PageDoc] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - one bad page should not stop ingestion
            log.warning("Could not read %s page %d: %s", path.name, i, exc)
            continue
        if len(text) >= min_chars:  # skip blank/cover pages
            pages.append(PageDoc(text=text, source=path.name, page=i))
    return pages


def load_directory(folder: Path) -> list[PageDoc]:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder}. Run: python scripts/download_docs.py")
    all_pages: list[PageDoc] = []
    for pdf in pdfs:
        pages = load_pdf(pdf)
        log.info("Loaded %-60s %4d pages with text", pdf.name, len(pages))
        all_pages.extend(pages)
    return all_pages
