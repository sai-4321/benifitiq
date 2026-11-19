"""Public knowledge-base documents: where they come from and how they are fetched.

All four are Medicare.gov publications by CMS, a U.S. federal agency, so they are public-domain
U.S. Government works. No patient data is involved. The URLs always serve the newest edition,
so each download records a SHA-256 fingerprint and page count in data/raw/manifest.json.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader

LICENSE = "Public domain (U.S. Government work, CMS / Medicare.gov)"
PUBLICATIONS_PAGE = "https://www.medicare.gov/publications"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BenefitIQ-portfolio-downloader/1.0"


@dataclass(frozen=True)
class SourceDoc:
    filename: str
    title: str
    url: str

    @property
    def product_number(self) -> str:
        return self.filename.split("-")[0]


SOURCES: tuple[SourceDoc, ...] = (
    SourceDoc("10050-Medicare-and-You.pdf", "Medicare & You (official handbook)",
              "https://www.medicare.gov/publications/10050-Medicare-and-You.pdf"),
    SourceDoc("10110-Your-Guide-to-Medicare-Preventive-Services.pdf", "Your Guide to Medicare Preventive Services",
              "https://www.medicare.gov/publications/10110-Your-Guide-to-Medicare-Preventive-Services.pdf"),
    SourceDoc("11109-your-guide-to-medicare-prescription-drug-coverage.pdf",
              "Your Guide to Medicare Prescription Drug Coverage",
              "https://www.medicare.gov/publications/11109-your-guide-to-medicare-prescription-drug-coverage.pdf"),
    SourceDoc("11045-medicare-coverage-of-dme.pdf", "Medicare Coverage of Durable Medical Equipment & Other Devices",
              "https://www.medicare.gov/publications/11045-medicare-coverage-of-dme.pdf"),
)


class HttpSession(Protocol):
    """The slice of requests.Session we use - lets tests pass a fake session (no network)."""

    def get(self, url: str, **kwargs): ...


@dataclass
class ManifestEntry:
    file: str
    title: str
    url: str
    pages: int
    sha256_16: str
    size_mb: float
    downloaded: str
    license: str = LICENSE


class NotAPdfError(ValueError):
    """The server answered, but not with a PDF (e.g. an HTML error or bot-check page)."""


def download_pdf(session: HttpSession, url: str, dest: Path, timeout: int = 60) -> None:
    """Stream a PDF to disk atomically: write to .part, then rename. A failed download never
    leaves a half-written file that later looks valid."""
    with session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower():
            raise NotAPdfError(f"expected a PDF but got {content_type or 'no Content-Type'}")
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as f:
            for block in r.iter_content(chunk_size=65536):
                f.write(block)
        tmp.replace(dest)


def describe(doc: SourceDoc, path: Path) -> ManifestEntry:
    """Validate the PDF (pypdf must open it) and fingerprint it."""
    pages = len(PdfReader(str(path)).pages)
    data = path.read_bytes()
    return ManifestEntry(
        file=doc.filename, title=doc.title, url=doc.url, pages=pages,
        sha256_16=hashlib.sha256(data).hexdigest()[:16], size_mb=round(len(data) / 1_048_576, 2),
        downloaded=date.today().isoformat(),
    )


def manual_fix_hint(doc: SourceDoc) -> str:
    return (f"Open {PUBLICATIONS_PAGE}, search for product number {doc.product_number}, "
            f"click 'Download PDF' and save it as data/raw/{doc.filename}")


def fetch_all(session: HttpSession, raw_dir: Path, force: bool = False) -> tuple[list[ManifestEntry], list[str]]:
    """Download (or reuse) every source and write manifest.json. Returns (entries, error lines)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []
    errors: list[str] = []
    for doc in SOURCES:
        dest = raw_dir / doc.filename
        try:
            if force or not dest.exists():
                download_pdf(session, doc.url, dest)
            entries.append(describe(doc, dest))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the others
            errors.append(f"{doc.filename}: {exc}\n     Manual fix: {manual_fix_hint(doc)}")
    (raw_dir / "manifest.json").write_text(json.dumps([asdict(e) for e in entries], indent=2), encoding="utf-8")
    return entries, errors
