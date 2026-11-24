"""Downloader tests with a fake HTTP session - no network, no Medicare.gov dependency in CI."""
from __future__ import annotations

import io
import json

import pytest
import requests
from pypdf import PdfWriter

from app.data.documents import SOURCES, NotAPdfError, download_pdf, fetch_all


def tiny_pdf(pages: int = 2) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "application/pdf", status: int = 200):
        self.body, self.status = body, status
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Answers each URL with a canned response and records what was requested."""

    def __init__(self, responses: dict[str, FakeResponse] | None = None, default: FakeResponse | None = None):
        self.responses, self.default, self.calls = responses or {}, default, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.get(url, self.default)


def test_sources_are_public_medicare_pdfs():
    assert len(SOURCES) == 4
    for doc in SOURCES:
        assert doc.url.startswith("https://www.medicare.gov/publications/")
        assert doc.url.endswith(".pdf") and doc.filename.endswith(".pdf")
        assert doc.product_number.isdigit()
        assert doc.url.split("/")[-1].startswith(doc.product_number)   # URL and local name refer to the same booklet


def test_download_sends_user_agent_and_timeout(tmp_path):
    session = FakeSession(default=FakeResponse(tiny_pdf()))
    download_pdf(session, "https://example.org/a.pdf", tmp_path / "a.pdf")
    _, kwargs = session.calls[0]
    assert "BenefitIQ" in kwargs["headers"]["User-Agent"]
    assert kwargs["timeout"] == 60 and kwargs["stream"] is True
    assert (tmp_path / "a.pdf").read_bytes().startswith(b"%PDF")


def test_html_page_is_rejected_and_nothing_written(tmp_path):
    session = FakeSession(default=FakeResponse(b"<html>blocked</html>", content_type="text/html"))
    with pytest.raises(NotAPdfError):
        download_pdf(session, "https://example.org/a.pdf", tmp_path / "a.pdf")
    assert list(tmp_path.iterdir()) == []      # no half-written file, no .part file


def test_fetch_all_writes_manifest_with_fingerprints(tmp_path):
    session = FakeSession(default=FakeResponse(tiny_pdf(pages=3)))
    entries, errors = fetch_all(session, tmp_path)
    assert errors == [] and len(entries) == 4
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert {m["file"] for m in manifest} == {d.filename for d in SOURCES}
    assert all(m["pages"] == 3 and len(m["sha256_16"]) == 16 and "Public domain" in m["license"] for m in manifest)


def test_existing_files_are_skipped_unless_forced(tmp_path):
    session = FakeSession(default=FakeResponse(tiny_pdf()))
    fetch_all(session, tmp_path)
    assert len(session.calls) == 4
    fetch_all(session, tmp_path)
    assert len(session.calls) == 4             # nothing re-downloaded
    fetch_all(session, tmp_path, force=True)
    assert len(session.calls) == 8


def test_one_failure_does_not_stop_the_others(tmp_path):
    bad = SOURCES[0]
    session = FakeSession(
        responses={bad.url: FakeResponse(b"", status=503)},
        default=FakeResponse(tiny_pdf()),
    )
    entries, errors = fetch_all(session, tmp_path)
    assert len(entries) == 3 and len(errors) == 1
    assert bad.filename in errors[0] and bad.product_number in errors[0]   # manual-fix hint names the product


def test_corrupt_pdf_is_reported(tmp_path):
    session = FakeSession(default=FakeResponse(b"%PDF-1.7 this is not really a pdf"))
    entries, errors = fetch_all(session, tmp_path)
    assert entries == [] and len(errors) == 4
