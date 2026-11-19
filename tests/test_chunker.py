"""Loader and chunker tests."""
from __future__ import annotations

from app.rag.chunker import CHUNK_SIZE, chunk_pages, doc_code, load_chunks, save_chunks
from app.rag.loader import PageDoc, clean_text, load_directory, load_pdf


def test_doc_code():
    assert doc_code("10050-Medicare-and-You.pdf") == "10050"
    assert doc_code("11045-medicare-coverage-of-dme.pdf") == "11045"


def test_chunks_respect_size_and_keep_metadata():
    text = " ".join(f"Sentence number {i} about Medicare Part B coverage rules." for i in range(200))
    chunks = chunk_pages([PageDoc(text=text, source="10050-Medicare-and-You.pdf", page=12)])
    assert len(chunks) > 5
    assert all(len(c.text) <= CHUNK_SIZE for c in chunks)
    assert all(c.page == 12 and c.source == "10050-Medicare-and-You.pdf" for c in chunks)
    assert chunks[0].id == "10050-p12-c0"
    assert len({c.id for c in chunks}) == len(chunks)  # ids are unique


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_pages([PageDoc(text=text, source="x.pdf", page=1)], chunk_size=200, overlap=50)
    first_tail = chunks[0].text.split()[-3:]
    assert any(tok in chunks[1].text.split() for tok in first_tail)


def test_chunks_never_cross_pages():
    pages = [PageDoc(text="A" * 50 + " page one text", source="d.pdf", page=1),
             PageDoc(text="B" * 50 + " page two text", source="d.pdf", page=2)]
    chunks = chunk_pages(pages)
    assert [c.page for c in chunks] == [1, 2]


def test_clean_text_joins_hyphenated_words():
    assert clean_text("equip-\nment  is   covered") == "equipment is covered"


def test_tiny_fragments_dropped():
    assert chunk_pages([PageDoc(text="Too short.", source="d.pdf", page=1)]) == []


def test_jsonl_roundtrip(tmp_path):
    text = " ".join(f"Rule {i}: Part B covers it." for i in range(100))
    chunks = chunk_pages([PageDoc(text=text, source="11045-medicare-coverage-of-dme.pdf", page=3)])
    path = tmp_path / "chroma" / "chunks.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(chunks)


def test_clean_text_normalises_quotes_dashes_and_blank_lines():
    assert clean_text("Medicare’s  Part B – covered\n\n\n\nnext") == "Medicare's Part B - covered\n\nnext"


def _pdf(path, pages):
    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        w.write(f)


def test_blank_pages_are_skipped(tmp_path):
    _pdf(tmp_path / "10050-x.pdf", 3)
    assert load_pdf(tmp_path / "10050-x.pdf") == []


def test_load_directory_without_pdfs_explains_fix(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="download_docs.py"):
        load_directory(tmp_path)


def test_unreadable_page_is_skipped_not_fatal(tmp_path, monkeypatch):
    _pdf(tmp_path / "d.pdf", 2)
    from pypdf import PageObject

    calls = {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("broken font")
        return "Medicare Part B covers durable medical equipment such as walkers."

    monkeypatch.setattr(PageObject, "extract_text", flaky)
    pages = load_pdf(tmp_path / "d.pdf")
    assert [p.page for p in pages] == [2]
