"""Split pages into small overlapping chunks.

Why ~800 characters with 120 overlap?
- Small enough that one chunk is about one topic (better retrieval precision).
- Big enough to hold a full rule ("Medicare covers X once every 12 months if...").
- Overlap keeps a sentence that crosses a chunk boundary findable from both sides.
Chunks never cross page boundaries, so every chunk has exactly one page to cite.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import PageDoc

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int

    def to_dict(self) -> dict:
        return asdict(self)


def doc_code(source: str) -> str:
    """'10050-Medicare-and-You.pdf' -> '10050' (the CMS product number)."""
    return source.split("-")[0].split(".")[0]


def chunk_pages(pages: list[PageDoc], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if len(piece) < 30:
                continue
            chunks.append(Chunk(id=f"{doc_code(page.source)}-p{page.page}-c{i}", text=piece,
                                source=page.source, page=page.page))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """One JSON object per line (JSONL). The BM25 index (BIQ-9) and evals read this file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
