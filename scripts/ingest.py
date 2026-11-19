"""Build the search indexes: PDFs -> pages -> chunks -> embeddings -> ChromaDB (+ chunks.jsonl for BM25).

Usage:  python scripts/ingest.py           (adds/updates chunks)
        python scripts/ingest.py --reset   (wipe the vector index first - use after changing
                                            EMBEDDING_PROVIDER, BEDROCK_EMBED_DIMENSIONS or chunk settings)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.llm_factory import get_embeddings  # noqa: E402
from app.rag.chunker import chunk_pages, save_chunks  # noqa: E402
from app.rag.loader import load_directory  # noqa: E402
from app.rag.vector_store import ChromaStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete the existing vector index first")
    args = parser.parse_args()

    s = get_settings()
    setup_logging(s.log_level)
    t0 = time.perf_counter()

    print(f"1/4 Loading PDFs from {s.raw_docs_path}")
    pages = load_directory(s.raw_docs_path)

    print(f"2/4 Chunking {len(pages)} pages")
    chunk_objs = chunk_pages(pages)
    chunks = [c.to_dict() for c in chunk_objs]
    per_doc = Counter(c["source"] for c in chunks)
    for src, n in per_doc.items():
        print(f"     {src:<62} {n:>5} chunks")
    avg = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)
    print(f"     total {len(chunks)} chunks, average {avg:.0f} characters")

    print("3/4 Saving chunk file for keyword (BM25) search")
    save_chunks(chunk_objs, s.chunks_file)

    print(f"4/4 Embedding with provider '{s.embedding_provider}' and writing to ChromaDB "
          "(about 1-3 minutes on cloud embeddings)")
    store = ChromaStore(s.chroma_path, get_embeddings())
    if args.reset:
        store.reset()
    store.add(chunks)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Vector index now holds {store.count()} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
