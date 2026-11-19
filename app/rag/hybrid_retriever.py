"""Hybrid retrieval = vector search (meaning) + BM25 keyword search (exact terms),
merged with Reciprocal Rank Fusion (RRF).

Why both?
- Vector search finds "walker" when the question says "mobility aid".
- BM25 finds exact terms embeddings blur: "Part B", "63 days", "DME", form numbers.
RRF score for a chunk = sum over retrievers of 1 / (60 + rank). It needs no score
normalisation, which is why Amazon OpenSearch (hybrid search) offers the same formula for hybrid queries.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.core.logging_config import get_logger
from app.rag.chunker import load_chunks
from app.rag.vector_store import ChromaStore

log = get_logger(__name__)
RRF_K = 60

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "and", "or", "in", "on", "for", "with", "does",
    "do", "what", "how", "when", "who", "which", "my", "i", "me", "it", "this", "that", "can", "if", "by",
    "at", "as", "from", "your", "you", "will", "should", "would", "about", "there", "their", "they",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def rrf_fuse(ranked_lists: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, dict[str, int]]]:
    """ranked_lists = {"vector": [id1, id2...], "bm25": [...]} -> [(id, score, {retriever: rank})] best first."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in ranked_lists.items():
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, {})[name] = rank
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(cid, round(score, 6), ranks[cid]) for cid, score in ordered]


class HybridRetriever:
    def __init__(self, chunks: list[dict], store: ChromaStore | None, candidates: int = 20):
        if not chunks:
            raise ValueError("No chunks loaded - run: python scripts/ingest.py")
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.store = store
        self.candidates = candidates
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    @classmethod
    def from_disk(cls, chunks_file: Path, store: ChromaStore | None) -> HybridRetriever:
        if not chunks_file.exists():
            raise FileNotFoundError(f"{chunks_file} not found - run: python scripts/ingest.py")
        return cls([c.to_dict() for c in load_chunks(chunks_file)], store)

    def bm25_search(self, query: str, n: int) -> list[str]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i]["id"] for i in top if scores[i] > 0]

    def vector_search(self, query: str, n: int) -> list[str]:
        if self.store is None:
            return []
        try:
            return [hit["id"] for hit in self.store.query(query, n)]
        except Exception as exc:  # noqa: BLE001 - keyword search still works if embeddings fail
            log.warning("Vector search failed, using keyword-only results: %s", exc)
            return []

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        fused = rrf_fuse(
            {"vector": self.vector_search(query, self.candidates), "bm25": self.bm25_search(query, self.candidates)}
        )
        results = []
        for cid, score, ranks in fused[:top_k]:
            chunk = self.by_id.get(cid)
            if chunk is None:
                continue
            results.append({**chunk, "rrf_score": score, "vector_rank": ranks.get("vector"),
                            "bm25_rank": ranks.get("bm25")})
        return results


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    from app.llm_factory import get_embeddings

    s = get_settings()
    store = ChromaStore(s.chroma_path, get_embeddings())
    retriever = HybridRetriever.from_disk(s.chunks_file, store)
    log.info("Retriever ready: %d chunks (vector index has %d)", len(retriever.chunks), store.count())
    return retriever
