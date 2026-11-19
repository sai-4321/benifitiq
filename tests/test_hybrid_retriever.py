"""Hybrid retrieval tests over five real-looking sample chunks (hash embeddings, no AWS)."""
from __future__ import annotations

from app.rag.hybrid_retriever import rrf_fuse, tokenize


def test_rrf_rewards_agreement():
    fused = rrf_fuse({"vector": ["a", "b", "c"], "bm25": ["b", "d"]})
    ids = [cid for cid, _, _ in fused]
    assert ids[0] == "b"  # found by both retrievers -> ranked first
    assert set(ids) == {"a", "b", "c", "d"}
    ranks = dict((cid, r) for cid, _, r in fused)
    assert ranks["b"] == {"vector": 2, "bm25": 1}


def test_tokenize_drops_stopwords():
    assert tokenize("What does Part B cover?") == ["part", "b", "cover"]


def test_keyword_search_finds_exact_term(retriever):
    ids = retriever.bm25_search("63 days late enrollment penalty", 3)
    assert ids[0] == "11109-p8-c2"


def test_hybrid_retrieve_returns_ranked_chunks(retriever):
    hits = retriever.retrieve("How often are screening mammograms covered?", top_k=3)
    assert hits[0]["id"] == "10110-p15-c1"
    assert {"rrf_score", "vector_rank", "bm25_rank", "source", "page", "text"} <= hits[0].keys()
    assert len(hits) <= 3


def test_hybrid_survives_vector_failure(retriever, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(retriever.store, "query", boom)
    hits = retriever.retrieve("walkers wheelchairs durable medical equipment", top_k=2)
    assert hits and hits[0]["id"] == "11045-p3-c0"  # keyword search still works


def test_rrf_scores_are_reciprocal_ranks():
    fused = dict((cid, score) for cid, score, _ in rrf_fuse({"vector": ["a"], "bm25": ["a"]}))
    assert fused["a"] == round(2 / 61, 6)


def test_no_store_means_keyword_only(sample_chunks):
    from app.rag.hybrid_retriever import HybridRetriever

    r = HybridRetriever(sample_chunks, store=None)
    hits = r.retrieve("late enrollment penalty Part D", top_k=1)
    assert hits[0]["id"] == "11109-p8-c2" and hits[0]["vector_rank"] is None


def test_empty_index_explains_fix():
    import pytest

    from app.rag.hybrid_retriever import HybridRetriever

    with pytest.raises(ValueError, match="ingest.py"):
        HybridRetriever([], store=None)


def test_from_disk_reads_chunks_file(settings, retriever):
    from app.rag.hybrid_retriever import HybridRetriever

    r = HybridRetriever.from_disk(settings.chunks_file, store=None)
    assert len(r.chunks) == 5


def test_vector_store_similarity_and_reset(retriever):
    hits = retriever.store.query("mammograms screening", k=2)
    assert hits[0]["id"] == "10110-p15-c1" and 0 < hits[0]["similarity"] <= 1
    retriever.store.reset()
    assert retriever.store.count() == 0 and retriever.store.query("anything") == []
