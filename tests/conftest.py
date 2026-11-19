"""Shared test fixtures. Tests never call AWS or need API keys:
- embeddings use the offline HashEmbeddings
- the member DB, vector index and cache live in a temporary folder
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ---- Point every setting at a temp folder BEFORE any app module reads settings ----
_TMP = Path(tempfile.mkdtemp(prefix="benefitiq_test_"))
os.environ.update({
    "BEDROCK_FALLBACK_MODEL_ID": "none",  # get_llm is patched in tests - Bedrock is never called
    "AWS_REGION": "us-east-1",
    "EMBEDDING_PROVIDER": "fake",
    "LANGSMITH_TRACING": "false",
    "CHROMA_DIR": str(_TMP / "chroma"),
    "MEMBER_DB_PATH": str(_TMP / "members.db"),
    "CACHE_DIR": str(_TMP / "cache"),
    "INJECTION_LLM_CHECK": "false",
    "HITL_CONFIDENCE_THRESHOLD": "0.6",
})

from app.config import get_settings  # noqa: E402

SAMPLE_CHUNKS = [
    {"id": "10050-p20-c0", "source": "10050-Medicare-and-You.pdf", "page": 20,
     "text": "Part A (Hospital Insurance) helps cover inpatient care in hospitals, skilled nursing facility care, "
             "hospice care, and home health care."},
    {"id": "10050-p21-c0", "source": "10050-Medicare-and-You.pdf", "page": 21,
     "text": "Part B (Medical Insurance) helps cover services from doctors and other health care providers, "
             "outpatient care, durable medical equipment, and many preventive services."},
    {"id": "10110-p15-c1", "source": "10110-Your-Guide-to-Medicare-Preventive-Services.pdf", "page": 15,
     "text": "Mammograms: Medicare covers screening mammograms once every 12 months for women 40 and older."},
    {"id": "11045-p3-c0", "source": "11045-medicare-coverage-of-dme.pdf", "page": 3,
     "text": "Durable medical equipment (DME) is reusable medical equipment, like walkers, wheelchairs, or "
             "hospital beds. You pay 20% of the Medicare-approved amount, and the Part B deductible applies."},
    {"id": "11109-p8-c2", "source": "11109-your-guide-to-medicare-prescription-drug-coverage.pdf", "page": 8,
     "text": "If you go 63 days or more in a row without Part D or other creditable prescription drug coverage, "
             "you may owe a late enrollment penalty."},
]


@pytest.fixture(scope="session")
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def member_db(settings):
    from app.data.synthetic import build_database

    build_database(settings.member_db)
    return settings.member_db


@pytest.fixture
def sample_chunks():
    return [dict(c) for c in SAMPLE_CHUNKS]


@pytest.fixture
def retriever(settings, sample_chunks):
    """A real HybridRetriever (BM25 + Chroma with hash embeddings) over the sample chunks."""
    from app.llm_factory import HashEmbeddings
    from app.rag.hybrid_retriever import HybridRetriever
    from app.rag.vector_store import ChromaStore

    store = ChromaStore(settings.chroma_path, HashEmbeddings(), collection="test_chunks")
    store.reset()
    store.add(sample_chunks)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    with settings.chunks_file.open("w", encoding="utf-8") as f:
        for c in sample_chunks:
            f.write(json.dumps(c) + "\n")
    return HybridRetriever(sample_chunks, store)


def make_llm(*json_replies: dict | str):
    """Fake chat model that returns the given replies in order (dicts are JSON-encoded)."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=[r if isinstance(r, str) else json.dumps(r) for r in json_replies])
