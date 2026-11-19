"""Thin wrapper around a persistent ChromaDB collection.

Production mapping on AWS: this class becomes an Amazon OpenSearch Serverless
vector collection (k-NN index with a vector field + a searchable text field), or a
Bedrock Knowledge Base that manages chunking/embedding/storage for you.
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

COLLECTION = "benefitiq_chunks"


class ChromaStore:
    def __init__(self, persist_dir: Path, embeddings: Embeddings, collection: str = COLLECTION):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embeddings = embeddings
        self.collection_name = collection
        self.collection = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        """Delete and recreate the collection (used by ingest --reset)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[dict], batch_size: int = 64) -> int:
        added = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embeddings.embed_documents([c["text"] for c in batch])
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=vectors,
                metadatas=[{"source": c["source"], "page": int(c["page"])} for c in batch],
            )
            added += len(batch)
        return added

    def query(self, text: str, k: int = 10) -> list[dict]:
        if self.count() == 0:
            return []
        vector = self.embeddings.embed_query(text)
        res = self.collection.query(query_embeddings=[vector], n_results=min(k, self.count()))
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=False
        ):
            out.append({"id": cid, "text": doc, "source": meta["source"], "page": int(meta["page"]),
                        "similarity": round(1.0 - float(dist), 4)})
        return out

    def count(self) -> int:
        return self.collection.count()
