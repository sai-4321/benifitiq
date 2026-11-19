"""Response cache. Same (masked) question + same settings -> return the saved answer for $0.

Only fully-completed, non-member, no-review answers are cached (see app/pipeline.py),
so no member-specific data is ever stored here.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import diskcache

from app.config import get_settings


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def make_key(masked_question: str, provider: str, prompt_version: str, top_k: int) -> str:
    raw = f"{normalize(masked_question)}|{provider}|{prompt_version}|k={top_k}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: str, ttl_seconds: int):
        self._cache = diskcache.Cache(directory)
        self.ttl = ttl_seconds

    def get(self, key: str) -> dict | None:
        return self._cache.get(key)

    def set(self, key: str, value: dict) -> None:
        self._cache.set(key, value, expire=self.ttl)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


@lru_cache(maxsize=1)
def get_cache() -> ResponseCache:
    s = get_settings()
    s.cache_path.mkdir(parents=True, exist_ok=True)
    return ResponseCache(str(s.cache_path), s.cache_ttl_seconds)
