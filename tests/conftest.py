"""Shared fixtures for Topic 3 smoke tests."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from ai.providers.base import LLMProvider, EmbeddingProvider
from ai.schemas import Article


class FakeLLM(LLMProvider):
    """Returns a fixed JSON response. No network."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "summary": "A new chip from a major vendor was announced today.",
            "topic": "Tech",
            "sentiment": "neutral",
        }
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(prompt)
        return json.dumps(self.payload)


class FakeEmbedder(EmbeddingProvider):
    """Deterministic 8-D unit vector from a hash; same input -> same output."""

    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed empty string.")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        v = rng.standard_normal(8).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_article() -> Article:
    return Article(
        title="Acme unveils new processor",
        url="https://example.com/news/acme-chip?utm_source=twitter",
        source="Example News",
        content="Acme Corp announced a new processor today aimed at AI workloads. "
                "The chip features improved energy efficiency over the previous "
                "generation. Industry analysts welcomed the news.",
    )
