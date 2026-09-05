"""Smoke tests for the provided Topic 3 AI module.

Exercises the public interface end-to-end with fake providers (no network).
Students MUST NOT delete or weaken these tests — they are part of the
grading contract. Add your own tests in tests/test_*.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai import (
    summarize_and_label, embed,
    url_canonicalize, content_hash, near_duplicate, jaccard,
    LabeledSummary, Topic, Sentiment,
)
from ai.providers.base import ProviderError


# --- summarize_and_label ---------------------------------------------------


def test_summarize_and_label_happy_path(fake_llm, sample_article):
    result = summarize_and_label(sample_article, llm=fake_llm)
    assert isinstance(result, LabeledSummary)
    assert result.topic == Topic.TECH
    assert result.sentiment == Sentiment.NEUTRAL
    assert result.summary  # non-empty


def test_summarize_and_label_rejects_empty_content(fake_llm, sample_article):
    sample_article.content = "   "
    with pytest.raises(ValueError):
        summarize_and_label(sample_article, llm=fake_llm)


def test_summarize_and_label_rejects_invalid_topic(fake_llm, sample_article):
    fake_llm.payload = {**fake_llm.payload, "topic": "Cryptocurrency"}
    with pytest.raises(ProviderError):
        summarize_and_label(sample_article, llm=fake_llm)


def test_summarize_and_label_rejects_invalid_sentiment(fake_llm, sample_article):
    fake_llm.payload = {**fake_llm.payload, "sentiment": "ecstatic"}
    with pytest.raises(ProviderError):
        summarize_and_label(sample_article, llm=fake_llm)


# --- embed ----------------------------------------------------------------


def test_embed_returns_unit_vector(fake_embedder):
    v = embed("breaking news today", embedder=fake_embedder)
    assert v.shape == (8,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_embed_rejects_empty(fake_embedder):
    with pytest.raises(ValueError):
        embed("", embedder=fake_embedder)


# --- url_canonicalize -----------------------------------------------------


def test_url_canonicalize_strips_tracking_params():
    a = url_canonicalize("https://example.com/post?utm_source=fb&id=42")
    b = url_canonicalize("https://example.com/post?id=42&fbclid=xyz")
    assert a == b == "https://example.com/post?id=42"


def test_url_canonicalize_drops_fragment_and_lowercases_host():
    out = url_canonicalize("HTTPS://Example.COM/article#section-2")
    assert out == "https://example.com/article"


def test_url_canonicalize_rejects_malformed():
    with pytest.raises(ValueError):
        url_canonicalize("not a url")


# --- content_hash ---------------------------------------------------------


def test_content_hash_is_whitespace_invariant():
    a = content_hash("hello   world\n\nfoo")
    b = content_hash("hello world foo")
    assert a == b


def test_content_hash_distinguishes_different_text():
    assert content_hash("a story") != content_hash("a different story")


# --- near_duplicate -------------------------------------------------------


def test_near_duplicate_same_text():
    assert near_duplicate("the quick brown fox jumps", "the quick brown fox jumps")


def test_near_duplicate_different_text():
    assert not near_duplicate(
        "Acme unveils a new processor for AI workloads.",
        "Local bakery wins national pastry award.",
    )


def test_jaccard_bounds():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_labeled_summary_rejects_extra_fields():
    """Pydantic ConfigDict(extra='forbid') enforces the schema contract."""
    with pytest.raises(Exception):  # pydantic.ValidationError
        LabeledSummary(
            summary="x", topic=Topic.TECH, sentiment=Sentiment.NEUTRAL,
            totally_unknown_field=42,  # type: ignore[call-arg]
        )
