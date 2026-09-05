"""
AI module for Topic 3 — AI News Briefing Service.

Public surface
--------------
summarize_and_label(article, *, llm=None) -> LabeledSummary
    Summarize an article and tag it with a topic + sentiment via LLM.

embed(text, *, embedder=None) -> np.ndarray
    Unit-normalized embedding for semantic dedup or topic clustering.

url_canonicalize(url) -> str
    Strip tracking params, normalize host/path/query.

content_hash(text) -> str
    Whitespace-normalized SHA-256 digest for cheap exact-dedup.

near_duplicate(a, b, *, threshold=0.7) -> bool
    Word k-shingle Jaccard similarity for semantic dedup.

Schemas: Article, LabeledSummary, Digest, DigestItem, Topic, Sentiment.
"""

from ai.schemas import (
    Article, LabeledSummary, Digest, DigestItem, Topic, Sentiment,
    LABELED_SUMMARY_SCHEMA,
)
from ai.llm import summarize_and_label
from ai.embedding import embed
from ai.dedup import url_canonicalize, content_hash, near_duplicate, jaccard

__all__ = [
    "Article", "LabeledSummary", "Digest", "DigestItem", "Topic", "Sentiment",
    "LABELED_SUMMARY_SCHEMA",
    "summarize_and_label",
    "embed",
    "url_canonicalize", "content_hash", "near_duplicate", "jaccard",
]
