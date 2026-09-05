"""Pydantic schemas for Topic 3 — AI News Briefing Service."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Topic(str, Enum):
    """Fixed taxonomy of news topics. The LLM must classify into one of these."""

    POLITICS = "Politics"
    TECH = "Tech"
    SPORTS = "Sports"
    BUSINESS = "Business"
    HEALTH = "Health"
    SCIENCE = "Science"
    WORLD = "World"
    ENTERTAINMENT = "Entertainment"
    OTHER = "Other"

    @classmethod
    def values(cls) -> list[str]:
        return [t.value for t in cls]


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


# JSON schema we ask the LLM to honour when summarizing + labeling an article.
LABELED_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topic": {"type": "string", "enum": Topic.values()},
        "sentiment": {
            "type": "string",
            "enum": [s.value for s in Sentiment],
        },
    },
    "required": ["summary", "topic", "sentiment"],
    "additionalProperties": False,
}


class Article(BaseModel):
    """A raw article ingested from a feed or a scrape.

    `published_at` is a datetime if known, else None. `content` is the body
    text (HTML stripped). `source` is a short human label like "BBC" or
    "TechCrunch".
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    source: str
    content: str
    published_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")  # datetime → isoformat string


class LabeledSummary(BaseModel):
    """The LLM's structured output for a single article."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str
    topic: Topic
    sentiment: Sentiment

    @field_validator("summary")
    @classmethod
    def _summary_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary must be non-empty")
        return v


class DigestItem(BaseModel):
    """One entry in a digest: an article plus its labeled summary."""

    model_config = ConfigDict(extra="forbid")

    article: Article
    labeled: LabeledSummary


class Digest(BaseModel):
    """A user-facing digest: items grouped by topic."""

    model_config = ConfigDict(extra="forbid")

    user: str
    generated_at: datetime
    items: list[DigestItem] = Field(default_factory=list)

    def by_topic(self) -> dict[Topic, list[DigestItem]]:
        out: dict[Topic, list[DigestItem]] = {}
        for it in self.items:
            out.setdefault(it.labeled.topic, []).append(it)
        return out
