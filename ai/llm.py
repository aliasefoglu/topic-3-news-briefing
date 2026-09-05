"""High-level LLM call: summarize and label an article."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ai.providers.base import LLMProvider, ProviderError
from ai.providers.factory import get_llm
from ai.schemas import Article, LabeledSummary, Topic, LABELED_SUMMARY_SCHEMA


_PROMPT_TEMPLATE = """You are a news editor preparing a daily digest.

Read the article and produce three things:

1. "summary": a concise 2-3 sentence summary in plain English. No editorialising,
   no rhetorical questions. Just the facts.
2. "topic": the single best-fit topic from this exact list:
   {topics}
   If nothing fits, use "Other".
3. "sentiment": "positive", "neutral", or "negative" — the overall tone of the
   reported events (NOT your opinion of the article).

Article title: {title}
Source: {source}

Article body:
---
{body}
---
"""


def summarize_and_label(
    article: Article,
    *,
    llm: LLMProvider | None = None,
) -> LabeledSummary:
    """Use an LLM to summarize and label an article.

    Raises
    ------
    ProviderError
        If the model errors or returns an unparseable / schema-invalid response.
    """
    if not article.content.strip():
        raise ValueError("Article content is empty.")

    llm = llm or get_llm()
    prompt = _PROMPT_TEMPLATE.format(
        topics=", ".join(Topic.values()),
        title=article.title,
        source=article.source,
        body=article.content[:6000],  # cap to keep prompts cheap
    )
    raw = llm.complete(prompt, json_schema=LABELED_SUMMARY_SCHEMA)
    payload = _parse_json(raw)

    try:
        return LabeledSummary.model_validate(payload)
    except ValidationError as e:
        raise ProviderError(f"LLM response failed schema validation: {e}") from e


def _parse_json(raw: str) -> dict[str, Any]:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        lines = lines[1:]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ProviderError(f"Could not parse JSON from LLM: {e}\nRaw: {raw[:300]!r}")
    if not isinstance(obj, dict):
        raise ProviderError(f"Expected JSON object, got {type(obj).__name__}")
    return obj
