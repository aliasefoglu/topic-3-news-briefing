"""Retry / timeout / caching wrapper around the provided `ai` package.

Per TOPIC.md's contract ("do not call provider SDKs directly from your
business logic"), this module is the *only* place in the SE layer allowed
to call `ai.summarize_and_label`. Everything else in `src/` depends on
`AIService`, never on `ai` directly (Dependency Inversion).
"""

from __future__ import annotations

import asyncio
import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import ai
from ai.providers.base import ProviderError
from src.models import Article as SEArticle
from src.models import LabeledSummary as SELabeledSummary
from src.models import SentimentEnum, TopicEnum
from src.storage.repository import Repository

logger = logging.getLogger(__name__)

_AI_CALL_TIMEOUT_SECONDS = 30.0


def _to_ai_article(article: SEArticle) -> ai.Article:
    """Translate our persistence-flavoured Article into the AI package's
    own schema. The two are deliberately decoupled: `ai.Article` is the
    provided package's frozen contract; `src.models.Article` carries
    fields (id, canonical_url, content_hash) the AI package never needs."""
    return ai.Article(
        title=article.title,
        url=article.url,
        source=article.source,
        content=article.content,
        published_at=article.published_at,
    )


def _from_ai_labeled_summary(labeled: ai.LabeledSummary) -> SELabeledSummary:
    return SELabeledSummary(
        summary=labeled.summary,
        topic=TopicEnum(labeled.topic.value),
        sentiment=SentimentEnum(labeled.sentiment.value),
    )


class AIService:
    """Wraps `ai.summarize_and_label` with retries, a timeout, structured
    logging, and a content-hash cache backed by PostgreSQL.

    Caching means re-running the digest for an already-seen article body
    never re-calls the LLM (TOPIC.md, "Tips for the SE layer").
    """

    def __init__(self, repository: Repository) -> None:
        self._repository = repository

    async def summarize_and_label(self, article: SEArticle) -> SELabeledSummary:
        if article.content_hash is None:
            raise ValueError(
                "article.content_hash must be set before calling AIService "
                "(run the article through Deduplicator.normalize first)."
            )

        cached = await self._repository.get_cached_summary(article.content_hash)
        if cached is not None:
            logger.info("AI cache hit for %r (%s)", article.title, article.content_hash[:12])
            return cached

        logger.info("Calling LLM for %r (%s)", article.title, article.content_hash[:12])
        labeled = await self._call_llm_with_retry(article)

        await self._repository.save_processed_article(
            canonical_url=article.canonical_url or article.url,
            content_hash=article.content_hash,
            labeled_summary=labeled,
        )
        return labeled

    async def _call_llm_with_retry(self, article: SEArticle) -> SELabeledSummary:
        @retry(
            reraise=True,
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(ProviderError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        async def _attempt() -> SELabeledSummary:
            ai_article = _to_ai_article(article)
            result = await asyncio.wait_for(
                asyncio.to_thread(ai.summarize_and_label, ai_article),
                timeout=_AI_CALL_TIMEOUT_SECONDS,
            )
            return _from_ai_labeled_summary(result)

        return await _attempt()
