import asyncpg
from typing import Optional
import json
from src.config import settings
from src.models import (LabeledSummary, SentimentEnum, TopicEnum, User)
class Repository:
    """It provides asynchronous access to the PostgreSQL database"""
    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """It creates a PostgreSQL connection pool for asynchronous access to the database."""
        if self.pool is not None:
            return  # Already connected

        # asyncpg expects a standard PostgreSQL DSN.
        database_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        try:
            self.pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=10)
        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to connect to the database: {e}") from e

    async def close(self) -> None:
        """It closes the PostgreSQL connection pool."""
        if self.pool is None:
            return  # Already closed
        try:
            await self.pool.close()
        finally:
            self.pool = None

    def _get_pool(self) -> asyncpg.Pool:
        """It returns the PostgreSQL connection pool, raising an error if not connected."""
        if self.pool is None:
            raise RuntimeError("Database connection pool is not initialized. Call connect() first.")
        return self.pool

    async def get_user(self, user_id: str) -> Optional[User]:
        """It returns a user by ID, or None if the user does not exist."""
        pool = self._get_pool()
        try:
            row = await pool.fetchrow("""SELECT user_id, preferred_topics, excluded_sources, preferred_length FROM users WHERE user_id = $1""", user_id)
        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to fetch user {user_id} from the database: {e}") from e
        if row is None:
            return None
        preferred_topics = json.loads(row["preferred_topics"])
        excluded_sources = json.loads(row["excluded_sources"])
        return User(
            user_id = row["user_id"],
            preferred_topics = [TopicEnum(topic) for topic in preferred_topics],
            excluded_sources = excluded_sources,
            preferred_length = row["preferred_length"]
        )

    async def save_user(self, user: User) -> None:
        """It adds a new user or updates an existing user in the database."""
        pool = self._get_pool()
        preferred_topics = [topic.value for topic in user.preferred_topics]
        try:
            await pool.execute(
                """INSERT INTO users (user_id, preferred_topics, excluded_sources, preferred_length)
                   VALUES ($1, $2::jsonb, $3::jsonb, $4)
                   ON CONFLICT (user_id) DO UPDATE
                   SET preferred_topics = EXCLUDED.preferred_topics,
                       excluded_sources = EXCLUDED.excluded_sources,
                       preferred_length = EXCLUDED.preferred_length""",
                user.user_id,
                json.dumps(preferred_topics),
                json.dumps(user.excluded_sources),
                user.preferred_length
            )
        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to save user {user.user_id} to the database: {e}") from e

    async def delete_user(self, user_id: str) -> None:
        """It deletes a user by ID from the database."""
        pool = self._get_pool()
        try:
            await pool.execute("""DELETE FROM users WHERE user_id = $1""", user_id)
        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to delete user {user_id} from the database: {e}") from e

    async def get_cached_summary(self, content_hash: str) -> Optional[LabeledSummary]:
        """It returns a cached summary by content hash, or None if not found."""
        pool = self._get_pool()
        try:
            row = await pool.fetchrow("""SELECT summary, topic, sentiment FROM processed_urls WHERE content_hash = $1""", content_hash)
        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to fetch cached summary for content hash {content_hash} from the database: {e}") from e
        if row is None or row["summary"] is None:
            return None
        return LabeledSummary(
            summary=row["summary"],
            topic=TopicEnum(row["topic"]),
            sentiment=SentimentEnum(row["sentiment"])
        )

    async def save_processed_article(self, canonical_url: str, content_hash: str, labeled_summary: LabeledSummary) -> None:
        """It Caches an AI-labeled article result.
        The content hash is used as the conflict target to prevent the same article content from being cached more than once."""
        pool = self._get_pool()
        try:
            await pool.execute(
                """INSERT INTO processed_urls (canonical_url, content_hash, summary, topic, sentiment)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (content_hash) DO UPDATE
                   SET summary = EXCLUDED.summary,
                       topic = EXCLUDED.topic,
                       sentiment = EXCLUDED.sentiment""",
                canonical_url,
                content_hash,
                labeled_summary.summary,
                labeled_summary.topic.value,
                labeled_summary.sentiment.value
            )
        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to save processed article for content hash {content_hash} to the database: {e}") from e

    async def is_url_processed(self, canonical_url: str) -> bool:
        """It checks if a canonical URL is already present in the processed_urls table."""
        pool = self._get_pool()
        try:
            result = await pool.fetchval("""SELECT EXISTS(SELECT 1 FROM processed_urls WHERE canonical_url = $1)""", canonical_url)
            return result is True
        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to check if URL {canonical_url} is processed in the database: {e}") from e