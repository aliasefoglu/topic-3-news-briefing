"""Unit tests for the PostgreSQL repository"""
import json
from unittest.mock import AsyncMock, patch
import asyncpg
import pytest
from src.models import LabeledSummary, SentimentEnum, TopicEnum, User
from src.storage.repository import Repository

@pytest.fixture
def repository() -> Repository:
    """It creates a Repository instance for testing."""
    return Repository()

@pytest.fixture
def mock_pool() -> AsyncMock:
    """It creates a mock asynchronous PostgreSQL connection pool."""
    return AsyncMock()

@pytest.fixture
def sample_user() -> User:
    """It creates a sample User instance for testing."""
    return User(
        user_id="ali",
        preferred_topics=[TopicEnum.TECH, TopicEnum.SCIENCE],
        excluded_sources=["Example News"],
        preferred_length="medium"
    )

@pytest.fixture
def sample_summary() -> LabeledSummary:
    """It creates a sample LabeledSummary instance for testing."""
    return LabeledSummary(
        summary="A technology article summary.",
        topic=TopicEnum.TECH,
        sentiment=SentimentEnum.POSITIVE
    )

## Connection Tests
@pytest.mark.asyncio
async def test_connect_creates_connection_pool(repository: Repository) -> None:
    """Connect creates a PostgreSQL connection pool with normalized settings."""
    mock_pool = AsyncMock()
    with patch("src.storage.repository.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)) as create_pool:
        await repository.connect()
    create_pool.assert_awaited_once()
    call_kwargs = create_pool.call_args.kwargs
    assert call_kwargs["dsn"].startswith("postgresql://")
    assert call_kwargs["min_size"] == 1
    assert call_kwargs["max_size"] == 10
    assert repository.pool is mock_pool

@pytest.mark.asyncio
async def test_connect_normalizes_asyncpg_database_url(repository: Repository) -> None:
    """Connect normalizes the database URL for asyncpg."""
    mock_pool = AsyncMock()
    with patch("src.storage.repository.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)) as create_pool:
        await repository.connect()
    dsn = create_pool.call_args.kwargs["dsn"]
    assert "postgresql+asyncpg://" not in dsn

@pytest.mark.asyncio
async def test_connect_is_idempotent(repository: Repository, mock_pool: AsyncMock) -> None:
    """Connect does not create a second pool when already connected."""
    repository.pool = mock_pool
    with patch("src.storage.repository.asyncpg.create_pool", new=AsyncMock()) as create_pool:
        await repository.connect()
    create_pool.assert_not_awaited()
    assert repository.pool is mock_pool

@pytest.mark.asyncio
async def test_connect_raises_runtime_error_on_database_failure(repository: Repository) -> None:
    """Connect converts PostgreSQL connection errors to RuntimeError."""
    with patch("src.storage.repository.asyncpg.create_pool", new=AsyncMock(side_effect=asyncpg.PostgresError("connection failed"))):
        with pytest.raises(RuntimeError, match="Failed to connect to the database"):
            await repository.connect()
    assert repository.pool is None

@pytest.mark.asyncio
async def test_close_closes_connection_pool(repository: Repository, mock_pool: AsyncMock) -> None:
    """Close closes the pool and clears the repository pool reference."""
    repository.pool = mock_pool
    await repository.close()
    mock_pool.close.assert_awaited_once()
    assert repository.pool is None

@pytest.mark.asyncio
async def test_close_is_safe_when_not_connected(repository: Repository) -> None:
    """Close does nothing when no connection pool exists."""
    await repository.close()
    assert repository.pool is None


@pytest.mark.asyncio
async def test_close_clears_pool_reference_when_close_fails(repository: Repository, mock_pool: AsyncMock) -> None:
    """Close clears the pool reference even if closing the pool fails."""
    repository.pool = mock_pool
    mock_pool.close.side_effect = asyncpg.PostgresError("close failed")
    with pytest.raises(asyncpg.PostgresError, match="close failed"):
        await repository.close()
    assert repository.pool is None

# User operations

@pytest.mark.asyncio
async def test_get_user_returns_user(repository: Repository, mock_pool: AsyncMock, sample_user: User) -> None:
    """Get user converts a database row into a User model."""
    repository.pool = mock_pool
    mock_pool.fetchrow.return_value = {
        "user_id": sample_user.user_id,
        "preferred_topics": json.dumps([topic.value for topic in sample_user.preferred_topics]),
        "excluded_sources": json.dumps(sample_user.excluded_sources),
        "preferred_length": sample_user.preferred_length,
    }
    result = await repository.get_user(sample_user.user_id)
    assert result == sample_user
    mock_pool.fetchrow.assert_awaited_once()
    sql, user_id = mock_pool.fetchrow.call_args.args
    assert "SELECT user_id, preferred_topics, excluded_sources, preferred_length" in sql
    assert "FROM users" in sql
    assert "WHERE user_id = $1" in sql
    assert user_id == sample_user.user_id

@pytest.mark.asyncio
async def test_get_user_returns_none_when_user_does_not_exist(repository: Repository, mock_pool: AsyncMock) -> None:
    """Get user returns None when the database has no matching user."""
    repository.pool = mock_pool
    mock_pool.fetchrow.return_value = None
    result = await repository.get_user("unknown")
    assert result is None
    mock_pool.fetchrow.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_user_raises_runtime_error_on_database_failure(repository: Repository, mock_pool: AsyncMock) -> None:
    """Get user converts PostgreSQL errors to RuntimeError."""
    repository.pool = mock_pool
    mock_pool.fetchrow.side_effect = asyncpg.PostgresError("database error")
    with pytest.raises(RuntimeError, match="Failed to fetch user"):
        await repository.get_user("ali")

@pytest.mark.asyncio
async def test_save_user_serializes_user_data(repository: Repository, mock_pool: AsyncMock, sample_user: User) -> None:
    """Save user serializes enum and list values before database insertion."""
    repository.pool = mock_pool
    await repository.save_user(sample_user)
    mock_pool.execute.assert_awaited_once()
    (sql, user_id, preferred_topics, excluded_sources, preferred_length) = mock_pool.execute.call_args.args
    assert "INSERT INTO users" in sql
    assert "VALUES ($1, $2::jsonb, $3::jsonb, $4)" in sql
    assert "ON CONFLICT (user_id) DO UPDATE" in sql
    assert user_id == sample_user.user_id
    assert json.loads(preferred_topics) == [topic.value for topic in sample_user.preferred_topics]
    assert json.loads(excluded_sources) == sample_user.excluded_sources
    assert preferred_length == sample_user.preferred_length

@pytest.mark.asyncio
async def test_save_user_uses_parameterized_query(repository: Repository, mock_pool: AsyncMock, sample_user: User) -> None:
    """Save user passes user data as SQL parameters instead of interpolating it."""
    repository.pool = mock_pool
    await repository.save_user(sample_user)
    sql = mock_pool.execute.call_args.args[0]
    assert "$1" in sql
    assert "$2::jsonb" in sql
    assert "$3::jsonb" in sql
    assert "$4" in sql
    assert sample_user.user_id not in sql
    assert "Example News" not in sql

@pytest.mark.asyncio
async def test_save_user_raises_runtime_error_on_database_failure(repository: Repository, mock_pool: AsyncMock, sample_user: User) -> None:
    """Save user converts PostgreSQL errors to RuntimeError."""
    repository.pool = mock_pool
    mock_pool.execute.side_effect = asyncpg.PostgresError("database error")
    with pytest.raises(RuntimeError, match="Failed to save user"):
        await repository.save_user(sample_user)

@pytest.mark.asyncio
async def test_delete_user_uses_user_id_as_parameter(repository: Repository, mock_pool: AsyncMock) -> None:
    """Delete user removes the requested user using a parameterized query."""
    repository.pool = mock_pool
    await repository.delete_user("ali")
    mock_pool.execute.assert_awaited_once()
    sql, user_id = mock_pool.execute.call_args.args
    assert "DELETE FROM users" in sql
    assert "WHERE user_id = $1" in sql
    assert user_id == "ali"

@pytest.mark.asyncio
async def test_delete_user_raises_runtime_error_on_database_failure(repository: Repository, mock_pool: AsyncMock) -> None:
    """Delete user converts PostgreSQL errors to RuntimeError."""
    repository.pool = mock_pool
    mock_pool.execute.side_effect = asyncpg.PostgresError("database error")
    with pytest.raises(RuntimeError, match="Failed to delete user"):
        await repository.delete_user("ali")

# Cached AI summary operations

@pytest.mark.asyncio
async def test_get_cached_summary_returns_labeled_summary(repository: Repository, mock_pool: AsyncMock, sample_summary: LabeledSummary) -> None:
    """Get cached summary converts a database row into LabeledSummary."""
    repository.pool = mock_pool

    mock_pool.fetchrow.return_value = {
        "summary": sample_summary.summary,
        "topic": sample_summary.topic.value,
        "sentiment": sample_summary.sentiment.value,
    }
    result = await repository.get_cached_summary("a" * 64)
    assert result == sample_summary
    mock_pool.fetchrow.assert_awaited_once()
    sql, content_hash = mock_pool.fetchrow.call_args.args
    assert "SELECT summary, topic, sentiment" in sql
    assert "FROM processed_urls" in sql
    assert "WHERE content_hash = $1" in sql
    assert content_hash == "a" * 64

@pytest.mark.asyncio
async def test_get_cached_summary_returns_none_when_not_found(repository: Repository, mock_pool: AsyncMock) -> None:
    """Get cached summary returns None when no matching row exists."""
    repository.pool = mock_pool
    mock_pool.fetchrow.return_value = None
    result = await repository.get_cached_summary("a" * 64)
    assert result is None

@pytest.mark.asyncio
async def test_get_cached_summary_returns_none_when_summary_is_missing(repository: Repository, mock_pool: AsyncMock) -> None:
    """Get cached summary returns None when the database row has no summary."""
    repository.pool = mock_pool
    mock_pool.fetchrow.return_value = {"summary": None, "topic": None, "sentiment": None}
    result = await repository.get_cached_summary("a" * 64)
    assert result is None

@pytest.mark.asyncio
async def test_get_cached_summary_raises_runtime_error_on_database_failure(repository: Repository, mock_pool: AsyncMock) -> None:
    """Get cached summary converts PostgreSQL errors to RuntimeError."""
    repository.pool = mock_pool
    mock_pool.fetchrow.side_effect = asyncpg.PostgresError("database error")
    with pytest.raises(RuntimeError, match="Failed to fetch cached summary"):
        await repository.get_cached_summary("a" * 64)

@pytest.mark.asyncio
async def test_save_processed_article_persists_labeled_summary(repository: Repository, mock_pool: AsyncMock, sample_summary: LabeledSummary) -> None:
    """Save processed article persists URL, hash, summary, topic, and sentiment."""
    repository.pool = mock_pool
    canonical_url = "https://example.com/article"
    content_hash = "a" * 64
    await repository.save_processed_article(
        canonical_url=canonical_url,
        content_hash=content_hash,
        labeled_summary=sample_summary,
    )
    mock_pool.execute.assert_awaited_once()
    (sql, saved_url, saved_hash, summary, topic, sentiment) = mock_pool.execute.call_args.args
    assert "INSERT INTO processed_urls" in sql
    assert "ON CONFLICT (content_hash) DO UPDATE" in sql
    assert saved_url == canonical_url
    assert saved_hash == content_hash
    assert summary == sample_summary.summary
    assert topic == sample_summary.topic.value
    assert sentiment == sample_summary.sentiment.value

@pytest.mark.asyncio
async def test_save_processed_article_raises_runtime_error_on_database_failure(repository: Repository, mock_pool: AsyncMock, sample_summary: LabeledSummary) -> None:
    """Save processed article converts PostgreSQL errors to RuntimeError."""
    repository.pool = mock_pool
    mock_pool.execute.side_effect = asyncpg.PostgresError("database error")
    with pytest.raises(RuntimeError, match="Failed to save processed article"):
        await repository.save_processed_article(
            canonical_url="https://example.com/article",
            content_hash="a" * 64,
            labeled_summary=sample_summary,
        )

# Processed URL operations

@pytest.mark.asyncio
async def test_is_url_processed_returns_true(repository: Repository, mock_pool: AsyncMock) -> None:
    """Is URL processed returns True when the URL exists."""
    repository.pool = mock_pool
    mock_pool.fetchval.return_value = True
    result = await repository.is_url_processed("https://example.com/article")
    assert result is True
    mock_pool.fetchval.assert_awaited_once()
    sql, canonical_url = mock_pool.fetchval.call_args.args
    assert "SELECT EXISTS" in sql
    assert "FROM processed_urls" in sql
    assert "WHERE canonical_url = $1" in sql
    assert canonical_url == "https://example.com/article"

@pytest.mark.asyncio
async def test_is_url_processed_returns_false(repository: Repository, mock_pool: AsyncMock) -> None:
    """Is URL processed returns False when the URL does not exist."""
    repository.pool = mock_pool
    mock_pool.fetchval.return_value = False
    result = await repository.is_url_processed("https://example.com/article")
    assert result is False
    mock_pool.fetchval.assert_awaited_once()

@pytest.mark.asyncio
async def test_is_url_processed_raises_runtime_error_on_database_failure(repository: Repository, mock_pool: AsyncMock) -> None:
    """Is URL processed converts PostgreSQL errors to RuntimeError."""
    repository.pool = mock_pool
    mock_pool.fetchval.side_effect = asyncpg.PostgresError("database error")
    with pytest.raises(RuntimeError, match="Failed to check if URL"):
        await repository.is_url_processed("https://example.com/article")