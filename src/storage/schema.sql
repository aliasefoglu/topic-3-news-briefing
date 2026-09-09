-- Topic 3: AI News Briefing Service
-- PostgreSQL database schema

CREATE TABLE IF NOT EXISTS users (
    user_id            TEXT PRIMARY KEY,
    preferred_topics   JSONB NOT NULL DEFAULT '[]'::jsonb,
    excluded_sources   JSONB NOT NULL DEFAULT '[]'::jsonb,
    preferred_length   TEXT NOT NULL DEFAULT 'medium'
                        CHECK (preferred_length IN ('short', 'medium', 'long'))
);

CREATE TABLE IF NOT EXISTS accounts (
    user_id            TEXT PRIMARY KEY
                        REFERENCES users (user_id) ON DELETE CASCADE,
    password_hash      TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS processed_urls (
    content_hash       CHAR(64) PRIMARY KEY,
    canonical_url      TEXT NOT NULL UNIQUE,
    summary            TEXT,
    topic              TEXT CHECK (topic IN (
                            'Politics', 'Tech', 'Sports', 'Business', 'Health',
                            'Science', 'World', 'Entertainment', 'Other'
                        )),
    sentiment          TEXT CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    processed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_processed_urls_canonical_url
    ON processed_urls (canonical_url);
