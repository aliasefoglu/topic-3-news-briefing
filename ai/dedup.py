"""Deduplication primitives for the news pipeline.

Two-stage dedup
---------------
Stage 1 (cheap): `url_canonicalize` + `content_hash` catch exact duplicates
                  (same URL with different tracking params, same body
                  reposted via syndication).

Stage 2 (semantic): `near_duplicate` catches re-written stories that share
                     content but not exact text. We use word k-shingle Jaccard
                     similarity — pure stdlib, no external deps. Students can
                     swap in MinHash/LSH (via `datasketch`) if they want
                     sub-linear dedup at scale.

Threshold guidance
------------------
Jaccard ≥ 0.85 → almost certainly the same story.
Jaccard 0.5–0.85 → related, possibly same event, different article.
Jaccard < 0.5 → unrelated.

Document the threshold you pick in your report.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse


# Tracking parameters we strip during URL canonicalization. The list is
# representative, not exhaustive — students may extend it.
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "dclid", "msclkid", "yclid",
    "mc_cid", "mc_eid",
    "ref", "ref_src", "referrer",
    "_ga", "_gl",
    "icid", "ncid", "cid",
})


def url_canonicalize(url: str) -> str:
    """Normalize a URL so equivalent URLs hash to the same string.

    - lowercase scheme and host
    - drop fragment
    - drop common tracking query parameters
    - sort remaining query parameters
    - drop a single trailing slash on the path (but keep "/" if that's all)

    Returns
    -------
    str
        The canonical form. Raises `ValueError` on malformed input.
    """
    if not url or not url.strip():
        raise ValueError("url must be non-empty")

    parts = urlparse(url.strip())
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"url is missing scheme or host: {url!r}")

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()

    # path: drop trailing slash unless the path is just "/"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # query: drop trackers, sort the rest
    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    kept.sort()
    query = urlencode(kept)

    return urlunparse((scheme, netloc, path, "", query, ""))


def content_hash(text: str) -> str:
    """Stable SHA-256 of whitespace-normalized text.

    Whitespace normalization (collapse all runs of whitespace to a single space)
    means re-formatted reposts hash to the same digest.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 5) -> set[str]:
    """k-shingles of word tokens. Lowercased, punctuation-stripped."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < k:
        # Fall back to single tokens for very short text.
        return set(tokens)
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Standard Jaccard similarity over two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def near_duplicate(
    a: str,
    b: str,
    *,
    threshold: float = 0.7,
    k: int = 5,
) -> bool:
    """Return True if texts `a` and `b` look like near-duplicates.

    Uses word k-shingle Jaccard similarity. Cheap and dependency-free.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if k < 1:
        raise ValueError("k must be >= 1")
    return jaccard(_shingles(a, k), _shingles(b, k)) >= threshold
