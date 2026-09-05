"""Demo harness for the Topic 3 AI module.

Two modes:

  python demo_ai.py             # uses real LLM provider from env
  python demo_ai.py --offline   # uses fake LLM (no network, no API keys)

The demo:
  1. Reads the two sample HTML files from data/html_samples/
  2. Strips HTML to plain text
  3. Builds Article objects
  4. Summarizes + labels each via the LLM
  5. Runs the cheap dedup pass (URL + content hash)
  6. Prints a Markdown digest grouped by topic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ai import (
    Article, Digest, DigestItem, Topic, Sentiment,
    summarize_and_label, url_canonicalize, content_hash, near_duplicate,
)
from ai.providers.base import LLMProvider, ProviderError


# --- offline fakes --------------------------------------------------------

class _OfflineLLM(LLMProvider):
    """Pulls a plausible label from the article body via simple keyword matching."""

    def complete(self, prompt: str, *, json_schema=None, max_tokens: int = 1024) -> str:
        # The article body is the chunk between '---' markers in the prompt.
        body_match = re.search(r"---\s*\n(.*)\n---", prompt, re.DOTALL)
        body = body_match.group(1).lower() if body_match else prompt.lower()

        if any(w in body for w in ("processor", "chip", "ai workload", "software")):
            topic = "Tech"
        elif any(w in body for w in ("astronomer", "telescope", "physics", "biology", "researcher")):
            topic = "Science"
        elif any(w in body for w in ("election", "minister", "parliament", "treaty")):
            topic = "Politics"
        elif any(w in body for w in ("market", "stock", "earnings", "revenue")):
            topic = "Business"
        else:
            topic = "World"

        # Naive sentiment
        if any(w in body for w in ("welcomed", "announced", "improved", "won")):
            sentiment = "positive"
        elif any(w in body for w in ("crisis", "disaster", "warned", "fell")):
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Take the first 2 sentences as a stand-in summary.
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        summary = " ".join(sentences[:2]).strip().capitalize() or "(no summary)"

        return json.dumps({"summary": summary, "topic": topic, "sentiment": sentiment})


# --- HTML helpers ---------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Naive HTML-to-text. Real code should use beautifulsoup4 or selectolax."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0
        self._title: str | None = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip > 0:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in ("p", "br", "h1", "h2", "h3", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title and self._title is None:
            self._title = data.strip()
        else:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        # collapse whitespace per line
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _parse_html(path: Path, source: str) -> Article:
    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8"))
    # Synthesize a plausible https URL from the filename so url_canonicalize
    # works the same way it would for real ingested URLs.
    url = f"https://example.com/news/{path.stem}"
    return Article(
        title=parser._title or path.stem,
        url=url,
        source=source,
        content=parser.text(),
        published_at=None,
    )


# --- digest rendering ------------------------------------------------------

_SENTIMENT_GLYPH = {
    Sentiment.POSITIVE: "+",
    Sentiment.NEUTRAL: "=",
    Sentiment.NEGATIVE: "-",
}


def render_markdown(digest: Digest) -> str:
    """Render a Digest as a Markdown string."""
    out = [f"# Daily digest for {digest.user}",
           f"_Generated {digest.generated_at.isoformat(timespec='seconds')}_",
           ""]
    by_topic = digest.by_topic()
    for topic in Topic:
        items = by_topic.get(topic, [])
        if not items:
            continue
        out.append(f"## {topic.value}")
        out.append("")
        for it in items:
            glyph = _SENTIMENT_GLYPH[it.labeled.sentiment]
            out.append(f"- **{it.article.title}** ({it.article.source})  [{glyph}]")
            out.append(f"  {it.labeled.summary}")
            out.append(f"  <{it.article.url}>")
            out.append("")
    return "\n".join(out)


# --- main -----------------------------------------------------------------

def run_demo(offline: bool) -> None:
    here = Path(__file__).parent
    samples = sorted((here / "data" / "html_samples").glob("*.html"))
    if not samples:
        print("!! No sample HTML in data/html_samples/", file=sys.stderr)
        sys.exit(2)

    # 1. Parse all sample articles.
    articles: list[Article] = [_parse_html(p, source="Sample News") for p in samples]

    # 2. Cheap dedup pass.
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    deduped: list[Article] = []
    for art in articles:
        u = url_canonicalize(art.url)
        h = content_hash(art.content)
        if u in seen_urls or h in seen_hashes:
            continue
        seen_urls.add(u)
        seen_hashes.add(h)
        deduped.append(art)

    # 3. Optional semantic dedup against already-kept items.
    final: list[Article] = []
    for art in deduped:
        if any(near_duplicate(art.content, k.content) for k in final):
            continue
        final.append(art)

    print(f"Parsed {len(articles)}, kept {len(final)} after dedup. "
          f"Mode = {'offline' if offline else 'online'}.\n")

    # 4. Summarize + label each.
    llm = _OfflineLLM() if offline else None
    items: list[DigestItem] = []
    for art in final:
        try:
            labeled = summarize_and_label(art, llm=llm)
        except (ProviderError, ValueError) as e:
            print(f"  ! Skipping {art.title!r}: {e}", file=sys.stderr)
            continue
        items.append(DigestItem(article=art, labeled=labeled))

    # 5. Render and print.
    digest = Digest(
        user="khagani",
        generated_at=datetime.now(timezone.utc),
        items=items,
    )
    print(render_markdown(digest))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", action="store_true",
                   help="Use a fake LLM (no API keys, no network).")
    args = p.parse_args()
    run_demo(offline=args.offline)


if __name__ == "__main__":
    main()
