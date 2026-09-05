# Topic 3 — AI News Briefing Service

> **What you receive:** a working AI module (LLM summary+label, embeddings, two-stage dedup helpers), a sample user profile, sample HTML pages, an end-to-end demo, and smoke tests.
> **What you build:** the full software-engineering layer around it (concurrent ingestion, scheduling, storage, dedup wiring, Markdown output, retries, logging, validation, tests, Docker, README, report).

---

## The problem

A scheduled service that fetches articles from multiple news sources concurrently, deduplicates near-identical stories, categorizes them by topic, and produces a personalized daily Markdown digest for a user given their interests.

## What the AI does

1. **Summary + label** (`ai.summarize_and_label`) takes an `Article`, asks the chosen LLM for a 2–3 sentence summary plus a `topic` (from a fixed enum: Politics, Tech, Sports, Business, Health, Science, World, Entertainment, Other) plus a `sentiment` (`positive` / `neutral` / `negative`). Returns a `LabeledSummary`.
2. **Embedding** (`ai.embed`) returns a unit-normalized vector — useful as the semantic stage of dedup or for topic clustering.
3. **Dedup primitives** (`ai.url_canonicalize`, `ai.content_hash`, `ai.near_duplicate`):
   - `url_canonicalize(url)` strips tracking params (`utm_*`, `fbclid`, `gclid`, …), lowercases scheme/host, drops fragment, sorts query.
   - `content_hash(text)` is a whitespace-normalized SHA-256 — catches exact reposts.
   - `near_duplicate(a, b, threshold=0.7)` is word k-shingle Jaccard — catches re-written versions of the same story. Pure stdlib; students can swap in MinHash/LSH (`datasketch`) if they want sub-linear scaling.

The LLM is provider-agnostic (Anthropic / OpenAI / Gemini) selected via env vars:

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=...

EMBEDDING_PROVIDER=openai      # or gemini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=...
```

(Anthropic does not offer a first-party embedding endpoint — pair Claude with an OpenAI or Gemini embedder.)

## What you build (the SE layer)

| Component | Required | Notes |
|---|---|---|
| `config.py` | yes | Read env, expose typed settings (`pydantic-settings` recommended). |
| Concurrent ingestion | **yes** | At least **5 sources** fetched via `asyncio` + `aiohttp`/`httpx`. At least one must be direct-scraped HTML (use `beautifulsoup4` or `selectolax`). |
| Two-stage dedup wiring | yes | Cheap (URL + content hash) first, then `near_duplicate` for survivors. Document your threshold. |
| User profile | yes | Stored in PostgreSQL (via `psycopg` or `asyncpg`) or JSON: preferred topics + excluded sources. |
| Output | yes | Markdown digest written to `digests/YYYY-MM-DD-<user>.md`. |
| CLI | yes | `python -m newsbrief run-daily --user <name>` runs the full pipeline. |
| Scheduling | recommended | APScheduler or a Docker entrypoint loop. |
| Retries | yes | Exponential backoff on every `ai.*` call and every fetch. |
| Validation | yes | Reject malformed feed entries; clean input. |
| Logging | yes | `logging` module, env-driven level. |
| Tests | yes | ≥60% coverage, all offline (mock HTTP). |
| Dockerfile | yes | Builds and runs end-to-end. |
| README | yes | Setup, env, run, test, parallel-vs-sequential timings. |

## How to run what we shipped

```bash
# (1) Install the AI-layer dependencies:
pip install numpy pydantic

# Optional, only needed if you actually call the providers:
pip install anthropic openai google-genai

# (2) Try the offline demo (no API keys, no network):
python demo_ai.py --offline

# (3) Run the smoke tests (offline, no network):
pytest tests/test_ai_smoke.py -v
```

Sample output of `python demo_ai.py --offline`:

```
Parsed 2, kept 2 after dedup. Mode = offline.

# Daily digest for khagani
_Generated 2026-05-06T..._

## Tech
- **Acme unveils new AI processor** (Sample News)  [+]
  Acme corp announced a new processor today aimed at AI workloads...
  <https://example.com/news/article1>

## Science
- **Astronomers detect unusual signal from nearby star** (Sample News)  [=]
  ...
```

## The contract (do not break)

- **Do not** edit any file under `ai/`. If you find a bug, file an issue with the instructor.
- **Do not** delete or weaken `tests/test_ai_smoke.py`. These tests are run during grading; they must pass on your final repo.
- **Do not** call provider SDKs directly from your business logic. Always go through `ai.summarize_and_label`, `ai.embed`, and the dedup helpers.

## Recommended folder layout for your project

```
your-project/
├── ai/                        # COPIED FROM HERE, unchanged
├── src/
│   ├── config.py
│   ├── models.py              # YOUR pydantic models: User, Source, ...
│   ├── services/
│   │   ├── ai_service.py      # retries, caching, logging around ai.*
│   │   └── fetch_service.py   # async fetch + HTML scrape
│   ├── core/
│   │   ├── dedup.py           # wires the two stages from ai.dedup
│   │   └── digest_builder.py
│   ├── concurrency/
│   │   └── pipeline.py        # asyncio.gather over the source list
│   ├── storage/
│   │   └── repository.py      # PostgreSQL for users, processed-URLs cache
│   ├── cli.py                 # `run-daily` etc.
│   └── scheduler.py           # APScheduler entry point (optional)
├── tests/                     # YOUR tests + the provided smoke tests
├── data/                      # COPIED FROM HERE
├── digests/                   # generated output
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Sample data

`data/` contains:
- `rss_feeds.txt` — 5 well-known public RSS feeds (one URL per line)
- `html_samples/article1.html`, `article2.html` — small sample pages for the offline demo and as templates for your scraper tests
- `user_profile.json` — sample user with preferred topics and excluded sources

## Tips for the SE layer

- **Use `asyncio.gather` with a Semaphore.** A `Semaphore(8)` is enough for 5 RSS feeds and stays polite to publishers. Document the speedup vs. sequential in your report.
- **The cheap dedup pass eliminates 50–80% of duplicates** in real feeds (syndicated stories share URLs and bodies). Run it before paying for any LLM call.
- **Cache labeled summaries by `content_hash`** — re-running the digest on the same article should never re-call the LLM.
- **Keep the threshold for `near_duplicate` documented and configurable.** 0.7 is a good default for word-5-shingles. Tune empirically.
- **Be defensive in your HTML scraper.** Wrap each fetch in a try/except so one broken source doesn't kill the run. Log the failure, continue.

## Free-tier API options

| Provider | Free tier? | Notes |
|---|---|---|
| Anthropic Claude | Limited trial credit | Best summary quality. |
| OpenAI GPT-4o-mini | Pay-as-you-go (cheap) | Smallest cost per article. |
| Google Gemini | Generous free tier | Both LLM and embeddings. |

A daily digest of 30 articles is well under $0.10 on any of these.
