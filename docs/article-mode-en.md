# Article Mode (News → Video)

Article Mode turns news topics, RSS/Atom feeds or single article URLs into
short-form videos with as much or as little automation as you want. It is
**automation-first**: an LLM scores each story, writes a source-grounded script,
reviews and (if needed) rewrites it, picks licensed stock visuals, and renders
the video. Human review is **optional**.

> ⚠️ **What Article Mode is not.** It does **not** guarantee truth and is **not**
> professional fact-checking. Generated videos are labelled *"Generated from
> available sources."* Always treat the output as an AI-generated news summary
> based on the reporting available at generation time.

The existing topic-to-video workflow is unchanged. Article Mode only activates
when `content_mode` is `article_url` or `article_feed`; the default `topic`
behaves exactly as before.

---

## 1. How it works

```
poll feeds → extract & cluster articles → AI story score → threshold gate
→ generate grounded script → automated AI review (+auto-rewrite)
→ visual search & selection → render → (optional) publish
```

Each stage is deterministic only where it protects the software or your
infrastructure (URL/SSRF checks, file validation, DB integrity, secret
redaction). Editorial judgement — *is this story credible enough, what is the
most likely account, how certain is it* — is the LLM's job, expressed as
**confidence scores** rather than pass/fail proof gates.

### "Source grounded" — what it means

The script-generation prompt is given the extracted article text, source
metadata and known uncertainties, and is instructed to use them as the factual
foundation, to **not invent** names/dates/quantities/quotations, and to express
uncertainty naturally when sources disagree. An automated reviewer then checks
the script against the same sources. This makes the output *traceable to the
listed sources* — not provably true.

### Why absolute truth cannot be guaranteed

Sources can be wrong, incomplete or contradictory; developing stories change;
and language models can still misread. Article Mode reduces these risks with
scoring, corroboration signals and review, but no automated system can certify
truth. That is why publishing is conservative by default and every video carries
its source list.

---

## 2. Clean install and startup

Use the repository lockfile so API, WebUI, voice, provider and test
dependencies match the validated set:

```bash
uv sync --all-extras --dev --frozen
uv lock --check
```

Required Article Mode packages are `feedparser` and `trafilatura`. Supported
runtime setup also includes Uvicorn/FastAPI, Streamlit, Edge/Azure/Gemini or
other configured TTS and LLM dependencies, FFmpeg, and the selected licensed
media providers. Optional provider packages are installed by `--all-extras`.

Start the API:

```bash
python main.py
# or
uvicorn app.asgi:app --host 0.0.0.0 --port 8080
```

Start the WebUI:

```bash
streamlit run webui/Main.py
```

Start the worker:

```bash
python -m app.services.article_worker --once
python -m app.services.article_worker --interval 900
```

For staging, use a separate config and storage root:

```bash
IA2_CONFIG_FILE=/tmp/ia2-article-config.toml \
IA2_STORAGE_DIR=/tmp/ia2-article-storage \
python -m app.services.article_worker --once --automated
```

---

## 3. Automation modes

Set `article_automation_mode` in `config.toml` (or per subscription):

| Mode | Polls | Scores | Generates | Renders | Publishes |
|------|:----:|:-----:|:---------:|:-------:|:---------:|
| **assisted**   | ✅ | ✅ | ✅ | ⏸ needs approval | ⏸ |
| **automated**  | ✅ | ✅ | ✅ | ✅ | ⏸ needs approval |
| **autonomous** | ✅ | ✅ | ✅ | ✅ | ✅ (when enabled) |

Autonomous mode pauses only on repeated technical failure or very high
AI-assessed risk. **Auto-publishing is disabled by default** and must be
explicitly enabled with `article_auto_publish_enabled = true`.

---

## 4. Configuration

All keys live under `[app]` in `config.toml` (see `config.example.toml` for the
full annotated block). Highlights:

```toml
[app]
# Storage (shared by the worker and API)
article_database_path = ""                 # default: storage/article/articles.db

# Fetch safety (untrusted content)
article_request_timeout = 20
article_max_bytes = 5242880                # 5 MB per fetch
article_max_text_length = 40000

# Ingestion defaults
article_default_freshness_hours = 72
article_poll_interval_minutes = 60
article_trusted_domains = ["reuters.com", "apnews.com"]
article_blocked_domains = []

# Media
article_image_provider = "pexels"          # reuses pexels_api_keys / pixabay_api_keys
article_voice_name = "no-voice"           # set a real TTS voice for production worker renders
article_media_mode = "images_only"         # images_only | videos_only | mixed
article_add_illustrative_label = true

# Automation thresholds (0..1)
article_automation_mode = "assisted"
article_minimum_story_score = 0.6
article_minimum_confidence_score = 0.6
article_minimum_visual_score = 0.4
article_auto_generate_enabled = true
article_auto_render_enabled = false
article_auto_publish_enabled = false       # keep off unless you mean it
article_auto_rewrite_attempts = 1
article_allow_single_source_stories = true
article_maximum_risk_for_auto_publish = "low"
article_max_generations_per_day = 20
article_max_publications_per_day = 10

# Sensitive topics
article_require_review_for_sensitive_topics = true
article_sensitive_categories = ["politics","health","finance","legal","crime","war","emergency","disaster","death"]
```

### Trusted sources

`article_trusted_domains` (and per-subscription `trusted_domains`) raise a
source's quality signal and help identify authoritative primary sources
(e.g. an official `.gov` announcement). They are a *preference*, not a hard gate.

### Corroboration

Multiple independent domains raise confidence, but a single credible source
(official announcement, press release, government publication, verified
statement, recognized specialist outlet) can still be produced when
`article_allow_single_source_stories = true`. Several URLs carrying the *same
wire article* are counted as **one** source, not independent corroboration.

### Image licensing

Only licensed stock providers (Pexels, Pixabay) are used. Search-engine image
scraping is not supported. An article's own lead image is never auto-reused
unless its license and attribution are known. Every selected asset persists its
`license_name`, `license_url`, `attribution_text` and public `source_page_url`
in `media_manifest.json`. Signed download URLs and API keys are never written to
artifacts or logs.

### Sensitive topics

Categories in `article_sensitive_categories` are handled more conservatively:
the reviewer preserves uncertainty language, and by default such stories require
human approval before publishing (`article_require_review_for_sensitive_topics`).

---

## 5. WebUI workflow

Open Streamlit and use the **Article Mode** tab. The UI uses the Article Mode
service/API layer; it does not run an infinite worker loop inside Streamlit.

The UI supports subscription management, candidate and cluster review, script
and scene inspection, narration edits, automated review, visual previews and
replacement, images-only/videos-only/mixed mode selection, aspect-ratio
selection, rendering, video preview, download and publishing through the
existing workflow. Mutating actions use forms and stable widget keys so
Streamlit reruns do not create duplicate generation tasks.

The UI labels confidence scores as AI assessments, not factual guarantees, and
shows a prominent warning when automatic publishing is enabled.

---

## 6. RSS subscription examples

Create a subscription via the API:

```bash
curl -X POST http://127.0.0.1:8080/api/v1/article-subscriptions \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "AI headlines",
        "query": "artificial intelligence",
        "language": "en",
        "rss_urls": [
          "https://feeds.arstechnica.com/arstechnica/technology-lab",
          "https://www.theverge.com/rss/index.xml"
        ],
        "trusted_domains": ["arstechnica.com", "theverge.com"],
        "poll_interval_minutes": 60,
        "platform": "tiktok"
      }'
```

---

## 7. The polling worker

The worker runs **independently of Streamlit**.

```bash
# Loop forever on the configured interval
python -m app.services.article_worker

# One poll + process pass, then exit
python -m app.services.article_worker --once

# Force a mode for this run
python -m app.services.article_worker --once --automated
python -m app.services.article_worker --autonomous

# Poll a single subscription and exit
python -m app.services.article_worker --subscription sub-xxxxxxxx

# Custom loop interval (seconds)
python -m app.services.article_worker --interval 900
```

Use Ctrl-C or your service manager's normal stop signal for safe shutdown. The
loop records failures per job, logs `article worker interrupted; shutting down`,
and exits cleanly.

For unattended renders, set `article_voice_name` to a real voice supported by
your configured TTS provider. If it is empty and no WebUI voice preference is
set, Article Mode falls back to `no-voice`, which creates local timed silent
audio. That fallback is useful for offline validation but is not a narrated
production video.

### Docker Compose worker (optional)

Add a service that shares the same image, config and storage volume so it reads
the same `config.toml` and `articles.db`:

```yaml
services:
  article-worker:
    image: influencer-automation-2.0:latest
    command: ["python", "-m", "app.services.article_worker"]
    volumes:
      - ./config.toml:/influencer-automation-2.0/config.toml:ro
      - ./storage:/influencer-automation-2.0/storage
    restart: unless-stopped
```

SQLite WAL makes concurrent reads from the API and writes from the worker safe
on a shared volume.

---

## 8. API examples

| Method & path | Purpose |
|---|---|
| `POST /api/v1/article-subscriptions` | Create a subscription |
| `GET /api/v1/article-subscriptions` | List subscriptions |
| `GET /api/v1/article-subscriptions/{id}` | Get one |
| `PUT /api/v1/article-subscriptions/{id}` | Update |
| `DELETE /api/v1/article-subscriptions/{id}` | Delete |
| `POST /api/v1/article-subscriptions/{id}/poll` | Poll now |
| `GET /api/v1/articles` | List article candidates (filter by subscription/status/domain/cluster) |
| `GET /api/v1/articles/{id}` | Get one article |
| `POST /api/v1/articles/{id}/assess` | AI-score the story |
| `POST /api/v1/articles/{id}/generate` | Prepare a script (and optionally render) |

```bash
# Prepare a script for review (assisted)
curl -X POST http://127.0.0.1:8080/api/v1/articles/article-123/generate \
  -H 'Content-Type: application/json' \
  -d '{"media_mode": "images_only", "render": false}'

# Generate and render
curl -X POST http://127.0.0.1:8080/api/v1/articles/article-123/generate \
  -H 'Content-Type: application/json' \
  -d '{"media_mode": "mixed", "image_source": "pexels", "video_aspect": "9:16", "render": true}'
```

RSS URLs are SSRF-checked on create/update; internal/loopback/link-local
addresses and non-`http(s)` schemes are rejected.

---

## 9. CLI examples

Existing commands are unchanged (`--video-subject`, `--video-script`, …). New
flags:

```bash
# Generate a video directly from an article URL (images only)
python cli.py --article-url "https://example.com/news/story" \
              --media-mode images_only --image-source pexels --video-aspect 9:16

# Generate from a stored article candidate
python cli.py --approve-article article-123 --media-mode mixed

# Poll a subscription
python cli.py --poll-subscription sub-xxxxxxxx

# List stored article candidates
python cli.py --list-articles
```

---

## 10. Reproducible validation

Run the local Article Mode release-candidate smoke:

```bash
python scripts/validate_article_mode.py
```

The command creates temporary config, storage and SQLite paths under `/tmp`,
disables publishing, and uses local fixture providers through the real
production seams. It exercises direct URL ingestion, RSS worker polling,
duplicate-safe second polling, AI assessment and review/rewrite, no-voice TTS,
subtitle generation, images-only rendering, mixed-media rendering, playable MP4
validation, narration-duration coverage, scene-order preservation, provenance
artifacts, and secret/signed-URL scrubbing. It prints task IDs, artifact
directories, MP4 paths and durations, and exits non-zero on failure.

This is **local integration validation**, not live-provider validation. To test
fully live providers, configure real LLM, TTS, Pexels/Pixabay credentials and
run direct URL, subscription polling and rendering against public sources with
publishing disabled.

---

## 11. Review & approval workflow

* **assisted** — the worker/API prepare scored candidates and scripts; you
  review scripts, replace visuals and approve before rendering/publishing.
* **automated** — videos render automatically; you approve before publishing.
* **autonomous** — videos render and publish automatically (only when
  `article_auto_publish_enabled = true`), pausing on repeated failure or high
  risk.

A video **never auto-publishes** when the story is sensitive (unless you opt in)
or when the AI-assessed risk exceeds `article_maximum_risk_for_auto_publish`.

Every generated task stores provenance you can inspect afterwards:
`article.json`, `sources.json`, `script_plan.json`, `media_manifest.json`,
`assessment.json`, `review.json`, `provenance.json`.

---

## 12. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Feed poll returns 0 articles | Feed unreachable, blocked domain, or all items already stored (dedupe). Check the poll run's `errors`. |
| "invalid rss url" on create | The URL is non-http(s) or resolves to an internal address (SSRF protection). |
| Story always skipped | Scores below `article_minimum_story_score` / `article_minimum_confidence_score`; lower thresholds or check the LLM is configured. |
| No visuals / black backgrounds | Image provider keys missing or search returned nothing; the pipeline falls back to a branded background so the render still completes. |
| TTS fails in the worker | Set `article_voice_name` to a supported real voice and verify provider credentials/network access. `no-voice` is an offline validation fallback. |
| FFmpeg errors | Ensure FFmpeg is on `PATH` or set `IMAGEIO_FFMPEG_EXE`; run `python scripts/validate_article_mode.py` to verify local audio/video rendering. |
| Provider credential errors | Check `pexels_api_keys`, `pixabay_api_keys`, LLM keys and provider-specific TTS keys; never paste keys into task metadata or prompts. |
| Nothing publishes | Expected by default — enable `article_auto_publish_enabled` and use `autonomous` mode; sensitive/high-risk stories still require approval. |
| Malformed LLM JSON | The layer auto-repairs/retries; persistent failures mean the model/endpoint needs attention. |

### Database backup and reset

By default the SQLite database is `storage/article/articles.db` unless
`article_database_path` is set. Stop the worker before maintenance.

```bash
# backup
cp storage/article/articles.db storage/article/articles.db.bak

# reset staging Article Mode state
rm storage/article/articles.db storage/article/articles.db-wal storage/article/articles.db-shm
```

Do not reset the production database without a verified backup.
