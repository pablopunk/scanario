# scanario

<p align="center">
  <img src="https://github.com/pablopunk/scanario/blob/main/src/assets/scanario.jpeg?raw=true" width="200px" />
</p>

**Turn phone photos into scanner-quality documents.**

Self-hosted document scanner that handles real-world mess — cluttered backgrounds, perspective distortion, shadows, even overlapping receipts. AI segments, math corrects, classical CV enhances. No generative hallucinations.

## Features

### 🎯 Smart Document Detection
- **Multi-backend isolation** — Runs both Gemini Nano Banana and rembg, scores results against actual image edges to pick the best mask
- **Receipt-over-document handling** — Intentionally designed to isolate the main sheet even when a small receipt overlaps it
- **True aspect ratio recovery** — Calculates the document's actual dimensions from perspective geometry, not naive width/height

### ✨ Faithful Enhancement
- **Gray mode** — Clean B&W scan with bleed-through suppression (near-white clamped to pure white)
- **Color mode** — Preserves original colors while removing shadows and background noise
- **No AI generation** — Uses classical CV (denoise, unsharp mask, LAB color space) so text/numbers/stamps never get hallucinated
- **Smart compression** — JPEG quality 85 strikes the balance: ~50% smaller files with visually identical quality

### 🖥️ Full-Featured Web UI
- **Drag & drop workflow** — Single or batch upload, full-screen preview carousel
- **Mode selector** — Tab-based Gray/Color selection with live preview
- **Results gallery** — Download individual PNGs or bulk "All as PDF"
- **Browser-scoped history** — Previous scans stored in localStorage (not server-side)
- **Dark/light/system theme** — Respects your preference

### ⚡ Production-Ready Architecture
- **Async job queue** — Celery + Redis handles multiple concurrent scans without blocking
- **API key auth** — Create and revoke keys via CLI
- **Auto-cleanup** — Jobs and files automatically deleted after 7 days (configurable)
- **Docker-first** — Single `docker compose up` deployment

### 🔌 Multiple Interfaces
- **REST API** — Submit jobs, poll status, download results, create PDFs
- **Web UI** — Zero-build vanilla JS, works on mobile and desktop
- **CLI** — `scanario scan`, `scanario pdf`, `scanario auth` for scripting

## How It Works

**The breakthrough:** Instead of asking AI to find 4 corners (imprecise), use AI to *isolate* the document, then use classical geometry to find corners.

```
┌─────────────────┐
│  Input Photo    │  Cluttered background, perspective distortion,
│  (phone camera) │  shadows, overlapping receipts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  1. Isolate     │  Multi-backend (Nano Banana + rembg) generates
│     Document    │  masks. Edge scoring picks best geometry.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Fit Quad    │  RANSAC on boundary lines → 4 intersections
│     (Corners)   │  = document corners with true aspect ratio
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Warp &      │  Perspective correction + flatten lighting +
│     Enhance     │  white clamp (removes bleed-through)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Clean Scan     │  Ready to download as PNG or PDF
└─────────────────┘
```

**Why this matters:**
- AI segmentation tolerates messy backgrounds
- Geometry math gives precise corners and true aspect ratio
- Classical enhancement preserves text fidelity (no LLM hallucinations)

## Screenshots

| Drop & Preview | Results & Download |
|:--:|:--:|
| *(placeholder)* | *(placeholder)* |

## Quick Start

### CLI

```bash
# Scan a single image
python -m scanario.main scan ~/photos/receipt.jpg --mode gray

# Create PDF from multiple scans
python -m scanario.main pdf page1.jpg page2.jpg -o output.pdf
```

### API

```bash
# Submit a scan job
curl -X POST http://localhost:8000/scan \
  -H "X-API-Key: your-key" \
  -F "file=@document.jpg" \
  -F "mode=gray"
# → {"job_id": "...", "status": "pending"}

# Check status
curl http://localhost:8000/jobs/{job_id} \
  -H "X-API-Key: your-key"
# → {"status": "completed", "files": [...]}

# Download result
curl "http://localhost:8000/images/{job_id}/03-enhanced-gray-input.jpg?api_key=your-key" \
  -o result.jpg
```

### Web UI

Open `http://localhost:8000`, enter your API key, drag images onto the dropzone, select mode, and scan.

## Deploy with Docker Compose

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: scanario-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data

  api:
    image: ghcr.io/pablopunk/scanario:latest
    container_name: scanario-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      SCANARIO_REDIS_URL: redis://redis:6379/0
      SCANARIO_DATA_DIR: /app/data
      SCANARIO_MAX_AGE_DAYS: 7
      SCANARIO_CLEANUP_INTERVAL_HOURS: 24
      GEMINI_API_KEY: ${GEMINI_API_KEY}
    volumes:
      - scanario_data:/app/data
    depends_on:
      - redis
    command: uvicorn scanario.api:app --host 0.0.0.0 --port 8000

  worker:
    image: ghcr.io/pablopunk/scanario:latest
    container_name: scanario-worker
    restart: unless-stopped
    environment:
      SCANARIO_REDIS_URL: redis://redis:6379/0
      SCANARIO_DATA_DIR: /app/data
      GEMINI_API_KEY: ${GEMINI_API_KEY}
    volumes:
      - scanario_data:/app/data
    depends_on:
      - redis
    command: celery -A scanario.worker.celery_app worker --loglevel=info --concurrency=2

  beat:
    image: ghcr.io/pablopunk/scanario:latest
    container_name: scanario-beat
    restart: unless-stopped
    environment:
      SCANARIO_REDIS_URL: redis://redis:6379/0
      SCANARIO_DATA_DIR: /app/data
    volumes:
      - scanario_data:/app/data
    depends_on:
      - redis
    command: celery -A scanario.worker.celery_app beat --loglevel=info

volumes:
  redis_data:
  scanario_data:
```

Set `GEMINI_API_KEY` in a `.env` file next to this `docker-compose.yml`, then:

```sh
docker compose up -d
```

API is at `http://localhost:8000`.

### Create an API key

All endpoints (except `/health`) require an API key. Create one from inside the `api` container:

```sh
docker compose exec api python -m scanario.main auth create
```

Send it on every request as either:

```
X-API-Key: <your-key>
Authorization: Bearer <your-key>
```
