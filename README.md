# scanario

<p align="center">
  <img src="https://github.com/pablopunk/scanario/blob/main/src/assets/scanario.jpeg?raw=true" width="200px" />
</p>

**Turn phone photos into scanner-quality documents.**

Self-hosted document scanner that handles real-world mess — cluttered backgrounds, perspective distortion, shadows, even overlapping receipts. AI segments, math corrects, classical CV enhances. No generative hallucinations.

## Features

- **Drop images, get a PDF** — Drag multiple photos into the browser and download a clean, scan-quality PDF
- **Works with messy photos** — Handles cluttered backgrounds, shadows, crumpled paper, even overlapping receipts
- **Your documents stay private** — Self-hosted, runs entirely on your machine, nothing sent to the cloud
- **Three ways to use it** — Web UI for quick scans, API for automation, CLI for scripting

## Quick Start

```bash
# CLI
python -m scanario.main scan photo.jpg
python -m scanario.main pdf page1.jpg page2.jpg -o output.pdf

# API
curl -X POST http://localhost:8000/scan -H "X-API-Key: $KEY" -F "file=@doc.jpg"

# Web UI
open http://localhost:8000
```

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
