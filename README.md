# scanario

<p align="center">
  <img src="https://github.com/pablopunk/scanario/blob/main/src/assets/scanario.jpeg?raw=true" width="200px" />
</p>

<p align="center">
  <img src="https://github.com/pablopunk/scanario/blob/main/src/assets/scanario.gif?raw=true" width="90%" />
</p>

> Self-hosted doc scanner. Take pictures with your phone, drop them into scanario, and get a clean PDF.

## Features

- **Drop images, get a PDF** — Drag multiple photos into the browser and download a clean, scan-quality PDF
- **Works with messy photos** — Handles cluttered backgrounds, shadows, crumpled paper, even overlapping receipts
- **Your documents stay private** — Self-hosted, runs entirely on your machine, nothing sent to the cloud
- **Three ways to use it** — Web UI for quick scans, API for automation, CLI for scripting

## Quick Start

```bash
# Create an API key
docker compose exec api python -m scanario.main auth create
```

### Web

Open `http://localhost:8000` → drag images → download PDF/image.

### API

```bash
curl -X POST http://localhost:8000/scan \
  -H "X-API-Key: $KEY" \
  -F "file=@doc.jpg" \
  -F "mode=gray"

# Returns: {"job_id": "...", "status": "pending"}
# Poll /jobs/{job_id} until completed, then download from /images/{job_id}/{filename}
```

### CLI

```bash
python -m scanario.main scan photo.jpg
python -m scanario.main pdf page1.jpg page2.jpg -o output.pdf
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


