from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scanario.auth import verify_key
from scanario.config import get_settings, validate_gemini_api_key
from scanario.job_state import delete_task_id, resolve_status, set_task_id
from scanario.storage import (
    create_job,
    delete_job,
    get_result_files,
    get_result_path,
    get_results_dir,
    get_upload_path,
    save_upload,
)
from scanario.worker import celery_app, create_pdf, process_scan

app = FastAPI(
    title="Scanario API",
    description="Document scanning API with job queue",
    version="1.0.0",
)

# Static files (UI)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text())
    return HTMLResponse(content="<h1>scanario API</h1><p>UI not found. API is at /docs</p>")

settings = get_settings()

# Validate GEMINI_API_KEY on startup (after settings loaded from .env)
validate_gemini_api_key()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

AUTH_HINT = (
    "Create one with:\n"
    "  docker compose exec api python -m scanario.main auth create\n"
    "Then send it as 'X-API-Key: <your-key>' or 'Authorization: Bearer <your-key>'."
)


def _extract_api_key(
    x_api_key: Optional[str],
    authorization: Optional[str],
) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    # Check headers first, then query parameter (for image URLs in browser)
    key = _extract_api_key(x_api_key, authorization)
    if not key:
        key = request.query_params.get("api_key")
    if not verify_key(key):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_or_invalid_api_key",
                "message": "This endpoint requires a valid API key.",
                "hint": AUTH_HINT,
            },
        )
    return key


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ScanResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed, unknown
    error: Optional[str] = None
    mode: Optional[str] = None
    backend: Optional[str] = None
    debug: bool = False
    result: Optional[dict] = None
    files: list[str] = []


class PDFResponse(BaseModel):
    job_id: str
    status: str
    message: str
    pages: int = 0


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Gated routes (all require_api_key)
# ---------------------------------------------------------------------------

@app.post("/scan", response_model=ScanResponse, dependencies=[Depends(require_api_key)])
async def scan_document(
    file: UploadFile = File(...),
    mode: str = Form(settings.default_mode),
    backend: str = Form(settings.default_backend),
    debug: bool = Form(False),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    job_id = create_job()
    data = await file.read()
    save_upload(job_id, data)

    task = process_scan.delay(job_id, mode, backend, debug)
    set_task_id(job_id, task.id)

    return ScanResponse(
        job_id=job_id,
        status="pending",
        message="Job queued for processing",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_api_key)])
async def get_job_status(job_id: str):
    result_files = get_result_files(job_id)
    upload_path = get_upload_path(job_id)
    has_results = bool(result_files)

    if not has_results and not upload_path.exists():
        raise HTTPException(404, "Job not found")

    info = resolve_status(job_id, has_results)

    return JobStatus(
        job_id=job_id,
        status=info["status"],
        error=info["error"],
        files=result_files,
    )


@app.get("/images/{job_id}/{filename}", dependencies=[Depends(require_api_key)])
async def get_image(job_id: str, filename: str):
    path = get_result_path(job_id, filename)
    if path is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def delete_job_endpoint(job_id: str):
    deleted = delete_job(job_id)
    if not deleted:
        raise HTTPException(404, "Job not found")
    delete_task_id(job_id)
    return {"status": "deleted", "job_id": job_id}


@app.get("/jobs", dependencies=[Depends(require_api_key)])
async def list_jobs():
    results_base = get_results_dir("").parent
    if not results_base.exists():
        return {"jobs": []}

    jobs = []
    for job_dir in results_base.iterdir():
        if job_dir.is_dir():
            files = get_result_files(job_dir.name)
            info = resolve_status(job_dir.name, bool(files))
            jobs.append(
                {
                    "job_id": job_dir.name,
                    "status": info["status"],
                    "error": info["error"],
                    "files": files,
                }
            )

    return {"jobs": sorted(jobs, key=lambda x: x["job_id"], reverse=True)[:50]}


@app.post("/pdf", response_model=PDFResponse, dependencies=[Depends(require_api_key)])
async def create_pdf_endpoint(
    files: list[UploadFile] = File(default=[]),
    existing_job_ids: list[str] = Form(default=[]),
    page_order: list[str] = Form(default=[]),
    mode: str = Form(settings.default_mode),
    backend: str = Form(settings.default_backend),
    debug: bool = Form(False),
):
    job_id = create_job()

    file_mapping: dict[int, str] = {}
    for i, file in enumerate(files):
        if file.content_type and file.content_type.startswith("image/"):
            data = await file.read()
            upload_path = get_upload_path(job_id).parent / f"upload_{i}.jpg"
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            upload_path.write_bytes(data)
            file_mapping[i] = str(upload_path)

    page_specs = []
    if page_order:
        for spec in page_order:
            if spec.startswith("file:"):
                idx = int(spec.split(":")[1])
                if idx in file_mapping:
                    page_specs.append({"type": "file", "path": file_mapping[idx]})
            elif spec.startswith("job:"):
                page_specs.append({"type": "job_id", "value": spec.split(":", 1)[1]})
    else:
        for idx, path in sorted(file_mapping.items()):
            page_specs.append({"type": "file", "path": path})
        for existing_id in existing_job_ids:
            page_specs.append({"type": "job_id", "value": existing_id})

    if not page_specs:
        delete_job(job_id)
        raise HTTPException(400, "No valid images or job IDs provided")

    task = create_pdf.delay(job_id, page_specs, mode, backend, debug)
    set_task_id(job_id, task.id)

    return PDFResponse(
        job_id=job_id,
        status="pending",
        message=f"PDF creation queued ({len(page_specs)} pages)",
        pages=len(page_specs),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
