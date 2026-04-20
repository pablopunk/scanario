from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import get_settings
from storage import (
    create_job,
    delete_job,
    get_result_files,
    get_result_path,
    get_results_dir,
    save_upload,
)
from worker import celery_app, process_scan, create_pdf

app = FastAPI(
    title="Scanario API",
    description="Document scanning API with job queue",
    version="1.0.0",
)

settings = get_settings()


class ScanResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/scan", response_model=ScanResponse)
async def scan_document(
    file: UploadFile = File(...),
    mode: str = settings.default_mode,
    backend: str = settings.default_backend,
    debug: bool = False,
):
    """Upload an image and start a scan job."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    job_id = create_job()
    data = await file.read()
    save_upload(job_id, data)
    
    # Queue the task
    task = process_scan.delay(job_id, mode, backend, debug)
    
    return ScanResponse(
        job_id=job_id,
        status="pending",
        message="Job queued for processing",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status and results of a job."""
    # Check if results exist (completed)
    result_files = get_result_files(job_id)
    
    if result_files:
        return JobStatus(
            job_id=job_id,
            status="completed",
            files=result_files,
        )
    
    # Check Celery task status
    # Note: We don't store task IDs, so we infer from file existence
    # For a production app, we'd store job state in Redis/DB
    from storage import get_upload_path
    upload_path = get_upload_path(job_id)
    if not upload_path.exists() and not result_files:
        raise HTTPException(404, "Job not found")
    
    return JobStatus(
        job_id=job_id,
        status="processing",
        files=[],
    )


@app.get("/images/{job_id}/{filename}")
async def get_image(job_id: str, filename: str):
    """Serve a result image."""
    path = get_result_path(job_id, filename)
    if path is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.delete("/jobs/{job_id}")
async def delete_job_endpoint(job_id: str):
    """Delete a job and all its data."""
    deleted = delete_job(job_id)
    if not deleted:
        raise HTTPException(404, "Job not found")
    return {"status": "deleted", "job_id": job_id}


@app.get("/jobs")
async def list_jobs():
    """List recent jobs (basic implementation)."""
    results_base = get_results_dir("").parent
    if not results_base.exists():
        return {"jobs": []}
    
    jobs = []
    for job_dir in results_base.iterdir():
        if job_dir.is_dir():
            files = get_result_files(job_dir.name)
            status = "completed" if files else "processing"
            jobs.append({
                "job_id": job_dir.name,
                "status": status,
                "files": files,
            })
    
    return {"jobs": sorted(jobs, key=lambda x: x["job_id"], reverse=True)[:50]}


@app.post("/pdf", response_model=PDFResponse)
async def create_pdf_endpoint(
    files: list[UploadFile] = File(default=[]),
    existing_job_ids: list[str] = [],
    page_order: list[str] = [],  # ["file:0", "job:abc123", "file:1"]
    mode: str = settings.default_mode,
    backend: str = settings.default_backend,
    debug: bool = False,
):
    """Create a PDF from multiple images.
    
    - files: New images to process and include
    - existing_job_ids: IDs of existing jobs to include outputs from
    - page_order: Optional ordering like ["file:0", "job:job_id", "file:1"]
    """
    from storage import save_upload, get_upload_path
    import shutil
    
    job_id = create_job()
    
    # Save uploaded files
    file_mapping = {}  # index -> saved path
    for i, file in enumerate(files):
        if file.content_type and file.content_type.startswith("image/"):
            data = await file.read()
            # Save to a temp location within job dir
            upload_path = get_upload_path(job_id).parent / f"upload_{i}.jpg"
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            upload_path.write_bytes(data)
            file_mapping[i] = str(upload_path)
    
    # Build page specs
    page_specs = []
    
    if page_order:
        # Use explicit ordering
        for spec in page_order:
            if spec.startswith("file:"):
                idx = int(spec.split(":")[1])
                if idx in file_mapping:
                    page_specs.append({"type": "file", "path": file_mapping[idx]})
            elif spec.startswith("job:"):
                job_id_ref = spec.split(":", 1)[1]
                page_specs.append({"type": "job_id", "value": job_id_ref})
    else:
        # Default order: all new files first, then existing jobs
        for idx, path in sorted(file_mapping.items()):
            page_specs.append({"type": "file", "path": path})
        for existing_id in existing_job_ids:
            page_specs.append({"type": "job_id", "value": existing_id})
    
    if not page_specs:
        # Cleanup and error
        from storage import delete_job
        delete_job(job_id)
        raise HTTPException(400, "No valid images or job IDs provided")
    
    # Queue PDF creation task
    task = create_pdf.delay(job_id, page_specs, mode, backend, debug)
    
    return PDFResponse(
        job_id=job_id,
        status="pending",
        message=f"PDF creation queued ({len(page_specs)} pages)",
        pages=len(page_specs),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
