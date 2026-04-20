import os
import sys
from pathlib import Path

from celery import Celery
from celery.signals import worker_ready

from scanario.config import get_settings
from scanario.storage import get_results_dir, get_upload_path

settings = get_settings()

celery_app = Celery(
    "scanario",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    worker_prefetch_multiplier=1,
)


def run_scanario(input_path: Path, output_dir: Path, mode: str, backend: str, debug: bool = False):
    """Import and run scanario processing."""
    # Import here to avoid loading heavy deps on worker startup
    from scanario import main as scanario
    import cv2
    
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    debug_dir = output_dir / "debug" if debug else None
    if debug_dir:
        debug_dir.mkdir(exist_ok=True)
    
    corners = scanario.detect_document(img, debug_dir=debug_dir, backend=backend)
    if corners is None:
        raise ValueError("Could not detect document corners")
    
    warped = scanario.warp_document(img, corners)
    enhanced = scanario.enhance_scan(warped, mode=mode)
    
    # Save with step prefixes
    slug = input_path.stem
    cv2.imwrite(str(output_dir / f"01-corners-{slug}.jpg"), 
                scanario.draw_corners(img, corners))
    cv2.imwrite(str(output_dir / f"02-warped-{slug}.jpg"), warped)
    cv2.imwrite(str(output_dir / f"03-enhanced-{mode}-{slug}.jpg"), enhanced)
    
    result = {
        "corners": corners.tolist(),
        "mode": mode,
        "backend": backend,
        "debug": debug,
    }
    
    return result


@celery_app.task(bind=True, max_retries=2)
def process_scan(self, job_id: str, mode: str, backend: str, debug: bool = False):
    """Process a scan job."""
    try:
        input_path = get_upload_path(job_id)
        output_dir = get_results_dir(job_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_scanario(input_path, output_dir, mode, backend, debug)
        
        return {
            "status": "completed",
            "job_id": job_id,
            "result": result,
            "files": [f.name for f in output_dir.iterdir()],
        }
    except Exception as exc:
        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=10)
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(exc),
        }


@celery_app.task(bind=True, max_retries=1)
def create_pdf(self, job_id: str, page_specs: list, mode: str, backend: str, debug: bool = False):
    """Create a PDF from multiple images.
    
    page_specs: list of dicts with 'type' ('file' or 'job_id') and 'value'
    """
    from scanario.pdf_utils import create_pdf_from_images
    from scanario.storage import get_results_dir, get_upload_path
    import cv2
    from scanario import main as scanario
    
    try:
        output_dir = get_results_dir(job_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        page_images = []
        
        for i, spec in enumerate(page_specs):
            if spec['type'] == 'file':
                # New file - process it
                upload_path = get_upload_path(job_id) / f"page_{i}.jpg"
                upload_path.parent.mkdir(parents=True, exist_ok=True)
                # File data should be saved before calling this task
                # For now, assume it's at a known path
                temp_input = Path(spec['path'])
                
                img = cv2.imread(str(temp_input))
                if img is None:
                    continue
                
                debug_dir = output_dir / f"debug_page_{i}" if debug else None
                if debug_dir:
                    debug_dir.mkdir(exist_ok=True)
                
                corners = scanario.detect_document(img, debug_dir=debug_dir, backend=backend)
                if corners is None:
                    continue
                
                warped = scanario.warp_document(img, corners)
                enhanced = scanario.enhance_scan(warped, mode=mode)
                
                page_path = output_dir / f"page_{i:03d}.jpg"
                cv2.imwrite(str(page_path), enhanced)
                page_images.append(page_path)
                
            elif spec['type'] == 'job_id':
                # Existing job - find best output
                existing_dir = get_results_dir(spec['value'])
                if not existing_dir.exists():
                    continue
                
                # Prefer enhanced image
                enhanced = list(existing_dir.glob('03-enhanced-*.jpg'))
                if enhanced:
                    page_images.append(enhanced[0])
                else:
                    # Any jpg will do
                    images = sorted(existing_dir.glob('*.jpg'))
                    if images:
                        page_images.append(images[0])
        
        if not page_images:
            raise ValueError("No valid pages to include in PDF")
        
        # Create PDF
        pdf_path = output_dir / "output.pdf"
        create_pdf_from_images(page_images, pdf_path)
        
        return {
            "status": "completed",
            "job_id": job_id,
            "pdf": "output.pdf",
            "pages": len(page_images),
            "files": [f.name for f in output_dir.iterdir()],
        }
        
    except Exception as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=5)
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(exc),
        }


@celery_app.task
def cleanup_old_jobs_task():
    """Scheduled task to clean up old jobs."""
    from scanario.storage import cleanup_old_jobs
    count = cleanup_old_jobs()
    return {"deleted_jobs": count}


# Schedule cleanup task
celery_app.conf.beat_schedule = {
    "cleanup-old-jobs": {
        "task": "scanario.worker.cleanup_old_jobs_task",
        "schedule": settings.cleanup_interval_hours * 3600,  # seconds
    },
}


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Ensure data directories exist."""
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
