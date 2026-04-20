import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

from scanario.config import get_settings


def get_data_dir() -> Path:
    return Path(get_settings().data_dir)


def get_upload_path(job_id: str) -> Path:
    settings = get_settings()
    return get_data_dir() / settings.uploads_dir / job_id / "input.jpg"


def get_results_dir(job_id: str) -> Path:
    settings = get_settings()
    return get_data_dir() / settings.results_dir / job_id


def create_job() -> str:
    """Create a new job directory and return job_id."""
    job_id = str(uuid4())
    upload_path = get_upload_path(job_id)
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    return job_id


def save_upload(job_id: str, data: bytes) -> Path:
    """Save uploaded image for a job."""
    path = get_upload_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def get_result_files(job_id: str) -> list[str]:
    """List all result files for a job."""
    results_dir = get_results_dir(job_id)
    if not results_dir.exists():
        return []
    return sorted([f.name for f in results_dir.iterdir() if f.is_file()])


def get_result_path(job_id: str, filename: str) -> Optional[Path]:
    """Get path to a specific result file if it exists."""
    path = get_results_dir(job_id) / filename
    if path.exists():
        return path
    return None


def delete_job(job_id: str) -> bool:
    """Delete all data for a job. Returns True if anything was deleted."""
    settings = get_settings()
    data_dir = get_data_dir()
    
    deleted = False
    upload_dir = data_dir / settings.uploads_dir / job_id
    results_dir = data_dir / settings.results_dir / job_id
    
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
        deleted = True
    if results_dir.exists():
        shutil.rmtree(results_dir)
        deleted = True
    
    return deleted


def cleanup_old_jobs() -> int:
    """Delete jobs older than max_age_days. Returns count of deleted jobs."""
    settings = get_settings()
    max_age = timedelta(days=settings.max_age_days)
    cutoff = datetime.now() - max_age
    
    deleted_count = 0
    data_dir = get_data_dir()
    
    for subdir_name in [settings.uploads_dir, settings.results_dir]:
        subdir = data_dir / subdir_name
        if not subdir.exists():
            continue
        for job_dir in subdir.iterdir():
            if not job_dir.is_dir():
                continue
            # Check modification time of the directory
            mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
            if mtime < cutoff:
                shutil.rmtree(job_dir)
                deleted_count += 1
    
    return deleted_count
