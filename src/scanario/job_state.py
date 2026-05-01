"""Track Celery task ids per job_id in Redis so the API can report real status."""

from __future__ import annotations

from typing import Optional

import redis

from scanario.config import get_settings


def _client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _key(job_id: str) -> str:
    return f"scanario:job:{job_id}:task_id"


def set_task_id(job_id: str, task_id: str, ttl_seconds: int = 60 * 60 * 24 * 30) -> None:
    _client().set(_key(job_id), task_id, ex=ttl_seconds)


def get_task_id(job_id: str) -> Optional[str]:
    try:
        return _client().get(_key(job_id))
    except redis.RedisError:
        return None


def delete_task_id(job_id: str) -> None:
    try:
        _client().delete(_key(job_id))
    except redis.RedisError:
        pass


def resolve_status(job_id: str, has_results: bool) -> dict:
    """Return a structured status dict for a job.

    States we expose: pending, processing, completed, failed, unknown.
    """
    from celery.result import AsyncResult  # local import to avoid api cold-start cost

    task_id = get_task_id(job_id)
    if not task_id:
        return {
            "status": "completed" if has_results else "unknown",
            "error": None,
        }

    result = AsyncResult(task_id)
    state = result.state  # PENDING, RECEIVED, STARTED, RETRY, SUCCESS, FAILURE, REVOKED

    if state in ("PENDING", "RECEIVED"):
        return {"status": "pending", "error": None}
    if state in ("STARTED", "RETRY"):
        return {"status": "processing", "error": None}
    if state == "SUCCESS":
        info = None
        try:
            info = result.info
        except Exception:
            pass
        if isinstance(info, dict) and info.get("status") == "failed":
            return {"status": "failed", "error": info.get("error") or "task failed"}
        return {"status": "completed", "error": None}
    if state in ("FAILURE", "REVOKED"):
        err = None
        try:
            info = result.info
            if isinstance(info, dict):
                err = info.get("error") or str(info)
            else:
                err = str(info) if info else state.lower()
        except Exception:
            err = state.lower()
        return {"status": "failed", "error": err}

    return {"status": "processing", "error": None}
