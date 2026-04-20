"""API-key authentication backed by Redis.

Keys are stored as a Redis set plus one hash per key for metadata. This means
keys survive restarts and are shared across the api, worker, and beat
containers without any extra database.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import redis

from scanario.config import get_settings

KEY_PREFIX = "sk_"
KEY_BYTES = 24  # 48 hex chars → plenty of entropy
KEYS_SET = "scanario:api_keys"


def _meta_key(key: str) -> str:
    return f"scanario:api_key:{key}"


@dataclass
class ApiKeyInfo:
    prefix: str
    label: str
    created_at: str


def _client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_hex(KEY_BYTES)


def create_key(label: str = "") -> str:
    r = _client()
    key = generate_key()
    r.sadd(KEYS_SET, key)
    r.hset(
        _meta_key(key),
        mapping={
            "label": label or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return key


def verify_key(key: Optional[str]) -> bool:
    if not key:
        return False
    try:
        return bool(_client().sismember(KEYS_SET, key))
    except redis.RedisError:
        return False


def list_keys() -> list[ApiKeyInfo]:
    r = _client()
    keys = sorted(r.smembers(KEYS_SET))
    out: list[ApiKeyInfo] = []
    for k in keys:
        meta = r.hgetall(_meta_key(k)) or {}
        out.append(
            ApiKeyInfo(
                prefix=k[: len(KEY_PREFIX) + 8] + "…",
                label=meta.get("label", ""),
                created_at=meta.get("created_at", ""),
            )
        )
    return out


def revoke_by_prefix(prefix: str) -> int:
    """Revoke every key whose full value starts with the given prefix."""
    if not prefix:
        return 0
    r = _client()
    matches = [k for k in r.smembers(KEYS_SET) if k.startswith(prefix)]
    if not matches:
        return 0
    pipe = r.pipeline()
    for k in matches:
        pipe.srem(KEYS_SET, k)
        pipe.delete(_meta_key(k))
    pipe.execute()
    return len(matches)


def has_any_key() -> bool:
    try:
        return _client().scard(KEYS_SET) > 0
    except redis.RedisError:
        return False
