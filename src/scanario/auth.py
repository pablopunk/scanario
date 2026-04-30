"""File-backed API-key authentication.

Keys are stored in SCANARIO_DATA_DIR/auth-keys.json so they survive restarts and
are shared across api/worker/beat via the mounted data volume, without relying
on Redis for credential storage.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scanario.config import get_settings

KEY_PREFIX = "sk_"
KEY_BYTES = 24  # 48 hex chars
AUTH_FILE = "auth-keys.json"


@dataclass
class ApiKeyInfo:
    prefix: str
    label: str
    created_at: str


def _auth_path() -> Path:
    settings = get_settings()
    path = Path(settings.data_dir) / AUTH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load() -> dict:
    path = _auth_path()
    if not path.exists():
        return {"keys": []}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "keys" not in data or not isinstance(data["keys"], list):
            return {"keys": []}
        return data
    except Exception:
        return {"keys": []}


def _save(data: dict) -> None:
    path = _auth_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_hex(KEY_BYTES)


def create_key(label: str = "") -> str:
    data = _load()
    key = generate_key()
    data["keys"].append(
        {
            "key": key,
            "label": label or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save(data)
    return key


def verify_key(key: Optional[str]) -> bool:
    if not key:
        return False
    data = _load()
    return any(entry.get("key") == key for entry in data["keys"])



def list_keys() -> list[ApiKeyInfo]:
    data = _load()
    out: list[ApiKeyInfo] = []
    for entry in sorted(data["keys"], key=lambda x: x.get("created_at", "")):
        k = entry.get("key", "")
        out.append(
            ApiKeyInfo(
                prefix=k[: len(KEY_PREFIX) + 8] + "…" if k else "",
                label=entry.get("label", ""),
                created_at=entry.get("created_at", ""),
            )
        )
    return out


def revoke_by_prefix(prefix: str) -> int:
    if not prefix:
        return 0
    data = _load()
    before = len(data["keys"])
    data["keys"] = [entry for entry in data["keys"] if not entry.get("key", "").startswith(prefix)]
    removed = before - len(data["keys"])
    if removed:
        _save(data)
    return removed


def has_any_key() -> bool:
    return bool(_load()["keys"])


def main() -> None:
    """Small, fast CLI for managing API keys without importing scan pipeline deps."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Manage scanario API keys")
    subparsers = parser.add_subparsers(dest="action", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("--label", help="Optional label for this key")

    subparsers.add_parser("list", help="List stored API keys (prefixes only)")

    revoke_parser = subparsers.add_parser("revoke", help="Revoke keys by prefix")
    revoke_parser.add_argument("prefix", help="Full key or any prefix (e.g. 'sk_abc123')")

    args = parser.parse_args()

    if args.action == "create":
        try:
            key = create_key(label=args.label or "")
        except Exception as exc:
            print(f"Error: could not write auth file ({exc}).", file=sys.stderr)
            print("Make sure SCANARIO_DATA_DIR is writable.", file=sys.stderr)
            sys.exit(1)
        print("✅ New API key created. Save it now – it will NOT be shown again:\n")
        print(f"   {key}\n")
        print("Send it on every request as either:")
        print("  X-API-Key: <key>")
        print("  Authorization: Bearer <key>")
        return

    if args.action == "list":
        keys = list_keys()
        if not keys:
            print("No API keys yet. Create one with: python -m scanario.auth create")
            return
        print(f"{'PREFIX':<20}  {'CREATED':<30}  LABEL")
        for key_info in keys:
            print(f"{key_info.prefix:<20}  {key_info.created_at:<30}  {key_info.label}")
        return

    if args.action == "revoke":
        n = revoke_by_prefix(args.prefix)
        print(f"Revoked {n} key(s).")
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
