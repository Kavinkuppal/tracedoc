"""Send contributions and deletion requests. Standard library HTTP only.

Every upload attempt is logged locally (uploads.jsonl in the user data directory) with the
exact payload, the endpoint, and the outcome.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from platformdirs import user_data_dir

from ..__about__ import NAME, __version__
from ..config import ShareConfig

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ShareError(RuntimeError):
    """Sharing could not happen (disabled, no endpoint, bad endpoint, server error)."""


def data_dir() -> Path:
    return Path(user_data_dir(NAME, appauthor=False))


def last_payload_path() -> Path:
    return data_dir() / "share-last-payload.json"


def upload_log_path() -> Path:
    return data_dir() / "uploads.jsonl"


def check_endpoint(config: ShareConfig) -> str:
    if not config.enabled:
        raise ShareError("data sharing is not enabled in this version")
    endpoint = config.endpoint.strip().rstrip("/")
    if not endpoint:
        raise ShareError("no data-sharing endpoint is configured")
    parsed = urlparse(endpoint)
    if parsed.scheme == "https" and parsed.hostname:
        return endpoint
    if parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS:
        return endpoint  # local testing only
    raise ShareError("the data-sharing endpoint must use https://")


def save_pending(payload: bytes) -> Path:
    path = last_payload_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _log(action: str, endpoint: str, payload: bytes | None, outcome: str) -> None:
    path = upload_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "endpoint": endpoint,
        "outcome": outcome,
        "payload": json.loads(payload) if payload else None,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _request(
    method: str, url: str, body: bytes | None, timeout: float, token: str | None = None
) -> int:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"{NAME}/{__version__}",  # no platform or host details
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as e:
        raise ShareError(f"server answered HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ShareError(f"could not reach the data-sharing server ({e})") from e


def upload(payload: bytes, config: ShareConfig) -> None:
    """POST exactly `payload` (the bytes `share --preview` shows)."""
    endpoint = check_endpoint(config)
    try:
        status = _request("POST", f"{endpoint}/v1/contributions", payload, config.timeout_s)
    except ShareError as e:
        _log("upload", endpoint, payload, f"failed: {e}")
        raise
    _log("upload", endpoint, payload, f"HTTP {status}")


def request_deletion(install_id: UUID, deletion_token: str, config: ShareConfig) -> None:
    """DELETE with the deletion token as the credential. The token is never logged."""
    endpoint = check_endpoint(config)
    url = f"{endpoint}/v1/installs/{install_id}"
    try:
        status = _request("DELETE", url, None, config.timeout_s, token=deletion_token)
    except ShareError as e:
        _log("delete", endpoint, None, f"failed: {e}")
        raise
    _log("delete", endpoint, None, f"HTTP {status}")
