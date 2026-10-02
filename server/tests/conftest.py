"""A throwaway Postgres for the server tests, and a network guard that allows only loopback.

Uses TRACEDOC_TEST_DB if set (CI provides a Postgres service). Otherwise starts a private
cluster in a temp directory with initdb/pg_ctl (found via PG_BIN, PATH, or Homebrew), listening
only on a Unix socket, and removes it afterwards.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_HOMEBREW = [Path(f"/opt/homebrew/opt/postgresql@{v}/bin") for v in (17, 16, 15, 14)]


def _pg_bin() -> Path | None:
    if os.environ.get("PG_BIN"):
        return Path(os.environ["PG_BIN"])
    found = shutil.which("initdb")
    if found:
        return Path(found).parent
    return next((p for p in _HOMEBREW if (p / "initdb").exists()), None)


@pytest.fixture(scope="session")
def dsn() -> Iterator[str]:
    if os.environ.get("TRACEDOC_TEST_DB"):
        yield os.environ["TRACEDOC_TEST_DB"]
        return
    pg = _pg_bin()
    if pg is None:
        pytest.skip("Postgres not available: set TRACEDOC_TEST_DB or PG_BIN")
    root = Path(tempfile.mkdtemp(prefix="tdpg"))  # short path: Unix socket paths are limited
    data, sock = root / "data", root
    subprocess.run(
        [pg / "initdb", "-D", data, "-A", "trust", "-U", "postgres", "--no-sync"],
        check=True,
        capture_output=True,
    )
    opts = f"-k {sock} -c listen_addresses='' -p 5499 -c fsync=off"
    subprocess.run(
        [pg / "pg_ctl", "-D", data, "-o", opts, "-w", "-l", root / "log", "start"],
        check=True,
        capture_output=True,
    )
    try:
        yield f"postgresql://postgres@/postgres?host={sock}&port=5499"
    finally:
        subprocess.run([pg / "pg_ctl", "-D", data, "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests may talk to the local test server and Postgres socket, nothing else."""
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and host not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError(f"network access to {host} is not allowed in tests")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(autouse=True)
def isolated_user_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tracedoc.config.user_config_dir", lambda *a, **k: str(tmp_path / "cfg"))
    monkeypatch.setattr("tracedoc.feedback.user_data_dir", lambda *a, **k: str(tmp_path / "data"))
    monkeypatch.setattr(
        "tracedoc.share.upload.user_data_dir", lambda *a, **k: str(tmp_path / "data")
    )
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
