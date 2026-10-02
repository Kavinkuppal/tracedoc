"""Phase 5 (server): validation, HTTPS-only, rate limits, deletion, no IPs, and end to end."""

from __future__ import annotations

import io
import json
import logging
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
import uvicorn
from fastapi.testclient import TestClient
from tracedoc_server.__main__ import LOG_CONFIG, server_config
from tracedoc_server.app import create_app
from tracedoc_server.db import Store
from tracedoc_server.ratelimit import RateLimiter
from typer.testing import CliRunner

from tracedoc import cli
from tracedoc.share import consent
from tracedoc.share.upload import upload_log_path

REPO = Path(__file__).resolve().parents[2]
CAPTURES = str(REPO / "tests" / "fixtures" / "traces" / "capture")
HTTPS = {"x-forwarded-proto": "https"}


@pytest.fixture
def store(dsn: str) -> Store:
    s = Store(dsn)
    s.migrate()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE contributions")
    return s


def _payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "install_id": str(uuid.uuid4()),
        "tool_version": "0.0.1",
        "runs": [
            {
                "task_type": "research",
                "framework": "langgraph",
                "framework_version": "1.2.12",
                "models_by_role": [{"role": "researcher", "models": ["claude-opus-5"]}],
                "graph": {"agent_count": 2, "tool_count": 1, "max_depth": 2},
                "status": "completed",
                "step_count": 5,
                "tokens_in": 100,
                "tokens_out": 20,
                "cost": 0.001,
                "duration_s": 1.5,
                "findings": [
                    {"check_id": "design.loop", "severity": "medium", "role": "researcher"}
                ],
            }
        ],
        "feedback": [{"check_id": "design.loop", "vote": "up"}],
    }
    data.update(overrides)
    return data


def test_accepts_and_stores_valid_contribution(store: Store) -> None:
    client = TestClient(create_app(store))
    body = _payload()
    r = client.post("/v1/contributions", json=body, headers=HTTPS)
    assert r.status_code == 201
    assert store.count(uuid.UUID(body["install_id"])) == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"prompt": "secret prompt text"},
        {"runs": [{**_payload()["runs"][0], "node_names": ["my_agent"]}]},
        {"install_id": "not-a-uuid"},
        {"schema_version": 99},
        {"feedback": [{"check_id": "x.invented", "vote": "up"}]},
    ],
)
def test_rejects_anything_outside_the_allow_list(store: Store, bad: dict[str, Any]) -> None:
    client = TestClient(create_app(store))
    r = client.post("/v1/contributions", json=_payload(**bad), headers=HTTPS)
    assert r.status_code == 422
    assert "secret prompt text" not in r.text and "my_agent" not in r.text  # no echo


def test_https_only(store: Store) -> None:
    client = TestClient(create_app(store))
    assert client.post("/v1/contributions", json=_payload()).status_code == 403
    assert client.post("/v1/contributions", json=_payload(), headers=HTTPS).status_code == 201


def test_rate_limited_per_install(store: Store) -> None:
    client = TestClient(create_app(store, limiter=RateLimiter(per_install=2)))
    body = _payload()
    codes = [
        client.post("/v1/contributions", json=body, headers=HTTPS).status_code for _ in range(3)
    ]
    assert codes == [201, 201, 429]
    other = client.post("/v1/contributions", json=_payload(), headers=HTTPS)
    assert other.status_code == 201  # limits are per install ID, not per address


def test_delete_removes_only_that_install(store: Store) -> None:
    client = TestClient(create_app(store))
    a, b = _payload(), _payload()
    for body in (a, a, b):
        client.post("/v1/contributions", json=body, headers=HTTPS)
    r = client.delete(f"/v1/installs/{a['install_id']}", headers=HTTPS)
    assert r.status_code == 200 and r.json()["rows"] == 2
    assert store.count(uuid.UUID(a["install_id"])) == 0
    assert store.count(uuid.UUID(b["install_id"])) == 1


def test_no_ip_address_stored(store: Store, dsn: str) -> None:
    client = TestClient(create_app(store))
    client.post(
        "/v1/contributions", json=_payload(), headers={**HTTPS, "x-forwarded-for": "203.0.113.7"}
    )
    with psycopg.connect(dsn) as conn:
        columns = [
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'contributions' ORDER BY ordinal_position"
            )
        ]
        rows = conn.execute("SELECT payload::text FROM contributions").fetchall()
    assert columns == ["id", "install_id", "schema_version", "received_at", "payload"]
    assert all("203.0.113.7" not in r[0] and "testclient" not in r[0] for r in rows)


# ---- a real server on localhost ---------------------------------------------------------------


class LiveServer:
    def __init__(self, store: Store) -> None:
        self.bodies: list[bytes] = []
        app = create_app(store, allow_http=True)

        @app.middleware("http")
        async def record(request, call_next):  # noqa: ANN001, ANN202 - test-only body capture
            if request.method == "POST":
                self.bodies.append(await request.body())
            return await call_next(request)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.server = uvicorn.Server(server_config(app, "127.0.0.1", self.port))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> LiveServer:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.05)
        assert self.server.started
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def live(store: Store) -> Iterator[LiveServer]:
    with LiveServer(store) as server:
        yield server


def test_server_logs_never_contain_client_addresses(store: Store) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    with LiveServer(store) as server:
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "tracedoc_server"):
            logging.getLogger(name).addHandler(handler)
        import urllib.request

        req = urllib.request.Request(
            f"{server.url}/v1/contributions",
            method="POST",
            data=json.dumps(_payload()).encode(),
            headers={"Content-Type": "application/json", "X-Forwarded-For": "203.0.113.7"},
        )
        assert urllib.request.urlopen(req).status == 201
    logs = stream.getvalue()
    assert "stored contribution" in logs  # the app does log...
    assert "127.0.0.1" not in logs and "203.0.113.7" not in logs  # ...but never addresses
    assert LOG_CONFIG["loggers"]["uvicorn.access"]["handlers"] == []


runner = CliRunner()


def test_end_to_end_opt_in_preview_revoke_delete(
    live: LiveServer, store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    config = tmp_path / "share.toml"
    config.write_text(f'[share]\nenabled = true\nendpoint = "{live.url}"\n')
    analyze = [
        "analyze",
        CAPTURES,
        "--no-judge",
        "--out",
        str(tmp_path / "r"),
        "--config",
        str(config),
    ]

    # 1. First report: terms shown, user agrees, the payload is uploaded and stored.
    result = runner.invoke(cli.app, analyze, input="y\n")
    assert result.exit_code == 0, result.output
    assert "Shared anonymized statistics" in result.output
    install_id = consent.load_state().install_id
    assert store.count(install_id) == 1

    # 2. Preview output matches what was sent, byte for byte.
    preview = runner.invoke(cli.app, ["share", "--preview"]).stdout_bytes
    assert live.bodies == [preview]

    # 3. Every upload is logged locally with its payload.
    log = [json.loads(line) for line in upload_log_path().read_text().splitlines()]
    assert log[-1]["action"] == "upload" and log[-1]["outcome"] == "HTTP 201"
    assert json.dumps(log[-1]["payload"], sort_keys=True) == json.dumps(
        json.loads(preview), sort_keys=True
    )

    # 4. Revoke: nothing more is sent.
    assert runner.invoke(cli.app, ["share", "--off"]).exit_code == 0
    assert runner.invoke(cli.app, analyze).exit_code == 0
    assert len(live.bodies) == 1 and store.count(install_id) == 1

    # 5. Opt back in, send again, then delete everything tied to the install ID.
    assert (
        runner.invoke(cli.app, ["share", "--on", "--config", str(config)], input="y\n").exit_code
        == 0
    )
    assert runner.invoke(cli.app, analyze).exit_code == 0
    assert store.count(install_id) == 2
    result = runner.invoke(cli.app, ["share", "--delete", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert store.count(install_id) == 0
    new_state = consent.load_state()
    assert new_state.install_id != install_id and new_state.choice == "no"
