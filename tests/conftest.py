"""Shared test setup. Every test runs with the network blocked."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


class NetworkBlockedError(RuntimeError):
    pass


def _blocked(*args: Any, **kwargs: Any) -> None:
    raise NetworkBlockedError("Network access is not allowed in tests")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="Run tests that call a real model.")
    parser.addoption("--browser", action="store_true", help="Run report tests in real browsers.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: calls a real model API; needs --live and a key")
    config.addinivalue_line("markers", "browser: renders the report in browsers; needs --browser")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker in ("live", "browser"):
        if config.getoption(f"--{marker}"):
            continue
        skip = pytest.mark.skip(reason=f"needs --{marker}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_user_dirs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests never read or write the real user config/data directories."""
    base = tmp_path_factory.mktemp("userdirs")
    monkeypatch.setattr("tracedoc.config.user_config_dir", lambda *a, **k: str(base / "config"))
    monkeypatch.setattr("tracedoc.feedback.user_data_dir", lambda *a, **k: str(base / "data"))
    monkeypatch.setattr("tracedoc.share.upload.user_data_dir", lambda *a, **k: str(base / "data"))


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if "live" in request.keywords or "browser" in request.keywords:
        yield  # these talk to a model API or drive a local browser; they police their own traffic
        return
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(var, "false")
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield
