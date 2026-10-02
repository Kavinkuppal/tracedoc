from __future__ import annotations

from pathlib import Path

import pytest

from tracedoc import config as config_module
from tracedoc.config import DEFAULT_REDACT_FIELDS, load_config


def test_defaults_without_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "user_config_path", lambda: tmp_path / "none.toml")
    cfg = load_config()
    assert cfg.redact.fields == DEFAULT_REDACT_FIELDS


def test_explicit_file_overrides_user_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = tmp_path / "user.toml"
    user.write_text('preview_chars = 100\n[redact]\nreplacement = "<x>"\n')
    explicit = tmp_path / "project.toml"
    explicit.write_text('preview_chars = 50\n[redact]\nfields = ["ssn"]\n')
    monkeypatch.setattr(config_module, "user_config_path", lambda: user)
    cfg = load_config(explicit)
    assert cfg.preview_chars == 50
    assert cfg.redact.fields == ["ssn"]
    assert cfg.redact.replacement == "<x>"
