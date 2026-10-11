"""Many input folders, searched recursively; non-trace files skipped; duplicates counted once."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from tests.helpers import TRACES
from tracedoc import cli
from tracedoc.analyze import collect_traces

CAPTURE = TRACES / "capture"
FILES = sorted(CAPTURE.glob("*.jsonl"))


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    """Two folders: a/ holds 2 traces in nested sub-folders, b/ holds the rest plus noise."""
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "x" / "y").mkdir(parents=True)
    shutil.copy(FILES[0], a / "x" / FILES[0].name)
    shutil.copy(FILES[1], a / "x" / "y" / FILES[1].name)
    (b / "deep").mkdir(parents=True)
    for f in FILES[2:]:
        shutil.copy(f, b / "deep" / f.name)
    (b / "labels.jsonl").write_text('{"text": "not a trace", "label": "other"}\n')
    (b / ".venv").mkdir()
    shutil.copy(FILES[0], b / ".venv" / "hidden.jsonl")  # hidden folders are not searched
    return a, b


def test_many_folders_searched_recursively(tmp_path: Path) -> None:
    a, b = _tree(tmp_path)
    result = collect_traces([a, b])
    assert len(result.traces) == len(FILES)
    assert result.files_read == len(FILES)
    assert [p.name for p in result.skipped_files] == ["labels.jsonl"]
    assert result.duplicates == 0
    assert "skipped 1 file(s) that are not traces" in result.describe()


def test_duplicates_through_symlinks_are_counted_once(tmp_path: Path) -> None:
    a, b = _tree(tmp_path)
    links = tmp_path / "links"
    links.mkdir()
    for f in FILES:
        (links / f"copy_{f.name}").symlink_to(f)
    result = collect_traces([a, b, links])
    assert len(result.traces) == len(FILES)
    assert result.duplicates == len(FILES)
    assert len({t.id for t in result.traces}) == len(FILES)


def test_order_does_not_depend_on_argument_order(tmp_path: Path) -> None:
    a, b = _tree(tmp_path)
    assert sorted(t.id for t in collect_traces([a, b]).traces) == sorted(
        t.id for t in collect_traces([b, a]).traces
    )


def test_langsmith_exports_found_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "exports" / "2026"
    nested.mkdir(parents=True)
    for f in sorted((TRACES / "langsmith").glob("*.json"))[:3]:
        shutil.copy(f, nested / f.name)
    (tmp_path / "exports" / "settings.json").write_text(json.dumps({"theme": "dark"}))
    result = collect_traces([tmp_path / "exports"], source="langsmith")
    assert len(result.traces) == 3
    assert [p.name for p in result.skipped_files] == ["settings.json"]


def test_cli_accepts_several_paths(tmp_path: Path, monkeypatch) -> None:
    from tracedoc import config as config_module

    monkeypatch.setattr(config_module, "user_config_path", lambda: tmp_path / "none.toml")
    a, b = _tree(tmp_path)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli.app, ["analyze", str(a), str(b), "--no-judge", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert f"Read {len(FILES)} run(s) from {len(FILES)} file(s)" in result.output
    report = json.loads((out / "report.json").read_text())
    assert report["scorecard"]["runs"] == len(FILES)
