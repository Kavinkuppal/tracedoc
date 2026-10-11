"""Thumbs up/down on findings, stored locally in the user data directory.

The HTML report cannot write to disk, so it shows a `tracedoc feedback <id> up` command for
each vote and can export all of a session's votes as a file for `tracedoc feedback --import`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from platformdirs import user_data_dir

from .__about__ import NAME

Vote = Literal["up", "down"]
FEEDBACK_FORMAT = "tracedoc-feedback"


def feedback_path() -> Path:
    return Path(user_data_dir(NAME, appauthor=False)) / "feedback.jsonl"


def record(
    finding_id: str, vote: Vote, check_id: str | None = None, path: Path | None = None
) -> None:
    path = path or feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "finding_id": finding_id,
        "check_id": check_id,
        "vote": vote,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Latest vote per finding."""
    path = path or feedback_path()
    votes: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                votes[row["finding_id"]] = row
    return votes


def import_file(source: Path, path: Path | None = None) -> int:
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("format") != FEEDBACK_FORMAT:
        raise ValueError(f"{source} is not a {FEEDBACK_FORMAT} export")
    count = 0
    for v in data.get("votes", []):
        if v.get("vote") in ("up", "down") and v.get("finding_id"):
            record(v["finding_id"], v["vote"], v.get("check_id"), path)
            count += 1
    return count


def check_for(finding_id: str, report_json: Path) -> str | None:
    """Look up a finding's check in a report.json. When the traces came from several systems,
    each system's report.json is in its own sub-folder, so those are searched too."""
    paths = (
        [report_json]
        if report_json.is_file()
        else sorted(report_json.parent.glob(f"*/{report_json.name}"))
    )
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for group in report.get("groups", []):
            for f in group.get("findings", []):
                if f.get("id") == finding_id:
                    return f.get("check_id")
    return None
