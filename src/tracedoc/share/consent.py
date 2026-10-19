"""Consent and the install ID, stored as share.json in the user config directory.

DRAFT TERMS: `TERMS` below is a plain-language summary, not legal text. It has not been
reviewed by a lawyer. Data sharing stays behind the `share.enabled` flag (off) until it is.
Changing the terms means bumping `TERMS_VERSION`, which makes tracedoc ask again.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..config import config_dir

TERMS_VERSION = 2
"""2: marked as a draft. Changing the terms means a new version, so tracedoc asks again."""
TERMS = """\
DRAFT: these terms are not in effect. Data sharing is not enabled in this version of tracedoc,
and the terms have not had a legal review.

tracedoc can share anonymized statistics about your analyses to help improve its checks.

What would be sent: the task type, framework and version, which public models were used per
role, the shape of the agent graph (counts and depth), each run's status, step count, tokens,
cost and duration, the IDs of the checks that fired with their severity and role, and your
thumbs up/down votes on findings. These are tied to a random install ID, not to you, plus a
one-way hash of a random deletion token that never leaves your machine.

What is never sent: prompts, inputs, outputs, tool arguments or results, code, node or tool
names, file paths, API keys, or anything that identifies you or your users.

You can see the exact payload with `tracedoc share --preview`, stop sharing with
`tracedoc share --off`, and delete everything already shared with `tracedoc share --delete`.
"""

Choice = Literal["yes", "no"]


def new_deletion_token() -> str:
    """256 random bits. Only its SHA-256 is ever sent with contributions; the token itself is
    sent only to authorize `share --delete`."""
    return secrets.token_urlsafe(32)


def deletion_key_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


@dataclass
class ShareState:
    install_id: uuid.UUID
    deletion_token: str
    choice: Choice | None = None
    terms_version: int | None = None
    decided_at: str | None = None
    votes_sent_until: str | None = None
    """Timestamp of the newest feedback vote already shared."""

    @property
    def deletion_key_hash(self) -> str:
        return deletion_key_hash(self.deletion_token)

    @property
    def sharing(self) -> bool:
        return self.choice == "yes" and self.terms_version == TERMS_VERSION

    @property
    def needs_asking(self) -> bool:
        return self.choice is None or self.terms_version != TERMS_VERSION


def state_path() -> Path:
    return config_dir() / "share.json"


def load_state(path: Path | None = None) -> ShareState:
    """Load the state, creating a random install ID (UUID4) and deletion token the first time."""
    path = path or state_path()
    if path.is_file():
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        state = ShareState(
            install_id=uuid.UUID(data["install_id"]),
            deletion_token=data.get("deletion_token") or new_deletion_token(),
            choice=data.get("choice"),
            terms_version=data.get("terms_version"),
            decided_at=data.get("decided_at"),
            votes_sent_until=data.get("votes_sent_until"),
        )
        if not data.get("deletion_token"):
            save_state(state, path)
        return state
    # Both random; derived from nothing identifying.
    state = ShareState(install_id=uuid.uuid4(), deletion_token=new_deletion_token())
    save_state(state, path)
    return state


def save_state(state: ShareState, path: Path | None = None) -> None:
    path = path or state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "install_id": str(state.install_id),
        "deletion_token": state.deletion_token,
        "choice": state.choice,
        "terms_version": state.terms_version,
        "decided_at": state.decided_at,
        "votes_sent_until": state.votes_sent_until,
    }
    # The deletion token is a credential: create the file owner-only (POSIX) from the start.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2) + "\n")
    with contextlib.suppress(OSError):
        path.chmod(0o600)  # tighten files created before this version


def record_choice(state: ShareState, choice: Choice, path: Path | None = None) -> ShareState:
    state.choice = choice
    state.terms_version = TERMS_VERSION
    state.decided_at = datetime.now(timezone.utc).isoformat()
    save_state(state, path)
    return state


def rotate_install_id(state: ShareState, path: Path | None = None) -> ShareState:
    """After a deletion, start a fresh, unlinked install ID and deletion token."""
    state.install_id = uuid.uuid4()
    state.deletion_token = new_deletion_token()
    state.votes_sent_until = None
    save_state(state, path)
    return state
