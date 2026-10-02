"""Postgres storage. One row per contribution; no IP addresses or request metadata stored."""

from __future__ import annotations

import hmac
import json
from typing import Any
from uuid import UUID

import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS installs (
    install_id        UUID        PRIMARY KEY,
    deletion_key_hash TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS contributions (
    id             BIGSERIAL PRIMARY KEY,
    install_id     UUID        NOT NULL,
    schema_version INTEGER     NOT NULL,
    received_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload        JSONB       NOT NULL
);
CREATE INDEX IF NOT EXISTS contributions_install_id ON contributions (install_id);
"""


class Store:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def migrate(self) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(SCHEMA)

    def add(
        self,
        install_id: UUID,
        deletion_key_hash: str,
        schema_version: int,
        payload: dict[str, Any],
    ) -> bool:
        """Store a contribution. The first one registers the install's deletion-key hash; later
        ones must carry the same hash. Returns False (nothing stored) on a mismatch."""
        with psycopg.connect(self.dsn) as conn, conn.transaction():
            conn.execute(
                "INSERT INTO installs (install_id, deletion_key_hash) VALUES (%s, %s) "
                "ON CONFLICT (install_id) DO NOTHING",
                (install_id, deletion_key_hash),
            )
            row = conn.execute(
                "SELECT deletion_key_hash FROM installs WHERE install_id = %s FOR UPDATE",
                (install_id,),
            ).fetchone()
            if row is None or not hmac.compare_digest(row[0], deletion_key_hash):
                return False
            conn.execute(
                "INSERT INTO contributions (install_id, schema_version, payload) "
                "VALUES (%s, %s, %s)",
                (install_id, schema_version, json.dumps(payload)),
            )
            return True

    def deletion_key_hash(self, install_id: UUID) -> str | None:
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "SELECT deletion_key_hash FROM installs WHERE install_id = %s", (install_id,)
            ).fetchone()
            return row[0] if row else None

    def delete_install(self, install_id: UUID) -> int:
        """Remove every contribution and the install record. Returns contributions removed."""
        with psycopg.connect(self.dsn) as conn, conn.transaction():
            cur = conn.execute("DELETE FROM contributions WHERE install_id = %s", (install_id,))
            conn.execute("DELETE FROM installs WHERE install_id = %s", (install_id,))
            return cur.rowcount

    def count(self, install_id: UUID) -> int:
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "SELECT count(*) FROM contributions WHERE install_id = %s", (install_id,)
            ).fetchone()
            return int(row[0]) if row else 0
