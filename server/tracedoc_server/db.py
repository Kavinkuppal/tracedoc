"""Postgres storage. One row per contribution; no IP addresses or request metadata stored."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import psycopg

SCHEMA = """
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

    def add(self, install_id: UUID, schema_version: int, payload: dict[str, Any]) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(
                "INSERT INTO contributions (install_id, schema_version, payload) "
                "VALUES (%s, %s, %s)",
                (install_id, schema_version, json.dumps(payload)),
            )

    def delete_install(self, install_id: UUID) -> int:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            cur = conn.execute("DELETE FROM contributions WHERE install_id = %s", (install_id,))
            return cur.rowcount

    def count(self, install_id: UUID) -> int:
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "SELECT count(*) FROM contributions WHERE install_id = %s", (install_id,)
            ).fetchone()
            return int(row[0]) if row else 0
