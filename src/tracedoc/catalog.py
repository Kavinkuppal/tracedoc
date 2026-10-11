"""The reviewed catalog of report text (catalog.toml), with strict template filling."""

from __future__ import annotations

import string
import sys
from functools import cache
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

CATALOG_FILE = Path(__file__).parent / "catalog.toml"


class CatalogError(KeyError):
    """A missing catalog entry or template placeholder: a bug in tracedoc, never user input."""


@cache
def catalog() -> dict[str, Any]:
    with CATALOG_FILE.open("rb") as f:
        return tomllib.load(f)


def placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def fill(template: str, **values: Any) -> str:
    """Fill {placeholders}; every placeholder must be given (extra values are ignored)."""
    missing = placeholders(template) - set(values)
    if missing:
        raise CatalogError(f"missing values {sorted(missing)} for template {template!r}")
    return template.format(**{k: v for k, v in values.items() if k in placeholders(template)})


def label(key: str, **values: Any) -> str:
    try:
        text = catalog()["labels"][key]
    except KeyError as e:
        raise CatalogError(f"no label {key!r}") from e
    return fill(text, **values) if values or placeholders(text) else text


def section(name: str) -> dict[str, Any]:
    return catalog()[name]


def text(table: str, key: str, **values: Any) -> str:
    try:
        template = catalog()[table][key]
    except KeyError as e:
        raise CatalogError(f"no entry {table}.{key}") from e
    return fill(template, **values)


def check_entry(check_id: str) -> dict[str, Any]:
    entry = catalog()["checks"].get(check_id) or catalog()["signals"].get(check_id)
    if entry is None:
        raise CatalogError(f"check {check_id!r} is not in the catalog")
    return entry


def check_title(check_id: str) -> str:
    return check_entry(check_id)["title"]


def check_templates(check_id: str, variant: str | None = None) -> tuple[str, str]:
    """(what_happened, what_to_change) templates for a rule finding."""
    entry = check_entry(check_id)
    if variant is not None:
        entry = entry.get(variant)
        if not isinstance(entry, dict):
            raise CatalogError(f"check {check_id!r} has no variant {variant!r}")
    try:
        return entry["what_happened"], entry["what_to_change"]
    except KeyError as e:
        raise CatalogError(f"check {check_id!r} variant {variant!r} lacks templates") from e


def limit(slot: str) -> int:
    return int(catalog()["limits"][slot])


def units(key: str, n: int, **values: Any) -> str:
    one, many = catalog()["units"][f"{key}_one"], catalog()["units"][f"{key}_many"]
    return fill(one if n == 1 else many, n=f"{n:,}", **values)
