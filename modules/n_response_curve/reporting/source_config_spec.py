"""Shared config/source-spec loading and file hashing for reporting recipes.

Several standalone reporting recipes each read one source's ``data_path`` and
``encoding`` out of the operator-facing TOML config and verify the resolved
file against a recorded SHA-256. This module holds that shared logic so the
recipes stay parameterized by their own ``SOURCE_NAME`` instead of carrying
independent copies of the same TOML-reading and hashing code.

This module is deliberately importable by the standalone reporting recipes and
is not on the release path.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

from n_response_curve.data.provenance import sha256_file as _sha256_file

__all__ = ["sha256_file", "load_source_spec"]


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*, read in fixed-size chunks."""

    return _sha256_file(path)


def load_source_spec(
    config_path: Path,
    source_name: str,
    *,
    relative_root: Path | None = None,
    resolve_path: bool = True,
    missing_sources_message: str | None = None,
    nonempty_requirement: str = "a nonempty string",
) -> tuple[Path, str]:
    """Resolve ``[sources.<source_name>].data_path`` and ``.encoding``.

    A relative ``data_path`` is resolved against *config_path*'s own directory
    unless a legacy recipe supplies its original ``relative_root``. The optional
    compatibility arguments let migrated callers retain their established error
    text and lexical path behavior while sharing the TOML parsing itself.
    """

    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    sources = config.get("sources")
    if not isinstance(sources, dict) and missing_sources_message is not None:
        raise ValueError(missing_sources_message)
    section = sources.get(source_name) if isinstance(sources, dict) else None
    if not isinstance(section, dict):
        raise ValueError(f"The configuration is missing [sources.{source_name}]")
    raw_path = section.get("data_path")
    encoding = section.get("encoding")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"[sources.{source_name}].data_path must be {nonempty_requirement}"
        )
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError(
            f"[sources.{source_name}].encoding must be {nonempty_requirement}"
        )
    source_path = Path(raw_path)
    if not source_path.is_absolute():
        root = (
            Path(relative_root)
            if relative_root is not None
            else Path(config_path).resolve().parent
        )
        source_path = root / source_path
    if resolve_path:
        source_path = source_path.resolve()
    return source_path, encoding
