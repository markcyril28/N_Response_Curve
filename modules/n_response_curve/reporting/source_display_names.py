"""Human-readable names for the registered source datasets, for figure text only.

Figures outside the promoted release package print a reader-facing name instead
of the registered ``source_name`` key. The key stays the identifier everywhere
it does identifier work: filenames, directory names, dict keys, table columns,
``run_manifest.json``, ``CHECKSUMS.sha256``, comparisons, and error messages.
Only rendered text — titles, subtitles, legend entries, axis tick labels,
annotations — passes through :func:`display_source_name`.

The map is hard-coded rather than configured on purpose. A display label is not
a governed input: it authorizes nothing, is not hashed into any release
identity, and a recipe that lost the key from its TOML would silently fall back
to the raw key. An unknown key returns unchanged, so a source added later keeps
today's behaviour until it is named here.

This module is deliberately importable by the standalone reporting recipes and
is not on the release path.
"""

from __future__ import annotations

__all__ = ["SOURCE_DISPLAY_NAMES", "display_source_name"]


# Registered source_name -> the name printed on a figure.
SOURCE_DISPLAY_NAMES: dict[str, str] = {
    "core_trial_data": "Literature Extracted Datasets",
}


def display_source_name(source_name: str) -> str:
    """Return the reader-facing name for ``source_name``.

    Unmapped keys are returned unchanged, so this is safe to apply to every
    rendered source label without enumerating the sources that keep their key.
    """

    return SOURCE_DISPLAY_NAMES.get(source_name, source_name)
