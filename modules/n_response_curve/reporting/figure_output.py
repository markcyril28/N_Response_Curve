"""Atomic figure writing shared by the standalone reporting recipes.

Every recipe that renders a directory of figures wrote the same nine lines: make
the parent, render to a hidden sibling under a fresh UUID, ``os.replace`` it
into place, and remove the temporary on any exit. The rename is what matters —
a reader that opens the destination sees either the previous complete JPEG or
the new complete one, never a half-written file, and an interrupted run leaves
the destination exactly as it found it.

The format and DPI are keyword arguments with the values every caller passed,
rather than a configured pair, because these recipes are outside the promoted
release package: their output contract is the bundle they write, not
``[outputs].figure_formats``. A recipe that needs a different format states it
at the call site, where the reader can see it.

The figure is deliberately **not** closed here. Callers own the figure's
lifetime, and several of them save one figure to more than one destination.

This module is importable by the standalone reporting recipes and is not on the
release path.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

__all__ = ["save_figure_atomically"]


def save_figure_atomically(
    figure: Any,
    destination: Path,
    *,
    image_format: str = "jpeg",
    dpi: int = 150,
) -> None:
    """Render *figure* to *destination* through a hidden sibling temporary.

    Missing parent directories are created. The temporary is removed whether or
    not the render succeeded, so a failure leaves no residue beside the
    destination.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        figure.savefig(temporary, format=image_format, dpi=dpi)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
