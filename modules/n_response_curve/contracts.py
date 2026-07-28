from __future__ import annotations


SUPPORTED_FIGURE_FORMATS = frozenset({"jpeg", "png"})
SUPPORTED_TABLE_FORMATS = frozenset({"csv", "parquet", "xlsx"})


__all__ = ["SUPPORTED_FIGURE_FORMATS", "SUPPORTED_TABLE_FORMATS"]
