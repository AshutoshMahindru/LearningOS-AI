"""g5.reference.M04 — tabular data cleaning lab source and quality runner.

This package lives under the fixture tree. It is not a platform API route.
Importing the package or ``g5.reference.M04.experiment`` must succeed without
pandas. Lab helpers in ``cleaning`` load pandas only when called.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CleaningResult",
    "assert_analysis_ready",
    "clean_orders",
    "load_raw",
    "raw_vs_clean_comparison",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import cleaning

        return getattr(cleaning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
