"""g5.reference.M04 — tabular data cleaning lab source and quality runner.

This package lives under the fixture tree. It is not a platform API route.
"""

from .cleaning import (
    CleaningResult,
    assert_analysis_ready,
    clean_orders,
    load_raw,
    raw_vs_clean_comparison,
)

__all__ = [
    "CleaningResult",
    "assert_analysis_ready",
    "clean_orders",
    "load_raw",
    "raw_vs_clean_comparison",
]
