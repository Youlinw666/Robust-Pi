"""CPU-only analysis primitives for M6 result tables."""

from .metrics import (
    build_analysis_report,
    compare_paired,
    summarize_results,
    wilson_interval,
)

__all__ = [
    "build_analysis_report",
    "compare_paired",
    "summarize_results",
    "wilson_interval",
]
