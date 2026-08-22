"""Tick Bars — close bar every N trades (ticks).

Extension bar type per AFML §2.3.
Placeholder — implement analogous to imbalance.py using numba kernel.
"""
from __future__ import annotations

import pandas as pd


def generate_tick_bars(
    df: pd.DataFrame,
    threshold: float = 1_000.0,
) -> pd.DataFrame:
    """Generate tick bars. Not yet implemented."""
    raise NotImplementedError("tick_bars not yet implemented")
