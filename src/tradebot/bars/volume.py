"""Volume Bars — close bar every N units of traded volume.

Extension bar type per AFML §2.3.
Placeholder — implement analogous to imbalance.py using numba kernel.
"""
from __future__ import annotations

import pandas as pd


def generate_volume_bars(
    df: pd.DataFrame,
    threshold: float = 1_000.0,
) -> pd.DataFrame:
    """Generate volume bars. Not yet implemented."""
    raise NotImplementedError("volume_bars not yet implemented")
