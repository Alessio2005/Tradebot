"""Write the frozen baselines that `test_bitequivalence.py` compares against.

    python -m tests.regression.regenerate_baselines

Run this ONLY after an intentional change of the algorithm behind a baseline,
and commit the new `.npy` files together with that change: a baseline that
moves without a reviewed diff is exactly what the regression tests exist to
catch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.regression._current import (
    GK_BASELINE,
    SB_BASELINE,
    current_gk_volatility,
    current_sequential_bootstrap,
)

BASELINES = Path(__file__).parent / "baselines"


def main() -> None:
    BASELINES.mkdir(exist_ok=True)
    for name, array in (
        (GK_BASELINE, current_gk_volatility()),
        (SB_BASELINE, current_sequential_bootstrap()),
    ):
        np.save(BASELINES / name, array)
        print(f"wrote {BASELINES / name} {array.shape} {array.dtype}")


if __name__ == "__main__":
    main()
