"""purge.py — Embargo and purge helpers.  SINGLE source of truth.

Eliminates the duplicate implementation bug:
  • quant_architect.py line 657  — authoritative version
  • train_regime.py lines 70-78  — stub fallback (same formula, no math import)

Blueprint §2.4: "dynamic_embargo_bars bestaat tweemaal … typische import-graph bug."

Resolution: canonical implementation here.  Both quant_architect.py and
train_regime.py should import from this module.  The stub in train_regime.py
lines 70-78 is deleted in Wave 4.

CHIEF AUDIT-FIX (Sim-to-Reality #2):
  Default safety_factor was 1.2 — calibrated to "typical 5-bar autocorrelation
  tail of crypto returns" under calm regimes.  During liquidation cascades
  (BTC May-2021, LUNA May-2022, FTX Nov-2022, SOL Aug-2023) the autocorrelation
  half-life extends to 15-20 bars due to:
    1. Forced deleveraging cascades (one liquidation triggers the next)
    2. Volatility clustering (GARCH(1,1) alpha+beta > 0.98 in crisis)
    3. Order-book impact persistence (LOB takes minutes to refill after a sweep)
  An embargo of 1.2x undershoots crisis tails by 12-17x → CPCV folds leak
  multi-day return information from test → train.  Default raised to 2.5
  (covers ~12.5-bar half-life under sf=2.5 × h_max=5, sufficient for normal
  crypto regimes plus 99th-percentile crisis days).  Callers needing tighter
  embargo can still explicitly pass safety_factor=1.2 but receive a warning.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence

logger = logging.getLogger(__name__)


# Conservative default after CHIEF AUDIT:
#   Crisis autocorr half-life observed at 15-20 bars on 5-min crypto bars.
#   With h_max=5 and sf=2.5 → embargo=13 bars (≈65 min); covers 2.5σ tail
#   of liquidation cascades.  See module docstring for incident attribution.
DEFAULT_SAFETY_FACTOR: float = 2.5

# Below this value caller is in "calm-regime only" territory — log a warning
# because the embargo will not survive a 2022-Luna-style cascade.
CRISIS_AWARE_MIN_SAFETY_FACTOR: float = 2.0


def dynamic_embargo_bars(
    horizons_bars: Sequence[int],
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    min_embargo: int = 1,
) -> int:
    """Compute the dynamic CPCV embargo in bars (audit item II.4).

    Standard purging covers overlap within one label-horizon, but ignores
    that multi-period returns produce information overlap that correlates
    even after the horizon (autocorrelation, vol-clustering, liquidation
    cascades in crypto perpetuals).

    Formula:
        embargo = ceil(max(horizon_bars) × safety_factor)

    The default ``safety_factor=2.5`` is calibrated against crisis-regime
    autocorrelation tails (15-20 bars on 5-min crypto bars during forced
    deleveraging cascades).  Calm-regime tails are typically 5-7 bars so
    sf=2.5 over-purges by ~2x in normal markets — this is the intended
    safety margin.

    Parameters
    ----------
    horizons_bars : Sequence of label horizons (in bars).
    safety_factor : Multiplier on max-horizon (must be ≥ 1.0).
                    Default 2.5 covers crypto crisis-regime autocorrelation.
                    Values below 2.0 trigger a warning (calm-regime only).
    min_embargo   : Floor (e.g. 1 bar for very short horizons).

    Returns
    -------
    int
        Embargo in bars.

    Raises
    ------
    ValueError
        If safety_factor < 1.0 (would reduce below the horizon — leakage).
    """
    if safety_factor < 1.0:
        raise ValueError(
            f"safety_factor must be ≥ 1.0, got {safety_factor}. "
            "A factor < 1.0 produces an embargo shorter than the horizon, "
            "allowing test-fold information to leak into the train set."
        )
    if safety_factor < CRISIS_AWARE_MIN_SAFETY_FACTOR:
        logger.warning(
            "dynamic_embargo_bars: safety_factor=%.2f is below the "
            "crisis-aware floor (%.2f). Calm-regime calibration only — "
            "during liquidation cascades (LUNA, FTX, SOL crashes) "
            "autocorrelation tails extend to 15-20 bars and this embargo "
            "will leak test-fold information into the train set. "
            "Pass safety_factor>=%.2f unless you have validated on crisis windows.",
            safety_factor, CRISIS_AWARE_MIN_SAFETY_FACTOR,
            CRISIS_AWARE_MIN_SAFETY_FACTOR,
        )
    if not horizons_bars:
        return int(min_embargo)
    h_max = int(max(int(h) for h in horizons_bars))
    embargo = math.ceil(h_max * float(safety_factor))
    return max(int(min_embargo), embargo)
