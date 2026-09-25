# tests/lookahead/test_funding_carry_causality.py
"""De positie van het carryboek op bar t gebruikt uitsluitend informatie <= t-1 (R-1).

Fase 11, stap 7.5. De hele keten wordt getoetst, van afrekening tot positie:
8h-afrekeningen -> `settlement_sums` (het dagpaneel) -> `known_carry` ->
`carry_weights` (rangschikking én vasthouden). Toets: verstoor elke afrekening
vanaf het sluitmoment van bar t-1, zo dat de rangschikking op die afrekeningen
om zou draaien. De positie op bar t mag niet veranderen.

De negatieve controle is hetzelfde boek met `lag_bars = 0`: de bekende carry
bevat dan de afrekeningen van de EIGEN bar. Die moet op dezelfde toets rood
worden. Zonder die controle bewijst de nul hieronder niets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.alpha.funding_carry_book import carry_weights, known_carry
from tradebot.data.funding_panel import settlement_sums

DAY_NS = np.int64(86_400 * 10**9)
HOUR_NS = np.int64(3_600 * 10**9)
SYMBOLS = ["AVAXUSDT", "BTCUSDT", "DOTUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT"]
N_DAYS = 60


def _ticks(seed: int = 3) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    base = np.int64(1_700_000_000) * 10**9 // DAY_NS * DAY_NS
    stamps = np.asarray([base + d * DAY_NS + h * HOUR_NS
                         for d in range(N_DAYS) for h in (0, 8, 16)], dtype="int64")
    return {s: (stamps, rng.normal(1e-4, 3e-4, stamps.size)) for s in SYMBOLS}


def _bar_ends(ticks) -> np.ndarray:
    first = next(iter(ticks.values()))[0][0] // DAY_NS * DAY_NS + DAY_NS
    return first + np.arange(N_DAYS - 1, dtype="int64") * DAY_NS


def _book(ticks, bar_end, *, lag_bars: int, holding_period: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(bar_end, unit="ns", utc=True))
    panel = pd.DataFrame({s: settlement_sums(e, r, bar_end)
                          for s, (e, r) in ticks.items()}, index=idx)
    return carry_weights(known_carry(panel, lag_bars=lag_bars),
                         holding_period=holding_period)


def _breaks(*, lag_bars: int, holding_period: int) -> int:
    """Het aantal bars waarop verstoring vanaf de close van t-1 de positie op t verandert."""
    ticks = _ticks()
    bar_end = _bar_ends(ticks)
    full = _book(ticks, bar_end, lag_bars=lag_bars, holding_period=holding_period)
    breaks = 0
    for k in range(3, len(bar_end)):
        cut = bar_end[k - 1]
        # Draai de rangschikking om op alles vanaf de close van bar k-1:
        # de huidige long-kant krijgt een enorme carry, de short-kant een negatieve.
        flip = {s: 1.0 if full.iloc[k][s] > 0 else -1.0 for s in SYMBOLS}
        disturbed = {s: (e, np.where(e >= cut, r + flip[s], r))
                     for s, (e, r) in ticks.items()}
        other = _book(disturbed, bar_end, lag_bars=lag_bars,
                      holding_period=holding_period)
        breaks += int(not other.iloc[k].equals(full.iloc[k]))
    return breaks


class TestThePositionIsCausal:
    def test_disturbing_everything_from_the_previous_close_changes_nothing(self) -> None:
        for h in (1, 3, 10):
            assert _breaks(lag_bars=1, holding_period=h) == 0

    def test_the_negative_control_reading_its_own_bar_goes_red(self) -> None:
        """`lag_bars = 0`: de bekende carry bevat de afrekeningen van bar t
        zelf. Op elke herbalanceringsbar draait de verstoring de positie om."""
        assert _breaks(lag_bars=0, holding_period=1) > 0
        assert _breaks(lag_bars=0, holding_period=3) > 0
