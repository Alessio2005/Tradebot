# src/tradebot/data/funding_panel.py
"""De dagelijkse fundingboeking uit de gecertificeerde 8h-reeks.

Phase 10, stap 1B. `features/base.py::load_certified_series` levert de
gecertificeerde 8h-fundingreeks; deze module aggregeert haar tot de dagbar die
`backtest/engine.py` verwacht. Zij leeft in `data/` en niet in `features/`,
net als `data/phase6_universe.py`: het resultaat is geen feature (geen
beslissingsinput met een burn-in-contract) maar een BOEKINGSPANEEL dat de
engine één-op-één als per-bar rate consumeert.

WAAROM GEEN `asof_join`
========================
`features/positioning.py::build_certified_micro_frame` koppelt dezelfde
8h-reeks al aan de dagbar, met `utils.time.asof_join(direction="backward")`.
Dat levert de LAATST BEKENDE rate op het beslismoment op — het juiste antwoord
voor een FEATURE (`FundingRateMean`, `FundingRateZScore`), en het VERKEERDE
antwoord voor een BOEKING. De engine boekt de rate die zij krijgt precies EEN
keer per bar op de notional (`accounting.py::apply_funding`,
`qty * mark_price * rate`), terwijl een dagbar tot DRIE 8-uurs afrekeningen
draagt. Een asof-join zou een van de drie boeken en de andere twee weggooien —
ongeveer een derde van de werkelijke funding. Deze module sommeert in plaats
daarvan alle afrekeningen die binnen de bar zijn gevallen.

CAUSALITEIT (R-1)
==================
Dit is geen voorspelling maar een GEREALISEERDE kostenpost. Funding die
binnen bar t wordt afgerekend, wordt betaald door een positie die TIJDENS
bar t wordt gehouden; die positie is op t-1 besloten (en L2 zet er al een
latentiebar tussen). R-1 bindt features en toestanden — grootheden die het
besluit op t VOEDEN — op data tot en met t-1. Een gerealiseerde kostenpost die
op t wordt afgerekend en op t wordt geboekt, is geen lookahead: zij is de
uitkomst van het besluit, niet een input ervoor.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..features.base import DataRegister, load_certified_series

__all__ = ["daily_funding_panel", "settlement_sums"]

#: Breedte van één dagbar. Een structurele eigenschap van `granularity="1d"`
#: (event_ts_ns en asof_ts_ns van elke daily bar liggen exact 86400s uiteen),
#: geen beleidsdrempel — vandaar geen `conf/`-entry.
_BAR_WIDTH = pd.Timedelta(days=1)


def daily_funding_panel(
    store: Any,
    register: DataRegister,
    *,
    symbols: Sequence[str],
    asset_class: str,
    funding_granularity: str,
    bar_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Bouw het dagelijkse fundingpaneel op het bar-raster van het prijspaneel.

    Voor een dagbar met sluitmoment (`asof_ts`) T is de waarde de SOM van elke
    gecertificeerde 8h-afrekening met `event_ts` in `[T - 1 dag, T)` — de
    afrekeningen die binnen die bar zijn gevallen. Een bar zonder afrekening
    wordt `0.0` en geen `NaN`: er was geen boeking, wat een ander feit is dan
    "geen besluit" (waarvoor `NaN` elders in dit project is gereserveerd) —
    `backtest/engine.py::build_slices` leest `funding.at[ts, symbol]` en een
    `NaN` daar zou zich stil door de hele P&L voortplanten.

    Het paneel staat op EXACT `bar_index`, in dezelfde volgorde als `symbols`:
    `build_slices` indexeert positioneel noch tolerant.
    """
    columns: dict[str, np.ndarray] = {}
    bar_end = bar_index.values.astype("datetime64[ns]")

    for symbol in symbols:
        df, _ = load_certified_series(
            store, register, asset_class=asset_class, dataset="funding",
            symbol=symbol, granularity=funding_granularity,
        )
        columns[symbol] = settlement_sums(
            df["event_ts_ns"].to_numpy(dtype="int64"),
            df["funding_rate"].to_numpy(dtype="float64"),
            bar_end.astype("int64"),
        )

    panel = pd.DataFrame(columns, index=bar_index)[list(symbols)]
    panel.index.name = bar_index.name
    return panel.astype("float64")


def settlement_sums(
    event_ts_ns: np.ndarray, rate: np.ndarray, bar_end_ns: np.ndarray,
) -> np.ndarray:
    """De som van de afrekeningen met `event_ts` in `[T - 1 dag, T)`, per bar.

    De pure kern van `daily_funding_panel`, apart zodat de causaliteitstoets
    (`tests/lookahead/test_funding_panel_causality.py`, fase 11 stap 5.4) hem
    rechtstreeks kan truncaren en perturberen, naast twee lekkende varianten
    die rood moeten worden. Het venster is links gesloten en rechts OPEN: de
    afrekening precies op het sluitmoment `T` hoort bij de bar erna. Zo gebruikt
    de waarde op de bar die op `T` sluit uitsluitend afrekeningen van vóór `T`.

    `event_ts_ns` moet oplopend zijn (`PitStore.load` sorteert op
    `event_ts_ns`). Een cumulatieve som plus binaire zoek geeft de vensteromsom
    in O(log n) per bar in plaats van een filter per bar.
    """
    event = np.asarray(event_ts_ns, dtype="int64")
    ends = np.asarray(bar_end_ns, dtype="int64")
    starts = ends - np.int64(_BAR_WIDTH.value)
    cum = np.concatenate(([0.0], np.cumsum(np.asarray(rate, dtype="float64"))))
    lo = np.searchsorted(event, starts, side="left")
    hi = np.searchsorted(event, ends, side="left")
    return np.asarray(cum[hi] - cum[lo], dtype="float64")
