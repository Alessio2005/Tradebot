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

__all__ = ["daily_funding_panel"]

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
    bar_start = bar_end - np.timedelta64(_BAR_WIDTH)

    for symbol in symbols:
        df, _ = load_certified_series(
            store, register, asset_class=asset_class, dataset="funding",
            symbol=symbol, granularity=funding_granularity,
        )
        event_ns = df["event_ts_ns"].to_numpy(dtype="int64")
        rate = df["funding_rate"].to_numpy(dtype="float64")
        event_dt = event_ns.astype("datetime64[ns]")

        # `event_dt` is oplopend (PitStore.load sorteert op event_ts_ns), dus
        # een cumulatieve som plus binaire zoek levert de raamsom in O(log n)
        # per bar in plaats van een filter per bar.
        cum = np.concatenate(([0.0], np.cumsum(rate)))
        lo = np.searchsorted(event_dt, bar_start, side="left")
        hi = np.searchsorted(event_dt, bar_end, side="left")
        columns[symbol] = cum[hi] - cum[lo]

    panel = pd.DataFrame(columns, index=bar_index)[list(symbols)]
    panel.index.name = bar_index.name
    return panel.astype("float64")
