"""Panel-opbouw uit de PIT-store — Phase 1.

De enige toegestane weg van de PIT-store naar een onderzoekspanel. Elke join
loopt via `utils.time.asof_join` met `direction="backward"` en een expliciete
tolerance; er bestaat hier geen andere merge.

Alles wat deze module teruggeeft draagt een `data_hash` van de reeksen waaruit
het is opgebouwd, zodat een resultaat later in de ledger te citeren is.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..utils.failfast import DataContractError, require
from ..utils.hashing import dataframe_content_hash
from ..utils.time import asof_join
from .pit_store import PitStore

__all__ = ["PanelResult", "join_funding_to_ohlcv", "load_ohlcv", "load_close_panel"]


@dataclass(frozen=True)
class PanelResult:
    """Een panel plus de provenance van elke reeks waaruit het is opgebouwd."""

    frame: pd.DataFrame
    #: symbool -> data_hash van de onderliggende PIT-reeks
    source_hashes: dict[str, str]
    #: hash over het samengestelde panel zelf
    panel_hash: str


def load_ohlcv(
    store: PitStore,
    symbol: str,
    granularity: str,
    *,
    asset_class: str = "crypto",
) -> pd.DataFrame:
    """Laad één OHLCV-reeks met een UTC-aware DatetimeIndex op `event_ts_ns`."""
    df = store.load(asset_class, "ohlcv", symbol, granularity)
    df = df.copy()
    df.index = pd.to_datetime(df["event_ts_ns"], unit="ns", utc=True)
    df.index.name = "event_ts"
    return df


def load_close_panel(
    store: PitStore,
    symbols: list[str] | tuple[str, ...],
    granularity: str,
    *,
    asset_class: str = "crypto",
) -> PanelResult:
    """Wide close-panel: één kolom per symbool, UTC-index.

    Symbolen met een verschillende listing-datum krijgen `NaN` vóór hun eerste
    bar. Dat is GEEN gat om te vullen maar het feit dat het instrument nog niet
    bestond; het invullen ervan zou survivorship bias introduceren.
    """
    require(len(symbols) > 0, "Leeg symboluniversum.", DataContractError)
    cols: dict[str, pd.Series] = {}
    hashes: dict[str, str] = {}
    for sym in symbols:
        df = load_ohlcv(store, sym, granularity, asset_class=asset_class)
        hashes[sym] = dataframe_content_hash(df.reset_index(drop=True))
        cols[sym] = df["close"]
    panel = pd.DataFrame(cols).sort_index()
    require(panel.index.is_monotonic_increasing,
            "Panel-index is niet oplopend.", DataContractError)
    return PanelResult(frame=panel, source_hashes=hashes,
                       panel_hash=dataframe_content_hash(panel.reset_index()))


def join_funding_to_ohlcv(
    store: PitStore,
    symbol: str,
    granularity: str,
    *,
    funding_granularity: str = "8h",
    tolerance: pd.Timedelta,
    asset_class: str = "crypto",
) -> pd.DataFrame:
    """Koppel de laatst BEKENDE funding rate aan elke OHLCV-bar.

    De koppeling gebruikt `asof_ts_ns` van de funding-reeks, dat gelijk is aan
    het settlement-moment. Een rate die om 08:00 UTC settelt kan daardoor niet
    aan een bar van 00:00 UTC diezelfde dag worden gekoppeld — precies de
    lookahead die stap 6 beschrijft.

    `tolerance` is verplicht en komt uit `conf/data/`. Zonder bovengrens zou een
    funding rate uit 2021 nog aan een bar uit 2026 kunnen worden gekoppeld.
    """
    ohlcv = load_ohlcv(store, symbol, granularity, asset_class=asset_class)
    funding = store.load(asset_class, "funding", symbol, funding_granularity)

    right = pd.DataFrame({
        "asof_ts": pd.to_datetime(funding["asof_ts_ns"], unit="ns", utc=True),
        "funding_rate": funding["funding_rate"].astype(float),
        "funding_interval_hours": funding["funding_interval_hours"],
    }).sort_values("asof_ts", kind="stable")

    return asof_join(ohlcv, right, asof_col="asof_ts", tolerance=tolerance)
