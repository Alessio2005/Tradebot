# src/tradebot/compliance/shadow_trader.py
"""Shadow trader — runs challenger model beside champion without placing orders.

Collects per-bar: challenger_signal vs. champion_signal, hypothetical fill
price (via PaperOMS), hypothetical PnL vs. realised champion PnL.

Output: artefacts/shadow/{challenger_version}_vs_{champion_version}.parquet
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ShadowRecord", "ShadowTrader"]


@dataclass
class ShadowRecord:
    """One bar's shadow observation."""

    ts: pd.Timestamp
    symbol: str
    champion_signal: float
    challenger_signal: float
    close_price: float
    hyp_fill_price: float
    hyp_pnl: float
    champion_pnl: float


class ShadowTrader:
    """Runs challenger model alongside champion — logs everything, places NO orders.

    Parameters
    ----------
    champion_version :
        Version string of the current champion model.
    challenger_version :
        Version string of the challenger model.
    output_root :
        Directory where shadow Parquet files are written.
    """

    def __init__(
        self,
        champion_version: str,
        challenger_version: str,
        output_root: Path | str = "artefacts/shadow",
    ) -> None:
        self.champion_version = champion_version
        self.challenger_version = challenger_version
        self._output_root = Path(output_root)
        self._records: List[ShadowRecord] = []
        self._shadow_position: float = 0.0  # hyp qty
        self._shadow_entry: float = 0.0

    # ------------------------------------------------------------------
    # Per-bar observation
    # ------------------------------------------------------------------

    def observe(
        self,
        ts: pd.Timestamp,
        symbol: str,
        champion_signal: float,
        challenger_signal: float,
        close_price: float,
        champion_pnl: float,
        next_open_price: float | None = None,
        spread_bps: float | None = None,
        slippage_bps: float = 5.0,
    ) -> ShadowRecord:
        """Record one bar's signals without placing any orders.

        Parameters
        ----------
        champion_signal :
            Signal in [-1, +1] from the champion model.
        challenger_signal :
            Signal in [-1, +1] from the challenger model.
        close_price :
            Bar close price (used as fallback fill reference).
        champion_pnl :
            Realised champion PnL this bar (from live OMS).
        next_open_price :
            CHIEF AUDIT 2026-05-23 (K4): Open van bar t+1.  Wanneer aangeleverd
            wordt de hypothetical fill geprijsd op het next-bar open i.p.v.
            same-bar close — dat matcht de live-executie waarbij signal aan
            einde van bar i bekend is en order op open van bar i+1 vult.
        spread_bps :
            Half-spread in basispunten (one-way).  Wanneer ``None`` valt de
            functie terug op ``slippage_bps`` (statisch, simplistisch).
        slippage_bps :
            Statische slippage-fallback in basispunten (default 5 bps).
        """
        # CHIEF AUDIT 2026-05-23 (K4): realistische fill-modelering.
        # De oude code gebruikte een hardcoded 5bps op de same-bar close →
        # same-bar fill (impossibility: signaal aan einde van bar = fill op
        # close van diezelfde bar). Live: signaal genereert order, fill volgt
        # op open van bar t+1 inclusief spread + slippage. Voor compliance-
        # grade modellering hoort hier full LOB-sim of bid/ask-shadow, maar
        # dat vereist data-stream-koppeling die buiten dit module valt.
        # TODO: full integration met tradebot.execution.slippage.compute_slippage
        # en tradebot.execution.spread.compute_dynamic_spread_arr zodra de
        # bar-DataFrame hier beschikbaar is.
        sign = 1 if challenger_signal > 0 else -1
        ref_price = float(next_open_price) if next_open_price is not None else float(close_price)
        if spread_bps is not None:
            cost_bps = float(spread_bps) + float(slippage_bps)
        else:
            cost_bps = float(slippage_bps)
            if next_open_price is None and spread_bps is None:
                # Eenmaal per ShadowTrader-instantie waarschuwen via debug-flag
                if not getattr(self, "_warned_simplistic_fill", False):
                    logger.warning(
                        "ShadowTrader.observe: using simplistic %.1f bps slippage on "
                        "same-bar close (no next_open / spread provided). For "
                        "compliance-grade reporting wire bar OHLCV + spread.",
                        slippage_bps,
                    )
                    self._warned_simplistic_fill = True
        hyp_fill = ref_price * (1.0 + 1e-4 * cost_bps * sign)

        # Hypothetical PnL: if challenger held a position
        hyp_pnl = 0.0
        if abs(self._shadow_position) > 1e-12 and self._shadow_entry > 0:
            hyp_pnl = self._shadow_position * (close_price - self._shadow_entry)
            self._shadow_entry = close_price  # mark-to-market reset

        # Update hypothetical position based on challenger signal
        if challenger_signal > 0.1:
            self._shadow_position = 1.0
            self._shadow_entry = hyp_fill
        elif challenger_signal < -0.1:
            self._shadow_position = -1.0
            self._shadow_entry = hyp_fill
        else:
            self._shadow_position = 0.0
            self._shadow_entry = 0.0

        rec = ShadowRecord(
            ts=ts,
            symbol=symbol,
            champion_signal=champion_signal,
            challenger_signal=challenger_signal,
            close_price=close_price,
            hyp_fill_price=hyp_fill,
            hyp_pnl=hyp_pnl,
            champion_pnl=champion_pnl,
        )
        self._records.append(rec)
        logger.debug(
            "Shadow[%s]: champion=%.3f challenger=%.3f hyp_pnl=%.2f",
            symbol, champion_signal, challenger_signal, hyp_pnl,
        )
        return rec

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, symbol: str) -> Path:
        """Persist shadow records for ``symbol`` as Parquet."""
        if not self._records:
            logger.warning("ShadowTrader: no records to save.")
            return Path()

        self._output_root.mkdir(parents=True, exist_ok=True)
        filename = f"{self.challenger_version}_vs_{self.champion_version}.parquet"
        out_path = self._output_root / symbol / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "ts": r.ts,
                "symbol": r.symbol,
                "champion_signal": r.champion_signal,
                "challenger_signal": r.challenger_signal,
                "close_price": r.close_price,
                "hyp_fill_price": r.hyp_fill_price,
                "hyp_pnl": r.hyp_pnl,
                "champion_pnl": r.champion_pnl,
            }
            for r in self._records
        ]
        df = pd.DataFrame(rows).set_index("ts")
        df.to_parquet(out_path)
        logger.info("ShadowTrader: saved %d records to %s.", len(df), out_path)
        return out_path

    def get_records(self) -> List[ShadowRecord]:
        return list(self._records)

    @property
    def n_observations(self) -> int:
        return len(self._records)
