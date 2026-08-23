"""Outlier-detectie — Phase 1, deliverable 7c.

Prijssprongen, nul-volume bars en negatieve spreads.

Deze module **verwijdert en winsoriseert niets**. Een uitschieter kan een echte
marktgebeurtenis zijn (een flash crash is data, geen fout) of datacorruptie (een
verkeerd geplaatste decimaal). Dat onderscheid is een onderzoeksbeslissing, geen
ingestion-beslissing. De validator meet en raiset; de behandeling hoort in de
config en in het data-register.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...utils.failfast import DataContractError, require

__all__ = ["OutlierReport", "detect_outliers", "enforce_outlier_policy"]

#: Hoeveel van de zwaarste sprongen in het rapport worden opgenomen. Dit is
#: PRESENTATIE, geen beleidsdrempel: het beinvloedt geen enkele beslissing en
#: hoort daarom niet in `conf/`.
_WORST_JUMPS_IN_REPORT = 5


@dataclass(frozen=True)
class OutlierReport:
    symbol: str
    granularity: str
    n_rows: int
    #: bars waarvan |log-return| de drempel overschrijdt
    n_price_jumps: int
    #: bars met volume == 0
    n_zero_volume: int
    #: bars met high < low
    n_inverted_range: int
    #: grootste absolute log-return in de reeks
    max_abs_log_return: float
    #: tijdstempels van de zwaarste sprongen
    worst_jumps: tuple[tuple[str, float], ...] = ()

    @property
    def total(self) -> int:
        return self.n_price_jumps + self.n_zero_volume + self.n_inverted_range


def detect_outliers(
    df: pd.DataFrame,
    *,
    symbol: str,
    granularity: str,
    max_abs_log_return: float,
    top_n: int = _WORST_JUMPS_IN_REPORT,
) -> OutlierReport:
    """Meet uitschieters. Muteert `df` niet.

    Parameters
    ----------
    max_abs_log_return : drempel op |ln(close_t / close_{t-1})|. Komt uit
        `conf/data/`; er is bewust geen default, want de juiste waarde verschilt
        per asset class en per granulariteit.
    """
    require("close" in df.columns,
            "detect_outliers vereist een close-kolom.",
            DataContractError, columns=list(df.columns)[:15])
    require(max_abs_log_return > 0,
            "max_abs_log_return moet positief zijn.",
            DataContractError, value=max_abs_log_return)

    close = df["close"].to_numpy(dtype=float)
    if close.size < 2:
        return OutlierReport(symbol, granularity, len(df), 0, 0, 0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(close[1:] / close[:-1])
    lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_lr = np.abs(lr)
    jump_mask = abs_lr > max_abs_log_return

    n_zero_vol = int((df["volume"] == 0).sum()) if "volume" in df.columns else 0
    n_inverted = (int((df["high"] < df["low"]).sum())
                  if {"high", "low"}.issubset(df.columns) else 0)

    worst: list[tuple[str, float]] = []
    if jump_mask.any():
        idx = np.argsort(-abs_lr)[:top_n]
        ev = df["event_ts_ns"].to_numpy()
        for i in idx:
            if abs_lr[i] > max_abs_log_return:
                ts = pd.Timestamp(int(ev[i + 1]), unit="ns", tz="UTC")
                worst.append((str(ts), float(lr[i])))

    return OutlierReport(
        symbol=symbol,
        granularity=granularity,
        n_rows=len(df),
        n_price_jumps=int(jump_mask.sum()),
        n_zero_volume=n_zero_vol,
        n_inverted_range=n_inverted,
        max_abs_log_return=float(abs_lr.max()),
        worst_jumps=tuple(worst),
    )


def enforce_outlier_policy(report: OutlierReport, *, allow_price_jumps: bool) -> None:
    """Raise wanneer de gemeten uitschieters het contract schenden.

    `n_inverted_range` en `n_zero_volume` zijn ALTIJD fataal: een bar met
    high < low is corruptie, en een bar met nul volume is geen verhandelbare
    observatie maar een placeholder — die stilzwijgend meenemen vertekent elke
    turnover- en kostenschatting.

    `n_price_jumps` is configureerbaar: op crypto is een dagelijkse beweging van
    30% een echte marktgebeurtenis, geen fout. `allow_price_jumps=True` betekent
    dat de sprongen zijn beoordeeld en als echt geaccepteerd — dat oordeel hoort
    in `docs/DATA_REGISTER.md` te staan.
    """
    require(report.n_inverted_range == 0,
            f"{report.n_inverted_range} bar(s) met high < low. Dat is geen "
            f"uitschieter maar corruptie, en het breekt elke range-gebaseerde "
            f"volatiliteitsschatter zonder dat die dat merkt.",
            DataContractError, symbol=report.symbol,
            granularity=report.granularity)
    require(report.n_zero_volume == 0,
            f"{report.n_zero_volume} bar(s) met volume 0. Een bar zonder handel "
            f"is geen verhandelbare observatie maar een placeholder; hem "
            f"meenemen vertekent turnover- en kostenschattingen.",
            DataContractError, symbol=report.symbol,
            granularity=report.granularity)
    if not allow_price_jumps:
        require(report.n_price_jumps == 0,
                f"{report.n_price_jumps} prijssprong(en) boven de drempel. "
                f"Zwaarste: {report.worst_jumps[:3]}. Beoordeel of dit echte "
                f"marktgebeurtenissen zijn; zo ja, leg dat vast in het "
                f"data-register en zet allow_price_jumps aan.",
                DataContractError, symbol=report.symbol,
                granularity=report.granularity,
                max_abs_log_return=round(report.max_abs_log_return, 4))
