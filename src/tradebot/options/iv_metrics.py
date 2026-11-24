"""IV-maten voor de strategieregels: ATM-IV, IV rank, IV percentile, IV/RV en skew.

IV RANK HEEFT HISTORIE NODIG, EN DIE BESTAAT NOG NIET
-----------------------------------------------------
IVR en IVP worden berekend over de laatste `ivr_lookback_days` (252) handelsdagen van de
dagelijkse ATM-IV (30 dagen), die de dienst zelf om `end_of_day` wegschrijft (`IVHistory`).
Zolang er minder waarden zijn, is de status `INSUFFICIENT_HISTORY`.

Dan geldt de vervangende regel (open vraag B, standaard aan): de verhouding
`iv_rv_ratio = ATM-IV(30) / RV(20)`, met RV de Yang-Zhang-vol over 20 dagen. Elke drempel op
IVR heeft een vaste tegenhanger op die verhouding (`iv_metrics.fallback` in de config). Elke
beslissing met de vervangende regel wordt als zodanig gemarkeerd. Op de dag dat er genoeg
historie is, schakelt de strategie vanzelf naar IVR. Dat is een vaste regel, geen training.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import pandas as pd

from ..schemas.options import IVMetricsConfig
from ..utils.failfast import DataContractError, require
from ..volatility.yang_zhang import get_yang_zhang_volatility

__all__ = ["IVCondition", "IVHistory", "IVMetrics", "IVRStatus", "IVRuleSettings",
           "check_iv_condition", "iv_rank", "iv_rule_settings", "realized_vol"]


class IVRStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_HISTORY = "insufficient_history"
    NO_ATM_IV = "no_atm_iv"


class IVCondition(StrEnum):
    HIGH = "high"            # IVR ≥ 30
    LOW = "low"              # IVR ≤ 30
    VERY_LOW = "very_low"    # IVR ≤ 20


@dataclass(frozen=True, slots=True)
class IVRuleSettings:
    lookback_days: int = 252
    high_ivr: float = 30.0
    low_ivr: float = 30.0
    very_low_ivr: float = 20.0
    fallback_enabled: bool = True
    high_ratio: float = 1.20
    low_ratio: float = 1.00
    very_low_ratio: float = 0.90


def iv_rule_settings(cfg: IVMetricsConfig) -> IVRuleSettings:
    fb = cfg.fallback
    return IVRuleSettings(cfg.ivr_lookback_days, cfg.high_ivr, cfg.low_ivr, cfg.very_low_ivr,
                          fb.enabled, fb.high_iv_ratio, fb.low_iv_ratio, fb.very_low_iv_ratio)


class IVHistory:
    """De dagelijkse ATM-IV(30) per onderliggende: één waarde per dag, oplopend in datum."""

    def __init__(self, rows: Iterable[tuple[date, float]] = (), max_len: int = 400) -> None:
        self.max_len = max_len
        self._rows: dict[date, float] = {}
        for d, v in rows:
            self.record(d, v)

    def record(self, day: date, atm_iv: float) -> None:
        require(0 < atm_iv < 10, "ATM-IV buiten (0, 10)", DataContractError, atm_iv=atm_iv)
        self._rows[day] = float(atm_iv)
        for d in sorted(self._rows)[:-self.max_len]:
            del self._rows[d]

    def values(self, until: date, n: int) -> list[float]:
        """De laatste `n` waarden tot en met `until`."""
        return [self._rows[d] for d in sorted(self._rows) if d <= until][-n:]

    def rows(self) -> list[tuple[date, float]]:
        return sorted(self._rows.items())

    def __len__(self) -> int:
        return len(self._rows)


def iv_rank(history: Sequence[float], now: float) -> tuple[float, float]:
    """(IVR, IVP) van `now` ten opzichte van `history`, beide in [0, 100]."""
    lo, hi = min(history), max(history)
    ivr = 50.0 if hi == lo else min(max((now - lo) / (hi - lo) * 100.0, 0.0), 100.0)
    ivp = sum(v < now for v in history) / len(history) * 100.0
    return ivr, ivp


def realized_vol(daily_bars: pd.DataFrame, window: int = 20) -> float | None:
    """Geannualiseerde Yang-Zhang-vol over de laatste `window` dagen, of None te weinig data."""
    if len(daily_bars) < window + 1:
        return None
    daily = float(get_yang_zhang_volatility(daily_bars, window=window).iloc[-1])
    return daily * math.sqrt(252.0) if daily > 0 else None


@dataclass(frozen=True, slots=True)
class IVMetrics:
    atm_iv: float | None
    ivr: float | None
    ivp: float | None
    status: IVRStatus
    iv_rv_ratio: float | None
    term_slope: float | None = None
    skew_25d: float | None = None

    @classmethod
    def compute(cls, atm_iv: float | None, history: IVHistory, today: date, rv: float | None,
                cfg: IVRuleSettings, term_slope: float | None = None,
                skew_25d: float | None = None) -> IVMetrics:
        ratio = atm_iv / rv if atm_iv is not None and rv else None
        if atm_iv is None:
            return cls(None, None, None, IVRStatus.NO_ATM_IV, None, term_slope, skew_25d)
        hist = history.values(today, cfg.lookback_days)
        if len(hist) < cfg.lookback_days:
            return cls(atm_iv, None, None, IVRStatus.INSUFFICIENT_HISTORY, ratio, term_slope,
                       skew_25d)
        ivr, ivp = iv_rank(hist, atm_iv)
        return cls(atm_iv, ivr, ivp, IVRStatus.OK, ratio, term_slope, skew_25d)


def check_iv_condition(m: IVMetrics, cond: IVCondition,
                       cfg: IVRuleSettings) -> tuple[bool, bool]:
    """(voldaan, vervangende regel gebruikt). Zonder IVR en zonder terugval: niet voldaan."""
    if m.status is IVRStatus.OK and m.ivr is not None:
        if cond is IVCondition.HIGH:
            return m.ivr >= cfg.high_ivr, False
        if cond is IVCondition.LOW:
            return m.ivr <= cfg.low_ivr, False
        return m.ivr <= cfg.very_low_ivr, False
    if not cfg.fallback_enabled or m.iv_rv_ratio is None:
        return False, False
    if cond is IVCondition.HIGH:
        return m.iv_rv_ratio >= cfg.high_ratio, True
    if cond is IVCondition.LOW:
        return m.iv_rv_ratio <= cfg.low_ratio, True
    return m.iv_rv_ratio <= cfg.very_low_ratio, True
