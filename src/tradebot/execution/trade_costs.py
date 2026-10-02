"""De ene kostendefinitie van de wekelijkse strategie (spec §6.2).

    C_i = c_fix + c_fund,i + c_imp,i     (rendementseenheden van het notioneel)

De stopslippage zit in de fillprijs (`labeling/barrier_fills.py`), niet in C_i.
Drie toepassingen, één bron:

* `label_costs`   -- (a) achteraf, voor het trainingsdoel: gerealiseerde funding
  en impact bij het referentie-notioneel, zodat het label niet van de eigen
  sizing van het model afhangt.
* `ex_ante_cost`  -- (b) bij het besluit, alleen data <= t: een conservatieve,
  trade-specifieke bovengrens (slechtste slippage, maximale houdtijd, alleen
  betaalde funding, impact bij het maximale notioneel).
* `per_leg`, `impact_fraction` -- (c) de P&L in het tradeboek, per order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schemas.config import ExecutionConfig, ImpactConfig, load_config
from ..utils.failfast import DataContractError, require
from .impact_model import ImpactParams, ImpactStatus, square_root_impact

__all__ = ["TradeCostModel", "load_impact_params"]

BPS = 1e-4


def load_impact_params(path: Path) -> ImpactParams:
    """`conf/execution/impact.yaml` als `ImpactParams`, op één plek."""
    imp = load_config(path, ImpactConfig)
    return ImpactParams(
        eta=imp.eta, kappa_d=imp.kappa_d, status=ImpactStatus(imp.status), method=imp.method,
        data_hash=imp.data_hash, sample_size=imp.sample_size, period_start=imp.period_start,
        period_end=imp.period_end, instruments=imp.instruments, eta_ci_low=imp.eta_ci_low,
        eta_ci_high=imp.eta_ci_high)


@dataclass(frozen=True)
class TradeCostModel:
    taker_fee: float
    half_spread: float
    stop_slippage: float
    impact: ImpactParams | None

    @classmethod
    def from_config(cls, exec_cfg: ExecutionConfig, *, stop_slippage_bps: float,
                    impact: ImpactParams | None) -> TradeCostModel:
        return cls(taker_fee=exec_cfg.taker_fee_bps * BPS,
                   half_spread=exec_cfg.assumed_half_spread_bps * BPS,
                   stop_slippage=float(stop_slippage_bps) * BPS, impact=impact)

    @property
    def per_leg(self) -> float:
        """Taker fee plus halve spread: elk been, geen maker-aanname (spec §12)."""
        return self.taker_fee + self.half_spread

    @property
    def fixed_round_trip(self) -> float:
        return 2.0 * self.per_leg

    def impact_fraction(self, notional: float, *, adv: float, sigma_daily: float) -> float:
        """Impact van één order als fractie van zijn notioneel; 0 zonder impactparameters."""
        if self.impact is None or notional == 0.0:
            return 0.0
        require(bool(np.isfinite(adv) and adv > 0.0 and np.isfinite(sigma_daily) and sigma_daily > 0.0),
                "Impact zonder geldige ADV of sigma.", DataContractError,
                adv=adv, sigma_daily=sigma_daily)
        return float(square_root_impact(order_notional=abs(notional), adv_notional=adv,
                                        sigma_daily=sigma_daily, params=self.impact).impact_fraction)

    def label_costs(
        self, side, entry_bar, exit_bar, *, funding: np.ndarray, adv: np.ndarray,
        sigma_daily: np.ndarray, reference_notional: np.ndarray,
    ) -> np.ndarray:
        """C_i^label (spec §6.2a): vast + gerealiseerde funding op bars e+1..x + impact bij N_ref."""
        side = np.asarray(side, dtype=np.float64)
        e = np.asarray(entry_bar, dtype=np.int64)
        x = np.asarray(exit_bar, dtype=np.int64)
        ref = np.asarray(reference_notional, dtype=np.float64)
        f = np.asarray(funding, dtype=np.float64)
        adv = np.asarray(adv, dtype=np.float64)
        sig = np.asarray(sigma_daily, dtype=np.float64)
        require(side.size == e.size == x.size == ref.size, "Eén waarde per trade.",
                DataContractError)
        out = np.empty(side.size, dtype=np.float64)
        for i in range(side.size):
            window = f[e[i] + 1: x[i] + 1]
            require(bool(np.isfinite(window).all()),
                    "Funding ontbreekt binnen de houdtijd van een trade.", DataContractError,
                    entry_bar=int(e[i]), exit_bar=int(x[i]))
            fund = side[i] * float(window.sum())
            imp = (self.impact_fraction(ref[i], adv=adv[e[i]], sigma_daily=sig[e[i]])
                   + self.impact_fraction(ref[i], adv=adv[x[i]], sigma_daily=sig[x[i]]))
            out[i] = self.fixed_round_trip + fund + imp
        return out

    def ex_ante_cost(
        self, *, side: float, funding_recent_mean: float, horizon_bars: int,
        max_notional: float, adv: float, sigma_daily: float,
    ) -> float:
        """Ĉ_i (spec §6.2b): een conservatieve bovengrens met alleen data <= t."""
        require(bool(np.isfinite(funding_recent_mean)), "Ex-ante funding onbekend.",
                DataContractError)
        paid_funding = horizon_bars * max(0.0, float(side) * float(funding_recent_mean))
        return (self.fixed_round_trip + self.stop_slippage + paid_funding
                + 2.0 * self.impact_fraction(max_notional, adv=adv, sigma_daily=sigma_daily))
