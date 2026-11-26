"""Welke strategieën draaien, in welke volgorde (detailontwerp 7.3, 7.4).

De volgorde is de veldvolgorde van `StrategiesConfig` en ligt vast. Een uitgezette strategie
wordt niet gebouwd. De short straddle/strangle (undefined risk) laadt alleen als ook
`risk.allow_undefined_risk` aan staat; anders is het een configfout, geen stille weglating.
Hetzelfde voor gamma scalping zonder `hedging.enabled`: een ongehedgede straddle is geen gamma
scalp.
"""
from __future__ import annotations

from ...schemas.options import OptionsConfig
from ...utils.failfast import ConfigContractError
from .common import Strategy, rules_from_config
from .credit import CreditSpread, IronCondor
from .debit import CalendarSpread, DebitVertical, LongVol
from .income import CashSecuredPut, CoveredCall
from .short_vol import ShortVol
from .vol_trading import GammaScalp
from .wheel import Wheel

__all__ = ["build_strategies"]


def build_strategies(cfg: OptionsConfig) -> tuple[Strategy, ...]:
    s, rules, eligible = cfg.strategies, rules_from_config(cfg), cfg.universe.share_eligible
    if s.short_straddle_strangle.enabled and not cfg.risk.allow_undefined_risk:
        raise ConfigContractError("short_straddle_strangle staat aan, maar "
                                  "risk.allow_undefined_risk is false: undefined risk geweigerd")
    if s.gamma_scalp.enabled and not cfg.hedging.enabled:
        raise ConfigContractError("gamma_scalp staat aan, maar hedging.enabled is false: "
                                  "zonder delta-hedge geen gamma scalping")
    csp = CashSecuredPut(s.cash_secured_put, rules, eligible)
    cc = CoveredCall(s.covered_call, rules, eligible)
    built: list[tuple[bool, Strategy]] = [
        (s.credit_spread.enabled, CreditSpread(s.credit_spread, rules)),
        (s.iron_condor.enabled, IronCondor(s.iron_condor, rules)),
        (s.debit_vertical.enabled, DebitVertical(s.debit_vertical, rules)),
        (s.calendar_spread.enabled, CalendarSpread(s.calendar_spread, rules)),
        (s.long_straddle_strangle.enabled, LongVol(s.long_straddle_strangle, rules)),
        (s.gamma_scalp.enabled, GammaScalp(s.gamma_scalp, rules, eligible)),
        (s.cash_secured_put.enabled, csp),
        (s.covered_call.enabled, cc),
        (s.wheel.enabled, Wheel(csp, cc)),
        (s.short_straddle_strangle.enabled, ShortVol(s.short_straddle_strangle, rules)),
    ]
    return tuple(strategy for on, strategy in built if on)
