"""Welke strategieën draaien, in welke volgorde (detailontwerp 7.3, 7.4).

De volgorde is de veldvolgorde van `StrategiesConfig` en ligt vast. Een uitgezette strategie
wordt niet gebouwd. De short straddle/strangle (undefined risk) laadt alleen als ook
`risk.allow_undefined_risk` aan staat; anders is het een configfout, geen stille weglating.
Hetzelfde voor gamma scalping zonder `hedging.enabled` of zonder `gamma_scalp` in
`hedging.strategies`: een ongehedgede straddle is geen gamma scalp.
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
    if s.gamma_scalp.enabled and not (cfg.hedging.enabled
                                      and "gamma_scalp" in cfg.hedging.strategies):
        raise ConfigContractError("gamma_scalp staat aan, maar hedging.enabled is false of "
                                  "gamma_scalp staat niet in hedging.strategies: zonder "
                                  "delta-hedge geen gamma scalping")
    csp = CashSecuredPut(s.cash_secured_put, rules, eligible)
    cc = CoveredCall(s.covered_call, rules, eligible)
    built: list[tuple[bool, Strategy]] = [
        (s.credit_spread.enabled, CreditSpread(s.credit_spread, rules)),
        (s.iron_condor.enabled, IronCondor(s.iron_condor, rules)),
        (s.debit_vertical.enabled, DebitVertical(s.debit_vertical, rules)),
        (s.calendar_spread.enabled, CalendarSpread(s.calendar_spread, rules)),
        # Gamma scalp vóór de long straddle: de scan geeft per onderliggende per dag één
        # entry, in deze volgorde. Erna kwam hij nooit aan bod, want op een hedgebare naam vond
        # LongVol (dezelfde ATM-straddle, zonder hedge) altijd eerst een kandidaat (replay
        # 2026-11-27). Op namen buiten `share_eligible` scant gamma scalp niet, dus daar houdt
        # LongVol de plek. Staat gamma scalp uit, dan verandert er niets.
        (s.gamma_scalp.enabled, GammaScalp(s.gamma_scalp, rules, eligible)),
        (s.long_straddle_strangle.enabled, LongVol(s.long_straddle_strangle, rules)),
        (s.cash_secured_put.enabled, csp),
        (s.covered_call.enabled, cc),
        (s.wheel.enabled, Wheel(csp, cc)),
        (s.short_straddle_strangle.enabled, ShortVol(s.short_straddle_strangle, rules)),
    ]
    return tuple(strategy for on, strategy in built if on)
