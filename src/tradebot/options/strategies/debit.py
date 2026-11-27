"""Debit-strategieën: debit vertical, calendar spread en long straddle/strangle (detailontwerp 7.4).

Bij elke debit-strategie is het maximale verlies het betaalde debit. "Waarde" is de mark van de
combo (wat sluiten nu oplevert, per aandeel). De expiratieregel uit 7.5 geldt ook hier: op de
handelsdag vóór een expiratie, na `close_before_expiry_time`, gaat alles dicht.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

from ...schemas.options import CalendarSpreadConfig, DebitVerticalConfig, LongVolConfig
from ..calendar import expiry_moment, year_fraction
from ..chain.builder import Chain
from ..contracts import Right
from ..iv_metrics import IVCondition, IVRStatus
from ..positions import (
    MULTIPLIER,
    Action,
    Book,
    Candidate,
    LegIntent,
    MarketState,
    StrategyPosition,
    Trend,
    Urgency,
    mark,
    position_greek,
)
from ..pricing.surface import forward
from .common import (
    Rules,
    close,
    dte,
    entry_blockers,
    expiry_close_due,
    iv_ok,
    otm_step,
    parse_time,
    pick_expiry,
    size,
    strike_at_delta,
    strike_nearest,
)

__all__ = ["CalendarSpread", "DebitVertical", "LongVol"]


def _manage_debit(pos: StrategyPosition, market: MarketState, rules: Rules,
                  take_profit_value: float, stop_value: float, close_dte: int | None) -> list[Action]:
    """Exits op waarde, als veelvoud van het debit, en op DTE (de vroegste expiratie)."""
    expiries = pos.expiries()
    if any(expiry_close_due(e, market.now, market, rules) for e in expiries):
        return [close(pos, Urgency.STOP, "expiry")]
    chain = market.chains.get(pos.underlying)
    debit = -pos.entry_credit
    m = mark(chain, pos.intents()) if chain is not None else None
    if m is not None and m >= take_profit_value * debit:
        return [close(pos, Urgency.NORMAL, "take_profit")]
    if m is not None and m <= stop_value * debit:
        return [close(pos, Urgency.STOP, "stop_loss")]
    if close_dte is not None and dte(expiries[0], market.today) <= close_dte:
        return [close(pos, Urgency.NORMAL, "dte")]
    return []


def _level(chain: Chain, expiry: date) -> float:
    f = forward(chain, expiry)
    return f if f is not None else chain.spot


class DebitVertical:
    name = "debit_vertical"

    def __init__(self, cfg: DebitVerticalConfig, rules: Rules) -> None:
        self.cfg, self.rules = cfg, rules

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, out = self.cfg, []
        for und in sorted(market.chains):
            chain, trend = market.chains[und], market.trend.get(und)
            if trend not in (Trend.UP, Trend.DOWN):
                continue
            right = Right.CALL if trend is Trend.UP else Right.PUT
            expiry = pick_expiry(chain, *cfg.dte, market.today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                continue
            ok, iv_trace = iv_ok(market, und, IVCondition(cfg.iv_condition), self.rules)
            long = strike_at_delta(chain, expiry, right, cfg.long_delta_target, cfg.long_delta)
            if not ok or long is None:
                continue
            found = None
            for n in range(1, cfg.max_width_strikes + 1):
                short = otm_step(chain, long, n)
                if short is None:
                    break
                legs = (LegIntent(long.contract, 1), LegIntent(short.contract, -1))
                debit = mark(chain, legs)
                width = abs(short.contract.strike - long.contract.strike)
                if debit is not None and 0 < debit <= cfg.max_debit_frac_width * width:
                    found = legs, debit, width
                    break
            if found is None:
                continue
            legs, debit, width = found
            units = size(debit * MULTIPLIER, market, book, und, self.rules, legs)
            if units == 0:
                continue
            kind = "bull_call" if right is Right.CALL else "bear_put"
            out.append(Candidate(self.name, und, legs, units, -debit,
                                 debit * MULTIPLIER * units, (width - debit) / debit,
                                 (f"trend:{trend.value}", kind, iv_trace,
                                  f"dte:{dte(expiry, market.today)}", f"width:{width:g}")))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        c = self.cfg
        return _manage_debit(pos, market, self.rules, 1 + c.take_profit_frac_debit,
                             c.stop_loss_frac_debit, c.close_dte)


class CalendarSpread:
    name = "calendar_spread"

    def __init__(self, cfg: CalendarSpreadConfig, rules: Rules) -> None:
        self.cfg, self.rules = cfg, rules

    def _iv_trace(self, market: MarketState, und: str) -> str | None:
        """None als de IV-voorwaarde niet geldt, anders de spoorregel."""
        m = market.iv.get(und)
        if m is None or m.term_slope is None or m.term_slope < self.cfg.min_term_slope:
            return None
        if m.status is IVRStatus.OK and m.ivp is not None:
            return "ivp:ivr:ok" if m.ivp <= self.cfg.max_ivp else None
        if (self.rules.iv.fallback_enabled and m.iv_rv_ratio is not None
                and m.iv_rv_ratio <= self.cfg.max_iv_rv_ratio_fallback):
            return "ivp:fallback:ok"
        return None

    @staticmethod
    def _atm_iv(chain: Chain, expiry: date) -> float | None:
        """Gemiddelde van call- en put-IV op de strike het dichtst bij de forward."""
        call = strike_nearest(chain, expiry, Right.CALL, _level(chain, expiry))
        put = chain.find(expiry, call.contract.strike, Right.PUT) if call is not None else None
        if call is None or put is None or call.iv is None or put.iv is None:
            return None
        return (call.iv + put.iv) / 2

    def _fwd_trace(self, market: MarketState, chain: Chain, und: str,
                   near: date, far: date) -> str:
        """Spoorregel van de forward-vol-regel; `fwd:ok` als die geldt, anders de reden.

        Zonder looptijd tussen de twee expiraties (`far <= near`) bestaat er geen forward-vol:
        `fwd:no_gap` (de config sluit dat al uit, zie `CalendarSpreadConfig`)."""
        if far <= near:
            return "fwd:no_gap"
        vs = market.vol.get(und)
        if vs is None or vs.rv is None:
            return "fwd:no_rv"
        iv1, iv2 = self._atm_iv(chain, near), self._atm_iv(chain, far)
        if iv1 is None or iv2 is None:
            return "fwd:no_iv"
        t1 = year_fraction(market.now, expiry_moment(near))
        t2 = year_fraction(market.now, expiry_moment(far))
        var_f = (iv2 ** 2 * t2 - iv1 ** 2 * t1) / (t2 - t1)
        if var_f < 0:
            return "fwd:neg_var"
        h1 = (near - market.today).days
        sigma_near = vs.rv.at_horizon(h1)
        sigma_gap = vs.rv.at_horizon((far - near).days)
        if (math.sqrt(var_f) <= self.cfg.max_fwd_vol_rv_ratio * sigma_gap
                and iv1 >= self.cfg.min_near_iv_rv_ratio * sigma_near):
            return "fwd:ok"
        return "fwd:rejected"

    def _entry_trace(self, market: MarketState, chain: Chain, und: str,
                     near: date, far: date) -> tuple[str, ...]:
        """De spoorregels van de toelatingsroute(s) die gelden; leeg als geen."""
        rule, out = self.cfg.entry_rule, []
        if rule in ("ivp", "either"):
            ivp = self._iv_trace(market, und)
            if ivp is not None:
                out.append(ivp)
        if rule in ("forward_vol", "either"):
            fwd = self._fwd_trace(market, chain, und, near, far)
            if fwd == "fwd:ok":
                out.append(fwd)
        return tuple(out)

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, out = self.cfg, []
        for und in sorted(market.chains):
            chain = market.chains[und]
            near = pick_expiry(chain, *cfg.short_dte, market.today)
            far = pick_expiry(chain, *cfg.long_dte, market.today)
            if near is None or far is None or entry_blockers(market, und, near, self.rules):
                continue
            trace = self._entry_trace(market, chain, und, near, far)
            short = strike_nearest(chain, near, Right.CALL, _level(chain, near))
            if not trace or short is None:
                continue
            long = chain.find(far, short.contract.strike, Right.CALL)
            if long is None:
                continue
            legs = (LegIntent(short.contract, -1), LegIntent(long.contract, 1))
            debit = mark(chain, legs)
            theta = position_greek(chain, legs, "theta")
            if debit is None or debit <= 0 or theta is None:
                continue
            units = size(debit * MULTIPLIER, market, book, und, self.rules, legs)
            if units == 0:
                continue
            out.append(Candidate(self.name, und, legs, units, -debit,
                                 debit * MULTIPLIER * units, theta / (debit * MULTIPLIER),
                                 (*trace, f"near:{near.isoformat()}", f"far:{far.isoformat()}",
                                  f"strike:{short.contract.strike:g}")))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        c = self.cfg
        return _manage_debit(pos, market, self.rules, c.take_profit_multiple_debit,
                             c.stop_loss_frac_debit, None)


class LongVol:
    """Long straddle of strangle: (a) vóór earnings, (b) bij lage IV."""

    name = "long_straddle_strangle"

    def __init__(self, cfg: LongVolConfig, rules: Rules) -> None:
        self.cfg, self.rules = cfg, rules
        self.pre_earnings_close = parse_time(cfg.pre_earnings_close_time)

    def _legs(self, chain: Chain, expiry: date) -> tuple[LegIntent, ...] | None:
        if self.cfg.style == "straddle":
            call = strike_nearest(chain, expiry, Right.CALL, _level(chain, expiry))
            put = (chain.find(expiry, call.contract.strike, Right.PUT)
                   if call is not None else None)
        else:
            lo, hi = self.cfg.strangle_delta
            call = strike_at_delta(chain, expiry, Right.CALL, (lo + hi) / 2, (lo, hi))
            put = strike_at_delta(chain, expiry, Right.PUT, (lo + hi) / 2, (lo, hi))
        if call is None or put is None:
            return None
        return LegIntent(put.contract, 1), LegIntent(call.contract, 1)

    def _setup(self, market: MarketState, und: str) -> tuple[date, str, str] | None:
        """(expiratie, variant, spoorregel) of None."""
        cfg, chain, today = self.cfg, market.chains[und], market.today
        ev = market.events.get(und)
        if cfg.earnings_variant and ev is not None and ev.usable(
                today, self.rules.max_calendar_data_age_days):
            earn = ev.next_earnings(today)
            lo, hi = cfg.earnings_window_days
            if earn is not None and lo <= (earn - today).days <= hi:
                after = [e for e in chain.expiries()
                         if e > earn and dte(e, today) >= cfg.earnings_min_dte]
                if after:
                    return after[0], "earnings", f"earnings:{earn.isoformat()}"
        if cfg.low_iv_variant:
            expiry = pick_expiry(chain, *cfg.low_iv_dte, today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                return None
            ok, trace = iv_ok(market, und, IVCondition(cfg.low_iv_condition), self.rules)
            if ok:
                return expiry, "low_iv", trace
        return None

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        out = []
        for und in sorted(market.chains):
            chain = market.chains[und]
            setup = self._setup(market, und)
            if setup is None:
                continue
            expiry, variant, trace = setup
            legs = self._legs(chain, expiry)
            debit = mark(chain, legs) if legs is not None else None
            vega = position_greek(chain, legs, "vega") if legs is not None else None
            if legs is None or debit is None or debit <= 0 or vega is None:
                continue
            units = size(debit * MULTIPLIER, market, book, und, self.rules, legs)
            if units == 0:
                continue
            out.append(Candidate(self.name, und, legs, units, -debit,
                                 debit * MULTIPLIER * units, vega / (debit * MULTIPLIER),
                                 (trace, self.cfg.style, f"dte:{dte(expiry, market.today)}"),
                                 variant))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def _before_earnings(self, pos: StrategyPosition, market: MarketState) -> bool:
        ev = market.events.get(pos.underlying)
        earn = ev.next_earnings(market.today - timedelta(days=1)) if ev is not None else None
        if earn is None:
            return False
        last = market.calendar.previous_trading_day(earn)
        cutoff = datetime.combine(last, self.pre_earnings_close, tzinfo=market.now.tzinfo)
        return market.now >= cutoff

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        c = self.cfg
        if pos.variant == "earnings" and self._before_earnings(pos, market):
            return [close(pos, Urgency.STOP, "pre_earnings")]
        return _manage_debit(pos, market, self.rules, c.take_profit_multiple_debit,
                             c.stop_loss_frac_debit, c.close_dte)
