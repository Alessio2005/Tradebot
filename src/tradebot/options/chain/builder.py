"""Ruwe quotes plus spot → een `Chain` met IV (mid, bid, ask) en Greeks per contract.

Per contract (detailontwerp 6.1):

1. quote ouder dan `max_quote_age_seconds` → gemarkeerd als `stale`;
2. IV op de mid (en met `spread_iv` ook op bid en ask), met de engine voor dit gebruik
   (`engine_for`: scan = BAW, beslissing = binomiaal, FD bij een discreet dividend);
3. Greeks alleen als de IV op de mid `OK` is. Er bestaat geen pad dat een IV invult.

De builder filtert niet: dat doet `chain.filters`, met een reden per afgewezen contract. Wel
kan `build` een `skip`-regel krijgen: een contract dat op zijn quote al afvalt (of `stale` is),
krijgt geen IV-solve en geen Greeks. Dat scheelt het meeste rekenwerk; de filter geeft zo'n
contract daarna dezelfde quote-reden.

IN BATCHES (`pricing.vectorized`, standaard aan)
------------------------------------------------
Dezelfde uitkomst als contract voor contract, met minder QuantLib-aanroepen:

* BAW-contracten (scan, geen discreet dividend) gaan samen door de numpy-kernel
  (`pricing.batch`, tegen QuantLib vastgepind): IV op mid (en bid/ask), Greeks bij een
  `OK` op de mid. Waar de kernel `TO_QUANTLIB` geeft of een Greek NaN is, rekent QuantLib
  dat ene getal zoals voorheen (met de terugvalengine);
* FD-contracten (discreet dividend) krijgen een startwaarde `σ₀`: de kernel-IV van een BAW
  op de spot min de contante waarde van de dividenden in het venster (tegen de rente, over
  de echte tijd tot de ex-datum), zonder discrete dividenden. Brent probeert eerst
  `[max(iv_min, σ₀ − 0,03), min(iv_max, σ₀ + 0,03)]`, dan het volle pad;
* alle andere contracten (de beslissingsengine) gaan door `price_one`.

De volgorde van `Chain.options` blijft die van `raw`.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time

import numpy as np
import numpy.typing as npt

from ...utils.failfast import DataContractError, require
from ..calendar import NY, year_fraction
from ..contracts import OptionContract, Quote, Right
from ..pricing.batch import BatchInputs, KernelIVStatus, baw_greeks, implied_vols
from ..pricing.engines import Method, Use, engine_for
from ..pricing.greeks import greeks
from ..pricing.implied_vol import IVResult, IVStatus, implied_vol
from ..pricing.settings import PricerSettings
from ..pricing.vanilla import OptionInputs
from ..rates import Carry
from ..units import GreeksPerShare

__all__ = ["Chain", "ChainBuilder", "PricedOption", "RawOption"]

#: De halve breedte van het Brent-interval rond de FD-startwaarde (in vol).
_FD_SEED_HALF_WIDTH = 0.03


@dataclass(frozen=True, slots=True)
class RawOption:
    contract: OptionContract
    quote: Quote


@dataclass(frozen=True, slots=True)
class PricedOption:
    contract: OptionContract
    quote: Quote
    iv_mid: IVResult
    iv_bid: IVResult
    iv_ask: IVResult
    greeks: GreeksPerShare | None
    method: Method
    stale: bool

    @property
    def iv(self) -> float | None:
        return self.iv_mid.value if self.iv_mid.status is IVStatus.OK else None

    @property
    def delta(self) -> float | None:
        return self.greeks.delta if self.greeks is not None else None


@dataclass(frozen=True, slots=True)
class Chain:
    underlying: str
    spot: float
    asof: datetime
    carry: Carry
    options: tuple[PricedOption, ...]

    def expiries(self) -> tuple[date, ...]:
        return tuple(sorted({o.contract.expiry for o in self.options}))

    def by_expiry(self, expiry: date, right: Right | None = None) -> tuple[PricedOption, ...]:
        """Contracten van één expiratie, op strike gesorteerd (optioneel één kant)."""
        rows = [o for o in self.options if o.contract.expiry == expiry
                and (right is None or o.contract.right is right)]
        return tuple(sorted(rows, key=lambda o: (o.contract.strike, o.contract.right.value)))

    def find(self, expiry: date, strike: float, right: Right) -> PricedOption | None:
        for o in self.options:
            c = o.contract
            if c.expiry == expiry and c.right is right and abs(c.strike - strike) < 1e-9:
                return o
        return None


class ChainBuilder:
    def __init__(self, settings: PricerSettings, max_quote_age_seconds: float,
                 use: Use = Use.SCAN, spread_iv: bool = False) -> None:
        self.settings = settings
        self.max_quote_age_seconds = max_quote_age_seconds
        self.use = use
        #: IV ook op bid en ask (de IV-spread). Kost twee extra solves per contract; de scan
        #: heeft hem niet nodig, rapportage wel.
        self.spread_iv = spread_iv

    def _unpriced(self, raw: RawOption, method: Method, stale: bool) -> PricedOption:
        none = IVResult(IVStatus.NO_QUOTE)
        return PricedOption(raw.contract, raw.quote, none, none, none, None, method, stale)

    def _sides(self, q: Quote) -> tuple[float | None, ...]:
        """De prijzen die een IV krijgen: de mid, met `spread_iv` ook bid en ask."""
        return (q.mid, q.bid, q.ask) if self.spread_iv else (q.mid,)

    def _assemble(self, raw: RawOption, method: Method, ivs: Sequence[IVResult],
                  g: GreeksPerShare | None) -> PricedOption:
        skipped = IVResult(IVStatus.NO_QUOTE)
        mid, bid, ask = (*ivs, skipped, skipped)[:3]
        return PricedOption(raw.contract, raw.quote, mid, bid, ask, g, method, False)

    def price_one(self, raw: RawOption, spot: float, asof: datetime, carry: Carry, *,
                  skip: Callable[[RawOption], bool] | None = None) -> PricedOption:
        c, q = raw.contract, raw.quote
        inp = OptionInputs(spot, c.strike, c.right, c.expiry, asof, carry)
        has_div = bool(inp.discrete_dividends)
        method = engine_for(self.use, has_div, self.settings.engines)
        stale = (asof - q.ts).total_seconds() > self.max_quote_age_seconds
        if stale or (skip is not None and skip(raw)):
            return self._unpriced(raw, method, stale)
        def solve(p: float | None) -> IVResult:
            return implied_vol(inp, p, method, self.settings.engines, self.settings.iv)

        mid = solve(q.mid)
        skipped = IVResult(IVStatus.NO_QUOTE)
        bid, ask = (solve(q.bid), solve(q.ask)) if self.spread_iv else (skipped, skipped)
        g = None
        if mid.status is IVStatus.OK and mid.value is not None:
            g = greeks(inp, mid.value, mid.method or method, self.settings.engines,
                       self.settings.bumps)
        return PricedOption(c, q, mid, bid, ask, g, method, stale)

    def build(self, underlying: str, spot: float, raw: Sequence[RawOption], asof: datetime,
              carry: Carry, *, skip: Callable[[RawOption], bool] | None = None) -> Chain:
        require(all(r.contract.underlying == underlying for r in raw),
                "een keten bevat één onderliggende", DataContractError, underlying=underlying)
        if not self.settings.vectorized:
            priced = tuple(self.price_one(r, spot, asof, carry, skip=skip) for r in raw
                           if not OptionInputs(spot, r.contract.strike, r.contract.right,
                                               r.contract.expiry, asof, carry).expired)
            return Chain(underlying, spot, asof, carry, priced)
        live = [(r, inp) for r in raw
                if not (inp := OptionInputs(spot, r.contract.strike, r.contract.right,
                                            r.contract.expiry, asof, carry)).expired]
        rows: list[PricedOption | None] = [None] * len(live)
        baw: list[int] = []
        fd: list[int] = []
        for i, (r, inp) in enumerate(live):
            method = engine_for(self.use, bool(inp.discrete_dividends), self.settings.engines)
            stale = (asof - r.quote.ts).total_seconds() > self.max_quote_age_seconds
            if stale or (skip is not None and skip(r)):
                rows[i] = self._unpriced(r, method, stale)
            elif method is Method.BAW:
                baw.append(i)
            elif method is Method.FD:
                fd.append(i)
            else:
                rows[i] = self.price_one(r, spot, asof, carry)
        for i, po in zip(baw, self._price_baw([live[i] for i in baw]), strict=True):
            rows[i] = po
        for i, po in zip(fd, self._price_fd([live[i] for i in fd]), strict=True):
            rows[i] = po
        return Chain(underlying, spot, asof, carry, tuple(r for r in rows if r is not None))

    def _price_baw(self, items: Sequence[tuple[RawOption, OptionInputs]]
                   ) -> list[PricedOption]:
        """BAW-contracten in één batch; `TO_QUANTLIB` en NaN-Greeks per stuk met QuantLib."""
        if not items:
            return []
        inps = [inp for _, inp in items]
        batch = BatchInputs.from_options(inps)
        sides = [self._sides(r.quote) for r, _ in items]
        ivs = list(zip(*(self._baw_ivs(batch, inps, [s[k] for s in sides])
                         for k in range(len(sides[0]))), strict=True))
        mids = [iv[0] for iv in ivs]
        kernel = np.array([m.status is IVStatus.OK and m.method is Method.BAW for m in mids])
        idx = np.flatnonzero(kernel)
        vols = np.array([mids[i].value for i in idx], dtype=np.float64)
        bg = baw_greeks(batch.subset(idx), vols, self.settings.bumps)
        g: list[GreeksPerShare | None] = [None] * len(items)
        for j, i in enumerate(idx):
            values = (bg.delta[j], bg.gamma[j], bg.vega[j], bg.theta[j], bg.rho[j])
            if all(math.isfinite(v) for v in values):
                g[i] = GreeksPerShare(*(float(v) for v in values))
        for i, m in enumerate(mids):
            if m.status is IVStatus.OK and m.value is not None and g[i] is None:
                g[i] = greeks(inps[i], m.value, m.method or Method.BAW, self.settings.engines,
                              self.settings.bumps)
        return [self._assemble(r, Method.BAW, iv, gi)
                for (r, _), iv, gi in zip(items, ivs, g, strict=True)]

    def _baw_ivs(self, batch: BatchInputs, inps: Sequence[OptionInputs],
                 prices: Sequence[float | None]) -> list[IVResult]:
        """De kernel-IV per prijs, als het `IVResult` dat `implied_vol` met BAW zou geven."""
        res = implied_vols(batch, _prices(prices), self.settings.iv)
        out: list[IVResult] = []
        for inp, p, status, v in zip(inps, prices, res.status, res.value, strict=True):
            if status is KernelIVStatus.TO_QUANTLIB:
                out.append(implied_vol(inp, p, Method.BAW, self.settings.engines,
                                       self.settings.iv))
            elif status is KernelIVStatus.OK:
                out.append(IVResult(IVStatus.OK, float(v), Method.BAW))
            else:
                out.append(IVResult(IVStatus(status.value)))
        return out

    def _price_fd(self, items: Sequence[tuple[RawOption, OptionInputs]]) -> list[PricedOption]:
        """FD-contracten met QuantLib, Brent eerst rond de startwaarde uit de kernel."""
        if not items:
            return []
        sides = [self._sides(r.quote) for r, _ in items]
        escrowed = [_escrowed(inp) for _, inp in items]
        has_seed = [i for i, e in enumerate(escrowed) if e is not None]
        seeds: list[list[float | None]] = [[None] * len(s) for s in sides]
        if has_seed:
            batch = BatchInputs.from_options([e for e in escrowed if e is not None])
            for k in range(len(sides[0])):
                res = implied_vols(batch, _prices([sides[i][k] for i in has_seed]),
                                   self.settings.iv)
                for i, status, v in zip(has_seed, res.status, res.value, strict=True):
                    if status is KernelIVStatus.OK:
                        seeds[i][k] = float(v)
        cfg = self.settings.iv
        out: list[PricedOption] = []
        for (r, inp), prices, seed in zip(items, sides, seeds, strict=True):
            ivs = [implied_vol(inp, p, Method.FD, self.settings.engines, cfg,
                               bracket=None if s0 is None else
                               (max(cfg.min_vol, s0 - _FD_SEED_HALF_WIDTH),
                                min(cfg.max_vol, s0 + _FD_SEED_HALF_WIDTH)))
                   for p, s0 in zip(prices, seed, strict=True)]
            mid = ivs[0]
            g = None
            if mid.status is IVStatus.OK and mid.value is not None:
                g = greeks(inp, mid.value, mid.method or Method.FD, self.settings.engines,
                           self.settings.bumps)
            out.append(self._assemble(r, Method.FD, ivs, g))
        return out


def _prices(prices: Sequence[float | None]) -> npt.NDArray[np.float64]:
    """`None` (geen quote) als NaN, zoals `implied_vols` het verwacht."""
    return np.array([np.nan if p is None else p for p in prices], dtype=np.float64)


def _escrowed(inp: OptionInputs) -> OptionInputs | None:
    """Dezelfde optie op `spot − Σ PV(dividend)` zonder discrete dividenden, voor de
    startwaarde van FD. PV tegen de rente over de echte tijd tot het begin van de ex-datum
    (New York). `None` als er dan geen positieve spot overblijft."""
    pv = sum(d.amount * math.exp(-inp.carry.rate * max(
        year_fraction(inp.now, datetime.combine(d.ex_date, time(0), tzinfo=NY)), 0.0))
        for d in inp.discrete_dividends)
    spot = inp.spot - pv
    if spot <= 0:
        return None
    return OptionInputs(spot, inp.strike, inp.right, inp.expiry, inp.now,
                        Carry(inp.carry.rate, inp.carry.div_yield))
