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
* alle andere contracten (de beslissingsengine) krijgen hetzelfde werk als in `price_one`.

PROCESPOOL (`pricing.workers`)
------------------------------
Het QuantLib-deel (FD, binomiaal en de terugval van de kernel) is per contract een
`_QuantLibJob`: picklebaar en zonder QuantLib-objecten. `solve_quantlib` (module-level, dus
ook met `spawn` of `forkserver` bruikbaar) rekent een lijst jobs uit. Bij `workers > 1` en
meer dan `pool_min_contracts` jobs gaan ze in vaste, aaneengesloten chunks naar een
`ProcessPoolExecutor`; `map` geeft ze in de oorspronkelijke volgorde terug. QuantLib's
`evaluationDate` is globaal per proces: daarom processen en geen threads. De workers starten
met `forkserver` waar dat bestaat (Linux, macOS; `fork` uit een proces met threads kan
vastlopen), anders met de standaardmethode. De pool komt er pas bij de eerste keer dat hij
nodig is, wordt hergebruikt en sluit met `close()`.

De volgorde van `Chain.options` blijft die van `raw`.
"""
from __future__ import annotations

import math
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from typing import cast

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

__all__ = ["Chain", "ChainBuilder", "PricedOption", "RawOption", "solve_quantlib"]

#: De halve breedte van het Brent-interval rond de FD-startwaarde (in vol).
_FD_SEED_HALF_WIDTH = 0.03
#: Chunks per worker: genoeg om ongelijke contracten te spreiden, weinig genoeg om het
#: heen-en-weer sturen klein te houden.
_CHUNKS_PER_WORKER = 2


def _mp_context() -> multiprocessing.context.BaseContext:
    """`forkserver` waar die bestaat, anders de standaard startmethode (`spawn`)."""
    if "forkserver" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context()


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


@dataclass(frozen=True, slots=True)
class _QuantLibJob:
    """Het QuantLib-werk van één contract: IV per prijs en (met `greeks`) de Greeks."""

    inp: OptionInputs
    method: Method
    prices: tuple[float | None, ...]
    #: per prijs een al bekende uitkomst (uit de kernel), of `None`: met QuantLib oplossen
    known: tuple[IVResult | None, ...]
    #: per prijs het smalle Brent-interval (FD-startwaarde) of `None`
    brackets: tuple[tuple[float, float] | None, ...]
    #: Greeks met QuantLib als de IV op de mid (de eerste prijs) `OK` is
    greeks: bool


def solve_quantlib(settings: PricerSettings, jobs: Sequence[_QuantLibJob]
                   ) -> list[tuple[tuple[IVResult, ...], GreeksPerShare | None]]:
    """De jobs met QuantLib, in volgorde: per job de IV's en de Greeks (of `None`)."""
    out: list[tuple[tuple[IVResult, ...], GreeksPerShare | None]] = []
    for job in jobs:
        ivs = tuple(k if k is not None else
                    implied_vol(job.inp, p, job.method, settings.engines, settings.iv, bracket=b)
                    for p, k, b in zip(job.prices, job.known, job.brackets, strict=True))
        mid = ivs[0]
        g = None
        if job.greeks and mid.status is IVStatus.OK and mid.value is not None:
            g = greeks(job.inp, mid.value, mid.method or job.method, settings.engines,
                       settings.bumps)
        out.append((ivs, g))
    return out


class ChainBuilder:
    def __init__(self, settings: PricerSettings, max_quote_age_seconds: float,
                 use: Use = Use.SCAN, spread_iv: bool = False) -> None:
        self.settings = settings
        self.max_quote_age_seconds = max_quote_age_seconds
        self.use = use
        #: IV ook op bid en ask (de IV-spread). Kost twee extra solves per contract; de scan
        #: heeft hem niet nodig, rapportage wel.
        self.spread_iv = spread_iv
        self._pool: ProcessPoolExecutor | None = None

    def close(self) -> None:
        """Sluit de procespool (als die er is); daarna maakt een nieuwe build hem opnieuw."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _solve(self, jobs: Sequence[_QuantLibJob]
               ) -> list[tuple[tuple[IVResult, ...], GreeksPerShare | None]]:
        """`solve_quantlib`, in de pool als die loont; de uitkomst in de volgorde van `jobs`."""
        workers = self.settings.workers
        if workers <= 1 or len(jobs) <= self.settings.pool_min_contracts:
            return solve_quantlib(self.settings, jobs)
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=workers, mp_context=_mp_context())
        size = math.ceil(len(jobs) / (workers * _CHUNKS_PER_WORKER))
        chunks = [jobs[i:i + size] for i in range(0, len(jobs), size)]
        parts = self._pool.map(solve_quantlib, [self.settings] * len(chunks), chunks)
        return [res for part in parts for res in part]

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
        ql: list[int] = []
        jobs: list[_QuantLibJob] = []
        for i, (r, inp) in enumerate(live):
            method = engine_for(self.use, bool(inp.discrete_dividends), self.settings.engines)
            stale = (asof - r.quote.ts).total_seconds() > self.max_quote_age_seconds
            if stale or (skip is not None and skip(r)):
                rows[i] = self._unpriced(r, method, stale)
            elif method is Method.BAW:
                baw.append(i)
            else:
                prices = self._sides(r.quote)
                todo = (None,) * len(prices)
                ql.append(i)
                jobs.append(_QuantLibJob(inp, method, prices, todo, todo, True))
        fd = [j for j, i in enumerate(ql) if jobs[j].method is Method.FD]
        for j, brackets in zip(fd, self._fd_brackets([jobs[j] for j in fd]), strict=True):
            jobs[j] = replace(jobs[j], brackets=brackets)
        baw_items = [live[i] for i in baw]
        baw_ivs = self._baw_kernel_ivs(baw_items)
        fallback = [j for j, ivs in enumerate(baw_ivs) if None in ivs]
        jobs += [_QuantLibJob(baw_items[j][1], Method.BAW, self._sides(baw_items[j][0].quote),
                              baw_ivs[j], (None,) * len(baw_ivs[j]), False) for j in fallback]
        solved = self._solve(jobs)
        n = len(ql)
        for i, job, (ivs, g) in zip(ql, jobs[:n], solved[:n], strict=True):
            rows[i] = self._assemble(live[i][0], job.method, ivs, g)
        done = {j: ivs for j, (ivs, _) in zip(fallback, solved[n:], strict=True)}
        resolved = [done[j] if j in done else cast(tuple[IVResult, ...], ivs)
                    for j, ivs in enumerate(baw_ivs)]
        for i, po in zip(baw, self._price_baw(baw_items, resolved), strict=True):
            rows[i] = po
        return Chain(underlying, spot, asof, carry, tuple(r for r in rows if r is not None))

    def _baw_kernel_ivs(self, items: Sequence[tuple[RawOption, OptionInputs]]
                        ) -> list[tuple[IVResult | None, ...]]:
        """De kernel-IV per contract en prijs, als het `IVResult` dat `implied_vol` met BAW
        zou geven; `None` waar de kernel `TO_QUANTLIB` geeft."""
        if not items:
            return []
        batch = BatchInputs.from_options([inp for _, inp in items])
        sides = [self._sides(r.quote) for r, _ in items]
        per_side = []
        for k in range(len(sides[0])):
            res = implied_vols(batch, _prices([s[k] for s in sides]), self.settings.iv)
            per_side.append([None if status is KernelIVStatus.TO_QUANTLIB else
                             IVResult(IVStatus.OK, float(v), Method.BAW)
                             if status is KernelIVStatus.OK else IVResult(IVStatus(status.value))
                             for status, v in zip(res.status, res.value, strict=True)])
        return list(zip(*per_side, strict=True))

    def _price_baw(self, items: Sequence[tuple[RawOption, OptionInputs]],
                   ivs: Sequence[tuple[IVResult, ...]]) -> list[PricedOption]:
        """BAW-contracten met hun IV's: Greeks in één batch, NaN-Greeks per stuk met QuantLib."""
        if not items:
            return []
        inps = [inp for _, inp in items]
        batch = BatchInputs.from_options(inps)
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
        todo = [i for i, m in enumerate(mids) if m.status is IVStatus.OK and g[i] is None]
        jobs = [_QuantLibJob(inps[i], Method.BAW, (items[i][0].quote.mid,), (mids[i],), (None,),
                             True) for i in todo]
        for i, (_, gi) in zip(todo, self._solve(jobs), strict=True):
            g[i] = gi
        return [self._assemble(r, Method.BAW, iv, gi)
                for (r, _), iv, gi in zip(items, ivs, g, strict=True)]

    def _fd_brackets(self, jobs: Sequence[_QuantLibJob]
                     ) -> list[tuple[tuple[float, float] | None, ...]]:
        """Per FD-contract en prijs het Brent-interval rond de startwaarde uit de kernel (een
        BAW op de escrowed spot), of `None` zonder startwaarde."""
        if not jobs:
            return []
        escrowed = [_escrowed(job.inp) for job in jobs]
        has_seed = [i for i, e in enumerate(escrowed) if e is not None]
        seeds: list[list[float | None]] = [[None] * len(job.prices) for job in jobs]
        if has_seed:
            batch = BatchInputs.from_options([e for e in escrowed if e is not None])
            for k in range(len(jobs[0].prices)):
                res = implied_vols(batch, _prices([jobs[i].prices[k] for i in has_seed]),
                                   self.settings.iv)
                for i, status, v in zip(has_seed, res.status, res.value, strict=True):
                    if status is KernelIVStatus.OK:
                        seeds[i][k] = float(v)
        cfg = self.settings.iv
        return [tuple(None if s0 is None else
                      (max(cfg.min_vol, s0 - _FD_SEED_HALF_WIDTH),
                       min(cfg.max_vol, s0 + _FD_SEED_HALF_WIDTH)) for s0 in seed)
                for seed in seeds]


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
