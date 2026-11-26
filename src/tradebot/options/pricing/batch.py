"""Een numpy-kernel voor een hele keten tegelijk: Europees, BAW, IV en BAW-Greeks.

QuantLib is de autoriteit (zie `vanilla`, `implied_vol`, `greeks`). Deze kernel rekent
hetzelfde in numpy, over arrays van contracten, en is pas bruikbaar omdat
`tests/unit/options/test_pricing_batch.py` hem tegen QuantLib vastpint. Geen afwijkend model:
elke formule hieronder is een poort van wat QuantLib 1.44 rekent.

TIJD EN PARAMETERS
------------------
Per contract komen `t_model` (hele dagen vanaf `eval_date`, Actual/365 Fixed) en de factor
`f = t_real / t_model` uit `ql_env.scaled_time`. QuantLib ziet dan

    σ_model = σ·√f,   r_model = r·f,   q_model = q·f,
    rD = 1 / exp(r_model·t_model),   dD = 1 / exp(q_model·t_model),   var = σ_model²·t_model

(`FlatForward` disconteert als `1 / compoundFactor`, `BlackConstantVol` geeft `vol·vol·t`).

EUROPEES
--------
`BlackCalculator` op de forward `F = S·dD/rD`: call `rD·(F·N(d1) − K·N(d2))`, put
`rD·(F·(N(d1) − 1) + K·(1 − N(d2)))`, met `d1 = ln(F/K)/√var + ½√var`.

BAW (`BaroneAdesiWhaleyApproximationEngine`)
--------------------------------------------
* een call met `dD >= 1` en een put met `rD >= 1` worden nooit vroeg uitgeoefend: de Europese
  prijs. Een call bij een negatieve rente (`rD > 1`) weigert QuantLib 1.44 ("not applicable
  with negative interest rates"); de kernel geeft daar NaN. Beide regels staan niet in de
  oudere bron; de pariteitstest met `r = 0` (en de rho-bump eronder) pint ze vast;
* anders de kritieke prijs `Sk`: de seed van Barone-Adesi-Whaley, daarna de Newton-iteratie
  van QuantLib tot `|LHS − RHS| / K <= 1e-6`;
* `Q` uit `n = 2·ln(dD/rD)/var` en `K = 2/var` als `rD` binnen 1000 ulps van 1 ligt, anders
  `−2·ln(rD) / (var·(1 − rD))`;
* call: `S < Sk` → Europees + `a·(S/Sk)^Q`, anders `S − K`; put: `S > Sk` → Europees +
  `a·(S/Sk)^Q`, anders `K − S`.

Waar QuantLib een fout zou gooien (een niet-positieve forward of een negatieve Black-prijs in
de iteratie, de foutmelding "forward + displacement must be positive"), of waar de iteratie
niet binnen `_MAX_CRITICAL_ITERATIONS` stappen convergeert, geeft de kernel NaN.

Ook NaN waar de seed ontspoort: bij een lage vol en een grote carry ligt de seed aan de
verkeerde kant van de strike, met `exp(h)` in de seed tot boven 1e20. De eerste Newton-stap
is dan een uitgedoofde aftrekking, en de uitkomst (een prijs of een fout) hangt af van de
laatste bit van `exp` en `log`: numpy en QuantLib verschillen daar. Gemeten (2026-11) over
185 000 punten met vol 1-10 %: kernel en QuantLib zijn het oneens vanaf `h ≈ 23`, QuantLib
faalt vanaf `h ≈ 4`. De kernel rekent daarom alleen bij `h <= _MAX_SEED_EXPONENT` (10), en
daar faalde geen enkel punt van een contract met `h(1 %) <= 10`. `h` daalt met σ: geldt de
grens bij `σ`, dan ook bij elke hogere vol. Een NaN is nooit een prijs: de aanroeper prijst
zo'n contract met QuantLib.

IMPLIED VOLATILITY
------------------
Dezelfde statuslogica als `implied_vol` met `Method.BAW`, hetzelfde bracket
`[min_vol, max_vol]` (prijs buiten het bereik → `NO_CONVERGENCE`) en een beveiligde
secant/bisectie op het bracket tot `|Δσ| <= 1e-12`. `KernelIVStatus.TO_QUANTLIB` krijgt een
contract waar de kernel niet zeker weet wat QuantLib doet:

* de kernel geeft NaN bij `min_vol`. QuantLib probeert dan 2 % en 5 % als ondergrens, of
  rekent in het ontspoorde gebied (zie boven) een prijs die de kernel niet kan nabootsen;
  daar kan Brent ook een tussenpunt treffen dat QuantLib weigert (dan de terugvalengine);
* de kernel faalt tijdens het oplossen of convergeert niet;
* de prijs ligt zo dicht bij een rand van het bracket dat afronding OK en NO_CONVERGENCE
  kan omdraaien.

Geen verzonnen IV: de aanroeper gebruikt het bestaande QuantLib-pad (dat dan ook de
terugvalengine kiest). Geeft de kernel bij `min_vol` een prijs, dan (zie boven) ook bij elke
vol in het bracket, en weigert QuantLib er geen enkel punt.

GREEKS
------
Precies de definities van `greeks` voor BAW: delta en gamma met spotbumps, vega en rho
centraal, theta als herwaardering na `min(1 dag, resterend/2)` met de geschaalde tijd van
het verschoven tijdstip (`BatchInputs.theta_*`).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from ..calendar import expiry_moment
from ..contracts import Right
from .greeks import Bumps
from .implied_vol import IVSettings
from .ql_env import scaled_time
from .vanilla import OptionInputs

__all__ = ["BatchGreeks", "BatchIV", "BatchInputs", "KernelIVStatus", "baw_greeks",
           "baw_price", "european_price", "implied_vols"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Tolerantie van QuantLib op `|LHS − RHS| / K` in de Newton-iteratie van de kritieke prijs.
_CRITICAL_TOLERANCE = 1.0e-6
#: QuantLib kent geen grens; een iteratie die zo lang duurt, gaat naar QuantLib.
_MAX_CRITICAL_ITERATIONS = 500
_QL_EPSILON = float(np.finfo(np.float64).eps)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

#: Boven deze exponent in de seed van de kritieke prijs rekent de kernel niet (zie boven).
_MAX_SEED_EXPONENT = 10.0
#: De IV-solver stopt bij een stap van ten hoogste deze grootte.
_IV_STEP_TOLERANCE = 1.0e-12
_MAX_IV_ITERATIONS = 100
#: Ligt `waarde(σ) − prijs` aan een rand van het bracket binnen deze fractie van de spot, dan
#: kan afronding de status (OK of NO_CONVERGENCE) omdraaien: dan beslist QuantLib.
_EDGE_FRACTION = 1.0e-9


@dataclass(frozen=True, slots=True)
class BatchInputs:
    """De arrays per contract (float64, `is_call` en `expired` bool).

    `t_model` en `factor` komen uit `scaled_time(now, expiry)`; `theta_t_model`,
    `theta_factor` en `theta_days` uit `scaled_time(now + step, expiry)` met
    `step = min(1 dag, resterend/2)` (`theta_days` is `step` in dagen). Voor een verlopen
    contract zijn alle tijden NaN.
    """

    spot: FloatArray
    strike: FloatArray
    is_call: BoolArray
    t_model: FloatArray
    factor: FloatArray
    rate: FloatArray
    div_yield: FloatArray
    expired: BoolArray
    theta_t_model: FloatArray
    theta_factor: FloatArray
    theta_days: FloatArray

    @classmethod
    def from_options(cls, options: Sequence[OptionInputs]) -> BatchInputs:
        """De arrays voor `options`, met de tijden zoals `ql_env.scaled_time` ze rekent.

        Een discreet dividend kent de kernel niet; dat blijft bij QuantLib (FD).
        """
        n = len(options)
        t_model, factor = np.full(n, np.nan), np.full(n, np.nan)
        th_t, th_f, th_days = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for i, o in enumerate(options):
            if o.expired:
                continue
            st = scaled_time(o.now, o.expiry)
            t_model[i], factor[i] = st.t_model, st.factor
            step = min(timedelta(days=1), (expiry_moment(o.expiry) - o.now) / 2)
            st_next = scaled_time(o.now + step, o.expiry)
            th_t[i], th_f[i] = st_next.t_model, st_next.factor
            th_days[i] = step / timedelta(days=1)
        return cls(
            spot=np.array([o.spot for o in options], dtype=np.float64),
            strike=np.array([o.strike for o in options], dtype=np.float64),
            is_call=np.array([o.right is Right.CALL for o in options], dtype=np.bool_),
            t_model=t_model, factor=factor,
            rate=np.array([o.carry.rate for o in options], dtype=np.float64),
            div_yield=np.array([o.carry.div_yield for o in options], dtype=np.float64),
            expired=np.array([o.expired for o in options], dtype=np.bool_),
            theta_t_model=th_t, theta_factor=th_f, theta_days=th_days,
        )

    def subset(self, index: npt.NDArray[np.intp]) -> BatchInputs:
        """Alleen de contracten op de posities `index`."""
        return BatchInputs(**{name: getattr(self, name)[index]
                              for name in BatchInputs.__dataclass_fields__})

    def shifted_in_time(self) -> BatchInputs:
        """Dezelfde contracten op het tijdstip `now + step` van de theta."""
        return replace(self, t_model=self.theta_t_model, factor=self.theta_factor)


@dataclass(frozen=True, slots=True)
class _Model:
    """Wat QuantLib uit de invoer en een vol maakt: discontering, variantie, forward."""

    spot: FloatArray
    strike: FloatArray
    is_call: BoolArray
    r_disc: FloatArray
    d_disc: FloatArray
    variance: FloatArray
    std_dev: FloatArray
    forward: FloatArray


def _model(inp: BatchInputs, vol: FloatArray) -> _Model:
    f, t = inp.factor, inp.t_model
    sigma = vol * np.sqrt(f)
    r_disc = 1.0 / np.exp(inp.rate * f * t)
    d_disc = 1.0 / np.exp(inp.div_yield * f * t)
    variance = sigma * sigma * t
    return _Model(inp.spot, inp.strike, inp.is_call, r_disc, d_disc, variance,
                  np.sqrt(variance), inp.spot * d_disc / r_disc)


def _cdf(x: FloatArray) -> FloatArray:
    return np.asarray(ndtr(x), dtype=np.float64)


def _pdf(x: FloatArray) -> FloatArray:
    exponent = -(x * x) / 2.0
    return np.where(exponent <= -690.0, 0.0, _INV_SQRT_2PI * np.exp(exponent))


def _black_calculator(is_call: BoolArray, forward: FloatArray, strike: FloatArray,
                      std_dev: FloatArray, discount: FloatArray) -> FloatArray:
    """`BlackCalculator(...).value()` voor een plain-vanilla payoff."""
    d1 = np.log(forward / strike) / std_dev + 0.5 * std_dev
    n1, n2 = _cdf(d1), _cdf(d1 - std_dev)
    call = discount * (forward * n1 - strike * n2)
    put = discount * (forward * (n1 - 1.0) + strike * (1.0 - n2))
    smooth = np.where(is_call, call, put)
    # Onder QL_EPSILON rekent BlackCalculator met d1 = ±∞: de verdisconteerde intrinsieke waarde.
    sign = np.where(is_call, 1.0, -1.0)
    degenerate = discount * np.maximum(sign * (forward - strike), 0.0)
    return np.where(std_dev >= _QL_EPSILON, smooth, degenerate)


def _black_formula(is_call: BoolArray, strike: FloatArray, forward: FloatArray,
                   std_dev: FloatArray) -> tuple[FloatArray, BoolArray]:
    """`blackFormula(type, K, F, σ√T)` zonder discontering, en waar QuantLib slaagt.

    QuantLib weigert een niet-positieve forward en een negatieve uitkomst.
    """
    sign = np.where(is_call, 1.0, -1.0)
    d1 = np.log(forward / strike) / std_dev + 0.5 * std_dev
    d2 = d1 - std_dev
    smooth = sign * (forward * _cdf(sign * d1) - strike * _cdf(sign * d2))
    result: FloatArray = np.where(std_dev == 0.0, np.maximum((forward - strike) * sign, 0.0),
                                  smooth)
    ok: BoolArray = (forward > 0.0) & (result >= 0.0)
    return result, ok


def _close_to_one(x: FloatArray) -> BoolArray:
    """QuantLib `close(x, 1.0, 1000)`."""
    diff = np.abs(x - 1.0)
    tol = 1000 * _QL_EPSILON
    close: BoolArray = (x == 1.0) | ((diff <= tol * np.abs(x)) & (diff <= tol))
    return close


def _q_exponent(m: _Model) -> FloatArray:
    """De exponent `Q` van de vroege-uitoefenpremie."""
    rd, var = m.r_disc, m.variance
    n = 2.0 * np.log(m.d_disc / rd) / var
    k = np.where(_close_to_one(rd), 2.0 / var, -2.0 * np.log(rd) / (var * (1.0 - rd)))
    root = np.sqrt((n - 1.0) * (n - 1.0) + 4 * k)
    return np.where(m.is_call, (-(n - 1.0) + root) / 2, (-(n - 1.0) - root) / 2)


def _critical_price(m: _Model, q: FloatArray) -> tuple[FloatArray, BoolArray]:
    """`BaroneAdesiWhaleyApproximationEngine::criticalPrice`; geeft `Sk` en waar het lukt.

    "Lukt" betekent ook: de seed is niet ontspoord (`h <= _MAX_SEED_EXPONENT`).
    """
    call, strike, rd, dd, var = m.is_call, m.strike, m.r_disc, m.d_disc, m.variance
    sv = np.sqrt(var)
    n = 2.0 * np.log(dd / rd) / var
    mm = -2.0 * np.log(rd) / var
    bt = np.log(dd / rd)
    root = np.sqrt((n - 1.0) * (n - 1.0) + 4.0 * mm)
    qu = np.where(call, (-(n - 1.0) + root) / 2.0, (-(n - 1.0) - root) / 2.0)
    su = strike / (1.0 - 1.0 / qu)
    h = np.where(call, -(bt + 2.0 * sv) * strike / (su - strike),
                 (bt - 2.0 * sv) * strike / (strike - su))
    si = np.where(call, strike + (su - strike) * (1.0 - np.exp(h)),
                  su + (strike - su) * np.exp(h))

    def state(si: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
        forward_si = si * dd / rd
        d1 = (np.log(forward_si / strike) + 0.5 * var) / sv
        black, ok = _black_formula(call, strike, forward_si, sv)
        temp = black * rd
        n_up, n_down = _cdf(d1), _cdf(-d1)
        lhs = np.where(call, si - strike, strike - si)
        rhs = np.where(call, temp + (1 - dd * n_up) * si / q, temp - (1 - dd * n_down) * si / q)
        bi = np.where(call,
                      dd * n_up * (1 - 1 / q) + (1 - dd * _pdf(d1) / sv) / q,
                      -dd * n_down * (1 - 1 / q) - (1 + dd * _pdf(-d1) / sv) / q)
        return lhs, rhs, bi, ok

    lhs, rhs, bi, ok = state(si)
    ok &= ~(h > _MAX_SEED_EXPONENT)
    for _ in range(_MAX_CRITICAL_ITERATIONS):
        active = ok & (np.abs(lhs - rhs) / strike > _CRITICAL_TOLERANCE)
        if not active.any():
            break
        step = np.where(call, (strike + rhs - bi * si) / (1 - bi),
                        (strike - rhs + bi * si) / (1 + bi))
        si = np.where(active, step, si)
        lhs_n, rhs_n, bi_n, ok_n = state(si)
        lhs, rhs, bi = (np.where(active, a, b) for a, b in ((lhs_n, lhs), (rhs_n, rhs),
                                                             (bi_n, bi)))
        ok = np.where(active, ok_n, ok)
    converged = ok & ~(np.abs(lhs - rhs) / strike > _CRITICAL_TOLERANCE)
    return si, converged


def european_price(inp: BatchInputs, vol: FloatArray) -> FloatArray:
    """De Europese prijs zoals `AnalyticEuropeanEngine` (NaN voor een verlopen contract)."""
    with np.errstate(all="ignore"):
        m = _model(inp, vol)
        return _black_calculator(m.is_call, m.forward, m.strike, m.std_dev, m.r_disc)


def baw_price(inp: BatchInputs, vol: FloatArray) -> FloatArray:
    """De BAW-prijs zoals `BaroneAdesiWhaleyApproximationEngine`.

    NaN waar QuantLib faalt of de kernel QuantLib niet zeker volgt (zie de moduledocstring).
    """
    with np.errstate(all="ignore"):
        m = _model(inp, vol)
        black = _black_calculator(m.is_call, m.forward, m.strike, m.std_dev, m.r_disc)
        q = _q_exponent(m)
        sk, ok = _critical_price(m, q)
        d1 = (np.log(sk * m.d_disc / m.r_disc / m.strike) + 0.5 * m.variance) / m.std_dev
        a = np.where(m.is_call, (sk / q) * (1.0 - m.d_disc * _cdf(d1)),
                     -(sk / q) * (1.0 - m.d_disc * _cdf(-d1)))
        premium = black + a * np.power(m.spot / sk, q)
        call = np.where(m.spot < sk, premium, m.spot - m.strike)
        put = np.where(m.spot > sk, premium, m.strike - m.spot)
        american = np.where(ok, np.where(m.is_call, call, put), np.nan)
        never_early = np.where(m.is_call, m.d_disc >= 1.0, m.r_disc >= 1.0)
        value = np.where(never_early, black, american)
        # QuantLib weigert een call bij een negatieve rente (rD > 1).
        return np.where(m.is_call & (m.r_disc > 1.0), np.nan, value)


class KernelIVStatus(StrEnum):
    """De waarden van `IVStatus`, plus één eigen status: dit contract gaat naar QuantLib."""

    OK = "ok"
    NO_QUOTE = "no_quote"
    NO_BID = "no_bid"
    BELOW_INTRINSIC = "below_intrinsic"
    ABOVE_UPPER_BOUND = "above_upper_bound"
    NO_CONVERGENCE = "no_convergence"
    EXPIRED = "expired"
    TO_QUANTLIB = "to_quantlib"


_CODES = tuple(KernelIVStatus)
_CODE = {s: np.int8(i) for i, s in enumerate(_CODES)}


@dataclass(frozen=True, slots=True)
class BatchIV:
    """De IV per contract (NaN tenzij OK) en de status als code in `KernelIVStatus`."""

    value: FloatArray
    code: npt.NDArray[np.int8]

    @property
    def status(self) -> tuple[KernelIVStatus, ...]:
        return tuple(_CODES[c] for c in self.code)


def _solve(inp: BatchInputs, target: FloatArray, cfg: IVSettings
           ) -> tuple[FloatArray, npt.NDArray[np.int8]]:
    """Het bracket `[min_vol, max_vol]` zoals `implied_vol._solve`, dan secant/bisectie.

    Een NaN bij `min_vol` of `max_vol` gaat naar QuantLib (zie de moduledocstring).
    """
    n = target.size
    code = np.full(n, _CODE[KernelIVStatus.TO_QUANTLIB])
    sigma = np.full(n, np.nan)

    def f(vol: FloatArray) -> FloatArray:
        return baw_price(inp, vol) - target

    a, b = np.full(n, cfg.min_vol), np.full(n, cfg.max_vol)
    fa, fb = f(a), f(b)
    edge = _EDGE_FRACTION * inp.spot
    known = ~np.isnan(fa) & ~np.isnan(fb) & (np.abs(fa) > edge) & (np.abs(fb) > edge)
    out_of_range = known & ((fa > 0) | (fb < 0))
    code[out_of_range] = _CODE[KernelIVStatus.NO_CONVERGENCE]
    active = known & ~out_of_range

    # Bracket [a, b] met f(a) < 0 < f(b); (x1, f1) het laatste punt, (x0, f0) het vorige.
    x0, f0, x1, f1 = a, fa, b, fb
    bisect = np.zeros(n, dtype=np.bool_)
    for _ in range(_MAX_IV_ITERATIONS):
        if not active.any():
            break
        with np.errstate(all="ignore"):
            secant = x1 - f1 * (x1 - x0) / (f1 - f0)
        use_secant = ~bisect & (secant > a) & (secant < b)
        x = np.where(active, np.where(use_secant, secant, 0.5 * (a + b)), x1)
        fx = f(x)
        active &= ~np.isnan(fx)                  # QuantLib zou hier falen: naar QuantLib
        below, above = active & (fx < 0), active & (fx > 0)
        a, b = np.where(below, x, a), np.where(above, x, b)
        done = active & ((np.abs(x - x1) <= _IV_STEP_TOLERANCE) | (fx == 0.0)
                         | (b - a <= _IV_STEP_TOLERANCE))
        sigma = np.where(done, x, sigma)
        code[done] = _CODE[KernelIVStatus.OK]
        # Halveert |f| niet, dan is de volgende stap een bisectie.
        bisect = np.abs(fx) > 0.5 * np.abs(f1)
        active &= ~done
        x0, f0 = np.where(active, x1, x0), np.where(active, f1, f0)
        x1, f1 = np.where(active, x, x1), np.where(active, fx, f1)
    return sigma, code


def implied_vols(inp: BatchInputs, prices: FloatArray, cfg: IVSettings = IVSettings()
                 ) -> BatchIV:
    """De BAW-IV per prijs (NaN = geen prijs), met de statuslogica van `implied_vol`."""
    n = prices.size
    code = np.full(n, -1, dtype=np.int8)

    def mark(mask: BoolArray, status: KernelIVStatus) -> None:
        code[(code < 0) & mask] = _CODE[status]

    upper = np.where(inp.is_call, inp.spot, inp.strike)
    lower = np.maximum(np.where(inp.is_call, inp.spot - inp.strike, inp.strike - inp.spot), 0.0)
    with np.errstate(invalid="ignore"):
        mark(inp.expired, KernelIVStatus.EXPIRED)
        mark(np.isnan(prices), KernelIVStatus.NO_QUOTE)
        mark(prices <= 0, KernelIVStatus.NO_BID)
        mark(prices < lower - cfg.tick, KernelIVStatus.BELOW_INTRINSIC)
        mark(prices > upper, KernelIVStatus.ABOVE_UPPER_BOUND)
    target = np.where(prices <= lower, lower + 0.5 * cfg.tick, prices)
    todo = np.flatnonzero(code < 0)
    sigma, sub_code = _solve(inp.subset(todo), target[todo], cfg)
    value = np.full(n, np.nan)
    value[todo] = sigma
    code[todo] = sub_code
    return BatchIV(value, code)


@dataclass(frozen=True, slots=True)
class BatchGreeks:
    """Greeks per aandeel, in de eenheden van `tradebot.options.units` (zie `greeks`).

    NaN waar een van de herwaarderingen NaN is (zie `baw_price`); dan rekent de aanroeper
    die Greeks met QuantLib. Bij `r = 0` geldt dat voor de rho van een call: de bump omlaag
    geeft een negatieve rente, en daar weigert QuantLib ook.
    """

    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    rho: FloatArray


def baw_greeks(inp: BatchInputs, vol: FloatArray, bumps: Bumps = Bumps()) -> BatchGreeks:
    """De BAW-Greeks van `greeks(..., Method.BAW)` bij volatiliteit `vol`, per contract."""
    base = baw_price(inp, vol)
    h = bumps.spot_frac * inp.spot
    up = baw_price(replace(inp, spot=inp.spot + h), vol)
    down = baw_price(replace(inp, spot=inp.spot - h), vol)
    delta = (up - down) / (2 * h)
    gamma = (up - 2 * base + down) / (h * h)

    b = np.minimum(bumps.vol, 0.5 * vol)
    vega = (baw_price(inp, vol + b) - baw_price(inp, vol - b)) / (2 * b) * 0.01

    r = inp.rate
    rho = (baw_price(replace(inp, rate=r + bumps.rate), vol)
           - baw_price(replace(inp, rate=r - bumps.rate), vol)) / (2 * bumps.rate) * 0.01

    theta = (baw_price(inp.shifted_in_time(), vol) - base) / inp.theta_days
    return BatchGreeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
