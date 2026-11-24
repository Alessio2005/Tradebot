"""De QuantLib-omgeving: globale toestand afschermen en de looptijd op seconden.

`ql.Settings.instance().evaluationDate` is GLOBALE toestand. Wie hem zet en niet terugzet,
verandert stilletjes elke volgende waardering in hetzelfde proces. Hij wordt hier daarom
uitsluitend gezet via `evaluation_date`, die de vorige waarde terugzet, ook na een
uitzondering. Er wordt niet parallel geprijsd in threads binnen één proces.

LOOPTIJD OP SECONDEN
--------------------
QuantLib rekent in hele dagen vanaf `evaluationDate`. De echte looptijd loopt tot 16:00 New
York op de expiratiedag (`calendar.expiry_moment`). Het verschil wordt verrekend door de
modelparameters te schalen met `f = T_echt / T_model`:

    σ_model = σ × sqrt(f),   r_model = r × f,   q_model = q × f

Zo zijn de totale variantie (σ²T) en de discontering (rT, qT) exact gelijk aan die over de
echte looptijd. Voor Black-Scholes is dat exact; voor Amerikaanse uitoefening is het een
goede benadering. Op de expiratiedag zelf zou QuantLib een verlopen optie zien; daar is
`evaluationDate` de dag ervoor, met dezelfde schaling.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import QuantLib as ql

from ...utils.failfast import DataContractError, require
from ..calendar import NY, expiry_moment, year_fraction

__all__ = ["DAY_COUNT", "ScaledTime", "evaluation_date", "make_process", "make_vol_process",
           "scaled_time", "to_ql_date"]

DAY_COUNT = ql.Actual365Fixed()
_CAL = ql.NullCalendar()


def to_ql_date(d: date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


@contextmanager
def evaluation_date(d: date) -> Iterator[ql.Date]:
    """Zet `evaluationDate` op `d` binnen het blok en zet daarna de vorige waarde terug."""
    settings = ql.Settings.instance()
    previous = settings.evaluationDate
    settings.evaluationDate = to_ql_date(d)
    try:
        yield settings.evaluationDate
    finally:
        settings.evaluationDate = previous


@dataclass(frozen=True, slots=True)
class ScaledTime:
    """De waarderingsdatum voor QuantLib en de twee looptijden (in jaren)."""

    eval_date: date
    t_real: float
    t_model: float

    @property
    def factor(self) -> float:
        return self.t_real / self.t_model


def scaled_time(now: datetime, expiry: date) -> ScaledTime:
    """Looptijd tot het expiratiemoment, en de QuantLib-datum waarop gewaardeerd wordt."""
    t_real = year_fraction(now, expiry_moment(expiry))
    require(t_real > 0, "de optie is verlopen", DataContractError, now=now.isoformat(),
            expiry=expiry.isoformat())
    eval_date = min(now.astimezone(NY).date(), expiry - timedelta(days=1))
    t_model = (expiry - eval_date).days / 365.0
    return ScaledTime(eval_date, t_real, t_model)


def make_process(spot: float, rate: float, div_yield: float, vol: float,
                 st: ScaledTime) -> ql.BlackScholesMertonProcess:
    """Een BSM-proces met geschaalde rente, yield en vol. Alleen binnen `evaluation_date`."""
    require(vol > 0, "volatiliteit moet positief zijn", DataContractError, vol=vol)
    process, set_vol = make_vol_process(spot, rate, div_yield, st)
    set_vol(vol)
    return process


def make_vol_process(spot: float, rate: float, div_yield: float, st: ScaledTime
                     ) -> tuple[ql.BlackScholesMertonProcess, Callable[[float], None]]:
    """Een BSM-proces waarvan de vol achteraf te zetten is (voor de IV-solver).

    Geeft het proces en een functie die de (ongeschaalde) vol zet. Alleen binnen
    `evaluation_date`.
    """
    require(spot > 0, "spot moet positief zijn", DataContractError, spot=spot)
    f = st.factor
    today = to_ql_date(st.eval_date)
    r = ql.YieldTermStructureHandle(ql.FlatForward(today, rate * f, DAY_COUNT))
    q = ql.YieldTermStructureHandle(ql.FlatForward(today, div_yield * f, DAY_COUNT))
    vol_quote = ql.SimpleQuote(0.2)
    v = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, _CAL, ql.QuoteHandle(vol_quote), DAY_COUNT))
    process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(spot)), q, r, v)
    scale = math.sqrt(f)

    def set_vol(vol: float) -> None:
        vol_quote.setValue(vol * scale)

    return process, set_vol
