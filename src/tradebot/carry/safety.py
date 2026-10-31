"""Veiligheid van het Kraken-boek: harde grenzen die het model niet kan overrulen.

Het model (`systematic.harvest.basis_decide`) beslist WAT het boek wil. Dit module beslist of
dat mag, los van het model, met vaste grenzen uit de serverconfig (`conf/live/u1.yaml`).
Elke grens hier kan een besluit alleen tegenhouden of risico laten AFBOUWEN, nooit risico
opbouwen. Wat er gebeurt als een grens bindt:

* `SafetyBreach` (harde grens, model- of configfout): het boek gaat stil, een mens kijkt.
  Kapitaal boven het plafond, een order boven de maximale grootte, bruto boven het plafond,
  een stap met meer verlies dan een gehedged boek kan maken.
* `Deferred` (de wereld klopt even niet): de stap gebeurt niet, de dienst probeert het bij
  de volgende ronde opnieuw. Data die niet vers is, een live koers die te ver van de
  besluitkoers ligt.
* Een hedge-gat (spot en perp van een munt lopen uiteen, bv. na een half uitgevoerde
  herbalancering): `repair_orders` sluit het OVERSCHOT van het grootste been, reduce-only.
  Nooit wordt het kleinere been aangevuld: dat zou risico opbouwen op een moment dat er al
  iets mis is.

`SingleInstance` houdt een tweede proces buiten de deur (een cron die overlapt, een
herstart terwijl de vorige nog draait): twee processen met dezelfde sleutel breken de nonce
van Kraken en kunnen dubbel handelen.
"""
from __future__ import annotations

import fcntl
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import TracebackType
from typing import Any

from ..utils.failfast import ConfigContractError, TradebotContractError, require
from .orders import LegOrder, unwind_orders
from .venue import VenueBook

__all__ = ["Deferred", "Limits", "SafetyBreach", "SingleInstance", "check_orders",
           "check_prices", "gross_after", "hedge_gaps", "load_limits", "margin_headroom",
           "marked_pnl", "repair_orders"]


class SafetyBreach(TradebotContractError):
    """Een harde grens bindt: het boek gaat stil tot een mens het onderzocht heeft."""


class Deferred(TradebotContractError):
    """De stap kan nu niet veilig; later opnieuw proberen (geen stilstand)."""


@dataclass(frozen=True)
class Limits:
    """De harde grenzen. Elke waarde is een plafond of een tolerantie, nooit een doel."""

    #: Het hoogste boekkapitaal dat de dienst accepteert (beschermt tegen een typo).
    max_capital_usd: float
    #: De grootste enkele order (USD-notional).
    max_order_usd: float
    #: (spot + perp) / kapitaal na de stap. H3 houdt maximaal 0,70 + 0,70 = 1,4.
    max_gross: float
    #: Hoe ver de live koers van de besluitkoers mag liggen voordat de stap wacht.
    max_price_deviation: float
    #: Een hedge-gat (|spot − perp| x koers) groter dan dit is een gat.
    hedge_tolerance_usd: float
    #: Het grootste verlies van de gehedgede posities tussen twee stappen, fractie van het
    #: kapitaal. Een gehedged boek beweegt met de basis (promilles), niet met de prijs.
    max_step_loss: float
    #: Marge-headroom = equity op de beurs / (perp-notional x initiële marge).
    margin_warn: float
    margin_critical: float
    #: Initiële marge van een perp (Kraken EU retail: 1 / 2:1).
    initial_margin: float = 0.5
    #: Toegestane klokafwijking tegen de servertijd van Kraken (seconden).
    max_clock_skew_s: float = 5.0

    def __post_init__(self) -> None:
        for f in fields(self):
            v = float(getattr(self, f.name))
            require(math.isfinite(v) and v > 0.0, "Een grens moet eindig en positief zijn.",
                    ConfigContractError, field=f.name, value=v)
        require(self.margin_critical < self.margin_warn, "De kritieke marge ligt onder de "
                "waarschuwing.", ConfigContractError)
        require(self.max_price_deviation < 0.5 and self.max_step_loss < 0.5,
                "Een tolerantie van 50 % of meer is geen grens.", ConfigContractError)

    @staticmethod
    def from_mapping(raw: Mapping[str, Any]) -> Limits:
        known = {f.name for f in fields(Limits)}
        unknown = sorted(set(raw) - known)
        require(not unknown, "Onbekende grenzen in de config.", ConfigContractError,
                unknown=unknown)
        return Limits(**{k: float(v) for k, v in raw.items()})


def load_limits(path: Path) -> Limits:
    """De grenzen uit een serverconfig (`limits:`)."""
    import yaml  # harde dependency (requirements.lock)

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    require(isinstance(raw.get("limits"), Mapping), "Een serverconfig zonder `limits:`.",
            ConfigContractError, path=str(path))
    return Limits.from_mapping(raw["limits"])


def check_orders(orders: Sequence[LegOrder], limits: Limits, capital_usd: float) -> list[str]:
    """De harde grenzen op een besluit. Geeft de overtredingen (leeg = in orde)."""
    out = []
    if capital_usd > limits.max_capital_usd:
        out.append(f"kapitaal {capital_usd:,.0f} boven het plafond {limits.max_capital_usd:,.0f}")
    for o in orders:
        if o.notional_usd > limits.max_order_usd:
            out.append(f"{o.symbol} {o.leg.value} {o.notional_usd:,.0f} USD boven de "
                       f"maximale order {limits.max_order_usd:,.0f}")
    return out


def gross_after(spot_frac: Sequence[float], perp_frac: Sequence[float]) -> float:
    return float(sum(spot_frac) + sum(perp_frac))


def check_prices(decision: Mapping[str, float], live: Mapping[str, float],
                 limits: Limits) -> list[str]:
    """De besluitkoers (close) tegen de live koers, per munt waarin gehandeld wordt."""
    out = []
    for sym, p in decision.items():
        q = live.get(sym)
        if q is None or not (math.isfinite(q) and q > 0.0):
            out.append(f"{sym}: geen live koers")
        elif abs(q / p - 1.0) > limits.max_price_deviation:
            out.append(f"{sym}: live {q:,.2f} tegen besluit {p:,.2f} "
                       f"({q / p - 1.0:+.1%})")
    return out


def hedge_gaps(book: VenueBook, prices: Mapping[str, float], tol_usd: float) -> dict[str, float]:
    """Per munt: spot − perp-short in munten, waar het verschil meer dan `tol_usd` waard is."""
    gaps = {}
    for sym in sorted(set(book.spot) | set(book.perp_short)):
        gap = float(book.spot.get(sym, 0.0)) - float(book.perp_short.get(sym, 0.0))
        px = prices.get(sym, math.nan)
        if not math.isfinite(px) or abs(gap) * px > tol_usd:
            gaps[sym] = gap
    return gaps


def repair_orders(gaps: Mapping[str, float], spot_price: Mapping[str, float],
                  perp_price: Mapping[str, float]) -> list[LegOrder]:
    """Sluit het overschot van het grootste been, reduce-only (zie de moduledocstring)."""
    spot = {s: g for s, g in gaps.items() if g > 0.0}       # te veel spot: verkopen
    perp = {s: -g for s, g in gaps.items() if g < 0.0}      # te veel short: terugkopen
    return unwind_orders(spot, perp, spot_price, perp_price)


def margin_headroom(book: VenueBook, perp_price: Mapping[str, float],
                    initial_margin: float) -> float:
    """Equity op de beurs gedeeld door de initiële marge van de shorts (inf zonder short)."""
    need = sum(q * perp_price[s] for s, q in book.perp_short.items()) * initial_margin
    return math.inf if need <= 0.0 else float(book.equity_usd) / need


def marked_pnl(spot: Mapping[str, float], perp: Mapping[str, float],
               then: Mapping[str, tuple[float, float]],
               now: Mapping[str, tuple[float, float]]) -> float:
    """De P&L van gehouden hoeveelheden tussen twee markeringen (spot, perp) per munt.

    Zonder funding en fees: dat is wat een gehedged boek met de PRIJS doet, en dat hoort
    bijna nul te zijn. Overboekingen van en naar de beurs tellen niet mee (alleen posities)."""
    pnl = 0.0
    for sym in set(spot) | set(perp):
        if sym not in then or sym not in now:
            continue
        (s0, p0), (s1, p1) = then[sym], now[sym]
        pnl += float(spot.get(sym, 0.0)) * (s1 - s0) - float(perp.get(sym, 0.0)) * (p1 - p0)
    return pnl


class SingleInstance:
    """Een exclusieve lock op een bestand (fcntl): een tweede proces krijgt `Deferred`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def __enter__(self) -> SingleInstance:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise Deferred(f"Een ander proces houdt {self.path}; deze ronde slaat over.") \
                from None
        self._fd = fd
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        assert self._fd is not None
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

