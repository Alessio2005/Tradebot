# src/tradebot/backtest/accounting.py
"""Dubbele boekhouding voor de authoritative engine — L10.

Phase 5, deliverable 2. Gebouwd VÓÓR de engine, omdat de fase-opdracht §10 dat
voorschrijft en omdat de volgorde ertoe doet: een engine die eerst rendementen
optelt en pas daarna een balans probeert te construeren, kan geen kosten meer
terugvinden die hij onderweg is kwijtgeraakt.

WAT HIER ANDERS IS DAN IN DE VIER LEGACY-ENGINES
------------------------------------------------
Geen van hen heeft een kasregister. Alle vier vermenigvuldigen een equity-getal
met `(1 + r)`:

    equity *= (1 + port_return - bar_costs)      # backtest/portfolio.py:474

Dat is geen boekhouding maar een rendementsreeks met een equity-label. Er is
geen positie in stuks, geen kas, geen tegenrekening, en dus ook geen manier om
te CONTROLEREN of een geboekte kostenpost ergens vandaan kwam. Audit §16.1 eist
letterlijk *"Explicit cash/position accounting registers"*; dit is die eis.

DE TWEE INVARIANTEN
-------------------
Beide worden na elke mutatie gecontroleerd, niet aangenomen.

1. **Balansidentiteit.**  `assets == liabilities + equity`

   Een perp wordt hier als VOLLEDIG GEFINANCIERD geboekt: een long van `Q` tegen
   `P` haalt `Q*P` uit de kas en zet `Q*P` in de positiewaarde. Dat is niet hoe
   margin werkt, en dat is bewust: margin verandert WELK deel van het vermogen
   vaststaat, niet HOEVEEL vermogen er is. Een gefinancierde weergave sluit
   exact, en de hefboomvraag is al beantwoord door L7 (`gross_cap`) voordat er
   een order bestaat. Wie later echte margin wil boeken, voegt een
   `margin_requirement`-rekening toe zonder deze identiteit aan te raken.

   Bij een short is de positiewaarde negatief; die negatieve waarde is de
   VERPLICHTING om terug te kopen, en verschijnt daarom aan de passiefzijde:

       assets      = cash + max(position_value, 0)
       liabilities = max(-position_value, 0)
       equity      = assets - liabilities = cash + position_value

2. **PnL-attributie.**  Elke euro verschil met de startequity is toegewezen:

       equity - initial_equity
           == realized_pnl + unrealized_pnl - fees_paid - funding_paid

   Dit is de invariant die "geen verdwenen kosten" (§13) afdwingbaar maakt. De
   TCA-roundtrip sluit hierop aan: als de som van de gedecomponeerde kosten niet
   gelijk is aan `fees_paid + funding_paid` plus het prijsverschil, is er een
   kostenpost die in geen van beide administraties staat.

WAAROM DIT GEEN TOLERANTIEPARAMETER IN CONFIGURATIE IS
------------------------------------------------------
`_TOL` compenseert uitsluitend float-afronding over opeenvolgende optellingen en
schaalt mee met de omvang van het boek. Het is geen beleidsdrempel en hoort
daarom niet in `conf/`: een boekhouding die pas sluit als je de tolerantie
ophoogt, sluit niet.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 15, 16.1, 19 (L10), 26.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, TradebotContractError, require

__all__ = [
    "AccountingError",
    "Fill",
    "Ledger",
    "LedgerSnapshot",
    "Position",
]


class AccountingError(TradebotContractError):
    """De boekhouding sluit niet.

    Nooit op te vangen. Een backtest waarvan de balans niet sluit, rapporteert
    een rendement dat nergens vandaan komt; dat is ernstiger dan een crash.
    """


#: Relatieve tolerantie op de invarianten. Zie de moduledocstring: dit is
#: float-ruis, geen beleid. Geschaald met de boekomvang zodat hij bij een
#: equity van 1e9 niet strenger is dan bij 1e5.
_REL_TOL = 1e-9
#: Absolute vloer voor boeken die naar nul lopen.
_ABS_TOL = 1e-6

#: Onder deze hoeveelheid geldt een positie als gesloten. Voorkomt dat
#: float-residu van 1e-18 stuks een symbool eeuwig in het register houdt.
_DUST_QTY = 1e-12

Liquidity = Literal["maker", "taker"]


@dataclass(frozen=True, slots=True)
class Fill:
    """Eén uitgevoerde (deel)transactie. De enige manier om posities te muteren.

    `qty` is GETEKEND: positief koopt, negatief verkoopt. Een fill zonder
    richting bestaat niet, en een fill met hoeveelheid nul is geen fill.

    `fee` is ABSOLUUT en niet-negatief, in quote-valuta. Hij wordt apart
    geadministreerd en nooit in `price` verwerkt: een fee die in de prijs is
    verstopt, is een fee die de TCA-decompositie niet kan terugvinden.
    """

    symbol: str
    ts: pd.Timestamp
    qty: float
    price: float
    fee: float
    liquidity: Liquidity
    order_id: str

    def __post_init__(self) -> None:
        require(
            isinstance(self.ts, pd.Timestamp) and self.ts.tz is not None,
            "Een fill zonder tijdzone-bewuste timestamp is niet causaal te "
            "controleren.",
            DataContractError,
            symbol=self.symbol, ts=str(self.ts),
        )
        require(
            np.isfinite(self.qty) and abs(self.qty) > _DUST_QTY,
            "Een fill met hoeveelheid nul of niet-eindig is geen fill.",
            DataContractError,
            symbol=self.symbol, qty=self.qty,
        )
        require(
            np.isfinite(self.price) and self.price > 0.0,
            "Een fill tegen een niet-positieve prijs.",
            DataContractError,
            symbol=self.symbol, price=self.price,
        )
        require(
            np.isfinite(self.fee) and self.fee >= 0.0,
            "Een negatieve fee is een rebate en moet als zodanig worden "
            "geboekt, niet als negatieve kosten.",
            DataContractError,
            symbol=self.symbol, fee=self.fee,
        )

    @property
    def notional(self) -> float:
        """Getekende notional: `qty * price`."""
        return float(self.qty) * float(self.price)

    def as_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts.isoformat(),
            "qty": float(self.qty),
            "price": float(self.price),
            "fee": float(self.fee),
            "liquidity": self.liquidity,
            "order_id": self.order_id,
        }


@dataclass(frozen=True, slots=True)
class Position:
    """Een positie in stuks, met de gewogen gemiddelde instapprijs.

    `avg_price` is de basis waartegen gerealiseerde winst wordt gemeten. Bij het
    VERGROTEN van een positie schuift hij mee; bij het VERKLEINEN niet — dan
    wordt het verschil gerealiseerd. Dat is de standaard gewogen-gemiddelde
    methode, en zij is hier gekozen boven FIFO omdat een perp geen partijen
    kent: er is één doorlopende positie per symbool.
    """

    qty: float
    avg_price: float

    def market_value(self, mark: float) -> float:
        return float(self.qty) * float(mark)

    def unrealized(self, mark: float) -> float:
        return float(self.qty) * (float(mark) - float(self.avg_price))


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """De volledige boekstaat op één moment. Serialiseerbaar, auditbaar."""

    ts: pd.Timestamp
    cash: float
    position_value: float
    assets: float
    liabilities: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    funding_paid: float
    gross_notional: float
    net_notional: float

    def as_record(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "cash": float(self.cash),
            "position_value": float(self.position_value),
            "assets": float(self.assets),
            "liabilities": float(self.liabilities),
            "equity": float(self.equity),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "fees_paid": float(self.fees_paid),
            "funding_paid": float(self.funding_paid),
            "gross_notional": float(self.gross_notional),
            "net_notional": float(self.net_notional),
        }


class Ledger:
    """Kas- en positieregister met dubbele boekhouding.

    Muteert uitsluitend via `apply_fill`, `apply_funding` en `mark`. Er is geen
    setter voor equity: dat getal is een AFGELEIDE en nooit een invoer. Dat is
    het verschil met `portfolio/legacy_sizing.py`, waar `self.equity` publiek
    muteerbaar was en de balans dus per definitie niet te controleren viel.

    `__slots__` maakt die eigenschap AFDWINGBAAR in plaats van afgesproken.
    Zonder slots is `ledger.equity = 1.0` gewoon een nieuwe attribuutbinding die
    de methode overschaduwt, en dan is "equity is afgeleid" een commentaarregel
    in plaats van een garantie. De unit-test die dit controleert, faalde
    aanvankelijk; dit is wat hem groen maakte.
    """

    __slots__ = (
        "_cash",
        "_fees",
        "_fills",
        "_funding",
        "_initial_equity",
        "_marks",
        "_positions",
        "_realized",
    )

    def __init__(self, initial_cash: float) -> None:
        require(
            np.isfinite(initial_cash) and initial_cash > 0.0,
            "Een boek begint met strikt positieve kas.",
            DataContractError,
            initial_cash=initial_cash,
        )
        self._initial_equity = float(initial_cash)
        self._cash = float(initial_cash)
        self._positions: dict[str, Position] = {}
        self._realized = 0.0
        self._fees = 0.0
        self._funding = 0.0
        #: Laatste bekende markprijs per symbool. Alleen gezet door `mark()`;
        #: `apply_fill` raakt hem NIET aan, want een fill is geen markprijs.
        self._marks: dict[str, float] = {}
        self._fills: list[Fill] = []

    # ------------------------------------------------------------------ #
    # Uitlezen
    # ------------------------------------------------------------------ #
    @property
    def initial_equity(self) -> float:
        return self._initial_equity

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def realized_pnl(self) -> float:
        return self._realized

    @property
    def fees_paid(self) -> float:
        return self._fees

    @property
    def funding_paid(self) -> float:
        return self._funding

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    def position(self, symbol: str) -> Position:
        return self._positions.get(str(symbol), Position(0.0, 0.0))

    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def unrealized_pnl(self, marks: Mapping[str, float] | None = None) -> float:
        m = self._resolve_marks(marks)
        return float(sum(p.unrealized(m[s]) for s, p in self._positions.items()))

    def position_value(self, marks: Mapping[str, float] | None = None) -> float:
        m = self._resolve_marks(marks)
        return float(sum(p.market_value(m[s]) for s, p in self._positions.items()))

    def equity(self, marks: Mapping[str, float] | None = None) -> float:
        return float(self._cash + self.position_value(marks))

    def _resolve_marks(self, marks: Mapping[str, float] | None) -> dict[str, float]:
        """Markprijzen voor elk symbool met een open positie, of een crash.

        Er is bewust GEEN terugval op de laatst bekende prijs wanneer een
        symbool met een open positie geen markprijs heeft. Een positie
        waarderen tegen een verouderde prijs is exact de stille degradatie die
        `docs/DEFERRED_ISSUES.md` en fase-opdracht §23 uitsluiten.
        """
        source = dict(self._marks) if marks is None else dict(marks)
        out: dict[str, float] = {}
        for symbol in self._positions:
            raw = source.get(symbol)
            price = float("nan") if raw is None else float(raw)
            require(
                np.isfinite(price) and price > 0.0,
                "Ontbrekende of ongeldige markprijs voor een symbool met een "
                "open positie. De boekhouding waardeert niet tegen een "
                "verouderde of aangenomen prijs.",
                DataContractError,
                symbol=symbol,
                price=price,
                missing=raw is None,
            )
            out[symbol] = price
        return out

    # ------------------------------------------------------------------ #
    # Muteren
    # ------------------------------------------------------------------ #
    def apply_fill(self, fill: Fill) -> float:
        """Boek één fill. Geeft de gerealiseerde winst van DEZE fill terug.

        De vier gevallen zijn expliciet uitgeschreven in plaats van in één
        formule gecomprimeerd, omdat de flip (van long naar short in één fill)
        de enige is waarin zowel realisatie als heropening plaatsvindt, en juist
        die is in `portfolio/legacy_sizing.py` nooit gemodelleerd — daar bestond
        `side` alleen als +1/-1/0 per bar.
        """
        require(
            isinstance(fill, Fill),
            "apply_fill verwacht een Fill.",
            DataContractError,
            got=type(fill).__name__,
        )
        symbol = str(fill.symbol)
        old = self._positions.get(symbol, Position(0.0, 0.0))
        q_old, p_avg = float(old.qty), float(old.avg_price)
        q_fill, price = float(fill.qty), float(fill.price)
        q_new = q_old + q_fill

        realized = 0.0
        if abs(q_old) <= _DUST_QTY:
            # 1. Openen vanuit vlak.
            new_avg = price
        elif (q_old > 0.0) == (q_fill > 0.0):
            # 2. Vergroten in dezelfde richting: gewogen gemiddelde.
            new_avg = (q_old * p_avg + q_fill * price) / q_new
        elif abs(q_fill) <= abs(q_old) + _DUST_QTY:
            # 3. Verkleinen of exact sluiten: realiseer op het gesloten deel.
            closed = min(abs(q_fill), abs(q_old))
            realized = closed * (price - p_avg) * (1.0 if q_old > 0.0 else -1.0)
            new_avg = p_avg if abs(q_new) > _DUST_QTY else 0.0
        else:
            # 4. Flip: sluit de hele oude positie, open de rest tegen `price`.
            realized = abs(q_old) * (price - p_avg) * (1.0 if q_old > 0.0 else -1.0)
            new_avg = price

        if abs(q_new) <= _DUST_QTY:
            q_new, new_avg = 0.0, 0.0
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(q_new, float(new_avg))

        # Dubbele boeking: de kas betaalt de notional en de fee; de
        # positiewaarde neemt de notional op. De gerealiseerde winst is het
        # verschil tussen de oude boekwaarde en de opbrengst en verschijnt
        # daarom NIET als aparte kasmutatie - hij zit al in de notional.
        self._cash -= fill.notional
        self._cash -= float(fill.fee)
        self._fees += float(fill.fee)
        self._realized += float(realized)
        self._fills.append(fill)
        return float(realized)

    def apply_funding(
        self, symbol: str, *, rate: float, mark_price: float, ts: pd.Timestamp
    ) -> float:
        """Boek één funding-afrekening. Geeft het betaalde bedrag terug (>0 = betaald).

        Conventie: bij een positieve rate betaalt de LONG. Het bedrag is
        `qty * mark_price * rate`; bij een short is `qty` negatief en is het
        bedrag dus negatief, oftewel ontvangen.

        `rate` wordt hier NIET afgekapt. `backtest/portfolio.py:408` klemde hem
        op +/-2 % met de motivering dat de exchange dat ook doet; die cap is een
        eigenschap van de VENUE en hoort daarom in de executielaag te worden
        toegepast, niet in de boekhouding. Een boekhouding die zijn invoer
        corrigeert, verbergt waar de correctie vandaan kwam.
        """
        require(
            np.isfinite(rate),
            "Niet-eindige funding rate.",
            DataContractError, symbol=str(symbol), rate=rate,
        )
        require(
            np.isfinite(mark_price) and mark_price > 0.0,
            "Funding tegen een niet-positieve markprijs.",
            DataContractError, symbol=str(symbol), mark_price=mark_price,
        )
        require(
            isinstance(ts, pd.Timestamp) and ts.tz is not None,
            "Funding zonder tijdzone-bewuste timestamp.",
            DataContractError, symbol=str(symbol), ts=str(ts),
        )
        pos = self._positions.get(str(symbol))
        if pos is None:
            return 0.0
        paid = float(pos.qty) * float(mark_price) * float(rate)
        self._cash -= paid
        self._funding += paid
        return paid

    def mark(self, prices: Mapping[str, float], ts: pd.Timestamp) -> LedgerSnapshot:
        """Waardeer het boek en controleer beide invarianten. Crasht bij afwijking.

        Dit is de enige plaats waar `_marks` wordt bijgewerkt, en dat gebeurt
        met de VOLLEDIGE mapping die de caller aanlevert - ook voor symbolen
        zonder positie, zodat een latere fill in dat symbool tegen een actuele
        prijs kan worden gewaardeerd.
        """
        require(
            isinstance(ts, pd.Timestamp) and ts.tz is not None,
            "mark() zonder tijdzone-bewuste timestamp.",
            DataContractError, ts=str(ts),
        )
        for symbol, price in prices.items():
            p = float(price)
            if np.isfinite(p) and p > 0.0:
                self._marks[str(symbol)] = p
        marks = self._resolve_marks(None)

        position_value = float(
            sum(p.market_value(marks[s]) for s, p in self._positions.items())
        )
        unrealized = float(
            sum(p.unrealized(marks[s]) for s, p in self._positions.items())
        )
        assets = self._cash + max(position_value, 0.0)
        liabilities = max(-position_value, 0.0)
        equity = assets - liabilities

        snapshot = LedgerSnapshot(
            ts=ts,
            cash=self._cash,
            position_value=position_value,
            assets=assets,
            liabilities=liabilities,
            equity=equity,
            realized_pnl=self._realized,
            unrealized_pnl=unrealized,
            fees_paid=self._fees,
            funding_paid=self._funding,
            gross_notional=float(
                sum(abs(p.market_value(marks[s])) for s, p in self._positions.items())
            ),
            net_notional=position_value,
        )
        self.verify(snapshot)
        return snapshot

    # ------------------------------------------------------------------ #
    # De controle
    # ------------------------------------------------------------------ #
    def verify(self, snapshot: LedgerSnapshot) -> None:
        """Controleer de twee invarianten. Een afwijking is een harde failure."""
        scale = max(
            abs(snapshot.equity), abs(snapshot.assets), self._initial_equity, 1.0
        )
        tol = max(_ABS_TOL, _REL_TOL * scale)

        balance_gap = snapshot.assets - (snapshot.liabilities + snapshot.equity)
        require(
            abs(balance_gap) <= tol,
            "De balans sluit niet: assets != liabilities + equity. Een backtest "
            "waarvan de balans niet sluit, rapporteert rendement dat nergens "
            "vandaan komt.",
            AccountingError,
            ts=snapshot.ts.isoformat(),
            assets=snapshot.assets,
            liabilities=snapshot.liabilities,
            equity=snapshot.equity,
            gap=balance_gap,
            tolerance=tol,
        )

        attributed = (
            snapshot.realized_pnl
            + snapshot.unrealized_pnl
            - snapshot.fees_paid
            - snapshot.funding_paid
        )
        pnl_gap = (snapshot.equity - self._initial_equity) - attributed
        require(
            abs(pnl_gap) <= tol,
            "De PnL-attributie sluit niet: het verschil met de startequity is "
            "niet volledig toegewezen aan realized, unrealized, fees en "
            "funding. Er is een kostenpost of opbrengst zonder tegenrekening.",
            AccountingError,
            ts=snapshot.ts.isoformat(),
            equity_change=snapshot.equity - self._initial_equity,
            attributed=attributed,
            realized_pnl=snapshot.realized_pnl,
            unrealized_pnl=snapshot.unrealized_pnl,
            fees_paid=snapshot.fees_paid,
            funding_paid=snapshot.funding_paid,
            gap=pnl_gap,
            tolerance=tol,
        )
