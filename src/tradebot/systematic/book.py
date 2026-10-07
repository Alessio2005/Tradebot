"""Het boek: doelgewichten -> trades -> kosten -> funding -> netto, bar voor bar.

SCREENINGSINSTRUMENT. Elke uitkomst draagt `backtest/vectorized.py::NOT_ADMISSIBLE`;
promotiebewijs komt uit `backtest/engine.py`. Wat dit boek wel exact doet, en waarin
het verschilt van de eerdere onderzoeksengines:

* **Drift.** Een gewicht is een fractie van de equity. Na een bar is het gewicht
  `w (1 + r_i) / (1 + r_boek)`, en de trade naar het doel vertrekt daarvandaan. Wie de
  trade als `|w_t − w_{t−1}|` telt, vergeet het bijhandelen dat een constant gewicht
  na elke prijsbeweging vraagt.
* **De kostentrap.** Per bar: bruto prijs-P&L, funding (een long betaalt positieve
  funding), fees, spread, extra slippage en vierkantswortelimpact, met de sluitende
  identiteit `netto = (1 + bruto + funding)(1 − kosten) − 1`. Die identiteit wordt na
  elke run gecontroleerd; een post die nergens vandaan komt, crasht.
* **Timing.** `lag = 1`: het besluit op de close van *t* wordt tegen die close
  uitgevoerd (de markt is 24/7 en `open(t+1) == close(t)` in deze data) en rendeert
  vanaf *t+1*. `lag = 2` voert hetzelfde besluit pas een bar later uit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from ..execution.impact_model import ImpactParams
from ..utils.failfast import DataContractError, require
from .market import BookMarket

__all__ = ["BookResult", "CostSpec", "run_book"]

BPS = 1e-4
#: Een trade kleiner dan dit (fractie van de equity) is numeriek stof en wordt niet gedaan.
DUST = 1e-9


@dataclass(frozen=True)
class CostSpec:
    """De kosten per eenheid verhandelde notional, en de boekgrootte voor de impact."""

    taker_fee: float
    half_spread: float
    impact: ImpactParams | None
    aum_usd: float
    extra_slippage: float = 0.0
    multiplier: float = 1.0
    #: Vervangt `impact.eta` (de vierkantswortel-Y) zonder de herkomst van de
    #: impactparameters te verliezen; `None` = de geconfigureerde eta.
    impact_eta: float | None = None

    def __post_init__(self) -> None:
        for name in ("taker_fee", "half_spread", "extra_slippage", "multiplier"):
            require(float(getattr(self, name)) >= 0.0, f"Negatieve kostenparameter {name}.",
                    DataContractError, value=getattr(self, name))
        require(self.aum_usd > 0.0, "De boekgrootte moet positief zijn.", DataContractError)
        require(self.impact_eta is None or (self.impact is not None and self.impact_eta > 0.0),
                "Een impact-eta-override vraagt impactparameters en een positieve eta.",
                DataContractError, impact_eta=self.impact_eta)

    @property
    def eta(self) -> float | None:
        if self.impact is None:
            return None
        return float(self.impact.eta if self.impact_eta is None else self.impact_eta)

    def scaled(self, *, multiplier: float | None = None, extra_slippage: float | None = None,
               aum_usd: float | None = None, impact_eta: float | None = None) -> CostSpec:
        return CostSpec(
            taker_fee=self.taker_fee, half_spread=self.half_spread, impact=self.impact,
            aum_usd=self.aum_usd if aum_usd is None else float(aum_usd),
            extra_slippage=self.extra_slippage if extra_slippage is None else float(extra_slippage),
            multiplier=self.multiplier if multiplier is None else float(multiplier),
            impact_eta=self.impact_eta if impact_eta is None else float(impact_eta))

    def as_record(self) -> dict[str, Any]:
        return {
            "taker_fee_bps": self.taker_fee / BPS, "half_spread_bps": self.half_spread / BPS,
            "extra_slippage_bps": self.extra_slippage / BPS, "multiplier": self.multiplier,
            "aum_usd": self.aum_usd,
            "impact": None if self.impact is None else {
                "eta": self.eta, "configured_eta": self.impact.eta,
                "status": str(self.impact.status.value)},
        }


@dataclass(frozen=True)
class BookResult:
    """Per bar: de P&L-trap, de turnover en de gehouden gewichten."""

    frame: pd.DataFrame
    held: pd.DataFrame
    trades: pd.DataFrame
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def net(self) -> pd.Series:
        return self.frame["net"]

    def window(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> BookResult:
        """Alle reeksen op `[start, end]` (inclusief), zonder iets te herberekenen."""
        sl = slice(pd.Timestamp(start, tz="UTC") if isinstance(start, str) else start,
                   pd.Timestamp(end, tz="UTC") if isinstance(end, str) else end)
        return BookResult(frame=self.frame.loc[sl], held=self.held.loc[sl],
                          trades=self.trades.loc[sl], audit=dict(self.audit))


def run_book(
    target: pd.DataFrame,
    rebalance: pd.Series,
    market: BookMarket,
    costs: CostSpec,
    *,
    lag: int,
    half_spread: pd.DataFrame | None = None,
    exit_on_missing_price: bool = False,
    trade_rate: float = 1.0,
) -> BookResult:
    """Simuleer het boek. `target` rij *t* = besluit op de close van *t*; NaN = geen besluit.

    `half_spread` (fractie per munt per bar, besluitbar = uitvoeringsbar) vervangt de
    vaste `costs.half_spread`. `exit_on_missing_price=True` is de delisting-regel van v2:
    verdwijnt de koers van een gehouden munt, dan wordt de positie gesloten tegen de laatste
    bekende close (rendement 0 op die bar), met kosten tegen de laatst bekende ADV en σ.
    Zonder die vlag is een gehouden positie zonder koers een crash (v1).

    `trade_rate` κ ∈ (0, 1] is de partiële aanpassing van Gârleanu-Pedersen (2013): op
    een herbalanceringsbar wordt een fractie κ van het gat tussen het gedrifte en het
    doelgewicht verhandeld. κ = 1 (default) is volledige aanpassing, zoals v1 en v2."""
    require(int(lag) >= 1, "Een vertraging van nul bars is lookahead.", DataContractError,
            lag=lag)
    require(0.0 < float(trade_rate) <= 1.0, "trade_rate moet in (0, 1] liggen.",
            DataContractError, trade_rate=trade_rate)
    require(bool(target.index.equals(market.index)) and list(target.columns) == list(market.symbols),
            "Doelgewichten en markt delen geen raster of kolommen.", DataContractError)
    require(bool(rebalance.index.equals(market.index)),
            "Het herbalanceringsschema staat niet op het marktraster.", DataContractError)

    shift = int(lag) - 1
    tgt = target.shift(shift).to_numpy(dtype=np.float64)
    reb = rebalance.shift(shift, fill_value=False).to_numpy(dtype=bool)
    ret = market.ret.to_numpy(dtype=np.float64)
    fund = market.funding.to_numpy(dtype=np.float64)
    price_ok = market.close.notna().to_numpy()
    sigma = market.sigma_daily.to_numpy(dtype=np.float64)
    adv = market.adv_usd.to_numpy(dtype=np.float64)
    if exit_on_missing_price:
        # Laatst bekende σ en laatst bekende POSITIEVE ADV, voor de afwikkeling van een
        # verdwenen of bevroren munt. FTTUSDT na de FTX-val (nov 2022): constante prijs,
        # nul volume, wekenlang -- de beurs wikkelt zo'n markt af tegen de laatste prijs;
        # een impact op ADV = 0 is daar ongedefinieerd, niet oneindig.
        sigma = market.sigma_daily.ffill().to_numpy(dtype=np.float64)
        adv = market.adv_usd.where(market.adv_usd > 0.0).ffill().to_numpy(dtype=np.float64)
    if half_spread is not None:
        require(bool(half_spread.index.equals(market.index))
                and list(half_spread.columns) == list(market.symbols),
                "Het spreadpaneel staat niet op het marktraster.", DataContractError)
        hs = half_spread.ffill().fillna(costs.half_spread).to_numpy(dtype=np.float64)
        require(bool((hs >= 0.0).all()), "Negatieve spread.", DataContractError)

    t_len, n = tgt.shape
    held = np.zeros((t_len, n))
    trades = np.zeros((t_len, n))
    cols = ("gross", "funding", "fees", "spread", "slippage", "impact", "net", "turnover",
            "gross_leverage", "net_leverage", "n_trades")
    out = np.zeros((t_len, len(cols)))
    fee_rate = costs.taker_fee * costs.multiplier
    spread_rate = costs.half_spread * costs.multiplier
    slip_rate = costs.extra_slippage * costs.multiplier
    eta = costs.eta
    n_forced_exits = 0

    h = np.zeros(n)
    equity = float(costs.aum_usd)
    for t in range(t_len):
        held[t] = h
        r = ret[t]
        active = np.abs(h) > 0.0
        vanished = active & ~np.isfinite(r)
        require(exit_on_missing_price or not bool(vanished.any()),
                "Een gehouden positie zonder rendement op deze bar.", DataContractError,
                bar=str(market.index[t]))
        r0 = np.where(np.isfinite(r), r, 0.0)
        gross = float(h @ r0)
        funding = -float(h @ fund[t])
        port = gross + funding
        require(1.0 + port > 0.0, "Het boek is op deze bar volledig weggevaagd.",
                DataContractError, bar=str(market.index[t]))
        drifted = h * (1.0 + r0) / (1.0 + port)
        equity_pre = equity * (1.0 + port)

        new = drifted
        if reb[t] and np.isfinite(tgt[t]).any():
            desired = np.where(np.isfinite(tgt[t]), tgt[t], 0.0)
            if exit_on_missing_price:
                # Een eerder besluit (lag >= 2) in een munt die op de uitvoeringsbar geen
                # koers meer heeft, is niet uit te voeren: de exitregel hieronder sluit hem.
                desired = np.where(price_ok[t], desired, 0.0)
            require(bool((desired[~price_ok[t]] == 0.0).all()),
                    "Een doelgewicht in een munt zonder prijs.", DataContractError,
                    bar=str(market.index[t]))
            delta = float(trade_rate) * (desired - drifted)
            delta[np.abs(delta) < DUST] = 0.0
            new = drifted + delta
        gone = ~price_ok[t] & (np.abs(new) > 0.0)
        if exit_on_missing_price and gone.any():
            new = np.where(gone, 0.0, new)
            n_forced_exits += int(gone.sum())
        trade = new - drifted
        traded = np.abs(trade)
        fee = fee_rate * traded.sum()
        spread = (float((hs[t] * traded).sum()) * costs.multiplier if half_spread is not None
                  else spread_rate * traded.sum())
        slip = slip_rate * traded.sum()
        impact = 0.0
        if eta is not None and traded.any():
            idx = traded > 0.0
            require(bool(np.isfinite(adv[t][idx]).all() and (adv[t][idx] > 0.0).all()
                         and np.isfinite(sigma[t][idx]).all()),
                    "Impact zonder geldige ADV of sigma op een handelsbar.", DataContractError,
                    bar=str(market.index[t]))
            notional = traded[idx] * equity_pre
            frac = eta * sigma[t][idx] * np.sqrt(notional / adv[t][idx])
            impact = float((traded[idx] * frac).sum()) * costs.multiplier
        cost = fee + spread + slip + impact
        require(cost < 1.0, "Kosten van meer dan de hele equity op één bar.", DataContractError)
        net = (1.0 + port) * (1.0 - cost) - 1.0
        equity = equity_pre * (1.0 - cost)
        scale = 1.0 + port
        out[t] = (gross, funding, -fee * scale, -spread * scale, -slip * scale, -impact * scale,
                  net, traded.sum(), np.abs(new).sum(), new.sum(), float((traded > 0).sum()))
        trades[t] = trade
        h = new

    frame = pd.DataFrame(out, index=market.index, columns=cols)
    identity = (frame["gross"] + frame["funding"] + frame["fees"] + frame["spread"]
                + frame["slippage"] + frame["impact"] - frame["net"]).abs().max()
    require(float(identity) < 1e-12, "De P&L-trap sluit niet.", DataContractError,
            residual=float(identity))
    return BookResult(
        frame=frame,
        held=pd.DataFrame(held, index=market.index, columns=market.symbols),
        trades=pd.DataFrame(trades, index=market.index, columns=market.symbols),
        audit={EVIDENCE_KEY: NOT_ADMISSIBLE, "engine": "systematic.book", "lag": int(lag),
               "costs": costs.as_record(), "n_forced_exits": n_forced_exits,
               "half_spread": "panel" if half_spread is not None else "constant",
               "trade_rate": float(trade_rate),
               "convention": "besluit op close t, uitgevoerd tegen close t+lag-1, "
                             "rendeert vanaf de bar erna"})
