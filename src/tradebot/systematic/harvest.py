"""Robuust boek v5: spot-perp-basiscarry -- long spot, short perp, de funding oogsten.

Ontwerp: `docs/superpowers/specs/2026-10-07-robust-book-v5-basis-design.md`.

WAAROM DIT EEN ANDERE BRON VAN RENDEMENT IS
===========================================
v2-v4 lieten zien wat er gebeurt als je funding cross-sectioneel oogst: long de munten
met lage funding, short die met hoge. Op de holdout verdiende dat boek 22,9 % per jaar
aan funding en verloor het meer op de prijs -- de short-kant zat in munten die bleven
stijgen. Een prijsweddenschap met een carry-label.

Hier is de prijs weggehedged: per munt dezelfde notional long op spot en short op de
perp. Wat overblijft is de funding die de short ontvangt, min de verandering van de basis
(perp/spot − 1) en de kosten. Dat is een vergoeding voor het leveren van hefboom aan
longs (He, Manela, Ross & von Wachter 2022; Schmeling, Schrimpf & Todorov 2023), geen
voorspelling. Ze kan verdwijnen (meer arbitragekapitaal, bear markets), maar ze hangt
niet af van het raden van een richting.

WAT HET BOEK EXACT DOET
=======================
* **Twee benen per munt.** `a` = spot-notional / equity (long), `b` = perp-notional /
  equity (short). Over bar *t*: spot-P&L `a·r_spot`, perp-P&L `−b·r_perp`, funding
  `+b·f` (een short ontvangt positieve funding). Daarna drijven beide benen met hun eigen
  prijs; gelijke hoeveelheden blijven gelijk, dus de hedge blijft staan zonder te handelen.
* **Band.** Een gehouden munt wordt alleen naar zijn doel teruggezet als de notional meer
  dan `band` (relatief) afwijkt, of als de twee benen meer dan `hedge_tolerance`
  uiteenlopen. Constante gewichten najagen zou elke dag beide benen verhandelen.
* **Kosten per been.** Spot: spot-fee + spot-halve-spread + impact op spot-ADV en
  spot-σ. Perp: perp-fee + perp-halve-spread + impact op perp-ADV en perp-σ. Met de
  identiteit `netto = (1 + bruto + funding + liquidatie)(1 − kosten) − 1`.
* **Marge en liquidatie.** Kapitaalmodel zonder portfolio margin: de spot staat in de
  spotwallet, de rest van de equity (`1 − Σa`) is USDT-marge in de futureswallet (cross).
  Een rally laat de spot stijgen en de wallet leeglopen. Zakt de wallet onder
  `margin_floor` × de perp-notional, dan gaat het HELE boek terug naar het doel (spot
  verkopen, wallet aanvullen). Binnen een bar is het ergste geval dat elke short
  tegelijk op zijn dag-high staat: raakt dat verlies de wallet min de onderhoudsmarge,
  dan wordt de futureswallet geliquideerd en is hij weg (wat er na de afwikkeling
  overblijft, gaat naar het verzekeringsfonds). De spot blijft staan en houdt zijn
  winst; de volgende herbalancering hedget opnieuw (en betaalt daarvoor). Het echte
  risico is dus een wick: de wallet weg, de spotwinst weer verdampt.
* **Delisting.** Verdwijnt de koers van één van de twee benen van een gehouden munt, dan
  worden beide benen gesloten: het verdwenen been tegen zijn laatste koers (rendement 0),
  met kosten op de laatst bekende ADV en σ.

SCREENINGSINSTRUMENT, zoals `book.py`: elke uitkomst draagt `NOT_ADMISSIBLE`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from ..schemas.robust_book_v2 import RobustBookV2Config
from ..utils.failfast import DataContractError, require
from .book import DUST, BookResult, CostSpec
from .breadth import IMPACT_EWMA_LAMBDA, BreadthMarket, bridge_data_holes
from .market import BARS_PER_YEAR

__all__ = [
    "BasisCosts",
    "BasisMarket",
    "basis_market",
    "carry_estimate",
    "combine_accounts",
    "harvest_targets",
    "load_basis_market",
    "run_basis",
]

FRAME_COLUMNS = ("gross", "funding", "liquidation", "fees", "spread", "slippage", "impact",
                 "net", "turnover", "gross_leverage", "net_leverage", "n_trades", "n_held",
                 "spot_notional", "futures_wallet")


@dataclass(frozen=True)
class BasisMarket:
    """Het perp-universum van v2 plus het spotbeen, op hetzelfde raster en dezelfde kolommen."""

    perp: BreadthMarket
    spot_close: pd.DataFrame
    spot_ret: pd.DataFrame
    spot_adv: pd.DataFrame
    spot_sigma: pd.DataFrame
    basis: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.perp.book.index

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.perp.book.symbols)

    def truncate(self, end: pd.Timestamp) -> BasisMarket:
        keep = self.index < end
        return BasisMarket(perp=self.perp.truncate(end), spot_close=self.spot_close.loc[keep],
                           spot_ret=self.spot_ret.loc[keep], spot_adv=self.spot_adv.loc[keep],
                           spot_sigma=self.spot_sigma.loc[keep], basis=self.basis.loc[keep])


def basis_market(perp: BreadthMarket, spot: Mapping[str, pd.DataFrame],
                 cfg: RobustBookV2Config) -> BasisMarket:
    """Leg het spotbeen op het perpraster. Een perp zonder spotpaar krijgt NaN: niet te hedgen."""
    idx, cols = perp.book.index, list(perp.book.symbols)
    close = spot["close"].reindex(index=idx, columns=cols)
    qv = spot["quote_volume"].reindex(index=idx, columns=cols)
    close, qv = bridge_data_holes(close, close), bridge_data_holes(qv, close)
    ret = close.pct_change(fill_method=None)
    adv = qv.rolling(cfg.universe.adv_window, min_periods=cfg.universe.adv_window).mean()
    sigma = np.sqrt((ret ** 2).ewm(alpha=1.0 - IMPACT_EWMA_LAMBDA, min_periods=30).mean())
    basis = perp.book.close / close - 1.0
    return BasisMarket(perp=perp, spot_close=close, spot_ret=ret, spot_adv=adv,
                       spot_sigma=sigma, basis=basis)


def load_basis_market(perp: BreadthMarket, spot_dir: Path, cfg: RobustBookV2Config) -> BasisMarket:
    spot = {k: pd.read_parquet(spot_dir / f"{k}.parquet") for k in ("close", "quote_volume")}
    return basis_market(perp, spot, cfg)


# --------------------------------------------------------------------------- #
# Het besluit
# --------------------------------------------------------------------------- #
def carry_estimate(m: BasisMarket, span: int) -> pd.DataFrame:
    """Geannualiseerde EWMA van de dagfunding, bekend op de close van *t*.

    `funding[t]` is de som van de afrekeningen in `[t − 1 dag, t)`: alle drie zijn op de
    close van *t* gepubliceerd. Vóór de notering is er geen funding (NaN, geen nul)."""
    raw = m.perp.book.funding.where(m.perp.book.close.notna())
    return raw.ewm(span=int(span), min_periods=int(span)).mean() * BARS_PER_YEAR


def harvest_targets(m: BasisMarket, *, span: int, enter_apr: float, exit_apr: float,
                    slots: int, notional: float, min_spot_adv_usd: float, max_abs_basis: float,
                    symbols: Sequence[str] | None = None,
                    carry_noise: pd.DataFrame | None = None) -> pd.DataFrame:
    """De gewenste hedge-notional per munt per besluitbar (fractie van de equity).

    Slots met hysterese: een gehouden munt blijft zolang zijn carry boven `exit_apr` ligt en
    beide benen verhandelbaar zijn; een vrij slot gaat naar de munt met de hoogste carry
    boven `enter_apr` die in het point-in-time-universum zit. Een gehouden munt wordt niet
    verdrongen door een betere: dat zou churn zijn, geen carry. Elk slot krijgt
    `notional / slots`; lege slots blijven cash.

    `symbols` beperkt de INSTAP tot die munten (de majors-variant). `carry_noise` (zelfde
    vorm) vermenigvuldigt de carryschatting: alleen voor de ruisstoets."""
    require(0.0 < notional <= 1.0 and slots >= 1 and exit_apr < enter_apr,
            "Ongeldige oogstparameters.", DataContractError,
            notional=notional, slots=slots, enter=enter_apr, exit=exit_apr)
    carry = carry_estimate(m, span)
    if carry_noise is not None:
        carry = carry * carry_noise.reindex(index=carry.index, columns=carry.columns).fillna(1.0)
    tradable = (m.perp.book.close.notna() & m.spot_close.notna()
                & (m.spot_adv >= min_spot_adv_usd) & (m.basis.abs() <= max_abs_basis)
                & (m.perp.book.adv_usd > 0.0) & m.perp.book.sigma_daily.notna()
                & m.spot_sigma.notna())
    enter = tradable & m.perp.universe & (carry >= enter_apr)
    if symbols is not None:
        enter &= pd.Series([s in set(symbols) for s in m.symbols], index=list(m.symbols))
    keep = tradable & (carry > exit_apr)
    c = carry.to_numpy(dtype=np.float64)
    en = enter.to_numpy(dtype=bool)
    kp = keep.to_numpy(dtype=bool)
    out = np.zeros(c.shape)
    held = np.zeros(c.shape[1], dtype=bool)
    size = float(notional) / int(slots)
    for t in range(c.shape[0]):
        held &= kp[t]
        free = int(slots) - int(held.sum())
        if free > 0:
            cand = np.flatnonzero(en[t] & ~held)
            if cand.size:
                # Hoogste carry eerst; gelijke carry: de kolomvolgorde (gesorteerd) beslist.
                order = cand[np.argsort(-c[t, cand], kind="stable")]
                held[order[:free]] = True
        out[t, held] = size
    return pd.DataFrame(out, index=m.index, columns=list(m.symbols))


# --------------------------------------------------------------------------- #
# Het boek
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BasisCosts:
    """De kosten van beide benen. `scaled` werkt op beide tegelijk."""

    perp: CostSpec
    spot: CostSpec

    def scaled(self, **kw: float) -> BasisCosts:
        return BasisCosts(perp=self.perp.scaled(**kw), spot=self.spot.scaled(**kw))

    def as_record(self) -> dict:
        return {"perp": self.perp.as_record(), "spot": self.spot.as_record()}


def _leg_cost(traded: np.ndarray, c: CostSpec, equity: float, sigma: np.ndarray,
              adv: np.ndarray, where: str) -> tuple[float, float, float, float]:
    fee = c.taker_fee * c.multiplier * traded.sum()
    spread = c.half_spread * c.multiplier * traded.sum()
    slip = c.extra_slippage * c.multiplier * traded.sum()
    impact = 0.0
    eta = c.eta
    if eta is not None and traded.any():
        idx = traded > 0.0
        require(bool(np.isfinite(adv[idx]).all() and (adv[idx] > 0.0).all()
                     and np.isfinite(sigma[idx]).all()),
                "Impact zonder geldige ADV of sigma op een handelsbar.", DataContractError,
                bar=where)
        frac = eta * sigma[idx] * np.sqrt(traded[idx] * equity / adv[idx])
        impact = float((traded[idx] * frac).sum()) * c.multiplier
    return fee, spread, slip, impact


def run_basis(target: pd.DataFrame, m: BasisMarket, costs: BasisCosts, *, lag: int,
              band: float, hedge_tolerance: float, maintenance_margin: float,
              margin_floor: float, skip: pd.Series | None = None) -> BookResult:
    """Simuleer het basisboek. `target` rij *t* = besluit op de close van *t* (notional per
    munt, ≥ 0), uitgevoerd tegen de close van *t + lag − 1*. `skip` (bool per bar) slaat
    de herbalancering over op die bars (stoets ontbrekende data); gedwongen exits gaan door."""
    require(int(lag) >= 1, "Een vertraging van nul bars is lookahead.", DataContractError)
    require(bool(target.index.equals(m.index)) and list(target.columns) == list(m.symbols),
            "Doel en markt delen geen raster of kolommen.", DataContractError)
    require(bool((target.fillna(0.0) >= 0.0).all().all()), "Een basisboek is long spot, "
            "short perp: een negatieve notional bestaat niet.", DataContractError)
    shift = int(lag) - 1
    tgt = target.shift(shift).fillna(0.0).to_numpy(dtype=np.float64)
    skp = (np.zeros(len(m.index), dtype=bool) if skip is None
           else skip.reindex(m.index).fillna(False).shift(shift, fill_value=False)
           .to_numpy(dtype=bool))
    pc = m.perp.book.close.to_numpy(dtype=np.float64)
    ph = m.perp.high.reindex(index=m.index, columns=list(m.symbols)).to_numpy(dtype=np.float64)
    rp = m.perp.book.ret.to_numpy(dtype=np.float64)
    rs = m.spot_ret.to_numpy(dtype=np.float64)
    fund = m.perp.book.funding.to_numpy(dtype=np.float64)
    p_ok = m.perp.book.close.notna().to_numpy()
    s_ok = m.spot_close.notna().to_numpy()
    p_sig = m.perp.book.sigma_daily.ffill().to_numpy(dtype=np.float64)
    s_sig = m.spot_sigma.ffill().to_numpy(dtype=np.float64)
    p_adv = m.perp.book.adv_usd.where(m.perp.book.adv_usd > 0.0).ffill().to_numpy(dtype=np.float64)
    s_adv = m.spot_adv.where(m.spot_adv > 0.0).ffill().to_numpy(dtype=np.float64)

    t_len, n = tgt.shape
    out = np.zeros((t_len, len(FRAME_COLUMNS)))
    held = np.zeros((t_len, n))
    trades = np.zeros((t_len, n))
    a = np.zeros(n)   # spot long, fractie van de equity
    b = np.zeros(n)   # perp short (als positieve grootte), fractie van de equity
    equity = float(costs.perp.aum_usd)
    n_forced = n_liq = 0
    for t in range(t_len):
        where = str(m.index[t])
        held[t] = (a + b) / 2.0
        r_s = np.where(np.isfinite(rs[t]), rs[t], 0.0)
        r_p = np.where(np.isfinite(rp[t]), rp[t], 0.0)
        gross = float(a @ r_s - b @ r_p)
        funding = float(b @ np.where(np.isfinite(fund[t]), fund[t], 0.0))
        # Liquidatietoets: elke short tegelijk op zijn high, tegen de wallet van gisteren.
        liq = 0.0
        liquidated = False
        if t > 0 and b.any():
            wallet = 1.0 - float(a.sum())
            up = np.where(np.isfinite(ph[t]) & np.isfinite(pc[t - 1]) & (b > 0),
                          ph[t] / np.where(b > 0, pc[t - 1], 1.0) - 1.0, 0.0)
            adverse = float(b @ np.clip(up, 0.0, None))
            if adverse >= wallet - maintenance_margin * float(b.sum()):
                n_liq += 1
                liquidated = True
                # De futureswallet is weg: het perpbeen verliest precies `wallet`, niet
                # `b·r_perp` (bruto bevat al −b·r_perp; dit is het verschil).
                liq = float(b @ r_p) - wallet
        port = gross + funding + liq
        require(1.0 + port > 0.0, "Het boek is op deze bar volledig weggevaagd.",
                DataContractError, bar=where)
        a_d = a * (1.0 + r_s) / (1.0 + port)
        b_d = np.zeros(n) if liquidated else b * (1.0 + r_p) / (1.0 + port)
        equity_pre = equity * (1.0 + port)

        new_a, new_b = a_d.copy(), b_d.copy()
        both = p_ok[t] & s_ok[t]
        n_forced += int((~both & ((a_d > 0.0) | (b_d > 0.0))).sum())
        if not skp[t]:
            want = np.where(both, tgt[t], 0.0)
            level = (a_d + b_d) / 2.0
            off = want <= 0.0
            enter = (want > 0.0) & (level <= 0.0)
            drift = (want > 0.0) & (level > 0.0) & (np.abs(level - want) > band * want)
            # Een munt waarvan de benen uiteenlopen (basisbeweging, of een geliquideerde
            # short) gaat in zijn geheel terug naar het doel: beide benen op `want`.
            unhedged = (want > 0.0) & (level > 0.0) & (
                np.abs(a_d - b_d) > hedge_tolerance * level)
            reset = enter | drift | unhedged
            new_a = np.where(off, 0.0, np.where(reset, want, a_d))
            new_b = np.where(off, 0.0, np.where(reset, want, b_d))
            # Marge-aanvulling, getoetst NA de trade: een rally (of een nieuwe instap)
            # heeft de futureswallet onder de vloer gebracht -> het hele boek naar het doel.
            if (1.0 - float(new_a.sum())) < margin_floor * float(new_b.sum()):
                new_a = np.where(want > 0.0, want, 0.0)
                new_b = new_a.copy()
        # Een munt zonder koers op een van beide benen wordt gesloten, ook op een
        # overgeslagen bar: het verdwenen been tegen zijn laatste koers.
        new_a = np.where(both, new_a, 0.0)
        new_b = np.where(both, new_b, 0.0)
        da, db = new_a - a_d, new_b - b_d
        da[np.abs(da) < DUST] = 0.0
        db[np.abs(db) < DUST] = 0.0
        new_a, new_b = a_d + da, b_d + db
        ta, tb = np.abs(da), np.abs(db)
        sf, ss, sl, si = _leg_cost(ta, costs.spot, equity_pre, s_sig[t], s_adv[t], where)
        pf, ps, pl, pi = _leg_cost(tb, costs.perp, equity_pre, p_sig[t], p_adv[t], where)
        fee, spread, slip, impact = sf + pf, ss + ps, sl + pl, si + pi
        cost = fee + spread + slip + impact
        require(cost < 1.0, "Kosten van meer dan de hele equity op één bar.", DataContractError)
        net = (1.0 + port) * (1.0 - cost) - 1.0
        equity = equity_pre * (1.0 - cost)
        sc = 1.0 + port
        a, b = new_a, new_b
        out[t] = (gross, funding, liq, -fee * sc, -spread * sc, -slip * sc, -impact * sc, net,
                  float(ta.sum() + tb.sum()), float(a.sum() + b.sum()), float(a.sum() - b.sum()),
                  float(((ta > 0) | (tb > 0)).sum()), float((a > 0).sum()), float(a.sum()),
                  1.0 - float(a.sum()))
        trades[t] = ta + tb

    frame = pd.DataFrame(out, index=m.index, columns=list(FRAME_COLUMNS))
    identity = (frame[["gross", "funding", "liquidation", "fees", "spread", "slippage",
                       "impact"]].sum(axis=1) - frame["net"]).abs().max()
    require(float(identity) < 1e-12, "De P&L-trap sluit niet.", DataContractError,
            residual=float(identity))
    return BookResult(
        frame=frame, held=pd.DataFrame(held, index=m.index, columns=list(m.symbols)),
        trades=pd.DataFrame(trades, index=m.index, columns=list(m.symbols)),
        audit={EVIDENCE_KEY: NOT_ADMISSIBLE, "engine": "systematic.harvest", "lag": int(lag),
               "costs": costs.as_record(), "n_forced_exits": n_forced,
               "n_liquidations": n_liq, "band": float(band),
               "hedge_tolerance": float(hedge_tolerance),
               "maintenance_margin": float(maintenance_margin),
               "margin_floor": float(margin_floor),
               "convention": "besluit op close t, uitgevoerd tegen close t+lag-1, "
                             "rendeert vanaf de bar erna; long spot, short perp"})


def combine_accounts(parts: Mapping[str, tuple[float, BookResult]]) -> BookResult:
    """Subrekeningen met vaste kapitaalaandelen, dagelijks herverdeeld (een interne
    overboeking spot <-> futures kost niets). Elke kolom is het gewogen gemiddelde; het
    netto rendement is dat exact."""
    shares = [s for s, _ in parts.values()]
    require(abs(sum(shares) - 1.0) < 1e-12 and all(s > 0 for s in shares),
            "Kapitaalaandelen moeten positief zijn en optellen tot 1.", DataContractError,
            shares=shares)
    cols = ("gross", "funding", "fees", "spread", "slippage", "impact", "net", "turnover",
            "gross_leverage", "net_leverage", "n_trades")
    frames = [(s, r.frame) for s, r in parts.values()]
    idx = frames[0][1].index
    require(all(f.index.equals(idx) for _, f in frames), "Subrekeningen op verschillende "
            "rasters.", DataContractError)
    frame = pd.DataFrame({c: sum(s * f[c] if c in f else 0.0 * f["net"] for s, f in frames)
                          for c in cols}, index=idx)
    frame["liquidation"] = sum(s * f["liquidation"] if "liquidation" in f else 0.0 * f["net"]
                               for s, f in frames)
    held = sum(s * r.held.reindex(columns=sorted(set().union(*[set(x.held.columns) for _, x
                                                             in parts.values()])),
                                  fill_value=0.0) for s, r in parts.values())
    trades = sum(s * r.trades.reindex(columns=held.columns, fill_value=0.0)
                 for s, r in parts.values())
    return BookResult(frame=frame, held=held, trades=trades, audit={
        EVIDENCE_KEY: NOT_ADMISSIBLE, "engine": "systematic.harvest.combine_accounts",
        "accounts": {k: {"share": s, "audit": r.audit} for k, (s, r) in parts.items()},
        "n_forced_exits": int(sum(r.audit.get("n_forced_exits", 0) for _, r in parts.values()))})
