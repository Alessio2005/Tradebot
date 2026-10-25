"""Robuust boek v6: de basiscarry van v5, met hefboom, in één unified-margin-account.

Ontwerp: `docs/superpowers/specs/2026-10-09-robust-book-v6-levered-carry-design.md`.

WAAROM HEFBOOM, EN WAAROM HIER
==============================
v5 vond de enige robuuste bron van rendement in deze repo: long spot, short perp, de
funding oogsten (Sharpe 5,7-6,8 op W_DEV, 11,5 op de ongemeten backcast). De Sharpe was
nooit het probleem; de kapitaalefficiëntie wel. In het tweewalletmodel van v5 (70 % spot,
30 % USDT-marge) stond gemiddeld maar 0,26-0,33 van de equity per been in de markt. Een
unified (portfolio-)margin-account gebruikt de spot zelf als onderpand voor de short: de
eigen equity draagt dan 1x per been, en wat daarboven gaat wordt in USDT geleend.

De eigenaar staat sinds 2026-10-09 hoog risico toe. Het risico van dit boek zit niet in
de vol (~1-2 %/jaar) maar in de staart. Dat ziet een backtest niet vanzelf, dus wordt het
hier expliciet gemodelleerd:

* **Financiering.** Lening = spot boven de eigen equity, `max(0, Σa − 1)`, tegen
  `max(vloer, k · BTC-carry)`. Een USDT-uitlener kan zelf de BTC-basis oogsten; lenen
  onder dat tarief is geen gratis geld. Daardoor betaalt hefboom alleen in munten met
  meer carry dan BTC.
* **Onderpand met haircut.** Spot telt als onderpand tegen `1 − h` (majors 5 %, de rest
  20 %). Een rally laat de short evenveel verliezen als de spot wint, maar de spot telt
  maar voor `1 − h` mee: de marge erodeert met `a · h · rally`.
* **uniMMR** = haircut-equity / onderhoudsmarge, met onderhoud `mmr` op de perp-notional
  en `mm_loan` op de lening. Binance liquideert een PM-account rond 1,05.
* **Liquidatietoets op de MARK price**, intraday, voor het hele boek in twee staten:
  - *up*: elke short op zijn mark-high, de spot hooguit even ver (de basis kan alleen
    pijn doen: `min(spot-high, mark-high)`);
  - *down*: elke short op zijn mark-low, de spot op het slechtste van spot-low en
    mark-low.

  Raakt de haircut-equity in een van beide staten de onderhoudsmarge, dan is het account
  geliquideerd: alles gesloten in die staat, min een liquidatievergoeding over de bruto
  notional. Er is geen funding voor die bar. Ontbreekt de mark price, dan geldt de last
  price, die verder wickt (conservatief).
* **Governor**, na elke trade:
  - bruto ≤ `gross_cap` (beide benen, het mandaat van `conf/risk/default.yaml`);
  - uniMMR ≥ `min_uni_mmr`.

  Wordt een van beide geschonden, dan gaat het boek naar het doel, geschaald met de
  grootste toegestane factor.

Alle andere mechaniek is die van `harvest.py`: twee benen, band, hedge-tolerantie,
delisting, kosten per been, de sluitende P&L-identiteit.

SCREENINGSINSTRUMENT, zoals `book.py` en `harvest.py`: elke uitkomst draagt
`NOT_ADMISSIBLE`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from ..utils.failfast import DataContractError, require
from .book import DUST, BookResult
from .harvest import BasisCosts, BasisMarket, _leg_cost, carry_estimate, harvest_targets
from .market import BARS_PER_YEAR

__all__ = [
    "FRAME_COLUMNS",
    "LeverArrays",
    "LeverBar",
    "LeverStep",
    "LeveredBook",
    "lever_arrays",
    "tranche_targets",
    "MarginMarket",
    "MarginSpec",
    "financing_rate",
    "governed_scale",
    "levered_targets",
    "load_margin_market",
    "margin_market",
    "run_levered",
    "uni_mmr",
]

FRAME_COLUMNS = ("gross", "funding", "financing", "liquidation", "fees", "spread", "slippage",
                 "impact", "net", "turnover", "gross_leverage", "net_leverage", "n_trades",
                 "n_held", "spot_notional", "loan", "uni_mmr", "stress_headroom", "governed")


@dataclass(frozen=True)
class MarginMarket:
    """De basismarkt van v5 plus de intraday-extremen die de liquidatietoets nodig heeft."""

    basis: BasisMarket
    mark_high: pd.DataFrame
    mark_low: pd.DataFrame
    spot_high: pd.DataFrame
    spot_low: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.basis.index

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.basis.symbols

    def truncate(self, end: pd.Timestamp) -> MarginMarket:
        keep = self.index < end
        return MarginMarket(basis=self.basis.truncate(end), mark_high=self.mark_high.loc[keep],
                            mark_low=self.mark_low.loc[keep], spot_high=self.spot_high.loc[keep],
                            spot_low=self.spot_low.loc[keep])


def margin_market(basis: BasisMarket, mark: Mapping[str, pd.DataFrame],
                  spot: Mapping[str, pd.DataFrame]) -> MarginMarket:
    """Leg mark-high/low en spot-high/low op het raster van de basismarkt.

    Een ontbrekende mark-waarde blijft NaN: `run_levered` valt dan terug op de last price.
    Een ontbrekend spot-extreem krijgt de spot-close (geen intraday-informatie)."""
    idx, cols = basis.index, list(basis.symbols)
    sc = basis.spot_close
    shi = spot["high"].reindex(index=idx, columns=cols)
    slo = spot["low"].reindex(index=idx, columns=cols)
    return MarginMarket(
        basis=basis,
        mark_high=mark["high"].reindex(index=idx, columns=cols),
        mark_low=mark["low"].reindex(index=idx, columns=cols),
        spot_high=shi.where(shi.notna(), sc), spot_low=slo.where(slo.notna(), sc))


def load_margin_market(basis: BasisMarket, spot_dir: Path, mark_dir: Path) -> MarginMarket:
    mark = {k: pd.read_parquet(mark_dir / f"{k}.parquet") for k in ("high", "low")}
    spot = {k: pd.read_parquet(spot_dir / f"{k}.parquet") for k in ("high", "low")}
    return margin_market(basis, mark, spot)


@dataclass(frozen=True)
class MarginSpec:
    """Het margemodel van het unified account. Alles als fractie (geen bp)."""

    haircut_major: float
    haircut_alt: float
    majors: tuple[str, ...]
    maintenance_margin: float
    loan_maintenance: float
    min_uni_mmr: float
    gross_cap: float
    liquidation_fee: float

    def __post_init__(self) -> None:
        for name in ("haircut_major", "haircut_alt", "maintenance_margin", "loan_maintenance",
                     "liquidation_fee"):
            v = float(getattr(self, name))
            require(0.0 <= v < 1.0, f"{name} moet in [0, 1) liggen.", DataContractError,
                    value=v)
        require(self.min_uni_mmr >= 1.0 and self.gross_cap > 0.0,
                "Een governor onder uniMMR 1 of zonder bruto-cap is geen governor.",
                DataContractError, min_uni_mmr=self.min_uni_mmr, gross_cap=self.gross_cap)

    def haircuts(self, symbols: Sequence[str]) -> np.ndarray:
        majors = set(self.majors)
        return np.array([self.haircut_major if s in majors else self.haircut_alt
                         for s in symbols], dtype=np.float64)

    def as_record(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else float(v))
                for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- #
# Besluit en financiering
# --------------------------------------------------------------------------- #
def levered_targets(m: BasisMarket, *, leverage: float, slots: int, span: int,
                    enter_apr: float, exit_apr: float, min_spot_adv_usd: float,
                    max_abs_basis: float, symbols: Sequence[str] | None = None,
                    carry_noise: pd.DataFrame | None = None,
                    max_sigma: float | None = None) -> pd.DataFrame:
    """De slotregel van v5, ongewijzigd, met `leverage` als notional per been.

    De slotkeuze hangt niet van de notional af (elk slot krijgt `notional / slots`), dus
    het hefboomdoel is exact `leverage` × het doel bij notional 1. De ADV-cap hoort NIET
    hier maar in `run_levered`: hij hangt af van de lopende equity, en die kent alleen het
    boek."""
    require(float(leverage) > 0.0, "Hefboom moet positief zijn.", DataContractError,
            leverage=leverage)
    one = harvest_targets(m, span=span, enter_apr=enter_apr, exit_apr=exit_apr, slots=slots,
                          notional=1.0, min_spot_adv_usd=min_spot_adv_usd,
                          max_abs_basis=max_abs_basis, symbols=symbols, carry_noise=carry_noise,
                          max_sigma=max_sigma)
    return one * float(leverage)


def tranche_targets(m: BasisMarket, *, slots: int, span: int, enter_apr: float,
                    exit_apr: float, min_spot_adv_usd: float, max_abs_basis: float,
                    financing: pd.Series, lever_spread: float,
                    carry_noise: pd.DataFrame | None = None,
                    max_sigma: float | None = None) -> pd.DataFrame:
    """v7: een basistranche uit eigen equity, plus een GEFINANCIERDE tranche waar de carry
    zijn leenrente verdient.

    * Basis: de slotregel van v5, ongewijzigd, `1 / slots` per munt (samen ≤ 1x).
    * Gefinancierde tranche: nog eens `1 / slots` op een munt die een basisslot heeft,
      aan zodra `carry ≥ r_fin + lever_spread`, en aan zolang `carry ≥ r_fin`
      (hysterese). Hefboom wordt dus alleen gebruikt waar hij verdient wat hij kost;
      bij 2021-tarieven (~31 %) vraagt een tranche ~41 % carry.

    `financing` is de rente die het boek ook betaalt (bekend op de close van *t*);
    `carry_noise` en `max_sigma` (de sprongrisicogrens van v8) werken als in
    `harvest_targets`: de tranche hangt aan het basisslot, dus valt mee weg."""
    require(float(lever_spread) >= 0.0, "Een negatieve spread leent onder de rente.",
            DataContractError, lever_spread=lever_spread)
    require(bool(financing.index.equals(m.index)), "Financiering en markt delen geen raster.",
            DataContractError)
    base = harvest_targets(m, span=span, enter_apr=enter_apr, exit_apr=exit_apr, slots=slots,
                           notional=1.0, min_spot_adv_usd=min_spot_adv_usd,
                           max_abs_basis=max_abs_basis, carry_noise=carry_noise,
                           max_sigma=max_sigma)
    carry = carry_estimate(m, span)
    if carry_noise is not None:
        carry = carry * carry_noise.reindex(index=carry.index, columns=carry.columns).fillna(1.0)
    c = carry.to_numpy(dtype=np.float64)
    r = financing.to_numpy(dtype=np.float64)[:, None]
    held = base.to_numpy() > 0.0
    with np.errstate(invalid="ignore"):
        enter = held & (c >= r + float(lever_spread))
        keep = held & (c >= r)
    on = np.zeros(c.shape[1], dtype=bool)
    extra = np.zeros(c.shape)
    for t in range(c.shape[0]):
        on = (on & keep[t]) | enter[t]
        extra[t] = on
    return base + pd.DataFrame(extra / int(slots), index=base.index, columns=base.columns)


def financing_rate(m: BasisMarket, *, floor_apr: float, multiplier: float, span: int,
                   reference: str = "BTCUSDT", constant_apr: float | None = None) -> pd.Series:
    """De jaarrente op geleende USDT, bekend op de close van *t*.

    `max(floor_apr, multiplier × carry_ref)`, met `carry_ref` de EWMA-funding van
    `reference` (dezelfde schatter als het besluit). Zonder carryschatting geldt de vloer.
    `constant_apr` vervangt het model (alleen voor een gevoeligheid)."""
    if constant_apr is not None:
        return pd.Series(float(constant_apr), index=m.index, name="financing_apr")
    require(reference in m.symbols, "De referentie voor de financiering ontbreekt.",
            DataContractError, reference=reference)
    carry = carry_estimate(m, span)[reference] * float(multiplier)
    out = np.maximum(float(floor_apr), carry.fillna(-np.inf).to_numpy())
    return pd.Series(out, index=m.index, name="financing_apr")


# --------------------------------------------------------------------------- #
# Marge
# --------------------------------------------------------------------------- #
def uni_mmr(a: np.ndarray, b: np.ndarray, h: np.ndarray, *, maintenance_margin: float,
            loan_maintenance: float) -> float:
    """Haircut-equity / onderhoudsmarge, met a, b fracties van de equity (equity = 1)."""
    mm = maintenance_margin * float(b.sum()) + loan_maintenance * max(float(a.sum()) - 1.0, 0.0)
    if mm <= 0.0:
        return float("inf")
    return (1.0 - float(a @ h)) / mm


def governed_scale(a: np.ndarray, b: np.ndarray, h: np.ndarray, spec: MarginSpec) -> float:
    """De grootste s ∈ (0, 1] waarbij s·(a, b) binnen de bruto-cap en de uniMMR-vloer ligt.

    De uniMMR daalt monotoon in s; bisectie is daarom exact tot op de tolerantie."""
    def ok(s: float) -> bool:
        return (s * float(a.sum() + b.sum()) <= spec.gross_cap + 1e-12
                and uni_mmr(s * a, s * b, h, maintenance_margin=spec.maintenance_margin,
                            loan_maintenance=spec.loan_maintenance)
                >= spec.min_uni_mmr - 1e-12)

    if ok(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    require(lo > 0.0, "Zelfs een minimaal boek past niet binnen de governor.",
            DataContractError)
    return lo


def _move(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den - 1.0


# --------------------------------------------------------------------------- #
# Het boek
# --------------------------------------------------------------------------- #
# EEN implementatie voor backtest en live (audit 2026-10-24, E-1). `run_levered` rekent de
# rijen voor (`LeverArrays`) en roept per bar `LeveredBook.step` aan; het live-pad
# (`tradebot.carry`) roept dezelfde `step` aan op de laatste bar, met een boek dat uit zijn
# opgeslagen staat is hersteld. Een tweede kopie van deze mechaniek in het live-pad zou
# precies het soort sim-to-real-gat zijn dat dit boek moet uitsluiten.
@dataclass(frozen=True)
class LeverBar:
    """Alles wat `LeveredBook.step` over één bar nodig heeft (rijen van lengte n)."""

    where: str
    first: bool
    tgt: np.ndarray
    skip: bool
    r_s: np.ndarray
    r_p: np.ndarray
    fund: np.ndarray
    mk_hi: np.ndarray
    mk_lo: np.ndarray
    s_hi: np.ndarray
    s_lo: np.ndarray
    pc_prev: np.ndarray
    sc_prev: np.ndarray
    rate_prev: float
    both: np.ndarray
    adv_min: np.ndarray
    s_sig: np.ndarray
    s_adv: np.ndarray
    p_sig: np.ndarray
    p_adv: np.ndarray


@dataclass(frozen=True)
class LeverArrays:
    """De voorberekende rijen van een markt: `bar(t)` geeft de invoer van bar *t*."""

    index: pd.DatetimeIndex
    symbols: tuple[str, ...]
    tgt: np.ndarray
    skp: np.ndarray
    pc: np.ndarray
    sc: np.ndarray
    mk_hi: np.ndarray
    mk_lo: np.ndarray
    s_hi: np.ndarray
    s_lo: np.ndarray
    rp: np.ndarray
    rs: np.ndarray
    fund: np.ndarray
    p_ok: np.ndarray
    s_ok: np.ndarray
    p_sig: np.ndarray
    s_sig: np.ndarray
    p_adv: np.ndarray
    s_adv: np.ndarray
    rate: np.ndarray
    adv_min: np.ndarray

    def bar(self, t: int) -> LeverBar:
        first = t == 0
        prev = 0 if first else t - 1
        return LeverBar(
            where=str(self.index[t]), first=first, tgt=self.tgt[t], skip=bool(self.skp[t]),
            r_s=self.rs[t], r_p=self.rp[t], fund=self.fund[t], mk_hi=self.mk_hi[t],
            mk_lo=self.mk_lo[t], s_hi=self.s_hi[t], s_lo=self.s_lo[t], pc_prev=self.pc[prev],
            sc_prev=self.sc[prev], rate_prev=0.0 if first else float(self.rate[t - 1]),
            both=self.p_ok[t] & self.s_ok[t], adv_min=self.adv_min[t], s_sig=self.s_sig[t],
            s_adv=self.s_adv[t], p_sig=self.p_sig[t], p_adv=self.p_adv[t])


def lever_arrays(target: pd.DataFrame, m: MarginMarket, *, lag: int, financing: pd.Series,
                 skip: pd.Series | None = None) -> LeverArrays:
    """Valideer de invoer van het hefboomboek en reken zijn rijen voor."""
    require(int(lag) >= 1, "Een vertraging van nul bars is lookahead.", DataContractError)
    b0 = m.basis
    require(bool(target.index.equals(m.index)) and list(target.columns) == list(m.symbols),
            "Doel en markt delen geen raster of kolommen.", DataContractError)
    require(bool((target.fillna(0.0) >= 0.0).all().all()), "Een basisboek is long spot, "
            "short perp: een negatieve notional bestaat niet.", DataContractError)
    require(bool(financing.index.equals(m.index)) and bool(np.isfinite(financing).all()),
            "De financieringsreeks moet het hele raster dekken.", DataContractError)
    shift = int(lag) - 1
    tgt = target.shift(shift).fillna(0.0).to_numpy(dtype=np.float64)
    skp = (np.zeros(len(m.index), dtype=bool) if skip is None
           else skip.reindex(m.index).fillna(False).shift(shift, fill_value=False)
           .to_numpy(dtype=bool))
    syms = list(m.symbols)
    last_hi = b0.perp.high.reindex(index=m.index, columns=syms).to_numpy(dtype=np.float64)
    last_lo = b0.perp.low.reindex(index=m.index, columns=syms).to_numpy(dtype=np.float64)
    mk_hi = m.mark_high.to_numpy(dtype=np.float64)
    mk_lo = m.mark_low.to_numpy(dtype=np.float64)
    p_adv = b0.perp.book.adv_usd.where(b0.perp.book.adv_usd > 0.0).ffill().to_numpy(dtype=np.float64)
    s_adv = b0.spot_adv.where(b0.spot_adv > 0.0).ffill().to_numpy(dtype=np.float64)
    return LeverArrays(
        index=pd.DatetimeIndex(m.index), symbols=tuple(syms), tgt=tgt, skp=skp,
        pc=b0.perp.book.close.to_numpy(dtype=np.float64),
        sc=b0.spot_close.to_numpy(dtype=np.float64),
        mk_hi=np.where(np.isfinite(mk_hi), mk_hi, last_hi),
        mk_lo=np.where(np.isfinite(mk_lo), mk_lo, last_lo),
        s_hi=m.spot_high.to_numpy(dtype=np.float64), s_lo=m.spot_low.to_numpy(dtype=np.float64),
        rp=b0.perp.book.ret.to_numpy(dtype=np.float64), rs=b0.spot_ret.to_numpy(dtype=np.float64),
        fund=b0.perp.book.funding.to_numpy(dtype=np.float64),
        p_ok=b0.perp.book.close.notna().to_numpy(), s_ok=b0.spot_close.notna().to_numpy(),
        p_sig=b0.perp.book.sigma_daily.ffill().to_numpy(dtype=np.float64),
        s_sig=b0.spot_sigma.ffill().to_numpy(dtype=np.float64), p_adv=p_adv, s_adv=s_adv,
        rate=financing.to_numpy(dtype=np.float64),
        adv_min=np.nan_to_num(np.fmin(s_adv, p_adv), nan=0.0))


@dataclass(frozen=True)
class LeverStep:
    """De uitkomst van één bar: de framerij, en de trades per been (fractie van `equity_pre`)."""

    row: tuple[float, ...]
    held: np.ndarray
    trades: np.ndarray
    da: np.ndarray
    db: np.ndarray
    equity_pre: float
    liquidated: bool


@dataclass(frozen=True)
class _Marked:
    """Het boek na de P&L van de bar en vóór het besluit (fracties van `equity_pre`)."""

    held: np.ndarray
    gross: float
    funding: float
    fin: float
    liq: float
    loan: float
    headroom: float
    liquidated: bool
    port: float
    a_d: np.ndarray
    b_d: np.ndarray
    equity_pre: float


class LeveredBook:
    """De toestand van het hefboomboek en zijn mechaniek per bar.

    `a` = spot long, `b` = perp short (positieve grootte), beide als fractie van de equity
    aan het einde van de vorige bar; `goal` het doel van een munt in beweging (v7);
    `equity` in USD. Zie `run_levered` voor de conventies."""

    def __init__(self, symbols: Sequence[str], costs: BasisCosts, *, band: float,
                 hedge_tolerance: float, margin: MarginSpec, adv_cap: float | None = None,
                 slice_step: float | None = None) -> None:
        require(adv_cap is None or 0.0 < float(adv_cap) <= 1.0, "Ongeldige ADV-cap.",
                DataContractError, adv_cap=adv_cap)
        require(slice_step is None or float(slice_step) > 0.0, "Ongeldige stapgrootte.",
                DataContractError, slice_step=slice_step)
        self.symbols = tuple(symbols)
        n = len(self.symbols)
        self.costs, self.band, self.hedge_tolerance = costs, float(band), float(hedge_tolerance)
        self.margin, self.adv_cap, self.slice_step = margin, adv_cap, slice_step
        self.h = margin.haircuts(self.symbols)
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.goal = np.full(n, np.nan)   # v7: het doel van een munt in beweging
        self.equity = float(costs.perp.aum_usd)
        self.n_forced = self.n_liq = self.n_governed = 0
        self.min_headroom = float("inf")

    def step(self, bar: LeverBar) -> LeverStep:
        """Eén bar van de backtest (en van het paper-/shadowboek): eerst de P&L van de
        markt over de bar volgens het model, dan het besluit op zijn close."""
        return self._decide(bar, self._mark(bar))

    def decide_live(self, bar: LeverBar, a_d: np.ndarray, b_d: np.ndarray,
                    equity: float) -> LeverStep:
        """Het besluit op de close van `bar` voor een boek waarvan de BEURS de stand geeft.

        `a_d`, `b_d` zijn de gemarkte benen (fractie van `equity`, de equity die de beurs
        meldt). Het model-P&L-deel van `step` vervalt: de beurs heeft de bar al
        afgerekend, inclusief funding, rente en een eventuele liquidatie of ADL."""
        require(math.isfinite(equity) and equity > 0.0, "De beurs meldt geen positieve "
                "equity.", DataContractError, equity=equity)
        a_d = np.asarray(a_d, dtype=np.float64)
        b_d = np.asarray(b_d, dtype=np.float64)
        require(a_d.shape == b_d.shape == (len(self.symbols),) and bool(
            (a_d >= 0.0).all() and (b_d >= 0.0).all()), "Ongeldige gemarkte benen.",
            DataContractError)
        self.equity = float(equity)
        loan = max(float(a_d.sum()) - 1.0, 0.0)
        return self._decide(bar, _Marked(held=(a_d + b_d) / 2.0, gross=0.0, funding=0.0,
                                         fin=0.0, liq=0.0, loan=loan, headroom=float("nan"),
                                         liquidated=False, port=0.0, a_d=a_d, b_d=b_d,
                                         equity_pre=float(equity)))

    def _mark(self, bar: LeverBar) -> _Marked:
        a, b, h = self.a, self.b, self.h
        margin = self.margin
        mmr, mml = margin.maintenance_margin, margin.loan_maintenance
        n = len(self.symbols)
        where = bar.where
        held = (a + b) / 2.0
        r_s = np.where(np.isfinite(bar.r_s), bar.r_s, 0.0)
        r_p = np.where(np.isfinite(bar.r_p), bar.r_p, 0.0)
        gross = float(a @ r_s - b @ r_p)
        funding = float(b @ np.where(np.isfinite(bar.fund), bar.fund, 0.0))
        loan = max(float(a.sum()) - 1.0, 0.0)
        fin = -loan * bar.rate_prev / BARS_PER_YEAR
        liq = 0.0
        headroom = float("nan")
        liquidated = False
        if not bar.first and (a.any() or b.any()):
            um = np.nan_to_num(_move(bar.mk_hi, bar.pc_prev), nan=0.0, posinf=0.0, neginf=0.0)
            dm = np.nan_to_num(_move(bar.mk_lo, bar.pc_prev), nan=0.0, posinf=0.0, neginf=0.0)
            us = np.nan_to_num(_move(bar.s_hi, bar.sc_prev), nan=0.0, posinf=0.0, neginf=0.0)
            ds = np.nan_to_num(_move(bar.s_lo, bar.sc_prev), nan=0.0, posinf=0.0, neginf=0.0)
            states = {"up": (np.minimum(us, um), um), "down": (np.minimum(ds, dm), dm)}
            ae0 = 1.0 - float(a @ h) + fin
            mm_s = mmr * float(b @ (1.0 + np.clip(um, 0.0, None))) + mml * loan
            heads = {k: ae0 + float((a * (1.0 - h)) @ sm - b @ pm) - mm_s
                     for k, (sm, pm) in states.items()}
            worst = min(heads, key=lambda k: heads[k])
            if mm_s > 0.0:
                headroom = heads[worst] / mm_s
                self.min_headroom = min(self.min_headroom, headroom)
            if heads[worst] < 0.0:
                self.n_liq += 1
                liquidated = True
                sm, pm = states[worst]
                e_s = 1.0 + float(a @ sm - b @ pm) + fin
                fee = margin.liquidation_fee * float(a @ (1.0 + sm) + b @ (1.0 + pm))
                e_after = e_s - fee
                require(e_after > 0.0, "Het account is bij de liquidatie volledig "
                        "weggevaagd.", DataContractError, bar=where, equity=e_after)
                funding = 0.0
                liq = (e_after - 1.0) - gross - fin
        port = gross + funding + fin + liq
        require(1.0 + port > 0.0, "Het boek is op deze bar volledig weggevaagd.",
                DataContractError, bar=where)
        if liquidated:
            a_d, b_d = np.zeros(n), np.zeros(n)
            self.goal[:] = np.nan
        else:
            a_d = a * (1.0 + r_s) / (1.0 + port)
            b_d = b * (1.0 + r_p) / (1.0 + port)
        return _Marked(held=held, gross=gross, funding=funding, fin=fin, liq=liq, loan=loan,
                       headroom=headroom, liquidated=liquidated, port=port, a_d=a_d, b_d=b_d,
                       equity_pre=self.equity * (1.0 + port))

    def _decide(self, bar: LeverBar, mk: _Marked) -> LeverStep:
        h, margin, band, hedge_tolerance = self.h, self.margin, self.band, self.hedge_tolerance
        adv_cap, slice_step, costs = self.adv_cap, self.slice_step, self.costs
        mmr, mml = margin.maintenance_margin, margin.loan_maintenance
        where = bar.where
        a_d, b_d, equity_pre, port = mk.a_d, mk.b_d, mk.equity_pre, mk.port
        gross, funding, fin, liq, loan = mk.gross, mk.funding, mk.fin, mk.liq, mk.loan
        new_a, new_b = a_d.copy(), b_d.copy()
        governed = False
        both = bar.both
        goal = self.goal
        self.n_forced += int((~both & ((a_d > 0.0) | (b_d > 0.0))).sum())
        if not bar.skip:
            want = np.where(both, bar.tgt, 0.0)
            if adv_cap is not None:
                want = np.minimum(want, float(adv_cap) * bar.adv_min / equity_pre)
            level = (a_d + b_d) / 2.0
            off = want <= 0.0
            enter = (want > 0.0) & (level <= 0.0)
            drift = (want > 0.0) & (level > 0.0) & (np.abs(level - want) > band * want)
            unhedged = (want > 0.0) & (level > 0.0) & (
                np.abs(a_d - b_d) > hedge_tolerance * level)
            if slice_step is None:
                reset = enter | drift | unhedged
                new_a = np.where(off, 0.0, np.where(reset, want, a_d))
                new_b = np.where(off, 0.0, np.where(reset, want, b_d))
                # De governor, getoetst NA de trade: bruto-cap en uniMMR-vloer. Geschonden
                # -> het hele boek naar het doel, geschaald met de grootste toegestane factor.
                if governed_scale(new_a, new_b, h, margin) < 1.0:
                    self.n_governed += 1
                    governed = True
                    base = np.where(want > 0.0, want, 0.0)
                    s = governed_scale(base, base, h, margin)
                    new_a, new_b = base * s, base * s
            else:
                # De governor op het DOEL: past het volledige doel niet, dan krimpt het doel.
                s_goal = governed_scale(want, want, h, margin)
                if s_goal < 1.0:
                    governed = True
                    want = want * s_goal
                # Uiteenlopende benen: gelijk op het huidige niveau (een kleine trade).
                a0 = np.where(unhedged, level, a_d)
                b0 = np.where(unhedged, level, b_d)
                start = (off & (level > 0.0)) | enter | drift
                goal = np.where(start | np.isfinite(goal), want, np.nan)
                moving = np.isfinite(goal)
                lvl = (a0 + b0) / 2.0
                gap = np.where(moving, goal - lvl, 0.0)
                new_lvl = lvl + np.clip(gap, -float(slice_step), float(slice_step))
                new_a = np.where(moving, new_lvl, a0)
                new_b = np.where(moving, new_lvl, b0)
                goal = np.where(moving & (np.abs(gap) <= float(slice_step)), np.nan, goal)
                # De governor op het BOEK (risico): drift buiten de grens krimpt meteen.
                s_book = governed_scale(new_a, new_b, h, margin)
                if s_book < 1.0:
                    governed = True
                    new_a, new_b = new_a * s_book, new_b * s_book
                self.n_governed += int(governed)
        # Een munt zonder koers op een van beide benen wordt gesloten, ook op een
        # overgeslagen bar: het verdwenen been tegen zijn laatste koers.
        new_a = np.where(both, new_a, 0.0)
        new_b = np.where(both, new_b, 0.0)
        self.goal = np.where(both, goal, np.nan)
        da, db = new_a - a_d, new_b - b_d
        da[np.abs(da) < DUST] = 0.0
        db[np.abs(db) < DUST] = 0.0
        new_a, new_b = a_d + da, b_d + db
        ta, tb = np.abs(da), np.abs(db)
        sf, ss, sl, si = _leg_cost(ta, costs.spot, equity_pre, bar.s_sig, bar.s_adv, where)
        pf, ps, pl, pi = _leg_cost(tb, costs.perp, equity_pre, bar.p_sig, bar.p_adv, where)
        fee, spread, slip, impact = sf + pf, ss + ps, sl + pl, si + pi
        cost = fee + spread + slip + impact
        require(cost < 1.0, "Kosten van meer dan de hele equity op één bar.", DataContractError)
        net = (1.0 + port) * (1.0 - cost) - 1.0
        self.equity = equity_pre * (1.0 - cost)
        scl = 1.0 + port
        self.a, self.b = new_a, new_b
        row = (gross, funding, fin, liq, -fee * scl, -spread * scl, -slip * scl,
               -impact * scl, net, float(ta.sum() + tb.sum()), float(new_a.sum() + new_b.sum()),
               float(new_a.sum() - new_b.sum()), float(((ta > 0) | (tb > 0)).sum()),
               float((new_a > 0).sum()), float(new_a.sum()), loan,
               uni_mmr(new_a, new_b, h, maintenance_margin=mmr, loan_maintenance=mml),
               mk.headroom, float(governed))
        return LeverStep(row=row, held=mk.held, trades=ta + tb, da=da, db=db,
                         equity_pre=equity_pre, liquidated=mk.liquidated)

    def audit(self, lag: int) -> dict:
        return {EVIDENCE_KEY: NOT_ADMISSIBLE, "engine": "systematic.leverage", "lag": int(lag),
                "costs": self.costs.as_record(), "margin": self.margin.as_record(),
                "n_forced_exits": self.n_forced, "n_liquidations": self.n_liq,
                "n_governed": self.n_governed,
                "min_stress_headroom": (self.min_headroom if np.isfinite(self.min_headroom)
                                        else float("nan")),
                "band": self.band, "hedge_tolerance": self.hedge_tolerance,
                "adv_cap": None if self.adv_cap is None else float(self.adv_cap),
                "slice_step": None if self.slice_step is None else float(self.slice_step),
                "convention": "besluit op close t, uitgevoerd tegen close t+lag-1, rendeert "
                              "vanaf de bar erna; long spot, short perp; lening = spot boven "
                              "de equity, rente bekend op de close ervoor; liquidatie op de "
                              "mark price"}


def run_levered(target: pd.DataFrame, m: MarginMarket, costs: BasisCosts, *, lag: int,
                band: float, hedge_tolerance: float, margin: MarginSpec,
                financing: pd.Series, skip: pd.Series | None = None,
                adv_cap: float | None = None, slice_step: float | None = None) -> BookResult:
    """Simuleer het hefboomboek. `target` rij *t* = besluit op de close van *t* (notional
    per been per munt, ≥ 0), uitgevoerd tegen de close van *t + lag − 1*.

    `financing` (jaarrente, bekend op de close van *t*) geldt voor de lening die over bar
    *t + 1* uitstaat. `skip` slaat de herbalancering over op die bars; gedwongen exits en
    liquidaties gaan door.

    `adv_cap`: geen been groter dan die fractie van de 30d-ADV van het DUNSTE been
    (min(spot, perp)), tegen de LOPENDE equity op de uitvoeringsbar
    (`conf/risk/default.yaml::adv_participation_cap`, een marktfeit). Wat een slot door de
    cap niet krijgt, blijft cash.

    `slice_step` (v7): gespreide uitvoering. Een munt die naar een nieuw doel moet (instap,
    uitstap, een resize buiten de band) beweegt per dag hooguit `slice_step` per been, tot
    hij er is; het doel volgt intussen het laatste besluit. In het √-impactmodel kost een
    trade ∝ q^1,5, dus k gelijke stappen kosten √k keer minder impact. Nooit gespreid:
    een verdwenen koers (meteen dicht), een liquidatie, en de governor op het boek zelf.
    Zonder `slice_step` is het boek exact dat van v6."""
    arr = lever_arrays(target, m, lag=lag, financing=financing, skip=skip)
    book = LeveredBook(arr.symbols, costs, band=band, hedge_tolerance=hedge_tolerance,
                       margin=margin, adv_cap=adv_cap, slice_step=slice_step)
    t_len, n = arr.tgt.shape
    out = np.zeros((t_len, len(FRAME_COLUMNS)))
    held = np.zeros((t_len, n))
    trades = np.zeros((t_len, n))
    for t in range(t_len):
        res = book.step(arr.bar(t))
        out[t], held[t], trades[t] = res.row, res.held, res.trades

    frame = pd.DataFrame(out, index=m.index, columns=list(FRAME_COLUMNS))
    identity = (frame[["gross", "funding", "financing", "liquidation", "fees", "spread",
                       "slippage", "impact"]].sum(axis=1) - frame["net"]).abs().max()
    require(float(identity) < 1e-12, "De P&L-trap sluit niet.", DataContractError,
            residual=float(identity))
    syms = list(arr.symbols)
    return BookResult(
        frame=frame, held=pd.DataFrame(held, index=m.index, columns=syms),
        trades=pd.DataFrame(trades, index=m.index, columns=syms), audit=book.audit(lag))
