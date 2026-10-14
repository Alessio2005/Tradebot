"""Robuust boek v2: het brede, survivorship-vrije universum en zijn sleeves.

Ontwerp: `docs/superpowers/specs/2026-10-07-robust-book-v2-breadth-design.md`. Alles hier
is causaal: een besluit op de close van *t* gebruikt alleen data die op die close bekend
is. Dat wordt getoetst in `tests/lookahead/test_breadth_causality.py`.

Het verschil met v1 zit in drie dingen:

* een **point-in-time-universum** (top-N naar ADV, met een vol-vloer tegen gekoppelde
  perps), zodat gedeliste munten meedoen zolang ze bestonden;
* **cross-sectionele** sleeves (momentum, funding-carry) die alleen bij breedte zinvol zijn;
* een **stromende EWMA-covariantie** voor het vol-doel. Een (T, N, N)-tensor over
  honderden munten past niet in het geheugen.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..schemas.robust_book_v2 import RobustBookV2Config
from ..utils.failfast import DataContractError, require
from .market import BARS_PER_YEAR, BookMarket, ewma_vol
from .signals import trend_score
from .sleeves import SleeveTargets, apply_caps

__all__ = [
    "BreadthMarket",
    "breadth_from_frames",
    "chl_half_spread",
    "combine_scaled",
    "load_breadth_market",
    "scale_to_vol",
    "sleeve_targets",
    "universe_mask",
]

#: RiskMetrics-lambda voor de dagvolatiliteit in het impactmodel (conf/model/volatility.yaml).
IMPACT_EWMA_LAMBDA = 0.94


@dataclass(frozen=True)
class BreadthMarket:
    """Een `BookMarket` plus wat alleen v2 nodig heeft."""

    book: BookMarket
    high: pd.DataFrame
    low: pd.DataFrame
    quote_volume: pd.DataFrame
    half_spread: pd.DataFrame
    universe: pd.DataFrame

    def subset(self, symbols: Sequence[str]) -> BreadthMarket:
        cols = list(symbols)
        return BreadthMarket(
            book=self.book.subset(tuple(cols)), high=self.high[cols], low=self.low[cols],
            quote_volume=self.quote_volume[cols], half_spread=self.half_spread[cols],
            universe=self.universe[cols])

    def truncate(self, end: pd.Timestamp) -> BreadthMarket:
        keep = self.book.index < end
        return BreadthMarket(
            book=self.book.truncate(end), high=self.high.loc[keep], low=self.low.loc[keep],
            quote_volume=self.quote_volume.loc[keep], half_spread=self.half_spread.loc[keep],
            universe=self.universe.loc[keep])


def chl_half_spread(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, *,
                    window: int, floor: float, cap: float) -> pd.DataFrame:
    """Halve spread uit de Abdi-Ranaldo (2017) CHL-schatter, causaal.

    `s² = 4·E[(c_{t−1} − η_{t−1})(c_{t−1} − η_t)]`, met c = log-close en η = log-midrange.
    Het paar dat op bar *t* eindigt, gebruikt alleen *t−1* en *t*: bekend op de close van *t*.
    Een negatief gemiddelde is geen negatieve spread maar ruis: dan geldt de vloer.
    """
    c = np.log(close)
    eta = (np.log(high) + np.log(low)) / 2.0
    prod = (c.shift(1) - eta.shift(1)) * (c.shift(1) - eta)
    mean = prod.rolling(window, min_periods=max(window // 2, 10)).mean()
    spread = np.sqrt((4.0 * mean).clip(lower=0.0))
    return (spread / 2.0).clip(lower=floor, upper=cap)


def universe_mask(close: pd.DataFrame, quote_volume: pd.DataFrame, cfg: RobustBookV2Config) -> pd.DataFrame:
    """U_t: genoteerd, oud genoeg, liquide genoeg, niet gekoppeld, top-N naar ADV (spec §2)."""
    u = cfg.universe
    history = close.notna().cumsum()
    adv = quote_volume.rolling(u.adv_window, min_periods=u.adv_window).mean()
    ret = np.log(close).diff()
    vol = ret.rolling(u.vol_window, min_periods=u.vol_window).std() * np.sqrt(BARS_PER_YEAR)
    eligible = (close.notna() & (history >= u.min_history_bars) & (adv >= u.min_adv_usd)
                & (vol >= u.min_ann_vol))
    score = adv.where(eligible)
    # Rang aflopend op ADV; gelijke ADV: de symboolnaam beslist (kolommen gesorteerd).
    cols = sorted(score.columns)
    s = score[cols]
    rank = s.rank(axis=1, ascending=False, method="first")
    return (rank <= u.top_n).reindex(columns=close.columns).fillna(False)


MAX_BRIDGED_GAP = 7


def bridge_data_holes(frame: pd.DataFrame, close: pd.DataFrame, *,
                      max_gap: int = MAX_BRIDGED_GAP) -> pd.DataFrame:
    """Vul interne gaten van ≤ `max_gap` bars binnen de noteringsperiode van `close` met ffill."""
    out = frame.copy()
    for s in close.columns:
        col = close[s]
        if col.notna().sum() == 0:
            continue
        inside = (col.index >= col.first_valid_index()) & (col.index <= col.last_valid_index())
        missing = col.isna() & inside
        if not missing.any():
            continue
        run_id = (~missing).cumsum()
        run_len = missing.groupby(run_id).transform("sum")
        short = missing & (run_len <= max_gap)
        filled = out[s].ffill()
        out[s] = out[s].where(~short, filled)
    return out


def breadth_from_frames(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                        quote_volume: pd.DataFrame, funding: pd.DataFrame,
                        cfg: RobustBookV2Config, *, source: str) -> BreadthMarket:
    """Vorm U_t en beperk de kolommen tot munten die ooit in U_t zaten.

    De beperking is survivorship-vrij: lidmaatschap van U_t is point-in-time, en een
    munt die ooit lid was, blijft in het paneel tot zijn laatste bar.

    Datagaten van ten hoogste `MAX_BRIDGED_GAP` bars BINNEN de noteringsperiode (de bron
    mist dan dagen voor tientallen munten tegelijk, bv. 2022-02-27 .. 03-01) worden
    overbrugd met de laatst bekende waarde: rendement 0 op de ontbrekende dagen, de
    beweging valt op de eerste bar erna. Een langer gat geldt als delisting plus
    herintrede. Het overbruggen kijkt alleen terug (ffill), maar weet wel dat de munt na
    het gat terugkomt; dat is een eigenschap van de bron, geen handelsinformatie.
    """
    close, high, low, quote_volume = (bridge_data_holes(x, close) for x in (
        close, high, low, quote_volume))
    universe = universe_mask(close, quote_volume, cfg)
    members = [s for s in close.columns if bool(universe[s].any())]
    require(len(members) >= 4, "Te weinig munten ooit in het universum.",
            DataContractError, n=len(members))
    close = close[members]
    ret = close.pct_change(fill_method=None)
    listed = close.notna()
    funding = funding[members].where(listed, 0.0).fillna(0.0)
    adv = quote_volume[members].rolling(cfg.universe.adv_window,
                                        min_periods=cfg.universe.adv_window).mean()
    sigma = np.sqrt((ret ** 2).ewm(alpha=1.0 - IMPACT_EWMA_LAMBDA, min_periods=30).mean())
    c = cfg.costs
    hs = chl_half_spread(close, high[members], low[members], window=c.spread_window,
                         floor=c.min_half_spread_bps * 1e-4, cap=c.max_half_spread_bps * 1e-4)
    book = BookMarket(close=close, ret=ret, funding=funding, adv_usd=adv, sigma_daily=sigma,
                      source_hashes={"breadth": source})
    return BreadthMarket(book=book, high=high[members], low=low[members],
                         quote_volume=quote_volume[members], half_spread=hs,
                         universe=universe[members] & sigma.notna() & adv.gt(0.0))


def load_breadth_market(panel_dir: Path, cfg: RobustBookV2Config) -> BreadthMarket:
    """Laad de Binance-panelen (`data/binance_vision.py::build_panels`)."""
    read = {k: pd.read_parquet(panel_dir / f"{k}.parquet")
            for k in ("close", "high", "low", "quote_volume", "funding")}
    return breadth_from_frames(read["close"], read["high"], read["low"], read["quote_volume"],
                               read["funding"], cfg, source=str(panel_dir))


# --------------------------------------------------------------------------- #
# Vol-doel met een stromende covariantie
# --------------------------------------------------------------------------- #
def scale_to_vol(raw: pd.DataFrame, ret: pd.DataFrame, rebalance: pd.Series, *,
                 span: int, vol_target: float, max_scale: float) -> pd.DataFrame:
    """Schaal elke besluitrij naar het vol-doel; rijen zonder besluit worden NaN.

    De EWMA-tweede-momentmatrix wordt bar voor bar bijgewerkt met het rendement van die
    bar (bekend op zijn close) en VÓÓR het besluit van die bar gebruikt: causaal. Een munt
    met minder dan `span//3` waarnemingen heeft nog geen covariantie; staat hij in het
    boek, dan wordt die rij niet geschaald (geen besluit).
    """
    x = ret.to_numpy(dtype=np.float64)
    w = raw.to_numpy(dtype=np.float64)
    reb = rebalance.to_numpy(dtype=bool)
    t_len, n = x.shape
    alpha = 2.0 / (span + 1.0)
    min_obs = max(span // 3, 10)
    s = np.zeros((n, n))
    norm = 0.0
    seen = np.zeros(n)
    out = np.full_like(w, np.nan)
    for t in range(t_len):
        row = x[t]
        ok = np.isfinite(row)
        seen += ok
        r = np.where(ok, row, 0.0)
        s *= 1.0 - alpha
        s += alpha * np.outer(r, r)
        norm = (1.0 - alpha) * norm + alpha
        if not reb[t]:
            continue
        wt = np.nan_to_num(w[t])
        nz = np.abs(wt) > 0.0
        if not nz.any():
            out[t] = 0.0
            continue
        if (seen[nz] < min_obs).any():
            continue
        cov = s[np.ix_(nz, nz)] / norm
        var = float(wt[nz] @ cov @ wt[nz]) * BARS_PER_YEAR
        if var <= 0.0:
            continue
        out[t] = wt * min(vol_target / np.sqrt(var), max_scale)
    return pd.DataFrame(out, index=raw.index, columns=raw.columns)


def _fixed_phase(index: pd.Index, valid: pd.Series, every: int) -> pd.Series:
    first = int(np.argmax(valid.to_numpy())) if bool(valid.any()) else len(index)
    pos = np.arange(len(index))
    return pd.Series((pos >= first) & ((pos - first) % every == 0), index=index)


def _legs(score: pd.DataFrame, inv_vol: pd.DataFrame) -> pd.DataFrame:
    """Bovenste helft long, onderste helft short; inverse-vol binnen elk been, elk been bruto 0,5."""
    rank = score.rank(axis=1, method="first")
    n = score.notna().sum(axis=1)
    half = (n // 2).to_numpy()[:, None]
    r = rank.to_numpy()
    longs = pd.DataFrame(r > (n.to_numpy()[:, None] - half), index=score.index,
                         columns=score.columns)
    shorts = pd.DataFrame(r <= half, index=score.index, columns=score.columns)
    lw = inv_vol.where(longs)
    sw = inv_vol.where(shorts)
    out = (lw.div(lw.sum(axis=1), axis=0).fillna(0.0) * 0.5
           - sw.div(sw.sum(axis=1), axis=0).fillna(0.0) * 0.5)
    # Minder dan vier namen: geen cross-sectie om op te handelen.
    return out.mul((n >= 4).astype(float), axis=0)


def _winsor_z(x: pd.DataFrame, clip: float) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0).replace(0.0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def sleeve_targets(name: str, m: BreadthMarket, cfg: RobustBookV2Config, *,
                   vol_span: int | None = None, max_scale: float | None = None,
                   xsmom_lookbacks: Sequence[int] | None = None,
                   trend_lookbacks: Sequence[int] | None = None,
                   carry_window: int | None = None,
                   rebalance_bars: int | None = None) -> SleeveTargets:
    """Bouw één sleeve (X1..X4) of referentie (R1, R2). Overrides zijn voor de batterij."""
    b = m.book
    span = int(vol_span or cfg.sizing.vol_span)
    kmax = float(max_scale or cfg.sizing.max_scale)
    sigma = ewma_vol(b.ret, span=span)
    sigma_ann = sigma * np.sqrt(BARS_PER_YEAR)
    u = m.universe & sigma.notna()
    inv = (1.0 / sigma_ann).where(u)
    idx = b.index

    def finish(raw: pd.DataFrame, rebalance: pd.Series) -> SleeveTargets:
        scaled = scale_to_vol(raw, b.ret, rebalance, span=span,
                              vol_target=cfg.sizing.portfolio_vol_target, max_scale=kmax)
        capped = apply_caps(scaled.fillna(0.0), per_asset_cap=cfg.sizing.per_asset_cap,
                            gross_cap=cfg.sizing.gross_cap).where(scaled.notna())
        reb = rebalance & capped.notna().all(axis=1)
        return SleeveTargets(name=name, weights=capped.where(reb, np.nan, axis=0), rebalance=reb)

    if name == "R1_BTC_HOLD":
        w = pd.DataFrame(0.0, index=idx, columns=b.symbols)
        w["BTCUSDT"] = 1.0
        w = w.where(b.live(90) & b.close.notna(), 0.0)
        return SleeveTargets(name=name, weights=w, rebalance=pd.Series(True, index=idx))
    if name == "R2_EW50_HOLD":
        n = m.universe.sum(axis=1).replace(0, np.nan)
        w = m.universe.astype(float).div(n, axis=0).fillna(0.0)
        return SleeveTargets(name=name, weights=w, rebalance=pd.Series(True, index=idx))

    if name == "X1_XSMOM":
        logp = np.log(b.close)
        parts = []
        for lb in (xsmom_lookbacks or cfg.xsmom.lookbacks):
            z = ((logp - logp.shift(int(lb))) / (sigma * np.sqrt(float(lb)))).where(u)
            parts.append(_winsor_z(z, cfg.xsmom.winsor))
        score = sum(p.fillna(0.0) for p in parts) / float(len(parts))
        score = score.where(u & parts[0].notna())
        every = int(rebalance_bars or cfg.xsmom.rebalance_bars)
        raw = _legs(score, inv)
        return finish(raw, _fixed_phase(idx, score.notna().sum(axis=1) >= 4, every))

    if name == "X2_XSCARRY":
        window = int(carry_window or cfg.xscarry.window_bars)
        listed = b.close.notna()
        carry = (b.funding.where(listed).rolling(window, min_periods=window).mean()
                 .shift(cfg.xscarry.signal_lag_bars)).where(u)
        every = int(rebalance_bars or cfg.xscarry.rebalance_bars)
        raw = _legs(-carry, inv)  # hoogste funding = laagste score = short
        return finish(raw, _fixed_phase(idx, carry.notna().sum(axis=1) >= 4, every))

    if name in ("X3_TREND_LS", "X4_TREND_LF"):
        score = trend_score(b.close, sigma, lookbacks=tuple(trend_lookbacks or cfg.trend.lookbacks),
                            z_clip=cfg.trend.z_clip).where(u)
        if name == "X4_TREND_LF":
            score = score.clip(lower=0.0)
        n = score.notna().sum(axis=1).replace(0, np.nan)
        raw = (score * (cfg.sizing.asset_vol_target / sigma_ann)).div(n, axis=0).fillna(0.0)
        every = int(rebalance_bars or 1)
        return finish(raw, _fixed_phase(idx, score.notna().sum(axis=1) >= 1, every))

    raise DataContractError(f"Onbekende sleeve {name!r}.")


def combine_scaled(name: str, parts: Sequence[SleeveTargets], m: BreadthMarket,
                   cfg: RobustBookV2Config, *, vol_span: int | None = None,
                   max_scale: float | None = None) -> SleeveTargets:
    """Gelijk gewogen som van sleeves (elk op het vol-doel), opnieuw naar het vol-doel."""
    require(len(parts) >= 1, "Een combinatie zonder onderdelen.", DataContractError)
    b = m.book
    total = sum(p.weights.ffill().fillna(0.0) for p in parts) / float(len(parts))
    total = total.where(b.close.notna(), 0.0)
    reb = pd.Series(True, index=b.index)
    scaled = scale_to_vol(total, b.ret, reb, span=int(vol_span or cfg.sizing.vol_span),
                          vol_target=cfg.sizing.portfolio_vol_target,
                          max_scale=float(max_scale or cfg.sizing.max_scale))
    capped = apply_caps(scaled.fillna(0.0), per_asset_cap=cfg.sizing.per_asset_cap,
                        gross_cap=cfg.sizing.gross_cap).where(scaled.notna())
    reb = capped.notna().all(axis=1)
    return SleeveTargets(name=name, weights=capped.where(reb, np.nan, axis=0), rebalance=reb)
