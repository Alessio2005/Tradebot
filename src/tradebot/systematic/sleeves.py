"""Van signaal naar doelgewicht: per-munt-sizing, portefeuille-vol-doel en limieten.

Een gewicht is een fractie van de equity in notioneel (signed). Het doelgewicht op rij
*t* is het besluit op de close van *t*; `book.run_book` voert het uit. Rijen met NaN
zijn bars zonder besluit (het boek laat de posities dan driften).

Sizing in drie stappen, gelijk voor elke sleeve (spec §4):

1. per munt `w_i = S_i · (τ_a / σ_i,jaar) / N`, met N het aantal munten waarover de
   sleeve op die bar iets zegt;
2. één portefeuillefactor `k = min(τ_p / σ̂_p, k_max)`, met `σ̂_p = sqrt(365 · wᵀΣw)`
   uit de causale EWMA-covariantie;
3. |w_i| ≤ `per_asset_cap` en Σ|w_i| ≤ `gross_cap`, beide door het HELE boek
   proportioneel te schalen, zodat een dollar-neutraal boek dollar-neutraal blijft.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schemas.robust_book import RobustBookConfig
from ..utils.failfast import DataContractError, require
from .market import BARS_PER_YEAR, BookMarket, ewma_covariance, ewma_vol
from .signals import carry_ranks, known_carry, trend_score

__all__ = [
    "SleeveTargets",
    "apply_caps",
    "baseline_btc",
    "baseline_equal_weight",
    "build_sleeve",
    "combine",
    "portfolio_scale",
    "size_by_inverse_vol",
]


@dataclass(frozen=True)
class SleeveTargets:
    """Doelgewichten per besluitbar en het herbalanceringsschema."""

    name: str
    weights: pd.DataFrame
    rebalance: pd.Series

    def __post_init__(self) -> None:
        require(bool(self.weights.index.equals(self.rebalance.index)),
                "Gewichten en herbalanceringsschema delen geen tijdas.", DataContractError,
                sleeve=self.name)


@dataclass(frozen=True)
class SizingParams:
    vol_span: int
    min_history_bars: int
    asset_vol_target: float
    portfolio_vol_target: float
    max_scale: float
    per_asset_cap: float
    gross_cap: float

    @classmethod
    def from_config(cls, cfg: RobustBookConfig) -> SizingParams:
        s = cfg.sizing
        return cls(vol_span=s.vol_span, min_history_bars=s.min_history_bars,
                   asset_vol_target=s.asset_vol_target,
                   portfolio_vol_target=s.portfolio_vol_target, max_scale=s.max_scale,
                   per_asset_cap=s.per_asset_cap, gross_cap=s.gross_cap)


def size_by_inverse_vol(
    score: pd.DataFrame, sigma_daily: pd.DataFrame, *, asset_vol_target: float,
) -> pd.DataFrame:
    """`S_i · (τ_a / σ_i,jaar) / N`; N = aantal munten met een eindige score op die bar."""
    sigma_ann = sigma_daily * np.sqrt(BARS_PER_YEAR)
    active = score.notna() & sigma_ann.gt(0.0)
    n = active.sum(axis=1).replace(0, np.nan)
    raw = score.where(active) * (asset_vol_target / sigma_ann)
    return raw.div(n, axis=0).where(active, 0.0).fillna(0.0)


def portfolio_scale(
    raw: pd.DataFrame, cov: np.ndarray, *, vol_target: float, max_scale: float,
) -> pd.DataFrame:
    """Schaal elke rij naar het vol-doel met de covariantie van die bar, k ≤ `max_scale`."""
    w = raw.to_numpy(dtype=np.float64)
    require(cov.shape == (w.shape[0], w.shape[1], w.shape[1]),
            "Covariantie en gewichten hebben een andere vorm.", DataContractError)
    out = np.zeros_like(w)
    for t in range(w.shape[0]):
        row = w[t]
        nz = np.abs(row) > 0.0
        if not nz.any():
            continue
        sub = cov[t][np.ix_(nz, nz)]
        if not np.isfinite(sub).all():
            continue
        var = float(row[nz] @ sub @ row[nz]) * BARS_PER_YEAR
        if var <= 0.0:
            continue
        k = min(vol_target / np.sqrt(var), max_scale)
        out[t] = row * k
    return pd.DataFrame(out, index=raw.index, columns=raw.columns)


def apply_caps(w: pd.DataFrame, *, per_asset_cap: float, gross_cap: float) -> pd.DataFrame:
    """Schaal een rij als geheel terug tot |w_i| ≤ per_asset_cap en Σ|w_i| ≤ gross_cap."""
    a = w.abs()
    f_asset = (per_asset_cap / a.max(axis=1)).clip(upper=1.0)
    f_gross = (gross_cap / a.sum(axis=1)).clip(upper=1.0)
    factor = pd.concat([f_asset, f_gross], axis=1).min(axis=1).fillna(1.0)
    return w.mul(factor, axis=0)


def _finish(name: str, raw: pd.DataFrame, cov: np.ndarray, p: SizingParams,
            rebalance: pd.Series) -> SleeveTargets:
    scaled = portfolio_scale(raw, cov, vol_target=p.portfolio_vol_target,
                             max_scale=p.max_scale)
    capped = apply_caps(scaled, per_asset_cap=p.per_asset_cap, gross_cap=p.gross_cap)
    return SleeveTargets(name=name, weights=capped.where(rebalance, np.nan, axis=0),
                         rebalance=rebalance)


def _daily(index: pd.Index) -> pd.Series:
    return pd.Series(True, index=index)


def build_sleeve(
    name: str,
    market: BookMarket,
    cfg: RobustBookConfig,
    *,
    sizing: SizingParams | None = None,
    lookbacks: Sequence[int] | None = None,
    carry_window: int | None = None,
    carry_rebalance: int | None = None,
) -> SleeveTargets:
    """Bouw één kandidaat-sleeve (C1..C4). De keyword-overrides bestaan voor de
    robuustheidsbatterij; de basisrun gebruikt uitsluitend `cfg`."""
    p = sizing or SizingParams.from_config(cfg)
    sigma = ewma_vol(market.ret, span=p.vol_span)
    live = market.live(p.min_history_bars)
    cov = ewma_covariance(market.ret, span=p.vol_span)
    idx = market.index

    if name in ("C1_TREND_LS", "C2_TREND_LF"):
        score = trend_score(market.close, sigma,
                            lookbacks=tuple(lookbacks or cfg.trend.lookbacks),
                            z_clip=cfg.trend.z_clip).where(live)
        if name == "C2_TREND_LF":
            score = score.clip(lower=0.0)
        raw = size_by_inverse_vol(score, sigma, asset_vol_target=p.asset_vol_target)
        return _finish(name, raw, cov, p, _daily(idx))

    if name == "C3_VOLMAN_CORE":
        core = [s for s in market.symbols if s in cfg.core.assets]
        score = pd.DataFrame(np.nan, index=idx, columns=market.symbols)
        score[core] = 1.0
        score = score.where(live)
        raw = size_by_inverse_vol(score, sigma, asset_vol_target=p.asset_vol_target)
        return _finish(name, raw, cov, p, _daily(idx))

    if name == "C4_CARRY_XS":
        window = int(carry_window or cfg.carry.window_bars)
        every = int(carry_rebalance or cfg.carry.rebalance_bars)
        listed = market.close.notna()
        carry = known_carry(market.funding, listed, window_bars=window,
                            signal_lag_bars=cfg.carry.signal_lag_bars)
        ok = (carry.notna() & live).sum(axis=1) >= 2
        require(bool(ok.any()), "Geen enkele bar met twee levende carry's.", DataContractError)
        first = int(np.argmax(ok.to_numpy()))
        positions = np.arange(len(idx))
        rebalance = pd.Series((positions >= first) & ((positions - first) % every == 0),
                              index=idx)
        ranks = carry_ranks(carry, live)
        inv = (1.0 / (sigma * np.sqrt(BARS_PER_YEAR))).where(live)
        long_leg = inv.where(ranks > 0.0)
        short_leg = inv.where(ranks < 0.0)
        raw = (long_leg.div(long_leg.sum(axis=1), axis=0).fillna(0.0) * 0.5
               - short_leg.div(short_leg.sum(axis=1), axis=0).fillna(0.0) * 0.5)
        return _finish(name, raw, cov, p, rebalance)

    raise DataContractError(f"Onbekende sleeve {name!r}.")


def baseline_btc(market: BookMarket, cfg: RobustBookConfig) -> SleeveTargets:
    """B1: 1,0x long BTC-perp, dagelijks terug naar 1,0."""
    w = pd.DataFrame(0.0, index=market.index, columns=market.symbols)
    w["BTCUSDT"] = 1.0
    w = w.where(market.close.notna(), 0.0)
    return SleeveTargets(name="B1_BTC_HOLD", weights=w, rebalance=_daily(market.index))


def baseline_equal_weight(market: BookMarket, cfg: RobustBookConfig) -> SleeveTargets:
    """B2: gelijk gewogen long over de levende munten, bruto 1,0."""
    live = market.live(cfg.sizing.min_history_bars)
    n = live.sum(axis=1).replace(0, np.nan)
    w = live.astype(float).div(n, axis=0).fillna(0.0)
    return SleeveTargets(name="B2_EW_HOLD", weights=w, rebalance=_daily(market.index))


def combine(
    name: str,
    parts: Sequence[SleeveTargets],
    market: BookMarket,
    cfg: RobustBookConfig,
    *,
    sizing: SizingParams | None = None,
) -> SleeveTargets:
    """Gelijk gewogen som van de sleeves (elk al op het vol-doel), dan opnieuw geschaald.

    Een sleeve die niet dagelijks herbalanceert, levert tussen zijn besluiten zijn laatste
    doelgewicht (vooruit gevuld); in de combinatie herbalanceert het boek dagelijks.
    """
    require(len(parts) >= 1, "Een combinatie zonder onderdelen.", DataContractError)
    p = sizing or SizingParams.from_config(cfg)
    cov = ewma_covariance(market.ret, span=p.vol_span)
    total = sum(s.weights.ffill().fillna(0.0) for s in parts) / float(len(parts))
    return _finish(name, total, cov, p, _daily(market.index))
