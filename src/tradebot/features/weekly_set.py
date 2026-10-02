"""De vaste featureset van de wekelijkse meta-label-strategie (spec §7, §17.4-5).

Elke kolom op bar t gebruikt uitsluitend data t/m de close van t; funding en
open interest krijgen daarbovenop één bar lag. Wat gefit wordt, wordt óf vóór
de eerste testperiode gefit (d*), óf rollend op een venster dat op t eindigt
(Kalman/OU, PCA). `tests/lookahead/test_weekly_features_causality.py` bewijst
dat door alles na t weg te laten en rij t te vergelijken.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..alpha.kalman_ou import KalmanOUMeanReversion
from ..data.weekly_market import WeeklyMarket
from ..schemas.config import FracDiffConfig, load_config
from ..utils.failfast import DataContractError, require
from ..volatility.yang_zhang import get_yang_zhang_volatility
from .base import repo_root
from .fracdiff import causal_min_frac_diff, frac_diff_ffd

__all__ = ["FEATURE_COLUMNS", "build_feature_panel", "ffd_threshold",
           "fit_common_d_star", "market_features"]

BAR_FEATURES = (
    "ffd_logp", "ret1_n", "ret5_n", "ret20_n", "dist_hi20", "dist_lo20",
    "kalman_z", "ou_halflife_log", "sigma_ewma", "yz_ratio", "vol_of_vol",
    "dollar_vol_ratio", "funding_z", "oi_chg1", "oi_chg5",
)
MARKET_FEATURES = ("mkt_ret5", "resid_z20", "avg_corr60")
FEATURE_COLUMNS = BAR_FEATURES + MARKET_FEATURES + ("side",)

KALMAN_WINDOW = 126
MIN_LIVE_SYMBOLS = 3


def ffd_threshold() -> float:
    """De FFD-gewichtsdrempel uit `conf/model/fracdiff.yaml` (één bron van waarheid)."""
    return float(load_config(repo_root() / "conf/model/fracdiff.yaml",
                             FracDiffConfig).weight_threshold)


def fit_common_d_star(
    log_close: Mapping[str, pd.Series], *, until: pd.Timestamp, min_obs: int = 250,
    threshold: float | None = None,
) -> float:
    """Eén d* voor alle symbolen: het maximum van de per-symbool d* vóór `until`.

    Alleen symbolen met ten minste `min_obs` bars vóór `until` tellen mee; het
    maximum maakt de reeks voor elk van hen stationair (ADF), en één d voor alle
    symbolen houdt de feature over symbolen vergelijkbaar.
    """
    thr = ffd_threshold() if threshold is None else float(threshold)
    found: list[float] = []
    for series in log_close.values():
        valid = series.dropna()
        n_before = int((valid.index < until).sum())
        if n_before >= min_obs:
            found.append(float(causal_min_frac_diff(valid, train_end_iloc=n_before,
                                                    threshold=thr)))
    require(bool(found), "Geen enkel symbool heeft genoeg historie voor d*.",
            DataContractError, until=str(until), min_obs=min_obs)
    return max(found)


def _rolling_kalman(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Kalman-OU opnieuw gefit op elk venster van 126 bars dat op t eindigt."""
    valid = close.dropna()
    z = np.full(len(close), np.nan)
    hl = np.full(len(close), np.nan)
    frame = valid.to_frame("close")
    where = close.index.get_indexer(valid.index)
    log_v = np.log(valid.to_numpy(dtype=np.float64))
    for pos in range(KALMAN_WINDOW - 1, len(valid)):
        model = KalmanOUMeanReversion(symbol="weekly", init_window=KALMAN_WINDOW)
        model.fit(frame.iloc[pos - KALMAN_WINDOW + 1: pos + 1])
        z[where[pos]] = model.current_zscore(float(log_v[pos]))
        hl[where[pos]] = model.current_halflife()
    return pd.Series(z, index=close.index), pd.Series(hl, index=close.index)


def _symbol_features(
    ohlcv: pd.DataFrame, sigma: pd.Series, funding: pd.Series,
    open_interest: pd.Series, *, d_star: float, threshold: float,
) -> pd.DataFrame:
    idx = ohlcv.index
    close = ohlcv["close"]
    logp = np.log(close)
    out = pd.DataFrame(index=idx)
    out["ffd_logp"] = frac_diff_ffd(logp.dropna(), d_star, threshold=threshold).reindex(idx)
    for h in (1, 5, 20):
        out[f"ret{h}_n"] = (logp - logp.shift(h)) / (sigma * np.sqrt(h))
    out["dist_hi20"] = (logp - np.log(ohlcv["high"].rolling(20).max())) / sigma
    out["dist_lo20"] = (logp - np.log(ohlcv["low"].rolling(20).min())) / sigma
    z, hl = _rolling_kalman(close)
    out["kalman_z"] = z
    out["ou_halflife_log"] = np.log(hl.clip(lower=1.0, upper=500.0))
    out["sigma_ewma"] = sigma
    valid = ohlcv[["open", "high", "low", "close"]].dropna()
    yz = get_yang_zhang_volatility(valid, window=20).reindex(idx)
    out["yz_ratio"] = yz / yz.rolling(120, min_periods=120).mean()
    out["vol_of_vol"] = np.log(sigma).diff().rolling(30, min_periods=30).std()
    turnover = ohlcv["turnover"]
    out["dollar_vol_ratio"] = turnover / turnover.rolling(30, min_periods=30).mean().shift(1)
    f3 = funding.rolling(3, min_periods=3).sum()
    f3_z = (f3 - f3.rolling(90, min_periods=90).mean()) / f3.rolling(90, min_periods=90).std()
    out["funding_z"] = f3_z.shift(1)
    log_oi = np.log(open_interest.where(open_interest > 0.0))
    out["oi_chg1"] = log_oi.diff(1).shift(1)
    out["oi_chg5"] = log_oi.diff(5).shift(1)
    return out


def market_features(log_returns: pd.DataFrame, *, window: int) -> dict[str, pd.DataFrame]:
    """PCA-marktfactor, residu-z-score en gemiddelde correlatie, rollend op `window` bars t/m t."""
    idx, cols = log_returns.index, list(log_returns.columns)
    values = log_returns.to_numpy(dtype=np.float64)
    mkt = np.full(values.shape, np.nan)
    resid = np.full(values.shape, np.nan)
    corr = np.full(values.shape, np.nan)
    for pos in range(window - 1, len(idx)):
        block = values[pos - window + 1: pos + 1]
        live = np.flatnonzero(np.isfinite(block).all(axis=0))
        if live.size < MIN_LIVE_SYMBOLS:
            continue
        x = block[:, live]
        std = x.std(axis=0, ddof=1)
        if np.any(std <= 0.0):
            continue
        zs = (x - x.mean(axis=0)) / std
        c = np.corrcoef(x, rowvar=False)
        _, vecs = np.linalg.eigh(c)
        pc1 = vecs[:, -1] if vecs[:, -1].sum() >= 0.0 else -vecs[:, -1]
        pc1 = pc1 / np.abs(pc1).sum()
        f = zs @ pc1
        fc = f - f.mean()
        beta = (zs * fc[:, None]).sum(axis=0) / float((fc ** 2).sum())
        e = zs - np.outer(f, beta)
        n_live = live.size
        mkt[pos, live] = f[-5:].sum() / (f.std(ddof=1) * np.sqrt(5.0))
        resid[pos, live] = e[-20:].sum(axis=0) / (e.std(axis=0, ddof=1) * np.sqrt(20.0))
        corr[pos, live] = (c.sum() - n_live) / (n_live * (n_live - 1))
    return {name: pd.DataFrame(arr, index=idx, columns=cols)
            for name, arr in (("mkt_ret5", mkt), ("resid_z20", resid), ("avg_corr60", corr))}


def build_feature_panel(
    market: WeeklyMarket, side: Mapping[str, pd.Series], *, d_star: float, corr_window: int,
    threshold: float | None = None,
) -> dict[str, pd.DataFrame]:
    """De featurematrix per symbool, op het raster van de markt, in `FEATURE_COLUMNS`-volgorde."""
    thr = ffd_threshold() if threshold is None else float(threshold)
    closes = pd.DataFrame({s: market.ohlcv[s]["close"] for s in market.symbols})
    mf = market_features(np.log(closes).diff(), window=corr_window)
    out: dict[str, pd.DataFrame] = {}
    for s in market.symbols:
        frame = _symbol_features(market.ohlcv[s], market.sigma_daily[s], market.funding[s],
                                 market.open_interest[s], d_star=d_star, threshold=thr)
        for name, panel in mf.items():
            frame[name] = panel[s]
        frame["side"] = side[s].reindex(market.grid)
        out[s] = frame[list(FEATURE_COLUMNS)]
    return out
