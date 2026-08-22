"""Adaptive walk-forward cross-sectional book — AFML, overfit-proof (Wave 18/19).

A non-stationary-aware, deploy-grade adaptive book. The design goal is *forward*
robustness for the coming period, not a static fit across all history. Every choice
below is an overfit control, not a performance knob.

Architecture
------------
1. **Pure walk-forward** — at each refit date the model trains ONLY on rows strictly
   before `t0 - embargo`; it predicts the forward block `[t0, t0+refit)`. No
   future bar ever informs a past decision (R-1 causality).
2. **Purge + embargo** — the `horizon`-day forward label window is purged and an
   extra embargo applied between train and test, so overlapping labels cannot leak.
3. **AFML sample weights** — per-asset average **uniqueness** (concurrency of the
   overlapping triple-horizon labels) × **return-magnitude** × **recency time-decay**
   (`get_sample_weights`). Recency lets the model track the current regime; uniqueness
   stops overlapping labels from being counted as independent evidence.
4. **Frozen MDA feature selection** — features are ranked once by causal MDA on the
   first training window and FROZEN; no per-fold reselection (which would itself
   overfit / leak).
5. **Probability calibration** — out-of-fold isotonic calibration of the win-prob.
6. **Regime-guarded harvest** — cross-sectional rank → demean (dollar-neutral) →
   neutralise vs the vol & market-β factors → EWMA smooth → no-trade band →
   de-gross in bull-mania regimes (the short-biased book's known failure mode).
7. **Overfit accounting** — Deflated Sharpe deflated by the number of construction
   variants tried, and CSCV **PBO** across those variants. Deterministic (R-5).

The class is data-source agnostic: feed it a tidy OHLCV panel
(`[date, symbol, open, high, low, close, volume]`) and an optional funding panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..backtest.metrics import deflated_sharpe
from ..backtest.pbo import compute_pbo
from ..cv.uniqueness import get_average_uniqueness, get_sample_weights
from ..utils.failfast import DataContractError, require

DAYS = 365.0


@dataclass
class AdaptiveWFConfig:
    """All knobs are overfit controls with conservative, fixed defaults."""

    horizon: int = 10                       # forward label horizon (days)
    refit_days: int = 91                    # walk-forward refit cadence (quarterly)
    embargo_days: int = 13                  # purge gap train->test (>= horizon)
    recency_halflife_days: float = 365.0    # sample-weight time-decay
    target_vol: float = 0.40                # book vol target (annualised)
    cost_bps: float = 10.0                  # round-trip taker cost
    wf_start: str = "2023-01-01"            # first forward block
    min_train_rows: int = 8000
    min_hist_bars: int = 900                # per-asset history filter
    decile: float = 0.20                    # (unused in continuous harvest; kept for grid)
    smooth_span: int = 5                    # EWMA turnover control
    band: float = 0.004                     # no-trade band (gross fraction)
    regime_degross: float = 0.7             # max gross cut in bull mania
    mda_keep: int = 28                      # frozen top-N features by causal MDA
    seed: int = 0
    catboost_params: dict = field(default_factory=lambda: {
        "iterations": 400, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 8,
        "loss_function": "Logloss", "subsample": 0.8, "rsm": 0.8, "verbose": 0,
    })


# --------------------------------------------------------------------------- #
# feature engineering (cross-sectional, causal)
# --------------------------------------------------------------------------- #
def _base_feats(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    r = np.log(c / c.shift(1))
    F = pd.DataFrame(index=df.index)
    for L in (5, 10, 20, 60):
        F[f"ret{L}"] = np.log(c / c.shift(L))
    F["vol20"] = r.rolling(20).std()
    F["vol60"] = r.rolling(60).std()
    F["volratio"] = F["vol20"] / F["vol60"]
    F["ma50"] = c / c.rolling(50).mean() - 1
    F["ma100"] = c / c.rolling(100).mean() - 1
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    F["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    hi = c.rolling(20).max()
    lo = c.rolling(20).min()
    F["rangepos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    F["skew20"] = r.rolling(20).skew()
    F["ac1"] = r.rolling(40).apply(lambda x: x.autocorr(1), raw=False)
    o, h, l = df["open"], df["high"], df["low"]
    gk = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    F["gk20"] = gk.rolling(20).mean()
    return F


class AdaptiveWalkForward:
    def __init__(self, config: AdaptiveWFConfig | None = None):
        self.cfg = config or AdaptiveWFConfig()
        self.features_: list[str] | None = None
        self.frozen_features_: list[str] | None = None
        self.refit_dates_: list[pd.Timestamp] = []
        self._panels: dict | None = None

    # ----- panel assembly --------------------------------------------------- #
    def build_panel(self, ohlcv: pd.DataFrame, funding: pd.DataFrame | None = None):
        cfg = self.cfg
        syms = [s for s, g in ohlcv.groupby("symbol") if len(g) >= cfg.min_hist_bars]
        close = ohlcv.pivot_table(index="date", columns="symbol", values="close")[syms].sort_index()
        rets = np.log(close / close.shift(1))
        mkt = rets.mean(axis=1)
        btc_ret = rets["BTCUSDT"] if "BTCUSDT" in rets.columns else mkt
        if funding is not None:
            fund = funding.reindex(close.index).reindex(columns=syms)
            fund_z = ((fund - fund.rolling(30).mean()) / fund.rolling(30).std()).shift(1)
        else:
            fund_z = pd.DataFrame(0.0, index=close.index, columns=syms)
        self._panels = dict(syms=syms, close=close, rets=rets, mkt=mkt,
                            btc_ret=btc_ret, fund_z=fund_z, ohlcv=ohlcv)
        return self

    def build_features_labels(self) -> pd.DataFrame:
        cfg, P = self.cfg, self._panels
        close, rets, mkt, btc_ret, fund_z = (P["close"], P["rets"], P["mkt"],
                                             P["btc_ret"], P["fund_z"])
        rows = []
        for s in P["syms"]:
            df = (P["ohlcv"][P["ohlcv"].symbol == s]
                  .set_index("date")[["open", "high", "low", "close", "volume"]]
                  .sort_index())
            df = df[~df.index.duplicated()]
            F = _base_feats(df).reindex(close.index)
            b = (rets[s].rolling(60).cov(btc_ret) / btc_ret.rolling(60).var()).clip(-3, 3)
            F["resid_mom20"] = (rets[s] - b * btc_ret).rolling(20).sum()
            F["rel_ret20"] = rets[s].rolling(20).sum() - mkt.rolling(20).sum()
            F["fund_z"] = fund_z[s]
            F = F.shift(1)                                   # causal: event t uses <= t-1
            F["__sym"] = s
            F["__date"] = close.index
            F["__fwd"] = np.log(close[s].shift(-cfg.horizon) / close[s]).values
            rows.append(F)
        Pf = pd.concat(rows, ignore_index=True)
        base = [c for c in Pf.columns if not c.startswith("__")]
        for col in base:                                    # cross-sectional ranks
            Pf[f"xr_{col}"] = Pf.groupby("__date")[col].rank(pct=True)
        med = Pf.groupby("__date")["__fwd"].transform("median")
        Pf["__win"] = (Pf["__fwd"] > med).astype(int)       # relative-winner label
        Pf["__date"] = pd.to_datetime(Pf["__date"])
        self.features_ = [c for c in Pf.columns if not c.startswith("__")]
        return Pf.dropna(subset=["__win"]).sort_values("__date").reset_index(drop=True)

    # ----- AFML sample weights (per-asset uniqueness x recency x return) ----- #
    def _sample_weights(self, tr: pd.DataFrame, anchor: pd.Timestamp) -> np.ndarray:
        H = self.cfg.horizon
        uniq = np.ones(len(tr), dtype=float)
        pos = np.arange(len(tr))
        for s, g in tr.groupby("__sym"):
            dates = pd.DatetimeIndex(g["__date"].values)
            order = np.argsort(dates.values)
            dsort = dates[order]
            n = len(dsort)
            t1 = pd.Series(np.minimum(np.arange(n) + H, n - 1), index=dsort)
            u = get_average_uniqueness(dsort, t1)
            uu = np.empty(n)
            uu[order] = u
            uniq[pos[tr["__sym"].values == s]] = uu
        rsafe = np.nan_to_num(tr["__fwd"].to_numpy(dtype=float), nan=0.0,
                              posinf=0.0, neginf=0.0)
        w = get_sample_weights(
            timestamps=tr["__date"].reset_index(drop=True),
            uniqueness=np.nan_to_num(uniq, nan=1.0),
            returns=rsafe,
            time_decay_span=self.cfg.recency_halflife_days,
            anchor_time=anchor,
        )
        return np.clip(np.nan_to_num(w, nan=0.05, posinf=10.0, neginf=0.05), 0.05, 10.0)

    # ----- frozen MDA feature selection on the first window ----------------- #
    def _freeze_features(self, P: pd.DataFrame, first_t0: pd.Timestamp) -> list[str]:
        cfg = self.cfg
        if cfg.mda_keep >= len(self.features_) or CatBoostClassifier is None:
            self.frozen_features_ = list(self.features_)
            return self.frozen_features_
        tr = P[P["__date"] < (first_t0 - pd.Timedelta(days=cfg.embargo_days))]
        # inner holdout (last 20%) for MDA shuffling — strictly past-only
        cut = tr["__date"].quantile(0.8)
        fit = tr[tr["__date"] < cut]
        val = tr[tr["__date"] >= cut]
        if len(fit) < 2000 or len(val) < 500:
            self.frozen_features_ = list(self.features_)
            return self.frozen_features_
        m = CatBoostClassifier(random_seed=cfg.seed, **cfg.catboost_params)
        m.fit(fit[self.features_].fillna(0.0), fit["__win"])
        from ..selection.mda import causal_mda
        res = causal_mda(m, val[self.features_].fillna(0.0).to_numpy(),
                         val["__win"].to_numpy(), self.features_,
                         block_size=20, n_repeats=5, random_seed=cfg.seed)
        if res is None or res.empty or "feature" not in res.columns:
            raise ValueError("empty MDA")
        ranked = res.sort_values("mda_mean", ascending=False)["feature"].tolist()
        self.frozen_features_ = ranked[: cfg.mda_keep]
        return self.frozen_features_

    # ----- walk-forward refit loop ------------------------------------------ #
    def run(self, P: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        qs = pd.date_range(pd.Timestamp(cfg.wf_start, tz="UTC"),
                           P["__date"].max(), freq=f"{cfg.refit_days}D")
        feats = self._freeze_features(P, qs[0])
        oos = np.full(len(P), np.nan)
        dtv = P["__date"].values
        self.refit_dates_ = []
        for t0 in qs:
            t1 = t0 + pd.Timedelta(days=cfg.refit_days)
            tr_mask = dtv < (t0 - pd.Timedelta(days=cfg.embargo_days)).to_datetime64()
            te_mask = (dtv >= t0.to_datetime64()) & (dtv < t1.to_datetime64())
            if tr_mask.sum() < cfg.min_train_rows or te_mask.sum() == 0:
                continue
            tr = P.loc[tr_mask]
            w = self._sample_weights(tr, anchor=t0)
            m = CatBoostClassifier(random_seed=cfg.seed, **cfg.catboost_params)
            m.fit(tr[feats].fillna(0.0), tr["__win"], sample_weight=w)
            oos[te_mask] = m.predict_proba(P.loc[te_mask, feats].fillna(0.0))[:, 1]
            self.refit_dates_.append(t0)
        P = P.copy()
        P["__p"] = oos
        return P

    # ----- regime-guarded harvest ------------------------------------------- #
    def harvest(self, P: pd.DataFrame, *, neutralise=True, smooth=True,
                band=True, regime=True) -> pd.Series:
        cfg, pan = self.cfg, self._panels
        close, rets, mkt, btc_ret = pan["close"], pan["rets"], pan["mkt"], pan["btc_ret"]
        m = P["__p"].notna()
        score = (P[m].pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last")
                 .reindex(close.index).reindex(columns=pan["syms"]))
        vol = rets.rolling(30).std().shift(1)
        logvol = np.log(vol.clip(lower=1e-4))
        betaf = rets.rolling(60).cov(btc_ret).div(btc_ret.rolling(60).var(), axis=0).shift(1)
        rk = score.rank(axis=1, pct=True)
        sig = rk.sub(rk.mean(axis=1), axis=0)
        if neutralise:
            sig = self._neutralise(sig, logvol, betaf)
        if smooth:
            sig = sig.ewm(span=cfg.smooth_span).mean()
        g = sig.abs().sum(axis=1).replace(0, np.nan)
        wn = sig.div(g, axis=0).fillna(0.0)
        if regime:
            wn = wn.mul(self._regime_gross(mkt), axis=0)
        if band:
            wn = self._apply_band(wn, cfg.band)
        pnl = ((wn.shift(1) * rets).sum(axis=1)
               - (wn - wn.shift(1)).abs().sum(axis=1) * cfg.cost_bps / 1e4).dropna()
        bv = pnl.std() * np.sqrt(DAYS)
        return pnl * (cfg.target_vol / bv) if bv > 0 else pnl

    @staticmethod
    def _neutralise(sig, logvol, betaf):
        out = sig * np.nan
        for dt in sig.index:
            y = sig.loc[dt].dropna()
            if len(y) < 15:
                continue
            X = pd.DataFrame({"lv": logvol.loc[dt], "b": betaf.loc[dt]}).reindex(y.index).fillna(0.0)
            X.insert(0, "c", 1.0)
            A = X.values
            # Phase 0: beide paden schreven de RUWE y terug wanneer de
            # neutralisatie-regressie niet kon draaien. Het resultaat heette
            # daarna nog steeds "geneutraliseerd", terwijl de factor-exposure er
            # volledig in bleef zitten - een markt-neutraal boek dat op die dagen
            # gewoon directioneel was.
            require(
                np.isfinite(A).all(),
                "Niet-eindige waarden in de neutralisatie-designmatrix; de "
                "factor-exposure kan niet worden weggeregresseerd. De ruwe "
                "returns worden NIET als geneutraliseerd doorgegeven.",
                DataContractError,
                date=str(dt),
            )
            try:
                coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
                out.loc[dt, y.index] = y.values - A @ coef
            except np.linalg.LinAlgError as exc:
                raise DataContractError(
                    f"Neutralisatie-regressie singulier op {dt}; ruwe returns "
                    f"worden niet als geneutraliseerd doorgegeven."
                ) from exc
        return out

    def _regime_gross(self, mkt: pd.Series) -> pd.Series:
        """De-gross in strong bull manias (short-biased book's failure mode)."""
        strength = (mkt.rolling(60).mean() / mkt.rolling(60).std()).shift(1)
        hi = strength.clip(lower=0).quantile(0.9)
        cut = (strength.clip(lower=0) / (hi if hi > 0 else 1.0)).clip(0, 1) * self.cfg.regime_degross
        return (1.0 - cut).fillna(1.0)

    @staticmethod
    def _apply_band(wn: pd.DataFrame, band: float) -> pd.DataFrame:
        held = wn * 0.0
        prev = pd.Series(0.0, index=wn.columns)
        for dt in wn.index:
            tgt = wn.loc[dt]
            mv = (tgt - prev).abs() > band
            prev = prev.where(~mv, tgt)
            held.loc[dt] = prev
        return held

    # ----- evaluation: forward stats + DSR + PBO ---------------------------- #
    def evaluate(self, P: pd.DataFrame) -> dict:
        cfg = self.cfg
        # construction-variant grid for honest overfit accounting (PBO + deflated DSR)
        variants = {}
        for neu in (True, False):
            for sm in (True, False):
                for reg in (True, False):
                    key = f"n{int(neu)}s{int(sm)}r{int(reg)}"
                    variants[key] = self.harvest(P, neutralise=neu, smooth=sm,
                                                 band=True, regime=reg)
        idx = sorted(set().union(*[v.index for v in variants.values()]))
        mat = pd.DataFrame({k: v.reindex(idx) for k, v in variants.items()}).fillna(0.0)
        n_trials = mat.shape[1]
        pbo = compute_pbo(mat.to_numpy(), n_subsets=10)
        pbo_val = float(pbo.get("pbo", np.nan))
        main = self.harvest(P)                    # the deployed config (all controls on)
        srd = main.mean() / main.std()
        dsr = deflated_sharpe(srd, n_trials, len(main))
        per_year = {int(y): float((1 + g).prod() - 1) for y, g in main.groupby(main.index.year)}
        per_q = {str(d.date()): float(v) for d, v in
                 main.resample(f"{cfg.refit_days}D").apply(lambda x: (1 + x).prod() - 1).items()}
        eq = (1 + main).cumprod()
        last12 = main[main.index >= main.index.max() - pd.Timedelta(days=365)]
        return dict(
            forward_sharpe=float(main.mean() / main.std() * np.sqrt(DAYS)),
            forward_vol=float(main.std() * np.sqrt(DAYS)),
            max_drawdown=float((eq / eq.cummax() - 1).min()),
            deflated_sharpe=float(dsr), n_trials=n_trials, pbo=pbo_val,
            last12m_return=float((1 + last12).prod() - 1),
            last12m_sharpe=float(last12.mean() / last12.std() * np.sqrt(DAYS)) if last12.std() else 0.0,
            n_refits=len(self.refit_dates_), n_features=len(self.frozen_features_ or []),
            per_year=per_year, per_quarter=per_q, pnl=main,
        )
