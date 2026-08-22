"""stress_and_optimise.py — Stress tests + HRP + Black-Litterman portfolio optimisation.

Leest de per-asset AssetTrack-bestanden en de portfolio equity-curve.
Doet:
  1. Stress tests op crypto-crisissperiodes binnen de OOS-data.
  2. HRP (Hierarchical Risk Parity) weights.
  3. Black-Litterman weights met per-asset alpha-views.
  4. Simuleert herijkte equity-curves en toont jaarreturn-vergelijking.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC  = _ROOT / "src"
for p in (_SRC, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SYMBOLS    = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
ART_DIR    = _ROOT / "artefacts"
BARS_PER_Y = 365.0 * 24   # 1h bars

# ─── Crypto stress-test windows (alle binnen OOS-periode sept 2021 – mei 2026) ─
STRESS_PERIODS = {
    "Terra/LUNA collapse":   ("2022-05-05", "2022-05-20"),
    "FTX collapse":          ("2022-11-06", "2022-11-16"),
    "2022 crypto bear":      ("2022-01-01", "2022-12-31"),
    "SVB banking crisis":    ("2023-03-08", "2023-03-20"),
    "2024 Aug correction":   ("2024-08-01", "2024-08-15"),
    "2025 Jan flush":        ("2025-01-15", "2025-02-01"),
}


# =============================================================================
# HELPERS
# =============================================================================

def load_tracks() -> dict[str, dict]:
    """Load AssetTrack en bouw dagelijkse return series per asset."""
    tracks = {}
    for sym in SYMBOLS:
        path = ART_DIR / "tracks" / f"{sym}.joblib"
        if not path.exists():
            continue
        t  = joblib.load(path)
        sr = np.asarray(t.signed_returns)
        lv = np.asarray(t.requested_leverage)
        ts = pd.DatetimeIndex(t.timestamps)
        bar_pnl = sr * lv          # leveraged bar return (unleveraged × kelly-lev)
        s = pd.Series(bar_pnl, index=ts, name=sym)
        daily = s.resample("D").sum()
        tracks[sym] = {
            "bar_pnl":  s,
            "daily":    daily,
            "n_trades": int(np.sum(np.diff(np.concatenate([[0], (np.asarray(t.side) != 0).astype(int), [0]])) == 1)),
        }
    return tracks


def equity_from_daily(daily: pd.Series, start: float = 1.0) -> pd.Series:
    eq = (1.0 + daily).cumprod() * start
    return eq


def sharpe_from_daily(daily: pd.Series) -> float:
    d = daily.dropna()
    if d.std() < 1e-10 or len(d) < 5:
        return 0.0
    return float(d.mean() / d.std() * math.sqrt(365.25))


def max_dd(equity: pd.Series) -> float:
    eq = equity.values
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / np.where(peak > 0, peak, 1.0)
    return float(dd.min())


# =============================================================================
# 1. STRESS TESTS
# =============================================================================

def run_stress_tests(tracks: dict) -> None:
    print("\n" + "=" * 72)
    print("STRESS TESTS")
    print("=" * 72)

    # Build per-asset daily series op gemeenschappelijke index
    all_daily = pd.DataFrame({s: tracks[s]["daily"] for s in SYMBOLS}).fillna(0.0)
    port_daily = all_daily.sum(axis=1) / len(SYMBOLS)   # equal-weight portfolio daily

    # Portfolio equity curve (equal weight — voor referentie)
    eq_port = equity_from_daily(port_daily)

    for name, (start, end) in STRESS_PERIODS.items():
        sl = slice(start, end)
        window = port_daily.loc[sl]
        if window.empty:
            print(f"\n  {name}: geen data in deze periode (buiten OOS-window)")
            continue

        eq_w = equity_from_daily(window, start=1.0)
        period_ret  = float(eq_w.iloc[-1] - 1.0)
        period_mdd  = max_dd(eq_w)
        period_days = len(window)

        print(f"\n  {name:30s}  [{start} -> {end}]")
        print(f"    Duur          : {period_days} handelsdagen")
        print(f"    Portfolio ret : {period_ret:+.2%}")
        print(f"    Max Drawdown  : {period_mdd:.2%}")

        # Per-asset breakdown
        for sym in SYMBOLS:
            sym_w = all_daily[sym].loc[sl]
            if sym_w.empty:
                continue
            sym_eq = equity_from_daily(sym_w, start=1.0)
            sym_ret = float(sym_eq.iloc[-1] - 1.0)
            sym_mdd = max_dd(sym_eq)
            print(f"    {sym:10s}   ret={sym_ret:+.2%}   MaxDD={sym_mdd:.2%}")


# =============================================================================
# 2. HRP — Hierarchical Risk Parity
# =============================================================================

def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0)


def hrp_weights(daily_rets: pd.DataFrame) -> pd.Series:
    """Compute HRP weights (López de Prado 2016).

    Stappen:
      1. Bereken correlatiematrix.
      2. Hierarchisch clusteren (Ward linkage op afstandsmatrix).
      3. Quasi-diagonaliseer covariantiematrix via leave-order.
      4. Recursive bisection: splits portfolio recursief op clusterbladen;
         wijs gewichten toe via inverse-variance per cluster.
    """
    cov  = daily_rets.cov().values
    corr = _cov_to_corr(cov)

    # Afstandsmatrix (1 - |corr|) → condensed form voor linkage
    dist     = np.sqrt((1.0 - corr) / 2.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    link  = linkage(condensed, method="ward")
    order = leaves_list(link)         # gesorteerde asset-volgorde

    # Recursive bisection
    n = len(order)
    weights = pd.Series(1.0, index=range(n))
    clusters = [list(order)]          # start met één cluster = alle assets

    while clusters:
        clusters_next = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            split   = len(cluster) // 2
            left    = cluster[:split]
            right   = cluster[split:]

            # Inverse-variance weight per helft
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r + 1e-12)   # gewicht rechter helft

            weights[left]  *= (1.0 - alpha)
            weights[right] *= alpha

            if len(left)  > 1: clusters_next.append(left)
            if len(right) > 1: clusters_next.append(right)
        clusters = clusters_next

    # Remap van volgorde-index naar kolomnamen
    result = pd.Series(0.0, index=daily_rets.columns)
    for i, col_idx in enumerate(range(n)):
        result.iloc[order[col_idx]] = weights[col_idx]
    result /= result.sum()
    return result


def _cluster_var(cov: np.ndarray, idx: list) -> float:
    sub = cov[np.ix_(idx, idx)]
    iv  = np.ones(len(idx)) / (np.diag(sub) + 1e-12)
    iv /= iv.sum()
    return float(iv @ sub @ iv)


# =============================================================================
# 3. BLACK-LITTERMAN
# =============================================================================

def black_litterman_weights(
    daily_rets: pd.DataFrame,
    alpha_views: dict[str, float],   # per-asset verwacht daily excess return (view)
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Black-Litterman posterior weights.

    Prior: CAPM-equilibrium op basis van gelijke marktgewichten.
    Views: per-asset alpha-verwachting (bijv. Sharpe × dagelijkse vol).

    Args:
        daily_rets   : dagelijkse return matrix
        alpha_views  : per-asset verwacht dagelijks excess return
        tau          : schaalparameter prior onzekerheid (typisch 0.01-0.1)
        risk_aversion: markt risico-aversie parameter λ

    Returns:
        Normalised portfolio weights (geen short, geen leverage > 1 per asset).
    """
    n    = len(daily_rets.columns)
    cov  = daily_rets.cov().values * 252  # jaarlijkse covariantie

    # Prior: equal-weight equilibrium
    w_eq  = np.ones(n) / n
    Pi    = risk_aversion * cov @ w_eq     # implied equilibrium returns

    # View matrix P = identity (absolute views op elk asset)
    P   = np.eye(n)
    Q   = np.array([alpha_views.get(s, 0.0) * 252 for s in daily_rets.columns])

    # Omega = diagonaal van τ × P Σ Pᵀ (view-onzekerheid proportioneel aan prior)
    Omega = np.diag(np.diag(tau * P @ cov @ P.T))

    # BL posterior expected returns
    tau_cov_inv = np.linalg.inv(tau * cov)
    Omega_inv   = np.linalg.inv(Omega)
    M_inv       = np.linalg.inv(tau_cov_inv + P.T @ Omega_inv @ P)
    mu_bl       = M_inv @ (tau_cov_inv @ Pi + P.T @ Omega_inv @ Q)

    # Mean-Variance optimal weights (unconstrained)
    cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(n))
    w_raw   = (1.0 / risk_aversion) * cov_inv @ mu_bl

    # Clip negatieve gewichten (geen short op portfolio-niveau)
    w_raw = np.maximum(w_raw, 0.0)
    total = w_raw.sum()
    if total < 1e-9:
        w_raw = np.ones(n) / n   # fallback: equal weight
    else:
        w_raw /= total

    return pd.Series(w_raw, index=daily_rets.columns)


# =============================================================================
# 4. GESIMULEERDE EQUITY + JAARRETURN
# =============================================================================

def simulate_weighted(
    daily_df: pd.DataFrame,
    weights: pd.Series,
    label: str,
    start_equity: float = 100_000.0,
) -> pd.Series:
    """Gewogen portfolio equity curve (dagelijks herbalanceren)."""
    w = weights.reindex(daily_df.columns).fillna(0.0)
    w /= w.sum()
    port = (daily_df * w).sum(axis=1)
    return equity_from_daily(port, start=start_equity)


def print_annual_table(
    results: dict[str, pd.Series],
    start_equity: float = 100_000.0,
) -> None:
    print("\n" + "=" * 72)
    print(f"JAARRETURN PER STRATEGIE  (startkapitaal EUR {start_equity:,.0f})")
    print("=" * 72)
    years  = list(range(2021, 2027))
    strats = list(results.keys())
    header = f"{'Jaar':<6}" + "".join(f"  {s:>18}" for s in strats)
    print(header)
    print("-" * len(header))
    for yr in years:
        row = f"{yr:<6}"
        for s, eq in results.items():
            idx   = eq.index
            mask  = idx.year == yr
            if not mask.any():
                row += f"  {'—':>18}"
                continue
            pos    = np.where(mask)[0]
            start  = float(eq.iloc[pos[0] - 1]) if pos[0] > 0 else start_equity
            end    = float(eq.iloc[pos[-1]])
            yr_ret = end / start - 1.0
            row += f"  {yr_ret:>+17.1%}"
        print(row)
    print("-" * len(header))
    for s, eq in results.items():
        total = float(eq.iloc[-1]) / start_equity - 1.0
        end_val = float(eq.iloc[-1])
        print(f"  {'Totaal':>10s} {s:>20s} : {total:+.1%}   (EUR {end_val:,.0f})")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("Laden van AssetTracks...")
    tracks = load_tracks()

    # Gemeenschappelijke dagelijkse return matrix
    daily_df = pd.DataFrame({s: tracks[s]["daily"] for s in SYMBOLS}).fillna(0.0)
    # Gebruik alleen bars die minstens één asset actief heeft
    active_mask = (daily_df != 0.0).any(axis=1)
    daily_active = daily_df[active_mask]

    # ── 1. Stress tests ──────────────────────────────────────────────────────
    run_stress_tests(tracks)

    # ── 2. HRP ───────────────────────────────────────────────────────────────
    hrp_w = hrp_weights(daily_active)
    print("\n" + "=" * 72)
    print("HRP GEWICHTEN (Hierarchical Risk Parity)")
    print("=" * 72)
    for sym, w in hrp_w.items():
        print(f"  {sym}: {w:.3f}")

    # ── 3. Black-Litterman ───────────────────────────────────────────────────
    # Alpha-views: Sharpe per asset × dagelijkse vol → verwacht daily excess return
    views = {}
    for sym in SYMBOLS:
        d = tracks[sym]["daily"]
        vol_d = float(d[d != 0].std()) if (d != 0).any() else 1e-4
        sharpe_d = sharpe_from_daily(d[d != 0])
        views[sym] = sharpe_d * vol_d / math.sqrt(252)   # daily excess return

    bl_w = black_litterman_weights(daily_active, alpha_views=views)
    print("\n" + "=" * 72)
    print("BLACK-LITTERMAN GEWICHTEN")
    print("=" * 72)
    for sym, w in bl_w.items():
        print(f"  {sym}: {w:.3f}  (alpha-view: {views[sym]*252:+.2%}/jaar)")

    # ── 4. Equal-weight referentie ───────────────────────────────────────────
    eq_w = pd.Series({s: 1.0 / len(SYMBOLS) for s in SYMBOLS})

    # ── 5. Simuleer equity curves ────────────────────────────────────────────
    start_eq = 100_000.0
    results = {
        "Equal Weight":        simulate_weighted(daily_df, eq_w,   "Equal Weight",  start_eq),
        "HRP":                  simulate_weighted(daily_df, hrp_w,  "HRP",           start_eq),
        "Black-Litterman":      simulate_weighted(daily_df, bl_w,   "BL",            start_eq),
    }

    # ── 6. Overzicht metrics per strategie ───────────────────────────────────
    print("\n" + "=" * 72)
    print("PORTFOLIO METRICS PER STRATEGIE")
    print("=" * 72)
    print(f"  {'Strategie':<22}  {'Sharpe':>7}  {'MaxDD':>7}  {'TotalRet':>9}  {'Weights'}")
    print("  " + "-" * 68)
    for label, eq in results.items():
        port_daily = eq.pct_change().dropna()
        sh  = sharpe_from_daily(port_daily)
        mdd = max_dd(eq)
        tr  = float(eq.iloc[-1]) / start_eq - 1.0
        if label == "Equal Weight":
            w_str = "1/3 | 1/3 | 1/3"
        elif label == "HRP":
            w_str = " | ".join(f"{hrp_w[s]:.2f}" for s in SYMBOLS)
        else:
            w_str = " | ".join(f"{bl_w[s]:.2f}" for s in SYMBOLS)
        print(f"  {label:<22}  {sh:>7.2f}  {mdd:>7.2%}  {tr:>+9.1%}  [{w_str}]")

    # ── 7. Jaarreturn tabel ──────────────────────────────────────────────────
    print_annual_table(results, start_equity=start_eq)

    # ── 8. Sla optimale gewichten op ────────────────────────────────────────
    import json
    out = {
        "hrp":             hrp_w.to_dict(),
        "black_litterman": bl_w.to_dict(),
        "equal_weight":    eq_w.to_dict(),
        "alpha_views_annual": {s: float(v * 252) for s, v in views.items()},
    }
    out_path = _ROOT / "reports" / "portfolio_optimisation.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nGewichten opgeslagen -> {out_path}")


if __name__ == "__main__":
    main()
