# src/tradebot/backtest/evaluation.py
"""Institutional realism helpers for backtest evaluation.

Extracted from ``evaluation_realism.py``.

Implements six corrections on the standard CPCV/Kelly/equity workflow:

  1. Deflated Sharpe Ratio (DSR) penalty accounting for ALL prior experiments
     (git-commit history + current Optuna run).
  2. Gap-Risk + CVaR aware Kelly sizing (vs. naïve target_risk / SL).
  3. Stationary block-bootstrap over CPCV path-Sharpes (Politis-Romano)
     to correct the correlated-paths bias.
  4. Hanging-barrier (timeout) monitor.
  5. Long/Short collision resolver with delta-neutral netting + horizon
     weighting (vs. binary winner-takes-all).
  6. Bar-by-Bar Mark-to-Market equity curve with Sharpe + Calmar.

All functions are pure utilities — no side-effects beyond logging.
No numba dependency: Pylance/mypy-friendly.
"""
from __future__ import annotations

import logging
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as _stats

logger = logging.getLogger("backtest.evaluation")


# =============================================================================
# 1. DEFLATED SHARPE — PRIOR-EXPERIMENT AWARE
# =============================================================================
def count_git_commits(
    repo_path: Path | None = None,
    fallback: int = 0,
) -> int:
    """Count the number of git commits in the repository.

    Quant firms use commit-count as proxy for the number of prior
    experiments/iterations preceding the current model. Each commit represents
    a (implicit) hypothesis test on the same data.

    Args:
        repo_path: path to the Git repository (default = cwd).
        fallback:  value when git is not available.

    Returns:
        Number of commits in the current branch (HEAD).
    """
    try:
        cwd = str(repo_path) if repo_path is not None else None
        result = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=cwd,
        )
        return int(result.decode("ascii").strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return int(fallback)


def _deflated_sharpe_penalty_legacy(
    hist_sharpes: Sequence[float],
    n_optuna_trials: int,
    n_prior_experiments: int = 0,
    cap_factor: float = 5.0,
) -> float:
    """Compute the Deflated Sharpe Ratio multiple-testing penalty (legacy).

    Follows López de Prado (2014):
        penalty = sigma(SR) * sqrt(2 * ln(N_eff))

    where ``N_eff`` is the effective number of hyperparameter combinations drawn.
    Unlike the naïve formula (only Optuna trials), ALL prior experiments are
    counted — typically approximated via the number of git commits (fixed prior
    penalty).

    .. deprecated::
        Use :func:`deflated_sharpe_ratio_p` for the true Bailey & LdP 2014
        DSR probability value.  This function returns a subtractive penalty
        and is kept for backward compatibility with callers in
        ``institutional_evaluation_report``.

    Args:
        hist_sharpes:         path/fold Sharpes from the current Optuna run.
        n_optuna_trials:      total planned trials in this study.
        n_prior_experiments:  number of historical experiments
                              (e.g. ``count_git_commits()``).
        cap_factor:           safety cap ``cap_factor * sigma_sr`` prevents
                              exploding penalties when ln(N) → ∞.

    Returns:
        Non-negative float subtracted directly from the raw Sharpe.
    """
    if len(hist_sharpes) < 2:
        return 0.0

    sigma_sr = float(np.std(np.asarray(hist_sharpes, dtype=np.float64)))
    if sigma_sr <= 0.0 or not math.isfinite(sigma_sr):
        return 0.0

    n_eff = max(int(n_optuna_trials), 1) + max(int(n_prior_experiments), 0)
    if n_eff < 2:
        return 0.0

    raw_penalty = sigma_sr * math.sqrt(2.0 * math.log(float(n_eff)))
    cap = cap_factor * sigma_sr
    return float(min(raw_penalty, cap))


# P0-20: backward-compat alias so existing callers of deflated_sharpe_penalty() keep working.
deflated_sharpe_penalty = _deflated_sharpe_penalty_legacy


def deflated_sharpe_ratio_p(
    sr_obs: float,
    hist_sharpes: Sequence[float],
    n_trials: int,
    n_obs: int,
) -> float:
    """Deflated Sharpe Ratio als kanswaarde (Bailey & López de Prado 2014).

    DSR_p = P(SR_obs > E[max(SR) | H0_no_skill])

    Formule (Bailey & LdP 2014, eq. 8):
        E[max(SR)] ≈ sigma_SR * (γ_E + sqrt(2*ln(N)))
        where γ_E = Euler-Mascheroni constant ≈ 0.5772

    Args:
        sr_obs:       observed (out-of-sample) Sharpe ratio.
        hist_sharpes: Sharpes from all prior trials/folds (distribution of SRs
                      under repeated testing — used to estimate sigma_SR).
        n_trials:     total number of independent trials / hyperparameter configs.
        n_obs:        number of observations (bars or trades) used to compute sr_obs.

    Returns:
        Float in [0, 1]. Promote only when DSR_p > 0.95 (Tier-1 gate).
        0.5 = skill equal to noise; 0.95 = statistically deflated above noise.
    """
    EULER_MASCHERONI = 0.5772156649
    if len(hist_sharpes) < 2 or n_trials < 2 or n_obs < 2:
        return 0.0
    sharpes_arr = np.asarray(hist_sharpes, dtype=np.float64)
    sigma_sr = float(np.std(sharpes_arr, ddof=1))
    if sigma_sr <= 0.0 or not math.isfinite(sigma_sr):
        return 0.0
    n_eff = max(int(n_trials), 2)
    e_max_sr = sigma_sr * (EULER_MASCHERONI + math.sqrt(2.0 * math.log(float(n_eff))))
    # Sharpe SE: var(SR) ≈ (1 + 0.5*SR²) / T for iid returns
    sr_se = math.sqrt((1.0 + 0.5 * sr_obs**2) / max(n_obs, 2))
    if sr_se <= 0.0:
        return 0.0
    z = (sr_obs - e_max_sr) / sr_se
    return float(_stats.norm.cdf(z))


# =============================================================================
# 2. GAP-RISK + TAIL-RISK STRESSED KELLY
# =============================================================================
@dataclass(frozen=True)
class KellySizingResult:
    """Output of ``gap_risk_kelly_size``."""

    leverage: float
    expected_risk: float
    gap_risk_factor: float
    cvar_factor: float
    capped: bool


def _empirical_cvar(
    returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Conditional Value-at-Risk on the (1-alpha) tail of *negative* returns."""
    if returns.size == 0:
        return 0.0
    losses = -np.asarray(returns, dtype=np.float64)
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        return 0.0
    var_q = float(np.quantile(losses, 1.0 - alpha))
    tail = losses[losses >= var_q]
    if tail.size == 0:
        return float(var_q)
    return float(tail.mean())


# AUDIT-FIX (N13 — per-asset gap_atr_multiple):
#   Uniform 1.5 underestimates SOL tail-risk (network outages 5-10% historically).
#   BTC: 1.0 (24/7 liquid), ETH: 1.5 (DeFi cascades), SOL: 2.5 (validator events).
ASSET_GAP_MULTIPLES: dict[str, float] = {
    "BTCUSDT": 1.0,
    "BTCBUSD": 1.0,
    "ETHUSDT": 1.5,
    "ETHBUSD": 1.5,
    "SOLUSDT": 2.5,
    "SOLBUSD": 2.5,
}
_DEFAULT_GAP_MULTIPLE: float = 1.5  # fallback for unknown assets


def get_gap_multiple_for_asset(symbol: str) -> float:
    """Return the asset-specific gap_atr_multiple.

    Looks up in ASSET_GAP_MULTIPLES by exact match or prefix match
    (e.g. "SOL" matches "SOLUSDT"). Fallback = 1.5.
    """
    if symbol in ASSET_GAP_MULTIPLES:
        return ASSET_GAP_MULTIPLES[symbol]
    for key, val in ASSET_GAP_MULTIPLES.items():
        if symbol.startswith(key[:3]):
            return val
    return _DEFAULT_GAP_MULTIPLE


def gap_risk_kelly_size(
    target_risk: float,
    atr_at_entry: float,
    price_at_entry: float,
    sl_width: float,
    max_leverage: float,
    kelly_divisor: float = 2.0,   # v3 SK-5: crypto mu-error correctie (2× conservatiever)
    *,
    historical_returns: np.ndarray | None = None,
    gap_atr_multiple: float = 1.5,
    cvar_alpha: float = 0.05,
    min_expected_risk: float = 1e-4,
    symbol: str | None = None,
) -> KellySizingResult:
    """Kelly sizing with gap-risk and tail-risk correction.

    Standard Kelly assumes the stop-loss is guaranteed to trigger at the SL
    price. In practice:

      * **Gap-risk**: a gap (overnight, weekend, news) can skip the SL.
        Effective worst-case = ``sl_width + gap_atr_multiple``.
      * **Tail-risk**: actual downside is fatter than Gaussian.
        We scale leverage by ``CVaR / mean_loss_assumed``.

    Args:
        target_risk:        desired fraction of equity per trade (e.g. 0.01).
        atr_at_entry:       ATR (in price units) at entry.
        price_at_entry:     entry price (>0).
        sl_width:           stop-loss width in ATR multiples.
        max_leverage:       hard leverage cap.
        kelly_divisor:      extra conservative factor voor crypto mu-estimation error.
                            Default 2.0 halveert de baseline leverage (v3 SK-5 correctie).
                            1.0 = geen extra correctie (legacy gedrag).
        historical_returns: prior trade returns for empirical CVaR
                            (None = fixed 1.5x penalty).
        gap_atr_multiple:   extra ATR steps the gap can skip.
        cvar_alpha:         tail quantile (0.05 = worst 5%).
        min_expected_risk:  lower bound for numerical stability.
        symbol:             when provided, auto-lookup per-asset multiple from
                            ASSET_GAP_MULTIPLES (overrides gap_atr_multiple
                            only when it is still at the default 1.5).

    Returns:
        ``KellySizingResult`` with leverage and associated factors.
    """
    # AUDIT-FIX (N13): auto-lookup per-asset multiple when symbol is given and
    # caller has not explicitly overridden gap_atr_multiple (≠ default 1.5).
    if symbol is not None and gap_atr_multiple == 1.5:
        gap_atr_multiple = get_gap_multiple_for_asset(symbol)

    safe_price = max(float(price_at_entry), 1e-4)
    base_risk = float(sl_width) * float(atr_at_entry) / safe_price
    gap_risk_factor = 1.0 + max(float(gap_atr_multiple), 0.0)
    stressed_risk = base_risk * gap_risk_factor

    if historical_returns is not None and historical_returns.size >= 20:
        cvar = _empirical_cvar(historical_returns, alpha=cvar_alpha)
        if cvar > 0.0 and stressed_risk > 0.0:
            cvar_factor = max(cvar / max(stressed_risk, min_expected_risk), 1.0)
        else:
            cvar_factor = 1.0
    else:
        # CHIEF AUDIT 2026-05-23 (H5): conservatieve prior CVaR-penalty bij
        # <20 trades. Voorkomt drawdown-shock in eerste productie-maand —
        # zonder bescherming zou cold-start een raw Kelly-leverage geven
        # gelijk aan een ervaren strategie, terwijl de fat-tail-realisatie
        # nog niet gekalibreerd is. Factor 1.3 = ~25% deflate op leverage.
        cvar_factor = 1.3

    expected_risk = max(stressed_risk * cvar_factor, min_expected_risk)
    raw_lev = float(target_risk) / (expected_risk * max(float(kelly_divisor), 1.0))
    capped = raw_lev > float(max_leverage)
    leverage = float(min(raw_lev, max_leverage))

    return KellySizingResult(
        leverage=leverage,
        expected_risk=expected_risk,
        gap_risk_factor=gap_risk_factor,
        cvar_factor=cvar_factor,
        capped=capped,
    )


# =============================================================================
# 3. STATIONARY BLOCK BOOTSTRAP — CPCV PATH SHARPES
# =============================================================================
def stationary_block_bootstrap_indices(
    n_obs: int,
    avg_block_size: float,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary block bootstrap indices.

    Returns a ``(n_boot, n_obs)`` int matrix with geometrically distributed
    block lengths (mean ``avg_block_size``).
    """
    if n_obs <= 0:
        return np.zeros((n_boot, 0), dtype=np.int64)
    p = 1.0 / max(float(avg_block_size), 1.0)
    out = np.empty((n_boot, n_obs), dtype=np.int64)
    for b in range(n_boot):
        idx = np.empty(n_obs, dtype=np.int64)
        i = 0
        while i < n_obs:
            start = int(rng.integers(0, n_obs))
            block_len = int(rng.geometric(p))
            block_len = max(block_len, 1)
            end = min(i + block_len, n_obs)
            for k in range(end - i):
                idx[i + k] = (start + k) % n_obs
            i = end
        out[b] = idx
    return out


def block_bootstrap_path_sharpes(
    path_returns: list[pd.Series],
    *,
    n_boot: int = 500,
    avg_block_size: float = 20.0,
    annualisation: float = math.sqrt(365.25),
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap distribution of path-Sharpes with block-resampling.

    CPCV paths share underlying observations → highly correlated. Naïve std
    across paths underestimates variance. Stationary block bootstrap restores
    the effective degrees of freedom.

    Args:
        path_returns:    list of trade-return Series (one per CPCV path).
        n_boot:          number of bootstrap replications.
        avg_block_size:  mean block length (Politis-Romano).
        annualisation:   factor to annualise daily Sharpe.
        seed:            RNG seed.

    Returns:
        Dict with ``sharpe_mean``, ``sharpe_std``, ``sharpe_lower_5``,
        ``sharpe_upper_95``, ``effective_dof``.
    """
    if not path_returns:
        return {
            "sharpe_mean": 0.0,
            "sharpe_std": 0.0,
            "sharpe_lower_5": 0.0,
            "sharpe_upper_95": 0.0,
            "effective_dof": 0.0,
        }

    pooled = pd.concat(path_returns).sort_index()
    pooled = pooled[~pooled.index.duplicated(keep="first")]
    if pooled.empty:
        return {
            "sharpe_mean": 0.0,
            "sharpe_std": 0.0,
            "sharpe_lower_5": 0.0,
            "sharpe_upper_95": 0.0,
            "effective_dof": 0.0,
        }

    rng = np.random.default_rng(seed)
    # Zero-padding is correct for event-driven strategies: non-trade days
    # contribute 0% to the live equity curve.  See comment in
    # compute_annualised_sharpe (execution/spread.py) for the full rationale.
    daily = pooled.resample("D").sum().fillna(0.0)
    daily_arr = daily.to_numpy(dtype=np.float64)
    n_obs = int(daily_arr.size)
    if n_obs < 2:
        return {
            "sharpe_mean": 0.0,
            "sharpe_std": 0.0,
            "sharpe_lower_5": 0.0,
            "sharpe_upper_95": 0.0,
            "effective_dof": float(n_obs),
        }

    boot_idx = stationary_block_bootstrap_indices(
        n_obs=n_obs,
        avg_block_size=avg_block_size,
        n_boot=n_boot,
        rng=rng,
    )
    samples = daily_arr[boot_idx]  # (n_boot, n_obs)
    means = samples.mean(axis=1)
    # P3.5-FIX: NumPy default ddof=0 (population std) understates volatility
    # by √((n-1)/n) per bootstrap sample.  Use ddof=1 (sample std) for
    # consistency with metrics.sharpe_ratio and portfolio._bootstrap_sharpe.
    stds = np.std(samples, axis=1, ddof=1)
    safe_std = np.where(stds > 1e-12, stds, np.nan)
    sharpes = (means / safe_std) * annualisation
    sharpes = sharpes[np.isfinite(sharpes)]

    if sharpes.size == 0:
        return {
            "sharpe_mean": 0.0,
            "sharpe_std": 0.0,
            "sharpe_lower_5": 0.0,
            "sharpe_upper_95": 0.0,
            "effective_dof": float(n_obs / max(avg_block_size, 1.0)),
        }

    return {
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "sharpe_lower_5": float(np.quantile(sharpes, 0.05)),
        "sharpe_upper_95": float(np.quantile(sharpes, 0.95)),
        "effective_dof": float(n_obs / max(avg_block_size, 1.0)),
    }


# =============================================================================
# 4. HANGING BARRIER (TIMEOUT EXIT) MONITOR
# =============================================================================
def compute_timeout_fraction(
    event_indices: np.ndarray,
    horizons: np.ndarray,
    t1_indices: np.ndarray,
    *,
    entry_offset: int = 2,
    n_total_bars: int,
    warn_threshold: float = 0.15,
    max_horizon: int | None = None,
    tolerance: int = 1,
) -> dict[str, float]:
    """Compute the fraction of events that exit on the horizon timeout.

    A high timeout rate (>15%) signals that ``t_max`` is too short or the
    ATR multiplier is too wide. Quant firms monitor this as a KPI.

    Args:
        event_indices: bar indices of entries.
        horizons:      per-event horizon (bars after entry) — e.g. ``t1 - entry``.
        t1_indices:    realised exit bar (from Triple Barrier).
        entry_offset:  bars between signal bar and entry bar (default 2).
        n_total_bars:  total number of bars in the dataset (for clamping).
        warn_threshold: log a warning above this fraction.
        max_horizon:   when provided, define timeout as
                       ``(t1 - entry) >= max_horizon - tolerance``. Otherwise
                       compare ``t1`` with event-specific
                       ``entry + entry_offset + horizon - 1``.
        tolerance:     bars margin when using ``max_horizon`` comparison.

    Returns:
        Dict with ``n_events``, ``n_timeouts``, ``timeout_fraction``.
    """
    n_events = int(event_indices.size)
    if n_events == 0:
        return {"n_events": 0.0, "n_timeouts": 0.0, "timeout_fraction": 0.0}

    actual_t1 = t1_indices.astype(np.int64)
    realised = actual_t1 - event_indices.astype(np.int64)

    if max_horizon is not None:
        is_timeout = realised >= (int(max_horizon) - int(tolerance))
    else:
        expected_timeout = (
            event_indices.astype(np.int64)
            + int(entry_offset)
            + horizons.astype(np.int64)
            - 1
        )
        expected_timeout = np.clip(expected_timeout, 0, max(n_total_bars - 1, 0))
        is_timeout = actual_t1 == expected_timeout

    n_timeouts = int(is_timeout.sum())
    frac = float(n_timeouts) / float(n_events)

    if frac >= warn_threshold:
        logger.warning(
            "Hanging-barrier alert: %.1f%% of %d events exit on horizon-timeout "
            "(>= %.0f%%). Consider increasing t_max or recalibrating pt_width/sl_width.",
            frac * 100.0, n_events, warn_threshold * 100.0,
        )
    else:
        logger.info(
            "Timeout-exits: %d/%d (%.1f%%) — within %.0f%% target.",
            n_timeouts, n_events, frac * 100.0, warn_threshold * 100.0,
        )

    return {
        "n_events": float(n_events),
        "n_timeouts": float(n_timeouts),
        "timeout_fraction": frac,
    }


# =============================================================================
# 5. LONG/SHORT COLLISION RESOLVER — DELTA-NEUTRAL / HORIZON-WEIGHTED NET
# =============================================================================
@dataclass(frozen=True)
class CollisionResolution:
    """Output of ``resolve_long_short_collision``."""

    side: str  # "FLAT", "LONG", "SHORT", "NET"
    weight_long: float
    weight_short: float
    net_weight: float   # signed: + = long, - = short
    horizon: int
    expected_return: float


def resolve_long_short_collision(
    *,
    long_active: bool,
    short_active: bool,
    prob_long: float,
    prob_short: float,
    threshold_long: float,
    threshold_short: float,
    horizon_long: int,
    horizon_short: int,
    ret_long: float,
    ret_short: float,
    duration_neutral: bool = True,
) -> CollisionResolution:
    """Resolve a LONG/SHORT signal collision to a net position.

    Replaces binary "winner-takes-all" behaviour with delta-neutral netting
    when both signals are active:

      * Identical signals on different horizons → retain the common directional
        exposure, weighted ~ ratio(prob/threshold).
      * Opposing signals on different horizons → the position gets a NET (signed)
        weight, with effective horizon = min(h_l, h_s) when
        ``duration_neutral=True``.

    Returns:
        ``CollisionResolution``.
    """
    if not long_active and not short_active:
        return CollisionResolution(
            side="FLAT",
            weight_long=0.0,
            weight_short=0.0,
            net_weight=0.0,
            horizon=0,
            expected_return=0.0,
        )

    ratio_l = float(prob_long / max(threshold_long, 1e-6)) if long_active else 0.0
    ratio_s = float(prob_short / max(threshold_short, 1e-6)) if short_active else 0.0

    if long_active and not short_active:
        return CollisionResolution(
            side="LONG",
            weight_long=ratio_l,
            weight_short=0.0,
            net_weight=ratio_l,
            horizon=int(horizon_long),
            expected_return=float(ret_long),
        )

    if short_active and not long_active:
        return CollisionResolution(
            side="SHORT",
            weight_long=0.0,
            weight_short=ratio_s,
            net_weight=-ratio_s,
            horizon=int(horizon_short),
            expected_return=float(ret_short),
        )

    # Both sides active — net.
    total_ratio = ratio_l + ratio_s
    if total_ratio <= 0.0:
        return CollisionResolution(
            side="FLAT",
            weight_long=0.0,
            weight_short=0.0,
            net_weight=0.0,
            horizon=0,
            expected_return=0.0,
        )

    w_l = ratio_l / total_ratio
    w_s = ratio_s / total_ratio
    net = w_l - w_s
    combined_ret = (w_l * float(ret_long)) - (w_s * float(ret_short))

    if duration_neutral:
        horizon = int(min(horizon_long, horizon_short))
    else:
        horizon = int(max(horizon_long, horizon_short))

    if abs(net) < 1e-6:
        side = "NET"
    elif net > 0.0:
        side = "LONG"
    else:
        side = "SHORT"

    return CollisionResolution(
        side=side,
        weight_long=float(w_l),
        weight_short=float(w_s),
        net_weight=float(net),
        horizon=horizon,
        expected_return=float(combined_ret),
    )


# =============================================================================
# 6. BAR-BY-BAR MARK-TO-MARKET EQUITY CURVE
# =============================================================================
@dataclass
class MTMResult:
    """Output of ``compute_bar_by_bar_mtm``."""

    equity_curve: pd.Series
    bar_returns: pd.Series
    sharpe_annualised: float
    calmar: float
    max_drawdown: float
    intra_drawdown_p95: float
    n_bars: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)


def compute_bar_by_bar_mtm(
    *,
    close: np.ndarray,
    bar_index: pd.DatetimeIndex,
    trade_entry_idx: np.ndarray,
    trade_exit_idx: np.ndarray,
    trade_sides: np.ndarray,            # +1 = LONG, -1 = SHORT
    trade_leverage: np.ndarray,
    account_size: float = 100_000.0,
    annualisation_days: float = 365.25,
    max_portfolio_leverage: float | None = None,
    funding_rates: np.ndarray | None = None,
    rebalance_cost_bps: float = 0.0,
) -> MTMResult:
    """Build a bar-by-bar Mark-to-Market equity curve.

    The trade-level equity curve hides intraday drawdowns: a trade that ends
    in profit may have been 15% underwater in the meantime. For institutional
    Sharpe/Calmar the unrealised PnL must be included.

    Implementation (vectorised, no Python loops over bars):

      * For each trade ``k``: position PnL on bar ``t`` = ``side_k * lev_k *
        (close[t] / close[entry_k] - 1) * equity_at_entry``.
      * Bar return = sum over closed + open trades of bar-PnL deltas.
      * Cumulative: ``equity_curve = account_size * cumprod(1 + bar_returns)``.

    PORTFOLIO-MARGIN-FIX (Item 8):
        Kelly sizing computes leverage per trade as if it is the only open
        position. With overlapping trades the gross exposure stacks up:
        4 trades at 1.5× → 6.0× gross leverage. Give
        ``max_portfolio_leverage`` to bound the gross portfolio exposure:
        new trades are scaled down so ``|sum(side · leverage)|`` across all
        bars stays under the cap. Trades are processed chronologically;
        trades that exceed the cap are scaled or dropped.

    FUNDING-FIX (Item 2):
        Per-bar funding cost = signed_exposure * funding_rate. Pass
        ``funding_rates`` as a per-bar array of decimal rates (NOT
        annualised) — 0.0 on all bars except the funding tick
        (typically every 8h on Bybit).

    REBALANCE-COST-FIX (Item 1):
        Per-bar one-sided fee on |Δ exposure|.
    """
    n_bars = int(close.size)
    if n_bars == 0 or trade_entry_idx.size == 0:
        empty = pd.Series([account_size], index=bar_index[:1])
        return MTMResult(
            equity_curve=empty,
            bar_returns=pd.Series([], dtype=np.float64),
            sharpe_annualised=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            intra_drawdown_p95=0.0,
            n_bars=0,
            diagnostics={"reason": 1.0},
        )

    entry = np.asarray(trade_entry_idx, dtype=np.int64)
    exit_ = np.asarray(trade_exit_idx, dtype=np.int64)
    sides = np.asarray(trade_sides, dtype=np.float64)
    lev = np.asarray(trade_leverage, dtype=np.float64).copy()
    close_arr = np.asarray(close, dtype=np.float64)

    # PORTFOLIO-MARGIN-FIX (Item 8): chronological pass that scales down new
    # trades so that rolling gross exposure (sum of |side · lev| across active
    # overlapping trades) stays under ``max_portfolio_leverage``.
    n_portfolio_capped = 0
    if max_portfolio_leverage is not None and float(max_portfolio_leverage) > 0.0:
        cap = float(max_portfolio_leverage)
        order = np.argsort(entry, kind="stable")
        events: list[tuple[int, int, float]] = []
        for k in order:
            i0 = int(max(entry[k], 0))
            i1 = int(min(exit_[k], n_bars - 1))
            if i0 > i1 or lev[k] <= 0.0:
                continue
            # GROSS-EXPOSURE-FIX: use abs() per position, NOT abs() of the net sum.
            # Long 2x + Short 2x → net 0x but gross 4x margin.
            current_gross = 0.0
            for (e_i0, e_i1, e_signed_lev) in events:
                if e_i0 <= i0 <= e_i1:
                    current_gross += abs(e_signed_lev)
            new_signed = sides[k] * lev[k]
            headroom = max(0.0, cap - current_gross)
            requested_gross_add = abs(new_signed)
            if requested_gross_add > headroom + 1e-9:
                if headroom <= 1e-9:
                    lev[k] = 0.0
                    n_portfolio_capped += 1
                    continue
                scale = headroom / requested_gross_add
                lev[k] = float(lev[k] * scale)
                new_signed = sides[k] * lev[k]
                n_portfolio_capped += 1
            events.append((i0, i1, float(new_signed)))

    # Per-bar gross exposure via prefix-sum.
    # CHIEF AUDIT 2026-05-23 (K3): align entry with bidirectional.py — signal at
    # end of bar i, position opens at start of bar i+1.  De oude code ``i0 =
    # entry_index`` prijsde de positie op ``close[i0]`` zelf — terwijl het
    # signaal pas aan het einde van bar i bekend is.  Dit is de dominante
    # oorzaak van het ``mtm vs bar_returns`` Sharpe-verschil (same-bar fill
    # geeft een "free lunch" op de signal-bar zelf).  De ``exit_index`` blijft
    # onveranderd: ``exit_[k]`` is al de t1-idx uit de TBM en geldt als de bar
    # waarop de barrière wordt geraakt (close-to-close grid). De daadwerkelijke
    # blootstellingsperiode is dus [i+1, t1] = exposure over ``t1 - i`` bars,
    # consistent met de bidirectional bar_returns slice ``[i+1, t1+1)``.
    add_at = np.zeros(n_bars + 1, dtype=np.float64)
    for k in range(entry.size):
        raw_i0 = int(max(entry[k], 0))
        i0 = min(raw_i0 + 1, n_bars - 1)
        i1 = int(min(exit_[k], n_bars - 1))
        if i0 > i1:
            continue
        add_at[i0] += sides[k] * lev[k]
        add_at[i1 + 1] -= sides[k] * lev[k]
    exposure = np.cumsum(add_at[:n_bars])

    # Bar return from the market (close-to-close) — true log-returns.
    # Using log-returns for compounding: equity = account * cumprod(exp(log_ret))
    # is replaced by cumprod(1 + simple_ret) below, so we convert back.
    # For crypto with >5% intrabar moves the difference between simple and log
    # is non-negligible (~0.5% per 10% move); compound error grows over time.
    safe_close = np.where(close_arr > 0.0, close_arr, np.nan)
    log_ret = np.zeros(n_bars, dtype=np.float64)
    log_ret[1:] = np.log(close_arr[1:] / safe_close[:-1])
    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

    # leveraged log-return → convert to simple for cumprod equity calc
    lev_log_ret = exposure * log_ret
    bar_returns = np.expm1(lev_log_ret)

    # FUNDING-FIX (Item 2)
    total_funding_cost: float = 0.0
    if funding_rates is not None and funding_rates.size == n_bars:
        funding_arr = np.asarray(funding_rates, dtype=np.float64)
        funding_arr = np.nan_to_num(funding_arr, nan=0.0, posinf=0.0, neginf=0.0)
        # CHIEF AUDIT 2026-05-23 (M15): funding sparsity guard. Bij directe
        # call (bypass AssetTrack-pipeline waar funding_rate_is_dense expliciet
        # wordt geset), kan de caller per-ongeluk een DENSE (ffill'd) funding-
        # reeks doorgeven. Sparse convention = ~1 tick per 8h (~3 ticks/dag
        # bij continue bars, ratio ≪ 0.25). Dense convention = volle 8h-rate
        # herhaald op elke bar → 32× over-booking bij 15-min bars. Detecteer
        # en waarschuw; correctie blijft caller-responsibility om geen stille
        # data te muteren.
        n_nonzero = int(np.count_nonzero(funding_arr))
        if n_bars > 0 and (n_nonzero / n_bars) > 0.25:
            import logging as _log_sparsity
            _log_sparsity.getLogger(__name__).warning(
                "compute_bar_by_bar_mtm: funding_arr non-zero op %.1f%% van "
                "de bars (%d/%d) — lijkt DENSE (ffill'd). Verwacht ≤ 5%% bij "
                "Bybit 8h-cadence. Caller moet _funding_dt schaling toepassen "
                "(funding_arr × bar_seconds/28800), anders wordt 8h-fee tot 32× "
                "geboekt. Funding-cost in deze run is mogelijk over-modelled.",
                100.0 * n_nonzero / max(n_bars, 1),
                n_nonzero, n_bars,
            )
        funding_pnl = exposure * funding_arr
        bar_returns = bar_returns - funding_pnl
        total_funding_cost = float(np.sum(funding_pnl))

    # REBALANCE-COST-FIX (Item 1)
    total_rebalance_cost: float = 0.0
    if rebalance_cost_bps > 0.0 and n_bars > 0:
        cost_unit = float(rebalance_cost_bps) * 1e-4
        delta_exp = np.zeros(n_bars, dtype=np.float64)
        delta_exp[0] = abs(exposure[0])
        if n_bars > 1:
            delta_exp[1:] = np.abs(np.diff(exposure))
        cost_arr = delta_exp * cost_unit
        bar_returns = bar_returns - cost_arr
        total_rebalance_cost = float(np.sum(cost_arr))

    equity = account_size * np.cumprod(1.0 + bar_returns)
    equity_series = pd.Series(equity, index=bar_index[:n_bars], name="mtm_equity")
    bar_ret_series = pd.Series(bar_returns, index=bar_index[:n_bars], name="bar_returns")

    daily = bar_ret_series.resample("D").sum()
    if daily.std() > 1e-12 and daily.size > 1:
        sharpe = float((daily.mean() / daily.std()) * math.sqrt(annualisation_days))
    else:
        sharpe = 0.0

    running_peak = np.maximum.accumulate(equity)
    drawdown = (equity - running_peak) / np.where(running_peak > 0.0, running_peak, 1.0)
    max_dd = float(drawdown.min()) if drawdown.size > 0 else 0.0

    if equity.size > 1 and bar_index.size > 1:
        duration = (bar_index[-1] - bar_index[0]).total_seconds() / (86400.0 * 365.25)
        if duration > 0.0 and equity[0] > 0.0:
            cagr = (equity[-1] / equity[0]) ** (1.0 / duration) - 1.0
        else:
            cagr = 0.0
    else:
        cagr = 0.0

    if max_dd < 0.0:
        calmar = float(cagr / abs(max_dd))
    else:
        calmar = 0.0

    intra_p95 = float(np.quantile(drawdown, 0.05)) if drawdown.size > 0 else 0.0

    diagnostics: dict[str, float] = {
        "cagr": float(cagr),
        "max_exposure": float(np.max(np.abs(exposure))) if exposure.size > 0 else 0.0,
        "avg_exposure": float(np.mean(np.abs(exposure))) if exposure.size > 0 else 0.0,
        "n_portfolio_capped": float(n_portfolio_capped) if max_portfolio_leverage is not None else 0.0,
        "max_portfolio_leverage_used": (
            float(max_portfolio_leverage) if max_portfolio_leverage is not None else 0.0
        ),
        "total_funding_cost": float(total_funding_cost),
        "total_rebalance_cost": float(total_rebalance_cost),
    }

    return MTMResult(
        equity_curve=equity_series,
        bar_returns=bar_ret_series,
        sharpe_annualised=sharpe,
        calmar=calmar,
        max_drawdown=max_dd,
        intra_drawdown_p95=intra_p95,
        n_bars=n_bars,
        diagnostics=diagnostics,
    )


# =============================================================================
# CONVENIENCE: aggregate all metrics in one report
# =============================================================================
def institutional_evaluation_report(
    *,
    path_returns: list[pd.Series],
    fold_sharpes_for_dsr: Sequence[float],
    n_optuna_trials: int,
    n_prior_experiments: int,
    timeout_stats: dict[str, float] | None = None,
    mtm: MTMResult | None = None,
) -> dict[str, float]:
    """Aggregate all institutional metrics into one report dict."""
    bs = block_bootstrap_path_sharpes(path_returns)
    dsr_pen = deflated_sharpe_penalty(
        fold_sharpes_for_dsr,
        n_optuna_trials=n_optuna_trials,
        n_prior_experiments=n_prior_experiments,
    )
    raw_mean = float(np.mean(fold_sharpes_for_dsr)) if fold_sharpes_for_dsr else 0.0

    report: dict[str, float] = {
        "raw_mean_sharpe": raw_mean,
        "dsr_penalty": dsr_pen,
        "deflated_sharpe": raw_mean - dsr_pen,
        "bootstrap_sharpe_mean": bs["sharpe_mean"],
        "bootstrap_sharpe_lower_5": bs["sharpe_lower_5"],
        "bootstrap_sharpe_upper_95": bs["sharpe_upper_95"],
        "effective_dof": bs["effective_dof"],
    }
    if timeout_stats is not None:
        report["timeout_fraction"] = float(timeout_stats.get("timeout_fraction", 0.0))
    if mtm is not None:
        report["mtm_sharpe"] = mtm.sharpe_annualised
        report["mtm_calmar"] = mtm.calmar
        report["mtm_max_drawdown"] = mtm.max_drawdown
        report["mtm_intra_drawdown_p95"] = mtm.intra_drawdown_p95
    return report


# =============================================================================
# 7. MONTE CARLO BOOTSTRAP SHARPE CI (Wave 17)
# =============================================================================
def _estimate_block_size(arr: np.ndarray, max_lag: int = 50) -> int:
    """Politis-White-style block-size estimator from autocorrelation decay.

    Returns the smallest k such that |rho(k)| < 2/sqrt(n) (Bartlett band),
    capped to [3, 60] bars.  This is a robust proxy for the integral
    time-scale of the return series.

    CHIEF AUDIT-FIX (Sim-to-Reality #4):
      The iid bootstrap (replace=True over individual obs) assumes serial
      independence — under crypto's vol-clustering and meta-label barrier
      overlap the effective DOF is overstated 20-40%, producing CIs that
      are systematically too tight.  Auto-calibrated block bootstrap fixes
      this with no caller change required.
    """
    n = len(arr)
    if n < 30:
        return 3
    x = arr - arr.mean()
    var0 = float(np.dot(x, x)) / max(n, 1)
    if var0 <= 0.0:
        return 3
    bartlett_band = 2.0 / math.sqrt(n)
    upper = int(min(max_lag, n // 4))
    for k in range(1, upper + 1):
        rho_k = float(np.dot(x[:-k], x[k:])) / (var0 * (n - k))
        if abs(rho_k) < bartlett_band:
            return int(max(3, min(60, k)))
    return int(max(3, min(60, upper)))


def monte_carlo_sharpe_ci(
    trade_returns: np.ndarray,
    n_sims: int = 1000,
    annualization: float = 252.0,
    confidence: float = 0.95,
    block_size: int | None = None,
) -> tuple[float, float, float]:
    """Stationary block-bootstrap CI for Sharpe ratio.

    CHIEF AUDIT-FIX (Sim-to-Reality #4):
      Previous implementation used iid resampling (``rng.choice`` over the
      flat return vector) which is only valid when returns are serially
      uncorrelated.  Crypto trade returns from CPCV paths exhibit:
        • Vol-clustering (GARCH α+β ≈ 0.95-0.99)
        • Triple-barrier overlap (adjacent trades share return paths)
        • Meta-label momentum (Judge clusters trades by regime)
      Net effect: iid bootstrap understates Sharpe-CI width by ~20-40 %,
      promoting overfitted strategies past the DSR gate.

      Switched to Politis-Romano stationary block bootstrap with
      auto-calibrated geometric block-size (mean = block_size).  When
      block_size=None the integral time-scale is estimated from the
      autocorrelation function (Politis-White-style heuristic).

    Parameters
    ----------
    trade_returns : array of trade-level returns.
    n_sims : number of bootstrap samples.
    annualization : Sharpe scaling factor.
    confidence : two-sided confidence level.
    block_size : geometric mean block length. None → auto-calibrate.

    Returns
    -------
    (lower, point, upper) Sharpe ratio CI.
    """
    arr = np.asarray(trade_returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return 0.0, 0.0, 0.0

    eff_block = int(block_size) if block_size is not None else _estimate_block_size(arr)
    eff_block = max(1, min(eff_block, n))
    p_continue = 1.0 - 1.0 / float(eff_block)  # geometric mean = eff_block

    rng = np.random.default_rng(seed=42)
    sim_sharpes = np.empty(n_sims, dtype=np.float64)
    sample = np.empty(n, dtype=np.float64)

    for i in range(n_sims):
        # Stationary block bootstrap (Politis-Romano 1994):
        #   - start at a uniform random index
        #   - each step, continue with prob p else jump to a fresh index
        idx = int(rng.integers(0, n))
        # Pre-draw the continuation Bernoullis and jump points for speed
        u = rng.random(n)
        jumps = rng.integers(0, n, size=n)
        for t in range(n):
            sample[t] = arr[idx]
            if u[t] < p_continue:
                idx = (idx + 1) % n
            else:
                idx = int(jumps[t])
        mean_ = float(np.mean(sample))
        std_  = float(np.std(sample, ddof=1))
        sim_sharpes[i] = (mean_ / std_ * math.sqrt(annualization)) if std_ > 0 else 0.0

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(sim_sharpes, alpha))
    upper = float(np.quantile(sim_sharpes, 1.0 - alpha))
    point = float(np.mean(arr) / np.std(arr, ddof=1) * math.sqrt(annualization)) if np.std(arr) > 0 else 0.0

    logger.info(
        "monte_carlo_sharpe_ci: n=%d eff_block=%d (auto=%s) → "
        "Sharpe %.3f CI[%.3f, %.3f]",
        n, eff_block, block_size is None, point, lower, upper,
    )
    return lower, point, upper


__all__ = [
    "ASSET_GAP_MULTIPLES",
    "CollisionResolution",
    "KellySizingResult",
    "MTMResult",
    "block_bootstrap_path_sharpes",
    "compute_bar_by_bar_mtm",
    "compute_timeout_fraction",
    "count_git_commits",
    "deflated_sharpe_penalty",           # backward-compat alias for _legacy
    "_deflated_sharpe_penalty_legacy",
    "deflated_sharpe_ratio_p",           # P0-20: true Bailey & LdP 2014 DSR
    "gap_risk_kelly_size",
    "get_gap_multiple_for_asset",
    "institutional_evaluation_report",
    "monte_carlo_sharpe_ci",
    "resolve_long_short_collision",
    "stationary_block_bootstrap_indices",
]
