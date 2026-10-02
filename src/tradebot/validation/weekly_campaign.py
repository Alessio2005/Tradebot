"""De meting van de wekelijkse strategie op de ontwikkelsample (spec §9, §14).

Volgorde, en waarom zij vastligt:
1. d* en k worden gekozen op data vóór de eerste testperiode; k op frequentie.
2. Drie modellen door dezelfde purged walk-forward (Taak 7).
3. Kandidaten per model; het ensemble met Kelly, de referentie ongefilterd met
   een vaste risicofractie; allemaal door hetzelfde boek en dezelfde risicolaag.
4. Negatieve controles: geschudde labels en de omgekeerde richting.
5. Inferentie: Lo-SE, blokbootstrap, Ledoit-Wolf-verschil, DSR bij de bevroren M,
   Wilson-interval op de trefkans, PBO over de vier varianten, Monte Carlo-drawdown.
6. Het oordeel komt uit de bevroren stop-criteria (Taak 10).
CPCV-paden worden gerapporteerd, niet gepoort.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..backtest.barrier_book import BookInputs, SizingRule, run_barrier_book
from ..backtest.pbo import compute_pbo
from ..cv.cpcv import CombinatorialPurgedCV, build_cpcv_return_paths
from ..cv.event_space import event_space_t1
from ..cv.walk_forward import WalkForwardCV
from ..data.weekly_market import WeeklyMarket, load_weekly_market
from ..execution.impact_model import ImpactParams, ImpactStatus
from ..features.registry import current_git_sha
from ..features.weekly_set import fit_common_d_star, market_features
from ..labeling.barrier_fills import round_trip_cost
from ..labeling.breakout import calibrate_k
from ..monitoring.prob_calibration import expected_calibration_error
from ..registry.preregistration import StopCriterion, require_preregistration
from ..registry.trial_counter import TrialCount, frozen_trial_count
from ..risk.binary_kelly import (
    break_even_probability,
    monte_carlo_drawdown_probability,
    posterior_lower_probability,
)
from ..risk.engine import RiskEngine, risk_config_hash
from ..schemas.config import (
    ExecutionConfig,
    ImpactConfig,
    RiskConfig,
    ValidationConfig,
    load_config,
)
from ..schemas.weekly_meta import WeeklyMetaConfig, weekly_meta_config
from ..train.light_models import MODEL_KINDS, fit_light_model
from ..train.meta_label import FoldPredictions, shuffled_targets, walk_forward_fit_predict
from ..train.weekly_dataset import WeeklyDataset, build_weekly_dataset
from ..utils.failfast import DataContractError, require
from .dsr import dsr_gate
from .holdout import development_slice
from .inference import block_bootstrap_ci, sharpe_difference_test, sharpe_with_se
from .weekly_verdict import Verdict, judge

__all__ = ["CampaignResult", "main", "run_campaign_on_market", "trade_candidates"]

BARS_PER_YEAR = 365.0
ARTEFACT = Path("artefacts/governance/weekly_meta_campaign.json")


@dataclass(frozen=True)
class CampaignResult:
    values: dict[str, float]
    verdict: Verdict
    record: dict[str, Any]


def trade_candidates(
    events: pd.DataFrame, folds: Sequence[FoldPredictions], *, cfg: WeeklyMetaConfig,
    cost_rt: float,
) -> pd.DataFrame:
    """Per OOS-event: kans, posterior-ondergrens, break-even en de handelsdrempel van zijn fold."""
    edges = np.linspace(0.0, 1.0, cfg.n_probability_bins + 1)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        cal = np.asarray(fold.extras["calibration_probability"], dtype=np.float64)
        phi = min(1.0, cfg.trades_per_week_target
                  / max(float(fold.extras["train_events_per_week"]), 1e-9))
        q = float(np.quantile(cal, 1.0 - phi))
        counts = np.histogram(cal, bins=edges)[0]
        for r, p in zip(fold.row_index, fold.probability, strict=True):
            ev = events.iloc[int(r)]
            barrier = cfg.barrier_sigma * float(ev["sigma"])
            p_be = break_even_probability(barrier, cost_rt)
            b = min(int(np.searchsorted(edges, p, side="right")) - 1, len(counts) - 1)
            rows.append({
                "symbol": ev["symbol"], "entry_bar": int(ev["event_bar"]) + 1,
                "exit_bar": int(ev["exit_bar"]), "side": float(ev["side"]),
                "fill_return": float(ev["fill_return"]), "barrier": barrier,
                "p": float(p),
                "p_low": posterior_lower_probability(float(p), float(counts[b]),
                                                     cfg.posterior_quantile),
                "p_be": p_be, "p_trade": max(p_be, q), "fold_id": int(fold.fold_id),
            })
    return pd.DataFrame(rows).sort_values(
        ["entry_bar", "symbol"], kind="stable").reset_index(drop=True)


def _unfiltered(events: pd.DataFrame, cfg: WeeklyMetaConfig, lo: int, hi: int) -> pd.DataFrame:
    ev = events[(events["event_bar"] >= lo) & (events["event_bar"] <= hi)]
    return pd.DataFrame({
        "symbol": ev["symbol"].to_numpy(), "entry_bar": ev["event_bar"].to_numpy() + 1,
        "exit_bar": ev["exit_bar"].to_numpy(), "side": ev["side"].to_numpy(),
        "fill_return": ev["fill_return"].to_numpy(),
        "barrier": cfg.barrier_sigma * ev["sigma"].to_numpy(),
        "p": 1.0, "p_low": 1.0, "p_trade": 0.0,
    }).sort_values(["entry_bar", "symbol"], kind="stable").reset_index(drop=True)


def _oos_auc(folds: Sequence[FoldPredictions]) -> float:
    y = np.concatenate([f.target for f in folds])
    p = np.concatenate([f.probability for f in folds])
    w = np.concatenate([f.uniqueness for f in folds])
    return float(roc_auc_score(y, p, sample_weight=w))


def _cpcv_path_sharpes(wd: WeeklyDataset, cfg: WeeklyMetaConfig, inputs: BookInputs,
                       risk_cfg: RiskConfig, fixed: SizingRule) -> list[float]:
    ds = wd.dataset
    order = np.argsort(ds.event_bar, kind="stable")
    ev, ex = ds.event_bar[order], ds.exit_bar[order]
    cv = CombinatorialPurgedCV(n_groups=cfg.cpcv_n_groups, n_test_groups=cfg.cpcv_n_test_groups,
                               purge_bars=0)
    size = len(order) // cfg.cpcv_n_groups
    bounds = [(g * size, len(order) if g == cfg.cpcv_n_groups - 1 else (g + 1) * size)
              for g in range(cfg.cpcv_n_groups)]
    fold_returns: dict[tuple, pd.Series] = {}
    for train_pos, test_pos, groups in cv.split(pd.DatetimeIndex(wd.grid[ev]),
                                               pd.Series(event_space_t1(ev, ex))):
        model = fit_light_model(ds, order[train_pos], ds.target, "ensemble", cfg)
        rows = order[test_pos]
        prob = model.predict_proba(ds.features.iloc[rows].to_numpy(dtype=np.float64))[:, 1]
        fp = FoldPredictions(fold_id=0, row_index=rows, probability=prob,
                             target=ds.target[rows], uniqueness=ds.uniqueness[rows],
                             n_train=int(train_pos.size), purge={},
                             feature_importance=model.feature_importance,
                             extras=model.fold_extras)
        cands = trade_candidates(wd.events, [fp], cfg=cfg, cost_rt=fixed.cost_rt)
        res = run_barrier_book(cands, inputs, RiskEngine(risk_cfg), fixed,
                               equity0=cfg.account_equity)
        pieces = []
        for g in groups:
            lo, hi = bounds[g]
            start, end = wd.grid[ev[lo]], wd.grid[int(ex[lo:hi].max())]
            pieces.append(res.returns.loc[start:end])
        fold_returns[tuple(groups)] = pd.concat(pieces)
    paths = build_cpcv_return_paths(fold_returns, n_groups=cfg.cpcv_n_groups)
    return [sharpe_with_se(p, bars_per_year=BARS_PER_YEAR).sharpe for p in paths]


def _sharpe_or_none(returns: pd.Series) -> float | None:
    """Sharpe van een informatief boek; `None` als het boek nooit handelde (nul variantie).

    Alleen voor de rapportage per model. De poortmetingen van het ensemble gaan wel
    rechtstreeks door `sharpe_with_se`: een ensemble dat nooit handelt is een kapotte
    campagne en crasht, het is geen oordeel.
    """
    r = returns.to_numpy(dtype=np.float64)
    if not np.nanstd(r) > 0.0:
        return None
    return float(sharpe_with_se(returns, bars_per_year=BARS_PER_YEAR).sharpe)


def _realized(taken: pd.DataFrame) -> np.ndarray:
    """Het werkelijke positierendement per trade: bij een risico-exit niet het barrièrerendement."""
    real = taken["realized_return"].to_numpy(dtype=np.float64)
    return np.where(np.isfinite(real), real, taken["fill_return"].to_numpy(dtype=np.float64))


def run_campaign_on_market(
    market: WeeklyMarket,
    cfg: WeeklyMetaConfig,
    *,
    criteria: Sequence[StopCriterion],
    trial_count: TrialCount,
    exec_cfg: ExecutionConfig,
    val_cfg: ValidationConfig,
    risk_cfg: RiskConfig,
    impact: ImpactParams | None,
) -> CampaignResult:
    cost_rt = round_trip_cost(exec_cfg)
    first_test = pd.Timestamp(cfg.first_test_start_utc)
    closes = {s: market.ohlcv[s]["close"] for s in market.symbols}
    d_star = fit_common_d_star({s: np.log(c) for s, c in closes.items()}, until=first_test)
    start = market.sigma_daily.dropna(how="all").index[0]
    k, rates = calibrate_k(closes, {s: market.sigma_daily[s] for s in market.symbols},
                           k_grid=cfg.k_grid, target_per_week=cfg.events_per_week_target,
                           start=start, end=first_test)
    wd = build_weekly_dataset(market, cfg, k=k, d_star=d_star, cost_rt=cost_rt)
    ds = wd.dataset
    first_pos = int(market.grid.searchsorted(first_test))
    embargo = cfg.horizon_bars + 1
    cv = WalkForwardCV(train_size=first_pos, test_size=cfg.test_bars, step=cfg.test_bars,
                       mode="anchored", min_train=first_pos, embargo_bars=embargo)
    n_bars = len(market.grid)
    oos_last = max(int(f.test_idx[-1]) for f in cv.split(n_bars))

    def run(kind: str, target: np.ndarray | None = None) -> list[FoldPredictions]:
        return walk_forward_fit_predict(
            ds, cv, lambda rows, y: fit_light_model(ds, rows, y, kind, cfg),
            n_bars=n_bars, embargo_bars=embargo, target=target)

    folds = {kind: run(kind) for kind in MODEL_KINDS}
    shuffle_aucs = [_oos_auc(run("ensemble", perm))
                    for perm in shuffled_targets(ds, cfg.seed, cfg.n_shuffle_replicates)]

    corr = market_features(np.log(pd.DataFrame(closes)).diff(),
                           window=cfg.corr_window)["avg_corr60"]
    inputs = BookInputs(market=market, avg_corr=corr, cost_rate=cost_rt / 2.0, impact=impact)
    kelly = SizingRule("kelly", cfg.kelly_multiple, cfg.baseline_risk_fraction,
                       cfg.resize_band, cost_rt)
    fixed = SizingRule("fixed", cfg.kelly_multiple, cfg.baseline_risk_fraction,
                       cfg.resize_band, cost_rt)
    window = slice(market.grid[first_pos], market.grid[oos_last])

    def book(cands: pd.DataFrame, sizing: SizingRule):
        return run_barrier_book(cands, inputs, RiskEngine(risk_cfg), sizing,
                                equity0=cfg.account_equity)

    books = {kind: book(trade_candidates(wd.events, folds[kind], cfg=cfg, cost_rt=cost_rt), kelly)
             for kind in MODEL_KINDS}
    baseline = book(_unfiltered(wd.events, cfg, first_pos, oos_last), fixed)
    reversed_wd = build_weekly_dataset(market, cfg, k=k, d_star=d_star, cost_rt=cost_rt,
                                       side_sign=-1.0)
    reversed_book = book(_unfiltered(reversed_wd.events, cfg, first_pos, oos_last), fixed)

    ens = books["ensemble"].returns.loc[window]
    base = baseline.returns.loc[window]
    ens_se = sharpe_with_se(ens, bars_per_year=BARS_PER_YEAR)
    base_se = sharpe_with_se(base, bars_per_year=BARS_PER_YEAR)
    rev_se = sharpe_with_se(reversed_book.returns.loc[window], bars_per_year=BARS_PER_YEAR)
    # `common_valid`: een vlakke dag (geen positie) is een waarneming van 0, geen gehalteerde
    # keten; beide boeken worden over dezelfde dagen vergeleken, de vlakke dagen inbegrepen.
    diff = sharpe_difference_test(ens, base, bars_per_year=BARS_PER_YEAR, seed=cfg.seed,
                                  align="common_valid")
    ci = block_bootstrap_ci(ens, bars_per_year=BARS_PER_YEAR, seed=cfg.seed)
    dsr = dsr_gate(ens.to_numpy(), trial_count=trial_count, config=val_cfg)

    taken = books["ensemble"].trades
    n_trades = int(len(taken))
    if n_trades:
        realized = _realized(taken)
        wins = int(((realized - cost_rt) > 0.0).sum())
        hit_low = float(stats.binomtest(wins, n_trades).proportion_ci(0.95, method="wilson").low)
        p_be_mean = float(np.mean([break_even_probability(b, cost_rt) for b in taken["barrier"]]))
        r_mult = (realized - cost_rt) / taken["barrier"].to_numpy(dtype=np.float64)
        years = (window.stop - window.start) / pd.Timedelta(days=365)
        per_year = n_trades / years
        mc = monte_carlo_drawdown_probability(
            r_mult, risk_fraction=float(taken["risk_fraction"].median()),
            n_trades=min(n_trades, max(1, int(round(per_year)))),
            drawdown=cfg.mc_max_drawdown, block_length=max(1, int(round(per_year / 26.0))),
            n_paths=cfg.mc_paths, seed=cfg.seed)
    else:
        hit_low, p_be_mean, mc = 0.0, 1.0, 1.0

    variants = np.column_stack([base.to_numpy()] + [books[k_].returns.loc[window].to_numpy()
                                                    for k_ in MODEL_KINDS])
    pbo = float(compute_pbo(variants, n_subsets=16)["pbo"])
    cpcv = _cpcv_path_sharpes(wd, cfg, inputs, risk_cfg, fixed)

    ens_folds = folds["ensemble"]
    y = np.concatenate([f.target for f in ens_folds])
    p = np.concatenate([f.probability for f in ens_folds])
    lo, _hi = cfg.shuffle_auc_band
    values = {
        "max_abs_shuffle_auc_deviation": float(max(abs(a - 0.5) for a in shuffle_aucs)),
        "reversed_minus_baseline_sharpe": float(rev_se.sharpe - base_se.sharpe),
        "ensemble_net_sharpe": float(ens_se.sharpe),
        "sharpe_diff_ci_low": float(diff.ci_low),
        "sharpe_ci_low": float(ci.low),
        "dsr": float(dsr.dsr),
        "hit_rate_ci_low_minus_break_even": float(hit_low - p_be_mean),
        "pbo": pbo,
        "mc_drawdown_probability_1y": float(mc),
        "n_trades": float(n_trades),
    }
    verdict = judge(criteria, values, stage="development")
    engine = RiskEngine(risk_cfg)
    record = {
        "verdict": verdict.as_dict(),
        "k": float(k), "k_rates": {str(kk): v for kk, v in rates.items()},
        "d_star": float(d_star),
        "n_events": int(len(ds)), "effective_n": float(ds.effective_n),
        "n_dropped_nan": wd.n_dropped_nan,
        "oos_window": [str(window.start), str(window.stop)],
        "oos_auc": {kind: _oos_auc(folds[kind]) for kind in MODEL_KINDS},
        "shuffle_aucs": shuffle_aucs, "shuffle_auc_band_low": lo,
        "brier_model": float(np.mean((p - y) ** 2)),
        "brier_base_rate": float(np.mean((y.mean() - y) ** 2)),
        "ece": float(expected_calibration_error(p, y)),
        "sharpe": {"ensemble": ens_se.sharpe, "ensemble_se": ens_se.se,
                   "baseline": base_se.sharpe, "reversed": rev_se.sharpe,
                   **{kind: _sharpe_or_none(books[kind].returns.loc[window])
                      for kind in MODEL_KINDS}},
        "sharpe_difference": {"delta": float(diff.delta_sharpe), "ci_low": float(diff.ci_low),
                              "ci_high": float(diff.ci_high), "p_value": float(diff.p_value)},
        "dsr": {"dsr": float(dsr.dsr), "passed": bool(dsr.passed), "M": int(trial_count.value)},
        "cpcv_path_sharpes": cpcv,
        "costs": {"fees": books["ensemble"].total_fees, "funding": books["ensemble"].total_funding,
                  "impact": books["ensemble"].total_impact, "round_trip": cost_rt},
        "n_risk_exits": int((taken["risk_exit_bar"] >= 0).sum()) if n_trades else 0,
        "risk_policy_hash": risk_config_hash(risk_cfg),
        "risk_audit_header": engine.audit_header(),
        "n_risk_resizes": books["ensemble"].n_resizes,
        "halted": books["ensemble"].halted,
    }
    return CampaignResult(values=values, verdict=verdict, record=record)


def main() -> None:
    """Draai de campagne op de gecertificeerde ontwikkelsample en schrijf het artefact."""
    root = Path.cwd()
    cfg = weekly_meta_config()
    lock = root / "artefacts/governance/holdout_lock.json"
    prereg_files = sorted((root / "artefacts/governance").glob("preregistration_*.json"))
    require(len(prereg_files) == 1, "Verwacht precies één bevroren preregistratie.",
            DataContractError, found=[p.name for p in prereg_files])
    prereg_path = prereg_files[0]
    prereg_id = json.loads(prereg_path.read_text(encoding="utf-8"))["preregistration_id"]
    prereg = require_preregistration(prereg_id, directory=root / "artefacts/governance")
    full = load_weekly_market(root, cfg.symbols)
    dev_index = development_slice(pd.DataFrame(index=full.grid), lock_path=lock).index
    market = full.truncate(dev_index[-1] + pd.Timedelta(hours=1))
    imp = load_config(root / "conf/execution/impact.yaml", ImpactConfig)
    impact = ImpactParams(
        eta=imp.eta, kappa_d=imp.kappa_d, status=ImpactStatus(imp.status), method=imp.method,
        data_hash=imp.data_hash, sample_size=imp.sample_size, period_start=imp.period_start,
        period_end=imp.period_end, instruments=imp.instruments, eta_ci_low=imp.eta_ci_low,
        eta_ci_high=imp.eta_ci_high)
    result = run_campaign_on_market(
        market, cfg, criteria=prereg.stop_criteria,
        trial_count=frozen_trial_count(prereg_path),
        exec_cfg=load_config(root / "conf/execution/fees.yaml", ExecutionConfig),
        val_cfg=load_config(root / "conf/validation/default.yaml", ValidationConfig),
        risk_cfg=load_config(root / "conf/risk/default.yaml", RiskConfig),
        impact=impact)
    record = {**result.record, "values": result.values, "preregistration_id": prereg_id,
              "git_sha": current_git_sha(), "source_hashes": full.source_hashes,
              "dev_last_bar": str(market.grid[-1])}
    (root / ARTEFACT).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(json.dumps(result.verdict.as_dict(), indent=2))


if __name__ == "__main__":
    main()
