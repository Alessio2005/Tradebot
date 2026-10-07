"""Het programma robuust boek v1: bevriezen, meten op W_DEV, één poortlezing.

    python -m tradebot.systematic.programme freeze   # boek M = 7, bevries de preregistratie
    python -m tradebot.systematic.programme run      # W_DEV: alle trials, batterij, poorten, score
    python -m tradebot.systematic.programme gate     # EEN lezing van het poortsample

De volgorde is afgedwongen: `run` weigert zonder bevroren preregistratie, en `run` ziet
het poortsample niet -- de markt wordt vóór elke berekening afgekapt op `gate_start`.
`gate` registreert de lezing in `artefacts/governance/holdout_lock.json` VOORDAT er één
getal uit het poortsample wordt berekend. Ontwerp:
`docs/superpowers/specs/2026-10-07-robust-book-design.md`.
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from ..data.weekly_market import load_weekly_market
from ..execution.trade_costs import load_impact_params
from ..features.base import DataRegister
from ..features.registry import current_git_sha
from ..registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from ..registry.preregistration import (
    freeze_metadata,
    freeze_preregistration,
    load_preregistration_spec,
    require_preregistration,
)
from ..registry.weekly_programme import certified_data_hashes
from ..schemas.config import ExecutionConfig, load_config
from ..schemas.robust_book import RobustBookConfig, robust_book_config
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from ..validation.holdout import gate_slice
from ..validation.inference import calibrate_block_length, circular_block_indices
from .book import BookResult, CostSpec, run_book
from .evaluate import (
    BOOTSTRAP_SEED,
    dsr_record,
    gate_z,
    pbo_record,
    regime_breakdown,
    robustness_score,
    summarize,
    yearly,
)
from .market import BARS_PER_YEAR, BookMarket, from_weekly_market
from .sleeves import (
    SizingParams,
    SleeveTargets,
    apply_caps,
    baseline_btc,
    baseline_equal_weight,
    build_sleeve,
    combine,
)

__all__ = ["CANDIDATES", "TRIALS", "freeze", "read_gate", "run_programme"]

ROOT = Path(__file__).resolve().parents[3]
PROGRAMME_UNIT = "robust_book_v1_programme"
SPEC_PATH = Path("conf/research/preregistration_robust_book.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v1")
LEDGER_PATH = Path("artefacts/governance/hypothesis_ledger.json")
LOCK_PATH = Path("artefacts/governance/holdout_lock.json")
PREREG_DIR = Path("artefacts/governance")

REFERENCES = ("B1_BTC_HOLD", "B2_EW_HOLD")
CANDIDATES = ("C1_TREND_LS", "C2_TREND_LF", "C3_VOLMAN_CORE", "C4_CARRY_XS", "C5_COMBO")
TRIALS = REFERENCES + CANDIDATES
SLEEVES = CANDIDATES[:-1]
TREND = ("C1_TREND_LS", "C2_TREND_LF")

HYPOTHESES = {
    "B1_BTC_HOLD": "Referentie: 1,0x long BTC-perp.",
    "B2_EW_HOLD": "Referentie: gelijk gewogen long over de levende munten.",
    "C1_TREND_LS": "Tijdreeks-momentum, long-short, log-raster 5..160 dagen, vol-getarget.",
    "C2_TREND_LF": "Tijdreeks-momentum, long-flat, zelfde raster, vol-getarget.",
    "C3_VOLMAN_CORE": "Vol-gemanaged long BTC+ETH, gelijk risico (marktpremie + vol-clustering).",
    "C4_CARRY_XS": "Dollar-neutrale funding-carry: short hoogste, long laagste bekende funding.",
    "C5_COMBO": "Gelijk-risicocombinatie van de sleeves met TRAIN-Sharpe > 0.",
}

N_SEEDS = 20
NOISE_FRACTION = 0.25
MISSING_FRACTION = 0.05
MC_PATHS = 2000


# --------------------------------------------------------------------------- #
# Invoer
# --------------------------------------------------------------------------- #
def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def programme_parameters(cfg: RobustBookConfig) -> dict[str, Any]:
    return json.loads(cfg.model_dump_json())


def cost_spec(root: Path, cfg: RobustBookConfig) -> CostSpec:
    exe = load_config(root / "conf/execution/fees.yaml", ExecutionConfig)
    return CostSpec(taker_fee=exe.taker_fee_bps * 1e-4,
                    half_spread=exe.assumed_half_spread_bps * 1e-4,
                    impact=load_impact_params(root / "conf/execution/impact.yaml"),
                    aum_usd=cfg.execution.aum_usd)


def load_market(root: Path, cfg: RobustBookConfig) -> BookMarket:
    return from_weekly_market(load_weekly_market(root, cfg.symbols))


def _prereg(root: Path, cfg: RobustBookConfig):
    register = DataRegister(root / "artefacts/governance/data_hashes.json")
    hashes = certified_data_hashes(register, cfg.symbols)
    return load_preregistration_spec(root / SPEC_PATH, data_hashes=hashes,
                                     parameters=programme_parameters(cfg)), hashes


# --------------------------------------------------------------------------- #
# Bevriezen
# --------------------------------------------------------------------------- #
def freeze(root: Path = ROOT, *, git_sha: str | None = None) -> Path:
    """Boek de zeven geplande trials en bevries de preregistratie. Eenmalig."""
    cfg = robust_book_config(root / "conf/model/robust_book.yaml")
    prereg, hashes = _prereg(root, cfg)
    require(prereg.planned_trials == cfg.planned_trials,
            "planned_trials in de preregistratie wijkt af van de config.", DataContractError,
            spec=prereg.planned_trials, config=cfg.planned_trials)
    sha = git_sha or current_git_sha()
    ledger = HypothesisLedger(root / LEDGER_PATH)
    already = [e for e in ledger.entries()
               if e.get("preregistration_id") == prereg.preregistration_id]
    require(not already, "Dit programma is al geboekt.", DataContractError,
            preregistration_id=prereg.preregistration_id)
    params = programme_parameters(cfg)
    ledger.append(LedgerEntry.from_config(
        wave=1, unit=PROGRAMME_UNIT, market="crypto", config=params, git_sha=sha,
        data_hash=hash_config(dict(hashes)), preregistration_id=prereg.preregistration_id,
        n_trials=cfg.planned_trials, result="interim",
        notes="Zeven geplande trials, geboekt vóór de eerste run: B1, B2 (referenties), "
              "C1 trend-LS, C2 trend-LF, C3 vol-gemanaged kern, C4 funding-carry, C5 combinatie."))
    return freeze_preregistration(prereg, git_sha=sha,
                                  ledger_total_at_freeze=ledger.total_n_hypotheses(),
                                  directory=root / PREREG_DIR)


# --------------------------------------------------------------------------- #
# Trials bouwen
# --------------------------------------------------------------------------- #
def _every(t: SleeveTargets, k: int) -> SleeveTargets:
    """Herbalanceer alleen elke k bars (vaste fase vanaf de eerste besluitbar)."""
    if k <= 1:
        return t
    reb = t.rebalance.to_numpy()
    pos = np.cumsum(reb) - 1
    mask = pd.Series(reb & (pos % k == 0), index=t.rebalance.index)
    return SleeveTargets(name=t.name, weights=t.weights.where(mask, np.nan, axis=0),
                         rebalance=mask)


def build_trial(
    name: str,
    market: BookMarket,
    cfg: RobustBookConfig,
    *,
    included: Sequence[str] = (),
    sizing: SizingParams | None = None,
    lookbacks: Sequence[int] | None = None,
    carry_window: int | None = None,
    carry_rebalance: int | None = None,
) -> SleeveTargets:
    if name == "B1_BTC_HOLD":
        return baseline_btc(market, cfg)
    if name == "B2_EW_HOLD":
        return baseline_equal_weight(market, cfg)
    kw = {"sizing": sizing, "lookbacks": lookbacks, "carry_window": carry_window,
          "carry_rebalance": carry_rebalance}
    if name == "C5_COMBO":
        require(len(included) > 0, "C5 zonder opgenomen sleeves.", DataContractError)
        parts = [build_sleeve(s, market, cfg, **kw) for s in included]
        return combine(name, parts, market, cfg, sizing=sizing)
    return build_sleeve(name, market, cfg, **kw)


def run_targets(t: SleeveTargets, market: BookMarket, costs: CostSpec, *, lag: int) -> BookResult:
    return run_book(t.weights, t.rebalance, market, costs, lag=lag)


def _w(res: BookResult, a: pd.Timestamp, b: pd.Timestamp) -> BookResult:
    return res.window(a, b)


def _sr(res: BookResult) -> float:
    r = res.net
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * math.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# De robuustheidsbatterij (spec §7) -- gevoeligheden, geen trials
# --------------------------------------------------------------------------- #
def _scaled_lookbacks(base: Sequence[int], factor: float) -> tuple[int, ...]:
    return tuple(sorted({max(2, int(round(x * factor))) for x in base}))


def perturbation_family(name: str, cfg: RobustBookConfig, included: Sequence[str]) -> dict[str, dict]:
    """De parameterverstoringen die op deze kandidaat van toepassing zijn."""
    parts = set(included) if name == "C5_COMBO" else {name}
    base = SizingParams.from_config(cfg)
    fam: dict[str, dict] = {}
    if parts & set(TREND):
        for f in (0.5, 0.75, 1.25, 1.5, 2.0):
            fam[f"lookbacks_x{f}"] = {"lookbacks": _scaled_lookbacks(cfg.trend.lookbacks, f)}
    for span in (20, 40, 90, 120):
        fam[f"vol_span_{span}"] = {"sizing": replace(base, vol_span=span)}
    if "C4_CARRY_XS" in parts:
        for win in (3, 14, 30):
            fam[f"carry_window_{win}"] = {"carry_window": win}
    for k in (1.0, 3.0):
        fam[f"max_scale_{k}"] = {"sizing": replace(base, max_scale=k)}
    return fam


def _noisy(t: SleeveTargets, market: BookMarket, cfg: RobustBookConfig, seed: int) -> SleeveTargets:
    rng = np.random.default_rng(seed)
    w = t.weights
    sd = w.std(skipna=True).fillna(0.0)
    noise = pd.DataFrame(rng.normal(0.0, 1.0, size=w.shape), index=w.index,
                         columns=w.columns) * (NOISE_FRACTION * sd)
    noisy = (w + noise).where(market.close.notna(), 0.0).where(w.notna())
    capped = apply_caps(noisy.fillna(0.0), per_asset_cap=cfg.sizing.per_asset_cap,
                        gross_cap=cfg.sizing.gross_cap).where(w.notna())
    return SleeveTargets(name=t.name, weights=capped, rebalance=t.rebalance)


def _missing(t: SleeveTargets, seed: int) -> SleeveTargets:
    rng = np.random.default_rng(seed)
    drop = rng.random(len(t.rebalance)) < MISSING_FRACTION
    reb = t.rebalance & ~pd.Series(drop, index=t.rebalance.index)
    return SleeveTargets(name=t.name, weights=t.weights.where(reb, np.nan, axis=0), rebalance=reb)


def _mc_drawdown(r: pd.Series, *, horizon: int, threshold: float) -> dict[str, float]:
    x = r.to_numpy()
    block = calibrate_block_length(x)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = circular_block_indices(len(x), block, MC_PATHS, rng)[:, :horizon]
    eq = np.cumprod(1.0 + x[idx], axis=1)
    peak = np.maximum(np.maximum.accumulate(eq, axis=1), 1.0)
    mdds = np.max(1.0 - eq / peak, axis=1)
    return {"horizon_bars": horizon, "block_length": int(block), "n_paths": MC_PATHS,
            "p_mdd_above_threshold": float((mdds > threshold).mean()),
            "threshold": threshold, "median_mdd": float(np.median(mdds)),
            "p95_mdd": float(np.quantile(mdds, 0.95))}


def battery(
    name: str,
    market: BookMarket,
    cfg: RobustBookConfig,
    costs: CostSpec,
    *,
    included: Sequence[str],
    base: BookResult,
    w_dev: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[dict[str, Any], dict[str, pd.Series]]:
    """Alle gevoeligheden van één kandidaat op W_DEV. Geeft ook de verstoringsfamilie terug."""
    a, b = w_dev
    lag = cfg.execution.lag_bars
    base_sr = _sr(_w(base, a, b))
    t_base = build_trial(name, market, cfg, included=included)
    out: dict[str, Any] = {"base_sharpe": base_sr}

    def sr_of(t: SleeveTargets, c: CostSpec = costs, lg: int = lag, mk: BookMarket = market) -> float:
        return _sr(_w(run_targets(t, mk, c, lag=lg), a, b))

    # Parameterverstoring -> plateau en PBO.
    family: dict[str, pd.Series] = {"base": _w(base, a, b).net}
    pert: dict[str, float] = {}
    for label, kw in perturbation_family(name, cfg, included).items():
        res = _w(run_targets(build_trial(name, market, cfg, included=included, **kw),
                             market, costs, lag=lag), a, b)
        family[label] = res.net
        pert[label] = _sr(res)
    out["perturbation"] = pert
    vals = np.array(list(pert.values()))
    out["plateau_fraction"] = float((vals > 0.5 * base_sr).mean()) if base_sr > 0 else 0.0
    out["min_perturbation_sharpe"] = float(vals.min())
    out["median_perturbation_sharpe"] = float(np.median(vals))

    out["costs"] = {
        "x0": sr_of(t_base, costs.scaled(multiplier=0.0)),
        "x2": sr_of(t_base, costs.scaled(multiplier=2.0)),
        "x3": sr_of(t_base, costs.scaled(multiplier=3.0)),
        "plus_10bp_slippage": sr_of(t_base, costs.scaled(extra_slippage=10e-4)),
    }
    out["capacity"] = {f"aum_{int(x):d}": sr_of(t_base, costs.scaled(aum_usd=x))
                       for x in (1e6, 1e7, 1e8)}
    out["delay"] = {f"lag_{k}": sr_of(t_base, lg=k) for k in (2, 3)}
    noise = [sr_of(_noisy(t_base, market, cfg, s)) for s in range(1, N_SEEDS + 1)]
    out["signal_noise"] = {"min": float(np.min(noise)), "median": float(np.median(noise)),
                           "max": float(np.max(noise)), "n_seeds": N_SEEDS,
                           "noise_fraction_of_weight_sd": NOISE_FRACTION}
    missing = [sr_of(_missing(t_base, s)) for s in range(1, N_SEEDS + 1)]
    out["missing_data"] = {"min": float(np.min(missing)), "median": float(np.median(missing)),
                           "max": float(np.max(missing)), "n_seeds": N_SEEDS,
                           "fraction_of_rebalances_skipped": MISSING_FRACTION}
    reb: dict[str, float] = {}
    if name == "C4_CARRY_XS":
        for k in (3, 14):
            reb[f"every_{k}"] = sr_of(build_trial(name, market, cfg, carry_rebalance=k))
    else:
        for k in (3, 7):
            reb[f"every_{k}"] = sr_of(_every(t_base, k))
    out["rebalance"] = reb

    universes: dict[str, tuple[str, ...]] = {
        f"without_{s}": tuple(x for x in market.symbols if x != s) for s in market.symbols}
    universes["btc_eth_only"] = ("BTCUSDT", "ETHUSDT")
    universes["without_btc_alts_only"] = tuple(x for x in market.symbols if x != "BTCUSDT")
    uni: dict[str, float] = {}
    for label, syms in universes.items():
        sub = market.subset(syms)
        if name in ("C3_VOLMAN_CORE",) and not set(cfg.core.assets) & set(syms):
            continue
        if name == "C5_COMBO" and "C3_VOLMAN_CORE" in included and not set(cfg.core.assets) & set(syms):
            continue
        uni[label] = sr_of(build_trial(name, sub, cfg, included=included), mk=sub)
    out["universe"] = uni
    loo = [v for k, v in uni.items() if k.startswith("without_") and k != "without_btc_alts_only"]
    out["leave_one_out_positive_fraction"] = float(np.mean([v > 0 for v in loo])) if loo else 0.0

    starts = {f"start_plus_{m}m": a + pd.DateOffset(months=m) for m in (3, 6, 9, 12)}
    ends = {f"end_minus_{m}m": b - pd.DateOffset(months=m) for m in (3, 6, 12)}
    out["start_dates"] = {k: _sr(_w(base, s, b)) for k, s in starts.items()}
    out["end_dates"] = {k: _sr(_w(base, a, e)) for k, e in ends.items()}
    r = _w(base, a, b).net
    out["monte_carlo_drawdown"] = {
        "one_year": _mc_drawdown(r, horizon=365, threshold=0.25),
        "full_window": _mc_drawdown(r, horizon=len(r), threshold=0.25)}
    return out, family


# --------------------------------------------------------------------------- #
# W_DEV
# --------------------------------------------------------------------------- #
def _gates(m: dict[str, float], prereg) -> dict[str, dict[str, Any]]:
    out = {}
    for c in prereg.stop_criteria:
        if c.metric in m and c.name != "promotion_requires_all_clear":
            v = float(m[c.metric])
            out[c.name] = {"metric": c.metric, "value": v, "operator": c.operator,
                           "threshold": c.threshold, "binds": bool(c.binds(v)) or not math.isfinite(v)}
    return out


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float,
                               ensure_ascii=False), encoding="utf-8")


def run_programme(root: Path = ROOT, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = robust_book_config(root / "conf/model/robust_book.yaml")
    prereg, hashes = _prereg(root, cfg)
    require_preregistration(prereg.preregistration_id, directory=root / PREREG_DIR)
    meta = freeze_metadata(prereg.preregistration_id, directory=root / PREREG_DIR)
    win = cfg.windows
    gate_start = _ts(win.gate_start)
    lock = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
    require(pd.Timestamp(lock["split_utc"]) == gate_start,
            "gate_start wijkt af van de bevroren holdout-split.", DataContractError,
            lock=lock["split_utc"], config=win.gate_start)
    # Het poortsample bestaat voor deze functie niet.
    market = load_market(root, cfg).truncate(gate_start)
    costs = cost_spec(root, cfg)
    lag = cfg.execution.lag_bars
    w_dev = (_ts(win.w_dev_start), _ts(win.w_dev_end))
    train = (_ts(win.w_dev_start), _ts(win.train_end))
    val = (_ts(win.validate_start), _ts(win.w_dev_end))
    sha = current_git_sha()

    results: dict[str, BookResult] = {}
    for name in REFERENCES + SLEEVES:
        log(f"run {name}")
        results[name] = run_targets(build_trial(name, market, cfg), market, costs, lag=lag)
    train_sr = {s: _sr(_w(results[s], *train)) for s in SLEEVES}
    included = tuple(s for s in SLEEVES if train_sr[s] > cfg.combo.inclusion_min_train_sharpe)
    log(f"C5 neemt op (TRAIN-Sharpe > {cfg.combo.inclusion_min_train_sharpe}): {included}")
    if included:
        results["C5_COMBO"] = run_targets(build_trial("C5_COMBO", market, cfg, included=included),
                                          market, costs, lag=lag)

    per_bar = [summarize(_w(results[n], *w_dev), bootstrap=False)["sharpe_per_bar"]
               for n in TRIALS if n in results]
    m_prog = int(meta["m_trials"])
    m_wide = m_prog + int(cfg.known_prior_trials)
    records: dict[str, dict[str, Any]] = {}
    for name in TRIALS:
        rec: dict[str, Any] = {
            "trial": name, "role": "reference" if name in REFERENCES else "candidate",
            "hypothesis": HYPOTHESES[name], "preregistration_id": prereg.preregistration_id,
            "git_sha": sha, "frozen_git_sha": meta["git_sha"], "data_hashes": dict(hashes),
            "parameters": programme_parameters(cfg), EVIDENCE_KEY: NOT_ADMISSIBLE,
            "windows": {"train": [win.w_dev_start, win.train_end],
                        "validate": [win.validate_start, win.w_dev_end],
                        "w_dev": [win.w_dev_start, win.w_dev_end],
                        "gate": [win.gate_start, "unread"]},
            "execution": {"lag_bars": lag, "costs": costs.as_record()},
        }
        if name == "C5_COMBO":
            rec["included"] = list(included)
            rec["train_sharpe_of_sleeves"] = train_sr
        if name not in results:
            rec["status"] = "not_constructible"
            records[name] = rec
            continue
        res = results[name]
        dev = summarize(_w(res, *w_dev))
        rec["w_dev"] = dev
        rec["train"] = summarize(_w(res, *train))
        rec["validate"] = summarize(_w(res, *val))
        rec["yearly"] = yearly(_w(res, *w_dev))
        rec["regimes"] = regime_breakdown(_w(res, *w_dev), market)
        rec["dsr"] = {
            f"m_{m_prog}": dsr_record(_w(res, *w_dev).net, n_trials=m_prog, trial_sharpes_per_bar=per_bar),
            f"m_{m_wide}": dsr_record(_w(res, *w_dev).net, n_trials=m_wide, trial_sharpes_per_bar=per_bar),
        }
        if name in CANDIDATES:
            log(f"batterij {name}")
            bat, family = battery(name, market, cfg, costs, included=included, base=res,
                                  w_dev=w_dev)
            rec["battery"] = bat
            rec["pbo"] = pbo_record(family)
            metrics = {
                "net_sharpe_w_dev": dev["sharpe"], "net_cagr_w_dev": dev["cagr"],
                "net_sharpe_train": rec["train"]["sharpe"],
                "net_sharpe_validate": rec["validate"]["sharpe"],
                "max_drawdown_w_dev": dev["max_drawdown"],
                "sharpe_ratio_2x_cost_over_base": bat["costs"]["x2"] / dev["sharpe"]
                if dev["sharpe"] > 0 else float("nan"),
                "sharpe_ratio_lag2_over_base": bat["delay"]["lag_2"] / dev["sharpe"]
                if dev["sharpe"] > 0 else float("nan"),
                "plateau_fraction": bat["plateau_fraction"],
                "min_perturbation_sharpe": bat["min_perturbation_sharpe"],
                "sharpe_ci_low_w_dev": dev["sharpe_ci_low"],
                "dsr_w_dev": rec["dsr"][f"m_{m_prog}"]["dsr"],
                "pbo": rec["pbo"]["pbo"],
            }
            rec["gate_metrics"] = metrics
            rec["gates"] = _gates(metrics, prereg)
            rec["n_binding_w_dev"] = int(sum(g["binds"] for g in rec["gates"].values()))
            rec["robustness_score"] = robustness_score({
                "sharpe": dev["sharpe"], "sharpe_train": rec["train"]["sharpe"],
                "sharpe_validate": rec["validate"]["sharpe"],
                "sharpe_2x_cost": bat["costs"]["x2"], "sharpe_lag2": bat["delay"]["lag_2"],
                "plateau_fraction": bat["plateau_fraction"],
                "p_sharpe_gt_0": dev["p_sharpe_gt_0"], "max_drawdown": dev["max_drawdown"],
                "pbo": rec["pbo"]["pbo"],
                "leave_one_out_positive_fraction": bat["leave_one_out_positive_fraction"]})
        records[name] = rec
        _dump(root / ARTEFACT_DIR / f"{name}.json", rec)
        daily = _w(res, *w_dev).frame
        daily.to_csv(root / ARTEFACT_DIR / f"{name}_daily_w_dev.csv", float_format="%.10g")

    scored = {n: records[n]["robustness_score"]["total"] for n in CANDIDATES
              if "robustness_score" in records[n]}
    selected = max(scored, key=lambda k: scored[k])
    summary = {
        "programme": "robust_book_v1", "preregistration_id": prereg.preregistration_id,
        "git_sha": sha, "m_programme": m_prog, "m_including_known_prior": m_wide,
        "c5_included": list(included), "robustness_scores": scored, "selected": selected,
        "passes_all_w_dev_gates": {n: records[n].get("n_binding_w_dev", None) == 0
                                   for n in CANDIDATES if n in records},
        "table": {n: {k: records[n]["w_dev"][k] for k in (
            "sharpe", "sharpe_se", "cagr", "ann_vol", "max_drawdown", "sortino", "calmar",
            "ann_turnover", "avg_gross_leverage", "sharpe_ci_low", "sharpe_ci_high")}
            for n in TRIALS if "w_dev" in records[n]},
        EVIDENCE_KEY: NOT_ADMISSIBLE,
    }
    _dump(root / ARTEFACT_DIR / "programme_w_dev.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# Het poortsample: één lezing, één kandidaat
# --------------------------------------------------------------------------- #
def read_gate(root: Path = ROOT, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = robust_book_config(root / "conf/model/robust_book.yaml")
    prereg, _ = _prereg(root, cfg)
    require_preregistration(prereg.preregistration_id, directory=root / PREREG_DIR)
    summary = json.loads((root / ARTEFACT_DIR / "programme_w_dev.json").read_text(encoding="utf-8"))
    require(summary["preregistration_id"] == prereg.preregistration_id,
            "De W_DEV-run hoort bij een andere preregistratie.", DataContractError)
    name = summary["selected"]
    included = tuple(summary["c5_included"])
    full = load_market(root, cfg)
    hypothesis_id = f"robust_book_v1/{prereg.preregistration_id}/{name}"
    # De lezing wordt geregistreerd VOORDAT er een getal uit het poortsample bestaat.
    gate_close = gate_slice(full.close, lock_path=root / LOCK_PATH, hypothesis_id=hypothesis_id)
    a = pd.Timestamp(gate_close.index[0])
    b = pd.Timestamp(gate_close.index[-1])
    costs = cost_spec(root, cfg)
    res = run_targets(build_trial(name, full, cfg, included=included), full, costs,
                      lag=cfg.execution.lag_bars)
    gate = summarize(_w(res, a, b))
    dev_rec = json.loads((root / ARTEFACT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    z = gate_z(gate["sharpe"], gate["sharpe_se"], dev_rec["w_dev"]["sharpe"])
    metrics = {"gate_sharpe_z_vs_dev": z, "max_drawdown_gate": gate["max_drawdown"]}
    gates = _gates(metrics, prereg)
    refs = {}
    for ref in REFERENCES:
        rr = run_targets(build_trial(ref, full, cfg), full, costs, lag=cfg.execution.lag_bars)
        refs[ref] = summarize(_w(rr, a, b))
    # Consistentiecheck: de W_DEV-cijfers van de volledige run zijn die van de afgekapte run.
    dev_again = _sr(_w(res, _ts(cfg.windows.w_dev_start), _ts(cfg.windows.w_dev_end)))
    require(abs(dev_again - dev_rec["w_dev"]["sharpe"]) < 1e-9,
            "De volledige run reproduceert de W_DEV-Sharpe niet: het poortsample lekte.",
            DataContractError, full=dev_again, truncated=dev_rec["w_dev"]["sharpe"])
    n_binding = int(sum(g["binds"] for g in gates.values())) + int(dev_rec["n_binding_w_dev"])
    verdict = "promote_to_paper_trading" if n_binding == 0 else (
        "falsified_on_gate" if any(g["binds"] for g in gates.values()) else "archived_w_dev_gates")
    out = {
        "hypothesis_id": hypothesis_id, "candidate": name, "included": list(included),
        "gate_window": [str(a.date()), str(b.date())], "gate": gate,
        "yearly": yearly(_w(res, a, b)), "references_on_gate": refs,
        "gate_metrics": metrics, "gate_gates": gates, "n_binding_total": n_binding,
        "verdict": verdict, "w_dev_sharpe": dev_rec["w_dev"]["sharpe"],
        "caveat": ("Voor een trendkandidaat is dit venster niet maagdelijk: het ongemergde "
                   "elegant-dijkstra-programma las 2024-01..2026-06-23 voor de trendfamilie "
                   "en jolly-feynman las 2026-06-24..2026-08-23 voor een BTC-trendregel."),
        EVIDENCE_KEY: NOT_ADMISSIBLE,
    }
    _dump(root / ARTEFACT_DIR / "gate_read.json", out)
    res.window(a, b).frame.to_csv(root / ARTEFACT_DIR / f"{name}_daily_gate.csv", float_format="%.10g")
    log(f"poort {name}: Sharpe {gate['sharpe']:.3f} (SE {gate['sharpe_se']:.3f}), z={z:.2f}, {verdict}")
    return out


def book_verdict(root: Path = ROOT, *, git_sha: str | None = None) -> None:
    """Boek het oordeel als amendement op de programma-entry (nul trials)."""
    cfg = robust_book_config(root / "conf/model/robust_book.yaml")
    prereg, hashes = _prereg(root, cfg)
    gate = json.loads((root / ARTEFACT_DIR / "gate_read.json").read_text(encoding="utf-8"))
    summary = json.loads((root / ARTEFACT_DIR / "programme_w_dev.json").read_text(encoding="utf-8"))
    params = programme_parameters(cfg)
    result = {"promote_to_paper_trading": "accepted", "falsified_on_gate": "falsified"}.get(
        gate["verdict"], "archived")
    HypothesisLedger(root / LEDGER_PATH).append(LedgerEntry.from_config(
        wave=1, unit="robust_book_v1_verdict", market="crypto", config=params,
        git_sha=git_sha or current_git_sha(), data_hash=hash_config(dict(hashes)),
        preregistration_id=prereg.preregistration_id, n_trials=0, result=result,
        amends=hash_config(params),
        metrics={"selected": gate["candidate"], "w_dev_sharpe": gate["w_dev_sharpe"],
                 "gate_sharpe": gate["gate"]["sharpe"], "verdict": gate["verdict"],
                 "robustness_scores": summary["robustness_scores"]},
        notes="Oordeel robuust boek v1 na W_DEV en één poortlezing."))


def main(argv: Sequence[str]) -> None:
    require(len(argv) == 1 and argv[0] in ("freeze", "run", "gate", "verdict"),
            "Gebruik: python -m tradebot.systematic.programme {freeze|run|gate|verdict}",
            DataContractError)
    cmd = argv[0]
    if cmd == "freeze":
        print(freeze())
    elif cmd == "run":
        print(json.dumps(run_programme(), indent=2, default=float))
    elif cmd == "gate":
        print(json.dumps(read_gate(), indent=2, default=float))
    else:
        book_verdict()


if __name__ == "__main__":
    main(sys.argv[1:])
