"""Robuust boek v2 (breedte): bevriezen, meten op W_DEV, één holdout-lezing.

    python -I -m tradebot.systematic.programme_v2 freeze
    python -I -m tradebot.systematic.programme_v2 run
    python -I -m tradebot.systematic.programme_v2 holdout
    python -I -m tradebot.systematic.programme_v2 verdict

Dezelfde discipline als v1 (`programme.py`): `run` weigert zonder bevroren
preregistratie en kapt de markt af op `holdout_start` vóór elke berekening; `holdout`
registreert de lezing in een EIGEN slot (`holdout_lock_binance_um.json`) voordat er één
getal uit de holdout bestaat.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from ..execution.trade_costs import load_impact_params
from ..features.registry import current_git_sha
from ..registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from ..registry.preregistration import (
    freeze_preregistration,
    load_preregistration_spec,
    require_preregistration,
)
from ..schemas.robust_book_v2 import RobustBookV2Config, robust_book_v2_config
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from ..validation.holdout import freeze_holdout, gate_slice
from ..validation.inference import calibrate_block_length, circular_block_indices
from .book import BookResult, CostSpec, run_book
from .breadth import BreadthMarket, combine_scaled, load_breadth_market, sleeve_targets
from .evaluate import (
    BOOTSTRAP_SEED,
    dsr_record,
    gate_z,
    pbo_record,
    robustness_score,
    summarize,
    yearly,
)
from .market import BARS_PER_YEAR
from .sleeves import SleeveTargets, apply_caps

__all__ = ["CANDIDATES", "TRIALS", "freeze", "read_holdout", "run_programme"]

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path("conf/model/robust_book_v2.yaml")
SPEC_PATH = Path("conf/research/preregistration_robust_book_v2.yaml")
PANEL_DIR = Path("data/binance_vision/panels")
MANIFEST = Path("artefacts/data/binance_um_manifest.json")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v2")
LEDGER_PATH = Path("artefacts/governance/hypothesis_ledger.json")
LOCK_PATH = Path("artefacts/governance/holdout_lock_binance_um.json")
PREREG_DIR = Path("artefacts/governance")
PROGRAMME_UNIT = "robust_book_v2_programme"

REFERENCES = ("R1_BTC_HOLD", "R2_EW50_HOLD")
SLEEVES = ("X1_XSMOM", "X2_XSCARRY", "X3_TREND_LS", "X4_TREND_LF")
CANDIDATES = (*SLEEVES, "X5_COMBO")
TRIALS = REFERENCES + CANDIDATES
HYPOTHESES = {
    "R1_BTC_HOLD": "Referentie: 1,0x long BTC-perp.",
    "R2_EW50_HOLD": "Referentie: gelijk gewogen long over het top-50-universum.",
    "X1_XSMOM": "Cross-sectioneel momentum 7-56 d, dollar-neutraal, wekelijks.",
    "X2_XSCARRY": "Funding-carry: short hoogste, long laagste bekende funding, dollar-neutraal.",
    "X3_TREND_LS": "Tijdreeks-momentum long-short over het universum.",
    "X4_TREND_LF": "Tijdreeks-momentum long-flat over het universum.",
    "X5_COMBO": "Gelijk-risicocombinatie van {X1, X2, X4} met TRAIN-Sharpe > 0.",
}
N_SEEDS = 10
MC_PATHS = 2000


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _params(cfg: RobustBookV2Config) -> dict[str, Any]:
    return json.loads(cfg.model_dump_json())


def _data_hashes(root: Path) -> tuple[tuple[str, str], ...]:
    m = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    panels = sorted((root / PANEL_DIR).glob("*.parquet"))
    out = [("binance_um/objects", m["objects_sha256"][:16])]
    out += [(f"binance_um/panel/{p.stem}", hashlib.sha256(p.read_bytes()).hexdigest()[:16])
            for p in panels]
    return tuple(sorted(out))


def _prereg(root: Path, cfg: RobustBookV2Config):
    hashes = _data_hashes(root)
    return load_preregistration_spec(root / SPEC_PATH, data_hashes=hashes,
                                     parameters=_params(cfg)), hashes


def cost_spec(root: Path, cfg: RobustBookV2Config) -> CostSpec:
    c = cfg.costs
    return CostSpec(taker_fee=c.taker_fee_bps * 1e-4, half_spread=c.min_half_spread_bps * 1e-4,
                    impact=load_impact_params(root / "conf/execution/impact.yaml"),
                    aum_usd=cfg.execution.aum_usd, impact_eta=c.impact_y)


# --------------------------------------------------------------------------- #
# Bevriezen
# --------------------------------------------------------------------------- #
def freeze(root: Path = ROOT, *, git_sha: str | None = None) -> Path:
    cfg = robust_book_v2_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    require(prereg.planned_trials == cfg.planned_trials, "planned_trials wijkt af.",
            DataContractError)
    sha = git_sha or current_git_sha()
    ledger = HypothesisLedger(root / LEDGER_PATH)
    require(not [e for e in ledger.entries()
                 if e.get("preregistration_id") == prereg.preregistration_id],
            "Dit programma is al geboekt.", DataContractError)
    ledger.append(LedgerEntry.from_config(
        wave=2, unit=PROGRAMME_UNIT, market="crypto", config=_params(cfg), git_sha=sha,
        data_hash=hash_config(dict(hashes)), preregistration_id=prereg.preregistration_id,
        n_trials=cfg.planned_trials, result="interim",
        notes="Zeven geplande trials, geboekt vóór de eerste meting op het Binance-universum: "
              "R1, R2 (referenties), X1 XS-momentum, X2 XS-carry, X3 trend-LS, X4 trend-LF, "
              "X5 combinatie."))
    path = freeze_preregistration(prereg, git_sha=sha,
                                  ledger_total_at_freeze=ledger.total_n_hypotheses(),
                                  directory=root / PREREG_DIR)
    lock = root / LOCK_PATH
    if not lock.exists():
        freeze_holdout(split_utc=f"{cfg.windows.holdout_start}T00:00:00+00:00", out=lock,
                       git_sha=sha)
    return path


# --------------------------------------------------------------------------- #
# Trials
# --------------------------------------------------------------------------- #
def build(name: str, m: BreadthMarket, cfg: RobustBookV2Config, *, included: Sequence[str] = (),
          **kw: Any) -> SleeveTargets:
    if name == "X5_COMBO":
        require(len(included) > 0, "X5 zonder opgenomen sleeves.", DataContractError)
        parts = [sleeve_targets(s, m, cfg, **kw) for s in included]
        return combine_scaled(name, parts, m, cfg, vol_span=kw.get("vol_span"),
                              max_scale=kw.get("max_scale"))
    return sleeve_targets(name, m, cfg, **kw)


def run(t: SleeveTargets, m: BreadthMarket, costs: CostSpec, *, lag: int) -> BookResult:
    return run_book(t.weights.reindex(columns=list(m.book.symbols)), t.rebalance, m.book,
                    costs, lag=lag, half_spread=m.half_spread, exit_on_missing_price=True)


def _sr(res: BookResult) -> float:
    r = res.net
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * math.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0


def _family(name: str, cfg: RobustBookV2Config, included: Sequence[str]) -> dict[str, dict]:
    parts = set(included) if name == "X5_COMBO" else {name}
    fam: dict[str, dict] = {}
    if "X1_XSMOM" in parts:
        for f in (0.5, 0.75, 1.5, 2.0):
            fam[f"xsmom_lookbacks_x{f}"] = {"xsmom_lookbacks": tuple(sorted(
                {max(2, round(x * f)) for x in cfg.xsmom.lookbacks}))}
    if parts & {"X3_TREND_LS", "X4_TREND_LF"}:
        for f in (0.5, 0.75, 1.5, 2.0):
            fam[f"trend_lookbacks_x{f}"] = {"trend_lookbacks": tuple(sorted(
                {max(2, round(x * f)) for x in cfg.trend.lookbacks}))}
    if "X2_XSCARRY" in parts:
        for win in (3, 14, 30):
            fam[f"carry_window_{win}"] = {"carry_window": win}
    for span in (20, 40, 90, 120):
        fam[f"vol_span_{span}"] = {"vol_span": span}
    for k in (2.0, 4.0):
        fam[f"max_scale_{k}"] = {"max_scale": k}
    return fam


def _mc(r: pd.Series, horizon: int) -> dict[str, float]:
    x = r.to_numpy()
    block = calibrate_block_length(x)
    idx = circular_block_indices(len(x), block, MC_PATHS,
                                 np.random.default_rng(BOOTSTRAP_SEED))[:, :horizon]
    eq = np.cumprod(1.0 + x[idx], axis=1)
    mdd = np.max(1.0 - eq / np.maximum(np.maximum.accumulate(eq, axis=1), 1.0), axis=1)
    return {"horizon_bars": horizon, "p_mdd_above_25pct": float((mdd > 0.25).mean()),
            "median_mdd": float(np.median(mdd)), "p95_mdd": float(np.quantile(mdd, 0.95))}


def _noisy(t: SleeveTargets, cfg: RobustBookV2Config, seed: int) -> SleeveTargets:
    rng = np.random.default_rng(seed)
    w = t.weights
    noise = pd.DataFrame(rng.normal(0.0, 1.0, size=w.shape), index=w.index,
                         columns=w.columns) * (0.25 * w.std(skipna=True).fillna(0.0))
    noisy = (w + noise).where(w.abs() > 0.0, 0.0).where(w.notna())
    capped = apply_caps(noisy.fillna(0.0), per_asset_cap=cfg.sizing.per_asset_cap,
                        gross_cap=cfg.sizing.gross_cap).where(w.notna())
    return SleeveTargets(name=t.name, weights=capped, rebalance=t.rebalance)


def _missing(t: SleeveTargets, seed: int) -> SleeveTargets:
    rng = np.random.default_rng(seed)
    reb = t.rebalance & ~pd.Series(rng.random(len(t.rebalance)) < 0.05, index=t.rebalance.index)
    return SleeveTargets(name=t.name, weights=t.weights.where(reb, np.nan, axis=0), rebalance=reb)


def _every(t: SleeveTargets, k: int) -> SleeveTargets:
    reb = t.rebalance.to_numpy()
    pos = np.cumsum(reb) - 1
    mask = pd.Series(reb & (pos % k == 0), index=t.rebalance.index)
    return SleeveTargets(name=t.name, weights=t.weights.where(mask, np.nan, axis=0), rebalance=mask)


def battery(name: str, m: BreadthMarket, cfg: RobustBookV2Config, costs: CostSpec, *,
            included: Sequence[str], base: BookResult, w_dev: tuple[pd.Timestamp, pd.Timestamp],
            alt_universes: dict[str, BreadthMarket]) -> tuple[dict[str, Any], dict[str, pd.Series]]:
    a, b = w_dev
    lag = cfg.execution.lag_bars
    base_sr = _sr(base.window(a, b))
    t0 = build(name, m, cfg, included=included)

    def sr(t: SleeveTargets, c: CostSpec = costs, lg: int = lag, mk: BreadthMarket = m) -> float:
        return _sr(run(t, mk, c, lag=lg).window(a, b))

    out: dict[str, Any] = {"base_sharpe": base_sr}
    family: dict[str, pd.Series] = {"base": base.window(a, b).net}
    pert = {}
    for label, kw in _family(name, cfg, included).items():
        res = run(build(name, m, cfg, included=included, **kw), m, costs, lag=lag).window(a, b)
        family[label] = res.net
        pert[label] = _sr(res)
    vals = np.array(list(pert.values()))
    out["perturbation"] = pert
    out["plateau_fraction"] = float((vals > 0.5 * base_sr).mean()) if base_sr > 0 else 0.0
    out["min_perturbation_sharpe"] = float(vals.min())
    out["costs"] = {"x0": sr(t0, costs.scaled(multiplier=0.0)),
                    "x2": sr(t0, costs.scaled(multiplier=2.0)),
                    "x3": sr(t0, costs.scaled(multiplier=3.0)),
                    "plus_10bp_slippage": sr(t0, costs.scaled(extra_slippage=10e-4)),
                    "impact_y_stress": sr(t0, costs.scaled(impact_eta=cfg.costs.impact_y_stress))}
    out["capacity"] = {f"aum_{int(x):d}": sr(t0, costs.scaled(aum_usd=x)) for x in (1e7, 5e7)}
    out["delay"] = {f"lag_{k}": sr(t0, lg=k) for k in (2, 3)}
    noise = [sr(_noisy(t0, cfg, s)) for s in range(1, N_SEEDS + 1)]
    out["signal_noise"] = {"min": float(np.min(noise)), "median": float(np.median(noise)),
                           "max": float(np.max(noise)), "n_seeds": N_SEEDS}
    miss = [sr(_missing(t0, s)) for s in range(1, N_SEEDS + 1)]
    out["missing_data"] = {"min": float(np.min(miss)), "median": float(np.median(miss)),
                           "max": float(np.max(miss)), "n_seeds": N_SEEDS}
    if name in ("X1_XSMOM", "X2_XSCARRY"):
        out["rebalance"] = {f"every_{k}": sr(build(name, m, cfg, included=included,
                                                  rebalance_bars=k)) for k in (3, 14)}
    else:
        out["rebalance"] = {f"every_{k}": sr(_every(t0, k)) for k in (3, 7)}
    uni = {}
    for label, mk in alt_universes.items():
        uni[label] = sr(build(name, mk, cfg, included=included), mk=mk)
    out["universe"] = uni
    out["universe_positive_fraction"] = float(np.mean([v > 0 for v in uni.values()])) if uni else 0.0
    out["start_dates"] = {f"start_plus_{k}m": _sr(base.window(a + pd.DateOffset(months=k), b))
                          for k in (6, 12, 18)}
    out["end_dates"] = {f"end_minus_{k}m": _sr(base.window(a, b - pd.DateOffset(months=k)))
                        for k in (6, 12)}
    r = base.window(a, b).net
    out["monte_carlo_drawdown"] = {"one_year": _mc(r, 365), "full_window": _mc(r, len(r))}
    return out, family


def regimes_v2(res: BookResult, m: BreadthMarket) -> dict[str, dict[str, float]]:
    """Causale regimes: BTC boven/onder MA200, BTC-30d-vol, gemiddelde correlatie met BTC."""
    b = m.book
    btc = b.close["BTCUSDT"]
    bull = (btc > btc.rolling(200, min_periods=200).mean()).shift(1)
    v30 = b.ret["BTCUSDT"].rolling(30, min_periods=30).std()
    hv = (v30 > v30.expanding(min_periods=200).median()).shift(1)
    c = b.ret.rolling(60, min_periods=60).corr(b.ret["BTCUSDT"]).where(m.universe).mean(axis=1)
    hc = (c > c.expanding(min_periods=200).median()).shift(1)
    r = res.net
    out = {}
    for k, mask in {"bull": bull == True, "bear": bull == False, "high_vol": hv == True,  # noqa: E712
                    "low_vol": hv == False, "high_corr": hc == True, "low_corr": hc == False}.items():  # noqa: E712
        mk = mask.reindex(r.index).fillna(False).astype(bool)
        chunk = r[mk]
        sd = float(chunk.std(ddof=1)) if len(chunk) > 2 else 0.0
        out[k] = {"share": float(mk.mean()),
                  "sharpe": float(chunk.mean() / sd * math.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0}
    return out


def _gates(metrics: dict[str, float], prereg) -> dict[str, dict[str, Any]]:
    out = {}
    for c in prereg.stop_criteria:
        if c.metric in metrics and c.name != "promotion_requires_all_clear":
            v = float(metrics[c.metric])
            out[c.name] = {"metric": c.metric, "value": v, "operator": c.operator,
                           "threshold": c.threshold,
                           "binds": bool(c.binds(v)) or not math.isfinite(v)}
    return out


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float,
                               ensure_ascii=False), encoding="utf-8")


def _alt_universes(root: Path, cfg: RobustBookV2Config, end: pd.Timestamp) -> dict[str, BreadthMarket]:
    out = {}
    for n in (30, 100):
        alt = cfg.model_copy(update={"universe": cfg.universe.model_copy(update={"top_n": n})})
        out[f"top_{n}"] = load_breadth_market(root / PANEL_DIR, alt).truncate(end)
    return out


def run_programme(root: Path = ROOT, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = robust_book_v2_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    require_preregistration(prereg.preregistration_id, directory=root / PREREG_DIR)
    w = cfg.windows
    cut = _ts(w.holdout_start)
    m = load_breadth_market(root / PANEL_DIR, cfg).truncate(cut)
    costs = cost_spec(root, cfg)
    lag = cfg.execution.lag_bars
    w_dev = (_ts(w.train_start), _ts(w.w_dev_end))
    train = (_ts(w.train_start), _ts(w.train_end))
    val = (_ts(w.validate_start), _ts(w.w_dev_end))
    sha = current_git_sha()

    results: dict[str, BookResult] = {}
    for name in REFERENCES + SLEEVES:
        log(f"run {name}")
        results[name] = run(build(name, m, cfg), m, costs, lag=lag)
    train_sr = {s: _sr(results[s].window(*train)) for s in SLEEVES}
    included = tuple(s for s in cfg.combo.candidates
                     if train_sr[s] > cfg.combo.inclusion_min_train_sharpe)
    log(f"X5 neemt op: {included}")
    if included:
        results["X5_COMBO"] = run(build("X5_COMBO", m, cfg, included=included), m, costs, lag=lag)

    ledger = HypothesisLedger(root / LEDGER_PATH)
    m_prog = int(ledger.total_n_hypotheses())
    m_wide = m_prog + int(cfg.known_prior_trials)
    per_bar = [summarize(results[n].window(*w_dev), bootstrap=False)["sharpe_per_bar"]
               for n in TRIALS if n in results]
    alt = _alt_universes(root, cfg, cut)
    records: dict[str, dict[str, Any]] = {}
    for name in TRIALS:
        rec: dict[str, Any] = {
            "trial": name, "hypothesis": HYPOTHESES[name],
            "role": "reference" if name in REFERENCES else "candidate",
            "preregistration_id": prereg.preregistration_id, "git_sha": sha,
            "data_hashes": dict(hashes), "parameters": _params(cfg),
            EVIDENCE_KEY: NOT_ADMISSIBLE,
            "windows": {"train": [w.train_start, w.train_end],
                        "validate": [w.validate_start, w.w_dev_end],
                        "holdout": [w.holdout_start, "unread"]},
            "execution": {"lag_bars": lag, "costs": costs.as_record(),
                          "half_spread": "Abdi-Ranaldo CHL, causaal, 60 d"},
        }
        if name == "X5_COMBO":
            rec["included"] = list(included)
            rec["train_sharpe_of_sleeves"] = train_sr
        if name not in results:
            rec["status"] = "not_constructible"
            records[name] = rec
            continue
        res = results[name]
        dev = summarize(res.window(*w_dev))
        rec.update({"w_dev": dev, "train": summarize(res.window(*train)),
                    "validate": summarize(res.window(*val)),
                    "yearly": yearly(res.window(*w_dev)),
                    "regimes": regimes_v2(res.window(*w_dev), m),
                    "n_forced_exits": res.audit["n_forced_exits"],
                    "dsr": {f"m_{k}": dsr_record(res.window(*w_dev).net, n_trials=k,
                                                 trial_sharpes_per_bar=per_bar)
                            for k in (m_prog, m_wide)}})
        if name in CANDIDATES:
            log(f"batterij {name}")
            bat, family = battery(name, m, cfg, costs, included=included, base=res,
                                  w_dev=w_dev, alt_universes=alt)
            rec["battery"] = bat
            rec["pbo"] = pbo_record(family)
            pos = dev["sharpe"] > 0
            metrics = {
                "net_sharpe_w_dev": dev["sharpe"], "net_cagr_w_dev": dev["cagr"],
                "net_sharpe_train": rec["train"]["sharpe"],
                "net_sharpe_validate": rec["validate"]["sharpe"],
                "max_drawdown_w_dev": dev["max_drawdown"],
                "sharpe_ratio_2x_cost_over_base": bat["costs"]["x2"] / dev["sharpe"] if pos else float("nan"),
                "sharpe_ratio_lag2_over_base": bat["delay"]["lag_2"] / dev["sharpe"] if pos else float("nan"),
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
                "sharpe_validate": rec["validate"]["sharpe"], "sharpe_2x_cost": bat["costs"]["x2"],
                "sharpe_lag2": bat["delay"]["lag_2"], "plateau_fraction": bat["plateau_fraction"],
                "p_sharpe_gt_0": dev["p_sharpe_gt_0"], "max_drawdown": dev["max_drawdown"],
                "pbo": rec["pbo"]["pbo"],
                "leave_one_out_positive_fraction": bat["universe_positive_fraction"]})
        records[name] = rec
        _dump(root / ARTEFACT_DIR / f"{name}.json", rec)
        res.window(*w_dev).frame.to_csv(root / ARTEFACT_DIR / f"{name}_daily_w_dev.csv",
                                        float_format="%.10g")
    scored = {n: records[n]["robustness_score"]["total"] for n in CANDIDATES
              if "robustness_score" in records[n]}
    selected = max(scored, key=lambda k: scored[k])
    summary = {
        "programme": "robust_book_v2", "preregistration_id": prereg.preregistration_id,
        "git_sha": sha, "m_programme": m_prog, "m_including_known_prior": m_wide,
        "x5_included": list(included), "robustness_scores": scored, "selected": selected,
        "passes_all_w_dev_gates": {n: records[n].get("n_binding_w_dev") == 0
                                   for n in CANDIDATES if n in records},
        "table": {n: {k: records[n]["w_dev"][k] for k in (
            "sharpe", "sharpe_se", "cagr", "ann_vol", "max_drawdown", "sortino", "calmar",
            "ann_turnover", "avg_gross_leverage", "sharpe_ci_low", "sharpe_ci_high")}
            for n in TRIALS if "w_dev" in records[n]},
        EVIDENCE_KEY: NOT_ADMISSIBLE,
    }
    _dump(root / ARTEFACT_DIR / "programme_w_dev.json", summary)
    return summary


def read_holdout(root: Path = ROOT, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = robust_book_v2_config(root / CONFIG)
    prereg, _ = _prereg(root, cfg)
    require_preregistration(prereg.preregistration_id, directory=root / PREREG_DIR)
    summary = json.loads((root / ARTEFACT_DIR / "programme_w_dev.json").read_text(encoding="utf-8"))
    require(summary["preregistration_id"] == prereg.preregistration_id,
            "De W_DEV-run hoort bij een andere preregistratie.", DataContractError)
    name = summary["selected"]
    included = tuple(summary["x5_included"])
    full = load_breadth_market(root / PANEL_DIR, cfg)
    hid = f"robust_book_v2/{prereg.preregistration_id}/{name}"
    sl = gate_slice(full.book.close, lock_path=root / LOCK_PATH, hypothesis_id=hid)
    a = pd.Timestamp(sl.index[0])
    b = min(pd.Timestamp(sl.index[-1]), _ts(cfg.windows.holdout_end))
    costs = cost_spec(root, cfg)
    res = run(build(name, full, cfg, included=included), full, costs, lag=cfg.execution.lag_bars)
    hold = summarize(res.window(a, b))
    dev_rec = json.loads((root / ARTEFACT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    dev_again = _sr(res.window(_ts(cfg.windows.train_start), _ts(cfg.windows.w_dev_end)))
    require(abs(dev_again - dev_rec["w_dev"]["sharpe"]) < 1e-9,
            "De volledige run reproduceert de W_DEV-Sharpe niet: de holdout lekte.",
            DataContractError, full=dev_again, truncated=dev_rec["w_dev"]["sharpe"])
    z = gate_z(hold["sharpe"], hold["sharpe_se"], dev_rec["w_dev"]["sharpe"])
    gates = _gates({"gate_sharpe_z_vs_dev": z, "max_drawdown_gate": hold["max_drawdown"]}, prereg)
    refs = {r: summarize(run(build(r, full, cfg), full, costs,
                             lag=cfg.execution.lag_bars).window(a, b)) for r in REFERENCES}
    n_binding = int(sum(g["binds"] for g in gates.values())) + int(dev_rec["n_binding_w_dev"])
    verdict = "promote_to_paper_trading" if n_binding == 0 else (
        "falsified_on_holdout" if any(g["binds"] for g in gates.values())
        else "archived_w_dev_gates")
    out = {"hypothesis_id": hid, "candidate": name, "included": list(included),
           "holdout_window": [str(a.date()), str(b.date())], "holdout": hold,
           "yearly": yearly(res.window(a, b)), "references_on_holdout": refs,
           "gate_metrics": {"gate_sharpe_z_vs_dev": z, "max_drawdown_gate": hold["max_drawdown"]},
           "gates": gates, "n_binding_total": n_binding, "verdict": verdict,
           "w_dev_sharpe": dev_rec["w_dev"]["sharpe"], EVIDENCE_KEY: NOT_ADMISSIBLE}
    _dump(root / ARTEFACT_DIR / "holdout_read.json", out)
    res.window(a, b).frame.to_csv(root / ARTEFACT_DIR / f"{name}_daily_holdout.csv",
                                  float_format="%.10g")
    log(f"holdout {name}: Sharpe {hold['sharpe']:.3f} (SE {hold['sharpe_se']:.3f}), "
        f"CAGR {hold['cagr']:.3f}, z={z:.2f}, {verdict}")
    return out


def book_verdict(root: Path = ROOT) -> None:
    cfg = robust_book_v2_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    hold = json.loads((root / ARTEFACT_DIR / "holdout_read.json").read_text(encoding="utf-8"))
    summary = json.loads((root / ARTEFACT_DIR / "programme_w_dev.json").read_text(encoding="utf-8"))
    result = {"promote_to_paper_trading": "accepted", "falsified_on_holdout": "falsified"}.get(
        hold["verdict"], "archived")
    HypothesisLedger(root / LEDGER_PATH).append(LedgerEntry.from_config(
        wave=2, unit="robust_book_v2_verdict", market="crypto", config=_params(cfg),
        git_sha=current_git_sha(), data_hash=hash_config(dict(hashes)),
        preregistration_id=prereg.preregistration_id, n_trials=0, result=result,
        amends=hash_config(_params(cfg)),
        metrics={"selected": hold["candidate"], "w_dev_sharpe": hold["w_dev_sharpe"],
                 "holdout_sharpe": hold["holdout"]["sharpe"], "verdict": hold["verdict"],
                 "robustness_scores": summary["robustness_scores"]},
        notes="Oordeel robuust boek v2 na W_DEV en één holdout-lezing."))


def main(argv: Sequence[str]) -> None:
    cmds = {"freeze": lambda: print(freeze()),
            "run": lambda: print(json.dumps(run_programme(), indent=2, default=float)),
            "holdout": lambda: print(json.dumps(read_holdout(), indent=2, default=float)),
            "verdict": book_verdict}
    require(len(argv) == 1 and argv[0] in cmds,
            "Gebruik: python -I -m tradebot.systematic.programme_v2 {freeze|run|holdout|verdict}",
            DataContractError)
    cmds[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
