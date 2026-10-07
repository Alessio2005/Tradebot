"""Robuust boek v3: v2 met een gecorrigeerd spreadmodel en partiële aanpassing.

    python -I -m tradebot.systematic.programme_v3 freeze
    python -I -m tradebot.systematic.programme_v3 run
    python -I -m tradebot.systematic.programme_v3 holdout
    python -I -m tradebot.systematic.programme_v3 verdict

POST-HOC, EN ZO GEBOEKT. v3 bestaat omdat v2 twee dingen liet zien:
1. de vooraf gekozen CHL-spreadschatter meet op dagbars volatiliteit, geen spread
   (BTCUSDT 17,7 bp, ETHUSDT 25 bp);
2. carry en de combinatie hebben bruto edge (1,13 en 1,49 vóór kosten) maar een omzet
   van 68-94x per jaar.
v3 verandert precies die twee dingen en niets anders. De ontwerpen van de sleeves en
het universum zijn die van v2. De drie trials worden vóór de run in de ledger geboekt
(M = 17); de holdout is voor de trendfamilie al één keer gelezen (v2/X4) en is voor v3
dus zwakker bewijs. Ontwerp: `docs/superpowers/specs/2026-10-07-robust-book-v3-costs-design.md`.
"""
from __future__ import annotations

import json
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
from ..schemas.robust_book_v2 import RobustBookV3Config, robust_book_v3_config
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from ..validation.holdout import gate_slice
from . import programme_v2 as v2
from .book import BookResult, CostSpec
from .breadth import BreadthMarket, load_breadth_market
from .evaluate import dsr_record, gate_z, pbo_record, robustness_score, summarize, yearly
from .sleeves import SleeveTargets

__all__ = ["CANDIDATES", "freeze", "read_holdout", "run_programme"]

ROOT = v2.ROOT
CONFIG = Path("conf/model/robust_book_v3.yaml")
SPEC_PATH = Path("conf/research/preregistration_robust_book_v3.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v3")
PROGRAMME_UNIT = "robust_book_v3_programme"

#: v3-kandidaat -> (v2-sleeve, opgenomen sleeves voor de combinatie).
SLEEVE_OF = {
    "Y2_CARRY": ("X2_XSCARRY", ()),
    "Y4_TREND_LF": ("X4_TREND_LF", ()),
    "Y5_COMBO": ("X5_COMBO", ("X2_XSCARRY", "X4_TREND_LF")),
}
CANDIDATES = tuple(SLEEVE_OF)
HYPOTHESES = {
    "Y2_CARRY": "v2-X2 funding-carry, met 3 bp spread en partiële aanpassing (kappa 0,2).",
    "Y4_TREND_LF": "v2-X4 trend long-flat, met 3 bp spread en partiële aanpassing.",
    "Y5_COMBO": "Gelijk-risicocombinatie van Y2 en Y4 (vast, geen inclusieregel).",
}


def _params(cfg: RobustBookV3Config) -> dict[str, Any]:
    return json.loads(cfg.model_dump_json())


def _prereg(root: Path, cfg: RobustBookV3Config):
    hashes = v2._data_hashes(root)
    return load_preregistration_spec(root / SPEC_PATH, data_hashes=hashes,
                                     parameters=_params(cfg)), hashes


def cost_spec(root: Path, cfg: RobustBookV3Config, *, half_spread_bps: float | None = None) -> CostSpec:
    return CostSpec(taker_fee=cfg.costs.taker_fee_bps * 1e-4,
                    half_spread=float(half_spread_bps or cfg.v3.half_spread_bps) * 1e-4,
                    impact=load_impact_params(root / "conf/execution/impact.yaml"),
                    aum_usd=cfg.execution.aum_usd, impact_eta=cfg.costs.impact_y)


def build(name: str, m: BreadthMarket, cfg: RobustBookV3Config, *, included: Sequence[str] = (),
          **kw: Any) -> SleeveTargets:
    """Het v2-doel van de sleeve, doorgetrokken naar elke bar: partiële aanpassing
    beweegt dagelijks een fractie kappa naar het laatste besluit."""
    sleeve, inc = SLEEVE_OF[name]
    t = v2.build(sleeve, m, cfg, included=inc or tuple(included), **kw)
    w = t.weights.ffill().where(m.book.close.notna(), 0.0)
    started = t.rebalance.cummax()
    return SleeveTargets(name=name, weights=w.where(started, np.nan, axis=0), rebalance=started)


def make_runner(cfg: RobustBookV3Config) -> Callable[..., BookResult]:
    def runner(t: SleeveTargets, m: BreadthMarket, costs: CostSpec, *, lag: int,
               trade_rate: float | None = None) -> BookResult:
        return v2.run(t, m, costs, lag=lag, spread_panel=False,
                      trade_rate=float(trade_rate or cfg.v3.trade_rate))
    return runner


def freeze(root: Path = ROOT, *, git_sha: str | None = None) -> Path:
    cfg = robust_book_v3_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    require(prereg.planned_trials == cfg.planned_trials, "planned_trials wijkt af.",
            DataContractError)
    sha = git_sha or current_git_sha()
    ledger = HypothesisLedger(root / v2.LEDGER_PATH)
    require(not [e for e in ledger.entries()
                 if e.get("preregistration_id") == prereg.preregistration_id],
            "Dit programma is al geboekt.", DataContractError)
    ledger.append(LedgerEntry.from_config(
        wave=3, unit=PROGRAMME_UNIT, market="crypto", config=_params(cfg), git_sha=sha,
        data_hash=hash_config(dict(hashes)), preregistration_id=prereg.preregistration_id,
        n_trials=cfg.planned_trials, result="interim",
        notes="Drie geplande trials, post-hoc na v2 en zo geboekt: Y2 carry, Y4 trend-LF, "
              "Y5 combinatie, met 3 bp spread en partiële aanpassing."))
    return freeze_preregistration(prereg, git_sha=sha,
                                  ledger_total_at_freeze=ledger.total_n_hypotheses(),
                                  directory=root / v2.PREREG_DIR)


def run_programme(root: Path = ROOT, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = robust_book_v3_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    require_preregistration(prereg.preregistration_id, directory=root / v2.PREREG_DIR)
    w = cfg.windows
    cut = v2._ts(w.holdout_start)
    m = load_breadth_market(root / v2.PANEL_DIR, cfg).truncate(cut)
    costs = cost_spec(root, cfg)
    runner = make_runner(cfg)
    lag = cfg.execution.lag_bars
    w_dev = (v2._ts(w.train_start), v2._ts(w.w_dev_end))
    train = (v2._ts(w.train_start), v2._ts(w.train_end))
    val = (v2._ts(w.validate_start), v2._ts(w.w_dev_end))
    sha = current_git_sha()
    results = {}
    for name in CANDIDATES:
        log(f"run {name}")
        results[name] = runner(build(name, m, cfg), m, costs, lag=lag)
    m_prog = int(HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses())
    m_wide = m_prog + int(cfg.known_prior_trials)
    per_bar = [summarize(results[n].window(*w_dev), bootstrap=False)["sharpe_per_bar"]
               for n in CANDIDATES]
    alt = v2._alt_universes(root, cfg, cut)
    stress = {"half_spread_stress": cost_spec(root, cfg,
                                              half_spread_bps=cfg.v3.half_spread_stress_bps)}
    kappa = {f"trade_rate_{k}": {"trade_rate": k} for k in cfg.v3.trade_rate_family}
    records = {}
    for name in CANDIDATES:
        res = results[name]
        sleeve, inc = SLEEVE_OF[name]
        dev = summarize(res.window(*w_dev))
        rec: dict[str, Any] = {
            "trial": name, "hypothesis": HYPOTHESES[name], "role": "candidate",
            "preregistration_id": prereg.preregistration_id, "git_sha": sha,
            "data_hashes": dict(hashes), "parameters": _params(cfg), EVIDENCE_KEY: NOT_ADMISSIBLE,
            "windows": {"train": [w.train_start, w.train_end],
                        "validate": [w.validate_start, w.w_dev_end],
                        "holdout": [w.holdout_start, "unread"]},
            "execution": {"lag_bars": lag, "trade_rate": cfg.v3.trade_rate,
                          "costs": costs.as_record()},
            "w_dev": dev, "train": summarize(res.window(*train)),
            "validate": summarize(res.window(*val)), "yearly": yearly(res.window(*w_dev)),
            "regimes": v2.regimes_v2(res.window(*w_dev), m),
            "n_forced_exits": res.audit["n_forced_exits"],
            "dsr": {f"m_{k}": dsr_record(res.window(*w_dev).net, n_trials=k,
                                         trial_sharpes_per_bar=per_bar) for k in (m_prog, m_wide)},
        }
        log(f"batterij {name}")
        bat, family = v2.battery(name, m, cfg, costs, included=inc, base=res, w_dev=w_dev,
                                 alt_universes=alt, builder=build, runner=runner,
                                 family_name=sleeve, run_family=kappa, extra_costs=stress)
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
            "dsr_w_dev": rec["dsr"][f"m_{m_prog}"]["dsr"], "pbo": rec["pbo"]["pbo"],
        }
        rec["gate_metrics"] = metrics
        rec["gates"] = v2._gates(metrics, prereg)
        rec["n_binding_w_dev"] = int(sum(g["binds"] for g in rec["gates"].values()))
        rec["robustness_score"] = robustness_score({
            "sharpe": dev["sharpe"], "sharpe_train": rec["train"]["sharpe"],
            "sharpe_validate": rec["validate"]["sharpe"], "sharpe_2x_cost": bat["costs"]["x2"],
            "sharpe_lag2": bat["delay"]["lag_2"], "plateau_fraction": bat["plateau_fraction"],
            "p_sharpe_gt_0": dev["p_sharpe_gt_0"], "max_drawdown": dev["max_drawdown"],
            "pbo": rec["pbo"]["pbo"],
            "leave_one_out_positive_fraction": bat["universe_positive_fraction"]})
        records[name] = rec
        v2._dump(root / ARTEFACT_DIR / f"{name}.json", rec)
        res.window(*w_dev).frame.to_csv(root / ARTEFACT_DIR / f"{name}_daily_w_dev.csv",
                                        float_format="%.10g")
    scored = {n: records[n]["robustness_score"]["total"] for n in CANDIDATES}
    selected = max(scored, key=lambda k: scored[k])
    summary = {
        "programme": "robust_book_v3", "preregistration_id": prereg.preregistration_id,
        "git_sha": sha, "m_programme": m_prog, "m_including_known_prior": m_wide,
        "robustness_scores": scored, "selected": selected,
        "passes_all_w_dev_gates": {n: records[n]["n_binding_w_dev"] == 0 for n in CANDIDATES},
        "table": {n: {k: records[n]["w_dev"][k] for k in (
            "sharpe", "sharpe_se", "cagr", "ann_vol", "max_drawdown", "sortino", "calmar",
            "ann_turnover", "avg_gross_leverage", "sharpe_ci_low", "sharpe_ci_high")}
            for n in CANDIDATES},
        EVIDENCE_KEY: NOT_ADMISSIBLE,
    }
    v2._dump(root / ARTEFACT_DIR / "programme_w_dev.json", summary)
    return summary


def read_holdout(root: Path = ROOT, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = robust_book_v3_config(root / CONFIG)
    prereg, _ = _prereg(root, cfg)
    require_preregistration(prereg.preregistration_id, directory=root / v2.PREREG_DIR)
    summary = json.loads((root / ARTEFACT_DIR / "programme_w_dev.json").read_text(encoding="utf-8"))
    require(summary["preregistration_id"] == prereg.preregistration_id,
            "De W_DEV-run hoort bij een andere preregistratie.", DataContractError)
    name = summary["selected"]
    full = load_breadth_market(root / v2.PANEL_DIR, cfg)
    hid = f"robust_book_v3/{prereg.preregistration_id}/{name}"
    sl = gate_slice(full.book.close, lock_path=root / v2.LOCK_PATH, hypothesis_id=hid)
    a = pd.Timestamp(sl.index[0])
    b = min(pd.Timestamp(sl.index[-1]), v2._ts(cfg.windows.holdout_end))
    costs = cost_spec(root, cfg)
    runner = make_runner(cfg)
    res = runner(build(name, full, cfg), full, costs, lag=cfg.execution.lag_bars)
    hold = summarize(res.window(a, b))
    dev_rec = json.loads((root / ARTEFACT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    dev_again = v2._sr(res.window(v2._ts(cfg.windows.train_start), v2._ts(cfg.windows.w_dev_end)))
    require(abs(dev_again - dev_rec["w_dev"]["sharpe"]) < 1e-9,
            "De volledige run reproduceert de W_DEV-Sharpe niet: de holdout lekte.",
            DataContractError, full=dev_again, truncated=dev_rec["w_dev"]["sharpe"])
    z = gate_z(hold["sharpe"], hold["sharpe_se"], dev_rec["w_dev"]["sharpe"])
    gates = v2._gates({"gate_sharpe_z_vs_dev": z, "max_drawdown_gate": hold["max_drawdown"]}, prereg)
    # Alleen de geselecteerde kandidaat leest de holdout. De andere kandidaten hier
    # doorrekenen zou een ongeregistreerde tweede (en derde) lezing zijn.
    n_binding = int(sum(g["binds"] for g in gates.values())) + int(dev_rec["n_binding_w_dev"])
    verdict = "promote_to_paper_trading" if n_binding == 0 else (
        "falsified_on_holdout" if any(g["binds"] for g in gates.values())
        else "archived_w_dev_gates")
    out = {"hypothesis_id": hid, "candidate": name, "holdout_window": [str(a.date()), str(b.date())],
           "holdout": hold, "yearly": yearly(res.window(a, b)),
           "gate_metrics": {"gate_sharpe_z_vs_dev": z, "max_drawdown_gate": hold["max_drawdown"]},
           "gates": gates, "n_binding_total": n_binding, "verdict": verdict,
           "w_dev_sharpe": dev_rec["w_dev"]["sharpe"],
           "caveat": "De holdout is voor de trendfamilie al gelezen in v2 (X4_TREND_LF).",
           EVIDENCE_KEY: NOT_ADMISSIBLE}
    v2._dump(root / ARTEFACT_DIR / "holdout_read.json", out)
    res.window(a, b).frame.to_csv(root / ARTEFACT_DIR / f"{name}_daily_holdout.csv",
                                  float_format="%.10g")
    log(f"holdout {name}: Sharpe {hold['sharpe']:.3f} (SE {hold['sharpe_se']:.3f}), "
        f"CAGR {hold['cagr']:.3f}, z={z:.2f}, {verdict}")
    return out


def book_verdict(root: Path = ROOT) -> None:
    cfg = robust_book_v3_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    hold = json.loads((root / ARTEFACT_DIR / "holdout_read.json").read_text(encoding="utf-8"))
    summary = json.loads((root / ARTEFACT_DIR / "programme_w_dev.json").read_text(encoding="utf-8"))
    result = {"promote_to_paper_trading": "accepted", "falsified_on_holdout": "falsified"}.get(
        hold["verdict"], "archived")
    HypothesisLedger(root / v2.LEDGER_PATH).append(LedgerEntry.from_config(
        wave=3, unit="robust_book_v3_verdict", market="crypto", config=_params(cfg),
        git_sha=current_git_sha(), data_hash=hash_config(dict(hashes)),
        preregistration_id=prereg.preregistration_id, n_trials=0, result=result,
        amends=hash_config(_params(cfg)),
        metrics={"selected": hold["candidate"], "w_dev_sharpe": hold["w_dev_sharpe"],
                 "holdout_sharpe": hold["holdout"]["sharpe"], "verdict": hold["verdict"],
                 "robustness_scores": summary["robustness_scores"]},
        notes="Oordeel robuust boek v3 na W_DEV en één holdout-lezing."))


def main(argv: Sequence[str]) -> None:
    cmds = {"freeze": lambda: print(freeze()),
            "run": lambda: print(json.dumps(run_programme(), indent=2, default=float)),
            "holdout": lambda: print(json.dumps(read_holdout(), indent=2, default=float)),
            "verdict": book_verdict}
    require(len(argv) == 1 and argv[0] in cmds,
            "Gebruik: python -I -m tradebot.systematic.programme_v3 {freeze|run|holdout|verdict}",
            DataContractError)
    cmds[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
