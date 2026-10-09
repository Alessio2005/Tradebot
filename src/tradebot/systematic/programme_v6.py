"""Robuust boek v6: de basiscarry van v5-H1 met hefboom, in één unified-margin-account.

    python -I -m tradebot.data.binance_vision mark-sync        # mark price (eenmalig)
    python -I -m tradebot.data.binance_vision mark-panels
    python -I -m tradebot.systematic.programme_v6 freeze      # ledger + prereg + vooruit-slot
    python -I -m tradebot.systematic.programme_v6 run         # W_DEV, batterij, selectie
    python -I -m tradebot.systematic.programme_v6 oos         # backcast 2020 + holdout, één keer
    python -I -m tradebot.systematic.programme_v6 forward     # vanaf 2027-04: het vooruit-sample

WAAROM v6 BESTAAT. v5 vond de enige robuuste bron van rendement in deze repo, de
spot-perp-basiscarry (Sharpe 5,7-6,8 op W_DEV, 11,5 op de ongemeten backcast), maar haalde
de CAGR niet: in het tweewalletmodel stond gemiddeld 0,26-0,33 van de equity per been in de
markt. De eigenaar staat sinds 2026-10-09 hoog risico toe. v6 zet de carry in een
unified-margin-account met hefboom tot de mandaatgrens (bruto 4,0) en modelleert het
staartrisico expliciet: financiering, haircut, liquidatie op de mark price, governor.
Ontwerp en alle vooraf gekozen parameters:
`docs/superpowers/specs/2026-10-09-robust-book-v6-levered-carry-design.md`.

HET BEWIJS, EERLIJK GEWOGEN. Drie trials (cumulatief M = 24), een dosis-responsreeks in
hefboom. Selectie: de laagste hefboom die alle W_DEV-poorten haalt. Daarna drie lezingen:
1. de BACKCAST 2020 (deels besmet: v5 las H3 daar) en
2. de HOLDOUT 2025-26 (besmet) kunnen alleen verwerpen;
3. het VOORUIT-sample vanaf 2026-10-01 bestond bij het bevriezen niet: de enige schone
   toets, eenmaal gelezen na minstens zes volle maanden.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from ..features.registry import current_git_sha
from ..registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from ..registry.preregistration import (
    freeze_preregistration,
    load_preregistration_spec,
    require_preregistration,
)
from ..schemas.robust_book_v2 import RobustBookV6Config, robust_book_v6_config
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from ..validation.holdout import backcast_gate_slice, freeze_holdout, gate_slice
from . import programme_v2 as v2
from . import programme_v5 as v5
from .book import BookResult
from .evaluate import dsr_record, gate_z, pbo_record, robustness_score, summarize, yearly
from .harvest import BasisCosts, BasisMarket
from .leverage import (
    MarginMarket,
    MarginSpec,
    financing_rate,
    levered_targets,
    load_margin_market,
    run_levered,
)
from .market import BARS_PER_YEAR

__all__ = ["CANDIDATES", "V6", "Programme", "freeze", "read_forward", "read_oos",
           "run_programme", "select", "simulate"]

ROOT = v2.ROOT
CONFIG = Path("conf/model/robust_book_v6.yaml")
SPEC_PATH = Path("conf/research/preregistration_robust_book_v6.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v6")
MARK_DIR = Path("data/binance_vision/mark_panels")
MARK_MANIFEST = Path("artefacts/data/binance_mark_manifest.json")
FORWARD_LOCK = Path("artefacts/governance/holdout_lock_forward_2026_10.json")
#: Het vooruit-sample leest uit eigen panelen (gebouwd met een latere `last_month`), zodat
#: de bevroren panelen en hun hashes onaangeroerd blijven.
FORWARD_ROOT = Path("data/binance_vision/forward")
CANDIDATES = ("K1_CARRY_PM_1X", "K2_CARRY_PM_1P5X", "K3_CARRY_PM_2X")
HYPOTHESES = {
    "K1_CARRY_PM_1X": "De slotregel van v5-H1 in een unified-margin-account, 1,0x notional per "
                      "been (bruto 2,0): kapitaalefficiëntie zonder lening.",
    "K2_CARRY_PM_1P5X": "Idem, 1,5x notional per been (bruto 3,0), lening boven de equity "
                        "tegen max(8 %, BTC-carry).",
    "K3_CARRY_PM_2X": "Idem, 2,0x notional per been (bruto 4,0 = de mandaatcap).",
}
N_SEEDS = 10
COMPONENTS_V6 = ("financing", "liquidation")


def _params(cfg: RobustBookV6Config) -> dict[str, Any]:
    return json.loads(cfg.model_dump_json())


def _data_hashes(root: Path, *, mark_dir: Path = MARK_DIR) -> tuple[tuple[str, str], ...]:
    out = list(v5._data_hashes(root))
    mark = json.loads((root / MARK_MANIFEST).read_text(encoding="utf-8"))
    out.append(("binance_mark/objects", mark["objects_sha256"][:16]))
    out += [(f"binance_mark/panel/{p.stem}", hashlib.sha256(p.read_bytes()).hexdigest()[:16])
            for p in sorted((root / mark_dir).glob("*.parquet"))]
    return tuple(sorted(out))


def _prereg(root: Path, cfg: RobustBookV6Config, prog: Programme | None = None):
    prog = prog or V6
    hashes = _data_hashes(root)
    return load_preregistration_spec(root / prog.spec, data_hashes=hashes,
                                     parameters=_params(cfg)), hashes


def _require_candidates(cfg: RobustBookV6Config, prog: Programme | None = None) -> None:
    prog = prog or V6
    require(tuple(cfg.leverage.candidates) == prog.candidates, "De kandidaten in de config "
            "wijken af van het programma.", DataContractError,
            config=list(cfg.leverage.candidates))


# --------------------------------------------------------------------------- #
# Markt, kosten, marge, financiering
# --------------------------------------------------------------------------- #
def load_market(root: Path, cfg: RobustBookV6Config, *, end: pd.Timestamp | None = None,
                top_n: int | None = None, panel_root: Path | None = None) -> MarginMarket:
    """De margemarkt. `panel_root` (alleen het vooruit-sample) wijst naar eigen panelen."""
    if panel_root is None:
        basis = v5.load_market(root, cfg, top_n=top_n)
        spot_dir, mark_dir = root / v5.SPOT_DIR, root / MARK_DIR
    else:
        from .breadth import load_breadth_market
        from .harvest import load_basis_market
        c = cfg if top_n is None else cfg.model_copy(update={
            "universe": cfg.universe.model_copy(update={"top_n": int(top_n)})})
        spot_dir, mark_dir = panel_root / "spot_panels", panel_root / "mark_panels"
        basis = load_basis_market(load_breadth_market(panel_root / "panels", c), spot_dir, c)
    mm = load_margin_market(basis, spot_dir, mark_dir)
    return mm if end is None else mm.truncate(end)


def costs_for(root: Path, cfg: RobustBookV6Config, **kw: float | None) -> BasisCosts:
    return v5.costs_for(root, cfg, **kw)


def margin_spec(cfg: RobustBookV6Config, *, stress: bool = False) -> MarginSpec:
    lv = cfg.leverage
    return MarginSpec(
        haircut_major=lv.haircut_major,
        haircut_alt=lv.haircut_alt_stress if stress else lv.haircut_alt,
        majors=tuple(lv.majors),
        maintenance_margin=lv.maintenance_margin_stress if stress else lv.maintenance_margin,
        loan_maintenance=lv.loan_maintenance, min_uni_mmr=lv.min_uni_mmr,
        gross_cap=lv.gross_cap, liquidation_fee=lv.liquidation_fee)


def financing(m: BasisMarket, cfg: RobustBookV6Config, mode: str = "base") -> pd.Series:
    """De leenrente: `base`, `stress` of `optimistic` (alleen ter informatie)."""
    lv, span = cfg.leverage, cfg.basis.funding_span
    if mode == "base":
        return financing_rate(m, floor_apr=lv.financing_floor_apr,
                              multiplier=lv.financing_multiplier, span=span)
    if mode == "stress":
        return financing_rate(m, floor_apr=lv.financing_stress_floor_apr,
                              multiplier=lv.financing_stress_multiplier, span=span)
    require(mode == "optimistic", "Onbekende financieringsmodus.", DataContractError, mode=mode)
    return financing_rate(m, floor_apr=0.0, multiplier=1.0, span=span,
                          constant_apr=lv.financing_optimistic_apr)


# --------------------------------------------------------------------------- #
# De kandidaten, met alle verstoringen van de batterij als overrides
# --------------------------------------------------------------------------- #
def simulate(name: str, m: MarginMarket, cfg: RobustBookV6Config, costs: BasisCosts, *,
             lag: int | None = None, ov: Mapping[str, Any] | None = None) -> BookResult:
    """Eén kandidaat. Overrides: funding_span, enter_apr, exit_apr, slots, band, noise_seed,
    missing_seed, every, financing ('base'|'stress'|'optimistic'), margin_stress (bool),
    governor_off (bool)."""
    require(name in CANDIDATES, "Onbekende v6-kandidaat.", DataContractError, name=name)
    o = dict(ov or {})
    lg = int(cfg.execution.lag_bars if lag is None else lag)
    b, lv = cfg.basis, cfg.leverage
    tgt = levered_targets(
        m.basis, leverage=float(lv.candidates[name]), slots=int(o.get("slots", b.slots)),
        span=int(o.get("funding_span", b.funding_span)),
        enter_apr=float(o.get("enter_apr", b.enter_apr)),
        exit_apr=float(o.get("exit_apr", b.exit_apr)),
        min_spot_adv_usd=b.min_spot_adv_usd, max_abs_basis=b.max_abs_basis,
        carry_noise=v5._noise(m.basis, int(o["noise_seed"])) if "noise_seed" in o else None)
    if "every" in o:
        tgt = v5._every(tgt, int(o["every"]))
    skip = v5._missing(m.index, int(o["missing_seed"])) if "missing_seed" in o else None
    margin = margin_spec(cfg, stress=bool(o.get("margin_stress", False)))
    if o.get("governor_off", False):
        margin = replace(margin, min_uni_mmr=1.0, gross_cap=1e9)
    return run_levered(tgt, m, costs, lag=lg, band=float(o.get("band", b.band)),
                       hedge_tolerance=b.hedge_tolerance, margin=margin,
                       financing=financing(m.basis, cfg, str(o.get("financing", "base"))),
                       skip=skip, adv_cap=lv.adv_participation_cap)


def _family(name: str = "", cfg: RobustBookV6Config | None = None) -> dict[str, dict[str, Any]]:
    """De parameterverstoringen van v5-H1, ongewijzigd (gelijk voor elke v6-kandidaat)."""
    return {
        "funding_span_14": {"funding_span": 14}, "funding_span_60": {"funding_span": 60},
        "thresholds_low": {"enter_apr": 0.10, "exit_apr": 0.03},
        "thresholds_high": {"enter_apr": 0.20, "exit_apr": 0.08},
        "band_0.25": {"band": 0.25}, "band_0.75": {"band": 0.75},
        "slots_5": {"slots": 5}, "slots_20": {"slots": 20},
    }


def _liq_count(res: BookResult) -> int:
    """Liquidaties BINNEN het venster (de frame), nooit uit de audit van de hele run: die
    bevat de opwarmperiode, en daarmee de backcast."""
    return int((res.frame["liquidation"] != 0.0).sum())


def _headroom(res: BookResult) -> float:
    h = res.frame["stress_headroom"].dropna()
    return float(h.min()) if len(h) else float("nan")


def summary(res: BookResult, *, bootstrap: bool = True) -> dict[str, Any]:
    """`evaluate.summarize` plus de v6-posten: financiering, liquidatie, lening, marge."""
    s = summarize(res, bootstrap=bootstrap)
    f = res.frame
    for c in COMPONENTS_V6:
        s[f"ann_{c}"] = float(f[c].mean() * BARS_PER_YEAR)
    s["n_liquidations"] = _liq_count(res)
    s["min_stress_headroom"] = _headroom(res)
    s["avg_loan"] = float(f["loan"].mean())
    s["max_loan"] = float(f["loan"].max())
    held = f["n_held"] > 0
    s["min_uni_mmr"] = float(f.loc[held, "uni_mmr"].min()) if held.any() else float("nan")
    s["avg_spot_notional"] = float(f["spot_notional"].mean())
    s["n_governed"] = int(f["governed"].sum())
    return s


def battery(name: str, m: MarginMarket, cfg: RobustBookV6Config, costs: BasisCosts,
            stress: BasisCosts, *, base: BookResult, w_dev: tuple[pd.Timestamp, pd.Timestamp],
            alt: Mapping[str, MarginMarket], prog: Programme | None = None,
            ) -> tuple[dict[str, Any], dict[str, pd.Series]]:
    """De batterij van v5 plus de hefboomstress, alles op W_DEV."""
    prog = prog or V6
    simulate = prog.simulate
    a, b = w_dev

    def sr(res: BookResult) -> float:
        return v2._sr(res.window(a, b))

    def full(**kw: Any) -> BookResult:
        mk = kw.pop("mk", m)
        return simulate(name, mk, cfg, kw.pop("c", costs), **kw).window(a, b)

    def run(**kw: Any) -> float:
        return v2._sr(full(**kw))

    def brief(res: BookResult) -> dict[str, float]:
        s = summary(res, bootstrap=False)
        return {k: s[k] for k in ("sharpe", "cagr", "max_drawdown", "n_liquidations",
                                  "min_stress_headroom", "ann_financing", "avg_loan",
                                  "max_gross_leverage")}

    base_sr = sr(base)
    out: dict[str, Any] = {"base_sharpe": base_sr}
    family: dict[str, pd.Series] = {"base": base.window(a, b).net}
    pert = {}
    for label, ov in prog.family(name, cfg).items():
        res = simulate(name, m, cfg, costs, ov=ov)
        family[label] = res.window(a, b).net
        pert[label] = sr(res)
    vals = np.array(list(pert.values()))
    out["perturbation"] = pert
    out["plateau_fraction"] = float((vals > 0.5 * base_sr).mean()) if base_sr > 0 else 0.0
    out["min_perturbation_sharpe"] = float(vals.min())
    x2 = full(c=costs.scaled(multiplier=2.0))
    out["costs"] = {"x0": run(c=costs.scaled(multiplier=0.0)), "x2": v2._sr(x2),
                    "x3": run(c=costs.scaled(multiplier=3.0)),
                    "plus_10bp_slippage": run(c=costs.scaled(extra_slippage=10e-4)),
                    "impact_y_stress": run(c=costs.scaled(impact_eta=cfg.costs.impact_y_stress)),
                    "half_spread_stress": run(c=stress)}
    out["costs_x2_cagr"] = summary(x2, bootstrap=False)["cagr"]
    out["capacity"] = {f"aum_{int(x):d}": brief(full(c=costs.scaled(aum_usd=x)))
                       for x in (1e7, 5e7)}
    out["delay"] = {f"lag_{k}": run(lag=k) for k in (2, 3)}
    noise = [run(ov={"noise_seed": s}) for s in range(1, N_SEEDS + 1)]
    out["signal_noise"] = {"min": float(np.min(noise)), "median": float(np.median(noise)),
                           "max": float(np.max(noise)), "n_seeds": N_SEEDS}
    miss = [run(ov={"missing_seed": s}) for s in range(1, N_SEEDS + 1)]
    out["missing_data"] = {"min": float(np.min(miss)), "median": float(np.median(miss)),
                           "max": float(np.max(miss)), "n_seeds": N_SEEDS}
    out["rebalance"] = {f"every_{k}": run(ov={"every": k}) for k in (3, 7)}
    uni = {label: run(mk=mk) for label, mk in alt.items()}
    out["universe"] = uni
    out["universe_positive_fraction"] = float(np.mean([v > 0 for v in uni.values()])) if uni else 0.0
    out["start_dates"] = {f"start_plus_{k}m": sr(base.window(a + pd.DateOffset(months=k), b))
                          for k in (6, 12, 18)}
    out["end_dates"] = {f"end_minus_{k}m": sr(base.window(a, b - pd.DateOffset(months=k)))
                        for k in (6, 12)}
    r = base.window(a, b).net
    out["monte_carlo_drawdown"] = {"one_year": v2._mc(r, 365), "full_window": v2._mc(r, len(r))}
    # De hefboomstress: de twee aannames die niet in de data staan, en de governor.
    out["financing_stress"] = brief(full(ov={"financing": "stress"}))
    out["financing_optimistic"] = brief(full(ov={"financing": "optimistic"}))
    out["margin_stress"] = brief(full(ov={"margin_stress": True}))
    out["governor_off"] = brief(full(ov={"governor_off": True}))
    if prog.battery_extra is not None:
        out.update(prog.battery_extra(full, brief))
    return out, family


# --------------------------------------------------------------------------- #
# Selectie
# --------------------------------------------------------------------------- #
def select(records: Mapping[str, Mapping[str, Any]],
           leverage: Mapping[str, float]) -> tuple[str, str]:
    """De vooraf vastgelegde selectie: de LAAGSTE hefboom zonder bindende W_DEV-poort;
    haalt geen enkele kandidaat alles, dan de hoogste robuustheidsscore."""
    clean = [n for n in records if int(records[n]["n_binding_w_dev"]) == 0]
    if clean:
        return min(clean, key=lambda n: (float(leverage[n]), n)), "lowest_leverage_all_clear"
    return (max(records, key=lambda n: float(records[n]["robustness_score"]["total"])),
            "robustness_score_none_clear")


# --------------------------------------------------------------------------- #
# Bevriezen, meten, lezen
# --------------------------------------------------------------------------- #
def freeze(root: Path = ROOT, *, git_sha: str | None = None,
           prog: Programme | None = None) -> Path:
    prog = prog or V6
    cfg = prog.load_config(root / prog.config)
    _require_candidates(cfg, prog)
    prereg, hashes = _prereg(root, cfg, prog)
    require(prereg.planned_trials == cfg.planned_trials == len(prog.candidates),
            "planned_trials wijkt af.", DataContractError)
    sha = git_sha or current_git_sha()
    ledger = HypothesisLedger(root / v2.LEDGER_PATH)
    require(not [e for e in ledger.entries()
                 if e.get("preregistration_id") == prereg.preregistration_id],
            "Dit programma is al geboekt.", DataContractError)
    ledger.append(LedgerEntry.from_config(
        wave=prog.wave, unit=f"{prog.name}_programme", market="crypto", config=_params(cfg),
        git_sha=sha, data_hash=hash_config(dict(hashes)),
        preregistration_id=prereg.preregistration_id, n_trials=cfg.planned_trials,
        result="interim", notes=prog.notes))
    path = freeze_preregistration(prereg, git_sha=sha,
                                  ledger_total_at_freeze=ledger.total_n_hypotheses(),
                                  directory=root / v2.PREREG_DIR)
    # Het vooruit-slot is er één, voor de hele carryfamilie; lezingen zijn per hypothese-id.
    if not (root / FORWARD_LOCK).exists():
        freeze_holdout(split_utc=f"{cfg.forward.start}T00:00:00+00:00",
                       out=root / FORWARD_LOCK, git_sha=sha)
    return path


def run_programme(root: Path = ROOT, *, log: Callable[[str], None] = print,
                  prog: Programme | None = None) -> dict[str, Any]:
    prog = prog or V6
    simulate = prog.simulate
    cfg = prog.load_config(root / prog.config)
    _require_candidates(cfg, prog)
    prereg, hashes = _prereg(root, cfg, prog)
    require_preregistration(prereg.preregistration_id, directory=root / v2.PREREG_DIR)
    w = cfg.windows
    cut = v2._ts(w.holdout_start)
    m = load_market(root, cfg, end=cut)
    costs = costs_for(root, cfg)
    stress = costs_for(root, cfg, perp_half_spread_bps=cfg.v3.half_spread_stress_bps,
                       spot_half_spread_bps=cfg.basis.spot_half_spread_stress_bps)
    w_dev = (v2._ts(w.train_start), v2._ts(w.w_dev_end))
    train = (v2._ts(w.train_start), v2._ts(w.train_end))
    val = (v2._ts(w.validate_start), v2._ts(w.w_dev_end))
    sha = current_git_sha()
    results = {}
    for name in prog.candidates:
        log(f"run {name}")
        results[name] = simulate(name, m, cfg, costs)
    m_prog = int(HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses())
    m_wide = m_prog + int(cfg.known_prior_trials)
    per_bar = [summarize(results[n].window(*w_dev), bootstrap=False)["sharpe_per_bar"]
               for n in prog.candidates]
    alt = {f"top_{n}": load_market(root, cfg, end=cut, top_n=n) for n in (30, 100)}
    records = {}
    for name in prog.candidates:
        res = results[name]
        dev = summary(res.window(*w_dev))
        rec: dict[str, Any] = {
            "trial": name, "hypothesis": prog.hypotheses[name], "role": "candidate",
            "leverage_per_leg": float(cfg.leverage.candidates[name]),
            "preregistration_id": prereg.preregistration_id, "git_sha": sha,
            "data_hashes": dict(hashes), "parameters": _params(cfg), EVIDENCE_KEY: NOT_ADMISSIBLE,
            "windows": {"train": [w.train_start, w.train_end],
                        "validate": [w.validate_start, w.w_dev_end],
                        "backcast": [cfg.backcast.start, "unread"],
                        "holdout": [w.holdout_start, "unread"],
                        "forward": [cfg.forward.start, "unread"]},
            "execution": {"lag_bars": cfg.execution.lag_bars, "costs": costs.as_record(),
                          "margin": margin_spec(cfg).as_record()},
            "w_dev": dev, "train": summary(res.window(*train)),
            "validate": summary(res.window(*val)), "yearly": yearly(res.window(*w_dev)),
            "regimes": v2.regimes_v2(res.window(*w_dev), m.basis.perp),
            "dsr": {f"m_{k}": dsr_record(res.window(*w_dev).net, n_trials=k,
                                         trial_sharpes_per_bar=per_bar) for k in (m_prog, m_wide)},
        }
        log(f"batterij {name}")
        bat, family = battery(name, m, cfg, costs, stress, base=res, w_dev=w_dev, alt=alt,
                              prog=prog)
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
            "n_liquidations_w_dev": float(dev["n_liquidations"]),
            "n_liquidations_margin_stress_w_dev": float(bat["margin_stress"]["n_liquidations"]),
            "net_sharpe_financing_stress_w_dev": bat["financing_stress"]["sharpe"],
            "net_cagr_financing_stress_w_dev": bat["financing_stress"]["cagr"],
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
        v2._dump(root / prog.artefact_dir / f"{name}.json", rec)
        res.window(*w_dev).frame.to_csv(root / prog.artefact_dir / f"{name}_daily_w_dev.csv",
                                        float_format="%.10g")
    selected, rule = select(records, cfg.leverage.candidates)
    cands = prog.candidates
    summary_out = {
        "programme": prog.name, "preregistration_id": prereg.preregistration_id,
        "git_sha": sha, "m_programme": m_prog, "m_including_known_prior": m_wide,
        "robustness_scores": {n: records[n]["robustness_score"]["total"] for n in cands},
        "selected": selected, "selection_rule": rule,
        "passes_all_w_dev_gates": {n: records[n]["n_binding_w_dev"] == 0 for n in cands},
        "binding_w_dev": {n: sorted(k for k, g in records[n]["gates"].items() if g["binds"])
                          for n in cands},
        "table": {n: {k: records[n]["w_dev"][k] for k in (
            "sharpe", "sharpe_se", "cagr", "ann_vol", "max_drawdown", "sortino", "calmar",
            "ann_turnover", "ann_funding", "ann_financing", "avg_gross_leverage",
            "avg_loan", "n_liquidations", "min_stress_headroom", "sharpe_ci_low",
            "sharpe_ci_high")} for n in cands},
        EVIDENCE_KEY: NOT_ADMISSIBLE,
    }
    v2._dump(root / prog.artefact_dir / "programme_w_dev.json", summary_out)
    return summary_out


def _read(res: BookResult, a: pd.Timestamp, b: pd.Timestamp, dev_sr: float) -> dict[str, Any]:
    w = res.window(a, b)
    base = {"window": [str(a.date()), str(b.date())], "yearly": None,
            "max_gross_leverage": float(w.frame["gross_leverage"].max()),
            "pct_bars_invested": float((w.frame["gross_leverage"] > 1e-12).mean()),
            "n_liquidations": _liq_count(w), "min_stress_headroom": _headroom(w)}
    if float(w.frame["net"].std(ddof=1)) == 0.0:
        nan = float("nan")
        return {**base, "invested": False, "note": v5.FLAT_WINDOW, "z_vs_dev": nan,
                "summary": {"sharpe": nan, "sharpe_se": nan, "cagr": 0.0, "max_drawdown": 0.0,
                            "ann_funding": 0.0, "n_obs": int(w.frame.shape[0])}}
    s = summary(w)
    return {**base, "invested": True, "summary": s, "yearly": yearly(w),
            "z_vs_dev": gate_z(s["sharpe"], s["sharpe_se"], dev_sr)}


def _selected(root: Path, prereg_id: str, prog: Programme | None = None,
              ) -> tuple[str, dict[str, Any], dict[str, Any]]:
    prog = prog or V6
    d = root / prog.artefact_dir
    summary_w = json.loads((d / "programme_w_dev.json").read_text(encoding="utf-8"))
    require(summary_w["preregistration_id"] == prereg_id,
            "De W_DEV-run hoort bij een andere preregistratie.", DataContractError)
    name = summary_w["selected"]
    dev_rec = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
    return name, summary_w, dev_rec


def read_oos(root: Path = ROOT, *, log: Callable[[str], None] = print,
             prog: Programme | None = None) -> dict[str, Any]:
    """De backcast en de holdout van de geselecteerde kandidaat, allebei, in deze volgorde.
    Elke lezing staat in haar slot VOORDAT de data terugkomt."""
    prog = prog or V6
    simulate = prog.simulate
    cfg = prog.load_config(root / prog.config)
    prereg, hashes = _prereg(root, cfg, prog)
    require_preregistration(prereg.preregistration_id, directory=root / v2.PREREG_DIR)
    name, summary_w, dev_rec = _selected(root, prereg.preregistration_id, prog)
    dev_sr = float(dev_rec["w_dev"]["sharpe"])
    full = load_market(root, cfg)
    hid = f"{prog.name}/{prereg.preregistration_id}/{name}"
    close = full.basis.perp.book.close
    back = backcast_gate_slice(close, lock_path=root / v5.BACKCAST_LOCK, hypothesis_id=hid)
    hold = gate_slice(close, lock_path=root / v2.LOCK_PATH, hypothesis_id=hid)
    res = simulate(name, full, cfg, costs_for(root, cfg))
    again = v2._sr(res.window(v2._ts(cfg.windows.train_start), v2._ts(cfg.windows.w_dev_end)))
    require(abs(again - dev_sr) < 1e-9,
            "De volledige run reproduceert de W_DEV-Sharpe niet: er lekte iets.",
            DataContractError, full=again, truncated=dev_sr)
    ba = max(pd.Timestamp(back.index[0]), v2._ts(cfg.backcast.start))
    bb = min(pd.Timestamp(back.index[-1]), v2._ts(cfg.backcast.end))
    ha = pd.Timestamp(hold.index[0])
    hb = min(pd.Timestamp(hold.index[-1]), v2._ts(cfg.windows.holdout_end))
    reads = {"backcast": _read(res, ba, bb, dev_sr), "holdout": _read(res, ha, hb, dev_sr)}
    metrics = {
        "backcast_sharpe_z_vs_dev": reads["backcast"]["z_vs_dev"],
        "backcast_max_drawdown": reads["backcast"]["summary"]["max_drawdown"],
        "backcast_n_liquidations": float(reads["backcast"]["n_liquidations"]),
        "backcast_net_sharpe": reads["backcast"]["summary"]["sharpe"],
        "holdout_sharpe_z_vs_dev": reads["holdout"]["z_vs_dev"],
        "holdout_max_drawdown": reads["holdout"]["summary"]["max_drawdown"],
        "holdout_n_liquidations": float(reads["holdout"]["n_liquidations"]),
    }
    gates = v2._gates(metrics, prereg)
    falsify = {c.name for c in prereg.stop_criteria if c.action == "falsify"}
    n_binding = int(sum(g["binds"] for g in gates.values())) + int(dev_rec["n_binding_w_dev"])
    verdict = "promote_to_paper_trading" if n_binding == 0 else (
        "falsified_out_of_sample" if any(g["binds"] for k, g in gates.items() if k in falsify)
        else "archived")
    out = {"hypothesis_id": hid, "candidate": name, "w_dev_sharpe": dev_sr, "reads": reads,
           "selection_rule": summary_w["selection_rule"],
           "gate_metrics": metrics, "gates": gates, "n_binding_w_dev": dev_rec["n_binding_w_dev"],
           "n_binding_total": n_binding, "verdict": verdict,
           "caveat": "De backcast is voor de carryfamilie deels besmet (v5 las H3 daar), de "
                     "holdout is besmet; beide kunnen alleen verwerpen. Het schone bewijs is "
                     "het vooruit-sample vanaf 2026-10-01.",
           EVIDENCE_KEY: NOT_ADMISSIBLE}
    v2._dump(root / prog.artefact_dir / "oos_read.json", out)
    for label, (x, y) in {"backcast": (ba, bb), "holdout": (ha, hb)}.items():
        res.window(x, y).frame.to_csv(root / prog.artefact_dir / f"{name}_daily_{label}.csv",
                                      float_format="%.10g")
    result = {"promote_to_paper_trading": "accepted",
              "falsified_out_of_sample": "falsified"}.get(verdict, "archived")
    HypothesisLedger(root / v2.LEDGER_PATH).append(LedgerEntry.from_config(
        wave=prog.wave, unit=f"{prog.name}_verdict", market="crypto", config=_params(cfg),
        git_sha=current_git_sha(), data_hash=hash_config(dict(hashes)),
        preregistration_id=prereg.preregistration_id, n_trials=0, result=result,
        amends=hash_config(_params(cfg)),
        metrics={"selected": name, "w_dev_sharpe": dev_sr,
                 "w_dev_cagr": float(dev_rec["w_dev"]["cagr"]),
                 "backcast_sharpe": v5._finite(metrics["backcast_net_sharpe"]),
                 "holdout_sharpe": v5._finite(reads["holdout"]["summary"]["sharpe"]),
                 "holdout_cagr": float(reads["holdout"]["summary"]["cagr"]),
                 "holdout_invested": reads["holdout"]["invested"], "verdict": verdict,
                 "robustness_scores": summary_w["robustness_scores"]},
        notes=f"Oordeel {prog.name} na W_DEV, de backcast en de holdout. Het vooruit-"
              "sample (vanaf 2026-10-01) volgt met de `forward`-lezing."))
    for label, r in reads.items():
        s = r["summary"]
        log(f"{label} {name}: Sharpe {s['sharpe']:.3f} (SE {s['sharpe_se']:.3f}), CAGR "
            f"{s['cagr']:.3f}, MDD {s['max_drawdown']:.3f}, liquidaties {r['n_liquidations']}, "
            f"z={r['z_vs_dev']:.2f}")
    log(f"oordeel: {verdict}")
    return out


def forward_window_ready(index: pd.DatetimeIndex, start: str, min_months: int) -> tuple[bool, pd.Timestamp]:
    """Zijn er minstens `min_months` VOLLE maanden na `start`? Kijkt alleen naar het raster
    (welke dagen bestaan), nooit naar een waarde: dit mag vóór de registratie."""
    s = pd.Timestamp(start, tz="UTC")
    end = s + pd.DateOffset(months=int(min_months))
    # Een dagbar met sluittijd T dekt [T − 1 dag, T): de laatste dag van de maand sluit op
    # de eerste van de volgende.
    return bool(len(index) and index.max() >= end), end


def read_forward(root: Path = ROOT, *, panel_root: Path | None = None,
                 log: Callable[[str], None] = print,
                 prog: Programme | None = None) -> dict[str, Any]:
    """Het vooruit-sample, eenmaal, op panelen met data na het bevriezen.

    De bevroren panelen blijven onaangeroerd: het vooruit-sample leest uit
    `FORWARD_ROOT` (panels/, spot_panels/, mark_panels/), gebouwd met een latere
    `last_month`. De preregistratie is de BEVROREN (uit de W_DEV-run), niet een herberekende:
    nieuwe data verandert de data-hashes. Daarom moet de W_DEV-Sharpe op de nieuwe panelen
    exact reproduceren; anders is de historie herschreven en weigert de lezing."""
    prog = prog or V6
    simulate = prog.simulate
    cfg = prog.load_config(root / prog.config)
    summary_w = json.loads((root / prog.artefact_dir / "programme_w_dev.json")
                           .read_text(encoding="utf-8"))
    prereg_id = summary_w["preregistration_id"]
    frozen = require_preregistration(prereg_id, directory=root / v2.PREREG_DIR)
    name, _, dev_rec = _selected(root, prereg_id, prog)
    dev_sr = float(dev_rec["w_dev"]["sharpe"])
    pr = root / (panel_root or FORWARD_ROOT)
    require((pr / "panels" / "close.parquet").is_file(), "Geen vooruit-panelen: bouw ze eerst "
            "met een latere last_month (zie de moduledocstring).", DataContractError,
            path=str(pr))
    full = load_market(root, cfg, panel_root=pr)
    ok, need = forward_window_ready(full.index, cfg.forward.start, cfg.forward.min_months)
    require(ok, "Nog geen zes volle maanden vooruit-data; de lezing wordt NIET geregistreerd.",
            DataContractError, last_bar=str(full.index.max()), needed=str(need))
    res = simulate(name, full, cfg, costs_for(root, cfg))
    again = v2._sr(res.window(v2._ts(cfg.windows.train_start), v2._ts(cfg.windows.w_dev_end)))
    require(abs(again - dev_sr) < 1e-9, "De vooruit-panelen reproduceren de W_DEV-Sharpe niet: "
            "de historie is veranderd.", DataContractError, full=again, frozen=dev_sr)
    hid = f"{prog.name}/{prereg_id}/{name}"
    fwd = gate_slice(full.basis.perp.book.close, lock_path=root / FORWARD_LOCK, hypothesis_id=hid)
    fa, fb = pd.Timestamp(fwd.index[0]), pd.Timestamp(fwd.index[-1])
    read = _read(res, fa, fb, dev_sr)
    metrics = {"forward_sharpe_z_vs_dev": read["z_vs_dev"],
               "forward_max_drawdown": read["summary"]["max_drawdown"],
               "forward_n_liquidations": float(read["n_liquidations"]),
               "forward_net_sharpe": read["summary"]["sharpe"]}
    gates = v2._gates(metrics, frozen)
    falsify = {c.name for c in frozen.stop_criteria if c.action == "falsify"}
    n_binding = int(sum(g["binds"] for g in gates.values()))
    verdict = "eligible_for_capital" if n_binding == 0 else (
        "falsified_forward" if any(g["binds"] for k, g in gates.items() if k in falsify)
        else "archived_forward")
    out = {"hypothesis_id": hid, "candidate": name, "w_dev_sharpe": dev_sr, "read": read,
           "gate_metrics": metrics, "gates": gates, "verdict": verdict,
           EVIDENCE_KEY: NOT_ADMISSIBLE}
    v2._dump(root / prog.artefact_dir / "forward_read.json", out)
    HypothesisLedger(root / v2.LEDGER_PATH).append(LedgerEntry.from_config(
        wave=prog.wave, unit=f"{prog.name}_forward", market="crypto", config=_params(cfg),
        git_sha=current_git_sha(), data_hash=hash_config(dict(_data_hashes(
            root, mark_dir=pr / "mark_panels"))), preregistration_id=prereg_id, n_trials=0,
        result={"eligible_for_capital": "accepted", "falsified_forward": "falsified"}.get(
            verdict, "archived"), amends=hash_config(_params(cfg)),
        metrics={"selected": name, "forward_sharpe": v5._finite(read["summary"]["sharpe"]),
                 "forward_cagr": float(read["summary"]["cagr"]), "verdict": verdict},
        notes=f"De schone vooruit-lezing van {prog.name}."))
    log(f"vooruit {name}: {json.dumps(metrics, default=float)} -> {verdict}")
    return out


@dataclass(frozen=True)
class Programme:
    """Wat een programma van deze familie onderscheidt; de orkestratie is gedeeld.

    v7 (`programme_v7.py`) gebruikt dezelfde freeze/run/lees-code met een eigen config,
    preregistratie, kandidaten en simulatie: één implementatie per stap."""

    name: str
    wave: int
    config: Path
    spec: Path
    artefact_dir: Path
    candidates: tuple[str, ...]
    hypotheses: Mapping[str, str]
    load_config: Callable[[Path], Any]
    simulate: Callable[..., BookResult]
    family: Callable[[str, Any], dict[str, dict[str, Any]]]
    notes: str
    battery_extra: Callable[..., dict[str, Any]] | None = None


V6 = Programme(
    name="robust_book_v6", wave=6, config=CONFIG, spec=SPEC_PATH, artefact_dir=ARTEFACT_DIR,
    candidates=CANDIDATES, hypotheses=HYPOTHESES, load_config=robust_book_v6_config,
    simulate=simulate, family=_family,
    notes="Drie geplande trials: K1/K2/K3 = de basiscarry van v5-H1 bij 1x/1,5x/2x "
          "notional per been in een unified-margin-account. Post-hoc na v5 en zo "
          "geboekt; hefboom op eigenaarsbesluit 2026-10-09.")


def main(argv: Sequence[str]) -> None:
    cmds = {"freeze": lambda: print(freeze()),
            "run": lambda: print(json.dumps(run_programme(), indent=2, default=float)),
            "oos": lambda: print(json.dumps(read_oos(), indent=2, default=float)),
            "forward": lambda: print(json.dumps(read_forward(), indent=2, default=float))}
    require(len(argv) == 1 and argv[0] in cmds,
            "Gebruik: python -I -m tradebot.systematic.programme_v6 {freeze|run|oos|forward}",
            DataContractError)
    cmds[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
