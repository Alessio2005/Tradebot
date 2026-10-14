"""Robuust boek v5: spot-perp-basiscarry, alleen en naast trend.

    python -I -m tradebot.data.binance_vision spot-sync      # spotbeen (eenmalig)
    python -I -m tradebot.data.binance_vision spot-panels
    python -I -m tradebot.systematic.programme_v5 freeze     # ledger + prereg + backcast-slot
    python -I -m tradebot.systematic.programme_v5 run        # W_DEV, batterij, selectie
    python -I -m tradebot.systematic.programme_v5 oos        # backcast 2020 + holdout, één keer

WAAROM v5 BESTAAT. v2-v4 vonden geen prijssignaal dat buiten W_DEV standhield; de
cross-sectionele carry (Y5) verdiende op de holdout 22,9 %/jaar funding en verloor meer op
de prijs. v5 oogst funding zonder prijsweddenschap: long spot, short perp
(`systematic/harvest.py`). Ontwerp en alle vooraf gekozen parameters:
`docs/superpowers/specs/2026-10-07-robust-book-v5-basis-design.md`.

HET BEWIJS, EERLIJK GEWOGEN. Drie trials (cumulatief M = 21). W_DEV kiest. Daarna twee
lezingen, allebei vooraf vastgelegd en allebei uitgevoerd, wat de eerste ook zegt:
1. de BACKCAST (2020-04 .. 2020-12): nooit gemeten, door geen enkel eerder programma
   gerapporteerd. Dit is de echte out-of-sampletoets;
2. de HOLDOUT (2025-07 .. 2026-09): besmet. In v4 zag ik dat de short-kant van de carry
   daar 22,9 %/jaar funding ontving; het idee voor v5 is dus niet onafhankelijk van die
   periode. Hij telt alleen als FALSIFICATIE: een ineenstorting daar verwerpt, een goed
   resultaat bevestigt niets.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
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
from ..schemas.robust_book_v2 import RobustBookV5Config, robust_book_v5_config
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from ..validation.holdout import (
    backcast_gate_slice,
    freeze_holdout,
    gate_slice,
    resume_registered_read,
)
from . import programme_v2 as v2
from . import programme_v3 as v3
from .book import BookResult, CostSpec
from .breadth import load_breadth_market
from .evaluate import dsr_record, gate_z, pbo_record, robustness_score, summarize, yearly
from .harvest import (
    BasisCosts,
    BasisMarket,
    combine_accounts,
    harvest_targets,
    load_basis_market,
    run_basis,
)

__all__ = ["CANDIDATES", "freeze", "read_oos", "run_programme", "simulate"]

ROOT = v2.ROOT
CONFIG = Path("conf/model/robust_book_v5.yaml")
SPEC_PATH = Path("conf/research/preregistration_robust_book_v5.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v5")
SPOT_DIR = Path("data/binance_vision/spot_panels")
SPOT_MANIFEST = Path("artefacts/data/binance_spot_manifest.json")
BACKCAST_LOCK = Path("artefacts/governance/holdout_lock_binance_backcast_2020.json")
PROGRAMME_UNIT = "robust_book_v5_programme"
TREND = "Y4_TREND_LF"
CANDIDATES = ("H1_BASIS", "H2_BASIS_TREND", "H3_BASIS_MAJORS")
HYPOTHESES = {
    "H1_BASIS": "Basiscarry op de point-in-time top-50: tien slots, long spot / short perp, "
                "instap bij >= 15 %/jaar verwachte funding, uitstap onder 5 %.",
    "H2_BASIS_TREND": "Half het kapitaal in H1, half in de trend-sleeve Y4 (ongewijzigd uit "
                      "v3), dagelijks herverdeeld.",
    "H3_BASIS_MAJORS": "H1, alleen BTCUSDT en ETHUSDT (twee slots van 35 %).",
}
N_SEEDS = 10
#: Waarom de lezing van H3 is hervat (commit 37a9a40): beide slots waren geregistreerd,
#: daarna crashte de samenvatting van de holdout op nul variantie (H3 nam daar geen positie).
RESUME_REASON = ("De berekening crashte na de registratie op een identiek-nul rendement in "
                 "de holdout (geen positie); de backcast-uitkomst is nooit getoond of "
                 "opgeslagen. Afgemaakt met een venster-zonder-positie-regel, verder ongewijzigd.")


def _params(cfg: RobustBookV5Config) -> dict[str, Any]:
    return json.loads(cfg.model_dump_json())


def _data_hashes(root: Path) -> tuple[tuple[str, str], ...]:
    out = list(v2._data_hashes(root))
    spot = json.loads((root / SPOT_MANIFEST).read_text(encoding="utf-8"))
    out.append(("binance_spot/objects", spot["objects_sha256"][:16]))
    out += [(f"binance_spot/panel/{p.stem}", hashlib.sha256(p.read_bytes()).hexdigest()[:16])
            for p in sorted((root / SPOT_DIR).glob("*.parquet"))]
    return tuple(sorted(out))


def _prereg(root: Path, cfg: RobustBookV5Config):
    hashes = _data_hashes(root)
    return load_preregistration_spec(root / SPEC_PATH, data_hashes=hashes,
                                     parameters=_params(cfg)), hashes


def costs_for(root: Path, cfg: RobustBookV5Config, *, perp_half_spread_bps: float | None = None,
              spot_half_spread_bps: float | None = None) -> BasisCosts:
    perp = v3.cost_spec(root, cfg, half_spread_bps=perp_half_spread_bps)
    spot = CostSpec(taker_fee=cfg.basis.spot_taker_fee_bps * 1e-4,
                    half_spread=float(spot_half_spread_bps or cfg.basis.spot_half_spread_bps) * 1e-4,
                    impact=perp.impact, aum_usd=perp.aum_usd, impact_eta=perp.impact_eta)
    return BasisCosts(perp=perp, spot=spot)


def load_market(root: Path, cfg: RobustBookV5Config, *, end: pd.Timestamp | None = None,
                top_n: int | None = None) -> BasisMarket:
    c = cfg if top_n is None else cfg.model_copy(update={
        "universe": cfg.universe.model_copy(update={"top_n": int(top_n)})})
    m = load_basis_market(load_breadth_market(root / v2.PANEL_DIR, c), root / SPOT_DIR, c)
    return m if end is None else m.truncate(end)


# --------------------------------------------------------------------------- #
# De kandidaten, met alle verstoringen van de batterij als overrides
# --------------------------------------------------------------------------- #
def _noise(m: BasisMarket, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(1.0 + 0.25 * rng.standard_normal((len(m.index), len(m.symbols))),
                        index=m.index, columns=list(m.symbols))


def _every(tgt: pd.DataFrame, k: int) -> pd.DataFrame:
    """Besluit alleen elke k-de bar; daartussen geldt het laatste besluit."""
    on = (np.arange(len(tgt)) % int(k)) == 0
    return tgt.where(pd.Series(on, index=tgt.index), np.nan, axis=0).ffill().fillna(0.0)


def _missing(index: pd.DatetimeIndex, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(len(index)) < 0.05, index=index)


def simulate(name: str, m: BasisMarket, cfg: RobustBookV5Config, costs: BasisCosts, *,
             lag: int | None = None, ov: Mapping[str, Any] | None = None) -> BookResult:
    """Eén kandidaat, met optionele overrides: funding_span, enter_apr, exit_apr, slots,
    band, trend_lookbacks, noise_seed, missing_seed, every."""
    require(name in CANDIDATES, "Onbekende v5-kandidaat.", DataContractError, name=name)
    o = dict(ov or {})
    lg = int(cfg.execution.lag_bars if lag is None else lag)
    b = cfg.basis
    majors = name == "H3_BASIS_MAJORS"
    tgt = harvest_targets(
        m, span=int(o.get("funding_span", b.funding_span)),
        enter_apr=float(o.get("enter_apr", b.enter_apr)),
        exit_apr=float(o.get("exit_apr", b.exit_apr)),
        slots=len(b.majors) if majors else int(o.get("slots", b.slots)),
        notional=b.notional, min_spot_adv_usd=b.min_spot_adv_usd, max_abs_basis=b.max_abs_basis,
        symbols=b.majors if majors else None,
        carry_noise=_noise(m, int(o["noise_seed"])) if "noise_seed" in o else None)
    if "every" in o:
        tgt = _every(tgt, int(o["every"]))
    skip = _missing(m.index, int(o["missing_seed"])) if "missing_seed" in o else None
    basis = run_basis(tgt, m, costs, lag=lg, band=float(o.get("band", b.band)),
                      hedge_tolerance=b.hedge_tolerance, maintenance_margin=b.maintenance_margin,
                      margin_floor=b.margin_floor, skip=skip)
    if name != "H2_BASIS_TREND":
        return basis
    kw = {"trend_lookbacks": tuple(o["trend_lookbacks"])} if "trend_lookbacks" in o else {}
    t = v3.build(TREND, m.perp, cfg, **kw)
    if "noise_seed" in o:
        t = v2._noisy(t, cfg, int(o["noise_seed"]))
    if "missing_seed" in o:
        t = v2._missing(t, int(o["missing_seed"]))
    if "every" in o:
        t = v2._every(t, int(o["every"]))
    trend = v3.make_runner(cfg)(t, m.perp, costs.perp, lag=lg)
    s = float(b.combo_trend_share)
    return combine_accounts({"basis": (1.0 - s, basis), "trend": (s, trend)})


def _family(name: str, cfg: RobustBookV5Config) -> dict[str, dict[str, Any]]:
    fam: dict[str, dict[str, Any]] = {
        "funding_span_14": {"funding_span": 14}, "funding_span_60": {"funding_span": 60},
        "thresholds_low": {"enter_apr": 0.10, "exit_apr": 0.03},
        "thresholds_high": {"enter_apr": 0.20, "exit_apr": 0.08},
        "band_0.25": {"band": 0.25}, "band_0.75": {"band": 0.75},
    }
    if name != "H3_BASIS_MAJORS":
        fam |= {"slots_5": {"slots": 5}, "slots_20": {"slots": 20}}
    if name == "H2_BASIS_TREND":
        for f in (0.5, 2.0):
            fam[f"trend_lookbacks_x{f}"] = {"trend_lookbacks": tuple(sorted(
                {max(2, round(x * f)) for x in cfg.trend.lookbacks}))}
    return fam


def battery(name: str, m: BasisMarket, cfg: RobustBookV5Config, costs: BasisCosts,
            stress: BasisCosts, *, base: BookResult, w_dev: tuple[pd.Timestamp, pd.Timestamp],
            alt: Mapping[str, BasisMarket]) -> tuple[dict[str, Any], dict[str, pd.Series]]:
    """Alle gevoeligheden van één kandidaat op W_DEV, zoals `programme_v2.battery`."""
    a, b = w_dev

    def sr(res: BookResult) -> float:
        return v2._sr(res.window(a, b))

    def run(**kw: Any) -> float:
        mk = kw.pop("mk", m)
        return sr(simulate(name, mk, cfg, kw.pop("c", costs), **kw))

    base_sr = sr(base)
    out: dict[str, Any] = {"base_sharpe": base_sr}
    family: dict[str, pd.Series] = {"base": base.window(a, b).net}
    pert = {}
    for label, ov in _family(name, cfg).items():
        res = simulate(name, m, cfg, costs, ov=ov)
        family[label] = res.window(a, b).net
        pert[label] = sr(res)
    vals = np.array(list(pert.values()))
    out["perturbation"] = pert
    out["plateau_fraction"] = float((vals > 0.5 * base_sr).mean()) if base_sr > 0 else 0.0
    out["min_perturbation_sharpe"] = float(vals.min())
    out["costs"] = {"x0": run(c=costs.scaled(multiplier=0.0)),
                    "x2": run(c=costs.scaled(multiplier=2.0)),
                    "x3": run(c=costs.scaled(multiplier=3.0)),
                    "plus_10bp_slippage": run(c=costs.scaled(extra_slippage=10e-4)),
                    "impact_y_stress": run(c=costs.scaled(impact_eta=cfg.costs.impact_y_stress)),
                    "half_spread_stress": run(c=stress)}
    out["capacity"] = {f"aum_{int(x):d}": run(c=costs.scaled(aum_usd=x)) for x in (1e7, 5e7)}
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
    return out, family


def _liquidations(res: BookResult) -> int:
    if "accounts" in res.audit:
        return int(sum(acc["audit"].get("n_liquidations", 0)
                       for acc in res.audit["accounts"].values()))
    return int(res.audit.get("n_liquidations", 0))


# --------------------------------------------------------------------------- #
# Bevriezen, meten, lezen
# --------------------------------------------------------------------------- #
def freeze(root: Path = ROOT, *, git_sha: str | None = None) -> Path:
    cfg = robust_book_v5_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    require(prereg.planned_trials == cfg.planned_trials, "planned_trials wijkt af.",
            DataContractError)
    sha = git_sha or current_git_sha()
    ledger = HypothesisLedger(root / v2.LEDGER_PATH)
    require(not [e for e in ledger.entries()
                 if e.get("preregistration_id") == prereg.preregistration_id],
            "Dit programma is al geboekt.", DataContractError)
    ledger.append(LedgerEntry.from_config(
        wave=5, unit=PROGRAMME_UNIT, market="crypto", config=_params(cfg), git_sha=sha,
        data_hash=hash_config(dict(hashes)), preregistration_id=prereg.preregistration_id,
        n_trials=cfg.planned_trials, result="interim",
        notes="Drie geplande trials: H1 basiscarry breed, H2 basis + trend, H3 basis majors. "
              "Post-hoc na v4 en zo geboekt."))
    path = freeze_preregistration(prereg, git_sha=sha,
                                  ledger_total_at_freeze=ledger.total_n_hypotheses(),
                                  directory=root / v2.PREREG_DIR)
    freeze_holdout(split_utc=f"{cfg.windows.train_start}T00:00:00+00:00",
                   out=root / BACKCAST_LOCK, git_sha=sha)
    return path


def run_programme(root: Path = ROOT, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = robust_book_v5_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
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
    for name in CANDIDATES:
        log(f"run {name}")
        results[name] = simulate(name, m, cfg, costs)
    m_prog = int(HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses())
    m_wide = m_prog + int(cfg.known_prior_trials)
    per_bar = [summarize(results[n].window(*w_dev), bootstrap=False)["sharpe_per_bar"]
               for n in CANDIDATES]
    alt = {f"top_{n}": load_market(root, cfg, end=cut, top_n=n) for n in (30, 100)}
    records = {}
    for name in CANDIDATES:
        res = results[name]
        dev = summarize(res.window(*w_dev))
        rec: dict[str, Any] = {
            "trial": name, "hypothesis": HYPOTHESES[name], "role": "candidate",
            "preregistration_id": prereg.preregistration_id, "git_sha": sha,
            "data_hashes": dict(hashes), "parameters": _params(cfg), EVIDENCE_KEY: NOT_ADMISSIBLE,
            "windows": {"train": [w.train_start, w.train_end],
                        "validate": [w.validate_start, w.w_dev_end],
                        "backcast": [cfg.backcast.start, "unread"],
                        "holdout": [w.holdout_start, "unread"]},
            "execution": {"lag_bars": cfg.execution.lag_bars, "costs": costs.as_record()},
            "w_dev": dev, "train": summarize(res.window(*train)),
            "validate": summarize(res.window(*val)), "yearly": yearly(res.window(*w_dev)),
            "regimes": v2.regimes_v2(res.window(*w_dev), m.perp),
            "n_forced_exits": res.audit["n_forced_exits"], "n_liquidations": _liquidations(res),
            "dsr": {f"m_{k}": dsr_record(res.window(*w_dev).net, n_trials=k,
                                         trial_sharpes_per_bar=per_bar) for k in (m_prog, m_wide)},
        }
        log(f"batterij {name}")
        bat, family = battery(name, m, cfg, costs, stress, base=res, w_dev=w_dev, alt=alt)
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
        "programme": "robust_book_v5", "preregistration_id": prereg.preregistration_id,
        "git_sha": sha, "m_programme": m_prog, "m_including_known_prior": m_wide,
        "robustness_scores": scored, "selected": selected,
        "passes_all_w_dev_gates": {n: records[n]["n_binding_w_dev"] == 0 for n in CANDIDATES},
        "table": {n: {k: records[n]["w_dev"][k] for k in (
            "sharpe", "sharpe_se", "cagr", "ann_vol", "max_drawdown", "sortino", "calmar",
            "ann_turnover", "ann_funding", "avg_gross_leverage", "sharpe_ci_low",
            "sharpe_ci_high")} for n in CANDIDATES},
        EVIDENCE_KEY: NOT_ADMISSIBLE,
    }
    v2._dump(root / ARTEFACT_DIR / "programme_w_dev.json", summary)
    return summary


#: Een venster waarin het boek nooit een positie hield, heeft een identiek-nul rendement:
#: de Sharpe is dan ONGEDEFINIEERD (geen nul), CAGR en drawdown zijn nul. Ongedefinieerde
#: poortmetriek bindt volgens `programme_v2._gates` (niet-eindig = bindend), zoals bevroren.
FLAT_WINDOW = "flat: geen positie in het hele venster"


def _read(res: BookResult, a: pd.Timestamp, b: pd.Timestamp, dev_sr: float) -> dict[str, Any]:
    w = res.window(a, b)
    base = {"window": [str(a.date()), str(b.date())], "yearly": None,
            "max_gross_leverage": float(w.frame["gross_leverage"].max()),
            "pct_bars_invested": float((w.frame["gross_leverage"] > 1e-12).mean())}
    if float(w.frame["net"].std(ddof=1)) == 0.0:
        nan = float("nan")
        return {**base, "invested": False, "note": FLAT_WINDOW, "z_vs_dev": nan,
                "summary": {"sharpe": nan, "sharpe_se": nan, "cagr": 0.0, "max_drawdown": 0.0,
                            "ann_funding": 0.0, "n_obs": int(w.frame.shape[0])}}
    s = summarize(w)
    return {**base, "invested": True, "summary": s, "yearly": yearly(w),
            "z_vs_dev": gate_z(s["sharpe"], s["sharpe_se"], dev_sr)}


def _finite(x: float) -> float | None:
    """JSON-veilig: een ongedefinieerde Sharpe is `null` in de ledger, geen NaN."""
    return float(x) if np.isfinite(x) else None


def read_oos(root: Path = ROOT, *, log: Callable[[str], None] = print,
             resume_reason: str | None = None) -> dict[str, Any]:
    """De twee vooraf vastgelegde lezingen van de geselecteerde kandidaat, allebei, in
    deze volgorde. Elke lezing staat in zijn slot VOORDAT de data terugkomt.

    `resume_reason`: de lezingen staan al in hun slot, maar de berekening crashte vóór er
    een uitkomst was. Dan wordt dezelfde lezing eenmaal afgemaakt via
    `resume_registered_read` (zichtbaar in het slot), niet opnieuw geregistreerd."""
    cfg = robust_book_v5_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    require_preregistration(prereg.preregistration_id, directory=root / v2.PREREG_DIR)
    summary = json.loads((root / ARTEFACT_DIR / "programme_w_dev.json").read_text(encoding="utf-8"))
    require(summary["preregistration_id"] == prereg.preregistration_id,
            "De W_DEV-run hoort bij een andere preregistratie.", DataContractError)
    name = summary["selected"]
    dev_rec = json.loads((root / ARTEFACT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    dev_sr = float(dev_rec["w_dev"]["sharpe"])
    full = load_market(root, cfg)
    hid = f"robust_book_v5/{prereg.preregistration_id}/{name}"
    close = full.perp.book.close
    if resume_reason is None:
        back = backcast_gate_slice(close, lock_path=root / BACKCAST_LOCK, hypothesis_id=hid)
        hold = gate_slice(close, lock_path=root / v2.LOCK_PATH, hypothesis_id=hid)
    else:
        bsplit = pd.Timestamp(resume_registered_read(
            root / BACKCAST_LOCK, hypothesis_id=hid, reason=resume_reason)["split_utc"])
        hsplit = pd.Timestamp(resume_registered_read(
            root / v2.LOCK_PATH, hypothesis_id=hid, reason=resume_reason)["split_utc"])
        back, hold = close.loc[close.index < bsplit], close.loc[close.index >= hsplit]
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
        "backcast_net_sharpe": reads["backcast"]["summary"]["sharpe"],
        "holdout_sharpe_z_vs_dev": reads["holdout"]["z_vs_dev"],
        "holdout_max_drawdown": reads["holdout"]["summary"]["max_drawdown"],
    }
    gates = v2._gates(metrics, prereg)
    falsify = {c.name for c in prereg.stop_criteria if c.action == "falsify"}
    n_binding = int(sum(g["binds"] for g in gates.values())) + int(dev_rec["n_binding_w_dev"])
    verdict = "promote_to_paper_trading" if n_binding == 0 else (
        "falsified_out_of_sample" if any(g["binds"] for k, g in gates.items() if k in falsify)
        else "archived")
    out = {"hypothesis_id": hid, "candidate": name, "w_dev_sharpe": dev_sr, "reads": reads,
           "gate_metrics": metrics, "gates": gates, "n_binding_w_dev": dev_rec["n_binding_w_dev"],
           "n_binding_total": n_binding, "verdict": verdict,
           "n_liquidations_full_run": _liquidations(res),
           "resumed_after_crash": resume_reason,
           "caveat": "De backcast is nooit gemeten. De holdout is besmet (fundingniveau gezien "
                     "in v4) en telt alleen als falsificatie.",
           EVIDENCE_KEY: NOT_ADMISSIBLE}
    v2._dump(root / ARTEFACT_DIR / "oos_read.json", out)
    for label, (x, y) in {"backcast": (ba, bb), "holdout": (ha, hb)}.items():
        res.window(x, y).frame.to_csv(root / ARTEFACT_DIR / f"{name}_daily_{label}.csv",
                                      float_format="%.10g")
    result = {"promote_to_paper_trading": "accepted",
              "falsified_out_of_sample": "falsified"}.get(verdict, "archived")
    HypothesisLedger(root / v2.LEDGER_PATH).append(LedgerEntry.from_config(
        wave=5, unit="robust_book_v5_verdict", market="crypto", config=_params(cfg),
        git_sha=current_git_sha(), data_hash=hash_config(dict(hashes)),
        preregistration_id=prereg.preregistration_id, n_trials=0, result=result,
        amends=hash_config(_params(cfg)),
        metrics={"selected": name, "w_dev_sharpe": dev_sr,
                 "backcast_sharpe": _finite(metrics["backcast_net_sharpe"]),
                 "holdout_sharpe": _finite(reads["holdout"]["summary"]["sharpe"]),
                 "holdout_invested": reads["holdout"]["invested"], "verdict": verdict,
                 "robustness_scores": summary["robustness_scores"]},
        notes="Oordeel robuust boek v5 na W_DEV, de backcast en de holdout."))
    for label, r in reads.items():
        s = r["summary"]
        log(f"{label} {name}: Sharpe {s['sharpe']:.3f} (SE {s['sharpe_se']:.3f}), CAGR "
            f"{s['cagr']:.3f}, MDD {s['max_drawdown']:.3f}, z={r['z_vs_dev']:.2f}")
    log(f"oordeel: {verdict}")
    return out


def main(argv: Sequence[str]) -> None:
    cmds = {"freeze": lambda: print(freeze()),
            "run": lambda: print(json.dumps(run_programme(), indent=2, default=float)),
            "oos": lambda: print(json.dumps(read_oos(), indent=2, default=float)),
            "oos-resume": lambda: print(json.dumps(read_oos(resume_reason=RESUME_REASON),
                                                   indent=2, default=float))}
    require(len(argv) == 1 and argv[0] in cmds,
            "Gebruik: python -I -m tradebot.systematic.programme_v5 "
            "{freeze|run|oos|oos-resume}",
            DataContractError)
    cmds[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
