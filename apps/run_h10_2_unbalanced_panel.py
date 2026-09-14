# apps/run_h10_2_unbalanced_panel.py
"""H-10.2 — het onevenwichtige paneel. Fase 10, stap 12.5 t/m 12.6.

Deze app is BEDRADING. De groeimeting staat in
`validation/phase10_unbalanced_panel.py`, de breedtepoort op de gewichten in
`portfolio/weights.py`.

WAT "BRUIKBAAR" HIER BETEKENT, en waarom dat de hele meting bepaalt. Het huidige
venster is `sigma.dropna(how="any") & adv.dropna(how="any")`: élk symbool moet een
ex-ante volatiliteit én een causale ADV hebben. Dat is de gebalanceerde eis, en
zij begint dus bij het laatst begonnen symbool. De breedtepoort vervangt "élk"
door "ten minste k", op exact dezelfde twee reeksen -- niet op de koersen, want de
risicolaag weigert een bar zonder vol-schatting en dan is een koers alleen geen
bruikbare bar.

DE LADDER DRAAIT HIER NIET VERDER DAN L0, EN DAT IS GEMETEN EN GEEN KEUZE.
`run_all_layers` op het ragged venster faalt met een `DataContractError` uit de
risicolaag: "Ontbrekende of niet-eindige ex-ante volatiliteitsschatting. De
risicolaag valt NIET terug op een laatste bekende waarde en NIET op een constante
vol." Dat is correct gedrag -- `phase5_baseline.py:170-184` bouwt
`sigma_hat={s: ... for s in symbols}` over ALLE symbolen op elke bar, dus een bar
waarop één naam nog niet bestaat, kan de soevereine laag per constructie niet
bereiken. Ragged doorvoeren tot L1 vraagt dat `RiskEngine.decide` een per-bar
SUBSET van symbolen aanvaardt, en dat is een wijziging in `backtest/` plus een
ontwerpbesluit over het risicocontract. Beide staan buiten de bestandenlijst van
deze stap. De vol-reeks opvullen om er toch door te komen is precies de stille
degradatie die die foutmelding verbiedt, en is hier dus niet gedaan.

DE TWEE L0-SHARPES STAAN NAAST ELKAAR EN NIET ALS VERBETERING. De stapopdracht
waarschuwt zelf: een Sharpe op een kort venster is niet vergelijkbaar met een
Sharpe op een lang venster. De twee vensters beslaan verschillende marktperiodes,
dus een verschil is een uitspraak over WELKE bars zijn toegevoegd. Elk getal komt
daarom met zijn Sharpe-triple (`n_obs`, `bars_per_year`, `t_years`) mee, en het
artefact draagt geen veld dat "verbetering" heet.

R-6-AANTEKENING. R-6 stelt 80 regels; dit bestand telt er 150: 34 docstring,
19 commentaar, 18 leeg en 79 code waarvan 25 importblok. Open gerapporteerd. De
54 resterende regels code zijn compositie; de rekenregels staan in `validation/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps"))

from run_phase5_baseline import load_market
from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import json_safe, load_baseline_configs
from tradebot.backtest.baseline_runner import CostModel, build_weight_tracks
from tradebot.backtest.vectorized import NOT_ADMISSIBLE, run_vectorized
from tradebot.data.pit_store import PitStore
from tradebot.features.base import DataRegister, load_certified_close_panel
from tradebot.features.registry import current_git_sha
from tradebot.portfolio.weights import normalise_weights
from tradebot.registry.ledger_reset import active_trial_count
from tradebot.registry.preregistration import (
    freeze_preregistration,
    load_preregistration_spec,
)
from tradebot.validation.holdout import development_slice
from tradebot.validation.inference import sharpe_with_se
from tradebot.validation.phase10_unbalanced_panel import measure_panel_growth

#: De vooraf geregistreerde breedte-eis, en de gebalanceerde eis waartegen zij
#: wordt gemeten. Eén configuratie, één trial (R-2).
MIN_SYMBOLS = 2
REFERENCE_MIN_SYMBOLS = 6
SPEC = "conf/experiment/h10_2_unbalanced_panel.yaml"
OUT = "artefacts/governance/phase10_h10_2.json"
LOCK = ROOT / "artefacts/governance/holdout_lock.json"
RESET = ROOT / "artefacts/governance/ledger_reset.json"
LEDGER = ROOT / "artefacts/governance/hypothesis_ledger.json"


def _l0_sharpe(weights, prices, *, idx, m, cost, cfg):
    """De L0-Sharpe op één venster, met zijn triple. NOOIT toelaatbaar bewijs."""
    w = normalise_weights(weights.loc[idx], min_symbols_per_bar=m).fillna(0.0)
    vec = run_vectorized(w, prices.loc[idx], cost_per_side=cost.per_side,
                         initial_equity=cfg["bt"].initial_equity)
    dev = development_slice(vec.returns.to_frame("r"), lock_path=LOCK)["r"]
    record = sharpe_with_se(dev, bars_per_year=cfg["bt"].bars_per_year).to_dict()
    record["evidence_class"] = NOT_ADMISSIBLE
    record["min_symbols_per_bar"] = int(m)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    market = load_market(ROOT, cfg)
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    panel = load_certified_close_panel(
        PitStore(ROOT / cfg["data"].pit_store_root), register,
        symbols=list(cfg["data"].symbols), granularity="1d", asset_class="crypto")
    tracks, _ = build_weight_tracks(
        panel, build_cross_sectional_momentum(cfg["alpha"]),
        lam=cfg["vol"].ewma_lambda, vol_burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor,
        gross_target=1.0, git_sha=current_git_sha())
    cost = CostModel(taker_fee_bps=cfg["exec"].taker_fee_bps,
                     half_spread_bps=cfg["exec"].assumed_half_spread_bps,
                     is_provisional=cfg["exec"].cost_assumption_is_provisional)

    # "Bruikbaar" = sigma EN adv aanwezig; zie de moduledocstring.
    usable = market["sigma"].notna() & market["adv"].notna()
    split = json.loads(LOCK.read_text(encoding="utf-8"))["split_utc"]
    development = usable.loc[usable.index < split]
    growth = measure_panel_growth(
        development, min_symbols_per_bar=MIN_SYMBOLS,
        reference_min_symbols_per_bar=REFERENCE_MIN_SYMBOLS,
        bars_per_year=cfg["bt"].bars_per_year)

    breadth = usable.sum(axis=1)
    windows = {"unbalanced": (MIN_SYMBOLS, usable.index[breadth >= MIN_SYMBOLS]),
               "balanced": (REFERENCE_MIN_SYMBOLS,
                            usable.index[breadth >= REFERENCE_MIN_SYMBOLS])}
    l0 = {name: {label: _l0_sharpe(tracks[name], market["prices"],
                                   idx=idx, m=m, cost=cost, cfg=cfg)
                 for label, (m, idx) in windows.items()} for name in sorted(tracks)}

    trials = active_trial_count(reset_path=RESET, ledger_path=LEDGER)
    prereg = load_preregistration_spec(
        ROOT / SPEC,
        data_hashes=[(s, register.hashes[s])
                     for s in sorted(register.hashes) if "/1d" in s],
        parameters={"min_symbols_per_bar": MIN_SYMBOLS,
                    "reference_min_symbols_per_bar": REFERENCE_MIN_SYMBOLS,
                    "usable_definition": "sigma.notna() & adv.notna()",
                    "bars_per_year": cfg["bt"].bars_per_year,
                    "m_new_frozen": trials.total})
    frozen = freeze_preregistration(prereg, git_sha=current_git_sha(),
                                    ledger_total_at_freeze=trials.archived_total)

    metrics = {
        "development_bars_added": growth.n_bars_added,
        "t2_hurdle_delta_development": growth.t2_hurdle_delta,
        "mean_breadth_added_bars_minus_existing":
            growth.mean_breadth_added - growth.mean_breadth_reference,
    }
    criteria = []
    for c in prereg.stop_criteria:
        value = metrics.get(c.metric)
        binds = None if value is None else bool(
            {"<=": value <= c.threshold, "<": value < c.threshold,
             ">=": value >= c.threshold, ">": value > c.threshold}[c.operator])
        criteria.append({"name": c.name, "metric": c.metric, "operator": c.operator,
                         "threshold": c.threshold, "action": c.action,
                         "measured": value, "binds": binds})
    n_binding = sum(1 for c in criteria if c["binds"])
    for c in criteria:
        if c["metric"] == "n_binding_stop_criteria":
            c["measured"] = n_binding
            c["binds"] = n_binding <= c["threshold"]

    payload = {"git_sha": current_git_sha(),
               "preregistration_id": prereg.preregistration_id,
               "preregistration_path": str(frozen),
               "ledger_total_at_freeze": trials.archived_total,
               "m_new_frozen": trials.total,
               "universe": list(cfg["data"].symbols),
               "sample": "development",
               "growth": growth.to_dict(),
               "l0_net_sharpe_by_window": l0,
               "l0_caveat": (
                   "The two windows span DIFFERENT market periods. A difference "
                   "between them is a statement about WHICH bars were added, not "
                   "about the system. Every record carries its own Sharpe triple; "
                   "neither supersedes the other."),
               "ladder_above_l0": {
                   "ran": False,
                   "reason": (
                       "RiskEngine refuses a bar without a finite ex-ante vol for "
                       "EVERY symbol (phase5_baseline.py:170-184 builds sigma_hat "
                       "over all symbols). Carrying a ragged panel to L1+ requires "
                       "RiskEngine.decide to accept a per-bar symbol subset: a "
                       "change in backtest/ plus a risk-contract decision, both "
                       "outside this step's file list. Filling the vol series to "
                       "get through is the silent degradation that error forbids."),
               },
               "stop_criteria": criteria,
               "n_binding_stop_criteria": n_binding,
               "trials": prereg.planned_trials,
               "artefact_path": args.out}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"development: {growth.n_bars_reference} -> {growth.n_bars} bars "
          f"({growth.n_bars_added:+d}, {100*growth.growth_fraction:+.2f} %), "
          f"t=2 hurdle {growth.t2_hurdle_reference:.4f} -> {growth.t2_hurdle:.4f} "
          f"({growth.t2_hurdle_delta:+.4f})")
    print(f"mean breadth {growth.mean_breadth_reference:.3f} -> {growth.mean_breadth:.3f}; "
          f"added bars carry {growth.mean_breadth_added:.3f}")
    print(f"{n_binding} stop criteria bind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
