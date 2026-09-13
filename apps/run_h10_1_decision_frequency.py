# apps/run_h10_1_decision_frequency.py
"""H-10.1 — de beslisfrequentie. Fase 10, stap 11.5 t/m 11.7.

Deze app is BEDRADING. De meting staat in
`validation/phase10_decision_frequency_measurement.py`, de records en hun
contracten in `validation/phase10_decision_frequency.py`, en de vasthoudoperatie
zelf in `portfolio/decision_frequency.py`.

VIER ORDENINGSBESLUITEN, want bij deze hypothese zit de fout in de volgorde.

(1) EERST SNIJDEN NAAR HET BRUIKBARE VENSTER, DAN VASTHOUDEN. De blokken liggen
    positioneel vanaf de eerste VERHANDELBARE bar, niet vanaf de eerste bar van
    het ruwe paneel. De beslisklok begint wanneer het handelen begint; een blok
    dat al loopt voordat er een positie kan worden ingenomen, meet een besluit
    dat niemand had kunnen nemen.

(2) HET GEWICHTSPANEEL WORDT VASTGEHOUDEN, NIET DE EXPOSURE BINNEN DE LADDER.
    `backtest/phase5_baseline.py` staat niet op de bestandenlijst van deze stap en
    wordt daarom niet aangeraakt. Dat kost niets: `exposures_from_weights`
    herschaalt elke rij door haar eigen piek-absolute waarde, en een vastgehouden
    blok heeft over dat blok een CONSTANTE piek. De twee operaties commuteren dus
    op L1/L2/L3. Vasthouden op het gewichtspaneel heeft er bovendien een voordeel
    bij: ook `L0_vectorized` ziet de vasthoudoperatie, en dat is nodig omdat
    `mean_turnover` uitsluitend op L0 wordt geproduceerd.

(3) DE HALTKETEN LOOPT NA HET VASTHOUDEN. `run_all_layers` laat
    `RiskEngine.decide` per bar over de al vastgehouden exposure lopen. Een halt
    is een RISICObesluit en wordt niet vastgehouden -- zou hij wel worden
    vastgehouden, dan overrulet de beslisfrequentie van L8 de soevereine
    risicolaag, en dat is precies de omkering die AD-6 verbiedt.

(4) DE LADDER DRAAIT OVER HET VOLLE BRUIKBARE VENSTER, DE SNEDE NAAR DE
    ONTWIKKELSAMPLE GEBEURT OP DE RESULTAATREEKSEN. De L3-engine is
    PADAFHANKELIJK: equity, high-water mark en drawdown bepalen de haltbeslissing.
    Een run die bij de split begint, start die toestand opnieuw op en meet een
    ANDER systeem -- niet hetzelfde systeem over een korter venster. Omdat de
    blokken positioneel vanaf de eerste verhandelbare bar liggen, hangt de
    blokindeling van een ontwikkelbar nooit van een latere bar af; de
    volle-venster-run is dus causaal en het poortsample wordt hier niet gelezen.

HET POORTSAMPLE WORDT HIER NIET AANGERAAKT. Geen enkel pad in dit bestand roept
`gate_slice` aan; uitsluitend `development_slice`, en dat mag herhaald (R-7 geldt
alleen voor het poortsample). Substap 11.8 leest de poort precies eenmaal, voor
de ene k die op de ontwikkelsample het gunstigst is, en dat is een aparte
handeling met een eigen onherroepelijke registratie.

R-6-AANTEKENING. R-6 stelt een grens van 80 regels aan een app en dit bestand
telt er 251 (`wc -l`, de maat die `scripts/check_file_size.py` gebruikt):
58 docstring, 23 commentaar, 23 leeg en 146 code, waarvan 44 regels
importblok. Dat is een OPEN schending en wordt niet weggeschreven als "past wel".
De 103 resterende regels code zijn compositie -- configuratie laden, per track en
per k de ladder draaien, het resultaat doorgeven en wegschrijven -- en er zit geen
statistiek in; elke berekening staat in `validation/`. Wat de app groot maakt is
het aantal parameters dat `run_all_layers` nodig heeft (veertien), en dat is de
handtekening van de ladder en niet iets dat hier valt te verkorten;
`apps/run_phase5_baseline.py` staat op 408 regels om dezelfde reden. De docstring
en het commentaar dragen de vier ordeningsbesluiten hierboven: die verplaatsen
naar de meetmodule zou de reden waarom de bedrading zo loopt, weghalen van de plek
waar zij loopt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps"))

# `load_market` komt uit de baseline-app en wordt NIET gekopieerd: een tweede
# kopie van de dataladder zou stil kunnen gaan afwijken van de run waarmee de
# nulmeting is gedaan, en dan meet deze stap een ander paneel. `main()` daar
# staat achter een `__name__`-guard, dus importeren draait hem niet.
from run_phase5_baseline import load_market
from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import (
    json_safe,
    load_baseline_configs,
)
from tradebot.backtest.baseline_runner import CostModel, build_weight_tracks
from tradebot.backtest.phase5_baseline import run_all_layers
from tradebot.backtest.vectorized import run_vectorized
from tradebot.data.pit_store import PitStore
from tradebot.execution.impact_model import ImpactParams, ImpactStatus
from tradebot.execution.order_router import (
    SpreadModel,
    SpreadStatus,
    VenueSpec,
)
from tradebot.features.base import DataRegister, load_certified_close_panel
from tradebot.features.registry import current_git_sha
from tradebot.portfolio.decision_frequency import hold_decision
from tradebot.registry.ledger_reset import active_trial_count
from tradebot.registry.preregistration import (
    freeze_preregistration,
    load_preregistration_spec,
)
from tradebot.schemas.config import (
    ImpactConfig,
    RiskConfig,
    load_config,
)
from tradebot.validation.phase10_decision_frequency import (
    LadderRead,
    measure_hold_inertness,
)
from tradebot.validation.phase10_decision_frequency_measurement import (
    measure_campaign,
)

#: De vooraf geregistreerde beslisfrequenties. Vier waarden, vier trials (R-2);
#: k = 1 is de identiteit en dus de referentie. Dit grid IS de hypothese en staat
#: daarom hier en niet in een instelbaar veld: een vijfde waarde toevoegen is een
#: vijfde trial en een nieuwe pre-registratie, geen configuratiewijziging.
K_GRID: tuple[int, ...] = (1, 2, 5, 10)
#: De pre-geregistreerde cel. Zie het configbestand voor waarom juist deze.
PRIMARY_TRACK = "long_only_equal_weight"
PRIMARY_LAYER = "L3_execution"
#: De laag waarop een kostenas van één getal per eenheid omzet bestaat. Op L3 is
#: impact onscheidbaar van spread, fees en funding, dus daar bestaat zij niet.
COST_AXIS_LAYER = "L0_vectorized"
ANCHOR = "first_bar"
SPEC = "conf/experiment/h10_1_decision_frequency.yaml"
OUT = "artefacts/governance/phase10_h10_1.json"
LOCK = ROOT / "artefacts/governance/holdout_lock.json"
RESET = ROOT / "artefacts/governance/ledger_reset.json"
LEDGER = ROOT / "artefacts/governance/hypothesis_ledger.json"


def _ladder_reads(track, k, *, weights, market, usable, risk, cost, params, cfg):
    """Eén ladderrun bij één k op één track, teruggegeven als `LadderRead`."""
    decision = weights.loc[usable].fillna(0.0)
    held = hold_decision(decision, k=k, anchor=ANCHOR)
    prices = market["prices"].loc[usable]
    rows = {r.layer: r for r in run_all_layers(
        track, held, prices, market["sigma"].loc[usable],
        market["sigma"].loc[usable] / float(np.sqrt(market["annualisation"])),
        market["adv"].loc[usable], market["volume"].loc[usable],
        market["funding"].loc[usable], dict(risk.clusters), risk_cfg=risk,
        impact=params,
        venue=VenueSpec(maker_fee_bps=cfg["exec"].maker_fee_bps,
                        taker_fee_bps=cfg["exec"].taker_fee_bps,
                        funding_cap_abs=0.02, min_notional=10.0, latency_bars=1),
        spread=SpreadModel(half_spread_bps=cfg["exec"].assumed_half_spread_bps,
                           status=SpreadStatus.SPREAD_ASSUMED,
                           source="conf/execution/fees.yaml"),
        initial_equity=cfg["bt"].initial_equity, cost_per_side=cost.per_side,
        bars_per_year=cfg["bt"].bars_per_year)}
    # De omzetREEKS staat niet in `LayerResult` -- die vat haar samen als
    # `mean_turnover`. `run_vectorized` levert haar wel, en de meetlaag toetst
    # het gemiddelde van deze reeks tegen dat auditgetal: twee routes naar
    # dezelfde omzet, zodat een reeks uit een andere bron opvalt (R-3).
    vec = run_vectorized(held, prices, cost_per_side=cost.per_side,
                         initial_equity=cfg["bt"].initial_equity)
    return LadderRead(
        k=k, primary_net_returns=rows[PRIMARY_LAYER].returns,
        cost_axis_gross_returns=vec.gross_returns,
        cost_axis_net_returns=vec.returns, cost_axis_turnover=vec.turnover,
        execution_audit=rows[PRIMARY_LAYER].audit,
        cost_axis_audit=rows[COST_AXIS_LAYER].audit,
        inertness=measure_hold_inertness(decision, held, k=k))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    risk = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
    imp = load_config(ROOT / "conf/execution/impact.yaml", ImpactConfig)
    market = load_market(ROOT, cfg)
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    tracks, _ = build_weight_tracks(
        load_certified_close_panel(
            PitStore(ROOT / cfg["data"].pit_store_root), register,
            symbols=list(cfg["data"].symbols), granularity="1d",
            asset_class="crypto"),
        build_cross_sectional_momentum(cfg["alpha"]),
        lam=cfg["vol"].ewma_lambda, vol_burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor,
        gross_target=1.0, git_sha=current_git_sha())
    usable = market["sigma"].dropna(how="any").index
    usable = usable[usable.isin(market["adv"].dropna(how="any").index)]
    cost = CostModel(taker_fee_bps=cfg["exec"].taker_fee_bps,
                     half_spread_bps=cfg["exec"].assumed_half_spread_bps,
                     is_provisional=cfg["exec"].cost_assumption_is_provisional)
    params = ImpactParams(
        eta=imp.eta, kappa_d=imp.kappa_d, status=ImpactStatus(imp.status),
        method=imp.method, data_hash=imp.data_hash, sample_size=imp.sample_size,
        period_start=imp.period_start, period_end=imp.period_end,
        instruments=imp.instruments, eta_ci_low=imp.eta_ci_low,
        eta_ci_high=imp.eta_ci_high)

    # Alle vier tracks, niet alleen de primaire. De pre-registratie stelt dat de
    # keten breekt omdat de track die HALTEERT geen besluit heeft om vast te
    # houden, terwijl de tracks met omzet NOOIT halteren. Die bewering hoort
    # gemeten in dit artefact te staan en niet ernaast: de drie niet-primaire
    # tracks leveren daarom de mechanische diagnostiek (`is_primary_track:
    # false`). Er wordt niets uit geselecteerd -- de primaire cel staat vooraf
    # vast -- en zij kosten dus geen trial.
    reads = {name: [_ladder_reads(name, k, weights=tracks[name], market=market,
                                  usable=usable, risk=risk, cost=cost,
                                  params=params, cfg=cfg) for k in K_GRID]
             for name in sorted(tracks)}
    trials = active_trial_count(reset_path=RESET, ledger_path=LEDGER)
    prereg = load_preregistration_spec(
        ROOT / SPEC,
        data_hashes=[(s, register.hashes[s])
                     for s in sorted(register.hashes) if "/1d" in s],
        parameters={"k_grid": list(K_GRID), "primary_track": PRIMARY_TRACK,
                    "primary_layer": PRIMARY_LAYER,
                    "cost_axis_layer": COST_AXIS_LAYER, "anchor": ANCHOR,
                    "cost_per_side_bps": cost.per_side * 1e4,
                    "bars_per_year": cfg["bt"].bars_per_year,
                    "eta": imp.eta, "eta_status": imp.status,
                    "m_new_frozen": trials.total})
    frozen = freeze_preregistration(
        prereg, git_sha=current_git_sha(),
        ledger_total_at_freeze=trials.archived_total)
    campaign = measure_campaign(
        reads, primary_track=PRIMARY_TRACK, primary_layer=PRIMARY_LAYER,
        cost_axis_layer=COST_AXIS_LAYER, anchor=ANCHOR, lock_path=LOCK,
        bars_per_year=cfg["bt"].bars_per_year,
        cost_per_side_bps=cost.per_side * 1e4, trial_count=trials.to_trial_count(),
        validation=cfg["val"], stop_criteria=prereg.stop_criteria)

    payload = {"git_sha": current_git_sha(),
               "preregistration_id": prereg.preregistration_id,
               "preregistration_path": str(frozen),
               "ledger_total_at_freeze": trials.archived_total,
               "m_new_frozen": trials.total,
               "universe": list(cfg["data"].symbols),
               "n_usable_bars": int(len(usable)),
               "campaign": campaign.to_dict(),
               "gate_read": None,
               "artefact_path": args.out}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(campaign.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
