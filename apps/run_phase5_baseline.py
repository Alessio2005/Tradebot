"""Phase 5, §20 — herwaardeer de Phase 3-baseline door de volledige keten.

De wetenschap staat in `backtest/phase5_baseline.py`; deze app doet argumenten,
data en artefact (R-6: apps <= 80 LOC).

    python apps/run_phase5_baseline.py
    python apps/run_phase5_baseline.py --with-transitions   # fase 10, stap 10

R-6-AANTEKENING (fase 10, stap 10). Dit bestand staat ver boven de 80 regels die
R-6 voor een app voorschrijft, en dat is een OPEN, GERAPPORTEERDE schending en
geen stilzwijgende. Twee feiten horen erbij. (1) De app stond op HEAD 8d62d5c al
op 138 regels en dus al boven R-6; stap 10 heeft die schending vergroot, niet
veroorzaakt. (2) `TRANSITIONS`, `_CAVEAT`, `_verdict` en `_transition_block` zijn
LOGICA en horen naar de letter van R-6 in
`src/tradebot/backtest/phase5_baseline.py` (305 regels, ruim onder de R-4-grens
van 800). Die module staat niet op de commitlijst van deze stap - die is door de
controller vastgelegd op zes paden en dit is er niet een van - en een pad
toevoegen in een boom waarin drie andere uitvoerders tegelijk committen, is niet
aan de uitvoerder. De verhuizing is daarom als BEVINDING gerapporteerd in
`.superpowers/sdd/fase_10_herstart_dagbars/stap-10-report.md` en wacht op een
uitspraak; zodra de commitlijst dat toelaat, verhuist dit blok in zijn geheel.

WAARSCHUWING OVER HET ARTEFACT. `--with-transitions` staat standaard UIT (zo legt
ruling P43(a) het vast), zodat de bestaande baselinerun ongewijzigd blijft. Het
gevolg daarvan is dat een run ZONDER de vlag de sleutel `layer_transitions` uit
het artefact VERWIJDERT - het artefact wordt in zijn geheel herschreven. En
`dvc.yaml` draait de stage `phase5_revaluation` op regel 179 juist zonder vlag,
dus de eerstvolgende `dvc repro` wist dit blok. Dat is een OPENSTAANDE BEVINDING
van stap 10: de reparatie is één woord in `dvc.yaml`, maar dat bestand staat niet
op de commitlijst van deze stap en wordt op dit moment door een andere uitvoerder
bewerkt. Zie `.superpowers/sdd/fase_10_herstart_dagbars/stap-10-report.md`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import load_baseline_configs
from tradebot.backtest.baseline_runner import CostModel, build_weight_tracks
from tradebot.backtest.phase5_baseline import LayerResult, run_all_layers, summarise
from tradebot.data.funding_panel import daily_funding_panel
from tradebot.data.pit_store import PitStore
from tradebot.execution.impact_model import ImpactParams, ImpactStatus
from tradebot.execution.order_router import SpreadModel, SpreadStatus, VenueSpec
from tradebot.features.base import (
    ASOF_INDEX_NAME,
    DataRegister,
    load_certified_close_panel,
    load_certified_series,
)
from tradebot.features.registry import current_git_sha
from tradebot.schemas.config import ImpactConfig, RiskConfig, load_config
from tradebot.validation.holdout import development_slice
from tradebot.validation.inference import sharpe_difference_test
from tradebot.volatility.ewma import ewma_volatility_panel

#: De bevroren ontwikkel/poort-grens. Stap 10 leest UITSLUITEND de
#: ontwikkelsample; dat is vrij van R7 en kost geen trial.
LOCK = ROOT / "artefacts/governance/holdout_lock.json"

#: De vijf laagovergangen van stap 10, als `(van, naar)` met elk
#: `(track, fidelity-laag)`. De afbeelding van de informele namen
#: `l1_l2_only` / `l1_l7_no_halt` / `l1_l7_with_halt` op de twee echte assen van
#: het artefact is ruling P42; die namen bestaan nergens in de codebase en zijn
#: hier NIET opnieuw afgeleid. `delta_sharpe = SR(naar) - SR(van)`.
TRANSITIONS: dict[str, tuple[tuple[str, str], tuple[str, str], str]] = {
    "T1_equal_weight_to_l1_l2": (
        ("long_only_equal_weight", "L0_vectorized"),
        ("xs_momentum_equal_weight", "L0_vectorized"),
        "Voegt de alfalaag iets toe aan gelijk gewicht?"),
    "T2_l1_l2_to_l1_l7_no_halt": (
        ("xs_momentum_equal_weight", "L0_vectorized"),
        ("xs_momentum_risk_parity", "L0_vectorized"),
        "Voegt de portefeuilleconstructie iets toe?"),
    "T3_l1_l7_no_halt_to_l1_l7_with_halt": (
        ("xs_momentum_risk_parity", "L0_vectorized"),
        ("xs_momentum_risk_parity", "L1_sovereign"),
        "Wat kost de haltketen?"),
    "T4_equal_weight_to_l1_l7_with_halt": (
        ("long_only_equal_weight", "L0_vectorized"),
        ("xs_momentum_risk_parity", "L1_sovereign"),
        "Is het volledige systeem beter dan het triviale alternatief?"),
    "T3b_long_only_halt_cost": (
        ("long_only_equal_weight", "L0_vectorized"),
        ("long_only_equal_weight", "L1_sovereign"),
        "Wat kost de haltketen op het spoor waar zij WEL vuurt?"),
}

#: Extra zin per overgang, bovenop de gegenereerde uitspraak. P42-addendum 1 eist
#: dat T3 zijn `n_sovereign_halted` naast zich draagt en dat T3b als SECUNDAIR
#: wordt gelabeld.
_CAVEAT: dict[str, str] = {
    "T3_l1_l7_no_halt_to_l1_l7_with_halt": (
        "LET OP — dit meet op dit spoor GEEN halt. Op xs_momentum_risk_parity "
        "registreert L3_execution n_sovereign_halted={halted_to} naast "
        "n_sovereign_clipped={clipped_to}: de soevereine laag is hier nooit "
        "gehalteerd, zij heeft GEKNIPT. Het antwoord op 'wat kost de "
        "haltketen?' luidt op dit spoor dus letterlijk: zij is nooit "
        "aangegaan, en dit getal is de prijs van vol-targeting en limieten, "
        "niet van een halt. De bindende beperkingen die L1_sovereign op dit "
        "spoor telt ({binding_to}, over het VOLLEDIGE venster van 1743 bars en "
        "dus niet over de {n_dev} ontwikkelbars hierboven) bevatten dan ook "
        "geen halt. Dat is bovendien structureel en niet toevallig: "
        "backtest/phase5_baseline.py::_sovereign_weights beoordeelt L1 per bar "
        "met een VERSE RiskState, zodat de drawdown daar per constructie nul is "
        "en een drawdown-halt op GEEN ENKEL spoor bij dit eindpunt kan vuren; "
        "{n_dropped} bars zijn hier door align='common_active' gevallen."),
    "T3b_long_only_halt_cost": (
        "SECUNDAIR EN NIET VOORGEREGISTREERD (P42-addendum 1). Dit blok bestaat "
        "omdat de voorgeregistreerde T3 inert is op zijn eigen spoor; het "
        "vervangt of hernummert T1-T4 niet en mag niet als vijfde "
        "voorgeregistreerde toets worden gelezen. EN DE VERWACHTING WAAROP HET "
        "BERUST, IS DOOR DE METING WEERLEGD (R-10). P42-addendum 1 wijst dit "
        "blok aan als het spoor 'waar de haltketen WEL vuurt'; hier vuurt zij "
        "niet. Bewijs uit de meting: align='common_active' laat een bar vallen "
        "zodra een van beide reeksen exact nul staat, en op dit paar zijn "
        "{n_dropped} bars gevallen tegen {n_dev} die bleven - het eindpunt "
        "L1_sovereign kent op dit spoor dus geen enkele vlakke bar. De oorzaak "
        "is mechanisch en niet toevallig: backtest/phase5_baseline.py::"
        "_sovereign_weights beoordeelt L1 per bar met een VERSE RiskState "
        "(equity = high_water_mark = day_start_equity), zodat de drawdown daar "
        "per constructie nul is en een drawdown-halt niet KAN vuren; die laag "
        "telt alleen bindende beperkingen ({binding_to}) en daar staat geen halt "
        "tussen. De haltteller n_sovereign_halted={halted_to} hoort bij "
        "L3_execution - de event-driven engine, die equity wel over bars heen "
        "bijhoudt - en dus bij een ANDER eindpunt dan hier is gemeten. Wat hier "
        "staat is daarom de prijs van vol-targeting en limieten op "
        "long_only_equal_weight, niet de prijs van een halt. De haltkosten zijn "
        "in dit artefact NIET gemeten: een blok dat ze wel meet zou op "
        "L3_execution moeten eindigen, waar zij niet van spread, impact, fees en "
        "funding te scheiden zijn."),
}


def load_market(root: Path, cfg: dict) -> dict:
    """Prijzen, volatiliteit, causale ADV, bar-volume en funding."""
    store = PitStore(root / cfg["data"].pit_store_root)
    register = DataRegister(root / "artefacts/governance/data_hashes.json")
    symbols = list(cfg["data"].symbols)
    prices = load_certified_close_panel(
        store, register, symbols=symbols, granularity="1d",
        asset_class="crypto").values
    sigma = ewma_volatility_panel(
        prices, lam=cfg["vol"].ewma_lambda, burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor)
    turnover = {}
    for symbol in symbols:
        df, _ = load_certified_series(store, register, asset_class="crypto",
                                      dataset="ohlcv", symbol=symbol,
                                      granularity="1d")
        idx = pd.DatetimeIndex(pd.to_datetime(df["asof_ts_ns"].to_numpy(),
                                              unit="ns", utc=True),
                               name=ASOF_INDEX_NAME)
        turnover[symbol] = pd.Series(df["turnover"].to_numpy(dtype="float64"),
                                     index=idx)
    volume = pd.DataFrame(turnover).reindex(prices.index)
    # Causaal: de turnover van bar t is pas op zijn close bekend.
    adv = volume.rolling(30, min_periods=30).mean().shift(1)
    funding = daily_funding_panel(
        store, register, symbols=symbols, asset_class="crypto",
        funding_granularity="8h", bar_index=prices.index)
    return {"prices": prices, "sigma": sigma, "adv": adv, "volume": volume,
            "funding": funding, "annualisation": cfg["vol"].annualisation_factor}


def _dev(series: pd.Series) -> pd.Series:
    """De ONTWIKKELsample van één reeks: alles vóór de bevroren split.

    Ruling P43(c): het artefact beslaat het VOLLEDIGE venster, poortsample
    inbegrepen, terwijl elk blok `"sample": "development"` moet dragen. Deze
    snede is daarom niet optioneel. Lezen van de ontwikkelsample is vrij van R7
    en kost geen trial; lezen van het poortsample zou een bestuursbreuk zijn.
    """
    return development_slice(series.to_frame("r"), lock_path=LOCK)["r"]


def _show(ec: object) -> str:
    """`None` is hier niet "geen oordeel" maar "deze laag zet dit veld niet".

    Alleen `L0_vectorized` draagt een `evidence_class` in dit artefact. Dat
    verschil moet leesbaar zijn en niet als het woord "None" in een
    bestuursartefact belanden.
    """
    return str(ec) if ec else "geen evidence_class in dit artefact"


def _audit(rows: list[LayerResult], track: str, layer: str, key: str) -> object:
    for row in rows:
        if row.track == track and row.layer == layer:
            return row.audit.get(key)
    return None


def _verdict(res, hurdle: float, t_years: float, analytic: list[float]) -> str:
    """De uitspraak, GEGENEREERD uit de gemeten getallen en niet met de hand.

    Een met de hand geschreven `note` kan de getallen naast zich tegenspreken
    zodra het artefact opnieuw wordt gedraaid. Deze zin kan dat niet: elke
    bewering hieronder is een vertakking op een gemeten grootheid, inclusief de
    bewering dat de twee intervalroutes het eens zijn.
    """
    straddles = res.ci_low <= 0.0 <= res.ci_high
    ratio = (analytic[1] - analytic[0]) / (res.ci_high - res.ci_low)
    # De brief (test 5) zegt het zelf: wijken de twee routes sterk af, dan IS
    # dat de bevinding. Een vaste, genoemde band in plaats van een zin die
    # instemming aanneemt.
    agree = 0.80 <= ratio <= 1.25
    return (
        f"delta_sharpe = {res.delta_sharpe:+.4f} met Ledoit-Wolf-SE {res.se:.4f}. "
        f"Het 95 %-bootstrap-t-interval [{res.ci_low:+.4f}, {res.ci_high:+.4f}] "
        + ("OMVAT NUL: dit is een NIET-bevinding en moet als zodanig worden "
           "gelezen, niet als een richting met zwakke steun. "
           if straddles else
           "omvat nul NIET. ")
        + f"p = {res.p_value:.4f} (tweezijdig, gestudentiseerde circulaire "
        f"blokbootstrap, n_boot={res.n_boot}, bloklengte={res.block_length}). "
        f"Het analytische interval [{analytic[0]:+.4f}, {analytic[1]:+.4f}] is "
        f"{ratio:.2f}x zo breed als het bootstrap-interval"
        + ("; binnen de band [0,80; 1,25] waarin de twee onafhankelijke routes "
           "als eensluidend gelden. " if agree else
           " en valt daarmee BUITEN de band [0,80; 1,25]. Dat verschil tussen de "
           "twee routes is zelf de bevinding en geen detail: de asymptotiek en "
           "de bootstrap meten hier niet hetzelfde. ")
        + f"t = {res.t_stat:+.3f} op n_effective = {res.n_effective} "
        f"gemeenschappelijk actieve bars (= {t_years:.2f} jaar, "
        f"{res.n_dropped} bars gevallen door align='common_active'). "
        f"De Sharpe-drempel die op DEZE horizon een t van 2 haalt is "
        f"2/sqrt({t_years:.2f}) = {hurdle:.2f}; |delta_sharpe| = "
        f"{abs(res.delta_sharpe):.4f} ligt daar "
        + ("ONDER. Een verschil onder die drempel is geen zwak bewijs maar "
           "GEEN BEWIJS." if abs(res.delta_sharpe) < hurdle else
           "BOVEN. Dat maakt het nog geen bevinding: het interval hierboven "
           "beslist, niet de vuistregel.")
    )


def _transition_block(name: str, rows: list[LayerResult], *,
                      bars_per_year: float) -> dict:
    """Eén laagovergang, met alles wat stap 10.4 verplicht stelt."""
    (t_van, l_van), (t_naar, l_naar), question = TRANSITIONS[name]
    by = {(r.track, r.layer): r for r in rows}
    a = _dev(by[(t_naar, l_naar)].returns)   # NAAR
    b = _dev(by[(t_van, l_van)].returns)     # VAN
    res = sharpe_difference_test(a, b, bars_per_year=bars_per_year,
                                 align="common_active")
    z = float(norm.ppf(1.0 - (1.0 - res.ci_level) / 2.0))
    analytic = [res.delta_sharpe - z * res.se, res.delta_sharpe + z * res.se]
    t_years = res.n_effective / float(bars_per_year)
    hurdle = 2.0 / math.sqrt(t_years)
    ec_van = by[(t_van, l_van)].audit.get("evidence_class")
    ec_naar = by[(t_naar, l_naar)].audit.get("evidence_class")
    # P42-addendum 2: elke note noemt de fidelity-laag van BEIDE eindpunten en
    # zegt dat een L0-eindpunt dit getal tot diagnostiek maakt.
    admissibility = (
        f"VAN {t_van}/{l_van} (evidence_class={_show(ec_van)}) NAAR "
        f"{t_naar}/{l_naar} (evidence_class={_show(ec_naar)}). "
    )
    if "L0_vectorized" in (l_van, l_naar):
        admissibility += (
            "Ten minste één eindpunt staat op L0_vectorized, dat in dit "
            "artefact evidence_class=NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE "
            "draagt. Dit getal is DIAGNOSTIEK over laagtoewijzing op de "
            "ontwikkelsample (stap 10: nul trials) en is NOOIT een toelaatbare "
            "prestatieclaim; het mag niet uit dit artefact worden gelicht en "
            "als bewijs worden geciteerd. "
        )
    caveat = _CAVEAT.get(name, "").format(
        halted_to=_audit(rows, t_naar, "L3_execution", "n_sovereign_halted"),
        clipped_to=_audit(rows, t_naar, "L3_execution", "n_sovereign_clipped"),
        binding_to=_audit(rows, t_naar, "L1_sovereign", "binding_constraints"),
        n_dev=res.n_effective,
        n_dropped=res.n_dropped,
    )
    return {
        "question": question,
        "from": {"track": t_van, "layer": l_van, "evidence_class": ec_van},
        "to": {"track": t_naar, "layer": l_naar, "evidence_class": ec_naar},
        "period": [str(a.index[0].date()), str(a.index[-1].date())],
        "delta_sharpe": res.delta_sharpe,
        "sharpe_from": res.sharpe_b,
        "sharpe_to": res.sharpe_a,
        "se_ledoit_wolf": res.se,
        "t_stat": res.t_stat,
        "ci_95_analytic": analytic,
        "ci_95_block_bootstrap": list(res.ci_95_block_bootstrap),
        # Gemeten, niet overgeschreven: `conf/validation/inference.yaml` zet
        # `block_length: null` (automatische kalibratie) en zegt er zelf bij dat
        # een vast getal "bedoeld [is] voor een reproductie van een eerdere
        # meting, niet voor productie". De brief toont 20 in zijn JSON-sjabloon;
        # hier staat de waarde die de toets DAADWERKELIJK heeft gebruikt, want
        # een veld dat 20 zegt terwijl er 5 is gerekend, is een onwaarheid in
        # een bestuursartefact.
        "block_length": res.block_length,
        "n_boot": res.n_boot,
        "seed": res.seed,
        "p_value": res.p_value,
        "n_bars": int(len(a.index.intersection(b.index))),
        "n_effective": res.n_effective,
        "n_dropped": res.n_dropped,
        "t_years": t_years,
        "sharpe_hurdle_t2": hurdle,
        "bars_per_year": float(bars_per_year),
        "align": res.align,
        "sample": "development",
        "note": f"{question} {admissibility}{_verdict(res, hurdle, t_years, analytic)} {caveat}".strip(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artefacts/baseline/phase5_revaluation.json")
    ap.add_argument("--with-transitions", action="store_true",
                    help="bereken de vijf laagovergangen van fase 10, stap 10 "
                         "(diagnostiek op de ontwikkelsample, nul trials)")
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    risk = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
    imp = load_config(ROOT / "conf/execution/impact.yaml", ImpactConfig)
    market = load_market(ROOT, cfg)
    tracks, _ = build_weight_tracks(
        load_certified_close_panel(
            PitStore(ROOT / cfg["data"].pit_store_root),
            DataRegister(ROOT / "artefacts/governance/data_hashes.json"),
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

    rows = []
    for name, weights in sorted(tracks.items()):
        rows.extend(run_all_layers(
            name, weights.loc[usable].fillna(0.0),
            market["prices"].loc[usable],
            market["sigma"].loc[usable],
            market["sigma"].loc[usable] / float(np.sqrt(market["annualisation"])),
            market["adv"].loc[usable], market["volume"].loc[usable],
            market["funding"].loc[usable],
            dict(risk.clusters), risk_cfg=risk, impact=params,
            venue=VenueSpec(maker_fee_bps=cfg["exec"].maker_fee_bps,
                            taker_fee_bps=cfg["exec"].taker_fee_bps,
                            funding_cap_abs=0.02, min_notional=10.0,
                            latency_bars=1),
            spread=SpreadModel(half_spread_bps=cfg["exec"].assumed_half_spread_bps,
                               status=SpreadStatus.SPREAD_ASSUMED,
                               source="conf/execution/fees.yaml"),
            initial_equity=cfg["bt"].initial_equity,
            cost_per_side=cost.per_side,
            bars_per_year=cfg["bt"].bars_per_year))

    # Ruling P43(b): de per-bar reeksen staan NIET in het artefact maar wel hier
    # in `rows`. De overgangen worden daarom VOOR het samenvatten gerekend; de
    # ruwe reeksen zelf worden bewust niet weggeschreven.
    payload: dict = {}
    if args.with_transitions:
        payload["layer_transitions"] = {
            name: _transition_block(name, rows,
                                    bars_per_year=cfg["bt"].bars_per_year)
            for name in TRANSITIONS
        }
        for name, block in payload["layer_transitions"].items():
            print(f"{name:38s} d={block['delta_sharpe']:+.4f} "
                  f"se={block['se_ledoit_wolf']:.4f} "
                  f"CI=[{block['ci_95_block_bootstrap'][0]:+.4f}, "
                  f"{block['ci_95_block_bootstrap'][1]:+.4f}] "
                  f"t={block['t_stat']:+.3f} p={block['p_value']:.4f} "
                  f"n_eff={block['n_effective']}")

    frame = summarise(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"git_sha": current_git_sha(), "n_bars": int(len(usable)),
         "period": [str(usable[0].date()), str(usable[-1].date())],
         "rows": frame.to_dict(orient="records"), **payload},
        indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
