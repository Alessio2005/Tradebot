"""Markdown-rendering van de H1 QLIKE-competitie — deliverable 13.

Gescheiden van de meting in `validation/vol_campaign.py` om dezelfde reden als
overal in deze codebase: een module die meet én opmaakt, verleidt tot het
aanpassen van de meting omdat de tabel er anders beter uitziet.

Wat dit bestand WEL doet, is elke uitkomst met zijn beperking naast zich zetten.
Een QLIKE-getal zonder de proxy waartegen het is gemeten, een p-waarde zonder de
power van de toets en een promotie zonder de kostenlabels zijn alle drie
onvolledig op een manier die de lezer niet kan zien.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["PROXY_ASSUMPTIONS", "build_h1_payload", "render_competition_report"]

_STATUS_LABEL = {
    "PROMOTED": "**PROMOTED**",
    "FALSIFIED": "FALSIFIED",
    "UNPROVEN": "UNPROVEN",
    "DESCOPED": "DESCOPED",
}


def _missing(value: float | None) -> bool:
    """Ontbrekend of NaN. Een streepje in de tabel, nooit een nul.

    Een nul zou hier een MEETWAARDE suggereren op een plek waar niet is gemeten
    -- een p-waarde van 0 is iets heel anders dan een toets die niet is
    gedraaid.
    """
    return value is None or math.isnan(float(value))


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if _missing(value) else f"{float(value):.{digits}f}"


def _sci(value: float | None) -> str:
    return "—" if _missing(value) else f"{float(value):.3g}"


def _sorted_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(outcomes, key=lambda o: (o["symbol"], o["horizon"], o["spec"]))


def render_competition_report(payload: Mapping[str, Any]) -> str:
    """Bouw `reports/GARCH_VS_EWMA_COMPETITION.md` uit het campagne-artefact."""
    lines: list[str] = []
    add = lines.append

    campaign = payload["campaign"]
    outcomes = _sorted_outcomes(campaign["outcomes"])
    universe = payload["universe"]
    counts = campaign["status_counts"]

    _header(add, payload, campaign, universe)
    _verdict_summary(add, campaign, counts)
    _preregistration(add, payload)
    _proxy_section(add, payload)
    _premise_section(add, outcomes, campaign)
    _arch_section(add, payload)
    _convergence_section(add, outcomes)
    _degenerate_section(add, outcomes)
    _loss_section(add, outcomes, campaign)
    _dm_section(add, outcomes)
    _mz_section(add, outcomes)
    _controls_section(add, campaign)
    _robustness_section(add, outcomes, campaign)
    _ledger_section(add, payload, campaign)
    _limitations(add, payload)

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def _header(add: Any, payload: Mapping[str, Any], campaign: Mapping[str, Any],
            universe: Mapping[str, Any]) -> None:
    add("# H1 — DE GARCH-FAMILIE TEGEN EWMA(0.94) OP QLIKE")
    add("")
    add("> **Deliverable 13** · Phase 6 stap 7 · Phase 7/8 Stage C-2 en C-3")
    add(f"> **git_sha:** `{payload['git_sha']}`")
    add(f"> **Pre-registratie:** `{payload['preregistration_id']}` "
        f"(bevroren {payload['preregistration_frozen_utc']}, "
        f"M bij bevriezing = {payload['ledger_total_at_freeze']})")
    add(f"> **Universum:** {len(universe['symbols'])} gecertificeerde reeksen · "
        f"{universe['n_bars']} bars · {universe['period_start']} t/m "
        f"{universe['period_end']}")
    add(f"> **Walk-forward:** {payload['fold_geometry']['n_folds']} folds · "
        f"train {payload['fold_geometry']['train_bars_min']} · test "
        f"{payload['fold_geometry']['test_bars_min']} · embargo "
        f"{payload['embargo_bars']} bars · "
        f"{payload['fold_geometry']['total_oos_bars']} OOS-bars per symbool")
    add("")


def _verdict_summary(add: Any, campaign: Mapping[str, Any],
                     counts: Mapping[str, int]) -> None:
    add("## 0. De uitkomst, eerst")
    add("")
    blocked = sum(
        1 for outcome in campaign["outcomes"]
        if "proxy_premise_violated" in outcome["verdict"]["binding"])
    if campaign["n_promoted"] == 0:
        add("**Geen enkel lid van de GARCH-familie is gepromoveerd. "
            "EWMA(0.94) blijft de productie-estimator.**")
        if blocked:
            add("")
            add(f"Dat is geen nulresultaat maar een MEETLATRESULTAAT: bij "
                f"{blocked} van de {len(campaign['outcomes'])} combinaties "
                f"bindt `proxy_premise_violated`. De gepre-registreerde proxy "
                f"blijkt een andere grootheid te meten dan de modellen "
                f"voorspellen, en op zo'n meetlat valt niets te promoveren en "
                f"evenmin iets te falsifiëren. §2.3 meet dat, met het "
                f"mechanisme erbij.")
    else:
        add(f"**{campaign['n_promoted']} van de {len(campaign['outcomes'])} "
            f"combinaties haalde elke poort.** Zie §10 voor wat daarvoor nog "
            f"moet gebeuren voordat er iets in productie verandert.")
    add("")
    add("| Oordeel | Aantal | Betekenis |")
    add("|---|---|---|")
    add(f"| PROMOTED | {counts.get('PROMOTED', 0)} | lagere QLIKE dan "
        "EWMA(0.94), significant, gepowerd én geconvergeerd |")
    add(f"| FALSIFIED | {counts.get('FALSIFIED', 0)} | getoetst en geen "
        "verbetering gevonden terwijl de toets er wel een had kunnen zien |")
    add(f"| UNPROVEN | {counts.get('UNPROVEN', 0)} | niet vast te stellen: de "
        "meetlat is aantoonbaar verschoven (§2.3), of de toets kon het "
        "verwachte effect niet zien |")
    add(f"| DESCOPED | {counts.get('DESCOPED', 0)} | niet vergelijkbaar: "
        "ARCH-poort dicht, te weinig convergentie of te veel randoplossingen, "
        "of een gedegenereerde forecast (§4.1) |")
    add("")
    add("Welk criterium waar bond:")
    add("")
    add("| Stop-criterium | Combinaties |")
    add("|---|---|")
    binding_counts: dict[str, int] = {}
    for outcome in campaign["outcomes"]:
        for name in outcome["verdict"]["binding"]:
            binding_counts[name] = binding_counts.get(name, 0) + 1
    for name, count in sorted(binding_counts.items(), key=lambda kv: -kv[1]):
        add(f"| `{name}` | {count} |")
    add("")
    add("Het onderscheid tussen `FALSIFIED` en `UNPROVEN` is geen nuance maar "
        "een regel uit de pre-registratie: no-go 8 van de fase verbiedt een "
        "falsificatie-oordeel over een model dat simpelweg te weinig data had. "
        "Welke van de twee geldt, hangt af van het gemeten minimaal "
        "detecteerbare effect in §6 — een getal, niet een indruk.")
    add("")


def _preregistration(add: Any, payload: Mapping[str, Any]) -> None:
    add("## 1. Wat vooraf vastlag")
    add("")
    add("| | |")
    add("|---|---|")
    add("| Nulhypothese | de verwachte QLIKE van elk familielid is ≥ die van "
        "EWMA(0.94) |")
    add("| Primaire maat | `oos_qlike` (proxy-robuust, Patton 2011) |")
    add(f"| Toets | Diebold-Mariano met HLN-correctie, tweezijdig, "
        f"α = {payload['alpha']} |")
    add(f"| Geplande trials | {payload['planned_trials']} "
        f"(4 varianten × 6 symbolen × 2 horizonnen) |")
    add(f"| Verwacht effect | {payload['expected_effect_sd']} SD van de "
        f"per-bar verliesverschilreeks (Hansen & Lunde 2005, bovengrens) |")
    add("")
    add("De vier stop-criteria en hun volgorde staan in "
        "`conf/research/preregistration_h1_garch_vs_ewma.yaml` en zijn in code "
        "afgedwongen in `validation/vol_competition.py::judge_challenger`. De "
        "volgorde is bindend: een gesloten ARCH-poort betekent dat een "
        "GARCH-structuur op die reeks niet gerechtvaardigd is, en dan valt er "
        "niets te falsifiëren.")
    add("")


def _proxy_section(add: Any, payload: Mapping[str, Any]) -> None:
    add("## 2. Waartegen QLIKE is gemeten — en wat daaraan ontbreekt")
    add("")
    add("### 2.1 Er is geen realized variance. Dat is een meting, geen aanname.")
    add("")
    har = payload["har_rv_adequacy"]
    measured = har["measured"]
    add(f"De Data Adequacy Gate meet **{measured['coverage_pct']:.2f} %** van de "
        f"dagen met voldoende {measured['required_granularity']}-dekking tegen "
        f"een eis van {measured['required_coverage_pct']:.0f} % "
        f"({measured['n_days_with_coverage']} van {measured['n_days_total']} "
        f"dagen). De gecertificeerde store bevat "
        f"{measured['observed_granularity']}.")
    add("")
    add(f"**Oordeel HAR-RV: `{har['verdict']}`.** Niet `FALSIFIED` — no-go 8 "
        "van de fase verbiedt een falsificatie-oordeel over een model dat de "
        "Data Adequacy Gate niet haalde. Er is geen enkele HAR-RV-parameter "
        "geschat en geen enkele parameterruimte doorzocht; de trial telt "
        "daarom **niet** mee in `M`.")
    add("")
    add("Wat daarmee onbeslist blijft, expliciet: **HAR-RV als Level 3-uitdager "
        "is niet getoetst.** Niet afgewezen, niet aangenomen — niet getoetst. "
        "Het spoor gaat pas open met een intraday-bron, en dat is een "
        "inkoopbesluit en geen technische keuze.")
    add("")
    add("### 2.2 De competitie draait dus op een dagelijkse range-proxy")
    add("")
    add("QLIKE vergelijkt een variantieforecast met een PROXY voor de "
        "gerealiseerde variantie. Die proxy is hier een range-estimator op "
        "dagbars, geen realized variance uit intraday returns. Dat is "
        "legitiem — QLIKE is robuust tegen proxy-ruis zolang de proxy "
        "conditioneel zuiver is (Patton 2011) — maar de EFFICIENTIE ligt veel "
        "lager dan die van 5-minuts-RV, en lagere efficiëntie is direct minder "
        "power. **Dit is een beperking van de competitie zelf en niet van de "
        "modellen die eraan meedoen.**")
    add("")
    add("| Proxy | Rol hier | Relatieve efficiëntie | Aanname |")
    add("|---|---|---|---|")
    for name, record in sorted(payload["proxy_records"].items()):
        role = ("**primair**" if name == payload["campaign"]["primary_proxy"]
                else "robuustheid")
        add(f"| `{name}` | {role} | "
            f"{record['relative_efficiency_vs_squared_return']:.1f}× | "
            f"{payload['proxy_assumptions'][name]} |")
    add("")
    add(f"De primaire proxy is **`{payload['campaign']['primary_proxy']}`**, en "
        "die keuze is vóór de run gemaakt op één grond: Rogers-Satchell is de "
        "enige range-estimator in dit rijtje die zuiver blijft bij een "
        "DRIFT binnen de bar. Parkinson en Garman-Klass veronderstellen "
        "driftloosheid; op een reeks die in een jaar verdrievoudigt of "
        "halveert, is dat de aanname die het eerst breekt. De overige proxies "
        "draaien mee als robuustheidscontrole (§9) op EXACT dezelfde fits, "
        "zodat het verschil de proxy is en niet een tweede campagne.")
    add("")
    add("Wat alle vier gemeen hebben: zij meten de variatie BINNEN de dag uit "
        "vier prijzen, terwijl de modellen de variantie van de "
        "CLOSE-TO-CLOSE-return voorspellen — dat is de reeks waarop zij zijn "
        "gefit. Op een driftloze GBM vallen die twee grootheden samen, en op "
        "die gelijkheid rust de hele proxykeuze. De volgende paragraaf meet of "
        "zij op deze data standhoudt. Zij houdt geen stand.")
    add("")


def _premise_section(add, outcomes, campaign: Mapping[str, Any]) -> None:
    """De belangrijkste paragraaf van dit rapport, en hij stond er eerst niet in."""
    add("### 2.3 De premisse onder de proxy — gemeten, en geschonden")
    add("")
    add("De pre-registratie rechtvaardigt de range-estimator met een "
        "voorwaarde: *\"QLIKE is robuust tegen proxy-ruis zolang de proxy "
        "conditioneel zuiver is (Patton 2011), en de range-estimators zijn dat "
        "onder een driftloze GBM binnen de dag\"*. Die voorwaarde is een "
        "PREMISSE, en een premisse is meetbaar.")
    add("")
    add("Waarom zij ertoe doet: QLIKE heeft zijn minimum op "
        "``forecast = E[proxy]``. Draagt de proxy een multiplicatieve factor "
        "ten opzichte van de grootheid die de modellen voorspellen — de "
        "variantie van de **close-to-close return**, want daarop zijn zij "
        "gefit — dan verschuift dat minimum mee. De competitie rangschikt dan "
        "op *kalibratie tegen een verschoven doel* in plaats van op "
        "voorspelkwaliteit, en het model met het toevallig passende NIVEAU "
        "wint.")
    add("")
    add("De referentie is de gekwadrateerde return. Die is de ruisigste proxy "
        "die er is, en tegelijk de enige die per constructie zuiver is voor "
        "precies de voorspelde grootheid: ``E[r_t² | F_{t-1}] = σ_t²``. Ruis "
        "maakt een schatter niet scheef, en het is de scheefheid die de "
        "rangorde breekt.")
    add("")
    add("| Symbool | proxy / r² | EWMA-forecast / r² | GARCH-forecast / r² "
        "(bereik) |")
    add("|---|---|---|---|")
    per_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for outcome in outcomes:
        per_symbol.setdefault(outcome["symbol"], []).append(outcome)
    for symbol, rows in sorted(per_symbol.items()):
        scored = [r for r in rows if r["proxy_scale"] is not None]
        if not scored:
            continue
        levels = [
            r["forecast_scale"] for r in scored
            if not _missing(r["forecast_scale"])
        ]
        add(f"| {symbol} | **{scored[0]['proxy_scale']:.2f}×** | "
            f"{_fmt(scored[0]['baseline_forecast_scale'], 2)}× | "
            f"{min(levels):.2f}× – {max(levels):.2f}× |")
    add("")
    add("Twee dingen staan hier naast elkaar, en samen verklaren zij de "
        "uitslag van §6 volledig:")
    add("")
    add("1. **De primaire proxy meet een grotere grootheid dan de modellen "
        "voorspellen.** Niet marginaal: de factor loopt op tot boven de twee. "
        "Op deze data is de intraday-variatie stelselmatig groter dan de "
        "close-to-close-variantie — prijzen zwiepen binnen de dag heen en weer "
        "en komen terug. Dat is een eigenschap van crypto-dagbars, geen "
        "meetfout, maar het maakt de range-estimator ongeschikt als meetlat "
        "voor een model dat close-to-close voorspelt.")
    add("2. **EWMA(0.94) is nagenoeg perfect gekalibreerd op de "
        "close-to-close-variantie** (rond 1,0×), terwijl de GARCH-varianten er "
        "systematisch boven zitten. Op een proxy die zelf naar boven is "
        "verschoven, is dat een voordeel dat niets met voorspelkwaliteit te "
        "maken heeft.")
    add("")
    add("Daarom staat er in `validation/vol_competition.py::judge_challenger` "
        "een criterium dat NIET in de bevroren pre-registratie stond: "
        "`proxy_premise_violated`. Het is toegevoegd nadat de eerste run de "
        "premisse als geschonden mat, en dat is normaal gesproken precies de "
        "manoeuvre die pre-registratie uitsluit. Wat het hier toelaatbaar "
        "maakt, is de RICHTING: dit criterium kan een `PROMOTED` en een "
        "`FALSIFIED` allebei alleen omzetten in `UNPROVEN`. Het kan de "
        "conclusie uitsluitend voorzichtiger maken en nooit gunstiger — en het "
        "toetst een voorwaarde die de pre-registratie zelf uitspreekt.")
    add("")
    add("**Wat §6 hierdoor NIET zegt.** De DM-uitslagen daar zijn de "
        "gepre-registreerde meting op de gepre-registreerde meetlat, en zij "
        "staan er onverkort in. Zij zijn geen bewijs voor of tegen de "
        "GARCH-familie, omdat de meetlat aantoonbaar een ander doel meet dan "
        "het model voorspelt.")
    add("")


def _arch_section(add: Any, payload: Mapping[str, Any]) -> None:
    add("## 3. De ARCH-poort — vóór de eerste fit")
    add("")
    add("Stap 3 van de fase: *\"De ARCH-test is de poortwachter.\"* Is er geen "
        "aantoonbare conditionele heteroskedasticiteit, dan is een "
        "GARCH-structuur op die reeks niet gerechtvaardigd en wordt zij "
        "gedescopeerd — zonder fit, zodat er geen QLIKE-getal ontstaat dat los "
        "van dit oordeel kan gaan reizen.")
    add("")
    add("| Symbool | Engle-ARCH p | Poort |")
    add("|---|---|---|")
    for symbol, p_value in sorted(payload["arch_p_values"].items()):
        add(f"| {symbol} | {_sci(p_value)} | "
            f"{'open' if p_value < payload['alpha'] else '**dicht**'} |")
    add("")


def _convergence_section(add: Any, outcomes: Sequence[Mapping[str, Any]]) -> None:
    add("## 4. Convergentie en randoplossingen")
    add("")
    add("Niet-convergentie is een RESULTAAT en geen probleem dat wordt "
        "weggevangen: een mislukte fit levert geen forecast, en die bars "
        "blijven leeg in plaats van te worden gevuld met de waarde van de "
        "titelverdediger. Een randoplossing is even informatief: op "
        "`persistence ≥ 0,999` bestaat de onvoorwaardelijke variantie niet en "
        "is de forecast een random walk in variantie.")
    add("")
    add("| Symbool | Variant | Fits | Geconvergeerd | Op de rand | Vergelijkbaar |")
    add("|---|---|---|---|---|---|")
    seen: set[tuple[str, str]] = set()
    for outcome in outcomes:
        key = (outcome["symbol"], outcome["spec"])
        if key in seen or outcome["convergence"] is None:
            continue
        seen.add(key)
        conv = outcome["convergence"]
        comparable = outcome["verdict"]["binding"] != [
            "convergence_too_low_to_compare"]
        add(f"| {outcome['symbol']} | `{outcome['spec']}` | {conv['n_fits']} | "
            f"{conv['convergence_ratio']:.0%} | {conv['boundary_ratio']:.0%} | "
            f"{'ja' if comparable else '**nee**'} |")
    add("")


def _degenerate_section(add, outcomes: Sequence[Mapping[str, Any]]) -> None:
    """De uitschieters die QLIKE bijna verbergt."""
    degenerate = [
        o for o in outcomes
        if "degenerate_forecast_level" in o["verdict"]["binding"]
    ]
    add("### 4.1 Forecasts die ophielden forecasts te zijn")
    add("")
    if not degenerate:
        add("Geen enkele combinatie overschreed de grens; elke forecast bleef "
            "binnen een plausibel veelvoud van de gemiddelde variantie.")
        add("")
        return
    add("Op h > 1 bestaat er voor EGARCH en APARCH geen analytische "
        "meerstaps-forecast: hun recursie loopt in ``ln σ²`` respectievelijk "
        "``σ^δ``, en de terugtransformatie heeft geen gesloten vorm. De "
        "forecast wordt daarom gesimuleerd. Dat legt een eigenschap van het "
        "model bloot die de analytische route zou hebben verborgen: de "
        "verwachting van ``exp`` van een zwaarstaartige random walk hoeft niet "
        "te bestaan, en dan schat de simulatie een moment dat er niet is.")
    add("")
    add("| Symbool | h | Variant | hoogste forecast / r² |")
    add("|---|---|---|---|")
    for outcome in degenerate:
        add(f"| {outcome['symbol']} | {outcome['horizon']} | "
            f"`{outcome['spec']}` | **{outcome['forecast_level_ratio']:.3g}×** |")
    add("")
    add("Deze combinaties zijn GEDESCOPEERD en niet gefalsifieerd. Het model "
        "heeft daar geen bruikbaar getal geleverd; dat afrekenen als "
        "\"slechter dan EWMA\" zou een oordeel vellen over een meting die niet "
        "bestaat.")
    add("")
    add("Waarom dit een aparte poort verdient: QLIKE groeit slechts "
        "LOGARITMISCH in een overschatting. Een handvol bars met een forecast "
        "van 10²⁵ verschuift het gemiddelde verlies nauwelijks, dus het getal "
        "blijft er bruikbaar uitzien. Zonder deze controle zou zo'n reeks "
        "gewoon meedoen in de rangorde.")
    add("")


def _loss_section(add: Any, outcomes: Sequence[Mapping[str, Any]],
                  campaign: Mapping[str, Any]) -> None:
    add("## 5. De verliezen")
    add("")
    add(f"Gemeten op de primaire proxy `{campaign['primary_proxy']}`, "
        "uitsluitend op de testvensters van de walk-forward. Beide modellen "
        "zien dezelfde bars: EWMA heeft na zijn burn-in overal een waarde, "
        "maar wordt hier niet gescoord op de trainbars waarop zijn uitdager "
        "per constructie niet beoordeeld mag worden.")
    add("")
    add("| Symbool | h | Variant | QLIKE | QLIKE EWMA | MSE-SD | MAE-SD | "
        "Bars |")
    add("|---|---|---|---|---|---|---|---|")
    for outcome in outcomes:
        if not outcome["losses"]:
            continue
        losses = outcome["losses"]
        base = outcome["baseline_losses"]
        add(f"| {outcome['symbol']} | {outcome['horizon']} | "
            f"`{outcome['spec']}` | {_fmt(losses['qlike']['mean'])} | "
            f"{_fmt(base['qlike']['mean'])} | "
            f"{_sci(losses['mse_sd']['mean'])} | "
            f"{_sci(losses['mae_sd']['mean'])} | "
            f"{losses['qlike']['n_usable']} |")
    add("")
    add("Een LAGERE QLIKE is beter; nul is een perfecte forecast. Het verschil "
        "in puntschatting beslist niets — zie de volgende paragraaf.")
    add("")


def _dm_section(add: Any, outcomes: Sequence[Mapping[str, Any]]) -> None:
    add("## 6. Diebold-Mariano met HLN-correctie, en de gemeten power")
    add("")
    add("`mean diff` is ``QLIKE(variant) − QLIKE(EWMA)``: negatief betekent dat "
        "de uitdager beter was. De p-waarde is tweezijdig; een verwerping in "
        "het NADEEL van de uitdager promoveert niets.")
    add("")
    add("`MDE` is het minimaal detecteerbare effect van DEZE toets, in "
        "standaarddeviaties van de per-bar verliesverschilreeks, berekend met "
        "de GEMETEN AR(1) in plaats van met een vooraf gekozen scenario. Ligt "
        "hij boven het verwachte effect, dan kon deze opzet het effect niet "
        "zien en is een niet-significante uitkomst `UNPROVEN`.")
    add("")
    add("De kolom `proxy/r²` staat er met opzet naast: waar zij ver van 1,00 "
        "ligt, is de p-waarde ernaast een meting op een verschoven meetlat "
        "(§2.3) en draagt zij geen oordeel.")
    add("")
    add("| Symbool | h | Variant | mean diff | HLN | p | AR(1) | MDE (SD) | "
        "proxy/r² | Oordeel |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for outcome in outcomes:
        verdict = outcome["verdict"]
        dm = verdict["dm"]
        scale = _fmt(outcome["proxy_scale"], 2)
        if dm is None:
            add(f"| {outcome['symbol']} | {outcome['horizon']} | "
                f"`{outcome['spec']}` | — | — | — | — | — | {scale} | "
                f"{_STATUS_LABEL[verdict['status']]} |")
            continue
        add(f"| {outcome['symbol']} | {outcome['horizon']} | "
            f"`{outcome['spec']}` | {_sci(dm['mean_loss_differential'])} | "
            f"{_fmt(dm['hln_statistic'], 2)} | {_sci(dm['p_value'])} | "
            f"{_fmt(dm['loss_differential_ar1'], 3)} | "
            f"{_fmt(verdict['mde_sd_units'], 3)} | {scale} | "
            f"{_STATUS_LABEL[verdict['status']]} |")
    add("")


def _mz_section(add: Any, outcomes: Sequence[Mapping[str, Any]]) -> None:
    add("## 7. Mincer-Zarnowitz — is de forecast zuiver?")
    add("")
    add("``RV_t = α + β·σ²_t + e_t`` met HAC-standaardfouten. De gezamenlijke "
        "nulhypothese is ``(α, β) = (0, 1)``: een zuivere forecast. Verwerping "
        "betekent een systematische onder- of overschatting, en die kan naast "
        "een goede QLIKE bestaan.")
    add("")
    add("| Symbool | h | Model | α | β | Wald p | R² | Zuiver |")
    add("|---|---|---|---|---|---|---|---|")
    seen_baseline: set[tuple[str, int]] = set()
    for outcome in outcomes:
        for record, is_baseline in (
            (outcome["mincer_zarnowitz"], False),
            (outcome["baseline_mincer_zarnowitz"], True),
        ):
            if record is None:
                continue
            key = (outcome["symbol"], outcome["horizon"])
            if is_baseline:
                if key in seen_baseline:
                    continue
                seen_baseline.add(key)
            add(f"| {outcome['symbol']} | {outcome['horizon']} | "
                f"`{record['model']}` | {_sci(record['alpha'])} | "
                f"{_fmt(record['beta'], 3)} | {_sci(record['p_value'])} | "
                f"{_fmt(record['r_squared'], 3)} | "
                f"{'ja' if record['unbiased'] else 'nee'} |")
    add("")


def _controls_section(add: Any, campaign: Mapping[str, Any]) -> None:
    add("## 8. De negatieve controles — op de TOETS, niet op de markt")
    add("")
    add("Stap 7: *\"een toets die ook op ruis significant is, meet niets\"*. "
        "Die eis heeft twee kanten, en één ervan alleen is misleidend.")
    add("")
    add("**POWER.** EWMA tegen een geschudde variant van zichzelf: dezelfde "
        "waarden, alleen de timing weg. Ziet DM-HLN dát verschil niet, dan kan "
        "hij het veel kleinere GARCH-EWMA-verschil zeker niet zien en betekent "
        "geen enkele niet-significante uitslag in dit rapport iets.")
    add("")
    add("**SIZE.** Per bar wisselen welke van twee echte verliesreeksen bij "
        "welk model hoort. Het verwachte verschil is dan per constructie nul, "
        "terwijl schaal, staarten en seriële structuur blijven staan. Verwerpt "
        "de toets daar veel vaker dan α, dan verwerpt hij op ruis en is elke "
        "significante uitslag hierboven verdacht.")
    add("")
    add("| Reeks | h | Geschud: p | Geschud significant | Verwerping op ruis | "
        "Plafond | Geslaagd |")
    add("|---|---|---|---|---|---|---|")
    for key, control in sorted(campaign["controls"].items()):
        symbol = key.split("|")[0]
        shuffle = control["shuffle_control"]
        add(f"| {symbol} | {control['horizon']} | {_sci(shuffle['p_value'])} | "
            f"{'ja' if shuffle['significant'] else '**nee**'} | "
            f"{control['swap_rejection_rate']:.1%} | "
            f"{control['size_ceiling']:.0%} | "
            f"{'ja' if control['passed'] else '**nee**'} |")
    add("")
    add(f"Elke controle draait op {next(iter(campaign['controls'].values()))['n_replicates']} "
        "replicaties en op de TITELVERDEDIGER, niet op de winnaar: zou zij op "
        "het winnende model draaien, dan hing haar oordeel af van wie won.")
    add("")
    failed = [
        key for key, control in sorted(campaign["controls"].items())
        if not control["shuffle_control"]["significant"]
    ]
    sizes_ok = all(
        control["swap_rejection_rate"] <= control["size_ceiling"]
        for control in campaign["controls"].values())
    add("**De uitkomst, en zij is ongemakkelijk.** De SIZE-kant slaagt "
        f"{'overal' if sizes_ok else 'niet overal'}: de toets verwerpt niet op "
        "ruis. De POWER-kant faalt op "
        f"**{len(failed)} van de {len(campaign['controls'])}** "
        "reeks/horizon-combinaties — daar onderscheidt DM-HLN een forecast "
        "waarvan de timing volledig is vernietigd, NIET van het origineel.")
    add("")
    add("Dat weegt zwaarder dan het lijkt. De pre-registratie leidt haar power "
        "AF uit het aantal observaties en de AR(1), en die berekening zegt dat "
        "de toets gepowerd is (§6, kolom MDE). De controle MEET hetzelfde en "
        "spreekt dat tegen. Waar berekening en meting botsen, wint de meting: "
        "een toets die het grootst denkbare verschil niet ziet, ziet het veel "
        "kleinere GARCH-EWMA-verschil zeker niet.")
    add("")
    add("Het gevolg is afgedwongen en niet alleen opgeschreven: "
        "`judge_challenger` weigert een `FALSIFIED` op een reeks waar deze "
        "controle faalt, en geeft `UNPROVEN` — het criterium "
        "`underpowered_test_cannot_falsify` uit de pre-registratie, nu gevoed "
        "door een meting in plaats van door een afleiding.")
    add("")


def _robustness_section(add: Any, outcomes: Sequence[Mapping[str, Any]],
                        campaign: Mapping[str, Any]) -> None:
    add("## 9. Robuustheid over de proxies")
    add("")
    add("Dezelfde fits, dezelfde bars, een andere realisatie. QLIKE hoort "
        "proxy-robuust te zijn zolang de proxy conditioneel zuiver is; deze "
        "tabel maakt van die stelling een meting. Zij draagt geen oordeel — "
        "het verdict staat op de primaire proxy, zoals vooraf vastgelegd. Zou "
        "een tweede proxy het oordeel mogen kantelen, dan was de proxykeuze "
        "achteraf gemaakt.")
    add("")
    header = " | ".join(f"p ({name})" for name in campaign["robustness_proxies"])
    add(f"| Symbool | h | Variant | p ({campaign['primary_proxy']}) | {header} |")
    add("|---|---|---|---|" + "---|" * len(campaign["robustness_proxies"]))
    for outcome in outcomes:
        if not outcome["robustness"]:
            continue
        cells = [
            _sci(outcome["robustness"][name]["p_value"])
            if name in outcome["robustness"] else "—"
            for name in campaign["robustness_proxies"]
        ]
        primary = outcome["verdict"]["dm"]
        add(f"| {outcome['symbol']} | {outcome['horizon']} | "
            f"`{outcome['spec']}` | "
            f"{_sci(primary['p_value']) if primary else '—'} | "
            + " | ".join(cells) + " |")
    add("")


def _ledger_section(add: Any, payload: Mapping[str, Any],
                    campaign: Mapping[str, Any]) -> None:
    add("## 10. De ledger: M is niet gegroeid, en dat is de bedoeling")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| M vóór deze run | {payload['ledger_before']} |")
    add(f"| Trials geboekt bij het BEVRIEZEN | "
        f"{payload['trials_booked_at_freeze']} "
        f"(`{payload['ledger_unit']}`, `{payload['ledger_config_hash']}`) |")
    add(f"| Trials in deze run gefit | {campaign['n_trials']} |")
    add(f"| M ná deze run | {payload['ledger_after']} |")
    add("")
    add("De 48 trials stonden al in `M` vóór de eerste fit: zij zijn geboekt "
        "toen de pre-registratie werd bevroren. Dat is precies goed — wie een "
        "parameterruimte vastlegt, heeft die kansen genomen, en `M` hoort niet "
        "pas te groeien als de uitkomst bevalt.")
    add("")
    add("Daarom boekt deze run ze NIET opnieuw. Zou hij dat doen, dan ging "
        f"`M` van {payload['ledger_before']} naar "
        f"{payload['ledger_before'] + campaign['n_trials']} voor onderzoek dat "
        "één keer is gedaan. Ondertellen maakt elke deflated Sharpe ratio erna "
        "te gunstig, dubbeltellen maakt hem te streng — beide getallen zijn "
        "even onwaar. Het oordeel gaat daarom als AMENDEMENT de ledger in: "
        "`n_trials = 0`, met `amends` naar de entry hierboven.")
    add("")
    add(f"Geplande trials: {campaign['n_planned_trials']}. Feitelijk gefit: "
        f"{campaign['n_trials']}.")
    if campaign["n_trials"] == campaign["n_planned_trials"]:
        add("Die twee zijn gelijk: geen enkele poort heeft een fit voorkomen, "
            "dus elke geboekte combinatie heeft daadwerkelijk een kans gehad "
            "om iets te vinden. De boeking bij het bevriezen was dus exact.")
    else:
        add("Het verschil bestaat uit combinaties waarop de ARCH-poort dicht "
            "stond; daar is geen parameter geschat. Zij zijn bij het bevriezen "
            "wél geboekt, dus `M` staat in zoverre aan de STRENGE kant. Dat "
            "wordt niet gecorrigeerd: een boeking terugdraaien op grond van "
            "wat de data later bleek te zijn, is precies de beweging die "
            "pre-registratie uitsluit.")
    add("")
    add("EWMA(0.94) telt niet mee: lambda komt uit `conf/model/volatility.yaml` "
        "en is niet gevarieerd. De negatieve controles tellen evenmin mee: zij "
        "toetsen de toets, niet de markt.")
    add("")


def _limitations(add: Any, payload: Mapping[str, Any]) -> None:
    add("## 11. Wat dit rapport NIET vaststelt")
    add("")
    add("1. **Geen realized variance.** De competitie is beslecht tegen een "
        "dagelijkse range-proxy. Met 5-minuts-RV zou de toets meer power "
        "hebben en zou een klein effect zichtbaar kunnen worden dat hier "
        "onzichtbaar blijft. Zie §2.")
    add("2. **HAR-RV is niet getoetst.** `UNPROVEN — insufficient data`, met "
        "0,00 % 5m-dekking. Het model is niet afgewezen.")
    add("3. **H1 loopt vast op dezelfde ontbrekende data als HAR-RV.** Een "
        "geldige QLIKE-competitie vraagt een proxy die zuiver is voor de "
        "voorspelde grootheid. De gekwadrateerde return is dat wel maar is te "
        "ruisig om een effect van 0,10 SD te zien; de range-estimators zijn "
        "efficiënt maar meten hier aantoonbaar een grotere grootheid (§2.3). "
        "Intraday realized variance zou beide oplossen, en die is er niet. Het "
        "H1-spoor is daarmee niet beslist maar GEBLOKKEERD, en de blokkade is "
        "een inkoopbesluit over data — geen modelleerkeuze.")
    add("4. **Geen economisch oordeel.** QLIKE is een statistische maat. Een "
        "lagere QLIKE zegt niets over rendement na kosten; dat vergt een run "
        "door de authoritative engine met turnover- en fees-delta, en die "
        "staat hier niet in.")
    add("5. **Twee horizonnen, geen meer.** h = 1 en h = 5 bars. Een langere "
        "horizon toevoegen vereist een nieuwe pre-registratie en verhoogt "
        "`M` opnieuw.")
    add("6. **Per symbool getoetst, niet gepoold.** De pre-registratie toetst "
        "per symbool en per horizon. De power-analyse bij bevriezing rekende "
        "met "
        f"{payload['preregistered_effective_series']:.3f} effectief "
        "onafhankelijke reeksen voor een gepoolde toets; de toetsen hier "
        "staan elk op één reeks en hebben dus MINDER power dan die scenario's "
        "suggereren. Dat is de conservatieve kant: het levert eerder "
        "`UNPROVEN` dan `FALSIFIED` op.")
    add("")
    add("---")
    add("")
    add(f"*Gegenereerd door `apps/run_vol_competition.py` op git_sha "
        f"`{payload['git_sha']}`. Artefact: "
        f"`{payload['artefact_path']}`.*")


#: Waarop elke proxy steunt. Staat hier zodat de tabel in §2 de aanname noemt
#: naast het efficiëntiegetal: een efficiëntere proxy die een aanname breekt die
#: op deze data niet houdt, is niet beter maar zekerder verkeerd.
PROXY_ASSUMPTIONS: Mapping[str, str] = {
    "squared_return": "geen — zuiver, maar de ruisigste van de vier",
    "parkinson": "geen drift binnen de bar",
    "garman_klass": "geen drift binnen de bar; gebruikt open en close",
    "rogers_satchell": "**zuiver ook mét drift** binnen de bar",
    "yang_zhang": "venster-estimator; geen per-bar proxy",
}


def build_h1_payload(
    *,
    git_sha: str,
    universe: Mapping[str, Any],
    fold_geometry: Mapping[str, Any],
    embargo_bars: int,
    alpha: float,
    planned_trials: int,
    expected_effect_sd: float,
    preregistration: Mapping[str, Any],
    adequacy_artefact: Mapping[str, Any],
    econometrics_artefact: Mapping[str, Any],
    proxy_records: Mapping[str, Mapping[str, Any]],
    campaign: Mapping[str, Any],
    ledger_before: int,
    ledger_after: int,
    trials_booked_at_freeze: int,
    ledger_unit: str,
    ledger_config_hash: str,
    artefact_path: str,
) -> dict[str, Any]:
    """Zet het artefact in elkaar dat zowel op schijf gaat als het rapport voedt.

    Eén payload voor beide, zodat een getal in het rapport niet kan afwijken van
    het getal in het artefact. Dat is geen theoretisch risico: twee keer
    dezelfde grootheid opbouwen is precies hoe rapporten gaan afwijken van de
    run die zij beschrijven.
    """
    power = preregistration["content"]["parameters"]["power_analysis"]
    # De ARCH-poortoordelen en het HAR-RV-oordeel komen uit de artefacten die
    # VOOR deze run zijn geschreven, en worden hier niet opnieuw berekend. Een
    # tweede berekening zou stil kunnen afwijken van de poort die feitelijk is
    # toegepast, en dan beschrijft het rapport een andere run dan er draaide.
    arch_p_values = {
        symbol: record["log_return"]["tests"]["engle_arch"]["p_value"]
        for symbol, record in econometrics_artefact["symbols"].items()
    }
    return {
        "git_sha": git_sha,
        "preregistration_id": preregistration["preregistration_id"],
        "preregistration_frozen_utc": preregistration["frozen_utc"],
        "ledger_total_at_freeze": preregistration["ledger_total_at_freeze"],
        "universe": dict(universe),
        "fold_geometry": dict(fold_geometry),
        "embargo_bars": int(embargo_bars),
        "alpha": alpha,
        "planned_trials": planned_trials,
        "expected_effect_sd": expected_effect_sd,
        "preregistered_effective_series": float(
            power["effective_independent_series"]),
        "har_rv_adequacy": dict(adequacy_artefact["verdicts"]["har_rv"]),
        "arch_p_values": dict(arch_p_values),
        "proxy_records": {k: dict(v) for k, v in proxy_records.items()},
        "proxy_assumptions": dict(PROXY_ASSUMPTIONS),
        "campaign": dict(campaign),
        "ledger_before": int(ledger_before),
        "ledger_after": int(ledger_after),
        "trials_booked_at_freeze": int(trials_booked_at_freeze),
        "ledger_unit": ledger_unit,
        "ledger_config_hash": ledger_config_hash,
        "artefact_path": artefact_path,
    }
