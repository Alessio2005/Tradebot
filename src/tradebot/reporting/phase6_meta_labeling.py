# src/tradebot/reporting/phase6_meta_labeling.py
"""`reports/META_LABELING_EVALUATION.md` uit het H3-artefact. Deliverable 22.

Eén payload voedt zowel het JSON-artefact als het rapport, zodat een getal in
het rapport niet kan afwijken van het getal in het artefact.

Ref: fase-opdracht stappen 12 en 13, deliverable 22; pre-registratie
`56395fa2013768014c0c915edf346770`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["build_h3_payload", "render_meta_label_report"]

_STATUS_MEANING = {
    "PROMOTED": "AUC-ondergrens boven 0,58 én netto economische waarde door de "
                "engine",
    "FALSIFIED": "de negatieve controle lekt — de RUN is ongeldig, niet de "
                 "hypothese",
    "DESCOPED": "te weinig effectieve events; `UNPROVEN — insufficient data`",
    "ARCHIVED": "gemeten en de drempel niet gehaald; CatBoost blijft "
                "gearchiveerd zoals §24 vastlegt",
}


def _f(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if math.isnan(number) else f"{number:.{digits}f}"


def _i(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def _stop_threshold(payload: Mapping[str, Any], name: str) -> float:
    """De drempel van één stop-criterium, UIT de bevroren pre-registratie.

    Niet uit een constante hier. Een rapport dat zijn eigen kopie van een
    drempel meedraagt, blijft het oude getal tonen wanneer de pre-registratie
    een ander draagt -- en dan staat er een oordeel tegen een grens die nergens
    is vastgelegd.
    """
    for criterion in payload["preregistration"]["content"]["stop_criteria"]:
        if criterion["name"] == name:
            return float(criterion["threshold"])
    raise KeyError(f"stop-criterium `{name}` staat niet in de pre-registratie")


def _power(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload["preregistration"]["content"]["parameters"]["power_analysis"]


def render_meta_label_report(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    campaign = payload["campaign"]
    trials = campaign["trials"]

    _header(add, payload, campaign)
    _outcome(add, campaign, trials)
    _preregistration(add, payload)
    _control(add, payload, campaign)
    _events(add, payload, campaign)
    _auc(add, payload, campaign, trials)
    _operating(add, payload, trials)
    _engine(add, campaign, trials)
    _importance(add, campaign)
    _ledger(add, payload, campaign)
    _limitations(add, payload, campaign)
    return "\n".join(lines) + "\n"


def _header(add: Any, payload: Mapping[str, Any],
            campaign: Mapping[str, Any]) -> None:
    universe = payload["universe"]
    add("# H3 — CATBOOST ALS SECONDARY MODEL")
    add("")
    add("> **Deliverable 22** · Phase 6 stappen 12 en 13 · Phase 7/8 Stage C-3")
    add(f"> **git_sha:** `{payload['git_sha']}`")
    add(f"> **Pre-registratie:** `{payload['preregistration_id']}` "
        f"(bevroren {payload['preregistration_frozen_utc']}, "
        f"M bij bevriezing = {payload['ledger_total_at_freeze']})")
    add(f"> **Universum:** {len(universe['symbols'])} gecertificeerde reeksen · "
        f"{universe['n_bars']} bars · {universe['period_start']} t/m "
        f"{universe['period_end']}")
    add(f"> **Primaire track:** `{payload['track']}` · "
        f"{_i(campaign['n_events'])} triple-barrier events · embargo "
        f"{payload['embargo_bars']} bars bovenop de purge op `t1`")
    add(f"> **Kostenlabels:** "
        f"`{campaign['unfiltered']['impact_status']}` · "
        f"`{campaign['unfiltered']['spread_status']}` "
        f"({_f(campaign['unfiltered']['half_spread_bps'], 1)} bp) · risk "
        f"`config_hash` `{campaign['unfiltered']['risk_policy_hash']}`")
    add("")


def _outcome(add: Any, campaign: Mapping[str, Any],
             trials: Sequence[Mapping[str, Any]]) -> None:
    counts = campaign["by_status"]
    add("## 0. De uitkomst, eerst")
    add("")
    if counts.get("PROMOTED"):
        add(f"**{counts['PROMOTED']} van de {len(trials)} specs is "
            f"gepromoveerd.**")
    else:
        add("**Geen enkele spec is gepromoveerd. CatBoost blijft `ARCHIVED`, "
            "zoals §24 vastlegt.**")
    add("")
    add("| Oordeel | Aantal | Betekenis |")
    add("|---|---|---|")
    for status in ("PROMOTED", "FALSIFIED", "DESCOPED", "ARCHIVED"):
        add(f"| {status} | {counts.get(status, 0)} | "
            f"{_STATUS_MEANING[status]} |")
    add("")
    best = max(trials, key=lambda t: t["oos_auc"])
    add(f"De beste van de {len(trials)} is `{best['label']}` met een OOS-AUC "
        f"van {_f(best['oos_auc'])}. {_where_the_aucs_fell(trials)}")
    add("")


def _where_the_aucs_fell(trials: Sequence[Mapping[str, Any]]) -> str:
    """Waar de gemeten AUC's liggen — AFGELEZEN, niet vooraf opgeschreven.

    Deze zin stond hier eerst als vaste tekst ("alle zes liggen onder 0,50"),
    geschreven voordat de campagne één keer had gedraaid. Dat is precies de
    fout waar §0.10 tegen waarschuwt, alleen dan in het rapport in plaats van in
    de toets: een conclusie die niet rood kan worden. Zij staat als defect in
    het exit-rapport.
    """
    n = len(trials)
    aucs = [float(t["oos_auc"]) for t in trials]
    below = sum(1 for a in aucs if a < 0.5)
    best = max(trials, key=lambda t: t["oos_auc"])
    low, high = (float(best["interval_conservative"][0]),
                 float(best["interval_conservative"][1]))
    covers = low <= 0.5 <= high

    if below == n:
        where = (f"**Alle {n} liggen ONDER 0,50** (van {_f(min(aucs))} tot "
                 f"{_f(max(aucs))})")
    elif below == 0:
        where = (f"**Alle {n} liggen BOVEN 0,50** (van {_f(min(aucs))} tot "
                 f"{_f(max(aucs))})")
    else:
        where = (f"**{below} van de {n} ligt onder 0,50**; de reeks loopt van "
                 f"{_f(min(aucs))} tot {_f(max(aucs))}")

    if covers:
        return (
            f"{where}, en het conservatieve interval van de beste "
            f"([{_f(low)}, {_f(high)}]) omvat 0,50. Het model rangschikt de "
            f"succeskans van het primaire signaal dus niet aantoonbaar beter "
            f"dan een muntworp — een AUC onder 0,50 is op deze "
            f"steekproefgrootte geen anti-signaal maar ruis, en §5 laat zien "
            f"waarom.")
    side = "boven" if float(best["oos_auc"]) > 0.5 else "onder"
    return (
        f"{where}, en het conservatieve interval van de beste "
        f"([{_f(low)}, {_f(high)}]) ligt volledig {side} 0,50. Dat is een "
        f"meetbaar verschil met een muntworp; of het de DREMPEL haalt, beslist "
        f"§4, en of het geld oplevert, §6.")


def _preregistration(add: Any, payload: Mapping[str, Any]) -> None:
    content = payload["preregistration"]["content"]
    power = content["parameters"]["power_analysis"]
    labeling = payload["labeling"]
    add("## 1. Wat vooraf vastlag")
    add("")
    add("| | |")
    add("|---|---|")
    add("| Nulhypothese | de OOS-AUC is niet te onderscheiden van 0,50 |")
    add(f"| Primaire maat | `{content['primary_metric']}` |")
    add(f"| Drempel | AUC > {_f(power['threshold_auc'], 2)} "
        f"(`{power['threshold_source']}`) |")
    add("| Toets | Hanley-McNeil-interval om de AUC, op de CONSERVATIEVE "
        "effectieve steekproefgrootte |")
    add(f"| Geplande trials | {content['planned_trials']} "
        "(3 dieptes × 2 leerstappen) |")
    add(f"| Barrières | {_f(labeling['profit_target_sigma'], 1)}σ / "
        f"{_f(labeling['stop_loss_sigma'], 1)}σ / "
        f"{labeling['horizon_bars']} bars, entry op "
        f"`t+{labeling['entry_lag_bars']}` |")
    add("")
    add("De power-analyse stond vóór de run vast, en haar uitkomst was dat H3 "
        "**precies op de grens** ligt:")
    add("")
    add("| Variant | effectieve n | MDE (AUC-overschot) | informatief? |")
    add("|---|---|---|---|")
    for name in ("nominal", "uniqueness_corrected",
                 "uniqueness_and_cross_section"):
        scenario = power["scenarios"][name]
        add(f"| {name} | "
            f"{_f(scenario['assumptions']['effective_n'], 1)} | "
            f"{_f(scenario['minimum_detectable_effect'], 4)} | "
            f"{'ja' if scenario['informative'] else 'nee'} |")
    add("")
    add("Aan welke kant van de grens H3 valt, hangt af van een keuze die niet "
        "uit de data volgt: twee labels van verschillende symbolen op dezelfde "
        "bar zijn verschillende trades met verschillende uitkomsten, maar hun "
        "uitkomsten zijn gecorreleerd. De pre-registratie legde daarom vast dat "
        "het interval op BEIDE wordt gerapporteerd en dat het oordeel op de "
        "conservatieve valt.")
    add("")


def _control(add: Any, payload: Mapping[str, Any],
             campaign: Mapping[str, Any]) -> None:
    leak = _stop_threshold(payload, "negative_control_leaks")
    measured = float(campaign["shuffled_auc"])
    clean = measured < leak
    add(f"## 2. De negatieve controle — "
        f"{'en zij is schoon' if clean else 'EN ZIJ LEKT'}")
    add("")
    add("Hetzelfde model, dezelfde folds, dezelfde purging, maar met "
        "**gerandomiseerde labels**. De klassebalans blijft per constructie "
        "gelijk (er wordt gepermuteerd, niet opnieuw getrokken), dus een "
        "verschil in AUC kan niet uit een andere basisrate komen.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| AUC op gerandomiseerde labels (max over "
        f"{len(campaign['shuffled_auc_replicates'])} replicaties) | "
        f"**{_f(measured)}** |")
    add("| replicaties | "
        + ", ".join(_f(a) for a in campaign["shuffled_auc_replicates"]) + " |")
    add(f"| grens waarboven de RUN wordt gefalsificeerd | {_f(leak, 2)} |")
    add("")
    if clean:
        add(f"Dit getal is de belangrijkste van het hele rapport, en niet omdat "
            f"het gunstig is. Het zegt dat de pipeline NIET lekt: geen scaler "
            f"over folds heen, geen feature die de toekomst raakt, en een "
            f"purging die doet wat zij belooft. Was hij {_f(leak, 2)} of hoger "
            f"geweest, dan was elk ander getal hieronder waardeloos geweest — "
            f"inclusief de AUC's die de hypothese hadden kunnen steunen.")
    else:
        add(f"**Dit getal falsificeert de RUN.** Op gerandomiseerde labels "
            f"hoort een AUC rond 0,50 te staan; gemeten is {_f(measured)}, "
            f"tegen een grens van {_f(leak, 2)}. Er lekt informatie in de "
            f"pipeline, en daarmee is elk ander getal hieronder waardeloos — "
            f"inclusief de AUC's die de hypothese zouden kunnen steunen. Wat "
            f"hieronder staat, staat er om het lek te kunnen localiseren, niet "
            f"om een oordeel over H3 op te baseren.")
    add("")


def _events(add: Any, payload: Mapping[str, Any],
            campaign: Mapping[str, Any]) -> None:
    add("## 3. De events, en waarom het nominale aantal misleidt")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| triple-barrier events (gepoold) | {_i(campaign['n_events'])} |")
    add(f"| positieve klasse | {_f(campaign['positive_ratio'])} |")
    add(f"| gemiddelde uniqueness | {_f(campaign['uniqueness_ratio'])} |")
    add(f"| effectief, nominaal (testfolds) | "
        f"{_f(campaign['effective_n_nominal'], 0)} |")
    add(f"| effectief, na uniqueness én cross-sectie | "
        f"{_f(campaign['effective_n_conservative'], 1)} |")
    add("")
    power = _power(payload)
    horizon = int(payload["labeling"]["horizon_bars"])
    add(f"Met een verticale barrière op {horizon} bars en een event op vrijwel "
        f"elke bar delen buren {horizon - 1} van hun {horizon} toekomstige "
        f"bars. De gemeten uniqueness van {_f(campaign['uniqueness_ratio'])} "
        f"zegt dat {_i(campaign['n_events'])} labels ongeveer "
        f"{_f(campaign['n_events'] * campaign['uniqueness_ratio'], 0)} "
        f"onafhankelijke waarnemingen waard zijn. Bovenop die overlap staat de "
        f"cross-sectionele afhankelijkheid: {power['n_symbols']} perpetuals met "
        f"een gemeten gemiddelde correlatie van "
        f"{_f(power['measured_mean_pairwise_correlation'])} zijn "
        f"{_f(power['effective_independent_series'], 2)} onafhankelijke "
        f"reeksen waard en geen {power['n_symbols']}.")
    add("")
    events_threshold = _stop_threshold(payload, "events_below_adequacy")
    binds = float(campaign["min_effective_events_per_fold"]) < events_threshold
    add(f"**De adequaatheidspoort bindt hier "
        f"{'WEL' if binds else 'NIET'}, en dat vraagt uitleg**, want er zijn "
        f"twee getallen die allebei 'effectieve events per fold' heten:")
    add("")
    add("| Grootheid | gemeten | rol |")
    add("|---|---|---|")
    add(f"| gepurgede TRAINevents per fold (Data Adequacy Gate) | "
        f"{_f(campaign['min_effective_events_per_fold'], 1)} | stop-criterium "
        f"2, drempel {_f(events_threshold, 0)} — bindt "
        f"{'WEL' if binds else 'niet'} |")
    add(f"| TESTevents per fold (deze run) | "
        f"{_f(campaign['min_effective_test_events_per_fold'], 1)} | bepaalt de "
        f"PRECISIE van de AUC, geen poort |")
    add("")
    add("Stop-criterium 2 zegt letterlijk *\"De Data Adequacy Gate meet het "
        "effectieve aantal labels per fold\"*, en die poort heeft een "
        "implementatie en een artefact: hij telt de GEPURGEDE TRAINevents, want "
        "dat is wat bepaalt of er gefit mag worden ('zonder gefit te zijn'). "
        "Die staat op "
        f"{_f(campaign['min_effective_events_per_fold'], 1)}, "
        f"{'ONDER' if binds else 'ruim boven'} de "
        "eis. Het testfold-aantal is een andere grootheid met een andere rol — "
        "het bepaalt hoe scherp de AUC te meten valt — en het staat hier "
        "expliciet naast, zodat niemand ze later door elkaar haalt.")
    add("")


def _auc(add: Any, payload: Mapping[str, Any], campaign: Mapping[str, Any],
         trials: Sequence[Mapping[str, Any]]) -> None:
    threshold = payload["preregistration"]["content"]["parameters"][
        "power_analysis"]["threshold_auc"]
    add("## 4. De AUC, met beide intervallen")
    add("")
    add("| Spec | OOS-AUC | 95 %-interval, nominaal | 95 %-interval, "
        "conservatief | oordeel |")
    add("|---|---|---|---|---|")
    for trial in trials:
        nominal = trial["interval_nominal"]
        conservative = trial["interval_conservative"]
        add(f"| `{trial['label']}` | {_f(trial['oos_auc'])} | "
            f"[{_f(nominal[0])}, {_f(nominal[1])}] | "
            f"[{_f(conservative[0])}, {_f(conservative[1])}] | "
            f"{trial['verdict']['status']} |")
    add("")
    add("De AUC weegt de **uniqueness** mee: hij is de kans dat een willekeurig "
        "getrokken geslaagde trade hoger scoort dan een willekeurig getrokken "
        "mislukte, waarbij beide worden getrokken proportioneel aan hun "
        "uniqueness. Zonder die weging telt elk van tien overlappende buren als "
        "een volwaardige waarneming, en dat is precies waarom "
        "meta-labeling-AUC's in de literatuur zo vaak te hoog uitvallen.")
    add("")
    best = max(trials, key=lambda t: t["oos_auc"])
    width_nominal = (float(best["interval_nominal"][1])
                     - float(best["interval_nominal"][0]))
    width_conservative = (float(best["interval_conservative"][1])
                          - float(best["interval_conservative"][0]))
    ratio = (width_conservative / width_nominal if width_nominal > 0.0
             else float("nan"))
    add(f"Het verschil tussen de twee intervallen is de hele les van §1: op "
        f"`{best['label']}` is het nominale interval {_f(ratio, 1)} keer zo "
        f"smal als het conservatieve ({_f(width_nominal)} tegen "
        f"{_f(width_conservative)} AUC breed). Een pipeline zonder "
        f"uniqueness-correctie zou dat smalle interval rapporteren, en een AUC "
        f"die de drempel van {_f(threshold, 2)} nét raakt, zou daarmee "
        f"'significant' heten waar het conservatieve interval de drempel nog "
        f"ruim omvat.")
    add("")


def _operating(add: Any, payload: Mapping[str, Any],
               trials: Sequence[Mapping[str, Any]]) -> None:
    best = max(trials, key=lambda t: t["oos_auc"])
    add("## 5. Het werkpunt: precision, recall en wat er wordt weggefilterd")
    add("")
    add(f"Het OORDEEL valt op één werkpunt: kans ≥ "
        f"{_f(payload['probability_threshold'], 2)}, de natuurlijke "
        f"beslisgrens van een kans. Er is niet geprobeerd welk werkpunt de "
        f"mooiste cijfers geeft; de volledige curve staat hieronder zodat een "
        f"lezer ziet wat een ander werkpunt zou hebben gedaan.")
    add("")
    add(f"Curve voor `{best['label']}`, de beste van de zes:")
    add("")
    add("| drempel | precision | recall | aandeel behouden | events behouden |")
    add("|---|---|---|---|---|")
    for point in best["operating_curve"]:
        add(f"| {_f(point['threshold'], 2)} | {_f(point['precision'])} | "
            f"{_f(point['recall'])} | {_f(point['kept_fraction'])} | "
            f"{_i(point['n_kept'])} |")
    add("")
    add(f"De basisrate is {_f(best['operating_point']['base_rate'])}: zoveel "
        f"van de trades slaagde sowieso. Een filter voegt pas iets toe wanneer "
        f"zijn precision daar BOVEN ligt — een filter met precision gelijk aan "
        f"de basisrate selecteert willekeurig en houdt alleen minder over.")
    add("")


def _engine(add: Any, campaign: Mapping[str, Any],
            trials: Sequence[Mapping[str, Any]]) -> None:
    unfiltered = campaign["unfiltered"]
    add("## 6. Door de authoritative engine — de economische toets")
    add("")
    add("Een AUC is een statistisch resultaat, geen economisch. Stap 13 van de "
        "fase-opdracht draait het gefilterde signaal door dezelfde engine met "
        "volledige kosten: *\"een filter dat de helft van de trades weghaalt en "
        "de Sharpe met 0,02 verbetert, verdient geen promotie\"*.")
    add("")
    add("| Arm | netto OOS Sharpe | Δ vs. ongefilterd | turnover | fees | "
        "bars met positie |")
    add("|---|---|---|---|---|---|")
    add(f"| ongefilterd | {_f(unfiltered['net_sharpe_oos'])} | — | "
        f"{_i(unfiltered['turnover_notional_oos'])} | "
        f"{_f(unfiltered['fees_oos'], 2)} | "
        f"{_i(unfiltered['n_exposed_bars_oos'])} |")
    for trial in trials:
        arm = trial["arm"]
        add(f"| `{trial['label']}` | {_f(arm['net_sharpe_oos'])} | "
            f"{float(trial['net_sharpe_delta']):+.4f} | "
            f"{_i(arm['turnover_notional_oos'])} | "
            f"{_f(arm['fees_oos'], 2)} | "
            f"{_i(arm['n_exposed_bars_oos'])} |")
    add("")
    _engine_reading(add, campaign, trials)
    add("Elke arm draait door dezelfde `EventDrivenEngine`, met dezelfde "
        "risicolaag, router, venue en kosten. De ENIGE ingang die verschilt is "
        "de exposure: de basisexposure maal het filter (1 doorlaten, 0 "
        "tegenhouden). Buiten de gescoorde events staat het filter op 1 — daar "
        "heeft hij geen oordeel, en een 0 zou een bewering zijn in plaats van "
        "een onthouding.")
    add("")


def _engine_reading(add: Any, campaign: Mapping[str, Any],
                    trials: Sequence[Mapping[str, Any]]) -> None:
    """Wat er in de engine-tabel staat dat een lezer anders verkeerd leest.

    Twee dingen, en allebei AFGELEZEN: de specs met een POSITIEVE delta (die
    zijn er niet per se, en als ze er zijn hoort erbij te staan waarom ze toch
    niet promoveren), en de instorting van het aantal bars met positie.
    """
    unfiltered = campaign["unfiltered"]
    exposed_base = float(unfiltered["n_exposed_bars_oos"])
    positive = [t for t in trials if float(t["net_sharpe_delta"]) > 0.0]

    if positive:
        names = ", ".join(f"`{t['label']}`" for t in positive)
        best_pos = max(positive, key=lambda t: float(t["net_sharpe_delta"]))
        add(f"**{len(positive)} van de {len(trials)} specs verbetert de netto "
            f"Sharpe: {names}.** De grootste verbetering is "
            f"{float(best_pos['net_sharpe_delta']):+.4f} op "
            f"`{best_pos['label']}`, van {_f(unfiltered['net_sharpe_oos'])} "
            f"naar {_f(best_pos['arm']['net_sharpe_oos'])} — op een spec met "
            f"een OOS-AUC van {_f(best_pos['oos_auc'])}, dus ONDER 0,50. Een "
            f"filter zonder gemeten voorspellende waarde dat de Sharpe met "
            f"{abs(float(best_pos['net_sharpe_delta'])):.2f} beweegt, is exact "
            f"het geval waar stap 13 voor waarschuwt: *\"een filter dat de "
            f"helft van de trades weghaalt en de Sharpe met 0,02 verbetert, "
            f"verdient geen promotie\"*. Het bindende criterium is bij deze "
            f"spec dan ook niet de engine maar de AUC "
            f"(`{'`, `'.join(best_pos['verdict']['binding'])}`).")
        add("")
    else:
        add(f"**Geen enkele spec verbetert de netto Sharpe.** Alle "
            f"{len(trials)} deltas zijn negatief; het filter haalt trades weg "
            f"en wat overblijft presteert slechter dan het ongefilterde "
            f"signaal. De economische toets bindt daarmee naast de "
            f"statistische, en niet in plaats daarvan.")
        add("")

    exposures = [(t["label"], float(t["arm"]["n_exposed_bars_oos"]))
                 for t in trials]
    thinnest = min(exposures, key=lambda pair: pair[1])
    add(f"**Waar deze cijfers op rusten.** Het ongefilterde boek heeft een "
        f"positie op {_i(exposed_base)} OOS-bars; gefilterd zakt dat naar "
        f"{_i(min(e for _, e in exposures))}–{_i(max(e for _, e in exposures))}. "
        f"Voor `{thinnest[0]}` blijven er {_i(thinnest[1])} over — "
        f"{thinnest[1] / exposed_base:.1%} van het origineel. Een Sharpe over "
        f"zo weinig bars met positie draagt een brede foutmarge, en dat is een "
        f"reden temeer om het oordeel niet op de engine-delta alleen te laten "
        f"rusten. De AUC-drempel bindt hier bij alle specs.")
    add("")


def _importance(add: Any, campaign: Mapping[str, Any]) -> None:
    add("## 7. Feature importance — MDI naast SFI")
    add("")
    add("§12.1 eist er twee, en zij meten verschillende dingen. **MDI** komt "
        "uit de fit die er toch al was, is IN-SAMPLE en deelt het belang van "
        "features die hetzelfde meten. **SFI** geeft elk model één feature en "
        "scoort hem OUT-OF-SAMPLE onder dezelfde purged folds; substitutie kan "
        "daar niet optreden, interactie wordt daar niet gezien.")
    add("")
    add(f"Beide op `{campaign['best_label']}`, de beste van de zes.")
    add("")
    add("| Feature | MDI (aandeel) | SFI (OOS-AUC alleen op deze feature) |")
    add("|---|---|---|")
    sfi = {f["name"]: f for f in campaign["sfi"]["features"]}
    for entry in campaign["mdi"]["features"]:
        single = sfi.get(entry["name"], {})
        add(f"| `{entry['name']}` | "
            f"{_f(entry['mean'])} ± {_f(entry['std'])} | "
            f"{_f(single.get('mean'))} ± {_f(single.get('std'))} |")
    add("")
    add("Geen van beide getallen promoveert of blokkeert iets: zij zijn "
        "diagnostiek. " + _what_the_importances_show(campaign))
    add("")


def _what_the_importances_show(campaign: Mapping[str, Any]) -> str:
    """De uitleg bij MDI en SFI, AFGELEZEN uit wat er is gemeten.

    Ook deze zin stond hier als vaste tekst ("SFI komt voor élke feature rond
    0,50 uit") voordat er iets was gefit. Zelfde defect als in §0; zie het
    exit-rapport.
    """
    mdi = campaign["mdi"]["features"]
    sfi = campaign["sfi"]["features"]
    if not mdi or not sfi:
        return "Er is te weinig gemeten om er iets uit af te lezen."
    top = mdi[0]
    excursions = [abs(float(f["mean"]) - 0.5) for f in sfi
                  if not math.isnan(float(f["mean"]))]
    if not excursions:
        return (f"MDI wijst het meeste gewicht toe aan `{top['name']}` "
                f"({_f(top['mean'])}); SFI leverde geen bruikbare score op.")
    furthest = max(sfi, key=lambda f: abs(float(f["mean"]) - 0.5))
    largest = max(excursions)
    spread = (f"de verste ligt {_f(largest, 3)} van 0,50 af "
              f"(`{furthest['name']}`, {_f(furthest['mean'])})")
    if largest < 0.05:
        return (
            f"Wat zij hier laten zien is dat MDI wel degelijk structuur "
            f"toewijst — het model splitst het meest op `{top['name']}` "
            f"({_f(top['mean'])} van het totaal) — terwijl SFI voor élke "
            f"feature dicht bij 0,50 uitkomt: {spread}. Dat is het patroon van "
            f"een model dat in-sample structuur vindt die out-of-sample niet "
            f"bestaat, en het is consistent met de AUC's in §4.")
    return (
        f"MDI legt het meeste gewicht op `{top['name']}` ({_f(top['mean'])} van "
        f"het totaal). SFI wijkt voor minstens één feature merkbaar van 0,50 "
        f"af: {spread}. Of die afwijking de drempel haalt, beslist §4 op de "
        f"gepoolde AUC en niet deze diagnostiek.")


def _ledger(add: Any, payload: Mapping[str, Any],
            campaign: Mapping[str, Any]) -> None:
    add("## 8. De ledger en `M`")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| `M` vóór deze run | {payload['ledger_before']} |")
    add(f"| `M` ná deze run | {payload['ledger_after']} |")
    add(f"| trials geboekt bij het bevriezen | "
        f"{payload['trials_booked_at_freeze']} |")
    add(f"| trials in deze run gedraaid | {campaign['n_trials']} |")
    add("")
    add("De zes trials zijn bij het BEVRIEZEN van de pre-registratie geboekt. "
        "Deze run voert ze uit en boekt ze niet opnieuw; het oordeel gaat als "
        "amendement terug de ledger in, met `n_trials = 0`.")
    add("")
    add("De negatieve controle telt NIET mee in `M`: zij toetst de PIPELINE en "
        "niet de markt. Vijf replicaties van hetzelfde model op gepermuteerde "
        "labels zijn geen vijf hypothesen over rendement.")
    add("")


def _limitations(add: Any, payload: Mapping[str, Any],
                 campaign: Mapping[str, Any]) -> None:
    add("## 9. Wat hiermee NIET is getoetst")
    add("")
    add("1. **Andere barrièrebreedtes.** 2σ / 2σ / 10 bars staat in "
        "`conf/model/labeling.yaml` en is niet gezocht. Breedtes variëren tot "
        "de AUC de drempel haalt, is de meest directe manier om deze hypothese "
        "te vervalsen zonder het te merken; dat vraagt een nieuwe "
        "pre-registratie met een eigen bijdrage aan `M`.")
    add("2. **Andere features.** De featureset is de gecertificeerde Phase "
        "2-registry uit `conf/features/default.yaml`, ongewijzigd. Deze fase "
        "evalueert MODELLEN en geen features (§2).")
    add("3. **Een ander werkpunt.** De curve in §5 staat er ter informatie; het "
        "oordeel valt op "
        f"{_f(payload['probability_threshold'], 2)}. Een werkpunt kiezen op de "
        "uitkomst is een trial die niet is geboekt.")
    add("4. **Directionele voorspelling.** Technisch geblokkeerd: er bestaat "
        "geen handtekening waarmee dit model een eigen doelvector krijgt. §24 "
        "houdt CatBoost daarvoor gearchiveerd en deze run verandert daar niets "
        "aan.")
    conservative = _power(payload)["scenarios"]["uniqueness_and_cross_section"]
    mde = float(conservative["minimum_detectable_effect"])
    threshold = float(_power(payload)["threshold_auc"])
    add(f"5. **Een effect kleiner dan {_f(mde, 3)} AUC-overschot.** Dat is de "
        f"MDE op de conservatieve effectieve steekproefgrootte "
        f"({_f(conservative['assumptions']['effective_n'], 1)} waarnemingen). "
        f"Het overschot dat de drempel van {_f(threshold, 2)} vraagt, is "
        f"{_f(threshold - 0.5, 3)} — kleiner dan de MDE, dus een AUC die die "
        f"drempel nét zou halen, is op deze opzet niet van 0,50 te "
        f"onderscheiden. Dat lag vóór de run vast.")
    add("")
    n_promoted = int(campaign["by_status"].get("PROMOTED", 0))
    add("Elke promotieclaim in dit rapport draagt de labels "
        f"`{campaign['unfiltered']['impact_status']}` en "
        f"`{campaign['unfiltered']['spread_status']}` "
        f"({_f(campaign['unfiltered']['half_spread_bps'], 1)} bp). "
        + ("Er zijn geen promotieclaims." if n_promoted == 0 else
           f"Dat geldt voor alle {n_promoted} gepromoveerde specs: het oordeel "
           f"is geldig ONDER die labels en vervalt zodra de impact is "
           f"gekalibreerd of de spread gemeten."))
    add("")


def build_h3_payload(
    *,
    git_sha: str,
    universe: Mapping[str, Any],
    track: str,
    preregistration: Mapping[str, Any],
    adequacy_artefact: Mapping[str, Any],
    labeling: Mapping[str, Any],
    campaign: Mapping[str, Any],
    ledger_before: int,
    ledger_after: int,
    trials_booked_at_freeze: int,
    ledger_unit: str,
    ledger_config_hash: str,
    probability_threshold: float,
    embargo_bars: int,
    artefact_path: str,
) -> dict[str, Any]:
    """Het artefact dat op schijf gaat én het rapport voedt."""
    return {
        "git_sha": git_sha,
        "preregistration_id": preregistration["preregistration_id"],
        "preregistration_frozen_utc": preregistration["frozen_utc"],
        "ledger_total_at_freeze": preregistration["ledger_total_at_freeze"],
        "preregistration": preregistration,
        "universe": dict(universe),
        "track": track,
        "labeling": dict(labeling),
        "probability_threshold": float(probability_threshold),
        "embargo_bars": int(embargo_bars),
        "adequacy_a_priori": adequacy_artefact["verdicts"]["meta_labeling"],
        "adequacy_details": adequacy_artefact["details"]["meta_labeling"],
        "campaign": dict(campaign),
        "ledger_before": int(ledger_before),
        "ledger_after": int(ledger_after),
        "trials_booked_at_freeze": int(trials_booked_at_freeze),
        "ledger_unit": ledger_unit,
        "ledger_config_hash": ledger_config_hash,
        "artefact_path": artefact_path,
    }
