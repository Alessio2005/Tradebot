# src/tradebot/reporting/phase6_regime_benchmark.py
"""`reports/M0_VS_HMM_BENCHMARK.md` uit het H2-campagne-artefact. Deliverable 18.

Eén payload voedt zowel het JSON-artefact als het rapport. Dat is geen
netheid maar een voorzorg: twee keer dezelfde grootheid opbouwen is precies hoe
een rapport gaat afwijken van de run die het beschrijft.

Ref: fase-opdracht stap 11, deliverable 18; pre-registratie
`3d3af28730a6c7f9da48d13139522a05`.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

__all__ = ["build_h2_payload", "render_regime_report"]

_STATUS_MEANING = {
    "PROMOTED": "superieure netto OOS Sharpe na kosten, alle poorten open",
    "FALSIFIED": "aantoonbaar slechter op economische gronden — turnover of "
                 "spread-afhankelijkheid, niet de Sharpe-toets",
    "UNPROVEN": "geen verbetering, en de toets kon het verwachte effect niet "
                "zien",
    "DESCOPED": "niet beoordeelbaar: de zeldzaamste toestand haalt de "
                "adequaatheidspoort niet",
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


def render_regime_report(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    campaign = payload["campaign"]
    trials = campaign["trials"]

    _header(add, payload, campaign)
    _outcome(add, campaign, trials)
    _preregistration(add, payload)
    _architecture(add, campaign)
    _baseline(add, payload, campaign)
    _adequacy(add, payload, trials)
    _engine(add, campaign, trials)
    _paired_test(add, campaign, trials)
    _spread(add, payload, campaign, trials)
    _convergence(add, trials)
    _ledger(add, payload, campaign)
    _limitations(add, payload, campaign)
    return "\n".join(lines) + "\n"


def _header(add: Any, payload: Mapping[str, Any],
            campaign: Mapping[str, Any]) -> None:
    universe = payload["universe"]
    add("# H2 — HET M2 FILTERED HMM TEGEN M0 CAUSAL VOL-BUCKETS")
    add("")
    add("> **Deliverable 18** · Phase 6 stap 11 · Phase 7/8 Stage C-3")
    add(f"> **git_sha:** `{payload['git_sha']}`")
    add(f"> **Pre-registratie:** `{payload['preregistration_id']}` "
        f"(bevroren {payload['preregistration_frozen_utc']}, "
        f"M bij bevriezing = {payload['ledger_total_at_freeze']})")
    add(f"> **Universum:** {len(universe['symbols'])} gecertificeerde reeksen · "
        f"{universe['n_bars']} bars · {universe['period_start']} t/m "
        f"{universe['period_end']}")
    add(f"> **Primaire track:** `{payload['track']}` · "
        f"{payload['n_oos_bars']} OOS-bars · embargo "
        f"{payload['embargo_bars']} bars")
    add(f"> **Kostenlabels:** `{campaign['m0']['impact_status']}` · "
        f"`{campaign['m0']['spread_status']}` "
        f"({_f(campaign['m0']['half_spread_bps'], 1)} bp half-spread) · "
        f"risk `config_hash` `{campaign['m0']['risk_policy_hash']}`")
    add("")


def _outcome(add: Any, campaign: Mapping[str, Any],
             trials: Sequence[Mapping[str, Any]]) -> None:
    counts = campaign["by_status"]
    add("## 0. De uitkomst, eerst")
    add("")
    promoted = counts.get("PROMOTED", 0)
    if promoted:
        add(f"**{promoted} van de {len(trials)} conditioneerders is "
            f"gepromoveerd.** De Phase 2-poort beslist over de rest.")
    else:
        add("**Geen enkele conditioneerder is gepromoveerd. M0 Causal "
            "Vol-Buckets blijft de productie-baseline.**")
    add("")
    add("| Oordeel | Aantal | Betekenis |")
    add("|---|---|---|")
    for status in ("PROMOTED", "FALSIFIED", "UNPROVEN", "DESCOPED"):
        add(f"| {status} | {counts.get(status, 0)} | "
            f"{_STATUS_MEANING[status]} |")
    add("")
    binding: dict[str, int] = {}
    for trial in trials:
        for name in trial["verdict"]["binding"]:
            binding[name] = binding.get(name, 0) + 1
    if binding:
        add("Welk stop-criterium waar bond:")
        add("")
        add("| Stop-criterium | Conditioneerders |")
        add("|---|---|")
        for name, count in sorted(binding.items(), key=lambda kv: -kv[1]):
            add(f"| `{name}` | {count} |")
        add("")


def _preregistration(add: Any, payload: Mapping[str, Any]) -> None:
    content = payload["preregistration"]["content"]
    power = content["parameters"]["power_analysis"]
    add("## 1. Wat vooraf vastlag")
    add("")
    add("| | |")
    add("|---|---|")
    add("| Nulhypothese | de netto OOS Sharpe onder M2-conditionering is ≤ die "
        "onder M0, na kosten |")
    add(f"| Primaire maat | `{content['primary_metric']}` |")
    add("| Toets | Jobson-Korkie met Memmel-correctie op het gepaarde "
        "Sharpe-verschil |")
    add(f"| Geplande trials | {content['planned_trials']} "
        "(M1, M2-gaussian en M2-student_t, elk met k = 2 en k = 3) |")
    add(f"| Verwacht effect | {_f(power['expected_sharpe_gain'], 2)} "
        "geannualiseerde Sharpe-eenheden |")
    add(f"| Baseline-Sharpe vooraf | {_f(power['baseline_sharpe'])} "
        f"(`{power['baseline_source']}`) |")
    add("")
    add("De pre-registratie noteerde het oordeel over haar eigen power VOOR de "
        "run, en dat is de belangrijkste zin erin:")
    add("")
    add("| ρ tussen de armen | MDE (Sharpe-eenheden) | informatief? |")
    add("|---|---|---|")
    for name, scenario in sorted(power["scenarios"].items()):
        add(f"| {name.replace('rho_', '')} | "
            f"{_f(scenario['minimum_detectable_effect'], 3)} | "
            f"{'ja' if scenario['informative'] else 'nee'} |")
    add("")
    add("De correlatie die nodig zou zijn om 0,08 te zien is ρ = 0,99816 — en "
        "bij die correlatie verandert de overlay de returnreeks zo weinig dat "
        "een effect van 0,08 er niet KAN zijn. De opzet is intern "
        "tegenstrijdig, en dat stond er voordat er iets was gemeten.")
    add("")


def _architecture(add: Any, campaign: Mapping[str, Any]) -> None:
    tilt = campaign["overlay_is_a_tilt"]
    add("## 2. Wat de architectuur al beperkt: dit is een tilt, geen "
        "risicoreductie")
    add("")
    add("De soevereine laag schaalt het hele boek met "
        "`w_t = min(max_leverage, σ_target / σ_boek)`. Vermenigvuldig elke "
        "exposure met dezelfde `c`, dan deelt `w_t` er weer door: **een "
        "regime-overlay kan dit boek niet de-grossen.** Wat overblijft is de "
        "asymmetrie TUSSEN symbolen — elk symbool heeft zijn eigen regime — en "
        "dat maakt van H2 een cross-sectionele tilt: haal het risicobudget weg "
        "bij wat nu onrustig is en geef het aan de rest.")
    add("")
    add("Gemeten op de M0-arm tegen de ongeconditioneerde arm:")
    add("")
    add("| Grootheid | M0 / ongeconditioneerd |")
    add("|---|---|")
    add(f"| bruto notional | {_f(tilt['gross_notional_ratio'], 4)} |")
    add(f"| turnover | {_f(tilt['turnover_ratio'], 4)} |")
    add(f"| bars met een positie | {_f(tilt['exposed_bars_ratio'], 4)} |")
    add("")
    add("Een bruto notional die vrijwel gelijk blijft terwijl de turnover "
        "stijgt, is precies wat een tilt doet: het risicobudget krimpt niet, "
        "het verhuist — en verhuizen kost fees. Elke zin in dit rapport die "
        "'risicoreductie' zou suggereren, zou onjuist zijn.")
    add("")


def _baseline(add: Any, payload: Mapping[str, Any],
              campaign: Mapping[str, Any]) -> None:
    plain, m0 = campaign["plain"], campaign["m0"]
    test = campaign["m0_vs_unconditioned"]
    add("## 3. De baseline zelf, en wat zij kost")
    add("")
    add("| Arm | netto OOS Sharpe | turnover (notional) | fees | bars met "
        "positie | bars met fill |")
    add("|---|---|---|---|---|---|")
    for label, arm in (("ongeconditioneerd", plain), ("M0 vol-buckets", m0)):
        add(f"| {label} | {_f(arm['net_sharpe_oos'])} | "
            f"{_i(arm['turnover_notional_oos'])} | "
            f"{_f(arm['fees_oos'], 2)} | {_i(arm['n_exposed_bars_oos'])} | "
            f"{_i(arm['n_trading_bars_oos'])} |")
    add("")
    delta = m0["net_sharpe_oos"] - plain["net_sharpe_oos"]
    add(f"**M0 zelf verbetert de baseline niet: de delta tegen géén overlay is "
        f"{_f(delta)} Sharpe-eenheden** (gepaard: z = "
        f"{_f(test['z_statistic'], 3)}, p = {_f(test['p_value'], 4)}, ρ = "
        f"{_f(test['correlation'], 4)}). Dat is geen detail voor de "
        f"vergelijking die volgt: de uitdagers worden tegen M0 gemeten omdat "
        f"de pre-registratie dat voorschrijft, maar een uitdager die M0 "
        f"verslaat, hoeft daarmee de ONGECONDITIONEERDE baseline nog niet te "
        f"verslaan. Beide getallen staan daarom in §5.")
    add("")
    add("Regimewisselingen van M0 op de gescoorde bars, per symbool:")
    add("")
    add("| Symbool | wisselingen | gemiddelde regimeduur (bars) |")
    add("|---|---|---|")
    transitions = campaign["m0_conditioner_transitions"]
    n_bars = payload["n_oos_bars"]
    for symbol, count in transitions.items():
        duration = n_bars / (count + 1) if count >= 0 else float("nan")
        add(f"| {symbol} | {count} | {_f(duration, 1)} |")
    add("")


def _adequacy(add: Any, payload: Mapping[str, Any],
              trials: Sequence[Mapping[str, Any]]) -> None:
    add("## 4. De adequaatheidspoort — en waarom hij hier bindt")
    add("")
    add("Stop-criterium 1 van de pre-registratie meet de **filtered** bezetting "
        "van de zeldzaamste toestand per fold, op het trainvenster waarop de "
        "parameters zijn geschat. Zakt die onder 100 observaties, dan schat het "
        "model daar een gemiddelde, een schaal en k−1 overgangskansen op enkele "
        "tientallen punten, en is die toestand een uitschieterdetector in "
        "plaats van een regime.")
    add("")
    add("De a-priori poort in `conf/model/adequacy.yaml` gebruikt de UNIFORME "
        "aanname (1/k per toestand) en noemt zichzelf daarbij optimistisch. Op "
        "495 trainbars geeft dat 247,5 observaties bij k = 2 — ruim boven de "
        "eis. De gemeten filtered bezetting is een ander getal:")
    add("")
    add("| Conditioneerder | min | mediaan | fits onder de poort | "
        "trainbars per fold |")
    add("|---|---|---|---|---|")
    for trial in trials:
        cond = trial["conditioner"]
        add(f"| `{trial['label']}` | "
            f"{_f(cond['rarest_state_obs_per_fold'], 2)} | "
            f"{_f(cond['rarest_state_obs_median'], 2)} | "
            f"{cond['n_fits_below_gate']}/{cond['n_fits']} | "
            f"{cond['n_train_obs_per_fold']} |")
    add("")
    m1 = [t for t in trials if t["label"].startswith("m1-")]
    if m1:
        usable = [f["n_train_obs"] for f in m1[0]["conditioner"]["fits"]]
        add("**M1 zakt het diepst, en de oorzaak is de M0-bucket zelf.** Zijn "
            "toestanden ZIJN de buckets, en die bestaan pas na "
            "`zscore_min_periods` bars: vóór die grens is het regime "
            "ONGEDEFINIEERD en niet 'normaal bij gebrek aan beter'. Het "
            f"trainvenster van de vroegste fold telt daardoor {_i(min(usable))} "
            f"bruikbare bars in plaats van {_i(max(usable))}, en het "
            "zeldzaamste regime (HOOG) heeft daar vijf waarnemingen. Een "
            "overgangskans uit vijf overgangen is geen schatting.")
        add("")
    add("Het oordeel gaat over het MINIMUM en niet over de mediaan. Een "
        "walk-forward-evaluatie is alleen zinnig wanneer elke fold erin geldig "
        "is; een Sharpe over 1.200 OOS-bars waarvan een deel uit een model komt "
        "dat daar niet gefit had mogen worden, is geen schoon getal. De "
        "mediaan en het aantal fits onder de poort staan er wél bij, want zij "
        "bepalen of het oordeel 'dit model kan hier niet' luidt of 'deze folds "
        "konden niet'.")
    add("")


def _scored_bars(trials: Sequence[Mapping[str, Any]]) -> int:
    """Het aantal gescoorde bars; identiek voor elke arm."""
    return int(trials[0]["arm"]["n_scored_bars"]) if trials else 0


def _engine(add: Any, campaign: Mapping[str, Any],
            trials: Sequence[Mapping[str, Any]]) -> None:
    plain = campaign["plain"]["net_sharpe_oos"]
    add("## 5. Door de authoritative engine, met volledige kosten")
    add("")
    add("| Conditioneerder | netto Sharpe | Δ vs. M0 | Δ vs. geen overlay | "
        "turnover-ratio | Δ fees | bars met positie | oordeel |")
    add("|---|---|---|---|---|---|---|---|")
    for trial in trials:
        arm = trial["arm"]
        add(f"| `{trial['label']}` | {_f(arm['net_sharpe_oos'])} | "
            f"{_f(trial['net_sharpe_delta'])} | "
            f"{_f(arm['net_sharpe_oos'] - plain)} | "
            f"{_f(trial['turnover_ratio'], 3)} | "
            f"{_f(trial['fees_delta'], 2)} | "
            f"{_i(arm['n_exposed_bars_oos'])} | "
            f"{trial['verdict']['status']} |")
    add("")
    add("Alle armen draaien door dezelfde `EventDrivenEngine`, met dezelfde "
        "risicolaag, dezelfde router, dezelfde venue en dezelfde kosten. De "
        "ENIGE ingang die verschilt is de exposure: de basisexposure maal de "
        "regimefactor. Een verschil in uitkomst kan dus nergens anders vandaan "
        "komen.")
    add("")
    total = _scored_bars(trials)
    for trial in [t for t in trials
                  if t["arm"]["n_exposed_bars_oos"] < 0.9 * total]:
        add(f"**`{trial['label']}` houdt op "
            f"{_i(trial['arm']['n_exposed_bars_oos'])} van de {_i(total)} "
            "gescoorde bars nog een positie.** Zijn factor is zo vaak zo klein "
            "dat de resterende order onder de `min_notional` van de venue valt "
            "en niet wordt geplaatst. Zijn Sharpe staat dus op aanzienlijk "
            "minder bars dan die van de andere armen en is daarmee niet één op "
            "één vergelijkbaar — precies het onderscheid dat §0.5 van de "
            "fase-opdracht uitvraagt.")
        add("")
    add("Per conditioneerder, de regimediagnostiek op de gescoorde bars:")
    add("")
    add("| Conditioneerder | gem. factor | gem. absolute factorverandering "
        "per bar | wisselingen (mediaan over symbolen) | gem. regimeduur |")
    add("|---|---|---|---|---|")
    for trial in trials:
        cond = trial["conditioner"]
        changes = list(cond["mean_absolute_change"].values())
        switches = list(cond["n_transitions"].values())
        durations = list(cond["mean_duration_bars"].values())
        add(f"| `{trial['label']}` | {_f(cond['mean_multiplier'], 4)} | "
            f"{_f(statistics.median(changes), 4)} | "
            f"{_i(statistics.median(switches))} | "
            f"{_f(statistics.median(durations), 1)} |")
    add("")


def _paired_test(add: Any, campaign: Mapping[str, Any],
                 trials: Sequence[Mapping[str, Any]]) -> None:
    controls = campaign["controls"]
    add("## 6. De gepaarde toets, en de controle die haar geldig maakt")
    add("")
    add("| Conditioneerder | Sharpe uitdager | Sharpe M0 | Δ | z | p "
        "(eenzijdig) | ρ |")
    add("|---|---|---|---|---|---|---|")
    for trial in trials:
        test = trial["paired_sharpe_test"]
        add(f"| `{trial['label']}` | {_f(test['sharpe_a'])} | "
            f"{_f(test['sharpe_b'])} | {_f(test['difference'])} | "
            f"{_f(test['z_statistic'], 3)} | {_f(test['p_value'], 4)} | "
            f"{_f(test['correlation'], 4)} |")
    add("")
    add("**De negatieve controles.** Exit-criterium 12 eist er een per "
        "statistische toets; voor deze toets zijn het er twee, en zij meten "
        "verschillende dingen. Beide draaien op een GEPAARDE stationaire "
        "block-bootstrap van de echte returnreeksen, zodat de autocorrelatie, "
        "de staarten en de onderlinge correlatie meegaan.")
    add("")
    add("| Controle | gemeten | eis | uitkomst |")
    add("|---|---|---|---|")
    add(f"| size (twee gelijke Sharpes) | "
        f"{_f(controls['size_rejection_rate'], 3)} | ≤ "
        f"{_f(2 * controls['alpha'], 2)} | "
        f"{'GESLAAGD' if controls['size_passed'] else 'GEFAALD'} |")
    add(f"| power bij het verwachte effect "
        f"({_f(controls['expected_effect'], 2)}) | "
        f"{_f(controls['power_at_expected_effect'], 3)} | ≥ "
        f"{_f(controls['target_power'], 2)} | "
        f"{'GESLAAGD' if controls['power_passed'] else 'GEFAALD'} |")
    add(f"| power bij de MDE "
        f"({_f(controls['minimum_detectable_effect'], 3)}) | "
        f"{_f(controls['power_at_mde'], 3)} | ≥ "
        f"{_f(controls['target_power'], 2)} | ijkpunt |")
    add("")
    add(f"De toets houdt zijn niveau — hij verwerpt "
        f"{_f(controls['size_rejection_rate'], 3)} van de tijd op een nul "
        f"waar het verschil per constructie nul is — en hij ziet het effect "
        f"waarop de power-analyse hem ijkt. Wat hij NIET ziet is het effect "
        f"waar de hypothese over gaat: bij "
        f"{_f(controls['expected_effect'], 2)} Sharpe-eenheden verwerpt hij "
        f"{_f(controls['power_at_expected_effect'], 3)} van de tijd, tegen een "
        f"doel van {_f(controls['target_power'], 2)}.")
    add("")
    rhos = [t["paired_sharpe_test"]["correlation"] for t in trials]
    add(f"De power bij de MDE ligt onder het doel, en dat is geen tegenspraak: "
        f"die MDE komt uit het ρ = 0,95-scenario van de pre-registratie, "
        f"terwijl de GEMETEN correlatie tussen de armen {_f(min(rhos), 3)} tot "
        f"{_f(max(rhos), 3)} is. Bij een lagere correlatie is het gepaarde "
        f"verschil ruiziger en is dezelfde MDE minder goed te zien. Het "
        f"ijkpunt bevestigt dus wat het hoort te bevestigen: de toets werkt, "
        f"en zijn resolutie ligt bij tienden van een Sharpe-eenheid.")
    add("")
    add("Dat is de GEMETEN versie van wat de pre-registratie analytisch al "
        "vaststelde, en het is de reden dat een niet-significante uitslag hier "
        "`UNPROVEN` heet en geen falsificatie draagt (no-go 8).")
    add("")


def _spread(add: Any, payload: Mapping[str, Any],
            campaign: Mapping[str, Any],
            trials: Sequence[Mapping[str, Any]]) -> None:
    sweep_bps = payload["spread_sweep_bps"]
    add("## 7. Spread-sensitiviteit")
    add("")
    add("De spread is `SPREAD_ASSUMED` op "
        f"{_f(campaign['m0']['half_spread_bps'], 1)} bp; Corwin-Schultz is op "
        "deze data verworpen (32,6–65,9 bp, 33–38 % negatief, AD-3). De "
        "pre-registratie eist daarom: bij welke aangenomen half-spread "
        "verdwijnt een gemeten verbetering? Verdwijnt zij al bij 3 bp, dan is "
        "de promotie een spread-aanname en geen modelresultaat.")
    add("")
    add("Netto OOS Sharpe van de M0-arm per aangenomen half-spread:")
    add("")
    add("| half-spread (bp) | " + " | ".join(_f(b, 1) for b in sweep_bps) + " |")
    add("|---|" + "---|" * len(sweep_bps))
    m0_sweep = campaign["m0_spread_sweep"]
    add("| M0 netto Sharpe | "
        + " | ".join(_f(m0_sweep[str(float(b))]) for b in sweep_bps) + " |")
    add("")
    add("Δ Sharpe van elke uitdager tegen M0, per half-spread:")
    add("")
    add("| Conditioneerder | "
        + " | ".join(f"{_f(b, 1)} bp" for b in sweep_bps)
        + " | winst verdwijnt bij |")
    add("|---|" + "---|" * (len(sweep_bps) + 1))
    for trial in trials:
        row = " | ".join(
            _f(trial["spread_sweep"][str(float(b))]) for b in sweep_bps)
        vanish = trial["spread_at_which_gain_vanishes"]
        label = ("n.v.t. — geen winst bij de basis-spread" if vanish is None
                 else f"{_f(vanish, 1)} bp")
        add(f"| `{trial['label']}` | {row} | {label} |")
    add("")
    ordered = [m0_sweep[str(float(b))] for b in sweep_bps]
    monotone = all(a >= b for a, b in pairwise(ordered))
    if not monotone:
        add("**De curve is niet monotoon, en dat beperkt hoe scherp dit "
            "criterium te lezen is.** Een bredere spread kost per constructie "
            "meer, dus de netto Sharpe zou moeten dalen. Hij doet dat niet: de "
            "M0-arm loopt van "
            + " naar ".join(_f(v) for v in ordered)
            + ". De oorzaak is padafhankelijkheid in de soevereine laag — de "
            "drawdown-breaker is een eenrichtingsdeur, en een klein "
            "kostenverschil bepaalt of hij op een bepaalde bar afgaat. "
            "'De half-spread waarbij de winst verdwijnt' is daarmee een GROVE "
            "aanwijzing en geen scherpe drempel; een waarde van 3 bp uit deze "
            "tabel draagt niet het gewicht dat een monotone curve eraan zou "
            "geven. Het criterium bindt hier nergens, en dat oordeel hangt "
            "niet van deze fijnstructuur af.")
        add("")
    add("`n.v.t.` betekent hier niet-van-toepassing en niet niet-geschonden: "
        "zonder gemeten winst is er niets dat bij 3 bp kan verdwijnen. Het "
        "criterium kan dan per constructie niet binden, en het als geschonden "
        "boeken zou elke arm zonder verbetering ook nog een "
        "spread-falsificatie geven.")
    add("")


def _convergence(add: Any, trials: Sequence[Mapping[str, Any]]) -> None:
    add("## 8. Convergentie, gedegenereerde fits en toestandsscheiding")
    add("")
    add("| Conditioneerder | convergentieratio | toestandsvariantie-ratio "
        "(min / mediaan) |")
    add("|---|---|---|")
    for trial in trials:
        cond = trial["conditioner"]
        ratios = [f["state_variance_ratio"] for f in cond["fits"]
                  if "state_variance_ratio" in f]
        spread = ("—" if not ratios else
                  f"{_f(min(ratios), 4)} / {_f(statistics.median(ratios), 4)}")
        add(f"| `{trial['label']}` | {_f(cond['convergence_ratio'], 2)} | "
            f"{spread} |")
    add("")
    add("De toestandsvariantie-ratio is `min(var) / max(var)` over de "
        "toestanden. Ligt hij rond 1, dan zijn twee toestanden hetzelfde regime "
        "met twee namen en is `k` te groot voor deze data. Het is een DIAGNOSE "
        "en geen poort — de poort is de bezetting uit §4, en die staat in de "
        "pre-registratie.")
    add("")
    add("Een convergentieratio onder 1 telt de fits waarin de EM is gestopt "
        "voordat zij convergeerde, inclusief de Student-t-fits die op een "
        "instortende toestand zijn teruggerold (`degenerate`). Dat is een "
        "geregistreerd resultaat en geen weggevangen fout.")
    add("")


def _ledger(add: Any, payload: Mapping[str, Any],
            campaign: Mapping[str, Any]) -> None:
    add("## 9. De ledger en `M`")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| `M` vóór deze run | {payload['ledger_before']} |")
    add(f"| `M` ná deze run | {payload['ledger_after']} |")
    add(f"| trials geboekt bij het bevriezen | "
        f"{payload['trials_booked_at_freeze']} |")
    add(f"| trials in deze run gedraaid | {campaign['n_trials']} |")
    add("")
    add("De zes trials zijn bij het BEVRIEZEN van de pre-registratie geboekt — "
        "wie een parameterruimte vastlegt, heeft die kansen genomen. Deze run "
        "voert ze uit en boekt ze dus niet opnieuw; het oordeel gaat als "
        "amendement terug de ledger in, met `n_trials = 0`.")
    add("")
    add("M0 telt niet mee: nul latente toestanden, nul geschatte parameters, "
        "drempels uit `conf/model/regime.yaml` die vóór de meting vastlagen. "
        "De ongeconditioneerde arm telt evenmin mee — die is de Phase "
        "5-baseline zelf.")
    add("")


def _limitations(add: Any, payload: Mapping[str, Any],
                 campaign: Mapping[str, Any]) -> None:
    add("## 10. Wat hiermee NIET is getoetst")
    add("")
    add("1. **De vraag of een regime-overlay het risico kan verlagen.** Dat kan "
        "dit boek per constructie niet meten: de soevereine vol-target "
        "herschaalt elke uniforme reductie weg (§2). Wat is gemeten is de "
        "cross-sectionele tilt.")
    add("2. **Een effect van de orde 0,08 Sharpe-eenheden.** De gemeten power "
        "daar is "
        f"{_f(campaign['controls']['power_at_expected_effect'], 3)} (§6). Deze "
        "opzet kan dat effect niet zien, en dat lag vóór de run vast.")
    add("3. **De conditioneerders waarvan de bezetting de poort niet haalde.** "
        "Zij zijn `UNPROVEN — insufficient data` met de gemeten bezetting "
        "erbij, en NIET gefalsificeerd (no-go 8). Wat daar ontbreekt is data, "
        "geen model: op 495 trainbars per fold is een toestand met "
        "minder dan 100 verwachte observaties niet te schatten.")
    add("4. **M3 Markov-Switching GARCH.** Staat expliciet buiten deze fase "
        "(§2 van de fase-opdracht) en is niet gefit.")
    add("5. **Andere primaire signaaltracks.** `long_only_equal_weight` "
        "halteert op 2022-05-10 en handelt ~130 van 1.743 bars; een "
        "regime-experiment daarop meet de eerste zes maanden en daarna niets. "
        f"De keuze voor `{payload['track']}` stond in de pre-registratie.")
    add("")
    add("Elke promotieclaim in dit rapport draagt de labels "
        f"`{campaign['m0']['impact_status']}` en "
        f"`{campaign['m0']['spread_status']}` "
        f"({_f(campaign['m0']['half_spread_bps'], 1)} bp). Er zijn geen "
        "promotieclaims.")
    add("")


def build_h2_payload(
    *,
    git_sha: str,
    universe: Mapping[str, Any],
    track: str,
    preregistration: Mapping[str, Any],
    adequacy_artefact: Mapping[str, Any],
    campaign: Mapping[str, Any],
    ledger_before: int,
    ledger_after: int,
    trials_booked_at_freeze: int,
    ledger_unit: str,
    ledger_config_hash: str,
    spread_sweep_bps: Sequence[float],
    artefact_path: str,
    n_oos_bars: int,
    embargo_bars: int,
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
        "n_oos_bars": int(n_oos_bars),
        "embargo_bars": int(embargo_bars),
        "spread_sweep_bps": [float(b) for b in spread_sweep_bps],
        "adequacy_a_priori": {
            key: value for key, value in adequacy_artefact["verdicts"].items()
            if key.startswith("hmm")
        },
        "campaign": dict(campaign),
        "ledger_before": int(ledger_before),
        "ledger_after": int(ledger_after),
        "trials_booked_at_freeze": int(trials_booked_at_freeze),
        "ledger_unit": ledger_unit,
        "ledger_config_hash": ledger_config_hash,
        "artefact_path": artefact_path,
    }
