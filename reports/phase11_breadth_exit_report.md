# FASE 11 — BREEDTE EN TIJDSCHAAL: EXITRAPPORT

> **Stand:** van de 21 exit-criteria zijn er 19 waar. Eén is niet van
> toepassing: criterium 16, omdat H-11.2 niet is geregistreerd. **Criterium 5
> is niet waar**: stap 1 staat niet als eigen PR op `main`. De reden en wat ervoor nodig
> is, staan in §2. Volgens de opdracht blokkeert dat de afsluiting, en dit
> rapport verzwijgt het niet.
>
> **Trials:** 0 besteed. H-11.2 is niet geregistreerd, omdat haar
> haalbaarheidspoort rood was onder elk boek dat het geldende beleid toelaat.

**Faseopdracht:** `Prompts-fases/fase_11_breedte_en_tijdschaal.md`
**Branch / PR:** `claude/wizardly-ride-c46qwc`,
[PR #6](https://github.com/Alessio2005/Tradebot/pull/6)
**Basis:** `main` op `28cc31b`
**Rapporten:** `reports/phase11_breadth_and_timescale.md` (breedte, klok,
haalbaarheid), `reports/phase11_ic_wall.md` (de muur),
`reports/phase11_breadth_owner_decision.md` (het besluit van de eigenaar)
**Artefact:** `artefacts/governance/phase11_breadth.json`

---

## 1. Wat deze fase heeft vastgesteld

1. **"N_eff" was twee grootheden onder één naam.** Onafhankelijke
   weddenschappen (participatieratio, begrensd door de rang) en Kish'
   ontwerpeffect heten nu elk bij hun eigen naam (AD-30). Op het dollar-neutrale
   residu van zes namen geeft de participatieratio 4,353 [4,144; 4,500] en het
   ontwerpeffect 123,58 [45,60; 335,91]. Het tweede getal is geen breedte (DI-35).
2. **Zes namen zijn directioneel 1,6 weddenschap**, en dollar-neutraal 4,4. Op
   het poortvenster is de directionele breedte aantoonbaar smaller (1,369 tegen
   1,601, de intervallen overlappen niet).
3. **De signaalklok van het momentumboek is 30 bars.** Dat boek betaalt voor 178
   herschikkingen per jaar om ongeveer 12 onafhankelijke besluiten uit te
   drukken.
4. **De IC-muur.** Bij de DSR-drempel vraagt een klok van één dag een IC van
   0,042 tot 0,077. Een klok van vijf dagen of meer vraagt minstens 0,094. Het
   domein heeft nooit meer dan 0,050 laten zien. De fundamentele wet geldt op
   de neutrale constructies binnen 2,5 % en 9 %. Op de directionele constructie
   wijkt zij af met een factor 1,8, afhankelijk van de voorspellingsruis.
5. **De beslisklok is een kostenhefboom en geen beslisbare hypothese.** De
   kostenwinst bij nul signaalverval is 0,19 Sharpe (0,23 onder het geldende
   beleid, en hooguit 0,53 bij de bruto-cap). Het kleinste zichtbare verschil is
   0,85. H-11.2 is niet geregistreerd.

## 2. Afvinklijst

| # | Criterium | Stand | Bewijs |
|---|---|---|---|
| 1 | Vingerafdruk begin en eind; elke nieuwe failure is van deze fase of met bewijs van de zusterfase | **waar** | §3: dezelfde vier failures als op `28cc31b`, geen nieuwe |
| 2 | `independent_bets` bestaat en delegeert naar `effective_n_assets`; geen vijfde implementatie | **waar** | `validation/breadth.py::independent_bets` valideert en roept `portfolio/covariance.py::effective_n_assets` aan; `test_breadth_definitions.py` |
| 3 | Rangbegrenzing getest, negatieve controle aantoonbaar rood | **waar** | `independent_bets` naar de ρ̄-formule laten wijzen maakte twee tests rood (met de hand gedraaid en teruggezet, commit `799e095`) |
| 4 | DI-35 vastgepind als `xfail(strict=True)`; gedrag van `inference.py` ongewijzigd | **waar** | `test_breadth_definitions.py`; de diff op `inference.py` is één docstring en één `#:`-regel; de LOC-cap 1.162 staat |
| 5 | Stap 1 als **eigen PR** op `main`, met stap 6.1 en 8.1 van de zusterfase in de beschrijving | **niet waar** | Zie hieronder |
| 6 | Elke breedte draagt venster, paneel, constructie, formulenaam en interval | **waar** | per rij `window`, `construction`, `independent_bets_ci`, `design_effect_ci`; blokken `panel` en `quantities` |
| 7 | Geen eerste moment van een signaal gemeten, niets dat van funding afhangt | **waar** | `first_moments_computed: false` in `breadth` en `feasibility`; §3.4 van het meetrapport **citeert** één gepubliceerde L3-Sharpe |
| 8 | τ_int per track gemeten; de constante-reeks-tak geeft `inf` en is getest | **waar** | `test_signal_clock.py::test_a_constant_series_has_no_finite_time` |
| 9 | De verklaring van H-10.1 staat in het rapport, niet in de ledger | **waar** | meetrapport §2.2; `hypothesis_ledger.json` ongewijzigd |
| 10 | Vasthoudimplementaties op equivalentie getest, risicopariteit zichtbaar anders; DI-36 geboekt | **waar** | `test_hold_equivalence.py`; DI-36 |
| 11 | `required_ic` weigert ongeldige invoer; `dsr_hurdle` reproduceert 1,868609; muur door simulatie geverifieerd, afwijking gerapporteerd | **waar** | `test_ic_wall.py`; muurrapport §2 |
| 12 | AD-29 en AD-30 geschreven; ADR volledig CRLF, eindigt op één CRLF | **waar** | nul kale LF, eindigt op precies één CRLF (nagemeten na de laatste commit) |
| 13 | AD-29 additief; `freeze_preregistration` ongewijzigd; DI-37 draagt de voorwaarde | **waar** | `assert_ic_wall_declared` in `validation/breadth.py`; `registry/` en `apps/freeze_preregistration.py` niet in de diff |
| 14 | Haalbaarheidsrekening met detectiegrens uit de echte kern in het rapport | **waar** | meetrapport §3; `sharpe_difference_test` op 200 nulpaden |
| 15 | H-11.2 geregistreerd dan en slechts dan wanneer de poort groen was | **waar** | poort `RED`, `red_under_every_book: true`; geen pre-registratie |
| 16 | Is H-11.2 geregistreerd: controles, decompositie, beslisregel | **n.v.t.** | niet geregistreerd; stap 8 is de vooraf geregistreerde handeling bij rood |
| 17 | `holdout_lock.json` ongewijzigd | **waar** | niet in de diff; `reads: []` |
| 18 | Trialsaldo 5 + zusterfase + deze fase ≤ 25, deze fase ≤ 1 | **waar** | §4 |
| 19 | Eigenaarsdocument met drie paden en hun kosten, zonder keuze | **waar** | `reports/phase11_breadth_owner_decision.md` |
| 20 | Geen bestand van de zusterfase gewijzigd | **waar** | §5 |
| 21 | ruff en mypy schoon; apps ≤ 80 LOC; nieuwe modules op budget 0; zes poortscripts exit 0 | **waar** | §6 |

**Criterium 5, uitgelegd.** Stap 1.7 vraagt voor stap 1 een eigen PR naar
`main`, zodat de zusterfase de hernoeming kan zien voordat zij haar stap 6.1
(de AD-20-telling) en 8.1 uitvoert. Deze sessie mag alleen naar
`claude/wizardly-ride-c46qwc` pushen. Een tweede branch vraagt expliciete
toestemming van de eigenaar, en die is er niet. Stap 1 staat daarom in
PR #6, in twee eigen commits die niets anders aanraken:

- `799e095` feat(validation): stap 1, het aantal weddenschappen krijgt een eigen naam
- `703acd9` docs(validation): stap 1, het ontwerpeffect zegt wat het is, DI-35 en DI-38

De beschrijving van PR #6 noemt stap 6.1 en 8.1 van de zusterfase bij naam.
**Wat er nodig is om criterium 5 waar te maken:** de eigenaar geeft toestemming
voor een tweede branch, of zet die twee commits zelf op een eigen PR. Ze
hangen alleen van `main` af. Of de eigenaar samenvoegt zonder dit criterium, is
aan de eigenaar.

## 3. Vingerafdruk

`python -m pytest -p no:cacheprovider`, Python 3.11.15, gepinde locks.

| moment | commit | collected | passed | skipped | xfailed | failed |
|---|---|---:|---:|---:|---:|---:|
| begin (stap 1.1) | `28cc31b` | 3.120 | 3.087 | 25 | 4 | 4 |
| eind | `096a90d` | 3.194 | 3.160 | 25 | 5 | 4 |

De vier failures aan het begin, alle vier van `main` en van de zusterfase
(§3.1 van `fase_11_meetbasis_en_carry.md`):

1. `tests/unit/test_docs_claim_only_what_exists.py::…::test_no_document_claims_a_path_that_does_not_exist` (`conf/env/` in `RISK_MANDATE.md`)
2. `tests/unit/test_regime_overlay.py::…::test_a_uniform_factor_is_neutralised_by_the_vol_target` (19.166,50 tegen 19.770,61)
3. `tests/regression/test_dust_breaks_relative_limits.py::test_a_book_that_straddles_the_dust_tolerance_is_decided` (beleidshash)
4. `tests/unit/test_regime_conditioning.py::…::test_the_multiplier_is_causal[hmm3-diag-student_t]`

Aan het eind zijn het **dezelfde vier**, met dezelfde namen. Er is geen
failure bijgekomen en er is geen test verdwenen. Het verschil van 74 tests is
volledig van deze fase, vastgesteld door de verzamelde test-ID's van `28cc31b` en
`096a90d` te vergelijken:

- 73 in de zeven testbestanden van deze fase, waarvan één `xfail(strict=True)`:
  `test_breadth_definitions.py::test_the_design_effect_is_not_breadth`, DI-35;
- 1 in `tests/unit/test_app_call_sites.py`, dat over de apps parametriseert en
  `run_breadth_measurement.py` nu meeneemt. Die test slaagt.

Het exitrapport zelf is documentatie buiten `docs/` en raakt geen test.

## 4. Trialstand

| post | trials |
|---|---:|
| `m_new` (bevroren budget, `ledger_reset.json`) | 25 |
| fase 10 (H-10.1, H-10.2 e.a., in de ledger) | 5 |
| zusterfase, gepland plafond (§5.4 van de opdracht) | ≤ 8 |
| **deze fase** | **0** |
| resterend, bij het plafond van de zusterfase | ≥ 12 |

Deze fase heeft niets aan `hypothesis_ledger.json` toegevoegd. Wat de
zusterfase werkelijk besteedt, staat na haar samenvoeging in de ledger. Het
saldo `5 + (zusterfase) + 0 ≤ 25` houdt voor elke waarde tot haar plafond.

## 5. Eigendom en gedeelde bestanden

**Bestanden van de zusterfase (§5.2):** geen gewijzigd. `git diff --name-only
28cc31b...HEAD` bevat geen pad onder `conf/risk/`, `src/tradebot/risk/`,
`src/tradebot/registry/`, `artefacts/baseline/`, `backtest/phase5_baseline.py`,
`apps/run_phase5_baseline.py`, de carry-modules, `live/`, `oms/`,
`pyproject.toml`, `.github/`, `tests/regression/`, `test_regime_overlay.py`,
`docs/DATA_REGISTER.md`, `RISK_MANDATE.md`, `CODE_REGISTER.md`,
`MEASUREMENT_DOMAIN.md` of `scripts/`. Deze fase leest wel uit
`artefacts/baseline/phase5_revaluation.json` en
`artefacts/governance/risk_config_registry.json`.

**Gedeelde bestanden, volgens hun protocol:**

| bestand | wat | hoe |
|---|---|---|
| `docs/ARCHITECTURAL_DECISIONS.md` | AD-29, AD-30 | één eigen commit (`b207127`), na AD-26; AD-27/28 vrij voor de zusterfase; CRLF |
| `docs/DEFERRED_ISSUES.md` | DI-35 t/m DI-38, en een "Bijgesteld"-blok voor DI-15/DI-21 | append-only, geen bestaande regel gewijzigd, eigen commits, CRLF |
| `src/tradebot/validation/inference.py` | één docstring, één `#:`-regel | gedrag ongewijzigd (criterium 4) |

**Bij een merge met de zusterfase.** In de ADR komen haar AD-27/28 vóór
AD-29/30. Bij een conflict houd je beide blokken, oplopend genummerd, en meet je
daarna CRLF en de ene afsluitende CRLF. In het DI-register kan haar
"Bijgesteld"-blok naast het mijne landen, direct onder dat van fase 10. Houd
beide.

**Nieuwe bestanden buiten de geplande lijst (R-10).** De opdracht plande de
campagne in `validation/breadth.py` en de app. Dat paste niet binnen R-4 en R-6
(`breadth.py` staat op 502 regels, de app op 65 van 80). Daarom zijn er vier
bestanden bijgekomen, alle vier nieuw en van niemand anders:
`src/tradebot/validation/phase11_breadth_measurement.py`,
`src/tradebot/validation/phase11_decision_clock_feasibility.py`,
`tests/unit/test_phase11_breadth_measurement.py` en
`tests/unit/test_decision_clock_feasibility.py`.

## 6. Poorten

| poort | uitkomst |
|---|---|
| `ruff check src/ apps/ tests/` | schoon |
| `mypy src/tradebot/schemas/ src/tradebot/utils/ apps/ --ignore-missing-imports` (de CI-scope) | schoon |
| `mypy` op de vijf modules van deze fase, `--ignore-missing-imports` | schoon |
| `scripts/check_file_size.py` | exit 0 |
| `scripts/check_hardcoded_params.py --strict` | exit 0; nieuwe modules op budget 0 |
| `scripts/check_banned_methods.py --strict` | exit 0 |
| `scripts/check_domain_consistency.py --strict` | exit 0 |
| `scripts/audit_fallbacks.py --strict` | exit 0 |
| `scripts/reachability_map.py --strict` | exit 0 |
| `apps/run_breadth_measurement.py` | 65 regels (≤ 80) |

**CI op PR #6.** De workflows die `pytest` draaien, zijn rood om de vier
failures van `main` in §3. Coverage staat onder de vloer; dat is op `main` ook
zo en is van de zusterfase (haar stap 9.3).
[Deze opmerking](https://github.com/Alessio2005/Tradebot/pull/6#issuecomment-5835497967) legt
dat uit.

## 7. Afwijkingen van de opdracht (R-10)

| opdracht | gemeten of gedaan | waar |
|---|---|---|
| de wet klopt "binnen ongeveer 15 %" | neutraal 2,5 % en 9 %; directioneel een factor 1,8 | muurrapport §2 |
| het tweede criterium van H-10.1 is "95 %-CI sluit nul uit na deflatie" | het is `dsr_development_best_k ≥ 0,95`, een DSR op de **niveau**-Sharpe. Ook dat criterium is rood | meetrapport §3.4 |
| de vasthoudoptie van de unit is een tweede implementatie | zij is met de geconfigureerde lookback voor geen k tussen 2 en 60 bruikbaar | DI-36, meetrapport §2.3 |
| de breakeven van H-10.1 beantwoordt V3 | de breakeven vraagt een eerste moment. Onder de aanname van de poort is zij per constructie 0 bp. Gerapporteerd is de kost per zijde waarbij de besparing zichtbaar wordt: 44,4 bp | meetrapport §3.5 |
| k\* als mediaan "zolang het venster is bereikt" | een afgekapte naam boven de mediaan laat de mediaan staan; dat is nu `median_is_determined` in code | meetrapport §2 |
| stap 7 wacht op de nieuwe ladder | de voorwaarde is niet vervuld. De poort is toch definitief rood, omdat het grootste boek onder het geldende beleid het niet kan omdraaien | meetrapport §3.3 |

## 8. Open handelingen

1. **Stap 7.1, na de zusterfase.** Zodra
   `artefacts/baseline/phase11_revaluation.json` met `9961e1613bc907a5` op
   `main` staat: zet `feasibility.ladder_artefact` in
   `conf/research/breadth.yaml` op dat pad en draai
   `python apps/run_breadth_measurement.py`. Dan wordt m gemeten in plaats van
   begrensd. Het oordeel kan niet omslaan; alleen de getallen van §3.3 worden
   scherper.
2. **Criterium 5**: zie §2.
3. **DI-35 tot en met DI-38** staan open met hun voorwaarde. DI-35 (de
   hernoeming in `inference.py`) wacht tot stage B van de zusterfase klaar is.

## 9. Wat `CHANGELOG.md` en `docs/PROJECT_STATE.md` na samenvoegen moeten zeggen

Deze fase heeft die twee bestanden niet bewerkt (§5.2). Na samenvoegen moeten ze
dit bevatten:

- **Nieuw:** `validation/breadth.py` (`independent_bets`, `dsr_hurdle`,
  `t_hurdle_sharpe`, `required_ic`, `simulate_wall`,
  `assert_ic_wall_declared`), `validation/signal_clock.py` (τ_int met
  Sokal-venster, `median_is_determined`), de campagnemodules
  `validation/phase11_breadth_measurement.py` en
  `validation/phase11_decision_clock_feasibility.py`, de app
  `apps/run_breadth_measurement.py` en de config `conf/research/breadth.yaml`.
- **Besluiten:** AD-29 (een pre-registratie noemt haar breedte, horizon en
  muur) en AD-30 (onafhankelijke weddenschappen en het ontwerpeffect zijn twee
  grootheden).
- **Uitgesteld:** DI-35 (hernoeming van het ontwerpeffect), DI-36 (twee
  vasthoudimplementaties, één onbruikbaar), DI-37 (AD-29 als poort in
  `freeze_preregistration`), DI-38 (ρ̄-breedte in de carry-units). DI-15 en
  DI-21 hebben nieuw bewijs gekregen, met ongewijzigde voorwaarden.
- **Hypothesen:** H-11.2 niet geregistreerd (haalbaarheidspoort rood), 0
  trials. De trialstand na deze fase is 5 + (zusterfase) van 25.
- **Stand van het programma:** de breedte van dit domein is gemeten (1,6
  directioneel, 4,4 dollar-neutraal), de IC-muur ligt boven alles wat het domein
  heeft laten zien, en de keuze tussen stoppen, verbreden binnen Bybit en een
  nieuw mandaat ligt bij de eigenaar (`reports/phase11_breadth_owner_decision.md`).
