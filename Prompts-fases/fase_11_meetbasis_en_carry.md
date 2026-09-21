# MASTER-PROMPT: FASE 11 — DE MEETBASIS DIE HET MANDAAT ONGELDIG MAAKTE, EN DE ENIGE KANDIDAAT DIE OVERBLIJFT

> **Status:** concept, ter vaststelling door de eigenaar.
> **Geschreven op 2026-09-20 tegen `f50f6ca`**, op een nulmeting en niet op
> aannames. Elk getal in §3 is op die datum gemeten in deze repository; waar
> een getal niet gemeten kon worden, staat dat er met de reden.
> **Voorganger:** `Prompts-fases/fase_10_herstart_dagbars.md` (stages A–C af,
> stage D half).
> **Taal:** dit document is Nederlands; code, docstrings, commits en
> artefactsleutels blijven Engels (R-9).

---

## 0. WAT DEZE FASE IS

Fase 10 heeft drie hypothesen beslist zonder het poortsample aan te raken, en
heeft daarmee gedaan wat zij moest doen. Tussen haar laatste meting en vandaag
is er echter iets gebeurd dat geen enkele van die metingen heeft meegemaakt: op
`f50f6ca` is het risicomandaat gewijzigd (AD-26), en die wijziging is de boom
in gekomen via een **automatische merge zonder conflict**. De merge-commit zegt
dat zelf, en dat is de reden dat dit document bestaat en niet iets anders.

Deze fase doet drie dingen, in deze volgorde, en zij mag ze niet omdraaien:

| | Stage | Wat het is | Trials |
|---|---|---|---:|
| **A** | **De meetbasis herstellen** | Elke gemeten uitspraak in deze repository is gedaan onder risicobeleid `1b60cb664fbf9a2a`. Het geldende beleid is `9961e1613bc907a5`. Zolang dat verschil er is, hoort bij geen enkel cijfer in `reports/` een configuratie die vandaag nog bestaat. | **0** |
| **B** | **De carrykandidaat** | Funding carry is de enige niet-gefalsificeerde kandidaat binnen het meetdomein, en het mandaat is expliciet gewijzigd om hem meetbaar te máken. Hij is nooit gemeten. | **≤ 8 van de 20 resterende** |
| **C** | **De opruiming die fase 10 openliet** | Stap 15.4/15.5, 16 en 17 zijn niet uitgevoerd; vier van fase 10's 32 exit-criteria staan daardoor open. | **0** |

### 0.1 Wat deze fase NIET is

**Zij is geen nieuwe zoektocht.** Er worden geen nieuwe families, geen nieuwe
featureblokken en geen nieuwe modelklassen geopend. Het budget dat stage B
mag besteden is een restant, geen nieuw krediet: fase 10 heeft 5 van de 25
bevroren trials gebruikt, en 20 is wat er ligt.

**Zij herijkt geen drempel op een uitkomst.** AD-26 is exogeen genomen en staat
vast. Stage A meet wat dat besluit met de bestaande cijfers doet; zij stelt het
besluit niet ter discussie en stelt het ook niet bij omdat een meting tegenvalt.
Dat is de enige onbeschadigde norm die fase 9 §9 aanwijst.

**Zij leest het poortsample niet, tenzij stage B daar met een positief
ontwikkelresultaat aankomt.** `holdout_lock.json` draagt vandaag nog
`reads: []`. Die ene lezing is er, hij is onbesteed, en hij wordt niet uitgegeven
aan een vraag die op de ontwikkelsample al gesloten is (R-7).

---

## 1. ROL EN CONTEXT

Je neemt een onderzoeks- en handelsplatform over dat rond één principe is
gebouwd: **een resultaat telt pas wanneer het de poort is gepasseerd die het had
kunnen tegenhouden.** Lees vóór de eerste regel code:

1. `docs/PROJECT_STATE.md` — wat waar is en wat aantoonbaar niet waar is.
2. `docs/MANDATE.md` + AD-22, AD-23, AD-24, AD-25, AD-26.
3. `docs/MEASUREMENT_CONTRACT.md` — het venster, de annualisatie, de SE's, de
   DSR-handtekening. Dit document herhaalt daar niets van; het verwijst ernaar.
4. `docs/MEASUREMENT_DOMAIN.md` + `conf/governance/measurement_domain.yaml` —
   drie bronnen, één frequentie.
5. `artefacts/governance/hypothesis_ledger.json`,
   `artefacts/governance/ledger_reset.json`, `artefacts/governance/holdout_lock.json`.
6. `docs/DEFERRED_ISSUES.md`, in het bijzonder DI-24 t/m DI-30.

Het universum is zes Bybit-perpetuals op dagbars. Er is op dit moment **geen
enkel model in deze repository waarvan is aangetoond dat het geld verdient.**
Dat blijft na deze fase mogelijk onveranderd, en "stop" is een toegestane
uitkomst.

---

## 2. DE BEVINDING DIE DEZE FASE DRAAGT

`artefacts/governance/risk_config_registry.json` kent drie geregistreerde
risicobeleidsregels:

| config_hash | git_sha | geregistreerd | `sigma_target` | `max_leverage` | `max_drawdown_pct` | `max_position_age_h` |
|---|---|---|---:|---:|---:|---:|
| `47821e47fe2cec30` | `417aaa6` | 2026-08-25 | 0,08 | 1,5 | 0,08 | 48 |
| `1b60cb664fbf9a2a` | `f4352e9` | 2026-08-25 | 0,08 | 1,5 | 0,08 | 48 |
| **`9961e1613bc907a5`** | `7f6181d` | **2026-09-12** | **0,20** | **4,0** | **0,25** | **720** |

`conf/risk/default.yaml` levert vandaag `9961e1613bc907a5`.
`artefacts/baseline/phase5_revaluation.json` — de vier-lagen-ladder waarop fase
10 haar drie hypothesen heeft beoordeeld — draagt op elke L3-rij
`risk_policy_hash: "1b60cb664fbf9a2a"`.

**Dit is geen administratief verschil.** De ladder is de meetlat: elke
L1/L2/L3-Sharpe, elke haltteller (1.566 van 1.743 bars), elke kostenuitsplitsing
en elke conclusie die daarop rust, is geproduceerd onder een risicolaag die op
0,08 vol-target en 1,5 leverage stond. Onder het geldende beleid staat dezelfde
laag op 0,20 en 4,0. Het boek dat de ladder meet, bestaat niet meer.

**En het is met de historie vast te stellen dat niemand het heeft nagerekend.**
`b18aeda` (de mandaatwijziging) is **geen voorouder** van `5dad1ab` (fase 10's
laatste meting):

```
git merge-base --is-ancestor b18aeda 5dad1ab   # exit 1 — NEE
```

De twee sporen liepen parallel en zijn op `f50f6ca` samengevoegd. De merge-commit
noemt dat ook: *"`conf/risk/default.yaml` merge-de AUTOMATISCH en nam de
branchwaarden over. Geen conflict, geen melding, geen review."*

### 2.1 Wat de repository er zélf van heeft gemerkt

Eén test. `tests/regression/test_dust_breaks_relative_limits.py` pint de
policy-hash en staat sinds de merge rood met exact de twee hashes uit de tabel
hierboven. De merge-commit heeft die test niet gezien, want zijn eigen
verificatie luidde `pytest tests/unit tests/integration` — en dit is
`tests/regression/`.

> **De les, en zij is groter dan de test.** Een risicoconfiguratie is in deze
> repository een gehashte, geregistreerde grootheid die met elk artefact
> meereist. Dat mechanisme heeft precies gedaan waarvoor het is gebouwd: het
> heeft de stille wijziging zichtbaar gemaakt. Het is alleen op één plek
> afgedwongen, en die plek stond buiten de verificatie die de wijziging
> begeleidde.

---

## 3. NULMETING (gemeten 2026-09-20 tegen `f50f6ca`)

Reproduceer deze tabel vóór je iets wijzigt. Wijkt jouw meting af, dan is dát je
eerste bevinding en die schrijf je op vóór je verdergaat (R-10).

**Omgeving van deze meting.** Python **3.11.15** met `requirements.lock` +
`requirements-dev.lock` (ruff 0.15.12, mypy 2.3.1 — de gepinde versies). De CI
draait op **3.13.15**. Dat verschil is DI-24 en het is hieronder waar het
uitmaakt expliciet aangegeven; waar de CI-logs van `f50f6ca` beschikbaar waren,
zijn beide metingen naast elkaar gezet.

### 3.1 De gedragsvingerafdruk

| | fase 10 §5.5 (`7507d6d`) | hier, lokaal 3.11 | CI 3.13 (`f50f6ca`) |
|---|---:|---:|---:|
| collected | 2.887 | **3.120** | — |
| passed | 2.860 | **3.087** | — |
| skipped | 23 | **25** | 14 posten |
| xfailed | 4 | **4** | 4 |
| xpassed | 0 | **0** | 0 |
| **failed** | **0** | **4** | **3** |

**Exit-criterium 4 van fase 10 eist 0 failed / 4 xfailed / 0 xpassed. Dat is
vandaag niet waar.** De vier lokale failures, met hun status:

| # | Test | Melding | Ook rood in CI? |
|---|---|---|:-:|
| 1 | `test_regime_overlay.py::…::test_a_uniform_factor_is_neutralised_by_the_vol_target` | 19.166,50 tegen 19.770,61 ± 197,71 (3,06 %) | **ja** |
| 2 | `test_dust_breaks_relative_limits.py::test_a_book_that_straddles_the_dust_tolerance_is_decided` | `'9961e1613bc907a5' == '1b60cb664fbf9a2a'` | **ja** |
| 3 | `test_docs_claim_only_what_exists.py::…::test_no_document_claims_a_path_that_does_not_exist` | `RISK_MANDATE.md` verwijst naar `conf/env/`, dat niet bestaat | **ja** |
| 4 | `test_regime_conditioning.py::…::test_the_multiplier_is_causal[hmm3-diag-student_t]` | `np.allclose(..., atol=0.0, rtol=0.0)` op twee arrays die identiek printen | **ja, intermitterend** |

> **Nummer 4 is niet stabiel, en dat is het defect.** Hij is een
> bit-gelijkheidstoets (`atol=0,0`, `rtol=0,0`) op een EM-fit uit `hmmlearn`,
> en hij geeft op identieke code niet steeds hetzelfde antwoord. Het bewijs
> staat in twee runs van **dezelfde job op dezelfde interpreter (3.13)**, met
> dezelfde testselectie:
>
> | `hygiene`-run | head | failures |
> |---|---|---|
> | job 105672758644 | `f50f6ca` | 2 — nummers 1 en 3, **zonder** hmm3 |
> | job 106152980220 | `80bc3dc` | 3 — nummers 1 en 3, **mét** hmm3 |
>
> Het verschil tussen die twee heads is één byte in een markdown-bestand plus
> één nieuw markdown-bestand. Dat kan een EM-fit niet raken.
>
> **De seed is het niet.** Lokaal op 3.11 faalt hij in de volledige suite,
> geïsoleerd met `-p no:randomly`, en bij `--randomly-seed` 1 t/m 6: zes van
> zes. De variatie zit dus niet in de testvolgorde of de RNG-seed, maar tussen
> omgevingen en tussen runs — precies de as waarop een bit-gelijkheidstoets
> geen marge heeft.
>
> **De eerste revisie van dit document noteerde hier "nee" in de CI-kolom en
> vermoedde een 3.11/BLAS-artefact.** Die lezing kwam van de CI-logs van
> `f50f6ca`, waarin hij inderdaad niet voorkomt — en zij is door de volgende
> run weerlegd. De meting is het antwoord (R-10), en de weerlegde voorspelling
> blijft hier staan omdat zij laat zien hoe een intermitterende poort zich
> voordoet: als een omgevingsverschil.
>
> **Wat stap 1.2 daarmee moet.** Niet `atol` oprekken, en niet de test
> markeren. Vaststellen *wat* er varieert: AD-17 legt vast dat het startpunt
> van elke EM **deterministisch** is en niet geseed, dus ofwel houdt die AD
> niet, ofwel leest er iets anders globale toestand. Een causaliteitstest die
> bij vlagen slaagt, bewijst geen causaliteit — hij meet ruis met een
> nulmarge.

**De CI meldt bovendien een blokkade die geen test is:**
`FAIL Required test coverage of 70.0% not reached. Total coverage: 60.66%`.

### 3.2 De poorten

Lokaal, met de gepinde tooling, alle zes exit 0:

| Poort | Uitkomst |
|---|---|
| `ruff check src/ apps/ tests/` | **All checks passed** |
| `mypy src/tradebot/schemas/ src/tradebot/utils/ apps/ --ignore-missing-imports` | **no issues in 64 source files** |
| `scripts/reachability_map.py --strict` | exit 0; klasse E = 3 geregistreerde modules |
| `scripts/check_file_size.py` | groen, 10 gecapte bestanden |
| `scripts/check_hardcoded_params.py --strict` | 298 literals, ratchet 313 |
| `scripts/audit_fallbacks.py --strict` | 39 advies, **0 blokkerend** |
| `scripts/check_banned_methods.py --strict` | 0 treffers |
| `scripts/check_domain_consistency.py` | 0 openstaande regels |

**En toch staat main rood.** De werkelijke stand van de workflows op `f50f6ca`:

| Workflow | Conclusie | Oorzaak |
|---|---|---|
| `research-gates` | **success** | — |
| `Docs` | success (op `371755e`) | — |
| `CI` | **failure** | de drie testfailures + coverage 60,66 % < 70 % |
| `hygiene` | **failure** | failures 1 en 3 |
| `inventory` | **failure** | de drie failures + dezelfde coverage-poort |
| `Security Scan` | **failure** | DI-26: `diskcache 5.6.3`, PYSEC-2026-2447, **zonder fixversie** — dit is een bewust rode poort |
| `Nightly Regression` | **failure** | DI-25: `gk_volatility.npy` en `sequential_bootstrap_indices.npy` ontbreken → *"De bit-equivalentietest heeft geen meetlat."* Plus een benchmarkjob die exit 5 geeft (nul tests verzameld) |
| `Paper-Trade Smoke` | cancelled | — |

> Twee van deze zeven zijn **eerlijk rood en horen rood te blijven** tot de
> eigenaar beslist (DI-25, DI-26). De andere vijf zijn werk.

### 3.3 De ladder, en onder welk beleid hij is gemeten

`artefacts/baseline/phase5_revaluation.json`, `git_sha 8d62d5c`, 1.743 bars
(2021-11-15 → 2026-08-23), **`risk_policy_hash 1b60cb664fbf9a2a` op elke
L3-rij** — dus niet het geldende beleid:

| Track | L0 | L1 + risk | L2 + latency | L3 + execution | `n_sovereign_halted` | `n_sovereign_clipped` |
|---|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | +0,0390 | +0,1192 | +0,1357 | **−0,7192** | 1.566 | 1.742 |
| `long_only_risk_parity` | +0,0125 | +0,1198 | +0,1304 | **−0,7447** | 1.566 | 1.742 |
| `xs_momentum_equal_weight` | +0,0985 | +0,2147 | −0,0516 | **−0,1681** | 0 | 1.742 |
| `xs_momentum_risk_parity` | −0,2267 | −0,0050 | −0,2633 | **−0,3691** | 0 | 1.742 |

> **Twee L3-cellen wijken af van fase 10 §5.2**, dat −0,695 en −0,721 noteert
> waar het artefact vandaag −0,7192 en −0,7447 geeft. Dat is geen tegenspraak
> maar een datering: §5.2 is gemeten tegen `5ae4a00` (2026-09-04) en het
> artefact is op `8d62d5c` (2026-09-11, fase 10 stap 8) opnieuw geschreven.
> `5ae4a00` is een voorouder van `8d62d5c`, dus de nieuwere waarden gelden.
> Fase 10 heeft haar eigen nulmeting dus al een keer ingehaald zonder dat §5.2
> is bijgewerkt — en dat is dezelfde klasse fout als die waar deze hele fase
> over gaat, alleen een orde kleiner.

**Let op de laatste kolom: de soevereine laag grijpt op 1.742 van 1.743 bars
in, op alle vier de tracks.** Welke limiet daar bindt, staat nergens
opgeschreven. Dat is dezelfde vraag als DI-27 en zij is nu meetbaar gemaakt door
de mandaatwijziging: `mean_gross` op L1 is 0,113 / 0,121 / 0,111 / 0,118 en de
gerealiseerde volatiliteit 0,0751 / 0,0761 / 0,0254 / 0,0227 — tegen een
`sigma_target` die toen 0,08 was en nu 0,20 is.

**De kostenuitsplitsing op L3**, en dit is het economische feit dat stage B
draagt:

| Track | fees | **funding** | impact | spread | aandeel funding |
|---|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | 38,36 | **261,67** | 20,21 | 6,97 | **80,0 %** |
| `long_only_risk_parity` | 43,06 | **279,72** | 18,34 | 7,83 | **80,2 %** |
| `xs_momentum_equal_weight` | 1.987,05 | 761,41 | 1.267,59 | 361,28 | 17,4 % |
| `xs_momentum_risk_parity` | 2.100,91 | 397,39 | 1.204,34 | 381,98 | 9,7 % |

Op de twee long-only-tracks is **funding vier vijfde van alle kosten**. Funding
wordt op de *positie* betaald en niet op de omzet — daarom bewoog hij in H-10.1
niet mee met de beslisfrequentie, en daarom is hij de grootste economische kracht
in dit boek. Het boek staat aan de betalende kant.

### 3.4 Fase 10's exit-criteria, stand vandaag

Van de 32 criteria zijn er vier aantoonbaar niet waar, en één is niet
beoordeelbaar:

| # | Criterium | Stand | Bewijs |
|---|---|---|---|
| 4 | suite meldt 0 failed / 4 xfailed / 0 xpassed | **onwaar** | §3.1 |
| 17 | `gate_by_state` overleeft vol-targeting | **n.v.t.** | H-10.3 is LAPSED; er is geen `gate_by_state` geschreven |
| 19 | concentratie na de poort begrensd door `max_weight_per_symbol` | **onwaar** | die sleutel bestaat niet; stap 17 is niet uitgevoerd |
| 28 | dagelijkse runner bestaat en is gedragsequivalent, AD is geschreven | **onwaar** | `apps/run_daily_decision.py` bestaat niet; `src/tradebot/live/` telt 13 modules |
| 30 | AD-2 en AD-3 zijn `PERMANENT` met een `Consequence`-sectie | **onwaar** | beide staan op *"actief tot er orderboek-/quote-data is"* |
| 32 | geen app > 80 LOC | **onwaar** | **33 van de 47 apps** staan boven 80 (grootste: `train_cpcv.py` 974) |

Criterium 26 (*elke buiten-domein-module heeft een verdict*) is **waar** — de
tabel in `docs/MEASUREMENT_DOMAIN.md` is volledig — maar **de verdicts zijn niet
uitgevoerd.** Stap 15.4 is niet gedraaid: er is geen `archive/`-directory en alle
zes de `ARCHIVED`-modules staan nog in `src/`:

| Module | verdict | LOC |
|---|---|---:|
| `features/microstructure.py` (de gearchiveerde helft) | ARCHIVED | 180 |
| `alpha/microstructure.py` | ARCHIVED | 106 |
| `bars/runs.py` | ARCHIVED | 254 |
| `bars/imbalance.py` | ARCHIVED | 175 |
| `bars/dollar.py` | ARCHIVED | 173 |
| `data/orderbook.py` | ARCHIVED | 150 |
| | | **1.038** |

### 3.5 Het trialbudget en het poortsample

| Grootheid | Waarde | Bron |
|---|---:|---|
| `archived_total` | 2.776 | `ledger_reset.json` |
| `m_new` (bevroren budget) | **25** | idem |
| besteed in fase 10 stage C | **5** (H-10.1: 4 · H-10.2: 1 · H-10.3: 0) | ledger-notities |
| **resterend** | **20** | — |
| poortsample-lezingen | **0** van 1 | `holdout_lock.json`: `reads: []` |
| vereiste ann. Sharpe voor DSR ≥ 0,95 bij M=25, N=1390 | **1,8686** | H-10.1-artefact |
| t = 2-drempel op W_DEV | **1,0249** | `MEASUREMENT_CONTRACT.md` §2 |

### 3.6 De carry, voor het eerst binnen het domein gemeten

Nieuw in dit document, en bewust **diagnostisch**: geen selectie, geen
parameterkeuze, nul trials. Gemeten rechtstreeks op de gecertificeerde store
(`data/pit_store`, datasets `funding` en `ohlcv`) over **W_DEV**
(2021-11-15 → 2025-09-04, 1.390 bars), funding per dag gesommeerd over de
8h-ticks. **De reproductie staat in bijlage A.**

| Symbool | carry (bps/dag) | sd | frac. dagen > 0 | prijs (bps/dag) | short bruto = carry − prijs | ann. Sharpe (short, bruto) |
|---|---:|---:|---:|---:|---:|---:|
| AVAXUSDT | +1,631 | 4,93 | 0,760 | −9,765 | **+11,39** | 0,422 |
| BTCUSDT | +2,139 | 3,03 | 0,869 | +3,979 | −1,84 | −0,126 |
| DOTUSDT | +1,280 | 4,02 | 0,727 | −17,835 | **+19,10** | 0,845 |
| ETHUSDT | +2,042 | 3,48 | 0,843 | −0,442 | +2,48 | 0,128 |
| LINKUSDT | +2,717 | 3,00 | 0,915 | −2,614 | +5,33 | 0,214 |
| SOLUSDT | −0,981 | 39,06 | 0,750 | −1,183 | +0,20 | 0,007 |

En de twee getallen waar deze fase op draait:

```
gelijkgewogen carrymandje   : +1,471 bps/dag · sd 7,67 · ann. Sharpe  3,665
gelijkgewogen prijsmandje   : −4,643 bps/dag · sd 388,78
verhouding sd prijs / carry : 50,7 x
```

Daar staan drie dingen tegelijk, en ze moeten alle drie worden gelezen:

1. **De carrystroom zelf is buitengewoon glad.** Ann. Sharpe 3,665 — ruim boven
   de DSR-drempel van 1,8686. In deze nulmeting is dat de enige grootheid die
   boven die drempel uitkomt; de beste gemeten track staat op +0,2147 (L1,
   `xs_momentum_equal_weight`) en het ruismaximum van een zoektocht over 55
   cellen lag in fase 10 §5.7 op +1,10.
2. **Het is geen verhandelbaar rendement.** Om die stroom te ontvangen moet je
   short staan, en de prijsrisico-standaarddeviatie is **50,7 keer** die van de
   carry. Wat je zonder hedge meet is een richtingswed met een carrygarnering.
3. **Over precies dit venster was short zijn op zichzelf al winstgevend.** Het
   gelijkgewogen prijsmandje deed −4,64 bps/dag, dus de ongehedgede short
   verdient gemiddeld ≈ +6,11 bps/dag, waarvan de carry **24 %** is. De overige
   76 % is de prijsdaling — en niemand heeft het teken van die daling vooraf
   geregistreerd. **Een ongehedged carryboek dat op W_DEV wint, is tot het
   tegendeel is bewezen een berenmarktartefact.**

**Kostenreferentie**: `conf/execution/fees.yaml` geeft taker 5,5 bps en
`assumed_half_spread_bps` 1,0 → **13,0 bps vaste kosten per round trip**. Bij
1,471 bps/dag betekent dat **8,8 dagen** houden om alleen die vaste kosten terug
te verdienen, impact niet meegerekend. `max_position_age_h` staat sinds AD-26 op
720 h (30 dagen); onder de oude 48 h was dit per constructie onmogelijk, en dát
is precies wat AD-26 stelt te hebben opgelost.

#### 3.6.1 Twee data-bevindingen die vóór stage B moeten worden opgelost

**(a) 41 dagen dragen niet drie funding-records.** Alle 41 zijn `SOLUSDT`: 39
dagen met 12 records, één met 9, één met 6. Zij liggen geconcentreerd rond
november 2022. `funding_interval_hours` staat op **elke** rij in de store op 8.
Optellen alsof het alle 8h-betalingen zijn geeft op 2022-11-10 een dagcarry van
**−1.236 bps**. Laat je SOL's drie extreemste dagen weg, dan gaat het
symboolgemiddelde van **−0,981** naar **+0,463** bps/dag en de sd van 39,06 naar
13,30. SOL's hele negatieve teken hangt aan die dagen.

Dit is geen uitbijterprobleem dat je wegwinsoriseert. Het is een vraag met een
feitelijk antwoord: **is `funding_interval_hours` op die rijen onjuist, of staan
er dubbelingen in de store?** Beantwoord hem tegen de bron voordat er één
carrygetal wordt gerapporteerd.

**(b) De repository heeft twee funding-routes en de ene is onbereikbaar.**
`src/tradebot/data/funding.py` (129 LOC) staat in `docs/CODE_REGISTER.md` als
**klasse E, geregistreerd onbereikbaar**; `features/funding_carry.py` noemt
`load_per_bar_funding_rate` alleen in zijn docstring. De feature-route is
`features/positioning.py`, die met een **backward asof-join de laatst bekende
rate** koppelt — niet de dagsom. De backtestkosten komen uit
`backtest/evaluation.py::compute_bar_by_bar_mtm`, dat een `funding_rates`-array
van de aanroeper krijgt. **Stel vast welke reeks de 261,67 aan L3-fundingkosten
heeft geproduceerd en of dat de gecertificeerde reeks is.** Twee definities van
dezelfde grootheid is een defect, ook als ze hetzelfde getal geven (R-3).

### 3.7 Wat de fase-9-diagnose werkelijk is

Kandidaat B uit fase 9 §6.3 — en daarmee de motivering onder AD-26's
`max_position_age_h: 720` — steunt op `reports/diag_funding_short_edge.csv`. Dat
bestand is **niet reproduceerbaar in deze repository en valt buiten het
meetdomein**, om drie onafhankelijke redenen:

1. **De bron bestaat hier niet.** `research/diag_funding_short_edge.py` leest
   `market_data_parquet/`. Die map staat op regel 67 van `.gitignore` en is in
   deze checkout afwezig. Het script kan niet draaien.
2. **De waarneming is fijner dan de dagbar.** Het script resample't
   **5-secondebars** naar 8h/24h/72h-horizonnen. AD-23 kent drie bronnen op één
   frequentie; dit is er geen van.
3. **Het rekent met een derde kostentabel.** `COST_BPS_PER_ASSET` in dat script
   zet ETH 3,0 · SOL 4,0 · AVAX 6,0 · LINK 6,0 · DOT 5,0 · BTC 2,0 bps round
   trip, tegen de 13,0 bps uit `conf/execution/fees.yaml`. Zijn `mean_ret` is al
   **netto** volgens die tabel. De getallen zijn dus niet vergelijkbaar met enig
   ander kostencijfer in deze repository.

Bovendien kent het vijf symbolen; BTCUSDT ontbreekt. Ter vergelijking, `all`-bucket,
carry omgerekend naar bps/dag, naast de in-domein-meting uit §3.6:

| Symbool | fase-9-CSV (5s-bron) | in-domein (PIT-store, W_DEV) |
|---|---:|---:|
| ETHUSDT | 1,96 | **2,43** |
| SOLUSDT | 0,48 | **−0,98** (zie §3.6.1a) |
| AVAXUSDT | 1,33 | **1,63** |
| LINKUSDT | 2,54 | **2,72** |
| DOTUSDT | 0,20 | **1,28** |
| BTCUSDT | — | **2,14** |

> **Dit weerlegt kandidaat B niet.** Het stelt vast dat hij nog nooit binnen het
> domein is gemeten, en dat de rekensom die in `conf/risk/default.yaml` staat
> (*"~1,95 bps/dag … ruim zeven dagen"*) uit een bron komt die AD-23 niet kent.
> Stage B meet hem opnieuw, vanaf de gecertificeerde store, met de kostenbasis
> van dit project. Wat er in dat commentaarblok staat is vanaf nu een te toetsen
> verwachting en geen vastgestelde grootheid (R-10).

---

## 4. DE DRIE VRAGEN VAN DEZE FASE

**V1 — Wat meet de ladder onder het geldende risicobeleid?**
Niet: "is de ladder beter geworden." De vraag is of de vier tracks onder
`9961e1613bc907a5` dezelfde uitspraken dragen als onder `1b60cb664fbf9a2a`, en
welke limiet op 1.742 van 1.743 bars bindt.

**V2 — Bestaat er een houdduur waarop funding carry, binnen het meetdomein en
met de kostenbasis van dit project, netto positief is — en overleeft dat de
scheiding tussen carry en prijsrichting?**
Bestaat die houdduur niet, dan is kandidaat B gefalsificeerd en is dat een
volwaardig resultaat (fase 9 §9). Bestaat hij wél maar uitsluitend ongehedged,
dan is de bevinding een richtingswed en géén carry-edge, en dan wordt hij als
zodanig geboekt.

**V3 — Wat blijft er van fase 10's stage D staan, en wat gaat eruit?**
De verdicts zijn geveld; zij zijn niet uitgevoerd. 1.038 regels dragen het label
`ARCHIVED` en staan nog in `src/`.

---

## 5. FENCES — WAT DEZE FASE NIET AANRAAKT

| Raak niet aan | Waarom |
|---|---|
| `docs/FALSIFICATION_REGISTER.md` F1–F20 | Onverkort bindend; de ledger-reset kocht dat expliciet af (AD-24 R2). Alleen een domeinmarkering mag worden toegevoegd. |
| `artefacts/governance/hypothesis_ledger.json`, bestaande entries | Append-only. Een oordeel wordt geamendeerd, nooit herschreven. |
| `holdout_lock.json` | Eén lezing, en alleen langs de route van stage B.4. |
| `m_new = 25` | Bevroren budget. Het wordt niet opgehoogd omdat er werk bij komt. |
| `conf/risk/default.yaml`, de waarden | AD-26 is exogeen. Stage A **meet** het gevolg; zij stelt geen getal bij. Blijkt een limiet aantoonbaar inert, dan is dat een bevinding voor de eigenaar en geen wijziging. |
| `adv_participation_cap`, `max_concentration` | Soort C resp. de enige werkzame spreidingsbescherming (AD-4, RISK_MANDATE §1). |
| De vier xfail-killgates | Zij toetsen alpha van `cm_carry`/`cm_tsmom` (energie-termijnstructuur, F20) — een ándere carry dan die van stage B. Niet aanraken, niet "meenemen". |
| `tests/regression/baselines/` | DI-25 is een eigenaarsbesluit over waar de bevroren meetlat woont. De poort blijft rood tot dat besluit valt. |
| De ruff-/mypy-versies | Gepind om precies de reden die DI-16 beschrijft. |

---

## 6. CONCRETE DELIVERABLES

| Stage | Artefact | Vorm |
|---|---|---|
| A | `reports/phase11_measurement_basis.md` | de her-afleiding, met beide policy-hashes naast elkaar |
| A | `artefacts/baseline/phase11_revaluation.json` | de ladder onder `9961e1613bc907a5` |
| A | `reports/phase11_binding_constraint.md` | welke limiet bindt op welk aandeel van de bars, per track en per laag |
| A | AD-27 | *"Elke meting draagt de policy-hash van de configuratie die haar produceerde"* |
| A | `docs/DEFERRED_ISSUES.md` | DI-27 en DI-30 gesloten óf met een gemeten voorwaarde doorgeschoven |
| B | `conf/research/preregistration_h11_funding_carry.yaml` | bevroren vóór de eerste fit |
| B | `artefacts/governance/phase11_h11_carry.json` | de campagne |
| B | `reports/phase11_h11_funding_carry.md` | het oordeel, met de carry/prijs-decompositie |
| B | ledger-amendement | verdict, `m_new`, `deflated_sharpe`, `t_hurdle_at_n` |
| C | `apps/run_daily_decision.py` | ≤ 80 LOC (R-6) |
| C | `archive/README.md` per gearchiveerde module | reden + heropeningsvoorwaarde |
| C | AD-2 / AD-3 | `PERMANENT`, met `Consequence` |
| C | `reports/phase11_exit_report.md` | de afvinklijst van §EXIT-CRITERIA |

---

# STAPSGEWIJZE UITVOERING

# STAGE A — DE MEETBASIS

*Geen hypothesen, nul trials. Deze stage herstelt de verbinding tussen wat er
gemeten is en de configuratie die vandaag geldt.*

---

### Stap 1: De vingerafdruk vastleggen en de vier failures scheiden

- [ ] **1.1 — Draai de volledige suite en leg de vingerafdruk vast** vóór enige
  wijziging: collected / passed / skipped / xfailed / xpassed / failed, plus de
  interpreterversie en de ruff-/mypy-versies. Schrijf hem in
  `reports/phase11_measurement_basis.md` §1. Zonder dit startpunt is niets in
  deze fase aantoonbaar gedragsneutraal.

- [ ] **1.2 — Stel per failure vast of hij van deze repository is.** Draai de
  vier tests uit §3.1 apart, met `-p no:randomly`. Failure 4 is aantoonbaar
  **intermitterend** (§3.1), en de seed is uitgesloten: zoek de variatie in de
  omgeving (BLAS-implementatie, threadaantal, `hmmlearn`/`scipy`-build) door hem
  op 3.13 herhaald te draaien, met `OMP_NUM_THREADS=1` naast de
  standaardinstelling. Boek het resultaat als een
  DI — een bit-gelijkheidstoets op een niet-bit-stabiele berekening is een
  defect van de **poort**, ook wanneer de onderliggende causaliteit klopt.
  Wat NIET mag: `atol` oprekken, de test markeren, of hem overslaan.

- [ ] **1.3 — Repareer failure 3, en alleen die.** `docs/RISK_MANDATE.md`
  verwijst naar `conf/env/`, dat niet bestaat. Corrigeer de verwijzing, of zet
  hem met een reden in `KNOWN_ABSENT` wanneer het document juist uitlegt dat het
  pad weg is. Dit is de enige failure die vóór de rest van stage A weg kan,
  omdat hij niets met de meetbasis te maken heeft.

- [ ] **1.4 — Raak failures 1 en 2 nog NIET aan.** Zij zijn de meetbasis zelf en
  worden in stap 2 en 3 beantwoord. Een test groen maken vóór je weet wat hij
  zegt, is precies de volgorde die dit project verbiedt.

- [ ] **1.5 — Commit.**

---

### Stap 2: De policy-hash wordt een poort, niet een veld

`test_dust_breaks_relative_limits.py` is vandaag het enige dat de
beleidswijziging heeft opgemerkt, en hij merkte hem op omdat iemand daar een
hash had vastgepind. Dat is toeval van dekking, geen mechanisme.

**Files:**
- Create: `tests/unit/test_measurement_carries_policy_hash.py`
- Modify: `src/tradebot/registry/` (de plek waar de hash wordt geregistreerd)
- Modify: `docs/ARCHITECTURAL_DECISIONS.md` (AD-27)

- [ ] **2.1 — Inventariseer wie de hash draagt.** Meet, per artefact in
  `artefacts/` en `reports/`, of er een `risk_policy_hash` in staat en welke.
  Leg de tabel vast. Gemeten op 2026-09-20 draagt `1b60cb664fbf9a2a` ten minste
  `phase5_revaluation.json`, `phase6_h2_regime_benchmark.json`,
  `phase6_h3_meta_labeling.json`, `preregistration_3d3af28…json` en vier
  rapporten; `9961e1613bc907a5` draagt alleen `phase4_stress.json` en het
  register zelf. Reproduceer dat, want het is jouw uitgangspunt.

- [ ] **2.2 — Schrijf de test eerst (R-11).** Eis: een artefact dat een
  risicobesluit bevat zonder de `config_hash` van de configuratie die het
  produceerde, is een `DataContractError`. Met een **negatieve controle** die
  bewijst dat de poort rood kan worden: voer een artefact met de verkeerde hash
  op en zie hem falen.

- [ ] **2.3 — Beslis wat er met de verouderde artefacten gebeurt, en schrijf het
  besluit op.** Er zijn twee geldige uitkomsten en het is een keuze:
  (a) elk artefact onder `1b60cb664fbf9a2a` krijgt een `SUPERSEDED_BY_POLICY`-
  markering en blijft staan als herkomst; of (b) het wordt opnieuw afgeleid.
  **Doe (a) voor alles en (b) alleen voor de ladder** — de ladder is de meetlat
  en die moet actueel zijn; een afgesloten hypothese opnieuw draaien is een
  besluit over een gesloten oordeel en valt buiten deze fase (zie DI-22).

- [ ] **2.4 — Repareer `test_dust_breaks_relative_limits`** door de hash uit het
  **register** af te leiden in plaats van hem te herschrijven naar de nieuwe
  literal. Een test die een hash hardcodeert, meet volgende keer weer toevallig.

- [ ] **2.5 — Schrijf AD-27**: *"Elke meting draagt de policy-hash van de
  configuratie die haar produceerde."* Afgewezen alternatief: de hash alleen in
  het register bijhouden — dat is precies wat er stond, en het heeft een
  mandaatwijziging door de hele meetbasis laten lopen zonder één melding.

- [ ] **2.6 — Commit.**

---

### Stap 3: Welke limiet bindt, en bindt de vol-target nog?

Dit is DI-27, DI-30 en failure 1 uit §3.1 — drie formuleringen van één vraag.

**Files:**
- Create: `apps/run_binding_constraint_audit.py` (≤ 80 LOC)
- Create: `src/tradebot/risk/binding_audit.py`
- Create: `tests/unit/test_binding_audit.py`
- Create: `reports/phase11_binding_constraint.md`

- [ ] **3.1 — Meet, per bar en per track, welke stap uit `constraint_order` het
  boek daadwerkelijk heeft veranderd.** `constraint_order` telt tien stappen
  (`halted` → `net_cap`). De uitvoer is een tabel: per limiet het aantal bars
  waarop hij bond, en de gemiddelde fractie waarmee hij het boek verkleinde.
  Zonder dat is "1.742 van 1.743 bars geclipt" een getal zonder inhoud.

- [ ] **3.2 — Doe dat onder BEIDE policies.** `1b60cb664fbf9a2a` en
  `9961e1613bc907a5`, op dezelfde bars, met dezelfde exposures. Het verschil
  tussen die twee kolommen **is** het gevolg van AD-26, en het is de eerste keer
  dat iemand dat meet.

- [ ] **3.3 — Beantwoord de vol-targetvraag expliciet.** De soevereine laag
  schaalt met `min(max_leverage, sigma_target / sigma_boek)`. Rapporteer op hoeveel
  bars `sigma_target / sigma_boek < max_leverage` (de vol-target bindt) en op
  hoeveel niet. Verwachting op grond van §3.3 — en dus toetsbaar en weerlegbaar:
  bij `sigma_boek ≈ 0,71` geannualiseerd geeft 0,20/0,71 ≈ 0,28, ruim onder 4,0,
  dus de vol-target **hoort** te binden en het boek hoort ~2,5× groter te zijn
  dan onder 0,08. Meet of dat zo is.

- [ ] **3.4 — Verklaar de 3,06 % uit failure 1.** Halveren van alle exposures
  hoort onder een bindende vol-target **exact** te worden teruggeschaald; de
  turnover-assertie in die test slaagt ook nog, en alleen de gross notional wijkt
  3,06 % af. Er is dus een limiet die *na* de vol-target staat en niet
  schaal-invariant is. `constraint_order` zet vier caps na `vol_target`
  (`per_asset_cap`, `adv_cap`, `concentration_cap`, `cluster_cap`) plus
  `gross_cap` en `net_cap`. **Wijs de verantwoordelijke aan met een meting, niet
  met een redenering.**

- [ ] **3.5 — Sluit DI-30 met een van twee uitkomsten**, en schrijf welke:
  (a) REGEL V geldt niet meer in de algemene vorm, en dan wordt AD-25 en de
  H2-rapportage bijgewerkt met de conditie waaronder hij wél geldt; of
  (b) REGEL V geldt nog en de 3,06 % is een aanwijsbare limiet, en dan wordt de
  test **preciezer** gemaakt — hij toetst dan de invariantie op de laag waar zij
  bestaat. Wat NIET mag: `rel=0.01` oprekken tot hij groen is.

- [ ] **3.6 — Commit.**

---

### Stap 4: De ladder opnieuw afleiden

- [ ] **4.1 — Draai `apps/run_phase5_baseline.py` onder het geldende beleid.**
  Exact dezelfde vier tracks, hetzelfde venster, dezelfde vier lagen. **Eén keer.**
  Er wordt geen variant geprobeerd, geen parameter gedraaid en niets gekozen; dit
  is een her-afleiding onder een exogeen gewijzigde configuratie en zij kost
  daarom nul trials (R-2). Registreer die redenering in het artefact, zodat een
  latere lezer kan controleren dat het geen zoektocht was.

- [ ] **4.2 — Schrijf `artefacts/baseline/phase11_revaluation.json`** met
  `risk_policy_hash: 9961e1613bc907a5` en met `supersedes` dat naar
  `phase5_revaluation.json` wijst. Het oude bestand blijft staan.

- [ ] **4.3 — Zet de twee ladders naast elkaar** in
  `reports/phase11_measurement_basis.md`, met per cel het verschil en met de SE
  van elke Sharpe ernaast (R-8). Elke conclusie van fase 10 die op een cel rust
  die significant verschuift, wordt bij naam genoemd.

- [ ] **4.4 — Trek de halt- en clipping-tellers opnieuw.** Onder 0,25 drawdown
  in plaats van 0,08 hoort `n_sovereign_halted` te dalen — de 1.566 gehalteerde
  bars komen van een halt op 8,31 % drawdown op 2022-05-10 (fase 10 §5.2; het
  aantal is in H-10.1 gecorrigeerd van 1.559 naar 1.566). Daalt hij, dan
  verandert het **actieve** venster van de L1+-Sharpes, en dan verandert de
  t-drempel waartegen zij worden beoordeeld. Reken die drempel opnieuw uit en
  noem hem. Fase 10 §5.2 wees hier al op: 184 actieve bars vragen een
  geannualiseerde Sharpe van 2,83 voor t = 2.

- [ ] **4.5 — Commit.**

---

### Stap 5: De funding-route ontdubbelen

- [ ] **5.1 — Stel vast welke reeks de L3-fundingkosten produceert.** Volg
  `backtest/evaluation.py::compute_bar_by_bar_mtm` terug naar zijn aanroeper en
  naar de reeks die hij als `funding_rates` krijgt. Vergelijk die met de
  gecertificeerde `funding`-dataset uit `data/pit_store`.

- [ ] **5.2 — Beslis over `data/funding.py`** (klasse E, 129 LOC, geregistreerd
  onbereikbaar). Twee uitkomsten: hij wordt de enige route en de andere verdwijnt,
  of hij verdwijnt. Een tweede implementatie van dezelfde grootheid is een defect,
  ook als zij hetzelfde getal geeft (R-3).

- [ ] **5.3 — Beantwoord §3.6.1(a).** 41 SOLUSDT-dagen dragen 6, 9 of 12
  funding-records terwijl `funding_interval_hours` overal 8 zegt. Stel tegen de
  bron vast of het interval onjuist is of de store dubbelingen bevat, en leg het
  antwoord vast in `docs/DATA_REGISTER.md`. **Er wordt geen rij verwijderd en
  geen waarde gewinsoriseerd** voordat dat antwoord er is.

- [ ] **5.4 — Schrijf de causaliteitstest met negatieve controle (R-1)** voor de
  dagaggregatie van funding: de waarde op bar `t` mag uitsluitend ticks ≤ `t−1`
  gebruiken. De negatieve controle moet aantoonbaar rood worden wanneer de tick
  van de eigen bar meedoet.

- [ ] **5.5 — Commit.**

---

# STAGE B — H-11.1: DE CARRYKANDIDAAT

*Eén hypothese. Zes vooraf geregistreerde trials van de twintig die er nog zijn.
Stage B begint pas wanneer stage A is afgerond, omdat haar kostenkant en haar
risicolaag daar worden vastgesteld.*

---

### Stap 6: De hypothese, en waarom zij zo is geformuleerd

> **H-11.1.** Er bestaat een houdduur `h` in dagbars waarop een **prijsneutraal
> geconstrueerd** carryboek op de zes perpetuals, na de kostenbasis van dit
> project, een netto rendement draagt waarvan het 95 %-interval nul uitsluit én
> waarvan de gedefleerde Sharpe de drempel bij `M_new = 25` haalt.

Drie woorden in die zin doen het werk:

**"prijsneutraal geconstrueerd."** §3.6 meet dat de prijscomponent een
standaardafwijking heeft die **50,7 keer** die van de carry is, en dat over W_DEV
**76 %** van het bruto rendement van een ongehedged short de prijsdaling is. Een
carryclaim die dat niet scheidt, meet de berenmarkt van 2022–2024. De hypothese
staat of valt bij de scheiding, en daarom zit zij in de formulering.

**"na de kostenbasis van dit project."** 13,0 bps vast per round trip
(`conf/execution/fees.yaml`) plus de impactterm uit `conf/execution/impact.yaml`
met status `IMPACT_UNCALIBRATED`. Niet de tabel uit
`research/diag_funding_short_edge.py` (§3.7). Eén kostendefinitie (R-3).

**"gedefleerde Sharpe bij `M_new = 25`."** De drempel is de bevroren
budgetwaarde, niet de lopende teller — dezelfde lezing die H-10.1 hanteerde. Op
W_DEV (N = 1390) vraagt `DSR ≥ 0,95` een geannualiseerde Sharpe van **1,8686**.

> **Wat vooraf vaststaat en wat de meting mag weerleggen (R-10).** Het
> carrymandje meet zelfstandig ann. Sharpe **3,665** — boven die drempel. Dat is
> géén voorspelling dat H-11.1 slaagt: het is de betalingsstroom zonder de
> positie die haar ontvangt, zonder kosten en zonder hedge. De verwachting die
> hier wordt geregistreerd is dat **de hedge en de kosten het verschil zijn
> tussen 3,665 en het antwoord**, en dat de meting dat verschil uitrekent in
> plaats van het te schatten.

- [ ] **Stap 6.1 — Schrijf de pre-registratie en bevries haar vóór de eerste
  fit.** `conf/research/preregistration_h11_funding_carry.yaml`, bevroren met
  `apps/freeze_preregistration.py`. Zij bevat minimaal:

  | Veld | Inhoud |
  |---|---|
  | `primary_cell` | de ene cel waarop het oordeel valt, vóór de meting benoemd |
  | `holding_periods` | `h ∈ {3, 5, 10}` dagbars |
  | `constructions` | 2: `cross_sectional_dollar_neutral` en `beta_hedged` |
  | `planned_trials` | **6** (3 × 2); `assert_within_budget(11)` tegen `m_new = 25`, `remaining = 14` |
  | `window` | `W_DEV` — 1.390 bars. `W_GATE` wordt niet aangeraakt |
  | `adequacy_floor` | ≥ 100 **niet-overlappende** perioden per symbool per cel, én de ρ̄-gedefleerde effectieve telling ernaast (AD-20: het oordeel valt op de conservatieve) |
  | `stop_criteria` | de vijf uit stap 8, elk met zijn vooraf vastgelegde actie |
  | `negative_controls` | de twee uit stap 7.4 |

- [ ] **Stap 6.2 — Reken de adequaatheid vooraf uit en leg hem vast.** W_DEV
  geeft per symbool `floor(1390/h)` niet-overlappende perioden: **463** bij
  h = 3, **278** bij h = 5, **139** bij h = 10. Alle drie boven de vloer van 100;
  `h = 20` zou **69,5** geven en is daarom **niet** gepland. Dat is de reden dat
  de rooster daar ophoudt, en niet dat 20 oninteressant is.

  > Dit is de les van H-10.3, vooraf toegepast: een poort waarvan vooraf
  > uitrekenbaar is dat zij niet gehaald kán worden, hoort niet in een
  > pre-registratie. Reken hem uit vóór je hem registreert.

- [ ] **Stap 6.3 — Commit de bevroren pre-registratie, apart.**

---

### Stap 7: De constructie

**Files:**
- Create: `src/tradebot/alpha/funding_carry_book.py`
- Create: `tests/unit/test_funding_carry_book.py`
- Create: `tests/lookahead/test_funding_carry_causality.py`
- Create: `apps/run_h11_funding_carry.py` (≤ 80 LOC)

- [ ] **7.1 — De signaalkant is causaal en parameterloos.** De carry die op bar
  `t` bekend is, is de som van de funding-ticks tot en met `t−1`, langs de route
  die stap 5 als enige heeft overgehouden. Geen z-score-drempel, geen venster
  dat gekozen moet worden: **elke drempel is een parameter en elke parameter is
  een trial** (R-2). De onconditionele vorm is de goedkoopste toets en zij is de
  enige die in het budget past.

- [ ] **7.2 — Arm 1: `cross_sectional_dollar_neutral`.** Rangschik de zes namen
  op de bekende carry, ga short in de bovenste helft en long in de onderste,
  dollar-neutraal. De marktrichting valt eruit per constructie; wat overblijft is
  het **verschil** in carry tussen namen. Let op wat §3.6 daarover al zegt: de
  spreiding tussen symbolen is 1,28 tot 2,72 bps/dag (SOL buiten beschouwing tot
  §3.6.1a beantwoord is), dus het te verhandelen verschil is **kleiner** dan de
  gemiddelde carry zelf. Dat is de prijs van neutraliteit en hij hoort in het
  rapport.

- [ ] **7.3 — Arm 2: `beta_hedged`.** Ontvang de carry op de volledige mand en
  hedge de prijscomponent met een causaal geschatte bèta tegen het
  gelijkgewogen mandje. De bèta wordt geschat op een **expanding** venster met
  uitsluitend data ≤ `t−1`; een rollend venster met een te kiezen lengte is een
  parameter en dus een trial.

- [ ] **7.4 — Twee negatieve controles, beide verplicht (R-1).**
  1. **Teken-omkering.** Ontvang de carry aan de verkeerde kant. Het resultaat
     moet aantoonbaar slechter zijn dan arm 1; is het dat niet, dan meet de
     constructie de carry niet.
  2. **Tijd-permutatie.** Schud de carryreeks binnen elk symbool. Het effect moet
     verdwijnen. Blijft het staan, dan zit het in de prijskant en niet in de
     carry, en dan is de hypothese op dat punt al beantwoord.

- [ ] **7.5 — De causaliteitstest met eigen negatieve controle.** De positie op
  bar `t` mag uitsluitend informatie ≤ `t−1` gebruiken, inclusief de bèta en
  inclusief de rangschikking. De negatieve controle zet één tick van de eigen bar
  terug en moet daarop rood worden.

- [ ] **7.6 — TDD, per substap (R-11).** Test eerst, zie hem falen met de
  verwachte fout, implementeer, zie hem slagen, commit.

---

### Stap 8: Het oordeel

- [ ] **8.1 — Rapporteer per cel de volledige inferentiekern** uit
  `validation/inference.py`: Sharpe, Lo-SE, t, block-bootstrap-interval, de op
  datum geclusterde paneel-t, N_eff en de gedefleerde lezing (R-8, AD-20). Geen
  getal zonder onzekerheid.

- [ ] **8.2 — Rapporteer de decompositie, en zet haar vóór het oordeel.** Per cel:
  welk deel van het bruto rendement is carry, welk deel prijs, welk deel kosten.
  Als het carrydeel onder de helft ligt, is de cel geen carrykandidaat, ongeacht
  wat de Sharpe zegt.

- [ ] **8.3 — Rapporteer de breakeven-kostenanalyse** zoals H-10.1 die voor de
  beslisfrequentie deed: bij welke kosten per round trip kantelt het teken van de
  cel? Die grens naast de 13,0 bps vaste kosten leggen is de enige manier om over
  de ongekalibreerde impactterm te praten zonder hem te kalibreren (AD-2).

- [ ] **8.4 — De vijf stop-criteria, elk met zijn vooraf vastgelegde actie:**

  | # | Criterium | Bindt wanneer | Actie |
  |---|---|---|---|
  | 1 | `no_positive_net_on_development` | geen enkele cel heeft `ci_low > 0` | `UNPROVEN` — en de poort wordt niet gelezen |
  | 2 | `deflated_interval_includes_zero` | `DSR < 0,95` bij `M_new = 25` | `UNPROVEN` |
  | 3 | `carry_is_not_the_source` | carrydeel < 50 % van het bruto in de beste cel | `REJECTED als carrykandidaat` — de bevinding wordt geboekt als richtingswed |
  | 4 | `negative_control_fails` | een van de twee controles uit 7.4 gedraagt zich niet zoals vereist | run ongeldig; **geen** oordeel, en de trials tellen wél |
  | 5 | `adequacy_floor_breached` | < 100 niet-overlappende perioden in een cel | die cel is `UNPROVEN — insufficient data`, en telt niet mee in het oordeel |

- [ ] **8.5 — De poortlezing, en wanneer zij NIET gebeurt.** Haalt de beste cel
  op de ontwikkelsample geen `ci_low > 0` **en** `DSR ≥ 0,95`, dan is de vraag
  gesloten en wordt `W_GATE` niet aangeraakt. Dat is geen overgeslagen substap
  maar de vooraf geregistreerde handeling (R-7). Wordt er wél gelezen, dan
  gebeurt dat **eenmaal**, voor de **ene vooraf benoemde primaire cel**, met de
  lezing geregistreerd in `holdout_lock.json` vóór de run.

- [ ] **8.6 — Boek het oordeel in de ledger** met `n_trials` en `amends` volgens
  de vorm die H-10.1/2/3 hebben moeten gebruiken, met `metrics.verdict`,
  `m_new`, `deflated_sharpe` en `t_hurdle_at_n`. Het veld `result` kent alleen
  `{accepted, archived, falsified, interim}`; `UNPROVEN` hoort in `metrics.verdict`
  en in de notities.

- [ ] **8.7 — Werk `docs/FALSIFICATION_REGISTER.md` bij** met een F-regel
  wanneer de uitkomst een falsificatie is, inclusief een heropeningsvoorwaarde
  die **ten minste één source-id uit `measurement_domain.yaml` noemt** of
  expliciet als buiten-domein is gemarkeerd. Zonder dat faalt
  `scripts/check_domain_consistency.py`, en terecht.

- [ ] **8.8 — Commit.**

> **Wat een positief resultaat hier NIET is.** Een geslaagde H-11.1 is geen
> promotie. Zij is één hypothese op de ontwikkelsample met hoogstens één
> poortlezing. De promotiepoorten in `validation/gates.py` staan er los van en
> worden in deze fase niet aangeraakt.

---

# STAGE C — DE OPRUIMING DIE FASE 10 OPENLIET

*Geen hypothesen, nul trials. Elke stap hieronder voert een verdict uit dat
fase 10 al heeft geveld.*

---

### Stap 9: De archiveringen (fase 10, stap 15.4/15.5)

- [ ] **9.1 — Verplaats per module**, in **aparte commits per verdict** (fase 10
  stap 15.5: een gebundelde commit maakt één foute archivering onomkeerbaar
  zonder de rest terug te draaien). De zes uit §3.4, samen 1.038 LOC.

- [ ] **9.2 — Schrijf per module een `README.md`** met de reden en de
  heropeningsvoorwaarde uit de tabel in `docs/MEASUREMENT_DOMAIN.md`. Neem die
  voorwaarde letterlijk over; formuleer hem niet opnieuw.

- [ ] **9.3 — Haal ze uit de coverage-paden in `pyproject.toml`** en controleer
  wat dat met de 60,66 % doet. Dit is de enige stap in deze fase die de
  coverage-poort **mag** bewegen, en de beweging moet worden gerapporteerd: 1.038
  regels met lage dekking uit de noemer halen verhoogt het percentage zonder dat
  er één test bij komt. **Zeg dat er expliciet bij**, anders leest een latere
  lezer een dekkingsverbetering die er niet is.

- [ ] **9.4 — Draai de volledige suite per verplaatsing.** `bars/_kernels.py`
  (330 LOC) is klasse A en wordt door `test_audit_fixes.py` gebruikt; volgens
  `docs/CODE_REGISTER.md` regel 94 breekt `bars/dollar.py` verplaatsen die test
  als de kernel meeverhuist. Verplaats de kernel dus **niet** mee zonder die test
  eerst aan te passen, en als je hem aanpast: laat zien dat hij nog kan falen.

- [ ] **9.5 — Commit per verdict.**

---

### Stap 10: De dagelijkse runner (fase 10, stap 16)

`src/tradebot/live/` telt 13 modules naast zijn `__init__.py`; `live/` + `oms/`
samen 21 `.py`-bestanden en 6.184 regels (`wc -l`; `docs/PROJECT_STATE.md` §3.2
noteert 21 modules / 6.173 LOC met een andere teller). Daarvan staat **nul** in
de authoritative afsluiting.

- [ ] **10.1 — Inventariseer eerst wat `live/` doet dat de runner óók moet doen.**
  Dit gaat vóór elke verwijdering. Minimaal: de haltketen en haar volgorde, de
  PIT-verificatie bij inlezen, de foutpaden en het fail-fast-gedrag, de logging
  en de artefactschrijving. De haltketen is de meest waardevolle logica in `live/`
  en mag onder geen voorwaarde verdwijnen.

- [ ] **10.2 — Schrijf `apps/run_daily_decision.py` (≤ 80 LOC, R-6).** Vier
  verantwoordelijkheden: lees de PIT-store tot en met gisteren; bereken de
  exposures voor vandaag; pas de haltketen toe in de bestaande volgorde; schrijf
  het besluit als artefact — **met de policy-hash erin** (AD-27). Geen loop, geen
  wachttijden, geen sessiebeheer.

- [ ] **10.3 — Bewijs de gelijkwaardigheid.** Draai de runner en de oude loop over
  dezelfde historische bar en toon aan dat exposures en haltbesluiten identiek
  zijn. Zonder die test is de vervanging niet aantoonbaar gedragsneutraal.

- [ ] **10.4 — Schrijf de AD** (`AD-28`, want AD-26 en AD-27 zijn vergeven):
  *"Eén besluitmoment per dag; `live/` wordt een runner."* Afgewezen alternatief:
  `live/` behouden met de loop op 24 uur — dat houdt de sessie-, reconnect- en
  latentiepaden in leven zonder dat iets ze test.

- [ ] **10.5 — Sluit DI-28 mee**: `live/circuit_breaker.py` en
  `apps/live_paper_trader.py` dragen eigen drempel-defaults naast
  `conf/risk/default.yaml`. Dat is de constructie die `RISK_CONTRACT` §8 verbiedt
  en zij is precies hoe deze repository ooit aan vijf plaatsen met vier waarden
  kwam.

- [ ] **10.6 — Commit.**

---

### Stap 11: De permanente limieten (fase 10, stap 17)

Fase 10 stap 17 noemt `conf/portfolio/constraints.yaml` en
`max_weight_per_symbol`. **Beide bestaan niet**, en dat is een correctie op de
stapopdracht en geen reden hem over te slaan (R-10):

* `conf/portfolio/` bestaat niet; `constraint_order` en alle limieten staan in
  `conf/risk/default.yaml`.
* De per-naam-bovengrens bestaat al **twee keer**: `max_position_pct` (0,80,
  absoluut) en `max_concentration` (0,40, als fractie van de gross). Een derde
  sleutel toevoegen die hetzelfde begrenst, is een tweede implementatie van
  dezelfde grootheid (R-3).

- [ ] **11.1 — Stel eerst vast of er een gat is.** Stap 3 heeft gemeten welke
  limiet op welk aandeel van de bars bindt. Is de concentratie na de toestands-
  of carrypoort al begrensd door `max_concentration`, dan is fase 10's
  criterium 19 vervuld door een bestaande limiet en wordt dat opgeschreven —
  niet met een nieuwe sleutel "opgelost".

- [ ] **11.2 — Voeg `max_notional_for_uncalibrated_impact` toe**, mét een
  commentaarblok dat de reden, de AD-verwijzing en de voorwaarde voor verhoging
  noemt (namelijk: een gekalibreerde `eta`). Positie in `constraint_order`:
  **vóór** de impactberekening, anders wordt een ongeldige functie geëvalueerd.
  Leg die reden in het commentaar vast.

- [ ] **11.3 — Schrijf de tests**: een notional onder de limiet passeert; een
  notional erboven wordt begrensd en niet stilzwijgend doorgelaten; en de
  volgorde in `constraint_order` is precies de vastgelegde.

- [ ] **11.4 — Werk AD-2 en AD-3 bij naar `PERMANENT`** met een expliciete
  `Consequence`-sectie. De grond is nu sterker dan in fase 10: beide dragen
  vandaag *"actief tot er orderboek-/quote-data is"*, en AD-23 heeft die data
  **buiten het meetdomein** geplaatst. Een heropeningsvoorwaarde die een bron
  noemt die per mandaat niet kan bestaan, is geen uitgestelde vraag maar een
  permanente eigenschap. Noem in `Consequence` welke uitspraken over kosten dit
  systeem **niet** kan doen, en verwijs naar de breakeven-route uit stap 8.3 als
  de manier waaróp er dan nog over kosten te praten valt.

- [ ] **11.5 — Commit.**

---

### Stap 12: De vier rode workflows

- [ ] **12.1 — `CI` / `hygiene` / `inventory`** worden groen door stage A stap 1
  en 2 plus de coverage-poort. Meet wat de coverage na stap 9.3 doet en beslis
  dan pas: de drempel verlagen is **geen** optie, dekking toevoegen aan wat er
  daadwerkelijk toe doet wel.

- [ ] **12.2 — `Nightly Regression`** heeft twee oorzaken. De ontbrekende
  baselines zijn DI-25 en dat is een eigenaarsbesluit: laat die poort rood. De
  benchmarkjob geeft exit 5 (nul tests verzameld) en dat is **geen** eigenaars-
  besluit maar een job die niets meet; repareer of verwijder hem, met de reden.

- [ ] **12.3 — `Security Scan`** blijft rood zolang `diskcache 5.6.3` /
  PYSEC-2026-2447 geen fixversie heeft (DI-26). Niet maskeren, niet `|| true`.
  Controleer wel of upstream inmiddels een versie heeft, en noteer de datum van
  die controle.

- [ ] **12.4 — Commit.**

---

# EXIT-CRITERIA

De fase is af wanneer **alle** onderstaande regels waar zijn. Dit is geen
samenvatting maar de afvinklijst; een regel die niet waar is, blokkeert de
afsluiting.

| # | Criterium | Verificatie |
|---|---|---|
| 1 | De vingerafdruk van vóór de fase staat vast en de suite meldt aan het eind **0 failed**, of elke resterende failure is met bewijs aangewezen als omgevingsverschil (DI-24) | stap 1 |
| 2 | Elk artefact met een risicobesluit draagt de `config_hash` van de configuratie die het produceerde; de poort daarop kan aantoonbaar rood worden | stap 2 |
| 3 | AD-27 is geschreven | stap 2.5 |
| 4 | Elk artefact onder `1b60cb664fbf9a2a` is `SUPERSEDED_BY_POLICY` gemarkeerd of opnieuw afgeleid; welke van de twee, staat per artefact opgeschreven | stap 2.3 |
| 5 | `test_dust_breaks_relative_limits` leidt de hash uit het register af en pint geen literal meer | stap 2.4 |
| 6 | Per limiet in `constraint_order` is gemeten op hoeveel bars hij bindt, onder **beide** policies | stap 3.1–3.2 |
| 7 | De vraag "bindt de vol-target" is met een getal beantwoord, niet met een redenering | stap 3.3 |
| 8 | De 3,06 % uit `test_a_uniform_factor_is_neutralised_by_the_vol_target` is toegeschreven aan een aanwijsbare limiet; de tolerantie is **niet** opgerekt | stap 3.4 |
| 9 | DI-27 en DI-30 zijn gesloten, elk met de uitkomst die de meting gaf | stap 3.5 |
| 10 | `artefacts/baseline/phase11_revaluation.json` bestaat, draagt `9961e1613bc907a5`, en de twee ladders staan met SE's naast elkaar | stap 4 |
| 11 | Elke conclusie van fase 10 die op een significant verschoven cel rust, is bij naam genoemd | stap 4.3 |
| 12 | Er is nog precies één funding-route; `data/funding.py` is óf de route óf weg | stap 5.2 |
| 13 | De 41 SOLUSDT-dagen met ≠ 3 records hebben een feitelijk antwoord in `docs/DATA_REGISTER.md`; geen rij is verwijderd vóór dat antwoord | stap 5.3 |
| 14 | De funding-aggregatie draagt een causaliteitstest met een negatieve controle die aantoonbaar rood wordt | stap 5.4 |
| 15 | `preregistration_h11_funding_carry.yaml` is bevroren vóór de eerste fit, met 6 geplande trials en de adequaatheidsvloer vooraf uitgerekend | stap 6 |
| 16 | H-11.1 draagt een `verdict`, een `m_new`, een `deflated_sharpe` en een `t_hurdle_at_n` | stap 8 |
| 17 | Het rapport draagt de carry/prijs/kosten-decompositie **vóór** het oordeel | stap 8.2 |
| 18 | Er is een breakeven-kostenanalyse naast het gemeten resultaat | stap 8.3 |
| 19 | Beide negatieve controles zijn gedraaid en hun uitkomst staat in het rapport | stap 7.4 |
| 20 | `holdout_lock.json` is óf ongewijzigd (`reads: []`) óf draagt precies één geregistreerde lezing voor de primaire cel | stap 8.5 |
| 21 | Het trialsaldo klopt: 5 (fase 10) + het werkelijk bestede aantal ≤ 25, en het restant staat in het rapport | stap 8.6 |
| 22 | De zes `ARCHIVED`-modules staan in `archive/`, elk met een `README.md` met reden en heropeningsvoorwaarde | stap 9 |
| 23 | Het coverage-effect van de archivering is apart gerapporteerd als noemer-effect | stap 9.3 |
| 24 | `apps/run_daily_decision.py` bestaat, is ≤ 80 LOC, en is aantoonbaar gedragsequivalent aan de oude loop | stap 10 |
| 25 | AD-28 is geschreven | stap 10.4 |
| 26 | De haltketen en `constraint_order` zijn ongewijzigd behalve de nieuwe limiet op haar vastgelegde positie, met de reden voor die positie in het commentaar | stap 10.1, 11.2 |
| 27 | DI-28 is gesloten: er is één bron van waarheid voor elke risicodrempel | stap 10.5 |
| 28 | AD-2 en AD-3 zijn `PERMANENT` met een `Consequence`-sectie | stap 11.4 |
| 29 | Fase 10's criteria 4, 19, 28 en 30 zijn waar geworden, of met een gemeten reden vervallen verklaard | §3.4 |
| 30 | `ruff` 0.15.12 en `mypy` zijn schoon; geen bestand > 800 LOC zonder cap; elke **nieuwe** app ≤ 80 LOC | continu |
| 31 | De zes poortscripts geven exit 0 | continu |
| 32 | `CI`, `hygiene` en `inventory` zijn groen; `Security Scan` en `Nightly Regression` zijn rood **met** een geregistreerd eigenaarsbesluit (DI-25, DI-26) en om geen andere reden | stap 12 |

---

# REGELS & HANDELINGSINSTRUCTIES

**R-1 — Causaliteit is een test, geen intentie.** Elke transformatie die een
tijdreeks aanraakt krijgt een truncatietest **en** een perturbatietest, elk met
een negatieve controle die bewijst dat de test rood kan worden.

**R-2 — Elke parameter kost een trial, en elke tak van een beslisboom ook.**
Registreer vóór de run. Een parameter die op grond van een uitkomst wordt
gewijzigd, kost retroactief ook een trial voor zijn voorganger.

**R-3 — Eén implementatie per grootheid.** Sharpe-SE, DSR en Sharpe-verschil
komen uitsluitend uit `validation/inference.py`. Dat geldt in deze fase ook voor
**de fundingreeks** en voor **de kostentabel**: een tweede implementatie is een
defect, ook als zij hetzelfde getal geeft.

**R-4 — Geen bestand boven 800 LOC.** Splits op verantwoordelijkheid, niet op
regelaantal.

**R-5 — Fail fast, met een reden.** Elke `require(...)` legt uit welke aanname
is geschonden en waarom die aanname bestaat.

**R-6 — Apps blijven onder 80 LOC.** Een app is een compositie, geen logica. De
33 bestaande overschrijdingen zijn DI-4 en worden in deze fase niet gerepareerd;
elke **nieuwe** app voldoet.

**R-7 — De poortsample wordt per hypothese ten hoogste eenmaal gelezen**, vooraf
geregistreerd in `holdout_lock.json`. Een tweede lezing maakt de sample tot
ontwikkeldata en dan is er geen poort meer.

**R-8 — Een getal zonder onzekerheid is geen bevinding.** Elke Sharpe, elk
verschil en elk conditioneel gemiddelde krijgt een SE, een interval en — op
paneeldata — de geclusterde en gedefleerde lezing ernaast.

**R-9 — Alles in dit document is Nederlands; code, docstrings, commits en
artefactsleutels blijven Engels.**

**R-10 — Een verwachting in dit document is geen resultaat.** Wijkt de meting
af, dan is de meting het antwoord en wordt de verwachting geciteerd als de
weerlegde voorspelling die zij was. Dit document bevat er drie met opzet: de
vol-target hoort te binden (§stap 3.3), het carrymandje hoort niet
verhandelbaar te zijn op zijn eigen Sharpe (§stap 6), en de halttellers horen te
dalen (§stap 4.4).

**R-11 — TDD, zonder uitzondering.** Een test die bij de eerste run slaagt, test
niet wat je denkt.

**R-12 — Commit per stap, niet per stage.**

**R-13 — Een meting zonder haar configuratie is geen meting.** Elk artefact dat
een risicobesluit bevat, draagt de `config_hash` van de configuratie die het
produceerde. Dit is de regel die deze fase heeft veroorzaakt; zij wordt in stap
2 een poort en staat hier zodat zij ook geldt voor wat er na deze fase komt.

**R-14 — De vingerafdruk is een poort, geen notitie.** Aan het begin en aan het
eind van elke stage wordt de volledige suite gedraaid en het resultaat
opgeschreven. Een failure die tussen twee stages verschijnt, hoort bij de stage
die ertussen zat — niet bij "het stond er al".

---

# STARTINSTRUCTIE

Begin bij **stap 1**. Werk de stappen in volgorde af. Stage A is een voorwaarde
voor stage B: zolang niet vaststaat welke limiet bindt en wat de ladder onder het
geldende beleid doet, is elke kostenuitspraak in stage B ongefundeerd. Stage C
mag parallel aan stage B, maar **nooit in dezelfde commit**.

Vóór de eerste regel code, vier handelingen in deze volgorde:

1. **Lees `docs/MANDATE.md`, AD-22 t/m AD-26, en
   `artefacts/governance/ledger_reset.json` volledig.** Alles hier veronderstelt
   ze.
2. **Reproduceer §3.** Niet steekproefsgewijs — de vingerafdruk, de acht
   poorten, de twee policy-hashes en de carrytabel uit §3.6. Wijkt iets af, dan
   is dát je eerste bevinding en die schrijf je op vóór je verdergaat.
3. **Controleer of `f50f6ca` nog de kop is.** Is er sinds 2026-09-20 gecommit,
   dan is §3 een historische meting en geen nulmeting, en moet je hem opnieuw
   doen voordat je hem gebruikt.
4. **Stel vast of dit document zelf klopt.** Fase 10 stap 14 vond drie
   feitelijke beweringen in zijn eigen stapopdracht die alle drie onjuist waren,
   en het juiste antwoord was toen de meting en niet de prompt. Dat geldt hier
   ook: §3.7 en stap 11 corrigeren de voorgaande prompt al, en er is geen reden
   aan te nemen dat dit document de laatste is dat het nodig heeft.

Rapporteer na elke stap: welke test faalde, met welke fout, wat de implementatie
werd, welke test slaagde, en de commit-SHA. Rapporteer na elke stage de stand van
het trialbudget, de vingerafdruk en de resterende exit-criteria.

Sla geen stap over omdat de uitkomst voorspelbaar lijkt. De verwachtingen in dit
document zijn expliciet gemaakt zodat zij kunnen worden weerlegd — niet zodat zij
de meting kunnen vervangen.

---

# BIJLAGE A — REPRODUCTIE VAN §3.6

Deze bijlage staat hier zodat §3.6 een meting is en geen bewering. Zij leest
uitsluitend de gecertificeerde store en schrijft niets. Draai hem vóór je stage
B begint; wijkt een getal af, dan is dát je eerste bevinding (R-10).

```python
# nulmeting_carry.py — diagnostisch, nul trials, schrijft niets.
import glob
import numpy as np
import pandas as pd

def load(dataset: str) -> pd.DataFrame:
    paths = sorted(glob.glob(
        f"data/pit_store/**/dataset={dataset}/**/*.parquet", recursive=True))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    frame["ts"] = pd.to_datetime(frame["event_ts_ns"], unit="ns", utc=True)
    frame["date"] = frame["ts"].dt.normalize()
    return frame

START = pd.Timestamp("2021-11-15", tz="UTC")   # W_DEV, MEASUREMENT_CONTRACT §2
END = pd.Timestamp("2025-09-04", tz="UTC")

funding, ohlcv = load("funding"), load("ohlcv")
window = funding[(funding["date"] >= START) & (funding["date"] <= END)]

# Bevinding 3.6.1(a): 41 dagen dragen geen drie 8h-ticks, alle 41 SOLUSDT.
per_day = window.groupby(["symbol", "date"]).size()
print(per_day[per_day != 3].value_counts().to_dict())

daily_carry = window.groupby(["symbol", "date"])["funding_rate"].sum()

bars = ohlcv[(ohlcv["date"] >= START) & (ohlcv["date"] <= END)]
bars = bars.sort_values(["symbol", "date"])
bars["logret"] = bars.groupby("symbol")["close"].transform(
    lambda s: np.log(s).diff())

rows = []
for symbol, group in bars.groupby("symbol"):
    carry = daily_carry.loc[symbol].reindex(group.set_index("date").index)
    price = group.set_index("date")["logret"]
    rows.append({
        "symbol": symbol,
        "n_days": len(group),
        "carry_bps_day": carry.mean() * 1e4,
        "carry_sd_bps_day": carry.std() * 1e4,
        "frac_positive": float((carry > 0).mean()),
        "price_bps_day": price.mean() * 1e4,
        "short_gross_bps_day": (carry - price).mean() * 1e4,
        "ann_sharpe_short_gross":
            (carry - price).mean() / (carry - price).std() * np.sqrt(365),
    })
print(pd.DataFrame(rows).round(4).to_string(index=False))

basket_carry = daily_carry.unstack(0).mean(axis=1) * 1e4
basket_price = bars.pivot_table(
    index="date", columns="symbol", values="logret").mean(axis=1) * 1e4
print("carry  :", round(basket_carry.mean(), 3), "bps/dag · sd",
      round(basket_carry.std(), 3), "· ann. Sharpe",
      round(basket_carry.mean() / basket_carry.std() * np.sqrt(365), 3))
print("prijs  :", round(basket_price.mean(), 3), "bps/dag · sd",
      round(basket_price.std(), 3))
print("sd-ratio:", round(basket_price.std() / basket_carry.std(), 1))
```

**Drie waarschuwingen bij deze bijlage**, en zij zijn de reden dat stap 5 vóór
stage B staat:

1. De dagsom hierboven is **niet** de aggregatie die de featurelaag gebruikt.
   `features/positioning.py` koppelt met een backward asof-join de laatst
   bekende rate. Welke van de twee de juiste is voor een carryboek, is een
   besluit dat in stap 5 valt en niet hier.
2. De prijscomponent is een **log**-rendement en de carry is een **simpel**
   tarief. Over één dag is het verschil verwaarloosbaar; over een houdduur van
   tien dagen is het dat niet. `MEASUREMENT_CONTRACT.md` §1 legt de
   rendementsconventie vast en stage B volgt die, niet deze bijlage.
3. Er staat hier **geen** kostenkant in. Elk getal is bruto.
