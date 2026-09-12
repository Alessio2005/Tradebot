# MASTER-PROMPT: FASE 9 — BESLUIT: ONTMANTELING & VOLATILITEITSSPOOR

> **Fase:** 9 van 10 · **Status:** besluitfase, geen bouwfase · **Prioriteit:** P0
> **Opvolger:** `fase_10_uitvoering_ontmanteling_en_volatiliteitsspoor.md` — die fase gaat pas open wanneer §3-V1 hier is beantwoord.
> **Bindende brondocumenten:** `DOORLICHTING TRADEBOT — repository-scan & levensvatbaarheidsvonnis` (2026-09-12) met Bijlage A t/m D
> **Nulmeting bij het schrijven van deze prompt:** `origin/main` = `8ad9143`, **98 commits**, laatste commit `2026-08-29 14:24:24 +0200`. **Er zijn geen commits bijgekomen sinds de doorlichting.** Het vonnis is dus onverkort actueel; niets in de repo weerspreekt het.
> **Opdrachtgever wil:** (1) maximaal opruimen wat overbodig is, (2) **alle sleeves eruit**, (3) verder werken met volatiliteitsmodellen (GARCH-familie) op een professioneler niveau.

---

## 0. ROL EN HARDE REGELS

Je acteert als **Head of Quantitative Research met mandaat tot ontmanteling**. Je opdracht is niet om dit systeem te redden en niet om het af te breken. Je opdracht is om vast te stellen **wat er na de doorlichting nog een meetbare vraag beantwoordt**, en alles wat dat niet doet te verwijderen.

Vijf regels, in deze volgorde bindend:

1. **Meting gaat boven bewering, ook boven deze prompt.** Wijkt een getal hier af van wat je meet, dan prevaleert jouw meting en noteer je de afwijking expliciet. De doorlichting deed dat zelf ook (§2.1: de eigen master-prompt werd gefalsificeerd door een volledige kloon).
2. **"Stoppen" blijft een toegestane uitkomst.** Deze prompt vraagt een besluit, geen plan. Luidt het antwoord op §3-V1 nee, dan is de deliverable een afsluitdocument en geen roadmap. Een prompt die alleen "doorgaan" als uitkomst toelaat, is een prompt die niets meet.
3. **Geen nieuwe code vóór §3-V1 is beantwoord.** De doorlichting wijst één patroon aan als hoofdoorzaak van 102.971 regels: de volgorde-fout (bouwen wat pas betekenis krijgt ná een bewezen edge, §5.2). Je mag die fout niet herhalen in een nieuwe smaak.
4. **Elke bewering krijgt een pad, een regelnummer, een sha of een commando met uitvoer.** Geen enkele uitzondering. Waar een getal niet vast te stellen is, schrijf je op welk bestand het zou bevatten.
5. **Een niet-uitgevoerde meting is geen negatieve meting.** `UNPROVEN` en `FALSIFIED` zijn verschillende oordelen en mogen nooit door elkaar worden gebruikt. Deze regel staat al in `conf/research/preregistration_h1_garch_vs_ewma.yaml` (`underpowered_test_cannot_falsify`) en geldt hier repo-breed.

---

## 1. WAT VASTSTAAT — NIET OPNIEUW ONDERZOEKEN

Deze uitkomsten zijn gemeten, gereproduceerd en niet in geschil. Ze zijn het uitgangspunt, niet het onderwerp. Herhaal dit werk niet; bouw erop voort.

| Vaststaand | Bron |
|---|---|
| Break-even bruto-edge per round trip: **13,0 bps** vast + impact → **16,3 (ETH) tot 57,8 (DOT) bps** | Bijlage C, C3 · `conf/execution/fees.yaml:10-16` |
| Beste bruto-edge die ergens in de repo gemeten is: **+4,43 bps**; mediaan van 120 cellen **−4,23 bps** | Bijlage C, C1 · `reports/short_barrier_sweep.csv` |
| Enige meting mét volledige herkomst: netto OOS Sharpe **−0,483**, SPA p = 1,00, DSR-p 0,00069, in de ledger als `falsified` | `reports/BASELINE_BENCHMARK.md` · `artefacts/baseline/phase3_baseline.json` |
| Enige echte forward-meting: `forward_sharpe 0,688`, `pbo 0,222`, `n_trials 8` | `reports/adaptive_wf_metrics.json` |
| De headline-cijfers (Sharpe 2,70 / 4,31) zijn niet reproduceerbaar: producer verwijderd in `8071dc5`, geen provenance-veld, niet in de ledger | Bijlage B §3 · Bijlage C, correctie (2) en (3) |
| De DSR-straf is materieel uitgeschakeld: M-proxy = `count_git_commits()` = 98 i.p.v. 2.776; `cap_factor` handmatig 5,0 → 1,0 | Doorlichting §3.5 · `src/tradebot/tune/objective.py:790,795-800` |
| De live-laag omzeilt de risicolaag: `rg RiskEngine src/tradebot/live/` → 0 treffers; `_max_gross_notional = inf`; fat-finger-guard reset na élke fill | Doorlichting §3.2 · `live/execution_controller.py:94-95`, `live/engine.py:655` |
| Onderzoeksgovernance (pre-registratie, ledger, kill-gates, falsificatieregister) **werkt** en heeft de auteur meermaals zijn eigen resultaat afgenomen | Bijlage B §6 |
| Promotie-/productiegovernance (promotion gates, champion/challenger, shadow, lineage, reconciliatie) is **nooit op een echte kandidaat gedraaid** | Bijlage B §6 |

**De consequentie die je moet internaliseren:** het probleem van dit systeem is nooit een tekort aan machinerie geweest. Het is het ontbreken van één verhandelbaar object waarvan de bruto-edge boven 16,3 bps uitkomt. Alles wat je hierna voorstelt, wordt daaraan getoetst en aan niets anders.

---

## 2. WAT JE ALS EERSTE CONTROLEERT (vóór alle overige werk)

### 2.1 Bijlage A — lokaliseren of afwezig verklaren

De doorlichting noemt in zijn kop uitsluitend **Bijlage B, C en D** (`faseB.md`, `faseC.md`, `faseD.md`). Een Bijlage A is in de opdracht genoemd maar is in het beschikbare materiaal **niet aangetroffen**. Doe precies dit, en niets anders:

1. Zoek naar `faseA*`, `bijlage*a*`, `MASTER_PROMPT_repo_scan_levensvatbaarheid.md` in de repo, in de omgeving van de doorlichting en bij de opdrachtgever.
2. **Gevonden:** verwerk hem als bindend brondocument en noteer welke beweringen uit A afwijken van B/C/D.
3. **Niet gevonden:** schrijf letterlijk op dat Bijlage A niet bestaat in het beschikbare materiaal, dat §1–§2 van de doorlichting (nulmeting en reproduceerbaarheid) de plaats ervan innemen, en ga verder op B t/m D. **Verzin geen inhoud voor A en presenteer geen reconstructie als bron.**

### 2.2 Kloondiepte — de val die de master-prompt zelf al eens maakte

`git rev-list --count HEAD` op een `--depth`-kloon geeft **50**; een volledige kloon geeft **98**. Dat verschil heeft in de vorige ronde tot een onjuiste bevinding geleid (doorlichting §2.1). Begin met:

```bash
git clone https://github.com/Alessio2005/Tradebot.git tradebot_full   # zonder --depth
cd tradebot_full && git rev-list --count HEAD    # moet 98 zijn, of meer
git log --oneline -1
```

Wijkt het aantal af van 98 of `8ad9143`, dan zijn er commits bijgekomen ná deze prompt: meet dan eerst wat die veranderd hebben en pas §1 aan voordat je verdergaat.

### 2.3 Reproduceer de vier rode poorten

```bash
pip install -e ".[dev]" && python apps/doctor.py && pytest -q
```

Verwacht: **4 failed, 25 skipped, ~2.341 passed**. De vier failures zijn geen hygiëne, het zijn de alpha-poorten (KG-B1/B2/B3 op `cm_carry` en `cm_tsmom`). Komen ze niet terug, dan is er iets aan de data of de code veranderd en moet je dat eerst uitzoeken.

---

## 3. DE DRIE VRAGEN DIE DEZE FASE BEANTWOORDT

Alles hierna dient deze drie vragen. Er is geen vierde.

**V1 — Bestaat er een verhandelbaar object waarvan de bruto-edge per round trip boven 20 bps uitkomt, out-of-sample, op data die niet is gebruikt om parameters te kiezen?**
20 bps = de kostendrempel uit C3 (16,3 bps op het goedkoopste symbool) met 20 % marge. Dit is de enige vraag waarvan het antwoord het vonnis kan omdraaien (doorlichting §4.2). Antwoord met een getal, een venster en een pre-registratie-id, of met "niet gemeten".

**V2 — Hoeveel van de 102.971 regels kan weg zonder dat het antwoord op V1 verandert?**
De doorlichting rekent voor dat ≈15.200 regels volstaan (Bijlage D.9). Toets die rekensom zelf en lever een schrapinventaris met bewijs per regel.

**V3 — Verandert een volatiliteitsmodel het antwoord op V1, en zo ja via welk mechanisme?**
Dit is de vraag waarop de opdrachtgever aanstuurt. §6 legt vast hoe je hem eerlijk stelt in plaats van hem te bevestigen.

---

## 4. SPOOR 1 — DE GRENDEL (eerst, onvoorwaardelijk, ±1 uur)

Dit spoor kent geen onderzoeksvraag en wacht op niets. Zolang `src/tradebot/live/` startbaar is, is er een pad waarlangs 4.609 regels ongetoetste code ongelimiteerd kan handelen op een positieboekhouding die niet met de beurs is afgestemd (doorlichting §3.2). Dat is het enige punt in de hele doorlichting dat directe schade kan veroorzaken.

Lever op, met een test per punt die aantoonbaar rood kan worden:

1. `live/engine.py` crasht bij opstarten zonder expliciete omgevingsvlag `I_UNDERSTAND_THIS_TRADES_REAL_MONEY`. Geen waarschuwing, geen logregel — een crash.
2. `_max_notional_per_symbol` en `_max_gross_notional` krijgen eindige waarden uit `conf/`, en de module crasht wanneer die sleutel ontbreekt (geen `inf`-default).
3. De fat-finger-reset op `live/engine.py:655` verdwijnt of wordt een geconfigureerde, geregistreerde uitzondering.
4. Credential-paden uit `conf/` en `infra/` verwijderd; `secret-binance.yaml` weg.
5. Eén test die het SIGUSR1-kill-switch-pad op een lopende engine aanroept — niet op `HaltStore`. De huidige 20+ kill-switch-tests raken de live-implementatie niet (Bijlage D.8).

**Acceptatie:** een reviewer kan in vijf minuten aanwijzen waarom een onbedoelde start nu geen order kan plaatsen. Lukt dat niet, dan is dit spoor niet af.

---

## 5. SPOOR 2 — DE ONTMANTELING

### 5.1 Uitgangspunt van de opdrachtgever: alle sleeves eruit

`src/tradebot/alpha/` telt **28 bestanden, 4.568 regels**. Van die 28 zijn er precies twee geen sleeve:

| Bestand | Regels | Wat het is | Vonnis |
|---|---|---|---|
| `adaptive_wf.py` | 362 | Walk-forward + PBO-harness. **Producent van `reports/adaptive_wf_metrics.json`, de enige echte forward-meting in de repo** | **BEHOUDEN, VERPLAATSEN** naar `backtest/` of `cv/` |
| `research_harness.py` | 161 | Onderzoeksrunner, geen marktclaim | BEHOUDEN of samenvoegen met bovenstaande |
| de overige 26 | 4.045 | Sleeves: carry, tsmom, xsmom, lowvol, strev, pead, quality, microstructure, mean_reversion, kalman_ou, macro_regime, de twee books, `factor_alpha.py` | **SCHRAPPEN** |

Waarvan 627 regels (`eq_*.py` + `fx_*.py`) sowieso buiten scope vallen: aandelen en FX in een crypto-perp-repo.

> **Dit is de belangrijkste val van dit spoor.** "Alle sleeves weg" en "de enige echte meting weg" liggen 362 regels uit elkaar. Verplaats `adaptive_wf.py` **vóór** je `alpha/` opruimt, en draai `reports/adaptive_wf_metrics.json` daarna opnieuw om te bewijzen dat de verplaatsing niets veranderde. Komt dat cijfer niet terug, dan is dat een bevinding op zichzelf.

Wat meeverdwijnt met de sleeves, en expliciet genoemd moet worden: `apps/paper_neutral_trader.py` en `apps/paper_multi_sleeve.py` (beide draaien uitsluitend op `alpha.neutral_book` / `alpha.multi_sleeve_book`), `apps/run_eq_units.py`, `apps/ingest_fx.py`, `apps/build_equity_universe.py`, `apps/ingest_edgar.py`, `apps/fetch_factors.py`, `apps/ingest_eia.py`, `apps/ingest_xasset.py`, plus de kill-gate-artefacten en tests die alleen die sleeves toetsen.

### 5.2 De rest van de schrapinventaris

Overneembaar uit Bijlage D, maar **elke regel opnieuw meten** met een AST-importscan (relatieve imports meegeteld, lazy in-functie-imports apart gemarkeerd):

| Categorie | Omvang | Grond |
|---|---|---|
| Nul productie-importers: `tca/` 723, `regime/` 922, `reporting/` 294, `featurestore/` 268 | **2.207** | D.4 |
| `src/tradebot/artefacts/` (twee `.md`, nul Python in `src/`) | 0 | D.4 — verplaats naar `docs/` |
| `compliance/` (champion/challenger, MRM, shadow) | 1.015 | 1 caller; `generate_mrm_report.py:61-68` rapporteert op hardcoded placeholders (`deflated_sharpe: 3.38`) |
| `miscellaneous/` pipeline-runners (3×) | 599 | Alle drie roepen het niet-bestaande `apps.backtest_portfolio` aan; `dvc.yaml` is de overlevende |
| `apps/live_trader.py` + `paper_monitor.py` | 702 | Echte-orderpad; `streamlit` staat in geen enkele dependency-lijst |
| `infra/k8s/` + Prometheus als **harde** dependency | 8 manifests + 1 dep | Kubernetes voor een proces dat nooit één dag heeft gelopen |
| `monitoring/drift*` | 523 | Drift meten in een verdeling die geen live model bedient |
| `tune/` (1.222 regels, **0 testbestanden**) | 1.222 | Optuna vergroot M vóór er een doel is; bevriezen of schrappen |
| `reports/`: 12 van de 15 metriek-JSON's | — | Sharpe-spreiding 2,61–4,31 zonder één provenance-veld; twee bit-identiek (`md5sum` gelijk) |
| `scripts/` (58 bestanden, 8.688 regels) | 8.688 | Ongedocumenteerde tweede apps-laag buiten elk contract — **inventariseer en verantwoord per bestand**; dit gebied is in Bijlage D niet per stuk beoordeeld |

**Fantoom-entrypoints repareren of verwijderen:** `pyproject.toml:88-89` exporteert `apps.backtest_portfolio:main` en `apps.make_tearsheet:main`; beide bestanden bestaan niet, waardoor `make backtest` en `make tearsheet` onmiddellijk falen.

### 5.3 De regels waaronder geschrapt wordt

1. **Eén annuleerbare tag vóór de eerste verwijdering:** `git tag pre-ontmanteling-<datum>` en push hem. Geen enkele schrapactie mag onherstelbaar zijn.
2. **Schrappen in kleine, thematische commits** — één categorie per commit, met de importscan-uitvoer in de commit-message. Niet één commit van −80.000 regels.
3. **Na elke commit:** `pytest -q` en `python apps/doctor.py`. Het testtotaal mag dalen (tests van geschrapte code verdwijnen mee) maar het aantal *failures* mag niet stijgen boven de vier bekende poort-failures.
4. **De gecertificeerde meetketen blijft draaibaar:** `data/pit_store` → features → labels → CV → backtest → `reports/BASELINE_BENCHMARK.md`. Breekt die keten, dan is de schrapactie fout en niet de keten.
5. **Wat je bevriest, bevries je zichtbaar:** verplaats het naar `frozen/` met een `README.md` die zegt onder welke gemeten voorwaarde het weer aan mag. Bevriezen zonder die voorwaarde is uitstel, geen besluit.

### 5.4 Wat er níét weg mag

De datalaag (`data/` + `pit_store` + schema/gaps/outliers-validatie), de backtest-boekhouding (`accounting`, `metrics`, `evaluation`), `execution/` (fees/spread/slippage/impact — hier komt C1–C3 vandaan), `cv/` (purged CPCV), `schemas/`, `utils/`, en het pre-registratie-/ledger-/kill-gate-mechanisme. Dat laatste is het enige onderdeel van dit project dat aantoonbaar de auteur tegen zichzelf heeft beschermd; het is zeldzamer dan een winstgevende strategie.

---

## 6. SPOOR 3 — HET VOLATILITEITSSPOOR

### 6.1 Wat er al staat (meet dit, geloof het niet op mijn woord)

De opdrachtgever vraagt om GARCH. **Die code staat er al, is getest, en is in productie dood.** Gemeten op `8ad9143`:

| Onderdeel | Omvang | Status |
|---|---|---|
| `src/tradebot/volatility/` | **1.770 regels** — `garch.py` (26 KB: `GarchSpec`, `fit_garch_window`, `walk_forward_variance_forecasts`, `ConvergenceSummary`), `har_rv.py`, `realized.py`, `yang_zhang.py`, `parkinson.py`, `garman_klass.py`, `rogers_satchell.py`, `ewma.py` | `arch==8.0.0` staat in `requirements.lock` |
| Productie-importers | Alleen `ewma.py` (12 treffers) en `garman_klass` (8). **`garch.py` en `har_rv.py`: nul productie-callers** — alle treffers in `src/`+`apps/` zijn docstrings, schema-configklassen en een rapportgenerator | `yang_zhang`, `parkinson`, `rogers_satchell`: nul importers |
| `src/tradebot/validation/vol_metrics.py` | QLIKE, MSE-variance/SD, MAE-SD, **Mincer-Zarnowitz**, **Diebold-Mariano met HLN-correctie**, Newey-West-lagkeuze | De professionele toolkit is aanwezig |
| `conf/research/preregistration_h1_garch_vs_ewma.yaml` | **Volledig bevroren pre-registratie** met power-analyse vooraf, 48 geplande trials, vijf stop-criteria, en een expliciete RV-proxy-beperking | **Nooit uitgevoerd** |
| `reports/GARCH_VS_EWMA_COMPETITION.md` | Wordt genoemd in `reporting/phase6_econometrics.py:263` | **Bestaat niet** |
| `tests/unit/test_garch_family.py` | 34 tests | Draait groen |

**De eerste bevinding van dit spoor is dus half gratis, en de andere helft moet je kennen voordat je plant.** De *wetenschapslaag* is compleet en hoeft niet gebouwd te worden: `garch.py`, `realized.py`, `ewma.py`, `walk_forward.py` en `vol_metrics.py` dragen precies de contracten die de pre-registratie eist, en de pre-registratie zelf is bevroren — `artefacts/governance/preregistration_cef1a3b9a6811d7bde1afc92a2a9503f.json`, bevroren 2026-08-25 op `git_sha 1b604a8`, met ledger-entry #11 (`result: interim`, `n_trials: 48`). Wat **wel** ontbreekt is de hele orkestratielaag eromheen: een competitie-engine, de h-staps EWMA-forecast, de keuze en verantwoording van de RV-proxy, de negatieve controle, een rapportschrijver, een runner-app, de wiring in Makefile/dvc/`[project.scripts]`, en de tests op de compositie. Geteld: **≈800–1.090 nieuwe regels** over zes nieuwe bestanden plus vier kleine wijzigingen. Reken dus niet op ‘alleen even draaien’ — reken op orkestratiewerk boven een afgebouwde wetenschapslaag.

### 6.2 De eerlijke beperking, vooraf opgeschreven

De pre-registratie zegt **zelf** wat de verwachte uitkomst is, en dat is geen detail:

> "Hansen & Lunde (2005) vonden over 330 modellen op wisselkoersen en aandelen dat vrijwel niets een GARCH(1,1) verslaat; de winst van GARCH boven RiskMetrics-EWMA is in de literatuur consistent klein en vaak niet significant. **OORDEEL VOORAF: deze toets is MARGINAAL informatief.**"

En de tweede beperking staat er ook: de gecertificeerde store bevat **nul rijen 1m/5m**, dus de RV-proxy is een daily range-estimator met veel lagere efficiëntie dan 5-minuts-RV, wat de power verder verkleint.

**De structurele beperking die nergens in de repo staat en die je wél moet opschrijven:** *een volatiliteitsmodel voorspelt variantie, geen richting.* Een betere σ̂ verandert de **noemer** van de positiegrootte. Hij creëert geen bruto-edge, en hij verlaagt de 13,0 bps vaste kosten per round trip met geen enkele basispunt. Op de rekening uit Bijlage C verandert een perfecte GARCH-fit het antwoord op V1 **niet**, tenzij hij langs één van de drie mechanismen in §6.3 loopt.

Wie dit overslaat, bouwt in een nieuwe smaak precies de volgorde-fout uit doorlichting §5.2.

### 6.3 De drie kandidaten — elk met zijn eigen toets

Je onderzoekt deze drie, in deze volgorde, en je stopt bij de eerste die de kostentoets niet haalt.

**Kandidaat A — σ̂ als risicolaag (uitvoeren wat al bevroren is)**
*Mechanisme:* betere ex-ante variantie → betere vol-targeting en tail-beheersing.
*Toets:* draai H1 exact volgens `preregistration_h1_garch_vs_ewma.yaml`: 4 modellen (GARCH(1,1), GJR, EGARCH, APARCH, Student-t) × 6 symbolen × 2 horizonnen = 48 trials, 1d, venster `2021-11-15` → `2026-08-23`, primaire metriek OOS-QLIKE, DM-HLN op α = 0,05, met de vier stop-criteria (ARCH-poort, significantie, power-deficit, convergentie ≥ 90 %).
*Lever op:* `reports/GARCH_VS_EWMA_COMPETITION.md` met de gemeten AR(1) van de verliesverschilreeks naast de drie power-scenario's, een ledger-entry, en een uitkomst die `PROMOTED` / `FALSIFIED` / `UNPROVEN — insufficient data` luidt.
*Blokkade die je vóór de run moet oplossen:* `src/tradebot/registry/hypothesis_ledger.py:37` kent `_VALID_RESULTS = {"accepted", "archived", "falsified", "interim"}` — er is **geen** `promoted` en **geen** `unproven`. De pre-registratie eist precies die twee verdicten (`promotion_requires_all_gates_clear`, `underpowered_test_cannot_falsify`). Kies expliciet en leg het vast: de frozenset uitbreiden, of `promoted → accepted` en `unproven → interim` mappen met het werkelijke verdict in `metrics`/`notes`. Wat niet mag is de uitkomst in het dichtstbijzijnde bestaande label persen en dat onvermeld laten — dan is de ledger geen register meer maar een afronding.
*Wat dit wél en niet beslist:* dit beslist welke estimator in productie hoort. **Het beslist niets over V1.** Noteer dat in het rapport zelf, zodat een latere lezer het niet als edge-bewijs leest.
*Extra werk dat hier hoort:* toets ook de **vol-targeting-anomalie** uit Bijlage C, C8 — `reports/vol_target_sweep.csv` laat `realized_vol` identiek 0,0742 zien bij targets 0,16 / 0,24 én 0,35. Een vol-target die niet bindt, is een kapotte risicolaag, en dat is met de aanwezige code en data direct te reproduceren.

**Kandidaat B — funding carry, met σ̂ als sizing- en poortmechanisme**
*Mechanisme:* dit is het enige crypto-perp-mechanisme in deze repo met een benoembare tegenpartij — bij positieve funding betalen leveraged longs de shorts voor het aanhouden van de positie. Dat is een antwoord op de vraag die volgens doorlichting §3.1 in 15.192 regels documentatie nergens beantwoord wordt: *wie verliest hier, en waarom accepteert hij dat?*
*Wat er al ligt:* `data/pit_store/asset_class=crypto/dataset=funding/` — **8h-granulariteit, 6 symbolen, 2020–2026, gecertificeerd, in git**. Plus `reports/diag_funding_short_edge.csv` (90 datarijen) met per symbool/bucket een gemeten `carry`.
*De rekensom die je als eerste maakt, vóór je één regel code schrijft:* die diagnose geeft voor ETH `carry ≈ 6,5e-5` per 8h in de `all`-bucket (≈ 0,65 bps per 8h ≈ 1,95 bps/dag) en `≈ 3,44e-4` (≈ 3,4 bps per 8h) in de `fund_z>=2`-bucket. Tegen een vaste 13,0 bps per round trip betekent dat: een positie moet **meerdere dagen** aangehouden worden voordat de carry de kosten dekt, en de bucket met de hoogste carry (`fund_z>=2`, N=133) heeft een **negatieve** `mean_ret` van −2,8e-4 — de prijsbeweging eet de carry op. Reken dit per symbool en per bucket exact door, met de impactterm uit C1, en rapporteer de houdduur waarbij carry − kosten − prijsdrift positief wordt. Bestaat die houdduur niet, dan is kandidaat B klaar en gefalsificeerd, en dat is een volwaardig resultaat.
*Rol van het volatiliteitsmodel hier:* σ̂ bepaalt de positiegrootte en de drempel waarboven je de carry niet meer wilt hebben (hoge vol = de kans dat de prijsbeweging de carry overtreft stijgt). Dat is een echte, toetsbare rol voor GARCH — en het is de enige van de drie kandidaten waarin de vol-modellering aan een verdienmechanisme vastzit in plaats van aan een noemer.
*Poort:* pre-registreer vóór de eerste run, met de echte M, en met het kostenmodel uit `conf/execution/fees.yaml` in de pre-registratie zelf.

**Kandidaat C — variantie als verhandelbaar object (variance/vol risk premium)**
*Mechanisme:* het best gedocumenteerde risicopremie-mechanisme in de literatuur; verkoper van variantie wordt betaald voor het dragen van gap-risico.
*Blokkade:* dit vereist opties- of gestructureerde volatiliteitsproducten. De gecertificeerde store bevat die niet, en `reports/TCA_CALIBRATION_REPORT.md:13-18` meldt dat zelfs `orderbook_l2` en `trades` afwezig zijn.
*Besluit dat je moet nemen en opschrijven:* **out of scope tenzij** er een gecertificeerde databron met PIT-garantie wordt geregeld. Schrijf op wat die bron zou moeten leveren, wat hij kost, en wat hij zou beantwoorden. Bouw er niets voor.

### 6.4 De kostentoets die elke kandidaat moet passeren

Geen kandidaat gaat door naar implementatie zonder dit tabelletje ingevuld, met de eigen gemeten getallen:

| | Kandidaat A | Kandidaat B | Kandidaat C |
|---|---|---|---|
| Bruto-edge per round trip (bps, OOS) | | | |
| Kosten per round trip (bps, uit C1/C3) | | | |
| Netto (bruto − kosten) | | | |
| Houdduur waarbij netto > 0 | | | |
| Aantal onafhankelijke trades in het OOS-venster | | | |
| Pre-registratie-id | | | |
| M op moment van de toets | | | |
| DSR-p (Bailey/LdP-**kans**, niet `raw − penalty`) | | | |

**Drempel:** netto > 0 én ≥ 3 symbolen met elk ≥ 100 trades én DSR-p ≥ 0,95 bij de echte M. Haalt geen enkele kandidaat dat, dan is het antwoord op V1 nee en is de deliverable het afsluitdocument uit §9.

---

## 7. WAT "PROFESSIONELER" MEETBAAR BETEKENT

De opdrachtgever vraagt om een professioneler niveau. Dat is geen stijl maar een lijst afdwingbare eisen. Onderstaande punten gelden repo-breed, ongeacht welke kandidaat overleeft.

1. **Eén canonieke metriekbron.** `reports/metrics/<run_id>.json` plus `reports/CANONICAL.json`. Verplichte velden, run crasht zonder: `run_id, generated_at_utc, git_sha, git_dirty, config_hash, dataset_hash, preregistration_id, sample:{in|out}, window_start, window_end, symbols[], n_trades_per_symbol{}, cost_model:{fees_bps, spread_bps, slippage_bps, impact_model}, gross_sharpe, net_sharpe, deflated_sharpe_probability, M_trials, pbo, spa_p_{lower,consistent,upper}, max_drawdown, calmar, avg_gross_leverage, ledger_entry_id`. Geen enkel getal uit `reports/` mag geciteerd worden als het niet in `CANONICAL.json` staat.
2. **De deflatie repareren.** `count_git_commits()` als M-proxy verdwijnt (`tune/objective.py:790`) en wordt vervangen door de ledger-M. `cap_factor` terug naar 5,0. `evaluation.py:896-906` gebruikt `metrics.py:168-228` (de Bailey/LdP-**kans**) in plaats van `_deflated_sharpe_penalty_legacy`. **Eén test die faalt zodra de M-proxy weer aan de git-historie hangt** — dat is precies de fout die niemand zag omdat niets erop toetste.
3. **`n_trades` betekent trades.** `run_hrp_backtest.py:375-377` telt `np.count_nonzero(t.side)` = bars met een positie. Hernoem het veld naar `n_active_bars` en voeg een echte trade-teller toe. Elke kostenberekening die op het oude veld leunde, is fout.
4. **Turnover en kosten in élke run.** `total_rebalance_cost` wordt een verplicht veld. Een backtestresultaat zonder kostenveld is geen resultaat.
5. **Eén engine, één latentieconventie.** `shift(2)`, afgedwongen door de test uit `reports/phase5_engine_diff.md`. De `shift(1)`-aanname in `baseline_runner.py` ("handelen op de close waarop je besloot") mag nergens meer voorkomen.
6. **Volatiliteitsvergelijkingen op de juiste metrieken.** QLIKE als primaire loss (robuust tegen proxy-ruis onder Patton 2011), Mincer-Zarnowitz voor bias/efficiëntie, Diebold-Mariano met HLN-correctie voor significantie, Newey-West voor de seriële correlatie. Dit staat al in `validation/vol_metrics.py` — gebruik het, schrijf geen tweede implementatie. Een vergelijking op RMSE van de standaarddeviatie alleen is niet publiceerbaar.
7. **Convergentie wordt gerapporteerd, nooit weggevangen.** Een niet-geconvergeerde GARCH-fit valt niet terug op EWMA; hij wordt geteld. Onder 90 % convergentie vergelijk je een selectie van makkelijke vensters met een estimator die overal een waarde geeft.
8. **Data-herkomst of een expliciete erkenning.** `.dvc/config` bevat alleen `[core] no_scm = True` — geen remote. Configureer er één, of schrijf op dat `market_data_parquet/` en `artefacts/tracks/*.joblib` **definitief verloren** zijn en dat elk resultaat dat erop leunde daarmee onherstelbaar is. Dat laatste is een geldig besluit; stilte erover is dat niet.
9. **`README.md` bestaat.** `pyproject.toml:13` verwijst ernaar, setuptools slikt de afwezigheid stil en levert een lege long-description. Eén pagina: wat dit is, wat het niet is, welk cijfer geldig is, en welk cijfer nadrukkelijk niet.
10. **De doctrine wordt bindend of verdwijnt.** `python scripts/audit_fallbacks.py --strict` vindt 37 × `SWALLOWED_EXCEPT` en classificeert ze alle 37 als *advies*. `ruff check src apps` geeft 211 errors; `make loc-check` faalt op `live/engine.py` (905 > 800). Een poort die niets blokkeert is documentatie. Kies per poort: blokkerend maken of verwijderen.

---

## 8. DELIVERABLES

1. **`reports/VERVOLGBESLUIT_<datum>.md`** — het besluitdocument. Bevat, in deze volgorde: het antwoord op V1/V2/V3 in maximaal drie zinnen elk; de ingevulde kostentoets-tabel uit §6.4; de schrapinventaris met regels en bewijs; en het besluit per kandidaat (`DOORGAAN` / `GESTOPT` / `UNPROVEN`).
2. **`reports/GARCH_VS_EWMA_COMPETITION.md`** — de uitkomst van H1, met ledger-entry en pre-registratie-id, als kandidaat A is uitgevoerd.
3. **De schrapcommits** — thematisch, elk met importscan-bewijs, achter de tag uit §5.3.
4. **`frozen/README.md`** — per bevroren onderdeel de gemeten voorwaarde waaronder het weer aan mag.
5. **Een bijgewerkt `docs/FALSIFICATION_REGISTER.md`** — elke negatieve uitkomst uit dit spoor komt erin. Dat register is het meest waardevolle bestand van deze repo; laat het niet verlopen.

---

## 9. STOPREGELS EN FALSIFICATIE

Leg deze vooraf vast, vóór de eerste run, in de pre-registratie:

* **Kandidaat A** valt op elk van de vier criteria in `preregistration_h1_garch_vs_ewma.yaml`. Een vijfde modelvariant "omdat de eerste vier het niet haalden" is de zet die pre-registratie uitsluit.
* **Kandidaat B** valt zodra er geen houdduur bestaat waarbij carry − kosten − prijsdrift positief is op ≥ 3 symbolen met elk ≥ 100 onafhankelijke trades.
* **Kandidaat C** is gestopt tot er een gecertificeerde databron ligt.
* **Het geheel** valt zodra §6.4 voor alle drie de kandidaten leeg of negatief blijft. In dat geval schrijf je het afsluitdocument: wat er bewezen is, wat er meegenomen kan worden naar een volgend project (de PIT-datalaag, de kostenboekhouding, het pre-registratiemechanisme), en wat er definitief niet werkte. Dat document is een volwaardige oplevering en geen nederlaag.

**Eén regel die boven alle andere staat:** verruim geen drempel achteraf omdat het resultaat tegenvalt. De auteur van deze repo heeft dat zelf al een keer weerstaan (`docs/WAVE_LOG.md:470-495`, `cm_tsmom` op ARCHIVED gezet tegen zijn eigen bedenkingen in). Die norm is de enige die in dit project onbeschadigd is; hij is het waard om gehandhaafd te worden.

---

## 10. VERBODEN ZETTEN

* Een nieuwe sleeve toevoegen, onder welke naam dan ook.
* Risicoparameters buiten `conf/` om verhogen. `miscellaneous/run_hrp_backtest.py:6-11` deed dat (vol-target 0,12 → 0,25; leverage 4,0 → 6,0; kelly 0,25 → 0,40; dd-breaker 0,20 → 0,30) en produceerde daarmee de Sharpe van 4,31.
* Een resultaat rapporteren zonder provenance-velden uit §7.1.
* M vergroten met een sweep vóór er een pre-registratie ligt.
* Infrastructuur bouwen (k8s, Prometheus, champion/challenger, shadow, MRM) vóór §6.4 groen is. Dat is de volgorde-fout, en hij kostte dit project ~1.560 regels plus 8 manifests.
* Een niet-uitgevoerde meting als negatief resultaat presenteren, of een onderpowerde toets als falsificatie.
* De vier bekende poort-failures groen maken door de poort te verzwakken.

---

## 11. MEETPROTOCOL

Lever bij het besluitdocument een bijlage met:

* Elk gebruikt commando met zijn letterlijke uitvoer.
* De AST-importscan (`import_scan.py` → `import_scan.json`), inclusief relatieve en lazy imports, met per module: regels, externe importers, src-module-importers, testbestanden.
* De sha waarop gemeten is, en `git status --short` (moet leeg zijn).
* Per afwijking van deze prompt: wat er anders was, en welke meting dat aantoonde.

Waar een getal niet vast te stellen was, schrijf je op in welk bestand het zou staan en waarom het er niet is. Dat is wat §7 van de doorlichting deed, en het is de reden dat dat rapport bruikbaar is.
