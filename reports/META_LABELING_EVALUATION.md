# H3 — CATBOOST ALS SECONDARY MODEL

> **Deliverable 22** · Phase 6 stappen 12 en 13 · Phase 7/8 Stage C-3
> **git_sha:** `df96805`
> **Pre-registratie:** `56395fa2013768014c0c915edf346770` (bevroren 2026-08-25T17:52:22.307814+00:00, M bij bevriezing = 2770)
> **Universum:** 6 gecertificeerde reeksen · 1743 bars · 2021-11-15 t/m 2026-08-23
> **Primaire track:** `xs_momentum_risk_parity` · 10.330 triple-barrier events · embargo 5 bars bovenop de purge op `t1`
> **Kostenlabels:** `IMPACT_UNCALIBRATED` · `SPREAD_ASSUMED` (1.0 bp) · risk `config_hash` `1b60cb664fbf9a2a`

## 0. De uitkomst, eerst

**Geen enkele spec is gepromoveerd. CatBoost blijft `ARCHIVED`, zoals §24 vastlegt.**

| Oordeel | Aantal | Betekenis |
|---|---|---|
| PROMOTED | 0 | AUC-ondergrens boven 0,58 én netto economische waarde door de engine |
| FALSIFIED | 0 | de negatieve controle lekt — de RUN is ongeldig, niet de hypothese |
| DESCOPED | 0 | te weinig effectieve events; `UNPROVEN — insufficient data` |
| ARCHIVED | 6 | gemeten en de drempel niet gehaald; CatBoost blijft gearchiveerd zoals §24 vastlegt |

De beste van de 6 is `catboost-d3-lr0.1` met een OOS-AUC van 0.4842. **Alle 6 liggen ONDER 0,50** (van 0.4768 tot 0.4842), en het conservatieve interval van de beste ([0.4113, 0.5572]) omvat 0,50. Het model rangschikt de succeskans van het primaire signaal dus niet aantoonbaar beter dan een muntworp — een AUC onder 0,50 is op deze steekproefgrootte geen anti-signaal maar ruis, en §5 laat zien waarom.

## 1. Wat vooraf vastlag

| | |
|---|---|
| Nulhypothese | de OOS-AUC is niet te onderscheiden van 0,50 |
| Primaire maat | `oos_auc` |
| Drempel | AUC > 0.58 (`ARCHITECTUUR_AUDIT_2026-08-22.md §24`) |
| Toets | Hanley-McNeil-interval om de AUC, op de CONSERVATIEVE effectieve steekproefgrootte |
| Geplande trials | 6 (3 dieptes × 2 leerstappen) |
| Barrières | 2.0σ / 2.0σ / 10 bars, entry op `t+1` |

De power-analyse stond vóór de run vast, en haar uitkomst was dat H3 **precies op de grens** ligt:

| Variant | effectieve n | MDE (AUC-overschot) | informatief? |
|---|---|---|---|
| nominal | 7200.0 | 0.0190 | ja |
| uniqueness_corrected | 1167.8 | 0.0472 | ja |
| uniqueness_and_cross_section | 246.2 | 0.1029 | nee |

Aan welke kant van de grens H3 valt, hangt af van een keuze die niet uit de data volgt: twee labels van verschillende symbolen op dezelfde bar zijn verschillende trades met verschillende uitkomsten, maar hun uitkomsten zijn gecorreleerd. De pre-registratie legde daarom vast dat het interval op BEIDE wordt gerapporteerd en dat het oordeel op de conservatieve valt.

## 2. De negatieve controle — en zij is schoon

Hetzelfde model, dezelfde folds, dezelfde purging, maar met **gerandomiseerde labels**. De klassebalans blijft per constructie gelijk (er wordt gepermuteerd, niet opnieuw getrokken), dus een verschil in AUC kan niet uit een andere basisrate komen.

| | |
|---|---|
| AUC op gerandomiseerde labels (max over 5 replicaties) | **0.5100** |
| replicaties | 0.4857, 0.5007, 0.5100, 0.4936, 0.4897 |
| grens waarboven de RUN wordt gefalsificeerd | 0.55 |

Dit getal is de belangrijkste van het hele rapport, en niet omdat het gunstig is. Het zegt dat de pipeline NIET lekt: geen scaler over folds heen, geen feature die de toekomst raakt, en een purging die doet wat zij belooft. Was hij 0.55 of hoger geweest, dan was elk ander getal hieronder waardeloos geweest — inclusief de AUC's die de hypothese hadden kunnen steunen.

## 3. De events, en waarom het nominale aantal misleidt

| | |
|---|---|
| triple-barrier events (gepoold) | 10.330 |
| positieve klasse | 0.4959 |
| gemiddelde uniqueness | 0.1591 |
| effectief, nominaal (testfolds) | 7200 |
| effectief, na uniqueness én cross-sectie | 241.6 |

Met een verticale barrière op 10 bars en een event op vrijwel elke bar delen buren 9 van hun 10 toekomstige bars. De gemeten uniqueness van 0.1591 zegt dat 10.330 labels ongeveer 1644 onafhankelijke waarnemingen waard zijn. Bovenop die overlap staat de cross-sectionele afhankelijkheid: 6 perpetuals met een gemeten gemiddelde correlatie van 0.7485 zijn 1.27 onafhankelijke reeksen waard en geen 6.

**De adequaatheidspoort bindt hier NIET, en dat vraagt uitleg**, want er zijn twee getallen die allebei 'effectieve events per fold' heten:

| Grootheid | gemeten | rol |
|---|---|---|
| gepurgede TRAINevents per fold (Data Adequacy Gate) | 442.7 | stop-criterium 2, drempel 100 — bindt niet |
| TESTevents per fold (deze run) | 85.9 | bepaalt de PRECISIE van de AUC, geen poort |

Stop-criterium 2 zegt letterlijk *"De Data Adequacy Gate meet het effectieve aantal labels per fold"*, en die poort heeft een implementatie en een artefact: hij telt de GEPURGEDE TRAINevents, want dat is wat bepaalt of er gefit mag worden ('zonder gefit te zijn'). Die staat op 442.7, ruim boven de eis. Het testfold-aantal is een andere grootheid met een andere rol — het bepaalt hoe scherp de AUC te meten valt — en het staat hier expliciet naast, zodat niemand ze later door elkaar haalt.

## 4. De AUC, met beide intervallen

| Spec | OOS-AUC | 95 %-interval, nominaal | 95 %-interval, conservatief | oordeel |
|---|---|---|---|---|
| `catboost-d3-lr0.03` | 0.4821 | [0.4688, 0.4954] | [0.4092, 0.5550] | ARCHIVED |
| `catboost-d3-lr0.1` | 0.4842 | [0.4709, 0.4976] | [0.4113, 0.5572] | ARCHIVED |
| `catboost-d4-lr0.03` | 0.4768 | [0.4635, 0.4901] | [0.4040, 0.5497] | ARCHIVED |
| `catboost-d4-lr0.1` | 0.4807 | [0.4674, 0.4940] | [0.4078, 0.5536] | ARCHIVED |
| `catboost-d6-lr0.03` | 0.4795 | [0.4662, 0.4928] | [0.4066, 0.5524] | ARCHIVED |
| `catboost-d6-lr0.1` | 0.4826 | [0.4693, 0.4959] | [0.4097, 0.5555] | ARCHIVED |

De AUC weegt de **uniqueness** mee: hij is de kans dat een willekeurig getrokken geslaagde trade hoger scoort dan een willekeurig getrokken mislukte, waarbij beide worden getrokken proportioneel aan hun uniqueness. Zonder die weging telt elk van tien overlappende buren als een volwaardige waarneming, en dat is precies waarom meta-labeling-AUC's in de literatuur zo vaak te hoog uitvallen.

Het verschil tussen de twee intervallen is de hele les van §1: op `catboost-d3-lr0.1` is het nominale interval 5.5 keer zo smal als het conservatieve (0.0267 tegen 0.1458 AUC breed). Een pipeline zonder uniqueness-correctie zou dat smalle interval rapporteren, en een AUC die de drempel van 0.58 nét raakt, zou daarmee 'significant' heten waar het conservatieve interval de drempel nog ruim omvat.

## 5. Het werkpunt: precision, recall en wat er wordt weggefilterd

Het OORDEEL valt op één werkpunt: kans ≥ 0.50, de natuurlijke beslisgrens van een kans. Er is niet geprobeerd welk werkpunt de mooiste cijfers geeft; de volledige curve staat hieronder zodat een lezer ziet wat een ander werkpunt zou hebben gedaan.

Curve voor `catboost-d3-lr0.1`, de beste van de zes:

| drempel | precision | recall | aandeel behouden | events behouden |
|---|---|---|---|---|
| 0.35 | 0.4929 | 0.7244 | 0.7290 | 5.261 |
| 0.40 | 0.4921 | 0.6440 | 0.6492 | 4.693 |
| 0.45 | 0.4879 | 0.5492 | 0.5585 | 4.045 |
| 0.50 | 0.4823 | 0.4551 | 0.4681 | 3.397 |
| 0.55 | 0.4780 | 0.3686 | 0.3825 | 2.778 |
| 0.60 | 0.4734 | 0.2875 | 0.3013 | 2.203 |
| 0.65 | 0.4741 | 0.2211 | 0.2314 | 1.691 |

De basisrate is 0.4961: zoveel van de trades slaagde sowieso. Een filter voegt pas iets toe wanneer zijn precision daar BOVEN ligt — een filter met precision gelijk aan de basisrate selecteert willekeurig en houdt alleen minder over.

## 6. Door de authoritative engine — de economische toets

Een AUC is een statistisch resultaat, geen economisch. Stap 13 van de fase-opdracht draait het gefilterde signaal door dezelfde engine met volledige kosten: *"een filter dat de helft van de trades weghaalt en de Sharpe met 0,02 verbetert, verdient geen promotie"*.

| Arm | netto OOS Sharpe | Δ vs. ongefilterd | turnover | fees | bars met positie |
|---|---|---|---|---|---|
| ongefilterd | -0.0998 | — | 2.698.250 | 1484.04 | 1.200 |
| `catboost-d3-lr0.03` | -0.4851 | -0.3853 | 2.045.739 | 1125.17 | 285 |
| `catboost-d3-lr0.1` | -0.6325 | -0.5328 | 906.015 | 498.32 | 114 |
| `catboost-d4-lr0.03` | -0.2387 | -0.1389 | 4.903.435 | 2696.89 | 753 |
| `catboost-d4-lr0.1` | -0.4693 | -0.3696 | 2.317.842 | 1274.82 | 281 |
| `catboost-d6-lr0.03` | -0.0548 | +0.0449 | 4.740.246 | 2607.13 | 702 |
| `catboost-d6-lr0.1` | -0.6019 | -0.5022 | 1.802.304 | 991.26 | 224 |

**1 van de 6 specs verbetert de netto Sharpe: `catboost-d6-lr0.03`.** De grootste verbetering is +0.0449 op `catboost-d6-lr0.03`, van -0.0998 naar -0.0548 — op een spec met een OOS-AUC van 0.4795, dus ONDER 0,50. Een filter zonder gemeten voorspellende waarde dat de Sharpe met 0.04 beweegt, is exact het geval waar stap 13 voor waarschuwt: *"een filter dat de helft van de trades weghaalt en de Sharpe met 0,02 verbetert, verdient geen promotie"*. Het bindende criterium is bij deze spec dan ook niet de engine maar de AUC (`auc_below_threshold`).

**Waar deze cijfers op rusten.** Het ongefilterde boek heeft een positie op 1.200 OOS-bars; gefilterd zakt dat naar 114–753. Voor `catboost-d3-lr0.1` blijven er 114 over — 9.5% van het origineel. Een Sharpe over zo weinig bars met positie draagt een brede foutmarge, en dat is een reden temeer om het oordeel niet op de engine-delta alleen te laten rusten. De AUC-drempel bindt hier bij alle specs.

Elke arm draait door dezelfde `EventDrivenEngine`, met dezelfde risicolaag, router, venue en kosten. De ENIGE ingang die verschilt is de exposure: de basisexposure maal het filter (1 doorlaten, 0 tegenhouden). Buiten de gescoorde events staat het filter op 1 — daar heeft hij geen oordeel, en een 0 zou een bewering zijn in plaats van een onthouding.

## 7. Feature importance — MDI naast SFI

§12.1 eist er twee, en zij meten verschillende dingen. **MDI** komt uit de fit die er toch al was, is IN-SAMPLE en deelt het belang van features die hetzelfde meten. **SFI** geeft elk model één feature en scoort hem OUT-OF-SAMPLE onder dezelfde purged folds; substitutie kan daar niet optreden, interactie wordt daar niet gezien.

Beide op `catboost-d3-lr0.1`, de beste van de zes.

| Feature | MDI (aandeel) | SFI (OOS-AUC alleen op deze feature) |
|---|---|---|
| `vol_expanding` | 0.1279 ± 0.0233 | 0.4750 ± 0.0529 |
| `vol_realized_60` | 0.0918 ± 0.0168 | 0.4903 ± 0.0504 |
| `micro_funding_mean_21` | 0.0917 ± 0.0232 | 0.5120 ± 0.0299 |
| `mom_logret_60` | 0.0884 ± 0.0120 | 0.4938 ± 0.0414 |
| `vol_parkinson_20` | 0.0727 ± 0.0083 | 0.5000 ± 0.0419 |
| `mom_ewma_spread` | 0.0659 ± 0.0105 | 0.4892 ± 0.0330 |
| `vol_realized_20` | 0.0604 ± 0.0071 | 0.4963 ± 0.0354 |
| `vol_realized_5` | 0.0592 ± 0.0103 | 0.5113 ± 0.0231 |
| `vol_ewma` | 0.0555 ± 0.0121 | 0.4858 ± 0.0322 |
| `mom_logret_20` | 0.0522 ± 0.0098 | 0.5146 ± 0.0337 |
| `mom_logret_5` | 0.0522 ± 0.0054 | 0.5121 ± 0.0196 |
| `micro_oi_logchg_7` | 0.0506 ± 0.0109 | 0.4892 ± 0.0215 |
| `micro_funding_zscore` | 0.0488 ± 0.0098 | 0.4941 ± 0.0296 |
| `micro_funding_mean_3` | 0.0476 ± 0.0090 | 0.4966 ± 0.0188 |
| `micro_oi_logchg_1` | 0.0349 ± 0.0045 | 0.4932 ± 0.0158 |

Geen van beide getallen promoveert of blokkeert iets: zij zijn diagnostiek. Wat zij hier laten zien is dat MDI wel degelijk structuur toewijst — het model splitst het meest op `vol_expanding` (0.1279 van het totaal) — terwijl SFI voor élke feature dicht bij 0,50 uitkomt: de verste ligt 0.025 van 0,50 af (`vol_expanding`, 0.4750). Dat is het patroon van een model dat in-sample structuur vindt die out-of-sample niet bestaat, en het is consistent met de AUC's in §4.

## 8. De ledger en `M`

| | |
|---|---|
| `M` vóór deze run | 2776 |
| `M` ná deze run | 2776 |
| trials geboekt bij het bevriezen | 6 |
| trials in deze run gedraaid | 6 |

De zes trials zijn bij het BEVRIEZEN van de pre-registratie geboekt. Deze run voert ze uit en boekt ze niet opnieuw; het oordeel gaat als amendement terug de ledger in, met `n_trials = 0`.

De negatieve controle telt NIET mee in `M`: zij toetst de PIPELINE en niet de markt. Vijf replicaties van hetzelfde model op gepermuteerde labels zijn geen vijf hypothesen over rendement.

## 9. Wat hiermee NIET is getoetst

1. **Andere barrièrebreedtes.** 2σ / 2σ / 10 bars staat in `conf/model/labeling.yaml` en is niet gezocht. Breedtes variëren tot de AUC de drempel haalt, is de meest directe manier om deze hypothese te vervalsen zonder het te merken; dat vraagt een nieuwe pre-registratie met een eigen bijdrage aan `M`.
2. **Andere features.** De featureset is de gecertificeerde Phase 2-registry uit `conf/features/default.yaml`, ongewijzigd. Deze fase evalueert MODELLEN en geen features (§2).
3. **Een ander werkpunt.** De curve in §5 staat er ter informatie; het oordeel valt op 0.50. Een werkpunt kiezen op de uitkomst is een trial die niet is geboekt.
4. **Directionele voorspelling.** Technisch geblokkeerd: er bestaat geen handtekening waarmee dit model een eigen doelvector krijgt. §24 houdt CatBoost daarvoor gearchiveerd en deze run verandert daar niets aan.
5. **Een effect kleiner dan 0.103 AUC-overschot.** Dat is de MDE op de conservatieve effectieve steekproefgrootte (246.2 waarnemingen). Het overschot dat de drempel van 0.58 vraagt, is 0.080 — kleiner dan de MDE, dus een AUC die die drempel nét zou halen, is op deze opzet niet van 0,50 te onderscheiden. Dat lag vóór de run vast.

Elke promotieclaim in dit rapport draagt de labels `IMPACT_UNCALIBRATED` en `SPREAD_ASSUMED` (1.0 bp). Er zijn geen promotieclaims.

