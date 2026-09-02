# FALSIFICATION REGISTER — bindend, append-only

> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-3).
> GEMETEN: de drie geciteerde ledger-amendementen bestaan in
> `artefacts/governance/hypothesis_ledger.json`. Bij die controle bleek
> het H2-amendement met een ONJUIST `config_hash` te zijn geciteerd; zie
> de correctie bij H2.

> Mandaat v3 §3. Her-testen van een item hieronder zonder aantoonbaar gewijzigde
> premisse = verspilde DSR-trials en een mandaatschending. Nieuwe falsificaties
> worden onderaan toegevoegd met bewijslink; bestaande regels worden nooit
> verwijderd of afgezwakt.
>
> Bewijsbronnen: `docs/TRUE_ALPHA_AUDIT_2026-06-08.md` (hierna AUDIT),
> `docs/SHORT_ALPHA_RESEARCH_2026-05-30.md` (hierna SHORT).

| # | Gefalsifieerde hypothese | Bewijs | Heropening alleen als… |
|---|--------------------------|--------|------------------------|
| F1 | Funding-rate als standalone alpha of conditioner (crypto) | AUDIT §11: IC ≈ 0 alle horizonten, n=176k; breadth maakt het erger (Sharpe −0.14…+0.18); conditioning schaadt (0.61→−1.03) | ander mechanisme (bv. funding-basis cross-exchange arb met executiebewijs) |
| F2 | Tick/intraday OFI & runs-imbalance op info-bars (crypto) | AUDIT §7: IC −0.008…−0.019 echt maar net −7…−13 @10bps (~32 bars/dag × 0.32 turnover) | kostenstructuur verandert (maker-rebates aantoonbaar haalbaar) |
| F3 | Per-asset directionele ML (long/short probs per naam) | AUDIT §10b, §12A: OOS AUC 0.504–0.523 op 5 én 99 namen; 198 modellen: boek-Sharpe −0.59; "tracks Sharpe 2.18" = constructie-artifact | fundamenteel andere featureklasse per markt (niet crypto-OHLCV) |
| F4 | Triple-barrier AUC als bewijs van edge | AUDIT §12 (Silent Killer): AUC 0.55–0.59 = vol/path-artifact, geen richting-informatie; elk verhandelbaar boek erop ≤ 0 | nooit — AUC zonder verhandelbaar boek telt per definitie niet |
| F5 | Markt-brede alt-data als directionele timer (F&G, TVL, DVOL, on-chain) | AUDIT §14B: timing-AUC 0.45, IC −0.10, breadth ≈ 1 → sample-starved | als conditioner/de-grosser mag het wél getest (geen timer) |
| F6 | Feature-maximalisatie op het XS-model (52 features) | AUDIT §14C: IC 0.064 ≤ baseline 0.069; kleinere samples: overfit (IC +0.015) | nieuwe informatie-bron, niet meer transformaties van dezelfde prijs |
| F7 | Multi-TF trend/RR-sweeps (crypto, 40 configs) | AUDIT §15A: geen config elk jaar positief; WF-winnaar verliest OOS (−0.28); RR-ratio irrelevant | trend in ANDERE asset-klasse (managed-futures-premie is academisch reëel) |
| F8 | Hand-made trend-heuristieken (MA-cross, XSMOM-long crypto) | AUDIT §10: Sharpe ≤ 0.3, MaxDD −90%+, ruïne onder leverage; crypto is XS-reversal, niet -momentum | n.v.t. binnen crypto; in equities/FX/commodities is XSMOM een ander, gedocumenteerd premium |
| F9 | Symmetrische long-mirror short-pipeline (TrendScan + PT2/SL1) | SHORT D1–D6: shorts onvoorwaardelijk −EV; Sharpe 2.92→1.31 bij meer shorts; AUDIT §10b: shorts toevoegen → −3.26 | nooit in deze vorm; short-edge komt uit carry/reversal/relative-value |
| F10 | Naïeve breedte ("meer namen/modellen = meer Sharpe") | AUDIT §12: 99 namen per-asset −0.59; §13: raw reversal +0.30 (12) → +0.06 (99); breadth meet nul-edge preciezer | breedte over markten/premia (mandaat v3) i.p.v. namen binnen één betafactor |
| F11 | Combination-tuning boven risk-parity | AUDIT §7: Sharpe-prop 0.71 / μσ² 0.69 < RP 0.94 — schattingsruis | bewezen stabiele μ-schatter (bv. shrinkage met OOS-bewijs) |
| F12 | Harvest-parameter-tuning als edge-bron | AUDIT §17: PBO 0.51 op 72 varianten (CSCV, S≥50) — config-ranking is ruis | n.v.t.; harvest simpel en vast houden |

## Appendix — nieuwe falsificaties (append hieronder, nooit hierboven wijzigen)

<!-- Template:
| F<n> | <hypothese> | <bewijs: script/artefact/doc §> | <heropeningsconditie> |
-->

| # | Gefalsifieerde hypothese | Bewijs | Heropening alleen als… |
|---|--------------------------|--------|------------------------|
| F13 | XSMOM 12-1 (JT1993) als netto-positieve unit op S&P-large-caps (~190 namen/dag, decile-spread $-neutraal, maandelijks, 2000–2026, 6bp+1bp/zijde) | W22 run 3: net Sharpe −0.34, gross −0.27, 9/27 jaren positief (`artefacts/wave20_runlog/w22_eq_units.log`; ledger c3cef72def149ffd) | breder/kleiner-cap universum met PIT-data; óf G4-MOM-loading-check toont lage loading (= constructiefout) zodra factordata bereikbaar |
| F14 | Low-vol als netto unit op S&P-large-caps — v1 $-neutraal vol-rank én v2 FP-conform beta-levered | v1 −0.82 (beta-bleed), v2 −0.12 (na correctie nog ≤0); 11/27 jaren positief (ledger 2b521b71d71864b9); G4-diag: restbèta +0.075, α −2%/jr t=−1.1 (geen artifact, premium afwezig) | beta-RANK BAB op breder universum; óf G4 met AQR BAB-factordata toont onverwachte α |
| F15 | Overnight-persistentie (LPS 2019) als close-to-close verhandelbare unit op S&P-large-caps (21d-formatie, maandelijks) | W23c: net −0.60 / gross −0.39; G4 α −3.0%/jr t=−3.01 (significant NEGATIEF); corr met strev −0.51 (`artefacts/wave20_runlog/w23c_eval.log`; ledger b395b1a409e98aa3) | uitvoering op open/close-auctions zelf (het LPS-construct — andere executieroute, G10-bewijs vereist); NOOIT als post-hoc sign-flip (§1.3) |

Geen falsificatie maar archivering (lat niet gehaald): `eq_strev_1m` net +0.39
(< 0.40) mét decay-staart 2024/25/26 = −3.6/−6.6/−7.2% — edge sterft richting
het live-venster (Wave-18-patroon). Formele G4 (FF5+MOM): **α +3.1%/jr,
t=2.47 PASS** — het premium bestaat, de rauwe oogst haalt de Sharpe-lat niet.
Heropening als **residual/industry-adjusted reversal** (Da-Liu-Schaumburg
2014) — geprobeerd in W24, zie F16.

| # | Gefalsifieerde hypothese | Bewijs | Heropening alleen als… |
|---|--------------------------|--------|------------------------|
| F16 | Residual (mkt-beta-252d) short-term reversal op S&P-large-caps (DLS2014-route) | W24: net −0.35, **gross −0.15** (raw strev gross was +0.56 — residualisatie vernietigt het signaal hier); G4 α −1.9% t=−1.8 (`w24_strev_resid.log`) | industry-adjusted variant met echte industriedata; óf small/mid-cap-universum |
| F17 | Quality GP/A (Novy-Marx 2013) als netto unit op S&P-large-caps, PIT-EDGAR 2010–2026 | W23b: net −0.25; G4: RMW-loading +0.114 (t=11.8) = getrouwe bouw, α −1.4% t=−1.65 (`w23b_quality.log`; 319 tickers met GrossProfit — financials vallen uit) | breder universum; óf composiet-quality (meerdere metrics) als ECHT ander construct |
| F18 | PEAD via announcement-return (BT1989/CJL1996), anchor = EDGAR-acceptance, S&P-large-caps 2010–2026 | W23a: net −0.18, gross −0.05; G4 α t=−0.6, R²=0.04 (`w23a_pead.log`; 23k events, 622 tickers) | 8-K item-2.02-anchor (echte persbericht-timestamp) op small/mid-caps; SUE-variant vereist quarterly-start-veld (her-ingest) |

**Hard patroon na F13–F18 (zes constructen, twee informatiebronnen):** het
S&P-large-cap-universum (~190 PIT-namen/dag) levert 2000/2010–2026 GEEN
netto verhandelbaar XS-premium van welke klassieke soort dan ook. Breedte
moet uit small/mid-caps (PIT-data nodig) of andere markten komen — niet uit
meer constructen op deze 190 namen (F10-logica).

| # | Gefalsifieerde hypothese | Bewijs | Heropening alleen als… |
|---|--------------------------|--------|------------------------|
| F19 | Crypto intraday klok-seizoenaliteit (uur-van-dag/dag-van-week/funding-venster 00-08-16 UTC) als netto unit op 6 USDT-perps (1h-bars uit 5s-data, 2021-06–2026-05) | W26: beste uur t=2.11 (max over 24 → NS na multiple-comparison), DOW max |t|=1.67, funding-venster t≈−0.5; beste aaneengesloten venster (in-sample max over 288) +7.4 bps/dag bruto < 20 bps/dag taker-round-trip (`scripts/w26_seasonality_diag.py`, `w26_diag.log`; ledger W26, n_trials=321) | maker-execution aantoonbaar haalbaar (zelfde conditie als F2); óf veel breder perp-universum met cross-sectionele klok-effecten (≥50 namen) |

| # | Gefalsifieerde hypothese | Bewijs | Heropening alleen als… |
|---|--------------------------|--------|------------------------|
| F20 | Energie-termijnstructuur-carry (Gorton-Rouwenhorst 2006 / Koijen et al. 2018) als netto unit op de EIA NYMEX Contract 1-4-archief — WTI/NG/HO/RBOB, tijdreeks-carry `(F1−F2)/F1`, slot-2 gehouden, maandelijkse herbalancering, 1994–2024 (n=7585, 30,1 jaar) | W28: net Sharpe **+0,100** (gross 0,148 — de rolkosten zijn het verschil), CAGR −0,65%/jr, 14/31 jaren positief (45,2%); **IS +0,454 → OOS −0,278, decay +161%**; G4-S3 α **−4,56%/jr t=−0,91** (S1 +3,89% t=0,87 → S2 +4,51% t=1,03 → S3 negatief zodra per-product passives meedoen); bootstrap P(S>0) 0,723 < 0,75; N_eff **1,54 op 4 producten**. 8 van 9 poorten rood. (`scripts/w28_cm_carry_eval.py`, `artefacts/killgates/cm_carry.json`; ledger W28) | een universum met échte breedte (N_eff ≫ 1,5 vraagt landbouw/metalen/vee-termijnstructuur, niet vier gecorreleerde energieproducten); **of** een termijnstructuurbron die ná 2024-04 doorloopt, zodat er überhaupt een recent OOS-venster bestaat. NOOIT door de horizon of het gehouden segment achteraf te wisselen — dat zijn trials, geen ontdekkingen. |

**Wat F20 toevoegt aan het patroon.** Dit is de zesde onafhankelijke keer dat
een gedocumenteerd premium in dit programma **IS-positief en OOS-negatief**
meet (F13, F16–F19, nu F20; en het is dezelfde vorm als de B-5-bevinding van
Setup B: Hong & Yogo +0,582 vóór 2008, −0,651 in de laatste tien jaar). De
carry-literatuur is niet weerlegd — de premie is er in de eerste helft van de
steekproef (IS +0,45). Wat gemeten wordt is dat hij in het regime dat een live
account zou verhandelen niet meer betaald wordt.

**En een tweede, aparte les — breedte was hier de bindende beperking, niet
het signaal.** N_eff = 1,54 op vier producten: WTI, HO en RBOB zijn vrijwel
hetzelfde risico (dezelfde raffinagecomplex-drijver) en NG is de enige echte
tweede bet. Grinold's IR ≈ IC·√breadth geeft met breadth ≈ 1,5 geen boek,
hoe goed de carry-schatter ook is. Dat is F10 opnieuw, nu in energie.

---

## INVALIDATIE - historische resultaten zonder gecertificeerde data-provenance

> **Toegevoegd 2026-08-29, Phase 2 exit-criterium 7** (uitgevoerd als Stage B
> van `Prompts-fases/fase_7_8_consolidatie_productie.md`). Dit register is
> append-only: geen enkele regel hierboven is gewijzigd, verwijderd of
> afgezwakt. Deze sectie voegt een STATUS toe, hij vervangt geen oordeel.

### Wat er is geherclassificeerd

| Groep | Aantal | Nieuwe status |
|---|---|---|
| Gefalsificeerde units `F1` t/m `F20` hierboven | **20** | `INVALID - no certified data provenance` |
| "Geaccepteerde" crypto-units uit Wave 20: **ML-XS**, **LOWVOL**, **REVERSAL-k10**, **CARRY** | **4** | `INVALID - no certified data provenance` |
| **Totaal geherclassificeerd** | **24** | |

De vier geaccepteerde units staan in de ledger onder
`w20_rebaseline_plus_4_crypto_units` (`result: accepted`, `n_trials: 5`), zelf
een RECONSTRUCTIE uit `docs/WAVE_LOG.md` regel 143 nadat het canonieke
ledgerbestand verloren ging.

### Waarom, en wat de status wel en niet betekent

Elk van deze 24 oordelen is geveld op data van voor Phase 1. Er is voor geen van
die runs een gecertificeerde `data_hash` uit de PIT-store, en dus is er geen
manier om vast te stellen op welke reeks het oordeel is gebaseerd, of om het te
reproduceren.

`INVALID` betekent hier **precies een ding: het bewijs is niet herleidbaar.**

* Het is **geen** herroeping. F1 t/m F20 blijven staan als falsificaties, en de
  mandaatregel dat hertesten zonder gewijzigde premisse een schending is, blijft
  onverkort gelden. Een gefalsificeerde hypothese wordt niet aantrekkelijker
  doordat het bewijs niet reproduceerbaar is.
* Het is **geen** promotie. De vier `accepted` units zijn hiermee juist strenger
  behandeld: zij verliezen hun status als bewijs.
* Het is **niet** `UNPROVEN - insufficient data`. Dat oordeel is voor een model
  dat de Data Adequacy Gate niet haalt (audit paragraaf 6, fase-no-go 13). Hier
  was de data er wel; de **herkomst** ontbreekt.

**Heropening vereist in alle 24 gevallen dezelfde weg:** een nieuwe
pre-registratie, gecertificeerde Phase 1-data met een `data_hash`, en de vijf
poorten uit `validation/gates.py`. Zo'n hertest telt opnieuw mee in `M`.

### Waarom hier geen ledger-entry tegenover staat

`registry/hypothesis_ledger.py` telt `M`, en `M` is de noemer van elke DSR in
dit platform. Een entry vereist `n_trials >= 1`, dus een invalidatie-entry zou
`M` met minstens 1 verhogen zonder dat er een configuratie is geprobeerd.

Deze herclassificatie is een GOVERNANCE-handeling, geen search. `M` ophogen voor
administratie maakt het getal minder eerlijk, niet strenger, en het zou een
precedent zetten waarin papierwerk een statistische noemer opblaast. Phase 2
exit-criterium 7 eist de status *"in het register"*, en dit register is
eveneens append-only - het is de juiste plaats.

`M` blijft daarom **2776**.
---

## H1 — GARCH-familie tegen EWMA(0.94): **geen falsificatie**

*Toegevoegd 2026-08-29, Phase 6 stap 7 / Phase 7/8 Stage C-3. Bewijs:
`reports/GARCH_VS_EWMA_COMPETITION.md`, artefact
`artefacts/governance/phase6_h1_competition.json`, ledger-amendement
`ae4823e95f6d5844` (`amends: cef1a3b9a6811d7b`).*

Deze regel staat hier omdat een LEGE plek in dit register even misleidend is als
een verkeerde regel. H1 is volledig gedraaid — 48 van 48 gepre-registreerde
combinaties gefit — en het oordeel luidt **`UNPROVEN`**, niet `FALSIFIED`.

| | |
|---|---|
| Gepromoveerd | 0 |
| Gefalsifieerd | **0** |
| `UNPROVEN` (proxy-premisse geschonden) | 36 |
| `DESCOPED` (convergentie/randoplossing, gedegenereerde forecast) | 12 |

**Waarom geen falsificatie.** De gepre-registreerde range-proxy meet aantoonbaar
een andere grootheid dan de modellen voorspellen: `mean(proxy)/mean(r²)` = 1,27
tot 2,24 tegen een toegestane afwijking van 15 %. QLIKE rangschikt op zo'n
meetlat naar kalibratie tegen een verschoven doel. Op een ongeldige meetlat is
een negatieve uitkomst even betekenisloos als een positieve. Daarbovenop faalt
de negatieve controle op de power-kant bij 8 van de 12
reeks/horizon-combinaties: daar onderscheidt DM-HLN zelfs een forecast waarvan
de timing volledig is vernietigd niet van het origineel.

**Wat dit betekent voor hertesten.** H1 mag opnieuw worden getoetst zodra er
intraday realized variance is (DI-18) — dat is een gewijzigde premisse en geen
herhaling van hetzelfde. Zo'n hertest vereist een NIEUWE pre-registratie en telt
opnieuw mee in `M`. Wat NIET mag: dezelfde competitie herhalen op een andere
range-proxy tot er een significante uitkomst verschijnt.

**EWMA(0.94) blijft de productie-estimator**, en dat is hier een niet-verworpen
nulhypothese en geen bewezen superioriteit.

### Naschrift bij "Waarom hier geen ledger-entry tegenover staat"

Die paragraaf hierboven constateerde terecht dat een governance-handeling niet
in de ledger kon zonder `M` op te blazen, omdat `n_trials >= 1` werd
afgedwongen. Dat gat is met AD-14 gesloten: een entry met `amends` en
`n_trials = 0` herziet een oordeel zonder de telling te raken. De 24
geherclassificeerde entries van Wave 28 blijven zoals zij zijn — die worden niet
alsnog geamendeerd, want dat zou een besluit uit die wave herschrijven — maar
elk NIEUW oordeel over eerder geboekt onderzoek hoort vanaf nu in de ledger,
zoals dat van H1 hierboven.

`M` blijft **2776**.


---

## H2 — de Markov-familie tegen M0 Causal Vol-Buckets: **geen falsificatie**

*Toegevoegd 2026-08-30, Phase 6 stap 11 / Phase 7/8 Stage C-3. Bewijs:
`reports/M0_VS_HMM_BENCHMARK.md`, artefact
`artefacts/governance/phase6_h2_regime_benchmark.json`, ledger-amendement
`beb6a14e4f362658` (`amends: 3d3af28730a6c7f9`).*

> **Gecorrigeerd 2026-09-01 (Phase 7/8, Stage E-3).** Hier stond
> `5e5e07579587cddd` als het `config_hash` van het H2-amendement. Die
> entry bestaat niet in `artefacts/governance/hypothesis_ledger.json`;
> het werkelijke amendement draagt `beb6a14e4f362658`. Het
> `amends`-veld klopte wel. Een register dat naar een niet-bestaande
> ledger-entry verwijst, is precies wat no-go 20 verbiedt — en het is
> alleen te vinden door de verwijzing daadwerkelijk op te zoeken, wat de
> reden is dat Stage E-3 dat automatiseert
> (`tests/unit/test_docs_claim_only_what_exists.py`).*

Alle zes gepre-registreerde conditioneerders zijn **gedescopeerd**, en géén
enkele is gefalsificeerd.

| | |
|---|---|
| Gepromoveerd | 0 |
| Gefalsifieerd | **0** |
| `UNPROVEN` | 0 |
| `DESCOPED` (bezetting onder de adequaatheidspoort) | 6 |

**Waarom geen falsificatie.** Stop-criterium 1 van pre-registratie
`3d3af28730a6c7f9da48d13139522a05` bindt op alle zes: de gemeten FILTERED
bezetting van de zeldzaamste toestand loopt van **5,0 tot 28,0 observaties per
fold** tegen een eis van 100. Op zo'n bezetting schat het model een gemiddelde,
een schaal en k−1 overgangskansen op enkele tientallen punten. No-go 8 van de
fase verbiedt een falsificatie-oordeel over een model dat de Data Adequacy Gate
niet haalt, en dat is hier precies de situatie: er is te weinig data, niet te
weinig model.

De a-priori poort in `conf/model/adequacy.yaml` gebruikt de UNIFORME aanname
(1/k per toestand) en noemt zichzelf daarbij optimistisch — 247,5 observaties
bij k = 2 op 495 trainbars. De gemeten bezetting is een orde lager. Dat verschil
is de kern van de bevinding: **de poort die vóór de fase groen stond, staat na
de meting rood, en de meting wint.**

**Twee nevenbevindingen die niet van de poort afhangen.**

1. **M0 verbetert de baseline zelf niet.** Netto OOS Sharpe −0,2501 tegen
   −0,0955 voor de ongeconditioneerde arm, over dezelfde 1.200 OOS-bars. De
   productie-baseline blijft M0 omdat de pre-registratie hem als baseline
   aanwijst, maar de vergelijking staat op een overlay die op deze data niets
   toevoegt.
2. **Een regime-overlay kan dit boek per constructie niet de-grossen.** De
   soevereine laag schaalt met `w_t = min(max_leverage, σ_target / σ_boek)`;
   een uniforme factor deelt daar weer uit. Gemeten: bruto notional 0,976× en
   turnover 1,140× die van de ongeconditioneerde arm. Wat H2 heeft gemeten is
   een cross-sectionele TILT, geen risicoreductie.

**Wat dit betekent voor hertesten.** H2 mag opnieuw worden getoetst zodra er
meer bars per fold zijn — een langer venster of een groter universum (DI-15) —
of onder een expliciet andere toestandsdefinitie. Beide zijn een gewijzigde
premisse en vereisen een NIEUWE pre-registratie met een eigen bijdrage aan `M`.
Wat NIET mag: `k` verlagen of `min_obs_per_state_per_fold` verruimen tot de
poort opengaat.

**M0 Causal Vol-Buckets blijft de productie-baseline**, en dat is hier een
niet-verworpen nulhypothese en geen bewezen superioriteit.

`M` blijft **2776**.

---

## H3 — CatBoost als meta-labeler: **geen falsificatie**

*Toegevoegd 2026-09-01, Phase 6 stappen 12 en 13 / Phase 7/8 Stage C-3. Bewijs:
`reports/META_LABELING_EVALUATION.md`, artefact
`artefacts/governance/phase6_h3_meta_labeling.json`, ledger-amendement
`b08585c4394488eb` (`amends: 56395fa201376801`).*

Alle zes gepre-registreerde specs zijn **`ARCHIVED`**, en géén enkele is
gefalsificeerd.

| | |
|---|---|
| Gepromoveerd | 0 |
| Gefalsifieerd | **0** |
| `UNPROVEN` | 0 |
| `DESCOPED` | 0 |
| `ARCHIVED` (AUC-drempel niet gehaald) | 6 |

**De negatieve controle is schoon, en dat is de voorwaarde voor al het andere.**
Hetzelfde model op gerandomiseerde labels geeft een AUC van **0,5100** (maximum
over vijf permutaties: 0,4857 · 0,5007 · 0,5100 · 0,4936 · 0,4897), tegen een
falsificatiegrens van 0,55. De pipeline lekt niet: geen scaler over folds heen,
geen feature die de toekomst raakt, en een purge op `t1` die doet wat zij
belooft. Was dit getal 0,55 of hoger geweest, dan was de RUN ongeldig geweest en
niet de hypothese — en dan had er hier niets gestaan.

**De adequaatheidspoort bindt niet.** 442,7 effectieve gepurgede trainevents per
fold tegen een eis van 100. Dit is dus géén `UNPROVEN — insufficient data` zoals
bij H1 en H2: er is genoeg data, er is gefit, en er is gemeten.

**Waarom dan toch geen falsificatie.** De zes OOS-AUC's liggen tussen **0,4768
en 0,4842** — alle zes onder 0,50 — en de conservatieve ondergrenzen tussen
0,4040 en 0,4113, ver onder de drempel van 0,58. Stop-criterium 3 bindt op alle
zes, en zijn gepre-registreerde actie is `archive`, niet `falsify`.

Dat is geen formaliteit maar de juiste lezing, en de reden staat in de
power-analyse die vóór de run vastlag:

| Variant | effectieve n | MDE (AUC-overschot) | informatief? |
|---|---|---|---|
| nominaal | 7.200 | 0,0190 | ja |
| na uniqueness | 1.167,8 | 0,0472 | ja |
| **na uniqueness én cross-sectie** | **246,2** | **0,1029** | **nee** |

Het overschot dat de drempel van 0,58 vraagt is 0,08, en dat is KLEINER dan de
MDE van 0,1029 op de steekproefgrootte waarop het oordeel valt. Een toets die
het effect dat zij zoekt niet kan detecteren, kan het ook niet verwerpen. Dat
lag vóór de run vast; het is geen verklaring achteraf.

**De twee lezingen wijzen niet dezelfde kant op, en dat hoort er te staan.** Op
het NOMINALE interval ligt de bovengrens van de beste spec op 0,4976 — onder
0,50, en dus zou een pipeline zonder uniqueness-correctie hier concluderen dat
het model significant SLECHTER dan willekeurig rangschikt. Dat interval is
ongeveer 5,5 keer zo smal als het conservatieve, en het is precies het interval
dat de pre-registratie vóór de run heeft afgewezen als te smal. De conservatieve
lezing omvat 0,50 ruim; het is ruis, geen anti-signaal.

**De economische toets bevestigt het beeld, maar draagt het oordeel niet.** Vijf
van de zes specs verslechteren de netto Sharpe (−0,5328 tot −0,1389). Eén,
`catboost-d6-lr0.03`, verbetert hem met **+0,0449** — van −0,0998 naar −0,0548 —
op een spec met een AUC van 0,4795, dus zonder gemeten voorspellende waarde. Dat
is exact het geval waar stap 13 voor waarschuwt. Het bindende criterium bij die
spec is dan ook de AUC en niet de engine. Bovendien zakt het aantal bars met
positie van 1.200 naar 114–753; een Sharpe over zo weinig bars draagt een brede
foutmarge.

De precision op het werkpunt (0,4747 tot 0,4847) ligt bij alle zes ONDER de
basisrate van 0,4961. Een filter dat selecteert op een kans die het niet kan
schatten, houdt trades over die het gemiddelde niet halen.

**Wat dit betekent voor hertesten.** H3 mag opnieuw worden getoetst zodra de
effectieve steekproef groter is — een langer venster, een groter universum
(DI-15), of een lagere labeloverlap door een kortere horizon. Alle drie zijn een
gewijzigde premisse en vereisen een NIEUWE pre-registratie met een eigen
bijdrage aan `M`. Wat NIET mag: de barrièrebreedtes variëren tot de AUC de
drempel haalt, een ander werkpunt kiezen op de uitkomst, of de drempel van 0,58
verlagen.

**CatBoost blijft `ARCHIVED` zoals §24 vastlegt**, en directionele CatBoost
blijft technisch geblokkeerd: `build_dataset` neemt geen `target`-argument, dus
er bestaat geen handtekening waarmee dit model een eigen doelvector krijgt.

`M` blijft **2776**.
