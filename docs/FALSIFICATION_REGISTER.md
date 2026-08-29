# FALSIFICATION REGISTER — bindend, append-only

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
