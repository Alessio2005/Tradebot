# FASE 10 — STAP 13: H-10.3, DE TOESTANDSPOORT

> **Verdict: LAPSED — deze stap vervalt, en dat is het antwoord.**
>
> De stapopdracht maakt H-10.3 voorwaardelijk: *"Zij hangt aan stap 9. Haalt de
> driedelige toestand de bezettings- en episodepoort niet, dan **vervalt deze
> stap** en is dat het antwoord."* De poort geeft `adequate = false`. Er is dus
> geen `gate_by_state` geschreven, geen poortsample gelezen en **nul trials
> besteed**.

**Gegenereerd:** 2026-09-14
**git_sha (meting):** `5dad1ab`
**Meetvenster:** ontwikkelsample, 1.390 sigma-bars waarvan **1.140 toegewezen**
(250 bars burn-in van het expanding kwantiel, die de poort correct eraf snijdt)
**Bron:** `artefacts/governance/phase10_h10_3.json`, geproduceerd door
`apps/run_h10_3_state_gate.py`; de poort zelf is
`validation/data_adequacy.py::assert_realised_occupancy` uit stap 9
**Trials:** **0** besteed van de 1 geplande
**Faseopdracht:** stap 13, met stap 9 als voorwaarde; R-2, R-8, R-10

---

## 1. Het poortoordeel

```
adequate : false
verdict  : "UNPROVEN — insufficient data"
krapste  : HIGH, fold 2, ETHUSDT — 0 bars, 0 episodes, bezetting 0,0
eis      : >= 100 bars EN >= 20 episodes per toestand per fold,
           bij een bezettingsfractie >= 0,10, op het MINIMUM over de folds
```

Dat is de voorwaarde uit stap 13. Zij is niet gehaald, dus de stap vervalt.

---

## 2. Waarom hij faalt — op beide armen, en niet marginaal

Bij 1.140 toegewezen bars en 8 folds houdt een fold **142–143 bars per symbool**.

**De barsarm.** De eis is 100 bars voor **elke** toestand, per fold, per symbool.
Drie toestanden × 100 = **300 bars**, waar er **142,5** zijn — een factor **2,11
meer dan er bestaan**.

**De episodearm**, onafhankelijk daarvan en met ruime marge. Gevraagd: 3 × 20 =
**60 episodes** per fold per symbool. HIGH heeft over het hele venster **80
episodes**, verdeeld over 6 symbolen en 8 folds: bij gelijke spreiding **1,67 per
symbool per fold**, tegen een vloer van 20. Een factor twaalf.

| arm | gevraagd per fold per symbool | beschikbaar | verhouding |
|---|---:|---:|---:|
| bars | 300 (3 × 100) | 142,5 | **2,11× te veel gevraagd** |
| episodes | 60 (3 × 20) | ≈ 1,67 (HIGH) | **≈ 36× te veel gevraagd** |

---

## 3. Het oordeel is correct, maar OVERBEPAALD — en dat is een bevinding over de poort

De barsarm is bij deze foldgeometrie door **geen enkele** driedelige toewijzing te
halen. Drie toestanden kunnen samen niet meer bars beslaan dan een fold lang is,
en de eis vraagt er ruim twee keer zoveel. De poort meet daar dus een eigenschap
van de **indeling**, niet van het regime:

| grootheid | waarde |
|---|---:|
| toegewezen bars | 1.140 |
| folds | 8 |
| bars per fold per symbool | 142,5 |
| gevraagd (3 × 100) | 300 |
| **haalbaar?** | **nee** |
| grootste `n_folds` waarbij het kán | **3** |
| drempel die bij 8 folds wél haalbaar is | **≤ 47,5 bars** |

**Er is hier geen drempel verzacht en geen foldaantal gewijzigd** om de poort te
laten slagen. Dat zou precies de aanpassing zijn die R-2 en R-10 verbieden. Het
blok `gate_satisfiability` in het artefact rekent de grens uit, zodat het oordeel
niet alleen zegt *dat* de poort faalt maar ook of hij had kunnen slagen.

De keuze tussen **"het regime is inadequaat"** en **"de poort is te streng
geconfigureerd voor deze foldgeometrie"** is een mandaatbesluit en ligt bij de
eigenaar. Voor stap 13 maakt het geen verschil: de episodearm faalt met een factor
twaalf en die arm is wél haalbaar in principe.

---

## 4. Waarom dit niet eerder is vastgesteld — de poort stond uit

`apps/run_state_diagnostics.py` riep `assert_realised_occupancy` aan, maar **achter
een `--check-adequacy`-vlag met default `False`** die nooit is meegegeven. En het
oordeel ging naar **stdout** en nooit naar het artefact.

De poort van stap 9 was dus geschreven, getest en groen — maar zijn uitspraak over
de **echte** toewijzing was nergens vastgelegd, terwijl stap 13 er volledig op
rust.

**Gerepareerd in dezelfde commit.** De vlag is verwijderd: de poort draait nu
altijd en zijn oordeel staat als `occupancy_gate` in
`artefacts/governance/phase10_state_diagnostics.json`. Een poort die standaard uit
staat is geen poort, en een oordeel dat alleen in stdout bestaat is geen meting
(R-8).

---

## 5. Wat dit sluit

De stapopdracht schreef de eerlijke verwachting op: HOOG draagt in de nulmeting
+136,4 % geannualiseerd rendement, dus dichtzetten zou **rendement** wegnemen
samen met de volatiliteit, en het Sharpe-effect kon makkelijk nul of negatief
zijn. Zij noemde dat expliciet *"een geldig en waardevol antwoord"* dat het
toestandsspoor **sluit** in plaats van het open te laten.

Dat antwoord komt hier, maar langs een andere route. Het spoor sluit niet op een
gemeten Sharpe-verschil, maar op de vaststelling dat **het paneel de driedelige
toestand niet kan dragen op de foldgeometrie waarmee dit programma valideert.**

Stap 6 had het al zichtbaar gemaakt en deze poort maakt het bindend:

| toestand | bezetting | episodes | gem. duur | ann. vol | ann. rendement | t (gedefleerd) |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 39,5 % | 163 | 17,4 | 63,7 % | +15,0 % | +0,29 |
| NORMAL | 48,1 % | 245 | 14,0 | 80,9 % | +41,4 % | +0,72 |
| **HIGH** | **12,4 %** | **80** | **10,6** | **118,4 %** | **−3,8 %** | **−0,02** |

De vol-separatie is er (63,7 % → 118,4 %), de richting niet: de gedefleerde t op
rendement is voor alle drie de toestanden binnen ±0,72, en voor HIGH praktisch
nul. Een toestand die de richting niet voorspelt en die te weinig episodes heeft
om per fold te valideren, levert geen handelbare poort.

---

## 6. Stage C, eindstand (R-10)

| hypothese | verdict | trials |
|---|---|---:|
| H-10.1 — de beslisfrequentie | UNPROVEN | 4 |
| H-10.2 — het onevenwichtige paneel | DESCOPED | 1 |
| H-10.3 — de toestandspoort | **LAPSED** | **0** |
| | **totaal** | **5 van 25** |

`remaining(booked=5) = 20`.

De stapopdracht verwachtte een eindstand van **25**: *"de eindstand van `M_new`
moet gelijk zijn aan het aantal daadwerkelijk doorlopen takken (verwacht: 25)."*
Gemeten: **5**. Dat is geen tekort. Twee hypothesen zijn goedkoop weerlegd en de
derde vervalt, dus er zijn **20 trials niet besteed** aan takken die de meting
niet nodig had. Het budget is een plafond, geen doel — en een programma dat zijn
plafond niet haalt omdat de vragen eerder gesloten waren, heeft geld
overgehouden in plaats van werk laten liggen.

Het poortsample is over de hele Stage C **niet één keer gelezen**:
`holdout_lock.json` draagt nog `reads: []`. De ene lezing die R-7 per hypothese
toestaat, is voor alle drie onbesteed.

---

## Bijlage — herleidbaarheid

**Nul trials.** Deze stap selecteert niets: de drempels komen uit
`conf/model/adequacy.yaml`, de foldgeometrie uit
`validation/adequacy_report.py::measure_fold_geometry`, en de toewijzing uit
`regime/state.py::assign_by_variance`. Er is geen keuze op grond van de uitkomst
gemaakt, en de geplande trial is dus niet besteed.

**Geen pre-registratie.** `conf/experiment/h10_3_state_gate.yaml` is niet
aangemaakt en dat is opzet: een pre-registratie legt een te toetsen hypothese
vast, en er is hier niets getoetst. `preregistration_id` in de ledger-entry wijst
daarom naar het artefact van deze stap.

**`result: archived` en niet `lapsed`.** `hypothesis_ledger.py:37` laat alleen
`{accepted, archived, falsified, interim}` toe. `LAPSED` staat in
`metrics.verdict` en in de `notes` — dezelfde beperking als bij H-10.1's
`UNPROVEN` en H-10.2's `DESCOPED`.

**Boekingsvorm.** `n_trials=0` met `amends`, om dezelfde reden als bij de twee
voorgaande hypothesen: `active_trial_count()` eist dat de lopende ledger-teller
gelijk blijft aan de bevroren `archived_total = 2776`. Hier is dat bovendien
inhoudelijk juist, want er ís geen trial besteed.

Reproductie:

```
python apps/run_h10_3_state_gate.py
python apps/run_state_diagnostics.py     # de poort draait nu standaard mee
python -m pytest tests/unit/test_realised_occupancy_gate.py -q
```

Het JSON-artefact is de bron van elk getal in dit rapport.
