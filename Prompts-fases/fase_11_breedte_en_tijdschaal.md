# MASTER-PROMPT: FASE 11 — BREEDTE EN TIJDSCHAAL: HOEVEEL ONAFHANKELIJKE WEDDENSCHAPPEN DIT DOMEIN KAN DRAGEN

> **Status:** concept, ter vaststelling door de eigenaar.
> **Geschreven op 2026-09-25 tegen `28cc31b`**, op een nulmeting en niet op
> aannames. Elk getal in §3 is op die datum in deze repository gemeten; de
> reproductie staat in bijlage A. Waar een getal een verwachting is en geen
> meting, staat dat erbij (R-10).
> **Zusterfase:** `Prompts-fases/fase_11_meetbasis_en_carry.md` — **wordt
> parallel uitgevoerd door een andere agent.** §5 is het contract tussen beide
> fasen. Lees het vóór de eerste regel code; het is geen bijlage maar een
> voorwaarde.
> **Voorganger:** `Prompts-fases/fase_10_herstart_dagbars.md` (H-10.1
> `UNPROVEN`, H-10.2 `DESCOPED`, H-10.3 vervallen).
> **Taal:** dit document is Nederlands; code, docstrings, commits en
> artefactsleutels blijven Engels (R-9).

---

## 0. WAT DEZE FASE IS

De fundamentele wet van actief beheer zegt dat de Sharpe van een boek ruwweg
het product is van twee dingen: hoe goed elke voorspelling is (de IC) en hoeveel
onafhankelijke voorspellingen er per jaar worden gedaan (de breedte):

```
SR_jaar  ≈  IC · √BR          BR = onafhankelijke weddenschappen per jaar
                              BR = (cross-sectionele breedte) × (temporele breedte)
```

Tien fasen hebben gewerkt aan de teller. Geen enkele heeft de noemer gemeten.
De cross-sectionele breedte is één keer bepaald, als deflatiefactor voor een
t-statistiek (fase 10 §3.4, `N_eff = 1,271`), en nooit als de grootheid die
bepaalt welk signaal de poort überhaupt kán halen. De temporele breedte, het
aantal onafhankelijke besluiten per jaar, is nooit gemeten. H-10.1 toetste de
beslisfrequentie op het enige besluitpaneel in deze repository dat **nul**
temporele breedte heeft (één unieke gewichtsrij op 1.743 bars). Dat was vóór de
registratie uit te rekenen (§3.4).

Deze fase meet de noemer, en doet dat met **nul trials**. Breedte en
beslisfrequentie zijn eigenschappen van het universum en van het besluitpaneel,
niet van een rendement. Ze zijn te meten met tweede momenten en omzet, zonder één
gemiddeld rendement te bekijken.

| | Stage | Wat het is | Trials |
|---|---|---|---:|
| **A** | **Eén breedte, met onzekerheid** | "N_eff" is in deze repository twee grootheden onder één naam, met vier implementaties. Op een dollar-neutraal boek van zes namen rapporteert de officiële versie **123,6** onafhankelijke weddenschappen (§2.1). | **0** |
| **B** | **De twee klokken** | Het besluitpaneel van `xs_momentum_equal_weight` verandert op 178 dagen per jaar; het signaal erachter ververst ongeveer 12 keer per jaar (§2.2). | **0** |
| **C** | **De muur** | Welke IC de poort vraagt bij de gemeten breedte, per constructie en per horizon. AD-29 maakt dat tot een veld dat elke nieuwe pre-registratie invult; de controle is eerst additief (stap 6). | **0** |
| **D** | **H-11.2 — de beslisklok volgt de signaalklok** | Uitsluitend als een vooraf uitgerekende haalbaarheidspoort groen is. De nulmeting voorspelt dat zij rood is (§3.7). | **≤ 1, verwacht 0** |
| **E** | **Breedte buiten het domein** | Een eigenaarsdocument: wat de gemeten muur betekent voor DI-15, DI-21 en F10, en welke besluiten alleen de eigenaar kan nemen. | **0** |

### 0.1 Wat deze fase NIET is

**Zij zoekt geen signaal.** Stages A, B en C rekenen uitsluitend met
correlaties, eigenwaarden, besluitpanelen en omzet. Er wordt geen gemiddeld
rendement, geen Sharpe en geen IC van enig signaal gemeten. Elk eerste moment dat
hier zou worden bekeken, is selectiedruk op elke volgende pre-registratie, ook
op die van de zusterfase (R-15).

**Zij verbreedt het universum niet.** De zes namen blijven zes. Meer namen van
dezelfde beurs is precies de breedte die F10 falsificeert ("namen binnen één
bètafactor"). Elke extra Bybit-naam is bovendien een overlever (DI-15), en de
PIT-store is gecertificeerd en in git (AD-12). De zusterfase leest de store
terwijl deze fase loopt. Wat een verbreding zou kosten en opleveren, staat in
stage E als besluit voor de eigenaar, niet als handeling.

**Zij raakt de carry niet aan.** Funding carry is de enige levende kandidaat,
en die is van de zusterfase (H-11.1). Deze fase berekent geen enkele grootheid
die van de fundingreeks afhangt.

**Zij verandert de meetbasis niet.** `W_DEV`, `W_GATE`, de annualisatie en de
DSR-handtekening staan in `docs/MEASUREMENT_CONTRACT.md`, en de bevroren
pre-registratie van de zusterfase hangt eraan. H-10.2 liet zien dat een ragged
paneel `W_DEV` met 25,8 % kan verlengen. Dat nu doorvoeren zou het venster
onder een lopende hypothese verschuiven.

**Zij besteedt het budget van de enige levende kandidaat niet** aan een vraag
waarvan vooraf uit te rekenen is dat zij onbeslisbaar is. Dat is de les van
H-10.3, toegepast op een hele fase in plaats van op één poort.

### 0.2 Waarom zij naast de zusterfase kan lopen, en waar niet

Stages A, B, C en E zijn **beleidsonafhankelijk**: ze werken op de datalaag en
op L0, en geen enkel getal erin gaat door de risicolaag. Ze kunnen beginnen op
het moment dat deze prompt is vastgesteld, ongeacht hoe ver de zusterfase is.

Stage D is dat niet. H-11.2 meet op L3, en L3 onder het geldende beleid
`9961e1613bc907a5` bestaat pas wanneer de zusterfase haar stap 4 heeft afgerond
(de ladder opnieuw afgeleid). Tot dat artefact op `main` staat, wordt H-11.2
niet geregistreerd. Zie §5.6.

---

## 1. ROL EN CONTEXT

Je neemt een onderzoeks- en handelsplatform over dat rond één principe is
gebouwd: **een resultaat telt pas wanneer het de poort is gepasseerd die het had
kunnen tegenhouden.** Lees vóór de eerste regel code, in deze volgorde:

1. `Prompts-fases/fase_11_meetbasis_en_carry.md` — volledig, en in het bijzonder
   §5 (fences), §6 (deliverables) en stap 6 t/m 8. Wat daar staat is van een
   andere uitvoerder; §5 hieronder zegt wat dat voor jou betekent.
2. `docs/PROJECT_STATE.md`, `docs/MANDATE.md` en AD-22 t/m AD-26.
3. `docs/MEASUREMENT_CONTRACT.md` — het venster, de annualisatie, de SE's en de
   DSR-handtekening. Dit document herhaalt ze niet.
4. `docs/FALSIFICATION_REGISTER.md`, in het bijzonder **F10** (naïeve breedte) en
   de tweede les onder **F20** (*"breedte was hier de bindende beperking, niet
   het signaal"*).
5. `reports/phase10_h10_1_decision_frequency.md` en
   `reports/phase10_h10_2_unbalanced_panel.md` — de twee hypothesen waarvan deze
   fase de onbeantwoorde helft oppakt.
6. `docs/DEFERRED_ISSUES.md`, DI-15 en DI-21 (universumbreedte) en de alinea
   *"Bijgesteld 2026-09-05"*: AD-23 gaat over waarnemingsfrequentie en zegt
   **niets** over breedte.

Het universum is zes Bybit-perpetuals op dagbars. Er is op dit moment **geen
enkel model in deze repository waarvan is aangetoond dat het geld verdient.**
Deze fase verandert dat niet en probeert dat ook niet. Zij stelt vast hoe hoog de
lat ligt voor elk model dat het ooit wil proberen, en of die lat binnen dit
domein te halen is.

---

## 2. DE TWEE BEVINDINGEN DIE DEZE FASE DRAGEN

### 2.1 "N_eff" is twee grootheden onder één naam

De repository kent vier functies die "N_eff" berekenen, met twee verschillende
formules:

| Implementatie | Formule | ruw, `W_DEV` | **dollar-neutraal, `W_DEV`** |
|---|---|---:|---:|
| `validation/inference.py:932` `effective_breadth` | `N / (1 + (N−1)·ρ̄)` | 1,276 | **123,58** |
| `alpha/cm_carry.py:291` `effective_breadth` | idem | 1,276 | **123,58** |
| `alpha/cm_tsmom.py:91` `effective_breadth` | idem | 1,276 | **123,58** |
| `portfolio/covariance.py:179` `effective_n_assets` | `(Σλ)² / Σλ²` | 1,601 | **4,353** |

"Dollar-neutraal" is het residu na aftrek van het gelijkgewogen mandje. Dat is
precies de constructie van arm 1 van de zusterfase
(`cross_sectional_dollar_neutral`).

**De eerste formule is niet fout. Zij meet iets anders dan haar naam zegt.**
`N / (1 + (N−1)·ρ̄)` is Kish' ontwerpeffect van een **gepoold, gelijkgewogen
gemiddelde**: hoeveel onafhankelijke waarnemingen het gemiddelde van N
gecorreleerde reeksen waard is. `clustered_mean` gebruikt haar voor precies dat
doel, en daar is zij juist. Op een paneel dat over de namen tot nul sommeert,
is het gelijkgewogen gemiddelde identiek nul en heeft het geen variantie. De
noemer gaat dan naar nul: bij ρ̄ = −0,1903 is `1 + 5·ρ̄ = 0,0485`, en de uitkomst
wordt 123,6. De controle in `inference.py` weigert pas bij een noemer ≤ 0, en
haar melding (*"Zo'n correlatiestructuur kan niet uit een geldige
correlatiematrix komen"*) is op dit punt onjuist: de residuele
correlatiematrix is geldig, alleen singulier.

**De breedte uit de fundamentele wet is een andere grootheid**: het aantal
onafhankelijke weddenschappen. Dat getal is begrensd door de rang van de
correlatiematrix, en die is bij een dollar-neutraal boek van zes namen ten
hoogste vijf. De participatieratio van de eigenwaarden respecteert die grens per
constructie ((Σλ)² ≤ r·Σλ²). De ρ̄-formule doet dat niet.

**Waarom dit nu telt, en niet pas in stage C.** De zusterfase leunt op twee
plekken op deze grootheid:

- **Haar stap 6.1** zet in de pre-registratie een `adequacy_floor` met *"de
  ρ̄-gedefleerde effectieve telling ernaast (AD-20: het oordeel valt op de
  conservatieve)"*. AD-20 vermenigvuldigt de nominale telling met
  `effectieve reeksen / aantal symbolen` en veronderstelt daarmee stilzwijgend
  dat die verhouding hoogstens 1 is. Defleert zij met de ρ̄ van het
  dollar-neutrale paneel van arm 1, dan is de "conservatieve" telling
  **groter** dan de nominale. Defleert zij met de ρ̄ van de ruwe rendementen
  (+0,74), dan klopt de richting en is er niets aan de hand.
- **Haar stap 8.1** eist voor elke cel *"de op datum geclusterde paneel-t, N_eff
  en de gedefleerde lezing"* uit `validation/inference.py`. Haar gedefleerde t
  is veilig, want `neff_deflation` kapt de factor af op 1,0 en blaast een t dus
  nooit op. Het getal dat haar rapport als "N_eff" afdrukt voor een
  dollar-neutraal boek is echter het ontwerpeffect, en dat kan groter zijn dan
  het aantal namen.

Dit is de enige bevinding in deze fase die de zusterfase raakt terwijl zij
loopt. Daarom gaat stap 1 als eerste, en apart, naar `main` (§5.6).

### 2.2 De beslisklok loopt vijftien keer sneller dan de signaalklok

`xs_momentum_equal_weight` vormt zijn signaal over **60 bars**
(`conf/model/alpha.yaml`, `lookback_bars: 60`) en herziet het **elke bar**
(`rebalance_every_bars: 1`). Gemeten op `W_DEV`, zonder één rendement:

| grootheid | waarde |
|---|---:|
| bars waarop het besluitpaneel verandert | 678 van 1.390 — **178 per jaar** |
| gemiddelde omzet (tweezijdig, `backtest/vectorized.py:149`) | 0,1531 per bar — **55,9 per jaar** |
| autocorrelatie van de gewichten op lag 1 (mediaan over de namen) | 0,946 |
| geïntegreerde autocorrelatietijd τ_int (mediaan, Sokal-venster c = 5) | **30,0 bars** |
| onafhankelijke besluiten per jaar (365 / τ_int) | **12,2** |
| aandeel omzetgedreven kosten op L3 (fees + impact + spread) | **82,6 %** |

Het boek betaalt dus voor 178 herschikkingen per jaar om ongeveer 12
onafhankelijke besluiten uit te drukken, en vier vijfde van zijn kosten is
omzet. Het is het spiegelbeeld van de long-only-tracks, waar vier vijfde van de
kosten funding is (zusterfase §3.3).

**En het verklaart H-10.1 achteraf, zonder een nieuwe meting.** H-10.1 legde de
beslisfrequentie op `long_only_equal_weight`: één unieke gewichtsrij, nul
wijzigingen, τ_int = ∞, temporele breedte nul. Een vasthoudoperatie op een
besluit dat nooit verandert is de identiteit, en dat stond vóór de registratie
in de gewichtsmatrix. De pre-registratie voorspelde het zelfs (*"vrijwel nul"*).
Wat ontbrak, was de conclusie dat een hypothese die vooraf per constructie niets
kan meten, niet geregistreerd hoort te worden.

### 2.3 Samen: de muur

Zet de gemeten breedte in de fundamentele wet en keer haar om. Dan volgt per
constructie en per horizon de IC die een signaal minimaal moet hebben om de poort
te kunnen halen (`W_DEV`, N = 1.390, M_new = 25):

| constructie | breedte | IC nodig, h = 1 | h = 10 | h = 30 |
|---|---:|---:|---:|---:|
| directioneel | 1,601 | 0,077 | 0,244 | 0,423 |
| dollar-neutraal | 4,353 | 0,047 | 0,148 | **0,257** |

De enige IC's die dit programma binnen het domein heeft gerapporteerd, komen uit
het diagnostische momentumraster van fase 10 §5.7: −0,024 tot **+0,050** over 55
cellen, bij een standaardfout van 0,011. Op welke horizon elke cel zijn IC mat,
staat daar niet. Als het maximum een dagelijkse cross-sectionele IC is, haalt het
de dagelijkse dollar-neutrale muur net (0,047), en alleen bij een beslisklok van
één dag. Bij die klok betaalt het boek vandaag 178 herschikkingen per jaar. Bij
elke langere klok staat het er ver onder.

> **Dit is een verwachting en nog geen resultaat (R-10).** De tabel gebruikt de
> fundamentele wet met een transfercoëfficiënt van 1 en zonder kosten. Een
> verkennende simulatie op de gemeten residuele covariantie (bijlage B) geeft
> een Sharpe binnen ongeveer 15 % van `IC·√(365·n)` voor n tussen de
> participatieratio (4,35) en N − 1 (5). Stap 5 preciseert dat. Wijkt de wet
> materieel af, dan is de simulatie de muur en niet de formule.

---

## 3. NULMETING (gemeten 2026-09-25 tegen `28cc31b`)

Reproduceer deze tabellen vóór je iets wijzigt. Wijkt jouw meting af, dan is dát
je eerste bevinding, en die schrijf je op vóór je verdergaat (R-10). Bijlage A
bevat het script; het leest de gecertificeerde store en schrijft niets.

### 3.0 Omgeving en vingerafdruk

Python **3.11.15** met `requirements.lock` + `requirements-dev.lock` (ruff
0.15.12, mypy 2.3.1, de gepinde versies). Dat is dezelfde interpreter als in de
nulmeting van de zusterfase; de CI draait op 3.13 (DI-24).

| collected | passed | skipped | xfailed | xpassed | **failed** |
|---:|---:|---:|---:|---:|---:|
| 3.120 | 3.087 | 25 | 4 | 0 | **4** |

Dat is **exact** de vingerafdruk van de zusterfase §3.1, met dezelfde vier
failures: `test_a_uniform_factor_is_neutralised_by_the_vol_target`,
`test_a_book_that_straddles_the_dust_tolerance_is_decided`,
`test_no_document_claims_a_path_that_does_not_exist` en de intermitterende
`test_the_multiplier_is_causal[hmm3-diag-student_t]`. Alle vier zijn van de
zusterfase (haar stap 1 t/m 3). Deze fase repareert er geen; zij mag er ook geen
bij maken (R-14). Draait de zusterfase al, dan is haar vingerafdruk intussen
veranderd en is de jouwe de nieuwe basis. Leg vast op welke `main`-commit je hem
nam.

### 3.1 Cross-sectionele breedte

Dagelijkse log-rendementen van de zes gecertificeerde closes, `W_DEV`
(2021-11-15 → 2025-09-04, 1.390 bars). Intervallen: circulaire blokbootstrap
over datums, bloklengte 5, 2.000 trekkingen, seed 20260925.

| constructie | N | ρ̄ | N_eff, ρ̄-formule | N_eff, participatieratio | 95 %-interval PR |
|---|---:|---:|---:|---:|---|
| directioneel (ruw) | 6 | +0,7403 | 1,276 | 1,601 | [1,518; 1,687] |
| dollar-neutraal (residu na EW-mandje) | 6 | −0,1903 | **123,58** | 4,353 | [4,143; 4,513] |
| bèta-gehedged tegen EW-mandje | 6 | −0,1872 | **93,58** | 4,615 | — |
| bèta-gehedged tegen BTC | 5 | +0,4312 | 1,835 | 2,825 | — |

Het 95 %-interval van de ρ̄-formule op de ruwe rendementen is [1,240; 1,313].

- De eigenwaarden van de ruwe correlatiematrix zijn 4,704 · 0,374 · 0,328 ·
  0,237 · 0,201 · 0,157. De eerste factor draagt **78,5 %** van de variantie.
- De residuele matrix na het EW-mandje heeft eigenwaarden 1,933 · 1,464 · 1,062
  · 0,966 · 0,576 · **0,000**: rang vijf, zoals de constructie vereist.
- De bèta's in de twee gehedgede rijen zijn geschat op het **volle** venster.
  Dat is een diagnostiek van tweede momenten en geen besluitgrootheid. Een
  handelbare hedge schat causaal; die van de zusterfase (arm 2) doet dat op een
  expanding venster. De BTC-bèta's zijn: ETH 1,112 · SOL 1,460 · AVAX 1,334 ·
  LINK 1,193 · DOT 1,129.
- **De bèta-hedge tegen BTC laat een altcoinfactor staan**: ρ̄ blijft +0,43 en de
  breedte komt niet boven 2,8. Neutraal tegen het EW-mandje is een andere, bredere
  constructie dan neutraal tegen BTC.

> **De 1,271 van fase 10 wordt gereproduceerd, maar niet op dit paneel.**
> `artefacts/governance/phase10_state_diagnostics.json` draagt ρ̄ = 0,7436 en
> N_eff = 1,2718 op het toestandsdiagnosepaneel (1.389 bars). Op de ruwe
> rendementen van `W_DEV` geeft dezelfde formule 1,276 (ρ̄ = 0,7403), op `W_FULL`
> 1,265 (ρ̄ = 0,7485). Het verschil zit in de constructie, niet in een fout. Het
> is wel de reden dat elk N_eff-getal in deze fase zijn venster en zijn paneel
> draagt.

### 3.2 Per jaar, en het poortvenster

| jaar | bars | ρ̄ | N_eff, ρ̄-formule | N_eff, PR |
|---|---:|---:|---:|---:|
| 2021 | 47 | 0,7194 | 1,305 | 1,635 |
| 2022 | 365 | 0,7981 | 1,202 | 1,430 |
| 2023 | 365 | 0,6534 | 1,406 | 1,900 |
| 2024 | 366 | 0,6982 | 1,336 | 1,738 |
| 2025 | 365 | 0,8075 | 1,191 | 1,406 |
| 2026 | 235 | 0,8263 | 1,169 | 1,351 |

2022 t/m 2026 reproduceren de reeks uit fase 10 §5.1 (1,20 · 1,41 · 1,34 · 1,19
· 1,17) exact op twee decimalen. Het jaar 2021 wijkt af: 1,305 hier tegen 1,40
daar. `W_FULL` begint pas op 2021-11-15, dus 2021 telt hier 47 bars. Op welk
venster fase 10 dat jaar heeft gemeten, is niet vastgesteld.

**Het poortvenster is het smalste venster van het programma:**

| venster | bars | ρ̄ | N_eff, ρ̄-formule | N_eff, PR | dollar-neutraal PR |
|---|---:|---:|---:|---:|---:|
| `W_DEV` | 1.390 | 0,7403 | 1,276 | 1,601 | 4,353 |
| `W_GATE` | 353 | 0,8200 | 1,176 | 1,369 | 3,801 |

Een hypothese waarvan het bewijs van breedte afhangt, meet op de poort een
**smaller** universum dan op de ontwikkelsample. Dat is geen reden om de poort te
verschuiven (fence), maar het hoort in elke haalbaarheidsrekening.

> **Openheid over deze meting.** Bovenstaande rij `W_GATE` bevat tweede momenten
> van het poortsample. Er is geen eerste moment gelezen en geen hypothese
> getoetst, en `holdout_lock.json` is niet aangeraakt. Er is ook precedent:
> `phase10_state_diagnostics.json` en de haltteller van H-10.1
> (`halt_counter_window: full_usable_window_including_gate_bars`) beslaan
> allebei het volle venster. De regel die deze fase volgt, staat in R-15:
> **tweede momenten op `W_FULL` mogen, eerste momenten op `W_GATE` nooit**
> buiten een geregistreerde lezing.

### 3.3 Vier implementaties van één naam

Zie de tabel in §2.1. Er zijn twee van de vier die deze fase **niet** mag
aanraken, en dat bepaalt de vorm van de reparatie:

- `alpha/cm_carry.py` en `alpha/cm_tsmom.py` dragen de vier xfail-killgates
  (F20). De zusterfase zet ze achter een fence en fase 10 deed dat ook. Hun kopie
  van de ρ̄-formule blijft staan en wordt als DI geboekt, niet gerepareerd.
- `portfolio/covariance.py` is door DI-10 **bewust ongewijzigd** gelaten, omdat
  `portfolio/legacy_sizing.py` ervan afhangt en de Phase 3-baseline herrekenbaar
  moet blijven. Juist daarom is hij de enige kandidaat voor de ene implementatie
  van "onafhankelijke weddenschappen". Hij wordt niet verplaatst en niet
  gewijzigd; de nieuwe laag **delegeert** naar hem (stap 1.4). Er is precedent
  voor een import uit `portfolio/` in `validation/`:
  `validation/phase10_decision_frequency_measurement.py`.

### 3.4 De besluitpanelen

Gewichtspanelen van de vier tracks, gebouwd met
`backtest/baseline_runner.py::build_weight_tracks` precies zoals H-10.1 dat doet,
beperkt tot `W_DEV`. Geen rendement.

| track | unieke rijen | bars met wijziging | omzet per bar | omzet per jaar | ac1 (mediaan) | τ_int (mediaan) | onafh. besluiten per jaar |
|---|---:|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | **1** | **0** | 0,0000 | 0 | — | ∞ | **0** |
| `long_only_risk_parity` | 1.390 | 1.389 | 0,0170 | 6,2 | 0,976 | 84,6 | 4,3 |
| `xs_momentum_equal_weight` | 284 | 678 | 0,1531 | 55,9 | 0,946 | 30,0 | 12,2 |
| `xs_momentum_risk_parity` | 1.390 | 1.389 | 0,1764 | 64,4 | 0,947 | 31,0 | 11,8 |

- Omzet is de definitie uit `backtest/vectorized.py:149`, tweezijdig, **zonder**
  de instapbar. H-10.1 rapporteert voor `long_only_equal_weight` 0,000719 omdat
  zijn gemiddelde de instap van 0 naar 1/6 per naam meetelt: 1,0 / 1.390 =
  0,000719. Het is dezelfde omzet.
- De twee momentumtracks wisselen per naam 21,9 keer per jaar van teken.
- Bij `long_only_risk_parity` en `xs_momentum_risk_parity` verandert het paneel
  op elke bar, maar dat komt door de volatiliteitsherschaling en niet door het
  signaal. `xs_momentum_equal_weight` is de enige track waarvan de wijzigingen
  **uitsluitend** van het signaal komen (284 unieke rijen). Dat is de grond voor
  de keuze van de primaire cel in stap 7, niet een uitkomst.

### 3.5 Waar de kosten vandaan komen

Uit `artefacts/baseline/phase5_revaluation.json`, L3. **Let op: dat artefact
draagt beleid `1b60cb664fbf9a2a` en de zusterfase leidt het opnieuw af** (haar
stap 4). De absolute bedragen gaan veranderen. De verwachting is dat de
verhoudingen blijven staan, maar dat is een verwachting (R-10), en stap 7.1 meet
haar opnieuw op het nieuwe artefact.

| track | fees | funding | impact | spread | **omzetgedreven** | funding |
|---|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | 38,36 | 261,67 | 20,21 | 6,97 | 20,0 % | **80,0 %** |
| `long_only_risk_parity` | 43,06 | 279,72 | 18,34 | 7,83 | 19,8 % | **80,2 %** |
| `xs_momentum_equal_weight` | 1.987,05 | 761,41 | 1.267,59 | 361,28 | **82,6 %** | 17,4 % |
| `xs_momentum_risk_parity` | 2.100,91 | 397,39 | 1.204,34 | 381,98 | **90,3 %** | 9,7 % |

De beslisklok kan alleen de omzetgedreven kolom raken. Op de long-only-tracks is
dat een vijfde van de kosten; daar valt voor een tijdschaalhypothese per
constructie weinig te winnen, en dat bevestigt de mechanistische lezing van
H-10.1 §3 (funding wordt op de positie betaald).

### 3.6 De muur

`IC_nodig = SR_nodig / √(breedte · 365 / h)`. De t = 2-drempel komt uit
`docs/MEASUREMENT_CONTRACT.md` §2. Voor de DSR-drempel bestaat **geen functie**:
1,8686 staat alleen als getal in de ledger-notitie van H-10.1 en in haar
rapport. Hier is hij gereproduceerd door `backtest/metrics.py::deflated_sharpe`
numeriek om te keren (DSR = 0,95 bij M = 25, N = 1.390), onder normaliteit:
scheefheid 0, kurtosis 3 en `sr_variance = 1/n_obs`, de gedocumenteerde
benadering. Uitkomst: 1,868609. Met de scheefheid en kurtosis van een
werkelijke rendementsreeks ligt de drempel anders, en elke hypothese rekent
hem daarom voor haar eigen reeks opnieuw uit.

**Bij de DSR-drempel (M_new = 25, N = 1.390: SR = 1,8686)**

| breedte | h = 1 | h = 5 | h = 10 | h = 30 | h = 60 |
|---|---:|---:|---:|---:|---:|
| directioneel, ρ̄-formule 1,276 | 0,0866 | 0,1936 | 0,2738 | 0,4742 | 0,6707 |
| directioneel, PR 1,601 | 0,0773 | 0,1728 | 0,2444 | 0,4234 | 0,5988 |
| dollar-neutraal, PR 4,353 | 0,0469 | 0,1048 | 0,1482 | 0,2568 | 0,3631 |

**Bij t = 2 op `W_DEV` (SR = 1,0249)**

| breedte | h = 1 | h = 5 | h = 10 | h = 30 | h = 60 |
|---|---:|---:|---:|---:|---:|
| directioneel, ρ̄-formule 1,276 | 0,0475 | 0,1062 | 0,1502 | 0,2601 | 0,3679 |
| directioneel, PR 1,601 | 0,0424 | 0,0948 | 0,1341 | 0,2322 | 0,3284 |
| dollar-neutraal, PR 4,353 | 0,0257 | 0,0575 | 0,0813 | 0,1408 | 0,1992 |

Onafhankelijke weddenschappen per jaar, dollar-neutraal: **1.589** (h = 1) ·
318 (h = 5) · 159 (h = 10) · 53 (h = 30) · 26,5 (h = 60).

Op `W_GATE` alleen (353 bars, 0,9671 jaar) vraagt t = 2 een geannualiseerde
Sharpe van **2,034**, bij een breedte die daar het laagst is (§3.2).

### 3.7 De haalbaarheid van een tijdschaalhypothese, vooraf

Dit is de rekening die H-10.1 en H-10.3 vóór hun registratie hadden moeten
maken, nu gemaakt vóór die van H-11.2. De rekening gebruikt uitsluitend omzet,
de covariantie van `W_DEV` en de gewichtspanelen, met
`portfolio/decision_frequency.py::hold_decision` voor het vasthouden. Er is geen
gemiddeld rendement in gebruikt.

| track | k | omzet per bar, k = 1 → k | daling | σ_boek | ρ(k = 1, k) | ΔSR uit kosten, L0-kostenas | idem, × 1,54 voor de L3-kostenmix | SE(ΔSR) ≈ | detecteerbaar, 95 % eenzijdig |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `xs_momentum_equal_weight` | 10 | 0,1531 → 0,0485 | −68,3 % | 0,233 | 0,876 | 0,106 | 0,164 | 0,255 | 0,419 |
| `xs_momentum_equal_weight` | 30 | 0,1531 → 0,0308 | −79,9 % | 0,233 | 0,635 | 0,124 | 0,192 | 0,438 | **0,720** |
| `xs_momentum_risk_parity` | 10 | 0,1764 → 0,0533 | −69,8 % | 0,220 | 0,873 | 0,133 | 0,205 | 0,258 | 0,425 |
| `xs_momentum_risk_parity` | 30 | 0,1764 → 0,0316 | −82,1 % | 0,220 | 0,645 | 0,156 | 0,241 | 0,432 | 0,710 |

- "ΔSR uit kosten" is de Sharpe-winst **bij nul signaalverval**: de omzetdaling
  maal 6,5 bp per zijde (`taker_fee_bps` 5,5 + `assumed_half_spread_bps` 1,0),
  geannualiseerd en gedeeld door σ_boek.
- De factor 1,54 is de verhouding tussen de omzetkosten op L3 (fees + impact +
  spread = 3.615,9) en die op de L0-kostenas (fees + spread = 2.348,3) op
  `xs_momentum_equal_weight`, uit §3.5.
- ρ(k = 1, k) en σ_boek zijn berekend als `w'Σw` over de gewichtspanelen, met Σ
  de covariantie van `W_DEV`.
- `SE(ΔSR) ≈ √((2 − 2ρ) / t_jaar)` is de benadering voor twee Sharpes rond nul.
  Stap 7.2 vervangt haar door de echte kern (R-3).

**Het getal waar het om draait:** bij k = 30 is de kostenwinst zonder enig
signaalverval hooguit **0,19**. Het kleinste verschil dat op `W_DEV` eenzijdig
op 95 % zichtbaar wordt, is **0,72**, nog vóór deflatie met M_new = 25. De
kostencomponent is bij k = 30 dus 4 tot 6 keer kleiner dan de detectiegrens, en
bij k = 10 2,5 tot 4 keer. Hoe groot het signaalverval is, is niet gemeten: dat
is het deel dat een trial zou kosten. Onder de hypothese kan het het effect
alleen verkleinen. Dat vasthouden de voorspelling nooit kan verbeteren, is
echter geen stelling; zie stap 7.3.

> **Verwachting (R-10):** de haalbaarheidspoort van stap 7 is **rood**, H-11.2
> wordt niet geregistreerd, en deze fase besteedt **nul** trials. Vraag V3 wordt
> dan beantwoord als kostenuitspraak ("de beslisklok is een kostenhefboom van
> ongeveer 0,12 tot 0,19 Sharpe op de omzetgedreven tracks, meetbaar als
> kostengrootheid maar op deze sample niet beslisbaar als Sharpe-verschil") en
> niet als hypothese. Wijkt stap 7 hiervan af, bijvoorbeeld omdat de L3-impact
> onder het nieuwe beleid een andere kostenmix geeft, dan is de meting het
> antwoord.

### 3.8 Budget en poortsample

| grootheid | waarde | bron |
|---|---:|---|
| `archived_total` | 2.776 | `ledger_reset.json` |
| `m_new` (bevroren budget) | **25** | idem |
| besteed in fase 10 | 5 | ledger-notities, H-10.1 en H-10.2 |
| **resterend vóór fase 11** | **20** | — |
| geclaimd door de zusterfase | 6 gepland, **≤ 8** toegestaan | haar §0 en stap 6.1 |
| **geclaimd door deze fase** | **≤ 1, verwacht 0** | §3.7 |
| niet toegewezen reserve | ≥ 11 | — |
| poortsample-lezingen | 0 (`reads: []`), één per hypothese | `holdout_lock.json`, `validation/holdout.py` |

### 3.9 Twee implementaties van "vasthouden"

`alpha/momentum.py::CrossSectionalMomentum._generate` houdt bij
`rebalance_every_bars > 1` de **signaalexposures** vast, met een kalender vanaf
het begin van het paneel en een `ffill`. `portfolio/decision_frequency.py::
hold_decision` houdt de **gewichten** vast, met positionele blokken vanaf de
eerste bar. Het zijn twee implementaties van dezelfde operatie (R-3).

Ze zijn niet overal equivalent. Bij gelijkgewogen sizing commuteren ze, omdat
normaliseren per rij identieke rijen identiek laat. Bij risicopariteit niet: de
sizing leest een volatiliteit die elke bar verandert, dus vasthouden vóór de
sizing geeft een ander boek dan vasthouden erna. `rebalance_every_bars` staat
op 1 en is nooit getoetst. `hold_decision` heeft een causaliteitstest met
negatieve controle en is door H-10.1 gebruikt. Stap 4 beslist.

---

## 4. DE DRIE VRAGEN VAN DEZE FASE

**V1 — Hoeveel onafhankelijke weddenschappen per jaar draagt dit domein?**
Met **één** definitie, per constructie (directioneel, dollar-neutraal,
bèta-gehedged) en per horizon, met een interval. Niet "hoe breed is het
universum", want dat is zes. De vraag is hoeveel van die zes er als afzonderlijke
voorspelling tellen, en hoe vaak per jaar.

**V2 — Welke IC vraagt de poort bij die breedte, en is die IC binnen dit domein
ooit gezien?**
Beantwoord met de muur uit §3.6, geverifieerd door simulatie en vastgelegd als
een veld dat elke nieuwe pre-registratie invult (AD-29). Het antwoord kan
zijn dat geen enkel signaal van het type dat dit domein kan maken de poort kan
halen. Dat is een volwaardig resultaat: het vertelt de eigenaar dat de volgende
beslissing een mandaatbeslissing is en geen onderzoeksvraag (stage E).

**V3 — Is de dagelijkse beslisklok een kostenlek, en is dat in dit domein
beslisbaar?**
Eerst de haalbaarheid (stap 7), daarna pas eventueel de hypothese. Is de
haalbaarheidspoort rood, dan wordt V3 beantwoord met de kostengrootheid en een
breakeven, en kost zij niets.

---

## 5. NAAST DE ZUSTERFASE — HET MERGECONTRACT

Twee uitvoerders werken tegelijk in één boom, op één budget, één ledger en één
poortsample. Deze repository heeft precies zo'n situatie al eens verkeerd laten
aflopen: `f50f6ca` voegde twee parallelle sporen **automatisch en zonder conflict**
samen, en dat heeft de meetbasis ongeldig gemaakt zonder dat iemand het zag. Dat
is de hele reden dat de zusterfase bestaat. Dit hoofdstuk voorkomt de tweede keer.

### 5.1 Wat er op het moment van schrijven bekend is

Op 2026-09-25 is `main` gelijk aan `28cc31b`: de commit die de prompt van de
zusterfase vastlegt. Op `origin` staat nog geen tak van de zusterfase. Wat zij
al heeft gedaan, is dus niet te zien. **Neem niets aan over haar voortgang.**
Draai vóór elke stage `git fetch origin` en `git branch -r`, en kijk wat er op
`main` en op haar tak staat.

### 5.2 Wie wat bezit

**Van de zusterfase — niet wijzigen, ook niet "even meenemen". Lezen en
importeren mag:**

| pad | waarom |
|---|---|
| `conf/risk/default.yaml`, `src/tradebot/risk/**`, `src/tradebot/registry/` (policy-hash; `preregistration.py` en `trial_budget.py` gebruik je, je wijzigt ze niet) | haar stages A en C |
| `backtest/phase5_baseline.py`, `apps/run_phase5_baseline.py`, `artefacts/baseline/**` | haar stap 4 leidt de ladder opnieuw af |
| `data/funding.py`, `features/positioning.py`, `features/funding_carry.py`, `alpha/funding_carry_book.py` | haar stap 5 en stage B |
| `src/tradebot/live/**`, `src/tradebot/oms/**`, `apps/run_daily_decision.py`, `archive/**` | haar stage C |
| `pyproject.toml` (coverage-paden), `.github/workflows/**` | haar stap 9.3 en 12 |
| `tests/regression/**`, `tests/unit/test_regime_overlay.py` | haar stap 2 en 3 |
| `docs/DATA_REGISTER.md`, `docs/RISK_MANDATE.md`, `docs/CODE_REGISTER.md`, `docs/MEASUREMENT_DOMAIN.md`, `scripts/reachability_map.py` | haar stap 1, 5 en 9 |
| `conf/research/preregistration_h11_funding_carry.yaml`, `artefacts/governance/phase11_h11_carry.json`, `reports/phase11_*` behalve de bestanden hieronder | haar artefacten |

**Van deze fase — alleen nieuwe bestanden:**

| pad | inhoud |
|---|---|
| `src/tradebot/validation/breadth.py` | `independent_bets`, `dsr_hurdle`, `required_ic`, `assert_ic_wall_declared`, de muursimulatie |
| `src/tradebot/validation/signal_clock.py` | τ_int en de diagnostiek van het besluitpaneel |
| `apps/run_breadth_measurement.py` (≤ 80 LOC) | stage A t/m C |
| `conf/research/breadth.yaml` | elke parameter van deze fase, gezet vóór het meten |
| `tests/unit/test_breadth_definitions.py`, `tests/unit/test_breadth.py`, `tests/unit/test_signal_clock.py`, `tests/unit/test_ic_wall.py`, `tests/unit/test_hold_equivalence.py` | |
| `artefacts/governance/phase11_breadth.json` | het meetartefact |
| `reports/phase11_breadth_and_timescale.md`, `reports/phase11_ic_wall.md`, `reports/phase11_breadth_owner_decision.md`, `reports/phase11_breadth_exit_report.md` | de rapporten |
| alleen als stap 7 groen is: `conf/research/preregistration_h11_2_decision_clock.yaml`, `apps/run_h11_2_decision_clock.py`, `artefacts/governance/phase11_h11_2.json`, `reports/phase11_h11_2_decision_clock.md` | H-11.2 |

**Gedeeld — aanraken volgens het protocol in de rechterkolom:**

| pad | protocol |
|---|---|
| `docs/ARCHITECTURAL_DECISIONS.md` | **CRLF-regeleindes**, eindigt vandaag op AD-26 en een CRLF. De zusterfase voegt AD-27 en AD-28 toe; deze fase AD-29 en AD-30. Voeg ze toe in **één eigen commit** aan het eind van de stage. Bij een conflict houd je beide blokken, oplopend genummerd, en controleer je daarna dat het bestand nog volledig CRLF is en op precies één CRLF eindigt. `80bc3dc` bestaat omdat een merge dat precies hier één keer fout deed. |
| `docs/DEFERRED_ISSUES.md` | append-only. De zusterfase sluit DI-27, DI-28 en DI-30 en voegt er vermoedelijk bij. Deze fase gebruikt nummers vanaf **DI-35** (§5.3). Wijzig geen regel die je niet zelf hebt geschreven. |
| `artefacts/governance/hypothesis_ledger.json` | append-only JSON. Een merge van twee appends geeft een tekstconflict in de lijst: behoud **beide** entries, herschik geen bestaande, en draai daarna `tests/unit/test_hypothesis_ledger.py`, `test_ledger_amendment.py`, `test_ledger_provenance.py` en `test_ledger_reset.py`. |
| `artefacts/governance/holdout_lock.json` | alleen via `gate_slice`, alleen voor H-11.2, en alleen als stap 8 dat voorschrijft. Bij een conflict behoud je elke `reads`-entry van beide kanten. Een verwijderde entry herstelt een verbruikte lezing (zie de moduledocstring van `validation/holdout.py`). |
| `docs/FALSIFICATION_REGISTER.md` | append-only, alleen onderaan, alleen als stap 8 een falsificatie oplevert. |
| `validation/inference.py` | **gedrag ongewijzigd zolang stage B van de zusterfase loopt**: haar artefacten moeten binnen haar fase reproduceerbaar blijven. Alleen docstrings mogen, en die gaan mee in stap 1. De hernoeming is DI-35. |
| `scripts/check_hardcoded_params.py` (ratchet 313, stand 298), `scripts/check_file_size.py` | één gedeelde marge. Nieuwe modules van deze fase staan op budget **0**: elke parameter komt uit `conf/research/breadth.yaml`. Verhoog geen cap en geen ratchet. |
| `CHANGELOG.md`, `docs/PROJECT_STATE.md` | niet bewerken. Het exitrapport van deze fase zegt wat erin hoort; de eigenaar voegt samen. |

### 5.3 Nummering

| soort | zusterfase | deze fase |
|---|---|---|
| hypothese | H-11.1 | **H-11.2** |
| AD | AD-27, AD-28 (in haar prompt vastgelegd) | **AD-29, AD-30** |
| DI | vanaf DI-31, niet vastgelegd | **vanaf DI-35**. Neem op het moment van schrijven het eerstvolgende nummer ≥ 35 dat **op `main` én op haar tak** nog vrij is, en noem dat nummer in de commit. |
| rapportprefix | `phase11_` | `phase11_breadth_`, `phase11_ic_wall`, `phase11_h11_2_` |

Staat AD-29 of AD-30 bij het schrijven al op `main` of op haar tak, neem dan
het eerstvolgende vrije nummer en werk elke verwijzing in dit document bij.
Verwijs niet naar een nummer dat een ander nog kan pakken.

### 5.4 Het trialbudget is één pot

`registry/trial_budget.py::assert_within_budget(planned, ...)` vergelijkt een
**cumulatief** gepland aantal met het bevroren `M_new = 25`. Fase 10 riep hem aan
met 4 en daarna met 5; de zusterfase rekent met 11 (5 + 6). Omdat de twee fasen
niet synchroon boeken, rekent deze fase met het **plafond** van de zusterfase en
niet met haar plan:

```
assert_within_budget(5 + 8 + planned_h11_2, reset_path=...)   # = 14 bij één trial
```

Staat haar ledger-amendement al op `main`, gebruik dan het getal dat zij
werkelijk heeft geboekt, als dat hoger is dan 8. Boek nooit tegen een
budgetstand die je niet op `main` hebt gelezen.

**Boekingsvorm.** H-10.1 en H-10.2 moesten boeken als `n_trials=0` met `amends`,
omdat `registry/ledger_reset.py::active_trial_count()` weigert zodra de lopende
teller boven `archived_total = 2776` komt (H-10.1-rapport, bijlage). Die
beperking staat nog. Deze fase lost haar niet op; zij is een open mandaatbesluit.

### 5.5 Het poortsample

`gate_slice` staat één lezing per **hypothese** toe. H-11.1 en H-11.2 hebben
dus elk hun eigen lezing en verbruiken elkaars lezing niet. Het zijn wel
dezelfde 353 bars: elke extra hypothese die ze leest, maakt ze iets minder
extern. H-11.2 leest het poortsample daarom alleen als de ontwikkelsample het
voorschrijft (stap 8.4), en de verwachting is dat het daar nooit komt (§3.7).

### 5.6 Volgorde en afhankelijkheden

```
stap 1  (de N_eff-bevinding)  ──► eigen PR naar main, zo snel mogelijk
stap 2–6 (stages A–C, 0 trials) ── parallel aan de zusterfase, beleidsonafhankelijk
stap 7  (haalbaarheid)         ──► wacht op: artefacts/baseline/phase11_revaluation.json
                                   op main, met risk_policy_hash 9961e1613bc907a5
stap 8  (H-11.2)               ──► alleen als stap 7 groen is
stap 9  (eigenaarsdocument)    ── kan na stap 6, onafhankelijk van D
```

**Waarom stap 1 voorrang heeft.** Het is de enige bevinding die een getal in een
lopend rapport van de zusterfase raakt (§2.1). Hoe eerder zij op `main` staat,
hoe groter de kans dat de zusterfase haar ziet vóór haar stap 6.1 (de
bevriezing) en 8.1 (het rapport). Stap 1 is
klein: een docstring, één nieuwe module die delegeert, tests en een DI. Houd hem
zo klein dat hij geen reviewrisico draagt.

### 5.7 Werkwijze in git

- Werk op een eigen tak. **Merge** `main` in je tak (geen rebase), en draai na
  elke merge de volledige suite. Force-push nooit.
- Vóór je een gedeeld bestand uit §5.2 aanraakt:
  `git fetch origin && git diff main...origin/ZUSTERTAK -- PAD`. Vervang
  `ZUSTERTAK` en `PAD`. Zie je daar een wijziging die nog niet op `main` staat,
  dan weet je waar het conflict gaat komen.
- Eén stap is één commit (R-12). Commits in gedeelde bestanden gaan **apart**
  van commits in eigen bestanden, zodat een conflict nooit een eigen wijziging
  meesleept.
- Een merge die zonder conflict slaagt, is **geen** bewijs dat er niets is
  veranderd. Na elke merge: draai de vingerafdruk (R-14) en
  `python scripts/check_domain_consistency.py`, en vergelijk
  `risk_config_registry.json` met wat je verwacht.

### 5.8 Wat je de zusterfase verschuldigd bent, en wat niet

- **Wel:** elke bevinding die haar raakt komt op `main`, met een DI en met een
  zin in de PR-beschrijving die haar stapnummer noemt. De enige route tussen
  twee uitvoerders loopt via `main` en via de eigenaar.
- **Niet:** haar bestanden repareren, haar pre-registratie aanpassen, of gedrag
  wijzigen van een module waarmee zij meet terwijl haar stage B loopt. Ook niet
  wanneer je zeker weet dat je gelijk hebt. Een correcte wijziging onder een
  lopende meting blijft een wijziging onder een lopende meting.

---

## 6. FENCES — WAT DEZE FASE NIET AANRAAKT

| Raak niet aan | Waarom |
|---|---|
| `docs/FALSIFICATION_REGISTER.md` F1–F20 | Onverkort bindend (AD-24 R2). F10 is voor deze fase de dragende regel: zij verbiedt de uitkomst "meer namen is meer breedte" als claim, niet de meting ervan. |
| Het universum en de PIT-store | Read-only. Geen nieuwe symbolen, geen nieuwe datasets (§0.1). |
| `docs/MEASUREMENT_CONTRACT.md`, `W_DEV`, `W_GATE` | De pre-registratie van H-11.1 hangt eraan. |
| `conf/risk/default.yaml` en de risicolaag | Van de zusterfase (§5.2). |
| Alles wat van de fundingreeks afhangt | H-11.1. Ook geen tweede momenten van carryconstructies. |
| `alpha/cm_carry.py`, `alpha/cm_tsmom.py` | Dragen de xfail-killgates. Hun N_eff-kopie wordt een DI en geen reparatie. |
| `portfolio/covariance.py`, `portfolio/legacy_sizing.py` | DI-10: bewust ongewijzigd. Delegeren mag, wijzigen niet. |
| Het gedrag van `validation/inference.py` | Zolang stage B van de zusterfase loopt (§5.2). |
| `m_new = 25`, bestaande ledger-entries, `holdout_lock.json` | Bevroren, append-only, één lezing per hypothese. |
| `conf/model/alpha.yaml` (`lookback_bars`, `rebalance_every_bars`) | Een baselineparameter. Wijzigen verandert de meetlat waartegen elke track wordt gemeten. Stap 4 beslist over de dubbele implementatie zonder de waarde aan te raken. |
| De ruff- en mypy-versies | Gepind (DI-16). |

---

## 7. CONCRETE DELIVERABLES

| Stage | Artefact | Vorm |
|---|---|---|
| A | `src/tradebot/validation/breadth.py::independent_bets` | delegeert naar `portfolio/covariance.py::effective_n_assets`; geen tweede implementatie |
| A | `tests/unit/test_breadth_definitions.py` | rangbegrenzing, plus de vastgepinde DI-35 als `xfail(strict=True)` |
| A | DI-35 | het ontwerpeffect heet "breedte"; hernoemen na het oordeel over H-11.1 |
| A | `artefacts/governance/phase11_breadth.json` §breadth | per constructie, per venster, per jaar, met interval |
| B | `src/tradebot/validation/signal_clock.py` | τ_int en de paneeldiagnostiek |
| B | DI-36 of een equivalentiebewijs | twee vasthoudimplementaties |
| C | `validation/breadth.py::required_ic`, `dsr_hurdle` + de simulatie | de muur, geverifieerd; de DSR-drempel voor het eerst als functie en niet als overgetypt getal |
| C | `reports/phase11_ic_wall.md` | de tabellen van §3.6, gemeten en gesimuleerd naast elkaar |
| C | AD-29 | *"Een pre-registratie noemt haar breedte, haar horizon en de IC die de poort daarbij vraagt"* |
| C | AD-30 | *"Onafhankelijke weddenschappen en het ontwerpeffect zijn twee grootheden"* |
| D | de haalbaarheidsrekening in `reports/phase11_breadth_and_timescale.md` | altijd, ook als zij rood is |
| D | H-11.2: pre-registratie, campagne, rapport, ledger-amendement | **alleen** als stap 7 groen is |
| E | `reports/phase11_breadth_owner_decision.md` | de besluiten die alleen de eigenaar kan nemen |
| — | `reports/phase11_breadth_exit_report.md` | de afvinklijst van §EXIT-CRITERIA |

---

# STAPSGEWIJZE UITVOERING

# STAGE A — ÉÉN BREEDTE, MET ONZEKERHEID

*Nul trials. Deze stage maakt van "N_eff" één grootheid met één naam en één
implementatie, en zet een interval om elke waarde.*

---

### Stap 1: De vingerafdruk, en de bevinding die als eerste naar main gaat

**Files:**
- Create: `src/tradebot/validation/breadth.py`
- Create: `tests/unit/test_breadth_definitions.py`
- Modify: `src/tradebot/validation/inference.py` (**alleen** de docstrings van
  `effective_breadth` en van `ClusteredMean.n_effective`)
- Modify: `docs/DEFERRED_ISSUES.md` (DI-35, DI-38)

- [ ] **1.1 — Draai de volledige suite en leg de vingerafdruk vast** vóór enige
  wijziging, naast die van §3.0. Staat de zusterfase al verder, dan is
  haar vingerafdruk de jouwe niet. Leg vast op welke `main`-commit je meet.

- [ ] **1.2 — Reproduceer §2.1 en §3.1** met bijlage A. Wijkt een getal af, dan
  is dat je eerste bevinding.

- [ ] **1.3 — Schrijf de tests eerst (R-11).** De kern:

```python
# tests/unit/test_breadth_definitions.py
"""Twee grootheden, twee namen.

`validation/inference.py::effective_breadth` is Kish' ontwerpeffect van een
gepoold, gelijkgewogen gemiddelde. Dat is de juiste grootheid om een gepoolde
t te defleren en de verkeerde om weddenschappen mee te tellen: op een paneel
dat over de namen tot nul sommeert, gaat haar noemer naar nul.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.validation.breadth import independent_bets
from tradebot.validation.inference import effective_breadth

RNG = np.random.default_rng(35)


def _one_factor(n_names: int = 6, n_obs: int = 2_000) -> np.ndarray:
    market = RNG.standard_normal((n_obs, 1))
    return 0.85 * market + 0.5 * RNG.standard_normal((n_obs, n_names))


def _dollar_neutral(panel: np.ndarray) -> np.ndarray:
    return panel - panel.mean(axis=1, keepdims=True)


def test_independent_bets_never_exceed_the_rank() -> None:
    residual = _dollar_neutral(_one_factor())
    corr = np.corrcoef(residual, rowvar=False)
    assert independent_bets(corr) <= np.linalg.matrix_rank(residual) + 1e-9


def test_clones_are_one_bet_and_independents_are_n() -> None:
    clones = np.repeat(RNG.standard_normal((2_000, 1)), 6, axis=1)
    clones = clones + 1e-6 * RNG.standard_normal(clones.shape)
    independents = RNG.standard_normal((2_000, 6))
    assert independent_bets(np.corrcoef(clones, rowvar=False)) < 1.05
    assert independent_bets(np.corrcoef(independents, rowvar=False)) > 5.8


@pytest.mark.xfail(strict=True, reason="DI-35: het ontwerpeffect is geen breedte")
def test_the_design_effect_is_not_breadth() -> None:
    residual = _dollar_neutral(_one_factor())
    corr = np.corrcoef(residual, rowvar=False)
    assert effective_breadth(corr) <= residual.shape[1]
```

  De derde test **pint het defect vast** in plaats van het te verbergen. Hij
  faalt vandaag met de verwachte fout. Wordt `effective_breadth` later
  gerepareerd of hernoemd, dan slaagt hij, en omdat de marker strict is, wordt de
  suite dan rood tot iemand de marker bewust weghaalt. Dat is hetzelfde patroon
  als de vier killgates.

  **Negatieve controle:** laat `independent_bets` tijdelijk de ρ̄-formule
  aanroepen en zie `test_independent_bets_never_exceed_the_rank` rood worden.
  Draai terug en commit de controle niet.

- [ ] **1.4 — Implementeer `independent_bets` als delegatie**, niet als tweede
  berekening:

```python
# src/tradebot/validation/breadth.py (uittreksel)
from ..portfolio.covariance import effective_n_assets


def independent_bets(correlation_matrix: np.ndarray) -> float:
    """Het aantal onafhankelijke weddenschappen: de participatieratio van de
    eigenwaarden, begrensd door de rang. De implementatie staat in
    `portfolio/covariance.py` en blijft daar (DI-10); dit is de naam waaronder
    de validatielaag haar gebruikt (AD-30)."""
    ...  # require(...) op vorm en symmetrie, dan: return effective_n_assets(c)
```

- [ ] **1.5 — Corrigeer de docstrings**, en alleen die. `effective_breadth` meet
  het ontwerpeffect van een gepoold gemiddelde. De melding in de weigertak
  (*"kan niet uit een geldige correlatiematrix komen"*) is onjuist voor een
  singuliere residuele matrix; noem in de docstring dat het gedrag in die tak
  onveranderd blijft tot DI-35. `ClusteredMean.n_effective` krijgt dezelfde
  aantekening. **Geen enkele regel code in `inference.py` verandert** (§5.2).

- [ ] **1.6 — Boek DI-35 en DI-38.**
  DI-35: *"`effective_breadth` meet het ontwerpeffect en heet breedte; op een
  dollar-neutraal paneel geeft zij 123,6 voor zes namen."* Noem de aanroepers,
  noem stap 6.1 (de AD-20-telling) en stap 8.1 van de zusterfase, en noem de
  sluiting: hernoemen naar een naam die zegt wat zij meet, **ná** het
  ledger-amendement van H-11.1, zodat de artefacten van die hypothese binnen
  haar fase reproduceerbaar blijven.
  DI-38: de kopieën in `alpha/cm_carry.py` en `alpha/cm_tsmom.py`, met als
  reden de fence op de killgates. Oordeel: bewust geaccepteerd.

- [ ] **1.7 — Commit, en open er een eigen PR voor.** De PR-beschrijving noemt
  stap 6.1 en stap 8.1 van `fase_11_meetbasis_en_carry.md` bij naam.

---

### Stap 2: Breedte per constructie, per jaar, per venster

**Files:**
- Modify: `src/tradebot/validation/breadth.py`
- Create: `conf/research/breadth.yaml`, `apps/run_breadth_measurement.py`
  (≤ 80 LOC), `tests/unit/test_breadth.py`
- Create (output): `artefacts/governance/phase11_breadth.json`,
  `reports/phase11_breadth_and_timescale.md` §1

- [ ] **2.1 — Zet elke parameter in `conf/research/breadth.yaml` vóór de eerste
  meting**: vensters, bloklengte, aantal bootstraptrekkingen, seed, de
  τ_int-vensterconstante van stap 3 en het horizonrooster van stap 5. Een
  parameter die na het meten wordt gezet, is een keuze op een uitkomst, ook als
  de uitkomst geen rendement is.

- [ ] **2.2 — Drie constructies**: directioneel (ruw), dollar-neutraal (residu
  na het EW-mandje) en bèta-gehedged (residu na de bèta tegen het EW-mandje). De
  bèta is diagnostisch en mag op het volle venster; zeg dat in het artefact.
  **Implementeer geen handelbare hedge.** Die is van de zusterfase (arm 2). Heeft
  zij haar hedge op `main` gezet voordat je hier bent, gebruik dan de hare (R-3)
  en rapporteer beide.

- [ ] **2.3 — Per venster en per jaar**: `W_DEV`, `W_GATE` en `W_FULL`, plus per
  kalenderjaar. Alleen tweede momenten (R-15).

- [ ] **2.4 — Een interval om elke breedte (R-8)**, met
  `validation/inference.py::circular_block_indices`. Dat is de bestaande
  blokbootstrap; schrijf er geen tweede (R-3).

- [ ] **2.5 — Rapporteer beide formules naast elkaar** voor elke rij, met hun
  eigen naam: "onafhankelijke weddenschappen" en "ontwerpeffect". Nooit
  "N_eff" zonder kwalificatie.

- [ ] **2.6 — Commit.**

---

# STAGE B — DE TWEE KLOKKEN

*Nul trials. De beslisklok en de signaalklok worden gemeten op het besluitpaneel,
niet op het rendement.*

---

### Stap 3: De signaalklok, gemeten zonder rendement

**Files:**
- Create: `src/tradebot/validation/signal_clock.py`, `tests/unit/test_signal_clock.py`
- Modify: `apps/run_breadth_measurement.py`

- [ ] **3.1 — Test eerst.** τ_int van een AR(1)-reeks met bekende φ benadert
  `(1 + φ) / (1 − φ)`. Een constante reeks geeft `inf` en **geen** getal; die
  tak wordt expliciet afgehandeld en niet door een deling door nul. Voeg een
  negatieve controle toe: een AR(1) met φ = 0,9 hoort ≈ 19 te geven, en een
  schatter die alleen lag 1 meeneemt, geeft ≈ 2,8. Laat zien dat de test op die
  afgeknotte schatter rood wordt.

- [ ] **3.2 — Meet per track** de grootheden van §3.4 op `W_DEV`: unieke rijen,
  bars met een wijziging, omzet, ac1, τ_int en onafhankelijke besluiten per jaar.
  Gebruik `build_weight_tracks` precies zoals `apps/run_h10_1_decision_frequency.py`,
  zodat de panelen dezelfde zijn als die van H-10.1.

- [ ] **3.3 — Schrijf de verklaring van H-10.1 op** in het rapport: temporele
  breedte nul op de primaire cel, en dat dit vóór de registratie uit de
  gewichtsmatrix te lezen was. **Niet in de ledger**: die is append-only en
  H-10.1's oordeel verandert er niet door.

- [ ] **3.4 — Commit.**

---

### Stap 4: Eén vasthoudoperatie

**Files:**
- Create: `tests/unit/test_hold_equivalence.py`

- [ ] **4.1 — Bewijs waar de twee implementaties samenvallen** (§3.9). Let op
  het anker. De kalender van de unit telt vanaf de eerste bar van het
  **featurepaneel**, inclusief de opwarmbars. `hold_decision` telt vanaf de
  eerste bar van het paneel dat hij krijgt, en H-10.1 geeft hem het
  **verhandelbare** venster (ordeningsbesluit 1). Veranker beide op dezelfde bar
  en toon dan aan dat gelijkgewogen sizing (`portfolio/equal_weight.py`,
  normalisatie per rij) een bit-identiek boek geeft. Toon ook aan dat
  risicopariteit een **ander** boek geeft. Een equivalentietest die ook daar
  groen is, test niets. Zonder gelijk anker verschillen de twee al door de fase
  van de kalender, en dat is een derde bevinding die hoort te worden opgeschreven.

- [ ] **4.2 — Beslis en schrijf het op.** De voorkeur van dit document is:
  `hold_decision` is de ene implementatie van "een besluit vasthouden" (hij heeft
  de causaliteitstest en de precedentie). `rebalance_every_bars` blijft op 1 en
  wordt DI-36, met als sluiting het verwijderen van de parameter uit de unit
  zodra een fase de baseline-unit om een andere reden opnieuw afleidt. Wijzig
  `conf/model/alpha.yaml` niet (§6).

- [ ] **4.3 — Commit.**

---

# STAGE C — DE MUUR

*Nul trials. De vraag welke IC de poort vraagt, wordt één functie, één
simulatie en één veld in elke volgende pre-registratie.*

---

### Stap 5: `required_ic`, en de simulatie die haar controleert

**Files:**
- Modify: `src/tradebot/validation/breadth.py`
- Create: `tests/unit/test_ic_wall.py`
- Create (output): `reports/phase11_ic_wall.md`

- [ ] **5.1 — `required_ic(sr_required, independent_bets, horizon_bars, bars_per_year)`.**
  Test eerst: de functie keert de wet exact om, en ze weigert
  (`DataContractError`, R-5) bij een breedte ≤ 0 of een horizon ≤ 0. Typ
  `sr_required` niet over. Voeg `dsr_hurdle(n_obs, n_trials, skew, kurtosis,
  sr_variance, bars_per_year)` toe, die `backtest/metrics.py::deflated_sharpe`
  numeriek omkeert (een wortelzoeker op DSR = 0,95) en dus geen tweede
  DSR-formule is (R-3). Haar eerste test reproduceert 1,868609 bij M = 25,
  N = 1.390 onder normaliteit (§3.6). Haar negatieve controle laat zien dat de
  drempel stijgt wanneer M stijgt of N daalt.

- [ ] **5.2 — De simulatie.** Genereer synthetische rendementen uit de gemeten
  covariantie van `W_DEV`, met een voorspelling van **bekende** IC, en meet de
  gerealiseerde Sharpe met de echte kern (`validation/inference.py::sharpe_with_se`).
  Doe dat per constructie, voor h ∈ het rooster uit `conf/research/breadth.yaml`
  en voor een IC-rooster rond de muur. Bijlage B is de verkenning; zij meet de IC
  op elke vijftigste dag en is daarom ruisig. De stap meet haar op elke dag, met
  een vaste seed.

  > **Beslisregel, vooraf.** Ligt de gesimuleerde Sharpe binnen 10 % van
  > `IC·√BR` over het hele rooster, dan is de formule de muur. Anders is de
  > gesimuleerde afbeelding de muur, en rapporteert `required_ic` die met de
  > afwijking erbij.

- [ ] **5.3 — Zet de enige in-domein-IC's ernaast die er al zijn**, geciteerd en
  niet opnieuw gemeten: fase 10 §5.7, −0,024 tot +0,050 bij een standaardfout van
  0,011, over 55 diagnostische cellen. **Meet geen nieuwe IC** (R-15).

- [ ] **5.4 — Schrijf `reports/phase11_ic_wall.md`** met de formule en de
  simulatie naast elkaar, voor `W_DEV` en `W_GATE`.

- [ ] **5.5 — Commit.**

---

### Stap 6: AD-29 en AD-30

- [ ] **6.1 — AD-29: *"Een pre-registratie noemt haar breedte, haar horizon en
  de IC die de poort daarbij vraagt."*** Implementeer dat als **additieve**
  controle `assert_ic_wall_declared(spec)` in `validation/breadth.py`. Nieuwe apps
  roepen haar aan vóór `freeze_preregistration`. Zij wordt **niet** in
  `registry/preregistration.py` ingebouwd en **niet** met terugwerkende kracht
  toegepast. Het afgewezen alternatief is haar nu verplicht maken in
  `freeze_preregistration`: dat breekt de bevriezing van H-11.1 halverwege een
  lopende fase. Het verplicht maken is DI-37, met als voorwaarde het
  ledger-amendement van H-11.1.

- [ ] **6.2 — AD-30: *"Onafhankelijke weddenschappen en het ontwerpeffect zijn
  twee grootheden."*** De eerste telt voorspellingen en hoort in de fundamentele
  wet. De tweede defleert een gepoold gemiddelde en hoort in `clustered_mean`.
  Het afgewezen alternatief is één functie met een vlag: dat is één naam voor
  twee grootheden, en dat is precies het defect.

- [ ] **6.3 — Commit AD-29 en AD-30 in één aparte commit**, volgens het
  CRLF-protocol van §5.2.

---

# STAGE D — H-11.2: DE BESLISKLOK VOLGT DE SIGNAALKLOK

*Hoogstens één trial, en alleen na twee poorten. De verwachting is nul.*

---

### Stap 7: De haalbaarheidspoort, vóór registratie

**Voorwaarde:** `artefacts/baseline/phase11_revaluation.json` staat op `main` en
draagt `risk_policy_hash: 9961e1613bc907a5`. Zolang dat niet zo is, wacht deze
stap, en de rest van de fase gaat door.

> **De hypothese, zoals zij geregistreerd zou worden.** *H-11.2. Op
> `xs_momentum_equal_weight` verhoogt het vasthouden van het besluit gedurende
> k\* = ⌈τ_int⌉ bars (de signaalklok uit stap 3) de netto Sharpe op L3 ten
> opzichte van k = 1, en het verschil overleeft de Ledoit–Wolf-toets met
> deflatie bij M_new = 25.*
>
> **Waarom deze cel.** Het is de enige track waarvan het besluitpaneel
> niet-ontaard is én uitsluitend door het signaal verandert (§3.4), en waarvan de
> kosten overwegend omzet zijn (§3.5). **Openheid:** de auteur van deze prompt
> kent de L3-Sharpe van alle vier de tracks bij k = 1
> (`phase5_revaluation.json`), hun omzet bij k ∈ {1, 2, 5, 10} (de diagnostiek
> van H-10.1) en hun omzet en tweede momenten bij k = 30 (§3.7). De auteur kent
> **geen** Sharpe bij enige k ≠ 1 op enige track behalve
> `long_only_equal_weight`, waar die per constructie gelijk is aan k = 1. Het
> H-10.1-artefact draagt voor de drie andere tracks
> `carries_no_performance_number: true`, en dat is nagekeken.
>
> **Waarom k\*.** k\* komt niet uit een rooster. Het is de klok van het signaal
> zelf, gemeten op het besluitpaneel voordat er een rendement bij k\* bestaat.
> De verwachting is k\* ≈ 30 (§3.4).

- [ ] **7.1 — Meet de kostenmix opnieuw** op het nieuwe ladderartefact en werk de
  factor 1,54 uit §3.7 bij.

- [ ] **7.2 — Reken de detectiegrens uit met de echte kern (R-3).** Vervang de
  benadering `√((2 − 2ρ) / t_jaar)`. Genereer rendementen **met gemiddelde
  nul** uit de covariantie van `W_DEV`, pas beide gewichtspanelen toe (k = 1 en
  k\*), en draai `validation/inference.py::sharpe_difference_test` op dat
  synthetische paar. De spreiding van het verschil onder de nulhypothese is de
  standaardfout die de echte toets zou gebruiken. Zo is er geen enkel echt
  gemiddeld rendement in de rekening gekomen.

- [ ] **7.3 — Pas de poort toe, zoals zij hier vooraf staat.**

  > **Groen** dan en slechts dan wanneer de kostenwinst bij nul signaalverval,
  > op de L3-kostenmix, groter is dan het kleinste verschil dat de toets van
  > H-10.1 (95 %-CI sluit nul uit na deflatie bij M_new = 25) kan onderscheiden.
  >
  > **Rood** in elk ander geval. Dan wordt H-11.2 niet geregistreerd, kost zij
  > nul trials, en wordt V3 beantwoord met de kostengrootheid, de breakeven
  > (`portfolio/decision_frequency.py::breakeven_cost_bps`) en de detectiegrens,
  > zoals §3.7 het formuleert.

  Deze poort veronderstelt dat vasthouden het signaal niet **verbetert**. Dat
  is geen stelling: een signaal met ruis die snel terugvalt, kan van vasthouden
  profiteren. Zeg dat in het rapport. Het maakt de poort niet ongeldig, want een
  hypothese die alleen haalbaar is als het signaal door vasthouden béter wordt,
  is een andere hypothese dan H-11.2.

- [ ] **7.4 — Schrijf de rekening op, ook en juist als de poort rood is**, in
  `reports/phase11_breadth_and_timescale.md` §3. Een niet-geregistreerde
  hypothese met een uitgerekende reden is een resultaat; een stilzwijgend
  overgeslagen hypothese is dat niet.

- [ ] **7.5 — Commit.**

---

### Stap 8: Alleen als stap 7 groen is — registratie, meting, oordeel

Is stap 7 rood, sla deze stap dan over en noteer in het exitrapport dat dat de
vooraf geregistreerde handeling is en geen overgeslagen stap.

- [ ] **8.1 — Pre-registratie, bevroren vóór de eerste run**, met
  `apps/freeze_preregistration.py`, in
  `conf/research/preregistration_h11_2_decision_clock.yaml`, met minimaal:
  `primary_cell` (`xs_momentum_equal_weight` / `L3_execution`, netto Sharpe,
  `W_DEV`), `k_star` (de gemeten waarde, geen rooster), `planned_trials: 1`,
  `assert_within_budget(14)` (§5.4), de IC-muurvelden van AD-29, de
  beslisregel van 8.4 als stopcriteria, elk met zijn actie, en de negatieve
  controles van 8.2.

- [ ] **8.2 — Twee negatieve controles (R-1), allebei verplicht.**
  1. **Identiteit.** Dezelfde code op `long_only_equal_weight` geeft door de
     volledige ladder een verschil van **exact** nul. Dat bewijst dat het pad
     geen bijwerking heeft.
  2. **Idempotentie.** Een paneel dat al met k\* is vastgehouden, nog eens met
     k\* vasthouden, geeft door de volledige ladder exact nul. Dat bewijst dat de
     ladder geen verborgen padafhankelijkheid van het vasthouden heeft (H-10.1,
     ordeningsbesluit 4).

- [ ] **8.3 — Meet en rapporteer de decompositie vóór het oordeel**: het verschil
  gesplitst in een kostendeel en een bruto-deel (signaalveroudering). Daarnaast
  de volledige inferentiekern (R-8) en de breakeventabel van H-10.1, met
  hergebruik van `breakeven_cost_bps` (R-3).

- [ ] **8.4 — De beslisregel van H-10.1, ongewijzigd overgenomen.**
  `CONFIRMED` vereist alle vier: (1) `delta_sharpe > 0` op de ontwikkelsample,
  (2) het 95 %-CI sluit nul uit na deflatie bij M = 25, (3) hetzelfde teken op
  de poortsample, eenmaal gelezen, vooraf geregistreerd, en (4) de breakeven ligt
  onder de vaste kosten van 6,5 bp per zijde. Ontbreekt er één, dan is het
  `UNPROVEN`. `REJECTED` is voor een aantoonbaar negatief effect. De poort wordt
  alleen gelezen als (1) en (2) gelden (R-7).

- [ ] **8.5 — Boek in de ledger** in de vorm van §5.4, met `metrics.verdict`,
  `m_new`, `deflated_sharpe` en `t_hurdle_at_n`.

- [ ] **8.6 — Commit.**

---

# STAGE E — BREEDTE BUITEN HET DOMEIN

*Nul trials, nul code. Een document voor de eigenaar.*

---

### Stap 9: Het eigenaarsdocument

**File:** `reports/phase11_breadth_owner_decision.md`

- [ ] **9.1 — De gemeten feiten, zonder bijvoeglijke naamwoorden**: het maximale
  aantal onafhankelijke weddenschappen per jaar per constructie; de muur; het
  poortvenster als smalste venster; de dalende breedte over de jaren (§3.2); en
  de IC's die het domein ooit heeft laten zien.

- [ ] **9.2 — Wat de breedte zou veranderen, en wat elk pad kost.** Drie paden,
  elk met zijn kosten en zonder aanbeveling:
  1. **Meer Bybit-perpetuals.** Formeel binnen de bronnen van AD-23, maar het
     verandert het universum van het meetcontract, en elke toegevoegde naam is
     een overlever (DI-15). De gemeten structuur zegt iets over wat het oplevert:
     de directionele breedte blijft rond 1,6, omdat de eerste factor 78,5 % van
     de variantie draagt. Alleen de dollar-neutrale breedte schaalt mee, met
     hoogstens N − 1. Dat is exact de breedte waarvan F10's bewijs zegt dat zij
     niet betaalt (*raw reversal +0,30 bij 12 namen → +0,06 bij 99*).
  2. **Een langer venster.** H-10.2: +25,8 % bars, maar smallere bars
     (gemiddelde breedte 3,265 tegen 6,000). Het vergt bovendien dat
     `RiskEngine.decide` een deelverzameling per bar accepteert. Dat is een
     wijziging in het risicocontract, na de zusterfase.
  3. **Breedte over markten.** Buiten het meetdomein (AD-23). Een mandaatbesluit,
     en de enige route die F10 en F20 open laten.

- [ ] **9.3 — Het besluit dat alleen de eigenaar kan nemen**, met de drie
  uitkomsten die de gemeten muur toelaat: stoppen; het universum binnen Bybit
  verbreden met de survivorship-aantekening op elke claim; of een nieuw mandaat.
  **Deze fase kiest niet.** Zij levert de getallen waarop gekozen kan worden.

- [ ] **9.4 — Werk DI-15 en DI-21 bij**, append-only, met de gemeten breedte als
  nieuw bewijs en met hun voorwaarden ongewijzigd.

- [ ] **9.5 — Commit.**

---

### Stap 10: Het exitrapport

- [ ] **10.1 — `reports/phase11_breadth_exit_report.md`**: de afvinklijst
  hieronder, de vingerafdruk aan begin en eind, de trialstand (fase 10 + de
  zusterfase + deze fase ≤ 25), en een lijst van alles wat `CHANGELOG.md` en
  `docs/PROJECT_STATE.md` na samenvoegen moeten zeggen (§5.2).

- [ ] **10.2 — Commit.**

---

# EXIT-CRITERIA

De fase is af wanneer **alle** onderstaande regels waar zijn. Een regel die niet
waar is, blokkeert de afsluiting.

| # | Criterium | Verificatie |
|---|---|---|
| 1 | De vingerafdruk aan begin en eind staat vast; elke failure die er aan het eind bij is gekomen, is van deze fase of met bewijs van de zusterfase | stap 1.1, R-14 |
| 2 | `independent_bets` bestaat, delegeert naar `portfolio/covariance.py::effective_n_assets`, en er is geen vijfde implementatie van een breedte bijgekomen | stap 1.4 |
| 3 | De rangbegrenzing is getest en de negatieve controle is aantoonbaar rood geweest | stap 1.3 |
| 4 | DI-35 staat vastgepind als `xfail(strict=True)`; het gedrag van `validation/inference.py` is ongewijzigd | stap 1.3, 1.5 |
| 5 | Stap 1 staat als eigen PR op `main`, met stap 6.1 en 8.1 van de zusterfase in de beschrijving | stap 1.7 |
| 6 | Elke breedte in het artefact draagt een venster, een paneel, een constructie, een formulenaam en een interval | stap 2 |
| 7 | Er is geen eerste moment van enig signaal gemeten, en geen enkele grootheid die van funding afhangt | R-15, §0.1 |
| 8 | τ_int is per track gemeten; de constante-reeks-tak geeft `inf` en is getest | stap 3 |
| 9 | De verklaring van H-10.1 staat in het rapport en niet in de ledger | stap 3.3 |
| 10 | De twee vasthoudimplementaties zijn op equivalentie getest, met het verschil bij risicopariteit zichtbaar; DI-36 is geboekt of de dubbeling is opgeheven | stap 4 |
| 11 | `required_ic` bestaat en weigert ongeldige invoer; `dsr_hurdle` keert `deflated_sharpe` om en reproduceert 1,868609; de muur is door simulatie met de echte kern geverifieerd, met de afwijking gerapporteerd | stap 5 |
| 12 | AD-29 en AD-30 zijn geschreven; `ARCHITECTURAL_DECISIONS.md` is nog volledig CRLF en eindigt op één CRLF | stap 6 |
| 13 | AD-29 is additief; `freeze_preregistration` is ongewijzigd; DI-37 draagt de voorwaarde | stap 6.1 |
| 14 | De haalbaarheidsrekening van stap 7 staat in het rapport, met de detectiegrens uit de echte kern, **ongeacht** haar uitkomst | stap 7.4 |
| 15 | H-11.2 is geregistreerd dan en slechts dan wanneer de poort van stap 7 groen was | stap 7.3 |
| 16 | Is H-11.2 geregistreerd: beide negatieve controles zijn gedraaid, de decompositie staat vóór het oordeel, en het oordeel volgt de beslisregel van H-10.1 | stap 8 |
| 17 | `holdout_lock.json` is ongewijzigd, of draagt precies één lezing voor H-11.2 | stap 8.4 |
| 18 | Het trialsaldo klopt: 5 + (werkelijk door de zusterfase) + (deze fase) ≤ 25, en deze fase besteedde ≤ 1 | §5.4 |
| 19 | Het eigenaarsdocument bestaat, noemt drie paden met hun kosten, en kiest er geen | stap 9 |
| 20 | Geen bestand uit de lijst "van de zusterfase" in §5.2 is door deze fase gewijzigd | `git diff --name-only` tegen de basis |
| 21 | `ruff` en `mypy` zijn schoon; elke nieuwe app ≤ 80 LOC; nieuwe modules op ratchetbudget 0; de zes poortscripts geven exit 0 | continu |

---

# REGELS & HANDELINGSINSTRUCTIES

**R-1 t/m R-14 uit `fase_11_meetbasis_en_carry.md` gelden hier onverkort**:
causaliteit is een test (R-1); elke parameter kost een trial (R-2); één
implementatie per grootheid (R-3); ≤ 800 LOC per bestand (R-4); fail fast met
een reden (R-5); apps ≤ 80 LOC (R-6); één poortlezing per hypothese (R-7); geen
getal zonder onzekerheid (R-8); Nederlands, met Engelse code (R-9); een
verwachting is geen resultaat (R-10); TDD zonder uitzondering (R-11); commit per
stap (R-12); een meting draagt haar configuratie (R-13); de vingerafdruk is een
poort (R-14). Deze fase voegt er drie aan toe:

**R-15 — Tweede momenten mogen, eerste momenten niet.** Stages A, B en C
rekenen met correlaties, eigenwaarden, besluitpanelen en omzet. Een gemiddeld
rendement, een Sharpe of een IC van een signaal is in die stages verboden, ook
"alleen om te kijken". Tweede momenten op `W_FULL` mogen, met het venster in het
artefact. Eerste momenten op `W_GATE` nooit, buiten een geregistreerde lezing.

**R-16 — Wat van de zusterfase is, raak je niet aan.** Ook niet wanneer je een
fout ziet. Een fout in haar bestanden wordt een DI met haar stapnummer erin, en
een zin in je PR. Ze wordt geen commit van jou.

**R-17 — Een haalbaarheid die vooraf uit te rekenen is, reken je vooraf uit.**
Een hypothese waarvan vóór de registratie te berekenen is dat zij niets kan
meten (H-10.1) of niet gehaald kan worden (H-10.3), wordt niet geregistreerd. De
rekening wordt wel opgeschreven, want zij is het resultaat.

---

# STARTINSTRUCTIE

Begin bij **stap 1**, en zorg dat stap 1 op `main` staat voordat je aan stap 2
begint. Werk daarna de stappen in volgorde af. Stage D begint pas wanneer de
voorwaarde van stap 7 op `main` waar is; tot die tijd lopen stage E en het
exitrapport door.

Vóór de eerste regel code, vier handelingen in deze volgorde:

1. **Lees §5 opnieuw, en daarna de prompt van de zusterfase.** Stel met
   `git fetch origin` en `git log origin/main` vast hoe ver zij is.
2. **Reproduceer §3 met bijlage A.** Niet steekproefsgewijs. Wijkt iets af, dan
   is dat je eerste bevinding.
3. **Controleer of `28cc31b` nog de kop van `main` is.** Is er sinds 2026-09-25
   gecommit, en dat is zeker zo als de zusterfase iets heeft samengevoegd, dan is
   §3 een historische meting. Meet opnieuw voordat je hem gebruikt, in het
   bijzonder §3.5 en §3.7, die van de ladder afhangen.
4. **Stel vast of dit document zelf klopt.** Deze prompt stelt al drie dingen
   over zijn voorgangers vast die daar niet stonden: het paneel achter de 1,271
   (§3.1), het venster van het jaar 2021 (§3.2) en dat de DSR-drempel van 1,8686
   nergens als functie bestaat (§3.6). Er is geen reden om aan te nemen dat hij
   zelf zonder fouten is.

Rapporteer na elke stap: welke test faalde, met welke fout, wat de
implementatie werd, welke test slaagde, en de commit-SHA. Rapporteer na elke
stage de trialstand, de vingerafdruk en de stand van `main` ten opzichte van je
tak.

De verwachtingen in dit document zijn expliciet gemaakt zodat ze kunnen worden
weerlegd, niet zodat ze de meting kunnen vervangen. Dat geldt het sterkst voor
de verwachting dat deze fase nul trials besteedt.

---

# BIJLAGE A — REPRODUCTIE VAN §3

Diagnostisch, nul trials, schrijft niets. Draai vanuit de repositoryroot met de
gepinde omgeving. Het script gebruikt uitsluitend loaders die de repository zelf
gebruikt, zodat de panelen die van de ladder zijn.

```python
# nulmeting_breedte.py -- tweede momenten en besluitpanelen, geen rendement.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "apps"))
from run_phase5_baseline import load_market
from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import load_baseline_configs
from tradebot.backtest.baseline_runner import build_weight_tracks
from tradebot.data.pit_store import PitStore
from tradebot.features.base import DataRegister, load_certified_close_panel
from tradebot.portfolio.covariance import effective_n_assets
from tradebot.portfolio.decision_frequency import hold_decision
from tradebot.validation.inference import effective_breadth

DEV = (pd.Timestamp("2021-11-15", tz="UTC"), pd.Timestamp("2025-09-04", tz="UTC"))
GATE = (pd.Timestamp("2025-09-05", tz="UTC"), pd.Timestamp("2026-08-23", tz="UTC"))

cfg = load_baseline_configs(ROOT)
market = load_market(ROOT, cfg)
logret = np.log(market["prices"]).diff()
day = logret.index.normalize()


def window(lo, hi):
    return logret[(day >= lo) & (day <= hi)].dropna(how="any")


def row(name, r):
    c = r.corr().to_numpy()
    rho = c[~np.eye(c.shape[0], dtype=bool)].mean()
    print(f"{name:32s} rho={rho:+.4f} design_effect={effective_breadth(c):8.3f} "
          f"independent_bets={effective_n_assets(c):.3f}")


dev, gate = window(*DEV), window(*GATE)
row("W_DEV directioneel", dev)                                   # §3.1
row("W_DEV dollar-neutraal", dev.sub(dev.mean(axis=1), axis=0))  # §2.1
row("W_GATE directioneel", gate)                                 # §3.2
row("W_GATE dollar-neutraal", gate.sub(gate.mean(axis=1), axis=0))

register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
panel = load_certified_close_panel(
    PitStore(ROOT / cfg["data"].pit_store_root), register,
    symbols=list(cfg["data"].symbols), granularity="1d", asset_class="crypto")
tracks, _ = build_weight_tracks(
    panel, build_cross_sectional_momentum(cfg["alpha"]),
    lam=cfg["vol"].ewma_lambda, vol_burn_in_bars=cfg["vol"].burn_in_bars,
    annualisation_factor=cfg["vol"].annualisation_factor,
    gross_target=1.0, git_sha="nulmeting")
usable = market["sigma"].dropna(how="any").index
usable = usable[usable.isin(market["adv"].dropna(how="any").index)]
cov = dev.cov().to_numpy()

for name in sorted(tracks):                                      # §3.4, §3.7
    w = tracks[name].loc[usable].fillna(0.0)
    wd = w[w.index.normalize() <= DEV[1]]
    turn = (wd - wd.shift(1)).abs().sum(axis=1).iloc[1:]
    unique = len(np.unique(np.round(wd.to_numpy(), 12), axis=0))
    print(f"{name:26s} unique={unique} changes={int((turn > 1e-12).sum())} "
          f"turnover={turn.mean():.4f}")
    for k in (10, 30):
        held = hold_decision(w, k=k, anchor="first_bar").loc[wd.index]
        tk = (held - held.shift(1)).abs().sum(axis=1).iloc[1:].mean()
        a, b = wd.to_numpy(), held.to_numpy()
        vaa = np.einsum("ti,ij,tj->t", a, cov, a).mean()
        vbb = np.einsum("ti,ij,tj->t", b, cov, b).mean()
        vab = np.einsum("ti,ij,tj->t", a, cov, b).mean()
        rho_ab = vab / np.sqrt(vaa * vbb) if vaa > 0 and vbb > 0 else float("nan")
        print(f"{'':26s} k={k:2d} turnover={tk:.4f} rho(k1,k)={rho_ab:.3f} "
              f"sigma_book={np.sqrt(vaa * 365):.3f}")
```

Wat het niet doet, en waarom:

1. **Geen τ_int.** De schatter hoort in `validation/signal_clock.py` met zijn
   eigen tests (stap 3). De waarden in §3.4 komen van een Sokal-venster met
   c = 5 in een verkenningsscript, en stap 3 reproduceert of weerlegt ze.
2. **Geen bootstrap.** De intervallen in §3.1 komen uit een verkenning met een
   eigen blokindexering. Stap 2.4 gebruikt de indexering van de inferentiekern,
   en kleine verschillen zijn daarom te verwachten.
3. **Geen kosten.** §3.5 komt rechtstreeks uit
   `artefacts/baseline/phase5_revaluation.json`, veld `audit` op elke
   L3-rij.

---

# BIJLAGE B — DE VERKENNENDE SIMULATIE ACHTER §2.3

Synthetisch. Zij gebruikt de residuele covariantie van `W_DEV`, geen rendement.
Een voorspelling met bekende IC wordt op het residu gezet, het boek wordt
dollar-neutraal gewogen, en de gerealiseerde Sharpe wordt vergeleken met
`IC·√(365·n)`:

| IC opgelegd | IC gemeten | Sharpe gerealiseerd | wet, n = PR 4,353 | wet, n = N − 1 = 5 | wet, n = N = 6 |
|---:|---:|---:|---:|---:|---:|
| 0,02 | 0,0185 | 0,894 | 0,739 | 0,792 | 0,868 |
| 0,05 | 0,0566 | 2,139 | 2,254 | 2,416 | 2,647 |
| 0,10 | 0,1010 | 4,366 | 4,028 | 4,317 | 4,729 |

De wet houdt op deze structuur ongeveer stand. De afwijking ligt binnen
ongeveer 15 %, en de juiste noemer ligt tussen de participatieratio en N − 1.
De IC is hier op elke vijftigste dag gemeten, en dat is de voornaamste bron van
ruis. Daarom is dit een verkenning en geen resultaat: stap 5.2 meet op elke
dag, met de echte kern en een vaste seed, en beslist met de vooraf vastgelegde
regel of de formule of de simulatie de muur is.
