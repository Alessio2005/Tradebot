# FASE 10 — STAP 11: H-10.1, DE BESLISFREQUENTIE

> **Verdict: UNPROVEN.** Twee van de drie beoordeelbare stop-criteria binden.
> `REJECTED` is niet van toepassing: dat is voorbehouden aan een aantoonbaar
> negatief effect en de bovengrens van het interval is exact nul.
>
> **Het poortsample is NIET gelezen, en dat is de vooraf geregistreerde
> handeling.** `holdout_lock.json` draagt nog `reads: []`; de ene lezing die R-7
> toestaat, is niet besteed. Zie §6.

**Gegenereerd:** 2026-09-13
**git_sha (meting):** `8df453b`
**Meetvenster:** ontwikkelsample W_DEV, 2021-11-15 → 2025-09-04, **1.390 bars**,
**3,8082 jaar**, 6 symbolen. Ladder gedraaid over het volle bruikbare venster van
**1.743 bars**; de snede naar de ontwikkelsample gebeurt op de resultaatreeksen
(zie §7, ordeningsbesluit 4).
**Pre-registratie:** `3bf8a8b17f610296ba488af5376136e9`, bevroren in
`artefacts/governance/preregistration_3bf8a8b17f610296ba488af5376136e9.json`,
`planned_trials: 4`
**Bron:** `artefacts/governance/phase10_h10_1.json`, geproduceerd door
`apps/run_h10_1_decision_frequency.py` op
`src/tradebot/validation/phase10_decision_frequency_measurement.py`
**Trials:** 4 (k ∈ {1, 2, 5, 10}), geboekt vóór het oordeel. `M_new = 25`
bevroren (AD-24); `remaining(booked=4) = 21`
**Faseopdracht:** stap 11 (`Prompts-fases/fase_10_herstart_dagbars.md`), R-2, R-3,
R-7, R-8, R-10

---

## 0. De beslisregel, zoals zij VOORAF stond

Uit `conf/experiment/h10_1_decision_frequency.yaml`, bevroren vóór de eerste run:

> `CONFIRMED` vereist alle vier: (1) `delta_sharpe > 0` op de ontwikkelsample,
> (2) het 95 %-CI sluit nul uit **na** deflatie met `M = 25`, (3) hetzelfde teken
> op de poortsample, en (4) `breakeven_cost_bps` ligt onder een plausibele
> kostenaanname. Ontbreekt er één, dan is het `UNPROVEN`. `REJECTED` is
> voorbehouden aan een aantoonbaar negatief effect.

---

## 1. De primaire cel — `long_only_equal_weight` / `L3_execution`, netto Sharpe

| k | haltes | gem. omzet | ann. Sharpe | Lo-SE | t | DSR (M=25) | Δ vs k=1 | 95 %-CI | p | breakeven |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **1** (ref) | 1566 | 0,000719424 | −0,8055 | 0,4909 | −1,641 | 9,879e-05 | 0 *(constructie)* | [0, 0] | — | — |
| 2 | 1566 | 0,000719424 | −0,8055 | 0,4909 | −1,641 | 9,879e-05 | +0,00000000 | [0, 0] | 1,000 | — |
| 5 | 1566 | 0,000719424 | −0,8055 | 0,4909 | −1,641 | 9,879e-05 | +0,00000000 | [0, 0] | 1,000 | — |
| 10 | 1566 | 0,000719424 | −0,8055 | 0,4909 | −1,641 | 9,879e-05 | +0,00000000 | [0, 0] | 1,000 | — |

Elke rij is identiek, en dat is geen afrondingsartefact maar een **exacte
gelijkheid**. De oorzaak staat in §2.

`breakeven_cost_bps` is voor elke k `null`. Dat is een uitkomst en geen ontbrekend
getal: het omzetverschil is exact nul, en dan bestaat er geen kostenniveau dat het
teken van het Sharpe-verschil omdraait. `breakeven_cost_bps` weigert dat geval met
een `DataContractError` in plaats van een oneindigheid terug te geven — de
bevinding is dat de kostenschakel leeg is, en dat mag niet als rekenkundig detail
worden weggeschreven.

De verschiltoets is bij **k = 1 niet aangeroepen**. Op twee identieke reeksen valt
zij in de schaal-invariante ontaardingstak (commit `812a232`) en geeft dan het
juiste getal om de verkeerde reden: nul omdat de covariantie degenereert, in plaats
van nul omdat er niets is vastgehouden. Er is daar dus ook geen p-waarde, want er
is niet getoetst.

`n_effective = 177` op de drie kandidaten, met `n_dropped = 1213`: de
`common_active`-uitlijning laat de gehalteerde bars vallen, en er blijven 177
gemeenschappelijk actieve bars over — **0,485 jaar**. De t = 2-drempel op die
horizon is een geannualiseerde Sharpe van **2,87**. Ook als daar een verschil had
gestaan, had dat venster het niet kunnen aantonen.

---

## 2. Waarom het verschil EXACT nul is

`hold_decision` houdt het besluit van de eerste bar van elk blok vast. Op deze
track is er geen besluit dat verandert:

| grootheid | gemeten |
|---|---:|
| unieke gewichtsrijen in het besluitpaneel | **1** op 1.743 bars |
| bars die het vasthouden beweegt, bij k = 2, 5 én 10 | **0** |

Een gelijkgewogen long-only boek op een universum dat niet van samenstelling
verandert **is** een constante vector: zes namen, elk 1/6, elke bar. Vasthouden van
een besluit dat niet verandert, is de identiteit. De drie kandidaat-k's zijn
daarom bit-identiek aan de referentie.

**Dat is een uitspraak over de allocator van deze track, niet over de
beslisfrequentie.** `hold_decision` is niet inert — §3 meet dat hij op de andere
drie tracks de omzet wél stevig verlaagt.

> **De voorspelling die vooraf in de pre-registratie stond, is uitgekomen** (R-10).
> Zij voorspelde op grond van de gewichtsmatrix — 7 unieke rijen op het volle
> paneel van 2.342 bars, 0,3 % bewegende bars — dat `turnover_delta` op de primaire
> cel vrijwel nul zou zijn en dat `breakeven_cost_bps` daar geen oplossing zou
> hebben. Gemeten is het scherper dan voorspeld: op het bruikbare venster van
> 1.743 bars is het **1** unieke rij en **exact** nul, niet "vrijwel".

---

## 3. De diagnostiek — en hier zit de sterkere bevinding

Alle vier tracks, alle vier k's. **Diagnostiek: er is niets uit geselecteerd** —
de primaire cel stond vooraf vast. Nul extra trials.

| track | primair | k | unieke rijen | bewogen bars | gem. omzet | omzetdaling | haltes |
|---|:-:|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | ✓ | 1 | 1 | 0 | 0,000719 | — | **1566** |
| `long_only_equal_weight` | ✓ | 10 | 1 | 0 | 0,000719 | 0,00 % | **1566** |
| `long_only_risk_parity` | | 1 | 1743 | 0 | 0,017697 | — | **1566** |
| `long_only_risk_parity` | | 2 | 1743 | 871 | 0,014183 | 19,86 % | **1566** |
| `long_only_risk_parity` | | 5 | 1743 | 1394 | 0,010262 | 42,01 % | **1566** |
| `long_only_risk_parity` | | 10 | 1743 | 1568 | 0,007472 | **57,78 %** | **1566** |
| `xs_momentum_equal_weight` | | 1 | 329 | 0 | 0,153697 | — | **0** |
| `xs_momentum_equal_weight` | | 10 | 329 | 1199 | 0,049201 | **67,99 %** | **0** |
| `xs_momentum_risk_parity` | | 1 | 1743 | 0 | 0,176944 | — | **0** |
| `xs_momentum_risk_parity` | | 10 | 1743 | 1568 | 0,053992 | **69,49 %** | **0** |

**De keten breekt niet op de eerste schakel maar aantoonbaar ook op de derde.**

`long_only_risk_parity` is het beslissende geval, en het was niet voorzien. Die
track **halteert** — 1566 bars, precies zoveel als de primaire — **én** het
vasthouden verlaagt daar de omzet met **57,78 %** bij k = 10. Beide schakels
bestaan er dus. En de haltteller blijft **exact 1566 bij elke k**. Niet
"ongeveer", niet "binnen de ruis": geen enkele halt beweegt.

De twee `xs_momentum`-tracks laten de omzet met 68–69 % dalen en halteren **nul**
bars, bij elke k. Over alle vier tracks samen is de haltteller dus **volledig
ongevoelig voor de beslisfrequentie**.

De hypothese stelde: minder besluiten → minder omzet → minder kosten → minder
drawdown → minder haltes. Schakel 1→2 is gemeten en bestaat (58 % omzetdaling).
Schakel 2→…→5 bestaat niet: een omzetdaling van 58 % op een halterende track
beweegt geen enkele halt. De nulhypothese van de pre-registratie noemt dit vooraf
als **weerlegging** en niet als bevestiging: *"Een daling van de omzet die NIET
door de haltes of door de netto Sharpe wordt gevolgd, weerlegt de hypothese en
bevestigt haar niet."*

**Waarom de haltes niet bewegen — de mechanistische lezing, en zij is niet
gemeten maar wel na te rekenen.** De L3-kosten van `long_only_equal_weight`
splitsen in fees 38,36, **funding 261,67**, spread 6,97 en impact 20,21. Funding
domineert met ruime marge, en funding wordt betaald op de **positie**, niet op de
omzet. Vasthouden verlaagt de omzetgebonden kosten en laat de funding ongemoeid;
de drawdown die de breaker laat vuren is dus grotendeels niet omzetgedreven. Dit
staat hier als **verklaring** en niet als bevinding: het volgt uit de
kostensplitsing van de nulmeting en is in deze stap niet als hypothese getoetst.

---

## 4. De breakeventabel (substap 11.7)

| k | `turnover_delta` | `delta_sharpe_gross` | `delta_sharpe_net` | `breakeven_cost_bps` |
|---:|---:|---:|---:|---:|
| 2 | 0,000000000000 | 0,0 | 0,0 | **geen** |
| 5 | 0,000000000000 | 0,0 | 0,0 | **geen** |
| 10 | 0,000000000000 | 0,0 | 0,0 | **geen** |

De tabel die de stapopdracht "de belangrijkste van deze stap" noemt, is op de
primaire cel **leeg, en dat is haar uitkomst**. De conclusie kan hier niet de vorm
"k = 5 is beter zodra de kosten boven X bp liggen" krijgen, omdat er geen X
bestaat: bij een omzetverschil van nul verschuift geen kostenaanname het antwoord.

`volatility_ratio = 1,0` op elke k, dus de gelijke-volatiliteitsaanname waarop
`breakeven_cost_bps` lineariseert is hier exact vervuld — triviaal, want de sporen
zijn identiek. De kostenas is `cost_per_side = 6,5 bp`
(`taker_fee_bps` 5,5 + `assumed_half_spread_bps` 1,0), en niet `eta`: `eta` treedt
uitsluitend op L3 op via de router en is daar onscheidbaar van spread, fees en
funding, dus daar bestaat "kosten per eenheid omzet" niet als één getal.

---

## 5. De stop-criteria, zoals gemeten

| criterium | eis | gemeten | bindt |
|---|---|---:|:-:|
| `no_improvement_on_development` | `delta_net_sharpe > 0` | 0,0 | **JA** |
| `deflated_interval_includes_zero` | `DSR ≥ 0,95` | 9,879e-05 | **JA** |
| `sign_flips_on_gate_sample` | zelfde teken op W_GATE | `null` | niet beoordeelbaar |
| `breakeven_above_plausible_cost` | `≤ 6,5 bp` | `null` | niet beoordeelbaar |
| `effect_is_demonstrably_negative` | `ci_high < 0` | 0,0 | **nee** |
| `promotion_requires_all_gates_clear` | 0 bindende criteria | 2 | **nee** |

**2 van de 3 beoordeelbare criteria binden → `UNPROVEN`.**

Het tweede criterium verdient een aparte lezing. Bij `M_new = 25` en N = 1390
vraagt `DSR ≥ 0,95` een geannualiseerde Sharpe van **1,8686**. De
pre-registratie voorspelde dat dit criterium "vrijwel zeker bindend" zou zijn en
weigerde het te verzachten. Het bindt, met een DSR van 9,9e-05 op een cel die
−0,8055 meet. Dat is geen eigenschap van deze hypothese maar de rekenkundige stand
van het programma: **geen enkele cel van de vier-lagen-ladder haalt de helft van
die drempel.**

> **Correctie op §5.6 van de stapopdracht.** Die tabel geeft de DSR-eis bij M = 25
> als **1,74**. Dat getal hoort bij N = 1615 en is exact gereproduceerd (nagerekend:
> 1,7333). Op de ontwikkelsample die deze stap gebruikt (N = 1390) is de eis
> **1,8686** — hoger, omdat het kortere venster een grotere standaardfout heeft.
> De stapopdracht noemt die 1,8686 niet.

---

## 6. Waarom het poortsample NIET is gelezen

Substap 11.8 schrijft voor het poortsample eenmaal te lezen, "alleen voor de k met
het gunstigste resultaat op de ontwikkelsample". Dat is hier **niet uitgevoerd**,
en dat is de handeling die de pre-registratie voorschrijft in plaats van een
overgeslagen stap. Haar eigen `rationale` op criterium 1 zegt het letterlijk:

> Een punt onder nul sluit de vraag af voordat de poortsample wordt aangeraakt, en
> dat is precies de bedoeling van R-7.

Het gunstigste resultaat op de ontwikkelsample is `k = 2` met een verschil van
**exact nul**. Er is dus geen effect om naar de poort te brengen. Een lezing zou
de ene onherroepelijke lezing die R-7 toestaat, besteden aan een vraag die de
ontwikkelsample al heeft gesloten — en daarna is er voor deze hypothese geen poort
meer.

`holdout_lock.json` draagt na deze stap nog steeds `reads: []`. Er is geen aanroep
van `validation/holdout.py::gate_slice` in het pad dat dit artefact produceert;
uitsluitend `development_slice`, en dat mag herhaald.

---

## 7. Vier ordeningsbesluiten die het resultaat dragen

1. **Eerst snijden naar het bruikbare venster, dan vasthouden.** De blokken liggen
   positioneel vanaf de eerste **verhandelbare** bar. De beslisklok begint wanneer
   het handelen begint.
2. **Het gewichtspaneel wordt vastgehouden, niet de exposure binnen de ladder.**
   `backtest/phase5_baseline.py` is niet aangeraakt. Dat kost niets:
   `exposures_from_weights` herschaalt elke rij door haar eigen piek-absolute
   waarde en een vastgehouden blok heeft een constante piek, dus de twee operaties
   commuteren op L1/L2/L3. Het levert bovendien op dat **L0** de vasthoudoperatie
   ziet, wat nodig is omdat `mean_turnover` alleen daar wordt geproduceerd.
3. **De haltketen loopt ná het vasthouden.** Een halt is een risicobesluit en wordt
   niet vastgehouden; zou hij dat wel worden, dan overrulet de beslisfrequentie van
   L8 de soevereine risicolaag.
4. **De ladder draait over het volle bruikbare venster; de snede naar de
   ontwikkelsample gebeurt op de resultaatreeksen.** De L3-engine is
   padafhankelijk (equity, high-water mark, drawdown, en dus de haltbeslissing);
   een run die bij de split begint, meet een **ander** systeem in plaats van
   hetzelfde systeem over een korter venster. Omdat de blokken positioneel vanaf de
   eerste verhandelbare bar liggen, hangt de blokindeling van een ontwikkelbar
   nooit van een latere bar af — de volle-venster-run is causaal.

---

## 8. Afwijkingen van de nulmeting van de stapopdracht (R-10)

| grootheid | stapopdracht | gemeten |
|---|---:|---:|
| `n_sovereign_halted`, `long_only_equal_weight`/L3 | 1.559 | **1.566** |
| DSR-eis bij M = 25 | 1,74 (N=1615) | **1,8686** (N=1390) |

De 1.559 staat ook in `docs/MEASUREMENT_CONTRACT.md:287` en in drie docstrings in
`validation/inference.py`. Verschil: 7 bars. Niet gecorrigeerd in die bestanden —
dat raakt documenten en modules buiten de bestandenlijst van deze stap.

De "184 actieve bars" uit §5.2 van de stapopdracht is op de ontwikkelsample
**177** (`n_effective`, met `n_dropped = 1213`). Het verschil is het venster: 1.390
ontwikkelbars tegen de 1.743 van het volle venster.

---

## Bijlage — herleidbaarheid

**Trials: 4.** k ∈ {1, 2, 5, 10}, vooraf vastgelegd als `planned_trials: 4` in de
bevroren pre-registratie. De poort erop is
`trial_budget.assert_within_budget(4, reset_path=...)` tegen het bevroren
`M_new = 25`; `remaining(booked=4) = 21`. De diagnostiek van §3 op de drie
niet-primaire tracks kost **nul** trials: er is niets uit geselecteerd.

**Boekingsvorm, met een bevinding.** De ledger-entry draagt `n_trials=0` met
`amends`, en niet `n_trials=4`. `registry.ledger_reset.active_trial_count()` eist
dat de **lopende** ledger-teller gelijk blijft aan de bevroren
`archived_total = 2776`; een boeking met `n_trials=4` tilt hem naar 2780 en laat
die functie weigeren — en daarmee elke DSR in fase 10. Gemeten in een
sandbox-kopie van beide artefacten:

| boeking | `total_n_hypotheses()` | `active_trial_count()` |
|---|---:|---|
| nulstand | 2776 | OK (`M_new=25`) |
| `n_trials=4` | 2780 | **DataContractError** |
| `n_trials=0` + `amends` | 2776 | OK (`M_new=25`) |

De vaste vorm waarmee deze repository vóór fase 10 boekte — wave 30:
`n_trials=6, result=interim`, daarna een amendement met het oordeel — is met AD-24
dus onbruikbaar geworden. De controle is **niet** verzwakt: zij vergelijkt met de
lopende teller in plaats van met het archief zoals het op het moment van
bevriezen was, en dat is aantoonbaar te streng, maar een governance-controle
losser zetten omdat zij in de weg staat is de aanpassing die R-2 en R-10
verbieden. De keuze is een mandaatbesluit en ligt bij de eigenaar. Stap 12 en 13
lopen tegen hetzelfde aan.

De lopende-tellerrekening van de stapopdracht (`M_new` 6 → 10) heeft in deze
repository **geen representatie**: `M_new = 25` is een bevroren budget en geen
teller, er stonden nul post-reset trials geboekt, en de 6 is nooit geboekt.

**`result: archived` en niet `unproven`.** `hypothesis_ledger.py:37` laat alleen
`{accepted, archived, falsified, interim}` toe. `UNPROVEN` staat daarom in
`metrics.verdict` en in de `notes`, en `archived` is het dichtstbijzijnde
toegestane label — dezelfde keuze die wave 30 voor H1, H2 en H3 maakte. Dat het
register het vooraf geregistreerde oordeel niet letterlijk kan opnemen, is een
bekende beperking en geen ronding die hier stil is toegepast.

**Omzetdefinitie (R-3).** `backtest/vectorized.py:149` —
`(weights - weights.shift(1)).abs().sum(axis=1)`, tweezijdig en **niet**
gehalveerd, zoals `VectorizedResult.turnover` hem op `L0_vectorized` levert. Dit
is de reeks waarop de backtest de kosten in rekening brengt, dus de enige die
dimensioneel bij een breakevenniveau past. De tweede definitie in deze repository,
`portfolio/constraints.py:280` (`0,5 * sum(|Δw|)`, eenzijdig), dient de
limiethandhaving en is hier niet gebruikt. Er is geen derde bijgekomen. De
meetlaag toetst het gemiddelde van de meegegeven reeks tegen het L0-auditveld
`mean_turnover` — twee routes naar dezelfde omzet, zodat een reeks uit een andere
bron opvalt.

Reproductie:

```
python apps/run_h10_1_decision_frequency.py
python -m pytest tests/unit/test_decision_frequency.py \
                tests/unit/test_phase10_decision_frequency.py \
                tests/lookahead/test_decision_frequency_causality.py -q
```

Het JSON-artefact is de bron van elk getal in dit rapport. Het draagt bewust
**geen** `verdict`: het oordeel is een lezing van die getallen tegen de vooraf
vastgelegde beslisregel, en dat hoort in dit rapport te staan en niet in een
bestand dat een renderer kan overschrijven — dezelfde afweging die `dvc.yaml`
voor `reports/phase10_state_diagnostics.md` heeft gemaakt.

**Deze stap is om dezelfde reden geen DVC-stage.** `gate_slice` muteert
`holdout_lock.json`, dus een stage die de poort leest zou een van zijn eigen
`deps` wijzigen en `dvc repro` zou hem eeuwig opnieuw willen draaien. Dat de poort
hier niet is gelezen, verandert dat niet: de stap is per constructie eenmalig.
