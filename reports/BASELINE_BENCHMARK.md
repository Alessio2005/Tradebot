# BASELINE BENCHMARK — Level 1

> **De meetlat.** Elk model dat vanaf Phase 4 wordt voorgesteld, moet aantonen
> dat het deze cijfers verslaat **ná transactiekosten**, gemeten op hetzelfde
> venster en door dezelfde gates.
>
> **Laag:** L1 → L2 → L4 → L8 → L10 · **Bindend brondocument:**
> `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 9.1, 11, 11.1, 13.1, 16.1, 17,
> 17.1, 21, 22, 26

| Provenance | |
|---|---|
| `git_sha` | `bac4427` |
| `preregistration_id` | `086430701e325429ecc1a4f93d600edb` (bevroren op `6940293`, 2026-08-23) |
| `config_hash` | vastgelegd in de pre-registratie (`parameters`, ingespoten uit `conf/`) |
| **`M` (eerlijke trial count)** | **2.715** = 2.711 bij bevriezing + 4 geplande trials |
| Data | 6 gecertificeerde `crypto/ohlcv/*/1d`-reeksen uit `artefacts/governance/data_hashes.json` |
| Artefact | `artefacts/baseline/phase3_baseline.json` |
| Ledger-entry | wave 29, `phase3_baseline_result`, resultaat **`falsified`** |

---

## 1. De uitkomst in één alinea

**De pre-geregistreerde hypothese is gefalsificeerd.** Cross-Sectional Momentum
met EWMA-gebaseerde Naive Risk Parity levert over het gecertificeerde
crypto-universum een netto out-of-sample Sharpe van **−0,48**. Drie van de vier
toetsbare stop-criteria binden; promotie naar `CANDIDATE` is geblokkeerd. Het
resultaat is als zodanig in de ledger geregistreerd.

Dat is een geldig en bruikbaar resultaat, geen mislukking van de fase. De
meetlat staat er nu, en hij staat op een eerlijke hoogte in plaats van op een
opgepoetste.

---

## 2. Wat er precies is gemeten

### 2.1 De keten

```
PIT-store (18 gecertificeerde reeksen, waarvan 6 gebruikt)
  → L1  close-panel op asof_ts + data_hash per symbool
  → L1  rolling_log_return(window=60, skip=1)  →  cross_sectional_rank(min_assets=3)
  → L2  EWMA-volatiliteit (λ = 0.94, causale seed, burn-in 60 bars)
  → L4  CrossSectionalMomentum  →  a_t ∈ [−1, +1], dollar-neutraal per bar
  → L8  allocator (risk parity of equal weight), genormaliseerd op gross = 1
  → L10 w.shift(1) · r_t, min turnover-kosten
```

Alle parameters komen uit `conf/model/alpha.yaml`, `conf/model/volatility.yaml`,
`conf/validation/default.yaml` en `conf/execution/fees.yaml`. Er staat geen
enkele vensterlengte of drempel als literal in `src/`.

### 2.2 Vensters

| | |
|---|---|
| Gecertificeerde reeks | 2020-03-26 → 2026-08-23 (2.342 bars) |
| Warmup (burn-in van de keten) | 61 bars |
| Verhandelbaar na warmup | 2.281 bars |
| Pre-geregistreerde `evaluation_start` | 2021-05-16 |
| **Gemeten OOS-venster** | **2021-10-13 → 2026-06-03** |
| OOS-bars | 1.615 |
| Walk-forward folds | 17 (config eist minimaal `n_splits = 6`) |
| Bars weggevallen op de embargo | 85 (17 × 5) |

**Het OOS-venster begint later dan `evaluation_start`, en dat is een afwijking
die hier expliciet wordt vermeld.** De pre-registratie legt vast vanaf wanneer
de strategie *kan* handelen (het eerste punt waarop drie symbolen een geldig
signaal én een geldige vol hebben). Purged Walk-Forward *meet* pas na het eerste
trainvenster van 500 bars plus embargo. Het venster eindigt op 2026-06-03 omdat
de laatste, onvolledige fold vervalt. Beide getallen staan hierboven; het OOS-
getal is het getal dat telt.

### 2.3 Kostenaanname — VOORLOPIG

| Component | Waarde | Bron |
|---|---:|---|
| Taker fee | 5,5 bps | `conf/execution/fees.yaml` (Bybit VIP-0 linear perps) |
| Half-spread | 1,0 bps | conservatieve aanname |
| **Totaal per eenheid turnover** | **6,5 bps** | |
| `cost_assumption_is_provisional` | `true` | |

De definitieve η-kalibratie op orderboekdata volgt in **Phase 5**. Tot die tijd
staat elk getal hieronder **bruto én netto**.

---

## 3. Resultaten

Alle vier de tracks draaien op exact dezelfde 1.615 OOS-bars en dezelfde
gross-exposure van 1,0, zodat de vergelijking over sizing gaat en niet over
hefboom.

| Track | Bruto Sharpe | **Netto Sharpe** | Max drawdown | Calmar | Turnover/dag | Netto totaalrendement |
|---|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` (1/N) | 0,156 | **0,156** | 0,835 | −0,167 | 0,0002 | −48,9 % |
| `long_only_risk_parity` | 0,135 | **0,129** | 0,822 | −0,161 | 0,017 | −47,3 % |
| `xs_momentum_equal_weight` | −0,041 | **−0,181** | 0,508 | −0,164 | 0,155 | −30,6 % |
| **`xs_momentum_risk_parity`** ← de hypothese | −0,284 | **−0,483** | 0,563 | −0,212 | 0,176 | −42,1 % |

### 3.1 De Sharpe is positief en het rendement is negatief — dat is geen fout

| Track | Ann. rekenkundig | Ann. volatiliteit | CAGR (meetkundig) | Variantie-drag |
|---|---:|---:|---:|---:|
| `long_only_equal_weight` | +11,3 % | 72,4 % | **−14,1 %** | 25,4 pp |
| `long_only_risk_parity` | +8,8 % | 67,8 % | **−13,5 %** | 22,3 pp |
| `xs_momentum_equal_weight` | −4,8 % | 26,4 % | −7,9 % | 3,2 pp |
| `xs_momentum_risk_parity` | −10,1 % | 21,0 % | −11,6 % | 1,5 pp |

De Sharpe gebruikt het **rekenkundige** gemiddelde; kapitaal groeit
**meetkundig**. Bij een geannualiseerde volatiliteit van 72 % is het verschil
ongeveer σ²/2 ≈ 26 procentpunt — precies de gemeten drag. Een 1/N-mandje crypto
met een Sharpe van +0,16 verloor over dit venster dus toch bijna de helft van
zijn kapitaal.

**Consequentie voor de gates.** De Sharpe is de pre-geregistreerde primaire
metriek en blijft dat; hij achteraf vervangen zou de pre-registratie waardeloos
maken. Maar een Sharpe-gate is op ongeschaalde crypto-exposure aantoonbaar
**geen** proxy voor kapitaalgroei. Dat is een argument vóór de
volatility-targeting-laag van Phase 4, niet ertegen — en het is precies het
soort bevinding waarvoor een baseline bestaat.

### 3.2 Wat de vergelijking laat zien

* **Risk parity verlaagt de drawdown, niet het verlies.** Tegenover
  equal weight: max drawdown van 0,835 naar 0,822 (long-only) en van 0,508 naar
  0,563 (momentum — hier dus juist hóger). Op deze data voegt de vol-schatting
  in de sizing geen meetbare waarde toe.
* **De momentum-tracks verliezen bruto al.** Bruto Sharpe −0,28 respectievelijk
  −0,04: het signaal heeft in dit venster geen edge, nog vóór er ook maar één
  bps aan kosten van af gaat. Kosten maken het erger (−0,28 → −0,48), maar ze
  zijn niet de oorzaak.
* **De kosten zijn wel substantieel.** Bij een turnover van 0,176 per dag en
  6,5 bps per eenheid kost de momentum-track ongeveer 4,2 % per jaar. Dat is
  ruim tweemaal het verschil tussen de twee allocators.

---

## 4. Statistische gates

### 4.1 Deflated Sharpe Ratio — met de eerlijke `M`

`M = 2.715`. Dat is de cumulatieve teller uit `artefacts/governance/hypothesis_ledger.json`
op het moment van **bevriezen** (2.711), plus de 4 vooraf getelde trials van
deze golf. Live uitlezen zou de resultaat-entry van de meting zichzelf laten
meetellen, waardoor een tweede run een andere DSR geeft.

| Track | Per-bar Sharpe | DSR-waarschijnlijkheid | Vereist | Slaagt |
|---|---:|---:|---:|---|
| `long_only_equal_weight` | +0,0082 | 6,9 × 10⁻⁴ | ≥ 0,95 | nee |
| `long_only_risk_parity` | +0,0068 | 5,6 × 10⁻⁴ | ≥ 0,95 | nee |
| `xs_momentum_equal_weight` | −0,0094 | 4,6 × 10⁻⁵ | ≥ 0,95 | nee |
| `xs_momentum_risk_parity` | −0,0253 | **2,7 × 10⁻⁶** | ≥ 0,95 | **nee** |

Geen enkele track komt in de buurt. Bij `M = 2.715` is de verwachte beste
Sharpe onder de nulhypothese aanzienlijk, en een negatieve gerealiseerde Sharpe
haalt die grens per definitie niet.

### 4.2 Hansen's SPA — geen superioriteit boven 1/N

| | |
|---|---|
| Benchmark | `long_only_equal_weight` (sectie 13.1) |
| Kandidaten | de overige drie tracks |
| Beste kandidaat | `long_only_risk_parity` |
| **SPA p-waarde** | **1,00** |
| Verwerpt de nulhypothese op α = 0,05 | **nee** |

Geen van de kandidaten verslaat de 1/N-referentie onder
multiple-testing-correctie. De p-waarde van 1,00 is geen randgeval: geen enkele
kandidaat heeft zelfs een positief gemiddeld verschil met de benchmark.

### 4.3 De pre-geregistreerde stop-criteria

| Criterium | Regel | Gemeten | Status | Actie |
|---|---|---:|---|---|
| `no_edge_after_costs` | `net_oos_sharpe ≤ 0,0` | −0,483 | **BINDING** | `falsify` |
| `dsr_indistinguishable_from_selection_noise` | `dsr_probability < 0,95` | 2,7 × 10⁻⁶ | **BINDING** | `archive` |
| `spa_no_superiority_over_benchmark` | `spa_p_value ≥ 0,05` | 1,00 | **BINDING** | `archive` |
| `costs_dominate_gross_edge` | `cost_drag_fraction > 0,5` | — | `NOT_APPLICABLE` | — |
| `promotion_requires_all_gates_clear` | `n_binding ≤ 0` | 3 | **BINDING** | promotie geblokkeerd |

`costs_dominate_gross_edge` is **niet te beoordelen**: de fractie
`(bruto − netto) / bruto` vereist een positieve bruto-edge om te eroderen, en
die is er niet (bruto Sharpe −0,284). Het criterium wordt daarom expliciet als
`NOT_APPLICABLE` gerapporteerd en **niet** als geslaagd geteld. `no_edge_after_costs`
bindt hier sowieso al en gaat voor.

---

## 5. Wat dit NIET zegt

Deze grenzen horen bij elke latere verwijzing naar dit rapport.

1. **Dit falsificeert niet "crypto momentum".** Het falsificeert *deze*
   configuratie — 60 bars lookback, 1 bar skip, dagelijks herbalanceren,
   6 symbolen, dit venster. De pre-registratie stond precies één configuratie
   toe, en dat is opzet: elke variant die je erbij zoekt tot er één werkt,
   verhoogt `M` en verlaagt de DSR van álle varianten.
2. **Het venster is ongunstig en kort.** 1.615 bars ≈ 4,4 jaar, en het bevat de
   bear van 2022. Zes assets is bovendien een dunne cross-sectie: een rang over
   drie tot zes namen heeft weinig ruimte om te discrimineren.
3. **Survivorship bias staat open (DI-15).** Het universum bestaat uit zes
   ex-post gekozen overlevers. Dat maakt de long-only referenties eerder te
   gunstig dan te ongunstig, en het maakt dit resultaat dus zo mogelijk nóg
   minder rooskleurig dan het lijkt. Deze vermelding is verplicht bij elke
   baseline-claim tot de bron met delisting-historie er is.
4. **De kosten zijn voorlopig.** Zie §2.3. De richting van de fout is echter
   bekend: 6,5 bps per eenheid turnover is conservatief laag voor crypto perps
   met marktorders, dus de netto-cijfers zijn eerder te gunstig.
5. **Walk-forward beschermt hier tegen niets.** De baseline fit niets, dus
   elke bar na de burn-in was al out-of-sample. Het schema is toegepast zodat
   een later model op precies dezelfde bars wordt gemeten — niet omdat het hier
   een overfit-risico wegneemt.

---

## 6. Reproductie

```bash
git checkout bac4427
python apps/run_baseline.py --preregistration-id 086430701e325429ecc1a4f93d600edb --wave 29
```

Twee runs op dezelfde `data_hash` leveren bit-identieke cijfers: `M` komt uit de
bevriezing en niet uit de live ledger, de SPA-bootstrap draait op een vaste
`block_length` uit `conf/validation/`, en de hele keten is truncatie-invariant
(33 tests in `tests/lookahead/test_baseline_causality.py`).

---

## 7. De meetlat voor Phase 4 tot en met 7

Een model dat deze baseline wil verslaan, moet **alle vier** aantonen:

| # | Eis | Deze baseline |
|---|---|---:|
| 1 | Netto OOS Sharpe > 0 op het OOS-venster | −0,483 |
| 2 | Netto OOS Sharpe > die van 1/N | 0,156 |
| 3 | DSR-waarschijnlijkheid ≥ 0,95 bij de dan geldende `M` | 2,7 × 10⁻⁶ |
| 4 | SPA p < 0,05 tegen de 1/N-referentie | 1,00 |

En daarbij: gemeten op hetzelfde OOS-venster, met dezelfde kostenaanname, en
met een `M` die élke gedraaide variant telt — ook de weggegooide.

**Waarschuwing bij eis 1 en 2.** Zoals §3.1 laat zien, is een positieve Sharpe
op ongeschaalde crypto-exposure verenigbaar met een verlies van bijna de helft
van het kapitaal. Phase 4 moet daarom naast de Sharpe ook de **meetkundige**
uitkomst rapporteren; de vol-targeting-laag die daar wordt gebouwd, is precies
het instrument dat het gat tussen die twee verkleint.
