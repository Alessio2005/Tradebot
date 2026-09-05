# MEETCONTRACT

> **Fase:** 10, stap 1.5 · **Status:** bindend · **Bewaakt door:** `docs/MANDATE.md` (B-2), AD-22, AD-23
> **Deliverable:** D2 · **Exit-criteria:** 2, 3, 11, 25
>
> **Geverifieerd tegen de codebase op 2026-09-05** (fase 10, stap 1). Alle
> vensterwaarden in §2 zijn op die datum gemeten tegen de gecertificeerde
> PIT-store, niet overgenomen uit eerdere documenten.

Dit document is de enige plaats waar de meetconventies van dit programma staan.
Elke stap in fase 10 **verwijst** hiernaar en herhaalt de conventie niet. Eén
plek, één waarheid: een tweede formulering is een defect, ook wanneer zij
hetzelfde zegt.

Wat hier niet in staat, is geen conventie maar een keuze die per meting moet
worden verantwoord.

---

## 1. Annualisatie

```
bars_per_year = 365
```

De markt is continu: er is geen weekend, geen handelspauze en geen feestdag in
de perpetual-futuresmarkt. Een handelsdagconventie (252) zou een kalenderjaar
met 113 dagen inkorten en elke geannualiseerde grootheid met een factor
`sqrt(365/252) = 1,204` opblazen.

Deze waarde staat in `conf/backtest/default.yaml` (`bars_per_year: 365.0`) en in
`conf/model/volatility.yaml` (`annualisation_factor: 365.0`). Zij is daar de
bron; dit document beschrijft haar, het definieert haar niet opnieuw.

---

## 2. Het meetvenster

### 2.1 De ene definitie

| | |
|---|---|
| **venster-id** | `W_FULL` |
| `period_start` | **2021-11-15** |
| `period_end` | **2026-08-23** |
| `n_obs` | **1743** |
| `bars_per_year` | 365 |
| `t_years` | **4,7753** (= 1743 / 365) |
| observatiefrequentie | 1 bar per dag |
| universum | 6 gecertificeerde USDT-perpetuals: BTC, ETH, SOL, AVAX, LINK, DOT |

`W_FULL` is de maximale **aaneengesloten** kalenderspanne waarop alle zes
gecertificeerde symbolen een gedefinieerde dagbar, een EWMA-σ̂ en een causale
30-bars ADV hebben. De span telt 1743 kalenderdagen inclusief en bevat 1743
bars: er zit geen gat in.

De constructie, gemeten en niet aangenomen:

| stap | bars | verlies | waarom |
|---|---:|---:|---|
| ruwe gecertificeerde paneelindex | 2342 | — | 2020-03-26 → 2026-08-23, ragged: alleen BTCUSDT dekt dit |
| alle zes symbolen aanwezig | 1773 | 569 | SOLUSDT begint als laatste, op 2021-10-16 |
| + EWMA-σ̂ gedefinieerd (`burn_in_bars: 60`) | 1773 | 0 | burn-in valt binnen de ragged aanloop |
| + causale ADV (`rolling(30).shift(1)`) | **1743** | 30 | 30 bars ADV-venster vanaf 2021-10-16 → 2021-11-15 |

### 2.2 Ruimte voor een tweede venster

Stap 4B splitst `W_FULL` in een ontwikkelvenster en een bevroren poortvenster.
Die splitsing wordt hier toegevoegd als extra rijen; `W_FULL` zelf verandert
daarbij **niet**, want zij is de unie van de twee.

| venster-id | period_start | period_end | n_obs | t_years | status |
|---|---|---|---:|---:|---|
| `W_FULL` | 2021-11-15 | 2026-08-23 | 1743 | 4,7753 | vastgelegd (dit document) |
| `W_DEV` | — | — | — | — | **te vullen in stap 4B** |
| `W_GATE` | — | — | — | — | **te vullen in stap 4B**, bevroren in `holdout_lock.json` |

### 2.3 Symbolen met kortere historie

`W_FULL` is per constructie gebalanceerd: binnen het venster heeft elk van de
zes namen elke bar. Er wordt **niet** opgevuld, niet geïnterpoleerd en niet
achterwaarts geëxtrapoleerd.

De prijs daarvan is zichtbaar en geboekt: 569 bars BTC-historie en de volledige
2020-2021-periode vallen buiten het venster. Het ongebalanceerde paneel — alle
2342 bars, met per bar wisselend aantal namen — is een **aparte hypothese**
(H-10.2, stap 12) en geen variant van dit venster. Wie het ongebalanceerde
paneel meet, meet niet op `W_FULL` en zegt dat erbij.

### 2.4 Reconciliatie van de drie vensterwaarden (Q10)

De fasetekst noemde drie waarden die naast elkaar zouden staan: `N = 1615`
(4,42 j), `T = 4,78 j` (1745 bars) en `1743` bars in de halttabel. Gemeten op de
gecertificeerde store zijn dat **geen drie vensters**:

| waarde | wat het werkelijk is | verdict |
|---|---|---|
| **1743** | `W_FULL` — de gebalanceerde, aaneengesloten zesnaamsspanne | **gekozen als het meetvenster** |
| **1742** | `W_FULL` minus de ene bar die de rendementsoperator opeet | afgeleid; label als `n_obs` van een rendementsreeks |
| **4,78 j / 1745 bars** | `W_FULL` zelf, afgerond en verkeerd overgeschreven: 1743/365 = **4,7753 ≈ 4,78**. Geen enkel artefact bevat 1745 | **ingetrokken** — het is `W_FULL`, niet een tweede venster |
| **1615** | **geen venster maar een steekproef**: de aaneengeregen out-of-sample-bars van de purged walk-forward in `phase3_baseline.json` / `phase4_stress.json` | behouden als **afgeleide steekproef**, met eigen `n_obs` |

Waarom 1615 geen venster kan zijn, gemeten:

* De 1615 bars lopen van **2021-10-13 tot 2026-06-03**, een kalenderspanne van
  1695 dagen. Er zitten **80 gaten** in: 17 trainblokken en 85 embargobars uit
  `_oos_mask(train_bars=500, test_bars=100, embargo_bars=5, min_splits=6)`.
  Een venster met gaten is een masker.
* Het getal is **niet stabiel**. Het verandert zodra de foldgeometrie, de
  embargolengte of de burn-in van de alpha-unit verandert — geen daarvan is een
  eigenschap van de meetbasis. Een contract dat op zo'n getal steunt, verandert
  mee met de strategie die het hoort te beoordelen.
* Het is een masker over het **ragged** 2342-barspaneel, niet over `W_FULL`.

Waarom 1743 het wel is: het is de enige van de vier die een aaneengesloten
kalenderspanne met alle zes gecertificeerde namen beschrijft, het volgt
uitsluitend uit data plus twee in `conf/` vastgelegde burn-ins, en het is al de
gedocumenteerde meetspanne in `docs/PROJECT_STATE.md`,
`reports/ECONOMETRIC_DIAGNOSTICS.md`, `reports/GARCH_VS_EWMA_COMPETITION.md`,
`reports/M0_VS_HMM_BENCHMARK.md`, `reports/META_LABELING_EVALUATION.md`,
`reports/phase5_cluster_concentration_audit.md`, `reports/phase5_exit_report.md`
en `reports/phase7_divergence_map.md`.

### 2.5 De afgeleide drempels

De t = 2-drempel is `2 / sqrt(t_years)` met `t_years = n_obs / 365` van de
steekproef waarop de Sharpe **daadwerkelijk** is gemeten — niet van het venster
waaruit die steekproef komt. Dat onderscheid is de hele reden dat dit hoofdstuk
bestaat.

| steekproef | n_obs | t_years | ann. Sharpe voor t = 2 | SE(ann. Sharpe) = 1/√t_years | NW-lags q | embargo (≥ 1 %) |
|---|---:|---:|---:|---:|---:|---:|
| `W_FULL` | 1743 | 4,7753 | **0,9152** | 0,4576 | 7 | **18 bars** |
| phase3/phase4 OOS-steekproef | 1615 | 4,4247 | **0,9508** | 0,4754 | 7 | 17 bars |

> **De correctie die revisie 2 aanbracht, preciezer.** Revisie 2 stelt dat
> 0,915 fout is en 0,951 juist. Gemeten klopt dat alleen als je de
> phase3-steekproef bedoelt. **0,9152 is de juiste drempel voor `W_FULL`** — het
> venster waarop de vier-lagen-ladder, de toestandstabel en elk rapport in
> `reports/` zijn gemeten. **0,9508 is de juiste drempel voor de
> phase3/phase4-steekproef** van 1615 OOS-bars. Beide getallen zijn goed; wat
> fout was, is ze door elkaar gebruiken en 0,915 aan een niet-bestaand
> 1745-barsvenster toeschrijven. Elke gerapporteerde Sharpe draagt daarom zijn
> eigen `n_obs` (§1 en §3.1 van de fasetekst), en de drempel wordt dááruit
> berekend.

> **Reikwijdte van de 1615-rij — dit is geen tweede standaard.** Zij bestaat
> voor één doel: het correct **teruglezen** van de bestaande
> phase3/phase4-artefacten (`artefacts/baseline/phase3_baseline.json`,
> `artefacts/risk/phase4_stress.json`), waarvan de Sharpes nu eenmaal op die
> OOS-steekproef zijn gemeten. Zij is **geen** drempel die een nieuwe meting mag
> kiezen. Elke meting in fase 10 draait op `W_FULL` en rapporteert haar eigen
> `n_obs`; wie 0,9508 op een nieuwe meting toepast, kiest de ruimere drempel
> zonder de steekproef te bezitten die haar rechtvaardigt.
>
> **Consequentie voor §5.6 van de fasetekst.** De tabel "DSR-drempel als functie
> van M" is berekend op **N = 1615** — verifieerbaar aan haar M = 2776-cel, die
> 2,47 luidt en daarmee exact de 1615-waarde is. Die tabel moet **op `W_FULL`
> worden herberekend voordat een latere stap haar gebruikt**: op 1743 bars ligt
> de M = 2776-drempel op **2,380**, niet op 2,474, en elke andere cel schuift
> mee. Stap 4A voert die herberekening uit, met de handtekening uit §6.

De DSR-drempel volgt dezelfde regel. Ter referentie, met normale momenten en de
implementatie van `backtest/metrics.py` zoals zij vóór stap 4A is:

| steekproef | n_obs | vereiste ann. Sharpe voor DSR ≥ 0,95 bij M = 2776 |
|---|---:|---:|
| `W_FULL` | 1743 | **2,380** |
| phase3/phase4 OOS-steekproef | 1615 | **2,474** |

Stap 4A herberekent deze kolom met de expliciete handtekening uit §6 hieronder;
tot dan zijn dit referentiewaarden onder de normale benadering, en dat feit
hoort in het artefact te staan en niet in iemands hoofd.

---

## 3. De standaardfout van een Sharpe

Niet `1/sqrt(T)`. Die uitdrukking veronderstelt i.i.d. normale rendementen;
dagelijkse crypto-P&L is scheef, dik-staartig én autogecorreleerd, en de drie
effecten werken niet dezelfde kant op.

Verplicht is de Lo (2002)-vorm met HAC-correctie:

```
SE(SR_hat) = sqrt( (1 + 0.5*SR^2 - g3*SR + ((g4 - 3)/4)*SR^2) / T ) * sqrt(eta_q)
```

met `g3` de scheefheid, `g4` de kurtosis, en `eta_q` de Newey-West-opslag over

```
q = floor( 4 * (T/100)^(2/9) )
```

Voor `T = 1743` is dat **q = 7**; voor `T = 1615` eveneens q = 7. De laglengte
is dus niet gevoelig voor de vensterkeuze van §2, maar wordt wel per steekproef
uit haar eigen `T` berekend en niet overgenomen.

De naïeve `1/sqrt(T)`-variant mag als **referentiekolom** worden
meegerapporteerd, nooit als de toets.

Eén implementatie, in `validation/inference.py` (stap 4A). Een tweede is een
defect, ook wanneer zij hetzelfde uitrekent (R-3).

---

## 4. Het verschil tussen twee Sharpes

Een Sharpe-verschil is **geen** gemiddeld rendementsverschil, en een gepaarde
t-toets op rendementsverschillen toetst de verkeerde grootheid.

De toets is **Ledoit-Wolf (2008)**, *Robust performance hypothesis testing with
the Sharpe ratio*: de HAC-geschatte covariantie van de vier momenten,
gestudentiseerd met een circulaire blokbootstrap (Politis-Romano), met

```
n_boot = 10_000
block_length = automatisch gekalibreerd
seed = geregistreerd (R-5: gelijke cfg + seed => bit-identieke output)
```

Dit geldt voor **elke** vergelijking in deze fase: de laagovergangen (stap 10),
EWMA tegen GARCH op de toestandsas (stap 8), geconditioneerd tegen
ongeconditioneerd (stap 13) en de panelen in stap 12.

**Uitlijning is onderdeel van de toets.** Een gepaarde vergelijking loopt
uitsluitend over bars waarop **beide** ketens actief zijn, en het aantal van die
bars hoort in de tabel. Dit is niet cosmetisch: `long_only_equal_weight`
halteert op 2022-05-10 en staat **1559 van de 1743 bars** gehalteerd, zodat de
L1-en-hoger-Sharpe over ongeveer 184 actieve bars (0,50 jaar) is berekend. De
t = 2-drempel op 0,50 jaar is een annualiseerde Sharpe van **2,83**. Een gepaard
verschil over verschillende actieve verzamelingen is geen gepaard verschil.

---

## 5. Paneelafhankelijkheid: clustering en N_eff-deflatie

Gemeten op `data/pit_store/`: **ρ̄ = 0,7442** over zes namen, **N_eff = 1,271**.
Een gepoolde toets over symbool-bars behandelt ~10.400 observaties als
onafhankelijk, terwijl er effectief ongeveer 1733 dagen × 1,271 namen aan
informatie in zit.

Twee verplichtingen, beide bij **elke** gepoolde grootheid:

1. **Cluster op datum.** Elke gepoolde standaardfout wordt geclusterd op de
   tijdsindex, niet op de symbool-bar.
2. **Rapporteer de deflatie.** Naast de gepoolde t staat haar
   N_eff-gedefleerde tegenhanger:

   ```
   t_eff = t_pooled * sqrt(N_eff / N) = t_pooled * 0.460
   ```

Een gepoolde t zonder zijn gedefleerde tegenhanger is geen bevinding (R-8).

---

## 6. De DSR-handtekening

`backtest/metrics.py::deflated_sharpe` krijgt in stap 4A de handtekening die de
statistiek vereist:

```python
deflated_sharpe(
    sr_hat, *, n_obs, n_trials, sr_variance, skew, kurtosis, bars_per_year
) -> DSRResult
```

met

```
SR_0 = sqrt(sr_variance) * [ (1 - gamma)*Phi^-1(1 - 1/M)
                             + gamma*Phi^-1(1 - 1/(M*e)) ]
```

`gamma` de Euler-Mascheroni-constante, en

```
DSR = Phi( (SR_hat - SR_0)*sqrt(T - 1)
           / sqrt(1 - g3*SR_hat + ((g4 - 1)/4)*SR_hat^2) )
```

`sr_variance` is de **empirische** variantie van de trial-Sharpes wanneer die
beschikbaar is, en anders de gedocumenteerde benadering `1/n_obs` — met dát
feit in het artefact.

**De familiecorrectie loopt uitsluitend via `M`.** Er komt geen tweede laag
overheen: geen Benjamini-Hochberg, geen Bonferroni. Twee correcties over
dezelfde familie zijn niet conservatiever maar onbepaald — de gecombineerde
grootte is niet meer te herleiden. Wie strenger wil zijn, verhoogt `M`.

---

## 7. De rendementsconventie

* Het instrument is de **perpetual future**, niet spot.
* De P&L is **`prijsrendement + funding`**, waarbij funding met het **teken van
  de positie** meeloopt en op de dagbar wordt geaccumuleerd.
* Kosten zijn `fee + spread/2 + impact`, met de impactparameters uit
  `execution/impact_model.py` (`eta = 2,991922`) en de labels AD-2
  (`IMPACT_UNCALIBRATED`) en AD-3 (`SPREAD_ASSUMED`), die per stap 17
  **permanent** worden.
* F9 falsificeert shorts op dit universum onvoorwaardelijk. Long-only is
  daarmee geen keuze maar een gegeven, en de fundingcomponent heeft dus één
  teken.

> **Een backtest die funding niet expliciet boekt, is op perpetuals geen
> backtest.** `backtest/accounting.py` boekt funding correct — met het teken van
> de positie, en met beide invarianten sluitend (geverifieerd in stap 1.7,
> `tests/unit/test_accounting.py`). De **aanvoer** was echter niet
> contractconform: `apps/run_phase5_baseline.py` leverde een fundingpaneel van
> nul aan `run_all_layers`, terwijl alle zes reeksen `crypto/funding/<symbool>/8h`
> hash-gecertificeerd aanwezig zijn — bedrading, geen databeperking.
>
> **Gesloten in stap 1B.** `data/funding_panel.py::daily_funding_panel` laadt de
> gecertificeerde 8h-reeks via `load_certified_series` en SOMMEERT de
> afrekeningen per dagbar (niet de laatst bekende rate via een asof-join, die
> ongeveer een derde van de werkelijke funding had geboekt — zie het
> stap-1B-rapport voor de valkuil). `apps/run_phase5_baseline.py` geeft dat
> paneel nu door in plaats van de nul-DataFrame. Gemeten op de gecertificeerde
> store: gemiddelde `|dagelijkse funding|` per symbool ligt tussen 2,6e-4 en
> 5,5e-4 (orde 1e-4, zoals verwacht voor een 8h-perp gesommeerd naar dagen).
> Alleen **L3** boekt funding — L0/L1/L2 zijn bit-identiek aan de prijs-only
> meting, geverifieerd per track. `cost_funding` op L3 ging van `0,0` naar
> 261,7 / 279,7 / 761,4 / 397,4 (long_only_equal_weight /
> long_only_risk_parity / xs_momentum_equal_weight / xs_momentum_risk_parity;
> in equity-eenheden op `initial_equity`), en de L3-netto-Sharpe verschoof met
> -0,024 / -0,024 / -0,065 / -0,038 (steeds negatiever: funding was en blijft
> een kostenpost op dit long-only boek). De cap `funding_cap_abs = 0,02` bond 8
> keer op 22.892 boekingen (0,035 %), veroorzaakt door enkele extreme
> fundingpieken (SOLUSDT tot -0,124 op één dag) en niet door een eenhedenfout.
> Zie het stap-1B-rapport voor de volledige tabel en de per-track cijfers.

---

## 8. Geen alfaclaim zonder residuele t

Zes namen met ρ̄ = 0,7442 vormen geen cross-sectie maar een gerichte weddenschap
op één factor. Daarom:

> **Geen enkele uitkomst in deze fase mag "alpha" worden genoemd voordat het
> rendement is geregresseerd op (a) het BTC-rendement en (b) het
> gelijkgewogen universumrendement, en de residuele alfa met haar HAC-t is
> gerapporteerd.**

De bruto Sharpe is een **beschrijving**. De residuele alfa is de **claim**. Het
verschil hoort in elk rapport te staan, ook — en juist — wanneer het de
conclusie omdraait.

---

## 9. Purge en embargo

Elke walk-forward- of CV-structuur in deze fase draagt:

* een **purge** gelijk aan de maximale featurelookback, en
* een **embargo** van minimaal 1 % van de steekproef.

Op `W_FULL` (n_obs = 1743) is dat **≥ 18 bars**. Op de phase3/phase4
OOS-steekproef (n_obs = 1615) was het ≥ 17 bars; de bestaande configuratie
(`conf/validation/default.yaml`, `embargo_bars: 5`) ligt onder beide en is
daarmee een openstaand punt voor de stappen die nieuwe folds bouwen.

Dit is bestaand beleid in `validation/`; het staat hier omdat de nieuwe
toestandsmodules er ook onder vallen en dat nergens stond.

---

## 10. Wat dit contract van elke meting eist

Een samenvatting in één regel. Let op de werkwoordstijd: dit is een **eis aan
de code die nog gebouwd moet worden**, geen beschrijving van een bestaande
poort.

> Elke gerapporteerde Sharpe draagt onlosmakelijk het drietal
> **`(n_obs, bars_per_year, t_years)`** met `t_years = n_obs / bars_per_year`,
> plus een SE volgens §3. Een Sharpe zonder dat drietal is geen getal maar een
> gerucht, en de serializer **moet** hem weigeren.

**Die weigering bestaat op dit moment niet.** `backtest/metrics.py` kent geen
serialisatiepoort die het drietal afdwingt; niets in de huidige code houdt een
Sharpe zonder `(n_obs, bars_per_year, t_years)` tegen. **Stap 4A bouwt haar**,
samen met de expliciete DSR-handtekening uit §6 en de ene implementatie van
Sharpe-SE, DSR en Sharpe-verschil in `validation/inference.py` (R-3).

Tot stap 4A rust deze regel dus op de auteur van elke meting en niet op een
poort. Wie hem in de tussentijd citeert als "de serializer weigert dat", citeert
een garantie die er nog niet is.

### 10.1 Openstaand defect voor stap 4A — de annualisatie-default

`src/tradebot/backtest/metrics.py:53` luidt:

```python
_BARS_PER_YEAR_DEFAULT = 365 * 24  # hourly bars; override via bars_per_year kwarg
```

De default is **8760** — uurbars. Onder AD-22 is elke bar een dagbar, en §1 van
dit contract eist **365**. Elke aanroeper die de `bars_per_year`-kwarg weglaat,
annualiseert daarom met 8760 in plaats van 365 en verschaalt zijn Sharpe met een
factor `sqrt(8760 / 365) = sqrt(24)` ≈ **4,9**.

Dit is een **openstaand defect, toegewezen aan stap 4A**, die `metrics.py`
herschrijft. Het wordt hier bewust **niet** gerepareerd: de default los
omzetten zou de uitkomst van elke bestaande aanroeper stil van waarde laten
veranderen, zonder dat de nieuwe handtekening uit §6 er al is om die verandering
zichtbaar te maken.

Tot stap 4A geldt daarom als contractregel: **geen enkele aanroep van
`metrics.py` mag `bars_per_year` weglaten.** Een Sharpe die op de default leunt,
is per dit contract ongeldig — en is bovendien vrijwel zeker een factor 4,9 te
hoog.

**En er is een derde annualisatie in omloop.** `src/tradebot/backtest/pbo.py:105`
definieert een eigen `_sharpe_ratio(returns, annualization=252.0)` — handelsdagen
— en `pbo.py:36` gebruikt die functie als de default-metriek van de
PBO-berekening. Daarmee staan er drie annualisaties naast elkaar in één
repository: **365** (dit contract en `conf/`), **8760** (de default van
`metrics.py`) en **252** (`pbo.py`). Dat is tegelijk een annualisatiedefect en
een schending van R-3 — één implementatie per statistische grootheid. Ook dit
gaat naar **stap 4A**, die Sharpe-SE, DSR en Sharpe-verschil samenbrengt in
`validation/inference.py`; het wordt hier niet gerepareerd.
