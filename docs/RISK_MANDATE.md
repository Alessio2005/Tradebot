# RISK MANDATE — van propfirm-compliance naar eigen kapitaal

> **Status:** bindend vanaf de `git_sha` van deze commit
> **Vervangt:** het accountprofiel uit `src/tradebot/artefacts/PROPFIRM_PARALLEL_AUDIT.md`
> (statische max-DD ~10 % + daily-loss ~5 %, drie accounts à ~$100k)
> **Raakt niet:** `docs/RISK_CONTRACT.md`. Dat is het *interfacecontract* van L7 en
> verandert hier met geen letter. Dit document gaat uitsluitend over de
> **getallen** in `conf/risk/default.yaml`.
> **Bewaakt door:** `tests/unit/test_config_contracts.py::TestRiskBudgetIsComplete`

---

## 0. Het besluit

Er wordt niet meer met propfirms gewerkt. Daarmee verdwijnt de enige reden
waarom een deel van het risicobudget zo strak stond: die drempels waren geen
risico-oordeel maar **contractnaleving**. Een propfirm-account gaat dood bij een
overschrijding, ongeacht of de positie verstandig was. Dat maakt "ver onder de
firmalijn blijven" rationeel zolang het contract bestaat — en zinloos zodra het
weg is.

Op eigen kapitaal is er geen tegenpartij die de rekening opzegt. Wat overblijft
is één echte grens: **ruïne**. Daar hoort een ander getal bij.

**Wat dit besluit NIET doet, en wat niemand er in mag lezen:**

Een ruimere risicolimiet creëert geen edge. De break-evendrempel staat op
**13,0 bps vast per round trip + impact → 16,3 (ETH) tot 57,8 (DOT) bps**
(`conf/execution/fees.yaml`, Bijlage C3 van de doorlichting). De beste
bruto-edge die ergens in deze repo gemeten is, is **+4,43 bps**; de mediaan van
120 cellen is **−4,23 bps** (`reports/short_barrier_sweep.csv`). De enige meting
met volledige herkomst is netto OOS Sharpe **−0,483**, in de ledger als
`falsified` (`artefacts/baseline/phase3_baseline.json`).

Een negatieve verwachting harder inzetten levert een grotere negatieve
verwachting. **Deze commit maakt niets verhandelbaar dat dat niet al was.** Wat
zij wel doet, staat in §4: twee limieten maakten een nog-niet-gefalsificeerde
kandidaat onmeetbaar, en dat is een meetfout, geen voorzichtigheid.

---

## 1. Herkomst per limiet — drie soorten, niet één

De belangrijkste bevinding van deze exercitie is dat `conf/risk/default.yaml`
drie soorten drempels door elkaar bevatte, met dezelfde toon en zonder
onderscheid. "De risicolagen zijn voor propfirm-regels gemaakt" geldt voor de
eerste soort, niet voor alle drie.

### Soort A — propfirm-afgeleid (mandaat weg ⇒ drempel vervalt)

Deze bestonden uitsluitend omdat een firma-contract ze eiste. Zij hebben geen
zelfstandige risico-onderbouwing.

| Sleutel | Was | Herkomst |
|---|---|---|
| `daily_loss_limit` | 0,03 | `conf/risk/default.yaml` noemde het zelf "een propfirm-lijn, en die is binair". Audit §4.1: buffer onder de firma-lijn van 5 %. |
| `max_drawdown_pct` | 0,08 | Audit §4.2: firma statisch 10 % → intern 6 %, "laat 4 % buffer voor slippage/gap". |
| `drawdown_breaker_levels` | 0,04 → 0,50 · 0,06 → 0,25 | Trappen die bestaan om de firmalijn onaangeroerd te laten. |
| `daily_var_limit_pct` | 0,02 | Audit §4.3 koppelt het ES-budget aan de daglimiet: "1-dags 99 %-ES < 0,5 × daily-limit". |
| `max_position_age_h` | 48 | Gedwongen flatten. Een funded account draagt geen positie die de firma niet kan afrekenen. |

### Soort B — interne "strengste wint" (fase-4-regel, géén propfirm)

Fase 4 vond dezelfde limiet op meerdere plaatsen met verschillende waarden en
koos consequent de strengste (`reports/phase4_entanglement_map.md` §5). Dat was
een *opruimregel*, geen risicomandaat. Zij mag herzien worden, maar het
wegvallen van de propfirm is daar niet de reden voor — de reden is dat er een
eigen risicobereidheid voor in de plaats komt, en die hoort hier expliciet te
staan in plaats van als bijproduct van een opruimactie.

| Sleutel | Was | Kandidaten waaruit "strengste" koos |
|---|---|---|
| `sigma_target` | 0,08 | 0,08 (`conf_config.yaml`) · 0,12 (`risk/portfolio.py`) |
| `max_leverage` | 1,5 | 1,5 · 2,0 · 4,0 |
| `gross_cap` | 1,5 | 1,5 · 2,0 · 4,0 |
| `net_cap` | 0,60 | afgeleid: 0,40 × gross |
| `max_position_pct` | 0,25 | — |
| `max_concentration` | 0,40 | 0,80 (`risk/position_limits.py`) · 0,40 (`portfolio/constraints.py`) |

### Soort C — marktfeit (geen mandaat ⇒ blijft staan)

Deze drempels zijn geen risicobereidheid maar een eigenschap van de markt. Ze
verruimen is niet moediger worden; het is de backtest laten rekenen met fills
die niet bestaan.

| Sleutel | Waarde | Waarom onaangeroerd |
|---|---|---|
| `adv_participation_cap` | 0,01 | Liquiditeit. Boven ~1 % van de ADV is de impactterm geen correctie meer maar het hele resultaat, en `conf/execution/impact.yaml` staat op `IMPACT_UNCALIBRATED`. Een ruimere ADV-cap vervalst de kostenkant van élke meting. |
| `max_cluster_concentration` | 0,60 | Vacuous op dit universum (AD-4): zes perps met gemiddelde paarsgewijze correlatie 0,735 zijn één cluster, dus `effective_relative_cap(1, 0,60) = 1,0`. Verruimen verandert niets; verwijderen zou hem onbruikbaar maken zodra er een tweede aantoonbaar cluster bijkomt. |

---

## 2. De nieuwe waarden

| Sleutel | Was | Wordt | Soort | Grond |
|---|---|---|---|---|
| `sigma_target` | 0,08 | **0,20** | B | Eigen-kapitaaltarget. 0,20 geannualiseerd ≈ 1,05 %/dag. |
| `max_leverage` | 1,5 | **4,0** | B | De ruimste waarde die vóór de consolidatie in de repo stond; `conf/conf_config.yaml:116` draagt hem nog als `max_portfolio_leverage_bidir`. |
| `gross_cap` | 1,5 | **4,0** | B | Idem. |
| `net_cap` | 0,60 | **2,0** | B | 0,50 × gross in plaats van 0,40 × gross. Zie §4.2: funding carry is per constructie directioneel. |
| `max_position_pct` | 0,25 | **0,80** | B | 6 × 0,80 = 4,8 > `gross_cap` 4,0, dus de gross-cap blijft binden. Zie §3. |
| `max_concentration` | 0,40 | **0,40** | B | **Ongewijzigd.** Niet propfirm-afgeleid, en volgens AD-4 de enige werkzame spreidingsbescherming op dit universum. Er is geen mandaatgrond om hem te verruimen. |
| `max_cluster_concentration` | 0,60 | 0,60 | C | Ongewijzigd, vacuous. |
| `adv_participation_cap` | 0,01 | 0,01 | C | Ongewijzigd, marktfeit. |
| `drawdown_breaker_levels` | 0,04 → 0,50 · 0,06 → 0,25 | **0,12 → 0,60 · 0,18 → 0,30** | A | Geschaald op de nieuwe halt; blijft getrapt en monotoon. |
| `max_drawdown_pct` | 0,08 | **0,25** | A | Ruïnelijn, niet comfortlijn. Zie §2.1 voor het plafond. |
| `daily_loss_limit` | 0,03 | **0,10** | A | Geen contractlijn meer; alleen nog containment. Zie §2.2. |
| `daily_var_limit_pct` | 0,02 | **0,05** | A | Houdt de verhouding van audit §4.3 vast: 0,5 × de daglimiet. |
| `max_position_age_h` | 48 | **720** | A | 30 dagen. Zie §4.1 — dit is de limiet die een levende kandidaat onmeetbaar maakte. |

### 2.1 Waarom `max_drawdown_pct` op 0,25 en niet hoger

Twee metingen zetten dit getal vast, van onderen en van boven.

**Van onderen — 0,08 was niet meetbaar.** De enige echte forward-meting in de
repo heeft `max_drawdown: −0,3817` (`reports/adaptive_wf_metrics.json`,
`forward_sharpe 0,688`, `pbo 0,222`). Een hard halt op 0,08 beëindigt dat pad
onherroepelijk en handmatig-reset-only ver voordat het venster uit is. Elke
kandidaat die zo'n pad reproduceert, is onder het oude budget niet "te
riskant" — hij is **niet te meten**. Dat is geen risicobeheersing maar een
meetfout met een risicomotivering.

**Van boven — 0,30 is een hard plafond.** `src/tradebot/risk/stress_test.py:68`
zet `GAP_DOWN_FRACTION = 0.30`: het S3-scenario schokt de equity met −30 % en
`tests/integration/test_risk_overrules_alpha.py:203` eist dat die schok de halt
trípt (`GAP_DOWN_FRACTION > cfg.max_drawdown_pct`). Boven 0,30 wordt dat
scenario non-bindend en toetst de stress-suite niets meer. 0,25 laat het
scenario zijn werk doen met 5 punten marge.

Merk op wat hieruit volgt en niet verzwegen mag worden: **het enige
forward-gemeten pad in deze repo breekt ook door 0,25 heen.** Dat is opzet. De
halt is een ruïnelijn; een strategie die hem raakt, hoort te stoppen. Het
verschil met 0,08 is dat zij nu eerst gemeten kan worden.

### 2.2 Waarom de daglimiet nooit de bindende limiet was

Bij `sigma_target = 0,08` draait het boek op een gerealiseerde volatiliteit van
**0,0742** geannualiseerd (`reports/vol_target_sweep.csv`). Met
`annualisation_factor: 365` is dat 0,0742 / √365 = **0,388 % per dag** (1σ).

De oude daglimiet van 3 % is daarmee **7,7 dagelijkse σ**. Onder normale
variantie vuurt hij nooit. Hij vuurt uitsluitend op een sprong — en
`conf/data/default.yaml` accepteert `|log-return|` tot 0,35 (≈ +42 % / −30 %)
expliciet als echte marktbeweging. De daily governor is dus altijd een
**gap-risicolijn** geweest en nooit een variantielijn, bij 0,03 net zo goed als
bij 0,10.

Dat verandert wat de nieuwe waarde betekent: 0,10 is geen tienmaal grotere
risicobereidheid, het is dezelfde functie (containment van een sprong, een bug
of een runaway-loop) zonder de contractbuffer die er geen doel meer bij heeft.

### 2.3 Het verruimen van `sigma_target` is op zichzelf inert — en dat is een openstaand defect

`reports/vol_target_sweep.csv` laat `realized_vol` **identiek 0,0742** zien bij
`target_vol` 0,16, 0,24 **én** 0,35 (sides=A), en identiek 0,1252 bij 0,24 en
0,35 (sides=ALL). Een vol-target die over een factor twee aan targets dezelfde
gerealiseerde volatiliteit oplevert, bindt niet: er zit een andere limiet vóór.

Dit is de anomalie die fase 9 §6.3 (kandidaat A) al aanwijst als direct
reproduceerbaar, en zij is met deze commit **niet opgelost**. Twee dingen horen
daarom expliciet gezegd:

1. Die sweep is gedraaid door `miscellaneous/run_hrp_backtest.py`, dat
   risicoparameters **buiten `conf/` om** verhoogde (vol-target 0,12 → 0,25;
   leverage 4,0 → 6,0; kelly 0,25 → 0,40; dd-breaker 0,20 → 0,30) — de verboden
   zet uit fase 9 §10. De caps in die sweep zijn dus niet de caps van deze
   config, en het getal 0,0742 mag niet als eigenschap van `conf/risk/` worden
   gelezen.
2. Ongeacht de oorzaak geldt: wie verwacht dat `sigma_target: 0,20` het boek
   2,5× groter maakt, moet dat **meten** en niet aannemen. Zolang §2.3 open
   staat, is de werkelijke schaal van het boek een open vraag.

**Openstaand, voor fase 10:** reproduceer de saturatie met de caps uit
`conf/risk/default.yaml` en stel vast welke limiet werkelijk bindt. Een
risicolaag waarvan de primaire schaalparameter niet bindt, is kapot, en dat
oordeel staat los van hoe ruim de getallen zijn.

---

## 3. Welke limieten binden nu, en welke niet

Een limiet die nooit bindt is documentatie, niet beheersing. Bij de nieuwe
waarden, in de volgorde uit `constraint_order`:

| Limiet | Bindt bij |
|---|---|
| `vol_target` | `min(4,0 ; 0,20/σ̂)` — de leverage-cap bindt pas bij σ̂ < 0,05 geannualiseerd, wat op dit universum niet voorkomt. De target bindt dus normaal; §2.3 is de open vraag of dat waar is. |
| `per_asset_cap` (0,80) | vanaf boek-gross ≈ 2,0 en hoger |
| `concentration_cap` (0,40) | onder boek-gross ≈ 2,0 |
| `adv_cap` (0,01) | afhankelijk van ordergrootte t.o.v. ADV; ongewijzigd |
| `cluster_cap` (0,60) | **nooit** — vacuous per AD-4 |
| `gross_cap` (4,0) | 6 × 0,80 = 4,8 > 4,0, dus bindt bij een vol boek |
| `net_cap` (2,0) | bij een directioneel boek boven 2,0 netto |

`per_asset_cap` en `concentration_cap` wisselen elkaar af rond gross ≈ 2,0. Dat
is bewust: onder die grens doet spreiding het werk, erboven de absolute
per-symboolgrens. Geen van beide is dood.

---

## 4. Wat dit besluit feitelijk mogelijk maakt

Dit is het enige deel van dit document met een meetbaar gevolg. De rest is
bereidheid; dit is bereikbaarheid.

### 4.1 `max_position_age_h: 48` maakte kandidaat B onmeetbaar

Fase 9 §6.3 houdt drie kandidaten over. Alleen kandidaat B (funding carry)
koppelt volatiliteitsmodellering aan een verdienmechanisme in plaats van aan een
noemer, en hij is **niet gefalsificeerd** — hij is niet gedraaid.

De rekensom uit die paragraaf: ETH-carry ≈ 6,5e-5 per 8h in de `all`-bucket
(≈ 0,65 bps per 8h ≈ 1,95 bps/dag) tegen 13,0 bps vaste kosten per round trip.
Daaruit volgt dat een positie **meerdere dagen** aangehouden moet worden voordat
de carry de kosten dekt — bij 1,95 bps/dag duurt het alleen al ruim **zes
dagen** voor de vaste 13,0 bps zijn terugverdiend, impact niet meegerekend.

`max_position_age_h: 48` sluit de positie na twee dagen. De limiet
**garandeerde** daarmee dat de enige overgebleven kandidaat zijn kosten niet kon
terugverdienen, en dat elke meting eraan negatief zou uitvallen om een reden die
niets met de markt te maken heeft. Een gefalsificeerde kandidaat B onder die
limiet zou een artefact van de config zijn geweest, geen marktuitspraak.

720 uur (30 dagen) is ruim genoeg om de houdduur waarbij
`carry − kosten − prijsdrift > 0` te vinden, of om vast te stellen dat die niet
bestaat. Dat laatste blijft een volwaardig, publiceerbaar resultaat — fase 9 §9
zegt het zelf — maar het moet dan uit de data komen en niet uit een
propfirm-flattenregel.

**Let op de tweede bron van waarheid.** `src/tradebot/live/circuit_breaker.py:85`
draagt een eigen `max_position_age_h: int = 48` als code-default en
`apps/live_paper_trader.py:296` geeft `48` hardcoded door. Dat is precies de
constructie die `docs/RISK_CONTRACT.md` §8 verbiedt ("nul defaults in
functiehandtekeningen"). Deze commit adresseert `conf/` en het schema; de
code-defaults staan in §6 als openstaand punt.

### 4.2 `net_cap: 0,60` verbood een directioneel carry-boek

Funding carry is per constructie directioneel: bij positieve funding betalen
leveraged longs de shorts, dus de positie die de carry ontvangt is
**systematisch short**. Een netto-cap van 0,40 × gross dwingt het boek richting
marktneutraal en snijdt daarmee juist de exposure weg die het mechanisme
oplevert.

`net_cap: 2,0` (0,50 × gross) laat een volledig directioneel boek van 2,0 toe en
blijft een onafhankelijke limiet: een ongehedged boek van 4,0 blijft verboden.

---

## 5. Wat deze commit expliciet NIET doet

1. **Geen drempel verruimd omdat een resultaat tegenviel.** Fase 9 §9 noemt dat
   de enige norm die in dit project onbeschadigd is. De grond is een gewijzigd
   mandaat — een exogeen feit — en per limiet is in §1 opgeschreven of het
   mandaat hem werkelijk zette. Soort B en C zijn daarom apart behandeld in
   plaats van meegesleept op de propfirm-motivering.
2. **Geen gate groen gemaakt door de gate te verzwakken.** De vier bekende
   poort-failures (KG-B1/B2/B3 op `cm_carry` en `cm_tsmom`) staan na deze commit
   onveranderd rood. Zij toetsen alpha, niet risico.
3. **Geen enkele edge-claim aangeraakt.** Geen backtest opnieuw gedraaid, geen
   metriek herschreven, geen ledger-entry gewijzigd. De cijfers in §0 blijven
   staan.
4. **De propfirm-machinerie niet verwijderd.** `risk/daily_loss_governor.py`
   (`PropfirmGovernor`, `PropfirmLimits`, `AccountRegime`) en
   `monitoring/cross_account.py` blijven staan, gemarkeerd als dormant. Zij zijn
   opt-in (`live/engine.py` bouwt de governor alleen wanneer `propfirm_limits`
   is meegegeven, default `None`) en dus inactief. Verwijderen is
   ontmantelingswerk en hoort in fase 10, achter de tag uit fase 9 §5.3.
5. **`adv_participation_cap` niet verruimd.** Zie §1, soort C.

---

## 6. Openstaand na deze commit

| # | Punt | Waar |
|---|---|---|
| 1 | De vol-target-saturatie reproduceren met de caps uit `conf/risk/` en de werkelijk bindende limiet benoemen | §2.3 · DI-18 · fase 9 §6.3 kandidaat A |
| 2 | De risicodrempels uit `live/circuit_breaker.py`, `apps/live_paper_trader.py` en `conf/config.yaml::circuit_breaker` halen; uitsluitend uit `conf/risk/` lezen | §4.1 · DI-19 · `docs/RISK_CONTRACT.md` §8 |
| 3 | `PropfirmGovernor` + `monitoring/cross_account.py` schrappen of naar `frozen/` met een gemeten heropeningsvoorwaarde | §5.4 · DI-20 · fase 9 §5.3 regel 5 |
| 4 | `conf/env/` bestaat niet, terwijl `conf/config.yaml` `defaults: - env: dev` declareert — een tweede, stille bron van risicowaarden zodra iemand hem aanmaakt | `conf/config.yaml:2` |
| 5 | `conf/config.yaml` `circuit_breaker` draagt nog een eigen `max_drawdown_pct: 0.08` en `max_daily_loss_pct: 0.03` naast `conf/risk/` | `conf/config.yaml:36-40` |

---

## 7. Wat een reviewer moet kunnen nazeggen

Als een van deze uitspraken niet meer waar is, is dit document geschonden:

1. Elke gewijzigde drempel staat in §2 met zijn soort (A/B/C) en zijn grond.
2. Geen enkele soort-C-drempel is gewijzigd.
3. `max_drawdown_pct < GAP_DOWN_FRACTION`, zodat het S3-stressscenario de halt
   nog trípt.
4. Elke de-grossing-trap ligt strikt onder `max_drawdown_pct` en is monotoon.
5. `daily_loss_limit < max_drawdown_pct` — één dag kan de ruïnelijn niet
   overschrijden.
6. `max_position_age_h` staat toe dat een positie lang genoeg leeft om de
   13,0 bps vaste kosten met carry terug te verdienen (> 7 dagen).
7. Het aantal falende tests is niet gestegen boven de vier bekende
   poort-failures.
