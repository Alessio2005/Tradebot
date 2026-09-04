# PHASE 7 — DIVERGENCE MAP: `live/` TEGENOVER DE PHASE 5-KETEN

> **Deliverable D-1** · Phase 7/8 Stage D · Phase 7 stap 1
> **Gemeten op:** 2026-09-01, tegen `df96805` plus de niet-gecommitte Stage C-3-boom
> **Methode:** statisch. Elke regel hieronder komt uit `grep`/`ls` op de boom zelf,
> niet uit een eerder rapport. Waar deze meting afwijkt van de meting van
> 2026-08-27 in `fase_7_8_consolidatie_productie.md`, staat de afwijking in §7 en
> **prevaleert deze meting** (§10, regel 1).

---

## 0. De uitkomst, eerst

De oorspronkelijke Phase 7-prompt vraagt om het opsporen van **duplicatie** tussen
`live/` en `backtest/`. Die vraag heeft hier geen antwoord, want er is geen
duplicatie. Er zijn **twee volledig gescheiden ketens** die elkaar op drie modules
raken, waarvan er één domeinlogica is.

**Gemeten:** `live/` importeert 18 `tradebot`-modules, de Phase 5-keten 19. De
doorsnede telt er **drie**, en twee daarvan zijn infrastructuur:

| Wat | Aantal |
|---|---|
| Modules die `live/` uit `tradebot` importeert | 18 |
| Modules die de Phase 5-keten importeert | 19 |
| Doorsnede | **3** — `alpha/base.py`, `schemas/config.py`, `utils/failfast.py` |
| Daarvan domeinlogica | **1** (`alpha/base.py`); de andere twee zijn configuratielading en de fail-fast-helper |
| Verwijzingen naar `RiskDecision`, `risk/engine.py` of `risk/kill_switches.py` in `live/` of `oms/` | **0** |
| Verwijzingen naar `EventDrivenEngine` of `backtest/accounting.py` in `live/` of `oms/` | **0** |

De Phase 5-keten importeert `risk/kill_switches.py` wél — via `risk/engine.py`.
De live-keten, die hem operationeel het hardst nodig heeft, niet.

De twee nullen onderaan zijn de belangrijkste meting van dit rapport. De soevereine
risicolaag die Phase 4 en 5 hebben gebouwd — de laag waarvan AD-1 zegt dat zij
als enige een limiet mag stellen — is in de live-keten **niet aangesloten**. Niet
gedeeltelijk, niet via een adapter: nul verwijzingen.

---

## 1. Wat er precies is gemeten

```
grep -rhoE "from \.\.[a-z_]+(\.[a-z_]+)*" src/tradebot/live/*.py | sort -u
grep -rn "RiskDecision|risk\.engine|kill_switches" src/tradebot/live/ src/tradebot/oms/
grep -rn "backtest\.|EventDrivenEngine|accounting"  src/tradebot/live/ src/tradebot/oms/
```

De derde en vierde rij van de tabel hierboven leveren alleen treffers op in
*commentaar en docstrings* — regels die over de backtest praten, niet code die
hem aanroept. Die treffers zijn met de hand nagelopen en tellen niet mee.

---

## 2. De kaart, per component

Per component: wat `live/` nu gebruikt, wat de Phase 5-tegenhanger is, en waar
het gedrag uiteenloopt. De laatste kolom is een **risico-oordeel**, geen meting;
de meting staat in de eerste twee.

| # | Component | `live/` gebruikt | Phase 5-tegenhanger | Divergentierisico |
|---|---|---|---|---|
| 1 | Orderconstructie & routing | `oms/router.py::place_order(order)` | `execution/order_router.py::build_orders(..., decision: RiskDecision)` | **De bypass.** Zie §3 |
| 2 | Risicobesluit & limieten | `live/execution_controller.py::_check_position_limits` | `risk/engine.py` → `risk/contract.py::RiskDecision` | **Twee limietstelsels.** Zie §4 |
| 3 | HALT-toestand | `live/circuit_breaker.py` — journaal op schijf, startweigering met een venster van 24 uur | `risk/kill_switches.py::HaltStore` (duurzaam, onherroepelijk, operator vereist) | **De HALT vervalt na 24 uur.** Zie §5 |
| 4 | Kosten & fills | `oms/paper_oms.py` → `execution/slippage.py` → `execution/market_impact.py` | `execution/impact_model.py` + `conf/execution/impact.yaml` | **eta verschilt een factor ~30.** Zie §6 |
| 5 | Accounting | `oms/position_tracker.py` | `backtest/accounting.py` | Geen invarianten in de live-variant. Zie §4 |
| 6 | Featureberekening | `features/pipeline.py::build_features` | `features/registry.py` → `features/base.py::FeaturePipeline` | **Twee featurestacks.** Zie §6 |
| 7 | Alphaberekening | `alpha/base.py`, `alpha/combination.py` | `alpha/base.py` | **Gedeeld** — de enige domeincomponent die dat is |
| 8 | Positiegrootte | `portfolio/constraints.py`, `portfolio/optimizer.py` | `portfolio/equal_weight.py`, `portfolio/risk_parity.py` | Andere sizers; niet vergelijkbaar zonder pariteitstest |

---

## 3. De bypass: twee routers, één contract

Dit is het verschil dat D-1 als scherpste aanwijst, en de meting bevestigt hem.

```
oms/router.py:140          async def place_order(self, order: Order) -> Fill:
execution/order_router.py:312                    decision: RiskDecision,
execution/order_router.py:326          isinstance(decision, RiskDecision),
```

`execution/order_router.py` draagt in zijn eigen moduledocstring uit waarom dat
argument verplicht is:

> *"Hier is het typesysteem de handhaving: zonder `RiskDecision` compileert de
> aanroep niet, en met de VERKEERDE `RiskDecision` crasht hij."*

`oms/router.py::place_order` neemt een `Order` en niets anders. Er is dus geen
typeniveau waarop een order zonder risicobesluit wordt tegengehouden — en omdat
`live/engine.py` `oms.router` importeert en niet `execution.order_router`, is dat
precies het pad dat een live order vandaag aflegt.

**Dit is no-go 6 en no-go 7 tegelijk**: een tweede implementatie van
executielogica, waarin de limiet niet ontbreekt-en-crasht maar simpelweg niet
wordt gevraagd.

---

## 4. Twee limietstelsels die vandaag toevallig overeenkomen

`live/circuit_breaker.py` draagt zijn drempels als eigen velden:

| `circuit_breaker.py` | waarde | `conf/risk/default.yaml` | waarde |
|---|---|---|---|
| `max_drawdown_pct` | 0.08 | `max_drawdown_pct` | 0.08 |
| `max_daily_loss_pct` | 0.03 | `daily_loss_limit` | 0.03 |
| `max_intraday_drawdown_pct` | 0.05 | — | *bestaat niet* |
| `max_var_breach` | 1.5 | — | *bestaat niet* |
| — | | `max_position_pct` | 0.25 |
| — | | `gross_cap` | 1.5 |
| — | | `drawdown_breaker_levels` | 0.04→0.50, 0.06→0.25 |
| — | | `constraint_order` | 4 stappen, geordend |

De eerste twee rijen zijn de kern van de bevinding: **de getallen komen overeen,
en er is geen mechanisme dat ze overeen hóudt.** Het zijn twee onafhankelijke
kopieën. Wie morgen `daily_loss_limit` in `conf/` verlaagt, verlaagt de
live-drempel niet mee, en niets in de boom merkt dat op. Een limietstelsel dat
alleen door toeval klopt, is geen limietstelsel.

`max_var_breach = 1.5` verdient een aparte waarschuwing: het getal is gelijk aan
`max_leverage` en aan `gross_cap`, maar het betekent iets volstrekt anders. Drie
identieke getallen met drie betekenissen in twee bestanden is een verwisseling
die op enig moment gemaakt gaat worden.

**De twee limieten die naar oneindig defaulten** (C4) staan in
`live/execution_controller.py`:

```
94:        self._max_notional_per_symbol: float = float("inf")
95:        self._max_gross_notional: float = float("inf")
```

De prompt noemt er één; het zijn er twee. Wie de setter vergeet, handelt zonder
notional-limiet *per symbool* én zonder *bruto* limiet — en krijgt geen fout.

**Accounting.** `backtest/accounting.py` sluit zijn invarianten na elke bar af
met `require(...)`-aanroepen; §13 noemt dat de constructie die "geen verdwenen
kosten" afdwingbaar maakt. `oms/position_tracker.py` bevat **nul** `require`- of
`assert`-aanroepen en draagt `initial_equity: float = 100_000.0` als default. De
live-keten heeft dus geen enkele boekhoudkundige sluitcontrole.

---

## 5. De HALT verloopt na 24 uur — hij wordt niet opgeheven, hij vervalt

**Correctie op een eerdere lezing van dit rapport.** Een eerste versie van deze
sectie stelde dat er "geen schrijfpad naar schijf" is. Dat is onjuist en de
werkelijkheid is subtieler:

* `_persist_trip` schrijft elke trip **wél** weg naar een append-only logboek,
  met `{"ts", "reason", "acknowledged": false}` (regel 266-278);
* `_check_prior_trips` **weigert** bij het opstarten te starten wanneer er een
  niet-geaccordeerde trip in dat logboek staat (regel 280-298).

Tot zover doet de live-keten precies wat D9 vraagt. Het probleem zit in één
regel:

```
live/circuit_breaker.py:284    cutoff_ts = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)
```

**De weigering geldt alleen voor trips van de laatste 24 uur.** Een systeem dat
halteert en 25 uur later wordt herstart, start gewoon op — met een trip die nog
altijd `acknowledged: false` draagt. De halt is niet opgeheven door een mens; hij
is verlopen door de klok.

Daarnaast is de halt-TOESTAND zelf in-memory (`_halt_reason`, `_halt_ts`, en
`state.circuit_breaker_active`). Na een herstart binnen het venster weigert het
proces te starten; buiten het venster start het op als niet-gehalteerd, zonder
enige herinnering aan de reden.

Zet dat naast `risk/kill_switches.py::HaltStore`, die zichzelf beschrijft als
*"Duurzame, onherroepelijke `HALTED`-toestand op schijf"*. Wat
`tests/unit/test_kill_switch_irreversibility.py` van die store afdwingt:

| Eigenschap | `HaltStore` (soeverein) | `CircuitBreaker` (live) |
|---|---|---|
| Overleeft een vers object | ja, getoetst | toestand niet; het logboek wel |
| Volledig equityherstel heft hem op | nee, getoetst | n.v.t. |
| Een nieuwe handelsdag heft hem op | nee, getoetst | **na 24 uur: ja** |
| Opheffen vereist een operator + motivering | ja, getoetst | nee — de tijd volstaat |
| Eerste oorzaak wordt nooit overschreven | ja, getoetst | niet afgedwongen |

Dit is daarmee geen ontbrekende persistentie maar iets wat lastiger te zien is:
persistentie die eruitziet alsof zij klopt en stilzwijgend vervalt. In de termen
van no-go 9 — *"De HALT-toestand overleeft een procesherstart niet"* — is de
live-keten rood, maar pas na een etmaal.

---

## 6. Vijf bevindingen die niet in de D-1-opdracht staan

De opdracht wijst `oms/router.py` versus `execution/order_router.py` aan als het
scherpste verschil. Deze meting vindt er vijf die daar niet in staan en die
zwaarder wegen voor exit-criterium **D7** (dagelijkse bit-identieke pariteit),
omdat zij niet een controle omzeilen maar **de getallen zelf veranderen**.

§6.1 en §6.2 zijn van 2026-09-01. §6.3 t/m §6.5 zijn van 2026-09-02, gemeten bij
de voorbereiding van D1, en zij veranderen de VOLGORDE van de fase: de wiring
die D-1 vraagt, kan niet als eerste stap.

### 6.1 De live-keten prijst impact met een andere eta

Er bestaan twee functies met **dezelfde naam** in twee modules:

| | `market_impact.square_root_impact` | `impact_model.square_root_impact` |
|---|---|---|
| `eta` | default `0.142` | verplichte `ImpactParams` |
| Provenance | geen | `data_hash`, `sample_size`, CI, `status` |
| Ontbrekende eta | plausibel getal | `ConfigContractError` |
| Aangeroepen door | `execution/slippage.py` → `oms/paper_oms.py` → **live** | `execution/order_router.py` → **backtest** |

`impact_model.py` legt in zijn eigen docstring uit waarom hij naast de ander
bestaat: *"Een default maakt hem onzichtbaar: een caller die `eta` vergeet, krijgt
geen fout maar een plausibel ogend getal."*

`from tradebot.execution import square_root_impact` levert de **onveilige**
variant — `execution/__init__.py` exporteert die uit `market_impact`.

De gemeten waarden lopen ver uiteen:

| Parameter | live-keten | Phase 5-keten |
|---|---|---|
| `eta` | `0.1` (`SlippageModel.impact_eta`) of `0.142` (`market_impact`) | **`2.991922`** (`conf/execution/impact.yaml`) |
| Slippage | `fixed_bps = 5.0`, `floor_bps = 5.0` | `assumed_half_spread_bps = 1.0` |
| Kostenlabel | geen | `IMPACT_UNCALIBRATED`, `SPREAD_ASSUMED` |

Een factor ~30 op eta en een factor 5 op de aangenomen spread. Zolang dit staat,
kán de dagelijkse pariteitstest van D7 niet slagen, en is elk verschil dat hij
meet verklaard door de configuratie in plaats van door het gedrag.

### 6.2 Er zijn twee featurestacks, en de live-docstring beweert het tegendeel

`live/feature_updater.py` regel 16 zegt:

> *"Both paths call the SAME build_features() function (features/pipeline.py)"*

Dat was waar in het Wave-tijdperk. **Vandaag is het onjuist.** Gemeten:

| Keten | Featurecode | Bron |
|---|---|---|
| live | `features/pipeline.py::build_features` (Hydra `DictConfig`, `PanelPipeline`) | `artefacts/features/*.parquet` |
| Phase 5/6 authoritative | `features/registry.py::build_default_registry` → `features/base.py::FeaturePipeline` | `data/pit_store` via `DataRegister` |

`phase5_baseline.py` importeert `features.base`; `run_meta_labeling.py` en
`run_regime_benchmark.py` bouwen hun matrix via `features/registry.py`. Geen van
drieën raakt `features/pipeline.py` aan. Omgekeerd raakt `live/feature_updater.py`
`features/registry.py` nergens aan.

Dit is een tweede implementatie van de featureberekening en daarmee een schending
van de PIT-regel uit §10: *"De live feature-berekening gebruikt exact dezelfde
causale code als de backtest."* De docstring die het tegendeel beweert valt
bovendien onder no-go 20.

### 6.3 De twee ketens draaien op verschillende bars — D7 is niet zwak maar ONGEDEFINIEERD

*Gemeten 2026-09-02, bij de voorbereiding van D1.*

| | live-keten | Phase 5-keten |
|---|---|---|
| Barresolutie | **5 seconden** (`live/feed.py::FeedConfig.bar_seconds = 5`; `interval="5s"`) | **dagbars** (1.743 bars, 2021-11-15 t/m 2026-08-23, gecertificeerde PIT-store) |
| Annualisatie van sigma | — | `conf/model/volatility.yaml::annualisation_factor = 365.0`, oftewel dagbars |

Een dag telt **17.280** bars van 5 seconden. De annualisatiefactor die bij die
resolutie hoort is 365 × 17.280 = **6.307.200**, niet 365 — een verhouding van
17.280×, en dus een factor **131,5** op sigma zelf.

Dat is geen detail, want `vol_target` staat in `constraint_order`
(`conf/risk/default.yaml`) en deelt door precies dat getal. Een `sigma_hat` die
op 5s-bars is geschat en door een op dagbars gekalibreerde vol-targeting gaat,
onderschat de geannualiseerde volatiliteit met ruwweg twee ordes van grootte —
en vol-targeting die door een te kleine sigma deelt, maakt posities **groter**.

**Wat dit met D7 doet.** D7 vraagt om *dagelijkse bit-identieke pariteit*. Het
exit-rapport §2.2 noemde D7 "niet toetsbaar" omdat een pariteitstest op
verschillende eta's en featurestacks de configuratie zou meten. Die formulering
is te mild. Tussen een keten op 5-secondebars en een keten op dagbars bestaat
geen correspondentie waarover "identiek" een betekenis heeft: er is geen paar
observaties om te vergelijken. D7 is niet zwak toetsbaar maar **niet
gedefinieerd**, en dat moet eerst worden opgelost met een besluit over de
resolutie — niet met een test.

### 6.4 De twee ketens handelen niet in hetzelfde universum

*Gemeten 2026-09-02.*

```
apps/live_paper_trader.py:44
    _SYMBOLS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]   # 5

conf/risk/default.yaml::clusters
    BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, DOTUSDT, LINKUSDT                 # 6
```

De live-keten handelt **vijf** namen; de soevereine policy is geschreven voor
**zes**. Het verschil is BTCUSDT — de grootste en meest liquide naam van het
universum, en de naam waarop elke spreidingslimiet het minst bindt.

`gross_cap`, `concentration_cap` en `cluster_cap` zijn grenzen over een BOEK.
Dezelfde grenswaarden op een boek van vijf namen binden anders dan op een boek
van zes: valt de grootste, minst gecorreleerde naam weg, dan stijgt de
concentratie van wat overblijft. De policy is dus niet alleen niet aangesloten
op de live-keten — hij is ook niet voor dat boek gekalibreerd.

### 6.5 Wat §6.3 en §6.4 betekenen voor de volgorde van D1

De D-1-opdracht luidt: sluit `live/` aan op `execution/order_router.py` en
`risk/engine.py`. `RiskEngine.decide` draait `constraint_order`, en daar staat
`vol_target` in. Die stap heeft een `sigma_hat` nodig **in de eenheid waarin de
policy is gekalibreerd**.

Dat kan vandaag niet. De live-keten heeft geen EWMA-sigma; zij heeft een
range-proxy (`live/engine.py:413`, `(high - low) / close`) op 5-secondebars —
en H1 heeft juist van die proxy vastgesteld dat hij aantoonbaar een andere
grootheid meet dan de modellen voorspellen (`mean(proxy)/mean(r²)` = 1,27–2,24,
`reports/GARCH_VS_EWMA_COMPETITION.md`).

**Daarom gaat de wiring niet eerst.** Zou je `RiskEngine` nu in de live-lus
hangen, dan neemt de soevereine laag een besluit over invoer waarvoor zij nooit
is gekalibreerd: een verkeerd geschaalde sigma, over een boek van vijf in plaats
van zes. Het resultaat is een limietstelsel dat er **aangesloten uitziet** en
verkeerde getallen produceert — en dat is gevaarlijker dan de huidige bypass,
want de bypass is zichtbaar en dit niet. Het zou bovendien exact het defect zijn
dat het exit-rapport §4.9 beschrijft: een poort die groen staat op de vorm.

De volgorde die hieruit volgt, en die vóór de bestaande volgorde in §9 komt:

1. **Besluit over de resolutie.** Draait de live-keten op dagbars, of wordt de
   policy op een intraday-resolutie gekalibreerd? Dit is een besluit voor de
   eigenaar van het risicoregime, geen implementatiekeuze.
2. **Universum gelijktrekken** — vijf namen of zes, in beide ketens dezelfde.
3. **Eén vol-estimator**, in de eenheid van het besluit uit stap 1.
4. **Dan pas** de wiring naar `risk/engine.py` en `execution/order_router.py`.

---

## 7. Afwijkingen van de meting van 2026-08-27

De prompt schrijft voor dat elke meting bij aanvang wordt herhaald en elke
afwijking genoteerd. Drie afwijkingen, alle drie in dezelfde richting — de
werkelijkheid is iets erger dan opgeschreven:

| # | Claim van 2026-08-27 | Meting 2026-09-01 |
|---|---|---|
| 1 | DI-7: *"vijf `asyncio.create_task(...)` in `live/`"* | **Vier** in `live/feed.py` (226, 228, 231, 236) en **één** in `monitoring/sharpe_monitor.py:78`. Totaal vijf, maar niet alle vijf in `live/` — de sluiting van DI-7 moet dus ook `monitoring/` raken |
| 2 | C4: *"`_max_gross_notional = float("inf")`"* | **Twee** oneindige defaults, niet één: ook `_max_notional_per_symbol` (regel 94) |
| 3 | C6: *"`min_confidence = 0.55`"* | In **twee** modules: `execution_controller.py:62` én `signal_runner.py:49`, met een werkend filter op `signal_runner.py:125` en `:254` |

Bevestigd zonder afwijking: C1 (eigen drempels in `circuit_breaker.py`), C2
(in-memory HALT), C3 (`_check_position_limits`), de vier `test_chaos.py`-skips,
`AlertSeverity` zonder `HALT`, PSI-drempels als module-constanten
(`drift.py:34-35`), en het ontbreken van `conf/monitoring/`.

---

## 8. Wat NIET divergent is

Een kaart die alleen problemen toont, overdrijft. Drie vaststellingen aan de
andere kant:

1. **`alpha/base.py` is gedeeld.** Beide ketens bouwen hun signaal op dezelfde
   abstractie; `live/model_signal.py` en `live/signal_runner.py` importeren
   dezelfde basisklassen als `backtest/phase5_baseline.py`.
2. **`schemas/config.py` en `utils/failfast.py` zijn gedeeld.** Beide ketens
   laden hun configuratie via hetzelfde gevalideerde schema en gebruiken dezelfde
   `require(...)`-helper. Dat is infrastructuur en geen gedrag, maar het betekent
   wel dat een contract dat in `schemas/config.py` wordt vastgelegd, beide kanten
   op werkt — het is het aangrijpingspunt waarlangs de rest kan worden
   aangesloten.
3. **De soevereine risicolaag is niet fout gekopieerd — hij is afwezig.** Dat is
   operationeel gevaarlijker, maar technisch aanzienlijk goedkoper te repareren
   dan twee uiteengelopen kopieën: er is niets samen te voegen, alleen aan te
   sluiten.

---

## 9. Wat dit betekent voor de Stage D-poorten

De middelste kolom is de METING van 2026-09-01, vóór er iets was gerepareerd.
De rechter is de stand nadat Stage D-2 en D-3 in dezelfde sessie zijn
uitgevoerd. Beide staan er, zodat de meting leesbaar blijft als meting en niet
achteraf wordt bijgewerkt tot zij gunstig oogt.

| Criterium | Bij de meting | Nu | Bewijs |
|---|---|---|---|
| **D1** — `live/` gebruikt de Phase 5-componenten, nul tweede implementaties | rood | **nog rood, en geblokkeerd** | §2 componenten 1, 4, 5 en 6 staan. **Nieuw (2026-09-02):** de wiring kan niet als eerste — §6.3 t/m §6.5 |
| **D2** — C1 t/m C6 gesloten; geen limiet defaultet naar oneindig | rood | **groen** | `tests/unit/test_live_limits_are_sovereign.py` (12) |
| **D3** — DI-7 gesloten | rood | **groen** | `tests/unit/test_background_tasks_are_held.py` (4); DI-7 staat als gesloten in het register |
| **D4** — `AlertSeverity` kent `HALT`; een `HALT` schakelt uit zonder te vragen | rood | **groen** | `tests/unit/test_alert_halt_severity.py` (11) |
| **D5** — drempels uit `conf/monitoring/`, gehasht vóór de klok | rood | **groen** | `conf/monitoring/default.yaml`, `apps/freeze_monitoring.py`, `tests/unit/test_monitoring_thresholds_are_frozen.py` (6) |
| **D6** — de vier `test_chaos.py`-skips | rood | **nog rood** | ongewijzigd; zij vragen de live-engine-integratie uit D1 |
| **D7** — dagelijkse bit-identieke pariteit | niet toetsbaar | **niet gedefinieerd** | §6.1 en §6.2 staan onaangeroerd. **Sterker (§6.3):** 5-secondebars tegen dagbars — er is geen paar observaties om identiek over te zijn |
| **D9** — `HALTED` overleeft een procesherstart | rood na 24 uur | **groen** | `tests/unit/test_live_halt_is_irreversible.py` (10), inclusief de 25-uurs-regressie |

> **Herzien 2026-09-02.** De volgorde hieronder is nog steeds juist, maar niet
> meer volledig: §6.3 t/m §6.5 zetten er drie stappen vóór. De barresolutie en
> het universum moeten gelijk zijn voordat de wiring naar `risk/engine.py`
> zinvol is, anders neemt de soevereine laag een besluit over invoer waarvoor
> zij niet is gekalibreerd.

**De volgorde die hieruit volgt.** §6.1 en §6.2 gaan vóór D7, want een
pariteitstest op twee verschillende featurestacks en twee verschillende
impactmodellen meet de configuratie en niet het gedrag. D1 gaat vóór D6, want de
vier chaos-skips vragen precies de engine-integratie die D1 moet leveren — ze nu
implementeren zou betekenen dat zij de tweede implementatie toetsen in plaats
van de eerste.

Wat in dezelfde sessie is gesloten, is telkens gesloten met een test die zijn
eigen negatieve controle draagt: elke detector is aantoonbaar rood geweest op de
toestand van vóór de reparatie.

---

## 10. Wat hiermee NIET is vastgesteld

1. **Of het gedrag daadwerkelijk uiteenloopt.** Dit rapport is *statisch*. Het
   stelt vast dat twee ketens verschillende code draaien met verschillende
   parameters; het meet niet hoe ver de uitkomsten uiteenlopen. Dat is precies
   wat `test_live_backtest_parity.py` (D-5) moet doen, en die test hoort te worden
   gebouwd nadat §6.1 en §6.2 zijn gesloten — niet ervoor, want dan meet hij een
   verschil waarvan de oorzaak al bekend is.
2. **Of de live-keten vandaag geld zou verliezen.** Niet gemeten en niet de vraag
   van deze stage; §6 van de fase-opdracht stelt vast dat operationele
   gereedheid, niet winstgevendheid, hier de maat is.
3. **Of `portfolio/optimizer.py` en `portfolio/risk_parity.py` hetzelfde doen.**
   Component 8 in §2 draagt daarom "niet vergelijkbaar zonder pariteitstest" en
   geen risico-oordeel.

---

## 11. Nameting na de inventarisatie — Phase 9, stap 15

> **Gemeten 2026-09-04** met `scripts/reachability_map.py` op commit `2542d62`.
> Deze sectie **kwantificeert** en **corrigeert**; zij lost niets op. De
> consolidatie hoort bij openstaand besluit 0 en 3 van de eigenaar
> (`docs/PROJECT_STATE.md` §5), en fence 5 van Phase 9 verbiedt expliciet dat
> een opruimfase een van de twee stacks weggooit.

### 11.1 De twee afsluitingen, gemeten

De opruiming maakte het mogelijk beide afsluitingen met hetzelfde instrument te
meten in plaats van te schatten:

| | Modules | LOC |
|---|---:|---:|
| authoritative (de tien DVC-seeds) | 193 | 53.096 |
| live (`apps/live_paper_trader.py`) | 146 | 37.534 |
| **doorsnede** | **113** | **29.879** |
| alleen authoritative | 80 | 23.217 |
| alleen live | 33 | 7.655 |

De doorsnede is groot omdat de live-APP veel van de authoritative keten
importeert — schemas, utils, features, execution. Dat is niet waar de
divergentie zit. Zij zit een laag dieper:

| | Modules | LOC |
|---|---:|---:|
| de pakketten `live/` + `oms/` | 21 | 6.173 |
| **daarvan ook in de authoritative afsluiting** | **0** | **0** |

**Nul.** De live-app deelt code met de authoritative keten; de live-*pakketten*
delen er niets mee. Dat is de scherpste formulering van §3.2 die uit een meting
volgt, en zij is scherper dan het beeld dat de app-afsluiting geeft.

### 11.2 Correctie op `PROJECT_STATE.md` §3.2

§3.2 stelt: *"Er zijn **nul** verwijzingen naar `RiskDecision`, `risk/engine.py`
of `risk/kill_switches.py` in `live/` of `oms/`."* Dat is sinds Stage D niet
meer waar, en het verschil is precies het interessante deel:

| Symbool | Imports in `live/` + `oms/` | |
|---|---:|---|
| `RiskDecision` | **0** | de besluitweg is NIET aangesloten |
| `RiskEngine` | **0** | idem |
| `execution/order_router.py` | **0** | de live-orderweg loopt er niet langs |
| `risk/kill_switches.py::HaltStore` | **1** | `live/circuit_breaker.py:29` |
| `risk/limits.py` | **1** | `live/execution_controller.py:39` |
| `risk/daily_loss_governor.py` | **1** | `live/engine.py:41` |

**Het HALT-pad staat op de soevereine laag; het BESLUIT-pad niet.** Dat is een
nauwkeuriger uitspraak dan "nul verwijzingen", en zij verklaart ook waarom §2
van `PROJECT_STATE.md` tegelijk kan zeggen dat de halt-state niet aan de
werkdirectory hangt: `live/circuit_breaker.py` deelt de HaltStore wél.

`oms/router.py::place_order` vraagt nog altijd geen risicobesluit:

```python
async def place_order(self, order: Order) -> Fill:
    """Route an order to paper or live exchange."""
    if not self._live_mode:
        return self._paper.place_order(order)
    return await self._live_place(order)
```

Er is geen `RiskDecision` tussen het signaal en de order. Dat blijft de kern van
openstaand besluit 3.

### 11.3 Wat de dekkingsmeting toevoegt

Nieuw ten opzichte van eerdere metingen: de twee ketens zijn niet even goed
getoetst.

| | Dekking |
|---|---:|
| `risk/` (soeverein, authoritative) | **86,8 %** |
| `live/` | **38,5 %** |
| `oms/` | **69,8 %** |
| waarvan `oms/router.py` — de orderweg zelf | **27,9 %** |

De laag waarvan `PROJECT_STATE.md` §2 zegt dat hij *"soeverein is in de
BACKTEST"* draagt 86,8 % dekking; de laag die in productie de orders plaatst,
draagt er 38,5 %. Phase 9 heeft de twee zuivere functies van `router.py`
afgedekt (ondertekening en lotgrootte-afronding) omdat die stil verkeerd kunnen
zijn; de async HTTP-weg staat onder een ratchet in `docs/CODE_REGISTER.md`.

### 11.4 De duplicatie, feitelijk

Onveranderd ten opzichte van §6.1 t/m §6.4, hier alleen bij elkaar gezet:

| | authoritative | live |
|---|---|---|
| featurestack | `features/registry.py` → `features/base.py` | `features/pipeline.py` |
| impactmodel | `eta = 2,991922` (gekalibreerd) | `eta = 0,1` / `0,142` |
| slippage | 1 bp aangenomen half-spread | 5 bp vast |
| barresolutie | dagbars | 5-secondebars (`FeedConfig.bar_seconds = 5`) |
| annualisatie | 365 | 6.307.200 — factor **131,5** op sigma |
| universum | zes namen | vijf namen (geen BTCUSDT) |
| risicobesluit | `RiskEngine.decide` → `RiskDecision` | geen |

### 11.5 Wat deze sectie NIET doet

Zij kiest niet. Welke van de twee featurestacks blijft, welke barresolutie
geldt en of `live/` wordt aangesloten of herbouwd, zijn besluiten van de
eigenaar van het risicoregime — openstaand besluit 0 en 3. Phase 9 heeft de
`live/`-modules om precies die reden **niet** gearchiveerd, hoewel zij met 38,5
% dekking en een app die nergens werd genoemd de goedkoopste kandidaten van de
hele inventarisatie waren. Ze weggooien zou het besluit nemen.
