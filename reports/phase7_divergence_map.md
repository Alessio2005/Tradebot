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

## 6. Twee bevindingen die niet in de D-1-opdracht staan

De opdracht wijst `oms/router.py` versus `execution/order_router.py` aan als het
scherpste verschil. Deze meting vindt er twee die daar niet in staan en die
zwaarder wegen voor exit-criterium **D7** (dagelijkse bit-identieke pariteit),
omdat zij niet een controle omzeilen maar **de getallen zelf veranderen**.

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
| **D1** — `live/` gebruikt de Phase 5-componenten, nul tweede implementaties | rood | **nog rood** | §2 componenten 1, 4, 5 en 6 staan; de wiring naar `execution/order_router.py` en de featurestack zijn niet aangeraakt |
| **D2** — C1 t/m C6 gesloten; geen limiet defaultet naar oneindig | rood | **groen** | `tests/unit/test_live_limits_are_sovereign.py` (12) |
| **D3** — DI-7 gesloten | rood | **groen** | `tests/unit/test_background_tasks_are_held.py` (4); DI-7 staat als gesloten in het register |
| **D4** — `AlertSeverity` kent `HALT`; een `HALT` schakelt uit zonder te vragen | rood | **groen** | `tests/unit/test_alert_halt_severity.py` (11) |
| **D5** — drempels uit `conf/monitoring/`, gehasht vóór de klok | rood | **groen** | `conf/monitoring/default.yaml`, `apps/freeze_monitoring.py`, `tests/unit/test_monitoring_thresholds_are_frozen.py` (6) |
| **D6** — de vier `test_chaos.py`-skips | rood | **nog rood** | ongewijzigd; zij vragen de live-engine-integratie uit D1 |
| **D7** — dagelijkse bit-identieke pariteit | niet toetsbaar | **nog niet toetsbaar** | §6.1 en §6.2 staan onaangeroerd; een pariteitstest zou de configuratie meten |
| **D9** — `HALTED` overleeft een procesherstart | rood na 24 uur | **groen** | `tests/unit/test_live_halt_is_irreversible.py` (10), inclusief de 25-uurs-regressie |

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
