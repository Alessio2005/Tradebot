# PROJECT STATE — Tradebot

> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-4),
> **herzien op 2026-09-02** na een externe review over de volledige Stage
> C/D/E-boom. Elk cijfer hieronder is op die datum gemeten, niet uit een eerder
> rapport overgenomen. Waar iets niet is gemeten, staat dat er.

> **Lees dit eerst als je het project overneemt.** De review vond elf defecten
> in het werk van deze fase, waarvan vijf in het haltpad — inclusief een halt
> die bij een gemeten degradatie niet afging terwijl drie exit-criteria er
> groen boven stonden. Zij zijn gerepareerd en met gedragstests afgedekt
> (`reports/phase7_8_exit_report.md` §4.9–§4.14). De les die je meeneemt: in
> deze repository is een groen criterium pas bewijs als de bijbehorende test
> rood kan worden om de reden waarvoor het criterium bestaat. Een AST-test die
> de vorm van een aanroep controleert, is dat niet.

> **Doel van dit document.** Eén plek waar staat wat dit platform is, wat er
> aantoonbaar waar is, wat er aantoonbaar NIET waar is, welke besluiten
> openstaan, en wat de volgende drie stappen zijn. Geschreven voor iemand die
> het project overneemt en die geen van de voorgaande fasen heeft meegemaakt.

---

## 1. Wat dit is

Een onderzoeks- en handelsplatform voor crypto-perpetuals, gebouwd rond één
principe: **een resultaat telt pas wanneer het de poort is gepasseerd die het
had kunnen tegenhouden.** Vrijwel alle infrastructuur in deze repository bestaat
om claims te kunnen weerleggen, niet om ze te produceren.

Het universum is zes Bybit-perpetuals — BTC, ETH, SOL, AVAX, LINK, DOT — op
dagbars, 1.743 bars van 2021-11-15 tot 2026-08-23, gecertificeerd in een
point-in-time store met per reeks een hash.

## 2. Wat aantoonbaar waar is

| | Bewijs |
|---|---|
| De data is point-in-time gecertificeerd | `artefacts/governance/data_hashes.json`; elke run draagt de hashes van de reeksen die hij las |
| Er is één authoritative backtest-engine | `backtest/engine.py`; de vier overlappende engines zijn in Phase 5 verwijderd na een pariteitsbewijs (`tests/integration/test_engine_parity.py`) |
| De risicolaag is soeverein in de BACKTEST | `risk/engine.py` → `RiskDecision`; `execution/order_router.py` weigert te bouwen zonder |
| Kill switches zijn onherroepelijk | `risk/kill_switches.py::HaltStore`; opheffen vereist een operator én een motivering |
| Kill switches gáán ook daadwerkelijk af | `tests/unit/test_external_monitors_can_actually_halt.py` rijdt een degradatie door een echte monitor een echte breaker in; vóór 2026-09-02 was dit alleen op vorm getoetst |
| De halt-state hangt niet aan de werkdirectory | `live/circuit_breaker.py` leidt beide paden af van `__file__`; subprocestest vanuit een andere map |
| Elke hypothese is vooraf geregistreerd | `artefacts/governance/preregistration_*.json`, bevroren vóór de eerste fit |
| `M` wordt eerlijk geteld | `artefacts/governance/hypothesis_ledger.json`, append-only, **M = 2776** |
| Drie fase-6 hypothesen zijn beslist | H1, H2, H3 — alle drie met een ledger-amendement en een rapport |
| Monitoringdrempels staan vast vóór de klok | `artefacts/governance/monitoring_config_hash.json` |

## 3. Wat aantoonbaar NIET waar is

Dit hoofdstuk is belangrijker dan het vorige.

### 3.1 Geen van de onderzochte modellen is gepromoveerd

| Hypothese | Oordeel | Reden |
|---|---|---|
| **H1** — GARCH-familie vs. EWMA(0,94) | `UNPROVEN` | De range-proxy meet aantoonbaar een andere grootheid dan de modellen voorspellen (`mean(proxy)/mean(r²)` = 1,27–2,24). EWMA blijft productie-estimator |
| **H2** — Filtered HMM vs. M0 | `UNPROVEN — insufficient data` | De bezetting van de zeldzaamste toestand loopt van 5,0 tot 28,0 observaties per fold tegen een eis van 100 |
| **H3** — CatBoost als meta-labeler | `ARCHIVED` | OOS-AUC 0,4768–0,4842, alle zes onder 0,50; conservatieve ondergrens max 0,4113 tegen een drempel van 0,58 |

**De vier Phase 3-tracks zijn netto negatief.** De ongefilterde baseline haalt
een netto OOS Sharpe van −0,0998 over 1.200 bars. Er is op dit moment geen
enkel model in deze repository waarvan is aangetoond dat het geld verdient.

### 3.2 De live-keten is niet de backtest-keten

Gemeten 2026-09-01 (`reports/phase7_divergence_map.md`): `live/` importeert 18
`tradebot`-modules, de Phase 5-keten 19, en de doorsnede is **drie** — waarvan
er één domeinlogica is.

> **Herzien 2026-09-04 (Phase 9, stap 15).** De zin die hier stond — *"er zijn
> nul verwijzingen naar `RiskDecision`, `risk/engine.py` of
> `risk/kill_switches.py` in `live/` of `oms/`"* — is sinds Stage D niet meer
> waar, en het verschil is precies het punt:
>
> | Symbool | imports in `live/` + `oms/` |
> |---|---:|
> | `RiskDecision` | **0** |
> | `RiskEngine` | **0** |
> | `execution/order_router.py` | **0** |
> | `risk/kill_switches.py::HaltStore` | 1 — `live/circuit_breaker.py:29` |
> | `risk/limits.py` | 1 — `live/execution_controller.py:39` |
> | `risk/daily_loss_governor.py` | 1 — `live/engine.py:41` |
>
> **Het HALT-pad staat op de soevereine laag; het BESLUIT-pad niet.** Dat is
> nauwkeuriger dan "nul verwijzingen" en het verklaart waarom §2 hierboven
> tegelijk kan zeggen dat de halt-state niet aan de werkdirectory hangt.
>
> Scherper nog, en gemeten met dezelfde scanner voor beide ketens: van de **21
> modules / 6.173 LOC** in de pakketten `live/` en `oms/` staat er **nul** in de
> authoritative afsluiting.

Concreet betekent dat:

* de live-orderweg (`oms/router.py::place_order`) vraagt geen risicobesluit;
* live prijst marktimpact met `eta = 0,1`/`0,142` waar de backtest de
  gekalibreerde `2,991922` gebruikt, en met 5 bp vaste slippage tegen 1 bp
  aangenomen half-spread;
* er zijn **twee featurestacks**: `features/pipeline.py` voor live, de Phase
  2-registry (`features/registry.py` → `features/base.py`) voor de authoritative
  keten.

Zolang dit staat, kan de dagelijkse bit-identieke pariteitstest van
exit-criterium D7 niet slagen, en meet elk verschil dat hij zou vinden de
configuratie in plaats van het gedrag.

**Herzien 2026-09-02, en het is fundamenteler dan hierboven.** De twee ketens
draaien niet op dezelfde bars en handelen niet hetzelfde boek:

* live draait op **5-secondebars** (`FeedConfig.bar_seconds = 5`), de Phase
  5-keten op **dagbars**. Een dag telt 17.280 5s-bars; de bijbehorende
  annualisatiefactor is 6.307.200 tegen de 365 in `conf/model/volatility.yaml`,
  een factor **131,5 op sigma**. `vol_target` deelt door precies dat getal.
* live handelt **vijf** namen (geen BTCUSDT), de policy is voor **zes**
  geschreven.

D7 is daarmee niet zwak toetsbaar maar **niet gedefinieerd**: er is geen paar
observaties waarover "bit-identiek" iets betekent. En het keert de volgorde om —
`RiskEngine` aansluiten vóór deze twee besluiten levert een limietstelsel dat
er aangesloten uitziet en verkeerd rekent. Zie `reports/phase7_divergence_map.md`
§6.3 t/m §6.5.

### 3.3 Het universum bestaat uit overlevers

DI-15: de publieke Bybit-API levert uitsluitend nog-verhandelde instrumenten
(gemeten: 833 van 833 met status `Trading`, nul delistings). Elke
momentum-achtige claim op dit universum is daarmee optimistisch vertekend. Het
contract dat delistings zou verwerken is gebouwd; de data ontbreekt.

### 3.4 Het runbook beschrijft een systeem dat niet meer bestaat

`docs/runbook.md` gaat over het Wave-tijdperk. Stage D-4 schrijft een
herschrijving voor, getoetst doordat een tweede persoon het systeem er
uitsluitend op start, halteert en herstart. **Die herschrijving is niet gedaan.**

### 3.5 De Phase 5-ladder was op een prijs-only P&L gemeten — gerepareerd in stap 1B

**Gemeten in fase 10, stap 1.7; gerepareerd in stap 1B.** Op perpetuals is de
P&L per contract `prijsrendement + funding`
(`docs/MEASUREMENT_CONTRACT.md` §7). De boekhoudlaag doet dat correct:
`backtest/accounting.py::apply_funding` boekt `qty * mark_price * rate`, dus
**met het teken van de positie**, en beide invarianten van de dubbele
boekhouding sluiten. Dat is geverifieerd met twee nieuwe tests in
`tests/unit/test_accounting.py`, elk bewezen rood-kunnend via een mutatie van
de fundingregel.

De **aanvoer** was echter niet contractconform. `apps/run_phase5_baseline.py`
laadde prijzen, volatiliteit, ADV en volume, maar **geen funding**, en gaf
`run_all_layers` een fundingpaneel van nul mee. In
`artefacts/baseline/phase5_revaluation.json` stond daardoor voor elke track
`cost_funding = 0.0`.

De data ontbrak niet: alle zes reeksen `crypto/funding/<symbool>/8h` zijn
hash-gecertificeerd aanwezig in `artefacts/governance/data_hashes.json`. Het
was bedrading, geen databeperking.

**Reparatie (stap 1B).** `src/tradebot/data/funding_panel.py::daily_funding_panel`
laadt de gecertificeerde 8h-fundingreeks via `features/base.py::load_certified_series`
en SOMMEERT — per dagbar — de afrekeningen die binnen die bar zijn gevallen
(tot drie 8h-settlements per dagbar). De voor de hand liggende kortere weg,
hergebruik van de bestaande `asof_join(direction="backward")`-koppeling in
`features/microstructure.py::build_certified_micro_frame`, is bewust NIET
gebruikt: die levert de laatst bekende rate op het beslismoment — het juiste
antwoord voor een feature, maar het zou hier ongeveer een derde van de
werkelijke funding hebben geboekt. `apps/run_phase5_baseline.py::load_market`
geeft het gesommeerde paneel nu door in plaats van de nul-DataFrame.

**Gemeten uitkomst.**

| | prijs-only (§5.2, vóór stap 1B) | met funding (na stap 1B) | Δ netto Sharpe |
|---|---:|---:|---:|
| L3 `long_only_equal_weight` | −0,695 | −0,719 | −0,024 |
| L3 `long_only_risk_parity` | −0,721 | −0,745 | −0,024 |
| L3 `xs_momentum_equal_weight` | −0,103 | −0,168 | −0,065 |
| L3 `xs_momentum_risk_parity` | −0,331 | −0,369 | −0,038 |

`cost_funding` op L3 ging van `0,0` (alle vier tracks) naar 261,7 / 279,7 /
761,4 / 397,4 respectievelijk (equity-eenheden). **L0, L1 en L2 zijn bit-identiek**
aan de prijs-only meting op alle vier tracks — funding raakt uitsluitend L3,
zoals de blast radius van stap 1B voorschreef. Eenhedencontrole: gemiddelde
`|dagelijkse funding|` per symbool op de gecertificeerde store ligt tussen
2,6e-4 en 5,5e-4 — orde 1e-4, zoals verwacht voor 8h-perp-funding gesommeerd
naar dagen. De venue-cap `funding_cap_abs = 0,02` bond 8 keer op 22.892
boekingen (0,035 %), veroorzaakt door enkele extreme fundingpieken (SOLUSDT tot
−0,124 op één dag) en niet door een eenhedenfout.

**Duiding.** De verschuiving is materieel maar verandert geen enkele
kwalitatieve conclusie van §5.2: elke track was al negatief of marginaal op L3
vóór funding, en blijft dat — nu met de juiste reden erbij. Funding werkt op
dit long-only boek structureel één kant op (F9 falsificeert shorts
onvoorwaardelijk) en verergert dus uitsluitend de reeds negatieve L3-uitkomst.
Zie `.superpowers/sdd/fase_10_herstart_dagbars/stap-1B-report.md` voor de
volledige zestien-cellen-tabel en de per-track uitsplitsing.

## 4. Waar de fasen staan

| Fase | Status |
|---|---|
| 0–5 | afgerond, met exit-rapport per fase in `reports/` |
| 6 — advanced research | **stap 14 (HRP) open**; stappen 0–13 en 15 gedaan voor H1/H2/H3 |
| 7/8 Stage A — fundament | afgerond |
| 7/8 Stage B — governance (D-1) | afgerond |
| 7/8 Stage C — Phase 6 afmaken | **grotendeels**; H1/H2/H3 beslist, HRP open |
| 7/8 Stage D — productie | **deels**: D2, D3, D4, D5, D9 groen; D1, D6, D7 open; de 60-daagse klok is niet gestart |
| 7/8 Stage E — oplevering | **deels**: E-2 t/m E-4 gedaan; E-1 en E-5 zie hieronder |
| 9 — opschoning, consolidatie & testmassa | **afgerond**, zie `reports/phase9_exit_report.md` |

## 5. Openstaande besluiten

Dit zijn keuzes die iemand met mandaat moet maken; ze zijn niet technisch op te
lossen.

1. ~~**Koopt dit project delisting-historie?** (DI-15)~~ **Beslist in fase 10 —
   zie AD-23** (mandaatbesluit B-2, `docs/MANDATE.md`). Het meetdomein is
   vastgelegd op drie gecertificeerde bronnen; een vierde bron toevoegen is
   geen openstaand besluit meer maar een domeinwijziging, en die valt buiten
   het mandaat. De survivorship-vertekening wordt daarmee een **permanent
   gelabelde beperking** (§3.3) in plaats van een inkoopvraag.
2. ~~**Koopt dit project intraday-data?** (DI-18)~~ **Beslist in fase 10 — zie
   AD-23** (mandaatbesluit B-2). Een grootheid die een fijnere waarneming dan
   één bar per dag vereist, is geen uitgestelde vraag maar een niet-bestaande
   vraag. DI-18 vervalt daarmee als besluit; H1 is **gesloten, niet
   geblokkeerd** (`reports/GARCH_VS_EWMA_COMPETITION.md` §9).
3. ~~**Wordt `live/` aangesloten of herbouwd?**~~ **Beslist in fase 10 — zie
   AD-22** (mandaatbesluit B-1). De handelsklok volgt de meetklok:
   `FeedConfig.bar_seconds` vervalt en er is nog één barresolutie. Welke van de
   twee routes — aansluiten of herbouwen — dat oplevert, is uitvoering en wordt
   in stap 16 uitgevoerd en in AD-26 vastgelegd; het is geen mandaatvraag meer.

   > De derde mandaatbeslissing, **B-3** (de ledger-reset, **AD-24**), stond
   > niet in deze lijst omdat zij hier nooit als besluit is opgeschreven — de
   > rekenkundige onbereikbaarheid van de promotiepoort bij `M = 2776` is in
   > `docs/DEFERRED_ISSUES.md` blijven staan als eigenschap in plaats van als
   > keuze. Zij is nu een besluit met een protocol (R1-R8).
4. **Wat draait er in de 60 dagen?** D-6 eist dat dit vooraf wordt vastgelegd.
   Omdat niets is gepromoveerd, is de champion de Phase 3-baseline plus de
   soevereine risicolaag — een systeem waarvan bekend is dat het geld verliest.
   Dat is een geldige keuze voor een operationele test, maar het moet expliciet
   zo worden geregistreerd zodat niemand het later als rendementsverwachting
   leest.
5. ~~**DI-3 en DI-4** (bestanden boven de LOC-limieten) schuiven sinds Phase 0
   door.~~ **Afgehandeld in Phase 9.** Beide zijn herbevestigd met een
   gecorrigeerde meting — DI-3 noteerde 2 bestanden >800 LOC waar er 9 staan,
   DI-4 noteerde 18 van 40 apps waar het er 27 zijn — en de ratchet die niet
   bestond is gebouwd: `scripts/check_file_size.py` met een cap per bestand,
   afgedwongen door `.github/workflows/inventory.yml` en met bewijs dat hij
   rood wordt.

## 6. De volgende drie stappen

> **Bijgewerkt 2026-09-04 na Phase 9.** De volgorde hieronder is ongewijzigd —
> de opruimfase heeft geen enkele openstaande onderzoeksvraag geraakt. Wat zij
> wel heeft veranderd, is dat de oppervlakte nu **gemeten** is in plaats van
> geschat, zodat stap 1 hieronder op cijfers kan steunen:
>
> * `docs/CODE_REGISTER.md` — 290 modules, 71.025 LOC, 60,6 % LOC-gewogen
>   dekking, met per pakket het entrypoint dat het bereikt en per module onder
>   de drempel een ratchet met datum en eigenaar.
> * `reports/phase9_inventory.md` — elke module, app en script met een verdict.
> * `.github/workflows/inventory.yml` — de poort die voorkomt dat er ongemerkt
>   een onbereikbare module bij komt.
> * `docs/runbook.md` §0.7 — alle 53 entrypoints (40 apps, 13 gereedschappen)
>   met LOC, DVC-stage en doel, plus de 48 onderzoeksbestanden via
>   `research/README.md`. Het aantal dat nergens werd genoemd is van **36 van
>   98** naar **0** gegaan.
>
> **Eén nieuwe post die vóór stap 2 hoort:** de dekking staat op **56,64 %**
> tegen een drempel van 70, met **108 modules / 31.903 LOC** onder een ratchet
> die op 2027-03-04 wordt herzien. Zij is in deze fase gestegen (van 55,82 %,
> met 89 statements méér gedekt), maar het gat naar de drempel is werk op
> zichzelf en geen bijvangst van een opruiming. Het raakt stap 1 rechtstreeks:
> `live/` draagt 38,5 % en `oms/router.py` — de weg waarlangs in productie een
> order de deur uit gaat — **27,9 %**.

0. **Besluit over de barresolutie en het universum** (§3.2, herziening
   2026-09-02). Dagbars of intraday? Vijf namen of zes? Dit zijn besluiten van
   de eigenaar van het risicoregime en geen implementatiekeuzes. Stap 1 en 3
   hangen erachter; alleen stap 2 kan zonder.
1. **Sluit §3.2.** Sluit `live/` aan op `execution/order_router.py`,
   `risk/engine.py` en de Phase 2-featureregistry — met één vol-estimator in de
   eenheid van besluit 0. Dit gaat vóór alles wat met pariteit of de klok te
   maken heeft, want een pariteitstest op twee verschillende ketens meet de
   configuratie.
2. **Maak Phase 6 stap 14 af** (HRP tegen Inverse Volatility, turnover-
   gecorrigeerd, onder de research-gate). Dat is de laatste openstaande
   deliverable van Stage C.
3. **Herschrijf het runbook en laat een tweede persoon het uitvoeren** (D-4,
   exit-criterium D12). Pas daarna heeft het starten van de 60-daagse klok zin.

## 7. Hoe je dit verifieert

```bash
python -m pytest tests -q          # verwacht: 0 failed, 4 xfailed, 0 xpassed
python -m ruff check src/          # CI-scope, hoort schoon te zijn
python apps/freeze_monitoring.py --check
dvc status                          # welke stages zijn stale
```

De vier verwachte `xfail`s zijn de pre-geregistreerde killgates op `cm_carry` en
`cm_tsmom` in `tests/killgates/test_expansion_killgates.py`; zie
`reports/phase0_fallback_register.md`.

> **Gewijzigd in fase 10, stap 1.8 (Q11).** Deze vier faalden tot dan toe
> permanent, en daarmee was **rood de normale toestand van de suite** — een
> echte regressie viel niet meer op. Zij staan nu op
> `@pytest.mark.xfail(strict=True)` met een `reason` die hun registerregel
> noemt. De suite is groen, en `strict=True` betekent dat een killgate die
> begint te **slagen** de suite rood maakt: precies de gebeurtenis die je wilt
> zien, want die betekent dat het gefalsifieerde gedrag is teruggekeerd of dat
> de killgate is uitgehold. Het aantal blijft vier; alleen de kleur van
> "normaal" is veranderd. Minder dan vier `xfail`s betekent dat een killgate is
> uitgeschakeld, niet dat er iets is opgelost.
