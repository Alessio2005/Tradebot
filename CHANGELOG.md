# CHANGELOG

> Gereconstrueerd 2026-06-10 (Wave 20): het originele bestand bleek leeg.
> Bronnen: docs/TRUE_ALPHA_AUDIT_2026-06-08.md, docs/data_dictionary.md,
> docs/WAVE_LOG.md, code-annotaties (AUDIT-tags in src/).

## Replay op echte DoltHub-optieketens; synthetische markt weg als replaybron (2026-11-27)

Op verzoek van Alessio: de synthetische marktdata (TSTA/TSTB) is geen replay- of demobron meer.
Replay en sweep draaien op echte einde-dag-ketens uit de openbare DoltHub-database
`post-no-preference/options` (februari 2019 t/m november 2026). Alessio accepteert dat dit niet
overal precies is; hieronder wat wel en niet klopt.

Nieuw:
- `options/data/dolthub.py` en `python -I -m tradebot.options dolthub --cache DIR --out DIR
  --start D --end D [--symbols ...] [--offline] [--rate R]`. Eén SQL-query per dag op de
  DoltHub-API, met retries; de ruwe rijen per dag in een cache (`YYYY-MM-DD.csv`, of `.empty`),
  die hervat waar hij stopte. Daarna bouwt hij een `Market` in het bestaande formaat.
- Koersen en dividenden: Yahoo's chart-API (`query2.finance.yahoo.com`), teruggerekend naar de
  koersen van toen (Yahoo corrigeert voor splits; DoltHub heeft de echte strikes, bv. AAPL vóór
  de 4:1-split van 31-08-2020). yfinance zelf werd vanuit de cloud geweigerd (rate limit op de
  crumb), de chart-API niet.
- Earnings: SEC EDGAR (`data.sec.gov`), de datum van elk 8-K met item 2.02. Dat is de dag
  van het persbericht (bij AAPL en MSFT na de slotbel). ETF's hebben geen earnings. Alleen
  AAPL en MSFT hebben een CIK in de tabel; een andere naam krijgt geen earnings (met een melding).
- `Market.daily_snapshot` ("15:15"): heeft een markt één snapshot per dag, dan zet
  `replay_config` de scantijd en het sluitmoment vóór expiratie op dat tijdstip. Op een ander
  tijdstip is er die dag geen keten. Snapshots zonder opties (alleen de koers) worden nu ook
  opgeslagen (`empty_snapshots` in `market.json`).
- `summary.json` heeft `buy_hold`: hetzelfde kapitaal op de eerste dag in SPY (slotkoers),
  dividenden als kas op de ex-datum, met dezelfde statistiek als het boek. Alleen als SPY in
  de markt staat.
- Kleine echte testmarkt `tests/fixtures/options/dolthub_mini` (SPY, AAPL, MSFT; 3 januari t/m
  28 februari 2023). `options-paper-smoke.yml` en `make options-smoke` spelen die af; `make
  options-replay` bouwt de hele DoltHub-markt en speelt hem af.
- De synthetische generator staat nu in `tests/unit/options/synthetic_market.py`, alleen als
  deterministische testfixture. `_expiries` en `_quote` zijn verhuisd naar `data/historical.py`
  (de IBKR-modelmarkt gebruikt ze). De CLI `python -m tradebot.options.data.synthetic` is weg.

Beperkingen van de data (keuzes van mij, die Alessio kan terugdraaien):
- **Eén snapshot per dag, op 15:15.** De quotes zijn van rond de slotkoers en de koers bij de
  keten is de slotkoers uit Yahoo. Alles van een dag gebeurt dus op één prijs; er is geen
  intraday-pad. Dat is iets optimistisch: een slotquote om 15:15.
- **Niet elke dag een keten.** In 2019 één per week (de zaterdag, met de quotes van vrijdag:
  die schuift naar vrijdag), daarna meestal maandag, woensdag en vrijdag. 1.336 data met
  ketens. Op elke handelsdag is er wel een snapshot.
- **Weinig expiraties en strikes.** Meestal drie expiraties (~2, ~4 en ~6-8 weken) en grove
  strikes; verre vleugels ontbreken vaak. Een strategie die een precieze delta of vleugel
  zoekt, vindt dan niets of iets verders.
- **Aanvullen.** Een contract dat uit de keten verdwijnt (door de rollende selectie van DoltHub,
  of op een dag zonder keten), wordt tot `fill_days` = 60 handelsdagen aangevuld, binnen
  |K/S − 1| ≤ 20 %. De quote is QuantLib (BAW, of FD bij een dividend) op de koers van vandaag,
  met de laatste DoltHub-IV en -spread van dat contract. Zo'n quote heeft volume en open
  interest 0: de filters laten hem niet toe voor een nieuwe entry, alleen een open positie
  kan erop waarderen en sluiten. Zonder dit konden posities niet sluiten en liepen ze naar
  expiratie of assignment (gezien in de proefreplay van 2019). Verder buiten de geld rekent de
  settlement op intrinsieke waarde.
- **Geen volume of open interest** in de bron: op de echte quotes vast 1.000 en 5.000, boven
  de filters. Fills zijn die van de paper-broker op bid/ask: optimistischer dan in een dunne
  echte keten.
- **IV en greeks van DoltHub** worden niet gebruikt voor de engine: die rekent ze uit de quotes
  met QuantLib (Amerikaans, met dividenden), zoals altijd. Alleen het aanvullen gebruikt de
  DoltHub-IV.
- **QQQ en IWM staan niet in DoltHub.** De replay draait op SPY, AAPL en MSFT; de config blijft
  ongewijzigd en `replay_config` neemt het universum van de markt.
- **Rente**: één vaste 3 % voor de hele periode (`--rate`), terwijl die in 2020-2021 bijna 0 was.
- **Oppervlak op echte ketens** (`pricing/surface.py`): een expiratie met butterfly-arbitrage
  valt nu alleen zelf uit het oppervlak (in `reason`); de rest blijft bruikbaar. Eerder legde
  één foute expiratie de hele keten stil. Op de DoltHub-ketens van 2023 was dat 83 % van de
  dagen, waardoor het vol-regime UNKNOWN bleef en er bijna niets opende; nu is 93 % van de
  echte ketens bruikbaar. De kalendercontrole blijft gelden voor wat overblijft. Dit geldt
  ook voor de IBKR-paper-dienst. Dit is de aanbevolen optie die aan Alessio is voorgelegd, en
  hij kan hem terugdraaien.
- **Aandelen uit een assignment worden afgebouwd** (`engine/service.py`, besluit van Alessio):
  bij een assignment of uitoefening buiten de wheel worden de geleverde aandelen een
  aandelen-leg van de positie. Ze worden geboekt op de slotkoers waartegen `settle_usd` al
  rekent. Is de positie dicht, dan sluit `_unwind_shares` ze de volgende ronde; anders gaan
  ze mee als de positie sluit. Eerder bleven ze los in het boek staan. In de DoltHub-replay liet
  een vroege assignment van een short SPY-call vóór ex-dividend daardoor −100 SPY open van 2020
  tot 2026 (eind-equity −11.000 USD). Dit geldt alleen voor de paper-broker van de
  replay: `execution/ibkr.py` geeft alleen FILL-uitvoeringen door, dus assignment en uitoefening
  via IBKR-paper zijn nog niet gebouwd en daar gebeurt dit (nog) niet (review 2026-11-27, N31).
  Het alarm
  `options_assignment` blijft. De risicopoort (`risk/gate.py`) laat daarom een sluitorder op
  aandelen toe op elk symbool, ook buiten `share_eligible` (zoals SPY). Opbouwen blijft
  beperkt tot `share_eligible`, en de sluitorder moet nog steeds precies de eigen leg afbouwen.
- **Netwerk**: de cloudomgeving moest `www.dolthub.com`, `query2.finance.yahoo.com` en
  `data.sec.gov` toelaten (met akkoord van Alessio).

### Reviewfixes engine, assignment en paper-broker (review 2026-11-27, strategieën N2-N31)

- **N2**: de aandelen uit een assignment worden alleen voor `wheel`, `covered_call` en
  `cash_secured_put` niet als aandelen-leg geboekt en niet teruggekocht: ze blijven losse aandelen
  (CSP → covered call; `CoveredCall.scan` ziet geen extra lot, geen naakte call) en geven geen
  `options_assignment`-alarm. Een geassignde CSP zet de kostprijs (`cost_basis`) als die er nog
  niet is.
- **N7**: een gedeeltelijke assignment of uitoefening rekent per positie `min(|q|, rest)` af en
  verkleint de leg (aandelen-leg voor het geleverde deel). Passen de legs daarna niet meer bij
  `units`, dan wordt `units` de grootste gemene deler, zodat een sluitorder klopt.
- **N8**: `position_closed` na een assignment leest `hedge_realized_usd` uit de bijgewerkte
  positie (na het boeken van de geleverde aandelen).
- **N6**: de paper-broker boekt dividend op aandelenposities (`PaperBroker.pay_dividends`, aan het
  einde van de dag vóór de ex-datum, na `early_assign`; long ontvangt, short betaalt). De engine
  boekt hetzelfde bedrag op de aandelen-legs (`hedge_realized_usd`) en journaalt `dividend_paid`
  en `dividend` (per positie); losse aandelen (wheel) raken alleen de equity. Buy-and-hold in
  `report.py` blijft ongewijzigd (kas op de ex-datum).
- **N29**: `_unwind_shares` verrekent eerst tegengestelde aandelen-legs van gesloten posities
  op hetzelfde symbool tegen de spot (`hedge_netted`, zonder commissie), zodat alleen het netto
  aantal verhandeld wordt.
- **N17** (engine-deel): `StrategyPosition.assigned` (bewaard in de staat) markeert een positie
  waarvan een assignment of uitoefening aandelen leverde; `_manage` sluit zo'n positie met rest
  meteen (STOP, reden `assigned`), voor elke strategie.
- **N31**: de claim "geldt ook voor IBKR-paper" hierboven is gecorrigeerd.

## Opruimen en optimaliseren, plus de risicogaten uit de review (2026-11-27)

Opruimen (geen gedragsverandering):
- Dependencies: `hydra-core` en `python-dateutil` uit de harde lijst (nergens geïmporteerd;
  dateutil komt via pandas), `pytest-asyncio` uit `[dev]` (geen async tests).
  `pytest-randomly` staat nu ook in `[dev]` (hij stond al in de lock, dus `.[dev]` gaf een
  andere testvolgorde dan CI). Uit `requirements-dev.lock`: pytest-asyncio en de dvc-restanten
  flufl.lock, ruamel.yaml en zc.lockfile. Locks met de hand bijgewerkt, geen andere pins
  verschoven. Verwijzingen naar ontbrekende docs (DEPENDENCY_CONTRACT, DEFERRED_ISSUES,
  RUNBOOK, `scripts/audit_fallbacks.py`) uit pyproject en de dev-lock gehaald.
- `params.yaml` weg: een DVC-shim zonder `dvc.yaml` en zonder lezer. De cache-sleutel van
  `paper-trade-ci.yml` hangt nu aan `conf/conf_config.yaml`.
- Dode options-code weg: `risk.scenarios.run_scenarios`/`ScenarioResult`, `Market.bars_until`,
  `ReplayFeed.times`, `PricerSettings.risk_free_rate`, de ongebruikte parameter `t` van
  `surface._forward`, de tweede `RiskGate` in de engine (`risk_gate`, alleen gebruikt voor
  `beta` en `scenarios`, die gelijk zijn aan die van `gate`). De tests rekenen hetzelfde na
  via `position_values`.
- Crypto niet meer in het options-importpad: `SingleInstance`/`Deferred` verhuisd naar
  `utils/single_instance.py` (`carry.safety` exporteert ze opnieuw, carry-code ongewijzigd);
  `tradebot.risk` en `tradebot.schemas` laden hun submodules lui (PEP 562). Gevolg:
  `import tradebot.options.cli` van 2,3 s naar 1,2 s, en de options-dienst laadt geen
  sklearn, hmmlearn of het carry-pakket meer.
- Risicopoort: de stringvergelijking `status.value != "closed"` vervangen door de enum.
- `options-paper-smoke.yml` draait nu ook op pull requests die options, `risk/` of de config
  raken: tot nu toe kon een PR de replay breken zonder rode check.
- Makefile: `install`/`install-dev` uit de lockfiles (waren ongepind), nieuw `make options-replay`.
  README: de fase-7-tekst bijgewerkt.

Risicogaten uit de review van 2026-11-27 (P1, punten 1-4); keuzes van mij:
- **Lopende entries tellen mee** (`Book.pending_entries`, alle niet-gevulde entry-orders,
  ook met beperkt risico). De poort telt ze als open positie voor `max_open_positions`,
  `max_new_positions_per_day` (die van vandaag), de Greeks, de scenario's en de stress; en
  `Book.open_max_loss` telt ze mee voor de sizing per onderliggende. `pending_undefined` is nu
  een afgeleide (de oneindige onder `pending_entries`).
- **HALT/UNWIND annuleert lopende entries** in de walk (`order_cancelled`, reden `halt`), in
  plaats van ze door te schuiven naar natural. Sluit-, rol- en bijstuurorders lopen door.
- **`size()` naast een open undefined-risk positie** geeft 0 eenheden in plaats van een
  OverflowError (`math.floor(-inf)`).
- **PDT-guard op elk uur**: onder 25.000 USD en met 3 daytrades in 5 handelsdagen geen entry
  meer, ook om 10:00 en 13:30. De 15:00-grens uit detailontwerp 8.5 greep nooit in omdat de
  scans eerder liggen. Afwijking van het ontwerp.
- **Gamma scalp vóór de long straddle** in de strategievolgorde. De scan neemt per
  onderliggende per dag één entry, in registervolgorde; LongVol (dezelfde ATM-straddle zonder
  hedge) pakte elke hedgebare naam eerst, dus gamma scalp kwam nooit aan bod en de hedger deed
  niets. Alleen van belang met `gamma_scalp.enabled`.

Gemeten, niet veranderd (beslissing voor Alessio):
- De replay is QuantLib-gebonden: in een profiel (1 worker) gaat 117 van de 140 s naar
  `Instrument.NPV` van de FD-engine (contracten met een dividend in de looptijd: IV-solve en
  bump-Greeks). De rest van de engine samen is minder dan 15 %, dus Python-optimalisaties
  geven hooguit een paar procent. Wat wel scheelt: het FD-rooster. 100×200 in plaats van
  200×400 is 3,3× sneller per prijs met een fout van hooguit 0,27 cent per aandeel (tegen een
  referentie van 800×1600), kleiner dan één tick. De standaard blijft 200×400.
- `limit_anchor: fair` annuleert entries op `max_edge` omdat de grens tussen twee ticks valt:
  op de synthetische spreads (halve spread 3 cent) is fair = mid, de grens ligt op
  0,5 × 3 = 1,5 cent en de walk stopt na 1 tick, terwijl de paper-broker pas vult bij
  0,40 × 3 = 1,2 cent, dus 2 ticks. Het anker werkt zoals ontworpen; het is de combinatie
  van `max_edge_frac`, de tick en het fill-model. Zelfde effect met het mid-anker op
  `max_concession` (0,6 × 3 = 1,8 cent).

## Options: punten uit de eindreview (2026-11-27)
- Risicopoort, aandelen (`review_stock`): een sluitorder moet tegengesteld zijn aan de eigen
  aandelen-leg van de gesloten positie (`Book.closing_shares`, niet het totaal van het symbool)
  en mag er niet groter dan zijn, anders `SafetyBreach`. Nieuwe check `stock_notional_usd`: |aandelen-legs na de order| × spot ≤
  `hedging.max_notional_frac` × kapitaal (het plafond van de hedger, nu ook in de poort), tenzij
  de order de aandelen verkleint. In HALT en UNWIND alleen afbouwen: minder aandelen op die
  onderliggende én minder delta; een hedge vanaf nul gaat in HALT niet meer door (het ontwerp
  zegt "bij een HALT alleen afbouwen"). De hedger stuurt zulke orders in HALT/UNWIND niet eens
  naar de poort, dus geen afgewezen `hedge_order` per ronde.
- `replay` en `sweep` weigeren een niet-lege `--out` (exit-code 2); `--overwrite` wist dan
  alleen de bekende bestanden van eerdere replay- of sweep-uitvoer (`state.json` met
  `mode: replay`, niets onbekends erin), nooit een hele boom, en weigert een map met `HALT`,
  `UNWIND` of `halt/` of met de markt erin. Een tweede run in dezelfde map verdubbelde de
  tellingen en de `stats`.
- Settlement per teken: rekent de broker precies de som van de legs af, dan rekent elke positie
  haar eigen leg af naar haar eigen teken (long `exercise`, short `assignment`); anders alleen
  posities met het teken van de afrekening. Tegengestelde posities op één contract die netto nul
  zijn (de paper-broker ziet ze niet) rekenen na de sluiting van de expiratiedag allebei af
  (`_settle_netted`). Het alarm `options_assignment` gaat ook af bij een uitoefening van een
  gehedgede strategie; het runbook zegt dat die aandelen niet automatisch afgebouwd worden.
- De undefined-risk stresscheck telt ook lopende, nog niet gevulde undefined-risk entries mee
  (`Book.pending_undefined`).
- Gamma-regel: zonder keten of met een leg zonder Greeks meldt hij één keer per positie
  `gamma_unknown` (journal) en `options_gamma_unknown` (alarm) in plaats van stil over te slaan.
- Calendar: forward-vol geeft `fwd:no_gap` als de verre expiratie niet na de nabije ligt (geen
  deling door nul meer); het schema eist dat `long_dte` helemaal na `short_dte` ligt.
- Schema: `short_straddle_strangle.max_per_underlying` hooguit 1 (meer had geen effect);
  `pricing.risk_free_rate` niet negatief (BAW weigert een call bij een negatieve rente).
- Rho gaat nooit onder een rente van nul: bij r < bump eenzijdig vanaf 0, in `greeks` en in de
  kernel gelijk. `greeks(..., BAW)` crashte eerder voor elke call bij r = 0.
- `stats.strategies.<naam>` krijgt `max_drawdown_usd`: de grootste daling van de cumulatieve
  gesloten P&L (in de volgorde van sluiten, vanaf 0); `largest_loss` blijft.
- Standaardreplay: journal byte-gelijk aan dat van `d141c0c` (3 posities, 25.303,60 USD).
  `summary.json` verschilt alleen door de nieuwe sleutel `max_drawdown_usd` (0,0 voor beide
  strategieën).

## Options: pro-strategieën, afronding en documentatie (2026-11-27)
- Eindstand van de branch op de standaard synthetische markt (4 cores): 3 posities, eind-equity
  25.303,60 USD (ongewijzigd), replay in 41 s tegen 171 s voor de code van vóór de branch,
  op dezelfde machine en hetzelfde moment gemeten (4,2×; het doel was 5×). De code van na A4 duurt
  daar ook 41 s: de latere fasen maken de replay niet trager. `stats`: totaalrendement 1,21 %, max drawdown 0,33 %, Sharpe
  4,31, Sortino 17,05; credit_spread 2 gesloten (+116,40), long_straddle_strangle 1 (+187,20).
- Alle nieuwe schakelaars staan uit of op een waarde die de standaardreplay niet raakt. Het
  runbook beschrijft ze per stuk, plus "wat voor een particulier niet haalbaar is". Short
  strangles blijven uit (`risk.allow_undefined_risk: false`); Alessio beslist.
- Het ontwerp (`docs/research/2026-11-26-options-pro-strategies-design.md`, sectie 5) noemt de
  afwijkingen van de bouw: versnelling 4,4× in plaats van ≥5×, vier `risk_decision`-waarden die in
  het vijfde cijfer verschillen door de FD-startwaarde, `limit_anchor` standaard `mid`,
  gamma-close standaard uit, drawdown-rem aan maar niet actief, en settlement dat posities in het
  journal sluit.

## Options: backteststatistiek en sweep (2026-11-27)
- `summary.json` en `ReplayResult` krijgen `stats` (`engine/report.py`): per strategie aantal
  gesloten posities, win rate, gemiddelde en totale P&L, grootste verlies (uit de fills en
  `position_closed` van het journal); voor het boek totaalrendement, max drawdown, Sharpe en
  Sortino op dagrendementen (√252, risicovrij 0). Randgevallen (geen gesloten posities, één
  equity-punt, stdev of neerwaartse afwijking 0) geven lege lijsten of `null`, nooit NaN.
- Nieuw: `python -I -m tradebot.options sweep --markets DIR... --out DIR [--workers N]`: dezelfde
  config op meerdere markten in een procespool (`forkserver`), `sweep.json` met de stats per markt.
  Robuustheidscheck, geen parameter-grid.
- Journal: de events `expiry`, `assignment` en `exercise` krijgen `position_id` en `settle_usd`
  (waarde van de afrekening: 0 bij expiratie, aantal × intrinsiek × 100 bij levering, tegen de
  slotkoers); rekent de laatste optieleg van een positie af, dan volgt ook `position_closed`
  (`reason` = het event, `exit_net` = `settle_usd`). Posities en equity veranderen niet.
- Een uitvoering van de broker (assignment, uitoefening, expiratie) rekent nu elke open positie met
  dat contract af (de hoeveelheid wordt in positievolgorde verdeeld; een rest geeft een alert
  `options_settlement_remainder` en een journal-regel `settlement_alert`), niet alleen de eerste.
  Zonder keten geeft `options_settlement_no_chain` een alert (settle_usd = 0).
- `positions_closed` in summary.json en sweep.json telt nu ook posities die door settlement
  sluiten (die schrijven nu `position_closed`).
- Alleen toegevoegde sleutels: de standaard-replay verandert niet (journal identiek, 3 posities,
  eind-equity 25.303,60 USD).

## Options: sizing op risico en drawdown-rem (2026-11-27)
- `sizing.daily_risk_frac` (standaard `null` = uit): eenheden = min(regel van 7.3,
  floor(daily_risk_frac × kapitaal / σ_unit)), σ_unit = √((Δ·S·σ_1d)² + 2·(½·Γ·S²·σ_1d²)² +
  (V·volvol_1d)²) in USD per eenheid (Greeks uit de keten via `position_greek`, σ_1d uit
  `risk.gamma.sigma_1d`, volvol_1d = `sizing.vol_of_vol_daily_points`, 1,0). Zonder Greeks of
  RV-verwachting: 0 eenheden (`strategies/common.py`).
- `sizing.drawdown_brake: [0.05, 0.10]` (standaard aan, `null` = uit): eenheden × een factor die
  lineair van 1 naar 0,5 loopt tussen die drawdowns van de equity-piek (max van
  `high_water_mark` en het startkapitaal) en daarna 0,5 blijft; naar beneden afgerond.
  `MarketState` draagt daarvoor `equity_usd` en `equity_peak_usd`. De standaard-replay verandert
  niet (grootste drawdown 0,33 %): journal identiek, 3 posities, eind-equity 25.303,60 USD.

## Options: gamma management per positie (2026-11-27)
- `risk.max_gamma_loss_frac_credit` (standaard `null` = uit, aanbevolen 0,5): een positie met
  credit waarvan ½·|Γ|·(S·σ_1d)² (short gamma, σ_1d = `RVForecast.at_horizon(1)` × √(1/252)) boven
  die fractie van de ontvangen credit komt, gaat dicht: CLOSE STOP `gamma_risk`, vóór de regels
  van de strategie (`risk/gamma.py`, `Engine._manage`). Uit = journal ongewijzigd (3 posities,
  eind-equity 25.303,60 USD).

## Options: fair-value-limieten voor entries (2026-11-27)
- `execution.limit_anchor: mid | fair` (standaard `mid`). Bij `fair` begint de limiet van een ENTRY
  op de combo-fair value Σ ratio × fair_leg, met fair_leg de QuantLib-waarde (scan-engine via
  `engine_for(Use.SCAN, ...)`) bij de vol van de gladde smile (`pricing.surface`) voor die strike en
  expiratie, en de mid van de leg zonder bruikbaar smilepunt. Die fair wordt begrensd tot
  [2·mid − natural, natural] en in ons voordeel op de combo-tick afgerond (`execution/orders.py`).
- De walk van zo'n entry concedeert nooit voorbij fair + `execution.max_edge_frac` (0,5) ×
  |natural − mid| richting natural, en nooit voorbij natural; daarna `order_cancelled` met reden
  `max_edge`. Ligt zelfs `limit_0` al voorbij dat plafond (fair voorbij de andere kant van het
  boek), dan gaat de entry niet uit: `entry_skipped` met reden `max_edge`. EXIT, STOP, ROLL en
  ADJUST houden de bestaande concessie.
- Een entry met 2 × |natural − mid| > `execution.max_spread_frac_fair` (0,25) × |fair| wordt niet
  gestuurd: journal `entry_skipped`, reden `spread_vs_fair`.
- Standaard (`mid`) is de synthetische replay ongewijzigd, het journal byte-gelijk (3 posities,
  eind-equity 25.303,60 USD).

## Options: aandelenorders en de delta-hedger (2026-11-26)
- `StockOrder` (purpose `HEDGE`, of `EXIT` bij het sluiten) in de `BrokerAdapter`
  (`submit_stock`): de paper-broker vult op de limiet als die spot ± de halve
  `paper.stock_spread_bps` (2) raakt, commissie `paper.commission_per_share` (0,005) met minimaal
  `paper.min_commission_per_order` (1,00); IBKR stuurt `Stock(sym, "SMART", "USD")` als LMT/DAY.
- `options/hedging.py`: per onderliggende op `universe.share_eligible` de netto delta van de open
  posities van `hedging.strategies`; boven `hedging.band_shares` (20) een order van −round(delta),
  begrensd op |aandelen| × spot ≤ `hedging.max_notional_frac` (0,5) × kapitaal, elke
  `hedging.interval_seconds` (300). De order gaat door `RiskGate.review_stock` (beta-delta; in HALT
  alleen als de absolute delta daalt). Fills worden de aandelen-leg van de positie; een gesloten
  positie gaat in dezelfde ronde terug naar nul aandelen. Journal: `hedge_order`, `hedge_fill`.
- Reconciliatie: op symbolen waarop gehedged is moet het verschil tussen broker en aandelen-legs
  hele lots van 100 zijn. Een covered call telt de hedge-aandelen niet als dekking.
- `hedging.enabled` staat standaard uit (en `hedging.strategies` is leeg tot gamma_scalp er met taak
  B4 bij komt): de standaard synthetische replay is ongewijzigd, het journal byte-gelijk (3 posities,
  eind-equity 25.303,60 USD).

## Options: horizon-RV, EWMA en VRP per looptijd (2026-11-26)
- `RVForecast.at_horizon(days)`: σ²(h) = σ_L² + (σ_0² − σ_L²)·(1 − e^{−κh})/(κh), h = kalenderdagen/365,
  σ_0 de bestaande blend, σ_L de lange RV (de middel-RV zonder lang venster), κ =
  `vol_regime.mean_reversion_kappa` (4,0 per jaar). `VolState.vrp_at(iv, dte_days)` geeft (ratio,
  punten) tegen σ(h).
- `vol_regime.ewma_lambda` (0,94) en `vol_regime.ewma_weight` (standaard 0: regime ongewijzigd). Bij
  gewicht > 0 telt de EWMA-variantie van de dagrendementen mee, met de andere gewichten herschaald.
- Gedragswijziging: `theta_cover` (theta-harvest-poort) rekent met σ(h) voor de DTE van de vroegste
  expiratie van de kandidaat in plaats van met de blend `rv.forecast`. `admit` kreeg daarvoor een
  `today`-parameter.
- Standaard synthetische replay na deze wijziging: nog steeds 3 posities en eind-equity 25.303,60 USD
  (±31 s). De poort laat dezelfde drie kandidaten toe; alleen hun dekking verschoof (1,05 → 1,02 en
  twee keer 1,12 → 1,14), omdat σ(h) bij de DTE van die kandidaten een paar procent van de blend
  afwijkt en de dekkingsgrens (1,0) niet wordt overschreden.

## Options: snellere replay (vectorisatie, startwaarde voor FD, procespool) (2026-11-26)
- `pricing.vectorized` (standaard aan): BAW-contracten in één numpy-kernel die tegen QuantLib
  is vastgepind, en de FD-solve begint met een startwaarde uit die kernel.
- `pricing.workers` (standaard `auto`, hooguit 4) en `pricing.pool_min_contracts`: het
  QuantLib-deel van de keten in een procespool.
- De heartbeat wordt overschreven en ingekort in plaats van bij het openen afgekapt (dat kostte
  in de replay ongeveer 5 s), en een smile bouwt zijn spline één keer in plaats van bij elke
  opvraging. De uitkomst verandert daar niet door.
- Standaard synthetische replay op 4 cores: 137 s vóór deze reeks, nu 31 s (4,4×; het doel was
  5×). Met `vectorized: false` en `workers: 1` 131 s; met `vectorized: true` en `workers: 1` 90 s.
  Alle runs: 3 posities, eind-equity 25.303,60 USD. Met `vectorized: false` is het journal
  byte-gelijk aan dat van vóór deze reeks. Met de standaard verschillen vier
  `risk_decision`-regels in de vierde decimaal van de gemeten Greeks, door de IV van de
  FD-solve met startwaarde (binnen `iv_accuracy`).

## Options: legs van open posities blijven altijd in de keten (2026-11-26)
Een leg van een open positie die door het liquiditeitsfilter valt (te weinig volume of open
interest, brede spread), viel ook uit de keten waarmee de engine waardeert, sluit en (paper)
vult. Na een grote beweging kon de bot zo zijn eigen positie niet meer sluiten. Nu worden die
legs altijd geprijsd en in de keten gehouden; de filtertelling in het journal blijft gelijk.
Verwant aan het strikebereik van de feed (open legs buiten ±15 % van spot), dat apart is opgelost.

## Options: illiquide contracten niet meer prijzen (2026-11-26)
De ketenbouwer slaat de IV-solve en Greeks over voor contracten die op hun quote al door het
filter vallen (open interest, volume, geen bid, spread) of `stale` zijn. Dat was het meeste
rekenwerk. De gefilterde keten en dus elke beslissing blijven gelijk; de standaardreplay geeft
dezelfde 3 trades en eindstand (25.303,60 USD) in 163 s in plaats van 221 s. Eén verschil in
het journal: een contract zonder IV dat ook op zijn quote afvalt, telt nu onder die quote-reden
in plaats van onder `no_iv`.

## Options: historische simulatie op echte koers en IV (2026-11-26)
- `options.data.historical` en `python -I -m tradebot.options historical`: een replay-markt uit
  IBKR-dagbars en IBKR's dagelijkse 30-daagse IV (alleen lezen), dividenden en earnings uit
  yfinance. Ketens met QuantLib uit het echte IV-niveau en de vaste smile; spot en IV per
  snapshot lineair tussen opening en slot naar het verstreken deel van de dag.
- `Market.iv_history` (in `market.json`, optioneel voor oudere markten) en `run_replay` zet die
  als begin-IV-historie: IVR is in de simulatie vanaf dag één bruikbaar.
- Runbook en detailontwerp (sectie 15) bijgewerkt. De uitkomst meet de regels, niet echte fills.

## Fix: IBKR-account in een andere basisvaluta (2026-11-25)
- `options.execution.ibkr.IBKRBroker.account` las alleen regels in USD. Een paper-account met
  EUR als basisvaluta gaf daardoor elke ronde `geen equity of cash van IBKR`, en na drie keer
  een HALT (`options_task_failures`). Nu worden equity, cash en buying power in één andere
  basisvaluta omgerekend naar USD met de IDEALPRO-koers `<valuta>.USD` (actuele koers, anders
  de slotkoers). Zonder bruikbare koers, of met regels in meer dan één valuta, blijft het een
  `DataContractError`.
- Engine-keuze bij een dividend op de expiratiedag: de keten koos BAW of FD op het venster
  `(vandaag, expiratie]`, de pricer neemt op de expiratiedag `(gisteren, expiratie]`. Valt de
  expiratie door een feestdag op de ex-dividenddag (SPY, donderdag 18 juni 2026: vrijdag
  Juneteenth), dan koos de keten BAW, weigerde de pricer, en haltte de engine na drie rondes.
  Nu is er één venster: `OptionInputs.discrete_dividends`, gebruikt door de ketenbouw, de
  IV-terugval, de scenario's en de marktgeneratoren. Gevonden met de historische simulatie.
- `IBKRBroker.executions`: de eerste ronde na een start vroeg executies "sinds" `datetime.min`,
  een tijd in jaar 1. IBKR weigert die (fout 10314) en gaf dan niets terug, zodat fills van
  vóór de herstart niet gelezen werden. Zonder geldige tijd gaat het filter nu zonder tijd.
- `IBKRBroker.connect` probeert tot 12 keer (elke 10 s) opnieuw zolang de gateway nog inlogt,
  in plaats van te crashen met een traceback (na een herstart van de pc).
- `options.execution.ib_noise`: IBKR-meldingen 200 (strike bestaat niet bij die expiratie),
  10090 en 10167 (vertraagde data) en "API connection failed" gaan niet meer naar de log;
  tick 90 (vertraagd "halted") krijgt de betekenis van tick 49, zodat `ib_async` 2.0.1 er geen
  `AssertionError` meer voor logt. Alle andere meldingen blijven.

## Options: volatility risk premium, theta harvesting en vol-regimes (2026-11-24)
Drie bronnen van rendement als vaste regels tussen `Strategy.scan` en de risicopoort. Alleen
nieuwe entries; sluiten, stops, rolls en bijsturingen gaan altijd door. Geen training: elk getal
staat in `conf/options/default.yaml`. Uitleg: `docs/research/2026-11-24-options-vol-regime.md`.
- `options.vol_regime`: de verwachte RV als vaste menging (in variantie) van de Yang-Zhang-RV
  over 5, 20 en 60 dagen, de VRP (ATM-IV(30) tegen die verwachting) en het regime per
  onderliggende: STRESS (backwardation of RV5/RV20 ≥ 1,5), RICH, NORMAL, CHEAP of UNKNOWN.
  STRESS op SPY geldt voor het hele boek.
- `options.selection`: per regime de strategieën die mogen openen (`vol_regime.allowed`), en
  voor kandidaten die premie ontvangen de theta-dekking `(θ/dag × 365) / (½·|Γ|·S²·σ_RV²) ≥ 1`
  plus een optionele vloer op θ per dag als fractie van het max verlies.
- Engine: een `vol_regime`-event bij elke wisseling, de regimes en de geblokkeerde kandidaten
  per poort in `scan_summary`, en de spoorregels `regime:...` en `theta:cover=...` per
  kandidaat.
- Config: secties `vol_regime` en `theta_harvest` (met standaardwaarden, dus oude configs
  laden ongewijzigd en met de poort aan).

## Options fase 7: IBKR-paper, deployment en de live-schakelaar (uit) (2026-11-24)
Detailontwerp: secties 10.6, 12.1, 13 en 14. Nog steeds alleen fictief geld.
- Dependency `ib_async==2.0.1` (open vraag D), met `aeventkit` en `nest-asyncio`, in de lock
  en in de lijst harde dependencies van `hygiene.yml`.
- `options.execution.ibkr`: de `BrokerAdapter` voor TWS/IB Gateway. Elke order is een combo
  (`BAG`) met de client-id als `orderRef`; `modify` wijzigt alleen de limiet. `connect`
  weigert elk account dat niet met `DU` begint en vraagt vertraagde data (open vraag A).
- `options.data.ibkr_feed`: ketens, koers en dagbars van IBKR in pakketten van 90 lijnen;
  earnings en dividenden van yfinance.
- `options.engine.paper` en `python -I -m tradebot.options paper --run-dir`: de dienst met de
  echte klok, één proces per run-map, SIGTERM rondt af. `--healthcheck` kijkt naar de
  heartbeat.
- `options.live`: de live-schakelaar met vijf voorwaarden die alle vijf waar moeten zijn; een
  killgate-test loopt alle 31 onvolledige combinaties af. Er is geen live-opdracht en geen
  config met `mode: live`; een test bewaakt dat.
- Deployment: `ib-gateway` (vast op `TRADING_MODE: paper`) en `options-paper` in
  `infra/compose/docker-compose.yml`, `infra/systemd/tradebot-options-paper.service`, nieuwe
  sleutels in `.env.example` en `docs/runbook-options.md`.
- Afwijking: assignments bij IBKR worden niet als aparte executie gelezen. De reconcile ziet
  de positiewijziging, blokkeert entries op die onderliggende en alarmeert.

## Options fase 6: paper-broker, engine, replay en CLI (2026-11-24)
Detailontwerp: secties 9 tot en met 12. Alles met fictief geld; er is geen live-pad.
- `options.execution`: combo-orders met stabiele position- en client-ids, de price-walk
  (start op mid, per tick richting natural, stop bij de maximale concessie), het
  `BrokerAdapter`-protocol en de `PaperBroker` (deterministische fills, commissie,
  expiratie met exercise en assignment, vroege assignment voor dividend).
- `options.engine.service`: de engine als toestandsmachine met taken (markt, walk, risico,
  manage, reconcile, scan, einde dag). Een `SafetyBreach` zet de `HaltStore`; drie fouten op
  rij in dezelfde taak ook. HALT-bestand blokkeert entries, UNWIND sluit alles. Een
  reconcile-verschil blokkeert entries voor die onderliggende tot het weg is.
- `options.engine.state`: staat atomisch op schijf met mode en account-hash; het journal is
  append-only JSONL zonder geheimen.
- `options.engine.replay` en `python -I -m tradebot.options replay --market --out`: de engine
  op een opgeslagen markt met een virtuele klok (elke 15 s in de sessie plus einde dag).
  Exit-code 1 bij een HALT of een short optie die na expiratie nog open staat.
- Workflow `options-paper-smoke.yml`: replay op de synthetische standaardmarkt.
- Config: `schedule` (scantijden, entryvenster, intervallen, einde dag), `account`
  (`capital_usd` 25.000, fictief) en `paper`.
- Afwijkingen: ketens worden in de engine gebouwd met BAW (`Use.SCAN`), ook voor de delta's
  waarop strikes gekozen worden; de risicopoort rekent pro forma met de beslisengine. De
  risicotaak rekent alleen opnieuw als markt of boek veranderde. In replay krijgt elke naam
  die de config niet kent beta 1,0. De ATM-IV-historie krijgt alleen een punt op een dag
  met een verse keten. `SingleInstance` en de `Alerter` zijn niet verplaatst; de dienst krijgt
  een alert-functie mee.

## Options fase 5: risicopoort, limieten, scenario's en PDT (2026-11-24)
Detailontwerp: sectie 8. Elke order gaat door één ingang, `RiskGate.review`; alleen de
poort kan een `ApprovedOrder` maken (een sentinel plus een AST-test die dat afdwingt).
- `options.risk.aggregate`: Greeks per onderliggende en totaal, beta-gewogen dollar-delta,
  verlies bij 5 % SPY-beweging (gamma), vega en theta. Aandelen tellen mee.
- `options.risk.scenarios`: herwaardering per leg op een rooster van spot- en
  vol-schokken; legs met discreet dividend via FD op een grof rooster.
- `options.risk.margin` (benadering van buying power), `options.risk.pdt` (de
  PDT-bewaking onder 25k) en `options.risk.monitor` (dagverlies, drawdown, afbouwplan).
- `options.risk.limits`: zeven benoemde controles tegen `risk.*`-fracties van het kapitaal.
- `options.risk.gate`: harde fouten (aantal, verkeerde onderliggende, verlopen, te grote
  order, limiet buiten het bid/ask-bereik) zijn een `SafetyBreach`; ontbrekende quotes zijn
  een weigering. Entries worden per eenheid verkleind tot ze passen. Volledig sluiten mag
  altijd, ook in HALT. Bijsturen mag alleen als het binnen de grenzen blijft of risico
  verkleint.
- Config: de rest van `risk` en een nieuwe sectie `execution` (ticks, `max_order_usd`).
- Afwijking: de prijscontrole vergelijkt de limiet met het bereik van gunstigst
  (kopen tegen bid, verkopen tegen ask) tot natuurlijk, plus één tick, in plaats van
  intrinsieke waarde tot natuurlijk. De intrinsieke ondergrens klopt niet voor combo's met
  short legs (een credit spread zou altijd buiten de grens vallen).

## Options fase 4: regelgebaseerde strategieën (2026-11-24)
Detailontwerp: sectie 7. Pure functies van een bevroren marktbeeld en het boek: geen klok,
geen I/O, geen broker, geen training.
- `options.positions`: posities, kandidaten, acties, `MarketState`, `Book`, de trend tegen
  de SMA50 en de mark van een combo.
- `options.strategies`: gedeelde keuzeregels (expiratie, strike bij delta, wings), poorten
  (events, IV-voorwaarde), sizing volgens 7.3 en de expiratieregel uit 7.5.
- Aan: credit spread, iron condor (met één bijsturing), debit vertical, calendar spread en
  long straddle/strangle (earnings en lage IV). Uit (open vraag C): cash-secured put,
  covered call en de wheel als toestandsmachine. Short straddle/strangle bestaat, maar de
  registry weigert hem zonder `risk.allow_undefined_risk`.
- Config: secties `strategies`, `sizing`, `schedule` en de eerste sleutels van `risk`.
- Afwijking van het detailontwerp: `min_credit_frac_width` staat op 0,20 (spread) en 0,15
  (condor) in plaats van ⅓. Credit gedeeld door breedte is ongeveer de kans op ITM, dus
  ongeveer de short delta; ⅓ is bij delta 0,25 en 0,16 onhaalbaar en zou beide
  strategieën stil laten staan.

## Options fase 3: ketens, filters, oppervlak, IV-maten en replay-data (2026-11-24)
Detailontwerp: secties 4 en 6. Geen strategie, geen order: alleen de marktkant.
- `options.chain`: ruwe quotes naar een keten met IV (mid, optioneel bid en ask) en Greeks
  per contract; filters in vaste volgorde met een telling per reden.
- `options.pricing.surface`: forward uit put-call-pariteit, smile in totale variantie,
  lineair over de tijd, kalender- en butterflycontrole. Smilepunten met een vega onder
  `pricing.surface_min_vega` tellen niet mee: bij centquotes is hun IV afrondingsruis.
- `options.iv_metrics`: IV rank en percentile over 252 dagen, met tot dan de vervangende
  regel IV/RV(20) (open vraag B); elke beslissing daarop is gemarkeerd.
- `options.events`: earnings en dividenden met hun ouderdom.
- `options.data`: een deterministische synthetische markt (GBM, sprong, dividend, vaste
  smile, centquotes) en de opslag ervan in parquet en JSON, ook voor het latere
  ketenarchief. `python -I -m tradebot.options.data.synthetic --out <map>`.
- Config: secties `filters` en `iv_metrics`.

## Options fase 2: QuantLib, contracten, kalender en pricing (2026-11-24)
Detailontwerp: `docs/research/2026-11-24-options-pivot-detailontwerp.md`, secties 2-5.
- **QuantLib 1.44** als harde dependency (lock, `hygiene.yml`, mypy-override).
- `tradebot.options`: eenheden in het type (`units`), contracten en quotes, de NYSE-kalender
  uit QuantLib met halve dagen uit de config, looptijden op seconden tot 16:00 New York.
- `tradebot.options.pricing`: Europees analytisch, binomiaal (LR/CRR), BAW en FD met discrete
  dividenden; `evaluationDate` alleen via een contextmanager; IV met expliciete foutcodes en
  een terugvalengine; Greeks per aandeel (vega, rho en theta met bump-and-reprice).
- `conf/options/default.yaml` en `schemas/options.py` (secties `universe` en `pricing`).
- Tests: de acceptatietests uit 5.7 van de leidraad (gesloten formule en pariteit tot 1e-10,
  Amerikaans tegen Europees, convergentie, Hull), een IV-round-trip als property-test en een
  benchmark van 2.000 contracten in minder dan 2 s.

## Crypto in standby, options worden de actieve setup (2026-11-24)
Ontwerp: `docs/research/2026-11-24-options-pivot-design.md`. Geen crypto-code, -config of
-test verwijderd en geen logica veranderd; alleen de activering.
- De U1-dienst staat in `infra/compose/docker-compose.yml` achter het profiel `crypto`.
- `kraken-shadow.yml` draait alleen nog met de hand; de k8s-cron `carry-shadow` heeft
  `suspend: true`.
- README, `pyproject.toml` en de module-docstrings van de crypto-packages zeggen standby.
- `docs/crypto-standby.md`: wat er in standby staat en hoe je het weer aanzet.
- `soupsieve` 2.9.2 naar 4.1.3 in `requirements.lock` (vier CVE's).

## Het Kraken-boek 24/7 op een server: veiligheid, watchdog, alarmen (2026-10-31)
Draaiboek: `docs/runbook-kraken-u1.md` (§3 server, §4 veiligheid). Geen bevroren poort,
oordeel of backtest veranderd.
- **`carry/safety.py`**: harde grenzen uit de serverconfig, los van het model.
  - Plafonds voor kapitaal, ordergrootte en bruto.
  - De live koers tegen de close, en verse data.
  - Een verliesgrens op de gehedgede posities per stap.
  - Een marge-headroom.
  - Hedge-gaten, met een reparatie die alleen afbouwt (het overschot van het grootste been).
  - Een lock zodat er maar één proces per staat draait.
  - Een grens kan risico tegenhouden of laten afbouwen, nooit opbouwen.
- **`carry.u1.step` voert order voor order uit.**
  - Bij een fout leest hij de beurs opnieuw, dicht hij een hedge-gat reduce-only en staat
    het boek stil.
  - Na het handelen legt hij de beurs nog eens tegen de verwachting.
  - Is de beurs niet te lezen, dan staat het boek stil met de stand uit de fills.
  - Nieuw: `resume` (een mens hervat; de beurs is de nieuwe verwachting), een logboek
    (JSONL) en de laatste 30 staten als backup.
  - De CLI leest pad, logboek en modus uit de serverconfig (`--config`).
- **`carry/service.py`**: de dienst. Elke ronde:
  - heartbeat (bestand en `HEARTBEAT_URL`);
  - lock;
  - UNWIND/HALT;
  - watchdog: reconciliatie, hedge, marge, klok;
  - de dagelijkse stap, met venster en deadline.
  - Tijdelijke fouten tellen; fatale fouten in de stap zetten het boek stil.
  - SIGTERM rondt de lopende ronde netjes af. `--once` draait één ronde, `--healthcheck`
    controleert de heartbeat.
- **`carry/alerts.py`**: webhook (Slack, Mattermost, Discord) en/of Telegram.
  - Ontdubbeld.
  - Gooit nooit een fout.
  - Lekt de token of webhook-URL niet in het log.
- **`KrakenVenue.prices()` en `.server_time()`**: publiek, ook in dry-run.
- **`PaperVenue`** houdt zijn eigen stand bij na een fill.
- **Uitrol:**
  - `conf/live/u1.example.yaml`;
  - `infra/compose/` (docker compose: read-only, geen capabilities, `restart:
    unless-stopped`, healthcheck, een volume voor de staat) met `.env.example`;
  - `infra/systemd/tradebot-u1.service` (`Restart=always`, verhard);
  - het image krijgt `/var/lib/tradebot`.
- **CI:** `docker.yml` bouwt het image en controleert healthcheck en compose. Op `main` draait
  hij één echte dry-run-ronde in de container. De infra-contracttest controleert compose,
  `.env.example` en de systemd-unit.
- **Het "te laat"-alarm** ging af vóór de stap draaide. De eerste echte ronde in de
  container (`docker.yml` op `main`, 16:03 UTC) liet dat zien. Na een herstart in de middag
  kwam er dus een vals alarm, terwijl de stap in dezelfde ronde slaagde. Nu komt het alarm
  alleen als de stap na de deadline mislukt.
- **pip-audit** werd rood op een nieuwe advisory, niet door deze diff: hydra-core 1.3.7
  (GHSA-wf6j-7hvr-gr93). Opgehoogd naar 1.3.8 met `pip-compile --upgrade-package
  hydra-core`; verder verandert de lock niet.

## Het Kraken-boek draait: feed, live-stap, dagelijkse shadow, vooruit-lezing (2026-10-27)
Draaiboek: `docs/runbook-kraken-u1.md`. Geen bevroren poort, oordeel, ledger-entry of
backtest veranderd.
- **`systematic.harvest.basis_decide`**: het besluit van één bar van H3 (band, hedge,
  margevloer, gedwongen exit, stof) staat nu op één plek, voor de backtest en het live-boek.
  Getoetst met een vingerafdruk van `run_basis` over 324 synthetische runs, met
  liquidaties en gedwongen exits: bit-identiek.
- **`data.kraken_feed`**: de spot-OHLC, de perp-candles en de funding van Kraken, in
  dezelfde panelen en `BasisMarket` als die van Binance.
  - Alleen afgesloten bars.
  - De index is de sluittijd van de bar.
  - De dagomzet moet plausibel zijn; dat vangt een verkeerde eenheid.
  - `breadth_from_frames` kreeg daarvoor `min_members`, standaard 4.
- **`carry.u1`**: het boek dat op Kraken kan.
  - `shadow` rekent het boek op de Kraken-data, met de kasregel van v9.
  - `step` reconcilieert eerst de beurs en past dan `basis_decide` toe op wat de beurs
    houdt. Het besluit wordt orders in de margeveilige volgorde. De stap meldt ook wat er
    van of naar het geldmarktfonds moet.
  - `unwind` sluit alles, eerst de shorts.
  - Standaard draait het boek dry-run. Live alleen met `--live` en sleutels in de omgeving.
- **`kraken-shadow.yml`**: de shadow elke dag op Actions, alleen lezend.
- **`programme_v9 forward`**: de vooraf vastgelegde vooruit-lezing op Kraken-data.
  - Vóór 2027-04-01 weigert hij, zonder te registreren.
  - De lezing staat in het slot vóór de berekening.
  - `research-run` (taak `v9-forward`) commit slot en ledger ook na een fout.
- **Correctie (v9):** de config, de module en het rapport zeiden dat het besluit "een dag
  vooraf (lag 1)" valt. De backtest besluit en handelt echter op dezelfde close.
  - De kasregel kan alleen rente wegnemen; de cijfers veranderen niet.
  - Live wacht een instap op de overboeking. Die vertraging is niet in U1 gerekend; de
    batterij van v5 begrenst haar op een Sharpe-verhouding van 0,99 bij lag 2.
- **Kleinere punten:**
  - `carry.orders.Deltas` (Protocol) en `runner.after_fills` zijn nu openbaar; het
    Kraken-boek gebruikt ze ook.
  - De docstrings van het carry-pakket noemen de Kraken-adapter.
  - Een test controleert dat elke `tradebot`-module die een workflow aanroept, bestaat.

## Robuust boek v9: H3 op Kraken EU met de kasrente (2026-10-25)
Rapport: `docs/research/2026-10-25-robust-book-v9-kraken-cash.md`. Eén trial (M = 32), in de
ledger en bevroren vóór de run; geen drempel versoepeld.
- **U1** = de bevroren dagreeks van v5-H3 (BTC/ETH-carry, de enige regelset die op Kraken
  EU kan), met de fees van Kraken en DTB3 − 50 bp op het kapitaal dat het boek niet
  gebruikt. De Sharpe-poorten meten het rendement BOVEN de T-bill.
- **Elk kalenderjaar positief:** +8,7 / +25,6 / +1,3 / +4,5 / +7,9 / +3,3 / +2,3 % (2020-04
  .. 2026-09), CAGR 8,0 %, max drawdown 0,33 %. Boven de T-bill +4,9 %/jaar, maar licht
  negatief in 2022, 2023, 2025 en 2026.
- **Gefalsificeerd** volgens de vooraf vastgelegde regels: g2 (CAGR W_DEV 8,75 % < 15 %) en
  de holdout (geen carry; het fonds verloor zijn 50 bp aan de T-bill). Voor een belegger in
  euro's met een dollarafdekking is 2022 −0,7 %.
- Nieuw: `systematic/programme_v9.py`, `conf/model/robust_book_v9.yaml`,
  `conf/research/preregistration_robust_book_v9.yaml`, `data/external/*.csv` (via
  `research-run`), `artefacts/research/robust_book_v9/`.

## Beurswissel Binance -> Kraken EU (2026-10-25)
Rapport: `docs/research/2026-10-25-venue-kraken-vs-bybit.md`. Geen bevroren poort, oordeel of
ledger-entry veranderd.
- **Bybit valt af:** Bybit EU (MiCA) biedt NL/EER-klanten geen perps; zonder perp-short
  bestaat het carryboek niet. **Kraken EU kan beide benen**, met 2:1 retail op perps, geen
  unified margin en alleen BTC/ETH (plus fiat/stablecoins) als perp-onderpand.
- **Welk boek past:** geen van de vier bevroren boeken in de Binance-indeling. Wel v5-H3
  (BTC/ETH) met de spot als onderpand in de futureswallet: dezelfde regels, perp/onderpand
  0,70. `conf/execution/venues.yaml` (profielen met bron en datum),
  `carry/venues.py` (blokkades, kosten per beurs, fee-overlay) en
  `artefacts/research/venue_comparison.json`.
- **Kosten:** 12 bp per paar (Kraken $1 mln maker) tot 85 bp (nieuw account, taker), tegen
  15,5 bp in het bevroren model. H3 over de hele cyclus: 6,2 % CAGR (Binance) -> 6,0 %
  (Kraken $1 mln taker) -> 5,3 % (nieuw account).
- **Kraken-adapter** `carry/kraken.py` (spot + futures, onderpand-overboeking, `reduceOnly`),
  standaard dry-run; spot-handtekening getoetst tegen de gepubliceerde vector van Kraken.
- **Onderzoeksdata via Actions:** `data/external_series.py` + `research-run.yml` (T-bill DTB3, €STR,
  dagfunding Kraken en Binance voor BTC/ETH, tot en met 2026-09-30).

## Productie-audit (2026-10-24) — eerlijke Sharpe, live-OMS, carry-uitvoeringspad, infra
Rapport: `docs/research/2026-10-24-production-readiness-audit.md`. Geen bevroren poort,
oordeel of ledger-entry veranderd; de backtest van v6–v8 is bit-identiek.
- **SK-1, de Sharpe van het carryboek is 3–5x te hoog.** `sqrt(365)` op dagrendementen met
  rho_1..3 ≈ 0,5. v8-T2 over de hele cyclus: 5,05 bevroren, **1,55 op maandrendementen**,
  1,08 op kwartaalrendementen; jaarvol 8,4 % i.p.v. 2,5 %. Bij de prijsstrategieën vallen
  beide samen. Nieuw: `validation/serial_sharpe.py` (maand, kwartaal, Lo met
  Bartlett-gewichten), extra velden in `systematic/evaluate.summarize`, en
  `systematic/restate.py` -> `artefacts/research/restatement.json`.
- **Live-OMS (E-3..E-6, E-9):** live fills werden nooit in het interne boek geboekt (de
  engine las equity en gewichten uit het papierboek); een HALT sloot alleen papierposities;
  een lege orderstatus werd "volledig gevuld tegen prijs 0"; een tweede noodsluiting
  plaatste niets (vaste `orderLinkId`, geen `reduceOnly`); een fout in het barpad stopte
  de engine met open posities. Alle vijf opgelost en gereproduceerd in tests.
- **Reconciliatie (E-7):** de `Reconciler` werd nergens gebruikt en keek alleen naar het
  interne boek. Nu elke bar in live-modus, over de unie van beide boeken; een blijvend
  verschil is een halt.
- **Carry-uitvoeringspad (E-1):** `systematic/leverage.run_levered` is opgesplitst in
  `LeveredBook.step` / `.decide_live`, bit-identiek over 16 scenario's. Nieuw pakket
  `tradebot.carry`: orders per been in een margeveilige volgorde, staat per symbool,
  venue-contract + paper-venue, reconciliatie (vangt ADL), kill switch, CLI. Shadowboek =
  backtest, bit voor bit, ook met dagelijkse save/load en een groeiend paneel.
- **Infra (E-2):** de images bouwden niet (Python 3.11 tegen >= 3.13, `COPY apps/` van een
  niet-bestaande map, geen lock) en de manifests riepen niet-bestaande modules aan. Nu één
  image uit de lock en een maandelijkse `cron-carry-shadow`; de stubs zijn weg en
  `tests/unit/test_infra_contract.py` toetst dat wat aangeroepen wordt, bestaat.
- **Strikte xfails opgelost (5 van 6):** de rolling Sharpe in de state-file (5s-bars, 252
  dagen, kon nooit iets schrijven); holdoutsloten zonder splittoets (v2–v8 en de
  vooruit-lezing, nu `programme_v2.require_lock_split`); de DSR-M live uit de ledger (nu
  `programme_v2.frozen_m`). Open: de fracdiff-gewichtsvolgorde (bevroren features).
- **Paper-trade-smoke** is hernoemd naar wat hij is (24 synthetische bars, geen orders) en
  geen promotiepoort meer.
- **pip-audit** werd rood op nieuwe advisories (niet door deze diff): aiohttp 3.14.4 (met
  yarl 1.25.1), certifi 2026.10.19 en pillow 13.0.0 via `pip-compile --upgrade-package`,
  setuptools 84.1.0 in `requirements-dev.lock`.

## P²-kwantielschatter hersteld (2026-10-18)
- **Fout:** `P2OnlineQuantile` (`train/quant_arch.py`) en
  `RollingRobustScaler._p2_update_single` (`train/_scalers.py`) rekenden de
  gewenste markerposities uit als `1 + 2q·n`, `1 + 4q·n`, `3 + 2q·n`. Jain &
  Chlamtac (1985) schrijven `1 + (n-1)q/2`, `1 + (n-1)q`, `1 + (n-1)(1+q)/2`
  voor. Daarnaast startten de WERKELIJKE posities op de gewenste startwaarden
  `1, 1+2q, 1+4q, 3+2q, 5` in plaats van op `1..5`.
- **Gevolg:** de schatting liep weg. De mediaan op een N(0,1)-steekproef kwam
  uit op 3,4. `RollingRobustScaler.fit` overschrijft de batch-mediaan en
  -spreiding met deze schatting, dus `center_` lag rond 3,0 in plaats van 0.
- **Na de fix:** mediaan −0,016 tegen −0,014 echt (n = 4000). `center_` en
  `scale_` liggen binnen 0,01 resp. 1,1 % van de batchwaarden.
- De drie `xfail`-tests die dit vastlegden, zijn nu gewone tests. Alleen het
  CatBoost-pad (`train/catboost.py`, `train/ensemble.py`) gebruikt de scaler.
  Het robuuste boek (`systematic/`) en de weekly-strategie
  (`train/light_models.py`) raken hem niet, en `artefacts/models/` bevat geen
  gefitte scaler: er verandert geen bevroren resultaat.

## DI-31 herzien (2026-10-14) — numpy's kernelkeuze, niet OpenBLAS
- **Symptoom:** `test_the_multiplier_is_causal[hmm3-diag-student_t]` faalde af
  en toe met `atol=0, rtol=0`. De volledige en de afgekapte run fitten
  bitidentieke trainvensters, maar de Student-t EM gaf soms andere bits terug.
- **Oorzaak:** `np.log(u[:, i])` op een kolom-VIEW in de M-stap, opgelost in
  `ad44372`. numpy 1.26 kiest voor float64-`log` de AVX-512-kernel alleen als
  `nomemoverlap` invoer en uitvoer disjunct ziet. Voor kolom `i` neemt die
  controle `stride * n` bytes, 8·i bytes voorbij het einde van `u`, en telt
  hij aansluitende bereiken als overlap. Landt de verse uitvoer in het
  heapblok direct na `u`, dan rekent numpy met scalaire libm: 1 ULP verschil
  op ~25 % van de elementen. 200 EM-iteraties maken daar 1e-12 van in de
  parameters en 1e-14 in de factor. Bij k = 2 haalt de overschrijding het
  volgende blok nooit.
- **Weerlegd: multithreaded OpenBLAS.** De oude regel drijft ook met OpenBLAS
  op één thread (8 van 20 herfits). In een verse Python-3.13-omgeving uit de
  lockfiles bouwt numpy 1.26.4 uit de sdist zonder BLAS, dus `@` raakt OpenBLAS
  daar niet. De OpenBLAS 0.3.30 die DI-31 noemde is die van scipy, en de EM
  roept die niet aan. `threadpool_limits` om de fit veranderde de
  heapgeschiedenis en daarmee waar de uitvoer landde. Dat verklaart waarom
  het de ene run repareerde en de pytest-run juist liet falen.
- **Gemeten (2026-10-14).** Herfits van de acht opgenomen trainvensters na
  een willekeurige heapverstoring:

  | omgeving | oude regel | HEAD |
  |---|---:|---:|
  | Py 3.11, numpy-wheel (OpenBLAS 0.3.23) | 22/30 | 0/120 |
  | Py 3.13, numpy zonder BLAS | 15/40 | 0/360 |

  De vergelijking van de causaliteitstest zelf, onder dezelfde verstoring,
  faalde met de oude regel 12/12 en 5/6 en op HEAD 0/48 en 0/12. Onder
  pytest zelf (`-p randomly`, `test_regime_conditioning.py` +
  `test_student_t_hmm.py`, Py 3.13) slaagt de causaliteitstest op HEAD in
  30 van 30 seeds; met de oude regel faalt hij in 8 van 10 seeds en de
  bewakingstest in alle 10. Een spion
  over een volledige `build_conditioner` vond geen enkele transcendente
  ufunc-aanroep (numpy, scipy, hmmlearn) met een niet-contigue invoer.
- **Geen numerieke wijziging en geen besluit nodig.** Tolerantie, markers en
  de bevroren H2-getallen blijven ongemoeid. De regel wordt bewaakt door
  `test_transcendentals_never_see_a_strided_view`. Het register
  `docs/DEFERRED_ISSUES.md` staat sinds `57d3fcf` niet meer in de repository;
  deze notitie en de toelichting bij `log_u` in `regime/student_t.py` zijn
  de bijgewerkte beschrijving.

## Robuust boek v6–v8 (2026-10-09) — hefboom, uitvoering, sprongrisico
- **Eigenaarsbesluit:** hoog risico toegestaan; hefboom niet langer uitgesloten.
  Mandaat ongewijzigd: bruto ≤ 4,0, ADV-participatie ≤ 1 %, max drawdown 25 %.
  Rapport: `docs/research/2026-10-09-robust-book-v6-v8-report.md`.
- **Nieuw boek `systematic/leverage.py`:** de basiscarry van v5 in een
  unified-margin-account.
  - Spot als onderpand met haircut; USDT-lening tegen `max(8 %, BTC-carry)`.
  - Liquidatietoets op de intraday **mark price**: nieuwe data, 895/900 perps,
    MD5-geverifieerd.
  - Governor op uniMMR ≥ 3 en bruto ≤ 4; ADV-cap tegen de lopende equity.
  - Gespreide uitvoering (v7), een gefinancierde tranche (v7) en een
    sprongrisicogrens (v8).
- **v6 (M = 24):** 1×/1,5×/2× per been op W_DEV gaf Sharpe 5,4 / 4,3 / 3,7 en CAGR
  12,1 / 13,6 / 14,0 %. Alle drie binden op G2 en G5. Hefboom is bijna NPV-nul, omdat
  de leenrente met de carry meebeweegt.
- **v7 (M = 26):** spreiding over 5 dagen plus een tranche alleen waar carry ≥
  rente + 10 pp. **S2 haalt op W_DEV alle poorten:** Sharpe 5,35, CAGR 16,7 %,
  MDD 1,1 %.
- **De stille killer:** in de holdout 2025–26 liet de carry-screen alleen
  pump-and-dump-microcaps door (dag-σ 0,24–1,05). Dat is adverse selectie, geen
  carry.
  - v6-K1: −8,6 % (z = −14,9), gefalsificeerd.
  - v7-S2: −5,7 % (z = −16,7), gefalsificeerd.
- **v8 (M = 27):** een sprongrisicogrens (dag-σ ≤ 0,20), afgeleid uit de bevroren
  kosten- en instapparameters.
  - T2: W_DEV Sharpe 5,33, CAGR 15,9 %, alle poorten vrij.
  - Backcast 2020: Sharpe 9,0, CAGR 24,7 %.
  - Holdout −1,1 % (DD 1,4 %), formeel gefalsificeerd door de bevroren z-regel.
  - Hele cyclus 2020-04 .. 2026-09: **Sharpe 5,05, CAGR 13,4 %, MDD 2,0 %**.
- **Vooruit-slot** (vanaf 2026-10-01), bevroren op 2026-10-09 vóór alle lezingen: de
  enige schone toets. Lezing vanaf april 2027 via `binance_vision forward` en
  `programme_v8 forward`.
- **Scenario:** met één beurs en ~3 %/jaar kans op een faillissement zakt de echte
  Sharpe naar ~0,8. Meerdere beurzen zijn een voorwaarde voor de doelstelling.
- **Reproduceerbaarheid:** v5 reproduceert bit-identiek op opnieuw opgehaalde data;
  v6 en v7 na elke refactor.

## Mandaatwijziging (2026-09-12) — eigen kapitaal i.p.v. propfirm
- **Risicobudget herijkt.** Er wordt niet meer met propfirms gewerkt; de
  drempels die contractnaleving waren zijn vervangen door een
  eigen-kapitaalbudget. `max_drawdown_pct` 0,08 → 0,25 · `daily_loss_limit`
  0,03 → 0,10 · de-grossing-trappen 0,04/0,06 → 0,12/0,18 ·
  `daily_var_limit_pct` 0,02 → 0,05 · `max_position_age_h` 48 → 720 ·
  `sigma_target` 0,08 → 0,20 · `max_leverage` en `gross_cap` 1,5 → 4,0 ·
  `net_cap` 0,60 → 2,0 · `max_position_pct` 0,25 → 0,80.
- **`adv_participation_cap` (0,01) en `max_concentration` (0,40) ONGEWIJZIGD.**
  Marktfeit resp. de enige werkzame spreidingsbescherming (AD-4); geen
  mandaatgrond om ze te verruimen.
- **`docs/RISK_MANDATE.md`** (nieuw): verantwoording per limiet, met een label
  A (propfirm-afgeleid) / B (interne "strengste wint") / C (marktfeit), zodat
  "de propfirm is weg" geen blanco cheque voor het hele bestand is. AD-26 in
  `docs/ARCHITECTURAL_DECISIONS.md`.
- **Twee limieten maakten een levende kandidaat onmeetbaar**, en dat is de enige
  meetbare winst van deze wijziging: `max_position_age_h: 48` sloot elke positie
  na twee dagen terwijl funding carry (~1,95 bps/dag) ruim zeven dagen nodig
  heeft om alleen de 13,0 bps vaste kosten terug te verdienen; en
  `max_drawdown_pct: 0,08` haltte de enige forward-meting in de repo
  (`max_drawdown −0,3817`) vóór het venster uit was.
- **Geen edge-claim geraakt.** Geen backtest opnieuw gedraaid, geen metriek
  herschreven, geen ledger-entry gewijzigd; de vier bekende poort-failures
  (KG-B1/B2/B3 op `cm_carry`/`cm_tsmom`) staan onveranderd rood.
- **Achterhaald verklaard:** `artefacts/PROPFIRM_PARALLEL_AUDIT.md`,
  `docs/STRATEGY_AUDIT_CRYPTO_ACCOUNTS_2026-06-14.md` en §4.5 van
  `docs/EXPANSION_RESEARCH_2026-08-10.md` ("keep the propfirm numbers — they are
  contractual"). `risk/daily_loss_governor.py` gemarkeerd DORMANT met een
  meetbare heropeningsvoorwaarde.
- **Tests mandaat-onafhankelijk gemaakt** i.p.v. herijkt op nieuwe literals: de
  kill-switch-scenario's leiden hun equity nu af uit `conf/risk/`, en de
  engine-pariteitstolerantie is genormaliseerd per eenheid bruto-exposure
  (gemeten stabiel op 11,7-15,0 bp/gross over een factor 2,5 aan boekgrootte,
  waar de absolute 3,0 bp dat per constructie niet was).
- Nieuw in DEFERRED_ISSUES: DI-27 (de vol-target bindt niet), DI-28 (drie
  tweede bronnen van waarheid voor risicodrempels), DI-29 (dode
  propfirm-machinerie).

## Wave 20 (2026-06-10) — mandaat v3 fundament
- Hypothesis-ledger (append-only, seed 2363) + falsificatieregister F1–F12.
- PIT multi-markt datalaag (`data/sources/`): stooq, kenfrench, edgar, fred,
  cboe, wiki_constituents; `asof_join` als enige gesanctioneerde PIT-merge.
- G4-factorlab: `risk/factor_alpha.py` (HAC/Newey-West residual alpha).
- Crypto-rebaseline bit-identiek gereproduceerd (Sharpe 1.15, CAGR +46.7%,
  G6 α +20.2%/jr t=2.65 — audit §9).
- W21/22-voorbereid: XS-harness + eq_xsmom/eq_strev/eq_lowvol + PIT-universum.

## Waves 14–19 (2026-06-09) — zie TRUE_ALPHA_AUDIT §12–17
- W15: XS-rework — pooled cross-sectioneel recept, OOS IC +0.069, modellen
  gepersisteerd (`artefacts/tracks_breadth/`). W19: AdaptiveWalkForward
  deploy-grade (`alpha/adaptive_wf.py`), PBO S≥50-les.

## Audit-fixes (2026-05-28 → 2026-06-08)
- **AUDIT A-1** — Fractional differentiation featureblok (AFML hf. 5):
  `src/tradebot/features/fracdiff.py`, `ffd_*` features met per-asset d*
  (zie docs/data_dictionary.md).
- **AUDIT D-2** — CPCV purge in bar-space: `CombinatorialPurgedCV` vereist
  expliciete `purge_bars` in intraday-regimes (≥24 bars/dag); kalenderdag-
  purge was te zwak (`src/tradebot/cv/cpcv.py`; afgedwongen met ValueError).
- **AUDIT E-4** — Funding-rate featureblok, causaal (t−1):
  `src/tradebot/features/funding_carry.py`, `funding_*` micro/meso-features.
- DSR-dimensionaliteitsfix (`metrics.deflated_sharpe`): expected-max
  geschaald door SE van de Sharpe-schatter (Bailey & LdP 2014);
  regressie-locked in `tests/regression/test_dsr_dimensional_fix.py`.

## Waves 5–13 — infrastructuur
- CPCV + purge/embargo, triple-barrier/TrendScan-labeling, HRP/RP-portfolio,
  TCA-laag, OMS + audit-log (R-8), circuit-breakers (R-7), paper-engine,
  MRM-rapportage, lookahead-testsuite, determinisme-suite (R-5).

### Reviewfixes risicopoort, monitor en PDT (2026-11-27)

- N3: de aandelen-sluitorder (`review_stock`, `closing`) is begrensd op wat de broker echt heeft
  (`Book.shares`: tegengesteld en niet groter, tenzij de broker precies de aandelen-legs heeft),
  weigert op een symbool met een reconciliatieverschil, en heeft een verplichte prijsband (zonder
  keten een weigering). `_share_breaks` laat de modulo-100-marge alleen nog gelden op wheel-,
  cash-secured-put- en covered-call-symbolen; elders is elk verschil een recon-break. Een
  crash-herstart die opnieuw +100 SPY zou kopen wordt zo geweigerd en gemeld.
- N9: `max_order_usd` geldt alleen voor OPEN, ADJUST en ROLL; een sluiting geeft geen HALT meer.
- N10: ADJUST/ROLL meten het boek met de lopende entries erbij, en een nieuwe entry telt lopende
  ADJUST/ROLL-orders mee (`Book.pending_changes`, `PendingChange`).
- Oud 5: ADJUST en ROLL toetsen `blocked_underlyings`, dagverlies en PDT (positie van vandaag);
  het maximale verlies na een ADJUST komt uit de nieuwe legs (`risk.limits.expiry_max_loss`).
- N18: `degross_plan` trekt de aandelen-legs van de te sluiten posities af bij het meten van de rest.
- N23: de PDT-guard telt daytrades plus entries van vandaag (gevuld en lopend) tegen de grens.
- N30: een CLOSE moet de legs van de positie precies naar nul brengen (anders `SafetyBreach`);
  `_after` houdt de aandelen-legs; een aandelen-exit wordt pas opnieuw gestuurd nadat de broker de
  geannuleerde order niet meer open heeft (`order_cancel_pending`).

### Reviewfixes pricing, oppervlak, dividendinput en orderlimieten

- N4: `Surface.atm_vol` extrapoleert alleen nog binnen `min(14 dagen, 30 % van de looptijd)` van
  de dichtstbijzijnde smile, anders `None`; `term_slope` is `None` tenzij 30 én 60 dagen door
  smiles aan twee kanten omsloten zijn (was: vlak, dus helling 0,0). Aanroepers (service,
  `debit.min_term_slope`, `classify`) verwerkten `None` al: geen helling = geen debit-entry,
  geen backwardation-STRESS. Gedragswijziging in de replay: een ketting met maar tot ~45 dagen
  aan expiraties geeft geen ATM-IV-30/60-helling meer.
- N11: theta de dag vóór een ex-datum laat het dividend niet meer uit het venster vallen: de
  spot van het herwaarderingstijdstip zakt met het dividend dat in de stap ex gaat.
- N5: de spotbump voor BAW-delta/gamma schaalt met 0,25·σ√T, geklemd op [0,05 %, 0,5 %] (was
  vast 0,5 %); bij langere looptijden ongewijzigd. Ook in de batch-kernel.
- N26: de forward uit put-call-pariteit corrigeert de Amerikaanse mids voor de
  vroege-uitoefenpremie (BAW min Europees bij de IV van de optie).
- N27: `combo_fair` gebruikt voor alle legs de mid zodra één expiratie geen smile heeft.
- N24: `execution.orders.tick_for` leest `execution.ticks` (nieuw: `penny`, `penny_symbols`);
  `ticks.default` is niet langer dode config.
- N5 (live): `ibkr_feed.project_dividends` projecteert het dividendschema (laatste bedrag, vaste
  cadans uit de historie, tot 365 dagen) met `CashDividend.estimated`.
- Oud #34: een dividend met ex-datum op de expiratiedag telt niet meer dubbel (zit al in de
  spot); de docstring van `pricing.engines` zegt nu waar `Use.DECISION` wel en niet gebruikt wordt.

### Reviewfixes IBKR-paper-robuustheid

- Oud #14: runbook en `live.py` zeggen nu eerlijk dat alleen `require_paper` (DU-hek) en `TRADING_MODE: paper` in het pad zitten; `gather`/`live_decision` zijn niet aangesloten.
- Oud #8: de paper-dienst (`Engine(persist=True)`) bewaart de staat atomisch na elke submit, fill, walk-wijziging en annulering, inclusief `done_today` (nieuw veld in `state.json`, ouder bestand: leeg) en de pogingenteller; vóór de broker-call staat de intentie al in de staat. Bij de start (en na herverbinden) neemt `_adopt_orders` open orders over op `orderRef` (`BrokerAdapter.adopt`, nieuw) of zet ze op CANCELLED (`order_orphaned`); er komt nooit een tweede order voor dezelfde client-id. De replay bewaart niet tussendoor (gedrag ongewijzigd).
- Oud #10: `run_paper` verbindt bij uitval opnieuw met backoff (5 s, verdubbelend tot 60 s), tickt niet zolang de verbinding weg is (de heartbeat gaat dus niet groen/blijft stilstaan) en geeft HALT na `IB_RECONNECT_TIMEOUT_SECONDS` (standaard 900); na herstel opnieuw adopteren en reconciliëren.
- Oud #11: de CLI koppelt `carry.alerts.Alerter` (webhook/Telegram uit de omgeving, altijd ook het log) als alert-callback aan `run_paper`, en `HEARTBEAT_URL` als dead man's switch (hoogstens eens per minuut, alleen bij een verbonden ronde).
