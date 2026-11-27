# Runbook: de options-engine (paper)

De options-engine handelt Amerikaanse aandelen- en ETF-opties met regelgebaseerde strategieën.
Alles draait met **fictief geld**: replay op opgeslagen data, of een IBKR-paper-account met
vertraagde marktdata. Live bestaat in de code als schakelaar, maar staat uit en kan vanuit
deze repository niet aan (zie "Live" onderaan).

## Snel: replay op echte DoltHub-ketens

```bash
# eenmalig: ketens ophalen (cache), koersen uit Yahoo, earnings uit SEC EDGAR
python -I -m tradebot.options dolthub --cache /tmp/dolthub --out /tmp/markt \
    --start 2019-02-11 --end 2026-11-26
python -I -m tradebot.options replay --market /tmp/markt --out /tmp/run
# zonder netwerk: de kleine testmarkt (SPY, AAPL, MSFT; jan-feb 2023)
python -I -m tradebot.options replay --market tests/fixtures/options/dolthub_mini --out /tmp/smoke
```

De bron is de openbare database `post-no-preference/options` op DoltHub: einde-dag-ketens
(bid, ask, IV) vanaf februari 2019. De download is één query per dag (ruim 2.000 data, zo'n
70 MB cache, ongeveer een half uur) en hervat waar hij stopte; `--offline` bouwt alleen uit de
cache. Nodig: toegang tot `www.dolthub.com`, `query2.finance.yahoo.com` en `data.sec.gov`.

Wat de data niet kan (en wat de uitkomst daardoor betekent), staat in `data/dolthub.py` en
in de CHANGELOG. Kort: één snapshot per dag om 15:15 (de scan en het sluiten vóór expiratie
schuiven mee), ketens niet elke dag (2019 wekelijks, daarna meestal ma/wo/vr), weinig
expiraties en strikes, geen volume of open interest, QQQ en IWM ontbreken, fills op bid/ask
zijn optimistisch. Contracten die uit de keten vallen, worden tot 60 handelsdagen aangevuld
met QuantLib op hun laatste DoltHub-IV; die quotes hebben open interest 0, dus alleen een
open positie kan erop waarderen en sluiten.

De synthetische markt (TSTA/TSTB) is geen replaybron meer; hij bestaat alleen nog als
testfixture in `tests/unit/options/synthetic_market.py`.

`/tmp/run` bevat dan:

- `journal.jsonl`: elke gebeurtenis (keten, kandidaat, risicobesluit, order, fill, positie).
- `state.json`: posities, orders, equity per dag.
- `summary.json`: begin- en eind-equity, aantal posities, en of er iets misging, plus `stats`
  (zie "Backteststatistiek").

Staat SPY in de markt, dan heeft `summary.json` ook `buy_hold`: hetzelfde kapitaal op de
eerste dag in SPY, dividenden als kas, met dezelfde statistiek als het boek.

De exit-code is 1 als de engine haltte of als er na de laatste dag nog een short optie open
staat die al expireerde. De workflow `options-paper-smoke.yml` doet dit op de testmarkt.

`replay` en `sweep` weigeren een `--out` die al bestaat en niet leeg is (exit-code 2, met de
reden op stderr): het journal wordt aangevuld, en een tweede run in dezelfde map zou elke
telling in `summary.json`/`sweep.json` verdubbelen. Kies een nieuwe map, of geef
`--overwrite`. Die wist alleen de bekende bestanden van eerdere uitvoer en nooit een hele boom:

- een replay-map: alleen `journal.jsonl`, `state.json`, `summary.json` en `heartbeat`, met
  `mode: replay` in `state.json` (een paper-run heeft `paper` en blijft dus staan);
- een sweep-map: `sweep.json` plus submappen die elk zo'n replay-map zijn.

Staat er iets anders in, staat er een `HALT`, `UNWIND` of `halt/` in (ook een gehaltte replay),
of is de map de markt of bevat hij die, dan wist `--overwrite` niets (exit-code 2). Ruim zo'n
map met de hand op.

### Backteststatistiek

`summary.json` bevat onder `stats` de statistiek van de run (`engine/report.py`):

- `strategies.<naam>`: aantal gesloten posities (`closed`), `win_rate` (P&L > 0), `avg_pnl`,
  `total_pnl`, `largest_loss` (0 als er geen verlies was) en `max_drawdown_usd` (de grootste
  daling vanaf de top van de cumulatieve gesloten P&L, in de volgorde van sluiten, vanaf 0),
  in USD. De P&L van een positie is de
  som van haar fills (ook bijsturingen) min commissie, plus de afrekening van vervallen,
  toegewezen of uitgeoefende legs (`settle_usd`), plus de laatste cumulatieve hedge-P&L
  (`hedge_realized_usd`, dus inclusief het afbouwen van de aandelen na het sluiten van de
  opties). Een positie telt als gesloten bij `position_closed`; de engine schrijft dat ook als
  de laatste leg door expiratie, assignment of uitoefening afrekent.
- `book`: `total_return`, `max_drawdown` (fractie van de equity-piek), `sharpe` en `sortino` op
  dagrendementen uit de equity per dag (met het startkapitaal als eerste punt), jaarfactor
  √252, risicovrij 0. Sortino gebruikt de neerwaartse afwijking t.o.v. 0.
- Randgevallen: zonder gesloten posities is `strategies` leeg; bij minder dan drie equity-punten
  (inclusief het startpunt), een equity ≤ 0 vóór het laatste punt of een spreiding van 0 zijn
  `sharpe` en `sortino` `null` (Sortino ook bij nul neerwaartse afwijking). Rendement en drawdown
  blijven gewoon berekend (een leeggelopen rekening geeft -100 %). Nooit NaN.

Op korte runs zijn dit kleine steekproeven: lees ze als een gezondheidscheck van de regels,
niet als bewijs van een edge.

### Sweep: dezelfde config op meerdere markten

```bash
python -I -m tradebot.options sweep --markets /tmp/markt1 /tmp/markt2 --out /tmp/sweep [--workers N] [--overwrite]
```

Dit is een robuustheidscheck, geen optimalisatie: er is geen parameter-grid, alleen dezelfde
config op meerdere opgeslagen markten (bijvoorbeeld `historical` voor verschillende periodes).
Elke markt krijgt een replay in `/tmp/sweep/<mapnaam>/` (mapnamen moeten verschillen);
`sweep.json` bevat per markt begin- en eind-equity, aantal posities, `halted` en de `stats` uit
het vorige hoofdstuk. `--workers` is het aantal markten tegelijk (standaard auto = min(4, cores),
nooit meer dan markten); bij meer dan één worker rekent de keten per markt in-proces en is de
uitkomst gelijk aan die van `--workers 1`. De exit-code is 1 als een markt haltte of een
verlopen short liet staan.

### Snelheid

Gemeten op de (toenmalige) synthetische standaardmarkt in een container met **4 cores** (wandkloktijd,
`replay` alleen, op dezelfde markt). Elke run: 3 posities, eind-equity 25.303,60 USD.

| Code en instelling | Tijd | Factor |
|---|---|---|
| vóór de vectorisatie (commit `b639b1a`) | 137 s | 1× |
| `pricing.vectorized: false`, `workers: 1` | 131 s | 1,05× |
| `pricing.vectorized: true`, `workers: 1` | 90 s | 1,5× |
| standaard (`vectorized: true`, `workers: auto` = 4), na taak A4 | 31 s | 4,4× |
| eindstand, vergelijkingsmeting 2026-11-27: vóór de branch (`a589399`) | 171 s | 1× |
| eindstand, vergelijkingsmeting 2026-11-27: code na A4 (`970b4cb`) | 41 s | 4,2× |
| eindstand, vergelijkingsmeting 2026-11-27: eindstand van de branch | 41 s | 4,2× |

Het doel was 5×; dat is niet gehaald. Bij 4 workers gaat ongeveer 23 s naar het QuantLib-deel
(FD met een discreet dividend: IV-solve en Greeks, ±84 s rekenwerk in één proces), de rest is
het serieel werk van de engine. De latere fasen maken de replay niet trager: op
dezelfde machine en hetzelfde moment gemeten duren de code van na A4 en de eindstand allebei 41 s,
en de code van vóór de branch 171 s (4,2×). Absolute tijden verschillen per meetmoment op deze
gedeelde cloudmachine (eerder 137 s en 31 s); vergelijk daarom alleen metingen van hetzelfde moment. Met `vectorized: false` is het journal byte-gelijk
aan dat van vóór de vectorisatie. Met `vectorized: true` zijn de beslissingen en de equity
gelijk, maar verschillen vier `risk_decision`-regels in het vijfde cijfer van de gemeten
Greeks: de FD-solve met startwaarde (smal Brent-interval) vindt de IV binnen `iv_accuracy`
(1e-6) op een iets ander punt dan het volle interval.

Uitkomst van de standaardreplay in de eindstand: 3 posities geopend en gesloten, eind-equity
25.303,60 USD (begin 25.000), niet gehaltd. `stats`: `book` met `total_return` 0,012144,
`max_drawdown` 0,003297, `sharpe` 4,31, `sortino` 17,05; `credit_spread` 2 gesloten
(win rate 1,0, gemiddeld +58,20, totaal +116,40, grootste verlies 0) en
`long_straddle_strangle` 1 gesloten (+187,20). Drie posities zijn een gezondheidscheck, geen
bewijs van een edge.

## Nieuwe schakelaars (pro-strategieën)

Alle schakelaars staan in `conf/options/default.yaml` (met toelichting) en in
`src/tradebot/schemas/options.py`; onbekende sleutels zijn een fout. Met de standaardwaarden
is de replay gelijk aan die van vóór deze branch (3 posities, 25.303,60 USD). Zet een
schakelaar aan door de sleutel in je eigen config te overschrijven (of in
`conf/options/default.yaml` te wijzigen) en daarna `replay` of `sweep` te draaien.

### Snelheid van de keten (`pricing.*`)

| Sleutel | Standaard | Wat het doet | Aanzetten |
|---|---|---|---|
| `pricing.vectorized` | `true` | BAW-contracten in één numpy-kernel (pariteit met QuantLib vastgepind); FD met startwaarde uit die kernel | `false` = elk contract apart door QuantLib (oud pad, byte-gelijk journal) |
| `pricing.workers` | `auto` | QuantLib-deel van de keten over een procespool: `auto` = min(4, cores), `1` = eigen proces | een getal, of `1` om uit te zetten |
| `pricing.pool_min_contracts` | `16` | pas boven dit aantal contracten per keten de pool gebruiken | lager of hoger getal |

### Volatility regime en relative value

| Sleutel | Standaard | Wat het doet | Aanzetten |
|---|---|---|---|
| `vol_regime.ewma_weight` | `0.0` (uit) | aandeel van de EWMA-variantie (`ewma_lambda: 0.94`) in de verwachte RV; andere gewichten worden herschaald | bijvoorbeeld `0.2` |
| `vol_regime.mean_reversion_kappa` | `4.0` | snelheid per jaar waarmee σ(h) van de blend naar de lange RV trekt; bepaalt VRP per looptijd en de theta-dekking. Staat altijd aan, geen fit | getal aanpassen |
| `relative_value.weight` | `0.0` (uit) | z-score van de VRP over het universum herweegt de rangscore met 1 ± w·z (credit +, debit −), begrensd op 0,5 tot 1,5; alleen de rangorde, geen poort | bijvoorbeeld `0.25` |

### Premium selling

| Sleutel | Standaard | Wat het doet | Aanzetten |
|---|---|---|---|
| `strategies.credit_spread.min_vrp_ratio`, `iron_condor.min_vrp_ratio`, `short_straddle_strangle.min_vrp_ratio` | `null` (uit) | VRP-poort per looptijd: gemiddelde IV van de short legs ≥ ratio × σ(h), h = de DTE | `1.10` aanbevolen |
| `strategies.credit_spread.short_delta_exit` | `null` (uit) | short leg met \|delta\| boven deze waarde: sluiten (STOP, `delta_breach`), vóór de ITM-stop | `0.40` aanbevolen |
| `strategies.short_straddle_strangle.*` | `enabled: false` | zie "Short strangle" hieronder | zie daar |
| `risk.max_undefined_stress_loss_frac` | `0.06` | grens op de som van het stressverlies (±20 % spot, min credit) van alle undefined-risk posities, de lopende (nog niet gevulde) undefined-risk entries en de nieuwe; werkt alleen als undefined risk aan staat | getal aanpassen |

**Short strangle: blijft uit.** Een short straddle of strangle heeft onbeperkt risico. Hij is
alleen mogelijk als zowel `strategies.short_straddle_strangle.enabled: true` als
`risk.allow_undefined_risk: true` staat (standaard allebei uit). Daarbovenop gelden de harde
limieten: allow-list `underlyings` (SPY, QQQ, IWM), `max_per_underlying: 1` (het schema
staat niet meer toe: een tweede positie zou toch 0 eenheden krijgen), alleen in het
regime RICH, het stressverlies-plafond hierboven, stop op 2 × credit,
`short_delta_exit: 0.30`, sluiten bij 21 DTE, en bij STRESS op SPY gaat alles met onbeperkt
risico direct dicht. **Alessio beslist of dit aan gaat**; dit is geen aanbeveling om het aan
te zetten.

### Volatility trading

| Sleutel | Standaard | Wat het doet | Aanzetten |
|---|---|---|---|
| `strategies.calendar_spread.entry_rule` | `ivp` | toelating van de calendar: `ivp` (oude IVP-regel), `forward_vol` of `either`. Forward-vol: σ_f² = (σ₂²T₂ − σ₁²T₁)/(T₂ − T₁) ≤ `max_fwd_vol_rv_ratio` (1,00) × σ(h₂−h₁) en IV near ≥ `min_near_iv_rv_ratio` (1,05) × σ(h₁). `long_dte` moet helemaal na `short_dte` liggen (het schema eist dat) | `forward_vol` of `either` |
| `strategies.gamma_scalp.enabled` | `false` | long ATM-straddle (30 tot 60 DTE) als IV ≤ `max_iv_rv_ratio` (0,90) × σ(h), gedelta-hedged. Exits: winstdoel (1,30 × debit, met de gerealiseerde hedge-P&L), stop (60 %), 14 DTE of IV terug op σ(h) | `enabled: true` én `hedging.enabled: true` |
| `hedging.enabled` | `false` | de delta-hedger: per onderliggende de netto delta van de posities van `hedging.strategies` (standaard `[gamma_scalp]`) terug in aandelen zodra \|delta\| > `band_shares` (20); plafond `max_notional_frac` (0,5) × kapitaal; elke `interval_seconds` (300). Alleen op `universe.share_eligible` (AAPL, MSFT): US-ETF's kunnen vanuit de EU niet als aandeel. Een hedge gaat door de risicopoort; die houdt hetzelfde plafond aan (`stock_notional_usd`) en controleert dat een sluitorder de aandelen-leg echt naar nul brengt. In HALT en UNWIND alleen afbouwen: de order moet de aandelen op die onderliggende in absolute waarde verkleinen én de delta van het boek verlagen; nieuwe aandelen (ook een hedge vanaf nul) niet | `true`, samen met `gamma_scalp` |
| `paper.stock_spread_bps`, `commission_per_share`, `min_commission_per_order` | `2`, `0.005`, `1.0` | kosten van aandelenorders in de paper-broker | alleen relevant met hedging |

### Executie en gamma management

| Sleutel | Standaard | Wat het doet | Aanzetten |
|---|---|---|---|
| `execution.limit_anchor` | `mid` | `fair` laat een entry beginnen op de QuantLib-waarde op de gladde smile in plaats van op de mid (alleen ENTRY) | `fair` |
| `execution.max_edge_frac` | `0.5` | bij `fair`: de walk concedeert nooit voorbij fair + frac × \|natural − mid\|. Ligt zelfs de eerste limiet al voorbij dat plafond, dan wordt de entry overgeslagen (`entry_skipped`, reden `max_edge`) | getal aanpassen |
| `execution.max_spread_frac_fair` | `0.25` | bij `fair`: geen entry als 2 × \|natural − mid\| > frac × \|fair\| (`entry_skipped`, reden `spread_vs_fair`) | getal aanpassen |
| `risk.max_gamma_loss_frac_credit` | `null` (uit) | positie met credit gaat dicht (STOP `gamma_risk`) als ½·\|Γ\|·(S·σ_1d)² > frac × credit. Zonder keten of met een leg zonder Greeks is het verlies onbekend: de regel sluit dan niet, maar meldt het één keer per positie (journal `gamma_unknown`, alarm `options_gamma_unknown`) | `0.5` aanbevolen |

### Sizing op risico

| Sleutel | Standaard | Wat het doet | Aanzetten |
|---|---|---|---|
| `sizing.daily_risk_frac` | `null` (uit) | eenheden ≤ floor(frac × kapitaal / σ_unit), met σ_unit de verwachte 1-dags P&L-σ per eenheid uit Δ, Γ en V; zonder Greeks 0 eenheden | bijvoorbeeld `0.005` |
| `sizing.vol_of_vol_daily_points` | `1.0` | verwachte dagbeweging van de IV in volpunten (vega-term van σ_unit) | getal aanpassen |
| `sizing.drawdown_brake` | `[0.05, 0.10]` (aan) | eenheden × factor die lineair van 1 naar 0,5 loopt tussen die drawdowns van de equity-piek | `null` = uit. In de standaardreplay niet actief (drawdown 0,33 %) |

## Wat voor een particulier niet haalbaar is

* **Market making** in de echte zin (beide kanten quoten, spread verdienen, rebates) is niet
  haalbaar: geen co-locatie, geen market-maker-rebates, en elke fill op onze limiet is
  waarschijnlijk adverse selection. Wat wel zinvol is, en wat deze engine doet: niet de spread
  weggeven (limieten op fair value), gamma beheersen en posities bijsturen.
* **Statistische arbitrage** in de echte zin (dispersie, honderden namen, snelle executie) past
  niet bij 25k en vijf onderliggenden. Wat wel kan: relative value, dus premie verkopen waar de
  VRP het rijkst is en kopen waar hij het goedkoopst is. Dat is geen arbitrage en heet ook zo
  niet in de code.
* **Delta hedging met aandelen** kan alleen op AAPL en MSFT (`share_eligible`). US-ETF's (SPY,
  QQQ, IWM) mogen vanuit de EU niet als aandeel gekocht worden (geen KID). Hedgen met
  synthetische aandelen of futures is bij 25k te grof (100 delta per stap tegen ±12 delta
  beweging per dag per straddle). ETF-posities sturen daarom bij met opties.
* **Aandelen uit assignment of uitoefening** van een gehedgede positie (bijvoorbeeld een
  gamma_scalp-leg die ITM expireert) horen bij geen aandelen-leg en worden **niet automatisch
  afgebouwd**: de reconciliatie accepteert hele lots van 100, dus ze blijven bij de broker
  staan, ongehedged. De exits (14 DTE, sluiten vóór expiratie) maken dit zeldzaam. Het alarm
  `options_assignment` gaat ook af bij een uitoefening van een strategie uit
  `hedging.strategies` (met hedging aan); sluit de aandelen dan met de hand.
* **Short strangles** hebben onbeperkt risico en blijven standaard uit
  (`risk.allow_undefined_risk: false`). Alessio beslist of ze aan gaan.

## Historische simulatie (echte koers en IV, ketens uit het model)

```bash
docker compose -f infra/compose/docker-compose.yml run --rm --no-deps -e IB_CLIENT_ID=18 \
  options-paper python -I -m tradebot.options historical \
  --out /var/lib/tradebot/historical/markt --start 2026-06-01 --end 2026-11-25
docker compose -f infra/compose/docker-compose.yml run --rm --no-deps options-paper \
  python -I -m tradebot.options replay \
  --market /var/lib/tradebot/historical/markt --out /var/lib/tradebot/historical/run
```

`historical` vraagt bij IB Gateway (alleen lezen, client-id 18) twee jaar dagbars en IBKR's
dagelijkse 30-daagse IV per onderliggende, en dividenden en earnings bij yfinance. De ketens
worden met QuantLib gereconstrueerd: het IV-niveau is echt, de smile, de spreads en het open
interest komen uit het model (`options.data.historical`). De IV van vóór de startdatum gaat mee
als IV-historie, zodat IVR vanaf dag één werkt. De uitkomst meet de regels op echte koers- en
IV-paden, **niet** echte fills. Een maand duurt ongeveer een kwartier.

## De paper-dienst tegen IBKR

Vereisten: een IBKR-paper-login (account-id begint met `DU`) en IB Gateway in paper-modus.

### Met Docker

```bash
cp infra/compose/.env.example infra/compose/.env     # TWS_USERID, TWS_PASSWORD, IB_ACCOUNT
docker compose -f infra/compose/docker-compose.yml up -d --build ib-gateway options-paper
docker compose -f infra/compose/docker-compose.yml logs -f options-paper
```

De gateway-container staat vast op `TRADING_MODE: paper`. De dienst verbindt via poort 4004.

### Met systemd

Zie de kop van `infra/systemd/tradebot-options-paper.service`. IB Gateway draait dan op
dezelfde machine in paper-modus op poort 4002.

### Wat de dienst doet

- Elke 15 seconden een ronde; in de sessie elke 5 minuten nieuwe ketens van IBKR.
- Entries alleen op de scantijden (10:00 en 13:30 New York), niet in de eerste 15 en de
  laatste 30 minuten, en niet op een halve dag na 12:30.
- Beheer van open posities elke minuut: winst nemen, stop-loss, sluiten voor expiratie,
  bijsturen of rollen volgens de strategie.
- Elke order gaat eerst door de risicopoort. Fouten die nooit mogen (te grote order, verkeerde
  onderliggende, prijs buiten bid/ask) zijn een `SafetyBreach` en zetten de HALT.
- Na de sluiting: ATM-IV vastleggen, expiraties en assignments verwerken, equity en staat
  bewaren.

## Ingrijpen

Alle bestanden staan in de run-map (`/var/lib/tradebot/options`).

| Wat | Hoe | Effect |
|---|---|---|
| Geen nieuwe entries | `touch HALT` | Beheer en sluiten gaan door |
| Alles sluiten | `touch UNWIND` | Elke positie wordt met voorrang gesloten |
| Hervatten | `rm HALT` of `rm UNWIND` | Volgende ronde weer normaal |
| Stoppen | `docker compose stop options-paper` of `systemctl stop ...` | Open orders geannuleerd, staat bewaard |

Een HALT door de dienst zelf (drawdown, drie fouten op rij, `SafetyBreach`) staat in
`halt/halt.json`. Die gaat pas weg met een expliciete vrijgave via de `HaltStore`
(`release(operator=..., justification=...)`), net als bij het crypto-boek.

## Alarmen en wat te doen

| Alarm | Betekenis | Doen |
|---|---|---|
| `options_halt` | De dienst haltte | Lees `halt/halt.json` en het journal; geef pas vrij als de oorzaak weg is |
| `options_recon_break` | Broker en staat verschillen | Vergelijk posities in TWS met `state.json`; entries op die onderliggende staan stil tot het weer klopt |
| `options_assignment` | Een short leg is assigned, of een long leg van een gehedgede strategie uitgeoefend | Controleer de aandelenpositie; sluit of laat de strategie verder beheren. Aandelen uit een gehedgede positie worden niet automatisch afgebouwd: sluit ze met de hand |
| `options_gamma_unknown` | Het gamma-verlies van een positie met credit is onbekend (geen keten of Greeks) | De gamma-regel sluit die positie niet; de stops van de strategie blijven gelden. Kijk waarom de Greeks ontbreken |
| `options_short_past_expiry` | Een short optie staat na expiratie nog in de staat | Controleer bij IBKR; dit hoort nooit te gebeuren |
| `options_task_*` | Een taak faalde | Lees het journal (`task_error`); drie op rij is HALT |

Assignments bij IBKR komen binnen als positiewijziging. De reconcile ziet het verschil,
blokkeert entries op die onderliggende en alarmeert.

## Configuratie

`conf/options/default.yaml` (schema: `src/tradebot/schemas/options.py`). Belangrijk:

- `account.capital_usd`: 25.000 (fictief). Alle risicogrenzen zijn fracties hiervan.
- `strategies.*.enabled`: CSP, covered call en de wheel staan uit bij 25.000.
- `risk.*`: grenzen op delta, vega, gamma, verlies per onderliggende, buying power,
  scenario-verlies, aantal posities, dagverlies (stop entries 2 %, afbouwen 4 %) en
  drawdown (HALT 10 %).

## Live

Live bestaat niet in het uitvoerpad. Wat er echt in zit, zijn twee hekken:

1. `require_paper` (`src/tradebot/options/live.py`): `IBKRBroker.connect` en elke `submit` weigeren een
   account dat niet met `DU` begint (`LiveRefused`).
2. `TRADING_MODE: paper` in `infra/compose/docker-compose.yml` en geen CLI-opdracht of config met
   `mode: live` (een test bewaakt dat).

`live.gather` en `live.live_decision` (de vijf voorwaarden: mode `live`, bevestigingszin,
`LIVE_APPROVED` met de account-hash, echt account, geslaagde paper-killgate) zijn een
ontwerp-schakelaar die nergens wordt aangeroepen: ze zijn getest maar niet aangesloten. Een
toekomstige live-stap moet ze eerst in het pad zetten; tot dan beschermen alleen bovenstaande twee
hekken. Ga er dus niet van uit dat vijf hekken actief zijn.
