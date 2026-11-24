# Ontwerp: van crypto naar aandelenopties als actieve setup (2026-11-24)

Dit is fase 0 van `prompts/options-pivot-prompt.md` (de leidraad in de projectbestanden). Het
document bevat **geen code**. Het legt vast wat er gebouwd wordt, in welke volgorde, en welke
keuzes nog bij de eigenaar liggen. Pas na akkoord begint fase 1.

Stand van de repo: `main` op `e237394`. Alles wat hieronder over de huidige code staat, is op die
commit nagekeken, niet uit de leidraad overgenomen.

## Samenvatting

- **Options vervangen crypto als actieve setup.** Nieuw subpackage `tradebot.options` met
  regelgebaseerde strategieën op US-genoteerde aandelen- en ETF-opties. Prijzen, IV en Greeks
  komen uit QuantLib, met Amerikaanse uitoefening en dividenden.
- **Crypto gaat in standby, niet weg.** Fase 1 zet de U1-dienst achter een compose-profiel
  `crypto`, zet de shadow-crons uit en maakt options de standaard in README en config. Geen
  crypto-module, -test of -config wordt verwijderd; de dekking blijft 100 % over het geheel.
- **Alles met fictief geld.** Replay en IBKR-paper (standaard 25.000 USD) zijn de enige modi. De
  live-schakelaar wordt gebouwd en met een killgate getest, maar staat nergens aan.
- **Geen training.** Strategieën zijn vaste regels uit config. Historische data dient hooguit om
  die regels te meten, met preregistratie.
- **Drie bevindingen die de leidraad nog niet had** (details in sectie 6):
  1. `ib_async` 2.1.0 eist `tzdata<2026`, terwijl de lock `tzdata==2026.3` heeft. Voorstel: pin
     `ib_async==2.0.1`, die die grens niet heeft.
  2. Een IBKR-paper-account krijgt realtime OPRA-data alleen via de abonnementen van een
     **gefinancierd live-account**. Zonder dat is paper alleen met vertraagde data mogelijk.
  3. Als Nederlandse retailklant mag je US-ETF's (SPY, QQQ, IWM) waarschijnlijk niet als aandeel
     kopen (PRIIPs/KID). Covered calls en de wheel op die ETF's werken dan live niet. Voor paper
     maakt het niets uit, maar het bepaalt het universum voor later.

## 1. Antwoorden en aannames voor Q1 tot en met Q13

| # | Vraag | Antwoord in dit ontwerp | Status |
|---|---|---|---|
| Q1 | Opties óp aandelen of aandelen én opties? | Opties op aandelen en ETF's. Het onderliggende wordt alleen gehouden voor covered calls en na assignment. | aanname |
| Q2 | US of Europa? | US-genoteerd (OPRA). Eurex/Euronext later achter dezelfde adapter. | aanname |
| Q3 | Indexopties (SPX, XSP)? | Nee in v1. De Europese pricer bestaat wel, dus later toevoegen is klein. | aanname |
| Q4 | Broker en data | IBKR via `ib_async`, paper-account. Realtime data via IBKR (zie 6.2 voor de voorwaarde). Geen betaalde historische feed in v1. | aanname, zie open vraag A |
| Q5 | Rekeningtype | Reg-T margin, kapitaal in config. Alleen defined risk plus volledig gedekte short opties. Naked short staat uit achter `allow_undefined_risk`. | aanname |
| Q6 | Historische data voor backtest? | Ja, alleen om vaste regels te meten, met preregistratie. Er is nu geen historische ketendata, dus **forward paper-trading is het eerste bewijs**. | aanname |
| Q7 | ML-dependencies naar een extra? | Nee in deze ronde. Fase 8 doet hooguit een voorstel. | beslist (standaard) |
| Q8 | Universum | SPY, QQQ, IWM, AAPL, MSFT, met de liquiditeitsfilters uit 4.1. | aanname, zie open vraag C |
| Q9 | Frequentie | Scan één keer per dag om 15:00 New York, beheer elke 60 s, risico elke 30 s. Geen 0DTE. | aanname |
| Q10 | Waar draait het? | Zelfde server, eigen compose-service `tradebot-options` met eigen state, HALT en UNWIND. IB Gateway als aparte container. | aanname |
| Q11 | Draait U1 live met geld? | Nee, er draait niets. | beantwoord |
| Q12 | Crypto-meetlopen door laten lopen? | Nee, alles uit. De forward-lezing vanaf 2027-04-01 wordt alleen genoemd in `docs/crypto-standby.md`. | beantwoord |
| Q13 | Kapitaal | Fictief, `capital_usd: 25000` in config. | beantwoord |

## 2. Overstapplan crypto naar standby (fase 1)

Nagekeken op `main`: de U1-dienst is de enige service in `infra/compose/docker-compose.yml`, de
U1-config staat op `mode: dry_run`, `kraken-shadow.yml` draait dagelijks om 06:23 UTC en
`infra/k8s/cron-carry-shadow.yaml` maandelijks op de 5e.

| Onderdeel | Wijziging in fase 1 |
|---|---|
| `infra/compose/docker-compose.yml` | Service `u1` krijgt `profiles: ["crypto"]`. Zonder profiel start er in fase 1 dus niets; vanaf fase 7 start `tradebot-options` zonder profiel. |
| `infra/systemd/tradebot-u1.service` | Blijft staan. Het runbook en `docs/crypto-standby.md` noemen `systemctl disable --now tradebot-u1`. |
| `infra/k8s/cron-carry-shadow.yaml` | `suspend: true`. |
| `.github/workflows/kraken-shadow.yml` | Het `schedule`-blok eruit, alleen `workflow_dispatch` blijft. |
| `.github/workflows/research-run.yml` | Gepland draaien uit, handmatig blijft. |
| `.github/workflows/paper-trade-ci.yml` | Ongewijzigd: de crypto-code bestaat nog en moet blijven werken. |
| `README.md`, `pyproject.toml` (`description`) | "Options-setup; crypto in standby". README krijgt een sectie *Crypto (standby)*. |
| Module-docstrings van `carry/`, `systematic/`, `train/`, `tune/`, `labeling/`, `cv/`, `selection/`, `featurestore/` | Eén regel: *standby sinds 2026-11, zie docs/crypto-standby.md*. Geen logica verandert. |
| Nieuw: `docs/crypto-standby.md` | Wat in standby staat, waarom, de laatste stand van het boek (verwijzing naar `2026-10-25-robust-book-v9-kraken-cash.md`), en de stappen om het weer aan te zetten. |
| Nieuwe test | `docker compose config --services` zonder profiel noemt `u1` niet, met `--profile crypto` wel. Plus een bestaande dry-run-smoke van de U1-dienst blijft groen. |

Fase 1 raakt geen Python in `src/`, behalve docstrings. Dat houdt de PR klein en de dekking
gelijk.

## 3. Module-indeling

De indeling uit sectie 4 van de leidraad blijft staan. Drie keuzes die uit de code volgen:

1. **OMS: naast zetten, niet uitbreiden.** `oms/order.py` is één leg per order en kent
   `OrderType.MARKET`. Een optieorder is een combo van legs en mag nooit market zijn. Uitbreiden
   zou de crypto-OMS veranderen, en dat verbiedt sectie 13. Dus: `options/execution/orders.py`
   met een eigen `ComboOrder` (alleen limiet) en een eigen append-only JSONL-journal. Het patroon
   van `oms/audit_log.py` en `oms/reconciler.py` wordt gevolgd, de klassen niet hergebruikt.
2. **Gedeelde onderdelen uit `carry/` worden eruit getrokken, niet geïmporteerd uit standby-code.**
   `SingleInstance` (`carry/safety.py`) en de alarmkanalen (`carry/alerts.py`) verhuizen naar
   `tradebot/utils/single_instance.py` en `tradebot/monitoring/channels.py`. De oude namen in
   `carry/` worden re-exports, dus de crypto-tests blijven ongewijzigd groen. Zo hangt de actieve
   setup niet af van een package dat in standby staat.
3. **Direct hergebruikt, want niet crypto-specifiek:** `risk/kill_switches.py` (`HaltStore`,
   drawdown-breaker, daily-loss-governor), `live/sigterm.py`, `monitoring/metrics.py`,
   `utils/failfast.py`, `utils/time.py`, `volatility/` (gerealiseerde vol voor IV-vs-RV) en
   `registry/` voor preregistratie.

Verder: de NYSE-feestdagen komen uit `ql.UnitedStates(ql.UnitedStates.NYSE)`, die QuantLib al
meelevert. Alleen de halve handelsdagen komen in een kleine tabel in config, met een test. Er komt
dus geen extra kalenderpakket.

Afhankelijkheidsrichting: `contracts` ← `pricing` ← `chain` ← `strategies` ← `risk` ← `engine`.
`execution` en `data` alleen via Protocols. Strategieën kennen de broker niet. Een test bewijst
dat de broker-adapter alleen via de risicopoort bereikbaar is.

### Pricing in het kort

- Scan over hele ketens: Barone-Adesi-Whaley. Beslissing en risico: binomiaal Leisen-Reimer (401
  stappen), FD bij discrete dividenden. Europees analytisch als referentie.
- Looptijd tot 16:00 New York op de expiratiedag, Actual/365 Fixed.
- Rente in v1 vlak uit config (`risk_free_rate`); een feed is een latere uitbreiding.
- Dividenden: discrete dividenden uit de dividendkalender (zie 4.1), anders een continue yield
  uit config.
- IV: `impliedVolatility` op mid, plus bid en ask apart. Buiten de arbitragegrenzen of zonder
  convergentie komt er een `IVStatus`-foutcode, nooit een NaN.
- Vega en rho voor Amerikaans via centrale bumps (1 volpunt, 1 bp). Eenheden zitten in het type.
- `evaluationDate` alleen via een contextmanager; geen threads binnen één proces.
- Acceptatietests zoals in sectie 5.7 van de leidraad, waaronder een benchmark van 2.000 contracten
  in minder dan 2 s.

## 4. Strategieën

### 4.1 Gemeenschappelijk

- Pure functies `scan(market, book)` en `manage(position, market)`, alle drempels in
  `conf/options/default.yaml`, gevalideerd met pydantic (`extra="forbid"`).
- Filters per leg: open interest ≥ 500, dagvolume ≥ 100, bid > 0, spread ≤ max(0,10 USD; 10 % van
  mid). Onderliggende: gemiddeld dagvolume ≥ 1 mln aandelen.
- **Earnings- en dividenddata uit `yfinance`**, die al een harde dependency is. Die bron is niet
  altijd betrouwbaar, dus de regel is streng: ontbreekt de datum of is hij ouder dan een dag, dan
  geen nieuwe entry op die onderliggende. Een betere bron is een latere keuze.
- Sizing: max verlies per positie ≤ 2 % van het kapitaal; covered call en CSP ≤ de toewijzing per
  onderliggende. Naar beneden afronden; 0 contracten is geen trade.
- Rangschikking deterministisch: credit gedeeld door max verlies (credit-strategieën), of
  verwachte winst gedeeld door debit (debit-strategieën). Gelijke stand: op contract-id.
- Elke short optie wordt uiterlijk op de handelsdag vóór expiratie gesloten.

### 4.2 Regels per strategie

De regels zijn die uit sectie 6.2 van de leidraad, zonder wijziging. Kort:

| Strategie | Entry | Exit |
|---|---|---|
| Covered call | 30-45 DTE, delta 0,20-0,30, strike boven kostprijs | 50 % winst; roll bij 21 DTE; terugkopen vóór ex-dividend als extrinsiek < dividend |
| Cash-secured put | 30-45 DTE, delta 0,20-0,30, IVR ≥ 30 | 50 %; sluit of roll bij 21 DTE; assignment → covered call |
| Wheel | toestandsmachine CSP → assignment → covered call | test per overgang |
| Credit spread | 30-45 DTE, short delta 0,20-0,30, credit ≥ ⅓ breedte, IVR ≥ 30, richting uit prijs vs 50d SMA | 50 %; stop bij 2 × credit; 21 DTE; short leg ITM |
| Debit vertical | 30-60 DTE, long delta 0,50-0,60, debit ≤ 50 % breedte, IVR ≤ 30 | +50-100 %; −50 %; 21 DTE |
| Iron condor | 30-45 DTE, short delta ±0,16, credit ≥ ⅓ breedte, IVR ≥ 30, geen earnings | 50 %; 2 × credit; 21 DTE; één bijsturing als short delta > 0,30 |
| Long straddle/strangle | 5-15 dagen vóór earnings, of IVR ≤ 20 | vóór de gebeurtenis, +25-50 %, −50 %, uiterlijk 7 DTE |
| Short straddle/strangle | **uit**, alleen met `allow_undefined_risk` | |
| Calendar | kort 20-35 DTE, lang 50-90 DTE, ATM, contango of vlak | +25 %; −50 %; vóór expiratie korte leg |

**IV rank heeft 252 handelsdagen ATM-IV-historie nodig.** Die bestaat niet: de dienst gaat hem
vanaf dag één zelf opbouwen. Tot er genoeg is, staat de IVR-status op "onvoldoende historie" en
doen de strategieën die IVR eisen (CSP, credit spread, debit vertical, iron condor, long
straddle via IVR) **niets**. In de eerste maanden van paper-trading handelen dus alleen de
covered call (zonder aandelen in bezit: niets), de calendar en de straddle rond earnings. Dat is
een verwacht en eerlijk gevolg van "niet trainen op historie", maar het maakt paper-trading de
eerste maanden stil. Zie open vraag B voor een alternatief.

## 5. Risico

De grenzen uit sectie 8.2 van de leidraad, bij C = 25.000 USD:

| Grens | Standaard | Bij C = 25k |
|---|---|---|
| Beta-weighted delta | ≤ 0,30 × C | ≈ 7.500 USD SPY-equivalent |
| Totale vega | ≤ 0,5 % van C per volpunt | 125 USD per volpunt |
| Gamma-verlies bij −5 % | ≤ 3 % van C | 750 USD |
| Max verlies per onderliggende | ≤ 5 % van C | 1.250 USD |
| Max verlies per positie | ≤ 2 % van C | 500 USD |
| Buying power | ≤ 50 % van beschikbaar | |
| Posities | max 10 open, max 3 nieuw per dag | |
| Dagverlies | 2 % geen entries, 4 % afbouwen | 500 / 1.000 USD |
| Drawdown | 10 % → HALT via `HaltStore` | 2.500 USD |
| Slechtste scenario | spot ±5/10/20 % × IV −10…+20 volpunten ≤ 8 % van C | 2.000 USD |

Gevolg van 500 USD max verlies per positie: een credit spread van 5 USD breed met 1,70 credit heeft
330 USD max verlies, dus 1 contract. Een cash-secured put vraagt strike × 100 aan cash: op SPY
in de orde van 60.000 USD, op AAPL of MSFT ruim boven 20.000 USD. Dat past niet in 25k, of het
legt bijna het hele kapitaal vast in één positie. Daarbij is het theoretische max verlies van een
CSP of covered call ook strike × 100, ver boven de 1.250 USD per onderliggende. **CSP, covered
call en wheel doen bij 25k op het huidige universum dus niets.** Zie open vraag C.

Gedrag: een grens kan alleen weigeren of verkleinen. Een order die een overschrijding verkleint,
mag altijd. `SafetyBreach` zet de dienst stil; `Deferred` slaat de ronde over.

**PDT-regel.** Een margin-account onder 25.000 USD mag bij IBKR maximaal 3 daytrades per 5 dagen.
Met precies 25k fictief kapitaal zakt de eerste verliesdag het account eronder. De engine telt
daarom zelf daytrades (openen en sluiten op dezelfde dag) en weigert een vierde binnen 5 dagen
zolang de equity onder 25k staat. De strategieën doen bijna nooit een daytrade, dus dit is een
vangnet.

## 6. Broker, data, kosten en dependencies

### 6.1 Dependencies

Nagekeken op PyPI op 2026-11-24:

| Pakket | Versie | Wheel | Bijzonderheid |
|---|---|---|---|
| `QuantLib` | 1.44 | `cp39-abi3-manylinux_2_28_x86_64`, 21 MB | Werkt op 3.13. Geen numpy-eis, dus geen botsing met `numpy==1.26.4`. Geen type-stubs: `ignore_missing_imports`. |
| `ib_async` | **2.0.1** voorgesteld | pure Python | 2.1.0 eist `tzdata<2026`, de lock heeft `tzdata==2026.3` (via pandas). 2.0.1 heeft die grens niet en hangt af van `eventkit` en `nest_asyncio`. |

Per pakket in fase 2 (QuantLib) en fase 7 (`ib_async`): `pyproject.toml`, de locks opnieuw, de
lijst harde dependencies in `hygiene.yml` (`QuantLib`, `ib_async`), de mypy-override, en pip-audit
groen. De Docker-image wordt ongeveer 21 MB groter. Als pip-audit op `eventkit` of `nest_asyncio`
iets vindt, stop ik en meld ik het, in plaats van een uitzondering toe te voegen.

`ib_async` komt pas in fase 7. Tot dan draait alles op de `PaperBroker` in replay, zonder
netwerk.

### 6.2 IBKR-paper en marktdata

- Een IBKR-paper-account (`DU…`) hoort altijd bij een live-account. Realtime marktdata in paper
  bestaat alleen als het **live-account** de abonnementen heeft en ze met het paper-account deelt.
- Voor US-opties is dat het OPRA-abonnement (non-professional), plus meestal een US-aandelenbundel.
  De bedragen zijn klein (enkele USD per maand), maar IBKR stelt een minimum aan het saldo van het
  live-account. **De exacte prijzen en het minimum moeten op de IBKR-site gecontroleerd worden;
  ik heb ze niet uit een officiële bron kunnen bevestigen.**
- Zonder live-account met abonnement krijgt paper vertraagde data (15 minuten). Voor een
  dagelijkse scan om 15:00 en exits op 50 % winst is dat werkbaar maar niet realistisch voor
  fills. De engine markeert vertraagde quotes en de rapporten melden het.
- Historische optieketens (ThetaData, Polygon/Massive, ORATS) kosten tientallen USD per maand of
  meer. Niet in v1: forward paper-trading is het bewijs (Q6).

### 6.3 Productbeperkingen voor een Nederlandse retailklant

US-ETF's als SPY, QQQ en IWM hebben geen EU-KID, dus een retailklant in de EU kan ze bij IBKR
meestal niet als aandeel kopen. Opties óp die ETF's zijn wel handelbaar, en een assignment levert
wel aandelen op. Gevolg voor later live: covered calls en de wheel op ETF's kunnen niet bewust
opgebouwd worden. In paper speelt dit niet, maar het universum houdt er rekening mee: de
strategieën die aandelen nodig hebben, krijgen een eigen lijst (`share_eligible`), los van het
optie-universum. **Ook dit moet bij IBKR gecontroleerd worden.**

## 7. Engine en veiligheid

Zoals sectie 7 en 10 van de leidraad. De belangrijkste punten:

- Modi `replay` (parquet-ketens, `PaperBroker`, deterministisch), `paper` (IBKR-paper) en
  `live`. Een state-bestand hoort bij één modus.
- Eén asyncio-proces met `SingleInstance`. Geen entries in de eerste en laatste 15 minuten.
- Alleen limietorders, multi-leg als IBKR-combo (BAG), price-walk vanaf mid, deterministische
  client-id's tegen dubbele orders na een herstart.
- Reconciliatie bij start en elke 5 minuten; een verschil stopt nieuwe entries en alarmeert.
- Assignment detecteren na de sluiting en bij start.
- Live vereist alle vijf voorwaarden (config `mode: live`, de omgevingsvariabele, account op de
  allowlist en niet `DU…`, een nieuw leeg state-bestand, `max_capital_usd` en `max_order_usd`
  gezet). De killgate test elke combinatie met één ontbrekende voorwaarde. Paper is overal de
  standaard; `mode: live` komt in geen enkele config of compose-bestand.
- IB Gateway draait als aparte container in compose (de officiële image wordt in fase 7 gekozen en
  op digest gepind), met alleen de paper-poort 4002 open naar `tradebot-options`.

## 8. Impact op CI

| Poort | Wat er verandert |
|---|---|
| `ci.yml` | Nieuwe tests vallen onder de bestaande markers. ruff en mypy over de nieuwe code. |
| `inventory.yml` | Locks bijgewerkt; dekking blijft `fail_under = 100`, ook over `tradebot.options`. |
| `hygiene.yml` | `QuantLib` en `ib_async` in de lijst harde dependencies; `schemas/options.py` onder `mypy --strict`. |
| `security-scan.yml` | pip-audit en bandit over de nieuwe locks en code. Geen uitzonderingen. |
| `docs.yml` | markdownlint over de nieuwe documenten (dit document is lokaal gecontroleerd met markdownlint-cli2 0.13.0). |
| `paper-trade-ci.yml` | Ongewijzigd. Nieuw ernaast: `options-paper-smoke` met de e2e-replay, op PR's en wekelijks. |
| `kraken-shadow.yml`, `research-run.yml` | Cron uit, handmatig blijft. |
| `docker.yml` | Image ~21 MB groter. |

De IBKR-adapter wordt getest met een fake van de `ib_async`-interface. `# pragma: no cover`
alleen voor de dunste netwerkschil, met reden op dezelfde regel.

## 9. Fasering

Elke fase is één draft-PR met groene CI. Volgorde zoals in sectie 12 van de leidraad:

0. Dit ontwerp. **Wacht op akkoord.**
1. Crypto in standby (sectie 2). Klein, geen nieuwe code.
2. QuantLib, `contracts`, `calendar`, `rates`, `pricing/` met alle pricingtests; `conf/options/`
   als standaardconfig; `SingleInstance` en alarmkanalen naar gedeelde modules.
3. Ketens, oppervlak, IVR-opslag, fixture-ketens met vast seed.
4. Strategieën, elk met unit- en scenariotests.
5. Risicolaag: aggregatie, grenzen, scenario's, marge, PDT-teller.
6. Engine met `PaperBroker`, replay, CLI, e2e en `options-paper-smoke`.
7. `ib_async`, IBKR-adapter, compose-service met IB Gateway, runbook, live-schakelaar met
   killgate. Alleen tegen een paper-account.
8. Voorstel over de ML-dependencies (Q7), zonder code.

Fase 2 tot en met 6 hebben geen IBKR-account nodig. Pas bij fase 7 moet het paper-account er zijn.

## 10. Risico's

| Risico | Gevolg | Maatregel |
|---|---|---|
| IVR-historie ontbreekt | Maandenlang weinig of geen trades in paper | Open vraag B |
| 25k is klein voor CSP/wheel | Die strategieën doen niets op het huidige universum | Open vraag C |
| Vertraagde data in paper | Fills en exits wijken af van live | Quotes markeren; open vraag A |
| `yfinance` voor earnings en dividenden is wankel | Gemiste earnings binnen de looptijd | Strenge regel: geen data, geen entry |
| QuantLib `evaluationDate` is globaal | Verkeerde waardering bij gelijktijdig gebruik | Contextmanager, geen threads, isolatietest |
| `ib_async` 2.0.1 is niet de nieuwste | Mogelijk mist een fix | Bij elke lock-update opnieuw kijken of 2.x de `tzdata`-grens loslaat |
| EU-productbeperkingen (PRIIPs) | Covered calls op ETF's live onmogelijk | `share_eligible`-lijst, controleren bij IBKR |
| Vroege assignment rond ex-dividend | Onverwachte aandelenpositie | Regel in covered call en credit spread, plus dagelijkse detectie |
| Schrijfwerk aan `carry/` voor gedeelde modules | Crypto-tests breken | Alleen verplaatsen met re-export, crypto-tests ongewijzigd |

## 11. Open vragen voor Alessio

Alle andere punten gebruiken de standaard uit sectie 1. Deze vier hebben een antwoord nodig vóór
de fase die erbij staat; tot dan ga ik uit van de aanname.

- **A. Marktdata (vóór fase 7).** Heb je, of wil je, een IBKR-live-account met het
  OPRA-abonnement, zodat paper realtime data krijgt? *Aanname: nee, paper met vertraagde data, en
  de rapporten zeggen dat erbij.*
- **B. IV rank zonder historie (vóór fase 4).** Mogen de strategieën die IVR eisen, zolang er
  minder dan 252 dagen historie is, een vervangende regel gebruiken die geen historie nodig heeft
  (bijvoorbeeld IV 30d boven gerealiseerde vol 20d)? Dat is een vaste regel, geen training.
  *Aanname: ja, met de vervangende regel apart gelogd en na 252 dagen automatisch vervangen door
  IVR.*
- **C. Universum bij 25k (vóór fase 4).** CSP, covered call en wheel passen niet op SPY, QQQ,
  IWM, AAPL en MSFT bij 25.000 USD. Wil je goedkopere namen toevoegen voor die strategieën (dan
  krijgen ze ook een eigen verliesmaat, zoals verlies bij −20 % in plaats van strike × 100), of ze
  in v1 laten slapen? *Aanname: laten slapen; spreads en iron condors op het huidige universum.*
- **D. `ib_async` 2.0.1 (vóór fase 7).** Akkoord met de pin op 2.0.1 vanwege de `tzdata`-botsing?
  *Aanname: ja.*
