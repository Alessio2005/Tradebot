# Detailontwerp: options-setup (2026-11-24)

Dit document werkt `2026-11-24-options-pivot-design.md` (het hoofdontwerp) uit tot op het niveau
van bestanden, typen, configsleutels, algoritmen en tests. Het bevat **geen productiecode**: de
codefragmenten zijn signaturen en pseudocode, bedoeld om over te beslissen. Waar dit document
een keuze maakt die in het hoofdontwerp als open vraag staat (A tot en met D), staat dat erbij.

Stand van de repo: `main` op `e237394`.

## Leeswijzer

| Sectie | Onderwerp | Fase |
|---|---|---|
| 1 | Fase 1 op bestandsniveau: crypto in standby | 1 |
| 2 | Datamodel en eenheden | 2 |
| 3 | Configuratie en schema | 2 |
| 4 | Kalender en tijd | 2 |
| 5 | Pricing, IV en Greeks | 2 |
| 6 | Ketens, filters, oppervlak en IV-maten | 3 |
| 7 | Strategieën: interface, selectie, sizing en beheer | 4 |
| 8 | Risicopoort | 5 |
| 9 | Engine: taken, toestanden en foutafhandeling | 6 |
| 10 | Orders, price-walk en de paper-broker | 6 |
| 11 | Staat en journal | 6 |
| 12 | IBKR-adapter | 7 |
| 13 | Live-schakelaar | 7 |
| 14 | Observability en alarmen | 6-7 |
| 15 | Replay-fixtures | 3, 6 |
| 16 | Testplan per fase | alle |
| 17 | Omvang per PR | alle |

## 1. Fase 1 op bestandsniveau: crypto in standby

Geen Python-logica verandert. Wat er verandert:

| Bestand | Wijziging |
|---|---|
| `infra/compose/docker-compose.yml` | Service `u1` krijgt `profiles: ["crypto"]`. Commentaar bovenaan: "crypto in standby, start met `docker compose --profile crypto up -d`". |
| `infra/k8s/cron-carry-shadow.yaml` | `spec.suspend: true`, met een regel commentaar naar `docs/crypto-standby.md`. |
| `.github/workflows/kraken-shadow.yml` | Het `schedule`-blok weg; `workflow_dispatch` blijft. |
| `.github/workflows/research-run.yml` | Geen wijziging: die draait al alleen handmatig (nagekeken; het hoofdontwerp zei hier ten onrechte "cron uit"). |
| `infra/systemd/tradebot-u1.service` | Geen wijziging. Uitzetten is een stap op de server (`systemctl disable --now tradebot-u1`), beschreven in `docs/crypto-standby.md`. |
| `README.md` | Nieuwe opening: Tradebot is een options-setup (in opbouw). Sectie *Crypto (standby)* met wat er staat, hoe je het aanzet, en een link naar de rapporten. De bestaande sectie *Stand (2026-10-27)* verhuist onder die kop. |
| `pyproject.toml` | `description = "Options-tradingsetup op US-aandelenopties (in opbouw); crypto-onderzoeksstack in standby."` Versie naar `0.5.0`. |
| `CHANGELOG.md` | Regel voor 0.5.0. |
| Module-docstrings `carry/__init__.py`, `systematic/__init__.py`, `train/__init__.py`, `tune/__init__.py`, `labeling/__init__.py`, `cv/__init__.py`, `selection/__init__.py`, `featurestore/__init__.py` | Eén regel: "Standby sinds 2026-11 (zie docs/crypto-standby.md): niet de actieve setup, wel onderhouden en getest." |
| `docs/runbook-kraken-u1.md` | Kop krijgt dezelfde standby-regel en een link. |
| Nieuw: `docs/crypto-standby.md` | Zie hieronder. |
| `tests/unit/test_infra_contract.py` | Nieuwe test `test_compose_starts_no_crypto_service_without_the_profile`: elke service zonder `profiles` is geen crypto-service, en `u1` heeft `profiles == ["crypto"]`. Nieuwe test `test_the_shadow_cron_is_suspended`. Nieuwe test `test_kraken_shadow_runs_only_by_hand`. De bestaande tests blijven ongewijzigd. |

De bestaande compose-test leest `services.u1` en blijft dus geldig. De dry-run-smoke van de
U1-dienst (`tests/unit/test_carry_service.py`) draait gewoon mee en bewijst dat de dienst nog
opstart.

Inhoud van `docs/crypto-standby.md`:

1. Wat in standby staat: de lijst uit sectie 1b van de leidraad.
2. Waarom: de beslissing van 2026-11-24, options vervangen crypto als actieve setup.
3. Laatste stand: verwijzingen naar `2026-10-25-robust-book-v9-kraken-cash.md` en
   `2026-10-25-venue-kraken-vs-bybit.md`. Geen nieuwe cijfers.
4. Weer aanzetten, in volgorde: `docker compose --profile crypto up -d u1`, of
   `systemctl enable --now tradebot-u1`; `suspend: false` in de k8s-cron; het `schedule`-blok
   terug in `kraken-shadow.yml`. De U1-config staat op `mode: dry_run` en blijft zo.
5. Wat er nog kan zonder iets aan te zetten: `python -I -m tradebot.carry.u1 shadow` met de
   hand, en de forward-lezing van v9 vanaf 2027-04-01 via `research-run`. Niet ingepland.

## 2. Datamodel en eenheden (`options/contracts.py`, `options/units.py`)

Alles is `@dataclass(frozen=True, slots=True)`. Geld is `float` in USD; datums zijn
`datetime.date`, tijden `datetime.datetime` met tijdzone `America/New_York`.

### 2.1 Eenheden

Eenheden staan in het type, zodat optellen van ongelijke grootheden een typefout is (mypy) en
een `require`-fout bij runtime.

| Type | Betekenis | Per |
|---|---|---|
| `DeltaShares` | delta in aandelen-equivalent | positie (al × multiplier × aantal × teken) |
| `GammaShares` | verandering van `DeltaShares` per 1 USD koersbeweging | positie |
| `VegaUSD` | USD per 1 volpunt (0,01 absolute vol) | positie |
| `ThetaUSD` | USD per kalenderdag | positie |
| `RhoUSD` | USD per 1 procentpunt rente | positie |
| `GreeksPerShare` | ruwe Greeks uit de pricer, per aandeel en per eenheid | contract, vóór schaling |

`GreeksPerShare.scale(qty: int, multiplier: int) -> PositionGreeks` is de enige overgang van
"per aandeel" naar "per positie". Een property-test bewijst: `scale(q) + scale(-q) == 0` en
`scale(a) + scale(b) == scale(a + b)`.

### 2.2 Contracten en quotes

```python
class Right(StrEnum): CALL = "C"; PUT = "P"

class OptionContract:
    underlying: str            # "SPY"
    expiry: date               # laatste handelsdag
    strike: float
    right: Right
    multiplier: int = 100
    exercise: Exercise = Exercise.AMERICAN
    con_id: int | None = None  # IBKR-contract-id, None in replay tot hij toegekend is
    def key(self) -> str: ...  # "SPY 2026-12-18 600 C", deterministisch

class Quote:
    bid: float | None
    ask: float | None
    last: float | None
    bid_size: int
    ask_size: int
    volume: int
    open_interest: int
    ts: datetime
    delayed: bool              # True bij vertraagde data (open vraag A)

class PricedOption:
    contract: OptionContract
    quote: Quote
    iv_mid: IVResult           # waarde of foutcode, zie 5.4
    iv_bid: IVResult
    iv_ask: IVResult
    greeks: GreeksPerShare | None   # None als iv_mid geen waarde heeft
    theo: float | None

class Chain:
    underlying: str
    spot: float
    asof: datetime
    options: tuple[PricedOption, ...]
    def expiries(self) -> tuple[date, ...]: ...
    def by_expiry(self, d: date) -> tuple[PricedOption, ...]: ...
```

### 2.3 Posities

```python
class Leg:
    contract: OptionContract | None   # None voor een aandelen-leg
    symbol: str                       # onderliggende bij een aandelen-leg
    qty: int                          # + long, − short, in contracten of aandelen
    entry_price: float                # per aandeel, zoals de broker rapporteert

class StrategyPosition:
    position_id: str                  # zie 10.3
    strategy: str                     # naam uit de registry
    underlying: str
    legs: tuple[Leg, ...]
    opened_at: datetime
    entry_net: float                  # + ontvangen credit, − betaald debit, totaal USD
    max_loss: float                   # USD, ≥ 0; inf alleen voor undefined risk
    adjustments: int                  # aantal bijsturingen (iron condor: max 1)
    phase: str                        # voor de wheel: "csp" / "shares" / "cc"
    status: PositionStatus            # OPEN / CLOSING / CLOSED
```

`max_loss` wordt bij de entry één keer berekend en daarna alleen bij een bijsturing opnieuw. Een
test per strategie legt de formule vast (sectie 7).

## 3. Configuratie en schema (`schemas/options.py`)

Twee bestanden: de strategie- en risicoconfig (`conf/options/default.yaml`, in git) en de
serverconfig (`conf/options/options.example.yaml`, voorbeeld; de echte staat op de server).
Beide worden geladen met pydantic-modellen op `StrictModel` (`extra="forbid"`, `frozen=True`)
uit `schemas/config.py`, onder `mypy --strict`. Een onbekende sleutel is een fout bij het laden,
niet een waarschuwing.

### 3.1 `conf/options/default.yaml` (schets, alle getallen zijn de standaard)

```yaml
universe:
  underlyings: [SPY, QQQ, IWM, AAPL, MSFT]
  share_eligible: [AAPL, MSFT]        # mag als aandeel gehouden worden (6.3 hoofdontwerp)
  beta_to_spy: {SPY: 1.0, QQQ: 1.15, IWM: 1.2, AAPL: 1.2, MSFT: 1.1}   # vaste tabel, jaarlijks herzien
  half_days: ["2026-11-27", "2026-12-24", "2027-11-26"]   # elk jaar bijwerken vanaf de NYSE-site

filters:
  min_open_interest: 500
  min_volume: 100
  max_spread_abs: 0.10
  max_spread_frac_mid: 0.10
  min_underlying_adv_shares: 1000000
  max_quote_age_seconds: 10
  earnings_blackout: true
  max_calendar_data_age_days: 1

pricing:
  scan_engine: baw                     # baw | binomial_lr | fd
  decision_engine: binomial_lr
  binomial_steps: 401
  fd_time_steps: 200
  fd_grid_points: 400
  vega_bump_vol: 0.01
  rho_bump_rate: 0.0001
  iv_accuracy: 1.0e-6
  iv_max_evaluations: 200
  iv_min: 1.0e-4
  iv_max: 5.0
  risk_free_rate: 0.04                 # vlak, handmatig bijgewerkt (zie 5.1)

iv_metrics:
  ivr_lookback_days: 252
  atm_tenor_days: 30
  rv_window_days: 20
  fallback:                            # open vraag B
    enabled: true
    high_iv_ratio: 1.20                # vervangt IVR >= 30
    low_iv_ratio: 1.00                 # vervangt IVR <= 30
    very_low_iv_ratio: 0.90            # vervangt IVR <= 20

strategies:
  iron_condor:
    enabled: true
    dte: [30, 45]
    short_delta: 0.16
    wing_width_strikes: [5, 10]
    min_credit_frac_width: 0.333
    min_ivr: 30
    take_profit_frac: 0.50
    stop_loss_multiple_credit: 2.0
    close_dte: 21
    adjust_short_delta: 0.30
    max_adjustments: 1
  credit_spread: {enabled: true, ...}
  debit_vertical: {enabled: true, ...}
  calendar_spread: {enabled: true, ...}
  long_straddle_strangle: {enabled: true, ...}
  cash_secured_put: {enabled: false, ...}    # open vraag C: slaapt bij 25k
  covered_call: {enabled: false, ...}
  wheel: {enabled: false, ...}
  short_straddle_strangle: {enabled: false, ...}

sizing:
  max_loss_frac_capital: 0.02
  ranking: credit_per_max_loss         # vast; zie 7.3

risk:                                  # sectie 8; fracties van capital_usd
  max_abs_beta_delta_frac: 0.30
  max_vega_frac: 0.005
  max_gamma_loss_5pct_frac: 0.03
  max_loss_per_underlying_frac: 0.05
  max_buying_power_frac: 0.50
  max_open_positions: 10
  max_new_positions_per_day: 3
  daily_loss_stop_entries_frac: 0.02
  daily_loss_degross_frac: 0.04
  drawdown_halt_frac: 0.10
  scenario_spot: [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
  scenario_vol_points: [-10, 0, 10, 20]
  max_scenario_loss_frac: 0.08
  allow_undefined_risk: false
  pdt_guard: true

execution:
  walk_interval_seconds: 15
  max_concession_entry: 0.60           # fractie van mid -> natuurlijke prijs
  max_concession_exit: 1.00
  ticks: {default: {below_3: 0.01, above_3: 0.05}, combo: 0.01}
  max_order_usd: 5000

schedule:                              # New York-tijd
  premarket_refresh: "09:00"
  no_entry_first_minutes: 15
  no_entry_last_minutes: 15
  scan_times: ["15:00"]
  manage_interval_seconds: 60
  risk_interval_seconds: 30
  reconcile_interval_seconds: 300
  close_before_expiry_time: "15:00"    # op de handelsdag vóór expiratie
  end_of_day: "16:15"

paper:                                 # alleen replay en PaperBroker
  fill_at_concession: 0.40
  commission_per_contract: 0.65
  seed: 20261124
```

De strategieblokken die hier met `...` staan, krijgen dezelfde vorm als `iron_condor`, met de
regels uit sectie 7.

### 3.2 `conf/options/options.example.yaml`

```yaml
mode: paper                            # replay | paper | live (live: zie sectie 13)
capital_usd: 25000.0
paths:
  state: /var/lib/tradebot-options/state.json
  journal: /var/lib/tradebot-options/journal.jsonl
  chains: /var/lib/tradebot-options/chains/        # parquet-snapshots per scan
  heartbeat: /var/lib/tradebot-options/heartbeat
  halt_file: /var/lib/tradebot-options/HALT
  unwind_file: /var/lib/tradebot-options/UNWIND
  lock: /var/lib/tradebot-options/options.lock
broker:
  host: ib-gateway
  port: 4002                           # IB Gateway paper
  client_id: 17
  market_data_type: delayed            # realtime | delayed (open vraag A)
  max_market_data_lines: 90            # onder de 100 die IBKR standaard toestaat
live:
  account_allowlist: []                # leeg: live kan niet
  max_capital_usd: 0
  max_order_usd: 0
metrics_port: 9108
strategy_config: /etc/tradebot/options-default.yaml
```

Geheimen komen uit de omgeving: `IBKR_ACCOUNT` (alleen om te controleren, nooit gelogd),
`TRADEBOT_OPTIONS_LIVE`, `ALERT_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`HEARTBEAT_URL`. De IB Gateway-inlog (`TWS_USERID`, `TWS_PASSWORD`) gaat alleen naar de
gateway-container, nooit naar `tradebot-options`. `infra/compose/.env.example` krijgt die namen,
en de bestaande test op `.env.example` krijgt een tweede variant voor `src/tradebot/options`.

## 4. Kalender en tijd (`options/calendar.py`)

- Handelsdagen: `ql.UnitedStates(ql.UnitedStates.NYSE).isBusinessDay`. Halve dagen (sluiting
  13:00) uit `universe.half_days`. Een test vergelijkt de QuantLib-feestdagen van 2026 en 2027
  met een vaste lijst in de test; wijkt QuantLib af na een update, dan faalt de test.
- Sessie: 09:30-16:00 New York, op halve dagen tot 13:00. Alle tijden in de engine zijn
  tz-aware; `utils/time.py` levert de klok, zodat tests een vaste klok kunnen injecteren.
- **Expiratiemoment**: 16:00 New York op de expiratiedag (ook op een halve dag 16:00 voor
  de waardering; de handel stopt dan om 13:00, en daarom sluit de engine korte posities de
  handelsdag ervoor).
- **Looptijd** `T = (expiratiemoment − nu) / (365 × 24 × 3600 s)`, Actual/365 Fixed op
  seconden. Voor een optie die over 2 dagen en 3 uur afloopt is `T` dus 51/8760, niet 2/365.
- Een helper `trading_days_between(a, b)` voor DTE-regels. **DTE in de strategieregels is in
  kalenderdagen**, zoals in de markt gebruikelijk; de helper is alleen voor de PDT-teller en IVR.

## 5. Pricing, IV en Greeks (`options/pricing/`)

### 5.1 Omgeving (`ql_env.py`)

```python
@contextmanager
def evaluation_date(d: date) -> Iterator[ql.Date]:
    settings = ql.Settings.instance()
    previous = settings.evaluationDate
    settings.evaluationDate = to_ql(d)
    try:
        yield settings.evaluationDate
    finally:
        settings.evaluationDate = previous

def make_process(spot: float, r: float, q: CurveOrDividends, vol: float,
                 asof: datetime) -> ql.GeneralizedBlackScholesProcess: ...
```

- Alle QuantLib-objecten worden binnen één `evaluation_date`-blok gemaakt en gebruikt, en niet
  buiten het blok bewaard.
- Omdat de looptijd op seconden telt en QuantLib op dagen rekent, wordt de expiratie omgezet naar
  een fractie via een eigen `ql.Actual365Fixed`-looptijd: de waardering gebeurt met
  `evaluationDate = vandaag` en een `exercise`-datum, waarna de resterende dagfractie van vandaag
  wordt verrekend door de volatiliteit te schalen (`σ_eff = σ × sqrt(T_sec / T_dag)`) en de
  rente en dividendyield met `T_sec / T_dag`. Dat geeft exact dezelfde totale variantie en
  discontering, dus exact voor Black-Scholes en een goede benadering voor Amerikaans. Op de
  expiratiedag zelf (QuantLib ziet dan een verlopen optie) is `evaluationDate` de vorige
  handelsdag met dezelfde schaling. Een test vergelijkt de prijs een dag vóór expiratie om
  09:31 en 15:59, en op de expiratiedag om 10:00, met de gesloten formule.
- Rente: vlak (`ql.FlatForward`) uit `pricing.risk_free_rate`. De bestaande `data/external`
  heeft de T-bill-reeks DTB3 tot 2026-09-30; een automatische feed is een latere uitbreiding.
  Een fout van 50 bp in de rente verschuift de prijs van een 45-daagse ATM-optie met ongeveer
  0,06 % van de spot; dat is klein ten opzichte van de spread.

### 5.2 Keuze van de engine

| Gebruik | Engine | Reden |
|---|---|---|
| Scan over hele ketens | `BaroneAdesiWhaleyApproximationEngine` | snel, nauwkeurig genoeg voor filters |
| Beslissing, sizing, risico | `BinomialVanillaEngine(process, "lr", 401)` | nauwkeurig, levert delta/gamma/theta |
| Discrete dividenden in de looptijd | `FdBlackScholesVanillaEngine` met `DividendSchedule` | binomiaal en BAW ondersteunen alleen een continue yield goed |
| Referentie en pariteit | `AnalyticEuropeanEngine` | gesloten formule |

De keuze zit in één functie `engine_for(use: Use, has_discrete_dividend: bool)`, met een test per
combinatie.

### 5.3 Dividenden

- Bron: `yfinance` (al een harde dependency) voor de verwachte ex-datum en het bedrag. Opgeslagen
  per onderliggende met de datum van ophalen.
- Valt een ex-datum binnen de looptijd: discrete dividenden en FD. Anders: continue yield 0.
- Ontbreekt de dividenddata of is ze ouder dan `max_calendar_data_age_days`: geen nieuwe entry op
  die onderliggende (bestaande posities worden wel beheerd, met de laatste bekende data, en een
  waarschuwing).

### 5.4 Implied volatility (`implied_vol.py`)

```python
class IVStatus(StrEnum):
    OK = "ok"
    NO_QUOTE = "no_quote"            # bid of ask ontbreekt
    NO_BID = "no_bid"                # bid == 0
    BELOW_INTRINSIC = "below_intrinsic"
    ABOVE_UPPER_BOUND = "above_upper_bound"   # call > spot, put > strike
    NO_CONVERGENCE = "no_convergence"
    EXPIRED = "expired"

class IVResult:
    status: IVStatus
    value: float | None              # alleen gezet bij OK
    method: str | None               # "american_engine" of "brent_on_baw"
```

Algoritme per contract en per prijs (mid, bid, ask):

1. `T ≤ 0` → `EXPIRED`. Prijs ontbreekt → `NO_QUOTE`. Bid = 0 → `NO_BID` (voor bid en mid).
2. Grenzen: prijs < intrinsieke waarde − 1 tick → `BELOW_INTRINSIC`; boven de bovengrens →
   `ABOVE_UPPER_BOUND`. Tussen intrinsiek − 1 tick en intrinsiek wordt de prijs op intrinsiek +
   0,5 tick gezet (een mid onder intrinsiek door afronding is normaal bij diep-ITM).
3. `option.impliedVolatility(price, process, 1e-6, 200, 1e-4, 5.0)` met de beslissingsengine.
4. Faalt dat: Brent (`scipy.optimize.brentq`) op `BAW(σ) − price` over `[1e-4, 5.0]`.
5. Faalt dat ook → `NO_CONVERGENCE`.

Een contract zonder `OK` op de mid krijgt geen Greeks en valt buiten elke strategie. Er bestaat
geen pad dat een standaard-IV invult.

### 5.5 Greeks (`greeks.py`)

| Greek | Europees | Amerikaans |
|---|---|---|
| delta, gamma, theta | analytisch | uit de binomiale engine |
| vega | analytisch | `(P(σ+0,01) − P(σ−0,01)) / 2`, per volpunt |
| rho | analytisch | `(P(r+1bp) − P(r−1bp)) / 2 × 100`, per procentpunt |

Theta uit QuantLib is per jaar; de module deelt door 365 voor "per kalenderdag". Bij BAW (scan)
zijn er geen Greeks uit de engine; daar worden delta en gamma met centrale verschillen in de spot
berekend (bump 0,5 % van spot). Een test controleert elke Greek tegen finite differences van de
prijs, binnen 1 % relatief of 1e-4 absoluut.

### 5.6 Prestatie

Een keten van 2.000 contracten in minder dan 2 s op één core in scanmodus: BAW plus IV per
contract. Gemeten in `tests/benchmark/test_options_chain_pricing.py`. Haalt BAW het niet, dan
is het plan: IV alleen voor contracten die de liquiditeitsfilters al halen (ongeveer een derde).

## 6. Ketens, filters, oppervlak en IV-maten (`options/chain/`, `options/pricing/surface.py`)

### 6.1 Pijplijn

1. Ruwe quotes plus spot → `ChainBuilder.build(raw, spot, asof) -> Chain`.
2. Per contract: quote-leeftijd ≤ `max_quote_age_seconds`, anders gemarkeerd als oud.
3. IV (mid, bid, ask) en Greeks volgens sectie 5.
4. Filters in vaste volgorde, met een reden per afgewezen contract:
   `stale_quote`, `no_iv`, `open_interest`, `volume`, `no_bid`, `spread_abs_and_frac`. De
   spreadregel is "spread ≤ max(0,10; 10 % van mid)", dus een contract valt pas af als het
   beide drempels overschrijdt.
5. Onderliggende-filters (eens per dag): gemiddeld dagvolume over 20 dagen, earnings in de
   looptijd, data-ouderdom.
6. De afwijzingen per reden gaan als telling naar het journal en de metrics. Zo is te zien of
   een strategie niets doet omdat de markt niets biedt of omdat een filter te streng is.

### 6.2 Oppervlak

- Per expiratie: OTM-IV's (puts onder de forward, calls erboven). Forward uit put-call-pariteit
  op de drie strikes rond ATM (mediaan), niet uit spot × e^{rT}, zodat dividenden en
  borrow-kosten vanzelf meetellen.
- Smile per expiratie: kubische spline in log-moneyness `k = ln(K/F)` op totale variantie
  `w = σ²T`, met minstens 5 punten; minder punten → geen smile voor die expiratie.
- Over de tijd: lineair in totale variantie bij gelijke `k`.
- Controles: kalender (`w` stijgt in `T` voor elke `k` op het raster) en butterfly (de
  Breeden-Litzenberger-dichtheid uit de callprijzen is ≥ −1e-6 op het raster). Faalt een
  controle, dan krijgt het oppervlak status `ARBITRAGE` en gebruikt geen strategie het. De
  ruwe IV's per contract blijven wel bruikbaar voor de strategieën die alleen een strike kiezen.
- SVI is in v1 niet nodig: de strategieën gebruiken ATM-IV, 25-delta skew en de term
  structure, en die volgen uit de ruwe smile.

### 6.3 IV-maten

| Maat | Definitie |
|---|---|
| `atm_iv(tenor)` | IV bij `k = 0`, lineair geïnterpoleerd in totale variantie tussen de twee expiraties rond `tenor` dagen |
| `skew_25d(expiry)` | IV(put delta −0,25) − IV(call delta 0,25) |
| `term_slope` | `atm_iv(60) − atm_iv(30)`; > 0 is contango |
| `ivr` | `(iv_now − min) / (max − min) × 100` over de laatste 252 handelsdagen van `atm_iv(30)` |
| `ivp` | percentage van die 252 dagen met `atm_iv(30) < iv_now` |
| `iv_rv_ratio` | `atm_iv(30) / RV20`, RV20 = Yang-Zhang over 20 dagen uit `tradebot.volatility.yang_zhang` |

De dienst schrijft elke handelsdag om `end_of_day` één `atm_iv(30)` per onderliggende in de staat.

**IVR zonder historie (open vraag B).** Zolang er minder dan 252 waarden zijn, is `ivr`
`INSUFFICIENT_HISTORY`. Met `iv_metrics.fallback.enabled: true` vervangt een strategie dan
"IVR ≥ 30" door "`iv_rv_ratio` ≥ 1,20", "IVR ≤ 30" door "≤ 1,00" en "IVR ≤ 20" door "≤ 0,90".
Elke beslissing die de vervangende regel gebruikt, staat zo in het journal. Op de dag dat de
historie 252 waarden heeft, schakelt de strategie vanzelf naar IVR. De drempels zijn vast en
staan vóór de eerste paper-dag in de preregistratie.

Een tweede mogelijkheid voor B, niet gekozen: IBKR levert historische 30-daagse IV per
onderliggende (`reqHistoricalData` met `whatToShow="OPTION_IMPLIED_VOLATILITY"`). Daarmee is
IVR meteen beschikbaar, maar het is IBKR's eigen IV-berekening en niet de onze, en het vergt
waarschijnlijk het data-abonnement uit open vraag A. Te heroverwegen als A ja wordt.

## 7. Strategieën (`options/strategies/`)

### 7.1 Interface

```python
class MarketState:                 # bevroren momentopname, gemaakt door de engine
    now: datetime
    chains: Mapping[str, Chain]
    iv: Mapping[str, IVMetrics]     # sectie 6.3, inclusief status
    trend: Mapping[str, Trend]      # spot vs SMA50: UP / DOWN / FLAT (|afstand| < 1 %)
    events: Mapping[str, Events]    # earnings- en ex-dividenddatums met hun ouderdom
    capital_usd: float

class Book:                         # alleen lezen
    positions: tuple[StrategyPosition, ...]
    shares: Mapping[str, int]
    cash_usd: float

class Candidate:
    strategy: str
    underlying: str
    legs: tuple[LegIntent, ...]     # contract, richting, verhouding (1 of 2 bij condor/calendar)
    quantity: int                   # na sizing
    limit_net: float                # mid van de combo, per eenheid
    max_loss: float
    rank_score: float
    rule_trace: tuple[str, ...]     # welke regels zijn gecontroleerd, voor het journal

class Action:                       # uit manage()
    kind: ActionKind                # CLOSE / ROLL / ADJUST / CLOSE_LEG
    position_id: str
    legs: tuple[LegIntent, ...]
    urgency: Urgency                # NORMAL (concessie 0,6) of STOP (concessie 1,0)
    reason: str                     # "take_profit", "stop_loss", "dte", "expiry", ...

class Strategy(Protocol):
    name: str
    def scan(self, market: MarketState, book: Book) -> list[Candidate]: ...
    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]: ...
```

Strategieën zijn pure functies: geen klok (alleen `market.now`), geen I/O, geen broker. Een test
roept elke strategie twee keer met dezelfde invoer aan en vergelijkt de uitvoer byte voor byte.

### 7.2 Strike- en expiratiekeuze (gedeeld, `strategies/select.py`)

- **Expiratie**: alle expiraties met DTE in het bereik; kies de expiratie met DTE het dichtst bij
  het midden van het bereik; gelijke stand: de vroegste. Alleen maandelijkse en wekelijkse
  expiraties die de ketenfilters halen.
- **Strike bij doel-delta**: de strike met `|delta − doel|` minimaal binnen het bereik; gelijke
  stand: de verste OTM. Delta komt uit de beslissingsengine (sectie 5.2), niet uit de scan.
- **Wing**: de strike `n` stappen verder OTM, waarbij `n` het kleinste getal uit
  `wing_width_strikes` is dat het credit-criterium haalt. Haalt geen breedte het, dan geen
  kandidaat.

### 7.3 Sizing en rangschikking

```text
eenheden = floor(min(max_loss_frac_capital × capital, ruimte_per_onderliggende)
                 / max_loss_per_eenheid)
eenheden = 0  → geen kandidaat
```

`ruimte_per_onderliggende` = `max_loss_per_underlying_frac × capital` minus het `max_loss` dat al
open staat op die onderliggende. Rangschikking: `credit / max_loss` voor credit-strategieën,
`(breedte − debit) / debit` voor debit verticals, `theta_usd / debit` voor calendars en
`vega_usd / debit` voor long straddles. De scores worden niet tussen strategieën vergeleken:
de engine neemt per scan de beste kandidaat per strategie, in de vaste strategievolgorde uit
config, en dan per onderliggende maximaal één nieuwe positie per dag.

### 7.4 Regels per strategie

Alle getallen zijn de standaard en staan in config. "Waarde" is de huidige mid van de combo om
de positie te sluiten.

**Credit spread (bull put of bear call).**

- Entry: trend UP → bull put; DOWN → bear call; FLAT → niets (een neutraal boek is de iron
  condor). DTE 30-45, short delta 0,20-0,30 (doel 0,25), credit ≥ ⅓ van de breedte, IVR ≥ 30
  (of de vervangende regel), geen earnings in de looptijd.
- `max_loss = (breedte − credit) × 100 × aantal`.
- Beheer, in deze volgorde (de eerste die geldt, wint):
  1. handelsdag vóór expiratie na `close_before_expiry_time` → CLOSE, STOP;
  2. waarde ≥ 2 × credit → CLOSE, STOP;
  3. short leg ITM met meer dan 1 % → CLOSE, STOP;
  4. waarde ≤ 50 % van credit → CLOSE, NORMAL;
  5. DTE ≤ 21 → CLOSE, NORMAL.

**Iron condor.**

- Entry: DTE 30-45, short delta ±0,16 aan beide kanten, wings 5 tot 10 strikes (gelijke breedte
  aan beide kanten), totaal credit ≥ ⅓ van de breedte, IVR ≥ 30, geen earnings in de looptijd.
- `max_loss = (breedte − credit) × 100 × aantal` (maar één kant kan verliezen).
- Beheer: 1, 2 en 4 en 5 als bij de credit spread, met de totale waarde. Bijsturen: als een short
  leg `|delta| > 0,30` en `adjustments < 1`: sluit de ongeteste kant (CLOSE_LEG, de twee legs van
  die kant) en open daar een nieuwe spread op delta 0,16 in dezelfde expiratie, als combo in één
  order. Lukt de nieuwe spread niet binnen de risicopoort, dan alleen sluiten. `adjustments` + 1,
  `max_loss` opnieuw berekend.

**Debit vertical (bull call of bear put).**

- Entry: trend UP → bull call; DOWN → bear put. DTE 30-60, long delta 0,50-0,60, short leg op de
  strike die het debit ≤ 50 % van de breedte maakt, IVR ≤ 30 (of de vervangende regel).
- `max_loss = debit × 100 × aantal`.
- Beheer: expiratie (1) als hierboven; waarde ≥ 2 × debit → CLOSE (100 % winst); waarde ≤ 0,5 ×
  debit → CLOSE, STOP; DTE ≤ 21 → CLOSE. Een doel van 50 % winst is een tweede, in config te
  kiezen waarde (`take_profit_frac_max_gain: 0.60` standaard na de reviewfixes (was `take_profit_frac_debit: 1.0`)).

**Calendar spread.**

- Entry: kort 20-35 DTE, lang 50-90 DTE, zelfde strike dichtst bij de forward, `term_slope ≥ 0`
  en `ivp(lange leg) ≤ 50`. Geen earnings tussen nu en de expiratie van de korte leg.
- `max_loss = debit × 100 × aantal` (bij Amerikaanse uitoefening is een assignment op de korte
  leg mogelijk; de lange leg dekt die, en de dagelijkse assignment-controle handelt hem af).
- Beheer: waarde ≥ 1,25 × debit → CLOSE; waarde ≤ 0,5 × debit → CLOSE, STOP; handelsdag vóór
  expiratie van de korte leg → CLOSE (beide legs).

**Long straddle of strangle.**

- Entry, twee varianten, elk apart aan of uit: (a) earnings over 5-15 kalenderdagen, (b) IVR ≤ 20
  (of de vervangende regel). Straddle ATM; strangle op delta ±0,25-0,30. Expiratie: de eerste na
  de earningsdatum met DTE ≥ 14, voor (b) DTE 30-45.
- `max_loss = debit × 100 × aantal`.
- Beheer: (a) uiterlijk de handelsdag vóór earnings, om 15:00 → CLOSE (IV-crush vermijden);
  waarde ≥ 1,25 × debit → CLOSE; ≤ 0,5 × debit → CLOSE, STOP; DTE ≤ 7 → CLOSE.

**Cash-secured put, covered call en wheel** (standaard uit, open vraag C).

- CSP: alleen op `share_eligible`, cash ≥ strike × 100 × aantal wordt in het boek gereserveerd en
  telt niet als vrij kapitaal. Delta 0,20-0,30, DTE 30-45, IVR ≥ 30.
  Beheer: 50 % → CLOSE; DTE ≤ 21 → roll naar de volgende maand op dezelfde delta als dat
  een credit oplevert, anders CLOSE.
- Covered call: alleen met ≥ 100 aandelen per contract in bezit. Strike ≥ kostprijs. Delta
  0,20-0,30, DTE 30-45. Beheer: 50 % → CLOSE; DTE ≤ 21 → roll; ITM én ex-dividend morgen én
  extrinsieke waarde < dividend → CLOSE of roll vandaag (vroege assignment voorkomen).
- `max_loss` voor CSP en covered call: in v1 `strike × 100 × aantal` (het theoretische maximum),
  met als gevolg dat ze bij 25k niet door de risicopoort komen. De alternatieve maat uit open
  vraag C (verlies bij −20 %) wordt pas gebouwd als die vraag ja is.
- Wheel, toestandsmachine per onderliggende:

| Van | Gebeurtenis | Naar | Actie |
|---|---|---|---|
| `idle` | scan vindt CSP | `csp` | open CSP |
| `csp` | CSP gesloten of verlopen zonder assignment | `idle` | geen |
| `csp` | assignment (aandelen ontvangen) | `shares` | kostprijs = strike − ontvangen premie |
| `shares` | scan vindt covered call | `cc` | open covered call |
| `cc` | call gesloten of verlopen | `shares` | geen |
| `cc` | call assigned (aandelen weg) | `idle` | P&L vastleggen |
| elk | UNWIND | `idle` | alles sluiten, aandelen verkopen met limiet |

Een test per overgang, en een test dat er geen andere overgangen bestaan.

**Short straddle of strangle.** Bestaat als klasse, maar de registry weigert hem te laden zolang
`risk.allow_undefined_risk` false is, en de live-schakelaar (sectie 13) weigert live met die vlag
aan tenzij hij ook in `live.allow_undefined_risk_ack` staat. Geen tests overslaan: de klasse heeft
dezelfde tests als de rest, en een extra test bewijst de weigering.

### 7.5 Gedeelde harde regels

- Elke short optie is gesloten vóór het einde van de handelsdag vóór expiratie. Lukt het niet
  binnen de normale walk, dan om 15:45 een laatste ronde met concessie 1,0, en daarna een
  kritiek alarm. Een e2e-test bewijst: na de laatste handelsdag is er geen open short optie.
- Geen entry in de eerste en laatste 15 minuten, en geen entry op een halve dag na 12:30.
- Geen entry op een onderliggende met een open reconciliatieverschil.

## 8. Risicopoort (`options/risk/`)

### 8.1 Eén ingang

```python
class RiskGate:
    def review(self, proposal: OrderProposal, book: Book, market: MarketState) -> Decision: ...

class Decision:
    verdict: Verdict                 # APPROVE / RESIZE / REJECT
    approved: ApprovedOrder | None   # alleen bij APPROVE of RESIZE
    checks: tuple[CheckResult, ...]  # elke grens, gemeten en drempel
```

`ApprovedOrder` kan alleen in `risk/gate.py` gemaakt worden (een module-private sentinel in de
constructor). `BrokerAdapter.submit` accepteert alleen een `ApprovedOrder`. Een test bewijst dat
geen andere module in `src/tradebot/options` `ApprovedOrder(` aanroept (AST-scan), en een tweede
test dat een `ApprovedOrder` zonder sentinel een fout geeft.

### 8.2 Algoritme

1. **Toestand**: HALT actief en de order vergroot risico → REJECT. UNWIND actief → alleen
   sluitorders.
2. **Hard (SafetyBreach)**: nominale waarde > `max_order_usd`, aantal ≤ 0, verlopen contract,
   limietprijs buiten [intrinsiek, natuurlijke prijs + 1 tick]. Dit zijn fouten in de code of de
   config: de dienst gaat in HALT.
3. **Aantallen**: open posities, nieuwe posities vandaag, PDT-teller (8.5).
4. **Pro forma**: het boek na de order wordt opnieuw gewaardeerd: Greeks (8.3), scenario's
   (8.4), max verlies per onderliggende, buying power (8.6).
5. **Vergelijk** met elke grens. Definities:
   - Een order is **risicoverkleinend** als hij voor elke grens die nu overschreden is de
     gemeten waarde verkleint, en voor geen enkele andere grens de waarde boven de drempel
     brengt. Een risicoverkleinende order wordt altijd goedgekeurd, ook in HALT.
   - Een order die een grens zou overschrijden, wordt met één eenheid tegelijk verkleind tot
     hij past (RESIZE) of tot 0 (REJECT). Verkleinen gebeurt alleen bij entries; een
     sluitorder wordt nooit verkleind.
6. Elke beslissing gaat met alle `CheckResult`s naar het journal.

### 8.3 Aggregatie

```text
leg_greeks      = greeks_per_share × multiplier × qty      (teken via qty)
delta_beta_spy  = Σ_onderliggende  delta_shares × spot_u × beta_u / spot_SPY
gamma_loss_5pct = Σ_onderliggende  0,5 × gamma_shares × (0,05 × spot_u)² × beta_u²
vega_usd        = Σ vega_usd         (zonder vol-beta in v1)
theta_usd       = Σ theta_usd
```

Aandelen tellen mee met delta 1 per aandeel. Per onderliggende en totaal.

### 8.4 Scenario's

Raster: spot × (1 + s) voor `s` in `scenario_spot` (en 0), IV + `v` volpunten voor `v` in
`scenario_vol_points`, voor elke onderliggende tegelijk (spot geschaald met beta ten opzichte van
SPY). Elke leg wordt **volledig herwaardeerd** met de beslissingsengine bij de nieuwe spot en
vol, op hetzelfde tijdstip (geen tijdsverloop). Verlies = huidige waarde − scenariowaarde. Het
slechtste scenario moet ≤ `max_scenario_loss_frac × capital` blijven. Rekenlast: 10 posities × 4
legs × 28 scenario's = 1.120 binomiale waarderingen per controle; bij 401 stappen ongeveer 1 s.
De risicotaak (elke 30 s) gebruikt daarom BAW, en de pro-forma-controle van een order de
binomiale engine.

### 8.5 PDT-teller

Een daytrade is: een positie die op dezelfde handelsdag geopend en (deels) gesloten wordt. De
staat houdt de datums bij. Is de equity < 25.000 USD en zijn er in de laatste 5 handelsdagen al 3,
dan: geen entry die vandaag ook gesloten zou kunnen worden. Concreet weigert de poort een entry
na 15:00 op een dag waarop de equity onder 25k staat en er al 3 daytrades zijn. Een sluitorder
voor een stop wordt nooit door de PDT-teller tegengehouden: een stop gaat vóór de PDT-regel, en
de waarschuwing gaat naar het journal.

### 8.6 Buying power (Reg-T-schatting, `risk/margin.py`)

| Strategie | Schatting |
|---|---|
| Credit spread, iron condor | `(breedte × 100 − credit) × aantal` (bij een condor alleen de bredere kant) |
| Debit vertical, calendar, long straddle | debit |
| Cash-secured put | `strike × 100 − premie` |
| Covered call | 0 bovenop de aandelen |

In paper en live vergelijkt de adapter de schatting met IBKR's `whatIfOrder` (marge-impact). Wijkt
IBKR meer dan 10 % hoger uit, dan geldt IBKR's getal en gaat er een waarschuwing naar het
journal. In replay geldt alleen de schatting.

### 8.7 Dagverlies, drawdown en HALT

- Dagverlies wordt gemarkeerd op de mid van elke positie, tegen de equity van de vorige sluiting.
  ≥ 2 %: geen entries tot de volgende handelsdag. ≥ 4 %: de engine stelt CLOSE-acties voor, in
  volgorde van grootste bijdrage aan het slechtste scenario, tot alle grenzen weer gehaald worden.
- Drawdown vanaf de high-water mark ≥ 10 %: `HaltStore.engage(HaltRecord(...))` uit
  `risk/kill_switches.py`. Die halt overleeft een herstart en wordt alleen handmatig opgeheven
  (`release()` met operator en motivering), precies zoals de bestaande doctrine.
- Het HALT-bestand (handmatig) en de `HaltStore` (automatisch) hebben hetzelfde effect: geen
  nieuw risico. Exits, stops en het sluiten vóór expiratie gaan door. Dat is een bewust verschil
  met de U1-dienst, die bij een halt helemaal stilstaat: een short optie die niet beheerd wordt
  rond expiratie, is zelf een risico.

## 9. Engine (`options/engine/`)

### 9.1 Toestanden

| Toestand | Wat er gebeurt | Volgende |
|---|---|---|
| `STARTING` | config laden, `SingleInstance`, live-schakelaar (13), staat laden | `RECONCILING` |
| `RECONCILING` | posities, orders en cash bij de broker tegen de staat | `RUNNING` of `BLOCKED` |
| `RUNNING` | alle taken uit 9.2 | `HALTED`, `UNWINDING`, `STOPPING` |
| `BLOCKED` | reconciliatieverschil: geen entries, wel exits op posities die kloppen, alarm | `RUNNING` na een schone reconciliatie |
| `HALTED` | geen nieuw risico, beheer en exits wel | alleen handmatig terug |
| `UNWINDING` | alles sluiten met combo-limietorders, concessie stijgend naar 1,0 | `STOPPED` als alles dicht is |
| `STOPPING` | SIGTERM: open orders annuleren, staat atomisch schrijven | einde |

### 9.2 Taken en tijden (New York)

| Taak | Wanneer | Doet |
|---|---|---|
| premarket | 09:00 op handelsdagen | kalender, earnings en dividenden verversen; lijst expiraties van morgen |
| scan | `scan_times` (15:00) | ketens ophalen, `ChainBuilder`, `scan()` per strategie, risicopoort, orders |
| manage | elke 60 s in de sessie | quotes voor open posities, `manage()`, risicopoort, orders |
| risk | elke 30 s in de sessie | aggregatie, BAW-scenario's, dagverlies, drawdown |
| reconcile | bij start en elke 300 s | broker tegen staat |
| walk | elke 15 s per open order | price-walk (10.2) |
| end_of_day | 16:15 | ATM-IV wegschrijven, assignment- en exercise-detectie, P&L, dagrapport |
| heartbeat | elke ronde | bestand plus `HEARTBEAT_URL` |

Alle taken zijn coroutines in één `asyncio`-loop, geplaatst door een kleine planner die de klok
uit `utils/time.py` gebruikt. In tests draait de planner op een virtuele klok, zodat een
handelsdag in milliseconden doorloopt. QuantLib draait synchroon in de loop (sectie 5.1: geen
threads); de scan van 15:00 blokkeert dus maximaal een paar seconden, en de manage-taak loopt
daarna in.

### 9.3 Fouten

| Fout | Gedrag |
|---|---|
| `SafetyBreach` | `HaltStore.engage`, kritiek alarm, toestand `HALTED` |
| `Deferred` (verouderde data, broker even weg) | deze ronde van deze taak overslaan, teller omhoog |
| `ReconciliationBreak` | toestand `BLOCKED`, alarm |
| Broker langer dan 5 minuten weg in de sessie | alarm; na terugkomst eerst `RECONCILING` |
| Onverwachte uitzondering in een taak | gelogd met traceback, alarm, taak gaat de volgende ronde door; drie keer achter elkaar in dezelfde taak → `HALTED` |
| SIGTERM | `STOPPING`, hergebruik van `live/sigterm.setup_signal_handlers` |

### 9.4 Hergebruik

- `SingleInstance`: verhuist naar `utils/single_instance.py`; `carry/safety.py` importeert hem
  vandaar en zet hem in `__all__`, zodat `from tradebot.carry.safety import SingleInstance`
  blijft werken.
- `Alerter`: verhuist naar `monitoring/channels.py` met dezelfde interface (`from_env`,
  `channels`, `send`, de redactie van geheimen); `carry/alerts.py` wordt een re-export.
- `HaltStore`, `HaltRecord`: direct uit `risk/kill_switches.py`.
- `live/sigterm.py`: direct.
- Niet hergebruikt: `carry/service.py` is een synchrone pollinglus met een dagelijkse stap; het
  patroon (paden, HALT/UNWIND, heartbeat, `load_service_config` die onbekende sleutels weigert)
  wordt gevolgd, de klasse niet.

## 10. Orders, price-walk en de paper-broker (`options/execution/`)

### 10.1 Typen

```python
class ComboOrder:
    client_id: str                   # 10.3
    position_id: str
    legs: tuple[ComboLeg, ...]       # contract (of aandeel), BUY/SELL, verhouding
    quantity: int                    # aantal combo-eenheden
    limit: float                     # netto per eenheid; + is betalen, − is ontvangen
    tif: str = "DAY"
    purpose: Purpose                 # ENTRY / EXIT / ROLL / ADJUST / UNWIND
    urgency: Urgency

class OrderStatus(StrEnum):
    NEW, SUBMITTED, PARTIAL, FILLED, CANCELLED, REJECTED, EXPIRED
```

Er is geen `MARKET`-type. Een enkele optie is een combo met één leg.

### 10.2 Price-walk

```text
mid       = som over legs van (verhouding × teken × mid_leg)
natural   = som over legs van (verhouding × teken × (ask bij kopen, bid bij verkopen))
limit_0   = mid, afgerond op de combo-tick in het nadeel van de markt
stap k    = mid + k × tick × richting
stop als  |limit_k − mid| > max_concession × |natural − mid|
```

Elke 15 s één stap, met de quotes van dat moment (mid en natural worden elke stap opnieuw
berekend). Na de laatste toegestane stap: annuleren. Een entry probeert het pas de volgende
scan weer; een exit de volgende manage-ronde, met de concessie die bij zijn urgentie hoort.
Voorbeeld: een iron condor met mid −1,50 (ontvangen) en natural −1,30, tick 0,01, concessie
0,6: de walk loopt van 1,50 naar 1,38 credit in 12 stappen, 3 minuten.

### 10.3 Client-id en idempotentie

```text
position_id = sha256(strategy | underlying | expiry(s) | strikes | opened_date)[:12]
client_id   = sha256(position_id | purpose | trading_date | poging)[:16]
```

`poging` telt per (positie, doel, dag). De client-id gaat bij IBKR in `orderRef`. Bij een
herstart haalt de engine eerst alle open orders op en koppelt ze via `orderRef`; een order met
een bekende client-id wordt niet opnieuw geplaatst. Property-test: dezelfde invoer geeft dezelfde
id, een andere poging een andere.

### 10.4 Validatie vóór verzending

Aantal > 0; elk contract niet verlopen en in de keten van vandaag; tick-veelvoud; multiplier 100;
nominale waarde ≤ `max_order_usd`; limiet binnen [intrinsieke waarde van de combo, natural + 1
tick]; voor een spread: alle legs op dezelfde onderliggende en in één combo. Een fout hier is
een `SafetyBreach` (8.2 stap 2).

### 10.5 PaperBroker (replay en tests)

- Deterministisch, geen randomness behalve via `paper.seed` (alleen voor de volgorde van
  gelijktijdige events).
- Fill-regel: een order vult volledig zodra `|limit − mid| ≥ fill_at_concession × |natural −
  mid|` (standaard 0,40), tegen de limietprijs. Geen gedeeltelijke fills in v1. Commissie 0,65
  USD per contract per leg.
- Expiratie: een long optie die op de expiratiedag ≥ 0,01 USD ITM sluit, wordt uitgeoefend; een
  short optie die ITM sluit, wordt assigned (dat zou nooit moeten gebeuren door 7.5, en een test
  bewijst dat de engine het dan detecteert en alarmeert).
- Vroege assignment: een short call die ITM is op de dag vóór ex-dividend en waarvan de
  extrinsieke waarde kleiner is dan het dividend, wordt assigned bij de opening van de ex-datum.
- Dezelfde `BrokerAdapter`-interface als de IBKR-adapter, en dezelfde contracttests (16).

### 10.6 BrokerAdapter-Protocol

```python
class BrokerAdapter(Protocol):
    async def connect(self) -> None: ...
    async def account(self) -> AccountSnapshot: ...      # equity, cash, buying power, is_paper, account_hash
    async def positions(self) -> list[BrokerPosition]: ...
    async def open_orders(self) -> list[BrokerOrder]: ...
    async def submit(self, order: ApprovedOrder) -> BrokerOrder: ...
    async def modify(self, client_id: str, new_limit: float) -> BrokerOrder: ...
    async def cancel(self, client_id: str) -> None: ...
    async def what_if(self, order: ComboOrder) -> MarginImpact | None: ...
    async def executions(self, since: datetime) -> list[Execution]: ...
```

`modify` mag alleen de limiet veranderen, niet het aantal of de legs; dat staat in de adapter én
in een test.

## 11. Staat en journal (`options/engine/state.py`, `options/data/store.py`)

### 11.1 Staat (JSON, schema-versie 1)

```json
{
  "schema_version": 1,
  "mode": "paper",
  "account_hash": "sha256 van het account-id, eerste 12 tekens",
  "created_at": "2026-12-01T09:00:00-05:00",
  "positions": [{"position_id": "…", "strategy": "iron_condor", "legs": ["…"], "entry_net": 165.0,
                 "max_loss": 335.0, "adjustments": 0, "phase": null, "status": "OPEN"}],
  "orders": [{"client_id": "…", "status": "SUBMITTED", "limit": -1.42, "step": 8}],
  "iv_history": {"SPY": [["2026-12-01", 0.142]]},
  "equity": [["2026-12-01", 25000.0]],
  "high_water_mark": 25000.0,
  "day_trades": ["2026-12-03"],
  "wheel": {"AAPL": "idle"}
}
```

- Atomisch: schrijven naar `state.json.tmp`, `fsync`, `os.replace`. Een test onderbreekt het
  schrijven en bewijst dat de oude staat leesbaar blijft.
- De `mode` en de `account_hash` in de staat moeten gelijk zijn aan de config en de broker; zo
  niet, dan weigert de dienst te starten (een paper-staat wordt nooit live gebruikt).
- Een migratie per schema-versie, met een test die een staat van versie 1 laadt.

### 11.2 Journal (JSONL, append-only)

Eén regel per gebeurtenis, met `ts`, `event`, `strategy`, `position_id`, `client_id`,
`underlying` waar van toepassing. Gebeurtenissen: `scan_summary` (kandidaten en afwijzingen per
reden), `candidate`, `risk_decision`, `order_submitted`, `order_modified`, `order_cancelled`,
`fill`, `assignment`, `exercise`, `position_opened`, `position_closed`, `iv_fallback_used`,
`halt`, `unwind`, `recon_break`, `recon_ok`, `state_saved`. Geen geheimen en nooit het volledige
account-id.

### 11.3 Ketenarchief

Elke scan schrijft de ruwe ketens (quotes, geen IV) als parquet naar `paths.chains`, per datum en
onderliggende. Dat is de bron voor latere replay en voor een backtest op eigen data (Q6), en kost
ongeveer 1 MB per onderliggende per dag. Over een jaar is dat genoeg om de regels op echte
ketens van eigen opname opnieuw te draaien.

## 12. IBKR-adapter (`options/execution/ibkr.py`, `options/data/feed.py`)

| Behoefte | `ib_async`-aanroep |
|---|---|
| Verbinden | `IB.connectAsync(host, port, clientId)`; paper-poort 4002 (Gateway) of 7497 (TWS) |
| Account | `accountSummaryAsync()`: `NetLiquidation`, `TotalCashValue`, `BuyingPower`; `managedAccounts()` voor de `DU`-controle |
| Ketenstructuur | `reqSecDefOptParamsAsync(symbol, "", "STK", conId)`: expiraties en strikes |
| Contracten | `qualifyContractsAsync(Option(...))` voor de con-ids, alleen voor het gefilterde raster |
| Quotes | `reqMktData(contract, "100,101,106", snapshot=True)` bij de scan; streaming alleen voor open posities |
| Vertraagd of realtime | `reqMarketDataType(1 of 3)` uit `broker.market_data_type` |
| Combo | `Contract(secType="BAG", comboLegs=[ComboLeg(conId, ratio, action, exchange="SMART")])`, `LimitOrder`, `orderRef = client_id` |
| Marge | `whatIfOrderAsync(contract, order)` |
| Posities en orders | `reqPositionsAsync()`, `reqAllOpenOrdersAsync()`, `reqExecutionsAsync()` |

- **Pacing**: IBKR staat standaard ongeveer 100 gelijktijdige marktdatalijnen en 50 berichten per
  seconde toe. De adapter heeft een token-bucket op 40 berichten per seconde en een semafoor op
  `max_market_data_lines`. Een scan vraagt dus niet de hele keten op, maar per onderliggende
  alleen de expiraties in een DTE-bereik van een strategie en de strikes binnen ±15 % van spot,
  in batches.
- **Snapshot-kosten**: realtime snapshots buiten een abonnement kosten bij IBKR per stuk; met
  vertraagde data (open vraag A) speelt dat niet. De adapter telt snapshots per dag als metric.
- **Herverbinden**: exponentieel tot 60 s, en daarna `RECONCILING`.
- **Testen**: een `FakeIB` die dezelfde methoden heeft en vooraf opgenomen antwoorden teruggeeft.
  De adapter is volledig gedekt tegen die fake; alleen de regel die `IB()` echt aanmaakt, heeft
  `# pragma: no cover  # echte netwerkverbinding, getest in paper`. Dezelfde contracttests als
  de `PaperBroker` (16).
- **IB Gateway in compose**: een aparte service met een image die op digest gepind wordt (de keuze
  van de image valt in fase 7; kandidaten zijn de community-images rond IBC). Alleen het interne
  netwerk, poort 4002 niet naar buiten, inlog uit een eigen `.env` die `tradebot-options` niet
  leest. IBKR verbreekt de sessie één keer per dag (en vraagt periodiek opnieuw 2FA); de engine
  behandelt dat als een gewone verbindingsonderbreking, en het runbook beschrijft de
  herinlog.

## 13. Live-schakelaar (`options/engine/live_switch.py`)

```python
def resolve_mode(cfg: ServerConfig, env: Mapping[str, str], account: AccountSnapshot,
                 state: StateFile) -> Mode:
    """Geeft REPLAY of PAPER, of LIVE als alle vijf voorwaarden gelden; anders LiveRefused."""
```

Voorwaarden voor LIVE, allemaal tegelijk:

1. `cfg.mode == "live"`;
2. `env["TRADEBOT_OPTIONS_LIVE"] == "I_UNDERSTAND_REAL_MONEY"`;
3. het account-id staat in `live.account_allowlist` **en** begint niet met `DU`, **en** de broker
   meldt hetzelfde account;
4. het state-bestand bestaat niet of is leeg en heeft `mode == "live"`;
5. `live.max_capital_usd > 0`, `live.max_order_usd > 0` en de broker-equity ligt onder
   `max_capital_usd`.

`cfg.mode == "live"` met één ontbrekende voorwaarde: de dienst **start niet** (geen stille
terugval naar paper, want wie live vraagt en paper krijgt, denkt dat hij live handelt). De
killgate-test zet alle 31 combinaties met minstens één ontbrekende voorwaarde op een rij en
bewijst dat geen ervan LIVE geeft; de 32e (alles waar, met een fake broker) geeft LIVE. Plus: een
`DU`-account met alle andere voorwaarden waar geeft geen LIVE. Geen enkele config, geen voorbeeld
en geen compose-bestand in de repo bevat `mode: live`; een test zoekt daarop.

## 14. Observability en alarmen

Prometheus op `metrics_port`, voorvoegsel `tradebot_options_`:

| Metric | Type |
|---|---|
| `delta_beta_spy_shares`, `vega_usd`, `gamma_loss_5pct_usd`, `theta_usd_day` | gauge |
| `positions_open`, `orders_open`, `new_positions_today` | gauge |
| `quote_age_seconds{underlying}` | gauge |
| `reconcile_ok` | gauge (0/1) |
| `pnl_day_usd`, `equity_usd`, `drawdown_frac` | gauge |
| `risk_decisions_total{verdict,limit}` | counter |
| `chain_rejects_total{reason}` | counter |
| `iv_status_total{status}` | counter |
| `snapshots_total` | counter |
| `engine_state` | gauge (enum als label) |

Alarmen via `monitoring/channels.Alerter` met vaste sleutels: `halt`, `unwind`, `recon_break`,
`assignment`, `broker_down`, `order_unfilled_final`, `expiry_short_open` (kritiek),
`task_error`, `iv_fallback_active` (één keer per dag, info), `delayed_data` (één keer bij start).

## 15. Replay-fixtures (`tests/fixtures/options/`)

Er is geen historische ketendata, dus de fixtures zijn **synthetisch** en alleen bedoeld om de
engine en de regels te testen, niet om rendement te meten.

Aanvulling 2026-11-26 (akkoord Alessio): een **historische simulatie** in `options.data.historical`.
Echte dagbars en IBKR's dagelijkse 30-daagse IV (`OPTION_IMPLIED_VOLATILITY`) per onderliggende,
ketens met QuantLib uit dat IV-niveau met de vaste smile van hierboven, en de IV van vóór de
start als IV-historie (IVR meteen bruikbaar, alleen in de simulatie). Dit meet de regels op echte
koers- en IV-paden, niet op echte quotes; het vervangt forward paper-trading (Q6) niet.

- Generator: `scripts/make_options_fixtures.py`, vast seed 20261124, in de repo, en een test die
  bewijst dat de generator twee keer hetzelfde bestand maakt (hash).
- Onderliggende: twee namen (`TST1` met dividend, `TST2` zonder), dagkoersen uit een GBM met
  vaste parameters plus één geplande sprong van −8 %.
- Ketens: elke handelsdag om 15:00 en elke 30 minuten voor beheer, strikes in stappen van 1 rond
  spot, expiraties wekelijks en maandelijks tot 90 dagen. Prijzen uit QuantLib met een vaste
  smile (`σ(k) = 0,20 + 0,10 k² − 0,05 k`), quotes = theo ± een halve spread uit de filterregel,
  OI en volume vast boven de drempels, behalve een paar contracten die expres afvallen.
- Scenario's in de set van 15 handelsdagen: een iron condor die winst neemt, een credit spread die
  op de stop gaat bij de sprong, een expiratie met een short leg die gesloten moet worden, een
  ex-dividend met vroege assignment op een (bewust ongedekte) testpositie, en een drawdown die
  HALT geeft.
- Grootte: onder 2 MB in totaal; de hygiene-test op de repo-root wordt niet geraakt (alles onder
  `tests/fixtures/`).

## 16. Testplan per fase

| Fase | Testbestanden | Wat ze bewijzen |
|---|---|---|
| 1 | `tests/unit/test_infra_contract.py` (uitgebreid) | U1 alleen met profiel; shadow-cron uit; kraken-shadow alleen handmatig |
| 2 | `tests/unit/options/test_units.py` | eenheden niet te mengen; `scale` lineair en antisymmetrisch (property) |
| 2 | `tests/unit/options/test_calendar.py` | feestdagen 2026-2027, halve dagen, looptijd op seconden |
| 2 | `tests/unit/options/test_ql_env.py` | `evaluationDate` hersteld, ook na een uitzondering; twee waarderingen beïnvloeden elkaar niet |
| 2 | `tests/unit/options/test_european.py` | QuantLib = gesloten BS (scipy) tot 1e-10; pariteit tot 1e-10 |
| 2 | `tests/unit/options/test_american.py` | call zonder dividend = Europees; put ≥ Europees; convergentie in stappen; referentiewaarden; pariteitsongelijkheid |
| 2 | `tests/property/options/test_iv_roundtrip.py` | prijs → IV → prijs binnen 1e-6 over een raster (hypothesis); foutcodes aan de randen |
| 2 | `tests/unit/options/test_greeks.py` | Greeks tegen finite differences; eenheden (theta per dag, vega per volpunt) |
| 2 | `tests/benchmark/test_options_chain_pricing.py` | 2.000 contracten < 2 s |
| 2 | `tests/unit/test_schemas_options.py` | onbekende sleutel = fout; standaardconfig laadt; geen `mode: live` in de repo |
| 3 | `tests/unit/options/test_chain_builder.py`, `test_filters.py` | elke afwijsreden; volgorde; telling |
| 3 | `tests/unit/options/test_surface.py` | forward uit pariteit; kalender- en butterfly-controle vangen een kapot oppervlak |
| 3 | `tests/unit/options/test_iv_metrics.py` | IVR/IVP; `INSUFFICIENT_HISTORY`; vervangende regel en de overgang op dag 252 |
| 3 | `tests/unit/options/test_fixtures_generator.py` | generator deterministisch |
| 4 | `tests/unit/options/strategies/test_<naam>.py` per strategie | elke entryregel aan en uit; elke beheerregel en hun volgorde; `max_loss`-formule; determinisme |
| 4 | `tests/unit/options/strategies/test_wheel_states.py` | elke overgang, geen andere |
| 4 | `tests/unit/options/strategies/test_registry.py` | short straddle weigert zonder vlag |
| 5 | `tests/unit/options/risk/test_aggregation.py`, `tests/property/options/test_risk_properties.py` | som per leg = totaal; spiegel = tegengesteld; verkleinend nooit geweigerd; vergrotend boven grens altijd |
| 5 | `tests/unit/options/risk/test_scenarios.py` | één long call en een iron condor tegen een handberekening |
| 5 | `tests/unit/options/risk/test_gate.py` | elke grens; RESIZE; HALT laat exits door; `ApprovedOrder` alleen uit de poort (AST) |
| 5 | `tests/unit/options/risk/test_pdt.py`, `test_margin.py` | PDT-teller; schattingen per strategie |
| 6 | `tests/unit/options/execution/test_price_walk.py`, `test_client_id.py`, `test_validation.py` | walk-stappen en stop; idempotente id's (property); elke validatieregel |
| 6 | `tests/unit/options/execution/test_broker_contract.py` | dezelfde gedragstests voor `PaperBroker` en (fase 7) de IBKR-adapter met `FakeIB` |
| 6 | `tests/unit/options/engine/test_state.py`, `test_scheduler.py`, `test_engine_states.py` | atomisch schrijven; migratie; mode-/accountcontrole; planner op virtuele klok; elke toestandsovergang |
| 6 | `tests/e2e/test_options_paper_smoke.py` (`@pytest.mark.e2e`) | 15 dagen replay: posities open, beheerd en dicht; geen short open na de laatste handelsdag; assignment gedetecteerd; HALT bij de drawdown; journal en rapport geschreven |
| 7 | `tests/unit/options/execution/test_ibkr.py` | adapter tegen `FakeIB`: contracten, combo's, `orderRef`, pacing, herverbinden |
| 7 | `tests/killgates/test_options_live_switch.py` | 31 combinaties geen LIVE, de 32e wel; `DU` nooit live; nooit xfail of skip |
| 7 | `tests/unit/test_infra_contract.py` (uitgebreid) | compose zonder profiel start `tradebot-options` en de gateway, met `--profile crypto` ook `u1`; systemd-unit; `.env.example` noemt elke variabele die `options/` leest |

Elke fase houdt de dekking op 100 %. De bestaande crypto-tests blijven ongewijzigd in de suites.

## 17. Omvang per PR (schatting)

| Fase | Nieuwe bestanden in `src/` | Ongeveer regels code + tests | Dependency |
|---|---|---|---|
| 1 | 0 | 100 + 40 | geen |
| 2 | 9 | 900 + 900 | QuantLib 1.44 |
| 3 | 5 | 600 + 600 | geen |
| 4 | 10 | 1.200 + 1.400 | geen |
| 5 | 5 | 700 + 800 | geen |
| 6 | 8 | 1.200 + 1.000 | geen |
| 7 | 4 | 700 + 700 | `ib_async` 2.0.1 |

Fase 4 is de grootste. Als die te groot wordt om goed te lezen, splits ik hem in twee PR's:
eerst de spreads en de iron condor, daarna calendar, straddle en de standby-strategieën.
