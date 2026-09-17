# MEETDOMEIN — het verdict per module

> **Fase 10, stap 15.** Exit-criterium 26.
> **Geverifieerd tegen de codebase op 2026-09-17**, op commit `8b31005`.
> Dit document voert AD-23 uit: elke module die een observatie fijner dan de
> dagbar veronderstelt, krijgt één van drie verdicts.

## De regel waar dit document aan gehoorzaamt

`conf/governance/measurement_domain.yaml` IS het meetdomein. Een grootheid is
toelaatbaar dan en slechts dan wanneer zij een functie is van een bron uit die
whitelist, geobserveerd op de frequentie die daar staat.

**En de discipline bij het invullen, letterlijk uit de stapopdracht:**

> *"Vul de tabel op grond van **wat de module leest**, nooit op grond van zijn
> naam. Een module archiveren die op de dagbar werkt, is even schadelijk als een
> module behouden die dat niet doet: het eerste vernietigt werkende code, het
> tweede houdt het mandaat leeg."*

## Wat de gecertificeerde store feitelijk bevat

Niet aangenomen maar uitgelezen, over alle partities van `data/pit_store`:

| dataset | granulariteit | kolommen |
|---|---|---|
| `ohlcv` | `1d` | `event_ts_ns`, `asof_ts_ns`, `open`, `high`, `low`, `close`, `volume`, `turnover`, `symbol` |
| `funding` | `8h` | `event_ts_ns`, `asof_ts_ns`, `funding_rate`, `funding_interval_hours`, `symbol` |
| `open_interest` | `1d` | `event_ts_ns`, `asof_ts_ns`, `open_interest`, `symbol` |

**Er is geen `bid_close`, geen `ask_close`, geen `taker_buy_volume`, geen
`tick_volume`, geen `real_volume` en geen `runs_imbalance`.** Dat is de
meetlat: elke functie die zo'n kolom nodig heeft, kan in dit domein niet worden
uitgerekend.

## De verdicts

| Module | LOC | Wat hij LEEST | Verdict | Heropeningsvoorwaarde |
|---|---:|---|---|---|
| `features/microstructure.py` regels 16–229 | 214 | `taker_buy_volume`, `taker_sell_volume`, `bid_close`, `ask_close`, `tick_volume`, `real_volume`, `runs_imbalance` | **`ARCHIVED`** | een gecertificeerde quote- of trade-print-bron in `measurement_domain.yaml` |
| `features/microstructure.py` regels 230–474 | 245 | `funding_rate`, `open_interest` via `asof_join` op het dagraster | **`RETAINED`**, hernoemd → `features/positioning.py` | n.v.t. — dit is de helft die de stapopdracht bedoelde |
| `alpha/microstructure.py` (`OFISignal`) | 103 | wikkelt `order_flow_imbalance` uit de gearchiveerde helft | **`ARCHIVED`** | idem als de helft die hij wikkelt |
| `bars/runs.py` | 254 | `bid_*`/`ask_*` inclusief `bid_depth`, `ask_depth` — AFML §2.2 runs bars | **`ARCHIVED`** | een gecertificeerde L2-boekbron |
| `bars/imbalance.py` | 175 | `bid_*`/`ask_*` — AFML §2.2 imbalance bars | **`ARCHIVED`** | idem |
| `bars/dollar.py` | 173 | `bid_*`/`ask_*` met fallback — AFML §2.3 dollar bars | **`ARCHIVED`** | idem |
| `execution/spread.py::detect_spread`, `::compute_dynamic_spread_arr` | 85 | `bid_close`, `ask_close`, `taker_buy_volume`, `tick_volume` — met stille terugval | **`DEMOTED`** (het bestand moet blijven, zie onder) | een gecertificeerde quote-bron; AD-3 blijft van kracht |
| `execution/spread.py::corwin_schultz_spread` | 60 | uitsluitend `high` en `low` van de dagbar | **`RETAINED`** | n.v.t. — hij BESTAAT omdat er geen quotes zijn |
| `data/orderbook.py` | 150 | L2-snapshots (`[prijs, hoeveelheid]`-lijsten), boekimbalans | **`ARCHIVED`** | een gecertificeerde L2-boekbron |
| `live/` (13 modules) | — | een besluitcyclus die niet de dagbar is | **zie stap 16** | zie stap 16 |
| `execution/impact.py` (`eta = 2.991922`) | — | ongekalibreerd; kalibratie vereist uitvoeringsdata | **`RETAINED`** met harde limiet | zie stap 17; AD-2 blijft van kracht |
| `volatility/realized.py` | — | range-estimators op de OHLC van de DAGBAR | **`RETAINED`, ongewijzigd** | n.v.t. — expliciet beschermd |
| `volatility/yang_zhang.py`, `rogers_satchell.py`, `garman_klass.py` | — | idem: open/high/low/close van de dagbar | **`RETAINED`, ongewijzigd** | n.v.t. |
| `data/crypto.py`, `data/ingestion/` | 501 + | leest het Bybit-publieke archief (trades) om de dagbar te MAKEN | **`RETAINED`** | n.v.t. — dit is de bron, niet een grootheid erop |

## De twee vallen, allebei aangetroffen

### Val 1 — de naam suggereert fijner, de inhoud is de dagbar

Deze modules moeten blijven, en dat is de reden dat §2.3 bestaat:

- `volatility/realized.py`, `yang_zhang.py`, `rogers_satchell.py`,
  `garman_klass.py` — "realized" en "intraday range" klinken naar intrabar-data,
  maar alle vier rekenen op `open`/`high`/`low`/`close` van de dagbar.
- `data/crypto.py` en `data/ingestion/` — lezen wél trades, maar als
  INGESTIE: zij produceren de gecertificeerde dagbar. De store die zij vullen
  bevat, gemeten, geen enkele trade-kolom.

- `execution/spread.py::corwin_schultz_spread` — heet naar een bid-ask spread,
  leest uitsluitend `high` en `low`. Corwin & Schultz (2012) ontwierpen de
  estimator juist om een spread te schatten ZONDER quotes, en de docstring zegt
  dat met zoveel woorden: *"De gecertificeerde store bevat GEEN bid/ask … Deze
  estimator is het alternatief dat wél op de data steunt."* Dit is de scherpste
  val in het hele bestand: de naam noemt precies de observatie die het domein
  niet heeft, terwijl de functie er niet naar kijkt.

### Val 2 — één bestand, twee domeinen

`execution/spread.py` is net als `features/microstructure.py` geen module met
één verdict. Gemeten op dezelfde 2.342 gecertificeerde dagbars:

| functie | uitkomst op de dagbar | oordeel |
|---|---|---|
| `detect_spread` | **0,006** (60 bp), afkomstig van de terugvalroute | `DEMOTED` — leest quotes als ze er zijn |
| `compute_dynamic_spread_arr` | 2.342 waarden, **25 unieke**, vastgelopen tegen een plafond van 0,006 | `DEMOTED` — idem |
| `corwin_schultz_spread` | 2.341 waarden, **1.578 uniek**, mediaan 65,1 bp | `RETAINED` — alleen `high`/`low` |
| `compute_annualised_sharpe` | n.v.t. — geen spread-logica | moet VERHUIZEN, zie onder |

Het bestand blijft dus staan. `DEMOTED` betekent hier: de twee quote-afhankelijke
functies verdwijnen uit elke actieve configuratie en dragen een expliciet label,
maar de code blijft waar zij staat omdat haar buren nodig zijn.

> **Terzijde, en buiten dit verdict.** Corwin–Schultz gééft op deze data een
> mediaan van 65 bp roundtrip waar de werkelijke BTCUSDT-perp-spread rond 1 bp
> ligt — twee ordes te hoog, een bekende eigenschap van de estimator op volatiele
> dagbars. Domeinconform en accuraat zijn twee verschillende vragen; stap 15
> beantwoordt alleen de eerste.

## De correctie op de stapopdracht

De stapopdracht schrijft voor `features/microstructure.py` één rij voor:

> *"naam suggereert een fijnere observatie; bevat feitelijk alleen dagelijkse
> funding en open interest → **`RETAINED`**, maar verplicht hernoemen →
> `features/positioning.py`. n.v.t. — de naam was de fout, niet de inhoud"*

**Die rij is door meting weerlegd.** Het bestand is twee bestanden, en het zegt
dat zelf op regel 189:

> *"Alles BOVEN deze regel is legacy (order-flow op intraday data, DI-12,
> Phase 3) … Alles HIERONDER is Phase 2-materiaal: `BaseFeature`-subklassen op de
> gecertificeerde `funding`- en `open_interest`-reeksen."*

De bovenste helft draait op de dagbar niet stuk maar **stil terug op een
vervangende kolom**. Gemeten op de gecertificeerde BTCUSDT-dagbars (2.342 bars,
geen enkele out-of-domain kolom aanwezig):

| functie | uitkomst op de dagbar | wat er feitelijk wordt gerekend |
|---|---|---|
| `order_flow_imbalance` | 2.342 eindige waarden, **1 unieke waarde: 0** | niets; de fallback vult nullen |
| `vpin` | **0 eindige waarden** (alles NaN) | niets — en dit is de enige die eerlijk faalt |
| `kyle_lambda` | 910 unieke waarden | `\|Δclose\|/volume`, bit-identiek aan de formule van Amihud — Kyle's λ wordt niet berekend |
| `bid_ask_spread` | 2.342 unieke waarden, gemiddeld **0,0445** | `(high−low)/close`, bit-identiek aan de range-proxy. Een "spread" van 4,45 % op BTCUSDT; de werkelijke perp-spread is ~1 bp |
| `amihud_illiquidity` | 876 unieke waarden | `\|log ret\|/volume` — dit is wél domein-conform |

Een uniform hernoemen naar `positioning.py` zou het enige waarschuwingsetiket
dat deze code draagt — het woord *microstructure* — weghalen van precies de
functies die microstructuurgetallen fabriceren uit dagbars. Dat is de omkering
van het mandaat, niet de uitvoering ervan.

**Het verdict is daarom een SPLITSING**, en zij haalt allebei de doelen van de
stapopdracht: de funding/OI-helft krijgt de naam `positioning.py` die de
stapopdracht voorschrijft (exit-criterium 27), en de order-flow-helft krijgt het
verdict dat het mandaat voorschrijft.

`amihud_illiquidity` verhuist mee naar de behouden kant: hij leest `close` en
`volume` en is daarmee domein-conform, ongeacht in welk hoofdstuk van AFML hij
staat. Dat is val 1, binnen één bestand.

## Wat de archivering kost — GECORRIGEERD, en de eerste meting was fout

**Correctie op de eerste versie van dit document.** Die telde per publiek
symbool hoe vaak het in `apps/`, `scripts/`, `conf/` en `dvc.yaml` voorkomt, vond
overal nul, en concludeerde dat 1.069 LOC "niets actiefs bereikt". Die telling
was juist en de conclusie fout: zij meet of een naam ergens LETTERLIJK staat, en
niet of de module TRANSITIEF bereikbaar is. `docs/CODE_REGISTER.md` had het
antwoord al in zijn kolom "Bereikt via" staan.

`scripts/reachability_map.py` beslist dit, en na de splitsing van stap 15.3
luidt het oordeel:

| module | klasse | LOC | betekenis |
|---|:---:|---:|---|
| `features/positioning.py` | **A** | 339 | bereikbaar vanuit een entrypoint |
| `features/microstructure.py` | **D** | 180 | test-only — dit is wat 15.3 opleverde |
| `alpha/microstructure.py` | **A** | 103 | bereikbaar |
| `bars/runs.py` | **A** | 254 | bereikbaar via `apps/build_features.py` |
| `bars/imbalance.py` | **A** | 175 | bereikbaar via `apps/build_features.py` |
| `bars/dollar.py` | **D** | 173 | geen afnemer |
| `execution/spread.py` | **A** | 314 | bereikbaar via `apps/tune_hparams.py` |
| `data/orderbook.py` | **D** | 150 | geen afnemer |

Vier van de zes kandidaten zijn klasse A. **Zij kunnen niet worden
gearchiveerd**: archiveren breekt een actief pad. Dat stap 15.3 de order-flow-helft
van klasse A naar klasse D bracht, is de enige archiveerbare winst die deze
stage tot nu toe heeft opgeleverd — en zij is meetbaar: 180 LOC.

Werkelijk onbereikbaar is 180 + 173 + 150 = **503 LOC**, en van die drie dragen
er twee een eerdere uitspraak in `CODE_REGISTER.md` dat verplaatsen tests breekt.

## De grootste domeinschending staat niet in de tabel hierboven

De reden dat `bars/runs.py` klasse A is, is belangrijker dan de module zelf:

```text
dvc.yaml::build_features
  -> apps/build_features.py:150   ingest_raw(sym_cfg, sym)  -> df_micro
  -> apps/build_features.py:153   build_features(...)
  -> features/pipeline.py:100     from .regime import FeaturePipeline
  -> features/regime.py:733       bar_fn = generate_runs_bars if use_runs
                                           else generate_imbalance_bars
  -> features/regime.py:741-743   df_micro / df_meso / df_macro
```

`features/regime.py:730` leest `feature_pipeline.target_micro_bars` met default
**30**, en bouwt daaruit een piramide van micro- (30 per dag), meso- (6) en
macrobars. `AD-23` schrijft `bars_per_day: 1`.

**Dit is een levende DVC-stage die dertig bars per dag genereert.** Niet een
module die dat zou kunnen, maar de eerste stage van de pijplijn — dezelfde stage
waarvan stap 14 de dependency repareerde. Het is de grootste schending van het
meetdomein in deze repository, en zij is geen kwestie van twee modules
archiveren: `regime.py` (1.060 LOC, klasse A), `features/pipeline.py`,
`apps/build_features.py` en beide barmodules vormen één keten.

> **Dit verdict wordt hier NIET geveld.** Stap 15 archiveert modules die buiten
> het domein vallen; deze keten is de actieve featurepijplijn van fasen 1 tot en
> met 3 en draagt de baseline waartegen alles is gemeten. Hem archiveren is geen
> opruiming maar het buiten gebruik stellen van de bestaande pijplijn, en dat is
> een besluit van de eigenaar. Vastgelegd als bevinding, met de keten erbij,
> zodat het besluit op de meting rust en niet op een naam.

## De vreemde eend in `execution/spread.py`

`compute_annualised_sharpe` (regel 195) staat in dit bestand maar is geen
spread-logica. Hij wordt aangeroepen door `tune/objective.py:631` en `:719`, de
Optuna-foldscoring. Hij is de tweede reden — naast `corwin_schultz_spread` — dat
dit bestand niet naar `archive/` kan.

> **Openstaande bevinding, buiten de scope van stap 15.** Deze functie
> annualiseert met `sqrt(365,25)`, terwijl `docs/MEASUREMENT_CONTRACT.md` en
> `conf/backtest/default.yaml` `bars_per_year = 365` vastleggen. Hetzelfde geldt
> voor `backtest/evaluation.py:358`, `:665`, `:846`, `backtest/metrics.py:28` en
> `compliance/champion_challenger.py:29`. Het verschil is
> `sqrt(365,25/365) = 1,000342`, dus 0,034 % — numeriek verwaarloosbaar, maar
> exit-criterium 2 eist **één** `bars_per_year` en er zijn er twee. Dit is een
> meetcontract-kwestie, geen domeinkwestie, en zij verandert opgeslagen
> backtestgetallen; daarom hier vastgelegd en niet hier gerepareerd.

## Wat dit document NIET vaststelt

- Het spreekt geen oordeel uit over de veertien dode alfamodules uit
  `docs/CHAIN_A_STATUS.md`. Negen daarvan meten aandelen, valuta of commodities
  en vallen buiten het domein via hun VERMOGENSTITEL, niet via hun
  observatiefrequentie. Dat is een aparte vraag met een ander antwoord.
- Het herroept geen regel uit `docs/FALSIFICATION_REGISTER.md`. B-2 laat F1–F20
  onaangetast.
- Het raakt geen poort. Er komt geen `force=`, `override=` of `warn_only=` bij.
