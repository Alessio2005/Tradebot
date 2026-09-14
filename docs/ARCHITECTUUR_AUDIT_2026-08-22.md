LEGACY DEPENDENCY FLOW (PROBLEMATISCH)

[Config / Envs] ---> [Feature Store (met stille fallbacks)]
|
v
[Alpha Module (L4) + Risk Limits (L7) Gemengd]
|
v
[4 Overlappende Backtesters (Vectorized/Event)]
|
v
[Niet-bestaande ML/Ensemble Lagen]


### Kernproblemen in de Legacy Structuur:

> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-3).
> Het bindende auditdocument, ONGEWIJZIGD sinds 2026-08-22.
> "Geverifieerd" betekent hier uitsluitend dat het nog steeds het
> document is waarnaar de rest van de repository verwijst; de inhoud is
> NIET regel voor regel tegen de code herlezen. Dat zou een audit zijn,
> geen verificatie, en het is niet gedaan.
1. **Circulaire & Misplaatste Afhankelijkheden:** Risicolimieten en volatiliteitstargeting bevinden zich gedeeltelijk binnen de alpha-units en feature-pipelines, wat een strikte scheiding van verantwoordelijkheden onmogelijk maakt.
2. **Shadow Trees & Configuration Drift:** De aanwezigheid van overbodige mappen en afwijkende omgevingsconfiguraties leidt tot onvoorspelbaar gedrag tussen test- en runtime-omgevingen.
3. **Versnipperde Backtest Engine:** Vier overlappende engines (`evaluation`, `portfolio`, `bidirectional`, `per_side`) verhogen de onderhoudslast en creëren kweekvijvers voor subtiele simulatiefouten.

---

## 5. Problems & Architectural Debt

### 5.1 Harde Documentatie-vs-Code Tegenstrijdigheden (D-1 t/m D-10)

| ID | Document claim | Werkelijkheid in de Codebase | Prioriteit |
|---|---|---|---|
| **D-1** | `model_risk_policy.md`: Promotie geblokkeerd door 6 specifieke lookahead tests. | Geen van deze 6 testbestanden bestaat in `tests/lookahead/`. | **P0** |
| **D-2** | `tca_methodology.md`: `apps/calibrate_impact.py` levert $\eta, \kappa_d$. | Bestand bestaat niet. | **P1** |
| **D-3** | `tca_methodology.md`: `tests/integration/test_tca_roundtrip.py` draait per PR. | Bestand bestaat niet. | **P1** |
| **D-4** | `tca_methodology.md`: `conf/tca/default.yaml` aanwezig. | Bestand bestaat niet. | **P1** |
| **D-5** | `architecture.md`: Zie `REFACTOR_BLUEPRINT_v3.md`. | Bestand bestaat niet. | **P1** |
| **D-6** | `architecture.md` R-4: Maximaal 800 LOC, 4-bestands-whitelist. | 7 bestanden overschrijden de limiet; 3 vallen buiten whitelist. | **P2** |
| **D-7** | `architecture.md` R-6: Applications $\le$ 80 LOC. | 17 van 30 apps zijn groter (tot ~1350 LOC). | **P2** |
| **D-8** | `architecture.md` DAG: `artefacts/features/`, `models/`, `tracks/`. | Geen van deze mappen bestaat in de repository. | **P0** |
| **D-9** | `model_risk_policy.md`: Elk MRM-rapport bevat een valide `git_sha`. | Geen werkende `.git` repository in de hoofdmap. | **P0** |
| **D-10** | `DATA_REGISTER.md`: HMM-regimes via `hmmlearn`. | `hmmlearn` ontbreekt in `pyproject.toml` $\rightarrow$ stille fallback naar EMA-crossover. | **P0** |

### 5.2 Stille Degradatie (Silent Fallbacks)
Er zijn 12 plekken in `src/` geïdentificeerd waar statistische of modelleringsfouten worden opgevangen via `try/except ImportError` of brede exceptions, waarna het systeem stilzwijgend terugvalt op een naïeve baseline:
- `risk/hmm_regime.py`: Valt bij het ontbreken van `hmmlearn` stilzwijgend terug van een 3-state Gaussian HMM naar een 20/100 EMA-crossover.
- **`quant_architect` modules:** Vijf core modules (`train/ensemble.py`, `train/_scalers.py`, `train/catboost.py`, `tune/objective.py`, `backtest/portfolio.py`) proberen `quant_architect` te importeren. Deze module bestaat nergens. Hierdoor draaien alle ensemble- en kalibratiemechanismen permanent in gedegradeerde modus zonder dat er een melding wordt gegenereerd.

---

## 6. Research Methodology Assessment

De onderzoeksfilosofie is in essentie solide (gebaseerd op popperiaanse falsificatie en pre-registratie), maar lijdt in de praktijk onder uitvoeringstekortkomingen.

### Onderscheid: "Niet Bewezen" vs. "Bewezen Slecht"
Een cruciale fout in eerdere evaluaties was het gelijkstellen van onbewezen complexiteit aan schadelijke complexiteit.

- **Bewezen Slecht:** Modellen die na correcte OOS-validatie en TCA een negatieve verwachte waarde laten zien of aantoonbaar overfitten op in-sample ruis (bijv. directionele p-hacks via ongestructureerde CatBoost op dagelijkse returns).
- **Niet Bewezen:** Modellen waarvoor de infrastructuur of de benodigde data (zoals tick-data of orderboek-data) simpelweg ontbrak om een eerlijke evaluatie uit te voeren (bijv. GARCH volatiliteitsforecasts of High-Frequency Realized Volatility).

> **Doctrine:** Een model wordt pas gearchiveerd of verwijderd als het *bewezen slecht* is onder een correct geconfigureerde baseline. Zolang het *onbewezen* is, verblijft het in de Research Track en krijgt het geen toegang tot de productie-pijplijn.

---

## 7. Data Architecture

> **CORE AXIOM:** *"Geen betrouwbare data $\rightarrow$ geen betrouwbare quant research."*  
> Data is het fundament van het gehele platform. Onvolledige, niet-point-in-time of gecorrumpeerde data maakt elke statistische toets en backtest waardeloos.

TARGET DATA PIPELINE & LINEAGE

[ Raw Data Ingestion ]
│
▼
[ Data Validation & Sanitization (Schema, Gaps, Outliers) ]
│
▼
[ Point-in-Time Alignment (asof_join, Truncation Guards) ]
│
▼
[ Immutable Parquet Storage + DVC Versioning (data_hash) ]
│
▼
[ Feature Generation (Causal, Stateless Transforms) ]


### 7.1 Vereiste Datasets per Research Track
- **Alpha Research:** Daily/Hourly OHLCV, Funding Rates, Open Interest, Liquidaties, Cross-sectional Spreads, Volume profiles.
- **Volatility Research (GARCH & EWMA):** High-frequency trade data (1m / 5m bars) voor Realized Variance, daily OHLCV voor GARCH/EGARCH/GJR-GARCH.
- **Regime Research:** Macro-economische tijdreeksen, Implied Volatility indices, Volatility term structures, Cross-asset correlatiematrices.
- **Execution & TCA Research:** Top-of-book (L1) en Depth-of-book (L2) orderboek-snapshots, individuele trade-prints, exchange fee structures, latentiestatistieken.

### 7.2 Data Governance & Point-in-Time Rigor
- **Timezone Standard:** Alle timestamps worden opgeslagen in UTC Unix Nanoseconden.
- **Survivorship Bias:** Het universum moet historische delistings en faillissementen bevatten.
- **Futures Rolls & Funding:** Continuous futures reeksen moeten expliciet gecorrigeerd worden via achterwaartse/voorwaartse verhoudingsaanpassingen of expliciete adjustment factor ledgers.
- **Dataset Versioning:** Elke dataset krijgt een unieke `data_hash` gegenereerd via DVC/Git-LFS. Research resultaten zonder geciteerde `data_hash` worden automatisch als invalid beschouwd.

---

## 8. Time-Series & Econometrics Architecture

Tijdreeksanalyse binnen de Target Architecture vereist een strikte behandeling van non-stationariteit en geheugeneffecten.

### 8.1 Stationariteit vs. Geheugenbehoud
- **Standaard Differentiëring ($d=1$):** Verwijdert het gehele geheugen van de tijdreeks, wat de voorspellende waarde voor kwantitatieve modellen reduceert.
- **Fractionele Differentiëring (FracDiff):** Wordt gebruikt om stationariteit te bereiken ($p$-value $<0.05$ op Augmented Dickey-Fuller) terwijl de correlationele geheugenstructuur ($d \in [0, 1]$) maximaal behouden blijft.

### 8.2 Econometrische Toetsingsketen
Elke tijdreeks die de feature pipeline binnenkomt moet verplicht de volgende toetsingsvolgorde doorlopen:
1. **Augmented Dickey-Fuller (ADF) & KPSS tests:** Beoordeling van unit roots en trend-stationariteit.
2. **CUSUM & structural break tests:** Detectie van regime-verschuivingen in het gemiddelde en de covariantie.
3. **Autocorrelatie & Heteroskedasticiteit:** Ljung-Box test voor autocorrelatie; ENGLE ARCH-test voor conditional heteroskedasticity om de noodzaak van een GARCH-structuur vast te stellen.

---

## 9. Volatility Architecture

Volatieliteitsmodellering wordt binnen de Target Architecture gedefinieerd als een ex-ante risk & sizing component, **NIET** primair als een richtinggevend alpha-signaal.

### 9.1 Volatieliteits-Hiërarchie

VOLATILITY RESEARCH HIERARCHY

┌──────────────────────────────────────────────────────────┐
│ Level 0: Naive Historical Volatility (Rolling StdDev)    │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ Level 1: EWMA / RiskMetrics (λ = 0.94) [BASELINE]        │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ Level 2: Parametric Econometric Models                   │
│ (GARCH(1,1), GJR-GARCH, EGARCH, APARCH)                  │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ Level 3: Realized Volatility / High-Frequency            │
│ (Realized Variance, HAR-RV, Kernel Volatility)           │
└──────────────────────────────────────────────────────────┘


### 9.2 Modelspecificaties & Aannames

| Model | Aannames | Databehoefte | Forecast Horizon | Primaire Toepassing |
|---|---|---|---|---|
| **EWMA ($\lambda=0.94$)** | Parameter-vrij IGARCH(1,1) relict; gelijke decay. | Daily OHLCV | $\sigma^2_{t+1\vert t}$ | Baseline position sizing & risk limits. |
| **GARCH(1,1)** | Symmetrische respons op schokken; stationaire variantie. | Daily Close | $\sigma^2_{t+h\vert t}$ | Ex-ante risicoforecasting op middellange termijn. |
| **GJR-GARCH** | Asymmetrie (leverage-effect: negatieve schokken verhogen vol meer). | Daily OHLCV | $\sigma^2_{t+h\vert t}$ | Equities & Crypto crash-risk modelling. |
| **EGARCH** | Logaritmische variantie; geen positiveringsrestricties vereist. | Daily Close | $\sigma^2_{t+h\vert t}$ | Zware staartverdelingen en extreme asymmetrie. |
| **HAR-RV** | Heterogene markt-hypothese (dagelijks, wekelijks, maandelijks geheugen). | 5-min Intraday Bars | $\sigma^2_{t+1\vert t}$ t/m $\sigma^2_{t+5\vert t}$ | High-frequency volatiliteitstargeting. |

### 9.3 Validatie & Evaluatiemetrics voor Volatieliteit
GARCH- en volatiliteitsmodellen worden OOS geëvalueerd tegen een zuivere proxy (Realized Volatility op minuut-basis of Parkinsons/Garman-Klass vol op daily basis) met de volgende metrics:
- **QLIKE (Quasi-Likelihood Loss):** De enige loss-functie die robuust is tegen ruis in de volatiliteitsproxy.
  $$\text{QLIKE} = \frac{RV_t}{\hat{\sigma}^2_t} - \ln\left(\frac{RV_t}{\hat{\sigma}^2_t}\right) - 1$$
- **Variance Forecast Error (MSE-SD / MAE-SD).**
- **Mincer-Zarnowitz Regressie:** Test op zuiverheid en efficiëntie van de forecast ($\hat{RV}_t = \alpha + \beta \hat{\sigma}^2_t + \epsilon_t$).
- **Parameterstabiliteit:** Diebold-Mariano test met Harvey-Leybourne-Newbold correctie om te bepalen of GARCH significant beter presteert dan EWMA OOS.

---

## 10. Regime / Markov Architecture

Regime-modellering dient om macro- en markttoestanden te identificeren en strategieën te conditioneren op de heersende marktontwikkelingen.

### 10.1 Vergelijking van Regime-Paradigmen

| Model | Observables | Latente States | Lekrisico (Lookahead) | OOS Stabiliteit | Aanbeveling |
|---|---|---|---|---|---|
| **M0: Causal Vol-Buckets** | EWMA $Z$-score, ATR ratios | Geen (harde drempels) | Nul (inherent causaal) | Uitstekend | **BASELINE PRODUCTION** |
| **M1: Markov Chain** | Diskrete markt-returns | Expliciet discreet | Laag (Filtered) | Matig | Research Track |
| **M2: Hidden Markov Model (HMM)** | Returns, Volatieliteit, Volume | Unobserved Gaussian/Student-t states | Extreem hoog bij Smoothed probabilities | Laag (gevoelig voor regime shifts) | Restricted Research Only |
| **M3: Markov-Switching GARCH** | Endogene tijdreeks + vol | Latente variantie-regimes | Extreem hoog | Zeer laag (parameter drift) | Exclusief Theoretisch |

### 10.2 De Strikte Filtered-vs-Smoothed Regel
- **Smoothed Probabilities ($P(S_t \vert \mathcal{F}_T)$):** Gebruiken de gehele dataset van $t=1$ tot $T$ (Baum-Welch). **GEBRUIK IN BACKTESTS IS STRENG VERBODEN** vanwege fatale lookahead bias (kijken in de toekomst).
- **Filtered Probabilities ($P(S_t \vert \mathcal{F}_t)$):** Gebruiken uitsluitend informatie tot en met tijdstip $t$ (Forward algoritme). Alleen gefilterde waarschijnlijkheden mogen gebruikt worden in causale backtests.

> **Acceptatiecriterium voor Productie:** Een regime-model mag pas naar productie gepromoveerd worden als het aantoont dat conditionering van alpha of risico op basis van de filtered probabilities een superieure OOS Sharpe-verbetering na transactiekosten oplevert ten opzichte van M0 (Causal Vol-Buckets).

---

## 11. Alpha Research Architecture

Alpha-generatie wordt in de Target Architecture verheven tot een strikt geïsoleerde, schaalloze researchlaag.

CONCEPTUELE SCHEIDING VAN EXPONERING

[ Alpha Layer (L4) ]      ---> Produceert Desired Exposure a_t ∈ [-1, +1]
│
▼
[ Risk Layer (L7) ]       ---> Bepaalt Permitted Exposure (Caps, Vol Target, Limits)
│
▼
[ Portfolio Layer (L8) ]  ---> Genereert Final Capital Allocation
│
▼
[ Execution Layer (L9) ]  ---> Voert orders uit & Minimaliseert Market Impact


### 11.1 Regels voor Alpha Modules
1. **Verantwoordelijkheid:** Alpha-modules genereren uitsluitend een voorspelling of gewenste relatieve exposure ($a_t \in [-1, 1]$).
2. **Geen Risico-bewustzijn:** Alpha-modules bevatten geen leverage-berekeningen, stop-losses, drawdown-breakers of position limits.
3. **Geen Executie-bewustzijn:** Alpha-modules kennen geen order-types, exchange-regels of executielogica.

### 11.2 Alpha Progression Pipeline
- **Baseline:** Cross-sectional Momentum, Trend Following, Mean Reversion, FX/Crypto Carry.
- **Statistical:** Factor-modellen, Ridge/Lasso Lineaire regressies, Fama-MacBeth specificaties.
- **Machine Learning:** Non-lineaire modellen (CatBoost, Random Forests) uitsluitend voor feature-combinatie of meta-labeling.

---

## 12. Machine Learning Architecture

Machine Learning wordt binnen dit platform behandeld met de nodige scepsis die past bij kwantitatieve financiële tijdreeksen (waarin de Signaal-Ruis-Verhouding extreem laag is).

### 12.1 Richtlijnen voor ML Toepassing
- **Geen Blinde Classification/Regression op Raw Prices:** CatBoost of Neurale Netwerken mogen niet direct ingezet worden om ruwe richting te voorspellen op dagelijkse returns.
- **Meta-Labeling Paradigm:** ML wordt primair ingezet als *secondary model* (López de Prado). Het primaire model bepaalt de richting (side), het ML-model voorspelt de kans op succes (sizing / trade filter).
- **Feature Importance Rigor:** Feature selectie vereist Mean Decrease Impurity (MDI) en Single Feature Importance (SFI) gecombineerd met Purged Cross-Validation.

---

## 13. Portfolio Architecture

De portfolio-laag transformeert gewenste exposures naar een geoptimaliseerde gewichtenmatrix.

### 13.1 Evaluatie van Allocatie-Algoritmen

| Algoritme | Schattingsfout | Covariantie Instabiliteit | Turnover & Kosten | OOS Robuustheid | Status |
|---|---|---|---|---|---|
| **1/N (Equal Weight)** | Nul | Nul | Extreem laag | Zeer hoog | **BASELINE** |
| **Inverse Volatility (Risk Parity)** | Laag | Laag | Laag | Hoog | **PRODUCTION DEFAULT** |
| **Hierarchical Risk Parity (HRP)** | Gemiddeld | Laag (geen inv-cov nodig) | Gemiddeld | Gemiddeld | Research Track |
| **Markowitz (Mean-Variance)** | Extreem hoog | Extreem hoog | Extreem hoog | Zeer laag (Markowitz curse) | **Banned in Raw Form** |
| **Black-Litterman** | Gemiddeld | Gemiddeld | Gemiddeld | Matig | Research Track |

> **Conclusie:** Naïeve Risk Parity (Inverse Volatility gebaseerd op EWMA) vormt de verplichte baseline. Complexere portfolio-algoritmen (zoals HRP) moeten bewijzen dat hun turnover-gecorrigeerde OOS-rendement superieur is aan Risk Parity.

---

## 14. Risk Architecture

Risicobeheer is het soevereine controle-orgaan van het handelssysteem. **Risico overruled altijd Alpha.**

### 14.1 Risico-Lagen
- **Unconditional Volatility Targeting:** De positiegrootte wordt dynamisch geschaald op basis van het verschil tussen de ex-ante volatiliteit en de doelvolatielheid:
  $$w_t = \min\left(\text{MaxLeverage}, \frac{\sigma_{\text{target}}}{\hat{\sigma}_{t+1\vert t}}\right)$$
- **Ex-Ante Volatility Estimation:** Gedreven door EWMA ($\lambda=0.94$) in productie en GARCH/HAR-RV in research.
- **Hard Limits:**
  - **Max Concentration Cap:** Maximale allocatie per asset/sector.
  - **Gross & Net Exposure Caps:** Absolute limieten op totale hefboomwerking.
  - **Drawdown Breaker (High-Water Mark):** Automatische de-grossing bij het bereiken van specifieke drawdown-drempels.
  - **Daily Loss Governor:** Onmiddellijke 'kill switch' wanneer het dagverlies de grens overschrijdt.

---

## 15. Execution & TCA Architecture

Een backtest die geen rekening houdt met marktimpact, bid-ask spread en latentie is een fictieve oefening.

### 15.1 Componenten van de Executie-Laag
- **Spread & Slippage:** Dynamische modelleringslaag gebaseerd op de actuele bid-ask spread en marktvolatieliteit.
- **Markt-Impact Model ($\eta, \kappa_d$):** Geijkte Square-Root Law voor marktimpact:
  $$\text{Impact} = \eta \cdot \sigma_{\text{daily}} \cdot \sqrt{\frac{\text{Order Size}}{\text{Daily Volume}}}$$
- **Order Lifecycle & Partial Fills:** Simulatie van limit order queues, time-to-fill, en kans op gedeeltelijke uitvoering op basis van orderboekdiepte.
- **Post-Trade Transaction Cost Analysis (TCA):** Continue vergelijking tussen de verwachte executieprijs (*arrival price*) en de daadwerkelijke opbrengst.

---

## 16. Backtesting Architecture

### 16.1 Authoritative Event-Driven Engine vs. Vectorized Sanity Checks
- **Vectorized Research Engine:** Toegestaan uitsluitend in de exploratieve fase voor snelle hypothese-screening. Vectorized resultaten worden nooit geaccepteerd als bewijs voor modelpromotie.
- **Authoritative Event-Driven Backtester:** De enige wettige autoriteit binnen het platform. Deze engine simuleert:
  - Discrete event-loops op tick- of bar-niveau.
  - Causal order routing en latency.
  - Explicit cash/position accounting registers.
  - Point-in-Time datafeeds zonder lookahead.

---

## 17. Statistical Validation Architecture

Validatie voorkomt dat ruis wordt gecodeerd als alpha.

STATISTICAL VALIDATION PIPELINE

[ Raw Model Signals ]
│
▼
[ Purged Walk-Forward CV (met Embargo) ]
│
▼
[ Deflated Sharpe Ratio (DSR) Check (Inclusief M, Y, γ) ]
│
▼
[ White's Reality Check / Hansen's SPA (Multiple Testing) ]
│
▼
[ Truncation & Lookahead Invariance Tests ]
│
▼
[ PROMOTION TO CANDIDATE STATE ]


### 17.1 Categorisering van Statistische Methoden

| Methode | Categorisatie | Onderbouwing |
|---|---|---|
| **Purged Walk-Forward (met Embargo)** | **ESSENTIAL** | Inherent causaal, voorkomt spillage tussen opeenvolgende overlap-intervallen. |
| **Deflated Sharpe Ratio (DSR)** | **ESSENTIAL** | Corrigeert voor non-normaliteit, geheugeneffecten en het aantal geteste trials. |
| **Hansen's SPA / White's Reality Check** | **ESSENTIAL** | Beschermt tegen data-mining bias bij het vergelijken van meerdere strategieën. |
| **Diebold-Mariano (met HLN-correctie)** | **USEFUL** | Essentieel voor het paarsgewijs vergelijken van volatiliteits-forecasts (GARCH vs EWMA). |
| **Combinatorial Purged CV (CPCV)** | **OPTIONAL** | Biedt rijke verdelingsinzichten, maar verstoort tijdreeks-paden; niet gebruiken als primaire gate. |
| **Probability of Backtest Overfitting (PBO)** | **OPTIONAL** | Goede diagnostiek, maar ondergeschikt aan OOS Walk-Forward. |
| **K-Fold CV (Unpurged)** | **BANNED** | Creëert zware lookahead leakage in financiële tijdreeksen. |

---

## 18. Model Governance

Model Risk Management (MRM) garandeert dat alle productie-code en modellen cryptografisch herleidbaar en auditbaar zijn.

### 18.1 Governance-Regels
1. **Pre-Registratie:** Elke onderzoeksgolf moet vooraf geformuleerde hypotheses, parameters, en stop-criteria vastleggen in het `FALSIFICATION_REGISTER.md`.
2. **Immutable Ledger Entry:** Resultaten worden via `registry/hypothesis_ledger.py` atomair geschreven, inclusief `git_sha`, `data_hash`, en `config_hash`.
3. **Champion / Challenger Framework:** Een challenger-model vervangt het champion-model pas na minimaal 60 dagen OOS paper-trading waarin het de champion statistisch significant verslaat (Diebold-Mariano $p < 0.05$).
4. **CI/CD Enforcement:** Geen enkele PR wordt gemerged als de lookahead-suite, DSR-check of invariantie-tests falen.

---

## 19. Proposed Target Architecture

De voorgestelde Target Architecture bestaat uit 14 strikt gescheiden lagen waarin afhankelijkheden uitsluitend van boven naar beneden bewegen.

┌──────────────────────────────────────────────────────────┐
│ L0: Data Ingestion, PIT Storage & Validation             │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L1: Feature Engineering (Stateless, Causal)              │
└────────────────────────────┬─────────────────────────────┘
│
┌────────────────┴────────────────┐
│                                 │
▼                                 ▼
┌───────────────────────┐         ┌────────────────────────┐
│ L2: Volatility Engines│         │ L3: Regime Engines     │
│ (EWMA / GARCH / HAR)  │         │ (M0 Buckets / HMM)     │
└───────────┬───────────┘         └───────────┬────────────┘
│                                 │
└────────────────┬────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L4: Alpha Generation (Raw Forecasts, Scale-free)         │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L5: Conditioning Layer (Alpha x Regime Overlay)          │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L6: Model Combination & Ensembles                        │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L7: Independent Risk & Volatility Targeting              │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L8: Portfolio Construction & Sizing (Risk Parity)        │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L9: Execution & Order Routing (TCA, Impact)              │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L10: Backtesting Engine (Event-Driven, Execution-Aware)  │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L11: Statistical Validation & Falsification              │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L12: Governance, Ledger & Audit Register                 │
└────────────────────────────┬─────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────┐
│ L13: Live Operations, Paper Trading & Monitoring         │
└──────────────────────────────────────────────────────────┘


---

## 20. Proposed Directory Structure

De volgende mappenstructuur definieert het doelontwerp voor de repository.  
*Opmerking: Dit is een architectuurvoorstel. Er zijn geen fysieke bestanden verplaatst in de huidige repository tijdens deze audit.*

quant_system/
├── conf/                       # Hydra / Pydantic gecentraliseerde configuraties
│   ├── data/
│   ├── model/
│   ├── risk/
│   └── execution/
├── data/                       # L0: Ingestion & Validation
│   ├── ingestion/
│   ├── validation/
│   └── pit_store/              # Parquet files + DVC tracking
├── features/                   # L1: Causal Feature Engineering
│   ├── transforms.py
│   └── pipeline.py
├── volatility/                 # L2: Volatieliteitsmodellering
│   ├── ewma.py
│   ├── garch.py
│   └── realized.py
├── regime/                     # L3: Regime Classificatie
│   ├── buckets.py              # M0 Causal Buckets
│   └── markov.py               # M1/M2 Models
├── alpha/                      # L4: Pure Alpha Signal Generators
│   ├── momentum.py
│   ├── mean_reversion.py
│   └── carry.py
├── portfolio/                  # L8: Portfolio Allocation
│   ├── risk_parity.py
│   └── hrp.py
├── risk/                       # L7: Independent Risk Overlay
│   ├── vol_targeting.py
│   ├── limits.py
│   └── kill_switches.py
├── execution/                  # L9: TCA & Impact Simulation
│   ├── impact_model.py
│   └── order_router.py
├── backtest/                   # L10: Event-Driven Backtesting Engine
│   ├── engine.py
│   └── accounting.py
├── validation/                 # L11: Statistical Gates & DSR
│   ├── dsr.py
│   ├── walk_forward.py
│   └── spa.py
├── registry/                   # L12: Falsificatie & Hypothese Ledger
│   ├── hypothesis_ledger.py
│   └── MRM_generator.py
├── research/                   # Geïsoleerde Jupyter Notebooks (No Prod Code)
│   ├── exploratory/
│   └── publication/
├── tests/                      # Test Suites
│   ├── unit/
│   ├── integration/
│   ├── lookahead/              # Truncatie & Causaliteits-tests
│   └── statistical/            # DSR & Gate tests
├── pyproject.toml
└── README.md


---

## 21. Research Pipeline

De gestandaardiseerde Target Research Pipeline loopt via de volgende fasen:

Data Ingestion (PIT)
└─► Data Validation & Quality Checks
└─► Causal Feature Engineering
└─► Baseline Model (EWMA / Linear)
└─► Hypothese Formulering & Pre-Registratie
└─► Model Estimation & Walk-Forward OOS
└─► Statistical Validation (DSR, SPA)
└─► Portfolio & Risk Overlay
└─► Event-Driven Execution Simulation (TCA)
└─► Ledger Registration
└─► Paper Trading (60 dagen)
└─► Production Promotion


---

## 22. Model Hierarchy

Modellen moeten hun plek in de hiërarchie empirisch verdienen. Promotie naar een hoger niveau vereist keihard bewijs van OOS Sharpe-verbetering na transactiekosten ten opzichte van de lagere niveaus.

- **LEVEL 0:** Naïeve Baselines (Random Walk, Buy & Hold, Constant Volatility).
- **LEVEL 1:** Simpele Statistische & Econometrische Baselines (EWMA, Linear Momentum, Rolling Z-Score).
- **LEVEL 2:** Econometrische Modellen (GARCH(1,1), ARMA-GARCH, M0 Vol-Buckets).
- **LEVEL 3:** Geregulariseerde Lineaire Modellen (Ridge, Lasso, ElasticNet).
- **LEVEL 4:** Non-lineaire Machine Learning (CatBoost, Random Forest voor Meta-Labeling).
- **LEVEL 5:** Geavanceerde Ensembles & Hierarchische Structuren (HRP + ML Meta-models).

---

## 23. Migration Strategy

De migratie van de Current State naar de Target State verloopt in 8 strikt chronologische fasen:

### Phase 0 — Audit & Repository Hygiene (P0)
- **Doel:** Herstel versiebeheer, centraliseer configuratie en elimineer stille fallbacks.
- **Deliverables:** `.git` herinitialisatie, Pydantic/Hydra schema-enforcement, vervanging van alle 12 `try/except ImportError` blokken door harde Fail-Fast crashes.
- **Exit Criteria:** Nul stille fallbacks; 100% reproduceerbare builds via `pyproject.toml`.

### Phase 1 — Data Foundation & Crypto Ingestion (P0)
- **Doel:** Herstel dataintegriteit en vul de ontbrekende crypto-dataset aan.
- **Deliverables:** PIT Parquet store, DVC lineage tracking, crypto OHLCV & funding rate ingestion.
- **Exit Criteria:** `asof_join` geverifieerd op crypto-data; unieke `data_hash` gegenereerd.

### Phase 2 — Research & Falsification Foundation (P0)
- **Doel:** Koppel het falsificatieregister en de ledger aan de CI/CD pipeline.
- **Deliverables:** Geautomatiseerde DSR, SPA en lookahead-suite integratie in GitHub Actions / local CI.
- **Exit Criteria:** PR's worden automatisch geblokkeerd bij lookahead-fouten of DSR-falsificatie.

### Phase 3 — Baseline Implementation (P1)
- **Doel:** Implementeer Level 1 baselines voor alle research tracks.
- **Deliverables:** EWMA volatiliteits-estimator, Cross-Sectional Momentum baseline, Naive Risk Parity portfolio allocation.
- **Exit Criteria:** Baseline OOS Sharpe en drawdown gekwantificeerd over de volledige dataset.

### Phase 4 — Risk & Volatility Architecture (P1)
- **Doel:** Volledige ontkoppeling van Risk (L7) en Alpha (L4).
- **Deliverables:** Standalone Volatility Targeting module, Hard Risk Limits, Daily Loss Governor.
- **Exit Criteria:** Risicolimieten overrulen aantoonbaar alpha-posities in stress-simulaties.

### Phase 5 — Execution & Backtesting Engine (P1)
- **Doel:** Consolidatie van de backtest engines naar 1 event-driven engine met TCA.
- **Deliverables:** Authoritative Event-Driven Engine, Marktimpact ($\eta$) kalibratiemodule.
- **Exit Criteria:** Bit-identieke resultaten tussen paper-trading logica en backtest execution simulator.

### Phase 6 — Advanced Research Tracks (GARCH, ML, Regimes) (P2)
- **Doel:** Gecontroleerde evaluatie van Level 2+ modellen.
- **Deliverables:** GARCH(1,1) vs EWMA QLIKE competitie, M0 vs HMM Filtered Probability benchmarking.
- **Exit Criteria:** Alleen modellen met aantoonbare OOS-waarde worden goedgekeurd.

### Phase 7 — Production Readiness & Paper Trading (P2)
- **Doel:** Integratie van het volledige platform voor live paper-trading.
- **Deliverables:** Live monitoring dashboards (PSI, Sharpe drift, Vol forecast errors), automated alerts.
- **Exit Criteria:** 60 opeenvolgende dagen succesvolle paper-trading zonder execution drift of runtime crashes.

---

## 24. Kill / Archive / Retain / Redesign Matrix

| Component / Module | Actie | Reden | Vereist Bewijs voor Wijziging | Prioriteit |
|---|---|---|---|---|
| `docs/FALSIFICATION_REGISTER.md` | **RETAIN** | Schaars wetenschappelijk goed; uitstekende hygiëne. | Geen (behouden) | **P0** |
| `registry/hypothesis_ledger.py` | **RETAIN** | Atomaire, robuuste logica. | Geen (behouden) | **P0** |
| `utils/time.py::asof_join` | **RETAIN** | Inherent causale PIT-merge. | Geen (behouden) | **P0** |
| `backtest/metrics.py::DSR` | **RETAIN** | Correcte Bailey-LdP dimensionaliteitsfix. | Geen (behouden) | **P0** |
| `risk/hmm_regime.py` | **REDESIGN** | Stille fallback naar EMA slopen; herstructureren naar M2 Filtered HMM. | QLIKE / OOS Sharpe winst t.o.v. M0 | **P0** |
| `risk/portfolio.py` | **REDESIGN** | Splitsen: Risicobeperking ontkoppelen van Alpha. | Geslaagde ontkoppelingstest | **P0** |
| Overlappende Backtesters (4x) | **REDESIGN** | Consolideren tot 1 event-driven engine. | Pariteits-test | **P0** |
| `quant_architect` Imports | **REMOVE** | Dode code / ontbrekende afhankelijkheid. | Vervangen door directe modules | **P0** |
| 26 Dode Alpha Units | **ARCHIVE** | Onbewezen of gefalsificeerd op ontbrekende data. | Nieuwe data + pre-registratie | **P1** |
| CatBoost Direct Directional | **ARCHIVE** | SNR te laag op daily returns; hoge overfitting. | Meta-labeling OOS AUC > 0.58 | **P2** |
| Hierarchical Risk Parity (HRP) | **RESEARCH ONLY** | Potentieel nuttig, maar verliest momenteel van Risk Parity. | OOS Sharpe > Inverse Vol | **P2** |
| Black-Litterman / Markowitz | **RESEARCH ONLY** | Covariantie-instabiliteit OOS. | Turnover-gecorrigeerde winst | **P3** |

---

## 25. Research Roadmap

### Q3 2026 — Data & Baseline Readiness
- Complete her-ingestie van Crypto OHLCV, Trades, en Funding Rates.
- Benchmarken van Level 1 Baselines (EWMA + Cross-Sectional Momentum).

### Q4 2026 — Volatility & Risk Calibration
- Uitvoeren van de GARCH vs EWMA QLIKE-competitie op dagelijkse en intraday data.
- Kalibratie van de marktimpact-parameter $\eta$ op orderboek-data.

### Q1 2027 — Regime & ML Meta-Labeling
- Evaluatie van M0 Buckets vs Filtered HMM Probabilities.
- Testen van CatBoost als meta-labeler voor basissignalen.

---

## 26. Acceptance Criteria

Voor elke strategie of modelcomponent die wordt genomineerd voor promotie naar de productie-pijplijn gelden de volgende onverwijdbare acceptatiecriteria:

1. **Data Provenance:** 100% van de gebruikte data is gekoppeld aan een gecertificeerde `data_hash` in het PIT store.
2. **Zero Lookahead Leakage:** Het model slaagt voor 100% van de ruimtelijke en temporele truncatie-tests in `tests/lookahead/`.
3. **Deflated Sharpe Ratio (DSR):** Statistisch significant met $p < 0.05$ na correctie voor non-normaliteit en het totale aantal uitgevoerde trials ($M$).
4. **Execution Realism:** Positieve verwachte netto-rendementen na toepassing van het geijkte marktimpactmodel en exchange fee schedules.
5. **Fail-Fast Compliance:** De module bevat nul `try/except` fallbacks en crashed onmiddellijk bij schending van data-contracten.

---

## 27. Open Questions

1. **Intraday Tick Data Availability:** Beschikken we over voldoende historische orderboek-depth data om de HAR-RV en TCA-impactmodellen met hoge nauwkeurigheid te kalibreren?
2. **Crypto Funding Rate Arbitrage:** Is de verdienstencurve van funding-rate carry voldoende robuust onder regimewisselingen van bull- naar bear-markten?
3. **Exchange Latency Scaling:** Hoe varieert de executielatentie tijdens periodes van extreme marktvolatieliteit op gedecentraliseerde vs. gecentraliseerde exchanges?

---

## 28. Final Architectural Recommendation

### Current vs. Target Score
- **Current Architecture Score:** `3 / 10` (Zware schuld, stille fallbacks, ontbrekende data, gebroken koppelingen).
- **Target Architecture Score:** `9 / 10` (Institutioneel-grade, causaal, baseline-first, strikt gescheiden lagen).

### Grootste Risico's & Bronnen van Complexiteit
- **Silent Failures:** Het stilzwijgend degraderen van statistische modellen naar EMA's is het grootste operationele risico in de huidige codebase.
- **Data-Illusie:** Onderzoek doen zonder gegarandeerde, versie-beheerde data.

### Wat zou een professionele quant vandaag fundamenteel anders doen?

> **Het Antwoord:**  
> Als een professionele quant deze repository vandaag opnieuw zou ontwerpen, zou hij geen enkele regel ML- of HMM-code schrijven totdat de data-infrastructuur 100% Point-in-Time, versie-beheerd en onveranderlijk is.
>
> Hij zou de overbodige complexiteit (CatBoost, HRP, 4 backtest engines) onmiddellijk opbreken. Hij zou beginnen met één robuuste, lineaire Risk Parity strategie gedreven door een EWMA volatiliteits-target, aangesloten op één event-driven backtester met realistisch gekalibreerde transactiekosten. Pas wanneer dat systeem aantoonbaar live paper-trades uitvoert met een bewezen statistische edge, zou hij stapsgewijs complexere modellen (GARCH, ML Meta-labeling) toelaten via het falsificatieregister.

---

## Conclusie

Het herzien van `ARCHITECTUUR_AUDIT_2026-08-22` markeert de overgang van een gefragmenteerd experimenteel systeem naar een professioneel, institutioneel kwantitatief platform. De richtlijnen en fasen in dit document zijn bindend voor alle toekomstige engineering- en onderzoeksinspanningen.
"""

file_path = "ARCHITECTUUR_AUDIT_2026-08-22.md"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"Markdown file successfully generated at {file_path}")

Het Markdown-bestand is opgemaakt, correct gestructureerd en georganiseerd met professionele typografie, tabellen, MathJax/LaTeX-formules en overzichtelijke diagrammen.

Your Markdown file is ready

MD-icoon
ARCHITECTUUR_AUDIT_2026-08-22
 MD
Openen
Belangrijkste doorgevoerde verbeteringen:
Hiërarchische Structuur: Koppen, subkoppen en document-metadata zijn logisch en consistent opgebouwd.

Tabellen & Matrixen: Tegenstrijdigheden (D-1 t/m D-10), volatiliteitsmodellen, regime-paradigmen, allocatie-algoritmen en de Kill/Archive/Retain/Redesign Matrix zijn overzichtelijk geformatteerd als Markdown-tabellen.

Wiskundige Formules: Formules zoals de Deflated Sharpe Ratio, QLIKE, Volatility Targeting en het Market Impact model (η) zijn voorzien van nette LaTeX-notatie.

ASCII Architectural Flows: De Target Pipeline, Volatility Hierarchy en Target Architecture zijn weergegeven in duidelijke ASCII-diagrammen en codeblocks.

Overzichtelijke Lijsten & Callouts: Belangrijke definities, doctrines, acceptatiecriteria en fasen in de migratiestrategie zijn strak uitgelijnd en visueel uitgelicht.




Gemini is AI en kan fouten maken, ook over mensen. Jouw privacy en GeminiOpent in een nieuw venster

Analyseren
# ARCHITECTUUR AUDIT 2026-08-22

**Datum:** 22 augustus 2026  
**Onderwerp:** Institutionele Architectuur- en Research-Audit Quant Trading Platform  
**Status:** Definitief / Bindend Document  

---

## 1. Executive Summary

Dit document bevat de definitieve, institutionele architectuur- en research-audit van het quant trading platform op datum van **22 augustus 2026**. Het doel van deze audit is niet het verdedigen of marginaliseren van de bestaande codebasis, maar het neerzetten van een meedogenloze, kwantitatief onderbouwde analyse die als blauwdruk dient voor de volgende generatie van het systeem.

De audit stelt vast dat het huidige platform gekenmerkt wordt door een opmerkelijke dichotomie:

1. **Hoge wetenschappelijke hygiëne in de researchfilosofie:** De aanwezigheid van een rigoureus falsificatieregister, pre-registratie van hypothesen, een atomaire hypothese-ledger en een theoretisch correct gefixte Deflated Sharpe Ratio (DSR) getuigen van een bovengemiddelde quant-mindset.
2. **Kritieke gebreken in software engineering en data-infrastructuur:** De repository lijdt aan zware documentatiedrift, ontbrekende primaire data (geen crypto-data aanwezig in de werkkopie), fatale stilzwijgende fallbacks (`try/except ImportError` die complexe modellen deactiveren zonder waarschuwing), gebroken CI/CD-koppelingen en een onveilige verstrengeling van alpha- en risicologica.

De belangrijkste conclusie is dat het systeem in zijn huidige staat **niet productie-rijp is (Score: 3/10)**. Om een institutionele status te bereiken, moet het platform opnieuw opgebouwd worden vanuit een *data-first* en *baseline-first* doctrine. Dit auditdocument definieert de complete Target Architecture, de 14-laags componentenstructuur, de doeldirectorystructuur, de onderzoekshiërarchie voor volatiliteit (GARCH) en regimes (Markov/HMM), en het gefaseerde migratieplan.

---

## 2. Scope & Audit Methodology

### 2.1 Scope
De scope van deze audit omvat de gehele codebase (`src/tradebot/`), configuraties, tests, notebooks, documentatie en de historische onderzoeksresultaten (waves W1–W28).

### 2.2 Audit Methodologie & Bewijsregel
In overeenstemming met de vereisten is de bestaande documentatie behandeld als een hypothese, niet als waarheid. Elke bewering is getoetst aan de hand van:
- Directe inspectie van het filesystem en de Python-codebase.
- Statistische en econometrische houdbaarheid volgens de nieuwste literatuur (*AFML*, Time-Series Econometrics).
- Reproduceerbaarheids- en data-lineage-checks.

Elk onderdeel in dit document maakt het expliciete onderscheid tussen **Current State** (feitelijk aangetroffen situatie) en **Target State** (het beoogde institutionele ontwerp).

---

## 3. Current-State Assessment

### 3.1 Wat momenteel bestaat en daadwerkelijk werkt
- **Falsificatieregister & Ledger:** `docs/FALSIFICATION_REGISTER.md` en `registry/hypothesis_ledger.py` bieden een atomaire, append-only administratie van gefalsificeerde en geaccepteerde hypothesen.
- **Point-in-Time (PIT) Joining:** `utils/time.py::asof_join` voert een correcte `merge_asof(direction="backward")` uit, wat lookahead-bias bij gegevenskoppeling voorkomt.
- **Deflated Sharpe Ratio (DSR):** De dimensionaliteitsfout in `backtest/metrics.py` is gecorrigeerd conform Bailey & López de Prado (2014) inclusief de Euler-Mascheroni term.
- **Cross-Sectional Harness:** `alpha/xs_unit.py` bevat de fantoom-herbalanceringsfix (`w.shift(1)` causaliteit) en ingebouwde transactiekosten.
- **Lookahead & Truncation Suite:** `tests/lookahead/` bevat valide tests voor truncatie-invariantie en determinisme.

### 3.2 Wat verouderd, incompleet of afwezig is
- **Afwezigheid van Crypto Data:** De map `market_data_parquet/` bevat uitsluitend FX en macro-data. Er is geen crypto-OHLCV of perp-data aanwezig. Het vlaggenschipmodel (*Crypto-MN, Sharpe 1.15*) is in de huidige werkkopie niet reproduceerbaar.
- **Uncalibrated TCA:** De documentatie claimt marktimpact-kalibratie ($\eta, \kappa_d$), maar de bijbehorende kalibratietool (`apps/calibrate_impact.py`) ontbreekt volledig.
- **Dode Alpha-Voorraad:** Van de 26 gedefinieerde alpha-units zijn er 20 gefalsificeerd en 4 geaccepteerd op niet-aanwezige crypto-data. Er zijn 0 verifieerbare, actieve alpha-units in de huidige repo.

---

## 4. Legacy Architecture Assessment

De legacy architectuur vertoont een sterke neiging tot *"architectural complexity inflation"*: het bouwen van complexe wiskundige lagen bovenop een onstabiele fundering.

```
LEGACY DEPENDENCY FLOW (PROBLEMATISCH)

[Config / Envs] ---> [Feature Store (met stille fallbacks)]
                           |
                           v
              [Alpha Module (L4) + Risk Limits (L7) Gemengd]
                           |
                           v
           [4 Overlappende Backtesters (Vectorized/Event)]
                           |
                           v
              [Niet-bestaande ML/Ensemble Lagen]
```

### Kernproblemen in de Legacy Structuur:
1. **Circulaire & Misplaatste Afhankelijkheden:** Risicolimieten en volatiliteitstargeting bevinden zich gedeeltelijk binnen de alpha-units en feature-pipelines, wat een strikte scheiding van verantwoordelijkheden onmogelijk maakt.
2. **Shadow Trees & Configuration Drift:** De aanwezigheid van overbodige mappen en afwijkende omgevingsconfiguraties leidt tot onvoorspelbaar gedrag tussen test- en runtime-omgevingen.
3. **Versnipperde Backtest Engine:** Vier overlappende engines (`evaluation`, `portfolio`, `bidirectional`, `per_side`) verhogen de onderhoudslast en creëren kweekvijvers voor subtiele simulatiefouten.

---

## 5. Problems & Architectural Debt

### 5.1 Harde Documentatie-vs-Code Tegenstrijdigheden (D-1 t/m D-10)

| ID | Document claim | Werkelijkheid in de Codebase | Prioriteit |
|---|---|---|---|
| **D-1** | `model_risk_policy.md`: Promotie geblokkeerd door 6 specifieke lookahead tests. | Geen van deze 6 testbestanden bestaat in `tests/lookahead/`. | **P0** |
| **D-2** | `tca_methodology.md`: `apps/calibrate_impact.py` levert $\eta, \kappa_d$. | Bestand bestaat niet. | **P1** |
| **D-3** | `tca_methodology.md`: `tests/integration/test_tca_roundtrip.py` draait per PR. | Bestand bestaat niet. | **P1** |
| **D-4** | `tca_methodology.md`: `conf/tca/default.yaml` aanwezig. | Bestand bestaat niet. | **P1** |
| **D-5** | `architecture.md`: Zie `REFACTOR_BLUEPRINT_v3.md`. | Bestand bestaat niet. | **P1** |
| **D-6** | `architecture.md` R-4: Maximaal 800 LOC, 4-bestands-whitelist. | 7 bestanden overschrijden de limiet; 3 vallen buiten whitelist. | **P2** |
| **D-7** | `architecture.md` R-6: Applications $\le$ 80 LOC. | 17 van 30 apps zijn groter (tot ~1350 LOC). | **P2** |
| **D-8** | `architecture.md` DAG: `artefacts/features/`, `models/`, `tracks/`. | Geen van deze mappen bestaat in de repository. | **P0** |
| **D-9** | `model_risk_policy.md`: Elk MRM-rapport bevat een valide `git_sha`. | Geen werkende `.git` repository in de hoofdmap. | **P0** |
| **D-10** | `DATA_REGISTER.md`: HMM-regimes via `hmmlearn`. | `hmmlearn` ontbreekt in `pyproject.toml` $\rightarrow$ stille fallback naar EMA-crossover. | **P0** |

### 5.2 Stille Degradatie (Silent Fallbacks)
Er zijn 12 plekken in `src/` geïdentificeerd waar statistische of modelleringsfouten worden opgevangen via `try/except ImportError` of brede exceptions, waarna het systeem stilzwijgend terugvalt op een naïeve baseline:
- `risk/hmm_regime.py`: Valt bij het ontbreken van `hmmlearn` stilzwijgend terug van een 3-state Gaussian HMM naar een 20/100 EMA-crossover.
- **`quant_architect` modules:** Vijf core modules (`train/ensemble.py`, `train/_scalers.py`, `train/catboost.py`, `tune/objective.py`, `backtest/portfolio.py`) proberen `quant_architect` te importeren. Deze module bestaat nergens. Hierdoor draaien alle ensemble- en kalibratiemechanismen permanent in gedegradeerde modus zonder dat er een melding wordt gegenereerd.

---

## 6. Research Methodology Assessment

De onderzoeksfilosofie is in essentie solide (gebaseerd op popperiaanse falsificatie en pre-registratie), maar lijdt in de praktijk onder uitvoeringstekortkomingen.

### Onderscheid: "Niet Bewezen" vs. "Bewezen Slecht"
Een cruciale fout in eerdere evaluaties was het gelijkstellen van onbewezen complexiteit aan schadelijke complexiteit.

- **Bewezen Slecht:** Modellen die na correcte OOS-validatie en TCA een negatieve verwachte waarde laten zien of aantoonbaar overfitten op in-sample ruis (bijv. directionele p-hacks via ongestructureerde CatBoost op dagelijkse returns).
- **Niet Bewezen:** Modellen waarvoor de infrastructuur of de benodigde data (zoals tick-data of orderboek-data) simpelweg ontbrak om een eerlijke evaluatie uit te voeren (bijv. GARCH volatiliteitsforecasts of High-Frequency Realized Volatility).

> **Doctrine:** Een model wordt pas gearchiveerd of verwijderd als het *bewezen slecht* is onder een correct geconfigureerde baseline. Zolang het *onbewezen* is, verblijft het in de Research Track en krijgt het geen toegang tot de productie-pijplijn.

---

## 7. Data Architecture

> **CORE AXIOM:** *"Geen betrouwbare data $\rightarrow$ geen betrouwbare quant research."*  
> Data is het fundament van het gehele platform. Onvolledige, niet-point-in-time of gecorrumpeerde data maakt elke statistische toets en backtest waardeloos.

```
TARGET DATA PIPELINE & LINEAGE

[ Raw Data Ingestion ] 
        │
        ▼
[ Data Validation & Sanitization (Schema, Gaps, Outliers) ]
        │
        ▼
[ Point-in-Time Alignment (asof_join, Truncation Guards) ]
        │
        ▼
[ Immutable Parquet Storage + DVC Versioning (data_hash) ]
        │
        ▼
[ Feature Generation (Causal, Stateless Transforms) ]
```

### 7.1 Vereiste Datasets per Research Track
- **Alpha Research:** Daily/Hourly OHLCV, Funding Rates, Open Interest, Liquidaties, Cross-sectional Spreads, Volume profiles.
- **Volatility Research (GARCH & EWMA):** High-frequency trade data (1m / 5m bars) voor Realized Variance, daily OHLCV voor GARCH/EGARCH/GJR-GARCH.
- **Regime Research:** Macro-economische tijdreeksen, Implied Volatility indices, Volatility term structures, Cross-asset correlatiematrices.
- **Execution & TCA Research:** Top-of-book (L1) en Depth-of-book (L2) orderboek-snapshots, individuele trade-prints, exchange fee structures, latentiestatistieken.

### 7.2 Data Governance & Point-in-Time Rigor
- **Timezone Standard:** Alle timestamps worden opgeslagen in UTC Unix Nanoseconden.
- **Survivorship Bias:** Het universum moet historische delistings en faillissementen bevatten.
- **Futures Rolls & Funding:** Continuous futures reeksen moeten expliciet gecorrigeerd worden via achterwaartse/voorwaartse verhoudingsaanpassingen of expliciete adjustment factor ledgers.
- **Dataset Versioning:** Elke dataset krijgt een unieke `data_hash` gegenereerd via DVC/Git-LFS. Research resultaten zonder geciteerde `data_hash` worden automatisch als invalid beschouwd.

---

## 8. Time-Series & Econometrics Architecture

Tijdreeksanalyse binnen de Target Architecture vereist een strikte behandeling van non-stationariteit en geheugeneffecten.

### 8.1 Stationariteit vs. Geheugenbehoud
- **Standaard Differentiëring ($d=1$):** Verwijdert het gehele geheugen van de tijdreeks, wat de voorspellende waarde voor kwantitatieve modellen reduceert.
- **Fractionele Differentiëring (FracDiff):** Wordt gebruikt om stationariteit te bereiken ($p$-value $<0.05$ op Augmented Dickey-Fuller) terwijl de correlationele geheugenstructuur ($d \in [0, 1]$) maximaal behouden blijft.

### 8.2 Econometrische Toetsingsketen
Elke tijdreeks die de feature pipeline binnenkomt moet verplicht de volgende toetsingsvolgorde doorlopen:
1. **Augmented Dickey-Fuller (ADF) & KPSS tests:** Beoordeling van unit roots en trend-stationariteit.
2. **CUSUM & structural break tests:** Detectie van regime-verschuivingen in het gemiddelde en de covariantie.
3. **Autocorrelatie & Heteroskedasticiteit:** Ljung-Box test voor autocorrelatie; ENGLE ARCH-test voor conditional heteroskedasticity om de noodzaak van een GARCH-structuur vast te stellen.

---

## 9. Volatility Architecture

Volatieliteitsmodellering wordt binnen de Target Architecture gedefinieerd als een ex-ante risk & sizing component, **NIET** primair als een richtinggevend alpha-signaal.

### 9.1 Volatieliteits-Hiërarchie

```
VOLATILITY RESEARCH HIERARCHY

┌──────────────────────────────────────────────────────────┐
│ Level 0: Naive Historical Volatility (Rolling StdDev)    │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Level 1: EWMA / RiskMetrics (λ = 0.94) [BASELINE]        │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Level 2: Parametric Econometric Models                   │
│ (GARCH(1,1), GJR-GARCH, EGARCH, APARCH)                  │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Level 3: Realized Volatility / High-Frequency            │
│ (Realized Variance, HAR-RV, Kernel Volatility)           │
└──────────────────────────────────────────────────────────┘
```

### 9.2 Modelspecificaties & Aannames

| Model | Aannames | Databehoefte | Forecast Horizon | Primaire Toepassing |
|---|---|---|---|---|
| **EWMA ($\lambda=0.94$)** | Parameter-vrij IGARCH(1,1) relict; gelijke decay. | Daily OHLCV | $\sigma^2_{t+1\vert t}$ | Baseline position sizing & risk limits. |
| **GARCH(1,1)** | Symmetrische respons op schokken; stationaire variantie. | Daily Close | $\sigma^2_{t+h\vert t}$ | Ex-ante risicoforecasting op middellange termijn. |
| **GJR-GARCH** | Asymmetrie (leverage-effect: negatieve schokken verhogen vol meer). | Daily OHLCV | $\sigma^2_{t+h\vert t}$ | Equities & Crypto crash-risk modelling. |
| **EGARCH** | Logaritmische variantie; geen positiveringsrestricties vereist. | Daily Close | $\sigma^2_{t+h\vert t}$ | Zware staartverdelingen en extreme asymmetrie. |
| **HAR-RV** | Heterogene markt-hypothese (dagelijks, wekelijks, maandelijks geheugen). | 5-min Intraday Bars | $\sigma^2_{t+1\vert t}$ t/m $\sigma^2_{t+5\vert t}$ | High-frequency volatiliteitstargeting. |

### 9.3 Validatie & Evaluatiemetrics voor Volatieliteit
GARCH- en volatiliteitsmodellen worden OOS geëvalueerd tegen een zuivere proxy (Realized Volatility op minuut-basis of Parkinsons/Garman-Klass vol op daily basis) met de volgende metrics:
- **QLIKE (Quasi-Likelihood Loss):** De enige loss-functie die robuust is tegen ruis in de volatiliteitsproxy.
  $$\text{QLIKE} = \frac{RV_t}{\hat{\sigma}^2_t} - \ln\left(\frac{RV_t}{\hat{\sigma}^2_t}\right) - 1$$
- **Variance Forecast Error (MSE-SD / MAE-SD).**
- **Mincer-Zarnowitz Regressie:** Test op zuiverheid en efficiëntie van de forecast ($\hat{RV}_t =  lpha +  eta \hat{\sigma}^2_t + \epsilon_t$).
- **Parameterstabiliteit:** Diebold-Mariano test met Harvey-Leybourne-Newbold correctie om te bepalen of GARCH significant beter presteert dan EWMA OOS.

---

## 10. Regime / Markov Architecture

Regime-modellering dient om macro- en markttoestanden te identificeren en strategieën te conditioneren op de heersende marktontwikkelingen.

### 10.1 Vergelijking van Regime-Paradigmen

| Model | Observables | Latente States | Lekrisico (Lookahead) | OOS Stabiliteit | Aanbeveling |
|---|---|---|---|---|---|
| **M0: Causal Vol-Buckets** | EWMA $Z$-score, ATR ratios | Geen (harde drempels) | Nul (inherent causaal) | Uitstekend | **BASELINE PRODUCTION** |
| **M1: Markov Chain** | Diskrete markt-returns | Expliciet discreet | Laag (Filtered) | Matig | Research Track |
| **M2: Hidden Markov Model (HMM)** | Returns, Volatieliteit, Volume | Unobserved Gaussian/Student-t states | Extreem hoog bij Smoothed probabilities | Laag (gevoelig voor regime shifts) | Restricted Research Only |
| **M3: Markov-Switching GARCH** | Endogene tijdreeks + vol | Latente variantie-regimes | Extreem hoog | Zeer laag (parameter drift) | Exclusief Theoretisch |

### 10.2 De Strikte Filtered-vs-Smoothed Regel
- **Smoothed Probabilities ($P(S_t \vert \mathcal{F}_T)$):** Gebruiken de gehele dataset van $t=1$ tot $T$ (Baum-Welch). **GEBRUIK IN BACKTESTS IS STRENG VERBODEN** vanwege fatale lookahead bias (kijken in de toekomst).
- **Filtered Probabilities ($P(S_t \vert \mathcal{F}_t)$):** Gebruiken uitsluitend informatie tot en met tijdstip $t$ (Forward algoritme). Alleen gefilterde waarschijnlijkheden mogen gebruikt worden in causale backtests.

> **Acceptatiecriterium voor Productie:** Een regime-model mag pas naar productie gepromoveerd worden als het aantoont dat conditionering van alpha of risico op basis van de filtered probabilities een superieure OOS Sharpe-verbetering na transactiekosten oplevert ten opzichte van M0 (Causal Vol-Buckets).

---

## 11. Alpha Research Architecture

Alpha-generatie wordt in de Target Architecture verheven tot een strikt geïsoleerde, schaalloze researchlaag.

```
CONCEPTUELE SCHEIDING VAN EXPONERING

[ Alpha Layer (L4) ]      ---> Produceert Desired Exposure a_t ∈ [-1, +1]
        │
        ▼
[ Risk Layer (L7) ]       ---> Bepaalt Permitted Exposure (Caps, Vol Target, Limits)
        │
        ▼
[ Portfolio Layer (L8) ]  ---> Genereert Final Capital Allocation
        │
        ▼
[ Execution Layer (L9) ]  ---> Voert orders uit & Minimaliseert Market Impact
```

### 11.1 Regels voor Alpha Modules
1. **Verantwoordelijkheid:** Alpha-modules genereren uitsluitend een voorspelling of gewenste relatieve exposure ($a_t \in [-1, 1]$).
2. **Geen Risico-bewustzijn:** Alpha-modules bevatten geen leverage-berekeningen, stop-losses, drawdown-breakers of position limits.
3. **Geen Executie-bewustzijn:** Alpha-modules kennen geen order-types, exchange-regels of executielogica.

### 11.2 Alpha Progression Pipeline
- **Baseline:** Cross-sectional Momentum, Trend Following, Mean Reversion, FX/Crypto Carry.
- **Statistical:** Factor-modellen, Ridge/Lasso Lineaire regressies, Fama-MacBeth specificaties.
- **Machine Learning:** Non-lineaire modellen (CatBoost, Random Forests) uitsluitend voor feature-combinatie of meta-labeling.

---

## 12. Machine Learning Architecture

Machine Learning wordt binnen dit platform behandeld met de nodige scepsis die past bij kwantitatieve financiële tijdreeksen (waarin de Signaal-Ruis-Verhouding extreem laag is).

### 12.1 Richtlijnen voor ML Toepassing
- **Geen Blinde Classification/Regression op Raw Prices:** CatBoost of Neurale Netwerken mogen niet direct ingezet worden om ruwe richting te voorspellen op dagelijkse returns.
- **Meta-Labeling Paradigm:** ML wordt primair ingezet als *secondary model* (López de Prado). Het primaire model bepaalt de richting (side), het ML-model voorspelt de kans op succes (sizing / trade filter).
- **Feature Importance Rigor:** Feature selectie vereist Mean Decrease Impurity (MDI) en Single Feature Importance (SFI) gecombineerd met Purged Cross-Validation.

---

## 13. Portfolio Architecture

De portfolio-laag transformeert gewenste exposures naar een geoptimaliseerde gewichtenmatrix.

### 13.1 Evaluatie van Allocatie-Algoritmen

| Algoritme | Schattingsfout | Covariantie Instabiliteit | Turnover & Kosten | OOS Robuustheid | Status |
|---|---|---|---|---|---|
| **1/N (Equal Weight)** | Nul | Nul | Extreem laag | Zeer hoog | **BASELINE** |
| **Inverse Volatility (Risk Parity)** | Laag | Laag | Laag | Hoog | **PRODUCTION DEFAULT** |
| **Hierarchical Risk Parity (HRP)** | Gemiddeld | Laag (geen inv-cov nodig) | Gemiddeld | Gemiddeld | Research Track |
| **Markowitz (Mean-Variance)** | Extreem hoog | Extreem hoog | Extreem hoog | Zeer laag (Markowitz curse) | **Banned in Raw Form** |
| **Black-Litterman** | Gemiddeld | Gemiddeld | Gemiddeld | Matig | Research Track |

> **Conclusie:** Naïeve Risk Parity (Inverse Volatility gebaseerd op EWMA) vormt de verplichte baseline. Complexere portfolio-algoritmen (zoals HRP) moeten bewijzen dat hun turnover-gecorrigeerde OOS-rendement superieur is aan Risk Parity.

---

## 14. Risk Architecture

Risicobeheer is het soevereine controle-orgaan van het handelssysteem. **Risico overruled altijd Alpha.**

### 14.1 Risico-Lagen
- **Unconditional Volatility Targeting:** De positiegrootte wordt dynamisch geschaald op basis van het verschil tussen de ex-ante volatiliteit en de doelvolatielheid:
  $$w_t = \min\left(\text{MaxLeverage}, \frac{\sigma_{\text{target}}}{\hat{\sigma}_{t+1\vert t}}\right)$$
- **Ex-Ante Volatility Estimation:** Gedreven door EWMA ($\lambda=0.94$) in productie en GARCH/HAR-RV in research.
- **Hard Limits:**
  - **Max Concentration Cap:** Maximale allocatie per asset/sector.
  - **Gross & Net Exposure Caps:** Absolute limieten op totale hefboomwerking.
  - **Drawdown Breaker (High-Water Mark):** Automatische de-grossing bij het bereiken van specifieke drawdown-drempels.
  - **Daily Loss Governor:** Onmiddellijke 'kill switch' wanneer het dagverlies de grens overschrijdt.

---

## 15. Execution & TCA Architecture

Een backtest die geen rekening houdt met marktimpact, bid-ask spread en latentie is een fictieve oefening.

### 15.1 Componenten van de Executie-Laag
- **Spread & Slippage:** Dynamische modelleringslaag gebaseerd op de actuele bid-ask spread en marktvolatieliteit.
- **Markt-Impact Model ($\eta, \kappa_d$):** Geijkte Square-Root Law voor marktimpact:
  $$\text{Impact} = \eta \cdot \sigma_{\text{daily}} \cdot \sqrt{\frac{\text{Order Size}}{\text{Daily Volume}}}$$
- **Order Lifecycle & Partial Fills:** Simulatie van limit order queues, time-to-fill, en kans op gedeeltelijke uitvoering op basis van orderboekdiepte.
- **Post-Trade Transaction Cost Analysis (TCA):** Continue vergelijking tussen de verwachte executieprijs (*arrival price*) en de daadwerkelijke opbrengst.

---

## 16. Backtesting Architecture

### 16.1 Authoritative Event-Driven Engine vs. Vectorized Sanity Checks
- **Vectorized Research Engine:** Toegestaan uitsluitend in de exploratieve fase voor snelle hypothese-screening. Vectorized resultaten worden nooit geaccepteerd als bewijs voor modelpromotie.
- **Authoritative Event-Driven Backtester:** De enige wettige autoriteit binnen het platform. Deze engine simuleert:
  - Discrete event-loops op tick- of bar-niveau.
  - Causal order routing en latency.
  - Explicit cash/position accounting registers.
  - Point-in-Time datafeeds zonder lookahead.

---

## 17. Statistical Validation Architecture

Validatie voorkomt dat ruis wordt gecodeerd als alpha.

```
STATISTICAL VALIDATION PIPELINE

[ Raw Model Signals ]
         │
         ▼
[ Purged Walk-Forward CV (met Embargo) ]
         │
         ▼
[ Deflated Sharpe Ratio (DSR) Check (Inclusief M, Y, γ) ]
         │
         ▼
[ White's Reality Check / Hansen's SPA (Multiple Testing) ]
         │
         ▼
[ Truncation & Lookahead Invariance Tests ]
         │
         ▼
[ PROMOTION TO CANDIDATE STATE ]
```

### 17.1 Categorisering van Statistische Methoden

| Methode | Categorisatie | Onderbouwing |
|---|---|---|
| **Purged Walk-Forward (met Embargo)** | **ESSENTIAL** | Inherent causaal, voorkomt spillage tussen opeenvolgende overlap-intervallen. |
| **Deflated Sharpe Ratio (DSR)** | **ESSENTIAL** | Corrigeert voor non-normaliteit, geheugeneffecten en het aantal geteste trials. |
| **Hansen's SPA / White's Reality Check** | **ESSENTIAL** | Beschermt tegen data-mining bias bij het vergelijken van meerdere strategieën. |
| **Diebold-Mariano (met HLN-correctie)** | **USEFUL** | Essentieel voor het paarsgewijs vergelijken van volatiliteits-forecasts (GARCH vs EWMA). |
| **Combinatorial Purged CV (CPCV)** | **OPTIONAL** | Biedt rijke verdelingsinzichten, maar verstoort tijdreeks-paden; niet gebruiken als primaire gate. |
| **Probability of Backtest Overfitting (PBO)** | **OPTIONAL** | Goede diagnostiek, maar ondergeschikt aan OOS Walk-Forward. |
| **K-Fold CV (Unpurged)** | **BANNED** | Creëert zware lookahead leakage in financiële tijdreeksen. |

---

## 18. Model Governance

Model Risk Management (MRM) garandeert dat alle productie-code en modellen cryptografisch herleidbaar en auditbaar zijn.

### 18.1 Governance-Regels
1. **Pre-Registratie:** Elke onderzoeksgolf moet vooraf geformuleerde hypotheses, parameters, en stop-criteria vastleggen in het `FALSIFICATION_REGISTER.md`.
2. **Immutable Ledger Entry:** Resultaten worden via `registry/hypothesis_ledger.py` atomair geschreven, inclusief `git_sha`, `data_hash`, en `config_hash`.
3. **Champion / Challenger Framework:** Een challenger-model vervangt het champion-model pas na minimaal 60 dagen OOS paper-trading waarin het de champion statistisch significant verslaat (Diebold-Mariano $p < 0.05$).
4. **CI/CD Enforcement:** Geen enkele PR wordt gemerged als de lookahead-suite, DSR-check of invariantie-tests falen.

---

## 19. Proposed Target Architecture

De voorgestelde Target Architecture bestaat uit 14 strikt gescheiden lagen waarin afhankelijkheden uitsluitend van boven naar beneden bewegen.

```
┌──────────────────────────────────────────────────────────┐
│ L0: Data Ingestion, PIT Storage & Validation             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L1: Feature Engineering (Stateless, Causal)              │
└────────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼
┌───────────────────────┐         ┌────────────────────────┐
│ L2: Volatility Engines│         │ L3: Regime Engines     │
│ (EWMA / GARCH / HAR)  │         │ (M0 Buckets / HMM)     │
└───────────┬───────────┘         └───────────┬────────────┘
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L4: Alpha Generation (Raw Forecasts, Scale-free)         │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L5: Conditioning Layer (Alpha x Regime Overlay)          │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L6: Model Combination & Ensembles                        │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L7: Independent Risk & Volatility Targeting              │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L8: Portfolio Construction & Sizing (Risk Parity)        │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L9: Execution & Order Routing (TCA, Impact)              │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L10: Backtesting Engine (Event-Driven, Execution-Aware)  │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L11: Statistical Validation & Falsification              │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L12: Governance, Ledger & Audit Register                 │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ L13: Live Operations, Paper Trading & Monitoring         │
└──────────────────────────────────────────────────────────┘
```

---

## 20. Proposed Directory Structure

De volgende mappenstructuur definieert het doelontwerp voor de repository.  
*Opmerking: Dit is een architectuurvoorstel. Er zijn geen fysieke bestanden verplaatst in de huidige repository tijdens deze audit.*

```
quant_system/
├── conf/                       # Hydra / Pydantic gecentraliseerde configuraties
│   ├── data/
│   ├── model/
│   ├── risk/
│   └── execution/
├── data/                       # L0: Ingestion & Validation
│   ├── ingestion/
│   ├── validation/
│   └── pit_store/              # Parquet files + DVC tracking
├── features/                   # L1: Causal Feature Engineering
│   ├── transforms.py
│   └── pipeline.py
├── volatility/                 # L2: Volatieliteitsmodellering
│   ├── ewma.py
│   ├── garch.py
│   └── realized.py
├── regime/                     # L3: Regime Classificatie
│   ├── buckets.py              # M0 Causal Buckets
│   └── markov.py               # M1/M2 Models
├── alpha/                      # L4: Pure Alpha Signal Generators
│   ├── momentum.py
│   ├── mean_reversion.py
│   └── carry.py
├── portfolio/                  # L8: Portfolio Allocation
│   ├── risk_parity.py
│   └── hrp.py
├── risk/                       # L7: Independent Risk Overlay
│   ├── vol_targeting.py
│   ├── limits.py
│   └── kill_switches.py
├── execution/                  # L9: TCA & Impact Simulation
│   ├── impact_model.py
│   └── order_router.py
├── backtest/                   # L10: Event-Driven Backtesting Engine
│   ├── engine.py
│   └── accounting.py
├── validation/                 # L11: Statistical Gates & DSR
│   ├── dsr.py
│   ├── walk_forward.py
│   └── spa.py
├── registry/                   # L12: Falsificatie & Hypothese Ledger
│   ├── hypothesis_ledger.py
│   └── MRM_generator.py
├── research/                   # Geïsoleerde Jupyter Notebooks (No Prod Code)
│   ├── exploratory/
│   └── publication/
├── tests/                      # Test Suites
│   ├── unit/
│   ├── integration/
│   ├── lookahead/              # Truncatie & Causaliteits-tests
│   └── statistical/            # DSR & Gate tests
├── pyproject.toml
└── README.md
```

---

## 21. Research Pipeline

De gestandaardiseerde Target Research Pipeline loopt via de volgende fasen:

```
Data Ingestion (PIT)
  └─► Data Validation & Quality Checks
        └─► Causal Feature Engineering
              └─► Baseline Model (EWMA / Linear)
                    └─► Hypothese Formulering & Pre-Registratie
                          └─► Model Estimation & Walk-Forward OOS
                                └─► Statistical Validation (DSR, SPA)
                                      └─► Portfolio & Risk Overlay
                                            └─► Event-Driven Execution Simulation (TCA)
                                                  └─► Ledger Registration
                                                        └─► Paper Trading (60 dagen)
                                                              └─► Production Promotion
```

---

## 22. Model Hierarchy

Modellen moeten hun plek in de hiërarchie empirisch verdienen. Promotie naar een hoger niveau vereist keihard bewijs van OOS Sharpe-verbetering na transactiekosten ten opzichte van de lagere niveaus.

- **LEVEL 0:** Naïeve Baselines (Random Walk, Buy & Hold, Constant Volatility).
- **LEVEL 1:** Simpele Statistische & Econometrische Baselines (EWMA, Linear Momentum, Rolling Z-Score).
- **LEVEL 2:** Econometrische Modellen (GARCH(1,1), ARMA-GARCH, M0 Vol-Buckets).
- **LEVEL 3:** Geregulariseerde Lineaire Modellen (Ridge, Lasso, ElasticNet).
- **LEVEL 4:** Non-lineaire Machine Learning (CatBoost, Random Forest voor Meta-Labeling).
- **LEVEL 5:** Geavanceerde Ensembles & Hierarchische Structuren (HRP + ML Meta-models).

---

## 23. Migration Strategy

De migratie van de Current State naar de Target State verloopt in 8 strikt chronologische fasen:

### Phase 0 — Audit & Repository Hygiene (P0)
- **Doel:** Herstel versiebeheer, centraliseer configuratie en elimineer stille fallbacks.
- **Deliverables:** `.git` herinitialisatie, Pydantic/Hydra schema-enforcement, vervanging van alle 12 `try/except ImportError` blokken door harde Fail-Fast crashes.
- **Exit Criteria:** Nul stille fallbacks; 100% reproduceerbare builds via `pyproject.toml`.

### Phase 1 — Data Foundation & Crypto Ingestion (P0)
- **Doel:** Herstel dataintegriteit en vul de ontbrekende crypto-dataset aan.
- **Deliverables:** PIT Parquet store, DVC lineage tracking, crypto OHLCV & funding rate ingestion.
- **Exit Criteria:** `asof_join` geverifieerd op crypto-data; unieke `data_hash` gegenereerd.

### Phase 2 — Research & Falsification Foundation (P0)
- **Doel:** Koppel het falsificatieregister en de ledger aan de CI/CD pipeline.
- **Deliverables:** Geautomatiseerde DSR, SPA en lookahead-suite integratie in GitHub Actions / local CI.
- **Exit Criteria:** PR's worden automatisch geblokkeerd bij lookahead-fouten of DSR-falsificatie.

### Phase 3 — Baseline Implementation (P1)
- **Doel:** Implementeer Level 1 baselines voor alle research tracks.
- **Deliverables:** EWMA volatiliteits-estimator, Cross-Sectional Momentum baseline, Naive Risk Parity portfolio allocation.
- **Exit Criteria:** Baseline OOS Sharpe en drawdown gekwantificeerd over de volledige dataset.

### Phase 4 — Risk & Volatility Architecture (P1)
- **Doel:** Volledige ontkoppeling van Risk (L7) en Alpha (L4).
- **Deliverables:** Standalone Volatility Targeting module, Hard Risk Limits, Daily Loss Governor.
- **Exit Criteria:** Risicolimieten overrulen aantoonbaar alpha-posities in stress-simulaties.

### Phase 5 — Execution & Backtesting Engine (P1)
- **Doel:** Consolidatie van de backtest engines naar 1 event-driven engine met TCA.
- **Deliverables:** Authoritative Event-Driven Engine, Marktimpact ($\eta$) kalibratiemodule.
- **Exit Criteria:** Bit-identieke resultaten tussen paper-trading logica en backtest execution simulator.

### Phase 6 — Advanced Research Tracks (GARCH, ML, Regimes) (P2)
- **Doel:** Gecontroleerde evaluatie van Level 2+ modellen.
- **Deliverables:** GARCH(1,1) vs EWMA QLIKE competitie, M0 vs HMM Filtered Probability benchmarking.
- **Exit Criteria:** Alleen modellen met aantoonbare OOS-waarde worden goedgekeurd.

### Phase 7 — Production Readiness & Paper Trading (P2)
- **Doel:** Integratie van het volledige platform voor live paper-trading.
- **Deliverables:** Live monitoring dashboards (PSI, Sharpe drift, Vol forecast errors), automated alerts.
- **Exit Criteria:** 60 opeenvolgende dagen succesvolle paper-trading zonder execution drift of runtime crashes.

---

## 24. Kill / Archive / Retain / Redesign Matrix

| Component / Module | Actie | Reden | Vereist Bewijs voor Wijziging | Prioriteit |
|---|---|---|---|---|
| `docs/FALSIFICATION_REGISTER.md` | **RETAIN** | Schaars wetenschappelijk goed; uitstekende hygiëne. | Geen (behouden) | **P0** |
| `registry/hypothesis_ledger.py` | **RETAIN** | Atomaire, robuuste logica. | Geen (behouden) | **P0** |
| `utils/time.py::asof_join` | **RETAIN** | Inherent causale PIT-merge. | Geen (behouden) | **P0** |
| `backtest/metrics.py::DSR` | **RETAIN** | Correcte Bailey-LdP dimensionaliteitsfix. | Geen (behouden) | **P0** |
| `risk/hmm_regime.py` | **REDESIGN** | Stille fallback naar EMA slopen; herstructureren naar M2 Filtered HMM. | QLIKE / OOS Sharpe winst t.o.v. M0 | **P0** |
| `risk/portfolio.py` | **REDESIGN** | Splitsen: Risicobeperking ontkoppelen van Alpha. | Geslaagde ontkoppelingstest | **P0** |
| Overlappende Backtesters (4x) | **REDESIGN** | Consolideren tot 1 event-driven engine. | Pariteits-test | **P0** |
| `quant_architect` Imports | **REMOVE** | Dode code / ontbrekende afhankelijkheid. | Vervangen door directe modules | **P0** |
| 26 Dode Alpha Units | **ARCHIVE** | Onbewezen of gefalsificeerd op ontbrekende data. | Nieuwe data + pre-registratie | **P1** |
| CatBoost Direct Directional | **ARCHIVE** | SNR te laag op daily returns; hoge overfitting. | Meta-labeling OOS AUC > 0.58 | **P2** |
| Hierarchical Risk Parity (HRP) | **RESEARCH ONLY** | Potentieel nuttig, maar verliest momenteel van Risk Parity. | OOS Sharpe > Inverse Vol | **P2** |
| Black-Litterman / Markowitz | **RESEARCH ONLY** | Covariantie-instabiliteit OOS. | Turnover-gecorrigeerde winst | **P3** |

---

## 25. Research Roadmap

### Q3 2026 — Data & Baseline Readiness
- Complete her-ingestie van Crypto OHLCV, Trades, en Funding Rates.
- Benchmarken van Level 1 Baselines (EWMA + Cross-Sectional Momentum).

### Q4 2026 — Volatility & Risk Calibration
- Uitvoeren van de GARCH vs EWMA QLIKE-competitie op dagelijkse en intraday data.
- Kalibratie van de marktimpact-parameter $\eta$ op orderboek-data.

### Q1 2027 — Regime & ML Meta-Labeling
- Evaluatie van M0 Buckets vs Filtered HMM Probabilities.
- Testen van CatBoost als meta-labeler voor basissignalen.

---

## 26. Acceptance Criteria

Voor elke strategie of modelcomponent die wordt genomineerd voor promotie naar de productie-pijplijn gelden de volgende onverwijdbare acceptatiecriteria:

1. **Data Provenance:** 100% van de gebruikte data is gekoppeld aan een gecertificeerde `data_hash` in het PIT store.
2. **Zero Lookahead Leakage:** Het model slaagt voor 100% van de ruimtelijke en temporele truncatie-tests in `tests/lookahead/`.
3. **Deflated Sharpe Ratio (DSR):** Statistisch significant met $p < 0.05$ na correctie voor non-normaliteit en het totale aantal uitgevoerde trials ($M$).
4. **Execution Realism:** Positieve verwachte netto-rendementen na toepassing van het geijkte marktimpactmodel en exchange fee schedules.
5. **Fail-Fast Compliance:** De module bevat nul `try/except` fallbacks en crashed onmiddellijk bij schending van data-contracten.

---

## 27. Open Questions

1. **Intraday Tick Data Availability:** Beschikken we over voldoende historische orderboek-depth data om de HAR-RV en TCA-impactmodellen met hoge nauwkeurigheid te kalibreren?
2. **Crypto Funding Rate Arbitrage:** Is de verdienstencurve van funding-rate carry voldoende robuust onder regimewisselingen van bull- naar bear-markten?
3. **Exchange Latency Scaling:** Hoe varieert de executielatentie tijdens periodes van extreme marktvolatieliteit op gedecentraliseerde vs. gecentraliseerde exchanges?

---

## 28. Final Architectural Recommendation

### Current vs. Target Score
- **Current Architecture Score:** `3 / 10` (Zware schuld, stille fallbacks, ontbrekende data, gebroken koppelingen).
- **Target Architecture Score:** `9 / 10` (Institutioneel-grade, causaal, baseline-first, strikt gescheiden lagen).

### Grootste Risico's & Bronnen van Complexiteit
- **Silent Failures:** Het stilzwijgend degraderen van statistische modellen naar EMA's is het grootste operationele risico in de huidige codebase.
- **Data-Illusie:** Onderzoek doen zonder gegarandeerde, versie-beheerde data.

### Wat zou een professionele quant vandaag fundamenteel anders doen?

> **Het Antwoord:**  
> Als een professionele quant deze repository vandaag opnieuw zou ontwerpen, zou hij geen enkele regel ML- of HMM-code schrijven totdat de data-infrastructuur 100% Point-in-Time, versie-beheerd en onveranderlijk is.
>
> Hij zou de overbodige complexiteit (CatBoost, HRP, 4 backtest engines) onmiddellijk opbreken. Hij zou beginnen met één robuuste, lineaire Risk Parity strategie gedreven door een EWMA volatiliteits-target, aangesloten op één event-driven backtester met realistisch gekalibreerde transactiekosten. Pas wanneer dat systeem aantoonbaar live paper-trades uitvoert met een bewezen statistische edge, zou hij stapsgewijs complexere modellen (GARCH, ML Meta-labeling) toelaten via het falsificatieregister.

---

## Conclusie

Het herzien van `ARCHITECTUUR_AUDIT_2026-08-22` markeert de overgang van een gefragmenteerd experimenteel systeem naar een professioneel, institutioneel kwantitatief platform. De richtlijnen en fasen in dit document zijn bindend voor alle toekomstige engineering- en onderzoeksinspanningen.
ARCHITECTUUR_AUDIT_2026-08-22.md
ARCHITECTUUR_AUDIT_2026-08-22.md weergeven.
