# Phase 9 — bereikbaarheidskaart en false-positive-audit (stap 6)

> **Gemeten 2026-09-04 op commit `97382cd`** met `scripts/reachability_map.py`,
> bewezen door `tests/unit/test_reachability_map.py` (17 passed).
>
> ```bash
> D:/venv/tradebot/Scripts/python.exe scripts/reachability_map.py \
>     --coverage coverage.json \
>     --json reports/phase9_reachability.json \
>     --markdown reports/phase9_reachability.md
> ```

De kaart zelf staat in `reports/phase9_reachability.md` (299 rijen). Dit
document doet twee dingen die de kaart niet kan: het verklaart waarom de meting
afwijkt van de nulmeting, en het put de drie false-positive-klassen uit die de
opdracht voorschrijft.

---

## 1. De meting wijkt af van de nulmeting

Stap 6 schrijft voor: *"Wijkt jouw uitkomst af, dan zoek je uit waarom vóór je
verdergaat."* Zij wijkt af, en fors.

| Klasse | Nulmeting | Gemeten | Δ modules | Δ LOC |
|---|---|---|---:|---:|
| **A** authoritative | 123 / 41.763 | **193 / 53.075** | +70 | +11.312 |
| **B** operationeel | 81 / 17.163 | **67 / 12.763** | −14 | −4.400 |
| **C** research | 8 / 1.108 | **8 / 1.108** | 0 | 0 |
| **D** test-only | 48 / 6.932 | **20 / 3.036** | −28 | −3.896 |
| **E** onbereikbaar | 39 / 4.521 | **11 / 1.505** | −28 | −3.016 |
| **totaal** | **299 / 71.487** | **299 / 71.487** | 0 | 0 |

Het totaal is identiek en klasse C komt tot op de LOC exact overeen. Het is dus
**dezelfde boom**; wat verschilt is de resolver. Twee edges verklaren het gat,
en beide zijn echte imports:

1. **`from pkg import submodule`.** `from tradebot.risk import var` importeert de
   module `tradebot.risk.var` zonder dat die naam ooit als dotted pad in de
   tekst staat. Een resolver die alleen `node.module` volgt, mist hem.
2. **Bovenliggende packages.** `import tradebot.risk.var` voert ook
   `tradebot/__init__.py` en `tradebot/risk/__init__.py` uit. Een `__init__.py`
   met re-exports trekt daarmee zijn hele pakket mee.

Het tweede punt is beslissend voor `risk/`. `src/tradebot/risk/__init__.py`
re-exporteert regel voor regel:

```python
from .beta_hedge import compute_btc_hedge_size, compute_rolling_betas
from .drawdown import BreakerState, DrawdownBreaker, DrawdownConfig, ...
from .factor_risk import FactorExposure, FactorRiskModel, compute_factor_risk
from .hmm_regime import HMMRegimeDetector, Regime
from .kelly import gap_risk_kelly_size, kelly_fraction, meta_label_kelly
from .liquidity_risk import LiquidityRiskAssessment, assess_liquidity_risk, ...
from .position_limits import PositionLimits, PositionViolation, ...
from .var import historical_cvar, historical_var, rolling_cvar, rolling_var, ...
```

Wie `tradebot.risk` importeert, importeert alle acht.

### Gevolg 1 — de `risk/`-spanning uit de opdracht bestaat niet

De opdracht schrijft: *"Let in het bijzonder op dat de helft van `risk/` in
klasse D valt"* en noemt acht modules, met de opdracht die spanning te benoemen
omdat `PROJECT_STATE.md` §2 de risicolaag *"soeverein in de BACKTEST"* noemt.

Alle acht zijn **klasse A**, bereikt vanuit `apps/run_data_adequacy.py` — een
DVC-stage-seed:

| Module | LOC | Nulmeting | Gemeten | Dekking |
|---|---:|:---:|:---:|---:|
| `risk/var.py` | 350 | D | **A** | 64,6 % |
| `risk/hmm_regime.py` | 270 | D | **A** | 98,2 % |
| `risk/factor_risk.py` | 179 | D | **A** | 95,5 % |
| `risk/kelly.py` | 170 | D | **A** | 96,9 % |
| `risk/drawdown.py` | 149 | D | **A** | 88,1 % |
| `risk/liquidity_risk.py` | 148 | D | **A** | 97,5 % |
| `risk/position_limits.py` | 105 | D | **A** | 95,3 % |
| `risk/beta_hedge.py` | 104 | D | **A** | 22,8 % |

Er is geen spanning tussen "soeverein" en "alleen door tests bereikt": zij zitten
in de authoritative keten, en het pakket draagt 86,8 % dekking. **Er is niets te
benoemen en niets op te lossen.**

Wat de meting wél zichtbaar maakt is een ander gat, en het is scherper dan het
gat dat de opdracht verwachtte: **twee van de acht staan onder de 70 %-drempel.**
`beta_hedge.py` op 22,8 % berekent BTC-hedgeomvang en rollende bèta's in de
authoritative keten; `var.py` op 64,6 % levert de VaR/CVaR-route. Zij gaan naar
de stap-11-lijst, niet naar een verwijderregister.

### Gevolg 2 — `labeling/meta.py` had niet verwijderd mogen worden

Stap 8 van de opdracht draagt op *"bijzondere aandacht voor de twee grootste"*
klasse-E-modules en noemt `labeling/meta.py` (1.181 LOC) als de grootste.

`labeling/meta.py` is **klasse A**, bereikt vanuit `apps/build_features.py`,
seed van de DVC-stage `build_features`. Verwijderen zou de authoritative DAG
hebben gebroken. De keten is:

```
apps/build_features.py -> tradebot.labeling -> labeling/__init__.py -> .meta
```

Dit is precies de faalklasse die de opdracht zelf in haar openingsregel
beschrijft: *"Een verwijderde module die nog een contract droeg, is een stille
degradatie met een schoner ogende `git ls-files`."*

---

## 2. De drie false-positive-klassen, uitgeput

### 2.1 `__init__.py`-bestanden — 15 vals alarm, alle 15 opgelost

De nulmeting merkt vijftien `__init__.py`-bestanden aan als onbereikbaar en
waarschuwt terecht dat `backtest/__init__.py` (152 LOC) en `train/__init__.py`
(110 LOC) *"te groot zijn om een leeg pakketmarkering te zijn"*.

Met bovenliggende-package-resolutie is **geen enkele van de dertig
`__init__.py`-bestanden in `src/` onbereikbaar**: 22 zijn A, 7 zijn B, en één is
D. Die ene is `tca/__init__.py`, en dat is geen false positive maar een juiste
uitkomst — het hele `tca/`-pakket is test-only, bereikt vanuit
`tests/integration/test_engine_parity.py`.

### 2.2 Dynamische registratie — leeg

```bash
grep -rn "importlib|__subclasses__|entry_points|pkgutil" src/
grep -rn "_target_" conf/
```

* **`importlib`**: drie treffers, geen ervan registreert een `tradebot`-module.
  `utils/failfast.py` gebruikt `importlib.util.find_spec` om de aanwezigheid van
  DERDE-partijpakketten te toetsen zonder `try/except ImportError`;
  `tradebot/__init__.py` gebruikt `importlib.metadata.version`.
* **`__subclasses__`**: nul treffers in `src/`.
* **Hydra `_target_`**: **nul** in `conf/`. De twee tekstuele treffers zijn
  `profit_target_sigma: 2.0` in `conf/model/labeling.yaml` en een commentaar over
  `_build_target_store()` in `conf/symbols/AVAXUSDT.yaml`. De scanner behandelt
  `_target_` desondanks correct — bewezen in de spec — zodat de poort niet
  stukgaat op de dag dat iemand er één toevoegt.

### 2.3 Aanroep vanaf de commandoregel — leeg

```bash
grep -rn "python -m tradebot\.|-m tradebot\." docs/ .github/ Makefile dvc.yaml
```

**Nul treffers.** Geen enkele module in `src/` wordt als
`python -m tradebot.x.y` aangeroepen. Alle tien DVC-`cmd`-regels roepen een app
aan, niet een pakketmodule:

```
python -m apps.data_sync    python -m apps.build_features
python -m apps.tune_hparams python -m apps.train_cpcv
python apps/run_phase5_baseline.py      python apps/run_data_adequacy.py
python apps/run_econometric_diagnostics.py   python apps/run_vol_competition.py
python apps/run_regime_benchmark.py     python apps/run_meta_labeling.py
```

Dat zijn exact de tien seeds die de opdracht noemt.

---

## 3. Een vierde false-positive-klasse die de opdracht niet noemt

De drie voorgeschreven klassen zijn uitgeput, maar de audit vond een vierde die
**twee van de elf klasse-E-modules redt**. De scanner meet importbereikbaarheid;
een test kan een bestand ook noemen zónder het te importeren.

```python
# tests/unit/test_risk_alpha_decoupling.py:163
def test_the_entangled_portfolio_module_no_longer_lives_in_risk(self) -> None:
    """Exit-criterium 3."""
    assert not (SRC / "risk" / "portfolio.py").exists()
    assert (SRC / "portfolio" / "legacy_sizing.py").exists()
    assert (SRC / "portfolio" / "covariance.py").exists()
```

Dit is het **Phase 4 exit-criterium 3**-bewijs: de verstrengelde module is uit
`risk/` verdwenen en in `portfolio/` beland. De test importeert niets — hij
toetst het bestaan van een pad. Voor de importgraaf zijn beide modules
onbereikbaar; voor de repository dragen zij een contract.

| Module | LOC | Scanner | Werkelijk verdict |
|---|---:|:---:|---|
| `portfolio/legacy_sizing.py` | 824 | E | **BEHOUDEN** — bestaansassertie + DI-10 + ratchetpost in `check_hardcoded_params.py` |
| `portfolio/covariance.py` | 194 | E | **BEHOUDEN** — bestaansassertie + DI-10 |

`legacy_sizing.py` draagt bovendien een regel in
`scripts/check_hardcoded_params.py`: `"portfolio/legacy_sizing.py": (10, "DI-10
legacy, research-only")`. Verwijderen zou die ratchetpost betekenisloos maken.

Volledige scan op deze klasse — elke bestaandsassertie op een `src/`-pad in
`tests/`:

```
tests/integration/test_sovereign_wiring.py:413   _AUTHORITATIVE_PATH (5 modules, alle A)
tests/unit/test_risk_alpha_decoupling.py:161-162 alpha/factor_alpha.py  (A)
tests/unit/test_risk_alpha_decoupling.py:166-168 portfolio/legacy_sizing.py, covariance.py
```

Meer zijn er niet. De klasse is uitgeput.

> **De les.** De opdracht draagt op de scanner te vertrouwen voor de klasse en
> de false positives handmatig na te lopen. Die volgorde is juist, en dit is het
> geval waarin zij een verwijdering van 1.018 LOC tegenhoudt die volgens het
> meetinstrument door mocht.

---

## 4. Wat er na de audit overblijft om te verwijderen

Van de elf klasse-E-modules vallen er twee af op een contract. De resterende
negen zijn **487 LOC**, en geen van hen heeft een afnemer: nul treffers in
`artefacts/governance/`, nul in `docs/ARCHITECTURAL_DECISIONS.md`, nul in
`docs/DEFERRED_ISSUES.md`, nul in `docs/PROJECT_STATE.md`.

| Module | LOC | Waarom het weg mag |
|---|---:|---|
| `alpha/decay_tracker.py` | 140 | Wave 18 IC-decay-tracker; nooit aangesloten, niet in `alpha/__init__.py` |
| `features/macro.py` | 110 | macro-featuretransformaties; er is geen macro-featurepad in de keten |
| `data/sources/cboe.py` | 61 | VIX-TS-bron; het pakketcontract eist een lookahead-test in `tests/lookahead/` (G6) en die bestaat niet |
| `logging_config.py` | 61 | docstring zegt *"Call setup_logging() once at application entry (apps/*.py)"* — **geen enkele app doet dat**; nul aanroepen in de hele repo |
| `alpha/eq_strev_resid.py` | 55 | Wave 24; `WAVE_LOG.md` en `EXPANSION_RESEARCH` boeken hem als **ARCHIVED → F16** |
| `types.py` | 25 | domein-NewTypes en Protocols; dubbele waarheid naast `schemas/`, door niets geïmporteerd |
| `bars/tick.py` | 16 | `raise NotImplementedError("tick_bars not yet implemented")`; niet in `bars/__init__.py` |
| `bars/volume.py` | 16 | idem |
| `_version.py` | 3 | docstring zegt *"Single source of truth for the package version"* — **onwaar**: `tradebot/__init__.py` leest de versie via `importlib.metadata` uit `pyproject.toml` |

Twee van de negen zijn een **dubbele waarheid** in de precieze zin die deze fase
bedoelt: `_version.py` en `types.py` beweren de bron te zijn van iets waarvan de
bron elders ligt.

**De opruimbare oppervlakte is daarmee 9 modules / 487 LOC, niet 39 / 4.521.**
Dat is geen tegenvaller maar de uitkomst van de meting: 28 van de 39 modules die
de nulmeting onbereikbaar noemde, zijn bereikbaar, en 2 van de resterende 11
dragen een contract.
