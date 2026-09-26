# Wekelijkse meta-label-strategie — Plan 1: onderzoek tot en met het oordeel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Een reproduceerbare, causaal geteste onderzoekspijplijn die op de gecertificeerde dagdata van zes perps meet of een CUSUM-doorbraak met een licht ML-filter, 1:1-barrières en Kelly-sizing na kosten een edge heeft — en die daar vooraf vastgelegde poorten over laat beslissen.

**Architecture:** Bestaande bouwstenen doen het zware werk: `labeling/cusum.py` (events), `labeling/vol_barriers.py` (labels), `features/fracdiff.py`, `volatility/ewma.py`, `alpha/kalman_ou.py`, `train/meta_label.py` (dataset, purge, walk-forward), `cv/cpcv.py`, `backtest/pbo.py`, `validation/inference.py` + `validation/dsr.py`, `registry/` (ledger, preregistratie) en `risk/engine.py` (het mandaat). Nieuwe code is lijm of een aantoonbaar ontbrekend stuk: richting uit CUSUM, fillprijzen bij de barrière, een vaste featureset, twee lichte modellen met Platt-kalibratie, binaire Kelly-rekenkunde, een klein tradeboek met exchange-side exits, en het oordeel.

**Tech Stack:** Python 3.13 (`D:\venv\tradebot`), pandas, numpy, scipy 1.17, scikit-learn 1.9, pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-weekly-meta-label-design.md` (lees §17 nadat Taak 1 hem heeft toegevoegd).

## Global Constraints

- Python: `D:/venv/tradebot/Scripts/python.exe`. Tests altijd met `-p no:randomly -q`.
- Na elke taak: `ruff check src/ tests/` schoon en `pytest -m "not slow" -p no:randomly -q` zonder nieuwe failures (nulmeting 2026-09-26: 2706 passed, 6 skipped, 0 failed).
- Geen hyperparameter-zoektocht, geen Optuna. Elke instelling staat in `conf/model/weekly_meta.yaml`.
- Causaliteit: een waarde op bar `t` gebruikt alleen data t/m de close van `t`; funding en open interest daarbovenop één bar lag. Elke nieuwe feature- of signaalfunctie krijgt een test die de invoer na `t` wijzigt en rij `t` vergelijkt.
- Instap volgt de bestaande `shift(2)`-conventie van `vol_barriers`: event op de close van `t`, fill op de close van `t+1`, barrières vanaf `t+2`.
- Kosten: één definitie. Round trip = `2 × (taker_fee_bps + assumed_half_spread_bps) / 1e4` uit `conf/execution/fees.yaml` (= 13 bps), plus stop-slippage op stops, plus funding en impact in het boek.
- `conf/risk/default.yaml`-waarden worden NIET gewijzigd (policy `9961e1613bc907a5`).
- `data/pit_store/` wordt niet aangeraakt. Geen netwerktoegang in dit plan.
- De holdout wordt ten hoogste één keer gelezen, alleen in Taak 14, alleen na `PASS`.
- Nieuwe bestanden: LF-regeleinden. Bestaande bestanden: behoud hun regeleinde (controleer met `file <pad>` vóór en na).
- Commit na elke taak, bericht in het Engels, afgesloten met `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Niet pushen.

---

## Wijzigingen ten opzichte van de spec (vastgelegd in Taak 1 als spec §17)

Bij het uitwerken bleek de spec op zeven punten niet uitvoerbaar zoals geschreven. Het plan volgt deze correcties:

1. **Data zoals gecertificeerd, niet bijgewerkt.** De PIT-store is append-only met jaarpartities (`PitStore.write` crasht bij een afwijkende bestaande partitie), dus dagen toevoegen aan 2026 vergt een wijziging aan het apparaat. Plan 1 gebruikt de store zoals hij is (laatste bar: asof 2026-08-23). Bijwerken hoort bij Plan 2 (live).
2. **Holdout = de laatste 60 bars:** split `2026-06-24T00:00:00+00:00`. Het huidige slot is nooit gelezen (`reads: []`) en wordt opnieuw bevroren.
3. **M via de lege ledger, niet via `freeze_reset`.** Na een reset eist `active_trial_count` dat de ledger niet meer groeit; een schone ledger telt zelf. Het programma boekt zijn 4 trials als eerste entry en bevriest `M = 4` in de preregistratie; de DSR leest hem met `frozen_trial_count`.
4. **GARCH valt uit de features.** Een GARCH-fit vraagt per symbool 250–500 bars burn-in (`conf/model/adequacy.yaml`); dat kost SOL en AVAX hun eerste 1–1,5 jaar. EWMA (RiskMetrics) blijft het volatiliteitsmodel voor barrières, CUSUM-drempels, sizing en risicolaag. Het vol-regime komt uit de Yang-Zhang-ratio en vol-of-vol.
5. **Kalman rollend opnieuw gefit.** `KalmanOUMeanReversion` houdt μ bewust statisch na `fit()`; zonder rollende refit trekt de z-score naar een oud prijsniveau. De OU-halfwaardetijd komt uit dezelfde fit (`current_halflife`).
6. **Exits op het barrièreniveau, in een klein tradeboek.** `backtest/engine.py` vult alleen op barprijzen en kan een stop bij de exchange niet uitdrukken. `backtest/barrier_book.py` gebruikt dezelfde `RiskEngine.decide`, dezelfde kostenparameters en `square_root_impact`, en wordt bewezen met een boekhoudidentiteit en een causaliteitstest.
7. **Posterior met uniforme prior:** `Beta(1 + p̂·n, 1 + (1 − p̂)·n)`, zodat hij ook bij `n = 0` gedefinieerd is (dan geen informatie, dus geen inzet boven break-even).

**Opsplitsing:** dit is Plan 1 (onderzoek tot en met oordeel en holdout-rooktest). Plan 2 — data bijwerken, papertrading, SPRT- en Beta-bewaking, orders met stops bij de exchange — wordt pas geschreven na een `PASS`.

---

## File Structure

| Pad | Verantwoordelijkheid | Taak |
|---|---|---|
| `conf/model/weekly_meta.yaml` | alle instellingen, vast | 1 |
| `src/tradebot/schemas/weekly_meta.py` | `WeeklyMetaConfig` + loader | 1 |
| `src/tradebot/labeling/cusum.py` (wijzig) | publieke `directional_cusum_filter` | 2 |
| `src/tradebot/labeling/breakout.py` | richting per bar, k-kalibratie op frequentie | 2 |
| `src/tradebot/labeling/barrier_fills.py` | fillrendementen bij de barrière, kostenbewust doel | 3 |
| `src/tradebot/data/weekly_market.py` | gecertificeerde dagmarkt op één raster | 4 |
| `tests/weekly_fixtures.py` | synthetische markt voor tests | 4 |
| `src/tradebot/features/weekly_set.py` | de vaste featureset | 5 |
| `src/tradebot/train/weekly_dataset.py` | events + labels + features → `MetaLabelDataset` | 6 |
| `src/tradebot/train/meta_label.py` (wijzig) | generieke `walk_forward_fit_predict`, `extras` op `FoldPredictions` | 7 |
| `src/tradebot/train/light_models.py` | LR, RF, ensemble, Platt op gepurgede binnensplit | 7 |
| `src/tradebot/risk/binary_kelly.py` | barrièretheorie, break-even, Kelly, posterior, correlatie, drawdown | 8 |
| `src/tradebot/backtest/barrier_book.py` | tradeboek met exchange-side exits en `RiskEngine` | 9 |
| `src/tradebot/cv/event_space.py` | exitposities in eventruimte voor CPCV | 10 |
| `src/tradebot/validation/weekly_verdict.py` | oordeel uit de bevroren stop-criteria | 10 |
| `src/tradebot/validation/weekly_campaign.py` | de meting op de ontwikkelsample | 11 |
| `conf/research/preregistration_weekly_meta.yaml` | de preregistratie | 12 |
| `src/tradebot/registry/weekly_programme.py` | holdout herbevriezen, trials boeken, preregistratie bevriezen | 12 |
| `src/tradebot/validation/weekly_holdout.py` | de eenmalige holdout-rooktest | 14 |

---

### Task 1: Configuratie, schema en spec-wijzigingen

**Files:**
- Create: `conf/model/weekly_meta.yaml`
- Create: `src/tradebot/schemas/weekly_meta.py`
- Modify: `docs/superpowers/specs/2026-09-26-weekly-meta-label-design.md` (voeg §17 toe)
- Test: `tests/unit/test_weekly_meta_config.py`

**Interfaces:**
- Produces: `WeeklyMetaConfig` (frozen pydantic model) en `weekly_meta_config(path=WEEKLY_META_CONFIG_PATH) -> WeeklyMetaConfig`. Velden zoals in de YAML hieronder.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_weekly_meta_config.py
"""Het contract van de wekelijkse strategie: laadt, is 1:1 per constructie, en weigert tegenstrijdigheden."""
from __future__ import annotations

import math

import pytest
import yaml

from tradebot.schemas.weekly_meta import (
    WEEKLY_META_CONFIG_PATH,
    WeeklyMetaConfig,
    weekly_meta_config,
)
from tradebot.utils.failfast import ConfigContractError


def _raw() -> dict:
    return yaml.safe_load(WEEKLY_META_CONFIG_PATH.read_text(encoding="utf-8"))


def test_the_committed_config_loads() -> None:
    cfg = weekly_meta_config()
    assert cfg.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT")
    assert cfg.barrier_sigma == pytest.approx(math.sqrt(5.0))
    assert cfg.planned_trials == 4


def test_one_barrier_width_means_one_to_one_by_construction() -> None:
    fields = set(WeeklyMetaConfig.model_fields)
    assert "barrier_sigma" in fields
    assert not {"profit_target_sigma", "stop_loss_sigma"} & fields


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("holdout_split_utc", "2021-06-01T00:00:00+00:00", "voor de holdout"),
        ("first_test_start_utc", "2022-01-01", "tijdzone"),
        ("k_grid", [2.0, 1.0], "oplopend"),
        ("trades_per_week_target", 9.0, "meer events"),
        ("cpcv_n_test_groups", 4, "deelbaar"),
        ("shuffle_auc_band", [0.55, 0.60], "omsluiten"),
        ("unknown_key", 1, "unknown_key"),
    ],
)
def test_a_contradiction_is_refused(tmp_path, key, value, match) -> None:
    raw = _raw()
    raw[key] = value
    path = tmp_path / "weekly_meta.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigContractError, match=match):
        weekly_meta_config(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_meta_config.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.schemas.weekly_meta'`

- [ ] **Step 3: Write the config and the schema**

```yaml
# conf/model/weekly_meta.yaml
# De wekelijkse meta-label-strategie. Gevalideerd tegen
# src/tradebot/schemas/weekly_meta.py::WeeklyMetaConfig (extra="forbid").
# Spec: docs/superpowers/specs/2026-09-26-weekly-meta-label-design.md
#
# Niets hieronder wordt gezocht. Elke waarde is een keuze die in de
# preregistratie wordt bevroren; een tweede waarde is een nieuwe trial.
symbols: [BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, LINKUSDT, DOTUSDT]

# Walk-forward: eerste testkwartaal en de holdout-grens (spec §9, §17.2).
first_test_start_utc: "2022-01-01T00:00:00+00:00"
holdout_split_utc: "2026-06-24T00:00:00+00:00"
test_bars: 91

# Events: k wordt uit dit rooster gekozen op frequentie, nooit op rendement.
events_per_week_target: 5.5
k_grid: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

# Eén barrièrebreedte, dus 1:1 per constructie. sqrt(5) daggrootte = één week.
barrier_sigma: 2.2360679774997896
horizon_bars: 10
stop_slippage_bps: 5.0

# Handelen: 1-2 trades per week voor het hele boek, via de kansdrempel.
trades_per_week_target: 1.5

# Modellen: licht en niet getuned (spec §8).
logreg_c: 0.1
forest_n_estimators: 500
forest_max_depth: 3
forest_min_samples_leaf: 50
calibration_fraction: 0.25

# Sizing (spec §10).
kelly_multiple: 0.25
posterior_quantile: 0.25
n_probability_bins: 10
baseline_risk_fraction: 0.01
resize_band: 0.25
corr_window: 60
account_equity: 100000.0

# Poorten (spec §1, §14). Moeten gelijk zijn aan de preregistratie (Taak 12 toetst dat).
mc_paths: 10000
mc_max_drawdown: 0.25
mc_max_probability_1y: 0.10
cpcv_n_groups: 6
cpcv_n_test_groups: 2
pbo_max: 0.25
min_trades: 100
n_shuffle_replicates: 5
shuffle_auc_band: [0.45, 0.55]
holdout_brier_margin: 0.0

planned_trials: 4
seed: 20260926
```

```python
# src/tradebot/schemas/weekly_meta.py
"""Het contract van de wekelijkse meta-label-strategie. Zie `conf/model/weekly_meta.yaml`.

Apart van `schemas/config.py`, dat al te groot is: dit domein hoort bij één
strategie en verandert met haar mee.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field, model_validator

from .config import StrictModel, load_config

__all__ = ["WEEKLY_META_CONFIG_PATH", "WeeklyMetaConfig", "weekly_meta_config"]

WEEKLY_META_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "weekly_meta.yaml")

Positive = Annotated[float, Field(gt=0.0)]
Fraction = Annotated[float, Field(gt=0.0, lt=1.0)]


class WeeklyMetaConfig(StrictModel):
    """Alle instellingen van de strategie. Niets hiervan wordt gezocht."""

    symbols: tuple[str, ...]
    first_test_start_utc: str
    holdout_split_utc: str
    test_bars: Annotated[int, Field(ge=28)]
    events_per_week_target: Positive
    k_grid: tuple[Positive, ...]
    #: Eén breedte voor winst- en verliesbarrière: 1:1 is een eigenschap van het
    #: schema, niet van twee getallen die toevallig gelijk staan.
    barrier_sigma: Positive
    horizon_bars: Annotated[int, Field(ge=1)]
    stop_slippage_bps: Annotated[float, Field(ge=0.0)]
    trades_per_week_target: Positive
    logreg_c: Positive
    forest_n_estimators: Annotated[int, Field(ge=10)]
    forest_max_depth: Annotated[int, Field(ge=1, le=5)]
    forest_min_samples_leaf: Annotated[int, Field(ge=1)]
    calibration_fraction: Fraction
    kelly_multiple: Annotated[float, Field(gt=0.0, le=0.5)]
    posterior_quantile: Fraction
    n_probability_bins: Annotated[int, Field(ge=2)]
    baseline_risk_fraction: Fraction
    resize_band: Annotated[float, Field(ge=0.0, lt=1.0)]
    corr_window: Annotated[int, Field(ge=20)]
    account_equity: Positive
    mc_paths: Annotated[int, Field(ge=1000)]
    mc_max_drawdown: Fraction
    mc_max_probability_1y: Fraction
    cpcv_n_groups: Annotated[int, Field(ge=2)]
    cpcv_n_test_groups: Annotated[int, Field(ge=1)]
    pbo_max: Fraction
    min_trades: Annotated[int, Field(ge=1)]
    n_shuffle_replicates: Annotated[int, Field(ge=1)]
    shuffle_auc_band: tuple[float, float]
    holdout_brier_margin: Annotated[float, Field(ge=0.0)]
    planned_trials: Annotated[int, Field(ge=2)]
    seed: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _consistent(self) -> WeeklyMetaConfig:
        first = pd.Timestamp(self.first_test_start_utc)
        split = pd.Timestamp(self.holdout_split_utc)
        if first.tzinfo is None or split.tzinfo is None:
            raise ValueError("first_test_start_utc en holdout_split_utc moeten een tijdzone dragen")
        if not first < split:
            raise ValueError("de eerste testperiode moet voor de holdout-split liggen")
        if list(self.k_grid) != sorted(set(self.k_grid)):
            raise ValueError("k_grid moet strikt oplopend zijn")
        if self.trades_per_week_target >= self.events_per_week_target:
            raise ValueError("er moeten meer events dan trades zijn: het filter kiest")
        if self.cpcv_n_groups % self.cpcv_n_test_groups != 0:
            raise ValueError("cpcv_n_groups moet deelbaar zijn door cpcv_n_test_groups")
        lo, hi = self.shuffle_auc_band
        if not 0.0 < lo < 0.5 < hi < 1.0:
            raise ValueError("shuffle_auc_band moet 0,5 omsluiten")
        return self


def weekly_meta_config(path: Path | str = WEEKLY_META_CONFIG_PATH) -> WeeklyMetaConfig:
    """Laad en valideer `conf/model/weekly_meta.yaml` (plat document, geen wrapper)."""
    return load_config(path, WeeklyMetaConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_meta_config.py -p no:randomly -q`
Expected: PASS (9 tests). Als een `match` faalt: kijk naar de exacte tekst die `validate_mapping` om de pydantic-fout zet en pas de test NIET aan maar de foutmelding in de validator, zodat hij het woord uit de test bevat.

- [ ] **Step 5: Voeg spec §17 toe**

Voeg onderaan `docs/superpowers/specs/2026-09-26-weekly-meta-label-design.md` toe (letterlijk de zeven punten uit de sectie "Wijzigingen ten opzichte van de spec" van dit plan), onder de kop:

```markdown
## 17. Wijzigingen bij het uitwerken (2026-09-26)

De implementatieplanning (`docs/superpowers/plans/2026-09-26-weekly-meta-label-research.md`)
liep op zeven punten tegen de werkelijkheid van de repository aan. Deze sectie gaat
voor waar zij §4, §7, §9, §10.4 of §12 tegenspreekt.
```

gevolgd door de zeven genummerde punten en de alinea over Plan 1 / Plan 2.

- [ ] **Step 6: Commit**

```bash
git add conf/model/weekly_meta.yaml src/tradebot/schemas/weekly_meta.py tests/unit/test_weekly_meta_config.py docs/superpowers/specs/2026-09-26-weekly-meta-label-design.md
git commit -m "feat(weekly): the strategy's config contract, 1:1 by construction, and the seven spec corrections"
```

---

### Task 2: Het primaire signaal — richting uit de CUSUM-doorbraak

**Files:**
- Modify: `src/tradebot/labeling/cusum.py` (nieuwe publieke functie naast `symmetric_cusum_filter`)
- Create: `src/tradebot/labeling/breakout.py`
- Test: `tests/unit/test_breakout.py`
- Test: `tests/lookahead/test_breakout_causality.py`

**Interfaces:**
- Consumes: `_directional_cusum_filter_kernel(prices, thresholds_up, thresholds_down) -> (up_idx, down_idx)` in `labeling/cusum.py`.
- Produces:
  - `directional_cusum_filter(prices: np.ndarray, thresholds_up: np.ndarray, thresholds_down: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - `breakout_side(close: pd.Series, sigma_daily: pd.Series, k: float) -> pd.Series` (waarden −1/0/+1, naam `"side"`)
  - `events_per_week(sides: Mapping[str, pd.Series], *, start: pd.Timestamp, end: pd.Timestamp) -> float`
  - `calibrate_k(close: Mapping[str, pd.Series], sigma: Mapping[str, pd.Series], *, k_grid: Sequence[float], target_per_week: float, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, dict[float, float]]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_breakout.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.labeling.breakout import breakout_side, calibrate_k, events_per_week
from tradebot.labeling.cusum import directional_cusum_filter, symmetric_cusum_filter
from tradebot.utils.failfast import DataContractError


def _walk(n: int = 600, drift: float = 0.0, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(drift, 0.02, n))), index=idx)
    return close, pd.Series(0.02, index=idx)


def test_the_directional_filter_splits_the_symmetric_one() -> None:
    close, sigma = _walk()
    p = np.log(close.to_numpy())
    thr = np.full(p.size, 0.05)
    up, down = directional_cusum_filter(p, thr, thr)
    assert not set(up) & set(down)
    assert sorted(set(up) | set(down)) == sorted(symmetric_cusum_filter(p, thr))


def test_an_uptrend_breaks_up_more_often() -> None:
    close, sigma = _walk(drift=0.01)
    side = breakout_side(close, sigma, 2.0)
    assert (side > 0).sum() > (side < 0).sum()
    assert set(np.unique(side)) <= {-1.0, 0.0, 1.0}


def test_a_higher_k_gives_no_more_events() -> None:
    close, sigma = _walk()
    counts = [int((breakout_side(close, sigma, k) != 0).sum()) for k in (1.0, 2.0, 3.0, 4.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_a_gap_after_the_first_valid_bar_crashes() -> None:
    close, sigma = _walk()
    close.iloc[300] = np.nan
    with pytest.raises(DataContractError, match="gat"):
        breakout_side(close, sigma, 2.0)


def test_leading_nans_are_the_listing_not_a_gap() -> None:
    close, sigma = _walk()
    close.iloc[:50] = np.nan
    side = breakout_side(close, sigma, 2.0)
    assert (side.iloc[:51] == 0.0).all()


def test_calibrate_k_picks_the_rate_closest_to_the_target() -> None:
    close, sigma = _walk(n=800)
    closes, sigmas = {"A": close}, {"A": sigma}
    start, end = close.index[0], close.index[-1]
    k, rates = calibrate_k(closes, sigmas, k_grid=(1.0, 2.0, 3.0, 4.0),
                           target_per_week=1.0, start=start, end=end)
    assert k in rates
    assert abs(rates[k] - 1.0) == min(abs(r - 1.0) for r in rates.values())
    assert rates[1.0] > rates[4.0]
    measured = events_per_week({"A": breakout_side(close.loc[close.index < end],
                                                   sigma.loc[sigma.index < end], k)},
                               start=start, end=end)
    assert measured == pytest.approx(rates[k])
```

```python
# tests/lookahead/test_breakout_causality.py
"""De richting op bar t leest niets van na t, en de drempel op t leest sigma van t-1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.labeling.breakout import breakout_side


@pytest.fixture
def walk() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 400))), index=idx)
    sigma = pd.Series(rng.uniform(0.015, 0.03, 400), index=idx)
    return close, sigma


@pytest.mark.parametrize("t", [60, 150, 300])
def test_the_side_on_bar_t_ignores_the_future(walk, t) -> None:
    close, sigma = walk
    full = breakout_side(close, sigma, 2.0)
    cut = breakout_side(close.iloc[: t + 1], sigma.iloc[: t + 1], 2.0)
    pd.testing.assert_series_equal(full.iloc[: t + 1], cut)


@pytest.mark.parametrize("t", [60, 150, 300])
def test_the_threshold_on_bar_t_uses_sigma_of_t_minus_one(walk, t) -> None:
    close, sigma = walk
    moved = sigma.copy()
    moved.iloc[t] *= 100.0
    a = breakout_side(close, sigma, 2.0)
    b = breakout_side(close, moved, 2.0)
    pd.testing.assert_series_equal(a.iloc[: t + 1], b.iloc[: t + 1])


def test_the_negative_control_reads_its_own_sigma_and_breaks(walk) -> None:
    """Een drempel op sigma[t] in plaats van sigma[t-1] moet door de tweede test worden gevangen."""
    close, sigma = walk
    from tradebot.labeling.cusum import directional_cusum_filter
    p = np.log(close.to_numpy())

    def leaky(s: pd.Series) -> np.ndarray:
        thr = 2.0 * s.to_numpy()
        up, down = directional_cusum_filter(p, thr, thr)
        out = np.zeros(p.size)
        out[up], out[down] = 1.0, -1.0
        return out

    moved = sigma.copy()
    changed = 0
    for t in range(60, 380, 7):
        moved.iloc[:] = sigma.to_numpy()
        moved.iloc[t] *= 100.0
        changed += int(not np.array_equal(leaky(sigma)[: t + 1], leaky(moved)[: t + 1]))
    assert changed > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_breakout.py tests/lookahead/test_breakout_causality.py -p no:randomly -q`
Expected: FAIL, `ImportError: cannot import name 'directional_cusum_filter'`

- [ ] **Step 3: Implement**

Voeg in `src/tradebot/labeling/cusum.py` direct ná `symmetric_cusum_filter` toe (en zet de naam in `__all__` als dat bestand er een heeft):

```python
def directional_cusum_filter(
    prices: np.ndarray,
    thresholds_up: np.ndarray,
    thresholds_down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Publieke, getypeerde ingang op de directionele kernel.

    Returns
    -------
    (up_indices, down_indices) : int32-arrays met de bars waarop de cumulatieve
        beweging de drempel omhoog resp. omlaag passeerde. Samen zijn zij
        exact `symmetric_cusum_filter` bij gelijke drempels.
    """
    if not (len(prices) == len(thresholds_up) == len(thresholds_down)):
        raise ValueError("prices en beide drempelreeksen moeten even lang zijn.")
    return _directional_cusum_filter_kernel(
        np.ascontiguousarray(prices, dtype=np.float64),
        np.ascontiguousarray(thresholds_up, dtype=np.float64),
        np.ascontiguousarray(thresholds_down, dtype=np.float64),
    )
```

```python
# src/tradebot/labeling/breakout.py
"""Het primaire signaal: een CUSUM-doorbraak op de log-slotkoers (spec §5).

Richting = richting van de doorbraak. De drempel op bar t is `k * sigma[t-1]`:
de volatiliteit van de eventbar zelf mag niet bepalen of die bar een event is.
`k` wordt vastgelegd op eventfrequentie en nooit op rendement; `calibrate_k`
leest daarom geen labels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from .cusum import directional_cusum_filter

__all__ = ["breakout_side", "calibrate_k", "events_per_week"]


def breakout_side(close: pd.Series, sigma_daily: pd.Series, k: float) -> pd.Series:
    """+1 / -1 op een doorbraakbar, 0 elders. Leidende NaN's zijn de notering."""
    require(close.index.equals(sigma_daily.index),
            "close en sigma delen geen index.", DataContractError)
    require(k > 0.0, "k moet positief zijn.", DataContractError, k=k)
    log_p = np.log(close.to_numpy(dtype=np.float64))
    thr = k * sigma_daily.shift(1).to_numpy(dtype=np.float64)
    ok = np.isfinite(log_p) & np.isfinite(thr) & (thr > 0.0)
    side = np.zeros(len(close), dtype=np.float64)
    if int(ok.sum()) >= 2:
        start = int(np.argmax(ok))
        require(
            bool(ok[start:].all()),
            "Een gat na de eerste geldige bar. Een CUSUM over een gat telt de "
            "sprong als één beweging; dat is een datadefect, geen signaal.",
            DataContractError,
            first_gap=str(close.index[start + int(np.argmin(ok[start:]))]),
        )
        up, down = directional_cusum_filter(log_p[start:], thr[start:], thr[start:])
        side[start + up] = 1.0
        side[start + down] = -1.0
    return pd.Series(side, index=close.index, name="side")


def events_per_week(
    sides: Mapping[str, pd.Series], *, start: pd.Timestamp, end: pd.Timestamp,
) -> float:
    """Events per week voor het hele boek in `[start, end)`."""
    weeks = (end - start) / pd.Timedelta(days=7)
    require(weeks > 0.0, "Een leeg venster.", DataContractError,
            start=str(start), end=str(end))
    total = 0
    for side in sides.values():
        window = side.loc[(side.index >= start) & (side.index < end)]
        total += int((window != 0.0).sum())
    return float(total / weeks)


def calibrate_k(
    close: Mapping[str, pd.Series],
    sigma: Mapping[str, pd.Series],
    *,
    k_grid: Sequence[float],
    target_per_week: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, dict[float, float]]:
    """De k uit `k_grid` waarvan de eventfrequentie in `[start, end)` het dichtst bij het doel ligt.

    Alleen data vóór `end` wordt gelezen. Bij gelijke afstand wint de grotere k
    (minder events, dus voorzichtiger).
    """
    require(set(close) == set(sigma), "close en sigma dekken andere symbolen.",
            DataContractError)
    rates: dict[float, float] = {}
    for k in k_grid:
        sides = {
            s: breakout_side(close[s].loc[close[s].index < end],
                             sigma[s].loc[sigma[s].index < end], float(k))
            for s in close
        }
        rates[float(k)] = events_per_week(sides, start=start, end=end)
    best = min(rates, key=lambda k: (abs(rates[k] - target_per_week), -k))
    return best, rates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_breakout.py tests/lookahead/test_breakout_causality.py tests/unit -k cusum -p no:randomly -q`
Expected: PASS; bestaande CUSUM-tests blijven groen.

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/labeling/cusum.py src/tradebot/labeling/breakout.py tests/unit/test_breakout.py tests/lookahead/test_breakout_causality.py
git commit -m "feat(labeling): the breakout side from the existing directional CUSUM kernel, with k fixed on frequency"
```

---

### Task 3: Fillrendementen bij de barrière en een kostenbewust doel

**Files:**
- Create: `src/tradebot/labeling/barrier_fills.py`
- Test: `tests/unit/test_barrier_fills.py`

**Interfaces:**
- Consumes: `BarrierLabels`, `label_triple_barrier` uit `labeling/vol_barriers.py`; `LabelingConfig`, `ExecutionConfig` uit `schemas/config.py`.
- Produces:
  - `round_trip_cost(exec_cfg: ExecutionConfig) -> float` (fractie, 0,0013)
  - `barrier_fill_returns(labels, open_, close, cfg: LabelingConfig, *, stop_slippage_bps: float) -> np.ndarray`
  - `with_cost_aware_target(labels, fill_returns, *, round_trip_cost: float) -> BarrierLabels` (vervangt `realized_return` en `meta_label`)
  - `select_events(labels, mask) -> BarrierLabels`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_barrier_fills.py
"""Een 1:1-trade met stops bij de exchange vult op het barrièreniveau, niet op de slotkoers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tradebot.labeling.barrier_fills import (
    barrier_fill_returns,
    round_trip_cost,
    select_events,
    with_cost_aware_target,
)
from tradebot.labeling.vol_barriers import label_triple_barrier
from tradebot.schemas.config import ExecutionConfig, LabelingConfig

CFG = LabelingConfig(profit_target_sigma=math.sqrt(5.0), stop_loss_sigma=math.sqrt(5.0),
                     horizon_bars=10, entry_lag_bars=1, min_sigma_obs=60)
SIG = 0.02
B = math.sqrt(5.0) * SIG
SLIP = 5.0


def _one_event(o2: float, h2: float, l2: float, c2: float, side: float = 1.0):
    n = 14
    o, h, l, c = (np.full(n, 100.0), np.full(n, 100.2), np.full(n, 99.8), np.full(n, 100.0))
    o[2], h[2], l[2], c[2] = o2, h2, l2, c2
    s = np.zeros(n)
    s[0] = side
    labels = label_triple_barrier(h, l, c, np.full(n, SIG), s, CFG)
    assert len(labels) == 1
    return labels, o, c


def test_a_long_target_fills_at_the_level() -> None:
    labels, o, c = _one_event(100.1, 105.0, 99.9, 104.0)
    assert labels.barrier_outcome[0] == 1
    ret = barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)
    assert ret[0] == pytest.approx(B)


def test_a_long_stop_fills_at_the_level_minus_slippage() -> None:
    labels, o, c = _one_event(100.1, 100.1, 95.0, 96.0)
    assert labels.barrier_outcome[0] == -1
    ret = barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)
    assert ret[0] == pytest.approx((1.0 - B) * (1.0 - SLIP / 1e4) - 1.0)


def test_a_gap_through_the_target_fills_at_the_open() -> None:
    labels, o, c = _one_event(106.0, 107.0, 105.5, 106.0)
    ret = barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)
    assert ret[0] == pytest.approx(0.06)


def test_both_barriers_in_one_bar_is_a_stop() -> None:
    labels, o, c = _one_event(100.0, 105.0, 95.0, 100.0)
    assert labels.barrier_outcome[0] == -1
    ret = barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)
    assert ret[0] < 0.0


def test_the_vertical_barrier_exits_at_the_close() -> None:
    labels, o, c = _one_event(100.0, 100.2, 99.8, 100.0)
    assert labels.barrier_outcome[0] == 0
    ret = barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)
    assert ret[0] == pytest.approx(0.0)


def test_a_short_is_the_mirror_image() -> None:
    labels, o, c = _one_event(99.9, 100.1, 95.0, 96.0, side=-1.0)
    assert labels.barrier_outcome[0] == 1
    assert barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)[0] == pytest.approx(B)
    labels, o, c = _one_event(100.1, 105.0, 99.9, 104.0, side=-1.0)
    assert labels.barrier_outcome[0] == -1
    expected = -((1.0 + B) * (1.0 + SLIP / 1e4) - 1.0)
    assert barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)[0] == pytest.approx(expected)


def test_the_target_is_net_of_the_round_trip() -> None:
    labels, o, c = _one_event(100.0, 100.2, 99.8, 100.0)
    out = with_cost_aware_target(labels, np.array([0.0012]), round_trip_cost=0.0013)
    assert out.meta_label[0] == 0 and out.realized_return[0] == pytest.approx(0.0012)
    out = with_cost_aware_target(labels, np.array([0.0014]), round_trip_cost=0.0013)
    assert out.meta_label[0] == 1


def test_the_round_trip_comes_from_the_fee_config() -> None:
    exec_cfg = ExecutionConfig(maker_fee_bps=2.0, taker_fee_bps=5.5, assumed_half_spread_bps=1.0)
    assert round_trip_cost(exec_cfg) == pytest.approx(0.0013)


def test_select_events_keeps_every_field_aligned() -> None:
    labels, _, _ = _one_event(100.0, 100.2, 99.8, 100.0)
    kept = select_events(labels, np.array([True]))
    dropped = select_events(labels, np.array([False]))
    assert len(kept) == 1 and len(dropped) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_barrier_fills.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.labeling.barrier_fills'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/labeling/barrier_fills.py
"""Wat een 1:1-trade met stops bij de exchange werkelijk oplevert (spec §6, §17.6).

`vol_barriers.label_triple_barrier` bepaalt WELKE barrière eerst raakt en op welke
bar. Zijn `realized_return` rekent met de slotkoers van die bar. Een stop of
take-profit bij de exchange vult op het barrièreniveau -- of op de open, als de
bar er met een gat voorbij opent. Een stop vult bovendien als marktorder, met
slippage tegen de positie in. Deze module rekent die fill uit, en zet het doel
op "netto na de round trip positief".
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..schemas.config import ExecutionConfig, LabelingConfig
from ..utils.failfast import DataContractError, require
from .vol_barriers import BarrierLabels

__all__ = ["barrier_fill_returns", "round_trip_cost", "select_events",
           "with_cost_aware_target"]


def round_trip_cost(exec_cfg: ExecutionConfig) -> float:
    """Eén definitie van de round trip: twee keer taker plus halve spread."""
    return 2.0 * (exec_cfg.taker_fee_bps + exec_cfg.assumed_half_spread_bps) / 1e4


def barrier_fill_returns(
    labels: BarrierLabels,
    open_: np.ndarray,
    close: np.ndarray,
    cfg: LabelingConfig,
    *,
    stop_slippage_bps: float,
) -> np.ndarray:
    """Positierendement per event bij een fill op de barrière (richting verrekend)."""
    open_ = np.asarray(open_, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    require(open_.size == close.size, "open en close zijn niet even lang.", DataContractError)
    lag = int(cfg.entry_lag_bars)
    slip = float(stop_slippage_bps) / 1e4
    out = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        s = float(labels.side[i])
        entry = float(close[int(labels.event_idx[i]) + lag])
        j = int(labels.exit_idx[i])
        outcome = int(labels.barrier_outcome[i])
        if outcome == 0:
            fill = float(close[j])
        else:
            o = float(open_[j])
            require(np.isfinite(o), "Een barrière-exit op een bar zonder open.",
                    DataContractError, exit_idx=j)
            if outcome == 1:
                level = entry * (1.0 + s * cfg.profit_target_sigma * float(labels.sigma[i]))
                gapped = o >= level if s > 0 else o <= level
                fill = o if gapped else level
            else:
                level = entry * (1.0 - s * cfg.stop_loss_sigma * float(labels.sigma[i]))
                gapped = o <= level if s > 0 else o >= level
                fill = (o if gapped else level) * (1.0 - s * slip)
        out[i] = s * (fill / entry - 1.0)
    return out


def with_cost_aware_target(
    labels: BarrierLabels, fill_returns: np.ndarray, *, round_trip_cost: float,
) -> BarrierLabels:
    """Labels met het fillrendement en het doel "netto na de round trip > 0"."""
    fill_returns = np.asarray(fill_returns, dtype=np.float64)
    require(fill_returns.size == len(labels), "Een fillrendement per event.",
            DataContractError, n=fill_returns.size, n_events=len(labels))
    return dataclasses.replace(
        labels,
        realized_return=fill_returns,
        meta_label=((fill_returns - float(round_trip_cost)) > 0.0).astype(np.int8),
    )


def select_events(labels: BarrierLabels, mask: np.ndarray) -> BarrierLabels:
    """Dezelfde labels, beperkt tot `mask`, met elk veld in de pas."""
    m = np.asarray(mask, dtype=bool)
    require(m.size == len(labels), "Het masker heeft niet één waarde per event.",
            DataContractError)
    return BarrierLabels(**{f.name: getattr(labels, f.name)[m]
                            for f in dataclasses.fields(labels)})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_barrier_fills.py -p no:randomly -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/labeling/barrier_fills.py tests/unit/test_barrier_fills.py
git commit -m "feat(labeling): fill 1:1 exits at the barrier, with stop slippage, and label net of the round trip"
```

---

### Task 4: De gecertificeerde dagmarkt op één raster

**Files:**
- Create: `src/tradebot/data/weekly_market.py`
- Create: `tests/weekly_fixtures.py`
- Test: `tests/unit/test_weekly_market.py`

**Interfaces:**
- Consumes: `PitStore`, `DataRegister`, `load_certified_series(store, register, *, asset_class, dataset, symbol, granularity) -> (df, data_hash)`, `ASOF_INDEX_NAME` (`features/base.py`), `daily_funding_panel` (`data/funding_panel.py`), `ewma_volatility` (`volatility/ewma.py`), `VolatilityConfig`.
- Produces:
  - `WeeklyMarket` (frozen dataclass): `grid: pd.DatetimeIndex`, `ohlcv: dict[str, pd.DataFrame]` (kolommen `open, high, low, close, turnover`), `sigma_daily`, `sigma_annual`, `funding`, `open_interest`, `adv_usd` (alle `pd.DataFrame`, kolom per symbool, index `grid`), `source_hashes: dict[str, str]`; methode `truncate(end) -> WeeklyMarket` (bars strikt vóór `end`).
  - `load_weekly_market(root: Path, symbols: Sequence[str]) -> WeeklyMarket`
  - In tests: `tests.weekly_fixtures.synthetic_market(n=500, symbols=("BTCUSDT","ETHUSDT","SOLUSDT"), seed=7) -> WeeklyMarket`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_weekly_market.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.weekly_fixtures import synthetic_market
from tradebot.data.weekly_market import load_weekly_market

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT")


def test_the_certified_market_is_one_gapless_daily_grid() -> None:
    market = load_weekly_market(ROOT, SYMBOLS)
    steps = market.grid[1:] - market.grid[:-1]
    assert (steps == pd.Timedelta(days=1)).all()
    assert str(market.grid.tz) == "UTC"
    for frame in (market.sigma_daily, market.funding, market.open_interest, market.adv_usd):
        assert list(frame.columns) == list(SYMBOLS)
        assert frame.index.equals(market.grid)
    # Drie gecertificeerde reeksen per symbool: ohlcv, funding, open interest.
    assert len(market.source_hashes) == 3 * len(SYMBOLS)


def test_daily_sigma_is_the_annual_ewma_de_annualised() -> None:
    market = load_weekly_market(ROOT, SYMBOLS)
    ratio = (market.sigma_annual / market.sigma_daily).stack().dropna()
    assert np.allclose(ratio.to_numpy(), np.sqrt(365.0))


def test_adv_is_known_one_bar_later() -> None:
    market = synthetic_market()
    turnover = market.ohlcv["BTCUSDT"]["turnover"]
    expected = turnover.rolling(30, min_periods=30).mean().shift(1)
    pd.testing.assert_series_equal(market.adv_usd["BTCUSDT"], expected, check_names=False)


def test_truncate_keeps_only_bars_strictly_before_end() -> None:
    market = synthetic_market()
    end = market.grid[300]
    cut = market.truncate(end)
    assert cut.grid[-1] == market.grid[299]
    pd.testing.assert_frame_equal(cut.sigma_daily, market.sigma_daily.iloc[:300])
    assert len(cut.ohlcv["ETHUSDT"]) == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_market.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tests.weekly_fixtures'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/data/weekly_market.py
"""De gecertificeerde dagmarkt van de wekelijkse strategie, op één raster (spec §4, §17.1).

Elke reeks komt via `load_certified_series`, dus langs het data-register; een
afwijkende hash crasht hier en niet pas in een rapport. Het raster is de unie
van de asof-tijden (sluitmomenten) en moet gatenloos dagelijks zijn. Vóór de
notering van een symbool staat er NaN: dat is geen gat maar het feit dat het
instrument nog niet bestond.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.base import ASOF_INDEX_NAME, DataRegister, load_certified_series
from ..schemas.config import VolatilityConfig, load_config
from ..utils.failfast import DataContractError, require
from ..volatility.ewma import ewma_volatility
from .funding_panel import daily_funding_panel
from .pit_store import PitStore

__all__ = ["OHLCV_COLUMNS", "WeeklyMarket", "load_weekly_market"]

OHLCV_COLUMNS = ("open", "high", "low", "close", "turnover")
ADV_WINDOW = 30


@dataclass(frozen=True)
class WeeklyMarket:
    grid: pd.DatetimeIndex
    ohlcv: dict[str, pd.DataFrame]
    sigma_daily: pd.DataFrame
    sigma_annual: pd.DataFrame
    funding: pd.DataFrame
    open_interest: pd.DataFrame
    adv_usd: pd.DataFrame
    source_hashes: dict[str, str]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.ohlcv)

    def truncate(self, end: pd.Timestamp) -> WeeklyMarket:
        """Alles strikt vóór `end`. Elke reeks hier is causaal, dus afkappen verandert niets vóór `end`."""
        grid = self.grid[self.grid < end]
        return WeeklyMarket(
            grid=grid,
            ohlcv={s: f.loc[grid] for s, f in self.ohlcv.items()},
            sigma_daily=self.sigma_daily.loc[grid],
            sigma_annual=self.sigma_annual.loc[grid],
            funding=self.funding.loc[grid],
            open_interest=self.open_interest.loc[grid],
            adv_usd=self.adv_usd.loc[grid],
            source_hashes=dict(self.source_hashes),
        )


def market_from_frames(
    ohlcv: dict[str, pd.DataFrame],
    funding: pd.DataFrame,
    open_interest: pd.DataFrame,
    *,
    vol: VolatilityConfig,
    source_hashes: dict[str, str],
) -> WeeklyMarket:
    """Bouw de afgeleide panels (sigma, ADV) uit ruwe frames op één raster."""
    grid = next(iter(ohlcv.values())).index
    for name, frame in ohlcv.items():
        require(frame.index.equals(grid), "OHLCV-frames delen geen raster.",
                DataContractError, symbol=name)
    closes = pd.DataFrame({s: f["close"] for s, f in ohlcv.items()})
    sigma_annual = pd.DataFrame({
        s: ewma_volatility(closes[s], lam=vol.ewma_lambda,
                           burn_in_bars=vol.burn_in_bars,
                           annualisation_factor=vol.annualisation_factor)
        for s in closes
    })
    turnover = pd.DataFrame({s: f["turnover"] for s, f in ohlcv.items()})
    return WeeklyMarket(
        grid=grid,
        ohlcv=ohlcv,
        sigma_daily=sigma_annual / np.sqrt(vol.annualisation_factor),
        sigma_annual=sigma_annual,
        funding=funding.reindex(columns=list(ohlcv)),
        open_interest=open_interest.reindex(columns=list(ohlcv)),
        # Causaal: de turnover van bar t is pas op zijn close bekend.
        adv_usd=turnover.rolling(ADV_WINDOW, min_periods=ADV_WINDOW).mean().shift(1),
        source_hashes=source_hashes,
    )


def _asof_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(df["asof_ts_ns"].to_numpy(), unit="ns", utc=True),
        name=ASOF_INDEX_NAME)


def load_weekly_market(root: Path, symbols: Sequence[str]) -> WeeklyMarket:
    """Laad OHLCV, funding en open interest langs het data-register."""
    vol = load_config(root / "conf/model/volatility.yaml", VolatilityConfig)
    store = PitStore(root / "data/pit_store")
    register = DataRegister(root / "artefacts/governance/data_hashes.json")
    frames: dict[str, pd.DataFrame] = {}
    oi: dict[str, pd.Series] = {}
    hashes: dict[str, str] = {}
    for sym in symbols:
        df, h = load_certified_series(store, register, asset_class="crypto",
                                      dataset="ohlcv", symbol=sym, granularity="1d")
        frames[sym] = pd.DataFrame(
            {c: df[c].to_numpy(dtype=np.float64) for c in OHLCV_COLUMNS},
            index=_asof_index(df))
        hashes[f"crypto/ohlcv/{sym}/1d"] = h
        odf, oh = load_certified_series(store, register, asset_class="crypto",
                                        dataset="open_interest", symbol=sym,
                                        granularity="1d")
        oi[sym] = pd.Series(odf["open_interest"].to_numpy(dtype=np.float64),
                            index=_asof_index(odf))
        hashes[f"crypto/open_interest/{sym}/1d"] = oh
        hashes[f"crypto/funding/{sym}/8h"] = register.certified_hash(
            "crypto", "funding", sym, "8h")
    grid = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])),
                            name=ASOF_INDEX_NAME)
    require(bool((grid[1:] - grid[:-1] == pd.Timedelta(days=1)).all()),
            "Het dagraster heeft gaten.", DataContractError)
    funding = daily_funding_panel(store, register, symbols=list(symbols),
                                  asset_class="crypto", funding_granularity="8h",
                                  bar_index=grid)
    return market_from_frames(
        {s: f.reindex(grid) for s, f in frames.items()},
        funding,
        pd.DataFrame({s: oi[s].reindex(grid) for s in symbols}),
        vol=vol,
        source_hashes=hashes,
    )
```

```python
# tests/weekly_fixtures.py
"""Een synthetische dagmarkt met de vorm van de echte, voor de tests van de wekelijkse strategie."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.data.weekly_market import WeeklyMarket, market_from_frames
from tradebot.schemas.config import VolatilityConfig


def synthetic_market(
    n: int = 500,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    seed: int = 7,
) -> WeeklyMarket:
    rng = np.random.default_rng(seed)
    grid = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC", name="asof_ts")
    common = rng.normal(0.0, 0.03, n)
    ohlcv = {}
    for s in symbols:
        r = 0.8 * common + 0.6 * rng.normal(0.0, 0.03, n)
        close = 100.0 * np.exp(np.cumsum(r))
        open_ = np.r_[100.0, close[:-1]]
        wick = np.abs(rng.normal(0.0, 0.01, n))
        ohlcv[s] = pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) * (1.0 + wick),
            "low": np.minimum(open_, close) * (1.0 - wick),
            "close": close,
            "turnover": rng.uniform(1e9, 2e9, n),
        }, index=grid)
    funding = pd.DataFrame(rng.normal(1e-4, 5e-5, (n, len(symbols))), index=grid,
                           columns=list(symbols))
    oi = pd.DataFrame(rng.uniform(1e6, 2e6, (n, len(symbols))), index=grid,
                      columns=list(symbols))
    vol = VolatilityConfig(ewma_lambda=0.94, burn_in_bars=60, annualisation_factor=365.0)
    return market_from_frames(ohlcv, funding, oi, vol=vol, source_hashes={})
```

Controleer vóór stap 4 dat `VolatilityConfig(...)` met deze drie velden valideert (de overige velden hebben defaults: `estimator`, `min_periods`, `forecast_*`). Zo niet: neem de ontbrekende velden letterlijk over uit `conf/model/volatility.yaml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_market.py -p no:randomly -q`
Expected: PASS (4 tests). Faalt de eerste test op `data_hashes`, dan is de store gewijzigd sinds certificering: stop en meld het; herstel het register niet.

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/data/weekly_market.py tests/weekly_fixtures.py tests/unit/test_weekly_market.py
git commit -m "feat(data): the certified daily market for the weekly strategy, on one gapless grid"
```

---

### Task 5: De vaste featureset

**Files:**
- Create: `src/tradebot/features/weekly_set.py`
- Test: `tests/unit/test_weekly_features.py`
- Test: `tests/lookahead/test_weekly_features_causality.py`

**Interfaces:**
- Consumes: `frac_diff_ffd(series, d, threshold)`, `causal_min_frac_diff(series, train_end_iloc)` (`features/fracdiff.py`); `KalmanOUMeanReversion(symbol, init_window)` met `fit(df)`, `current_zscore(log_price)`, `current_halflife()` (`alpha/kalman_ou.py`); `get_yang_zhang_volatility(df, window)` (`volatility/yang_zhang.py`); `WeeklyMarket` (Taak 4).
- Produces:
  - `FEATURE_COLUMNS: tuple[str, ...]` (19 kolommen, volgorde vast)
  - `ffd_threshold() -> float` (uit `conf/model/fracdiff.yaml`, `weight_threshold` = 1e-4)
  - `fit_common_d_star(log_close: Mapping[str, pd.Series], *, until: pd.Timestamp, min_obs: int = 250, threshold: float | None = None) -> float`
  - `market_features(log_returns: pd.DataFrame, *, window: int) -> dict[str, pd.DataFrame]` (sleutels `mkt_ret5`, `resid_z20`, `avg_corr60`)
  - `build_feature_panel(market: WeeklyMarket, side: Mapping[str, pd.Series], *, d_star: float, corr_window: int, threshold: float | None = None) -> dict[str, pd.DataFrame]`

**Waarom de FFD-drempel uit `conf/` komt:** gemeten op 2026-09-26 is het FFD-venster bij d = 0,4 **1.458 bars** onder de moduledefault 1e-5, en **282 bars** onder de 1e-4 uit `conf/model/fracdiff.yaml` (d = 0,3: 2.275 tegen 388). Met 1e-5 houden SOL en AVAX vrijwel geen events over. De burn-in van ~200–400 bars na notering blijft een kostenpost en wordt per symbool gerapporteerd (`n_dropped_nan`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_weekly_features.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.features.weekly_set import (
    FEATURE_COLUMNS,
    build_feature_panel,
    fit_common_d_star,
    market_features,
)
from tradebot.utils.failfast import DataContractError


def _zeros(market) -> dict[str, pd.Series]:
    return {s: pd.Series(0.0, index=market.grid) for s in market.symbols}


def test_every_symbol_gets_exactly_the_declared_columns() -> None:
    market = synthetic_market()
    panel = build_feature_panel(market, _zeros(market), d_star=0.4, corr_window=60)
    assert set(panel) == set(market.symbols)
    for frame in panel.values():
        assert tuple(frame.columns) == FEATURE_COLUMNS
        assert frame.index.equals(market.grid)
        # Na de langste burn-in (FFD 282 bij d = 0,4 en drempel 1e-4) is alles eindig.
        assert np.isfinite(frame.iloc[300:].to_numpy()).all()


def test_d_star_reads_only_data_before_until() -> None:
    market = synthetic_market(n=600)
    logs = {s: np.log(market.ohlcv[s]["close"]) for s in market.symbols}
    until = market.grid[400]
    d0 = fit_common_d_star(logs, until=until)
    moved = {s: x.where(x.index < until, x + 5.0) for s, x in logs.items()}
    assert fit_common_d_star(moved, until=until) == d0
    assert 0.0 < d0 <= 0.9


def test_d_star_refuses_too_little_history() -> None:
    market = synthetic_market(n=300)
    logs = {s: np.log(market.ohlcv[s]["close"]) for s in market.symbols}
    with pytest.raises(DataContractError, match="historie"):
        fit_common_d_star(logs, until=market.grid[100])


def test_the_average_correlation_recovers_an_equicorrelated_rho() -> None:
    rng = np.random.default_rng(11)
    n, rho = 400, 0.6
    common = rng.normal(size=n)
    data = {f"S{i}": np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(size=n)
            for i in range(4)}
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    out = market_features(pd.DataFrame(data, index=idx) * 0.02, window=60)
    assert out["avg_corr60"].iloc[100:].mean().mean() == pytest.approx(rho, abs=0.1)
```

```python
# tests/lookahead/test_weekly_features_causality.py
"""Rij t van elke feature is gelijk als alles na t wordt weggelaten."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.features.weekly_set import build_feature_panel


def _zeros(market) -> dict[str, pd.Series]:
    return {s: pd.Series(0.0, index=market.grid) for s in market.symbols}


@pytest.mark.parametrize("t", [230, 320, 410])
def test_row_t_does_not_see_the_future(t) -> None:
    market = synthetic_market(n=460)
    full = build_feature_panel(market, _zeros(market), d_star=0.4, corr_window=60)
    cut_market = market.truncate(market.grid[t] + pd.Timedelta(hours=1))
    cut = build_feature_panel(cut_market, _zeros(cut_market), d_star=0.4, corr_window=60)
    for s in market.symbols:
        a = full[s].iloc[t].to_numpy(dtype=float)
        b = cut[s].iloc[t].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-10, atol=1e-12), s


def test_funding_carries_one_bar_of_lag() -> None:
    market = synthetic_market(n=460)
    t = 300
    moved_funding = market.funding.copy()
    moved_funding.iloc[t] += 0.01
    moved = type(market)(**{**market.__dict__, "funding": moved_funding})
    a = build_feature_panel(market, _zeros(market), d_star=0.4, corr_window=60)["BTCUSDT"]
    b = build_feature_panel(moved, _zeros(moved), d_star=0.4, corr_window=60)["BTCUSDT"]
    assert a["funding_z"].iloc[t] == pytest.approx(b["funding_z"].iloc[t])
    assert a["funding_z"].iloc[t + 1] != pytest.approx(b["funding_z"].iloc[t + 1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_features.py tests/lookahead/test_weekly_features_causality.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.features.weekly_set'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/features/weekly_set.py
"""De vaste featureset van de wekelijkse meta-label-strategie (spec §7, §17.4-5).

Elke kolom op bar t gebruikt uitsluitend data t/m de close van t; funding en
open interest krijgen daarbovenop één bar lag. Wat gefit wordt, wordt óf vóór
de eerste testperiode gefit (d*), óf rollend op een venster dat op t eindigt
(Kalman/OU, PCA). `tests/lookahead/test_weekly_features_causality.py` bewijst
dat door alles na t weg te laten en rij t te vergelijken.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..alpha.kalman_ou import KalmanOUMeanReversion
from ..data.weekly_market import WeeklyMarket
from ..schemas.config import FracDiffConfig, load_config
from ..utils.failfast import DataContractError, require
from ..volatility.yang_zhang import get_yang_zhang_volatility
from .base import repo_root
from .fracdiff import causal_min_frac_diff, frac_diff_ffd

__all__ = ["FEATURE_COLUMNS", "build_feature_panel", "ffd_threshold",
           "fit_common_d_star", "market_features"]

BAR_FEATURES = (
    "ffd_logp", "ret1_n", "ret5_n", "ret20_n", "dist_hi20", "dist_lo20",
    "kalman_z", "ou_halflife_log", "sigma_ewma", "yz_ratio", "vol_of_vol",
    "dollar_vol_ratio", "funding_z", "oi_chg1", "oi_chg5",
)
MARKET_FEATURES = ("mkt_ret5", "resid_z20", "avg_corr60")
FEATURE_COLUMNS = BAR_FEATURES + MARKET_FEATURES + ("side",)

KALMAN_WINDOW = 126
MIN_LIVE_SYMBOLS = 3


def ffd_threshold() -> float:
    """De FFD-gewichtsdrempel uit `conf/model/fracdiff.yaml` (één bron van waarheid)."""
    return float(load_config(repo_root() / "conf/model/fracdiff.yaml",
                             FracDiffConfig).weight_threshold)


def fit_common_d_star(
    log_close: Mapping[str, pd.Series], *, until: pd.Timestamp, min_obs: int = 250,
    threshold: float | None = None,
) -> float:
    """Eén d* voor alle symbolen: het maximum van de per-symbool d* vóór `until`.

    Alleen symbolen met ten minste `min_obs` bars vóór `until` tellen mee; het
    maximum maakt de reeks voor elk van hen stationair (ADF), en één d voor alle
    symbolen houdt de feature over symbolen vergelijkbaar.
    """
    thr = ffd_threshold() if threshold is None else float(threshold)
    found: list[float] = []
    for series in log_close.values():
        valid = series.dropna()
        n_before = int((valid.index < until).sum())
        if n_before >= min_obs:
            found.append(float(causal_min_frac_diff(valid, train_end_iloc=n_before,
                                                    threshold=thr)))
    require(bool(found), "Geen enkel symbool heeft genoeg historie voor d*.",
            DataContractError, until=str(until), min_obs=min_obs)
    return max(found)


def _rolling_kalman(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Kalman-OU opnieuw gefit op elk venster van 126 bars dat op t eindigt."""
    valid = close.dropna()
    z = np.full(len(close), np.nan)
    hl = np.full(len(close), np.nan)
    frame = valid.to_frame("close")
    where = close.index.get_indexer(valid.index)
    log_v = np.log(valid.to_numpy(dtype=np.float64))
    for pos in range(KALMAN_WINDOW - 1, len(valid)):
        model = KalmanOUMeanReversion(symbol="weekly", init_window=KALMAN_WINDOW)
        model.fit(frame.iloc[pos - KALMAN_WINDOW + 1: pos + 1])
        z[where[pos]] = model.current_zscore(float(log_v[pos]))
        hl[where[pos]] = model.current_halflife()
    return pd.Series(z, index=close.index), pd.Series(hl, index=close.index)


def _symbol_features(
    ohlcv: pd.DataFrame, sigma: pd.Series, funding: pd.Series,
    open_interest: pd.Series, *, d_star: float, threshold: float,
) -> pd.DataFrame:
    idx = ohlcv.index
    close = ohlcv["close"]
    logp = np.log(close)
    out = pd.DataFrame(index=idx)
    out["ffd_logp"] = frac_diff_ffd(logp.dropna(), d_star, threshold=threshold).reindex(idx)
    for h in (1, 5, 20):
        out[f"ret{h}_n"] = (logp - logp.shift(h)) / (sigma * np.sqrt(h))
    out["dist_hi20"] = (logp - np.log(ohlcv["high"].rolling(20).max())) / sigma
    out["dist_lo20"] = (logp - np.log(ohlcv["low"].rolling(20).min())) / sigma
    z, hl = _rolling_kalman(close)
    out["kalman_z"] = z
    out["ou_halflife_log"] = np.log(hl.clip(lower=1.0, upper=500.0))
    out["sigma_ewma"] = sigma
    valid = ohlcv[["open", "high", "low", "close"]].dropna()
    yz = get_yang_zhang_volatility(valid, window=20).reindex(idx)
    out["yz_ratio"] = yz / yz.rolling(120, min_periods=120).mean()
    out["vol_of_vol"] = np.log(sigma).diff().rolling(30, min_periods=30).std()
    turnover = ohlcv["turnover"]
    out["dollar_vol_ratio"] = turnover / turnover.rolling(30, min_periods=30).mean().shift(1)
    f3 = funding.rolling(3, min_periods=3).sum()
    f3_z = (f3 - f3.rolling(90, min_periods=90).mean()) / f3.rolling(90, min_periods=90).std()
    out["funding_z"] = f3_z.shift(1)
    log_oi = np.log(open_interest.where(open_interest > 0.0))
    out["oi_chg1"] = log_oi.diff(1).shift(1)
    out["oi_chg5"] = log_oi.diff(5).shift(1)
    return out


def market_features(log_returns: pd.DataFrame, *, window: int) -> dict[str, pd.DataFrame]:
    """PCA-marktfactor, residu-z-score en gemiddelde correlatie, rollend op `window` bars t/m t."""
    idx, cols = log_returns.index, list(log_returns.columns)
    values = log_returns.to_numpy(dtype=np.float64)
    mkt = np.full(values.shape, np.nan)
    resid = np.full(values.shape, np.nan)
    corr = np.full(values.shape, np.nan)
    for pos in range(window - 1, len(idx)):
        block = values[pos - window + 1: pos + 1]
        live = np.flatnonzero(np.isfinite(block).all(axis=0))
        if live.size < MIN_LIVE_SYMBOLS:
            continue
        x = block[:, live]
        std = x.std(axis=0, ddof=1)
        if np.any(std <= 0.0):
            continue
        zs = (x - x.mean(axis=0)) / std
        c = np.corrcoef(x, rowvar=False)
        _, vecs = np.linalg.eigh(c)
        pc1 = vecs[:, -1] if vecs[:, -1].sum() >= 0.0 else -vecs[:, -1]
        pc1 = pc1 / np.abs(pc1).sum()
        f = zs @ pc1
        fc = f - f.mean()
        beta = (zs * fc[:, None]).sum(axis=0) / float((fc ** 2).sum())
        e = zs - np.outer(f, beta)
        n_live = live.size
        mkt[pos, live] = f[-5:].sum() / (f.std(ddof=1) * np.sqrt(5.0))
        resid[pos, live] = e[-20:].sum(axis=0) / (e.std(axis=0, ddof=1) * np.sqrt(20.0))
        corr[pos, live] = (c.sum() - n_live) / (n_live * (n_live - 1))
    return {name: pd.DataFrame(arr, index=idx, columns=cols)
            for name, arr in (("mkt_ret5", mkt), ("resid_z20", resid), ("avg_corr60", corr))}


def build_feature_panel(
    market: WeeklyMarket, side: Mapping[str, pd.Series], *, d_star: float, corr_window: int,
    threshold: float | None = None,
) -> dict[str, pd.DataFrame]:
    """De featurematrix per symbool, op het raster van de markt, in `FEATURE_COLUMNS`-volgorde."""
    thr = ffd_threshold() if threshold is None else float(threshold)
    closes = pd.DataFrame({s: market.ohlcv[s]["close"] for s in market.symbols})
    mf = market_features(np.log(closes).diff(), window=corr_window)
    out: dict[str, pd.DataFrame] = {}
    for s in market.symbols:
        frame = _symbol_features(market.ohlcv[s], market.sigma_daily[s], market.funding[s],
                                 market.open_interest[s], d_star=d_star, threshold=thr)
        for name, panel in mf.items():
            frame[name] = panel[s]
        frame["side"] = side[s].reindex(market.grid)
        out[s] = frame[list(FEATURE_COLUMNS)]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_features.py tests/lookahead/test_weekly_features_causality.py -p no:randomly -q`
Expected: PASS. Faalt de causaliteitstest op één kolom, dan lekt die kolom: repareer de feature, niet de test. `yz_ratio` is de meest waarschijnlijke kandidaat — `get_yang_zhang_volatility` wordt op het afgekapte frame opnieuw berekend en moet causaal zijn; is hij dat niet, vervang hem door `volatility.realized.yang_zhang_variance` op een rollend venster.

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/features/weekly_set.py tests/unit/test_weekly_features.py tests/lookahead/test_weekly_features_causality.py
git commit -m "feat(features): the fixed weekly feature set on existing fracdiff, Kalman-OU and Yang-Zhang, proven causal row by row"
```

---

### Task 6: Van markt naar `MetaLabelDataset`

**Files:**
- Create: `src/tradebot/train/weekly_dataset.py`
- Test: `tests/unit/test_weekly_dataset.py`

**Interfaces:**
- Consumes: `breakout_side` (Taak 2); `label_triple_barrier`, `barrier_fill_returns`, `with_cost_aware_target`, `select_events`, `round_trip_cost` (Taak 3); `build_feature_panel` (Taak 5); `build_dataset(features, labels) -> MetaLabelDataset` (`train/meta_label.py`).
- Produces:
  - `WeeklyDataset` (frozen): `dataset: MetaLabelDataset`, `events: pd.DataFrame` (kolommen `symbol, event_bar, exit_bar, side, fill_return, sigma, target`, rij-voor-rij gelijk aan `dataset`), `grid`, `k`, `d_star`, `n_dropped_nan: dict[str, int]`.
  - `labeling_config(cfg: WeeklyMetaConfig) -> LabelingConfig`
  - `build_weekly_dataset(market, cfg, *, k, d_star, cost_rt, side_sign=1.0) -> WeeklyDataset`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_weekly_dataset.py
from __future__ import annotations

import numpy as np
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.train.weekly_dataset import build_weekly_dataset

CFG = weekly_meta_config()


@pytest.fixture(scope="module")
def built():
    market = synthetic_market(n=700)
    return market, build_weekly_dataset(market, CFG, k=2.0, d_star=0.4, cost_rt=0.0013)


def test_events_align_row_for_row_with_the_dataset(built) -> None:
    _, wd = built
    ds, ev = wd.dataset, wd.events
    assert len(ev) == len(ds) > 50
    assert np.array_equal(ev["event_bar"].to_numpy(), ds.event_bar)
    assert np.array_equal(ev["exit_bar"].to_numpy(), ds.exit_bar)
    assert np.array_equal(ev["side"].to_numpy(), ds.side)
    assert np.array_equal(ev["target"].to_numpy(), ds.target)
    assert np.array_equal(ds.features["side"].to_numpy(), ds.side)


def test_every_training_row_is_finite(built) -> None:
    _, wd = built
    assert np.isfinite(wd.dataset.features.to_numpy(dtype=float)).all()
    assert sum(wd.n_dropped_nan.values()) > 0  # de burn-in kost events, en dat wordt geteld


def test_the_target_is_net_of_costs(built) -> None:
    _, wd = built
    ev = wd.events
    assert ((ev["fill_return"] - 0.0013 > 0.0).astype(int) == ev["target"]).all()


def test_reversing_the_side_flips_the_events(built) -> None:
    market, wd = built
    rev = build_weekly_dataset(market, CFG, k=2.0, d_star=0.4, cost_rt=0.0013, side_sign=-1.0)
    a = wd.events.set_index(["symbol", "event_bar"])["side"]
    b = rev.events.set_index(["symbol", "event_bar"])["side"]
    common = a.index.intersection(b.index)
    assert len(common) > 0
    assert (a.loc[common] == -b.loc[common]).all()


def test_no_label_reaches_past_a_truncated_market(built) -> None:
    market, _ = built
    cut = market.truncate(market.grid[500])
    wd = build_weekly_dataset(cut, CFG, k=2.0, d_star=0.4, cost_rt=0.0013)
    assert int(wd.dataset.exit_bar.max()) < 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_dataset.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.train.weekly_dataset'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/train/weekly_dataset.py
"""Van gecertificeerde markt naar de gepoolde meta-label-dataset (spec §5-§7).

De volgorde van rijen is die van `train.meta_label.build_dataset`: symbolen
alfabetisch, binnen een symbool op eventbar. `events` volgt precies die
volgorde, zodat het tradeboek per rij de fill en de sigma terugvindt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.weekly_market import WeeklyMarket
from ..features.weekly_set import build_feature_panel
from ..labeling.barrier_fills import barrier_fill_returns, select_events, with_cost_aware_target
from ..labeling.breakout import breakout_side
from ..labeling.vol_barriers import BarrierLabels, label_triple_barrier
from ..schemas.config import LabelingConfig
from ..schemas.weekly_meta import WeeklyMetaConfig
from .meta_label import MetaLabelDataset, build_dataset

__all__ = ["WeeklyDataset", "build_weekly_dataset", "labeling_config"]


@dataclass(frozen=True)
class WeeklyDataset:
    dataset: MetaLabelDataset
    events: pd.DataFrame
    labels: dict[str, BarrierLabels]
    grid: pd.DatetimeIndex
    k: float
    d_star: float
    n_dropped_nan: dict[str, int]


def labeling_config(cfg: WeeklyMetaConfig) -> LabelingConfig:
    return LabelingConfig(profit_target_sigma=cfg.barrier_sigma,
                          stop_loss_sigma=cfg.barrier_sigma,
                          horizon_bars=cfg.horizon_bars, entry_lag_bars=1,
                          min_sigma_obs=60)


def build_weekly_dataset(
    market: WeeklyMarket,
    cfg: WeeklyMetaConfig,
    *,
    k: float,
    d_star: float,
    cost_rt: float,
    side_sign: float = 1.0,
) -> WeeklyDataset:
    """Events, 1:1-labels met barrièrefills, en features; rijen met een NaN-feature vallen af."""
    lab_cfg = labeling_config(cfg)
    sides = {s: side_sign * breakout_side(market.ohlcv[s]["close"], market.sigma_daily[s], k)
             for s in market.symbols}
    features = build_feature_panel(market, sides, d_star=d_star, corr_window=cfg.corr_window)
    labels: dict[str, BarrierLabels] = {}
    dropped: dict[str, int] = {}
    rows: list[pd.DataFrame] = []
    for s in sorted(market.symbols):
        o = market.ohlcv[s]
        raw = label_triple_barrier(
            o["high"].to_numpy(), o["low"].to_numpy(), o["close"].to_numpy(),
            market.sigma_daily[s].to_numpy(), sides[s].to_numpy(), lab_cfg)
        fills = barrier_fill_returns(raw, o["open"].to_numpy(), o["close"].to_numpy(),
                                     lab_cfg, stop_slippage_bps=cfg.stop_slippage_bps)
        lab = with_cost_aware_target(raw, fills, round_trip_cost=cost_rt)
        finite = np.isfinite(features[s].iloc[lab.event_idx].to_numpy(dtype=float)).all(axis=1)
        dropped[s] = int((~finite).sum())
        lab = select_events(lab, finite)
        labels[s] = lab
        rows.append(pd.DataFrame({
            "symbol": s, "event_bar": lab.event_idx, "exit_bar": lab.exit_idx,
            "side": lab.side, "fill_return": lab.realized_return, "sigma": lab.sigma,
            "target": lab.meta_label.astype(np.int64),
        }))
    dataset = build_dataset(features, labels)
    return WeeklyDataset(dataset=dataset, events=pd.concat(rows, ignore_index=True),
                         labels=labels, grid=market.grid, k=float(k),
                         d_star=float(d_star), n_dropped_nan=dropped)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_dataset.py -p no:randomly -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/train/weekly_dataset.py tests/unit/test_weekly_dataset.py
git commit -m "feat(train): the weekly meta-label dataset from breakout events, barrier fills and the fixed features"
```

---

### Task 7: Lichte modellen en een generieke purged walk-forward

**Files:**
- Modify: `src/tradebot/train/meta_label.py` (`FoldPredictions.extras`, nieuwe `walk_forward_fit_predict`, `walk_forward_predictions` delegeert)
- Create: `src/tradebot/train/light_models.py`
- Test: `tests/unit/test_light_models.py`
- Regressie: `tests/unit/test_meta_label.py` blijft ongewijzigd groen

**Interfaces:**
- Consumes: `MetaLabelDataset`, `purged_training_index(dataset, fold, *, embargo_bars)`, `WalkForwardCV` (`cv/walk_forward.py`).
- Produces:
  - `FoldPredictions.extras: Mapping[str, Any]` (default `{}`)
  - `walk_forward_fit_predict(dataset, cv, fit: Callable[[np.ndarray, np.ndarray], Any], *, n_bars: int, embargo_bars: int, target: np.ndarray | None = None) -> list[FoldPredictions]` — `fit(train_rows, labels)` geeft een model met `predict_proba(X)` en `feature_importance` (of `get_feature_importance()`), optioneel `fold_extras`.
  - `MODEL_KINDS = ("logreg", "forest", "ensemble")`
  - `inner_calibration_split(event_bar, exit_bar, *, fraction) -> (fit_pos, cal_pos)`
  - `fit_light_model(dataset, rows, labels, kind, cfg) -> CalibratedModel`; `CalibratedModel.predict_proba(X) -> (n, 2)`, `.feature_importance`, `.fold_extras` met `calibration_probability`, `calibration_target`, `train_events_per_week`, `n_fit`, `n_calibration`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_light_models.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.train.light_models import MODEL_KINDS, fit_light_model, inner_calibration_split
from tradebot.train.meta_label import MetaLabelDataset, shuffled_targets, walk_forward_fit_predict
from tradebot.utils.failfast import DataContractError

CFG = weekly_meta_config().model_copy(update={"forest_n_estimators": 60})


def _dataset(n: int = 900, signal: float = 1.2, seed: int = 3) -> MetaLabelDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    p = 1.0 / (1.0 + np.exp(-signal * x[:, 0]))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    ev = np.arange(n, dtype=np.int64)
    return MetaLabelDataset(
        features=pd.DataFrame(x, columns=["a", "b", "c"]), target=y,
        event_bar=ev, exit_bar=ev + 5, side=np.ones(n),
        symbol=np.array(["BTCUSDT"] * n, dtype=object), uniqueness=np.full(n, 0.2))


def _cv() -> WalkForwardCV:
    return WalkForwardCV(train_size=400, test_size=100, step=100, mode="anchored",
                         min_train=400, embargo_bars=6)


def _run(ds, kind, target=None):
    return walk_forward_fit_predict(
        ds, _cv(), lambda rows, y: fit_light_model(ds, rows, y, kind, CFG),
        n_bars=int(ds.exit_bar.max()) + 1, embargo_bars=6, target=target)


def test_the_inner_split_is_chronological_and_purged() -> None:
    ev = np.arange(100)
    ex = ev + 5
    fit, cal = inner_calibration_split(ev, ex, fraction=0.25)
    assert ev[cal].min() > ev[fit].max()
    assert (ex[fit] < ev[cal].min()).all()
    assert cal.size == 25


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_a_real_signal_is_found_out_of_sample(kind) -> None:
    ds = _dataset()
    folds = _run(ds, kind)
    rows = np.concatenate([f.row_index for f in folds])
    prob = np.concatenate([f.probability for f in folds])
    assert roc_auc_score(ds.target[rows], prob) > 0.65
    assert ((prob >= 0.0) & (prob <= 1.0)).all()


def test_the_probabilities_are_calibrated_on_average() -> None:
    ds = _dataset()
    folds = _run(ds, "ensemble")
    rows = np.concatenate([f.row_index for f in folds])
    prob = np.concatenate([f.probability for f in folds])
    assert prob.mean() == pytest.approx(ds.target[rows].mean(), abs=0.05)


def test_shuffled_labels_carry_no_signal() -> None:
    ds = _dataset()
    perm = shuffled_targets(ds, seed=1, n_replicates=1)[0]
    folds = _run(ds, "ensemble", target=perm)
    rows = np.concatenate([f.row_index for f in folds])
    prob = np.concatenate([f.probability for f in folds])
    assert 0.40 < roc_auc_score(perm[rows], prob) < 0.60


def test_a_non_permutation_target_is_refused() -> None:
    ds = _dataset()
    with pytest.raises(DataContractError, match="permutatie"):
        _run(ds, "logreg", target=np.ones_like(ds.target))


def test_fold_extras_carry_the_calibration_set() -> None:
    ds = _dataset()
    folds = _run(ds, "ensemble")
    extras = folds[0].extras
    assert extras["calibration_probability"].size == extras["n_calibration"] > 0
    assert extras["train_events_per_week"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_light_models.py -p no:randomly -q`
Expected: FAIL, `ImportError: cannot import name 'walk_forward_fit_predict'`

- [ ] **Step 3a: Refactor `train/meta_label.py` zonder gedragswijziging**

Wijzig de import `from dataclasses import dataclass` naar `from dataclasses import dataclass, field`, voeg `from collections.abc import Callable` toe aan de bestaande `collections.abc`-import, en zet `"walk_forward_fit_predict"` in `__all__`. Voeg aan `FoldPredictions` als laatste veld toe:

```python
    feature_importance: np.ndarray
    #: Wat de fit naast zijn kansen meegeeft (bijv. de kalibratieset). Leeg voor CatBoost.
    extras: Mapping[str, Any] = field(default_factory=dict)
```

Vervang de body van `walk_forward_predictions` en voeg de generieke functie en een helper toe:

```python
def _importance(model: Any) -> np.ndarray:
    if hasattr(model, "get_feature_importance"):
        return np.asarray(model.get_feature_importance(), dtype=np.float64)
    return np.asarray(model.feature_importance, dtype=np.float64)


def walk_forward_fit_predict(
    dataset: MetaLabelDataset,
    cv: WalkForwardCV,
    fit: Callable[[np.ndarray, np.ndarray], Any],
    *,
    n_bars: int,
    embargo_bars: int,
    target: np.ndarray | None = None,
) -> list[FoldPredictions]:
    """Purged walk-forward met een willekeurig model: `fit(train_rows, labels)`.

    Dezelfde purge, hetzelfde embargo en dezelfde permutatie-eis als
    `walk_forward_predictions`; alleen de fit is ingeplugd.
    """
    labels = dataset.target if target is None else np.asarray(target)
    require(
        labels.shape == dataset.target.shape,
        "Een vervangend doel met een andere vorm dan het echte doel.",
        DataContractError,
    )
    require(
        int(labels.sum()) == int(dataset.target.sum()),
        "Het vervangende doel is geen permutatie van het echte doel. Deze "
        "ingang bestaat alleen voor de negatieve controle; hij is geen route "
        "om dit model iets anders te laten voorspellen.",
        DataContractError, n_positive=int(labels.sum()),
        n_positive_expected=int(dataset.target.sum()),
    )
    out: list[FoldPredictions] = []
    for fold in cv.split(n_bars):
        train_rows, purge = purged_training_index(
            dataset, fold, embargo_bars=embargo_bars)
        test_rows = np.flatnonzero(
            (dataset.event_bar >= int(fold.test_idx[0]))
            & (dataset.event_bar <= int(fold.test_idx[-1])))
        if train_rows.size == 0 or test_rows.size == 0:
            continue
        model = fit(train_rows, labels)
        probability = model.predict_proba(
            dataset.features.iloc[test_rows].to_numpy(dtype="float64"))[:, 1]
        out.append(FoldPredictions(
            fold_id=fold.fold_id, row_index=test_rows,
            probability=probability, target=labels[test_rows],
            uniqueness=dataset.uniqueness[test_rows],
            n_train=int(train_rows.size), purge=purge,
            feature_importance=_importance(model),
            extras=dict(getattr(model, "fold_extras", {})),
        ))
    require(
        bool(out),
        "Geen enkele fold leverde een voorspelling op.",
        DataContractError, n_bars=n_bars,
    )
    return out


def walk_forward_predictions(
    dataset: MetaLabelDataset,
    cv: WalkForwardCV,
    spec: MetaLabelSpec,
    cfg: MetaLabelConfig,
    *,
    n_bars: int,
    embargo_bars: int,
    target: np.ndarray | None = None,
) -> list[FoldPredictions]:
    """Purged walk-forward over de bar-as; één CatBoost-fit per fold.

    (Docstring van de oorspronkelijke functie hier ongewijzigd laten staan.)
    """
    def fit(rows: np.ndarray, labels: np.ndarray) -> Any:
        return fit_secondary_model(dataset.features.iloc[rows], labels[rows],
                                   dataset.uniqueness[rows], spec, cfg)

    return walk_forward_fit_predict(dataset, cv, fit, n_bars=n_bars,
                                    embargo_bars=embargo_bars, target=target)
```

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_meta_label.py -p no:randomly -q`
Expected: PASS, exact hetzelfde aantal tests als vóór de wijziging.

- [ ] **Step 3b: Implement `train/light_models.py`**

```python
# src/tradebot/train/light_models.py
"""Lichte, niet-getunede secondary models met Platt-kalibratie (spec §8).

Twee modellen, instellingen vast in `conf/model/weekly_meta.yaml`:
een L2-logistische regressie en een ondiepe random forest met
`max_features=1` en `max_samples` = gemiddelde uniqueness (AFML hoofdstuk 4/6).
Het ensemble middelt hun GEKALIBREERDE kansen.

Kalibratie gebeurt op het laatste deel van het trainvenster, chronologisch en
gepurged: geen fitrij waarvan het label de kalibratieset in loopt. Een K-fold
kalibratie zou toekomst binnen het trainvenster gebruiken.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..schemas.weekly_meta import WeeklyMetaConfig
from ..utils.failfast import DataContractError, require
from .meta_label import MetaLabelDataset

__all__ = ["MODEL_KINDS", "CalibratedModel", "fit_light_model", "inner_calibration_split"]

ModelKind = Literal["logreg", "forest", "ensemble"]
MODEL_KINDS: tuple[ModelKind, ...] = ("logreg", "forest", "ensemble")
_EPS = 1e-6
_DAYS_PER_WEEK = 7.0


def inner_calibration_split(
    event_bar: np.ndarray, exit_bar: np.ndarray, *, fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(fit-, kalibratie-)posities binnen de trainrijen: laatste `fraction` op tijd, gepurged."""
    order = np.argsort(event_bar, kind="stable")
    n_cal = max(1, int(round(fraction * order.size)))
    cal = order[-n_cal:]
    cal_start = int(event_bar[cal].min())
    fit = order[:-n_cal]
    return fit[exit_bar[fit] < cal_start], cal


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


@dataclass
class _Platt:
    model: LogisticRegression

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _fit_platt(p_raw: np.ndarray, y: np.ndarray, w: np.ndarray) -> _Platt:
    require(np.unique(y).size == 2,
            "Een kalibratieset met één klasse; er valt geen Platt-schaling te schatten.",
            DataContractError)
    model = LogisticRegression(C=1e6, max_iter=1000)  # praktisch ongestraft
    model.fit(_logit(p_raw).reshape(-1, 1), y, sample_weight=w)
    return _Platt(model)


@dataclass
class CalibratedModel:
    kind: ModelKind
    members: list[Any]
    calibrators: list[_Platt]
    feature_importance: np.ndarray
    fold_extras: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = [cal(m.predict_proba(x)[:, 1])
                 for m, cal in zip(self.members, self.calibrators, strict=True)]
        p = np.mean(probs, axis=0)
        return np.column_stack([1.0 - p, p])


def _fit_member(kind: str, x: np.ndarray, y: np.ndarray, w: np.ndarray,
                cfg: WeeklyMetaConfig) -> tuple[Any, np.ndarray]:
    if kind == "logreg":
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=cfg.logreg_c, max_iter=2000))
        model.fit(x, y, logisticregression__sample_weight=w)
        coef = np.abs(model.named_steps["logisticregression"].coef_[0])
        return model, coef / coef.sum()
    model = RandomForestClassifier(
        n_estimators=cfg.forest_n_estimators, max_depth=cfg.forest_max_depth,
        max_features=1, min_samples_leaf=cfg.forest_min_samples_leaf,
        max_samples=float(np.clip(w.mean(), 0.05, 1.0)),
        class_weight="balanced_subsample", bootstrap=True,
        random_state=cfg.seed, n_jobs=1)
    model.fit(x, y, sample_weight=w)
    return model, np.asarray(model.feature_importances_, dtype=np.float64)


def fit_light_model(
    dataset: MetaLabelDataset, rows: np.ndarray, labels: np.ndarray,
    kind: ModelKind, cfg: WeeklyMetaConfig,
) -> CalibratedModel:
    """Fit op `rows`, kalibreer op hun laatste deel; ziet niets buiten `rows`."""
    require(kind in MODEL_KINDS, "Onbekend modeltype.", DataContractError, kind=kind)
    rows = np.asarray(rows, dtype=np.int64)
    fit_pos, cal_pos = inner_calibration_split(
        dataset.event_bar[rows], dataset.exit_bar[rows], fraction=cfg.calibration_fraction)
    fit_rows, cal_rows = rows[fit_pos], rows[cal_pos]
    x = dataset.features.to_numpy(dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    w = dataset.uniqueness
    require(np.unique(y[fit_rows]).size == 2,
            "Een fitvenster met één klasse.", DataContractError)
    members, cals, imps = [], [], []
    for member in (("logreg", "forest") if kind == "ensemble" else (kind,)):
        model, imp = _fit_member(member, x[fit_rows], y[fit_rows], w[fit_rows], cfg)
        cals.append(_fit_platt(model.predict_proba(x[cal_rows])[:, 1], y[cal_rows], w[cal_rows]))
        members.append(model)
        imps.append(imp)
    out = CalibratedModel(kind, members, cals, np.mean(imps, axis=0))
    ev = dataset.event_bar[rows]
    weeks = (int(ev.max()) - int(ev.min()) + 1) / _DAYS_PER_WEEK
    out.fold_extras = {
        "calibration_probability": out.predict_proba(x[cal_rows])[:, 1],
        "calibration_target": y[cal_rows],
        "train_events_per_week": float(rows.size / weeks),
        "n_fit": int(fit_rows.size),
        "n_calibration": int(cal_rows.size),
    }
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_light_models.py tests/unit/test_meta_label.py -p no:randomly -q`
Expected: PASS. Faalt de kalibratietest op het ensemble met een afwijking boven 0,05: meld het getal, verhoog de tolerantie niet.

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/train/meta_label.py src/tradebot/train/light_models.py tests/unit/test_light_models.py
git commit -m "feat(train): light LR/RF models with purged Platt calibration, through a generic purged walk-forward"
```

---

### Task 8: Kansrekening voor een 1:1-barrière

**Files:**
- Create: `src/tradebot/risk/binary_kelly.py`
- Test: `tests/unit/test_binary_kelly.py`

**Interfaces:**
- Consumes: `circular_block_indices(n_obs, block_length, n_boot, rng)` (`validation/inference.py`).
- Produces:
  - `first_passage_up_probability(mu: float, sigma: float, barrier: float) -> float`
  - `no_exit_probability(a: float, n_terms: int = 50) -> float`
  - `break_even_probability(barrier: float, cost: float) -> float`
  - `kelly_fraction_binary(p: float, barrier: float, cost: float) -> float`
  - `posterior_lower_probability(p_hat: float, n: float, quantile: float) -> float`
  - `correlation_scale(k_same: int, rho: float) -> float`
  - `drawdown_probability(kelly_multiple: float, drawdown: float) -> float`
  - `monte_carlo_drawdown_probability(r_multiples: np.ndarray, *, risk_fraction: float, n_trades: int, drawdown: float, block_length: int, n_paths: int, seed: int) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_binary_kelly.py
from __future__ import annotations

import math

import numpy as np
import pytest

from tradebot.risk.binary_kelly import (
    break_even_probability,
    correlation_scale,
    drawdown_probability,
    first_passage_up_probability,
    kelly_fraction_binary,
    monte_carlo_drawdown_probability,
    no_exit_probability,
    posterior_lower_probability,
)


def test_without_drift_the_barrier_is_a_coin() -> None:
    assert first_passage_up_probability(0.0, 0.03, 0.07) == pytest.approx(0.5)
    p = first_passage_up_probability(0.004, 0.03, 0.07)
    assert p + first_passage_up_probability(-0.004, 0.03, 0.07) == pytest.approx(1.0)
    assert p > 0.5


def test_first_passage_matches_a_simulated_brownian_motion() -> None:
    rng = np.random.default_rng(0)
    mu, sigma, b, dt = 0.5, 1.0, 0.5, 1e-3
    n_paths, n_steps = 4000, 20000
    x = np.zeros(n_paths)
    hit = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    for _ in range(n_steps):
        x[alive] += mu * dt + sigma * math.sqrt(dt) * rng.normal(size=int(alive.sum()))
        up, down = alive & (x >= b), alive & (x <= -b)
        hit[up] = 1.0
        alive &= ~(up | down)
        if not alive.any():
            break
    assert hit.mean() == pytest.approx(first_passage_up_probability(mu, sigma, b), abs=0.03)


def test_the_vertical_barrier_share_of_the_spec() -> None:
    assert no_exit_probability(math.sqrt(0.5)) == pytest.approx(0.108, abs=1e-3)
    assert no_exit_probability(0.5) < no_exit_probability(1.0) < no_exit_probability(3.0) <= 1.0


def test_break_even_and_kelly_meet_at_zero() -> None:
    p_be = break_even_probability(0.09, 0.0025)
    assert p_be == pytest.approx(0.5 + 0.0025 / 0.18)
    assert kelly_fraction_binary(p_be, 0.09, 0.0025) == pytest.approx(0.0, abs=1e-12)
    assert kelly_fraction_binary(0.6, 0.09, 0.0) == pytest.approx(0.2)
    assert kelly_fraction_binary(0.4, 0.09, 0.0) == 0.0


def test_the_posterior_lower_bound_shrinks_toward_the_estimate_with_data() -> None:
    lo_small = posterior_lower_probability(0.58, 10, 0.25)
    lo_big = posterior_lower_probability(0.58, 2000, 0.25)
    assert lo_small < lo_big < 0.58
    assert posterior_lower_probability(0.58, 0, 0.25) == pytest.approx(0.25)


def test_the_equicorrelated_kelly_correction() -> None:
    assert correlation_scale(1, 0.75) == 1.0
    assert correlation_scale(3, 0.75) == pytest.approx(1.0 / 2.5)
    assert correlation_scale(3, -0.2) == 1.0


def test_the_fractional_kelly_drawdown_formula() -> None:
    assert drawdown_probability(0.25, 0.25) == pytest.approx(0.75 ** 7)
    assert drawdown_probability(0.5, 0.25) == pytest.approx(0.75 ** 3)


def test_monte_carlo_drawdown_is_certain_or_impossible_at_the_extremes() -> None:
    losses = np.full(50, -1.0)
    wins = np.full(50, 1.0)
    kw = dict(risk_fraction=0.1, n_trades=10, drawdown=0.25, block_length=2, n_paths=500, seed=1)
    assert monte_carlo_drawdown_probability(losses, **kw) == 1.0
    assert monte_carlo_drawdown_probability(wins, **kw) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_binary_kelly.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.risk.binary_kelly'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/risk/binary_kelly.py
"""Kansrekening voor een 1:1-barrière (spec §10, §17.7).

Een trade met symmetrische barrières op ±b is een binaire weddenschap. Zonder
drift is de kans om eerst boven uit te komen precies 1/2; met drift mu en
volatiliteit sigma is zij 1 / (1 + exp(-2 mu b / sigma^2)). Break-even, Kelly,
de onzekerheid in p, de correlatie tussen gelijktijdige trades en de kans op
een drawdown volgen daaruit in gesloten vorm; de Monte Carlo controleert de
laatste op de echte tradeverdeling.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.special import expit

from ..utils.failfast import DataContractError, require
from ..validation.inference import circular_block_indices

__all__ = [
    "break_even_probability", "correlation_scale", "drawdown_probability",
    "first_passage_up_probability", "kelly_fraction_binary",
    "monte_carlo_drawdown_probability", "no_exit_probability",
    "posterior_lower_probability",
]


def first_passage_up_probability(mu: float, sigma: float, barrier: float) -> float:
    """P(+b eerst) voor een Brownse beweging met drift, barrières op ±b."""
    require(sigma > 0.0 and barrier > 0.0, "sigma en barrier moeten positief zijn.",
            DataContractError, sigma=sigma, barrier=barrier)
    return float(expit(2.0 * mu * barrier / sigma ** 2))


def no_exit_probability(a: float, n_terms: int = 50) -> float:
    """P(sup_{s<=1} |W_s| < a) voor standaard-Brownse beweging (reeksformule)."""
    require(a > 0.0, "a moet positief zijn.", DataContractError, a=a)
    k = np.arange(n_terms)
    terms = ((-1.0) ** k) / (2 * k + 1) * np.exp(-((2 * k + 1) ** 2) * math.pi ** 2 / (8.0 * a * a))
    return float(min(1.0, 4.0 / math.pi * terms.sum()))


def break_even_probability(barrier: float, cost: float) -> float:
    """p_be = 1/2 + c / (2b): winst b - c, verlies b + c."""
    require(barrier > 0.0 and 0.0 <= cost < barrier, "Nodig: 0 <= cost < barrier.",
            DataContractError, barrier=barrier, cost=cost)
    return 0.5 + cost / (2.0 * barrier)


def kelly_fraction_binary(p: float, barrier: float, cost: float) -> float:
    """Kelly-fractie van het vermogen dat verloren gaat bij de stop; 0 zonder edge."""
    require(0.0 <= p <= 1.0, "p buiten [0, 1].", DataContractError, p=p)
    require(barrier > 0.0 and 0.0 <= cost < barrier, "Nodig: 0 <= cost < barrier.",
            DataContractError, barrier=barrier, cost=cost)
    eps = cost / barrier
    f = (p * (1.0 - eps) - (1.0 - p) * (1.0 + eps)) / ((1.0 - eps) * (1.0 + eps))
    return float(max(f, 0.0))


def posterior_lower_probability(p_hat: float, n: float, quantile: float) -> float:
    """Kwantiel van Beta(1 + p_hat n, 1 + (1 - p_hat) n): uniforme prior plus n waarnemingen."""
    require(0.0 <= p_hat <= 1.0 and n >= 0.0 and 0.0 < quantile < 1.0,
            "Ongeldige posterior-invoer.", DataContractError, p_hat=p_hat, n=n, q=quantile)
    return float(stats.beta.ppf(quantile, 1.0 + p_hat * n, 1.0 + (1.0 - p_hat) * n))


def correlation_scale(k_same: int, rho: float) -> float:
    """Multivariate Kelly bij k gelijk-gecorreleerde, gelijk-renderende bets: 1 / (1 + (k-1) rho)."""
    require(k_same >= 1, "Ten minste één positie.", DataContractError, k_same=k_same)
    return 1.0 / (1.0 + (k_same - 1) * max(float(rho), 0.0))


def drawdown_probability(kelly_multiple: float, drawdown: float) -> float:
    """P(ooit onder (1 - drawdown) maal het huidige niveau) onder fractie-Kelly: x^(2/c - 1).

    Geldt vanaf een gegeven moment, in een continu model met juist geschatte
    edge. Over een horizon met veel nieuwe pieken is de kans op minstens één
    zo'n drawdown hoger; daarvoor is `monte_carlo_drawdown_probability`.
    """
    require(0.0 < kelly_multiple < 2.0 and 0.0 < drawdown < 1.0,
            "Nodig: 0 < c < 2 en 0 < drawdown < 1.", DataContractError)
    return float((1.0 - drawdown) ** (2.0 / kelly_multiple - 1.0))


def monte_carlo_drawdown_probability(
    r_multiples: np.ndarray,
    *,
    risk_fraction: float,
    n_trades: int,
    drawdown: float,
    block_length: int,
    n_paths: int,
    seed: int,
) -> float:
    """Kans op een maximale drawdown >= `drawdown` binnen `n_trades` trades.

    `r_multiples` is het rendement per eenheid risico, in tijdvolgorde (winst
    ~ +1, verlies ~ -1). Circulaire blokbootstrap uit `validation.inference`,
    zodat trades die dicht op elkaar liggen hun samenhang houden.
    """
    r = np.asarray(r_multiples, dtype=np.float64)
    require(r.size > 0 and bool(np.isfinite(r).all()), "Lege of niet-eindige trades.",
            DataContractError)
    require(1 <= n_trades <= r.size, "De horizon vraagt meer trades dan er gemeten zijn.",
            DataContractError, n_trades=n_trades, n_measured=int(r.size))
    rng = np.random.default_rng(seed)
    idx = circular_block_indices(r.size, max(1, int(block_length)), int(n_paths), rng)
    draws = r[idx[:, :n_trades]]
    log_eq = np.cumsum(np.log1p(np.clip(risk_fraction * draws, -0.999999, None)), axis=1)
    peak = np.maximum.accumulate(np.maximum(log_eq, 0.0), axis=1)
    worst = 1.0 - np.exp((log_eq - peak).min(axis=1))
    return float((worst >= drawdown).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_binary_kelly.py -p no:randomly -q`
Expected: PASS (8 tests). De Brownse-bewegingstest duurt enkele seconden.

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/risk/binary_kelly.py tests/unit/test_binary_kelly.py
git commit -m "feat(risk): closed-form probability for a 1:1 barrier -- the null, break-even, Kelly, posterior, correlation, drawdown"
```

---

### Task 9: Het tradeboek met exits op de barrière

**Files:**
- Create: `src/tradebot/backtest/barrier_book.py`
- Test: `tests/unit/test_barrier_book.py`

**Interfaces:**
- Consumes: `RiskEngine.decide(desired_exposure, market_state, risk_state) -> RiskDecision` (`risk/engine.py`), `MarketState`, `RiskState` (`risk/contract.py`), `square_root_impact(*, order_notional, adv_notional, sigma_daily, params) -> ImpactEstimate` en `ImpactParams` (`execution/impact_model.py`), `kelly_fraction_binary`, `correlation_scale` (Taak 8), `WeeklyMarket` (Taak 4).
- Produces:
  - `CANDIDATE_COLUMNS = ("symbol", "entry_bar", "exit_bar", "side", "fill_return", "barrier", "p", "p_low", "p_trade")`
  - `SizingRule(mode: Literal["kelly", "fixed"], kelly_multiple: float, fixed_risk_fraction: float, resize_band: float, cost_rt: float)`
  - `BookInputs(market: WeeklyMarket, avg_corr: pd.DataFrame, cost_rate: float, impact: ImpactParams | None)` — `cost_rate` is per kant (taker + halve spread).
  - `BookResult(returns: pd.Series, equity: pd.Series, trades: pd.DataFrame, total_fees: float, total_funding: float, total_impact: float, total_pnl: float, n_resizes: int, halted: bool)`
  - `run_barrier_book(candidates: pd.DataFrame, inputs: BookInputs, risk: RiskEngine, sizing: SizingRule, *, equity0: float) -> BookResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_barrier_book.py
"""Het boek: juiste P&L bij een fill op de barrière, kosten, funding, filters, en geen blik vooruit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.backtest.barrier_book import BookInputs, SizingRule, run_barrier_book
from tradebot.risk.engine import RiskEngine
from tradebot.schemas.config import RiskConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
RISK = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)


def _market():
    m = synthetic_market(n=200)
    # Lage volatiliteit en diepe ADV: de risicolaag bindt niet, zodat de rekensom zichtbaar blijft.
    m.sigma_annual.loc[:, :] = 0.05
    m.sigma_daily.loc[:, :] = 0.05 / np.sqrt(365.0)
    m.adv_usd.loc[:, :] = 1e12
    m.funding.loc[:, :] = 0.0
    return m


def _inputs(m, cost_rate=0.0):
    corr = pd.DataFrame(0.0, index=m.grid, columns=list(m.symbols))
    return BookInputs(market=m, avg_corr=corr, cost_rate=cost_rate, impact=None)


def _cand(**over) -> pd.DataFrame:
    row = dict(symbol="BTCUSDT", entry_bar=100, exit_bar=104, side=1.0, fill_return=0.05,
               barrier=0.05, p=0.6, p_low=0.58, p_trade=0.52)
    row.update(over)
    return pd.DataFrame([row])


FIXED = SizingRule(mode="fixed", kelly_multiple=0.25, fixed_risk_fraction=0.01,
                   resize_band=0.25, cost_rt=0.0)


def test_a_target_hit_books_exactly_the_barrier_move() -> None:
    m = _market()
    res = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=100_000.0)
    t = res.trades.iloc[0]
    entry = m.ohlcv["BTCUSDT"]["close"].iloc[100]
    assert t["entry_price"] == pytest.approx(entry)
    assert res.equity.iloc[-1] - 100_000.0 == pytest.approx(t["qty"] * entry * 0.05)


def test_costs_are_charged_on_both_legs() -> None:
    m = _market()
    free = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=100_000.0)
    paid = run_barrier_book(_cand(), _inputs(m, cost_rate=0.00065), RiskEngine(RISK), FIXED,
                            equity0=100_000.0)
    t = paid.trades.iloc[0]
    entry = t["entry_price"]
    fill = entry * 1.05
    expected = abs(t["qty"]) * (entry + fill) * 0.00065
    assert paid.total_fees == pytest.approx(expected)
    assert free.equity.iloc[-1] - paid.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_a_long_pays_positive_funding() -> None:
    m = _market()
    m.funding.loc[m.grid[102], "BTCUSDT"] = 0.001
    res = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=100_000.0)
    assert res.total_funding > 0.0


def test_the_books_balance() -> None:
    m = _market()
    m.funding.loc[:, :] = 1e-4
    cands = pd.concat([_cand(), _cand(symbol="ETHUSDT", side=-1.0, fill_return=-0.05,
                                      entry_bar=101, exit_bar=103)], ignore_index=True)
    res = run_barrier_book(cands, _inputs(m, cost_rate=0.00065), RiskEngine(RISK), FIXED,
                           equity0=100_000.0)
    change = res.equity.iloc[-1] - 100_000.0
    assert change == pytest.approx(res.total_pnl - res.total_fees - res.total_funding
                                   - res.total_impact, rel=1e-9)


def test_the_filter_and_the_one_position_per_symbol_rule() -> None:
    m = _market()
    below = run_barrier_book(_cand(p=0.50), _inputs(m), RiskEngine(RISK), FIXED, equity0=1e5)
    assert below.trades.empty
    overlap = pd.concat([_cand(), _cand(entry_bar=102, exit_bar=106)], ignore_index=True)
    res = run_barrier_book(overlap, _inputs(m), RiskEngine(RISK), FIXED, equity0=1e5)
    assert len(res.trades) == 1


def test_simultaneous_same_direction_trades_are_scaled_for_correlation() -> None:
    m = _market()
    inputs = _inputs(m)
    inputs.avg_corr.loc[:, :] = 0.75
    pair = pd.concat([_cand(), _cand(symbol="ETHUSDT")], ignore_index=True)
    res = run_barrier_book(pair, inputs, RiskEngine(RISK), FIXED, equity0=1e5)
    solo = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=1e5)
    assert res.trades["risk_fraction"].iloc[0] == pytest.approx(
        solo.trades["risk_fraction"].iloc[0] / 1.75)


def test_the_book_never_reads_an_exit_before_it_happens() -> None:
    m = _market()
    a = run_barrier_book(_cand(exit_bar=150, fill_return=0.05), _inputs(m), RiskEngine(RISK),
                         FIXED, equity0=1e5)
    b = run_barrier_book(_cand(exit_bar=160, fill_return=-0.05), _inputs(m), RiskEngine(RISK),
                         FIXED, equity0=1e5)
    pd.testing.assert_series_equal(a.returns.iloc[:150], b.returns.iloc[:150])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_barrier_book.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.backtest.barrier_book'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/backtest/barrier_book.py
"""Een tradeboek met stops en take-profits bij de exchange (spec §10-§12, §17.6).

WAAROM NIET `backtest/engine.py`
================================
De authoritative engine vult orders op de prijs van een bar. Een 1:1-trade met
stops bij de exchange sluit op het BARRIÈRENIVEAU, midden in een bar; dat kan de
engine niet uitdrukken zonder zijn fillmodel te veranderen. Dit boek gebruikt
wel dezelfde soevereine risicolaag (`RiskEngine.decide`), dezelfde
kostenparameters en `square_root_impact`. `tests/unit/test_barrier_book.py`
bewijst de boekhouding (P&L, kosten, funding sluiten op de equity) en de
causaliteit (een exit wordt pas op zijn exitbar gelezen).

DE VOLGORDE OP BAR d
====================
1. Mark-to-market van close d-1 naar close d, of naar de fill als de trade op d
   sluit. Funding over bar d op de positie van d-1 (long betaalt positieve funding).
2. Nieuwe trades met entry op d (event op d-1): toegelaten als p >= p_trade en
   het symbool geen open positie heeft; grootte uit binaire Kelly of een vaste
   risicofractie, maal de correlatiecorrectie.
3. Het gewenste boek gaat door `RiskEngine.decide`; toegestane exposures worden
   posities, tegen kosten. Bestaande posities worden alleen herschaald buiten
   `resize_band`, of naar nul.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..data.weekly_market import WeeklyMarket
from ..execution.impact_model import ImpactParams, square_root_impact
from ..risk.binary_kelly import correlation_scale, kelly_fraction_binary
from ..risk.contract import MarketState, RiskState
from ..risk.engine import RiskEngine
from ..utils.failfast import DataContractError, require

__all__ = ["CANDIDATE_COLUMNS", "BookInputs", "BookResult", "SizingRule", "run_barrier_book"]

CANDIDATE_COLUMNS = ("symbol", "entry_bar", "exit_bar", "side", "fill_return",
                     "barrier", "p", "p_low", "p_trade")


@dataclass(frozen=True)
class SizingRule:
    mode: Literal["kelly", "fixed"]
    kelly_multiple: float
    fixed_risk_fraction: float
    resize_band: float
    #: De round trip die Kelly als kosten ziet (fractie).
    cost_rt: float


@dataclass(frozen=True)
class BookInputs:
    market: WeeklyMarket
    avg_corr: pd.DataFrame
    #: Kosten per kant: taker + halve spread, als fractie van het notioneel.
    cost_rate: float
    impact: ImpactParams | None


@dataclass(frozen=True)
class BookResult:
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    total_fees: float
    total_funding: float
    total_impact: float
    total_pnl: float
    n_resizes: int
    halted: bool


class _Ledger:
    def __init__(self, equity: float) -> None:
        self.equity = equity
        self.fees = self.funding = self.impact = self.pnl = 0.0

    def trade_cost(self, notional: float, inputs: BookInputs, adv: float, sigma: float) -> None:
        fee = abs(notional) * inputs.cost_rate
        imp = 0.0
        if inputs.impact is not None and abs(notional) > 0.0:
            imp = square_root_impact(order_notional=abs(notional), adv_notional=adv,
                                     sigma_daily=sigma, params=inputs.impact).cost(abs(notional))
        self.fees += fee
        self.impact += imp
        self.equity -= fee + imp


def _base_fraction(row: pd.Series, sizing: SizingRule) -> float:
    if sizing.mode == "fixed":
        return sizing.fixed_risk_fraction
    return sizing.kelly_multiple * kelly_fraction_binary(float(row.p_low), float(row.barrier),
                                                         sizing.cost_rt)


def run_barrier_book(
    candidates: pd.DataFrame,
    inputs: BookInputs,
    risk: RiskEngine,
    sizing: SizingRule,
    *,
    equity0: float,
) -> BookResult:
    """Speel de kandidaten af in tijdvolgorde en geef het dagelijkse boekrendement."""
    missing = sorted(set(CANDIDATE_COLUMNS) - set(candidates.columns))
    require(not missing, "Kandidaten missen kolommen.", DataContractError, missing=missing)
    m = inputs.market
    cols = {s: j for j, s in enumerate(m.symbols)}
    close = pd.DataFrame({s: m.ohlcv[s]["close"] for s in m.symbols}).to_numpy(np.float64)
    fund = m.funding.reindex(columns=list(m.symbols)).to_numpy(np.float64)
    sig_a = m.sigma_annual.reindex(columns=list(m.symbols)).to_numpy(np.float64)
    sig_d = m.sigma_daily.reindex(columns=list(m.symbols)).to_numpy(np.float64)
    adv = m.adv_usd.reindex(columns=list(m.symbols)).to_numpy(np.float64)
    corr = inputs.avg_corr.reindex(index=m.grid, columns=list(m.symbols)).to_numpy(np.float64)
    cand = candidates.reset_index(drop=True)
    by_entry: dict[int, list[int]] = {}
    for i, e in enumerate(cand["entry_bar"].to_numpy(dtype=np.int64)):
        by_entry.setdefault(int(e), []).append(i)

    n = len(m.grid)
    book = _Ledger(float(equity0))
    state = RiskState(equity=book.equity, high_water_mark=book.equity,
                      day_start_equity=book.equity)
    qty: dict[str, float] = {}
    open_row: dict[str, int] = {}
    entry_px: dict[str, float] = {}
    trades: list[dict] = []
    returns = np.zeros(n)
    equity = np.full(n, float(equity0))
    resizes = 0

    for d in range(1, n):
        start = book.equity
        # 1. Mark-to-market, funding en exits op de barrière of de verticale close.
        for sym in list(qty):
            j, q = cols[sym], qty[sym]
            prev = close[d - 1, j]
            row = cand.loc[open_row[sym]]
            if int(row.exit_bar) == d:
                fill = entry_px[sym] * (1.0 + float(row.fill_return) * float(row.side))
                move = q * (fill - prev)
                book.trade_cost(q * fill, inputs, adv[d, j], sig_d[d, j])
                del qty[sym], open_row[sym], entry_px[sym]
            else:
                move = q * (close[d, j] - prev)
            book.pnl += move
            book.equity += move
            if np.isfinite(fund[d, j]):
                paid = q * prev * fund[d, j]
                book.funding += paid
                book.equity -= paid
        if state.halted:
            returns[d] = (book.equity - start) / start
            equity[d] = book.equity
            continue

        # 2. Het gewenste boek: bestaande posities plus toegelaten nieuwe trades.
        desired = {s: qty[s] * close[d, cols[s]] / book.equity for s in qty}
        pending: dict[str, tuple[int, float]] = {}
        chosen = [i for i in by_entry.get(d, [])
                  if cand.loc[i].symbol not in qty
                  and float(cand.loc[i].p) >= float(cand.loc[i].p_trade)
                  and np.isfinite(adv[d, cols[cand.loc[i].symbol]])
                  and np.isfinite(sig_a[d, cols[cand.loc[i].symbol]])]
        seen: set[str] = set()
        chosen = [i for i in chosen if not (cand.loc[i].symbol in seen or seen.add(cand.loc[i].symbol))]
        for i in chosen:
            row = cand.loc[i]
            same = (sum(1 for q in qty.values() if np.sign(q) == row.side)
                    + sum(1 for k in chosen if cand.loc[k].side == row.side))
            rho = corr[d - 1, cols[row.symbol]]
            f = _base_fraction(row, sizing) * correlation_scale(same, rho if np.isfinite(rho) else 1.0)
            if f <= 0.0:
                continue
            desired[row.symbol] = float(np.clip(row.side * f / float(row.barrier), -1.0, 1.0))
            pending[row.symbol] = (i, f)
        if desired:
            market = MarketState(asof_ts=m.grid[d],
                                 sigma_hat={s: float(sig_a[d, cols[s]]) for s in desired},
                                 adv_usd={s: float(adv[d, cols[s]]) for s in desired})
            state = dataclasses.replace(state, equity=book.equity,
                                        high_water_mark=max(state.high_water_mark, book.equity),
                                        day_start_equity=start)
            decision = risk.decide(desired, market, state)
            state = decision.risk_state_out
            # 3. Toegestane exposures worden posities.
            for sym in sorted(set(desired) | set(qty)):
                j = cols[sym]
                px = close[d, j]
                target_w = float(decision.permitted_exposure.get(sym, 0.0))
                cur_q = qty.get(sym, 0.0)
                if sym in qty and sym not in pending and target_w != 0.0:
                    cur_w = cur_q * px / book.equity
                    if abs(target_w - cur_w) <= sizing.resize_band * abs(cur_w):
                        continue
                target_q = target_w * book.equity / px
                delta = target_q - cur_q
                if delta == 0.0:
                    continue
                book.trade_cost(delta * px, inputs, adv[d, j], sig_d[d, j])
                if target_q == 0.0:
                    qty.pop(sym, None)
                    open_row.pop(sym, None)
                    entry_px.pop(sym, None)
                    continue
                qty[sym] = target_q
                if sym in pending:
                    i, f = pending[sym]
                    open_row[sym] = i
                    entry_px[sym] = px
                    trades.append({"row": i, "symbol": sym, "entry_bar": d,
                                   "exit_bar": int(cand.loc[i].exit_bar),
                                   "side": float(cand.loc[i].side), "qty": target_q,
                                   "entry_price": px, "weight": target_w,
                                   "risk_fraction": abs(target_w) * float(cand.loc[i].barrier),
                                   "requested_risk_fraction": f,
                                   "fill_return": float(cand.loc[i].fill_return),
                                   "barrier": float(cand.loc[i].barrier),
                                   "p": float(cand.loc[i].p), "p_low": float(cand.loc[i].p_low)})
                else:
                    resizes += 1
        returns[d] = (book.equity - start) / start
        equity[d] = book.equity

    return BookResult(
        returns=pd.Series(returns, index=m.grid, name="book_return"),
        equity=pd.Series(equity, index=m.grid, name="equity"),
        trades=pd.DataFrame(trades),
        total_fees=book.fees, total_funding=book.funding, total_impact=book.impact,
        total_pnl=book.pnl, n_resizes=resizes, halted=bool(state.halted),
    )
```

Twee punten om bij de uitvoering te controleren, niet te raden:
1. `RiskEngine.decide` kan voor een symbool zonder cluster in `conf/risk/default.yaml` crashen; de tests gebruiken daarom echte symboolnamen. Crasht hij op iets anders (bijv. een verplicht `cluster`-veld in `MarketState`), lees `risk/engine.py::_validated` en geef precies wat hij vraagt.
2. Een positie die `decide` naar nul zet terwijl haar trade nog open is (halt, drawdown), sluit hier op de close van d. Dat is een risico-exit en geen barrière-exit; de trade blijft in `trades` staan met zijn geplande `exit_bar`. Voeg in dat geval `"risk_exit_bar": d` toe aan de betreffende trade-dict en laat de verdictcode trades met een `risk_exit_bar` meetellen tegen hun werkelijke P&L, niet tegen `fill_return`.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_barrier_book.py -p no:randomly -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/backtest/barrier_book.py tests/unit/test_barrier_book.py
git commit -m "feat(backtest): a trade book that exits at the barrier, sized by binary Kelly, through the sovereign RiskEngine"
```

---

### Task 10: Het oordeel en CPCV in eventruimte

**Files:**
- Create: `src/tradebot/cv/event_space.py`
- Create: `src/tradebot/validation/weekly_verdict.py`
- Test: `tests/unit/test_weekly_verdict.py`

**Interfaces:**
- Consumes: `StopCriterion` met `.name`, `.metric`, `.action`, `.binds(value)` (`registry/preregistration.py`).
- Produces:
  - `event_space_t1(event_bar: np.ndarray, exit_bar: np.ndarray) -> np.ndarray`
  - `DEVELOPMENT_METRICS: tuple[str, ...]` en `HOLDOUT_METRICS: tuple[str, ...]`
  - `Verdict(status: Literal["PASS", "INVALID", "UNPROVEN", "FALSIFIED"], binding: tuple[str, ...], values: dict[str, float])` met `.as_dict()`
  - `judge(criteria: Sequence[StopCriterion], values: Mapping[str, float], *, stage: Literal["development", "holdout"]) -> Verdict`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_weekly_verdict.py
from __future__ import annotations

import numpy as np
import pytest

from tradebot.cv.event_space import event_space_t1
from tradebot.registry.preregistration import StopCriterion
from tradebot.utils.failfast import DataContractError
from tradebot.validation.weekly_verdict import judge

C = [
    StopCriterion("negative_control_shuffle", "max_abs_shuffle_auc_deviation", ">", 0.05, "archive", "r"),
    StopCriterion("negative_control_reversed", "reversed_minus_baseline_sharpe", ">=", 0.0, "archive", "r"),
    StopCriterion("no_edge_after_costs", "ensemble_net_sharpe", "<=", 0.0, "falsify", "r"),
    StopCriterion("filter_adds_nothing", "sharpe_diff_ci_low", "<=", 0.0, "archive", "r"),
    StopCriterion("dsr_below_threshold", "dsr", "<", 0.95, "archive", "r"),
    StopCriterion("insufficient_trades", "n_trades", "<", 100.0, "descope", "r"),
    StopCriterion("holdout_brier_worse", "holdout_brier_diff_ci_low", ">", 0.0, "archive", "r"),
    StopCriterion("promotion_requires_all_clear", "n_binding_stop_criteria", "<=", 0.0, "promote", "r"),
]
GOOD = {"max_abs_shuffle_auc_deviation": 0.01, "reversed_minus_baseline_sharpe": -0.4,
        "ensemble_net_sharpe": 1.4, "sharpe_diff_ci_low": 0.1, "dsr": 0.97, "n_trades": 250.0}


def test_all_clear_is_a_pass() -> None:
    v = judge(C, GOOD, stage="development")
    assert v.status == "PASS" and v.binding == ()


@pytest.mark.parametrize(
    ("metric", "value", "status"),
    [
        ("max_abs_shuffle_auc_deviation", 0.08, "INVALID"),
        ("reversed_minus_baseline_sharpe", 0.1, "INVALID"),
        ("ensemble_net_sharpe", -0.2, "FALSIFIED"),
        ("sharpe_diff_ci_low", -0.1, "UNPROVEN"),
        ("dsr", 0.90, "UNPROVEN"),
        ("n_trades", 40.0, "UNPROVEN"),
    ],
)
def test_each_gate_can_go_red(metric, value, status) -> None:
    v = judge(C, {**GOOD, metric: value}, stage="development")
    assert v.status == status
    assert len(v.binding) == 1


def test_a_missing_measurement_crashes() -> None:
    values = dict(GOOD)
    del values["dsr"]
    with pytest.raises(DataContractError, match="meting"):
        judge(C, values, stage="development")


def test_the_holdout_stage_reads_only_holdout_criteria() -> None:
    assert judge(C, {"holdout_brier_diff_ci_low": -0.01}, stage="holdout").status == "PASS"
    assert judge(C, {"holdout_brier_diff_ci_low": 0.02}, stage="holdout").status == "UNPROVEN"


def test_event_space_t1_points_at_the_last_event_inside_the_label() -> None:
    ev = np.array([0, 2, 3, 7, 9])
    ex = np.array([4, 3, 8, 12, 10])
    assert event_space_t1(ev, ex).tolist() == [2, 2, 3, 4, 4]
    with pytest.raises(DataContractError):
        event_space_t1(np.array([3, 1]), np.array([4, 5]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_verdict.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.cv.event_space'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/cv/event_space.py
"""Exitposities in eventruimte voor `CombinatorialPurgedCV.split`.

De splitter verwacht `t1` als POSITIE in de eventreeks, niet als bar. Voor een
op tijd gesorteerd paneel is dat voor event i de positie van het laatste event
dat start op of vóór de exitbar van i.
"""
from __future__ import annotations

import numpy as np

from ..utils.failfast import DataContractError, require

__all__ = ["event_space_t1"]


def event_space_t1(event_bar: np.ndarray, exit_bar: np.ndarray) -> np.ndarray:
    ev = np.asarray(event_bar, dtype=np.int64)
    ex = np.asarray(exit_bar, dtype=np.int64)
    require(ev.size == ex.size, "Eén exit per event.", DataContractError)
    require(bool(np.all(np.diff(ev) >= 0)), "Events moeten op tijd gesorteerd zijn.",
            DataContractError)
    require(bool(np.all(ex > ev)), "Een exit op of vóór zijn eigen event.", DataContractError)
    return np.searchsorted(ev, ex, side="right") - 1
```

```python
# src/tradebot/validation/weekly_verdict.py
"""Het oordeel valt op de BEVROREN stop-criteria van de preregistratie (spec §14).

Geen drempel staat in deze module. Een negatieve controle die bindt maakt de run
ongeldig; een falsify-criterium falsifieert; elk ander bindend criterium laat de
hypothese onbewezen. Een criterium zonder meting crasht: een ontbrekend getal
mag nooit als "niet bindend" worden gelezen.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..registry.preregistration import StopCriterion
from ..utils.failfast import DataContractError, require

__all__ = ["DEVELOPMENT_METRICS", "HOLDOUT_METRICS", "Verdict", "judge"]

DEVELOPMENT_METRICS = (
    "max_abs_shuffle_auc_deviation", "reversed_minus_baseline_sharpe",
    "ensemble_net_sharpe", "sharpe_diff_ci_low", "sharpe_ci_low", "dsr",
    "hit_rate_ci_low_minus_break_even", "pbo", "mc_drawdown_probability_1y",
    "n_trades",
)
HOLDOUT_METRICS = ("holdout_brier_diff_ci_low",)
Status = Literal["PASS", "INVALID", "UNPROVEN", "FALSIFIED"]


@dataclass(frozen=True)
class Verdict:
    status: Status
    binding: tuple[str, ...]
    values: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "binding": list(self.binding), "values": self.values}


def judge(
    criteria: Sequence[StopCriterion],
    values: Mapping[str, float],
    *,
    stage: Literal["development", "holdout"],
) -> Verdict:
    in_stage = [c for c in criteria
                if c.action != "promote"
                and c.name.startswith("holdout_") == (stage == "holdout")]
    missing = sorted({c.metric for c in in_stage} - set(values))
    require(not missing, "Een stop-criterium zonder meting.", DataContractError,
            missing=missing, stage=stage)
    measured = {c.metric: float(values[c.metric]) for c in in_stage}
    require(all(np.isfinite(v) for v in measured.values()),
            "Een niet-eindige meting.", DataContractError, values=measured)
    binding = [c for c in in_stage if c.binds(measured[c.metric])]
    if any(c.name.startswith("negative_control") for c in binding):
        status: Status = "INVALID"
    elif any(c.action == "falsify" for c in binding):
        status = "FALSIFIED"
    elif binding:
        status = "UNPROVEN"
    else:
        status = "PASS"
    return Verdict(status=status, binding=tuple(c.name for c in binding), values=measured)
```

Controleer dat `StopCriterion(name, metric, operator, threshold, action, rationale)` positioneel werkt (dataclass-volgorde in `registry/preregistration.py`); zo niet, zet de test om naar keywords.

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_verdict.py -p no:randomly -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tradebot/cv/event_space.py src/tradebot/validation/weekly_verdict.py tests/unit/test_weekly_verdict.py
git commit -m "feat(validation): judge on the frozen stop criteria, and event-space t1 for CPCV"
```

---

### Task 11: De meting op de ontwikkelsample

**Files:**
- Create: `src/tradebot/validation/weekly_campaign.py`
- Test: `tests/unit/test_weekly_campaign.py`

**Interfaces:**
- Consumes: alles uit Taak 2–10; `WalkForwardCV`; `shuffled_targets`; `CombinatorialPurgedCV`, `build_cpcv_return_paths` (`cv/cpcv.py`); `compute_pbo` (`backtest/pbo.py`); `sharpe_with_se`, `block_bootstrap_ci`, `sharpe_difference_test` (`validation/inference.py`); `dsr_gate` (`validation/dsr.py`); `TrialCount`, `frozen_trial_count` (`registry/trial_counter.py`); `require_preregistration`, `development_slice`; `ExecutionConfig`, `ValidationConfig`, `RiskConfig`, `ImpactConfig`; `risk_config_hash`.
- Produces:
  - `trade_candidates(events: pd.DataFrame, folds: Sequence[FoldPredictions], *, cfg, cost_rt: float) -> pd.DataFrame` (kolommen `CANDIDATE_COLUMNS` + `p_be`, `fold_id`)
  - `CampaignResult(values: dict[str, float], verdict: Verdict, record: dict)`
  - `run_campaign_on_market(market, cfg, *, criteria, trial_count, exec_cfg, val_cfg, risk_cfg, impact) -> CampaignResult`
  - `main() -> None` (CLI: `python -m tradebot.validation.weekly_campaign`)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_weekly_campaign.py
"""De campagne van begin tot eind op een synthetische markt: elke poortmeting bestaat en is eindig."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.registry.preregistration import StopCriterion
from tradebot.registry.trial_counter import TrialCount
from tradebot.schemas.config import ExecutionConfig, RiskConfig, ValidationConfig, load_config
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.validation.weekly_campaign import run_campaign_on_market, trade_candidates
from tradebot.validation.weekly_verdict import DEVELOPMENT_METRICS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def result():
    market = synthetic_market(n=1100, seed=21)
    cfg = weekly_meta_config().model_copy(update={
        "first_test_start_utc": str(market.grid[500]),
        "holdout_split_utc": str(market.grid[-1] + pd.Timedelta(days=1)),
        "forest_n_estimators": 40, "n_shuffle_replicates": 1, "mc_paths": 1000,
        "k_grid": (1.0, 1.5, 2.0),
    })
    criteria = [StopCriterion(f"c_{m}", m, "<", -1e9, "archive", "synthetic")
                for m in DEVELOPMENT_METRICS]
    return run_campaign_on_market(
        market, cfg, criteria=criteria,
        trial_count=TrialCount(value=4, source="frozen", origin="test",
                               seed_total=0, registered_total=4),
        exec_cfg=load_config(ROOT / "conf/execution/fees.yaml", ExecutionConfig),
        val_cfg=load_config(ROOT / "conf/validation/default.yaml", ValidationConfig),
        risk_cfg=load_config(ROOT / "conf/risk/default.yaml", RiskConfig),
        impact=None)


def test_every_gate_metric_is_measured_and_finite(result) -> None:
    for metric in DEVELOPMENT_METRICS:
        assert metric in result.values
        assert np.isfinite(result.values[metric]), metric


def test_the_record_carries_its_policy_and_its_choices(result) -> None:
    rec = result.record
    assert rec["risk_policy_hash"]
    assert rec["risk_audit_header"]["config_hash"] == rec["risk_policy_hash"]
    assert rec["k"] > 0.0 and 0.0 < rec["d_star"] <= 0.9
    assert rec["verdict"]["status"] in {"PASS", "INVALID", "UNPROVEN", "FALSIFIED"}


def test_candidates_never_use_a_threshold_below_break_even() -> None:
    from tradebot.train.meta_label import FoldPredictions
    events = pd.DataFrame({"symbol": ["BTCUSDT"] * 3, "event_bar": [10, 11, 12],
                           "exit_bar": [15, 16, 17], "side": [1.0, -1.0, 1.0],
                           "fill_return": [0.05, -0.05, 0.0], "sigma": [0.03] * 3,
                           "target": [1, 0, 0]})
    fold = FoldPredictions(fold_id=0, row_index=np.array([0, 1, 2]),
                           probability=np.array([0.40, 0.55, 0.70]), target=np.array([1, 0, 0]),
                           uniqueness=np.ones(3), n_train=100, purge={},
                           feature_importance=np.ones(19),
                           extras={"calibration_probability": np.linspace(0.3, 0.7, 50),
                                   "train_events_per_week": 5.5})
    cands = trade_candidates(events, [fold], cfg=weekly_meta_config(), cost_rt=0.0013)
    assert (cands["p_trade"] >= cands["p_be"]).all()
    assert (cands["entry_bar"] == events["event_bar"] + 1).all()
    assert (cands["p_low"] < cands["p"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_campaign.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.validation.weekly_campaign'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/validation/weekly_campaign.py
"""De meting van de wekelijkse strategie op de ontwikkelsample (spec §9, §14).

Volgorde, en waarom zij vastligt:
1. d* en k worden gekozen op data vóór de eerste testperiode; k op frequentie.
2. Drie modellen door dezelfde purged walk-forward (Taak 7).
3. Kandidaten per model; het ensemble met Kelly, de referentie ongefilterd met
   een vaste risicofractie; allemaal door hetzelfde boek en dezelfde risicolaag.
4. Negatieve controles: geschudde labels en de omgekeerde richting.
5. Inferentie: Lo-SE, blokbootstrap, Ledoit-Wolf-verschil, DSR bij de bevroren M,
   Wilson-interval op de trefkans, PBO over de vier varianten, Monte Carlo-drawdown.
6. Het oordeel komt uit de bevroren stop-criteria (Taak 10).
CPCV-paden worden gerapporteerd, niet gepoort.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..backtest.barrier_book import BookInputs, SizingRule, run_barrier_book
from ..backtest.pbo import compute_pbo
from ..cv.cpcv import CombinatorialPurgedCV, build_cpcv_return_paths
from ..cv.event_space import event_space_t1
from ..cv.walk_forward import WalkForwardCV
from ..data.weekly_market import WeeklyMarket, load_weekly_market
from ..execution.impact_model import ImpactParams, ImpactStatus
from ..features.registry import current_git_sha
from ..features.weekly_set import fit_common_d_star, market_features
from ..labeling.barrier_fills import round_trip_cost
from ..labeling.breakout import calibrate_k
from ..monitoring.prob_calibration import expected_calibration_error
from ..registry.preregistration import StopCriterion, require_preregistration
from ..registry.trial_counter import TrialCount, frozen_trial_count
from ..risk.binary_kelly import break_even_probability, monte_carlo_drawdown_probability, posterior_lower_probability
from ..risk.engine import RiskEngine, risk_config_hash
from ..schemas.config import ExecutionConfig, ImpactConfig, RiskConfig, ValidationConfig, load_config
from ..schemas.weekly_meta import WeeklyMetaConfig, weekly_meta_config
from ..train.light_models import MODEL_KINDS, fit_light_model
from ..train.meta_label import FoldPredictions, shuffled_targets, walk_forward_fit_predict
from ..train.weekly_dataset import WeeklyDataset, build_weekly_dataset
from ..utils.failfast import DataContractError, require
from .dsr import dsr_gate
from .holdout import development_slice
from .inference import block_bootstrap_ci, sharpe_difference_test, sharpe_with_se
from .weekly_verdict import Verdict, judge

__all__ = ["CampaignResult", "main", "run_campaign_on_market", "trade_candidates"]

BARS_PER_YEAR = 365.0
ARTEFACT = Path("artefacts/governance/weekly_meta_campaign.json")


@dataclass(frozen=True)
class CampaignResult:
    values: dict[str, float]
    verdict: Verdict
    record: dict[str, Any]


def trade_candidates(
    events: pd.DataFrame, folds: Sequence[FoldPredictions], *, cfg: WeeklyMetaConfig,
    cost_rt: float,
) -> pd.DataFrame:
    """Per OOS-event: kans, posterior-ondergrens, break-even en de handelsdrempel van zijn fold."""
    edges = np.linspace(0.0, 1.0, cfg.n_probability_bins + 1)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        cal = np.asarray(fold.extras["calibration_probability"], dtype=np.float64)
        phi = min(1.0, cfg.trades_per_week_target / max(float(fold.extras["train_events_per_week"]), 1e-9))
        q = float(np.quantile(cal, 1.0 - phi))
        counts = np.histogram(cal, bins=edges)[0]
        for r, p in zip(fold.row_index, fold.probability, strict=True):
            ev = events.iloc[int(r)]
            barrier = cfg.barrier_sigma * float(ev["sigma"])
            p_be = break_even_probability(barrier, cost_rt)
            b = min(int(np.searchsorted(edges, p, side="right")) - 1, len(counts) - 1)
            rows.append({
                "symbol": ev["symbol"], "entry_bar": int(ev["event_bar"]) + 1,
                "exit_bar": int(ev["exit_bar"]), "side": float(ev["side"]),
                "fill_return": float(ev["fill_return"]), "barrier": barrier,
                "p": float(p),
                "p_low": posterior_lower_probability(float(p), float(counts[b]), cfg.posterior_quantile),
                "p_be": p_be, "p_trade": max(p_be, q), "fold_id": int(fold.fold_id),
            })
    return pd.DataFrame(rows).sort_values(["entry_bar", "symbol"], kind="stable").reset_index(drop=True)


def _unfiltered(events: pd.DataFrame, cfg: WeeklyMetaConfig, lo: int, hi: int) -> pd.DataFrame:
    ev = events[(events["event_bar"] >= lo) & (events["event_bar"] <= hi)]
    return pd.DataFrame({
        "symbol": ev["symbol"].to_numpy(), "entry_bar": ev["event_bar"].to_numpy() + 1,
        "exit_bar": ev["exit_bar"].to_numpy(), "side": ev["side"].to_numpy(),
        "fill_return": ev["fill_return"].to_numpy(),
        "barrier": cfg.barrier_sigma * ev["sigma"].to_numpy(),
        "p": 1.0, "p_low": 1.0, "p_trade": 0.0,
    }).sort_values(["entry_bar", "symbol"], kind="stable").reset_index(drop=True)


def _oos_auc(folds: Sequence[FoldPredictions]) -> float:
    y = np.concatenate([f.target for f in folds])
    p = np.concatenate([f.probability for f in folds])
    w = np.concatenate([f.uniqueness for f in folds])
    return float(roc_auc_score(y, p, sample_weight=w))


def _cpcv_path_sharpes(wd: WeeklyDataset, cfg: WeeklyMetaConfig, inputs: BookInputs,
                       risk_cfg: RiskConfig, fixed: SizingRule) -> list[float]:
    ds = wd.dataset
    order = np.argsort(ds.event_bar, kind="stable")
    ev, ex = ds.event_bar[order], ds.exit_bar[order]
    cv = CombinatorialPurgedCV(n_groups=cfg.cpcv_n_groups, n_test_groups=cfg.cpcv_n_test_groups,
                               purge_bars=0)
    size = len(order) // cfg.cpcv_n_groups
    bounds = [(g * size, len(order) if g == cfg.cpcv_n_groups - 1 else (g + 1) * size)
              for g in range(cfg.cpcv_n_groups)]
    fold_returns: dict[tuple, pd.Series] = {}
    for train_pos, test_pos, groups in cv.split(pd.DatetimeIndex(wd.grid[ev]),
                                               pd.Series(event_space_t1(ev, ex))):
        model = fit_light_model(ds, order[train_pos], ds.target, "ensemble", cfg)
        rows = order[test_pos]
        prob = model.predict_proba(ds.features.iloc[rows].to_numpy(dtype=np.float64))[:, 1]
        fp = FoldPredictions(fold_id=0, row_index=rows, probability=prob,
                             target=ds.target[rows], uniqueness=ds.uniqueness[rows],
                             n_train=int(train_pos.size), purge={},
                             feature_importance=model.feature_importance,
                             extras=model.fold_extras)
        cands = trade_candidates(wd.events, [fp], cfg=cfg, cost_rt=fixed.cost_rt)
        res = run_barrier_book(cands, inputs, RiskEngine(risk_cfg), fixed, equity0=cfg.account_equity)
        pieces = []
        for g in groups:
            lo, hi = bounds[g]
            start, end = wd.grid[ev[lo]], wd.grid[int(ex[lo:hi].max())]
            pieces.append(res.returns.loc[start:end])
        fold_returns[tuple(groups)] = pd.concat(pieces)
    paths = build_cpcv_return_paths(fold_returns, n_groups=cfg.cpcv_n_groups)
    return [sharpe_with_se(p, bars_per_year=BARS_PER_YEAR).sharpe for p in paths]


def run_campaign_on_market(
    market: WeeklyMarket,
    cfg: WeeklyMetaConfig,
    *,
    criteria: Sequence[StopCriterion],
    trial_count: TrialCount,
    exec_cfg: ExecutionConfig,
    val_cfg: ValidationConfig,
    risk_cfg: RiskConfig,
    impact: ImpactParams | None,
) -> CampaignResult:
    cost_rt = round_trip_cost(exec_cfg)
    first_test = pd.Timestamp(cfg.first_test_start_utc)
    closes = {s: market.ohlcv[s]["close"] for s in market.symbols}
    d_star = fit_common_d_star({s: np.log(c) for s, c in closes.items()}, until=first_test)
    start = market.sigma_daily.dropna(how="all").index[0]
    k, rates = calibrate_k(closes, {s: market.sigma_daily[s] for s in market.symbols},
                           k_grid=cfg.k_grid, target_per_week=cfg.events_per_week_target,
                           start=start, end=first_test)
    wd = build_weekly_dataset(market, cfg, k=k, d_star=d_star, cost_rt=cost_rt)
    ds = wd.dataset
    first_pos = int(market.grid.searchsorted(first_test))
    embargo = cfg.horizon_bars + 1
    cv = WalkForwardCV(train_size=first_pos, test_size=cfg.test_bars, step=cfg.test_bars,
                       mode="anchored", min_train=first_pos, embargo_bars=embargo)
    n_bars = len(market.grid)
    oos_last = max(int(f.test_idx[-1]) for f in cv.split(n_bars))

    def run(kind: str, target: np.ndarray | None = None) -> list[FoldPredictions]:
        return walk_forward_fit_predict(
            ds, cv, lambda rows, y: fit_light_model(ds, rows, y, kind, cfg),
            n_bars=n_bars, embargo_bars=embargo, target=target)

    folds = {kind: run(kind) for kind in MODEL_KINDS}
    shuffle_aucs = [_oos_auc(run("ensemble", perm))
                    for perm in shuffled_targets(ds, cfg.seed, cfg.n_shuffle_replicates)]

    corr = market_features(np.log(pd.DataFrame(closes)).diff(), window=cfg.corr_window)["avg_corr60"]
    inputs = BookInputs(market=market, avg_corr=corr, cost_rate=cost_rt / 2.0, impact=impact)
    kelly = SizingRule("kelly", cfg.kelly_multiple, cfg.baseline_risk_fraction, cfg.resize_band, cost_rt)
    fixed = SizingRule("fixed", cfg.kelly_multiple, cfg.baseline_risk_fraction, cfg.resize_band, cost_rt)
    window = slice(market.grid[first_pos], market.grid[oos_last])

    def book(cands: pd.DataFrame, sizing: SizingRule):
        return run_barrier_book(cands, inputs, RiskEngine(risk_cfg), sizing, equity0=cfg.account_equity)

    books = {kind: book(trade_candidates(wd.events, folds[kind], cfg=cfg, cost_rt=cost_rt), kelly)
             for kind in MODEL_KINDS}
    baseline = book(_unfiltered(wd.events, cfg, first_pos, oos_last), fixed)
    reversed_wd = build_weekly_dataset(market, cfg, k=k, d_star=d_star, cost_rt=cost_rt, side_sign=-1.0)
    reversed_book = book(_unfiltered(reversed_wd.events, cfg, first_pos, oos_last), fixed)

    ens = books["ensemble"].returns.loc[window]
    base = baseline.returns.loc[window]
    ens_se = sharpe_with_se(ens, bars_per_year=BARS_PER_YEAR)
    base_se = sharpe_with_se(base, bars_per_year=BARS_PER_YEAR)
    rev_se = sharpe_with_se(reversed_book.returns.loc[window], bars_per_year=BARS_PER_YEAR)
    diff = sharpe_difference_test(ens, base, bars_per_year=BARS_PER_YEAR, seed=cfg.seed,
                                  align="common_valid")
    ci = block_bootstrap_ci(ens, bars_per_year=BARS_PER_YEAR, seed=cfg.seed)
    dsr = dsr_gate(ens.to_numpy(), trial_count=trial_count, config=val_cfg)

    taken = books["ensemble"].trades
    n_trades = int(len(taken))
    if n_trades:
        wins = int(((taken["fill_return"] - cost_rt) > 0.0).sum())
        hit_low = float(stats.binomtest(wins, n_trades).proportion_ci(0.95, method="wilson").low)
        p_be_mean = float(np.mean([break_even_probability(b, cost_rt) for b in taken["barrier"]]))
        r_mult = ((taken["fill_return"] - cost_rt) / taken["barrier"]).to_numpy()
        years = (window.stop - window.start) / pd.Timedelta(days=365)
        per_year = n_trades / years
        mc = monte_carlo_drawdown_probability(
            r_mult, risk_fraction=float(taken["risk_fraction"].median()),
            n_trades=min(n_trades, max(1, int(round(per_year)))),
            drawdown=cfg.mc_max_drawdown, block_length=max(1, int(round(per_year / 26.0))),
            n_paths=cfg.mc_paths, seed=cfg.seed)
    else:
        hit_low, p_be_mean, mc = 0.0, 1.0, 1.0

    variants = np.column_stack([base.to_numpy()] + [books[k].returns.loc[window].to_numpy()
                                                    for k in MODEL_KINDS])
    pbo = float(compute_pbo(variants, n_subsets=16)["pbo"])
    cpcv = _cpcv_path_sharpes(wd, cfg, inputs, risk_cfg, fixed)

    ens_folds = folds["ensemble"]
    y = np.concatenate([f.target for f in ens_folds])
    p = np.concatenate([f.probability for f in ens_folds])
    lo, _hi = cfg.shuffle_auc_band
    values = {
        "max_abs_shuffle_auc_deviation": float(max(abs(a - 0.5) for a in shuffle_aucs)),
        "reversed_minus_baseline_sharpe": float(rev_se.sharpe - base_se.sharpe),
        "ensemble_net_sharpe": float(ens_se.sharpe),
        "sharpe_diff_ci_low": float(diff.ci_low),
        "sharpe_ci_low": float(ci.low),
        "dsr": float(dsr.dsr),
        "hit_rate_ci_low_minus_break_even": float(hit_low - p_be_mean),
        "pbo": pbo,
        "mc_drawdown_probability_1y": float(mc),
        "n_trades": float(n_trades),
    }
    verdict = judge(criteria, values, stage="development")
    engine = RiskEngine(risk_cfg)
    record = {
        "verdict": verdict.as_dict(),
        "k": float(k), "k_rates": {str(kk): v for kk, v in rates.items()},
        "d_star": float(d_star),
        "n_events": int(len(ds)), "effective_n": float(ds.effective_n),
        "n_dropped_nan": wd.n_dropped_nan,
        "oos_window": [str(window.start), str(window.stop)],
        "oos_auc": {kind: _oos_auc(folds[kind]) for kind in MODEL_KINDS},
        "shuffle_aucs": shuffle_aucs, "shuffle_auc_band_low": lo,
        "brier_model": float(np.mean((p - y) ** 2)),
        "brier_base_rate": float(np.mean((y.mean() - y) ** 2)),
        "ece": float(expected_calibration_error(p, y)),
        "sharpe": {"ensemble": ens_se.sharpe, "ensemble_se": ens_se.se,
                   "baseline": base_se.sharpe, "reversed": rev_se.sharpe,
                   **{kind: sharpe_with_se(books[kind].returns.loc[window],
                                           bars_per_year=BARS_PER_YEAR).sharpe
                      for kind in MODEL_KINDS}},
        "sharpe_difference": {"delta": float(diff.delta_sharpe), "ci_low": float(diff.ci_low),
                              "ci_high": float(diff.ci_high), "p_value": float(diff.p_value)},
        "dsr": {"dsr": float(dsr.dsr), "passed": bool(dsr.passed), "M": int(trial_count.value)},
        "cpcv_path_sharpes": cpcv,
        "costs": {"fees": books["ensemble"].total_fees, "funding": books["ensemble"].total_funding,
                  "impact": books["ensemble"].total_impact, "round_trip": cost_rt},
        "risk_policy_hash": risk_config_hash(risk_cfg),
        "risk_audit_header": engine.audit_header(),
        "n_risk_resizes": books["ensemble"].n_resizes,
        "halted": books["ensemble"].halted,
    }
    return CampaignResult(values=values, verdict=verdict, record=record)


def main() -> None:
    """Draai de campagne op de gecertificeerde ontwikkelsample en schrijf het artefact."""
    root = Path.cwd()
    cfg = weekly_meta_config()
    lock = root / "artefacts/governance/holdout_lock.json"
    prereg_files = sorted((root / "artefacts/governance").glob("preregistration_*.json"))
    require(len(prereg_files) == 1, "Verwacht precies één bevroren preregistratie.",
            DataContractError, found=[p.name for p in prereg_files])
    prereg_path = prereg_files[0]
    prereg_id = json.loads(prereg_path.read_text(encoding="utf-8"))["preregistration_id"]
    prereg = require_preregistration(prereg_id, directory=root / "artefacts/governance")
    full = load_weekly_market(root, cfg.symbols)
    dev_index = development_slice(pd.DataFrame(index=full.grid), lock_path=lock).index
    market = full.truncate(dev_index[-1] + pd.Timedelta(hours=1))
    imp = load_config(root / "conf/execution/impact.yaml", ImpactConfig)
    impact = ImpactParams(
        eta=imp.eta, kappa_d=imp.kappa_d, status=ImpactStatus(imp.status), method=imp.method,
        data_hash=imp.data_hash, sample_size=imp.sample_size, period_start=imp.period_start,
        period_end=imp.period_end, instruments=imp.instruments, eta_ci_low=imp.eta_ci_low,
        eta_ci_high=imp.eta_ci_high)
    result = run_campaign_on_market(
        market, cfg, criteria=prereg.stop_criteria,
        trial_count=frozen_trial_count(prereg_path),
        exec_cfg=load_config(root / "conf/execution/fees.yaml", ExecutionConfig),
        val_cfg=load_config(root / "conf/validation/default.yaml", ValidationConfig),
        risk_cfg=load_config(root / "conf/risk/default.yaml", RiskConfig),
        impact=impact)
    record = {**result.record, "values": result.values, "preregistration_id": prereg_id,
              "git_sha": current_git_sha(), "source_hashes": full.source_hashes,
              "dev_last_bar": str(market.grid[-1])}
    (root / ARTEFACT).write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n",
                                 encoding="utf-8")
    print(json.dumps(result.verdict.as_dict(), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_campaign.py -p no:randomly -q`
Expected: PASS (3 tests), binnen enkele minuten. Crasht `compute_pbo` op een te korte reeks of `dsr_gate` op `MIN_OBS_FOR_DSR`: vergroot `n` in de fixture, verzwak de functies niet. Crasht `CombinatorialPurgedCV` op zijn embargo-eis, lees de melding: `embargo_bars=None` hoort `t_max + 1` af te leiden.

- [ ] **Step 5: Run the full suite and commit**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest -m "not slow" -p no:randomly -q` en `ruff check src/ tests/`
Expected: geen nieuwe failures, ruff schoon.

```bash
git add src/tradebot/validation/weekly_campaign.py tests/unit/test_weekly_campaign.py
git commit -m "feat(validation): the weekly campaign -- purged walk-forward, controls, inference, PBO, CPCV paths, and the frozen verdict"
```

---

### Task 12: De preregistratie en het bevriezen van het programma

**Files:**
- Create: `conf/research/preregistration_weekly_meta.yaml`
- Create: `src/tradebot/registry/weekly_programme.py`
- Test: `tests/unit/test_weekly_programme.py`

**Interfaces:**
- Consumes: `freeze_holdout(*, split_utc, out, git_sha)` (`validation/holdout.py`); `load_preregistration_spec(path, *, data_hashes, parameters)`, `freeze_preregistration(prereg, *, git_sha, ledger_total_at_freeze, directory)`; `HypothesisLedger`, `LedgerEntry.from_config(...)`; `DataRegister.certified_hash(asset_class, dataset, symbol, granularity)`; `hash_config` (`utils/hashing.py`); `current_git_sha`.
- Produces:
  - `PROGRAMME_UNIT = "weekly_meta_programme"`
  - `refreeze_unread_holdout(lock_path: Path, *, split_utc: str, git_sha: str) -> HoldoutLock`
  - `programme_parameters(cfg: WeeklyMetaConfig) -> dict`
  - `certified_data_hashes(register: DataRegister, symbols) -> tuple[tuple[str, str], ...]`
  - `book_and_freeze(*, spec_path, cfg, register, ledger_path, prereg_dir, git_sha) -> Path`
  - CLI: `python -m tradebot.registry.weekly_programme freeze`

- [ ] **Step 1: Write the preregistration**

```yaml
# conf/research/preregistration_weekly_meta.yaml
# PRE-REGISTRATIE — wekelijkse meta-label-strategie, Plan 1.
# Bevroren door `python -m tradebot.registry.weekly_programme freeze`, VOOR de
# eerste fit op echte data. `parameters` staat hier bewust niet: die worden bij
# het bevriezen uit conf/model/weekly_meta.yaml ingespoten, samen met M = 4.
wave: weekly_meta_v1

title: >-
  CUSUM-doorbraak met een licht, gekalibreerd ML-filter op 1:1-barrières en
  binaire Kelly-sizing, op zes crypto-perpetuals op dagbars.

hypothesis: >-
  Een ensemble van een L2-logistische regressie en een ondiepe random forest,
  gekalibreerd met Platt op een gepurgede binnensplit, dat alleen de doorbraken
  handelt waarvan de gekalibreerde kans boven break-even en boven de
  frequentiedrempel ligt, levert na kosten, funding en impact een hogere netto
  Sharpe dan hetzelfde CUSUM-signaal zonder filter, met een DSR van ten minste
  0,95 bij M = 4, onder purged walk-forward met embargo.

null_hypothesis: >-
  Het filter voegt na kosten niets toe aan het ongefilterde CUSUM-signaal, of
  de gefilterde strategie heeft geen netto Sharpe die zich onderscheidt van de
  beste van vier pogingen.

universe: [BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, LINKUSDT, DOTUSDT]
granularity: 1d
period_start: "2020-03-26"
period_end: "2026-06-23"
evaluation_start: "2022-01-01"
primary_metric: ensemble_net_sharpe

# referentie (ongefilterd), logreg, forest, ensemble.
planned_trials: 4

data_series:
  - crypto/ohlcv/BTCUSDT/1d
  - crypto/ohlcv/ETHUSDT/1d
  - crypto/ohlcv/SOLUSDT/1d
  - crypto/ohlcv/AVAXUSDT/1d
  - crypto/ohlcv/LINKUSDT/1d
  - crypto/ohlcv/DOTUSDT/1d
  - crypto/funding/BTCUSDT/8h
  - crypto/funding/ETHUSDT/8h
  - crypto/funding/SOLUSDT/8h
  - crypto/funding/AVAXUSDT/8h
  - crypto/funding/LINKUSDT/8h
  - crypto/funding/DOTUSDT/8h
  - crypto/open_interest/BTCUSDT/1d
  - crypto/open_interest/ETHUSDT/1d
  - crypto/open_interest/SOLUSDT/1d
  - crypto/open_interest/AVAXUSDT/1d
  - crypto/open_interest/LINKUSDT/1d
  - crypto/open_interest/DOTUSDT/1d

stop_criteria:
  - name: negative_control_shuffle
    metric: max_abs_shuffle_auc_deviation
    operator: ">"
    threshold: 0.05
    action: archive
    rationale: >-
      Op geschudde labels hoort de OOS-AUC binnen 0,45-0,55 te liggen. Daarbuiten
      lekt de pijplijn informatie en is de run ongeldig.
  - name: negative_control_reversed
    metric: reversed_minus_baseline_sharpe
    operator: ">="
    threshold: 0.0
    action: archive
    rationale: >-
      Het omgekeerde primaire signaal moet slechter zijn dan het echte. Is het dat
      niet, dan meet de constructie de doorbraak niet.
  - name: no_edge_after_costs
    metric: ensemble_net_sharpe
    operator: "<="
    threshold: 0.0
    action: falsify
    rationale: Een niet-positieve netto Sharpe betekent geen edge na kosten.
  - name: filter_adds_nothing
    metric: sharpe_diff_ci_low
    operator: "<="
    threshold: 0.0
    action: archive
    rationale: >-
      De ondergrens van het Ledoit-Wolf-interval op (ensemble - ongefilterd) moet
      boven nul liggen; anders is niet aangetoond dat het ML-filter iets toevoegt.
  - name: sharpe_interval_includes_zero
    metric: sharpe_ci_low
    operator: "<="
    threshold: 0.0
    action: archive
    rationale: De blokbootstrap-ondergrens van de netto Sharpe moet boven nul liggen.
  - name: dsr_below_threshold
    metric: dsr
    operator: "<"
    threshold: 0.95
    action: archive
    rationale: Onder DSR 0,95 bij M = 4 is de Sharpe niet te onderscheiden van selectieruis.
  - name: hit_rate_not_above_break_even
    metric: hit_rate_ci_low_minus_break_even
    operator: "<="
    threshold: 0.0
    action: archive
    rationale: >-
      De Wilson-ondergrens van het trefpercentage moet boven het gemiddelde
      break-even-percentage 1/2 + c/(2b) van de genomen trades liggen.
  - name: overfit_probability
    metric: pbo
    operator: ">="
    threshold: 0.25
    action: archive
    rationale: Een PBO van 0,25 of meer over de vier varianten is te veel overfitkans.
  - name: drawdown_risk
    metric: mc_drawdown_probability_1y
    operator: ">"
    threshold: 0.10
    action: archive
    rationale: >-
      De Monte Carlo-kans op een drawdown van 25 % (de mandaatgrens) binnen een jaar
      mag bij kwart-Kelly niet boven 10 % liggen. De fractie wordt daarna NIET
      bijgesteld; een andere fractie is een nieuwe preregistratie.
  - name: insufficient_trades
    metric: n_trades
    operator: "<"
    threshold: 100
    action: descope
    rationale: Onder 100 OOS-trades is geen enkele trefkansuitspraak te dragen.
  - name: holdout_brier_worse
    metric: holdout_brier_diff_ci_low
    operator: ">"
    threshold: 0.0
    action: archive
    rationale: >-
      Rooktest op de holdout: ligt het 95 %-interval van (Brier model - Brier
      basisfrequentie) volledig boven nul, dan is het model daar aantoonbaar
      slechter dan niets weten, en gaat het niet naar papertrading.
  - name: promotion_requires_all_clear
    metric: n_binding_stop_criteria
    operator: "<="
    threshold: 0
    action: promote
    rationale: Doorgaan naar Plan 2 vereist dat geen enkel criterium hierboven bindt.
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_weekly_programme.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.registry.hypothesis_ledger import HypothesisLedger
from tradebot.registry.preregistration import load_preregistration_spec
from tradebot.registry.trial_counter import frozen_trial_count
from tradebot.registry.weekly_programme import (
    book_and_freeze,
    certified_data_hashes,
    programme_parameters,
    refreeze_unread_holdout,
)
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.features.base import DataRegister
from tradebot.utils.failfast import DataContractError
from tradebot.validation.weekly_verdict import DEVELOPMENT_METRICS, HOLDOUT_METRICS

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "conf/research/preregistration_weekly_meta.yaml"
CFG = weekly_meta_config()


def _empty_ledger(tmp_path: Path) -> Path:
    path = tmp_path / "hypothesis_ledger.json"
    path.write_text(json.dumps({"version": 1, "seed_total": 0, "seed_note": "t", "entries": []}),
                    encoding="utf-8")
    return path


def test_the_spec_loads_with_every_criterion_measured() -> None:
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    prereg = load_preregistration_spec(SPEC, data_hashes=certified_data_hashes(register, CFG.symbols),
                                       parameters=programme_parameters(CFG))
    decisive = {c.metric for c in prereg.stop_criteria if c.action != "promote"}
    assert decisive == set(DEVELOPMENT_METRICS) | set(HOLDOUT_METRICS)
    assert prereg.planned_trials == CFG.planned_trials


def test_the_thresholds_match_the_config() -> None:
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    prereg = load_preregistration_spec(SPEC, data_hashes=certified_data_hashes(register, CFG.symbols),
                                       parameters=programme_parameters(CFG))
    by = {c.name: c.threshold for c in prereg.stop_criteria}
    assert by["overfit_probability"] == CFG.pbo_max
    assert by["drawdown_risk"] == CFG.mc_max_probability_1y
    assert by["insufficient_trades"] == CFG.min_trades
    assert by["negative_control_shuffle"] == pytest.approx(0.5 - CFG.shuffle_auc_band[0])
    assert by["holdout_brier_worse"] == CFG.holdout_brier_margin


def test_booking_then_freezing_fixes_m(tmp_path) -> None:
    ledger = _empty_ledger(tmp_path)
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    path = book_and_freeze(spec_path=SPEC, cfg=CFG, register=register, ledger_path=ledger,
                           prereg_dir=tmp_path, git_sha="abc1234")
    assert HypothesisLedger(ledger).total_n_hypotheses() == 4
    assert frozen_trial_count(path).value == 4
    with pytest.raises(DataContractError, match="eerste"):
        book_and_freeze(spec_path=SPEC, cfg=CFG, register=register, ledger_path=ledger,
                        prereg_dir=tmp_path, git_sha="abc1234")


def test_an_unread_holdout_can_be_refrozen_and_a_read_one_cannot(tmp_path) -> None:
    lock = tmp_path / "holdout_lock.json"
    lock.write_text(json.dumps({"split_utc": "2025-09-05T00:00:00+00:00", "git_sha": "x",
                                "frozen_utc": "t", "reads": []}), encoding="utf-8")
    refreeze_unread_holdout(lock, split_utc=CFG.holdout_split_utc, git_sha="abc1234")
    assert json.loads(lock.read_text(encoding="utf-8"))["split_utc"] == CFG.holdout_split_utc
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["reads"] = [{"hypothesis_id": "h"}]
    lock.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DataContractError, match="gelezen"):
        refreeze_unread_holdout(lock, split_utc="2026-07-01T00:00:00+00:00", git_sha="abc1234")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_programme.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.registry.weekly_programme'`

- [ ] **Step 4: Implement**

```python
# src/tradebot/registry/weekly_programme.py
"""Het bevriezen van het wekelijkse programma, vóór de eerste fit (spec §9, §17.2-3).

Drie handelingen, in deze volgorde en elk precies één keer:
1. Het ongelezen holdout-slot krijgt de nieuwe split (2026-06-24).
2. De vier geplande trials worden als EERSTE entry in de (lege) ledger geboekt.
3. De preregistratie wordt bevroren met M = 4 in haar parameters, zodat
   `frozen_trial_count` en daarmee de DSR hem uit het bevroren artefact leest.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..features.base import DataRegister
from ..features.registry import current_git_sha
from ..schemas.weekly_meta import WeeklyMetaConfig, weekly_meta_config
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from ..validation.holdout import HoldoutLock, freeze_holdout
from .hypothesis_ledger import HypothesisLedger, LedgerEntry
from .preregistration import freeze_preregistration, load_preregistration_spec

__all__ = ["PROGRAMME_UNIT", "book_and_freeze", "certified_data_hashes",
           "programme_parameters", "refreeze_unread_holdout"]

PROGRAMME_UNIT = "weekly_meta_programme"
_SERIES = (("ohlcv", "1d"), ("funding", "8h"), ("open_interest", "1d"))


def refreeze_unread_holdout(lock_path: Path, *, split_utc: str, git_sha: str) -> HoldoutLock:
    """Een nieuwe split, alleen als het oude slot nooit is gelezen."""
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    require(payload.get("reads") == [],
            "Het poortsample is al gelezen; een nieuwe split zou die lezing ongedaan maken.",
            DataContractError, reads=payload.get("reads"))
    lock_path.unlink()
    return freeze_holdout(split_utc=split_utc, out=lock_path, git_sha=git_sha)


def programme_parameters(cfg: WeeklyMetaConfig) -> dict[str, Any]:
    return {"weekly_meta": cfg.model_dump(mode="json"), "M": cfg.planned_trials,
            "M_seed_total": 0, "M_registered_total": cfg.planned_trials}


def certified_data_hashes(register: DataRegister, symbols: Sequence[str]) -> tuple[tuple[str, str], ...]:
    out = []
    for dataset, gran in _SERIES:
        for sym in symbols:
            out.append((f"crypto/{dataset}/{sym}/{gran}",
                        register.certified_hash("crypto", dataset, sym, gran)))
    return tuple(sorted(out))


def book_and_freeze(
    *, spec_path: Path, cfg: WeeklyMetaConfig, register: DataRegister,
    ledger_path: Path, prereg_dir: Path, git_sha: str,
) -> Path:
    """Boek de geplande trials als eerste ledger-entry en bevries daarna de preregistratie."""
    ledger = HypothesisLedger(ledger_path)
    require(ledger.total_n_hypotheses() == 0,
            "De programmaboeking hoort de eerste entry van de ledger te zijn; er staat al iets in.",
            DataContractError, total=ledger.total_n_hypotheses())
    params = programme_parameters(cfg)
    hashes = certified_data_hashes(register, cfg.symbols)
    prereg = load_preregistration_spec(spec_path, data_hashes=hashes, parameters=params)
    require(prereg.planned_trials == cfg.planned_trials,
            "planned_trials in de preregistratie wijkt af van de config.", DataContractError,
            spec=prereg.planned_trials, config=cfg.planned_trials)
    ledger.append(LedgerEntry.from_config(
        wave=1, unit=PROGRAMME_UNIT, market="crypto", config=params, git_sha=git_sha,
        data_hash=hash_config(dict(hashes)), preregistration_id=prereg.preregistration_id,
        n_trials=cfg.planned_trials, result="interim",
        notes="Vier geplande trials, geboekt vóór de eerste fit: referentie, logreg, forest, ensemble."))
    return freeze_preregistration(prereg, git_sha=git_sha,
                                  ledger_total_at_freeze=ledger.total_n_hypotheses(),
                                  directory=prereg_dir)


def main(argv: Sequence[str]) -> None:
    require(list(argv) == ["freeze"], "Gebruik: python -m tradebot.registry.weekly_programme freeze",
            DataContractError)
    root = Path.cwd()
    cfg = weekly_meta_config()
    sha = current_git_sha()
    refreeze_unread_holdout(root / "artefacts/governance/holdout_lock.json",
                            split_utc=cfg.holdout_split_utc, git_sha=sha)
    path = book_and_freeze(
        spec_path=root / "conf/research/preregistration_weekly_meta.yaml", cfg=cfg,
        register=DataRegister(root / "artefacts/governance/data_hashes.json"),
        ledger_path=root / "artefacts/governance/hypothesis_ledger.json",
        prereg_dir=root / "artefacts/governance", git_sha=sha)
    print(f"holdout split {cfg.holdout_split_utc}; preregistratie {path.name}; M = {cfg.planned_trials}")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 5: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_programme.py tests/unit/test_preregistration.py tests/unit/test_ledger_provenance.py -p no:randomly -q`
Expected: PASS. `test_ledger_provenance` blijft groen omdat de echte ledger in deze taak niet wordt aangeraakt.

- [ ] **Step 6: Commit**

```bash
git add conf/research/preregistration_weekly_meta.yaml src/tradebot/registry/weekly_programme.py tests/unit/test_weekly_programme.py
git commit -m "feat(registry): the weekly preregistration, and a freeze that books M = 4 before the first fit"
```

---

### Task 13: Bevriezen en meten (operationeel, eenmalig)

Geen nieuwe code. Deze taak raakt de echte governance-artefacten en draait de campagne één keer. **Vraag de eigenaar om een expliciete go voordat stap 2 begint**: vanaf daar is de ontwikkelsample geboekt.

**Files:**
- Modify (door de CLI): `artefacts/governance/holdout_lock.json`, `artefacts/governance/hypothesis_ledger.json`
- Create (door de CLI): `artefacts/governance/preregistration_<id>.json`, `artefacts/governance/weekly_meta_campaign.json`

- [ ] **Step 1: Schone boom, volledige suite groen**

Run: `git status --short` (leeg) en `D:/venv/tradebot/Scripts/python.exe -m pytest -m "not slow" -p no:randomly -q`
Expected: geen output van git; 0 failed.

- [ ] **Step 2: Bevries het programma**

Run: `D:/venv/tradebot/Scripts/python.exe -m tradebot.registry.weekly_programme freeze`
Expected: `holdout split 2026-06-24T00:00:00+00:00; preregistratie preregistration_<32 hex>.json; M = 4`

- [ ] **Step 3: Controleer en commit het bevriezen apart**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_ledger_provenance.py tests/unit/test_ledger_reset.py tests/unit/test_measurement_carries_policy_hash.py tests/unit/test_holdout.py -p no:randomly -q`
Expected: PASS. (`test_the_restart_is_documented_in_the_artefact` blijft groen: `seed_total` is nog 0.)

```bash
git add artefacts/governance/holdout_lock.json artefacts/governance/hypothesis_ledger.json artefacts/governance/preregistration_*.json
git commit -m "governance(weekly): freeze the holdout split, book M = 4, and freeze the preregistration -- before the first fit"
```

- [ ] **Step 4: Draai de campagne**

Run: `D:/venv/tradebot/Scripts/python.exe -m tradebot.validation.weekly_campaign`
Expected: een JSON-oordeel met `status` in `PASS | INVALID | UNPROVEN | FALSIFIED` en de bindende criteria.

- [ ] **Step 5: Boek het oordeel als amendement en commit**

Voeg met één Python-aanroep een amendement toe (`n_trials=0`, `amends` = de `config_hash` van de `weekly_meta_programme`-entry), met `result` = `interim` bij PASS, `falsified` bij FALSIFIED, anders `archived`, en `metrics = {"verdict": <status>, **values}`:

```python
import json
from pathlib import Path
from tradebot.features.registry import current_git_sha
from tradebot.registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from tradebot.schemas.weekly_meta import weekly_meta_config

gov = Path("artefacts/governance")
art = json.loads((gov / "weekly_meta_campaign.json").read_text(encoding="utf-8"))
ledger = HypothesisLedger(gov / "hypothesis_ledger.json")
booked = next(e for e in ledger.entries() if e["unit"] == "weekly_meta_programme")
status = art["verdict"]["status"]
result = {"PASS": "interim", "FALSIFIED": "falsified"}.get(status, "archived")
ledger.append(LedgerEntry.from_config(
    wave=1, unit="weekly_meta_campaign", market="crypto",
    config={"weekly_meta": weekly_meta_config().model_dump(mode="json"), "stage": "development"},
    git_sha=current_git_sha(), data_hash=booked["data_hash"],
    preregistration_id=art["preregistration_id"], n_trials=0, result=result,
    metrics={"verdict": status, **art["values"]},
    notes="Oordeel op de ontwikkelsample; bindend: " + ", ".join(art["verdict"]["binding"]),
    amends=booked["config_hash"]))
```

Run daarna de volledige suite (de sweep in `test_measurement_carries_policy_hash.py` leest nu het campagne-artefact en moet de policy-hash accepteren).

```bash
git add artefacts/governance/weekly_meta_campaign.json artefacts/governance/hypothesis_ledger.json
git commit -m "governance(weekly): the development verdict, booked as an amendment with zero trials"
```

- [ ] **Step 6: Rapporteer aan de eigenaar**

Geef het oordeel, de bindende criteria, de netto Sharpe met SE, de DSR, het aantal trades, het trefpercentage tegen break-even, de PBO, de Monte Carlo-drawdownkans en de CPCV-padverdeling. Bij iets anders dan `PASS`: stop hier (spec §14); Taak 14 wordt niet uitgevoerd en er wordt aan geen enkele parameter gedraaid.

---

### Task 14: De holdout-rooktest (alleen na `PASS`)

**Files:**
- Create: `src/tradebot/validation/weekly_holdout.py`
- Test: `tests/unit/test_weekly_holdout.py`

**Interfaces:**
- Consumes: `gate_slice(frame, *, lock_path, hypothesis_id)` (`validation/holdout.py`); `build_weekly_dataset`, `fit_light_model`, `trade_candidates`, `run_barrier_book`, `judge`; `circular_block_indices`.
- Produces:
  - `holdout_brier_difference(p: np.ndarray, y: np.ndarray, base_rate: float, *, seed: int, n_boot: int = 2000) -> tuple[float, float]` (punt, 95%-ondergrens van Brier(model) − Brier(basisfrequentie))
  - `run_holdout_smoke(root: Path) -> dict` en CLI `python -m tradebot.validation.weekly_holdout`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_weekly_holdout.py
from __future__ import annotations

import json

import numpy as np
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.weekly_holdout import holdout_brier_difference, require_pass


def test_a_perfect_model_beats_the_base_rate() -> None:
    y = np.array([0, 1] * 30)
    point, low = holdout_brier_difference(y.astype(float), y, 0.5, seed=1)
    assert point < 0.0 and low < 0.0


def test_a_confidently_wrong_model_is_significantly_worse() -> None:
    y = np.array([0, 1] * 30)
    point, low = holdout_brier_difference(1.0 - y.astype(float), y, 0.5, seed=1)
    assert point > 0.0 and low > 0.0


def test_the_holdout_is_refused_without_a_pass(tmp_path) -> None:
    art = tmp_path / "campaign.json"
    art.write_text(json.dumps({"verdict": {"status": "UNPROVEN", "binding": ["dsr_below_threshold"]}}),
                   encoding="utf-8")
    with pytest.raises(DataContractError, match="PASS"):
        require_pass(art)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_holdout.py -p no:randomly -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradebot.validation.weekly_holdout'`

- [ ] **Step 3: Implement**

```python
# src/tradebot/validation/weekly_holdout.py
"""De eenmalige holdout-rooktest (spec §9, R7).

Alleen na een ontwikkeloordeel PASS. De lezing wordt geregistreerd VOORDAT er
een uitkomst wordt bekeken (`gate_slice`). Het model wordt gefit op alle
ontwikkelevents waarvan het label vóór de split eindigt, en scoort ALLE
holdout-events: bij 4-16 trades is een trefkans geen bewijs, een
kalibratietoets op ~50 events wel een rooktest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.weekly_market import load_weekly_market
from ..features.registry import current_git_sha
from ..labeling.barrier_fills import round_trip_cost
from ..registry.preregistration import require_preregistration
from ..schemas.config import ExecutionConfig, load_config
from ..schemas.weekly_meta import weekly_meta_config
from ..train.light_models import fit_light_model
from ..train.weekly_dataset import build_weekly_dataset
from ..utils.failfast import DataContractError, require
from .holdout import gate_slice
from .inference import circular_block_indices
from .weekly_verdict import judge

__all__ = ["holdout_brier_difference", "require_pass", "run_holdout_smoke"]

CAMPAIGN = Path("artefacts/governance/weekly_meta_campaign.json")


def require_pass(campaign_path: Path) -> dict[str, Any]:
    record = json.loads(campaign_path.read_text(encoding="utf-8"))
    require(record["verdict"]["status"] == "PASS",
            "De holdout wordt alleen gelezen na een ontwikkeloordeel PASS (R7).",
            DataContractError, status=record["verdict"]["status"])
    return record


def holdout_brier_difference(
    p: np.ndarray, y: np.ndarray, base_rate: float, *, seed: int, n_boot: int = 2000,
) -> tuple[float, float]:
    """Brier(model) - Brier(basisfrequentie): punt en 95%-ondergrens (circulaire blokbootstrap)."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    d = (p - y) ** 2 - (base_rate - y) ** 2
    idx = circular_block_indices(d.size, max(1, d.size // 10), n_boot, np.random.default_rng(seed))
    return float(d.mean()), float(np.quantile(d[idx].mean(axis=1), 0.05))


def run_holdout_smoke(root: Path) -> dict[str, Any]:
    cfg = weekly_meta_config()
    record = require_pass(root / CAMPAIGN)
    prereg_id = record["preregistration_id"]
    prereg = require_preregistration(prereg_id, directory=root / "artefacts/governance")
    market = load_weekly_market(root, cfg.symbols)
    gate = gate_slice(pd.DataFrame(index=market.grid),
                      lock_path=root / "artefacts/governance/holdout_lock.json",
                      hypothesis_id=prereg_id)
    split_pos = int(market.grid.searchsorted(gate.index[0]))
    cost_rt = round_trip_cost(load_config(root / "conf/execution/fees.yaml", ExecutionConfig))
    wd = build_weekly_dataset(market, cfg, k=record["k"], d_star=record["d_star"], cost_rt=cost_rt)
    ds = wd.dataset
    train = np.flatnonzero(ds.exit_bar < split_pos)
    test = np.flatnonzero(ds.event_bar >= split_pos)
    require(test.size > 0, "Geen enkel volledig gelabeld event in de holdout.", DataContractError)
    model = fit_light_model(ds, train, ds.target, "ensemble", cfg)
    p = model.predict_proba(ds.features.iloc[test].to_numpy(dtype=np.float64))[:, 1]
    y = ds.target[test]
    point, low = holdout_brier_difference(p, y, float(ds.target[train].mean()), seed=cfg.seed)
    verdict = judge(prereg.stop_criteria, {"holdout_brier_diff_ci_low": low}, stage="holdout")
    out = {"verdict": verdict.as_dict(), "n_events": int(test.size),
           "brier_difference": point, "brier_difference_ci_low": low,
           "preregistration_id": prereg_id, "git_sha": current_git_sha()}
    (root / "artefacts/governance/weekly_meta_holdout.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run_holdout_smoke(Path.cwd()), indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_weekly_holdout.py -p no:randomly -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit de code**

```bash
git add src/tradebot/validation/weekly_holdout.py tests/unit/test_weekly_holdout.py
git commit -m "feat(validation): the one-shot holdout smoke test, refused without a development PASS"
```

- [ ] **Step 6 (alleen bij PASS, na go van de eigenaar): Lees de holdout één keer**

Run: `D:/venv/tradebot/Scripts/python.exe -m tradebot.validation.weekly_holdout`
Commit `artefacts/governance/holdout_lock.json` (nu met één lezing) en `artefacts/governance/weekly_meta_holdout.json`, en boek een ledger-amendement zoals in Taak 13 stap 5 met `unit="weekly_meta_holdout"`. Rapporteer. Bij `PASS` volgt het schrijven van Plan 2 (live); bij iets anders stopt het programma hier.

---

## Self-review (uitgevoerd bij het schrijven)

- **Spec-dekking:** §4 data → T4; §5 events en k → T2; §6 labels, 1:1, pessimistische dubbele touch (bestaand in `vol_barriers`), kostenbewust doel, uniqueness (bestaand in `build_dataset`) → T3, T6; §7 features → T5 (GARCH vervallen, §17.4); §8 modellen en kalibratie → T7; §9 holdout, M, walk-forward, CPCV, negatieve controles, inferentie → T11, T12, T13, T14; §10 barrièretheorie, break-even, handelsdrempel, Kelly, posterior, correlatie, drawdown → T8, T9, T11; §11 mandaat via `RiskEngine`, exits op de barrière → T9; §12 kosten → T3, T9; §13 live-bewaking → Plan 2; §14 stopregels → T10, T12, T13.
- **Bewust niet in Plan 1:** SPRT en Beta-posterior-bewaking live, papertrading, orders bij de exchange, data bijwerken (Plan 2).
- **Typeconsistentie:** `fit_light_model(dataset, rows, labels, kind, cfg)` in T7, T11, T14; `trade_candidates(events, folds, *, cfg, cost_rt)` in T11 en de CPCV-helper; `run_barrier_book(candidates, inputs, risk, sizing, *, equity0)` in T9, T11; `judge(criteria, values, *, stage)` in T10, T11, T14; `build_weekly_dataset(market, cfg, *, k, d_star, cost_rt, side_sign)` in T6, T11, T14.
