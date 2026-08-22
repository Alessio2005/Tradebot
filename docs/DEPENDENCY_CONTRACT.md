# DEPENDENCY CONTRACT

> **Geverifieerd tegen de codebase op 2026-08-22, Phase 0.**
> Bindend brondocument: `ARCHITECTUUR_AUDIT_2026-08-22.md` — D-10, sectie 5.2, sectie 26 (AC-5).

---

## De regel

> **Elke dependency hieronder is HARD. Ontbreekt er een, dan crasht het platform
> bij import. Er bestaat geen codepad waarin een ontbrekende dependency leidt tot
> een naïevere benadering.**

Dit is geen stijlvoorkeur. Vóór Phase 0 ontbraken `hmmlearn`, `arch`,
`statsmodels` en `pydantic` volledig in `pyproject.toml`, terwijl ze op de
ontwikkelmachine wél geïnstalleerd waren. De fallbacks waren daardoor **latent**:
ze zouden pas vuren op een schone build — precies de omgeving waarin niemand
meekijkt. De "HMM regime detector" zou daar een 20/100 EMA-crossover draaien,
terwijl elk rapport en elke ledger-entry hem als HMM registreerde.

`scripts/audit_fallbacks.py --strict` dwingt dit statisch af en draait als
blokkerende stap in `.github/workflows/hygiene.yml`.

---

## Harde afhankelijkheden per laag

| Laag | Dependency | Waarvoor | Gevolg bij afwezigheid |
|---|---|---|---|
| **L0** Data | `pandas`, `numpy` | alle tijdreeksen | crash bij import |
| L0 | `pyarrow` | Parquet PIT-store | crash bij import |
| L0 | `pandera` | schema-contracten op stage-grenzen | crash bij import |
| L0 | `requests`, `aiohttp` | Bybit V5 REST-ingestion | crash bij import |
| L0 | `dvc` | data-lineage, `data_hash` ↔ DVC-hash | crash bij import |
| L0 | `PyYAML` | config- en registerbestanden | crash bij import |
| **L1** Features | `scipy` | Hilbert-fase, hiërarchische clustering, Spearman | crash bij import |
| L1 | `numba` | JIT-kernels (FFD-gewichten, rolling stats) | crash bij import |
| L1 | `statsmodels` | **ADF/KPSS-gate**, `min_frac_diff`, Ljung-Box | crash bij import |
| L1 | `scikit-learn` ≥1.6 | PCA-orthogonalisatie, `sklearn.frozen.FrozenEstimator` | crash bij import |
| **L2** Volatility | `arch` | GARCH/EGARCH/GJR (Phase 6) | crash bij import |
| **L3** Regime | `hmmlearn` | 3-state Gaussian HMM | crash bij import |
| **L4–L6** Alpha/ML | `catboost` | meta-labeling Judge, SFI | crash bij import |
| L4–L6 | `optuna` ≥3.4 | hyperparameter-search | crash bij import |
| L4–L6 | `joblib` | parallelle folds | crash bij import |
| **L9** Execution | `aiohttp` | order-routing, positie-query | crash bij import |
| **L13** Live/Ops | `websockets` | L1/aggtrade feed | crash bij import |
| L13 | `prometheus-client` | metrics-export | crash bij import |
| **Cross-cutting** | `pydantic` ≥2 | config-contracten (`extra="forbid"`, `frozen=True`) | crash bij import |
| Cross-cutting | `hydra-core`, `omegaconf` | compositie van `conf/` | crash bij import |

---

## Wat *wel* optioneel mag zijn

Precies één categorie: **capability-probes waarbij afwezigheid betekent dat de
capability feitelijk niet in gebruik is** — en er dus geen model, geen schatting
en geen statistische claim degradeert.

Uitsluitend toegestaan via `tradebot.utils.failfast.has_module`, dat
`importlib.util.find_spec` gebruikt en dus geen `try/except` bevat.

| Module | Gebruikt in | Waarom dit géén degradatie is |
|---|---|---|
| `torch` | `train/seeded.py` | Zonder torch bestaat er geen torch-model om te seeden. |
| `cupy` | `train/seeded.py` | Idem voor GPU-RNG. |
| `mlflow` | `registry/experiment.py` | De JSONL-ledger is de **autoriteit**; MLflow is een extra spiegel. |
| `ccxt` | optionele extra `[ingestion]` | Alternatieve venue-adapter. De Phase 1-ingestion gebruikt Bybit V5 rechtstreeks via `requests`, zodat geen enkel ingestion-pad van een optionele package afhangt. |

**Verboden voorbeeld**, expliciet benoemd in de docstring van `has_module`:

```python
if has_module("hmmlearn"):   # FOUT
    return hmm_regimes(x)
return ema_crossover(x)
```

Dat is D-10 in een nieuw jasje. Gebruik `require_dependency`.

---

## Handhaving

| Mechanisme | Waar | Wat het blokkeert |
|---|---|---|
| `scripts/audit_fallbacks.py --strict` | CI + `tests/unit/test_no_silent_fallbacks.py` | elke `try/except ImportError`, bare except, `except Exception` zonder re-raise, en warn-then-degrade |
| `requirements.lock` | CI-stap "Install from lockfile" | build reproduceerbaar uit `pyproject.toml` + lockfile, zonder handmatige installatie |
| CI-stap "Verify dependency contract is closed" | `.github/workflows/hygiene.yml` | faalt wanneer één van de 17 harde dependencies na een lockfile-install ontbreekt |
| `require_dependency(...)` | `src/tradebot/utils/failfast.py` | het enige toegestane idioom voor een import die zou kúnnen ontbreken |

---

## Reproduceerbare build

```bash
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.lock
pip install -e . --no-deps
pytest tests/unit -q
```

`requirements.lock` is gegenereerd met `pip-compile --strip-extras pyproject.toml`
en bevat de volledige transitieve closure (133 pakketten, gepind).
