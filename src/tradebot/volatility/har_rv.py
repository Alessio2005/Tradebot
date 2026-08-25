"""HAR-RV (Heterogeneous Autoregressive Realized Volatility) — Corsi 2009.

Long-memory volatility model that empirically outperforms EWMA on crypto.
Wave 18 microstructure upgrade.

CHIEF AUDIT 2026-05-23 (P-12) — CPCV USAGE CONTRACT
====================================================
``har_rv_fit`` is an OLS regression on the FULL input series.  If the caller
passes the entire backtest dataset, the β coefficients are fit using bars from
EVERY future fold — the resulting in-sample forecasts contain look-ahead.

CORRECT USAGE inside a CPCV / walk-forward loop:
    - Slice ``realized_var`` to the current TRAIN fold only, OR
    - Pass ``fit_indices`` explicitly so β is estimated on the TRAIN rows
      while forecasts are produced for ALL rows (test bars are then valid
      because their β did not see them).

A runtime warning fires when the input series exceeds 5000 bars without
``fit_indices`` being supplied — this is a strong heuristic for the
"someone passed the whole dataset" mistake.

PHASE 6 STAP 0 — NON-NEGATIVITEIT VAN DE VARIANTIEFORECAST
===========================================================
``test_har_rv_forecast_non_negative`` was rood op `84273ca`: op een RV-reeks die
vrijwel overal nul is (één spike) produceerde de ongeconstrainde OLS een
intercept en hellingen waarvan de gefitte waarden net onder nul uitkwamen.

Dat is geen afrondingsdetail maar een specificatiefout. HAR-RV in NIVEAUS is een
lineair model op een grootheid die per definitie niet-negatief is; de OLS-oplossing
kent die restrictie niet. HAR-RV is Level 3 in de vol-hiërarchie (§9.1) en zijn
forecast is de noemer van QLIKE:

    QLIKE = RV_t / sigma^2_t - ln(RV_t / sigma^2_t) - 1

Een negatieve noemer maakt ``ln`` ongedefinieerd en de hele QLIKE-competitie uit
Stap 7 zou dan op een kapotte metriek rusten.

De gekozen oplossing is **non-negative least squares** op [1, RV_d, RV_w, RV_m]:
alle vier de coëfficiënten worden beperkt tot >= 0. Dat is geen truc om een test
groen te krijgen maar de econometrisch juiste restrictie — de HAR-componenten
zijn gewichten waarmee volatiliteit op drie horizonnen doorwerkt in de verwachting
van morgen, en een negatief gewicht is niet interpreteerbaar. Corsi (2009)
rapporteert dan ook uitsluitend positieve beta's.

Waarom deze constructie en niet een clip op nul: met alle regressoren >= 0 (de
input is variantie) en alle coëfficiënten >= 0 is ``X @ beta`` een som van
niet-negatieve producten, en dus EXACT niet-negatief in IEEE-754. Er is geen
epsilon, geen drempel uit conf/ en geen geval waarin de garantie net niet geldt.
Een clip zou daarentegen een materieel negatieve forecast stilzwijgend op nul
zetten — dezelfde klasse stille degradatie die §26-AC5 verbiedt.

De restrictie is bovendien niet stil: ``HARRVResult.n_coefficients_clamped``
telt hoeveel coëfficiënten door de restrictie op nul zijn gezet, zodat elk
rapport kan laten zien hoe vaak zij bindt. Bindt zij vaak, dan is dat een
resultaat over de data, geen verborgen ingreep.

De invoercontract-check (`realized_var` eindig en >= 0) hoort bij dezelfde
garantie: zonder haar zou een aanroeper die per ongeluk rendementen in plaats van
gekwadrateerde rendementen doorgeeft, de non-negativiteit alsnog kunnen breken.
"""
from __future__ import annotations

import logging
import warnings
from typing import NamedTuple

import numpy as np
from scipy.optimize import nnls

from ..utils.failfast import DataContractError, require

logger = logging.getLogger(__name__)


class HARRVResult(NamedTuple):
    """HAR-RV fit results."""
    c: float       # intercept                        (>= 0, NNLS-restrictie)
    beta_d: float  # daily RV coefficient             (>= 0)
    beta_w: float  # weekly RV (5-bar) coefficient    (>= 0)
    beta_m: float  # monthly RV (21-bar) coefficient  (>= 0)
    forecast: np.ndarray  # 1-step-ahead forecasts (same length as input)
    r_squared: float
    #: Aantal coëfficiënten dat de non-negativiteitsrestrictie op exact nul
    #: heeft gezet. 0 betekent dat de restrictie niet bond en de fit gelijk is
    #: aan de ongeconstrainde OLS. Hoger dan 0 is een RESULTAAT over de reeks
    #: dat in het rapport hoort, geen verborgen ingreep.
    n_coefficients_clamped: int = 0


def har_rv_fit(
    realized_var: np.ndarray,
    daily_window: int = 1,
    weekly_window: int = 5,
    monthly_window: int = 21,
    fit_indices: np.ndarray | None = None,
) -> HARRVResult:
    """Fit HAR-RV model to realized variance series (Corsi 2009).

    Model: RV_t = c + β_d * RV_{t-1} + β_w * RV̄_{t-1:t-5} + β_m * RV̄_{t-1:t-21} + ε_t

    Args:
        realized_var : 1D array of realized variance (squared returns or RV estimator).
        daily_window : lag for daily component (default 1).
        weekly_window: lags for weekly avg (default 5).
        monthly_window: lags for monthly avg (default 21).
        fit_indices  : CHIEF AUDIT 2026-05-23 (P-12) — integer indices into
                       ``realized_var`` that define the TRAIN slice on which β
                       is estimated.  Forecasts are still produced for ALL
                       rows of ``realized_var`` (so the test rows have a
                       prediction).  ``None`` keeps the legacy global-fit
                       behaviour but triggers a warning for large inputs.

    Returns:
        HARRVResult with fitted coefficients and 1-step forecasts.
    """
    rv = np.asarray(realized_var, dtype=np.float64)
    n = len(rv)
    min_obs = monthly_window + 2
    if n < min_obs:
        raise ValueError(f"HAR-RV needs at least {min_obs} observations, got {n}")

    # Phase 6 stap 0: invoercontract. `realized_var` is een VARIANTIE-reeks.
    # Zonder deze check kan een aanroeper die rendementen doorgeeft in plaats
    # van gekwadrateerde rendementen de non-negativiteitsgarantie van de
    # forecast breken, want die garantie steunt erop dat elke regressor >= 0 is.
    require(
        bool(np.all(np.isfinite(rv))),
        "HAR-RV kreeg een niet-eindige realized-variance reeks. Er wordt NIET "
        "geïmputeerd of gefilterd: een gat in de RV-proxy is een databevinding "
        "die in het dekkingsrapport hoort, geen ruis om weg te vangen.",
        DataContractError,
        n_non_finite=int(np.count_nonzero(~np.isfinite(rv))),
        n_obs=n,
    )
    require(
        bool(np.all(rv >= 0.0)),
        "HAR-RV kreeg NEGATIEVE realized variance. Variantie is niet-negatief; "
        "de meest waarschijnlijke oorzaak is dat er rendementen zijn "
        "doorgegeven in plaats van gekwadrateerde rendementen.",
        DataContractError,
        min_value=float(np.min(rv)),
        n_negative=int(np.count_nonzero(rv < 0.0)),
    )

    # CHIEF AUDIT 2026-05-23 (P-12): warn loudly when the caller passes a
    # large series without specifying ``fit_indices``.  A 5000-bar threshold
    # is a heuristic that catches the "passed the whole CPCV dataset"
    # mistake without spamming legitimate per-fold callers (TRAIN fold is
    # typically smaller).
    if fit_indices is None and n > 5000:
        warnings.warn(
            f"har_rv_fit called on {n} bars without fit_indices — β will be "
            "estimated using EVERY bar, including future folds.  Inside a "
            "CPCV / walk-forward loop you MUST pass fit_indices=<train_idx> "
            "or slice realized_var to the train fold beforehand.",
            UserWarning,
            stacklevel=2,
        )

    # Construct feature matrix
    start = monthly_window
    y = rv[start:]
    T = len(y)

    # Daily: RV_{t-1}
    rv_d = rv[start - daily_window : start - daily_window + T]

    # Weekly: rolling mean over [t-weekly_window, t-1]
    rv_w = np.array([
        np.mean(rv[max(0, start + i - weekly_window) : start + i])
        for i in range(T)
    ])

    # Monthly: rolling mean over [t-monthly_window, t-1]
    rv_m = np.array([
        np.mean(rv[max(0, start + i - monthly_window) : start + i])
        for i in range(T)
    ])

    X = np.column_stack([np.ones(T), rv_d, rv_w, rv_m])

    # CHIEF AUDIT 2026-05-23 (P-12): fit β on ONLY the TRAIN rows when
    # fit_indices is provided.  Forecasts are then produced for the FULL
    # series so test bars receive an unbiased β-prediction.
    if fit_indices is not None:
        fit_idx_arr = np.asarray(fit_indices, dtype=np.int64)
        # Convert original-array indices into X/y row indices (X is shifted
        # forward by ``start`` bars).
        local_idx = fit_idx_arr[fit_idx_arr >= start] - start
        local_idx = local_idx[local_idx < T]
        if local_idx.size < 5:
            logger.warning(
                "HAR-RV fit_indices yielded %d usable rows (<5) — falling "
                "back to full-series fit.", local_idx.size,
            )
            X_fit, y_fit = X, y
        else:
            X_fit, y_fit = X[local_idx], y[local_idx]
    else:
        X_fit, y_fit = X, y

    # Non-negative least squares.
    #
    # Phase 0: `except LinAlgError: beta = [mean(y), 0, 0, 0]` degradeerde het
    # HAR-RV-model (Level 2) naar een CONSTANTE gemiddelde-voorspelling
    # (Level 0) - met beta_d = beta_w = beta_m = 0 verdwijnt de volledige
    # heterogene-autoregressiestructuur. Elke QLIKE-vergelijking tegen EWMA zou
    # dan feitelijk EWMA-vs-constante zijn. Die fallback blijft weg.
    #
    # Phase 6 stap 0: de restrictie beta >= 0. Zie de moduledocstring — dit is
    # wat de non-negativiteit van de forecast EXACT maakt in plaats van bij
    # benadering, omdat X >= 0 door het invoercontract hierboven.
    try:
        beta, _residual_norm = nnls(X_fit, y_fit)
    except (np.linalg.LinAlgError, RuntimeError) as exc:
        raise DataContractError(
            "HAR-RV non-negative-least-squares-fit convergeerde niet. Er wordt "
            "NIET teruggevallen op een ongeconstrainde OLS of op een constante "
            "gemiddelde-voorspelling; een niet-convergerende fit is een "
            "resultaat dat wordt geregistreerd."
        ) from exc

    c, beta_d, beta_w, beta_m = float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])
    n_clamped = int(np.count_nonzero(beta <= 0.0))

    forecasts_in_sample = X @ beta

    # De garantie hierboven is een redenering; dit is de meting. Faalt zij, dan
    # is een aanname onder het model gebroken en mag er geen forecast naar de
    # QLIKE-competitie ontsnappen.
    require(
        bool(np.all(forecasts_in_sample >= 0.0)),
        "HAR-RV produceerde een negatieve variantieforecast ondanks de "
        "NNLS-restrictie en het niet-negatieve invoercontract. Dit hoort "
        "onmogelijk te zijn; QLIKE zou hierop ongedefinieerd worden.",
        DataContractError,
        min_forecast=float(np.min(forecasts_in_sample)),
        beta=[c, beta_d, beta_w, beta_m],
    )

    # Pad with NaN for the warm-up period
    full_forecast = np.full(n, np.nan, dtype=np.float64)
    full_forecast[start:] = forecasts_in_sample

    # R-squared (on the FIT rows so the score reflects the train-slice fit)
    fit_pred = X_fit @ beta
    ss_res = float(np.sum((y_fit - fit_pred)**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return HARRVResult(
        c=c, beta_d=beta_d, beta_w=beta_w, beta_m=beta_m,
        forecast=full_forecast, r_squared=r2,
        n_coefficients_clamped=n_clamped,
    )


def har_rv_feature(
    realized_var: np.ndarray,
    daily_window: int = 1,
    weekly_window: int = 5,
    monthly_window: int = 21,
) -> np.ndarray:
    """Return HAR-RV 1-step-ahead forecast as a feature series.

    Suitable for use as a volatility feature in the ML pipeline.
    Returns NaN for the first monthly_window bars (warm-up).
    """
    result = har_rv_fit(realized_var, daily_window, weekly_window, monthly_window)
    return result.forecast
