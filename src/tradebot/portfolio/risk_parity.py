# src/tradebot/portfolio/risk_parity.py
"""Equal Risk Contribution (ERC) / Risk Parity portfolio.

Each asset contributes equally to total portfolio variance.

Reference: Maillard, Roncalli & Teïletche (2010) "The Properties of
Equally Weighted Risk Contribution Portfolios".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

__all__ = ["erc_weights", "risk_contributions"]


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute marginal risk contribution of each asset.

    RC_i = w_i * (Σw)_i / (w'Σw)
    """
    sigma_w = cov @ weights
    port_var = float(weights @ sigma_w)
    if port_var < 1e-12:
        return np.ones(len(weights)) / len(weights)
    return weights * sigma_w / port_var


def erc_weights(
    returns: pd.DataFrame,
    max_weight: float = 0.5,
    tol: float = 1e-8,
) -> pd.Series:
    """Equal Risk Contribution portfolio weights.

    Parameters
    ----------
    returns :
        Returns DataFrame (rows=time, cols=assets).
    max_weight :
        Per-asset maximum weight.
    tol :
        Optimiser tolerance.

    Returns
    -------
    pd.Series of weights summing to 1.
    """
    assets = returns.columns.tolist()
    n = len(assets)

    # Drop NaN rows rather than filling with 0 — fillna(0) decorrelates
    # assets artificially during data gaps (phantom diversification).
    clean = returns.dropna(how="any")
    if len(clean) < 5:
        clean = returns.fillna(returns.mean())
    lw = LedoitWolf()
    lw.fit(clean.values)
    cov = lw.covariance_

    w0 = np.ones(n) / n
    target_rc = 1.0 / n  # equal contribution

    def objective(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov)
        return float(np.sum((rc - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, max_weight)] * n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": 1000},
    )

    if not result.success:
        logger.warning("ERC solve failed: %s — inverse-vol fallback.", result.message)
        vol = np.sqrt(np.diag(cov)) + 1e-9
        w = 1.0 / vol
        w /= w.sum()
        return pd.Series(w, index=assets, name="erc_weight")

    w = np.maximum(result.x, 0.0)
    w /= w.sum()
    return pd.Series(w, index=assets, name="erc_weight")


# =========================================================================== #
# PHASE 3 - L8 NAIVE RISK PARITY (production default)
# =========================================================================== #
# Alles BOVEN deze regel is de ERC-optimalisatie (Equal Risk Contribution):
# een SLSQP-solve op een Ledoit-Wolf covariantie. Die code wordt gebruikt door
# `portfolio/optimizer.py` en is hier ONGEWIJZIGD gelaten, maar hij is voor
# Phase 3 GEEN toegestane allocator, om twee redenen:
#
#   1. Hij bevat twee stille degradaties. `returns.fillna(returns.mean())` bij
#      te weinig schone rijen vult gaten met het sample-gemiddelde; en bij een
#      mislukte solve logt hij een warning en valt terug op inverse vol. Dat
#      laatste is precies het scenario uit audit sectie 5.2: het model dat je
#      denkt te draaien is niet het model dat draait.
#   2. Audit sectie 13.1 wijst voor Phase 3 NAIVE Risk Parity aan als production
#      default, niet ERC. Level 2+ allocators moeten hun plek verdienen, en dat
#      gebeurt in Phase 6.
#
# Alles HIERONDER is de Level-1 allocator: gewichten omgekeerd evenredig aan de
# CAUSALE EWMA-volatiliteit uit L2. Geen optimalisatie, geen covariantie, geen
# solver die kan falen - en dus geen fallback die stil kan degraderen.
# --------------------------------------------------------------------------- #
from ..utils.failfast import DataContractError as _DataContractError
from ..utils.failfast import require as _require
from .equal_weight import normalise_to_gross as _normalise_to_gross


def _require_finite_vol_where_needed(
    vol: pd.DataFrame, needed: np.ndarray, *, context: str
) -> None:
    """Crash zodra er een positie wordt gevraagd zonder bruikbare vol-schatting.

    Dit is het hart van deliverable 6. De verleiding is om het asset dan maar
    op gelijk gewicht te zetten of over te slaan; beide zijn de stille
    degradatie uit audit sectie 5.2. Een ontbrekende vol-schatting betekent dat
    de allocator niet weet hoe groot deze positie mag zijn, en dat is geen
    situatie om doorheen te rekenen.
    """
    values = vol.to_numpy(dtype="float64")
    unusable = needed & (~np.isfinite(values) | (values <= 0.0))
    if not bool(unusable.any()):
        return
    rows = np.flatnonzero(unusable.any(axis=1))
    first = int(rows[0])
    cols = [str(vol.columns[j]) for j in np.flatnonzero(unusable[first])]
    _require(
        False,
        "Er wordt een positie gevraagd in een asset zonder bruikbare "
        "vol-schatting (ontbrekend, niet-eindig of nul). Er is GEEN fallback: "
        "het asset stilzwijgend op gelijk gewicht zetten is precies de stille "
        "degradatie uit audit sectie 5.2, en een vol van nul zou een oneindig "
        "gewicht opleveren.",
        _DataContractError,
        context=context,
        n_offending_cells=int(unusable.sum()),
        first_offending_row=str(vol.index[first]),
        first_offending_symbols=cols,
    )


def inverse_volatility_long_only(
    vol: pd.DataFrame, *, gross_target: float
) -> pd.DataFrame:
    """Naive Risk Parity, long-only: `w_i ~ 1 / sigma_i`.

    De referentie-allocator zonder alpha. Elk asset dat op `t` een geldige
    vol-schatting heeft, krijgt een gewicht omgekeerd evenredig aan die
    schatting; assets zonder schatting (nog in hun burn-in, of nog niet
    gelanceerd) krijgen geen positie. Dat laatste is geen degradatie maar de
    juiste uitkomst: er is nog geen basis om ze te sizen.
    """
    _require(
        len(vol.columns) > 0,
        "Leeg universum; er valt niets te verdelen.",
        _DataContractError,
    )
    values = vol.to_numpy(dtype="float64")
    usable = np.isfinite(values) & (values > 0.0)
    raw = np.where(usable, 1.0 / np.where(usable, values, 1.0), 0.0)
    return _normalise_to_gross(
        pd.DataFrame(raw, index=vol.index, columns=vol.columns),
        gross_target=gross_target,
    )


def sized_by_risk_parity(
    exposures: pd.DataFrame, vol: pd.DataFrame, *, gross_target: float
) -> pd.DataFrame:
    """Naive Risk Parity op een alpha-view: `w_i ~ a_i / sigma_i`.

    De gewenste exposure uit L4 bepaalt de RICHTING en de relatieve sterkte; de
    causale EWMA-vol uit L2 bepaalt hoe groot die view mag worden vertaald in
    kapitaal. Een asset met de dubbele volatiliteit krijgt bij dezelfde view het
    halve gewicht, zodat elke positie ex ante ongeveer evenveel risico draagt.

    Ontbreekt de vol-schatting terwijl er WEL een view is, dan crasht deze
    functie. Ontbreekt de view, dan is er geen positie en is de vol irrelevant -
    dat onderscheid is precies waarom de controle op de gevraagde cellen kijkt
    en niet op het hele panel.
    """
    _require(
        list(exposures.columns) == list(vol.columns),
        "Exposure- en vol-panel beschrijven een verschillend universum.",
        _DataContractError,
        exposure_columns=list(exposures.columns),
        vol_columns=list(vol.columns),
    )
    _require(
        bool(exposures.index.equals(vol.index)),
        "Exposure- en vol-panel staan niet op dezelfde tijdas.",
        _DataContractError,
    )
    a = exposures.to_numpy(dtype="float64")
    needed = np.isfinite(a) & (a != 0.0)
    _require_finite_vol_where_needed(vol, needed, context="sized_by_risk_parity")

    sigma = vol.to_numpy(dtype="float64")
    raw = np.where(needed, np.divide(a, sigma, where=needed, out=np.zeros_like(a)), 0.0)
    return _normalise_to_gross(
        pd.DataFrame(raw, index=exposures.index, columns=exposures.columns),
        gross_target=gross_target,
    )
