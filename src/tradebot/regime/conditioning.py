# src/tradebot/regime/conditioning.py
"""Van een regimemodel naar een exposure-factor. H2, stap 11.

DE AFBEELDING IS VOOR ALLE DRIE DEZELFDE, EN DAAROM IS DIT EEN VERGELIJKING
===========================================================================
De vraag van H2 is of het M2 Filtered HMM als CONDITIONERINGSLAAG beter is dan
M0 Causal Vol-Buckets. Dat is alleen een vraag over MODELLEN als de afbeelding
van model naar positie voor beide identiek is::

    a_geconditioneerd[t] = a_basis[t] * (1 - p_hoog[t])

met `p_hoog` de kans (of, bij M0, de zekerheid) dat bar `t` in het onrustigste
regime valt:

    M0  1{bucket_t = HOOG}                     indicator, nul geschatte parameters
    M1  P(S_t = HOOG | S_{t-1})                eenstapsvoorspelling, k(k-1) parameters
    M2  P(S_t = onrustigste toestand | F_t)    filtered posterior, latente toestanden

Er is GEEN vrije parameter in die afbeelding. Geen multiplier per bucket, geen
drempel op de kans, geen schaling. Dat is een bewuste beperking: elke vrije
parameter hierin zou een trial zijn die niet in pre-registratie
`3d3af28730a6c7f9da48d13139522a05` staat, en zij zou de winnaar kunnen bepalen
zonder dat een van de regimemodellen iets had gedaan. Wat de modellen mogen
doen is uitsluitend `p_hoog` schatten.

ALLE DRIE ZIEN PRECIES `F_t`, EN DAT IS WAAR ZE VERSCHILLEN
============================================================
M0 en M2 SCHATTEN de huidige toestand; M1 VOORSPELT de volgende. Alle drie
gebruiken uitsluitend informatie tot en met de close van `t`. Dat M1 daarmee de
bucket van `t` niet gebruikt, is geen handicap die hier is opgelegd -- het is
wat een Markov-keten op waargenomen toestanden IS. Zou hij de bucket van `t`
zelf gebruiken, dan is hij M0 met extra stappen.

De positie die op deze factor volgt, bestaat pas op `t+2`: de engine legt
`latency_bars = 1` bovenop de beslisbar (§0.2). Die verschuiving staat in de
engine en niet hier, precies zoals in `buckets.py` beargumenteerd -- wie hem
hier zou inbouwen, verschuift ook de diagnostiek en maakt de regime-occupancy
onvergelijkbaar tussen de armen.

BUITEN HET OOS-MASKER IS DE FACTOR EXACT 1, IN ELKE ARM
========================================================
De walk-forward levert twaalf testvensters; alleen daar is een fit
out-of-sample beoordeeld. Op alle andere bars krijgt ELKE arm factor 1 en zijn
de armen dus bitidentiek. Zonder die regel zou de M2-arm op de trainbars een
in-sample geconditioneerde equity-curve opbouwen en zou het Sharpe-verschil
deels uit die bars komen -- bars waarop M0 en M2 niet vergelijkbaar zijn omdat
de een niets schat en de ander alles al gezien heeft.

Ref: pre-registratie `3d3af28730a6c7f9da48d13139522a05`; fase-opdracht stap 11,
deliverable 18; ARCHITECTUUR_AUDIT_2026-08-22.md 10.1, 10.2.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..cv.walk_forward import WalkForwardCV, WalkForwardFold
from ..schemas.config import AdequacyConfig, M2HmmConfig
from ..utils.failfast import DataContractError, require
from .buckets import VolBucket
from .markov import (
    FilteredProbabilities,
    HmmParameters,
    HmmSpec,
    MarkovChain,
    filtered_occupancy,
    fit_hmm,
    forward_filter,
    require_filtered,
)

__all__ = [
    "CONDITIONER_SPECS",
    "ConditionerResult",
    "ConditionerSpec",
    "build_conditioner",
    "m0_multiplier",
    "oos_mask",
]


@dataclass(frozen=True)
class ConditionerSpec:
    """Eén trial uit de H2-parameterruimte."""

    family: Literal["m1", "m2"]
    n_states: int
    distribution: Literal["gaussian", "student_t"] = "gaussian"
    covariance_type: Literal["diag", "full"] = "diag"

    @property
    def label(self) -> str:
        if self.family == "m1":
            return f"m1-k{self.n_states}"
        return f"hmm{self.n_states}-{self.covariance_type}-{self.distribution}"

    @property
    def hmm_spec(self) -> HmmSpec:
        require(
            self.family == "m2",
            "Alleen een M2-trial heeft een HmmSpec; M1 schat geen latente "
            "toestanden.",
            DataContractError, family=self.family,
        )
        return HmmSpec(n_states=self.n_states,
                       covariance_type=self.covariance_type,
                       distribution=self.distribution)


def _spec_grid(
    n_states_grid: Sequence[int] = (2, 3), covariance_type: str = "diag",
) -> tuple[ConditionerSpec, ...]:
    """De zes gepre-registreerde trials, in vaste volgorde."""
    out: list[ConditionerSpec] = []
    for k in n_states_grid:
        out.append(ConditionerSpec("m1", k))
    for distribution in ("gaussian", "student_t"):
        for k in n_states_grid:
            out.append(ConditionerSpec(
                "m2", k, distribution=distribution,  # type: ignore[arg-type]
                covariance_type=covariance_type))    # type: ignore[arg-type]
    return tuple(out)


#: De zes trials van pre-registratie `3d3af28730a6c7f9da48d13139522a05`. Zij
#: staan hier als CONSTANTE en niet als iets dat een aanroeper mag doorgeven:
#: een campagne die zijn eigen ruimte samenstelt, kan er stilzwijgend een
#: zevende bij zetten en dan klopt `M` niet meer.
CONDITIONER_SPECS: tuple[ConditionerSpec, ...] = _spec_grid()


@dataclass(frozen=True)
class ConditionerResult:
    """De factor van één trial, met de diagnostiek die het rapport uitvraagt."""

    label: str
    #: (n_bars, n_symbols), waarden in [0, 1]. Buiten het masker exact 1.
    values: pd.DataFrame
    #: Het OOS-masker waarop deze factor iets anders dan 1 mag zijn.
    mask: np.ndarray
    #: Kans op het onrustigste regime, uitsluitend op de gescoorde bars.
    high_probability: pd.DataFrame
    #: Regimewisselingen per symbool op de gescoorde bars (argmax-toestand).
    n_transitions: Mapping[str, int]
    mean_duration_bars: Mapping[str, float]
    #: Gemiddelde |verandering| van de factor per bar. Dit is de grootheid die
    #: TURNOVER veroorzaakt, en die is voor een continue kans beter gedefinieerd
    #: dan een telling van wisselingen.
    mean_absolute_change: Mapping[str, float]
    #: Stop-criterium 1: de zeldzaamste FILTERED toestandsbezetting per fold,
    #: gemeten op het trainvenster waarop de parameters zijn geschat.
    rarest_state_obs_per_fold: float
    n_train_obs_per_fold: int
    #: Convergentie over alle (symbool, fold)-fits van deze trial.
    convergence_ratio: float
    fits: tuple[Mapping[str, Any], ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_scored_bars": int(self.mask.sum()),
            "n_transitions": dict(self.n_transitions),
            "mean_duration_bars": dict(self.mean_duration_bars),
            "mean_absolute_change": dict(self.mean_absolute_change),
            "rarest_state_obs_per_fold": self.rarest_state_obs_per_fold,
            "n_train_obs_per_fold": self.n_train_obs_per_fold,
            "convergence_ratio": self.convergence_ratio,
            "mean_multiplier": float(
                self.values.to_numpy()[self.mask].mean()),
            "fits": [dict(f) for f in self.fits],
        }


def oos_mask(
    n: int, cv: WalkForwardCV,
) -> tuple[np.ndarray, tuple[WalkForwardFold, ...]]:
    """Het masker van de testvensters, plus de folds die het opleveren.

    Dezelfde constructie als in H1: er wordt uitsluitend gescoord op bars die
    voor de betreffende fit out-of-sample zijn, en beide armen worden op
    dezelfde bars gescoord.
    """
    folds = tuple(cv.split(n))
    require(
        len(folds) > 0,
        "De walk-forward levert geen enkele fold; er is niets om out-of-sample "
        "op te beoordelen.",
        DataContractError, n_bars=n, train_size=cv.train_size,
        test_size=cv.test_size,
    )
    mask = np.zeros(n, dtype=bool)
    for fold in folds:
        mask[fold.test_idx] = True
    return mask, folds


def m0_multiplier(buckets: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """M0: ``1 - 1{bucket = HOOG}``. Nul geschatte parameters.

    Een ONGEDEFINIEERDE bucket binnen het masker crasht. Buiten het masker is
    hij onschadelijk -- daar geldt factor 1 voor elke arm -- maar op een bar
    waarop M0 wordt beoordeeld zou "geen oordeel" stilzwijgend "geen reductie"
    betekenen, en dat is een bewering over de markt in plaats van een
    onthouding.
    """
    require(
        len(buckets) == mask.size,
        "Het bucketpaneel en het OOS-masker staan niet op dezelfde tijdas.",
        DataContractError, n_buckets=len(buckets), n_mask=int(mask.size),
    )
    scored = buckets.to_numpy(dtype="float64")[mask]
    require(
        bool(np.all(np.isfinite(scored))),
        "Een ongedefinieerde M0-bucket op een bar die wordt beoordeeld. Daar "
        "zou 'geen oordeel' als 'geen reductie' worden gelezen, en dat is een "
        "bewering in plaats van een onthouding.",
        DataContractError,
        n_undefined=int(np.count_nonzero(~np.isfinite(scored))),
    )
    values = pd.DataFrame(
        1.0, index=buckets.index, columns=buckets.columns, dtype="float64")
    is_high = (buckets == float(VolBucket.HIGH)).to_numpy()
    values.values[mask & np.ones(len(buckets), dtype=bool)] = np.where(
        is_high[mask], 0.0, 1.0)
    return values


def _state_variance(params: HmmParameters) -> np.ndarray:
    """De VARIANTIE per toestand, ook wanneer de emissie een t is.

    Voor een Student-t is `covars` de SCHAAL en is de variantie
    ``Sigma * nu / (nu - 2)``. Rangschikken op de schaal zou toestanden met
    verschillende `nu` op de verkeerde volgorde zetten, en dan wijst "de
    onrustigste toestand" de verkeerde aan.
    """
    covars = np.asarray(params.covars, dtype=np.float64)
    if params.spec.covariance_type == "full":
        scale = np.array([np.trace(covars[i])
                          for i in range(params.spec.n_states)])
    else:
        scale = covars.sum(axis=1)
    if params.dof is None:
        return scale
    dof = np.asarray(params.dof, dtype=np.float64)
    return scale * dof / (dof - 2.0)


def _high_state(params: HmmParameters) -> int:
    """De index van de onrustigste toestand. De EM ordent niet."""
    return int(np.argmax(_state_variance(params)))


def _durations(states: np.ndarray) -> float:
    """Gemiddelde regimeduur in bars over een gescoorde reeks."""
    if states.size == 0:
        return 0.0
    changes = int(np.count_nonzero(states[1:] != states[:-1]))
    return float(states.size / (changes + 1))


def _m2_fold(
    observations: np.ndarray, spec: ConditionerSpec, fold: WalkForwardFold,
    *, symbol: str, adequacy: AdequacyConfig, m2_cfg: M2HmmConfig,
    n_obs_per_fold: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, HmmParameters, float]:
    """Eén (symbool, fold): fit op het trainvenster, filter tot het testeinde."""
    train_end = fold.train_end
    params = fit_hmm(
        observations[:fold.test_end], spec.hmm_spec, adequacy, symbol=symbol,
        fold_id=fold.fold_id, train_end=train_end,
        n_obs_per_fold=n_obs_per_fold, seed=m2_cfg.seed, m2_cfg=m2_cfg)
    high = _high_state(params)

    # De poort van stop-criterium 1 meet de FILTERED bezetting op de bars
    # waarop is GESCHAT; daar staan de parameters op, en daar telt het.
    train_filtered: FilteredProbabilities = require_filtered(
        forward_filter(observations[:train_end], params),
        context=f"H2-adequaatheid {spec.label} {symbol} fold {fold.fold_id}")
    rarest = float(filtered_occupancy(train_filtered).min() * train_end)

    filtered = require_filtered(
        forward_filter(observations[:fold.test_end], params),
        context=f"H2-conditionering {spec.label} {symbol} fold {fold.fold_id}")
    rows = filtered.values[fold.test_idx]
    return rows[:, high], filtered.most_likely_state[fold.test_idx], params, rarest


def _m1_fold(
    states: np.ndarray, spec: ConditionerSpec, fold: WalkForwardFold,
) -> tuple[np.ndarray, np.ndarray, MarkovChain]:
    """Eén (symbool, fold) voor M1: transitiematrix uit het trainvenster.

    De voorspelling voor bar `t` gebruikt de WAARGENOMEN toestand op `t-1`. Dat
    is de enige niet-triviale invulling: met de toestand op `t` zelf zou M1
    samenvallen met M0 en zou er niets te vergelijken zijn.
    """
    chain = MarkovChain.fit(
        states, n_states=spec.n_states, train_end=fold.train_end)
    high = spec.n_states - 1
    previous = states[fold.test_idx - 1].astype(np.int64)
    probabilities = chain.trans_mat[previous]
    return probabilities[:, high], np.argmax(probabilities, axis=1), chain


def _observed_states(buckets: pd.Series, n_states: int) -> np.ndarray:
    """M0-buckets als waargenomen toestanden voor M1.

    Bij `k = 3` zijn dat de drie buckets zelf. Bij `k = 2` wordt
    gebinariseerd naar {niet-HOOG, HOOG}: de toestand die de conditionering
    gebruikt, blijft dan dezelfde grootheid en alleen de resolutie verandert.
    """
    values = buckets.to_numpy(dtype="float64")
    if n_states == 3:
        return values
    return np.where(np.isnan(values), np.nan,
                    (values == float(VolBucket.HIGH)).astype(np.float64))


def build_conditioner(
    spec: ConditionerSpec,
    *,
    returns: pd.DataFrame,
    buckets: pd.DataFrame,
    cv: WalkForwardCV,
    adequacy: AdequacyConfig,
    m2_cfg: M2HmmConfig,
) -> ConditionerResult:
    """De exposure-factor van één trial over het volledige venster."""
    require(
        bool(returns.index.equals(buckets.index)),
        "Returns en M0-buckets staan niet op dezelfde tijdas. Een stilzwijgende "
        "reindex is de onopvallendste manier om een model een bar te laten "
        "zien die er niet was.",
        DataContractError, n_returns=len(returns), n_buckets=len(buckets),
    )
    require(
        tuple(returns.columns) == tuple(buckets.columns),
        "Returns en M0-buckets dragen niet dezelfde symbolen.",
        DataContractError, returns=list(returns.columns),
        buckets=list(buckets.columns),
    )
    mask, folds = oos_mask(len(returns), cv)
    n_train = int(folds[0].train_end)
    n_obs_per_fold = [f.train_end for f in folds]

    values = pd.DataFrame(
        1.0, index=returns.index, columns=returns.columns, dtype="float64")
    high_probability = pd.DataFrame(
        np.nan, index=returns.index, columns=returns.columns, dtype="float64")
    states = pd.DataFrame(
        np.nan, index=returns.index, columns=returns.columns, dtype="float64")
    fits: list[Mapping[str, Any]] = []
    rarest = np.inf
    converged = 0

    for symbol in returns.columns:
        observations = returns[symbol].to_numpy(dtype="float64")[:, None]
        observed = _observed_states(buckets[symbol], spec.n_states)
        for fold in folds:
            if spec.family == "m2":
                probability, argmax, params, fold_rarest = _m2_fold(
                    observations, spec, fold, symbol=symbol,
                    adequacy=adequacy, m2_cfg=m2_cfg,
                    n_obs_per_fold=n_obs_per_fold)
                rarest = min(rarest, fold_rarest)
                converged += int(params.converged)
                variance = _state_variance(params)
                fits.append({"symbol": symbol, **params.as_record(),
                             "rarest_state_obs": fold_rarest,
                             "high_state": _high_state(params),
                             # Ligt deze verhouding rond 1, dan zijn twee
                             # toestanden hetzelfde regime met twee namen en
                             # is `k` te groot voor deze data. Het is een
                             # DIAGNOSE en geen poort: de poort is de
                             # bezetting, en die staat in de pre-registratie.
                             "state_variance_ratio": float(
                                 variance.min() / variance.max())})
            else:
                probability, argmax, chain = _m1_fold(observed, spec, fold)
                # M1 schat geen latente toestanden; de bezetting is de
                # WAARGENOMEN telling van de zeldzaamste toestand.
                counts = np.bincount(
                    observed[:fold.train_end][
                        np.isfinite(observed[:fold.train_end])
                    ].astype(np.int64), minlength=spec.n_states)
                rarest = min(rarest, float(counts.min()))
                converged += 1
                fits.append({"symbol": symbol, "fold_id": fold.fold_id,
                             **dict(chain.as_record()),
                             "rarest_state_obs": float(counts.min())})
            values.iloc[fold.test_idx, values.columns.get_loc(symbol)] = (
                1.0 - probability)
            high_probability.iloc[
                fold.test_idx,
                high_probability.columns.get_loc(symbol)] = probability
            states.iloc[fold.test_idx, states.columns.get_loc(symbol)] = argmax

    scored_states = {
        symbol: states[symbol].to_numpy()[mask] for symbol in returns.columns}
    return ConditionerResult(
        label=spec.label, values=values, mask=mask,
        high_probability=high_probability,
        n_transitions={
            s: int(np.count_nonzero(v[1:] != v[:-1]))
            for s, v in scored_states.items()},
        mean_duration_bars={
            s: _durations(v) for s, v in scored_states.items()},
        mean_absolute_change={
            s: float(np.abs(np.diff(
                values[s].to_numpy()[mask])).mean()) if mask.sum() > 1 else 0.0
            for s in returns.columns},
        rarest_state_obs_per_fold=float(rarest),
        n_train_obs_per_fold=n_train,
        convergence_ratio=converged / len(fits) if fits else 0.0,
        fits=tuple(fits),
    )
