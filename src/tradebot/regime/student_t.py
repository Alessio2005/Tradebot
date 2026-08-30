# src/tradebot/regime/student_t.py
"""Student-t emissies voor het M2 HMM. Deliverable 15, twee van de zes H2-trials.

WAAROM DIT BESTAND BESTAAT
===========================
`hmmlearn` heeft geen Student-t HMM. De pre-registratie H2
(`3d3af28730a6c7f9da48d13139522a05`, bevroren 2026-08-25) heeft er wel twee:
`hmm2-diag-student_t` en `hmm3-diag-student_t`. Die zes trials zijn bij het
BEVRIEZEN in `M` geboekt -- terecht, want wie een parameterruimte vastlegt heeft
die kansen genomen. Wat er niet mag gebeuren, is dat `M` twee trials draagt
waarvoor nooit iets is gemeten.

`markov.py` weigerde `distribution="student_t"` met de motivering dat de variant
"in een eigen pre-registratie hoort, met een eigen bijdrage aan M". Die
motivering was op deze pre-registratie feitelijk onjuist: hij STAAT er al in en
is al geboekt. De blokkade is daarom vervangen door deze EM. Wat er wél in stond
en blijft staan: een Gaussische fit met een t-filter (of andersom) is een DERDE
model dat in geen enkele pre-registratie voorkomt, en dat blijft verboden.

DE SCHAALMENGSEL-REPRESENTATIE, EN WAAROM DE EM DAN GEWOON WERKT
=================================================================
Een multivariate t is een normale met een willekeurige schaal::

    u_t | S_t = i  ~  Gamma(nu_i / 2, nu_i / 2)
    y_t | u_t, S_t = i  ~  N(mu_i, Sigma_i / u_t)

Integreer `u` uit en je hebt `t_nu(mu_i, Sigma_i)`. Dat is geen truc maar de
definitie, en zij levert de EM gratis: naast de toestandskansen `gamma_t(i)` is
er per bar een verwachte schaal::

    E[u_t | y_t, S_t = i]  =  (nu_i + d) / (nu_i + delta_ti)

met `delta_ti` de Mahalanobis-afstand. Een uitschieter krijgt daarmee een KLEIN
gewicht in de M-stap -- dat is precies wat "zware staarten" hier betekent en de
reden dat deze variant op crypto-returns een andere fit kan opleveren dan de
Gaussische. `nu` zelf komt uit de ECME-stap van Liu & Rubin (1995): één
1-D-nulpunt per toestand, begrensd door `conf/model/regime.yaml`.

DE WARME START, EN WAAROM DIE GEEN DETAIL IS
=============================================
De EM start van de UITKOMST van de Gaussische fit op hetzelfde venster, en die
parameters komen als ARGUMENT binnen -- dit module fit zelf geen Gaussische HMM
en kent `hmmlearn` niet. Zonder die gedeelde start zou het verschil tussen
`hmm2-diag-gaussian` en `hmm2-diag-student_t` deels het STARTPUNT zijn: beide
EM's zijn lokaal en landen vanuit een ander punt in een ander optimum. Met de
warme start is het enige verschil de emissieverdeling, en dat is de vergelijking
die de pre-registratie bedoelt.

De keerzijde staat in het rapport en niet in een voetnoot: de likelihood van een
warm gestarte EM is niet gegarandeerd hoger dan die van zijn startpunt, want het
startpunt is een optimum van een ANDER model. `StudentTFit.gaussian_loglikelihood`
draagt daarom het vergelijkingsgetal mee, zodat het rapport kan tonen of de extra
parameter iets deed in plaats van dat aan te nemen.

WAT HIER NIET STAAT
====================
De FILTER. Die staat in `markov.py::forward_filter` en is voor beide
emissieverdelingen dezelfde recursie -- alleen de dichtheid verschilt. Dat is
opzettelijk: er is precies één causale route naar een backtest, en die route mag
niet per verdeling een eigen implementatie krijgen.

Ref: Liu & Rubin (1995), *ML estimation of the t distribution using EM and its
extensions, ECM and ECME*; Peel & McLachlan (2000); fase-opdracht stap 9,
deliverable 15.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, special

from ..schemas.config import M2HmmConfig
from ..utils.failfast import DataContractError, require

__all__ = ["StudentTFit", "fit_student_t_hmm", "student_t_log_density"]

#: Minimaal aantal trainobservaties per toestand. Onder deze grens schat de EM
#: een gemiddelde, een schaal, een vrijheidsgraad en k-1 overgangskansen op een
#: handvol punten; dat is geen toestand maar een uitschieterdetector. De
#: ADEQUAATHEIDSpoort in `conf/model/adequacy.yaml` is strenger (100) en gaat
#: voor; deze grens is de ondergrens waaronder de EM zelf niet gedefinieerd is.
_MIN_OBS_PER_STATE = 10


def student_t_log_density(
    observations: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    dof: np.ndarray,
    covariance_type: str,
) -> np.ndarray:
    """``log p(y_t | S_t = i)`` onder een multivariate t. Vorm ``(n, k)``.

    `covars` is de SCHAALmatrix, niet de variantie: de variantie is
    ``Sigma * nu / (nu - 2)`` en bestaat alleen voor ``nu > 2``. Dat onderscheid
    is de meest voorkomende fout in een t-implementatie en is de reden dat
    `test_large_dof_converges_to_the_gaussian` bestaat -- die test faalt zodra
    de twee door elkaar lopen.
    """
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[:, None]
    n, d = observations.shape
    k = means.shape[0]
    dof = np.asarray(dof, dtype=np.float64)
    require(
        dof.shape == (k,),
        "Er hoort per toestand precies een vrijheidsgraad te zijn.",
        DataContractError, n_states=k, dof_shape=dof.shape,
    )
    require(
        bool(np.all(dof > 2.0)),
        "Een vrijheidsgraad van 2 of minder. De variantie van de emissie "
        "bestaat daar niet, en een toestand zonder tweede moment maakt de "
        "vergelijking tussen toestanden betekenisloos.",
        DataContractError, dof=dof.tolist(),
    )
    out = np.empty((n, k), dtype=np.float64)
    for i in range(k):
        delta = observations - means[i]
        if covariance_type == "diag":
            scale = np.asarray(covars[i], dtype=np.float64)
            require(
                bool(np.all(scale > 0.0)),
                "Een toestand met een niet-positieve schaal. De dichtheid is "
                "daar niet gedefinieerd; dit is een gedegenereerde fit en geen "
                "getal om doorheen te rekenen.",
                DataContractError, state=i, scale=scale.tolist(),
            )
            quad = np.sum(delta**2 / scale, axis=1)
            logdet = float(np.sum(np.log(scale)))
        else:
            matrix = np.asarray(covars[i], dtype=np.float64)
            sign, logdet = np.linalg.slogdet(matrix)
            require(
                sign > 0,
                "Een toestand met een niet-positief-definiete schaalmatrix.",
                DataContractError, state=i, sign=float(sign),
            )
            solved = np.linalg.solve(matrix, delta.T).T
            quad = np.einsum("ij,ij->i", delta, solved)
        nu = float(dof[i])
        out[:, i] = (
            special.gammaln(0.5 * (nu + d))
            - special.gammaln(0.5 * nu)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
            - 0.5 * (nu + d) * np.log1p(quad / nu)
        )
    return out


@dataclass(frozen=True)
class StudentTFit:
    """Wat één Student-t EM opleverde, inclusief wat er misging."""

    n_states: int
    start_prob: np.ndarray      # (k,)
    trans_mat: np.ndarray       # (k, k)
    means: np.ndarray           # (k, d)
    covars: np.ndarray          # (k, d) diag, (k, d, d) full -- SCHAAL
    dof: np.ndarray             # (k,)
    converged: bool
    n_iter_used: int
    loglikelihood: float
    #: De likelihood van de Gaussische fit waaruit is warmgestart, op dezelfde
    #: bars. Zonder dit getal is niet te zeggen of de staartparameter iets deed.
    gaussian_loglikelihood: float
    #: `True` zodra een toestand tegen `dof_min` of `dof_max` aan ligt. Dat is
    #: een bevinding die wordt gerapporteerd, geen waarde die wordt weggerond.
    dof_at_bound: bool
    #: `True` wanneer de EM is gestopt omdat een toestand instortte. De
    #: teruggegeven parameters zijn dan die van de LAATSTE geldige iteratie en
    #: `converged` is `False`. Zelfde contract als `fit_garch_window`: een
    #: numerieke mislukking is een RESULTAAT dat wordt geteld, geen fout die
    #: de campagne afbreekt en geen waarde die stilzwijgend wordt gerepareerd.
    degenerate: bool
    loglikelihood_history: tuple[float, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "converged": self.converged,
            "n_iter_used": self.n_iter_used,
            "loglikelihood": self.loglikelihood,
            "gaussian_loglikelihood": self.gaussian_loglikelihood,
            "loglikelihood_gain_over_gaussian": (
                self.loglikelihood - self.gaussian_loglikelihood),
            "dof": self.dof.tolist(),
            "dof_at_bound": self.dof_at_bound,
            "degenerate": self.degenerate,
        }


def _forward_backward(
    log_b: np.ndarray, start_prob: np.ndarray, trans_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """De E-stap: ``gamma_t(i)``, ``sum_t xi_t(i,j)`` en de log-likelihood.

    Dit is de SMOOTHED posterior en dat is hier legitiem: hij wordt uitsluitend
    op het TRAINVENSTER gebruikt om parameters te schatten, precies zoals
    `hmmlearn` dat voor de Gaussische variant doet. De weg naar een backtest
    loopt via `markov.py::forward_filter`, die geen backward-pass kent.
    """
    n, k = log_b.shape
    shift = log_b.max(axis=1, keepdims=True)
    b = np.exp(log_b - shift)

    alpha = np.empty((n, k))
    scale = np.empty(n)
    current = start_prob * b[0]
    for t in range(n):
        if t > 0:
            current = (alpha[t - 1] @ trans_mat) * b[t]
        total = float(current.sum())
        require(
            total > 0.0,
            "De forward-recursie liep vast: alle toestanden hebben kans nul op "
            "deze bar. Dat is een gedegenereerde fit, geen numeriek detail.",
            DataContractError, bar=t,
        )
        scale[t] = total
        alpha[t] = current / total

    beta = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        beta[t] = trans_mat @ (b[t + 1] * beta[t + 1]) / scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi_sum = np.zeros((k, k))
    for t in range(n - 1):
        xi = (alpha[t][:, None] * trans_mat
              * (b[t + 1] * beta[t + 1])[None, :]) / scale[t + 1]
        xi_sum += xi
    loglik = float(np.sum(np.log(scale)) + float(shift.sum()))
    return gamma, xi_sum, loglik


def _solve_dof(
    weight: np.ndarray, log_u: np.ndarray, u: np.ndarray, d: int,
    nu_old: float, cfg: M2HmmConfig,
) -> float:
    """De ECME-stap voor `nu`: één 1-D-nulpunt, begrensd door `conf/`.

    De vergelijking (Liu & Rubin 1995, eq. 22) is monotoon dalend in `nu`, dus
    een tekenwisseling op het interval betekent precies één oplossing. Is er
    geen tekenwisseling, dan ligt het optimum OP een grens; dat wordt
    teruggegeven en als `dof_at_bound` gerapporteerd in plaats van weggerond.
    """
    total = float(weight.sum())
    constant = float(np.sum(weight * (log_u - u)) / total)
    constant += float(special.digamma(0.5 * (nu_old + d))
                      - np.log(0.5 * (nu_old + d)))

    def objective(nu: float) -> float:
        return float(-special.digamma(0.5 * nu) + np.log(0.5 * nu)
                     + 1.0 + constant)

    low, high = cfg.dof_min, cfg.dof_max
    f_low, f_high = objective(low), objective(high)
    if f_low * f_high > 0.0:
        return low if f_low < 0.0 else high
    return float(optimize.brentq(objective, low, high, xtol=1e-8))


def fit_student_t_hmm(
    observations: np.ndarray,
    *,
    start: Mapping[str, np.ndarray],
    gaussian_loglikelihood: float,
    cfg: M2HmmConfig,
) -> StudentTFit:
    """EM voor een HMM met Student-t emissies. Uitsluitend op het trainvenster.

    De aanroeper snijdt het trainvenster af en levert de warme start; dit module
    ziet niets anders. Dat is dezelfde arbeidsverdeling als bij `fit_hmm` en de
    reden dat hier geen `train_end` staat: een functie die zelf mag kiezen
    hoeveel zij ziet, is een functie die op een dag te veel ziet.

    `start` draagt `start_prob`, `trans_mat`, `means` en `covars` van de
    Gaussische fit op DEZELFDE bars; `gaussian_loglikelihood` is haar
    log-likelihood, zodat het rapport kan tonen wat de staartparameter opleverde.
    """
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[:, None]
    n, d = observations.shape
    n_states = int(np.asarray(start["means"]).shape[0])
    require(
        bool(np.all(np.isfinite(observations))),
        "Student-t EM op een venster met niet-eindige observaties. Er wordt "
        "niet geimputeerd; een gat is een databevinding.",
        DataContractError, n_obs=n,
        n_non_finite=int(np.count_nonzero(~np.isfinite(observations))),
    )
    require(
        n >= _MIN_OBS_PER_STATE * n_states,
        "Te weinig observaties voor deze EM. Er wordt niet 'zo goed als het "
        "gaat' gefit: onder deze grens is elke geschatte parameter ruis.",
        DataContractError, n_obs=n, n_states=n_states,
        required=_MIN_OBS_PER_STATE * n_states,
    )

    start_prob = np.array(start["start_prob"], dtype=np.float64)
    trans_mat = np.array(start["trans_mat"], dtype=np.float64)
    means = np.array(start["means"], dtype=np.float64)
    covars = np.array(start["covars"], dtype=np.float64)
    dof = np.full(n_states, float(cfg.dof_init))

    history: list[float] = []
    converged = False
    degenerate = False
    used = 0
    # De laatste parameters waarvan de E-stap GEZOND was: elke toestand droeg
    # daar nog minstens een observatie aan verantwoordelijkheid. Stort er later
    # een in, dan is dit het punt waarnaar wordt teruggerold -- de ingestorte
    # parameters zelf teruggeven zou een filter opleveren die op een schaal van
    # 1e-128 rekent en getallen produceert die niets meer meten.
    last_healthy = (start_prob.copy(), trans_mat.copy(), means.copy(),
                    covars.copy(), dof.copy())
    for iteration in range(1, cfg.n_iter + 1):
        used = iteration
        log_b = student_t_log_density(
            observations, means, covars, dof, cfg.covariance_type)
        gamma, xi_sum, loglik = _forward_backward(log_b, start_prob, trans_mat)

        # De verwachte schaal per bar en per toestand. Een uitschieter krijgt
        # hier een klein gewicht -- dat IS het staartgedrag.
        u = np.empty((n, n_states))
        for i in range(n_states):
            delta = observations - means[i]
            if cfg.covariance_type == "diag":
                quad = np.sum(delta**2 / covars[i], axis=1)
            else:
                quad = np.einsum(
                    "ij,ij->i", delta, np.linalg.solve(covars[i], delta.T).T)
            u[:, i] = (dof[i] + d) / (dof[i] + quad)

        # EEN TOESTAND DIE INSTORT IS EEN UITKOMST, GEEN FOUT
        # ----------------------------------------------------
        # De t-mengselverdeling heeft een bekende ontaarding: krijgt een
        # toestand nog maar een handvol punten die dicht op zijn gemiddelde
        # liggen, dan gaat `u` omhoog, de schaal naar nul en de likelihood naar
        # oneindig. Dat is geen optimum maar een singulariteit, en zij treedt op
        # wanneer `k` groter is dan het aantal regimes dat de data draagt.
        #
        # De EM stopt daar en geeft de parameters van de LAATSTE GELDIGE
        # iteratie terug, met `converged = False` en `degenerate = True`. Dat is
        # hetzelfde contract als `fit_garch_window`: de mislukking wordt geteld
        # in de convergentieratio en de campagne descopeert erop. De grens is
        # geen drempel maar het DOMEIN -- minder dan een observatie aan
        # verantwoordelijkheid, of een schaal die niet strikt positief is,
        # betekent dat de M-stap niet gedefinieerd is.
        responsibility = gamma.sum(axis=0)
        if float(responsibility.min()) < 1.0:
            degenerate = True
            break
        last_healthy = (start_prob.copy(), trans_mat.copy(), means.copy(),
                        covars.copy(), dof.copy())
        history.append(loglik)

        new_start = gamma[0] / gamma[0].sum()
        denom = gamma[:-1].sum(axis=0)
        if float(denom.min()) <= 0.0:
            degenerate = True
            break
        new_trans = xi_sum / denom[:, None]
        new_trans /= new_trans.sum(axis=1, keepdims=True)

        weight = gamma * u
        new_means = (weight.T @ observations) / weight.sum(axis=0)[:, None]
        new_covars = np.empty_like(covars)
        new_dof = np.empty_like(dof)
        for i in range(n_states):
            delta = observations - new_means[i]
            total = float(gamma[:, i].sum())
            if cfg.covariance_type == "diag":
                new_covars[i] = (weight[:, i] @ (delta**2)) / total
            else:
                new_covars[i] = (delta * weight[:, i][:, None]).T @ delta / total
            new_dof[i] = _solve_dof(
                gamma[:, i], np.log(u[:, i]), u[:, i], d, float(dof[i]), cfg)
        scale_diagonal = (new_covars if cfg.covariance_type == "diag"
                          else np.diagonal(new_covars, axis1=1, axis2=2))
        if not (np.all(np.isfinite(new_covars)) and np.all(scale_diagonal > 0.0)):
            degenerate = True
            break
        total_scale = scale_diagonal.sum(axis=1)
        if float(total_scale.min() / total_scale.max()) < cfg.min_scale_ratio:
            degenerate = True
            break
        start_prob, trans_mat = new_start, new_trans
        means, covars, dof = new_means, new_covars, new_dof

        if len(history) >= 2:
            delta_ll = history[-1] - history[-2]
            if abs(delta_ll) <= cfg.em_tolerance * max(abs(history[-2]), 1.0):
                converged = True
                break

    if degenerate:
        start_prob, trans_mat, means, covars, dof = last_healthy
    final_log_b = student_t_log_density(
        observations, means, covars, dof, cfg.covariance_type)
    _, _, final_loglik = _forward_backward(final_log_b, start_prob, trans_mat)
    history.append(final_loglik)
    at_bound = bool(np.any(np.isclose(dof, cfg.dof_min))
                    or np.any(np.isclose(dof, cfg.dof_max)))
    return StudentTFit(
        n_states=n_states, start_prob=start_prob, trans_mat=trans_mat,
        means=means, covars=covars, dof=dof, converged=converged,
        n_iter_used=used, loglikelihood=final_loglik,
        gaussian_loglikelihood=float(gaussian_loglikelihood),
        dof_at_bound=at_bound, degenerate=degenerate,
        loglikelihood_history=tuple(history),
    )
