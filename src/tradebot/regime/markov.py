"""M1 Markov Chain en M2 Filtered HMM — deliverable 15. Filtered only.

DE REGEL (audit 10.2)
======================
    Smoothed probabilities `P(S_t | F_T)` gebruiken de VOLLEDIGE dataset.
    GEBRUIK IN BACKTESTS IS STRENG VERBODEN.
    Uitsluitend filtered probabilities `P(S_t | F_t)` uit het
    forward-algoritme zijn toegestaan.

Dit module maakt die regel technisch afdwingbaar in plaats van iets waar je aan
moet denken. De twee grootheden hebben hier VERSCHILLENDE TYPES:
:class:`FilteredProbabilities` gaat naar de backtest, :class:`SmoothedProbabilities`
draagt geen array-interface, geen `.values`, geen `__array__`, en kan dus niet
per ongeluk in numerieke code belanden. Wie hem er toch in wil, moet
`as_diagnostic_array()` aanroepen -- een naam die je niet per ongeluk typt en die
een grep onmiddellijk vindt.

DE VAL DIE DIT NODIG MAAKT, GEMETEN
====================================
`hmmlearn.predict_proba` en `score_samples` geven de SMOOTHED posterior. Dat is
de voor de hand liggende API, hij heet niet "smoothed", en niets waarschuwt.

Gemeten op hmmlearn 0.3.3, twee overlappende toestanden, 600 bars:

    max |P(S_t | F_600) - P(S_t | F_400)| over t < 400   =  0,0033

De posterior op bar 399 VERANDERT wanneer je er 200 latere bars aan toevoegt.
Dat is per definitie informatie uit de toekomst.

En let op waarom dit gevaarlijker is dan het lijkt: op GOED GESCHEIDEN toestanden
was dezelfde meting exact 0,000000, omdat de posteriors daar tegen 0 en 1 aan
zitten en er niets te smoothen valt. Wie de API op schone synthetische data
uitprobeert, concludeert dus dat `predict_proba` veilig is -- en gebruikt hem
vervolgens op echte data, waar de toestanden wel overlappen. Een val die alleen
zichtbaar is op moeilijke data is erger dan een val die altijd zichtbaar is.

Die 0,0033 meet hoe ver de smoothed posterior VERSCHUIFT als je er data achter
plakt. De omvang van het VOORDEEL zelf is een andere en veel grotere grootheid:
met dezelfde parameters op dezelfde 600 bars is

    max |P(S_t | F_T) - P(S_t | F_t)|   =  0,52

Meer dan een halve eenheid kansmassa. Op zo'n verschil kantelt een
regime-oordeel; dit is geen afronding maar een andere beslissing.
:meth:`SmoothedProbabilities.max_absolute_difference` berekent hem, zodat het
M0-vs-HMM-rapport de vermeden voorsprong als GEMETEN getal kan noemen in plaats
van als voorschrift.

Daarom wordt het forward-algoritme hier ZELF geïmplementeerd. Het is twintig
regels; het alternatief is vertrouwen op een bibliotheek waarvan de meest
voor de hand liggende methode het verboden getal teruggeeft.

DE PARAMETERS ZIJN OUT-OF-SAMPLE, DE FILTER LOOPT DOOR
=======================================================
Zelfde constructie als bij `volatility/garch.py`: :func:`fit_hmm` schat
uitsluitend op het trainvenster, en :func:`forward_filter` laat de recursie
daarna doorlopen over de WAARGENOMEN testbars met bevroren parameters. Dat is
geen lek maar precies wat het model in productie zou doen.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md 10.1, 10.2, 19 (L3), 24; Hamilton (1989);
Rabiner (1989); fase-opdracht stap 9.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..schemas.config import AdequacyConfig
from ..utils.failfast import (
    CausalityViolationError,
    DataContractError,
    require,
    require_dependency,
)
from ..validation.data_adequacy import assess_hmm, require_adequacy

__all__ = [
    "DIAGNOSTICS_ONLY",
    "DiagnosticsToken",
    "FilteredProbabilities",
    "HmmParameters",
    "HmmSpec",
    "MarkovChain",
    "SmoothedProbabilities",
    "filtered_occupancy",
    "fit_hmm",
    "forward_filter",
    "require_filtered",
    "smoothed_probabilities",
]


@dataclass(frozen=True)
class DiagnosticsToken:
    """Het bewijs dat een aanroeper WEET dat hij de verboden grootheid vraagt.

    Er is geen default-instantie in een functiehandtekening. Wie smoothed
    probabilities wil, moet dit object construeren en een reden meegeven, en die
    reden verschijnt in het object dat eruit komt. Dat maakt het onmogelijk om
    de smoothed variant per ongeluk te gebruiken en triviaal om in een review te
    zien dat iemand het bewust deed.
    """

    reason: str

    def __post_init__(self) -> None:
        require(
            len(self.reason.strip()) >= 20,
            "Een DiagnosticsToken zonder inhoudelijke reden. Smoothed "
            "probabilities zijn in backtests STRENG VERBODEN (audit §10.2); "
            "wie ze voor diagnostiek gebruikt, schrijft op waarvoor.",
            CausalityViolationError,
            reason=self.reason,
        )


#: De enige reden die dit module zelf gebruikt: het rapport wil laten zien
#: HOEVEEL de smoothed variant van de filtered afwijkt, want dat verschil is de
#: omvang van het lek dat de regel voorkomt.
DIAGNOSTICS_ONLY = DiagnosticsToken(
    reason="rapportage van het verschil tussen filtered en smoothed, om de "
           "omvang van het vermeden lek te kwantificeren")


@dataclass(frozen=True)
class HmmSpec:
    """De M2-specificatie. Onderdeel van de H2-parameterruimte, telt mee in M."""

    n_states: int
    covariance_type: Literal["diag", "full"] = "diag"
    distribution: Literal["gaussian", "student_t"] = "gaussian"
    n_iter: int = 200

    def __post_init__(self) -> None:
        require(
            self.n_states >= 2,
            "Een HMM met minder dan twee toestanden is geen regimemodel.",
            DataContractError, n_states=self.n_states,
        )
        require(
            self.distribution == "gaussian",
            "Student-t emissies vereisen een eigen EM-stap; hmmlearn levert "
            "die niet en een Gaussische fit met een Student-t filter is een "
            "ANDER model dan beide. Er wordt niet stilzwijgend teruggevallen "
            "op Gaussisch: die variant is een aparte trial en hoort in een "
            "eigen pre-registratie, met een eigen bijdrage aan M.",
            DataContractError, distribution=self.distribution,
        )

    @property
    def label(self) -> str:
        return f"hmm{self.n_states}-{self.covariance_type}-{self.distribution}"


@dataclass(frozen=True)
class HmmParameters:
    """Bevroren parameters uit één trainvenster. Alles wat de filter nodig heeft."""

    spec: HmmSpec
    symbol: str
    fold_id: int
    start_prob: np.ndarray      # (k,)
    trans_mat: np.ndarray       # (k, k)
    means: np.ndarray           # (k, d)
    covars: np.ndarray          # (k, d) voor diag, (k, d, d) voor full
    converged: bool
    n_train_obs: int
    loglikelihood: float

    def __post_init__(self) -> None:
        k = self.spec.n_states
        require(
            self.trans_mat.shape == (k, k),
            "Transitiematrix met de verkeerde vorm.",
            DataContractError, shape=self.trans_mat.shape, n_states=k,
        )
        require(
            bool(np.allclose(self.trans_mat.sum(axis=1), 1.0)),
            "De rijen van de transitiematrix sommeren niet naar 1.",
            DataContractError, row_sums=self.trans_mat.sum(axis=1).tolist(),
        )
        require(
            bool(np.isclose(self.start_prob.sum(), 1.0)),
            "De startverdeling sommeert niet naar 1.",
            DataContractError, total=float(self.start_prob.sum()),
        )

    @property
    def stationary_distribution(self) -> np.ndarray:
        """De linker-eigenvector bij eigenwaarde 1. Voedt de adequaatheidspoort.

        Dit is de verwachte bezetting per toestand op lange termijn, en dus de
        beste schatting vooraf van hoeveel observaties de ZELDZAAMSTE toestand
        per fold krijgt. `assess_hmm` gebruikt hem in plaats van de uniforme
        aanname, die de zeldzaamste toestand stelselmatig overschat.
        """
        values, vectors = np.linalg.eig(self.trans_mat.T)
        index = int(np.argmin(np.abs(values - 1.0)))
        vector = np.real(vectors[:, index])
        total = vector.sum()
        require(
            abs(total) > 1e-12,
            "De transitiematrix heeft geen bruikbare stationaire verdeling.",
            DataContractError, eigenvalues=np.real(values).tolist(),
        )
        return np.asarray(vector / total, dtype=np.float64)

    def as_record(self) -> dict[str, Any]:
        return {
            "spec": self.spec.label, "symbol": self.symbol,
            "fold_id": self.fold_id, "converged": self.converged,
            "n_train_obs": self.n_train_obs,
            "loglikelihood": self.loglikelihood,
            "stationary_distribution": self.stationary_distribution.tolist(),
            "persistence": np.diag(self.trans_mat).tolist(),
        }


@dataclass(frozen=True)
class FilteredProbabilities:
    """``P(S_t | F_t)`` — het ENIGE dat een backtest mag zien.

    Rij `t` gebruikt uitsluitend observaties tot en met `t`. Dat is het contract,
    en `tests/lookahead/test_filtered_only_enforcement.py` meet het door de reeks
    af te kappen en te eisen dat de rijen tot aan de snede identiek blijven.
    """

    values: np.ndarray          # (n, k)
    symbol: str
    spec_label: str
    loglikelihood: float
    #: `True` zodra `require_filtered` hem heeft goedgekeurd; puur informatief.
    causal: bool = field(default=True, init=False)

    @property
    def most_likely_state(self) -> np.ndarray:
        """Argmax per bar. NIET Viterbi: Viterbi is een pad over de HELE reeks
        en dus smoothed. De argmax van de filtered posterior is causaal."""
        return np.asarray(np.argmax(self.values, axis=1), dtype=np.int64)


@dataclass(frozen=True)
class SmoothedProbabilities:
    """``P(S_t | F_T)`` — DIAGNOSTIEK. Onbereikbaar voor de backtest.

    Deze klasse draagt met opzet GEEN `values`, geen `__array__`, geen
    `__len__` en geen `__iter__`. Een poging om hem als array te gebruiken
    crasht met een `TypeError` in plaats van stilzwijgend te werken, en de
    enige uitweg heet `as_diagnostic_array()` -- een naam die niemand per
    ongeluk typt en die elke grep en elke review meteen laat zien.
    """

    _values: np.ndarray
    symbol: str
    spec_label: str
    token: DiagnosticsToken

    def as_diagnostic_array(self) -> np.ndarray:
        """De smoothed posterior, expliciet opgevraagd voor diagnostiek."""
        return self._values.copy()

    def max_absolute_difference(self, filtered: FilteredProbabilities) -> float:
        """Hoe ver wijkt smoothed af van filtered? De omvang van het lek.

        Dit getal hoort in `reports/M0_VS_HMM_BENCHMARK.md`. Het maakt van de
        regel uit §10.2 een gemeten grootheid in plaats van een voorschrift.
        """
        return float(np.max(np.abs(self._values - filtered.values)))


def require_filtered(probabilities: Any, *, context: str) -> FilteredProbabilities:
    """De poort. Elke backtest- en conditioneringsingang loopt hierlangs.

    Een `SmoothedProbabilities` crasht met `CausalityViolationError`; een kale
    ndarray ook, want daaraan is niet te zien waar hij vandaan komt en de
    hele constructie hangt op dat onderscheid.
    """
    if isinstance(probabilities, SmoothedProbabilities):
        raise CausalityViolationError(
            f"{context}: er werd geprobeerd SMOOTHED probabilities "
            f"P(S_t | F_T) een backtest-pad in te voeren. Die gebruiken de "
            f"volledige dataset -- inclusief bars die op t nog niet bestonden "
            f"-- en zijn STRENG VERBODEN (audit §10.2). De smoothed variant "
            f"werd opgevraagd met de reden: '{probabilities.token.reason}'. "
            f"Gebruik `forward_filter()`."
        )
    require(
        isinstance(probabilities, FilteredProbabilities),
        f"{context}: verwacht `FilteredProbabilities` en kreeg "
        f"{type(probabilities).__name__}. Een kale array wordt geweigerd omdat "
        "er niet aan te zien is of hij causaal is berekend; dat onderscheid is "
        "de hele reden dat deze twee grootheden verschillende types hebben.",
        CausalityViolationError,
        got=type(probabilities).__name__, context=context,
    )
    return probabilities


# =========================================================================== #
# Emissies en het forward-algoritme
# =========================================================================== #
def _gaussian_log_density(
    observations: np.ndarray, means: np.ndarray, covars: np.ndarray,
    covariance_type: str,
) -> np.ndarray:
    """``log p(y_t | S_t = i)`` voor elke bar en elke toestand. Vorm (n, k)."""
    n, d = observations.shape
    k = means.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for i in range(k):
        delta = observations - means[i]
        if covariance_type == "diag":
            variance = np.asarray(covars[i], dtype=np.float64)
            require(
                bool(np.all(variance > 0.0)),
                "Een toestand met niet-positieve variantie. De emissiedichtheid "
                "is daar niet gedefinieerd; dit is een gedegenereerde fit en "
                "geen getal om doorheen te rekenen.",
                DataContractError, state=i, variance=variance.tolist(),
            )
            quad = np.sum(delta**2 / variance, axis=1)
            logdet = float(np.sum(np.log(variance)))
        else:
            matrix = np.asarray(covars[i], dtype=np.float64)
            sign, logdet = np.linalg.slogdet(matrix)
            require(
                sign > 0,
                "Een toestand met een niet-positief-definiete covariantie.",
                DataContractError, state=i, sign=float(sign),
            )
            solved = np.linalg.solve(matrix, delta.T).T
            quad = np.einsum("ij,ij->i", delta, solved)
        out[:, i] = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)
    return out


def forward_filter(
    observations: np.ndarray, params: HmmParameters,
) -> FilteredProbabilities:
    """Het forward-algoritme met schaling. De ENIGE causale route.

    ``alpha_t(i) = P(S_t = i | y_0..y_t)``. Elke stap gebruikt uitsluitend het
    verleden; er is geen backward-pass en er is dus niets dat de toekomst kan
    binnenlaten. De schaling per bar voorkomt underflow en levert en passant de
    log-likelihood op als som van de logs van de schaalfactoren.
    """
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[:, None]
    require(
        bool(np.all(np.isfinite(observations))),
        "Forward-filter op observaties met niet-eindige waarden. Er wordt niet "
        "geimputeerd; een gat is een databevinding.",
        DataContractError, symbol=params.symbol,
        n_non_finite=int(np.count_nonzero(~np.isfinite(observations))),
    )
    n = observations.shape[0]
    k = params.spec.n_states
    log_b = _gaussian_log_density(
        observations, params.means, params.covars, params.spec.covariance_type)
    # Stabiliseren per bar: een constante per rij valt weg in de normalisatie.
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))

    alpha = np.empty((n, k), dtype=np.float64)
    log_scale_total = 0.0
    current = params.start_prob * b[0]
    for t in range(n):
        if t > 0:
            current = (alpha[t - 1] @ params.trans_mat) * b[t]
        total = float(current.sum())
        require(
            total > 0.0,
            "De forward-recursie liep vast: alle toestanden hebben kans nul op "
            "deze bar. Dat betekent dat de observatie onder elke toestand "
            "onmogelijk is -- een gedegenereerde fit, geen numeriek detail.",
            DataContractError, symbol=params.symbol, bar=t,
        )
        alpha[t] = current / total
        log_scale_total += float(np.log(total)) + float(log_b[t].max())
    return FilteredProbabilities(
        values=alpha, symbol=params.symbol, spec_label=params.spec.label,
        loglikelihood=log_scale_total,
    )


def smoothed_probabilities(
    observations: np.ndarray, params: HmmParameters, token: DiagnosticsToken,
) -> SmoothedProbabilities:
    """``P(S_t | F_T)`` via forward-backward. **Nooit in een backtest.**

    Vereist een expliciete :class:`DiagnosticsToken` met een reden. Het
    resultaat is een `SmoothedProbabilities` en die kan niet als array worden
    gebruikt; zie de klassedocstring.
    """
    require(
        isinstance(token, DiagnosticsToken),
        "Smoothed probabilities zonder DiagnosticsToken. Zij gebruiken de "
        "volledige dataset en zijn in backtests STRENG VERBODEN (§10.2).",
        CausalityViolationError, got=type(token).__name__,
    )
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[:, None]
    n = observations.shape[0]
    k = params.spec.n_states
    log_b = _gaussian_log_density(
        observations, params.means, params.covars, params.spec.covariance_type)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))

    alpha = np.empty((n, k))
    scale = np.empty(n)
    current = params.start_prob * b[0]
    for t in range(n):
        if t > 0:
            current = (alpha[t - 1] @ params.trans_mat) * b[t]
        scale[t] = current.sum()
        alpha[t] = current / scale[t]

    beta = np.ones((n, k), dtype=np.float64)
    for t in range(n - 2, -1, -1):
        beta[t] = params.trans_mat @ (b[t + 1] * beta[t + 1]) / scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return SmoothedProbabilities(
        _values=gamma, symbol=params.symbol, spec_label=params.spec.label,
        token=token,
    )


def filtered_occupancy(filtered: FilteredProbabilities) -> np.ndarray:
    """Verwachte bezetting per toestand, uit de FILTERED posterior.

    De adequaatheidspoort vraagt hierom: het gemiddelde van de filtered
    posterior is het verwachte aantal bars per toestand, en dat is wat bepaalt
    of een toestand genoeg observaties heeft om iets over te zeggen.
    """
    return np.asarray(filtered.values.mean(axis=0), dtype=np.float64)


# =========================================================================== #
# Fit
# =========================================================================== #
def fit_hmm(
    observations: np.ndarray,
    spec: HmmSpec,
    cfg: AdequacyConfig,
    *,
    symbol: str,
    fold_id: int,
    train_end: int,
    n_obs_per_fold: Sequence[int],
    seed: int,
) -> HmmParameters:
    """Schat op ``observations[:train_end]``. Nooit op de volledige reeks.

    De EM-stap komt uit `hmmlearn`; de FILTER komt uit dit module. Die scheiding
    is bewust: `hmmlearn.predict_proba` geeft de smoothed posterior en is
    daarmee onbruikbaar voor een backtest -- zie de moduledocstring voor de
    meting.
    """
    hmm_mod = require_dependency(
        "hmmlearn.hmm",
        needed_for="het M2 Filtered HMM (H2)",
        install_hint="pip install hmmlearn",
    )
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[:, None]
    require(
        0 < train_end < observations.shape[0],
        "HMM-fit op de volledige sample. `train_end` moet strikt kleiner zijn "
        "dan de reekslengte; anders is er geen out-of-sample venster.",
        DataContractError, symbol=symbol, train_end=train_end,
        n_total=int(observations.shape[0]),
    )
    train = observations[:train_end]
    require(
        bool(np.all(np.isfinite(train))),
        "HMM-fit op een venster met niet-eindige observaties.",
        DataContractError, symbol=symbol, fold_id=fold_id,
    )
    # De poort staat VOOR de fit. Zij toetst de zeldzaamste toestand in de
    # kleinste fold; onder de uniforme aanname, die optimistisch is.
    require_adequacy(assess_hmm(n_obs_per_fold, spec.n_states, cfg))

    model = hmm_mod.GaussianHMM(
        n_components=spec.n_states, covariance_type=spec.covariance_type,
        n_iter=spec.n_iter, random_state=seed,
    )
    model.fit(train)
    return HmmParameters(
        spec=spec, symbol=symbol, fold_id=fold_id,
        start_prob=np.asarray(model.startprob_, dtype=np.float64),
        trans_mat=np.asarray(model.transmat_, dtype=np.float64),
        means=np.asarray(model.means_, dtype=np.float64),
        covars=np.asarray(model.covars_ if spec.covariance_type == "full"
                          else model.covars_.reshape(spec.n_states, -1),
                          dtype=np.float64),
        converged=bool(model.monitor_.converged),
        n_train_obs=int(train_end),
        loglikelihood=float(model.score(train)),
    )


# =========================================================================== #
# M1 — Markov Chain op waargenomen toestanden
# =========================================================================== #
@dataclass(frozen=True)
class MarkovChain:
    """M1: een Markov-keten op WAARGENOMEN toestanden, bijvoorbeeld M0-buckets.

    Geen latente toestanden en dus geen forward-algoritme nodig: de toestand op
    `t` is gewoon bekend. Wat wel wordt geschat is de transitiematrix, en dat
    gebeurt uitsluitend op het trainvenster.

    M1 staat tussen M0 en M2 in en beantwoordt een aparte vraag: voegt de
    OVERGANGSSTRUCTUUR iets toe, los van de latente-toestandsmachinerie van het
    HMM? Verslaat M2 wel M0 maar M1 niet, dan zit de winst in de latente
    toestanden; verslaan beide M0 ongeveer even veel, dan zit zij in de
    persistentie en is het HMM overbodige complexiteit.
    """

    trans_mat: np.ndarray
    n_states: int
    n_train_obs: int
    #: Overgangen die in het trainvenster NIET zijn waargenomen. Hun geschatte
    #: kans is nul, en dat is een meting en geen modelkeuze: er is geen
    #: smoothing toegepast die het zou verhullen.
    unobserved_transitions: int

    @classmethod
    def fit(
        cls, states: np.ndarray, *, n_states: int, train_end: int,
    ) -> MarkovChain:
        states = np.asarray(states)
        require(
            0 < train_end < states.size,
            "M1-fit op de volledige sample.",
            DataContractError, train_end=train_end, n_total=int(states.size),
        )
        train = states[:train_end]
        usable = train[np.isfinite(train.astype(np.float64))].astype(np.int64)
        require(
            usable.size >= 2,
            "M1 heeft minstens twee opeenvolgende gedefinieerde toestanden nodig.",
            DataContractError, n_usable=int(usable.size),
        )
        counts = np.zeros((n_states, n_states), dtype=np.float64)
        for a, b in itertools.pairwise(usable):
            counts[a, b] += 1.0
        row_totals = counts.sum(axis=1, keepdims=True)
        require(
            bool(np.all(row_totals > 0.0)),
            "Een toestand komt in het trainvenster voor zonder ENKELE "
            "waargenomen overgang eruit. De rij van de transitiematrix is dan "
            "niet gedefinieerd; er wordt niet gesmooth om dat te verbergen, "
            "want het is een adequaatheidsbevinding.",
            DataContractError,
            empty_rows=np.flatnonzero(row_totals.ravel() == 0.0).tolist(),
        )
        return cls(
            trans_mat=counts / row_totals, n_states=n_states,
            n_train_obs=int(usable.size),
            unobserved_transitions=int(np.count_nonzero(counts == 0.0)),
        )

    def predict_next(self, current_state: int) -> np.ndarray:
        """``P(S_{t+1} | S_t)``. Causaal per constructie."""
        require(
            0 <= current_state < self.n_states,
            "Onbekende toestand aan M1 aangeboden.",
            DataContractError, state=current_state, n_states=self.n_states,
        )
        return np.asarray(self.trans_mat[current_state], dtype=np.float64)

    def as_record(self) -> Mapping[str, Any]:
        return {
            "n_states": self.n_states, "n_train_obs": self.n_train_obs,
            "persistence": np.diag(self.trans_mat).tolist(),
            "unobserved_transitions": self.unobserved_transitions,
        }
