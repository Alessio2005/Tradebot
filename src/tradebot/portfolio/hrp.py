# src/tradebot/portfolio/hrp.py
"""Hierarchical Risk Parity (HRP) — RESEARCH ONLY, technisch geblokkeerd.

Algoritme: López de Prado (2018) AFML §16.
  1. Covariantiematrix (Ledoit-Wolf shrinkage).
  2. Correlatie → afstandsmatrix D = sqrt(0.5 * (1 - rho)).
  3. Hiërarchische clustering (Ward linkage op D).
  4. Quasi-diagonalisatie (bladvolgorde uit de linkage).
  5. Recursieve bisectie: alloceer per cluster op inverse variantie.

===========================================================================
DE RESEARCH-GATE — Phase 6 deliverable 23, gebouwd in Phase 7/8 Stage C-1
===========================================================================
`fase_6_advanced_research.md` no-go 15 luidde: *"HRP is productie-toegankelijk
zonder bewijs."* Die conditie was **actief**. Dit bestand bevatte geen `raise`,
geen vlag en geen markering; `hrp_weights(returns)` gaf gewoon gewichten terug
die elke aanroeper in een boek kon zetten.

Deliverable 23 schrijft voor: *"HRP blijft bestaan maar is technisch geblokkeerd
voor productiegebruik totdat turnover-gecorrigeerde OOS Sharpe > Inverse
Volatility is aangetoond; **de gate is een crash, geen vlag**."*

WAAROM EEN TOKEN EN GEEN ONVOORWAARDELIJKE `raise`
==================================================
Een harde `raise` zou deliverable 24 (`reports/HRP_VS_INVERSE_VOL.md`)
onmogelijk maken: je kunt HRP niet met Inverse Volatility vergelijken als je hem
niet kunt draaien. *Research-gated* betekent daarom precies twee dingen:

* **onderzoek is mogelijk**, maar alleen met een expliciet
  `HrpResearchGate`-token dat een reden en een pre-registratie-ID draagt. Er is
  geen default-instantie in enige handtekening, dus per ongeluk lukt niet;
* **productie is onmogelijk**, want de uitkomst draagt permanent
  `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`. `registry/promotion.py` roept
  `reject_vectorized_evidence()` aan op elke stagewijziging, en die functie
  crasht op dat label — hij toetst de MARKERING, niet de engine, dus HRP-output
  wordt automatisch geweigerd zonder dat daar een tweede mechanisme voor nodig is.

Dat token-idioom is niet nieuw hier: `regime/markov.py::DiagnosticsToken` doet
hetzelfde voor smoothed probabilities.

WAT DE GATE ZOU OPENEN — EN WAAROM DAT ONWAARSCHIJNLIJK IS
==========================================================
`ADMISSION_CRITERIA` staat hieronder als code, niet als proza. Beide eisen
moeten gelden, en `fase_6` §0.7 zegt vooraf waarom dat zwaar wordt:

> De `cluster_cap` is **vacuous** op dit universum. Zes symbolen, gemiddelde
> paarsgewijze correlatie 0,735, één cluster. **HRP is een clustering-allocator
> op een universum waarvan bewezen is dat het geen clusterstructuur heeft.**

Vindt HRP niettemin een verbetering, dan is scepsis geboden: op zes assets met
rho ~ 0,73 is het verschil met Inverse Volatility enkele basispunten aan
gewichten, en de kans op overfitting op de recursieve bisectie-ordening is
aanzienlijk. `HrpAllocation.tree_order` wordt daarom meegegeven, zodat
deliverable 24 de stabiliteit van die ordening over folds kan MÉTEN in plaats
van aannemen.

Ref: audit §13.1 (HRP is Research Track; Markowitz ruw is banned);
`fase_6_advanced_research.md` deliverable 23/24, no-go 11 en 15;
`reports/phase5_cluster_concentration_audit.md`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from ..backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from ..utils.failfast import DataContractError, require

logger = logging.getLogger(__name__)

__all__ = [
    "ADMISSION_CRITERIA",
    "HRPOptimizer",
    "HrpAllocation",
    "HrpResearchGate",
    "hrp_weights",
]

#: Minimale lengte van de reden op een `HrpResearchGate`. Lang genoeg dat
#: `"test"` of `"onderzoek"` niet volstaat, kort genoeg om geen ritueel te zijn.
_MIN_REASON_CHARS = 40

#: De twee eisen die samen de productiegate zouden openen. Als CODE, zodat een
#: toekomstige promotie ze moet aanraken in plaats van eromheen te schrijven.
#: Beide, niet één van beide.
ADMISSION_CRITERIA: tuple[str, ...] = (
    "turnover-gecorrigeerde OOS Sharpe van HRP > die van Inverse Volatility, "
    "gemeten door backtest/engine.py met config_hash 1b60cb664fbf9a2a en de "
    "shift(2)-conventie",
    "de HRP-boomordening is STABIEL over de walk-forward folds; een ordening "
    "die per fold wisselt, alloceert op ruis",
)


@dataclass(frozen=True)
class HrpResearchGate:
    """Het bewijs dat de aanroeper WEET dat HRP niet is toegelaten.

    Er is geen default-instantie in enige handtekening in dit bestand. Wie HRP
    wil draaien, construeert dit object en schrijft op waarom. Die reden reist
    mee in `HrpAllocation`, dus zij verschijnt in elk artefact dat eruit volgt.
    """

    reason: str
    #: De bevroren pre-registratie waaronder dit onderzoek valt. Een HRP-run
    #: zonder pre-registratie is een search die niet in `M` terechtkomt.
    preregistration_id: str

    def __post_init__(self) -> None:
        require(
            len(self.reason.strip()) >= _MIN_REASON_CHARS,
            f"HrpResearchGate zonder inhoudelijke reden (minimaal "
            f"{_MIN_REASON_CHARS} tekens, kreeg "
            f"{len(self.reason.strip())}). HRP is RESEARCH ONLY en technisch "
            f"geblokkeerd voor productie (fase-6 no-go 15). Wie hem draait, "
            f"schrijft op waarvoor.",
            DataContractError,
            reason=self.reason,
        )
        require(
            bool(str(self.preregistration_id).strip()),
            "HrpResearchGate zonder preregistration_id. Een HRP-run is een "
            "trial; zonder bevroren pre-registratie telt hij niet mee in M, en "
            "dan is elke DSR die erop volgt te optimistisch.",
            DataContractError,
        )


@dataclass(frozen=True)
class HrpAllocation:
    """HRP-gewichten, permanent gemarkeerd als niet-toelaatbaar bewijs.

    `evidence_class` is een `Literal` met precies één toegestane waarde. Er is
    dus geen constructie waarin een HRP-allocatie zichzelf als promotiebewijs
    aanbiedt — dezelfde vorm als `backtest/vectorized.py::VectorizedResult`.
    """

    weights: pd.Series
    #: Bladvolgorde uit de linkage. Deliverable 24 meet hiermee of de ordening
    #: stabiel is over folds; een wisselende ordening alloceert op ruis.
    tree_order: tuple[int, ...]
    linkage_method: str
    research_gate: HrpResearchGate
    evidence_class: Literal["NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE"] = NOT_ADMISSIBLE
    audit: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """De vorm waarin dit een artefact of ledger-entry in gaat.

        `EVIDENCE_KEY` staat erin omdat `is_vectorized_evidence()` op platte
        dicts werkt: dat is de vorm waarin een resultaat de promotion gate
        feitelijk bereikt.
        """
        return {
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "tree_order": list(self.tree_order),
            "linkage_method": self.linkage_method,
            EVIDENCE_KEY: self.evidence_class,
            "research_reason": self.research_gate.reason,
            "preregistration_id": self.research_gate.preregistration_id,
            "admission_criteria_not_yet_met": list(ADMISSION_CRITERIA),
            **self.audit,
        }


# =============================================================================
# Kern-algoritme — ONGEWIJZIGD. De gate zit eromheen, niet erin.
# =============================================================================

def _cov_ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    # Drop rows where ANY asset has NaN rather than filling with 0.
    # fillna(0.0) artificially decorrelates assets during data gaps
    # (missing bar → zero covariance with all other assets), which
    # produces phantom diversification and incorrect HRP allocations.
    clean = returns.dropna(how="any")
    if len(clean) < 5:
        clean = returns.fillna(returns.mean())
    lw = LedoitWolf()
    lw.fit(clean.values)
    return lw.covariance_


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-9, 1e-9, std)
    return cov / np.outer(std, std)


def _distance_matrix(corr: np.ndarray) -> np.ndarray:
    """Correlatie → afstand: D = sqrt(0.5 * (1 - rho))."""
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def _quasi_diagonalise(link: np.ndarray, n_assets: int) -> list[int]:
    """Extract leaf order from a scipy linkage matrix."""
    return list(hierarchy.leaves_list(link))


def _recursive_bisect(
    cov: np.ndarray,
    sorted_idx: list[int],
    weights: np.ndarray,
) -> None:
    """In-place recursive bisection allocation (modifies ``weights``)."""
    if len(sorted_idx) == 1:
        return

    mid = len(sorted_idx) // 2
    left  = sorted_idx[:mid]
    right = sorted_idx[mid:]

    # Inverse-variance contribution of each cluster
    var_left  = _cluster_var(cov, left)
    var_right = _cluster_var(cov, right)

    total = var_left + var_right + 1e-12
    alpha_left  = var_right / total    # left gets share proportional to right var
    alpha_right = var_left  / total

    weights[left]  *= alpha_left
    weights[right] *= alpha_right

    _recursive_bisect(cov, left, weights)
    _recursive_bisect(cov, right, weights)


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    """Minimum-variance portfolio variance within a cluster."""
    sub_cov = cov[np.ix_(idx, idx)]
    inv_var = 1.0 / (np.diag(sub_cov) + 1e-12)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


# =============================================================================
# De gepoorte ingang
# =============================================================================

def hrp_weights(
    returns: pd.DataFrame,
    *,
    research_gate: HrpResearchGate,
    linkage_method: str = "ward",
) -> HrpAllocation:
    """Bereken HRP-gewichten. RESEARCH ONLY.

    Parameters
    ----------
    returns
        DataFrame met asset-returns (rijen = tijd, kolommen = assets).
        Minimaal 5 rijen; 30 of meer is aan te raden.
    research_gate
        **VERPLICHT en keyword-only.** Er is geen default. Zie
        `HrpResearchGate` en de moduledocstring: HRP is technisch geblokkeerd
        voor productie tot `ADMISSION_CRITERIA` beide zijn aangetoond.
    linkage_method
        Linkage-methode voor `scipy.cluster.hierarchy` (default `"ward"`).

    Returns
    -------
    HrpAllocation
        Draagt permanent `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`.

    Raises
    ------
    DataContractError
        Bij een ontbrekende of ongeldige `research_gate`, of bij een lege
        returns-matrix.
    """
    require(
        isinstance(research_gate, HrpResearchGate),
        f"hrp_weights() vereist een HrpResearchGate, kreeg "
        f"{type(research_gate).__name__}. HRP is RESEARCH ONLY en technisch "
        f"geblokkeerd voor productiegebruik (fase-6 deliverable 23). De gate "
        f"opent pas wanneer BEIDE criteria zijn aangetoond: "
        f"{'; '.join(ADMISSION_CRITERIA)}.",
        DataContractError,
    )
    assets = list(returns.columns)
    require(
        len(assets) > 0,
        "hrp_weights() kreeg een matrix zonder kolommen; er valt niets te "
        "alloceren.",
        DataContractError,
    )
    n = len(assets)

    if n == 1:
        return HrpAllocation(
            weights=pd.Series([1.0], index=assets, name="hrp_weight"),
            tree_order=(0,), linkage_method=linkage_method,
            research_gate=research_gate,
            audit={"n_assets": 1, "note": "eén asset; geen clustering mogelijk"},
        )

    cov  = _cov_ledoit_wolf(returns)
    corr = _corr_from_cov(cov)
    dist = _distance_matrix(corr)

    condensed = squareform(dist, checks=False)
    link = hierarchy.linkage(condensed, method=linkage_method)
    sorted_idx = _quasi_diagonalise(link, n)

    weights = np.ones(n) / n
    _recursive_bisect(cov, sorted_idx, weights)
    weights = weights / weights.sum()

    # De gemiddelde paarsgewijze correlatie reist mee. Op dit universum is hij
    # ~0,73 (fase-6 §0.7), en dat getal is de belangrijkste context bij elke
    # HRP-uitkomst: clustering op een universum zonder clusterstructuur.
    off_diag = corr[~np.eye(n, dtype=bool)]
    return HrpAllocation(
        weights=pd.Series(weights, index=assets, name="hrp_weight"),
        tree_order=tuple(int(i) for i in sorted_idx),
        linkage_method=linkage_method,
        research_gate=research_gate,
        audit={
            "n_assets": n,
            "n_obs": int(len(returns)),
            "mean_abs_pairwise_corr": float(np.abs(off_diag).mean()),
        },
    )


class HRPOptimizer:
    """Stateful HRP-optimizer. RESEARCH ONLY — zie de moduledocstring.

    De gate wordt bij CONSTRUCTIE geëist en niet pas bij `optimize()`. Zou hij
    pas bij de aanroep worden gevraagd, dan kan een object rondgereikt worden
    dat er onschuldig uitziet en pas diep in een aanroepketen crasht.
    """

    def __init__(
        self,
        *,
        research_gate: HrpResearchGate,
        linkage_method: str = "ward",
    ) -> None:
        require(
            isinstance(research_gate, HrpResearchGate),
            f"HRPOptimizer vereist een HrpResearchGate, kreeg "
            f"{type(research_gate).__name__}. HRP is technisch geblokkeerd "
            f"voor productiegebruik (fase-6 no-go 15).",
            DataContractError,
        )
        self.research_gate = research_gate
        self.linkage_method = linkage_method
        self._last: HrpAllocation | None = None

    def optimize(self, returns: pd.DataFrame) -> HrpAllocation:
        self._last = hrp_weights(
            returns, research_gate=self.research_gate,
            linkage_method=self.linkage_method)
        return self._last

    @property
    def allocation(self) -> HrpAllocation | None:
        return self._last

    @property
    def weights(self) -> pd.Series | None:
        return None if self._last is None else self._last.weights
