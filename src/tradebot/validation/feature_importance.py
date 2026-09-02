# src/tradebot/validation/feature_importance.py
"""MDI en SFI onder Purged Walk-Forward met embargo. Deliverable 21.

WAAROM ER TWEE ZIJN, EN WAAROM DE EERSTE ALLEEN NIET GENOEG IS
================================================================
**MDI** (mean decrease impurity) leest af hoeveel elke feature bijdroeg aan de
splitsingen van het model. Hij is goedkoop -- hij komt uit de fit die er toch al
was -- en hij heeft twee bekende gebreken die AFML hoofdstuk 8 uitschrijft:

* hij is IN-SAMPLE. Een feature die het trainvenster prachtig verklaart en
  daarbuiten niets doet, krijgt een hoge MDI;
* hij is *substitutie-gevoelig*. Twee features die hetzelfde meten, delen hun
  belang en lijken allebei half zo belangrijk als ze zijn. Op dit universum is
  dat geen randgeval: `vol_realized_20`, `vol_ewma` en `vol_parkinson_20` meten
  dezelfde grootheid met drie schatters.

**SFI** (single feature importance) heeft geen van beide problemen: elk model
krijgt ÉÉN feature en wordt OUT-OF-SAMPLE gescoord, onder dezelfde purged
walk-forward. Substitutie kan niet optreden, want er is niets om mee te delen.
Wat SFI niet ziet is INTERACTIE -- een feature die alleen in combinatie met een
andere werkt, scoort hier laag.

De twee samen zeggen daarom meer dan elk apart, en dat is precies waarom §12.1
ze allebei eist. Waar zij het oneens zijn, staat de reden in het rapport in
plaats van dat er een winnaar wordt gekozen.

DE POORT IS NIET DE IMPORTANCE
===============================
Geen van beide getallen promoveert of blokkeert iets. Zij zijn DIAGNOSTIEK: zij
verklaren waar een AUC vandaan komt en of hij op iets rust dat plausibel is. Het
oordeel valt op de OOS-AUC tegen de drempel uit de pre-registratie, en op de
economische toets door de engine.

Ref: AFML hoofdstuk 8; ARCHITECTUUR_AUDIT_2026-08-22.md §12.1.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cv.walk_forward import WalkForwardCV
from ..schemas.config import MetaLabelConfig
from ..train.meta_label import (
    FoldPredictions,
    MetaLabelDataset,
    MetaLabelSpec,
    walk_forward_predictions,
)
from ..utils.failfast import DataContractError, require

__all__ = ["ImportanceResult", "mdi_importance", "pooled_predictions",
           "sfi_importance", "weighted_auc"]


def weighted_auc(
    target: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None,
) -> float:
    """AUC via de rangvorm van de Mann-Whitney U-statistiek.

    Met gewichten wordt de AUC de kans dat een willekeurig getrokken positief
    event hoger scoort dan een willekeurig getrokken negatief, waarbij beide
    worden getrokken PROPORTIONEEL AAN HUN UNIQUENESS. Dat is de juiste maat
    wanneer buren negen van hun tien bars delen: zonder weging telt elk van die
    buren als een volwaardige waarneming, en dat is precies waarom
    meta-labeling-AUC's in de literatuur zo vaak te hoog uitvallen.

    Gelijke scores krijgen een half punt -- de standaardconventie, en zij houdt
    de AUC van een constante voorspeller op exact 0,5 in plaats van op 0 of 1.
    """
    target = np.asarray(target)
    score = np.asarray(score, dtype=np.float64)
    weight = (np.ones_like(score) if weight is None
              else np.asarray(weight, dtype=np.float64))
    require(
        target.shape == score.shape == weight.shape,
        "AUC op reeksen van verschillende lengte.",
        DataContractError, n_target=target.size, n_score=score.size,
        n_weight=weight.size,
    )
    positive = target == 1
    require(
        bool(positive.any()) and bool((~positive).any()),
        "Een AUC op één klasse bestaat niet. Dat is een adequaatheidsbevinding "
        "en geen numeriek detail.",
        DataContractError, n_positive=int(positive.sum()),
        n_negative=int((~positive).sum()),
    )
    pos_score, pos_weight = score[positive], weight[positive]
    neg_score, neg_weight = score[~positive], weight[~positive]
    order = np.argsort(neg_score, kind="stable")
    neg_sorted, neg_weight_sorted = neg_score[order], neg_weight[order]
    cumulative = np.concatenate([[0.0], np.cumsum(neg_weight_sorted)])
    lower = cumulative[np.searchsorted(neg_sorted, pos_score, side="left")]
    equal = (cumulative[np.searchsorted(neg_sorted, pos_score, side="right")]
             - lower)
    concordant = float(np.sum(pos_weight * (lower + 0.5 * equal)))
    return concordant / float(pos_weight.sum() * neg_weight.sum())


@dataclass(frozen=True)
class ImportanceResult:
    """Belang per feature, met de spreiding over folds erbij.

    De spreiding is niet decoratief: een feature die in één fold alles verklaart
    en in de rest niets, is een ander verhaal dan een die overal middelmatig
    meedoet, en het gemiddelde maakt die twee gelijk.
    """

    method: str
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    #: ``(n_folds, n_features)`` -- voor MDI EN voor SFI. De rij is de fold, de
    #: kolom de feature; `names[j]` hoort bij kolom `j`.
    per_fold: np.ndarray

    def ranked(self) -> list[tuple[str, float, float]]:
        order = np.argsort(-self.mean)
        return [(self.names[i], float(self.mean[i]), float(self.std[i]))
                for i in order]

    def as_record(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "features": [
                {"name": name, "mean": mean, "std": std}
                for name, mean, std in self.ranked()
            ],
        }


def mdi_importance(
    folds: Sequence[FoldPredictions], names: Sequence[str],
) -> ImportanceResult:
    """Gemiddelde MDI over folds, genormaliseerd naar een som van 1 per fold.

    Normaliseren per fold en niet over het geheel: CatBoost's importances zijn
    al per fit relatief, en folds met meer trainrijen zouden anders zwaarder
    wegen om een reden die niets met de features te maken heeft.
    """
    require(
        bool(folds),
        "MDI zonder folds. Er is niets gefit om belang uit af te lezen.",
        DataContractError,
    )
    rows = []
    for fold in folds:
        values = np.asarray(fold.feature_importance, dtype=np.float64)
        require(
            values.size == len(names),
            "De importance-vector past niet op de featurenamen.",
            DataContractError, n_values=int(values.size), n_names=len(names),
        )
        total = float(values.sum())
        rows.append(values / total if total > 0.0 else values)
    matrix = np.vstack(rows)
    return ImportanceResult(
        method="MDI", names=tuple(names), mean=matrix.mean(axis=0),
        std=matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1
        else np.zeros(matrix.shape[1]),
        per_fold=matrix,
    )


def sfi_importance(
    dataset: MetaLabelDataset,
    cv: WalkForwardCV,
    spec: MetaLabelSpec,
    cfg: MetaLabelConfig,
    *,
    n_bars: int,
    embargo_bars: int,
) -> ImportanceResult:
    """OOS-AUC per feature, elk model op ÉÉN feature, zelfde purged folds.

    Dit is de dure kant van hoofdstuk 8 -- één fit per feature per fold -- en
    het is de reden dat hij wordt gedraaid voor de BESTE spec en niet voor alle
    zes. Zes keer dezelfde diagnose op zes bijna gelijke modellen voegt niets
    toe aan het begrip van waar de AUC vandaan komt.
    """
    names = list(dataset.features.columns)
    per_fold: list[list[float]] = []
    for name in names:
        single = MetaLabelDataset(
            features=dataset.features[[name]], target=dataset.target,
            event_bar=dataset.event_bar, exit_bar=dataset.exit_bar,
            side=dataset.side, symbol=dataset.symbol,
            uniqueness=dataset.uniqueness,
        )
        folds = walk_forward_predictions(
            single, cv, spec, cfg, n_bars=n_bars, embargo_bars=embargo_bars)
        per_fold.append([
            weighted_auc(f.target, f.probability, f.uniqueness)
            if len(set(f.target.tolist())) == 2 else float("nan")
            for f in folds
        ])
    # TRANSPONEREN. `per_fold` wordt hierboven per FEATURE opgebouwd, dus als
    # (n_features, n_folds), terwijl `mdi_importance` hetzelfde veld vult als
    # (n_folds, n_features). Twee orientaties onder een naam is een val voor de
    # eerste lezer die op fold indexeert; `ImportanceResult.per_fold` legt de
    # vorm nu vast en beide methoden leveren hem zo aan.
    matrix = np.asarray(per_fold, dtype=np.float64).T
    usable = np.count_nonzero(np.isfinite(matrix), axis=0)
    require(
        bool(np.any(usable > 0)),
        "Geen enkele fold leverde voor ook maar een feature een bruikbare AUC. "
        "Er valt dan geen belang af te lezen.",
        DataContractError, n_features=int(matrix.shape[1]),
    )
    # `nanstd(..., ddof=1)` geeft NaN plus een RuntimeWarning zodra een feature
    # in maar EEN fold een bruikbare AUC opleverde. Dat is geen spreiding die
    # ontbreekt maar een spreiding die niet bestaat; `mdi_importance` doet
    # hetzelfde met zijn `matrix.shape[0] > 1`-tak.
    std = np.zeros(matrix.shape[1], dtype=np.float64)
    enough = usable > 1
    if np.any(enough):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            std[enough] = np.nanstd(matrix[:, enough], axis=0, ddof=1)
    with warnings.catch_warnings():
        # Een feature zonder enkele bruikbare fold houdt NaN als gemiddelde --
        # "niet gemeten", en dat mag niet als een getal worden gelezen.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
    return ImportanceResult(
        method="SFI", names=tuple(names), mean=mean, std=std,
        per_fold=matrix,
    )


def pooled_predictions(
    folds: Sequence[FoldPredictions],
) -> Mapping[str, np.ndarray]:
    """Alle testrijen van alle folds achter elkaar, in foldvolgorde."""
    return {
        "target": np.concatenate([f.target for f in folds]),
        "probability": np.concatenate([f.probability for f in folds]),
        "uniqueness": np.concatenate([f.uniqueness for f in folds]),
        "row_index": np.concatenate([f.row_index for f in folds]),
    }
