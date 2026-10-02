# src/tradebot/train/meta_label.py
"""CatBoost als SECONDARY model. Deliverable 20, stap 12.

WAT DIT MODEL WEL EN NIET MAG DOEN, EN WAAROM DAT EEN CONSTRUCTIE IS
=====================================================================
§24 van de audit archiveerde CatBoost voor DIRECTIONELE voorspelling. De enige
route terug is als meta-labeler: het primaire Phase 3-signaal bepaalt de
richting, en dit model voorspelt uitsluitend de KANS DAT DIE TRADE SLAAGT.

De pre-registratie noemt dat "technisch geblokkeerd", en dat is hier letterlijk
genomen. Er bestaat geen handtekening waarmee je dit model een eigen doelvector
geeft: :func:`build_dataset` leidt het doel af uit :class:`BarrierLabels` en
neemt geen `target`-argument. Wie een richting of een rendement zou willen
voorspellen, moet een ANDER bestand schrijven, en dat is zichtbaar in een diff.

De richting zit al VERWERKT in het label voordat het bestaat: `meta_label` is 1
wanneer de trade van het primaire model geld opleverde, gezien vanuit de
POSITIE. Een model dat daarop leert, kan per constructie niet leren waar de
markt heen gaat -- alleen wanneer het primaire signaal gelijk had.

HET SCHEMA IS WALK-FORWARD, EN DE PURGE GEBEURT OP `t1`
========================================================
Elke fold traint op het rollende venster van `train_bars` bars dat VOOR zijn
testvenster ligt, en op niets anders. Data van ná het testvenster komt de fit
niet in -- niet als trainrij, niet als validatieset. Dat is wat
`scheme: purged_walk_forward` in de bevroren pre-registratie betekent, en het is
strenger dan de purged K-fold van AFML hoofdstuk 7, die het hele complement van
de testfold als trainmateriaal gebruikt.

Binnen dat venster is de purge exact. Een triple-barrier-label van bar `t`
gebruikt bars tot en met `t + 1 + H`. De bevroren pre-registratie draagt
`embargo_bars: 5` en `label_horizon_bars: 1` uit `conf/validation/default.yaml`,
en die 1 slaat op het PHASE 3-baselinelabel, niet op de horizon van 10 bars
hier. Vijf bars embargo zou dus te weinig zijn -- als de embargo het enige
mechanisme was.

Dat is hij niet. Elk trainevent waarvan het interval `[t, t1]` het testvenster
RAAKT, valt weg; dat is horizon-bewust en exact, omdat `exit_bar` per event
bekend is. De embargo van 5 bars komt daar bovenop als buffer tegen seriële
correlatie in de aanloop naar het venster. De combinatie is strikt sterker dan
"embargo >= horizon"; :func:`purged_training_index` telt per fold hoeveel events
er in elke categorie sneuvelen, zodat het rapport dat kan laten zien in plaats
van beweren -- inclusief `n_after_test`, het aantal events dat het schema
weggooit en dat onder K-fold in de fit zou hebben gezeten.

WAAROM ER GEEN SCALER IN DEZE PIPELINE ZIT
===========================================
Geen omissie maar een gevolg: CatBoost splitst op ordeningen en is invariant
onder elke monotone transformatie per feature. Een per-fold gefitte scaler zou
hier niets veranderen behalve de indruk van zorgvuldigheid wekken. Wat de regel
"scaler per fold gefit" beschermt -- dat er niets wordt gefit op data buiten het
trainvenster -- geldt onverkort en wordt getest: dit module krijgt de testrijen
nooit te zien, en `test_the_model_never_sees_the_test_window` bewijst dat door de
testrijen te vervangen en te eisen dat het model niet verandert.

Ontbrekende waarden worden evenmin geimputeerd. CatBoost verwerkt NaN zelf, en
een imputatie zou een schatting toevoegen die niemand heeft gevraagd.

Ref: AFML hoofdstuk 3 (meta-labeling), 4 (uniqueness) en 7 (purged CV);
ARCHITECTUUR_AUDIT_2026-08-22.md §12.1, §24; pre-registratie
`56395fa2013768014c0c915edf346770`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..cv.walk_forward import WalkForwardCV, WalkForwardFold
from ..labeling.vol_barriers import BarrierLabels
from ..schemas.config import MetaLabelConfig, meta_label_config
from ..utils.failfast import DataContractError, require, require_dependency

__all__ = [
    "META_LABEL_GRID",
    "shuffled_targets",
    "FoldPredictions",
    "MetaLabelDataset",
    "MetaLabelSpec",
    "build_dataset",
    "fit_secondary_model",
    "meta_label_grid",
    "purged_training_index",
    "walk_forward_fit_predict",
    "walk_forward_predictions",
]


@dataclass(frozen=True)
class MetaLabelSpec:
    """Eén trial uit de H3-parameterruimte: een diepte en een leerstap."""

    depth: int
    learning_rate: float

    @property
    def label(self) -> str:
        return f"catboost-d{self.depth}-lr{self.learning_rate:g}"


def meta_label_grid(cfg: MetaLabelConfig) -> tuple[MetaLabelSpec, ...]:
    """De gepre-registreerde ruimte, in vaste volgorde."""
    return tuple(
        MetaLabelSpec(depth=int(depth), learning_rate=float(rate))
        for depth in cfg.depths
        for rate in cfg.learning_rates
    )


#: De zes trials van `56395fa2013768014c0c915edf346770`, uit `conf/`. Hij staat
#: hier als CONSTANTE en niet als iets dat een aanroeper mag samenstellen: een
#: campagne die zijn eigen ruimte bouwt, kan er stilzwijgend een zevende bij
#: zetten en dan klopt `M` niet meer.
META_LABEL_GRID: tuple[MetaLabelSpec, ...] = meta_label_grid(
    meta_label_config())


@dataclass(frozen=True)
class MetaLabelDataset:
    """Events, features en labels van ALLE symbolen op één tijdas.

    De pooling over symbolen is geen gemak maar de opzet: één meta-labeler voor
    het hele universum, zoals §12.1 voorschrijft. `symbol` blijft mee zodat het
    rapport per symbool kan uitsplitsen en de purging over de tijdas kan lopen
    zonder symbolen door elkaar te halen.
    """

    features: pd.DataFrame
    #: Binair. Er is geen andere doelvector mogelijk; zie de moduledocstring.
    target: np.ndarray
    #: Bar-index van het event en van de exit, op het gedeelde raster.
    event_bar: np.ndarray
    exit_bar: np.ndarray
    side: np.ndarray
    symbol: np.ndarray
    #: Gemiddelde uniqueness per event (AFML hoofdstuk 4). Weegt de fit én
    #: bepaalt de effectieve steekproefgrootte in het rapport.
    uniqueness: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.features)
        for name in ("target", "event_bar", "exit_bar", "side", "symbol",
                     "uniqueness"):
            require(
                len(getattr(self, name)) == n,
                f"`{name}` heeft niet evenveel rijen als de featurematrix.",
                DataContractError, n_features=n,
                n_other=len(getattr(self, name)), field=name,
            )
        require(
            n > 0,
            "Een lege dataset. Er valt niets te fitten en niets te toetsen.",
            DataContractError,
        )
        unique_targets = set(np.unique(self.target).tolist())
        require(
            unique_targets <= {0, 1},
            "Het doel is niet binair. Dit model voorspelt uitsluitend de "
            "SUCCESKANS van het primaire signaal; een continu doel zou een "
            "rendement of een richting zijn, en dat is precies wat §24 heeft "
            "gearchiveerd.",
            DataContractError, observed=sorted(unique_targets)[:6],
        )
        require(
            set(np.unique(self.side).tolist()) <= {-1.0, 1.0},
            "Een event zonder richting. Het primaire signaal bepaalt de "
            "richting; een event zonder richting is geen trade.",
            DataContractError,
        )
        require(
            bool(np.all(self.exit_bar > self.event_bar)),
            "Een event dat sluit op of vóór zijn eigen bar. Dat is lookahead "
            "in de labeling, geen boekhoudkundig detail.",
            DataContractError,
        )
        require(
            bool(np.all((self.uniqueness > 0.0) & (self.uniqueness <= 1.0))),
            "Uniqueness buiten (0, 1]. AFML hoofdstuk 4 definieert hem als een "
            "fractie van gedeelde bars.",
            DataContractError,
        )

    def __len__(self) -> int:
        return int(len(self.features))

    @property
    def positive_ratio(self) -> float:
        return float(self.target.mean())

    @property
    def effective_n(self) -> float:
        """De uniqueness-gewogen steekproefgrootte."""
        return float(self.uniqueness.sum())


def _average_uniqueness(event: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """AFML 4.2: het gemiddelde aandeel van een event in zijn eigen bars.

    Een bar die door `c` events wordt gedeeld, draagt `1/c` bij aan elk van die
    events. Met een horizon van tien bars en een event op vrijwel elke bar delen
    buren negen van hun tien toekomstige bars, en dat is de reden dat het
    nominale aantal labels de steekproefgrootte fors overschat.
    """
    n_bars = int(exit_.max()) + 1
    concurrency = np.zeros(n_bars, dtype=np.float64)
    for start, stop in zip(event, exit_, strict=True):
        concurrency[start:stop + 1] += 1.0
    out = np.empty(event.size, dtype=np.float64)
    for i, (start, stop) in enumerate(zip(event, exit_, strict=True)):
        out[i] = float(np.mean(1.0 / concurrency[start:stop + 1]))
    return out


def build_dataset(
    features: Mapping[str, pd.DataFrame],
    labels: Mapping[str, BarrierLabels],
) -> MetaLabelDataset:
    """Bouw de gepoolde dataset uit per-symbool features en barrièrelabels.

    Er is GEEN `target`-argument. Het doel komt uit `BarrierLabels.meta_label`
    en nergens anders vandaan; dat is de technische blokkade die de
    pre-registratie eist.
    """
    require(
        set(features) == set(labels),
        "Features en labels dekken niet dezelfde symbolen.",
        DataContractError, features=sorted(features), labels=sorted(labels),
    )
    frames: list[pd.DataFrame] = []
    target: list[np.ndarray] = []
    event: list[np.ndarray] = []
    exit_: list[np.ndarray] = []
    side: list[np.ndarray] = []
    symbol: list[np.ndarray] = []
    uniqueness: list[np.ndarray] = []
    for name in sorted(features):
        matrix, label = features[name], labels[name]
        require(
            len(label) > 0,
            "Een symbool zonder enkel event. Het primaire signaal handelde daar "
            "nooit, en dan valt er niets te meta-labelen.",
            DataContractError, symbol=name,
        )
        require(
            int(label.exit_idx.max()) < len(matrix),
            "Een exit voorbij het einde van de featurematrix; labels en "
            "features komen van verschillende reeksen.",
            DataContractError, symbol=name,
        )
        frames.append(matrix.iloc[label.event_idx])
        target.append(label.meta_label.astype(np.int64))
        event.append(label.event_idx.astype(np.int64))
        exit_.append(label.exit_idx.astype(np.int64))
        side.append(label.side.astype(np.float64))
        symbol.append(np.full(len(label), name, dtype=object))
        uniqueness.append(_average_uniqueness(label.event_idx, label.exit_idx))
    return MetaLabelDataset(
        features=pd.concat(frames, axis=0),
        target=np.concatenate(target), event_bar=np.concatenate(event),
        exit_bar=np.concatenate(exit_), side=np.concatenate(side),
        symbol=np.concatenate(symbol),
        uniqueness=np.concatenate(uniqueness),
    )


def purged_training_index(
    dataset: MetaLabelDataset, fold: WalkForwardFold, *, embargo_bars: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Trainrijen van deze fold: het WALK-FORWARD venster, gepurged op `t1`.

    Het trainvenster is dat van de fold zelf -- `[fold.train_idx[0],
    test_start - embargo_bars)` -- en dus uitsluitend VERLEDEN ten opzichte van
    het testvenster. Daarbinnen valt weg wat het testvenster raakt:

    * events buiten het rollende venster van `train_bars` bars;
    * events in de embargozone direct VOOR het testvenster;
    * events in het venster waarvan het label het testvenster IN loopt
      (`exit_bar >= test_start`) -- dat is de purge, en zij is exact omdat
      `exit_bar` per event bekend is;
    * alles vanaf `test_start`, inclusief alles NA het testvenster.

    WAAROM DIE LAATSTE REGEL ER STAAT
    ==================================
    Deze functie hield eerder ook de events NA het testvenster vast
    (`keep = (before & ~overlapping) | after`). Dat is purged K-fold zoals AFML
    hoofdstuk 7 hem beschrijft, en op zichzelf een verdedigbaar schema -- maar
    het is NIET het schema dat is gepre-registreerd, en het brak drie dingen
    tegelijk:

    1. `56395fa2013768014c0c915edf346770` draagt `scheme: purged_walk_forward`
       met `train_bars: 500`. Onder de oude regel werd `train_bars` nooit
       gebruikt: elke fold trainde op ~9.650 van de 10.330 events in plaats van
       op de ~2.900 van een venster van 500 bars maal zes symbolen.
    2. De Data Adequacy Gate -- de poort die de fit AUTORISEERT -- mat precies
       die ~2.900 events per fold. Poort en run waren het dus oneens over wat
       een fold is, en de poort stond groen voor een fit die zij niet had
       gemeten.
    3. Het gefilterde signaal gaat in stap 13 door de authoritative engine. Een
       filter dat op bar `t` een trade tegenhoudt op grond van een model dat op
       data van ná `t` is gefit, levert geen OOS Sharpe op maar een
       vooruitkijkende. Dat raakt de economische toets, niet alleen de
       statistische.

    De meting staat in `reports/META_LABELING_EVALUATION.md`; het defect staat
    in het exit-rapport.
    """
    test_start = int(fold.test_idx[0])
    test_end = int(fold.test_idx[-1])
    train_lo = int(fold.train_idx[0])
    embargo_start = test_start - int(embargo_bars)

    in_window = ((dataset.event_bar >= train_lo)
                 & (dataset.event_bar < embargo_start))
    overlapping = in_window & (dataset.exit_bar >= test_start)
    keep = in_window & ~overlapping

    inside = (dataset.event_bar >= test_start) & (dataset.event_bar <= test_end)
    embargoed = ((dataset.event_bar >= embargo_start)
                 & (dataset.event_bar < test_start))
    return np.flatnonzero(keep), {
        "n_candidates": int(len(dataset)),
        "n_kept": int(keep.sum()),
        "n_inside_test": int(inside.sum()),
        "n_purged_by_overlap": int(overlapping.sum()),
        "n_embargoed": int(embargoed.sum()),
        #: Alles na het testvenster. Onder walk-forward hoort dit NOOIT in de
        #: fit; het staat in het artefact zodat een lezer ziet hoeveel data het
        #: schema weggooit -- en zodat een terugkeer naar K-fold zichtbaar is
        #: als dit getal ooit in `n_kept` opduikt.
        "n_after_test": int((dataset.event_bar > test_end).sum()),
        "n_before_train_window": int((dataset.event_bar < train_lo).sum()),
    }


def fit_secondary_model(
    features: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    spec: MetaLabelSpec,
    cfg: MetaLabelConfig,
) -> Any:
    """Fit CatBoost op precies deze rijen. Ziet niets anders.

    De testrijen worden niet doorgegeven -- niet als validatieset, niet voor
    early stopping. Een `eval_set` op het testvenster is de meest voorkomende
    manier waarop een 'out-of-sample' AUC in-sample wordt: het aantal iteraties
    wordt dan op de testdata gekozen.
    """
    catboost = require_dependency(
        "catboost",
        needed_for="het secondary model van H3",
        install_hint="pip install catboost",
    )
    require(
        len(features) == len(target) == len(weights),
        "Features, doel en gewichten hebben niet dezelfde lengte.",
        DataContractError, n_features=len(features), n_target=len(target),
        n_weights=len(weights),
    )
    require(
        len(set(np.unique(target).tolist())) == 2,
        "Een trainvenster met maar één klasse. Er valt geen kans te schatten; "
        "dat is een adequaatheidsbevinding en geen modelprobleem.",
        DataContractError, classes=sorted(set(np.unique(target).tolist())),
    )
    model = catboost.CatBoostClassifier(
        depth=spec.depth, learning_rate=spec.learning_rate,
        iterations=cfg.iterations, l2_leaf_reg=cfg.l2_leaf_reg,
        loss_function="Logloss", random_seed=cfg.seed,
        thread_count=cfg.thread_count, verbose=False,
        allow_writing_files=False,
    )
    model.fit(features.to_numpy(dtype="float64"), target,
              sample_weight=weights)
    return model


@dataclass(frozen=True)
class FoldPredictions:
    """Wat één fold opleverde: kansen op zijn testrijen, plus de purge-telling."""

    fold_id: int
    row_index: np.ndarray
    probability: np.ndarray
    target: np.ndarray
    uniqueness: np.ndarray
    n_train: int
    purge: Mapping[str, int]
    feature_importance: np.ndarray
    #: Wat de fit naast zijn kansen meegeeft (bijv. de kalibratieset). Leeg voor CatBoost.
    extras: Mapping[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id, "n_test": int(self.row_index.size),
            "n_train": self.n_train, "positive_ratio": float(
                self.target.mean()) if self.target.size else float("nan"),
            **{f"purge_{k}": v for k, v in self.purge.items()},
        }


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
    """Purged walk-forward over de bar-as; één fit per fold.

    `target` bestaat uitsluitend voor de NEGATIEVE CONTROLE: daar wordt hetzelfde
    model op gerandomiseerde labels gedraaid. Hij is geen route naar een eigen
    doelvector -- de aanroeper kan er alleen een PERMUTATIE van het bestaande
    doel in stoppen, en `walk_forward_predictions` controleert dat.
    """
    def fit(rows: np.ndarray, labels: np.ndarray) -> Any:
        return fit_secondary_model(
            dataset.features.iloc[rows], labels[rows],
            dataset.uniqueness[rows], spec, cfg)

    return walk_forward_fit_predict(dataset, cv, fit, n_bars=n_bars,
                                    embargo_bars=embargo_bars, target=target)


def shuffled_targets(
    dataset: MetaLabelDataset, seed: int, n_replicates: int,
) -> list[np.ndarray]:
    """Permutaties van het echte doel, voor de negatieve controle.

    Permuteren en niet opnieuw trekken: de klassebalans blijft daarmee exact
    gelijk, zodat een verschil in AUC niet uit een andere basisrate kan komen.
    """
    rng = np.random.default_rng(seed)
    return [rng.permutation(dataset.target) for _ in range(int(n_replicates))]


def feature_names(dataset: MetaLabelDataset) -> Sequence[str]:
    return list(dataset.features.columns)
