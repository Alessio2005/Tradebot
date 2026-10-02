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

# De testfolds zijn ~100x kleiner dan de echte: bij uniqueness 0,2 ziet elke boom ~58 rijen en
# kan hij met de productiewaarde `forest_min_samples_leaf: 50` niet splitsen (constante kans,
# AUC 0,51). Alleen deze testconfig krijgt daarom een kleinere leaf; `conf/model/weekly_meta.yaml`
# blijft zoals vastgelegd. Gemeten over seeds 3-7: leaf 5 geeft AUC 0,69-0,76, leaf 10 0,62-0,70.
CFG = weekly_meta_config().model_copy(
    update={"forest_n_estimators": 60, "forest_min_samples_leaf": 5})


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
