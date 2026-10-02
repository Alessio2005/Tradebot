"""Lichte, niet-getunede secondary models met Platt-kalibratie (spec §8).

Twee modellen, instellingen vast in `conf/model/weekly_meta.yaml`:
een L2-logistische regressie en een ondiepe random forest met
`max_features=1` en `max_samples` = gemiddelde uniqueness (AFML hoofdstuk 4/6).
Het ensemble middelt hun GEKALIBREERDE kansen.

Kalibratie gebeurt op het laatste deel van het trainvenster, chronologisch en
gepurged: geen fitrij waarvan het label de kalibratieset in loopt. Een K-fold
kalibratie zou toekomst binnen het trainvenster gebruiken.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..schemas.weekly_meta import WeeklyMetaConfig
from ..utils.failfast import DataContractError, require
from .meta_label import MetaLabelDataset

__all__ = ["MODEL_KINDS", "CalibratedModel", "fit_light_model", "inner_calibration_split"]

ModelKind = Literal["logreg", "forest", "ensemble"]
MODEL_KINDS: tuple[ModelKind, ...] = ("logreg", "forest", "ensemble")
_EPS = 1e-6
_DAYS_PER_WEEK = 7.0


def inner_calibration_split(
    event_bar: np.ndarray, exit_bar: np.ndarray, *, fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(fit-, kalibratie-)posities binnen de trainrijen: laatste `fraction` op tijd, gepurged."""
    order = np.argsort(event_bar, kind="stable")
    n_cal = max(1, int(round(fraction * order.size)))
    cal = order[-n_cal:]
    cal_start = int(event_bar[cal].min())
    fit = order[:-n_cal]
    return fit[exit_bar[fit] < cal_start], cal


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


@dataclass
class _Platt:
    model: LogisticRegression

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _fit_platt(p_raw: np.ndarray, y: np.ndarray, w: np.ndarray) -> _Platt:
    require(np.unique(y).size == 2,
            "Een kalibratieset met één klasse; er valt geen Platt-schaling te schatten.",
            DataContractError)
    model = LogisticRegression(C=1e6, max_iter=1000)  # praktisch ongestraft
    model.fit(_logit(p_raw).reshape(-1, 1), y, sample_weight=w)
    return _Platt(model)


@dataclass
class CalibratedModel:
    kind: ModelKind
    members: list[Any]
    calibrators: list[_Platt]
    feature_importance: np.ndarray
    fold_extras: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = [cal(m.predict_proba(x)[:, 1])
                 for m, cal in zip(self.members, self.calibrators, strict=True)]
        p = np.mean(probs, axis=0)
        return np.column_stack([1.0 - p, p])


def _fit_member(kind: str, x: np.ndarray, y: np.ndarray, w: np.ndarray,
                cfg: WeeklyMetaConfig) -> tuple[Any, np.ndarray]:
    if kind == "logreg":
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=cfg.logreg_c, max_iter=2000))
        model.fit(x, y, logisticregression__sample_weight=w)
        coef = np.abs(model.named_steps["logisticregression"].coef_[0])
        return model, coef / coef.sum()
    model = RandomForestClassifier(
        n_estimators=cfg.forest_n_estimators, max_depth=cfg.forest_max_depth,
        max_features=1, min_samples_leaf=cfg.forest_min_samples_leaf,
        max_samples=float(np.clip(w.mean(), 0.05, 1.0)),
        class_weight="balanced_subsample", bootstrap=True,
        random_state=cfg.seed, n_jobs=1)
    model.fit(x, y, sample_weight=w)
    return model, np.asarray(model.feature_importances_, dtype=np.float64)


def fit_light_model(
    dataset: MetaLabelDataset, rows: np.ndarray, labels: np.ndarray,
    kind: ModelKind, cfg: WeeklyMetaConfig,
) -> CalibratedModel:
    """Fit op `rows`, kalibreer op hun laatste deel; ziet niets buiten `rows`."""
    require(kind in MODEL_KINDS, "Onbekend modeltype.", DataContractError, kind=kind)
    rows = np.asarray(rows, dtype=np.int64)
    fit_pos, cal_pos = inner_calibration_split(
        dataset.event_bar[rows], dataset.exit_bar[rows], fraction=cfg.calibration_fraction)
    fit_rows, cal_rows = rows[fit_pos], rows[cal_pos]
    x = dataset.features.to_numpy(dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    w = dataset.uniqueness
    require(np.unique(y[fit_rows]).size == 2,
            "Een fitvenster met één klasse.", DataContractError)
    members, cals, imps = [], [], []
    for member in (("logreg", "forest") if kind == "ensemble" else (kind,)):
        model, imp = _fit_member(member, x[fit_rows], y[fit_rows], w[fit_rows], cfg)
        cals.append(_fit_platt(model.predict_proba(x[cal_rows])[:, 1], y[cal_rows], w[cal_rows]))
        members.append(model)
        imps.append(imp)
    out = CalibratedModel(kind, members, cals, np.mean(imps, axis=0))
    ev = dataset.event_bar[rows]
    weeks = (int(ev.max()) - int(ev.min()) + 1) / _DAYS_PER_WEEK
    out.fold_extras = {
        "calibration_probability": out.predict_proba(x[cal_rows])[:, 1],
        "calibration_target": y[cal_rows],
        "train_events_per_week": float(rows.size / weeks),
        "n_fit": int(fit_rows.size),
        "n_calibration": int(cal_rows.size),
    }
    return out
