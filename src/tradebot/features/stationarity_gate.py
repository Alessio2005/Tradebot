"""Stationarity gate for features (AUDIT A-3 / AFML §5.5).

Every column that enters the ML training loop has to be stationary in the
weak sense. ``check_feature_stationarity`` runs an Augmented Dickey-Fuller
test on each numeric feature column and returns the offenders. The build
pipeline raises (or downgrades to a warning) based on policy.

Why a gate, not just a metric:
  An I(1) feature inflates the t-statistic of every model that learns from
  it; the train-time loss looks great while the OOS Sharpe collapses on the
  first regime change. Catching I(1) leak at build-time stops that.

This module degrades gracefully when ``statsmodels`` is not installed: it
emits a warning and skips the check rather than failing the pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

# Phase 0: `try: from statsmodels... except ImportError` liet de ADF-gate
# ZICHZELF OVERSLAAN en gaf een lege StationarityReport terug waarin elke
# kolom als 'skipped' stond. Elke aanroeper zag een geslaagde gate. statsmodels
# is nu een harde dependency; de gate kan niet meer stilzwijgend uitvallen.
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


@dataclass
class StationarityReport:
    n_features_checked: int
    n_non_stationary:   int
    p_values:           dict[str, float] = field(default_factory=dict)
    non_stationary:     list[str] = field(default_factory=list)
    skipped:            list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.n_non_stationary == 0


def check_feature_stationarity(
    df: pd.DataFrame,
    feature_cols: Iterable[str] | None = None,
    p_threshold: float = 0.05,
    min_obs: int = 100,
    maxlag: int = 10,
    exclude_prefixes: tuple[str, ...] = ("ts", "timestamp", "fold_id",
                                          "label", "bin", "side", "y_"),
    drop_if_non_stationary: bool = False,
) -> tuple[pd.DataFrame, StationarityReport]:
    """Run ADF on every numeric feature column.

    Parameters
    ----------
    df :
        Feature matrix. The function only inspects numeric columns.
    feature_cols :
        Explicit subset to test. If ``None``, every numeric column not
        starting with one of ``exclude_prefixes`` is included.
    p_threshold :
        ADF rejection level. Features with p > threshold are flagged
        non-stationary.
    min_obs :
        Skip the column if non-NaN length is below this floor (too small
        to make a reliable inference).
    maxlag :
        Lag count passed to ``adfuller`` (the function uses
        ``autolag=None`` for determinism).
    exclude_prefixes :
        Identifier / target columns to skip.
    drop_if_non_stationary :
        When True, the offending columns are removed from the returned
        frame. When False (default), the frame is returned untouched and
        the caller decides what to do.

    Returns
    -------
    (df_out, report) — ``df_out`` either equals ``df`` or has the bad
    columns dropped (if ``drop_if_non_stationary``).
    """
    if feature_cols is None:
        candidate_cols = [
            c for c in df.columns
            if not any(c.startswith(p) for p in exclude_prefixes)
        ]
    else:
        candidate_cols = list(feature_cols)

    numeric_cols: list[str] = []
    for c in candidate_cols:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        numeric_cols.append(c)

    report = StationarityReport(
        n_features_checked=0,
        n_non_stationary=0,
    )

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < min_obs:
            report.skipped.append(col)
            continue
        if series.nunique() < 5:
            # Constants and quasi-constants are degenerate inputs to ADF.
            report.skipped.append(col)
            continue
        try:
            _, p_value, *_ = adfuller(
                series.astype(np.float64),
                maxlag=maxlag,
                regression="c",
                autolag=None,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("ADF failed on %s: %s", col, exc)
            report.skipped.append(col)
            continue
        # CHIEF AUDIT-FIX (Sim-to-Reality #17):
        #   ADF can return NaN when the series is near-constant but passes the
        #   nunique>=5 gate (e.g. a feature dominated by a few quantised values
        #   plus tiny float noise).  Comparing NaN > threshold is always False,
        #   so the feature passed through the gate silently — a classic
        #   non-stationary-feature-trained-as-stationary bug.  Treat NaN as
        #   "non-stationary unless proven otherwise" (conservative direction).
        p_value_f = float(p_value)
        if not np.isfinite(p_value_f):
            logger.warning(
                "stationarity_gate: ADF returned non-finite p-value for %s "
                "(near-constant after dedup) — treating as non-stationary.",
                col,
            )
            report.skipped.append(col)
            report.non_stationary.append(col)
            continue
        report.p_values[col] = p_value_f
        report.n_features_checked += 1
        if p_value_f > p_threshold:
            report.non_stationary.append(col)

    report.n_non_stationary = len(report.non_stationary)

    if report.n_non_stationary:
        logger.warning(
            "AUDIT A-3: %d non-stationary feature(s) detected (p > %.3f). "
            "Worst 5: %s",
            report.n_non_stationary,
            p_threshold,
            sorted(
                ((c, report.p_values[c]) for c in report.non_stationary),
                key=lambda kv: -kv[1],
            )[:5],
        )
    else:
        logger.info(
            "AUDIT A-3: stationarity gate PASS — %d feature(s) tested.",
            report.n_features_checked,
        )

    if drop_if_non_stationary and report.non_stationary:
        df_out = df.drop(columns=report.non_stationary)
    else:
        df_out = df
    return df_out, report


__all__ = ["check_feature_stationarity", "StationarityReport"]
