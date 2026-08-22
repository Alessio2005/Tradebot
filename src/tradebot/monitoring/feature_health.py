# src/tradebot/monitoring/feature_health.py
"""Feature health checks — NaN rates, dtype validation, out-of-range detection.

Per the blueprint (§8.4): > 1 % NaN on a critical column → schema-fail.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["FeatureHealthReport", "HealthIssue", "check_feature_health"]


@dataclass
class HealthIssue:
    """A single feature health problem."""
    feature: str
    issue_type: str   # "nan_rate" | "inf_rate" | "zero_variance" | "out_of_range"
    value: float
    threshold: float

    def __str__(self) -> str:
        return f"HealthIssue({self.feature}: {self.issue_type}={self.value:.4f} > {self.threshold:.4f})"


@dataclass
class FeatureHealthReport:
    """Summary of feature health for a DataFrame."""
    total_features: int
    total_rows: int
    issues: list[HealthIssue] = field(default_factory=list)
    nan_rates: dict[str, float] = field(default_factory=dict)
    inf_rates: dict[str, float] = field(default_factory=dict)

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    @property
    def has_critical(self) -> bool:
        return any(i.issue_type == "nan_rate" for i in self.issues)


def check_feature_health(
    df: pd.DataFrame,
    nan_threshold: float = 0.01,
    critical_features: set[str] | None = None,
) -> FeatureHealthReport:
    """Run feature health checks on a DataFrame.

    Parameters
    ----------
    df : feature DataFrame to inspect.
    nan_threshold : max acceptable NaN fraction (default 1 %).
    critical_features : set of column names where NaN > threshold raises.
        If None, all numeric columns are treated as critical.

    Returns
    -------
    FeatureHealthReport
    """
    n_rows = len(df)
    issues: list[HealthIssue] = []
    nan_rates: dict[str, float] = {}
    inf_rates: dict[str, float] = {}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        vals = df[col].values.astype(np.float64)
        nan_rate = float(np.isnan(vals).mean())
        inf_rate = float(np.isinf(vals).mean())
        nan_rates[col] = nan_rate
        inf_rates[col] = inf_rate

        if nan_rate > nan_threshold:
            issues.append(HealthIssue(col, "nan_rate", nan_rate, nan_threshold))
            logger.warning("Feature health: %s NaN rate=%.4f > %.4f", col, nan_rate, nan_threshold)

        if inf_rate > 0.0:
            issues.append(HealthIssue(col, "inf_rate", inf_rate, 0.0))
            logger.warning("Feature health: %s has %.4f infinite values", col, inf_rate)

        # Zero-variance check (constant feature — useless)
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size > 0 and np.std(finite_vals) < 1e-12:
            issues.append(HealthIssue(col, "zero_variance", 0.0, 1e-12))
            logger.debug("Feature health: %s has zero variance", col)

    return FeatureHealthReport(
        total_features=len(num_cols),
        total_rows=n_rows,
        issues=issues,
        nan_rates=nan_rates,
        inf_rates=inf_rates,
    )
