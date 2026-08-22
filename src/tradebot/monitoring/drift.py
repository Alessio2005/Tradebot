# src/tradebot/monitoring/drift.py
"""Feature distribution drift detection — PSI, KS test, Wasserstein distance.

Drift signals (from blueprint §8.4):
  PSI > 0.2      → critical drift (model retraining required)
  PSI 0.1–0.2   → moderate drift (monitor closely)
  PSI < 0.1      → stable

Minimum-sample gate: a PSI computed on < 50 observations is noisy and
unreliable.  The guard returns ``nan`` rather than a false alert.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PSI_CRITICAL",
    "PSI_MODERATE",
    "DriftResult",
    "ReferenceStability",
    "assess_reference_stability",
    "check_feature_drift",
    "ks_drift",
    "psi",
    "wasserstein_drift",
]

PSI_MODERATE: float = 0.10
PSI_CRITICAL: float = 0.20
_MIN_SAMPLES: int = 50

# CHIEF AUDIT-FIX (Sim-to-Reality #16):
#   Drift PSI uses ``reference`` bins as the baseline distribution.  If the
#   reference window itself spans a regime shift (e.g. May-Nov 2022 spans
#   the LUNA collapse, FTX collapse, and the subsequent capitulation low),
#   the baseline is bimodal/multimodal and *anything* in production looks
#   non-drifted relative to that wide envelope.  We add an internal split
#   test: divide the reference into halves, compute the PSI between the
#   halves, and flag references with intra-window PSI > critical.
#   These references should NOT be used until a regime-stable subset is
#   selected.
REFERENCE_INTRA_PSI_LIMIT: float = PSI_CRITICAL


@dataclass
class ReferenceStability:
    """Diagnostic for the train/reference window itself.

    Attributes
    ----------
    feature : feature name.
    intra_psi : PSI between the first and second half of the reference window.
    n : reference sample count.
    is_stable : True if the reference passes the intra-window PSI gate.
    """

    feature: str
    intra_psi: float
    n: int
    is_stable: bool


def assess_reference_stability(
    reference: dict[str, np.ndarray],
    n_bins: int = 10,
    limit: float = REFERENCE_INTRA_PSI_LIMIT,
) -> list[ReferenceStability]:
    """Detect regime-shift contamination in the reference window.

    CHIEF AUDIT-FIX (Sim-to-Reality #16):
      Splits each feature's reference into halves and computes PSI between
      them.  When intra-reference PSI exceeds ``limit`` the baseline is
      bimodal — any downstream drift detection against this reference is
      unreliable.  Call this BEFORE :func:`check_feature_drift` and refuse
      to monitor features that fail.

    Parameters
    ----------
    reference : dict {feature_name: values}.
    n_bins : PSI histogram bin count.
    limit : intra-window PSI threshold above which the reference is
        unstable.  Default = ``PSI_CRITICAL`` (0.20).

    Returns
    -------
    List of ReferenceStability results.
    """
    out: list[ReferenceStability] = []
    for feat, vals in reference.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n = arr.size
        if n < 2 * _MIN_SAMPLES:
            out.append(ReferenceStability(
                feature=feat,
                intra_psi=float("nan"),
                n=n,
                is_stable=False,  # not enough data → conservative fail
            ))
            continue
        half = n // 2
        intra = psi(arr[:half], arr[half:], n_bins=n_bins)
        is_stable = math.isfinite(intra) and intra < limit
        if not is_stable:
            logger.warning(
                "Reference unstable for %s: intra-window PSI=%.3f >= %.3f. "
                "Train/reference window contains a regime shift — drift "
                "detection against this baseline is unreliable. Re-select "
                "a homogeneous sub-window before monitoring.",
                feat, intra if math.isfinite(intra) else -1.0, limit,
            )
        out.append(ReferenceStability(
            feature=feat,
            intra_psi=float(intra),
            n=n,
            is_stable=is_stable,
        ))
    return out


@dataclass
class DriftResult:
    """Result of a single-feature drift test.

    Attributes
    ----------
    feature : feature name.
    psi_value : Population Stability Index (nan if insufficient data).
    ks_pvalue : Kolmogorov-Smirnov p-value (nan if insufficient data).
    wasserstein : normalised Wasserstein-1 distance.
    is_drifted : True if PSI >= PSI_MODERATE.
    is_critical : True if PSI >= PSI_CRITICAL.
    n_reference : number of samples in reference window.
    n_current : number of samples in current window.
    """

    feature: str
    psi_value: float
    ks_pvalue: float
    wasserstein: float
    is_drifted: bool
    is_critical: bool
    n_reference: int
    n_current: int


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two distributions.

    PSI = Σ (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

    Parameters
    ----------
    reference : 1-D array from the reference/baseline window.
    current : 1-D array from the current/monitoring window.
    n_bins : number of histogram buckets (default 10).

    Returns
    -------
    float : PSI value >= 0; nan if insufficient data.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if ref.size < _MIN_SAMPLES or cur.size < _MIN_SAMPLES:
        logger.debug("psi: insufficient data (ref=%d, cur=%d), returning nan.", ref.size, cur.size)
        return float("nan")

    # Build bins on reference distribution (causal: bins from ref only)
    _, bin_edges = np.histogram(ref, bins=n_bins)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    ref_counts = np.histogram(ref, bins=bin_edges)[0].astype(np.float64)
    cur_counts = np.histogram(cur, bins=bin_edges)[0].astype(np.float64)

    # Clip to avoid log(0): add epsilon
    eps = 1e-9
    ref_pct = (ref_counts + eps) / (ref.size + eps * n_bins)
    cur_pct = (cur_counts + eps) / (cur.size + eps * n_bins)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns (statistic, p_value).  Low p-value => distributions differ.
    """
    from scipy import stats  # lazy — scipy is a core dep

    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if ref.size < _MIN_SAMPLES or cur.size < _MIN_SAMPLES:
        return float("nan"), float("nan")

    stat, pval = stats.ks_2samp(ref, cur)
    return float(stat), float(pval)


def wasserstein_drift(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """Normalised Wasserstein-1 distance between two 1-D distributions.

    Normalised by the standard deviation of the reference so that the
    value is scale-invariant.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if ref.size < _MIN_SAMPLES or cur.size < _MIN_SAMPLES:
        return float("nan")

    from scipy.stats import wasserstein_distance  # lazy

    raw = float(wasserstein_distance(ref, cur))
    sigma = float(np.std(ref, ddof=1))
    return raw / max(sigma, 1e-12)


def check_feature_drift(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    n_bins: int = 10,
    assess_reference: bool = True,
) -> list[DriftResult]:
    """Run all drift tests for a set of features.

    Parameters
    ----------
    reference : dict {feature_name: values} for the baseline window.
    current : dict {feature_name: values} for the current window.
    n_bins : PSI bin count.
    assess_reference : when True, calls :func:`assess_reference_stability`
        first and logs a warning for each feature whose reference window
        contains an internal regime shift.  CHIEF AUDIT-FIX (Sim-to-Reality
        #16).

    Returns
    -------
    list of DriftResult sorted by PSI descending.
    """
    if assess_reference:
        stability = assess_reference_stability(reference, n_bins=n_bins)
        unstable = [s.feature for s in stability if not s.is_stable]
        if unstable:
            logger.warning(
                "check_feature_drift: %d feature(s) have unstable reference "
                "windows — drift detection against these baselines is "
                "unreliable: %s",
                len(unstable), unstable[:10],
            )
    results: list[DriftResult] = []
    all_features = set(reference) | set(current)

    for feat in sorted(all_features):
        ref_vals = np.asarray(reference.get(feat, []), dtype=np.float64)
        cur_vals = np.asarray(current.get(feat, []), dtype=np.float64)

        psi_val = psi(ref_vals, cur_vals, n_bins=n_bins)
        _, ks_pval = ks_drift(ref_vals, cur_vals)
        w_dist = wasserstein_drift(ref_vals, cur_vals)

        is_drifted = math.isfinite(psi_val) and psi_val >= PSI_MODERATE
        is_critical = math.isfinite(psi_val) and psi_val >= PSI_CRITICAL

        if is_critical:
            logger.warning("CRITICAL drift: feature=%s PSI=%.4f", feat, psi_val)
        elif is_drifted:
            logger.info("Moderate drift: feature=%s PSI=%.4f", feat, psi_val)

        results.append(DriftResult(
            feature=feat,
            psi_value=psi_val,
            ks_pvalue=ks_pval,
            wasserstein=w_dist,
            is_drifted=is_drifted,
            is_critical=is_critical,
            n_reference=int(ref_vals.size),
            n_current=int(cur_vals.size),
        ))

    return sorted(results, key=lambda r: (-(r.psi_value if math.isfinite(r.psi_value) else -1)))
