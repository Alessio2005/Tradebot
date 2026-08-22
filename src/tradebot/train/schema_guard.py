# src/tradebot/train/schema_guard.py
"""Feature schema guard and entropy gate.

Extracted from ``quant_architect.py`` (audit items V.8 and V.9).

Contents:
    SchemaFingerprint   — immutable SHA-256 fingerprint of a feature pipeline
    FeatureSchemaGuard  — kill-switch against feature drift
    SchemaMismatchError — hard-fail signal for the orchestrator
    EntropyGateDecision — result dataclass for EntropyGate.evaluate()
    EntropyGate         — neutral-arm forcing when H(p) > τ
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("train.schema_guard")


# =============================================================================
# V.8 FEATURE SCHEMA GUARD  — SHA-256 kill-switch
# =============================================================================
@dataclass(frozen=True)
class SchemaFingerprint:
    """Immutable fingerprint of a feature pipeline."""

    sha256: str
    feature_count: int
    feature_names: tuple[str, ...]


class FeatureSchemaGuard:
    """Kill-switch against feature drift (audit V.8).

    When a new feature is added or the order changes in the data pipeline,
    column indices can subtly shift and the CatBoost agent silently receives
    wrong input. This guard locks the schema:

      1. ``stamp(feature_names)`` produces a SHA-256 hash + length.
      2. On every model-load we store that fingerprint in the artifact.
      3. ``check(live_feature_names)`` compares the live pipeline against the
         stored schema; a mismatch raises ``SchemaMismatchError`` → the
         orchestrator must immediately block all trading.

    The hash covers both the feature names and their **order** — two pipelines
    with identical names but different order produce different hashes, which is
    correct because CatBoost is position-dependent.
    """

    def __init__(self) -> None:
        self._stamped: SchemaFingerprint | None = None

    @staticmethod
    def fingerprint(feature_names: Sequence[str]) -> SchemaFingerprint:
        """Compute a deterministic fingerprint."""
        names = tuple(str(n) for n in feature_names)
        canonical = json.dumps(list(names), separators=(",", ":"), sort_keys=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return SchemaFingerprint(
            sha256=digest, feature_count=len(names), feature_names=names
        )

    def stamp(self, feature_names: Sequence[str]) -> SchemaFingerprint:
        """Store the current fingerprint (e.g. during train()-time)."""
        fp = self.fingerprint(feature_names)
        self._stamped = fp
        return fp

    def check(self, live_feature_names: Sequence[str]) -> None:
        """Verify that the live pipeline matches.

        Raises:
            SchemaMismatchError: on hash, count or name mismatch.
        """
        if self._stamped is None:
            raise SchemaMismatchError(
                "FeatureSchemaGuard: no schema stamped — call stamp() first."
            )
        live_fp = self.fingerprint(live_feature_names)
        if live_fp.sha256 != self._stamped.sha256:
            missing = set(self._stamped.feature_names) - set(live_fp.feature_names)
            extra = set(live_fp.feature_names) - set(self._stamped.feature_names)
            raise SchemaMismatchError(
                f"FeatureSchemaGuard: schema-hash mismatch.\n"
                f"  trained sha256 = {self._stamped.sha256[:16]}…\n"
                f"  live    sha256 = {live_fp.sha256[:16]}…\n"
                f"  trained_count={self._stamped.feature_count}, "
                f"live_count={live_fp.feature_count}\n"
                f"  missing (in live): {sorted(missing)[:5]}{'…' if len(missing) > 5 else ''}\n"
                f"  extra   (in live): {sorted(extra)[:5]}{'…' if len(extra) > 5 else ''}"
            )

    @property
    def stamped(self) -> SchemaFingerprint | None:
        return self._stamped


class SchemaMismatchError(RuntimeError):
    """Hard-fail signal — orchestrator must pause trading."""


# =============================================================================
# V.9 ENTROPY GATE  — neutral-arm forcing when H(p) > τ
# =============================================================================
@dataclass(frozen=True)
class EntropyGateDecision:
    """Result of :meth:`EntropyGate.evaluate`."""

    pass_through: bool   # True = signal passes; False = force NEUTRAL
    entropy: float
    threshold: float
    action: str          # "PASS", "NEUTRAL"


class EntropyGate:
    """Protection against low-edge regimes (audit V.9).

    When the CatBoost agent delivers its prediction with high entropy (model is
    effectively *undecided*), the Bandit is still inclined to bet on small noise
    differences. This gate forces the system to the **NEUTRAL arm** whenever
    ``H(p) > τ``::

        H(p) = -Σᵢ pᵢ · log(pᵢ)

    Threshold τ is **dynamic**: rolling X-percentile of the own entropy
    distribution. In calm regimes H is typically lower → the threshold lower too
    → more signals pass. In chaotic regimes the gate trims the top-N%.

    Args:
        history_size:    rolling window for threshold computation.
        tau_percentile:  percentile threshold (default 80% — top-20% blocked).
        absolute_floor:  lower bound on τ (e.g. log(2)/2 = 0.346 = ½ of max
                         entropy of a binary model).
    """

    def __init__(
        self,
        history_size: int = 500,
        tau_percentile: float = 80.0,
        absolute_floor: float | None = None,
    ) -> None:
        if not 50.0 <= tau_percentile <= 99.5:
            raise ValueError("tau_percentile must be in [50, 99.5].")
        self.history_size: int = int(max(history_size, 10))
        self.tau_percentile: float = float(tau_percentile)
        self.absolute_floor: float | None = (
            float(absolute_floor) if absolute_floor is not None else None
        )
        self._history: list[float] = []

    @staticmethod
    def shannon_entropy(probabilities: Sequence[float], eps: float = 1e-12) -> float:
        """Compute H(p) in nats.

        Robust against zero probabilities (clipped with eps) and unnormalised
        inputs (re-normalised before integration).
        """
        p = np.asarray(probabilities, dtype=np.float64).flatten()
        p = np.where(p > 0.0, p, 0.0)
        s = float(p.sum())
        if s <= 0.0:
            return 0.0
        p = p / s
        p_safe = np.clip(p, eps, 1.0)
        return float(-np.sum(p_safe * np.log(p_safe)))

    def _current_threshold(self) -> float:
        """Threshold τ — rolling percentile or absolute floor."""
        if len(self._history) < 30:
            if self.absolute_floor is not None:
                return self.absolute_floor
            return float(math.log(2.0))  # max entropy of a binary model = 0.693
        tau = float(np.percentile(self._history, self.tau_percentile))
        if self.absolute_floor is not None:
            tau = max(tau, self.absolute_floor)
        return tau

    def evaluate(self, probabilities: Sequence[float]) -> EntropyGateDecision:
        """Decide pass-through vs. neutral.

        Args:
            probabilities: probability vector p (e.g. [prob_loss, prob_win]).

        Returns:
            ``EntropyGateDecision`` — caller checks ``.pass_through``.
        """
        h = self.shannon_entropy(probabilities)
        tau = self._current_threshold()

        # Add the OBSERVATION after threshold determination — prevents circularity.
        self._history.append(h)
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]

        passed = h <= tau
        return EntropyGateDecision(
            pass_through=bool(passed),
            entropy=float(h),
            threshold=float(tau),
            action="PASS" if passed else "NEUTRAL",
        )


__all__ = [
    "EntropyGate",
    "EntropyGateDecision",
    "FeatureSchemaGuard",
    "SchemaFingerprint",
    "SchemaMismatchError",
]
