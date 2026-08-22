# src/tradebot/monitoring/live_drift_monitor.py
"""F4 — Live feature-drift monitor (PSI vs training reference).

Loaded once per symbol at engine startup; samples the training features
parquet (artefacts/features/{SYM}.parquet) to build a reference distribution
for every feat_* column.  On each FeaturePipeline refresh the latest
``current_window`` rows of the live features are compared against this
reference via :py:func:`tradebot.monitoring.drift.check_feature_drift`.

Output:
  - WARNING log line per refresh listing top-N PSI features above threshold
  - artefacts/paper_trade/drift_report.jsonl (one JSON per refresh)
  - prometheus gauge feature_drift_psi_max{symbol="..."}

Notes:
  - PSI < 0.1 stable, 0.1-0.2 moderate, > 0.2 critical retraining signal.
  - Reference window = last ``ref_window`` rows of training parquet.
  - Current window = trailing ``current_window`` rows from live refresh.
  - Drift checks are EXPENSIVE (~100ms for 100 features) — only run every
    ``check_every_n`` refreshes.  Default check_every_n=1 (every ~45 min).
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import pandas as pd

from .drift import PSI_CRITICAL, PSI_MODERATE, check_feature_drift

logger = logging.getLogger(__name__)

__all__ = ["LiveDriftMonitor", "LiveDriftMonitorConfig"]


class LiveDriftMonitorConfig:
    """Configuration for the live drift monitor.

    Parameters
    ----------
    artefacts_dir :
        Root where ``features/{SYM}.parquet`` reference and the output
        ``drift_report.jsonl`` live.
    ref_window :
        Number of rows from the tail of training parquet used as reference.
        Default 2000 ≈ 2-3 months of micro-bar data.
    current_window :
        Maximum current-window rows aggregated across refreshes before
        comparison.  Default 100 ≈ 4-5 micro bars of recent live state.
    check_every_n :
        Run the PSI check every N refreshes (1 = every refresh).
    drift_report_path :
        JSONL file written one record per check; None = disabled.
    top_n_logged :
        Number of worst-PSI features to include in the per-refresh log line.
    """

    def __init__(
        self,
        artefacts_dir: Path,
        ref_window: int = 2000,
        current_window: int = 100,
        check_every_n: int = 1,
        drift_report_path: Optional[Path] = None,
        top_n_logged: int = 10,
    ) -> None:
        self.artefacts_dir = Path(artefacts_dir)
        self.ref_window = int(ref_window)
        self.current_window = int(current_window)
        self.check_every_n = max(1, int(check_every_n))
        self.drift_report_path = drift_report_path
        self.top_n_logged = int(top_n_logged)


class LiveDriftMonitor:
    """Per-symbol reference distribution + per-refresh PSI check."""

    def __init__(self, config: LiveDriftMonitorConfig, symbols: list[str]) -> None:
        self._cfg = config
        self._refresh_counts: Dict[str, int] = {s: 0 for s in symbols}
        self._reference: Dict[str, Dict[str, np.ndarray]] = {}
        # Buffer of recent live feature rows per symbol (deque of dicts).
        self._current: Dict[str, Deque[pd.Series]] = {
            s: deque(maxlen=config.current_window) for s in symbols
        }
        for sym in symbols:
            self._load_reference(sym)

    # ------------------------------------------------------------------
    # Reference loading
    # ------------------------------------------------------------------

    def _load_reference(self, symbol: str) -> None:
        """Load and tail-trim training features parquet for ``symbol``."""
        ref_path = self._cfg.artefacts_dir / "features" / f"{symbol}.parquet"
        if not ref_path.exists():
            logger.warning(
                "LiveDriftMonitor [%s]: reference parquet not found at %s — "
                "drift monitoring disabled for this symbol.",
                symbol, ref_path,
            )
            self._reference[symbol] = {}
            return
        df = pd.read_parquet(ref_path)
        df = df.tail(self._cfg.ref_window)
        ref: Dict[str, np.ndarray] = {}
        for col in df.columns:
            if not col.startswith("feat_"):
                continue
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 50:  # _MIN_SAMPLES in drift module
                ref[col] = arr
        self._reference[symbol] = ref
        logger.info(
            "LiveDriftMonitor [%s]: reference loaded — %d feat_* columns "
            "(ref_window=%d rows).",
            symbol, len(ref), self._cfg.ref_window,
        )

    # ------------------------------------------------------------------
    # Per-refresh observe + (periodically) compare
    # ------------------------------------------------------------------

    def observe(self, symbol: str, feature_row: pd.DataFrame) -> None:
        """Append the last row of ``feature_row`` to the current buffer
        and trigger a drift check if the refresh counter modulo
        ``check_every_n`` reaches 0.
        """
        if symbol not in self._reference or not self._reference[symbol]:
            return
        if feature_row is None or feature_row.empty:
            return
        last = feature_row.iloc[-1]
        self._current[symbol].append(last)
        self._refresh_counts[symbol] += 1
        if self._refresh_counts[symbol] % self._cfg.check_every_n != 0:
            return
        if len(self._current[symbol]) < 50:
            return  # not enough live samples yet
        self._run_check(symbol)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _run_check(self, symbol: str) -> None:
        ref = self._reference[symbol]
        if not ref:
            return
        # Build current arrays per feature.
        cur_df = pd.DataFrame(list(self._current[symbol]))
        cur: Dict[str, np.ndarray] = {}
        for col in ref:
            if col not in cur_df.columns:
                continue
            arr = pd.to_numeric(cur_df[col], errors="coerce").to_numpy(dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 50:
                cur[col] = arr

        if not cur:
            return

        # Restrict reference to the feature set we actually have live.
        ref_subset = {k: v for k, v in ref.items() if k in cur}

        results = check_feature_drift(
            ref_subset, cur, n_bins=10, assess_reference=False,
        )

        # Filter on finite PSI; sort by descending PSI.
        finite = [r for r in results if np.isfinite(r.psi_value)]
        finite.sort(key=lambda r: r.psi_value, reverse=True)

        n_critical = sum(1 for r in finite if r.is_critical)
        n_moderate = sum(1 for r in finite if r.is_drifted and not r.is_critical)
        top = finite[: self._cfg.top_n_logged]
        top_str = ", ".join(f"{r.feature}={r.psi_value:.3f}" for r in top)

        if n_critical > 0:
            logger.warning(
                "LiveDriftMonitor [%s]: %d CRITICAL drifts (PSI >= %.2f), "
                "%d moderate. Top: %s",
                symbol, n_critical, PSI_CRITICAL, n_moderate, top_str,
            )
        elif n_moderate > 0:
            logger.info(
                "LiveDriftMonitor [%s]: %d moderate drifts (PSI >= %.2f). "
                "Top: %s",
                symbol, n_moderate, PSI_MODERATE, top_str,
            )
        else:
            logger.debug(
                "LiveDriftMonitor [%s]: no drift (max PSI=%.3f).",
                symbol, top[0].psi_value if top else 0.0,
            )

        # Emit machine-readable record (one line per check)
        if self._cfg.drift_report_path is not None:
            self._cfg.drift_report_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                "symbol": symbol,
                "n_features_checked": len(finite),
                "n_critical": n_critical,
                "n_moderate": n_moderate,
                "max_psi": float(top[0].psi_value) if top else 0.0,
                "top": [
                    {"f": r.feature, "psi": float(r.psi_value), "ks_p": float(r.ks_pvalue)}
                    for r in top
                ],
            }
            with open(self._cfg.drift_report_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
