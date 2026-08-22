# src/tradebot/registry/experiment.py
"""Experiment tracker — MLflow-adapter over ModelCatalog.

Logs per Optuna trial:
  - hparams dict
  - CV metrics (logloss, sharpe, DSR)
  - feature_hash, git_sha, dvc_hash
  - artifacts: model.joblib, calibrator.joblib

Compatible with MLflow UI AND standalone JSONL catalog.
No MLflow server required in CI (file://-backend).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ExperimentTracker", "TrialRecord"]

_JSONL_FILENAME = "experiment_log.jsonl"


class TrialRecord:
    """Lightweight record of one Optuna trial (MLflow run equivalent)."""

    __slots__ = (
        "run_id", "experiment_name", "trial_number",
        "params", "metrics", "tags", "artifact_paths",
        "start_time", "end_time", "status",
    )

    def __init__(
        self,
        run_id: str,
        experiment_name: str,
        trial_number: int,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        artifact_paths: list[str],
        start_time: float,
        end_time: float,
        status: str,
    ) -> None:
        self.run_id          = run_id
        self.experiment_name = experiment_name
        self.trial_number    = trial_number
        self.params          = params
        self.metrics         = metrics
        self.tags            = tags
        self.artifact_paths  = artifact_paths
        self.start_time      = start_time
        self.end_time        = end_time
        self.status          = status

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrialRecord":
        return cls(**d)


class ExperimentTracker:
    """MLflow-compatible experiment tracker backed by JSONL.

    Falls back to file-backed JSONL when MLflow is not installed.
    When MLflow IS installed, logs to ``file://`` URI so no server is needed.

    Parameters
    ----------
    experiment_name :
        Name for grouping runs (e.g. ``"BTCUSDT_cpcv"``).
    tracking_dir :
        Root directory for JSONL log and MLflow artifacts.
    use_mlflow :
        If True and MLflow is installed, also log to MLflow.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_dir: str | Path,
        use_mlflow: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_dir    = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path     = self.tracking_dir / _JSONL_FILENAME
        self._use_mlflow     = use_mlflow and self._try_import_mlflow()
        self._active_run: Optional[TrialRecord] = None

    # ------------------------------------------------------------------
    # MLflow opt-in
    # ------------------------------------------------------------------

    def _try_import_mlflow(self) -> bool:
        try:
            import mlflow  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            logger.debug("MLflow not installed — using JSONL backend only.")
            return False

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        trial_number: int,
        params: dict[str, Any],
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """Start a new trial run and return a unique run_id."""
        run_id = hashlib.md5(
            f"{self.experiment_name}_{trial_number}_{time.time()}".encode()
        ).hexdigest()[:12]

        self._active_run = TrialRecord(
            run_id=run_id,
            experiment_name=self.experiment_name,
            trial_number=trial_number,
            params=params,
            metrics={},
            tags=tags or {},
            artifact_paths=[],
            start_time=time.time(),
            end_time=0.0,
            status="RUNNING",
        )

        if self._use_mlflow:
            import mlflow
            mlflow.set_tracking_uri(f"file://{self.tracking_dir / 'mlruns'}")
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=f"trial_{trial_number}")
            mlflow.log_params(params)
            if tags:
                mlflow.set_tags(tags)

        return run_id

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        if self._active_run is None:
            raise RuntimeError("Call start_run() first.")
        self._active_run.metrics.update(metrics)
        if self._use_mlflow:
            import mlflow
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self._active_run is None:
            raise RuntimeError("Call start_run() first.")
        self._active_run.artifact_paths.append(str(path))
        if self._use_mlflow:
            import mlflow
            mlflow.log_artifact(str(path))

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run is None:
            return
        self._active_run.end_time = time.time()
        self._active_run.status   = status
        self._append_jsonl(self._active_run)
        if self._use_mlflow:
            import mlflow
            mlflow.end_run(status=status)
        self._active_run = None

    # ------------------------------------------------------------------
    # JSONL persistence
    # ------------------------------------------------------------------

    def _append_jsonl(self, record: TrialRecord) -> None:
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def load_runs(self) -> pd.DataFrame:
        """Load all trial records as a DataFrame."""
        if not self._jsonl_path.exists():
            return pd.DataFrame()
        records = []
        with self._jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def best_run(self, metric: str = "sharpe", higher_is_better: bool = True) -> Optional[TrialRecord]:
        """Return the TrialRecord with the best value of ``metric``."""
        df = self.load_runs()
        if df.empty:
            return None
        df = df[df["status"] == "FINISHED"]
        if df.empty:
            return None
        metric_vals = df["metrics"].apply(
            lambda m: m.get(metric, np.nan) if isinstance(m, dict) else np.nan
        )
        idx = metric_vals.idxmax() if higher_is_better else metric_vals.idxmin()
        return TrialRecord.from_dict(df.loc[idx].to_dict())
