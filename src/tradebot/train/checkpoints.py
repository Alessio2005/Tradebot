# src/tradebot/train/checkpoints.py
"""Per-fold model checkpoint save / restore for CPCV training.

Design
------
Each CPCV fold produces one checkpoint file:

    <checkpoint_dir>/<study_name>/fold_<fold_id>_<side>.joblib

The checkpoint contains a ``FoldCheckpoint`` dataclass with:
  - the trained model (RegimeCatAgent or CalibratedClassifierCV)
  - fold metadata (fold_id, side, oos_logloss, n_train, n_test)
  - a sha256 hash of the training feature names for lineage tracking

On resume, ``load_fold_checkpoint`` reconstructs the same object so
downstream CPCV path-reconstruction can re-use previously trained folds.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

__all__ = ["FoldCheckpoint", "list_checkpoints", "load_fold_checkpoint", "save_fold_checkpoint"]


def _feature_hash(feature_names: list[str]) -> str:
    """SHA-256 fingerprint of an ordered list of feature names."""
    payload = json.dumps(feature_names, sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class FoldCheckpoint:
    """Serialisable container for one trained CPCV fold.

    Attributes
    ----------
    fold_id : int
        Zero-based fold index.
    side : str
        "LONG" or "SHORT".
    model : Any
        Trained model (RegimeCatAgent / CalibratedClassifierCV / etc.).
    feature_names : list[str]
        Ordered list of features the model was trained on.
    feature_hash : str
        SHA-256[:16] fingerprint of ``feature_names`` for lineage detection.
    oos_logloss : float
        Out-of-sample log-loss on the held-out test fold.
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    seed : int
        Random seed used for this fold.
    extra : dict
        Arbitrary extra metadata (e.g. optuna trial number, bar counts).
    """

    fold_id: int
    side: str
    model: Any
    feature_names: list[str] = field(default_factory=list)
    feature_hash: str = ""
    oos_logloss: float = float("nan")
    n_train: int = 0
    n_test: int = 0
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_hash and self.feature_names:
            self.feature_hash = _feature_hash(self.feature_names)

    def validate_features(self, feature_names: list[str]) -> None:
        """Raise ``ValueError`` if feature list differs from stored fingerprint."""
        incoming_hash = _feature_hash(feature_names)
        if incoming_hash != self.feature_hash:
            raise ValueError(
                f"Feature hash mismatch for fold {self.fold_id} — "
                f"stored={self.feature_hash}, incoming={incoming_hash}. "
                "The feature pipeline changed since this checkpoint was created."
            )


def _checkpoint_path(
    checkpoint_dir: Path,
    study_name: str,
    fold_id: int,
    side: str,
) -> Path:
    sub = checkpoint_dir / study_name
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"fold_{fold_id:04d}_{side.upper()}.joblib"


def save_fold_checkpoint(
    checkpoint: FoldCheckpoint,
    checkpoint_dir: str | Path,
    study_name: str = "default",
    compress: int = 3,
) -> Path:
    """Persist a ``FoldCheckpoint`` to disk via joblib.

    Parameters
    ----------
    checkpoint : FoldCheckpoint
        The fold to persist.
    checkpoint_dir : str | Path
        Root directory for checkpoints.
    study_name : str
        Sub-directory name (one per Optuna study / experiment run).
    compress : int
        joblib compression level (0=none, 9=max; default 3).

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    path = _checkpoint_path(Path(checkpoint_dir), study_name, checkpoint.fold_id, checkpoint.side)
    joblib.dump(checkpoint, path, compress=compress)
    logger.info(
        "Saved checkpoint: fold=%d side=%s logloss=%.4f -> %s",
        checkpoint.fold_id,
        checkpoint.side,
        checkpoint.oos_logloss,
        path,
    )
    return path


def load_fold_checkpoint(
    fold_id: int,
    side: str,
    checkpoint_dir: str | Path,
    study_name: str = "default",
) -> FoldCheckpoint:
    """Load a ``FoldCheckpoint`` from disk.

    Parameters
    ----------
    fold_id : int
        Zero-based fold index.
    side : str
        "LONG" or "SHORT".
    checkpoint_dir : str | Path
        Root checkpoint directory.
    study_name : str
        Sub-directory name.

    Returns
    -------
    FoldCheckpoint

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    path = _checkpoint_path(Path(checkpoint_dir), study_name, fold_id, side)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: fold={fold_id}, side={side}, path={path}"
        )
    ckpt: FoldCheckpoint = joblib.load(path)
    logger.info(
        "Loaded checkpoint: fold=%d side=%s logloss=%.4f from %s",
        ckpt.fold_id,
        ckpt.side,
        ckpt.oos_logloss,
        path,
    )
    return ckpt


def list_checkpoints(
    checkpoint_dir: str | Path,
    study_name: str = "default",
    side: str | None = None,
) -> list[Path]:
    """List all checkpoint files for a study, optionally filtered by side.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Root checkpoint directory.
    study_name : str
        Sub-directory name.
    side : str | None
        Filter to "LONG" or "SHORT"; ``None`` returns all.

    Returns
    -------
    list[Path]
        Sorted list of checkpoint paths.
    """
    sub = Path(checkpoint_dir) / study_name
    if not sub.exists():
        return []
    pattern = f"fold_*_{side.upper()}.joblib" if side else "fold_*.joblib"
    return sorted(sub.glob(pattern))
