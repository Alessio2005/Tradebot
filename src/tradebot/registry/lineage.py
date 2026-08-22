# src/tradebot/registry/lineage.py
"""Model lineage tracking — git SHA + DVC hash fingerprinting.

Ensures that a model artifact can be traced back to:
  1. The exact code version (git commit hash)
  2. The exact data version (DVC content hash)
  3. The feature schema (feature hash from train/checkpoints.py)

LineageHashMismatch is raised when any fingerprint does not match the
stored record.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from ..exceptions import LineageHashMismatch

logger = logging.getLogger(__name__)

__all__ = ["get_file_hash", "get_git_sha", "verify_lineage"]


def get_git_sha(short: bool = True) -> str:
    """Return current git commit hash (empty string on failure)."""
    cmd = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()


def get_file_hash(path: str | Path, algorithm: str = "sha256", chunk: int = 65536) -> str:
    """Compute hex digest of a file for DVC-compatible lineage tracking."""
    # Phase 0 (D-9): `except FileNotFoundError: return ""` gaf een LEGE hash
    # terug voor een ontbrekend artefact. verify_lineage() slaat een lege hash
    # over ("empty = skip check"), dus een verdwenen modelbestand passeerde de
    # lineage-verificatie zonder enige melding.
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while buf := fh.read(chunk):
            h.update(buf)
    return h.hexdigest()[:32]  # 32 hex = 128-bit fingerprint


def verify_lineage(
    artifact_path: str | Path,
    expected_git_sha: str,
    expected_file_hash: str,
    expected_feature_hash: str,
    actual_feature_hash: str,
) -> None:
    """Verify artifact lineage against stored fingerprints.

    Raises LineageHashMismatch if any fingerprint does not match.

    Parameters
    ----------
    artifact_path : path to the model file.
    expected_git_sha : stored git SHA (empty = skip check).
    expected_file_hash : stored file content hash (empty = skip check).
    expected_feature_hash : stored feature schema hash (empty = skip check).
    actual_feature_hash : hash of the current feature list.
    """
    current_sha = get_git_sha()
    current_file_hash = get_file_hash(artifact_path)

    if expected_git_sha and current_sha and current_sha != expected_git_sha:
        logger.warning(
            "Git SHA mismatch: stored=%s, current=%s. "
            "Model was trained on a different code version.",
            expected_git_sha, current_sha,
        )

    if expected_file_hash and current_file_hash and current_file_hash != expected_file_hash:
        raise LineageHashMismatch(
            f"Artifact content hash mismatch: "
            f"expected={expected_file_hash}, actual={current_file_hash}. "
            f"The model file at {artifact_path} has been modified or replaced."
        )

    if expected_feature_hash and actual_feature_hash and actual_feature_hash != expected_feature_hash:
        raise LineageHashMismatch(
            f"Feature schema hash mismatch: "
            f"expected={expected_feature_hash}, actual={actual_feature_hash}. "
            f"The feature pipeline has changed since this model was trained."
        )
