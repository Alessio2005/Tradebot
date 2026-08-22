"""pruning.py — Optuna pruner configuration.

Centralises the MedianPruner settings so that tune_hparams.py and
tests share the same pruning regime without copy-pasting magic numbers.
"""
from __future__ import annotations

import optuna


def create_median_pruner(
    n_startup_trials: int = 20,
    n_warmup_steps: int = 10,
    interval_steps: int = 1,
) -> optuna.pruners.MedianPruner:
    """Return a MedianPruner suitable for CPCV-based HPO.

    Parameters
    ----------
    n_startup_trials:
        Number of trials to run before pruning starts (to bootstrap the
        median estimate).  CHIEF AUDIT 2026-05-23 (P-10): bumped van 5
        naar 20. Bij 200 trials betekent n_startup_trials=5 dat trials
        6-25 op een mediaan van slechts 5 samples worden afgesneden —
        de median is dan extreem high-variance en goede trials worden
        regelmatig vroegtijdig gesnoeid. 20 geeft een veel stabielere
        baseline (variance van de median ~4× kleiner) zonder noemenswaardig
        verlies aan totale compute (10% van een 200-trial budget).
    n_warmup_steps:
        Number of reporting steps within a trial before the pruner is
        allowed to fire.  Prevents premature pruning during early folds.
    interval_steps:
        Check the pruning condition every this many steps (default: every
        fold report).
    """
    return optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps,
    )


# Backward-compat alias (tune/__init__.py public API)
make_pruner = create_median_pruner
