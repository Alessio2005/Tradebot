"""Optuna sampler factory — TPE (default), GP, Random, QMC.

Centralises sampler construction so that apps/tune_hparams.py
only needs to pass a config flag, not construct samplers directly.
"""
from __future__ import annotations

import logging
from typing import Literal

import optuna

logger = logging.getLogger(__name__)

SamplerName = Literal["tpe", "gp", "random", "qmc", "nsgaii"]


def make_sampler(
    name: SamplerName = "tpe",
    seed: int = 42,
    n_startup_trials: int = 10,
    multivariate: bool = True,
) -> optuna.samplers.BaseSampler:
    """Create an Optuna sampler by name.

    Args:
        name              : Sampler type.
        seed              : Random seed for reproducibility.
        n_startup_trials  : Random exploration phase before model kicks in.
        multivariate      : TPE multivariate mode (considers parameter correlations).

    Returns:
        Configured Optuna sampler instance.
    """
    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            multivariate=multivariate,
        )
    if name == "gp":
        try:
            return optuna.samplers.GPSampler(seed=seed)
        except AttributeError:
            logger.warning("GPSampler not available in this Optuna version; falling back to TPE.")
            return make_sampler("tpe", seed=seed)
    if name == "qmc":
        try:
            return optuna.samplers.QMCSampler(seed=seed)
        except AttributeError:
            logger.warning("QMCSampler not available; falling back to TPE.")
            return make_sampler("tpe", seed=seed)
    if name == "nsgaii":
        return optuna.samplers.NSGAIISampler(seed=seed)
    # default: random
    return optuna.samplers.RandomSampler(seed=seed)
