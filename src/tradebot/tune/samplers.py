"""Optuna sampler factory — TPE (default), GP, Random, QMC.

Centralises sampler construction so that apps/tune_hparams.py
only needs to pass a config flag, not construct samplers directly.
"""
from __future__ import annotations

import logging
from typing import Literal

import optuna

from ..utils.failfast import ConfigContractError, require

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
    # Phase 0: beide takken vielen bij een oudere Optuna-versie stilzwijgend
    # terug op TPE. De config vroeg om `gp` of `qmc` en kreeg TPE - een andere
    # zoekstrategie, dus een andere trial-verdeling, terwijl elk rapport de
    # geconfigureerde sampler noemde. De trial-teller (M) werd daarmee ook aan de
    # verkeerde sampler toegeschreven.
    if name == "gp":
        require(
            hasattr(optuna.samplers, "GPSampler"),
            "conf vraagt sampler='gp' maar deze Optuna-versie kent geen "
            "GPSampler. Er wordt NIET stilzwijgend op TPE teruggevallen: "
            "kies expliciet een andere sampler of upgrade Optuna.",
            ConfigContractError,
            optuna_version=optuna.__version__,
        )
        return optuna.samplers.GPSampler(seed=seed)
    if name == "qmc":
        require(
            hasattr(optuna.samplers, "QMCSampler"),
            "conf vraagt sampler='qmc' maar deze Optuna-versie kent geen "
            "QMCSampler. Er wordt NIET stilzwijgend op TPE teruggevallen.",
            ConfigContractError,
            optuna_version=optuna.__version__,
        )
        return optuna.samplers.QMCSampler(seed=seed)
    if name == "nsgaii":
        return optuna.samplers.NSGAIISampler(seed=seed)
    # default: random
    return optuna.samplers.RandomSampler(seed=seed)
