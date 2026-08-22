"""tradebot — medium-frequency crypto market-making research stack.

Package layout: src-layout (PEP 621), strangler-fig refactor v2.

Sub-packages
------------
bars            : volume/dollar/tick/runs imbalance bar generators
cv              : CPCV, sequential bootstrap, walk-forward CV
data            : ingestion, macro features, funding rates
execution       : market impact, slippage, fees
features        : FeatureEngineer, FeatureOrthogonalizer, Numba kernels
labeling        : triple-barrier, meta-labeling, fixed-horizon
monitoring      : drift detection, feature health, calibration monitoring, alerts
registry        : model catalog, lineage, promotion
reporting       : AFML-style tearsheet generation
risk            : portfolio risk manager, Kelly sizing, VaR, drawdown breaker
schemas         : Pandera DataFrame schemas (stage boundaries)
train           : CatBoost agents, bandit ensemble, calibration, checkpoints
tune            : Optuna sampler/storage factories
volatility      : Garman-Klass, Parkinson, EWMA, jump-diffusion estimators

Do NOT import from train_regime.py in new code — use the sub-modules here.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("tradebot")
except PackageNotFoundError:
    __version__ = "0.4.0"  # fallback for editable installs without metadata

__version_tuple__: tuple[int, ...] = tuple(int(x) for x in __version__.split(".")[:3])
