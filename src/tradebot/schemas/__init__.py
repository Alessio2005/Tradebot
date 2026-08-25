"""Pandera schema contracts for every stage boundary.

PHASE 5: ``portfolio.py`` (``PortfolioResultSchema``) and ``tracks.py``
(``TrackSchema``) are gone. They described the stage boundary between the
per-symbol legacy engines and ``PortfolioBacktester``, and both sides of that
boundary were removed after the parity proof — see
``reports/phase5_engine_diff.md`` and ``backtest/__init__.py``.

The authoritative engine's own result contract is
``backtest.engine.BacktestResult``, and its per-bar state contract is
``backtest.accounting.LedgerSnapshot``. Both are frozen dataclasses with an
``as_record()``, so they need no separate pandera model.
"""
from .bars import BarsSchema
from .events import EventsSchema
from .features import FeaturesSchema
from .folds import FoldsSchema
from .hparams import HParamsSchema
from .labels import LabelsSchema
from .oos_predictions import OOSPredictionSchema

__all__ = [
    "BarsSchema",
    "EventsSchema",
    "FeaturesSchema",
    "FoldsSchema",
    "HParamsSchema",
    "LabelsSchema",
    "OOSPredictionSchema",
]
