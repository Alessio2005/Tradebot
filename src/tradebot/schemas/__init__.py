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

# Lui geladen (PEP 562, opruim-PR 2026-11-27): `import tradebot.schemas.x` laadt
# niet meer elke submodule, en dus niet meer zware deps (pandera, sklearn, hmmlearn) voor wie
# maar één module nodig heeft (de options-dienst). `from tradebot.schemas import Naam`
# werkt ongewijzigd: `__getattr__` importeert de submodule bij het eerste gebruik.
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bars import BarsSchema
    from .events import EventsSchema
    from .features import FeaturesSchema
    from .folds import FoldsSchema
    from .hparams import HParamsSchema
    from .labels import LabelsSchema
    from .oos_predictions import OOSPredictionSchema

#: publieke naam -> (submodule, naam in die submodule)
_LAZY: dict[str, tuple[str, str]] = {
    "BarsSchema": ("bars", "BarsSchema"),
    "EventsSchema": ("events", "EventsSchema"),
    "FeaturesSchema": ("features", "FeaturesSchema"),
    "FoldsSchema": ("folds", "FoldsSchema"),
    "HParamsSchema": ("hparams", "HParamsSchema"),
    "LabelsSchema": ("labels", "LabelsSchema"),
    "OOSPredictionSchema": ("oos_predictions", "OOSPredictionSchema"),
}

__all__ = [
    "BarsSchema",
    "EventsSchema",
    "FeaturesSchema",
    "FoldsSchema",
    "HParamsSchema",
    "LabelsSchema",
    "OOSPredictionSchema",
]


def __getattr__(name: str) -> Any:
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
