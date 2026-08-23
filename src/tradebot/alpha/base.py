# src/tradebot/alpha/base.py
"""AlphaSignal protocol + SignalResult dataclass.

Every signal in the alpha library must implement this interface so the
ICWeightedCombiner and research_harness can treat them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = ["SignalResult", "AlphaSignal"]


@dataclass(frozen=True)
class SignalResult:
    """Single-bar prediction from one alpha signal.

    Attributes
    ----------
    symbol :
        Ticker string (e.g. ``"BTCUSDT"``).
    timestamp :
        Bar close time (UTC-aware).
    signal :
        Normalised signal in [-1, +1].  +1 = maximum long conviction.
    confidence :
        Calibrated probability in [0, 1].  0.5 = no view.
    horizon_bars :
        Number of forward bars this signal targets.
    signal_id :
        Reproducible identifier: ``f"{symbol}_{algo}_{params_hash}"``.
    """

    symbol: str
    timestamp: pd.Timestamp
    signal: float
    confidence: float
    horizon_bars: int
    signal_id: str


@runtime_checkable
class AlphaSignal(Protocol):
    """Protocol every alpha signal must satisfy."""

    signal_id: str

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the signal on historical bars ``df``.

        ``df`` must have a UTC-aware DatetimeIndex and contain at minimum
        ``open``, ``high``, ``low``, ``close``, ``volume`` columns.
        All data in ``df`` is treated as in-sample — no lookahead possible
        at fit time (the last row represents bar t-1 relative to live bar t).
        """
        ...

    def predict(self, df: pd.DataFrame) -> SignalResult:
        """Predict on the LATEST bar of ``df``.

        Only the most-recent row is used for the live signal.  Historical
        rows in ``df`` are used solely for rolling/stateful computations.
        Must be called after ``fit``.
        """
        ...

    def feature_names(self) -> list[str]:
        """Return the list of feature names consumed by this signal."""
        ...


# =========================================================================== #
# PHASE 3 - L4 ALPHA UNIT CONTRACT
# =========================================================================== #
# Alles BOVEN deze regel is het legacy `AlphaSignal`-protocol: een per-bar
# `predict()` die naast het signaal ook een `confidence` en een `horizon_bars`
# teruggeeft. Zesentwintig bestaande units hangen eraan; die code is hier
# bewust ONGEWIJZIGD gelaten en wordt in Phase 5/6 herzien.
#
# Alles HIERONDER is het formele L4-contract uit audit sectie 11.1. Het verschil
# is niet cosmetisch:
#
#   * een `AlphaUnit` levert UITSLUITEND `a_t in [-1, +1]` - de GEWENSTE
#     exposure. Geen positiegrootte, geen leverage, geen stop-loss, geen
#     orderlogica. Wat er met `a_t` gebeurt, is de zaak van L7 (risk) en L8
#     (portfolio), en die twee lagen zijn soeverein.
#   * de input is een GECERTIFICEERD feature-artefact dat zijn eigen
#     `data_hash` draagt. Een unit kan dus niet op ongecertificeerde data
#     draaien zonder dat dat in het resultaat zichtbaar is.
#   * de scheiding wordt niet gevraagd maar AFGEDWONGEN. `__init_subclass__`
#     scant de AST van de module waarin de unit is gedefinieerd en crasht bij
#     elke import uit `risk/`, `portfolio/`, `execution/` of `oms/`.
#
# Waarom een AST-scan en niet een afspraak? Omdat risicologica die in Phase 3 in
# een alpha-unit sluipt, in Phase 4 niet meer wordt teruggevonden: hij ziet er
# tegen die tijd uit als onderdeel van het signaal.
# --------------------------------------------------------------------------- #
import abc as _abc
import ast as _ast
import sys as _sys
from collections.abc import Mapping as _Mapping
from pathlib import Path as _Path
from types import MappingProxyType as _MappingProxyType
from typing import Any as _Any
from typing import ClassVar as _ClassVar

import numpy as _np

from ..utils.failfast import DataContractError as _DataContractError
from ..utils.failfast import require as _require

#: Lagen waaruit `alpha/` NOOIT mag importeren (audit sectie 11.1).
FORBIDDEN_LAYERS: tuple[str, ...] = ("risk", "portfolio", "execution", "oms")

#: Het pakket waarbinnen de scheiding geldt.
_ALPHA_PACKAGE = "tradebot.alpha"


def _imported_layers(source: str, filename: str) -> set[str]:
    """De tradebot-lagen waaruit `source` importeert, absoluut en relatief."""
    tree = _ast.parse(source, filename=filename)
    found: set[str] = set()

    def _record(dotted: str) -> None:
        parts = [p for p in dotted.split(".") if p]
        if not parts:
            return
        # `tradebot.risk.limits` -> risk ; `risk.limits` (relatief) -> risk
        head = parts[1] if parts[0] == "tradebot" and len(parts) > 1 else parts[0]
        if head in FORBIDDEN_LAYERS:
            found.add(head)

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                _record(alias.name)
        elif isinstance(node, _ast.ImportFrom):
            _record(node.module or "")
    return found


def assert_alpha_isolation(module_name: str) -> None:
    """Crash wanneer een alpha-module uit een verboden laag importeert.

    Statisch, op de AST: een `import` binnen een functie of achter een `if`
    telt net zo goed mee. Dat is opzet - een conditionele import van `risk/` is
    geen uitzondering maar precies de manier waarop de scheiding in de praktijk
    verwatert.
    """
    module = _sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    _require(
        path is not None,
        "Kan de bron van deze alpha-module niet vinden; de laagscheiding is "
        "dan niet statisch te bewijzen.",
        _DataContractError,
        module=module_name,
    )
    assert path is not None
    source = _Path(path).read_text(encoding="utf-8")
    offending = sorted(_imported_layers(source, path))
    _require(
        not offending,
        "Een alpha-module importeert uit een verboden laag. Alpha kent geen "
        "risico en geen executie (audit sectie 11.1): een unit levert a_t en "
        "verder niets. Risicologica die hier binnensluipt, wordt in Phase 4 "
        "niet meer teruggevonden.",
        _DataContractError,
        module=module_name,
        forbidden=offending,
        allowed_layers_excluded=list(FORBIDDEN_LAYERS),
    )


@dataclass(frozen=True)
class AlphaOutput:
    """De ENIGE toegestane output van een AlphaUnit.

    `exposures` is de gewenste exposure `a_t in [-1, +1]` per symbool, met een
    `asof_ts`-index. Er staat bewust geen positiegrootte, geen notional en geen
    hefboom in: die bestaan pas nadat L7 en L8 hun zegje hebben gedaan.
    """

    unit: str
    exposures: pd.DataFrame
    burn_in_period: int
    params: _Mapping[str, _Any]
    data_hashes: tuple[tuple[str, str], ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.exposures.columns)

    def drop_burn_in(self) -> pd.DataFrame:
        return self.exposures.iloc[self.burn_in_period :]


class AlphaUnit(_abc.ABC):
    """Het formele L4-contract. Een subklasse levert `a_t` en verder niets.

    Subklassen implementeren `_generate`; `generate` is het contract en wordt
    niet overschreven. De bereikcontrole gebruikt `signal_floor` en
    `signal_cap` uit `conf/model/alpha.yaml`, die bij constructie worden
    meegegeven en meegaan in de parameters van het resultaat.
    """

    name: _ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: _Any) -> None:
        super().__init_subclass__(**kwargs)
        # De scheiding wordt afgedwongen op het moment dat de klasse ontstaat,
        # dus bij import van de module. Een unit die uit risk/ importeert komt
        # niet eens tot een eerste aanroep.
        if cls.__module__.startswith(_ALPHA_PACKAGE):
            assert_alpha_isolation(cls.__module__)

    def __init__(
        self, *, signal_floor: float, signal_cap: float, params: _Mapping[str, _Any]
    ) -> None:
        _require(
            bool(type(self).name),
            "AlphaUnit zonder `name`; zonder stabiele identiteit is het "
            "resultaat niet in de ledger te plaatsen.",
            _DataContractError,
            cls=type(self).__name__,
        )
        _require(
            signal_cap > signal_floor,
            "signal_cap moet groter zijn dan signal_floor.",
            _DataContractError,
            signal_floor=signal_floor,
            signal_cap=signal_cap,
        )
        self._floor = float(signal_floor)
        self._cap = float(signal_cap)
        merged = dict(params)
        merged.update({"signal_floor": self._floor, "signal_cap": self._cap})
        self._params: _Mapping[str, _Any] = _MappingProxyType(merged)

    def __repr__(self) -> str:
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self._params.items()))
        return f"{type(self).__name__}({rendered})"

    @property
    def params(self) -> _Mapping[str, _Any]:
        return self._params

    @property
    def unit_id(self) -> str:
        rendered = ",".join(f"{k}={self._params[k]!r}" for k in sorted(self._params))
        return f"{self.name}({rendered})"

    @property
    @_abc.abstractmethod
    def burn_in_period(self) -> int:
        """Aantal bars waarin er per constructie geen exposure bestaat."""

    @_abc.abstractmethod
    def _generate(self, features: pd.DataFrame) -> pd.DataFrame:
        """Map het feature-panel op de gewenste exposure. Pure functie."""

    def generate(self, features: _Any) -> AlphaOutput:
        """Bereken `a_t` en dwing het volledige L4-contract af."""
        _require(
            hasattr(features, "values") and hasattr(features, "data_hashes"),
            "generate() eist een gecertificeerd feature-artefact met een "
            "`data_hash`. Een kaal DataFrame draagt geen provenance, en een "
            "alpha-resultaat zonder provenance is INVALID (sectie 7.2).",
            _DataContractError,
            unit=self.name,
            got=type(features).__name__,
        )
        panel = features.values
        exposures = self._generate(panel)
        self._validate(panel, exposures)
        return AlphaOutput(
            unit=self.name,
            exposures=exposures,
            burn_in_period=int(self.burn_in_period),
            params=self._params,
            data_hashes=tuple(features.data_hashes),
        )

    def _validate(self, panel: pd.DataFrame, exposures: object) -> None:
        _require(
            isinstance(exposures, pd.DataFrame),
            "_generate moet een DataFrame teruggeven.",
            _DataContractError,
            unit=self.name,
            got=type(exposures).__name__,
        )
        assert isinstance(exposures, pd.DataFrame)
        _require(
            bool(exposures.index.equals(panel.index)),
            "De exposure-index wijkt af van het feature-panel; een alpha-unit "
            "mag rijen niet herordenen, toevoegen of laten vallen.",
            _DataContractError,
            unit=self.name,
        )
        _require(
            list(exposures.columns) == list(panel.columns),
            "Het universum is in de alpha-unit gewijzigd.",
            _DataContractError,
            unit=self.name,
        )
        for col in exposures.columns:
            _require(
                exposures[col].dtype == _np.dtype("float64"),
                "Elke exposure-kolom is float64.",
                _DataContractError,
                unit=self.name,
                column=str(col),
                dtype=str(exposures[col].dtype),
            )
        arr = exposures.to_numpy(dtype="float64")
        _require(
            not bool(_np.isinf(arr).any()),
            "De unit produceerde plus/min oneindig als exposure.",
            _DataContractError,
            unit=self.name,
        )
        finite = arr[_np.isfinite(arr)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            _require(
                lo >= self._floor and hi <= self._cap,
                "Een exposure valt BUITEN het toegestane bereik. Een alpha-unit "
                "die buiten [-1, +1] komt, spreekt over positiegrootte in plaats "
                "van over richting en overtreedt daarmee sectie 11.1.",
                _DataContractError,
                unit=self.name,
                observed_min=lo,
                observed_max=hi,
                floor=self._floor,
                cap=self._cap,
            )
        burn_in = int(self.burn_in_period)
        head = arr[: min(burn_in, len(arr))]
        _require(
            (int(_np.isfinite(head).sum()) if head.size else 0) == 0,
            "Er bestaat een exposure binnen de burn-in van de unit. Een unit die "
            "handelt voordat zijn features bestaan, handelt op een opgevulde "
            "waarde.",
            _DataContractError,
            unit=self.name,
            burn_in=burn_in,
        )
        # Missing-policy: NaN NA de burn-in mag uitsluitend waar de FEATURE zelf
        # ontbreekt. Een unit die zijn eigen gaten maakt, verplaatst het
        # invulbesluit naar een consument die de oorzaak niet kent.
        feature_nan = panel.isna().to_numpy()
        exposure_nan = _np.isnan(arr)
        offending = exposure_nan & ~feature_nan
        offending[: min(burn_in, len(offending))] = False
        _require(
            not bool(offending.any()),
            "NaN in de exposure na de burn-in zonder ontbrekende feature. Er is "
            "geen impliciete missing-policy: ontbreekt een feature, dan is er "
            "geen view, en dat wordt doorgegeven in plaats van ingevuld.",
            _DataContractError,
            unit=self.name,
            n_offending=int(offending.sum()),
        )
