"""Domain-level NewTypes and Protocols for tradebot."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

BarTimestamp = pd.Timestamp
UTCTimestamp = pd.Timestamp
SideLiteral = str  # "LONG" | "SHORT"

@runtime_checkable
class HasOHLCV(Protocol):
    @property
    def columns(self) -> pd.Index: ...
    def __len__(self) -> int: ...

@runtime_checkable
class HasClose(Protocol):
    @property
    def index(self) -> pd.DatetimeIndex: ...
    def to_numpy(self, dtype: type = ...) -> np.ndarray: ...

FeatureMap = dict[str, list[str]]
