"""Exitposities in eventruimte voor `CombinatorialPurgedCV.split`.

De splitter verwacht `t1` als POSITIE in de eventreeks, niet als bar. Voor een
op tijd gesorteerd paneel is dat voor event i de positie van het laatste event
dat start op of vóór de exitbar van i.
"""
from __future__ import annotations

import numpy as np

from ..utils.failfast import DataContractError, require

__all__ = ["event_space_t1"]


def event_space_t1(event_bar: np.ndarray, exit_bar: np.ndarray) -> np.ndarray:
    ev = np.asarray(event_bar, dtype=np.int64)
    ex = np.asarray(exit_bar, dtype=np.int64)
    require(ev.size == ex.size, "Eén exit per event.", DataContractError)
    require(bool(np.all(np.diff(ev) >= 0)), "Events moeten op tijd gesorteerd zijn.",
            DataContractError)
    require(bool(np.all(ex > ev)), "Een exit op of vóór zijn eigen event.", DataContractError)
    return np.searchsorted(ev, ex, side="right") - 1
