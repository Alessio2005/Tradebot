"""Eén proces per map: een exclusieve bestandslock (fcntl).

Woonde in `tradebot.carry.safety`; verhuisd zodat de options-dienst de lock kan gebruiken
zonder het crypto-pakket te laden. `carry.safety` importeert beide namen hiervandaan, dus
`from tradebot.carry.safety import Deferred, SingleInstance` blijft werken.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType

from .failfast import TradebotContractError

__all__ = ["Deferred", "SingleInstance"]


class Deferred(TradebotContractError):
    """De stap kan nu niet veilig; later opnieuw proberen (geen stilstand)."""


class SingleInstance:
    """Een exclusieve lock op een bestand (fcntl): een tweede proces krijgt `Deferred`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def __enter__(self) -> SingleInstance:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise Deferred(f"Een ander proces houdt {self.path}; deze ronde slaat over.") \
                from None
        self._fd = fd
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        assert self._fd is not None
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None
