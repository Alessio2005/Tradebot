"""Ruis van ib_async uit de log: alleen de bekende, betekenisloze meldingen."""
from __future__ import annotations

import logging

from ib_async import wrapper as ib_wrapper

from tradebot.options.execution.ib_noise import DropIrrelevant, quiet_ib


def record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("ib_async.wrapper", logging.ERROR, __file__, 1, msg, None, None)


def test_only_the_known_noise_is_dropped():
    f = DropIrrelevant()
    assert not f.filter(record("Error 200, reqId 5: No security definition has been found"))
    assert not f.filter(record("Error 10090, reqId 7: Part of requested market data is not"))
    assert not f.filter(record("Warning 10167, reqId 8: Displaying delayed market data"))
    assert not f.filter(record("API connection failed: TimeoutError()"))
    assert f.filter(record("Error 201, reqId 9: Order rejected"))
    assert f.filter(record("Error 2000, reqId 9: iets anders"))


def test_quiet_ib_installs_once_and_maps_tick_90_to_halted():
    quiet_ib()
    quiet_ib()
    for name in ("ib_async.wrapper", "ib_async.client"):
        assert sum(isinstance(x, DropIrrelevant) for x in logging.getLogger(name).filters) == 1
    assert ib_wrapper.GENERIC_TICK_MAP[90] == ib_wrapper.GENERIC_TICK_MAP[49] == "halted"
