"""Graceful shutdown signal handlers — Wave 15 P1-25.

Call ``setup_signal_handlers(loop)`` once, just before starting the asyncio
event loop in your entry-point script.  On SIGTERM or SIGINT the loop is
stopped cleanly so that ``async with`` blocks, finally clauses, and engine
shutdown routines all run.

Usage example::

    import asyncio
    from tradebot.live.sigterm import setup_signal_handlers

    async def main() -> None:
        ...

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    setup_signal_handlers(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

logger = logging.getLogger(__name__)


def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register graceful shutdown on SIGTERM/SIGINT (Wave 15 P1-25).

    Parameters
    ----------
    loop :
        The running (or about-to-run) asyncio event loop.
    """

    def _handle_signal(sig: int) -> None:
        logger.info(
            "[SIGTERM/SIGINT] received signal %d — initiating graceful shutdown...", sig
        )
        loop.stop()

    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))
        logger.debug("Signal handlers registered: SIGTERM, SIGINT (POSIX).")
    else:
        # Windows does not support loop.add_signal_handler;
        # use synchronous signal.signal as fallback.
        signal.signal(signal.SIGTERM, lambda s, f: _handle_signal(s))
        # SIGINT (Ctrl+C) is already handled by asyncio on Windows by default.
        logger.debug("Signal handlers registered: SIGTERM (Windows fallback).")
