"""De paper-dienst: de engine met de echte klok tegen een IBKR-paper-account.

Eén proces per run-map (`SingleInstance`). Elke `tick_seconds` een ronde; in de sessie
eerst elke `refresh_seconds` nieuwe marktdata. Een fout bij het ophalen telt als een
taakfout (drie op rij is HALT). Stoppen gaat via `stop` (SIGTERM in de CLI): open orders
worden geannuleerd en de staat wordt bewaard.

Live bestaat hier niet: `IBKRBroker.connect` weigert elk account dat geen paper-account is.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from ...carry.safety import SingleInstance
from ...schemas.options import OptionsConfig
from ..calendar import NY
from ..data.ibkr_feed import IBKRFeed
from ..execution.ib_noise import quiet_ib
from ..execution.ibkr import IBKRBroker
from .service import Alert, Engine, Paths, _no_alert
from .state import EngineState, load_state

__all__ = ["PAPER_MODE", "now_ny", "run_paper"]

PAPER_MODE = "paper"


def now_ny() -> datetime:
    return datetime.now(NY)


async def run_paper(cfg: OptionsConfig, ib: Any, run_dir: Path, stop: asyncio.Event,
                    account: str = "", host: str = "127.0.0.1", port: int = 4002,
                    client_id: int = 17, clock: Callable[[], datetime] = now_ny,
                    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                    tick_seconds: float = 15.0, refresh_seconds: float = 300.0,
                    alert: Alert = _no_alert) -> Engine:
    quiet_ib()
    paths = Paths(run_dir)
    with SingleInstance(run_dir / "lock"):
        broker = IBKRBroker(ib, host, port, client_id, account)
        await broker.connect()
        feed = IBKRFeed(ib, cfg.universe.underlyings, cfg.pricing.risk_free_rate)
        state = (load_state(paths.state, PAPER_MODE, broker.account_hash)
                 or EngineState(PAPER_MODE, broker.account_hash, clock().isoformat()))
        engine = Engine(cfg, feed, broker, paths, state, alert)
        last: datetime | None = None
        try:
            while not stop.is_set():
                now = clock()
                if engine.cal.is_open(now) and (
                        last is None or (now - last).total_seconds() >= refresh_seconds):
                    last = now
                    await engine._run("feed", now, feed.refresh)
                await engine.tick(now)
                await sleep(tick_seconds)
            await engine.stop(clock())
        finally:                          # ook bij een fout: de procespool van de keten dicht
            engine.builder.close()
        return engine
