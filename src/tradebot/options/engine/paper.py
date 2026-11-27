"""De paper-dienst: de engine met de echte klok tegen een IBKR-paper-account.

Eén proces per run-map (`SingleInstance`). Elke `tick_seconds` een ronde; in de sessie
eerst elke `refresh_seconds` nieuwe marktdata. Een fout bij het ophalen telt als een
taakfout (drie op rij is HALT). Valt de verbinding weg, dan probeert `run_paper` met
backoff opnieuw te verbinden en tickt niet (de heartbeat blijft dus uit); na `reconnect_timeout`
seconden volgt HALT. Na herstel adopteert en reconcilieert de engine opnieuw. Stoppen gaat via `stop` (SIGTERM in
de CLI): open orders worden geannuleerd en de staat wordt bewaard.

Live bestaat hier niet: `IBKRBroker.connect` weigert elk account dat geen paper-account is.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from ...schemas.options import OptionsConfig
from ...utils.single_instance import SingleInstance
from ..calendar import NY
from ..data.ibkr_feed import IBKRFeed
from ..execution.ib_noise import quiet_ib
from ..execution.ibkr import IBKRBroker
from .service import Alert, Engine, EngineMode, Paths, _no_alert
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
                    alert: Alert = _no_alert, reconnect_timeout: float = 900.0,
                    reconnect_base: float = 5.0, reconnect_cap: float = 60.0,
                    pulse: Callable[[], None] | None = None) -> Engine:
    quiet_ib()
    paths = Paths(run_dir)
    with SingleInstance(run_dir / "lock"):
        broker = IBKRBroker(ib, host, port, client_id, account)
        await broker.connect()
        feed = IBKRFeed(ib, cfg.universe.underlyings, cfg.pricing.risk_free_rate)
        state = (load_state(paths.state, PAPER_MODE, broker.account_hash)
                 or EngineState(PAPER_MODE, broker.account_hash, clock().isoformat()))
        engine = Engine(cfg, feed, broker, paths, state, alert, persist=True)
        last: datetime | None = None
        down_since: datetime | None = None
        retries = 0
        try:
            while not stop.is_set():
                now = clock()
                if not ib.isConnected():
                    # Geen tick (dus ook geen heartbeat: de healthcheck wordt rood) zolang de
                    # verbinding weg is; opnieuw verbinden met backoff, HALT na de time-out.
                    if down_since is None:
                        down_since = now
                        engine._event(now, "ib_disconnected")
                        alert("critical", "options_ib_down", "verbinding met IB Gateway weg")
                    try:
                        await broker.connect(attempts=1)
                    except (TimeoutError, OSError) as e:
                        waited = (now - down_since).total_seconds()
                        if waited >= reconnect_timeout and not engine.halted:
                            engine._engage_halt(now, "options_ib_disconnected",
                                                f"geen verbinding met IB Gateway sinds "
                                                f"{down_since.isoformat()}: {e!r}", waited,
                                                reconnect_timeout)
                        await sleep(min(reconnect_base * 2 ** retries, reconnect_cap))
                        retries += 1
                        continue
                    engine._event(clock(), "ib_reconnected", down_seconds=round(
                        (now - down_since).total_seconds(), 1))
                    alert("warning", "options_ib_up", "verbinding met IB Gateway hersteld")
                    down_since, retries = None, 0
                    engine._failures.clear()          # uitval telt niet als taakfout
                    if engine.mode not in (EngineMode.HALTED, EngineMode.UNWINDING):
                        engine.mode = EngineMode.STARTING   # opnieuw adopteren en reconciliëren
                    last = None
                if engine.cal.is_open(now) and (
                        last is None or (now - last).total_seconds() >= refresh_seconds):
                    last = now
                    await engine._run("feed", now, feed.refresh)
                await engine.tick(now)
                if pulse is not None and ib.isConnected():
                    pulse()                       # de dead man's switch, alleen bij een gezonde ronde
                await sleep(tick_seconds)
            await engine.stop(clock())
        finally:                          # ook bij een fout: de procespool van de keten dicht
            engine.builder.close()
        return engine
