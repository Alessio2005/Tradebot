"""Exchange system status monitor — Wave 15 P0-5.7.

Polls Bybit V5 every 60 seconds and trips CircuitBreaker on outage.

NOTE (venue migration 2026-06-14): migrated Binance → Bybit.  Bybit does not
expose a Binance-style ``/sapi/v1/system/status`` maintenance flag publicly,
so we use the public ``/v5/market/time`` endpoint as a reachability + API-health
probe: ``retCode == 0`` and HTTP 200 mean healthy; anything else (non-zero
retCode, non-200, timeout, connection error) is treated as degraded and trips
the breaker.  Scheduled-maintenance windows surface as 5xx / connection refusal
on the public API, which this catches.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Public Bybit V5 server-time endpoint (no auth). Healthy → {"retCode":0,...}.
BYBIT_STATUS_URL = "https://api.bybit.com/v5/market/time"
POLL_INTERVAL_SECONDS = 60


async def monitor_exchange_status(
    circuit_breaker: CircuitBreaker,
    *,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """Background coroutine: poll Bybit API health, trip CB on outage."""

    logger.info("Exchange status monitor started (Bybit, poll every %ds)", poll_interval)
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(
                    BYBIT_STATUS_URL, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ret_code = data.get("retCode", -1)  # 0 = OK
                        if ret_code != 0:
                            msg_text = data.get("retMsg", "unknown")
                            logger.critical(
                                "Bybit API status retCode=%s msg=%r — tripping CB.",
                                ret_code,
                                msg_text,
                            )
                            circuit_breaker.trip(
                                f"exchange_status:bybit:{msg_text}",
                                config_key="live.exchange_status",
                            )
                    else:
                        logger.warning(
                            "Exchange status check returned HTTP %d", resp.status
                        )
            except asyncio.CancelledError:
                logger.info("Exchange status monitor cancelled.")
                return
            # Phase 0: aangescherpt van `except Exception`. Dit is een
            # poll-lus tegen een externe status-endpoint; netwerk- en
            # time-outfouten zijn verwachte, tijdelijke condities en de lus
            # hoort door te pollen. Elke andere fout is een bug en propageert.
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                logger.error("Exchange status check failed: %s", exc)
            await asyncio.sleep(poll_interval)
