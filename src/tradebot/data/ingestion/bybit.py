"""Bybit V5 publieke REST-client — Phase 1.

Eén client voor alle crypto-bronnen. Geen enkele methode geeft een lege of
gedeeltelijke reeks terug: een gefaalde of afgebroken pagination crasht.

Bybit V5-eigenaardigheden die hier worden afgehandeld:
  * responses zijn DESCENDING in tijd en gecapt op 1000 (kline) / 200 (funding,
    open interest) rijen per call, dus er wordt achterwaarts gepagineerd via de
    `end`/`endTime`-cursor;
  * `retCode != 0` is een applicatiefout met HTTP 200 — die moet expliciet
    worden gecontroleerd, anders leest de pipeline een foutmelding als data;
  * timestamps zijn Unix MILLIseconden als string.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from ...utils.failfast import DataContractError, require

__all__ = ["BybitV5Client", "MS_TO_NS"]

logger = logging.getLogger(__name__)

#: Bybit levert Unix milliseconden; de PIT-store slaat nanoseconden op.
MS_TO_NS = 1_000_000

_BASE_URL = "https://api.bybit.com"
_HTTP_TIMEOUT_S = 20
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_S = 1.5
_THROTTLE_S = 0.12
#: Harde bovengrens op het aantal pagina's per reeks. Beschermt tegen een
#: cursor die niet opschuift; zonder deze grens zou dat een oneindige lus zijn.
_MAX_PAGES = 400


class BybitV5Client:
    """Dunne, fail-fast wrapper rond de publieke Bybit V5 market-endpoints."""

    def __init__(self, base_url: str = _BASE_URL, category: str = "linear") -> None:
        self.base_url = base_url.rstrip("/")
        self.category = category
        self._session = requests.Session()

    # ------------------------------------------------------------------ HTTP
    def _get(self, endpoint: str, params: dict) -> dict:
        """Eén GET met retry-op-transport. Crasht na uitputting van de pogingen."""
        url = f"{self.base_url}{endpoint}"
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = self._session.get(url, params=params, timeout=_HTTP_TIMEOUT_S)
            except (requests.Timeout, requests.ConnectionError) as exc:
                # Transportfouten zijn tijdelijk; een retry is geen degradatie.
                # Bij uitputting wordt WEL geraist - nooit een lege reeks.
                last_exc = exc
                logger.warning("Bybit %s poging %d/%d: %s",
                               endpoint, attempt, _MAX_ATTEMPTS, exc)
                time.sleep(_BACKOFF_BASE_S ** attempt)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                logger.warning("Bybit %s HTTP %d, poging %d/%d",
                               endpoint, resp.status_code, attempt, _MAX_ATTEMPTS)
                time.sleep(_BACKOFF_BASE_S ** attempt)
                continue

            require(resp.status_code == 200,
                    "Bybit gaf een niet-herstelbare HTTP-status.",
                    DataContractError, endpoint=endpoint,
                    status=resp.status_code, body=resp.text[:200])

            payload = resp.json()
            # retCode != 0 is een APPLICATIEfout met HTTP 200. Zonder deze
            # controle leest de pipeline een foutmelding als data.
            require(payload.get("retCode") == 0,
                    "Bybit retCode != 0; de respons bevat geen data maar een "
                    "foutmelding.",
                    DataContractError, endpoint=endpoint,
                    retCode=payload.get("retCode"),
                    retMsg=str(payload.get("retMsg"))[:200], params=str(params))
            return payload

        raise DataContractError(
            f"Bybit {endpoint} onbereikbaar na {_MAX_ATTEMPTS} pogingen: "
            f"{last_exc}. Er wordt GEEN lege of gedeeltelijke reeks "
            f"teruggegeven."
        )

    # -------------------------------------------------------------- paginate
    def _paginate_backward(
        self,
        endpoint: str,
        base_params: dict,
        *,
        start_ms: int,
        end_ms: int,
        ts_key: str | int,
        cursor_param: str,
        limit: int,
    ) -> list:
        """Pagineer achterwaarts tot `start_ms` is bereikt.

        Crasht wanneer de cursor niet opschuift of het paginabudget op is: dat
        is een gedeeltelijke reeks, en die is erger dan geen reeks.
        """
        rows: list = []
        cursor = end_ms
        for page in range(_MAX_PAGES):
            params = {**base_params, cursor_param: cursor, "limit": limit}
            payload = self._get(endpoint, params)
            batch = (payload.get("result") or {}).get("list") or []
            if not batch:
                break

            rows.extend(batch)
            oldest = int(batch[-1][ts_key])
            if oldest <= start_ms:
                break
            new_cursor = oldest - 1
            require(new_cursor < cursor,
                    "Bybit-cursor schuift niet op; dit zou een oneindige lus "
                    "zijn en levert anders een gedeeltelijke reeks.",
                    DataContractError, endpoint=endpoint, cursor=cursor)
            cursor = new_cursor
            time.sleep(_THROTTLE_S)
        else:
            raise DataContractError(
                f"Bybit {endpoint} paginabudget ({_MAX_PAGES}) uitgeput voor "
                f"{base_params.get('symbol')}. Er wordt geen gedeeltelijke "
                f"reeks teruggegeven."
            )

        require(bool(rows),
                "Bybit leverde nul rijen over het gevraagde venster.",
                DataContractError, endpoint=endpoint,
                symbol=base_params.get("symbol"),
                start=str(pd.Timestamp(start_ms, unit="ms", tz="UTC")),
                end=str(pd.Timestamp(end_ms, unit="ms", tz="UTC")))
        return rows

    # ----------------------------------------------------------------- kline
    def kline(self, symbol: str, interval: str, start_ms: int, end_ms: int
              ) -> pd.DataFrame:
        """OHLCV-bars. `interval`: "D" voor daily, "5" voor 5m, etc."""
        rows = self._paginate_backward(
            "/v5/market/kline",
            {"category": self.category, "symbol": symbol, "interval": interval,
             "start": start_ms},
            start_ms=start_ms, end_ms=end_ms, ts_key=0,
            cursor_param="end", limit=1000,
        )
        df = pd.DataFrame(rows, columns=[
            "start_ms", "open", "high", "low", "close", "volume", "turnover"])
        return df.astype({"start_ms": "int64", "open": float, "high": float,
                          "low": float, "close": float, "volume": float,
                          "turnover": float})

    # --------------------------------------------------------------- funding
    def funding_history(self, symbol: str, start_ms: int, end_ms: int
                        ) -> pd.DataFrame:
        rows = self._paginate_backward(
            "/v5/market/funding/history",
            {"category": self.category, "symbol": symbol, "startTime": start_ms},
            start_ms=start_ms, end_ms=end_ms,
            ts_key="fundingRateTimestamp", cursor_param="endTime", limit=200,
        )
        df = pd.DataFrame(rows)
        return pd.DataFrame({
            "settle_ms": df["fundingRateTimestamp"].astype("int64"),
            "funding_rate": df["fundingRate"].astype(float),
        })

    # --------------------------------------------------------- open interest
    def open_interest(self, symbol: str, interval: str, start_ms: int,
                      end_ms: int) -> pd.DataFrame:
        """`interval`: "5min" | "15min" | "30min" | "1h" | "4h" | "1d"."""
        rows = self._paginate_backward(
            "/v5/market/open-interest",
            {"category": self.category, "symbol": symbol,
             "intervalTime": interval, "startTime": start_ms},
            start_ms=start_ms, end_ms=end_ms, ts_key="timestamp",
            cursor_param="endTime", limit=200,
        )
        df = pd.DataFrame(rows)
        return pd.DataFrame({
            "ts_ms": df["timestamp"].astype("int64"),
            "open_interest": df["openInterest"].astype(float),
        })

    # ----------------------------------------------------------- instruments
    def instrument_launch_ms(self, symbol: str) -> int:
        """Launch-timestamp van het instrument, voor het levensloop-contract."""
        payload = self._get("/v5/market/instruments-info",
                            {"category": self.category, "symbol": symbol})
        lst = (payload.get("result") or {}).get("list") or []
        require(bool(lst),
                "Bybit kent dit symbool niet; het universum in conf/ verwijst "
                "naar een instrument dat op deze venue niet bestaat.",
                DataContractError, symbol=symbol, category=self.category)
        return int(lst[0]["launchTime"])
