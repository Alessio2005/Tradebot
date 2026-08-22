# src/tradebot/data/tradfi_macro.py
"""TradFi macro data fetcher (Yahoo Finance, Deribit DVOL, Fear & Greed).

Extracted from ``data_macro.py``.

Data sources:
  * Yahoo Finance  — global yield curve, credit spread, DXY (via UUP ETF)
  * Deribit API    — DVOL (BTC Volatility Index, crypto-native VIX-equivalent)
  * alternative.me — Crypto Fear & Greed Index

Deliberately NOT included (removed from legacy):
  * ^VIX / ^VIX3M  — SP500 implied vol; not representative of crypto IV.
  * ^ADD            — NYSE Advance-Decline; irrelevant for crypto.
  * SqueezeMetrics GEX/DIX — SP500 options gamma; structurally disconnected
                              from crypto microstructure.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger("data.tradfi_macro")


# Per-indicator publication lag in hours (crypto bars = 1h resolution typical).
# CPI, NFP: 08:30 ET on release day ≈ 12h UTC lag from midnight of prior day.
# FOMC: 18:00 UTC on announcement day ≈ 18h lag.
# GDP: 08:30 ET first estimate ≈ 12h lag.
# Default: 12h = "safe minimum" for daily macro to intraday crypto bars.
_MACRO_LAG_HOURS: dict[str, int] = {
    "cpi":               12,   # 08:30 ET release
    "core_cpi":          12,
    "pce":               12,
    "nfp":               12,   # Non-Farm Payrolls 08:30 ET Friday
    "unemployment":      12,
    "fomc":              18,   # 18:00 UTC announcement
    "fed_funds":         18,
    "gdp":               12,
    "gdp_growth":        12,
    "retail_sales":      12,
    "ism_manufacturing": 12,
    "ism_services":      12,
}
_DEFAULT_MACRO_LAG_HOURS: int = 12


class MacroDataFetcher:
    """Fetch and process macro-economic market data for crypto assets (BTC, ETH, SOL).

    Args:
        data_dir: directory where ``macro_tradfi_daily.parquet`` is saved.
    """

    def __init__(self, data_dir: str = "market_data_parquet/macro"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ── Yahoo Finance Tickers ─────────────────────────────────────────────
        # ^US2Y does not exist on Yahoo Finance — replaced by ^IRX (13-week T-Bill).
        # DX-Y.NYB can be flaky; UUP (Invesco DB Dollar Index ETF) is a stable proxy.
        self.tickers: dict[str, str] = {
            "10Y_Yield": "^TNX",   # 10-Year Treasury Yield — interest rate cycle
            "2Y_Yield":  "^IRX",   # 13-Week T-Bill — short rate / curve spread
            "HYG":       "HYG",    # High Yield / Junk Bonds ETF — credit risk proxy
            "TLT":       "TLT",    # 20+ Year Treasuries ETF — duration regime
            "DXY":       "UUP",    # Dollar Index via Invesco DB ETF (more stable)
        }

        self._yf_timeout: int       = 30
        self._yf_max_retries: int   = 3
        self._yf_retry_delay: float = 5.0
        self._dvol_lookback_days: int = 365 * 5  # 5 years

    # =========================================================================
    # 1. DERIBIT DVOL — Crypto-native Volatility Index
    # =========================================================================
    def fetch_dvol(self, currency: str = "BTC") -> pd.DataFrame:
        """Fetch the Deribit Volatility Index (DVOL) for BTC (or ETH).

        DVOL is the crypto equivalent of ^VIX: a model-free implied vol computed
        from the full Deribit options smile for the given asset. Resolution: daily
        (86400 seconds). Output: 'dvol_{currency}' column.
        """
        logger.info("Downloading Deribit DVOL (%s) daily data...", currency)
        url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

        end_ts   = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        start_ts = int(
            (pd.Timestamp.now(tz="UTC") - pd.DateOffset(days=self._dvol_lookback_days))
            .timestamp() * 1000
        )

        all_rows: list = []
        current_start = start_ts
        resolution = 86400  # daily in seconds

        while current_start < end_ts:
            params = {
                "currency":        currency.upper(),
                "start_timestamp": current_start,
                "end_timestamp":   end_ts,
                "resolution":      resolution,
                "count_back":      500,
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                result = r.json().get("result", {})
                data   = result.get("data", [])

                if not data:
                    break

                all_rows.extend(data)

                last_ts = int(data[-1][0])
                if last_ts <= current_start:
                    break
                current_start = last_ts + resolution * 1000

                time.sleep(0.15)

            except Exception as e:
                logger.error("DVOL fetch error (%s): %s", currency, e)
                break

        if not all_rows:
            logger.warning("DVOL (%s): no data received.", currency)
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=["ts_ms", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        col_name = f"dvol_{currency.lower()}"
        df[col_name] = df["close"].astype(float)
        logger.info(" -> DVOL %s: %d daily records.", currency, len(df))
        return df[[col_name]]

    # =========================================================================
    # 2. FEAR & GREED INDEX — Crypto-native sentiment
    # =========================================================================
    def fetch_fear_greed(self) -> pd.DataFrame:
        """Fetch the Crypto Fear & Greed Index from alternative.me.

        Composite indicator (0 = Extreme Fear, 100 = Extreme Greed) based on
        volatility, volume, social media, dominance and trends. Daily frequency,
        free API, maximum ≈2000 days of history available.
        """
        logger.info("Downloading Crypto Fear & Greed Index from alternative.me...")
        url = "https://api.alternative.me/fng/?limit=2000&format=json"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; CryptoMacroFetcher/1.0)"}
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            payload = r.json()
            data    = payload.get("data", [])

            if not data:
                logger.warning("Fear & Greed: empty payload received.")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(int), unit="s", utc=True
            )
            df["fear_greed"] = df["value"].astype(float)
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="last")]

            logger.info(" -> Fear & Greed: %d daily records.", len(df))
            return df[["fear_greed"]]

        except Exception as e:
            logger.error("Fear & Greed fetch error: %s", e)
            return pd.DataFrame()

    # =========================================================================
    # 3. YAHOO FINANCE — Global TradFi macro
    # =========================================================================
    def _extract_close_price(
        self, df_raw: pd.DataFrame | None, name: str
    ) -> pd.DataFrame:
        """Safely extract the Close price from a yfinance DataFrame."""
        if df_raw is None or df_raw.empty:
            return pd.DataFrame()

        try:
            if isinstance(df_raw.columns, pd.MultiIndex):
                if "Close" in df_raw.columns.get_level_values(0):
                    df_ticker = df_raw["Close"].copy()
                    if isinstance(df_ticker, pd.DataFrame):
                        df_ticker = df_ticker[[df_ticker.columns[0]]]
                    else:
                        df_ticker = df_ticker.to_frame()
                else:
                    df_ticker = df_raw.iloc[:, 0].to_frame()
            elif "Close" in df_raw.columns:
                df_ticker = df_raw[["Close"]].copy()
            else:
                df_ticker = df_raw.iloc[:, 0].to_frame()

            df_ticker.columns = [name]
            return df_ticker
        except Exception as e:
            logger.error("Error extracting data for %s: %s", name, e)
            return pd.DataFrame()

    def _download_ticker_with_retry(
        self, name: str, ticker: str, start_date: str
    ) -> pd.DataFrame:
        """Download one ticker with retry logic and timeout."""
        for attempt in range(1, self._yf_max_retries + 1):
            try:
                df_raw = yf.download(
                    ticker,
                    start=start_date,
                    progress=False,
                    auto_adjust=False,
                    timeout=self._yf_timeout,
                )
                if df_raw is None:
                    logger.warning(" -> %s (%s): no data (attempt %d).", name, ticker, attempt)
                    continue
                df_clean = self._extract_close_price(df_raw, name)
                if not df_clean.empty:
                    logger.info(" -> %s (%s) downloaded (%d rows).", name, ticker, len(df_clean))
                    return df_clean
                else:
                    logger.warning(
                        " -> %s (%s): empty DataFrame (attempt %d).", name, ticker, attempt
                    )
            except Exception as e:
                logger.warning(
                    " -> %s (%s): error at attempt %d/%d: %s",
                    name, ticker, attempt, self._yf_max_retries, e,
                )
                if attempt < self._yf_max_retries:
                    time.sleep(self._yf_retry_delay)

        logger.error(
            " -> %s (%s): definitively failed after %d attempts.", name, ticker, self._yf_max_retries
        )
        return pd.DataFrame()

    def fetch_daily_data(self, start_date: str = "2010-01-01") -> pd.DataFrame:
        """Fetch all macro data sources and combine them.

        Order:
          1. Yahoo Finance TradFi tickers (yields, credit, DXY)
          2. Deribit DVOL (BTC crypto-native IV)
          3. Crypto Fear & Greed Index (alternative.me)

        Tickers that fail after all retries are gracefully skipped.
        """
        logger.info(
            "Downloading TradFi macro data from Yahoo Finance since %s...", start_date
        )

        raw_dfs: list[pd.DataFrame] = []
        for name, ticker in self.tickers.items():
            df_clean = self._download_ticker_with_retry(name, ticker, start_date)
            if not df_clean.empty:
                raw_dfs.append(df_clean)

        if not raw_dfs:
            logger.error("No single TradFi macro ticker could be downloaded.")
            return pd.DataFrame()

        df_macro = pd.concat(raw_dfs, axis=1)
        df_macro.index = pd.to_datetime(df_macro.index, utc=True)

        df_dvol = self.fetch_dvol(currency="BTC")
        if not df_dvol.empty:
            df_macro = df_macro.join(df_dvol, how="outer")
        else:
            logger.warning("DVOL not available — DVOL features will be skipped.")

        df_fg = self.fetch_fear_greed()
        if not df_fg.empty:
            df_macro = df_macro.join(df_fg, how="outer")
        else:
            logger.warning("Fear & Greed not available — F&G feature will be skipped.")

        df_macro = df_macro.dropna(how="all").ffill()
        logger.info(
            "Macro data ready: %d days, %d columns: %s",
            len(df_macro), df_macro.shape[1], list(df_macro.columns),
        )
        return df_macro

    # =========================================================================
    # 4. FEATURE COMPUTATION
    # =========================================================================
    def calculate_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived indicators from the fetched macro data.

        Feature groups:
          A. YIELD CURVE & RATES       — yield curve spread, steepness
          B. CREDIT-SPREAD REGIME      — HYG/TLT ratio, credit momentum
          C. DOLLAR REGIME             — DXY momentum, zscore
          D. CRYPTO-NATIVE VOLATILITY  — DVOL zscore, IV regime
          E. CRYPTO SENTIMENT          — Fear & Greed zscore, extremes
        """
        if df is None or df.empty:
            return pd.DataFrame()

        logger.info("Calculating Crypto-Optimised Macro Features...")
        df = df.copy()

        # ── A. YIELD CURVE & RATES ───────────────────────────────────────────
        if {"10Y_Yield", "2Y_Yield"}.issubset(df.columns):
            df["feat_macro_yc_spread"]    = df["10Y_Yield"] - df["2Y_Yield"]
            df["feat_macro_yc_steepness"] = df["feat_macro_yc_spread"].diff(20)

        # ── B. CREDIT-SPREAD REGIME ──────────────────────────────────────────
        if {"HYG", "TLT"}.issubset(df.columns):
            df["feat_macro_credit_spread"]   = df["HYG"] / df["TLT"].replace(0, 1e-9)
            df["feat_macro_credit_momentum"] = df["feat_macro_credit_spread"].pct_change(10)

        # ── C. DOLLAR REGIME ────────────────────────────────────────────────
        if "DXY" in df.columns:
            df["feat_macro_dxy_momentum"] = df["DXY"].pct_change(5)
            dxy_roll = df["DXY"].rolling(window=20)
            df["feat_macro_dxy_zscore"] = (
                (df["DXY"] - dxy_roll.mean()) / dxy_roll.std().replace(0, 1e-9)
            )

        # ── D. CRYPTO-NATIVE VOLATILITY (Deribit DVOL) ──────────────────────
        if "dvol_btc" in df.columns:
            dvol_roll = df["dvol_btc"].rolling(window=20)
            df["feat_macro_dvol_zscore"] = (
                (df["dvol_btc"] - dvol_roll.mean()) / dvol_roll.std().replace(0, 1e-9)
            )
            df["feat_macro_dvol_regime"]   = np.clip(df["feat_macro_dvol_zscore"], -3.0, 3.0)
            dvol_pct = df["dvol_btc"].rolling(252, min_periods=30).rank(pct=True)
            df["feat_macro_dvol_pct"]      = dvol_pct.fillna(0.5)
            df["feat_macro_dvol_momentum"] = df["dvol_btc"].pct_change(5)

        # ── E. CRYPTO SENTIMENT (Fear & Greed) ──────────────────────────────
        if "fear_greed" in df.columns:
            df["feat_macro_fg_norm"]          = (df["fear_greed"] - 50.0) / 50.0
            df["feat_macro_fg_extreme_fear"]  = (df["fear_greed"] < 20).astype(float)
            df["feat_macro_fg_extreme_greed"] = (df["fear_greed"] > 80).astype(float)
            df["feat_macro_fg_momentum"]      = df["fear_greed"].diff(14)

        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        return df[feature_cols]

    # =========================================================================
    # 5. PIPELINE ENTRY POINT
    # =========================================================================
    def run_pipeline(self) -> None:
        """Run the full ingestion and processing pipeline."""
        df_raw = self.fetch_daily_data()

        if df_raw is None or df_raw.empty:
            logger.error("No data received. Pipeline stopped.")
            return

        df_features = self.calculate_macro_features(df_raw)

        # Prevent look-ahead bias: apply per-indicator publication-lag shifts.
        # CHIEF AUDIT 2026-05-23 (FIX 4 / double publication-lag):
        # The previous implementation called ``.shift(lag_bars)`` where
        # ``lag_bars`` is the lag expressed in HOURS but used as an INTEGER
        # number of ROWS.  Because yfinance feeds this fetcher with DAILY
        # data, ``.shift(12)`` shifted CPI/NFP twelve **days** into the
        # future (not 12 hours), and ``data/macro.py`` then shifted those
        # already-lagged columns AGAIN by 12h in ``_merge_asof_lagged``.
        # The fix: shift the INDEX by the lag expressed as a Timedelta so
        # the resulting lag is exactly ``lag_h`` hours regardless of the
        # source's native frequency.  ``data/macro.py`` still applies its
        # own ``_PUBLICATION_LAG`` (12h) in the merge, so the *effective*
        # lag after the merge will be approximately (lag_h + 12h).  The
        # 12h merge-side lag is intentionally retained for clock-drift
        # safety; suppressing it would require a per-column tolerance
        # rework in ``_merge_asof_lagged``.  Document this clearly so the
        # behaviour is auditable.
        logger.info(
            "Applying per-indicator publication-lag shifts (FIX 4 / time-shift)..."
        )
        df_shifted_frames: list[pd.DataFrame] = []
        for col in df_features.columns:
            col_lower = col.lower().replace(" ", "_").replace("-", "_")
            lag_h = _DEFAULT_MACRO_LAG_HOURS
            for key, hours in _MACRO_LAG_HOURS.items():
                if key in col_lower:
                    lag_h = hours
                    break
            s = df_features[[col]].copy()
            # Time-based shift: push the timestamp forward by lag_h hours so a
            # value originally indexed at 2024-01-15 00:00 UTC (the date the
            # CPI print refers to) becomes observable at 2024-01-15 12:00 UTC
            # (the wall-clock publication time).  This is independent of the
            # bar frequency of the source.
            s.index = s.index + pd.Timedelta(hours=lag_h)
            df_shifted_frames.append(s)

        if df_shifted_frames:
            df_features = pd.concat(df_shifted_frames, axis=1).sort_index()
        df_features = df_features.dropna(how="all")

        save_path = self.data_dir / "macro_tradfi_daily.parquet"

        df_export = df_features.reset_index()
        rename_map = {"Date": "timestamp", "date": "timestamp", "index": "timestamp"}
        df_export.rename(columns=rename_map, inplace=True)

        try:
            df_export.to_parquet(save_path, engine="pyarrow", compression="zstd", index=False)
            logger.info("Success! %d records saved to %s", len(df_export), save_path)
        except Exception as e:
            logger.error("Error saving parquet: %s", e)


__all__ = ["MacroDataFetcher"]
