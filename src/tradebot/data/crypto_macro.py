# src/tradebot/data/crypto_macro.py
"""Multi-asset crypto macro fetcher (Bybit linear-perp funding rates).

Extracted from ``data_macro_crypto.py``.

NOTE (venue migration 2026-06-14): migrated Binance Futures
(``fapi/v1/fundingRate``) → **Bybit V5** (``/v5/market/funding/history``,
``category=linear``).  Bybit returns at most 200 records per call in
DESCENDING time order with fields ``fundingRate`` + ``fundingRateTimestamp``;
we walk backwards via ``endTime`` until ``fetch_years`` of history is covered.

Fetches per-symbol funding rates from Bybit linear perps for BTC/ETH/SOL and
computes:
  * ``feat_macro_funding_zscore``     — per-asset rolling ECDF (per-asset feature)
  * ``feat_macro_market_funding``     — cross-asset mean funding regime
  * ``feat_macro_btc_dominance_proxy``— BTC funding minus alt-funding (sentiment skew)
  * ``feat_macro_funding_dispersion`` — std of per-asset funding zscores

Output:
  * ``macro_crypto_{symbol}.parquet``  — per asset (merged in train pipeline)
  * ``macro_crypto_market.parquet``    — cross-asset shared features
  * ``macro_crypto_daily.parquet``     — alias for BTC (legacy compat)

Backward compat: the old ``CryptoMacroFetcher(symbol="BTCUSDT")`` API still
works (single-symbol). New ``MultiCryptoMacroFetcher(symbols=[...])`` adds
cross-asset features.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..utils.failfast import DataContractError

logger = logging.getLogger("data.crypto_macro")


# =============================================================================
# SINGLE-ASSET FUNDING FETCHER (legacy + utility)
# =============================================================================
class CryptoMacroFetcher:
    """Single-symbol funding rate fetcher.

    Preserves the old API so that the training pipeline can import unchanged.
    The multi-asset orchestrator (``MultiCryptoMacroFetcher``) uses this class
    internally per symbol.
    """

    def __init__(
        self,
        data_dir: str = "market_data_parquet/macro",
        symbol: str = "BTCUSDT",
        fetch_years: int = 5,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.symbol   = symbol
        self.base_url = "https://api.bybit.com"
        self.category = "linear"

        self.start_ts = int(
            (pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=fetch_years))
            .timestamp() * 1000
        )

    def _paginate_backward(self, limit: int = 200) -> list:
        """Backward-paginate Bybit funding history from NOW to ``start_ts``.

        Bybit V5 ``/v5/market/funding/history`` returns <= ``limit`` (max 200)
        records per call in DESCENDING ``fundingRateTimestamp`` order.  We walk
        backwards by setting ``endTime`` to (oldest_seen_ts - 1) until we pass
        ``start_ts`` or the API returns an empty page.
        """
        endpoint = "/v5/market/funding/history"
        all_data: list = []
        end_ts = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

        for _guard in range(2000):  # hard cap: 2000 * 200 = 400k records
            params = {
                "category": self.category,
                "symbol": self.symbol,
                "endTime": end_ts,
                "limit": limit,
            }
            try:
                r = requests.get(
                    f"{self.base_url}{endpoint}", params=params, timeout=10
                )
                r.raise_for_status()
                payload = r.json()

                if payload.get("retCode", -1) != 0:
                    logger.error(
                        "[%s] Bybit funding/history retCode=%s msg=%s",
                        self.symbol, payload.get("retCode"), payload.get("retMsg"),
                    )
                    break

                rows = (payload.get("result") or {}).get("list") or []
                if not rows:
                    break

                all_data.extend(rows)

                # Rows are DESC → last element is the oldest in this page.
                oldest_ts = int(rows[-1]["fundingRateTimestamp"])
                if oldest_ts <= self.start_ts:
                    break

                end_ts = oldest_ts - 1
                time.sleep(0.1)

            except Exception as e:
                # Phase 0: dit `break` gaf de tot dan toe opgehaalde pagina's
                # terug alsof de historie compleet was. Een funding-reeks met
                # een stil gat is erger dan geen funding-reeks: de carry-P&L
                # wordt dan systematisch te gunstig geschat.
                raise DataContractError(
                    f"[{self.symbol}] Bybit funding-pagination afgebroken op "
                    f"{endpoint} na {len(all_data)} rijen: {e}. Er wordt GEEN "
                    f"gedeeltelijke historie teruggegeven."
                ) from e

        return all_data

    def fetch_funding_rate(self) -> pd.DataFrame:
        data = self._paginate_backward(limit=200)

        df = pd.DataFrame(data)
        if not df.empty:
            df["timestamp"]   = pd.to_datetime(
                df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True
            )
            df["fundingRate"] = df["fundingRate"].astype(float)
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df[["fundingRate"]]
        return pd.DataFrame()

    def calculate_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-asset rolling ECDF on funding rate → feat_macro_funding_zscore in [-1, 1]."""
        logger.info("[%s] Calculating Crypto Macro Regime Features...", self.symbol)
        if df.empty:
            return df
        df = df.sort_index().ffill()

        if "fundingRate" in df.columns:
            funding = df["fundingRate"]
            try:
                ecdf = funding.rolling("30D", min_periods=10).rank(pct=True)
            except (AttributeError, TypeError):
                ecdf = funding.rolling("30D", min_periods=10).apply(
                    lambda w: (w.rank(pct=True).iloc[-1] if len(w) > 0 else np.nan),
                    raw=False,
                )

            df["feat_macro_funding_zscore"] = 2.0 * ecdf - 1.0

        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        # FUNDING-FIX (Item 2): preserve the RAW fundingRate alongside the zscore
        # so the backtester can apply the funding fee on realised PnL.
        keep_cols: list[str] = ["feat_macro_funding_zscore"]
        if "fundingRate" in df.columns:
            keep_cols.append("fundingRate")
        return df[keep_cols]

    def run_pipeline(self) -> pd.DataFrame:
        """Single-asset pipeline. Writes macro_crypto_{symbol}.parquet."""
        logger.info("[%s] Downloading Macro Data (Target: 5 Years)...", self.symbol)
        df_fund = self.fetch_funding_rate()

        logger.info(
            "[%s] Fetch results: Fund=%d data points.", self.symbol, len(df_fund)
        )

        if df_fund.empty:
            logger.error("[%s] Funding rate stream failed. Abort.", self.symbol)
            return pd.DataFrame()

        df_features = self.calculate_macro_features(df_fund)

        per_asset_path = self.data_dir / f"macro_crypto_{self.symbol}.parquet"
        out_df = df_features.reset_index()
        if "index" in out_df.columns:
            out_df.rename(columns={"index": "timestamp"}, inplace=True)
        out_df.to_parquet(per_asset_path, engine="pyarrow", compression="zstd", index=False)
        logger.info("[%s] %d records -> %s", self.symbol, len(out_df), per_asset_path.name)

        # LEGACY: BTC also writes the old filename so the training pipeline works
        # with unchanged paths for BTC.
        if self.symbol.upper() == "BTCUSDT":
            legacy_path = self.data_dir / "macro_crypto_daily.parquet"
            out_df.to_parquet(
                legacy_path, engine="pyarrow", compression="zstd", index=False
            )
            logger.info("[%s] Legacy alias -> %s", self.symbol, legacy_path.name)

        return df_features


# =============================================================================
# MULTI-ASSET FUNDING ORCHESTRATOR + CROSS-ASSET MARKET FEATURES
# =============================================================================
class MultiCryptoMacroFetcher:
    """Fetch funding for N crypto symbols and compute cross-asset features.

    Cross-asset features (in ``macro_crypto_market.parquet``):
      * ``feat_macro_market_funding``        — cross-asset mean of per-asset
                                               funding-zscores. Market-wide sentiment.
      * ``feat_macro_btc_dominance_proxy``   — BTC funding-zscore minus mean(alt-zscores).
                                               Positive = BTC overheat; negative =
                                               alts overheat (typical alt-season skew).
      * ``feat_macro_funding_dispersion``    — std of per-asset funding-zscores.
                                               High = diverging sentiments between assets
                                               (regime-shift detector).
    """

    DEFAULT_SYMBOLS: list[str] = [  # noqa: RUF012
        "BTCUSDT", "ETHUSDT", "SOLUSDT",
        "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    ]

    def __init__(
        self,
        data_dir: str = "market_data_parquet/macro",
        symbols: list[str] | None = None,
        fetch_years: int = 5,
    ):
        self.data_dir   = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.symbols: list[str] = list(symbols) if symbols else list(self.DEFAULT_SYMBOLS)
        self.fetch_years = fetch_years

    def run_pipeline(self) -> dict[str, pd.DataFrame]:
        """Fetch all assets + build market-wide aggregates. Writes all outputs to disk."""
        per_asset: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            fetcher = CryptoMacroFetcher(
                data_dir=str(self.data_dir), symbol=sym, fetch_years=self.fetch_years
            )
            df_feat = fetcher.run_pipeline()
            if not df_feat.empty:
                per_asset[sym] = df_feat

        if not per_asset:
            logger.error("All macro streams failed. No market features created.")
            return per_asset

        logger.info("Computing cross-asset market features (dominance, dispersion)...")
        market_df = self._build_market_features(per_asset)

        market_path = self.data_dir / "macro_crypto_market.parquet"
        market_out  = market_df.reset_index()
        if "index" in market_out.columns:
            market_out.rename(columns={"index": "timestamp"}, inplace=True)
        market_out.to_parquet(
            market_path, engine="pyarrow", compression="zstd", index=False
        )
        logger.info(
            "Market-wide crypto macro: %d records -> %s (%s)",
            len(market_out), market_path.name, list(market_df.columns),
        )

        return per_asset

    def _build_market_features(
        self, per_asset: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Combine per-asset funding-zscores into marketwide features.

        Resampling: all assets aligned on a common 8h grid (Bybit majors'
        funding interval). Ffill within 24h tolerance to cover gaps.
        """
        zscores: dict[str, pd.Series] = {}
        for sym, df in per_asset.items():
            if "feat_macro_funding_zscore" not in df.columns or df.empty:
                continue
            s = df["feat_macro_funding_zscore"].copy()
            s_raw = s.resample("8h").mean()
            s = s_raw.ffill(limit=3)
            # CHIEF AUDIT 2026-05-23 (FIX 7 / silent funding ffill):
            # ``ffill(limit=3)`` silently masks up to 24h of feed downtime.
            # When more than 30% of the resulting series comes from ffilled
            # values (rather than fresh 8h reads) the resulting zscore is
            # dominated by stale data and may mislead the model.  Log a
            # warning so the operator can investigate the upstream feed
            # without breaking the backwards-compatible signature.
            n_total = int(s.notna().sum())
            n_fresh = int(s_raw.notna().sum())
            if n_total > 0:
                ffill_ratio = 1.0 - (n_fresh / n_total)
                if ffill_ratio > 0.30:
                    logger.warning(
                        "Funding zscore for %s is %.0f%% ffilled (>30%%): "
                        "%d/%d 8h-buckets are forward-filled — upstream funding "
                        "feed may be lagging or down.",
                        sym, ffill_ratio * 100.0, n_total - n_fresh, n_total,
                    )
            zscores[sym] = s

        if not zscores:
            return pd.DataFrame()

        z_mat = pd.DataFrame(zscores).sort_index()

        feat_market = z_mat.mean(axis=1).rename("feat_macro_market_funding")

        if "BTCUSDT" in z_mat.columns and z_mat.shape[1] >= 2:
            alt_cols = [c for c in z_mat.columns if c != "BTCUSDT"]
            alt_mean = z_mat[alt_cols].mean(axis=1)
            feat_dom = (z_mat["BTCUSDT"] - alt_mean).rename(
                "feat_macro_btc_dominance_proxy"
            )
        else:
            feat_dom = pd.Series(dtype=float, name="feat_macro_btc_dominance_proxy")

        feat_disp = z_mat.std(axis=1, ddof=0).rename("feat_macro_funding_dispersion")

        market_df = pd.concat([feat_market, feat_dom, feat_disp], axis=1)
        market_df = market_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return market_df


__all__ = [
    "CryptoMacroFetcher",
    "MultiCryptoMacroFetcher",
]
