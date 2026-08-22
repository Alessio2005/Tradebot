"""ingestion.py — Raw data ingestion: engine routing + sync + load.

Extracted from train_regime.py lines 3582-3605 (per-symbol engine routing
inside train_pipeline).

Design:
  • ingest_raw() is async because the underlying engine.sync_raw_data()
    is async (it fetches from the Bybit public archive).
  • Engine routing: data.crypto.CryptoIngestionEngine (asset-agnostic Bybit
    linear-perp ingestion).
  • Raises RuntimeError when no engine is available — callers must handle
    per-symbol errors without killing the entire pipeline run.

Strangler-fig: apps/build_features.py uses this function directly.
"""
from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Engine resolution
# ---------------------------------------------------------------------------
# Phase 0: `.crypto` is een interne module binnen data/. De try/except zette de
# engine op None en logde "Bybit data ingestion is disabled" - waarna de hele
# ingestion-pijplijn stil bleef staan zonder dat een aanroeper dat merkte.
from .crypto import CryptoIngestionEngine as _BybitEngine

logger = logging.getLogger(__name__)


def _resolve_engine(sym: str, cfg_dict: dict[str, Any]):
    """Return an instantiated engine for `sym` or raise RuntimeError."""
    bybit_target: str = cfg_dict.get("data", {}).get("bybit_symbol", "")

    if sym == bybit_target or "USDT" in sym.upper():
        logger.info("[%s] Routing to Bybit Ingestion Engine.", sym)
        return _BybitEngine(cfg_dict)

    raise RuntimeError(
        f"[{sym}] No suitable data engine for this symbol. "
        f"Expected bybit_symbol={bybit_target!r} or USDT suffix."
    )


async def ingest_raw(cfg: DictConfig, sym: str) -> pd.DataFrame:
    """Sync Bybit raw data and return the loaded DataFrame.

    Parameters
    ----------
    cfg:
        Hydra DictConfig — already merged with per-symbol overrides by the
        caller (apps/build_features.py loads conf/symbols/{sym}.yaml before
        calling this function).
    sym:
        Trading-pair symbol, e.g. "BTCUSDT".

    Returns
    -------
    pd.DataFrame
        Raw tick/bar data as loaded from the local storage backend.

    Raises
    ------
    RuntimeError
        If no engine is configured for the symbol, or if the loaded data
        is empty after sync.
    """
    cfg_dict: dict[str, Any] = cast(
        dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
    )

    # Ensure target symbol is set in the engine config hierarchy.
    cfg_dict.setdefault("model", {}).setdefault("symbols", {})["target"] = sym

    processor = _resolve_engine(sym, cfg_dict)

    logger.info("[%s] Syncing raw data from Bybit...", sym)
    await processor.sync_raw_data(symbol_override=sym)

    df_micro = await processor.storage.load_data(sym)

    if df_micro is None or df_micro.empty:
        raise RuntimeError(
            f"[{sym}] No data loaded after sync. "
            "Check the Bybit public archive availability and date range in config."
        )

    logger.info("[%s] Loaded %d raw bars.", sym, len(df_micro))
    return df_micro
