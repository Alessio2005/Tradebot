# src/tradebot/data/__init__.py
"""Data sub-package — ingestion, macro, crypto data."""
from .crypto import BybitPublicClient, CryptoIngestionEngine, ParquetStorage
from .crypto_macro import CryptoMacroFetcher, MultiCryptoMacroFetcher
from .open_interest import OpenInterestFetcher, load_per_bar_open_interest
from .tradfi_macro import MacroDataFetcher

__all__ = [
    # crypto ingestion
    "BybitPublicClient",
    "ParquetStorage",
    "CryptoIngestionEngine",
    # macro
    "MacroDataFetcher",
    "CryptoMacroFetcher",
    "MultiCryptoMacroFetcher",
    # open interest
    "OpenInterestFetcher",
    "load_per_bar_open_interest",
]
