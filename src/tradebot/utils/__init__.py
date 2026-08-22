"""tradebot.utils — low-level helpers with zero business logic.

arrays.py     : numba_array() — single source of truth for Pandas → Numba.
parquet_io.py : write_validated_parquet() — schema-validated stage output.

These two modules eliminate _f64c / _i32c (blueprint §2.3 root-cause).
"""
from tradebot.utils.arrays import numba_array, validate_or_die
from tradebot.utils.parquet_io import read_validated_parquet, write_validated_parquet

__all__ = [
    "numba_array",
    "read_validated_parquet",
    "validate_or_die",
    "write_validated_parquet",
]
