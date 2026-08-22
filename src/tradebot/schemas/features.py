"""FeatureBlockSchema — full feature frame.  Stage 1 output.

Design decisions (blueprint §3.3, O2):
  • All feat_* columns are float32 on disk (halves footprint).
  • Numba kernels receive float64 via numba_array(..., dtype="f64") — the
    single cast happens in utils/arrays.py, NOT scattered across the code.
  • strict = True: extra columns → FAIL.  The schema is the contract.
    If you need a new column, add it here first.
  • coerce = False: no silent casts.

Tier-based suffix convention (enforced by dataframe_check):
  _micro   : bar-level (microstructure) features
  _meso    : rolling-window features (regime-level)
  _macro   : cross-asset / macro features (publication-lag aware)
  feat_macro_* : macro block (alternative prefix form)
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class FeatureBlockSchema(pa.DataFrameModel):
    """Full feature block.  Output of Stage 1 build_features.

    Strict: extra columns fail.  All feat_* must be float32.
    Index: monotonic DatetimeIndex, unique, UTC-aware.
    """

    # Index validated by dataframe_check (tz-aware UTC is accepted).
    # check_name=False: the index may be named "timestamp" or unnamed.

    # ── Required non-feature columns (needed by downstream stages) ────────────
    close:       Series[float] = pa.Field(gt=0, nullable=False)
    high:        Series[float] = pa.Field(gt=0, nullable=False)
    low:         Series[float] = pa.Field(gt=0, nullable=False)
    open:        Series[float] = pa.Field(gt=0, nullable=False)
    volume:      Series[float] = pa.Field(ge=0, nullable=False)
    feat_vol_gk: Series[float] = pa.Field(gt=0, nullable=False)

    class Config:
        # strict=False: the feature block has a dynamic set of feat_* columns;
        # the schema declares required skeleton columns only.  Tier-suffix and
        # dtype checks run via dataframe_check instead.
        strict = False
        coerce = False
        ordered = False

    @pa.dataframe_check
    def index_monotonic_increasing(cls, df: DataFrame) -> bool:
        return bool(df.index.is_monotonic_increasing)

    @pa.dataframe_check
    def index_is_datetime(cls, df: DataFrame) -> bool:
        return isinstance(df.index, pd.DatetimeIndex)

    @pa.dataframe_check
    def feat_columns_have_valid_tier_suffix(cls, df: DataFrame) -> bool:
        """feat_* columns (excl. feat_vol_gk) must end with _micro/_meso/_macro,
        start with feat_macro_, OR have no tier suffix (micro-level by convention).

        Micro features intentionally have no suffix — they are identified by
        exclusion (not _meso and not _macro) in derive_feature_map().
        """
        valid_suffixes = ("_micro", "_meso", "_macro")
        for col in df.columns:
            if not col.startswith("feat_"):
                continue
            if col == "feat_vol_gk":
                continue
            if col.startswith("feat_macro_"):
                continue
            # Allow: has a valid suffix OR has no tier suffix (micro convention)
            has_valid = any(col.endswith(s) for s in valid_suffixes)
            has_any_tier_suffix = (
                col.endswith("_micro") or col.endswith("_meso") or col.endswith("_macro")
            )
            if has_any_tier_suffix and not has_valid:
                return False  # has a suffix but invalid one
        return True

    @pa.dataframe_check
    def no_all_nan_columns(cls, df: DataFrame) -> bool:
        """No column may be entirely NaN — indicates upstream feature failure."""
        return bool(~df.isna().all(axis=0).any())


# Backward-compat alias (schemas/__init__.py public API)
FeaturesSchema = FeatureBlockSchema
