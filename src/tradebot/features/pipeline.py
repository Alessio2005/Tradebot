"""pipeline.py — Adapter around legacy ``features.FeaturePipeline``.

Extracted from train_regime.py lines 3607-3814 (the per-symbol feature build
section inside train_pipeline).

Why a wrapper?
  ``FeaturePipeline.transform`` (legacy ``features.py:739``) returns a 4-tuple
  ``(X_micro, X_meso, X_macro, df_merged)`` whose semantics are opaque without
  context. This adapter:
    1. Names the unpacking explicitly.
    2. Derives the canonical feature_map (micro/meso/macro buckets) from
       ``df_merged.columns`` using the same suffix-rules as train_regime.py:3773-3794.
    3. Persists the fitted PCA orthogonalizers for downstream determinism
       (Stage 3 must reproduce the exact PCA projection seen during Stage 2).

Strangler-fig: once ``features.py`` is fully decomposed into
``tradebot.bars.*`` + ``tradebot.features.{ta,orthogonalize}``, this module
becomes the single import surface for Stage 1.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Feature-map derivation (canonical rules from train_regime.py:3773-3794)
# =============================================================================

def derive_feature_map(df_merged: pd.DataFrame) -> dict[str, list[str]]:
    """Bucket ``feat_*`` columns into micro / meso / macro by suffix/prefix.

    Rules (mirror train_regime.py exactly):
      • ``feat_macro_*`` or ``*_macro``                 → macro
      • ``*_meso``                                       → meso
      • everything else starting with ``feat_``         → micro
    """
    all_cols: list[str] = df_merged.columns.tolist()

    feats_micro = [
        c for c in all_cols
        if c.startswith("feat_")
        and not c.endswith("_meso")
        and not c.endswith("_macro")
        and not c.startswith("feat_macro_")
    ]
    feats_meso = [c for c in all_cols if c.startswith("feat_") and c.endswith("_meso")]
    feats_macro = [
        c for c in all_cols
        if c.startswith("feat_")
        and (c.endswith("_macro") or c.startswith("feat_macro_"))
    ]

    return {
        "micro": sorted(feats_micro),
        "meso":  sorted(feats_meso),
        "macro": sorted(feats_macro),
    }


# =============================================================================
# Pipeline driver
# =============================================================================

def build_features(
    cfg: DictConfig,
    sym: str,
    df_micro: pd.DataFrame,
    orthogonalizer_save_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Run FeaturePipeline.transform and return ``(df_merged, feat_map)``.

    Parameters
    ----------
    cfg:
        Hydra DictConfig (already merged with per-symbol overrides).
    sym:
        Trading-pair symbol (e.g. "BTCUSDT").
    df_micro:
        Raw intraday bars from ``data.ingestion.ingest_raw``.
    orthogonalizer_save_dir:
        If provided, the fitted orthogonalizers are persisted as
        ``{dir}/orthogonalizer_{tier}_{sym}.pkl`` (mirrors train_regime.py:3801-3814).
        REQUIRED for Stage 3 reproducibility.

    Returns
    -------
    df_merged: pd.DataFrame
        Bars + micro/meso/macro features with DatetimeIndex.
    feat_map: dict[str, list[str]]
        Canonical feature_map with micro/meso/macro buckets.
    """
    from .regime import FeaturePipeline

    logger.info("[%s] Running FeaturePipeline.transform...", sym)

    engine = FeaturePipeline(cfg, sym)
    result = engine.transform(df_micro)

    # Defensive unpack: the contract is (X_micro, X_meso, X_macro, df_merged).
    if not (isinstance(result, tuple) and len(result) == 4):
        raise RuntimeError(
            f"[{sym}] FeaturePipeline.transform returned {type(result).__name__} "
            f"(len={len(result) if hasattr(result, '__len__') else 'N/A'}); "
            "expected 4-tuple (X_micro, X_meso, X_macro, df_merged)."
        )

    _X_micro, _X_meso, _X_macro, df_merged = result

    if df_merged is None or len(df_merged) == 0:
        raise ValueError(
            f"[{sym}] FeaturePipeline returned empty df_merged. "
            "Check bar configuration (min_bars, timeframe, burn-in)."
        )

    # Derive the canonical feature_map (suffix-rules).
    feat_map = derive_feature_map(df_merged)

    # ── Persist fitted orthogonalizers for Stage 3 determinism ────────────────
    # Mirrors train_regime.py:3801-3814 naming convention.
    if orthogonalizer_save_dir is not None:
        orthogonalizer_save_dir.mkdir(parents=True, exist_ok=True)
        for tier in ("micro", "meso", "macro"):
            attr_name = f"orthogonalizer_{tier}"
            orth = getattr(engine, attr_name, None)
            if orth is None:
                continue
            if not getattr(orth, "is_fitted", False):
                logger.debug("[%s] Orthogonalizer '%s' not fitted — skipping persist.", sym, tier)
                continue
            out_path = orthogonalizer_save_dir / f"orthogonalizer_{tier}_{sym}.pkl"
            joblib.dump(orth, out_path)
            logger.info(
                "[%s] Orthogonalizer '%s' saved → %s (%d components).",
                sym, tier, out_path.name,
                len(getattr(orth, "output_names", [])),
            )

    logger.info(
        "[%s] FeaturePipeline complete: %d bars, %d cols (%d micro / %d meso / %d macro).",
        sym, len(df_merged), df_merged.shape[1],
        len(feat_map["micro"]), len(feat_map["meso"]), len(feat_map["macro"]),
    )
    return df_merged, feat_map


# =============================================================================
# Re-loadable orthogonalizers (for Stage 3 inference)
# =============================================================================

def load_orthogonalizers(
    sym: str,
    orthogonalizer_dir: Path,
) -> dict[str, Any]:
    """Load previously saved orthogonalizers for downstream stages.

    Returns ``{'micro': obj, 'meso': obj, 'macro': obj}`` with missing tiers
    set to None.  Matches the file naming used by ``build_features``.
    """
    out: dict[str, Any] = {"micro": None, "meso": None, "macro": None}
    for tier in out:
        path = orthogonalizer_dir / f"orthogonalizer_{tier}_{sym}.pkl"
        if path.exists():
            out[tier] = joblib.load(path)
    return out


# =========================================================================== #
# PHASE 3 - L1 CAUSAL TRANSFORM PIPELINE
# =========================================================================== #
# Alles BOVEN deze regel is de legacy-adapter rond `features.regime`-
# FeaturePipeline (Level 2+, DI-12/Phase 6): een strangler-fig wrapper zonder
# provenance-contract. Die code is hier bewust ONGEWIJZIGD gelaten - hem nu
# aanraken zou `features/__init__.py` en `apps/build_features.py` breken.
#
# Alles HIERONDER is Phase 3: de L1-compositie van de toestandsloze transforms
# uit `features/transforms.py` tot EEN artefact, met verplichte propagatie van
# de `data_hash` van elke bronreeks.
#
# Waarom een tweede pipeline-abstractie naast `features.base.FeaturePipeline`?
# Omdat ze op iets anders werken. `base.FeaturePipeline` (Phase 2) componeert
# `BaseFeature`-objecten op EEN symbool: rijen zijn bars, kolommen zijn features.
# `PanelPipeline` (hier) componeert transforms op een CROSS-SECTIE: rijen zijn
# bars, kolommen zijn symbolen. Een rangschikking over assets bestaat niet in de
# eerste vorm, en dat is precies wat Cross-Sectional Momentum nodig heeft.
# --------------------------------------------------------------------------- #
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass
from typing import Any as _Any

import numpy as np

from ..utils.failfast import DataContractError as _DataContractError
from ..utils.failfast import require as _require
from ..utils.hashing import DATA_HASH_LENGTH as _HASH_LENGTH
from ..utils.hashing import hash_config as _hash_config
from .base import ASOF_INDEX_NAME as _ASOF_INDEX_NAME
from .base import FEATURE_DTYPE as _FEATURE_DTYPE
from .base import CertifiedPanel as _CertifiedPanel
from .transforms import burn_in_of as _burn_in_of
from .transforms import resolve_transform as _resolve_transform

#: Versie van de hash-receptuur van het L1-artefact. Wijzigt de samenstelling
#: van de payload, dan verschuiven alle hashes zichtbaar in plaats van
#: onopgemerkt. Zie `features.registry.FEATURE_HASH_VERSION` voor de L3-variant.
PANEL_HASH_VERSION = "phase3.l1.v1"


@_dataclass(frozen=True)
class TransformStep:
    """Een transform plus zijn parameters. Beide gaan mee in de hash."""

    name: str
    params: _Mapping[str, _Any]

    def __post_init__(self) -> None:
        _resolve_transform(self.name)  # crasht op een onbekende naam

    @property
    def burn_in_period(self) -> int:
        return _burn_in_of(self.name, self.params)

    def as_dict(self) -> dict[str, _Any]:
        return {
            "name": self.name,
            "params": {k: self.params[k] for k in sorted(self.params)},
        }


@_dataclass(frozen=True)
class PanelFeature:
    """Het L1-artefact: een panel plus de volledige herkomst ervan.

    `values` draagt de uitkomst; `data_hashes` de gecertificeerde bronreeksen;
    `steps` de exacte transformketen; `panel_hash` de identiteit van het geheel.
    Een consument die dit artefact citeert, citeert daarmee ook zijn data.
    """

    values: pd.DataFrame
    data_hashes: tuple[tuple[str, str], ...]
    steps: tuple[TransformStep, ...]
    burn_in_period: int
    panel_hash: str

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.values.columns)

    def drop_burn_in(self) -> pd.DataFrame:
        """Kap de burn-in af. Het alternatief is hem expliciet documenteren."""
        return self.values.iloc[self.burn_in_period :]


class PanelPipeline:
    """Compositie van L1-transforms met verplichte `data_hash`-propagatie.

    De pipeline voegt geen wiskunde toe. Hij dwingt af dat:

      * elke stap een BEKENDE transform is (een typo crasht in plaats van
        stilzwijgend een stap over te slaan);
      * de burn-in van de keten vooraf bekend is - de som van de burn-ins van
        de stappen, want ze werken sequentieel op elkaars output;
      * het resultaat float64 blijft en dezelfde asof-index houdt;
      * het artefact niet los te maken is van de data waarop het is berekend.

    De keten is bewust NIET zelf-optimaliserend en kent geen conditionele
    stappen: dezelfde stappen in dezelfde volgorde, elke run.
    """

    def __init__(self, steps: _Sequence[TransformStep]) -> None:
        _require(
            len(steps) > 0,
            "Lege PanelPipeline. Een keten zonder stappen levert het ruwe panel "
            "terug, wat later als 'signaal' zou worden gelezen.",
            _DataContractError,
        )
        for s in steps:
            _require(
                isinstance(s, TransformStep),
                "Elke stap in een PanelPipeline moet een TransformStep zijn.",
                _DataContractError,
                got=type(s).__name__,
            )
        self._steps: tuple[TransformStep, ...] = tuple(steps)

    @property
    def steps(self) -> tuple[TransformStep, ...]:
        return self._steps

    @property
    def burn_in_period(self) -> int:
        """De burn-ins tellen OP: elke stap werkt op de output van de vorige."""
        return sum(int(s.burn_in_period) for s in self._steps)

    def panel_hash(
        self, *, data_hashes: _Sequence[tuple[str, str]], git_sha: str
    ) -> str:
        """Identiteit van het artefact: keten + brondata + codeversie."""
        _require(
            bool(data_hashes),
            "Een panel_hash zonder input-data_hash is betekenisloos: het artefact "
            "zou niet aan een gecertificeerde dataset te koppelen zijn.",
            _DataContractError,
        )
        _require(
            bool(git_sha),
            "Een panel_hash zonder git_sha is niet auditbaar.",
            _DataContractError,
        )
        return _hash_config(
            {
                "hash_version": PANEL_HASH_VERSION,
                "steps": [s.as_dict() for s in self._steps],
                "burn_in_period": self.burn_in_period,
                "input_data_hashes": [list(p) for p in sorted(data_hashes)],
                "git_sha": git_sha,
            },
            length=_HASH_LENGTH,
        )

    def transform(self, panel: _CertifiedPanel, *, git_sha: str) -> PanelFeature:
        _require(
            isinstance(panel, _CertifiedPanel),
            "PanelPipeline eist een CertifiedPanel; een kaal DataFrame draagt "
            "geen provenance en mag geen L1-artefact voeden.",
            _DataContractError,
            got=type(panel).__name__,
        )
        values = panel.values
        for step in self._steps:
            values = _resolve_transform(step.name)(values, **dict(step.params))
            _require(
                bool(values.index.equals(panel.values.index)),
                "Een transform heeft de index gewijzigd. Een L1-transform mag "
                "rijen niet herordenen, toevoegen of laten vallen.",
                _DataContractError,
                step=step.name,
            )
            _require(
                list(values.columns) == list(panel.values.columns),
                "Een transform heeft de kolommen gewijzigd; het universum moet "
                "de keten ongeschonden doorkomen.",
                _DataContractError,
                step=step.name,
            )
        for col in values.columns:
            _require(
                values[col].dtype == _FEATURE_DTYPE,
                "Impliciete dtype-conversie in de transformketen; L1 blijft "
                "float64.",
                _DataContractError,
                column=str(col),
                dtype=str(values[col].dtype),
            )
        arr = values.to_numpy(dtype="float64")
        _require(
            not bool(np.isinf(arr).any()),
            "De transformketen produceerde plus/min oneindig. Dat is geen waarde "
            "maar een deling door nul, en het propageert door elk later gewicht.",
            _DataContractError,
        )
        burn_in = self.burn_in_period
        head = arr[: min(burn_in, len(arr))]
        _require(
            (int(np.isfinite(head).sum()) if head.size else 0) == 0,
            "Er bestaat een EINDIGE waarde binnen de burn-in van de keten. Dat "
            "kan alleen wanneer een stap zijn opstartfase heeft opgevuld.",
            _DataContractError,
            burn_in=burn_in,
        )
        values.index.name = _ASOF_INDEX_NAME
        return PanelFeature(
            values=values,
            data_hashes=panel.data_hashes,
            steps=self._steps,
            burn_in_period=burn_in,
            panel_hash=self.panel_hash(
                data_hashes=panel.data_hashes, git_sha=git_sha
            ),
        )
