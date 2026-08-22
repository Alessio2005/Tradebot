"""features/scaling.py — Oscillator detection + SymQS overlay.

Extracted from train_regime.py lines 88-188 (_OSCILLATOR_TOKENS,
_BINARY_PASSTHROUGH_TOKENS, _build_oscillator_masks, _apply_symqs_overlay).

Also re-exports ``RollingRobustScaler`` from ``agent.py`` as the single
authoritative import path for downstream code.

Design (blueprint §3.1 / §3.3):
  • _build_oscillator_masks — pure function, no state.
  • _apply_symqs_overlay    — stateless transform, safe to call per-fold.
  • RollingRobustScaler     — stateful, fitted per-symbol in Stage 1 and
    serialised to an artefact; loaded read-only in Stage 2/3.

Strangler-fig: once train_regime.py imports from here and the equivalence
test passes, delete lines 88-188 in train_regime.py.
"""
from __future__ import annotations

import logging

import numpy as np

from ..utils.failfast import CausalityViolationError

logger = logging.getLogger(__name__)


# =============================================================================
# OSCILLATOR TOKEN REGISTRY  (blueprint §3.1 — BLUEPRINT-FIX I.1)
# =============================================================================
# Oscillators (RSI, MACD, stochastic-like indicators) carry a meaningful
# zero line.  RollingRobustScaler compresses this to median/IQR but loses
# the zero-crossing semantics.  SymmetricQuantileScaler restores it by
# scaling the negative and positive halves independently.  We run SymQS NOT
# instead of RRS but ON TOP of RRS output: RRS remains the fallback for all
# non-oscillator features (volume, ATR, vol-percentiles).

_OSCILLATOR_TOKENS: tuple[str, ...] = (
    "rsi", "macd", "stoch", "cci", "willr", "williams",
    "momentum", "mom_", "_mom", "roc_", "_roc", "ppo",
    "tsi", "uo_", "ultimate_osc", "fisher",
    # signed cycle / phase indicators
    "phase", "_signal", "delta_signal",
    # custom z-scores and signed frames carrying a zero line
    "_zscore", "_z_", "_centered",
)

_BINARY_PASSTHROUGH_TOKENS: tuple[str, ...] = (
    "is_", "_is", "flag_", "_flag", "regime_", "_regime",
    "weekend", "session_", "hour_of_day", "day_of_week",
)


# =============================================================================
# PUBLIC API
# =============================================================================

def build_oscillator_masks(
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Classify columns by name → (osc_mask, passthrough_mask).

    Parameters
    ----------
    feature_names : Column names in the same order as the feature matrix.

    Returns
    -------
    osc_mask         : bool[n_features] — True ⇒ column undergoes SymQS.
    passthrough_mask : bool[n_features] — True ⇒ binary/discrete column
                       that must NOT be rescaled by SymQS.
    """
    n = len(feature_names)
    osc         = np.zeros(n, dtype=bool)
    passthrough = np.zeros(n, dtype=bool)
    if n == 0:
        return osc, passthrough

    lowered = [str(name).lower() for name in feature_names]
    for j, lname in enumerate(lowered):
        if any(tok in lname for tok in _BINARY_PASSTHROUGH_TOKENS):
            passthrough[j] = True
            continue
        if any(tok in lname for tok in _OSCILLATOR_TOKENS):
            osc[j] = True

    return osc, passthrough


def apply_symqs_overlay(
    X_full_scaled: np.ndarray,
    X_full_raw: np.ndarray,
    feature_names: list[str],
    train_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Overwrite oscillator columns with SymmetricQuantileScaler output.

    All paths produce an ``np.ndarray`` with the same dtype and shape as
    ``X_full_scaled``.  Falls back to a no-op when quant_architect is not
    available (legacy behaviour preserved).

    Parameters
    ----------
    X_full_scaled : Matrix after RollingRobustScaler has been applied.
    X_full_raw    : Same matrix BEFORE scaling — source for SymQS fit.
    feature_names : Column names in order.
    train_indices : CHIEF AUDIT 2026-05-23 (P2.2) — integer indices of the
        training rows within X_full_raw.  When provided, SymQS is FIT on
        only those rows (causal, no leakage across CPCV folds) and then
        TRANSFORMED on the full matrix.  When None (legacy), fit_transform
        on the full X — distributional leakage at regime-shift boundaries.
        Always pass train_indices when calling inside a CPCV fold loop.

    Returns
    -------
    np.ndarray
        New matrix where only oscillator columns have been SymQS-rescaled;
        all other columns come unmodified from ``X_full_scaled``.
    """
    # Phase 0 stap 5: SymmetricQuantileScaler leeft in tradebot.train.quant_arch,
    # niet in het niet-bestaande pakket `quant_architect`. De oude except-tak zette
    # _qarch_ok=False en gaf X_full_scaled ONGEWIJZIGD terug: de oscillator-kolommen
    # werden dan nooit symmetrisch geschaald, zonder enige melding.
    from ..train.quant_arch import SymmetricQuantileScaler

    if X_full_scaled.shape[1] == 0:
        return X_full_scaled

    osc_mask, passthrough_mask = build_oscillator_masks(feature_names)
    if not bool(osc_mask.any()):
        return X_full_scaled

    out     = X_full_scaled.copy()
    osc_idx = np.where(osc_mask)[0]
    osc_pass = passthrough_mask[osc_idx]
    sub_raw  = np.asarray(X_full_raw[:, osc_idx], dtype=np.float64)
    if sub_raw.size == 0:
        return out

    sym_scaler = SymmetricQuantileScaler(
        n_bins=21,
        passthrough_mask=osc_pass.tolist(),
    )
    if train_indices is not None:
        # Causal path: fit on training rows only, transform full matrix.
        # Prevents distributional leakage: test-fold quantile boundaries
        # are never seen during the fit phase.
        sub_train = sub_raw[train_indices]
        sym_scaler.fit(sub_train)
        sub_scaled = sym_scaler.transform(sub_raw)
    else:
        # Phase 0: het legacy-pad deed fit_transform op de VOLLEDIGE X. Dat is
        # distributionele leakage over foldgrenzen heen: de quantiel-grenzen van
        # de testfold worden meegenomen in de fit. Het pad waarschuwde alleen en
        # leverde vervolgens gewoon een lekkende matrix op. Dat is geen
        # implementatiedetail maar een lookahead-lek (audit sectie 7.2).
        raise CausalityViolationError(
            "apply_symqs_overlay() zonder train_indices fit de "
            "SymmetricQuantileScaler op de volledige sample, inclusief de "
            "testfold. Geef train_indices=fold_train_idx mee zodat de fit "
            "uitsluitend op trainingsrijen plaatsvindt."
        )

    out[:, osc_idx] = sub_scaled.astype(out.dtype, copy=False)
    logger.info(
        "SymQS overlay applied to %d oscillator column(s) "
        "(%d passthrough, %d non-oscillator).",
        int(osc_mask.sum()),
        int(passthrough_mask.sum()),
        int((~osc_mask & ~passthrough_mask).sum()),
    )
    return out


# ── Backward-compat aliases (leading-underscore monolith names) ───────────────
_build_oscillator_masks = build_oscillator_masks
_apply_symqs_overlay    = apply_symqs_overlay


# ── Re-export RollingRobustScaler (single import path) ────────────────────────
# Phase 0: de oude try/except zette RollingRobustScaler op None wanneer de import
# faalde. Elke aanroeper kreeg dan een TypeError diep in de pipeline in plaats van
# een duidelijke ImportError op de importsite. Dit is een interne module binnen
# hetzelfde pakket; hij kan niet legitiem ontbreken.
from ..train._scalers import RollingRobustScaler  # noqa: F401
