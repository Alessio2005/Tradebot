"""Volatility estimator suite (AFML §3)."""
from .ewma import (
    ewma_variance_causal,
    ewma_volatility,
    ewma_volatility_panel,
    get_ewma_volatility,  # GEDEPRECIEERD: crasht
)
from .garman_klass import (
    get_garman_klass_volatility,
    get_jump_adjusted_volatility,
    gt_garman_klass_volatilility,  # backward-compat alias  # noqa: F401
)
from .parkinson import get_parkinson_volatility
from .rogers_satchell import get_rogers_satchell_volatility
from .yang_zhang import get_yang_zhang_volatility

__all__ = [
    "ewma_variance_causal",
    "ewma_volatility",
    "ewma_volatility_panel",
    "get_ewma_volatility",
    "get_garman_klass_volatility",
    "get_jump_adjusted_volatility",
    "get_parkinson_volatility",
    "get_rogers_satchell_volatility",
    "get_yang_zhang_volatility",
]
