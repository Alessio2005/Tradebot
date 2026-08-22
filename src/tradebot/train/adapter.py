"""adapter.py — Thin wrapper adapting raw CatBoostClassifier to the predict_binary
interface expected by ContextualBanditEnsemble.

Kept in a dedicated module so joblib can deserialize pickled ensembles from any
entrypoint (train_cpcv, backtest_portfolio, live agent) without ImportError.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class CatBoostModelAdapter:
    """Wraps a raw CatBoostClassifier and exposes the predict_binary / _align_input_custom
    interface that ContextualBanditEnsemble.predict_greybox_strategy requires."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.min_conf_long  = getattr(model, "min_conf_long",  0.50)
        self.min_conf_short = getattr(model, "min_conf_short", 0.99)

    def predict_binary(
        self,
        x_micro: Any,
        x_meso: Any = None,
        x_macro: Any = None,
        base_feature_map: Dict | None = None,
        is_pre_scaled: bool = False,
    ) -> Dict[str, float]:
        parts = []
        for x in (x_micro, x_meso, x_macro):
            if x is not None:
                arr = np.asarray(x)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim == 2 and arr.shape[1] > 0:
                    parts.append(arr.astype(np.float32))
        if not parts:
            return {"prob_win": 0.0, "prob_loss": 1.0}
        X = np.hstack(parts)
        try:
            probs = self.model.predict_proba(X)[0]
            prob_win = float(probs[1]) if len(probs) > 1 else float(probs[0])
            return {"prob_win": prob_win, "prob_loss": 1.0 - prob_win}
        except Exception:
            return {"prob_win": 0.0, "prob_loss": 1.0}

    def _align_input_custom(self, arr: Any, target_feats: List, timeframe: str) -> Any:
        return arr
