# src/tradebot/train/calibration.py
"""Per-path Platt calibration for CPCV probability estimates.

Migrated from quant_architect.py (audit-item II.3).

PathSpecificPlattCalibrator fits one (A, B) sigmoid pair per CPCV path
rather than a single global sigmoid.  Weighting uses:

    w_t = exp(-lambda * (T - t) / T)  *  (1 / max(sigma_t, eps))

so that recent high-volatility samples receive the lowest weight and recent
low-volatility samples receive the highest weight.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


__all__ = ["PathPlattParams", "PathSpecificPlattCalibrator"]


@dataclass
class PathPlattParams:
    """Per-pad sigmoid-parameters opgeslagen na fit.

    P(y=1 | f) = 1 / (1 + exp(A·f + B))
    """

    A: float = 0.0
    B: float = 0.0
    n_samples: int = 0
    pos_rate: float = 0.5


class PathSpecificPlattCalibrator:
    """Weighted Platt scaling **per CPCV-pad** (audit-item II.3).

    Standaard Platt fit één globale (A, B) over alle CPCV-folds samen — en
    middelt daarmee de unieke regime-blootstelling van elk pad uit. In de
    praktijk is iedere CPCV-pad een afzonderlijke "what-if" trajectorie met
    eigen klasse-prior en eigen vol-regime. Wij fitten ``(A_path, B_path)``
    onafhankelijk per pad en gebruiken Recency-+Volatility-Adjusted gewichten.

    Wegingsfunctie:
        w_t = exp(-λ · (T - t) / T)  ·  (1 / max(σ_t, ε))

    De recency-decay lambda staat default op 1.0 (zachte voorkeur recente
    samples) en kan via ``recency_lambda`` aangepast worden. ``vol_floor``
    voorkomt division-by-zero bij vlakke regimes.

    Inference-tijd: bij prediction op pad ``p`` toepasen ``predict_proba(scores, p)``
    — als ``p`` onbekend is, valt de calibrator terug op een globale fit.

    Args:
        recency_lambda: decay-rate van het recency-gewicht.
        vol_floor:      minimale σ in de vol-adjustment (anti zero-divide).
        max_iter:       max Newton-Raphson iteraties per pad.
        tol:            convergentie-tolerantie op ‖∇‖₂.
    """

    def __init__(
        self,
        recency_lambda: float = 1.0,
        vol_floor: float = 1e-4,
        max_iter: int = 50,
        tol: float = 1e-7,
        input_is_probability: bool = True,
    ) -> None:
        self.recency_lambda: float = float(recency_lambda)
        self.vol_floor: float = float(vol_floor)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        # CHIEF AUDIT 2026-05-28 (L-1 fix): Platt scaling is defined on real-
        # valued decision scores f ∈ (-∞, ∞).  Both the training pipeline and
        # the live ModelSignal feed *probabilities* (∈ (0, 1)) as ``scores``.
        # Fitting a sigmoid 1/(1+exp(A·f+B)) on a feature compressed into [0, 1]
        # and clustered near the base rate produces a degenerate (A, B) that
        # maps every live input to a near-constant ~0.72 — the silent killer
        # that destroyed live directional discrimination.  When
        # ``input_is_probability`` is True we map p → logit(p) before fitting
        # and before inference, restoring the proper Platt domain and making
        # the calibration monotone and well-conditioned.
        self.input_is_probability: bool = bool(input_is_probability)
        self._params: dict[int, PathPlattParams] = {}
        self._global: PathPlattParams = PathPlattParams()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        """Stable logit: log(p / (1 - p)) with clipping to avoid ±inf."""
        pc = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        return np.log(pc / (1.0 - pc))

    def _to_score(self, f: np.ndarray) -> np.ndarray:
        """Map raw input to a Platt-domain score (logit when input is a prob)."""
        if self.input_is_probability:
            return self._logit(f)
        return np.asarray(f, dtype=np.float64)

    # ------------------------------------------------------------------
    @staticmethod
    def _newton_platt(
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        max_iter: int,
        tol: float,
    ) -> tuple[float, float]:
        """Numerieke MAP-fit van (A, B) met Newton-Raphson.

        Negative log-likelihood (Bishop 2006, eq. 4.90):
            L = Σ w_i · [t_i · log(p_i) + (1 - t_i) · log(1 - p_i)]
        waarbij t_i het Platt-pseudo-target is (afhankelijk van pos/neg-rate).
        """
        f = scores.astype(np.float64).flatten()
        y = labels.astype(np.float64).flatten()
        w = weights.astype(np.float64).flatten()
        if f.size == 0:
            return 0.0, 0.0

        n_pos = float(np.sum(w * y))
        n_neg = float(np.sum(w * (1.0 - y)))
        if n_pos <= 0.0 or n_neg <= 0.0:
            # Eén-klasse fold: terugvallen op een neutrale identity-mapping.
            return -1.0, 0.0

        hi = (n_pos + 1.0) / (n_pos + 2.0)
        lo = 1.0 / (n_neg + 2.0)
        t = np.where(y > 0.5, hi, lo)

        A = 0.0
        B = math.log((n_neg + 1.0) / (n_pos + 1.0))

        for _it in range(max_iter):
            fApB = A * f + B
            # Numerically stable sigmoid — bewaar als ndarray voor type-clarity.
            p_arr: np.ndarray = np.where(
                fApB >= 0.0,
                1.0 / (1.0 + np.exp(-fApB)),
                np.exp(fApB) / (1.0 + np.exp(fApB)),
            )
            p = p_arr
            # Gradient
            d = p - t
            g1 = float(np.sum(w * d * f))
            g2 = float(np.sum(w * d))
            # Hessian (positive definite)
            h_diag = p * (1.0 - p) * w
            h11 = float(np.sum(h_diag * f * f)) + 1e-12
            h22 = float(np.sum(h_diag)) + 1e-12
            h12 = float(np.sum(h_diag * f))

            det = h11 * h22 - h12 * h12
            if abs(det) < 1e-18:
                break

            dA = -(h22 * g1 - h12 * g2) / det
            dB = -(-h12 * g1 + h11 * g2) / det
            A_new = A + dA
            B_new = B + dB
            if math.sqrt(g1 * g1 + g2 * g2) < tol:
                A, B = A_new, B_new
                break
            A, B = A_new, B_new

        return float(A), float(B)

    # ------------------------------------------------------------------
    def fit_per_path(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        path_ids: np.ndarray,
        bar_volatility: np.ndarray | None = None,
    ) -> PathSpecificPlattCalibrator:
        """Fit één (A, B) per uniek pad-id.

        Args:
            scores:   ongekalibreerde model-output f(x), shape (T,).
            labels:   binaire targets y ∈ {0, 1}, shape (T,).
            path_ids: int array met CPCV-pad-id per sample, shape (T,).
            bar_volatility: optionele σ_t per sample voor vol-weging; None ⇒
                            uniforme vol.

        Returns:
            self.
        """
        f = np.asarray(scores, dtype=np.float64).flatten()
        y = np.asarray(labels, dtype=np.float64).flatten()
        p_ids = np.asarray(path_ids, dtype=np.int64).flatten()
        if f.size != y.size or f.size != p_ids.size:
            raise ValueError(
                f"PathSpecificPlatt: shape-mismatch — scores={f.size}, "
                f"labels={y.size}, paths={p_ids.size}."
            )
        if f.size == 0:
            self._fitted = True
            return self

        # L-1 fix: map probability inputs into the Platt score domain (logit)
        # so the sigmoid fit is well-conditioned and monotone.
        f = self._to_score(f)

        if bar_volatility is None:
            sigma = np.ones_like(f)
        else:
            sigma = np.asarray(bar_volatility, dtype=np.float64).flatten()
            if sigma.size != f.size:
                raise ValueError("bar_volatility moet zelfde lengte als scores hebben.")

        # Globale fallback fit
        T = float(f.size)
        idx = np.arange(f.size, dtype=np.float64)
        recency_w_global = np.exp(-self.recency_lambda * (T - 1.0 - idx) / max(T, 1.0))
        vol_w_global = 1.0 / np.maximum(sigma, self.vol_floor)
        w_global = recency_w_global * vol_w_global
        A_g, B_g = self._newton_platt(
            f, y, w_global, self.max_iter, self.tol
        )
        A_g, B_g = self._guard_params(A_g, B_g, "global")
        self._global = PathPlattParams(
            A=A_g, B=B_g, n_samples=int(f.size), pos_rate=float(np.mean(y))
        )

        # Per-pad fit
        unique_paths = np.unique(p_ids)
        self._params = {}
        for path in unique_paths:
            mask = p_ids == path
            if not mask.any():
                continue
            f_p = f[mask]
            y_p = y[mask]
            sig_p = sigma[mask]
            local_idx = np.arange(f_p.size, dtype=np.float64)
            T_p = float(f_p.size)
            recency_w = np.exp(
                -self.recency_lambda * (T_p - 1.0 - local_idx) / max(T_p, 1.0)
            )
            vol_w = 1.0 / np.maximum(sig_p, self.vol_floor)
            w = recency_w * vol_w
            A_p, B_p = self._newton_platt(
                f_p, y_p, w, self.max_iter, self.tol
            )
            A_p, B_p = self._guard_params(A_p, B_p, f"path {int(path)}")
            self._params[int(path)] = PathPlattParams(
                A=A_p,
                B=B_p,
                n_samples=int(f_p.size),
                pos_rate=float(np.mean(y_p)) if y_p.size > 0 else 0.5,
            )

        self._fitted = True
        logger.info(
            "PathSpecificPlattCalibrator: gefit op %d paden (globale prior=%.3f).",
            len(self._params), self._global.pos_rate,
        )
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _guard_params(A: float, B: float, name: str) -> tuple[float, float]:
        """Reject inverted / degenerate Platt fits → fall back to identity.

        ``_newton_platt`` fits p = sigmoid(A·f + B), so the calibrated
        probability is monotone *increasing* in the score f iff A > 0.  A fit
        with A <= 0 means higher model confidence maps to an equal-or-LOWER
        calibrated probability — an inverted/flat calibrator that silently
        flips or kills the signal (the live ~0.72 failure mode).  In that case
        we fall back to the logit-domain identity (A=1, B=0 ⇒ predict_proba
        returns the input probability unchanged), which is safe: the live
        signal then reflects the raw model prob and is gated by min_conf
        rather than a broken transform.
        """
        _MIN_SLOPE = 1e-3
        if not np.isfinite(A) or not np.isfinite(B) or A < _MIN_SLOPE:
            logger.warning(
                "PathSpecificPlatt[%s]: degenerate/inverted fit (A=%.4g, B=%.4g) "
                "— falling back to identity calibration (raw prob passthrough).",
                name, A, B,
            )
            return 1.0, 0.0
        return float(A), float(B)

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        path_ids: np.ndarray,
        bar_volatility: np.ndarray | None = None,
    ) -> PathSpecificPlattCalibrator:
        """Alias for fit_per_path — matches sklearn-style API."""
        return self.fit_per_path(scores, labels, path_ids, bar_volatility)

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        scores: np.ndarray,
        path_id: int | None = None,
    ) -> np.ndarray:
        """Calibreer scores → kansen.

        Args:
            scores:  ongekalibreerde model-output, shape (n,).
            path_id: pad-id; None of onbekend ⇒ globale (A, B).

        Returns:
            float64 array van kansen ∈ (0, 1), shape (n,).
        """
        if not self._fitted:
            raise RuntimeError("PathSpecificPlattCalibrator niet gefit.")
        params = self._params.get(int(path_id)) if path_id is not None else None
        if params is None:
            params = self._global

        # L-1 fix: identical score domain as fit() (logit when input is a prob).
        f = self._to_score(np.asarray(scores, dtype=np.float64).flatten())
        z = params.A * f + params.B
        # CHIEF AUDIT 2026-05-28 (L-1 ROOT CAUSE): ``_newton_platt`` fits the
        # model p = sigmoid(A·f + B) = sigmoid(z).  The previous implementation
        # returned sigmoid(-z) here, i.e. ≈ (1 − p_fit) — a systematic SIGN
        # INVERSION.  That is precisely why a raw model prob of 0.29 mapped to
        # ~0.72 (≈ 1 − 0.29) live, with both LONG and SHORT clearing the gate.
        # predict_proba must use the SAME sign as the fit objective: p = σ(z).
        out = np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        return np.clip(out, 1e-7, 1.0 - 1e-7)
