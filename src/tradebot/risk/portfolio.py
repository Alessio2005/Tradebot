# src/tradebot/risk/portfolio.py
"""Cross-asset portfolio risk manager (vol-target, corr-scaler, caps, DD breaker).

Migrated from portfolio_risk.py (root-level legacy).

Classes / Functions
-------------------
RiskState           : snapshot of all risk metrics at one bar.
SizingDecision      : leverage decision with diagnostics.
effective_n_assets  : number of effective independent assets (Herfindahl).
PortfolioRiskManager: stateful rolling-window risk manager.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Phase 0: `..execution.market_impact` is een INTERNE module binnen dit pakket en
# kan niet legitiem ontbreken. De try/except zette de vlag _MARKET_IMPACT_AVAILABLE
# op False, waarna het model stilzwijgend zonder de betreffende correctie draaide.
from ..execution.market_impact import (
    ledoit_wolf_shrunk_corr,
    negative_skew_crisis_multiplier,
)

logger = logging.getLogger(__name__)




# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class RiskState:
    """Snapshot van alle risk-metrics op één bar.

    Wordt door de backtester per bar gelogd → equity-curve diagnostics.
    """
    timestamp: pd.Timestamp | None = None
    equity: float = 1.0
    peak_equity: float = 1.0
    drawdown: float = 0.0
    realized_vol: float = 0.0           # Annualized portfolio vol (rolling)
    avg_pairwise_corr: float = 0.0
    vol_target_multiplier: float = 1.0
    corr_multiplier: float = 1.0
    dd_breaker_active: bool = False
    gross_leverage: float = 0.0
    net_leverage: float = 0.0


@dataclass
class SizingDecision:
    """Resultaat van size_position()."""
    raw_leverage: float                  # Vóór alle caps
    scaled_leverage: float               # Na vol+corr+DD scaler
    final_leverage: float                # Na exposure caps (= echte fillsize)
    blocked: bool = False                # True als DD-breaker actief is
    reason: str = ""


# =============================================================================
# CORE COVARIANCE / CORRELATION HELPERS
# =============================================================================
def _safe_corr(
    returns: np.ndarray,
    min_obs: int = 20,
    ewma_lambda: float = 0.94,
    *,
    use_shrinkage: bool = True,
    shrinkage_target: str = "const_corr",
) -> np.ndarray:
    """Correlatie-matrix met EWMA-gewogen covariantie (RiskMetrics λ=0.94).

    AUDIT-FIX (N22 — pairwise complete observations):
      Pairwise complete observaties worden gehandhaafd: elk asset-paar gebruikt
      alleen de gemeenschappelijke niet-NaN bars.

    AUDIT-FIX (Round 3 — EWMA covariance, λ=0.94):
      Vervangt rolling Pearson (gelijke gewichten over venster) door exponentieel
      gewogen covariantie (RiskMetrics standaard). Voordelen:
        • Reageert 5-10× sneller op regime-omslag (crypto panic, flash-crash);
        • Lagere asymptotische overschatting van correlatie na reset;
        • Stabiel voor sparse NaN-perioden (gewicht valt snel weg).
      λ=0.94 ≡ half-life ≈ 11 bars (standaard Bybit 15-min bars).
      Zet ewma_lambda=1.0 om terug te vallen op het uniforme Pearson-gedrag.

    Args:
        returns:      shape (T, N). Mag NaN bevatten.
        min_obs:      minimum aantal geldige observaties per paar (NaN-guard).
        ewma_lambda:  vervalconstante 0 < λ ≤ 1 (0.94 = RiskMetrics default).

    Returns:
        (N, N) correlatie-matrix in [-1, 1]. Identity als T te klein.
    """
    if returns.ndim != 2 or returns.shape[0] < min_obs or returns.shape[1] == 0:
        n = returns.shape[1] if returns.ndim == 2 else 0
        return np.eye(max(n, 1))

    # AUDIT-FIX (Issue 6 — Ledoit-Wolf shrinkage):
    #   Voor multi-asset (N ≥ 2) en wanneer market_impact-module geladen is,
    #   schakelen we standaard over op LW-shrunk corr.  Dit dempt de
    #   correlatie-explosie die EWMA pas reproduceert na ~λ-decay-bars
    #   (te traag bij flash-crashes).
    if use_shrinkage and returns.shape[1] >= 2:
        try:
            corr_lw, _delta = ledoit_wolf_shrunk_corr(
                returns,
                target=shrinkage_target,
                min_obs=min_obs,
            )
            return corr_lw
        except Exception:  # pragma: no cover
            # Numeriek of import-issue → val terug op EWMA-pad hieronder.
            pass

    lam = float(np.clip(ewma_lambda, 0.0, 1.0))

    try:
        n_t, n_assets = returns.shape

        if lam >= 1.0 - 1e-9:
            # Fallback: gelijke gewichten = origineel Pearson pairwise gedrag (N22)
            df = pd.DataFrame(returns)
            corr_df = df.corr(method="pearson", min_periods=min_obs).fillna(0.0)
            corr = corr_df.to_numpy(dtype=np.float64)
            if corr.ndim == 0:
                return np.array([[1.0]])
            corr = np.where(np.isfinite(corr), corr, 0.0)
            np.fill_diagonal(corr, 1.0)
            return corr

        # ── EWMA gewogen covariantie ─────────────────────────────────────────
        # Gewichten: w_t ∝ λ^(T-1-t), nieuwste bar krijgt gewicht λ^0 = 1.
        # Normalisering zodat Σw=1 → gewogen covariantie is schaalvrij.
        raw_w: np.ndarray = np.power(
            lam, np.arange(n_t - 1, -1, -1, dtype=np.float64)
        )  # shape (T,): [λ^(T-1), λ^(T-2), ..., λ^0]
        raw_w /= raw_w.sum()  # normaliseer

        # NaN-masker: behandel NaN-bars per asset als missing weight.
        # Pairwise: cov[i,j] gebruikt alleen bars waar BEIDE assets geldig zijn.
        valid = np.isfinite(returns)  # (T, N) bool

        cov = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(i, n_assets):
                mask_ij = valid[:, i] & valid[:, j]
                if mask_ij.sum() < min_obs:
                    # Minder dan min_obs gemeenschappelijke obs → behandel als ongecorr.
                    cov[i, j] = 0.0
                    cov[j, i] = 0.0
                    if i == j:
                        cov[i, i] = 1.0
                    continue
                w_ij = raw_w * mask_ij.astype(np.float64)
                w_sum = w_ij.sum()
                if w_sum < 1e-15:
                    cov[i, j] = 0.0
                    cov[j, i] = 0.0
                    if i == j:
                        cov[i, i] = 1.0
                    continue
                w_ij_norm = w_ij / w_sum
                mu_i = float(np.dot(w_ij_norm, returns[:, i]))
                mu_j = float(np.dot(w_ij_norm, returns[:, j]))
                r_i = returns[:, i] - mu_i
                r_j = returns[:, j] - mu_j
                c_ij = float(np.dot(w_ij_norm, r_i * r_j))
                cov[i, j] = c_ij
                cov[j, i] = c_ij

        # Converteer covariantie → correlatie
        std_arr = np.sqrt(np.diag(cov))
        std_safe = np.where(std_arr > 1e-9, std_arr, 1.0)
        corr = cov / np.outer(std_safe, std_safe)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    except Exception:
        return np.eye(returns.shape[1])


def _avg_offdiag(corr: np.ndarray) -> float:
    """Gemiddelde van de off-diagonal elementen (de echte cross-asset correlatie)."""
    n = corr.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    vals = corr[mask]
    if vals.size == 0:
        return 0.0
    return float(np.mean(np.abs(vals)))


def _dual_window_ewma_corr(
    returns_df: pd.DataFrame,
    fast_window: int = 50,
    slow_window: int = 500,
    blend_weight: float = 0.3,
) -> np.ndarray:
    """Dual-window EWMA correlation: fast reacts to regime shifts (Wave 16).

    Blends fast (50-bar) and slow (500-bar) EWMA correlations.
    In crisis: fast dominates; in calm: slow stabilizes.
    """
    fast_cov = returns_df.ewm(span=fast_window, adjust=False).cov().iloc[-len(returns_df.columns):]
    slow_cov = returns_df.ewm(span=slow_window, adjust=False).cov().iloc[-len(returns_df.columns):]

    def cov_to_corr(cov_df: pd.DataFrame) -> np.ndarray:
        cov = cov_df.values
        std = np.sqrt(np.diag(cov))
        std = np.where(std > 0, std, 1.0)
        return cov / np.outer(std, std)

    fast_corr = cov_to_corr(fast_cov)
    slow_corr = cov_to_corr(slow_cov)
    blended = blend_weight * fast_corr + (1.0 - blend_weight) * slow_corr
    return blended


def effective_n_assets(corr: np.ndarray) -> float:
    """Effectief aantal onafhankelijke assets (Bouchaud-style).

    N_eff = (Σ λᵢ)² / Σ λᵢ²  waarbij λᵢ de eigenvalues van corr zijn.
    Bij identity (geen correlatie): N_eff = N. Bij volledig gecorreleerd: N_eff → 1.
    """
    if corr.shape[0] < 1:
        return 1.0
    try:
        # Symmetriseer voor eigenvalsh stabiliteit
        c = 0.5 * (corr + corr.T)
        evals = np.linalg.eigvalsh(c)
        evals = np.maximum(evals, 1e-10)
        s1 = float(evals.sum())
        s2 = float((evals ** 2).sum())
        if s2 <= 0:
            return float(corr.shape[0])
        return s1 * s1 / s2
    except Exception:
        return float(corr.shape[0])


# =============================================================================
# PORTFOLIO RISK MANAGER
# =============================================================================
class PortfolioRiskManager:
    """Stateful cross-asset risk manager.

    Gebruik:
        prm = PortfolioRiskManager(symbols=["BTCUSDT","ETHUSDT","SOLUSDT"], ...)
        for bar_t in range(T):
            # 1. Voor elke asset: vraag een leverage-decision
            d_btc = prm.size_position("BTCUSDT", raw_leverage=1.5, side=+1)
            ...
            # 2. Aan einde bar: feed realized returns
            prm.update(returns_per_asset={"BTCUSDT": r_btc, ...},
                       active_signed_exposure={"BTCUSDT": d_btc.final_leverage, ...})

    De manager is bewust async-vrij en heeft geen globale state — kan veilig
    in unit-tests gemockt worden.
    """

    def __init__(
        self,
        symbols: list[str],
        target_annual_vol: float = 0.12,
        vol_window_bars: int = 500,
        corr_window_bars: int = 1000,
        fast_corr_window_bars: int | None = 50,  # Wave 16 P0-17: dual-window EWMA corr default ON
        max_avg_pairwise_corr: float = 0.85,
        max_gross_leverage: float = 4.0,
        max_net_leverage: float = 2.5,
        max_per_asset_leverage: float = 2.0,
        dd_breaker_threshold: float = 0.20,
        dd_resume_threshold: float = 0.10,
        dd_breaker_lookback_bars: int = 5000,
        # Wave 16 P0-19: kelly_fraction=None means no additional portfolio-level
        # fraction is applied — kelly.py's own divisor (now 1/6) already handles
        # the fractional sizing. Set to a float in (0, 1] only when raw_leverage
        # comes from a naive Kelly without CVaR/gap-risk stress.
        kelly_fraction: float | None = None,
        starting_equity: float = 1.0,
        bars_per_year: float = 365.0 * 24 * 12,    # 5min crypto bars; recalibreer indien anders
        # G4 — ADV cap: maximum position notional as fraction of estimated daily volume.
        # None = no cap (default; appropriate when equity << daily ADV).
        # Set to e.g. 0.01 (1% of ADV) when running at > $500k equity to prevent
        # market impact. The caller is responsible for providing adv_per_asset_usd
        # at size_position() call time (via per_asset_cap_override or future ADV API).
        # At current scale (avg_gross_lev=6.5%, equity~$100k) this cap is not binding.
        max_adv_fraction: float | None = None,
    ) -> None:
        self.symbols: list[str] = list(symbols)
        self.target_annual_vol: float = float(target_annual_vol)
        self.vol_window_bars: int = int(vol_window_bars)
        self.corr_window_bars: int = int(corr_window_bars)
        # AUDIT-FIX (issue #6 — N_eff Correlation Scaling):
        #   corr_window_bars=1000 bij 5-min bars = 83h lag → te traag voor
        #   crypto-paniek.  fast_corr_window_bars biedt een sneller venster
        #   (bv. 100 bars = ~8h) dat parallel aan het lange venster wordt
        #   berekend.  De _correlation_metrics() geeft altijd het meest
        #   conservatieve resultaat (laagste N_eff) van beide vensters.
        #   Aanbeveling per asset:
        #     BTC/ETH : fast_corr_window_bars=200  (~16h bij 5min bars)
        #     SOL     : fast_corr_window_bars=100  (~8h  bij 5min bars)
        self.fast_corr_window_bars: int | None = (
            int(fast_corr_window_bars) if fast_corr_window_bars is not None else None
        )
        self.max_avg_pairwise_corr: float = float(max_avg_pairwise_corr)
        self.max_gross_leverage: float = float(max_gross_leverage)
        self.max_net_leverage: float = float(max_net_leverage)
        self.max_per_asset_leverage: float = float(max_per_asset_leverage)
        self.dd_breaker_threshold: float = float(dd_breaker_threshold)
        self.dd_resume_threshold: float = float(dd_resume_threshold)
        self.dd_breaker_lookback_bars: int = int(dd_breaker_lookback_bars)
        # Wave 16 P0-19: None → 1.0 (pass-through; kelly.py divisor already applied)
        self.kelly_fraction: float = (
            1.0 if kelly_fraction is None else float(np.clip(kelly_fraction, 0.0, 1.0))
        )
        self.bars_per_year: float = float(bars_per_year)
        # G4 ADV cap: stored for future use; enforcement requires caller to supply
        # adv_per_asset_usd at size_position() time (not yet wired in backtest).
        self.max_adv_fraction: float | None = (
            float(max_adv_fraction) if max_adv_fraction is not None else None
        )

        # State
        self.equity: float = float(starting_equity)
        self.peak_equity: float = float(starting_equity)
        self.dd_breaker_active: bool = False
        self.equity_curve: list[float] = [self.equity]
        self.timestamps: list[pd.Timestamp] = []

        # AUDIT-FIX (Issue 6 — Crisis Multiplier hysterese-state):
        # Wanneer BTC's rolling skewness onder ``_skew_panic`` valt, wordt
        # ``_crisis_state_active`` True → vol-target multiplier × stress-factor.
        # Resume pas zodra skew boven ``_skew_resume`` herstelt.
        #
        # CRISIS-HYSTERESIS-FIX (Silent Killer — 2026-05-18):
        #   Oude thresholds (_skew_panic=-1.5, _skew_resume=-0.5) lagen binnen
        #   het normale fluctuatiebereik van de 500-bar rolling skew van 1h BTC
        #   returns (typisch [-2.0, +2.0] in niet-crisis perioden).  Gevolg:
        #   crisis-multiplier togglede op ~50% van alle bars heen-en-weer
        #   (ACTIVE→RESUMED→ACTIVE elke bar) → timing-asymmetrische leverage-
        #   suppressie → structureel lagere Sharpe door false positives.
        #
        #   Fix: drempels verplaatst naar respectievelijk -2.0 en +0.5, een
        #   hysteresiskloof van 2.5 i.p.v. 1.0.  De multiplier activeert nu
        #   alleen bij echte crash-skew (< -2.0), zoals COVID-crash (mrt 2020),
        #   LUNA-collapse (mei 2021) of FTX-implosie (nov 2022).  Normale
        #   bear-market left-skew triggert niet meer.
        self._crisis_state_active: bool = False
        self._skew_panic: float = -2.0   # was -1.5: te gevoelig → 50% false-positive rate
        self._skew_resume: float = 0.5   # was -0.5: hysteresiskloof 1.0 → 2.5 (breder)
        self._crisis_mult_active: float = 1.5  # extra cov-stress in panic
        self._crisis_anchor_symbol: str = "BTCUSDT"  # bron voor skew-meting

        # Rolling per-asset returns (deque-equivalent via list slice)
        self._returns_history: dict[str, list[float]] = {s: [] for s in self.symbols}
        # Aktive signed exposures (laatste bekende per-asset weight)
        self._active_signed: dict[str, float] = {s: 0.0 for s in self.symbols}
        # GK-vol tracking (optioneel) — gevuld via update(gk_vol_per_asset=...).
        # Garman-Klass reageert direct op intrabar H/L; std() ijlt gemiddeld
        # 3-5 bars na. Cruciaal voor vol-targeting in crypto: we willen leverage
        # terugschroeven vóór een crash de equity raakt, niet erna.
        self._gk_vol_history: dict[str, list[float]] = {s: [] for s in self.symbols}
        self._use_gk_vol: bool = False   # wordt True zodra eerste GK-waarden binnenkomen

        # v3 T1.4: HMM regime detector voor LONG-positie capping in Bear-regime
        self._hmm_detector: Optional[object] = None
        self._hmm_returns_buffer: list = []   # recente returns voor regime-detectie
        self._hmm_buffer_max: int = 500        # max 500 bars in buffer

        # v3 T1.3: portfolio-level CVaR scaling
        self._cvar_scale_enabled: bool = False
        self._asset_cvar: dict = {}
        self._baseline_cvar: float = 0.01   # 1% baseline risk

    # -----------------------------------------------------------------------
    # ROLLING METRICS
    # -----------------------------------------------------------------------
    def _returns_matrix(self) -> np.ndarray:
        """Returns matrix (T, N) voor de rolling window."""
        cols: list[list[float]] = []
        # Laat zoveel rijen als de kortste reeks
        T = min((len(self._returns_history[s]) for s in self.symbols), default=0)
        for s in self.symbols:
            r = self._returns_history[s][-T:] if T > 0 else []
            cols.append(r)
        if T == 0:
            return np.zeros((0, len(self.symbols)))
        arr = np.array(cols, dtype=np.float64).T  # (T, N)
        return arr

    def _realized_portfolio_vol(self) -> float:
        """Annualized realized vol van het *huidige gewogen* portfolio.

        Prioriteit:
          1. Garman-Klass vol (wanneer update() met gk_vol_per_asset gevoerd is).
             GK reageert direct op intrabar H/L → vol-target schroeft leverage
             terug vóór een crash de equity raakt, niet 3-5 bars erna.
          2. Fallback: close-to-close std (origineel gedrag, ijlt na).

        Gewichten: laatste actieve exposures; bij geen positie equal-weighted.
        """
        w = np.array([self._active_signed.get(s, 0.0) for s in self.symbols], dtype=np.float64)
        if np.allclose(w, 0.0):
            w = np.ones(len(self.symbols), dtype=np.float64) / max(len(self.symbols), 1)
        w_abs = np.abs(w)
        w_norm = float(w_abs.sum())

        # ── Pad 1: Garman-Klass weighted portfolio vol ─────────────────────
        # AUDIT-FIX (M4 — Full covariance i.p.v. diagonal approximation):
        #   Oude formule σ_p ≈ Σᵢ|wᵢ|σᵢ ignoreert off-diagonal correlaties.
        #   Bij ρ=0.95 onderschat zij de werkelijke port-vol met factor √1.6 ≈
        #   1.27 → vol-target multiplier scaleert te ver op tijdens correlatie-
        #   crises (precies wanneer je IN moet schalen).
        #   Nieuwe formule: σ_p² = wᵀ Σ w met Σ = D · R · D, waarbij D = diag(σᵢ)
        #   en R uit de rolling pair-correlatie. Wanneer R nog niet beschikbaar
        #   is (cold start) valt code terug op de oude diagonale benadering.
        if self._use_gk_vol:
            gk_spot: list[float] = []
            for s in self.symbols:
                hist = self._gk_vol_history[s]
                # Mediaan over laatste vol_window_bars — één spike-bar mag de
                # leveragebeslissing niet domineren.
                n_hist = min(len(hist), self.vol_window_bars)
                if n_hist > 0:
                    gk_spot.append(float(np.median(hist[-n_hist:])))
                else:
                    gk_spot.append(0.0)
            gk_arr: np.ndarray = np.asarray(gk_spot, dtype=np.float64)

            # Probeer full-Σ pad: wᵀ (D R D) w = (Dw)ᵀ R (Dw).
            sigma_per_bar = 0.0
            ret_full = self._returns_matrix()
            if ret_full.shape[0] >= 30 and ret_full.shape[1] == len(self.symbols):
                ret_window = (
                    ret_full[-self.corr_window_bars:]
                    if ret_full.shape[0] > self.corr_window_bars
                    else ret_full
                )
                R: np.ndarray = _safe_corr(ret_window)
                if R.shape == (len(self.symbols), len(self.symbols)):
                    Dw: np.ndarray = w_abs * gk_arr
                    var_p = float(Dw @ R @ Dw)
                    if var_p > 0.0:
                        # Normaliseer door |w| zodat resultaat per-unit-leverage is,
                        # consistent met de oude diagonal-benadering die door
                        # w_norm deelde.
                        if w_norm > 1e-9:
                            sigma_per_bar = float(np.sqrt(var_p)) / w_norm
                        else:
                            sigma_per_bar = float(np.sqrt(var_p))

            # Fallback op diagonaal als full-Σ niet beschikbaar (cold start)
            if sigma_per_bar <= 1e-9:
                if w_norm > 1e-9:
                    sigma_per_bar = float((w_abs @ gk_arr) / w_norm)
                else:
                    sigma_per_bar = float(np.mean(gk_arr))

            if sigma_per_bar > 1e-9:
                return sigma_per_bar * float(np.sqrt(self.bars_per_year))
            # GK gaf 0 (cold start of alle nul) → fall through naar std

        # ── Pad 2: close-to-close std (fallback) ───────────────────────────
        ret = self._returns_matrix()
        if ret.shape[0] < 30:
            return 0.0
        if ret.shape[0] > self.vol_window_bars:
            ret = ret[-self.vol_window_bars:]
        port_ret = ret @ w
        if port_ret.size < 2:
            return 0.0
        sigma_per_bar = float(np.std(port_ret, ddof=1))
        return sigma_per_bar * float(np.sqrt(self.bars_per_year))

    def _correlation_metrics(self) -> tuple[float, float]:
        """Returns (avg_offdiag_abs_corr, effective_n_assets).

        AUDIT-FIX (issue #6 — fast corr window):
          Berekent correlatie op twee vensters als fast_corr_window_bars is
          ingesteld: het lange venster (corr_window_bars) voor structurele
          correlatie, en het korte venster voor regime-detectie.  Geeft het
          meest conservatieve resultaat (hoogste avg_corr / laagste N_eff)
          zodat de multiplier altijd het worst-case scenario weerspiegelt.
        """
        ret_full = self._returns_matrix()

        def _metrics_from(ret: np.ndarray) -> tuple[float, float]:
            if ret.shape[0] < 30 or ret.shape[1] < 2:
                return 0.0, float(len(self.symbols))
            corr = _safe_corr(ret)
            return _avg_offdiag(corr), effective_n_assets(corr)

        # Lange venster (structureel)
        ret_long = ret_full[-self.corr_window_bars:] if ret_full.shape[0] > self.corr_window_bars else ret_full
        avg_long, neff_long = _metrics_from(ret_long)

        if self.fast_corr_window_bars is not None and ret_full.shape[0] >= self.fast_corr_window_bars:
            ret_fast = ret_full[-self.fast_corr_window_bars:]
            avg_fast, neff_fast = _metrics_from(ret_fast)
            # Conservatief: hoogste correlatie, laagste N_eff
            avg  = max(avg_long,  avg_fast)
            neff = min(neff_long, neff_fast)
        else:
            avg  = avg_long
            neff = neff_long

        return avg, neff

    # -----------------------------------------------------------------------
    # MULTIPLIERS
    # -----------------------------------------------------------------------
    def vol_target_multiplier(self, in_transition: bool = False) -> float:
        """Schaal-factor zodat realized portfolio-vol → target_annual_vol.

        Als realized vol = 0 (cold start) of target_vol = 0: 1.0 (geen scaling).

        AUDIT-FIX (N23 — cap 2.0→2.5 met transitie-guard):
          Oude cap=2.0 schiet te snel vol in calm-before-storm regimes:
          bij realized_vol=0.04 (ultra-rustig) en target=0.12 → mult=3.0
          maar cap fires at 2.0 → leverage blijft ondermaats voor een
          breakout.  Met M4-fix (full-Σ vol) is realized_vol nu hoger →
          cap raakt nog vaker.

          Nieuwe cap: 2.5 normaal, 1.5 als in_transition=True (caller
          kan dit signaleren via feat_bars_since_break < 30).

        Args:
            in_transition: True wanneer een regime-transitie gedetecteerd
                is (bv. feat_bars_since_break < 30).  Verlaagt de cap naar
                1.5× om in snel-bewegende markten leverage-ophoping te
                voorkomen.
        """
        rv = self._realized_portfolio_vol()
        if rv <= 1e-6 or self.target_annual_vol <= 0.0:
            return 1.0

        # AUDIT-FIX (Issue 6 — Crisis Multiplier overlay):
        #   Wanneer BTC's recente skewness diep negatief is (panic/liquidatie),
        #   schalen we de "effectieve" realized vol omhoog → vol-target trekt
        #   leverage instant terug.  Hysterese voorkomt flap rond een drempel.
        crisis_mult, self._crisis_state_active = self._compute_crisis_multiplier()
        rv_eff = rv * float(crisis_mult)

        mult = self.target_annual_vol / rv_eff
        cap = 1.5 if in_transition else 2.5
        return float(np.clip(mult, 0.0, cap))

    # -----------------------------------------------------------------------
    def _compute_crisis_multiplier(self) -> tuple[float, bool]:
        """Bereken stress-overlay multiplier o.b.v. neg-skew van anchor-asset.

        Returns:
            (multiplier, new_state_active) — multiplier ≥ 1.0.
            Wanneer market_impact-module niet beschikbaar is, returnt
            altijd (1.0, False) zodat het gedrag identiek is aan voor
            de fix.
        """
        anchor = self._crisis_anchor_symbol
        if anchor not in self._returns_history:
            # Fallback: gebruik eerste asset uit symbols-lijst
            if not self.symbols:
                return 1.0, False
            anchor = self.symbols[0]

        hist = np.asarray(self._returns_history.get(anchor, []), dtype=np.float64)
        if hist.size < 30:
            return 1.0, False

        window = min(self.vol_window_bars, hist.size)
        # CRISIS-ZERO-PADDING-FIX (Silent Killer — 2026-05-18):
        #   Op de union-grid portfolio-timestamp-reeks wordt _returns_history
        #   voor de anchor-asset gevuld met 0.0 op elke GAP-bar (event-bar van
        #   EEN ANDER asset, waarbij anchor-asset geen signed_return heeft).
        #   In een 500-bar venster met ~80% nullen is σ artificeel laag terwijl
        #   de 3e centrale moment dominanten door slechts enkele grote outliers.
        #   Gevolg: skewness = E[X³]/σ³ → extreme waarden < -2.0 op bijna elke
        #   bar → crisis-multiplier toggled continu (50-100% van de bars).
        #
        #   Fix: filter de nul-retournen eruit vóór de skew-berekening.  Alleen
        #   de eigen CUSUM-event-bars van de anchor-asset (waarbij de prijs
        #   werkelijk bewogen is) tellen mee.  Minimum 30 niet-nul observaties
        #   vereist; anders: 1.0 (geen stress-overlay).
        hist_nonzero = hist[-window:]
        hist_nonzero = hist_nonzero[hist_nonzero != 0.0]
        if hist_nonzero.size < 30:
            return 1.0, self._crisis_state_active
        try:
            mult, new_state = negative_skew_crisis_multiplier(
                hist_nonzero,
                skew_panic=self._skew_panic,
                skew_resume=self._skew_resume,
                state_active=self._crisis_state_active,
                multiplier_active=self._crisis_mult_active,
                multiplier_calm=1.0,
            )
        except Exception:  # pragma: no cover
            return 1.0, self._crisis_state_active
        if new_state and not self._crisis_state_active:
            logger.warning(
                "[Portfolio Risk] Crisis multiplier ACTIVE (%s skew -> panic): "
                "vol-target multiplier x %.2f",
                anchor, float(mult),
            )
        elif self._crisis_state_active and not new_state:
            logger.info(
                "[Portfolio Risk] Crisis multiplier RESUMED (%s skew herstelt).",
                anchor,
            )
        return float(mult), bool(new_state)

    def correlation_multiplier(self) -> float:
        """Trek leverage terug bij hoge cross-asset correlatie.

        Berekening: wanneer alle assets perfect gecorreleerd zijn, hebben we
        N keer hetzelfde risico ipv N onafhankelijke risico's. We schalen
        gross leverage met sqrt(N_eff / N) — dit komt overeen met de claim
        "houd portfolio-vol gelijk alsof we N onafhankelijke assets handelen".

        Daarnaast harde grens op `max_avg_pairwise_corr`: zodra de gemiddelde
        |corr| > drempel, schalen we extra (lineair tot 0.5×).

        AUDIT-FIX (issue #6 — N_eff Panic Response):
          In een crypto-paniek (bv. liquidatie-cascade) loopt de pairwise
          correlatie snel op naar 0.95–0.99.  Op dat moment is N_eff ≈ 1.0:
          er is nog maar één "onafhankelijke" asset.  De oude code liet de
          multiplier dan uitkomen op sqrt(1/N) × 0.5 (bij corr=0.99) → bv.
          sqrt(1/3) × 0.5 ≈ 0.29 voor een 3-asset portfolio.

          Dat is te traag: de floor van 0.5 in de correlatie-throttle beschermde
          nooit helemaal.  Nieuwe logica:
            * N_eff < 1.5  →  return 0.0  (volledig sluiten, crisis-modus)
            * N_eff < 2.0  →  extra lineaire afbouw richting 0
          Dit zorgt dat bij quasi-volledige correlatie de positie-grootte naar
          nul gaat voordat de DD-breaker wordt geraakt.
        """
        avg_corr, n_eff = self._correlation_metrics()
        n = max(len(self.symbols), 1)

        # CRYPTO-RECALIBRATION (M4 — panic threshold for structural high correlations):
        #   Voor N=5 crypto assets is N_eff ≈ 1.29 bij baseline avg_corr=0.85 en
        #   N_eff ≈ 1.08 bij echte paniek (avg_corr=0.95).  De oude formule
        #   0.40×N+0.10 gaf panic=2.10 voor N=5 — altijd actief bij crypto, want
        #   N_eff overschrijdt 2.10 nooit in normale markten.  Resultaat: 100%
        #   positie-blokkering waardoor de strategie nauwelijks handelt.
        #
        #   Nieuwe thresholds zijn calibrated op eigenvalue-spectra van een
        #   uniforme correlatiematrix (formule: N_eff = N^2 / Σλ_i^2):
        #     r=0.85 (baseline)  → N_eff ≈ 1.29  (moet NIET firen)
        #     r=0.93 (stress)    → N_eff ≈ 1.10  (soft zone)
        #     r=0.96 (crisis)    → N_eff ≈ 1.05  (nabij panic)
        #     r=0.98 (liquidatie)→ N_eff ≈ 1.02  (panic: blokkeer)
        panic_threshold: float = 1.03   # Fires only at avg_corr > 0.97 (liquidation cascade)
        soft_threshold:  float = 1.12   # Soft reduction for avg_corr ≈ 0.92–0.97

        # ── Paniek-pad: N_eff zo laag dat diversificatie illusoir is ──────
        if n_eff < panic_threshold:
            logger.warning(
                "[Portfolio Risk] N_eff=%.3f < panic=%.2f (N=%d) — correlatie-paniek. "
                "correlation_multiplier → 0.0 (alle nieuwe posities geblokkeerd).",
                n_eff, panic_threshold, n,
            )
            return 0.0

        # ── Zachte afbouw bij N_eff in [panic_threshold, soft_threshold) ──
        if n_eff < soft_threshold:
            # Lineair van 0.0 (bij panic) naar volledige sqrt-scaling (bij soft)
            panic_fraction = (n_eff - panic_threshold) / max(soft_threshold - panic_threshold, 1e-9)
            base_mult = float(np.sqrt(max(n_eff, 1.0) / n)) * float(panic_fraction)
        else:
            # Normaal pad: sqrt-scaling op effective N
            base_mult = float(np.sqrt(max(n_eff, 1.0) / n))

        # Correlatie-throttle bij oversampling
        if avg_corr > self.max_avg_pairwise_corr:
            # Lineair van 1.0× bij drempel naar 0.5× bij corr=1.0
            slope = (1.0 - 0.5) / max(1.0 - self.max_avg_pairwise_corr, 1e-3)
            penalty = 1.0 - slope * (avg_corr - self.max_avg_pairwise_corr)
            penalty = float(np.clip(penalty, 0.5, 1.0))
            base_mult *= penalty
        return float(np.clip(base_mult, 0.0, 1.0))

    def dd_state(self) -> bool:
        """Update + retourneer of de DD-breaker actief is."""
        # Lookback peak: peak_equity over de laatste N bars (rolling, niet all-time)
        if self.dd_breaker_lookback_bars > 0 and len(self.equity_curve) > self.dd_breaker_lookback_bars:
            peak = max(self.equity_curve[-self.dd_breaker_lookback_bars:])
        else:
            peak = self.peak_equity
        peak = max(peak, 1e-9)
        dd = 1.0 - self.equity / peak

        # Hysteresis: schakel aan boven threshold, schakel uit pas wanneer DD weer
        # onder resume_threshold zakt.
        if not self.dd_breaker_active and dd >= self.dd_breaker_threshold:
            self.dd_breaker_active = True
            logger.warning(
                "[Portfolio Risk] DD-breaker AAN: dd=%.2f%% >= threshold=%.2f%%",
                dd * 100.0, self.dd_breaker_threshold * 100.0,
            )
        elif self.dd_breaker_active and dd <= self.dd_resume_threshold:
            self.dd_breaker_active = False
            logger.info(
                "[Portfolio Risk] DD-breaker UIT: dd=%.2f%% <= resume=%.2f%%",
                dd * 100.0, self.dd_resume_threshold * 100.0,
            )
        return self.dd_breaker_active

    # -----------------------------------------------------------------------
    # SIZING
    # -----------------------------------------------------------------------
    def size_position(
        self,
        symbol: str,
        raw_leverage: float,
        side: int,
        per_asset_cap_override: float | None = None,
    ) -> SizingDecision:
        """Bepaal de uiteindelijke leverage voor een nieuwe positie in `symbol`.

        Args:
            symbol: ticker.
            raw_leverage: leverage uit ``gap_risk_kelly_size`` of vergelijkbaar
                per-trade Kelly-resultaat.
            side: +1 (long), -1 (short), 0 (flat — retourneert blocked).
            per_asset_cap_override: voor symbolen met striktere cap (zie
                `conf/symbols/*.yaml`'s `max_leverage`).
        """
        if side == 0 or raw_leverage <= 0.0:
            return SizingDecision(raw_leverage, 0.0, 0.0, blocked=True, reason="flat")

        # 1. DD-breaker
        if self.dd_state():
            return SizingDecision(
                raw_leverage, 0.0, 0.0, blocked=True, reason="dd_breaker_active"
            )

        # 2. Kelly-fraction.
        # Wave 16 P0-19: default kelly_fraction=None → resolves to 1.0 here
        # because raw_leverage from kelly.py already uses divisor=6 (1/6-Kelly).
        # Apply additional scaling only when caller explicitly sets a fraction.
        scaled = raw_leverage * self.kelly_fraction

        # v3 T1.3: portfolio-level CVaR scaling
        # Als het asset een verhoogde tail-risk vertoont (CVaR > 2× baseline),
        # krimp het kapitaalbeslag exponentieel om staartverliezen te beperken.
        if self._cvar_scale_enabled and hasattr(self, '_asset_cvar'):
            _asset_cvar_val = self._asset_cvar.get(symbol, None)
            _baseline_cvar  = self._baseline_cvar
            if _asset_cvar_val is not None and _baseline_cvar > 0:
                _cvar_ratio = _asset_cvar_val / _baseline_cvar
                if _cvar_ratio > 2.0:
                    # Exponentieel inklimpen: bij 2× CVaR → factor 0.5, bij 4× CVaR → factor 0.25
                    _cvar_factor = 1.0 / max(_cvar_ratio / 2.0, 1.0)
                    scaled = scaled * _cvar_factor
                    logger.debug("[%s] CVaR scaling: ratio=%.2f → factor=%.2f", symbol, _cvar_ratio, _cvar_factor)

        # 3. Vol-target + correlatie-scaler
        vol_mult = self.vol_target_multiplier()
        corr_mult = self.correlation_multiplier()
        scaled *= vol_mult * corr_mult

        # 4. Per-asset cap
        per_cap = per_asset_cap_override if per_asset_cap_override is not None else self.max_per_asset_leverage
        scaled_capped = float(min(scaled, per_cap))

        # 5. Aggregate caps (gross + net)
        proposed_signed = side * scaled_capped
        # Build proposed exposure dict
        proposed: dict[str, float] = dict(self._active_signed)
        proposed[symbol] = proposed_signed

        gross = sum(abs(v) for v in proposed.values())
        net = sum(proposed.values())

        if gross > self.max_gross_leverage and gross > 1e-9:
            shrink = self.max_gross_leverage / gross
            scaled_capped *= shrink
            proposed[symbol] = side * scaled_capped
            gross = sum(abs(v) for v in proposed.values())
            net = sum(proposed.values())

        if abs(net) > self.max_net_leverage and abs(net) > 1e-9:
            # Trim alleen het nieuwe positie naar zoveel als netto cap toelaat
            other_net = sum(v for k, v in proposed.items() if k != symbol)
            allowed = self.max_net_leverage * np.sign(net) - other_net
            allowed = side * max(0.0, side * allowed)
            scaled_capped = float(min(scaled_capped, abs(allowed)))

        # v3 T1.4: HMM regime cap — reduceer LONG met 50% in Bear-regime
        if side > 0 and self._hmm_detector is not None:
            try:
                if len(self._hmm_returns_buffer) >= 20:
                    import pandas as _pd_hmm
                    _recent = _pd_hmm.Series(self._hmm_returns_buffer[-200:])
                    _regime = self._hmm_detector.current_regime(_recent)
                    _long_cap = self._hmm_detector.get_long_cap(int(_regime))
                    scaled_capped = scaled_capped * _long_cap
                    if _long_cap < 1.0:
                        logger.debug("HMM Bear-regime: LONG-cap %.1f×", _long_cap)
            except Exception as _hmm_exc:
                logger.debug("HMM regime cap fout (doorgaan): %s", _hmm_exc)

        return SizingDecision(
            raw_leverage=float(raw_leverage),
            scaled_leverage=float(scaled),
            final_leverage=float(max(scaled_capped, 0.0)),
            blocked=False,
            reason=f"vol_mult={vol_mult:.3f} corr_mult={corr_mult:.3f}",
        )

    def set_hmm_detector(self, detector) -> None:
        """Koppel een HMMRegimeDetector voor LONG-regime capping (v3 T1.4)."""
        self._hmm_detector = detector
        logger.info("HMM regime detector gekoppeld aan PortfolioRiskManager.")

    def update_hmm_buffer(self, portfolio_return: float) -> None:
        """Voeg meest recente portfolio-return toe aan HMM-buffer."""
        self._hmm_returns_buffer.append(float(portfolio_return))
        if len(self._hmm_returns_buffer) > self._hmm_buffer_max:
            self._hmm_returns_buffer.pop(0)

    def update_cvar_estimates(self, asset_cvar_dict: dict, baseline_cvar: float = 0.01) -> None:
        """Update per-asset CVaR schattingen voor portfolio-level risk capping (v3 T1.3).

        Args:
            asset_cvar_dict: {symbool: cvar_value} — 99% CVaR van OOS residuen.
            baseline_cvar:   Referentie CVaR (default 1%).
        """
        self._asset_cvar = dict(asset_cvar_dict)
        self._baseline_cvar = max(float(baseline_cvar), 1e-6)
        self._cvar_scale_enabled = bool(asset_cvar_dict)
        logger.info("CVaR estimates bijgewerkt: %d assets, baseline=%.4f", len(self._asset_cvar), self._baseline_cvar)

    # -----------------------------------------------------------------------
    # FEEDBACK
    # -----------------------------------------------------------------------
    def update(
        self,
        timestamp: pd.Timestamp | None,
        returns_per_asset: dict[str, float],
        active_signed_exposure: dict[str, float] | None = None,
        gk_vol_per_asset: dict[str, float] | None = None,
        bar_costs: float = 0.0,
        # AUDIT-FIX (N24 — equity-curve mutation anti-pattern):
        #   Oude code in portfolio_backtest.py muteert extern:
        #     self.risk_manager.equity *= (1.0 - costs)
        #     self.risk_manager.equity_curve[-1] = self.risk_manager.equity
        #   Dit creëert:
        #     (a) DD-breaker false trigger: peak berekend op gesplitste updates
        #     (b) Sharpe inconsistentie: equity_curve != portfolio_returns lijst
        #     (c) Refactoring-bom: toekomstige invariants in update() stille breken
        #
        #   Fix: pass bar_costs (rebalance_cost + funding_cost als fractionele
        #   return) naar update().  equity wordt in één atomaire update gezet:
        #     equity *= (1 + port_r - bar_costs)
        #   De externe mutatie in portfolio_backtest.py wordt daarna verwijderd.
    ) -> RiskState:
        """Voed bar-returns toe + update equity, peak en breaker-state.

        Args:
            returns_per_asset    : per-asset bar-return (price-return × sign(pos)).
            active_signed_exposure: optionele update van de actieve gewichten.
            gk_vol_per_asset     : optioneel — Garman-Klass σ per asset voor
                                   deze bar (fractioneel, uit get_garman_klass_volatility).
                                   Wanneer aangeleverd, schakelt de manager over
                                   op GK-gebaseerde vol-targeting (sneller dan std).
                                   Typisch: gk_vol_per_asset={"BTCUSDT": feat_vol_gk_val}.
            bar_costs            : fractie van equity die deze bar als kosten
                                   wordt afgetrokken (rebalance + funding).
                                   0.0 = geen extra kosten (default = backward compat).
        """
        # 0. GK-vol bijhouden (vóór returns zodat vol_target_multiplier deze bar
        #    al de nieuwe vol ziet als de aanroeper dat wil).
        if gk_vol_per_asset is not None:
            self._use_gk_vol = True
            cap = max(self.vol_window_bars, self.corr_window_bars) * 4
            for s in self.symbols:
                v = float(gk_vol_per_asset.get(s, 0.0))
                if np.isfinite(v) and v >= 0.0:
                    self._gk_vol_history[s].append(v)
                    if len(self._gk_vol_history[s]) > cap:
                        self._gk_vol_history[s] = self._gk_vol_history[s][-cap:]

        # 1. Append per-asset returns
        for s in self.symbols:
            r = float(returns_per_asset.get(s, 0.0))
            if not np.isfinite(r):
                r = 0.0
            self._returns_history[s].append(r)
            # Cap history op 4× corr_window om geheugen te beperken
            cap = max(self.vol_window_bars, self.corr_window_bars) * 4
            if len(self._returns_history[s]) > cap:
                self._returns_history[s] = self._returns_history[s][-cap:]

        # 2. Update active exposure
        if active_signed_exposure is not None:
            for s in self.symbols:
                self._active_signed[s] = float(active_signed_exposure.get(s, 0.0))

        # 3. Bereken portfolio-bar-return uit per-asset returns × actieve exposure
        port_r = 0.0
        for s in self.symbols:
            port_r += self._active_signed.get(s, 0.0) * float(returns_per_asset.get(s, 0.0))

        # 4. Equity update — atomair: bruto PnL en kosten in één stap (N24-fix)
        # bar_costs is de fractie rebalance+funding die deze bar wordt afgetrokken.
        # Geen externe mutatie nodig; peak en equity_curve zijn altijd consistent.
        net_r = port_r - float(bar_costs)
        self.equity *= (1.0 + net_r)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)
        if timestamp is not None:
            self.timestamps.append(timestamp)

        # 5. State snapshot
        avg_corr, _ = self._correlation_metrics()
        return RiskState(
            timestamp=timestamp,
            equity=self.equity,
            peak_equity=self.peak_equity,
            drawdown=1.0 - self.equity / max(self.peak_equity, 1e-9),
            realized_vol=self._realized_portfolio_vol(),
            avg_pairwise_corr=avg_corr,
            vol_target_multiplier=self.vol_target_multiplier(),
            corr_multiplier=self.correlation_multiplier(),
            dd_breaker_active=self.dd_breaker_active,
            gross_leverage=sum(abs(v) for v in self._active_signed.values()),
            net_leverage=sum(self._active_signed.values()),
        )

    # -----------------------------------------------------------------------
    # METRICS
    # -----------------------------------------------------------------------
    def equity_metrics(self) -> dict[str, float]:
        """Sharpe / Calmar / MaxDD samenvatting van de equity-curve."""
        eq = np.array(self.equity_curve, dtype=np.float64)
        if eq.size < 2:
            return {"sharpe": 0.0, "max_dd": 0.0, "calmar": 0.0, "total_return": 0.0}
        rets = np.diff(eq) / eq[:-1]
        rets = rets[np.isfinite(rets)]
        if rets.size == 0:
            return {"sharpe": 0.0, "max_dd": 0.0, "calmar": 0.0, "total_return": 0.0}
        mu = float(rets.mean())
        sigma = float(rets.std(ddof=1)) if rets.size > 1 else 1e-12
        sharpe = (mu / sigma) * float(np.sqrt(self.bars_per_year)) if sigma > 0 else 0.0

        # Max drawdown
        running_peak = np.maximum.accumulate(eq)
        dd = 1.0 - eq / np.maximum(running_peak, 1e-9)
        max_dd = float(dd.max()) if dd.size else 0.0

        total_ret = float(eq[-1] / eq[0] - 1.0)
        # Calmar = annualised return / max DD
        n_years = max(eq.size / self.bars_per_year, 1e-9)
        ann_ret = (eq[-1] / eq[0]) ** (1.0 / n_years) - 1.0
        calmar = (ann_ret / max_dd) if max_dd > 1e-9 else 0.0

        return {
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
            "calmar": float(calmar),
            "total_return": float(total_ret),
        }
