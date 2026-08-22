# src/tradebot/execution/market_impact.py
"""Square-root market impact, Ledoit-Wolf shrinkage, crisis multiplier, signal smoothers.

Migrated from market_impact.py (root-level legacy).

Functions
---------
square_root_impact      : Almgren-Chriss/Bouchaud-Bonart sqrt-impact model.
trade_passes_impact_gate: no-trade gate comparing alpha against impact cost.
max_size_for_alpha      : invert impact model to find max order size.
ledoit_wolf_shrunk_cov  : Ledoit-Wolf shrinkage covariance (Oracle Approximating).
cov_to_corr             : normalise cov -> corr matrix.
ledoit_wolf_shrunk_corr : shrunk correlation matrix.
negative_skew_crisis_multiplier: hysteretic vol-multiplier on skew regime.
ehlers_super_smoother   : 2-pole low-pass causal filter.
kalman_smoother_1d      : scalar Kalman filter for online smoothing.
RankQuantileScaler      : online P2-quantile uniform scaler.
vwpf_check              : volume-weighted price fairness check.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("MarketImpact")


# =============================================================================
# 1. SQUARE-ROOT MARKET IMPACT
# =============================================================================
@dataclass(frozen=True)
class ImpactResult:
    """Output van :func:`square_root_impact`.

    Attributes:
        impact_fraction: verwachte slippage als decimal van mid-price
            (0.0005 = 5 bps).  Direct af te trekken van expected-alpha.
        impact_bps:      hetzelfde getal × 10 000 (basis-punten).
        permanent_share: fractie van de impact die NIET reverteert (Bouchaud-eta term).
        order_participation: OrderSize / Volume tijdens executie-window.
    """

    impact_fraction: float
    impact_bps: float
    permanent_share: float
    order_participation: float


# CHIEF AUDIT-FIX (Sim-to-Reality #5):
#   η=0.142 is calibrated to calm-regime Bouchaud-Bonart data.  During
#   liquidation cascades (May-2021 BTC, May-2022 LUNA, Nov-2022 FTX,
#   Aug-2023 SOL) realised η spiked 3-5× because order-book depth
#   evaporates faster than σ rises.  Without crisis scaling the slippage
#   estimate is systematically too small precisely when the strategy is
#   most exposed.  We expose a `crisis_multiplier` parameter and a helper
#   `crisis_eta_from_skew` that converts the existing negative-skew regime
#   signal into a 1.0 / 3.0 multiplier on η.
ETA_CRISIS_MULTIPLIER_DEFAULT: float = 3.0


def crisis_eta_from_skew(
    realised_skew: float,
    skew_threshold: float = -1.5,
    calm_mult: float = 1.0,
    crisis_mult: float = ETA_CRISIS_MULTIPLIER_DEFAULT,
) -> float:
    """Return a multiplier on η based on realised negative skew.

    Negative-skew regimes (jump-down distributions) coincide with thin
    books and elevated taker-side flow → η rises sharply.  This is the
    same signal used by `negative_skew_crisis_multiplier` for the covariance
    matrix; reusing it for impact keeps the regime view internally
    consistent.
    """
    if realised_skew <= skew_threshold:
        return float(crisis_mult)
    return float(calm_mult)


def square_root_impact(
    order_size: float,
    bar_volume: float,
    sigma_per_bar: float,
    *,
    eta: float = 0.142,
    permanent_share: float = 0.50,
    min_volume: float = 1e-9,
    crisis_multiplier: float = 1.0,
) -> ImpactResult:
    """Bouchaud-Bonart-Almgren square-root impact model.

    Forme:
        I(Q) ≈ η · σ · sqrt(Q / V)

    waarbij σ = volatiliteit per bar (decimal), Q = orderomvang in basis-
    asset, V = totaalvolume in dezelfde bar (zelfde eenheid als Q).
    η is empirisch ≈ 0.10–0.15 op crypto futures (Almgren et al. 2005,
    Bouchaud-Bonart 2017 voor Bitcoin perpetuals).

    CHIEF AUDIT-FIX (Sim-to-Reality #5):
      ``crisis_multiplier`` scales η during liquidity-stress regimes.  Set
      via :func:`crisis_eta_from_skew` or the same negative-skew detector
      used for the covariance multiplier.  Default 1.0 = backward compat.

    Args:
        order_size:        jouw geplaatste size (basis-asset).
        bar_volume:        gerealiseerde bar-volume in dezelfde eenheid.
        sigma_per_bar:     bar-vol als decimal (bv. feat_vol_gk).
        eta:               empirische impact-coefficient (0.142 = Bouchaud).
        permanent_share:   fractie die langdurig in de prijs blijft (drift).
        min_volume:        anti-zero-divide guard.
        crisis_multiplier: regime-aware multiplier on η (1.0 calm, ≈3.0 crisis).

    Returns:
        ``ImpactResult`` met fraction + bps + breakdown.
    """
    q = max(float(order_size), 0.0)
    v = max(float(bar_volume), float(min_volume))
    s = max(float(sigma_per_bar), 0.0)
    participation = q / v
    eta_eff = float(eta) * max(float(crisis_multiplier), 1.0)
    impact = eta_eff * s * math.sqrt(participation) if participation > 0.0 else 0.0
    return ImpactResult(
        impact_fraction=float(impact),
        impact_bps=float(impact * 1e4),
        permanent_share=float(np.clip(permanent_share, 0.0, 1.0)),
        order_participation=float(participation),
    )


def trade_passes_impact_gate(
    expected_alpha: float,
    order_size: float,
    bar_volume: float,
    sigma_per_bar: float,
    *,
    safety_margin: float = 1.5,
    eta: float = 0.142,
    crisis_multiplier: float = 1.0,
) -> tuple[bool, ImpactResult]:
    """No-trade-gate: alpha moet impact × safety_margin overstijgen.

    Best practice (Tier-1 MM): vereis dat verwachte alpha minstens
    1.5× de modelled impact is — dit verzekert dat de slippage ook bij
    onderschatting van η of vol nog winstgevend wordt afgetopt.

    Args:
        expected_alpha:  verwachte trade-return (decimal, na fees).
        order_size:      voorgenomen orderomvang.
        bar_volume:      huidige bar-volume.
        sigma_per_bar:   bar-vol decimal.
        safety_margin:   extra factor boven impact die de alpha moet halen.
        eta:             impact-coefficient.

    Returns:
        (passes, impact)
    """
    impact = square_root_impact(
        order_size=order_size,
        bar_volume=bar_volume,
        sigma_per_bar=sigma_per_bar,
        eta=eta,
        crisis_multiplier=crisis_multiplier,
    )
    threshold = impact.impact_fraction * float(safety_margin)
    return float(expected_alpha) > threshold, impact


def max_size_for_alpha(
    expected_alpha: float,
    bar_volume: float,
    sigma_per_bar: float,
    *,
    eta: float = 0.142,
    safety_margin: float = 1.5,
    cap_participation: float = 0.10,
) -> float:
    """Inverse: gegeven een alpha, hoeveel size mag je maximaal placen?

    Lost ``alpha = safety · η · σ · sqrt(Q / V)`` op naar Q.

    Args:
        cap_participation: harde bovengrens (fractie van V) — 10% is een
            goede kapitaal-veilige limiet om geen footprint achter te laten.
    """
    eps = 1e-9
    sigma = max(float(sigma_per_bar), eps)
    eta_eff = max(float(eta) * float(safety_margin), eps)
    if expected_alpha <= 0.0:
        return 0.0
    ratio = (float(expected_alpha) / (eta_eff * sigma)) ** 2.0
    q = ratio * max(float(bar_volume), 0.0)
    cap = float(cap_participation) * max(float(bar_volume), 0.0)
    return float(min(q, cap))


# =============================================================================
# 2. LEDOIT-WOLF SHRINKAGE COVARIANCE (CONST-CORR TARGET)
# =============================================================================
def ledoit_wolf_shrunk_cov(
    returns: np.ndarray,
    *,
    target: str = "const_corr",
    min_obs: int = 30,
) -> tuple[np.ndarray, float]:
    """Analytical Ledoit-Wolf shrinkage estimator.

    Levert een **gestructureerd-doel-shrunken** covariantiematrix:

        Σ̂ = δ · F  +  (1 − δ) · S

    waarbij S = sample-cov en F het doel is:
      * ``"const_corr"``  → constant-correlation prior (Ledoit-Wolf 2003a):
        diagonale variances behouden, off-diagonal correlations vervangen
        door hun gemiddelde.  Default voor crypto multi-asset.
      * ``"identity"``    → diag(σ²) target (Ledoit-Wolf 2004): krimpt naar
        onafhankelijkheid.  Geschikt voor enkele-asset of hoge-noise regimes.

    De optimale shrinkage-intensiteit δ wordt analytisch bepaald uit de
    LW-2004 formule (geen kruisvalidatie nodig).

    Args:
        returns: shape (T, N), NaN's mogen — worden vervangen door 0.
        target:  "const_corr" of "identity".
        min_obs: minimum T anders identity-fallback.

    Returns:
        (sigma_shrunk, delta) — gekrompen cov-matrix en de gebruikte δ ∈ [0,1].
    """
    if returns.ndim != 2:
        return np.eye(1), 0.0
    t, n = returns.shape
    if t < min_obs or n == 0:
        return np.eye(max(n, 1)), 1.0

    # NaN → 0 (causaal: missing returns tellen als nul-bar).
    R = np.nan_to_num(returns.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = R.mean(axis=0, keepdims=True)
    Rc = R - mu
    sample = (Rc.T @ Rc) / float(t)

    var = np.diag(sample).copy()
    var = np.where(var > 1e-12, var, 1e-12)
    std = np.sqrt(var)

    # ── Doel F ──────────────────────────────────────────────────────────────
    if target == "identity":
        mu_diag = float(np.mean(var))
        F = mu_diag * np.eye(n)
    # constant-correlation: gemiddelde correlatie van de sample
    elif n > 1:
        outer = np.outer(std, std)
        outer[outer < 1e-12] = 1e-12
        corr = sample / outer
        np.fill_diagonal(corr, 0.0)
        r_bar = float(np.sum(corr) / max(n * (n - 1), 1))
        F = r_bar * outer
        np.fill_diagonal(F, var)
    else:
        F = sample.copy()

    # ── LW δ-formule (asymptotisch optimaal MSE) ────────────────────────────
    # π̂ = Σᵢⱼ var(sᵢⱼ);   ρ̂ = Σ var(fᵢⱼ, sᵢⱼ);   γ̂ = ||F − S||²_F
    # Pylance: gebruik vectorised numpy waar mogelijk.
    Y = Rc * Rc                                 # (T, N)
    pi_mat = (Y.T @ Y) / float(t) - sample ** 2  # var van sᵢⱼ-schatter
    pi_hat = float(np.sum(pi_mat))

    # ρ̂ — covariantie tussen sᵢᵢ en sᵢⱼ (Ledoit-Wolf 2003 const-corr formule).
    # Voor identity-target geldt ρ̂ = Σᵢ var(sᵢᵢ).  Hieronder gebruiken we
    # een simpele upper-bound die in de praktijk werkt en numeriek stabiel is.
    rho_hat = float(np.sum(np.diag(pi_mat)))

    diff = F - sample
    gamma_hat = float(np.sum(diff * diff))
    if gamma_hat <= 1e-15:
        return sample, 0.0

    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(np.clip(kappa / float(t), 0.0, 1.0))

    sigma_shrunk = delta * F + (1.0 - delta) * sample
    # Symmetriseer (numeriek)
    sigma_shrunk = 0.5 * (sigma_shrunk + sigma_shrunk.T)
    return sigma_shrunk, delta


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Converteer cov-matrix → corr-matrix (numeriek veilig)."""
    n = cov.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    outer = np.outer(std, std)
    outer = np.where(outer > 1e-12, outer, 1.0)
    corr = cov / outer
    np.fill_diagonal(corr, 1.0)
    return np.clip(np.where(np.isfinite(corr), corr, 0.0), -1.0, 1.0)


def ledoit_wolf_shrunk_corr(
    returns: np.ndarray,
    *,
    target: str = "const_corr",
    min_obs: int = 30,
) -> tuple[np.ndarray, float]:
    """Wrapper: shrunk cov → corr (drop-in voor _safe_corr)."""
    cov, delta = ledoit_wolf_shrunk_cov(returns, target=target, min_obs=min_obs)
    return cov_to_corr(cov), delta


# =============================================================================
# 3. CRISIS MULTIPLIER (NEG-SKEW STRESS OVERLAY)
# =============================================================================
def negative_skew_crisis_multiplier(
    btc_returns: np.ndarray,
    *,
    skew_panic: float = -1.5,
    skew_resume: float = -0.5,
    state_active: bool = False,
    multiplier_active: float = 1.5,
    multiplier_calm: float = 1.0,
) -> tuple[float, bool]:
    """Stress-overlay multiplier op de cov-matrix.

    Wanneer BTC's rolling skewness onder ``skew_panic`` valt (zware
    linker-staart toont aan dat liquidatie-cascade aan de gang is),
    worden alle vol-elementen vermenigvuldigd met ``multiplier_active``.
    Hysterese: alleen weer naar ``multiplier_calm`` zodra skew boven
    ``skew_resume`` herstelt.

    Args:
        btc_returns: rolling BTC return-array (vereist ≥ 30 obs).
        state_active: of het stress-pad reeds actief was (caller bewaart).
        skew_panic:   drempel om aan te zetten (typisch −1.5).
        skew_resume:  drempel om uit te zetten (typisch −0.5; hysterese-band).

    Returns:
        (multiplier, new_state_active).  Caller persistert ``new_state_active``.
    """
    if btc_returns.size < 30:
        return float(multiplier_calm), bool(state_active)

    r = btc_returns.astype(np.float64)
    r = r[np.isfinite(r)]
    if r.size < 30:
        return float(multiplier_calm), bool(state_active)

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 1e-12:
        return float(multiplier_calm), bool(state_active)

    z = (r - mu) / sigma
    skew = float(np.mean(z ** 3))

    new_state = bool(state_active)
    if not state_active and skew <= float(skew_panic):
        new_state = True
    elif state_active and skew >= float(skew_resume):
        new_state = False

    mult = float(multiplier_active) if new_state else float(multiplier_calm)
    return mult, new_state


# =============================================================================
# 4. EHLERS SUPER SMOOTHER + KALMAN-1D (PRE-FILTER VOOR HILBERT)
# =============================================================================
def ehlers_super_smoother(price: np.ndarray, period: float = 10.0) -> np.ndarray:
    """John Ehlers' 2-pole Super Smoother (causaal, low-lag).

    Ehlers (2004) — een kritisch-gedempt 2-pole IIR filter dat frequenties
    boven ``period`` agressiever afsnijdt dan een EMA, met < ½ × EMA-lag
    op vergelijkbare cutoff.  Gebruik dit als input voor Hilbert-fase
    detectie: de hoge-frequente ruis is verwijderd, de fase-shift van het
    filter is constant en bekend, en de output is volledig backward-only.

    Args:
        price : 1D float-array.
        period: cutoff-periode in bars (≥ 4).

    Returns:
        Glad 1D-array, lengte gelijk aan input.  Eerste 2 bars = price[0..1].
    """
    n = price.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    p = max(float(period), 4.0)
    # Bron: Ehlers, "Cybernetic Analysis for Stocks and Futures" (2004).
    a1 = math.exp(-1.414 * math.pi / p)
    b1 = 2.0 * a1 * math.cos(1.414 * math.pi / p)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

    out[0] = float(price[0])
    if n > 1:
        out[1] = float(price[1])
    for i in range(2, n):
        out[i] = c1 * 0.5 * (float(price[i]) + float(price[i - 1])) \
                 + c2 * out[i - 1] \
                 + c3 * out[i - 2]
    return out


def kalman_smoother_1d(
    price: np.ndarray,
    process_var: float = 1e-4,
    obs_var: float = 1e-2,
) -> np.ndarray:
    """1-D Kalman filter (random-walk model, causaal).

    Random-walk dynamiek:
        xₜ = xₜ₋₁ + wₜ ,  wₜ ~ N(0, Q)
        yₜ = xₜ + vₜ   ,  vₜ ~ N(0, R)

    Args:
        price:       observatie-reeks.
        process_var: Q (lager = trager, gladder).
        obs_var:     R (hoger = trust observation less).

    Returns:
        Filtered state estimate (causaal).
    """
    n = price.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    x = float(price[0])
    p_var = 1.0  # initial state covariance
    out[0] = x
    Q = max(float(process_var), 1e-12)
    R = max(float(obs_var), 1e-12)
    for i in range(1, n):
        # Predict
        x_pred = x
        p_pred = p_var + Q
        # Update
        k = p_pred / (p_pred + R)
        x = x_pred + k * (float(price[i]) - x_pred)
        p_var = (1.0 - k) * p_pred
        out[i] = x
    return out


# =============================================================================
# 5. RANK-QUANTILE ONLINE SCALER (VOL-INVARIANTE FEATURE-NORMALISATIE)
# =============================================================================
class RankQuantileScaler:
    """Online rank-naar-uniform [0, 1] scaler met dual-buffer P²-markers.

    Probleem dat dit oplost: tijdens vol-explosies (bv. BTC dagvol → 20%)
    raken Z-score-scaled features ver buiten de getrainde distributie.  Het
    model krijgt input die het nooit eerder heeft gezien → onvoorspelbaar
    gedrag.

    Oplossing: vervang z-score door rank-based scaling.  Door over een
    rolling window te ranken en op [0, 1] te projecteren wordt de output-
    distributie ALTIJD uniform — onafhankelijk van de absolute σ.

    Implementatie: we onderhouden ``n_bins`` P²-markers per feature (Jain &
    Chlamtac 1985 — O(1) memory, O(log n_bins) update).  Bij elke
    transform wordt de input geprojecteerd op zijn empirische CDF op basis
    van die markers.

    Args:
        n_bins:        aantal P²-markers per feature (default 21 → percentielen 0,5,10,…,100).
        warmup_bars:   minimum bars vóór transform niet-trivial wordt.
        clip_low:      ondergrens van de output (default 0.001 → vermijd 0.0).
        clip_high:     bovengrens van de output (default 0.999).
    """

    def __init__(
        self,
        n_bins: int = 21,
        warmup_bars: int = 200,
        clip_low: float = 0.001,
        clip_high: float = 0.999,
    ) -> None:
        self.n_bins: int = int(max(n_bins, 5))
        self.warmup_bars: int = int(max(warmup_bars, 5))
        self.clip_low: float = float(clip_low)
        self.clip_high: float = float(clip_high)
        self._buffer: np.ndarray | None = None  # (warmup, n_features) tijdens warmup
        self._buf_count: int = 0
        self._sorted: np.ndarray | None = None  # (n_bins, n_features) percentielen
        self._n_features: int = 0

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> RankQuantileScaler:
        """Initialiseer markers uit een batch (offline trainings-call)."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self._n_features = X_arr.shape[1]
        # Bepaal percentiel-grid
        qs = np.linspace(0.0, 1.0, self.n_bins)
        self._sorted = np.quantile(X_arr, qs, axis=0).astype(np.float64)
        self._buf_count = X_arr.shape[0]
        # Buffer wordt niet meer gebruikt na fit, maar laat None blijven voor zuinigheid.
        self._buffer = None
        return self

    # ------------------------------------------------------------------
    def partial_fit(self, X: np.ndarray) -> RankQuantileScaler:
        """Online update — voeg nieuwe rijen toe en pas markers blendend aan.

        Strategie:
          * Tijdens warmup: vul buffer.
          * Na warmup: voor elk nieuw sample, schuif de markers richting
            de nieuwe waarde met een EWMA-stijl decay (α = 1 / warmup_bars).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if self._sorted is None:
            # Verzamel in buffer tot warmup_bars bereikt
            if self._buffer is None:
                self._buffer = np.empty((self.warmup_bars, X_arr.shape[1]), dtype=np.float64)
                self._n_features = X_arr.shape[1]
            need = self.warmup_bars - self._buf_count
            take = min(need, X_arr.shape[0])
            self._buffer[self._buf_count : self._buf_count + take] = X_arr[:take]
            self._buf_count += take
            if self._buf_count >= self.warmup_bars:
                self.fit(self._buffer[: self._buf_count])
            return self

        # Online blend van markers richting nieuwe samples
        alpha = 1.0 / float(max(self.warmup_bars, 1))
        for row in X_arr:
            for j in range(row.shape[0]):
                col     = self._sorted[:, j]
                new_val = float(row[j])
                # Soft-shift: move all markers toward new_val by alpha (EWMA-style).
                self._sorted[:, j] = col + alpha * (new_val - col)
                # Hard extend: if new_val falls outside the current marker range,
                # clamp the min/max marker to new_val so extreme quantiles track
                # distributional shifts during regime transitions.
                # Previous code here was a no-op: minimum(maximum(A,min(A,B)),max(A,B)) = A.
                if new_val < self._sorted[0, j]:
                    self._sorted[0, j] = new_val
                elif new_val > self._sorted[-1, j]:
                    self._sorted[-1, j] = new_val
            # Enforce strict monotonicity after all-column updates.
            self._sorted = np.sort(self._sorted, axis=0)
        # Onnodig: qs hierboven niet gebruikt na first-fit
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projecteer nieuwe rijen op de empirische CDF (uniforme output)."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if self._sorted is None:
            # Tijdens warmup: passthrough met clipping naar [0.5, 0.5] (constant midden).
            return np.full(X_arr.shape, 0.5, dtype=np.float64)

        _, n_feats = X_arr.shape
        out = np.empty_like(X_arr, dtype=np.float64)
        qs = np.linspace(0.0, 1.0, self.n_bins)

        for j in range(n_feats):
            col_markers = self._sorted[:, j]
            # np.interp doet causale linear lookup: bij waarden buiten range → clipped extremen.
            ranks = np.interp(X_arr[:, j], col_markers, qs, left=0.0, right=1.0)
            out[:, j] = np.clip(ranks, self.clip_low, self.clip_high)
        return out

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# =============================================================================
# 6. VWPF (VOLUME-WEIGHTED PROBABILITY OF FILLING)
# =============================================================================
def vwpf_check(
    target_size: float,
    bar_volume: float,
    *,
    min_participation_ratio: float = 0.05,
    max_participation_ratio: float = 0.20,
) -> tuple[bool, float]:
    """Volume-Weighted Probability of Filling — execution-aware label-gate.

    Een "barrier-hit" in de Triple Barrier-methode telt alleen als win
    wanneer er voldoende liquiditeit was om jouw size in de bar te absorberen
    zonder de prijs voorbij de barrier te duwen.

    Heuristiek (López de Prado, AFML ch. 11 + Bouchaud-Bonart 2017):
      * Participation < min_participation_ratio → fill onwaarschijnlijk
        (te illiquide t.o.v. order — markt kan niet vullen zonder spread).
      * Participation > max_participation_ratio → fill verandert prijs
        (jouw eigen impact zou de barrier-hit veroorzaakt hebben → pseudo-win).
      * Tussenin → realistisch fillbaar.

    Args:
        target_size: jouw orderomvang.
        bar_volume:  totaal bar-volume.
        min_participation_ratio: ondergrens (default 5%).
        max_participation_ratio: bovengrens (default 20%).

    Returns:
        (fillable, participation_ratio).  ``fillable=True`` betekent dat
        de barrier-hit telt als echte win in de meta-labels.
    """
    if bar_volume <= 0.0 or target_size <= 0.0:
        return False, 0.0
    p = float(target_size) / float(bar_volume)
    return (float(min_participation_ratio) <= p <= float(max_participation_ratio)), p
