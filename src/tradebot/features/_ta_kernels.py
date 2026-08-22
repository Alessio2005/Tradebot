"""Internal Numba kernels and FFD helpers for features/ta.py.

Migrated from legacy ta_features.py. All Numba kernels preserved verbatim.
"""
# ta_features.py — Technical analysis feature engineering utilities.
from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------------------------
# Optionele afhankelijkheden: KPSS-test, scipy (spearman + hiërarchische
# clustering), sklearn (PCA / StandardScaler) en scipy.signal (Hilbert).
# Alle optionele imports degraderen graceful naar een no-op / fallback zodat de
# module importeerbaar blijft in minimale omgevingen (unit tests, CI).
# ---------------------------------------------------------------------------
try:  # KPSS — voor stationariteitsbevestiging (geen rejection-only beslissing)
    from statsmodels.tsa.stattools import kpss as _kpss_test
    _KPSS_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _kpss_test = None  # type: ignore[assignment]
    _KPSS_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    PCA = None  # type: ignore[assignment,misc]
    StandardScaler = None  # type: ignore[assignment,misc]
    logger_import = logging.getLogger("TA_Engine")
    logger_import.warning(
        "scikit-learn niet gevonden. "
        "FeatureOrthogonalizer degradeert naar no-op. "
        "Installeer via: pip install scikit-learn"
    )

try:  # scipy — hiërarchische clustering + Spearman dendrogram
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    fcluster = None     # type: ignore[assignment]
    linkage = None      # type: ignore[assignment]
    squareform = None   # type: ignore[assignment]
    _spearmanr = None   # type: ignore[assignment]

try:  # Hilbert-transform voor causale fase-detectie
    from scipy.signal import hilbert as _scipy_hilbert
    from scipy.signal.windows import hann as _scipy_hann_window
    _HILBERT_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _scipy_hilbert = None       # type: ignore[assignment]
    _scipy_hann_window = None   # type: ignore[assignment]
    _HILBERT_AVAILABLE = False

# Typed wrappers — Pylance vriendelijke facades die None-short-circuit doen
# (voorkomt "None is not callable" waarschuwingen in call-sites).
if _KPSS_AVAILABLE and _kpss_test is not None:
    kpss_func: Callable[..., Any] | None = _kpss_test
else:  # pragma: no cover
    kpss_func = None

if _SCIPY_AVAILABLE and _spearmanr is not None:
    spearmanr_func: Callable[..., Any] | None = _spearmanr
else:  # pragma: no cover
    spearmanr_func = None

if _HILBERT_AVAILABLE and _scipy_hilbert is not None:
    hilbert_func: Callable[..., Any] | None = _scipy_hilbert
else:  # pragma: no cover
    hilbert_func = None

# Let op: zorg dat volatility.py in je path staat.
try:
    from ..volatility import get_garman_klass_volatility, get_jump_adjusted_volatility
except ImportError:
    logging.warning(
        "volatility.py niet gevonden. Garman-Klass functies zullen falen."
    )
    get_garman_klass_volatility = None  # type: ignore[assignment]
    get_jump_adjusted_volatility = None  # type: ignore[assignment]

logger = logging.getLogger("TA_Engine")

DEFAULT_BASE_WINDOW = 75
DEFAULT_FFD_THRESHOLD = 1e-4
# Harde ondergrens voor de dynamische FFD-threshold: lager dan 1e-6 levert
# honderden gewichten op (geheugenexplosie) zonder meetbaar voordeel in
# autocorrelatie-retentie — de gewichten vallen sneller dan 1e-6 in de staart.
DEFAULT_FFD_THRESHOLD_FLOOR = 1e-6
DEFAULT_FFD_THRESHOLD_CEIL = 1e-3
DEFAULT_VOLUME_CANDIDATES = ("real_volume", "tick_volume", "volume")


# =========================================================
# 0. NUMBA KERNELS (OPTIMIZED)
# =========================================================


@njit(cache=True)
def roll_autocorr_lag1(returns: np.ndarray, window: int) -> np.ndarray:
    """Berekent rolling Pearson autocorrelatie met lag 1.

    Gedeprecieerd voor kleine windows. Pearson autocorr is extreem gevoelig
    voor één uitschieter: een enkele 5-σ return kan ρ van 0.0 naar 0.4
    tillen. Voor robuuste momentum-features: zie :func:`roll_autocorr_spearman`.
    """
    n = len(returns)
    out = np.full(n, np.nan)

    for i in range(window, n):
        x = returns[i - window + 1 : i]
        y = returns[i - window + 2 : i + 1]

        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        std_x = float(np.std(x))
        std_y = float(np.std(y))

        if std_x > 1e-9 and std_y > 1e-9:
            cov = np.mean((x - mu_x) * (y - mu_y))
            out[i] = cov / (std_x * std_y)
        else:
            out[i] = 0.0

    return out


@njit(cache=True)
def _rankdata_average(arr: np.ndarray) -> np.ndarray:
    """Average-rank van een 1-D array (numba-kompatibel).

    Gelijke waarden krijgen de gemiddelde rank (identiek gedrag aan
    ``scipy.stats.rankdata(method='average')``). Stabiele O(n log n)
    via argsort — de feature-hot-path blijft snel in numba.
    """
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        # Verzamel een run van gelijke sorteerwaarden.
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        # Gemiddelde rank van de tie-groep (1-based → 0-based mix).
        avg_rank = 0.5 * (float(i) + float(j)) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks


@njit(cache=True)
def roll_autocorr_spearman(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling Spearman rank autocorrelatie (lag 1).

    Spearman = Pearson op de ranks. Robuust tegen fat-tail uitschieters:
    één extreme return → rank blijft bounded [1, window], dus de
    correlatie beweegt hooguit 1/window per bar. Ideaal voor kleine
    momentum-windows waar Pearson volledig vastgenageld wordt door
    een enkele outlier.

    Returns:
        float64 array, NaN voor de eerste ``window`` bars.
    """
    n = len(returns)
    out = np.full(n, np.nan)

    for i in range(window, n):
        x = returns[i - window + 1 : i]
        y = returns[i - window + 2 : i + 1]

        rx = _rankdata_average(x)
        ry = _rankdata_average(y)

        mu_x = np.mean(rx)
        mu_y = np.mean(ry)
        dx = rx - mu_x
        dy = ry - mu_y

        num = np.sum(dx * dy)
        den_x = np.sqrt(np.sum(dx * dx))
        den_y = np.sqrt(np.sum(dy * dy))

        if den_x > 1e-9 and den_y > 1e-9:
            out[i] = num / (den_x * den_y)
        else:
            out[i] = 0.0

    return out


@njit(cache=True)
def _ewma_std(x: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentieel gewogen σ (causale, parameterloze variant).

    Gebruikt als input voor :func:`compute_dynamic_ffd_threshold_series`.
    Halflife in bars → α = 1 − exp(−ln 2 / halflife). We berekenen σ via
    EWMA(x²) − EWMA(x)² (Welford-stijl, numeriek stabiel).
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or halflife <= 0.0:
        return out

    alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
    mu = 0.0
    var = 0.0
    for i in range(n):
        xi = x[i]
        delta = xi - mu
        mu += alpha * delta
        var = (1.0 - alpha) * (var + alpha * delta * delta)
        if var > 0.0:
            out[i] = math.sqrt(var)

    return out


def compute_hilbert_phase(
    close: np.ndarray,
    smoothing_span: int = 20,
    rolling_window: int = 128,
    *,
    pre_filter: str = "ehlers",
    ehlers_period: float = 10.0,
    kalman_process_var: float = 1e-4,
    kalman_obs_var: float = 1e-2,
) -> np.ndarray:
    """Strikt causale Hilbert-transform fase op een EMA-gladgestreken prijs.

    Motivatie: spectrale fasefilters (FFT, complexe DFT) falen op financiële
    data omdat de dominante "cyclus" continu driftt. De Hilbert-transform
    daarentegen levert een *instantane* fase per bar — geen aanname van
    vaste periode. Toegepast op een low-pass gefilterde prijs (bv. EMA-20)
    isoleert het de lange-swing component zonder hoog-frequente ruis.

    FIX (issue #13 — Hilbert Causaliteit):
        ``scipy.signal.hilbert`` voert een FFT uit over de **volledige**
        invoer-array. Het analytisch signaal op tijd t hangt daardoor af
        van data op t+1, t+2, … (FFT spreidt informatie over de hele
        reeks). Een trailing EMA + globale FFT was dus alleen op de laatste
        sample causaal; alle eerdere punten lekten future info. We passen
        de Hilbert nu in een **rolling backward-only window** toe: voor
        elke t wordt het analytisch signaal berekend over [t-w+1, t] en
        nemen we de fase van het laatste sample. Daarmee is elk punt
        wiskundig strikt afhankelijk van uitsluitend zijn verleden, voor
        zowel training als live inference.

    Args:
        close          : Close-prijzen.
        smoothing_span : EMA-span voor low-pass filtering.
        rolling_window : Causale window-grootte voor de Hilbert FFT.
                         Moet ≥ 32 zijn voor numerieke stabiliteit.

    Returns:
        float64 array met fase in [−π, π], NaN gedurende burn-in.
    """
    n = int(close.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    burn_in = max(int(smoothing_span), int(rolling_window))
    if n <= burn_in or hilbert_func is None:
        return out

    # AUDIT-FIX (Issue 8 — Pre-Filter Hilbert):
    #   EMA als low-pass voegt expliciete phase-lag toe (~½ × span bars). Voor
    #   MFT-fase-detectie is die lag dodelijk: tegen de tijd dat de fase-omslag
    #   gedetecteerd wordt, is de alpha al verdampt.
    #
    #   Alternatieven (volgorde van afnemende lag):
    #     * Ehlers Super Smoother — 2-pole IIR, kritisch gedempt → ½ × EMA-lag
    #       voor vergelijkbare cutoff. Default.
    #     * Kalman 1-D random-walk — adaptief, geen vaste cutoff. Tweede keuze.
    #     * EMA — backward-compat fallback (legacy gedrag).
    pre_filter_name = str(pre_filter).lower().strip()
    if pre_filter_name in ("ehlers", "super_smoother"):
        try:
            from ..execution.market_impact import ehlers_super_smoother as _ehlers
            ema = _ehlers(close.astype(np.float64), period=float(ehlers_period))
        except ImportError:
            # Fall through naar EMA (backward compat)
            pre_filter_name = "ema"
    if pre_filter_name == "kalman":
        try:
            from ..execution.market_impact import kalman_smoother_1d as _kalman
            ema = _kalman(
                close.astype(np.float64),
                process_var=float(kalman_process_var),
                obs_var=float(kalman_obs_var),
            )
        except ImportError:
            pre_filter_name = "ema"
    if pre_filter_name == "ema" or pre_filter_name not in ("ehlers", "super_smoother", "kalman"):
        # EMA (causale low-pass): α = 2 / (N+1). Backward-compat fallback.
        alpha = 2.0 / (float(smoothing_span) + 1.0)
        ema = np.empty(n, dtype=np.float64)
        ema[0] = float(close[0])
        for i in range(1, n):
            ema[i] = alpha * float(close[i]) + (1.0 - alpha) * ema[i - 1]

    w = int(rolling_window)

    # AUDIT-FIX (issue #4 — Hilbert Gibbs Phenomenon):
    #   De Hilbert-transform is een lineaire fase-rotatie in het frequentie-
    #   domein.  Bij een abrupte rechter rand (het signaal stopt ineens) ziet
    #   de FFT een rechthoekvenster, wat Gibbs-ringing veroorzaakt:
    #     * Overshoots van ~9% bij de rechterrand (Wilbraham-Gibbs constante)
    #     * Neerslag in de gextraheerde fase op positie t (de huidige bar)
    #   Een Hann-venster demt de amplitude aan de randen soepel naar nul en
    #   elimineert daarmee de ringing bijna volledig (sidelobe-attenuation ~31 dB
    #   vs. -13 dB voor rechthoek).
    #
    #   Trade-off: het Hann-venster versmalt de effectieve frequentiebreedte
    #   (ENBW ≈ 1.5 bins) en vermindert de amplitude-schaal.  Omdat we alleen
    #   de *fase* (np.angle) extraheren en niet de amplitude, is deze trade-off
    #   volledig acceptabel — fase-detectie profiteert puur van minder ringing.
    #
    #   Fallback: als _scipy_hann_window niet beschikbaar is (import mislukt),
    #   gebruik dan het originele rechthoekvenster met een runtime-waarschuwing.
    _hann: np.ndarray | None
    if _HILBERT_AVAILABLE and _scipy_hann_window is not None:
        _hann = np.asarray(_scipy_hann_window(w), dtype=np.float64)
    else:
        import warnings as _warnings
        _warnings.warn(
            "compute_hilbert_phase: scipy.signal.windows.hann niet beschikbaar "
            "— rechthoekvenster gebruikt (Gibbs-ringing mogelijk aanwezig).",
            RuntimeWarning,
            stacklevel=2,
        )
        _hann = None

    try:
        for t in range(burn_in, n):
            window_slice = ema[t - w + 1 : t + 1]
            # Per-window detrend (zero-mean is Hilbert-vereiste).
            local_mean = window_slice.mean()
            detrended_w = window_slice - local_mean
            # Hann-venster toepassen vóór de Hilbert-transform om Gibbs te dempen.
            if _hann is not None:
                detrended_w = detrended_w * _hann
            analytic_w = hilbert_func(detrended_w)
            # Fase op het *laatste* sample (= huidige bar) — backward-only.
            out[t] = float(np.angle(np.asarray(analytic_w))[-1])
    except Exception:
        # Bij onverwachte numerieke fout (zelden, op exotische arrays):
        # geef NaN-array terug i.p.v. half-gevulde data te leveren.
        return np.full(n, np.nan, dtype=np.float64)

    return out


def compute_dynamic_ffd_threshold(
    returns_np: np.ndarray,
    base: float = DEFAULT_FFD_THRESHOLD,
    floor: float = DEFAULT_FFD_THRESHOLD_FLOOR,
    ceil: float = DEFAULT_FFD_THRESHOLD_CEIL,
) -> float:
    """Volatility-adaptive FFD weight truncation threshold.

    Probleem: een vaste threshold (bv. 1e-4) kapt FFD-gewichten altijd op
    dezelfde staart af. In hoge-volatility-regimes (macro shocks, news)
    zijn verre-lags informatief — we laten 'memory' weglekken precies
    wanneer het kritiek is.

    Oplossing: schaal de threshold omgekeerd evenredig met de recente
    volatiliteit. In hoge-vol regimes → lagere threshold → meer gewichten
    bewaard → langere geheugen-staart.

    Formule:
        τ_dyn = clip( base × σ_ref / σ_recent ,  floor,  ceil )
    waarbij σ_ref het mediaan-σ over de hele serie is en σ_recent het
    EWMA-σ over de laatste ~halflife bars.
    """
    if returns_np.size < 10:
        return float(base)

    vol_ewma = _ewma_std(returns_np.astype(np.float64), halflife=100.0)
    sigma_recent = float(vol_ewma[-1])
    sigma_ref = float(np.median(vol_ewma[vol_ewma > 0.0])) if np.any(vol_ewma > 0.0) else sigma_recent

    if sigma_recent <= 0.0 or sigma_ref <= 0.0 or not np.isfinite(sigma_recent / sigma_ref):
        return float(base)

    scale = sigma_ref / sigma_recent  # < 1 in high-vol regimes → lagere τ
    tau = float(base) * scale
    return float(np.clip(tau, floor, ceil))


@njit(cache=True)
def roll_cycle_phase(price: np.ndarray, window: int) -> np.ndarray:
    """Berekent de cyclische fase van de prijs met Z-scores en atan2.

    FIX (issue #10 — Zero-Variance Bars):
    Wanneer een bar sluit na 1 tick (gigantische order, extreem fragmenteel
    regime), geldt H=L=O=C → rolling σ=0. De oude implementatie wees
    stilzwijgend 0.0 toe, wat niet te onderscheiden was van een geldig
    signaal op atan2(0,0)=0 en CatBoost ten onrechte liet leren dat dit
    een "neutrale fase" is. We geven nu NaN terug zodat de caller deze
    bars expliciet kan flaggen (``feat_bar_is_degenerate``).
    """
    n = len(price)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n):
        slice_p = price[i - window + 1 : i + 1]
        mu, sigma = np.mean(slice_p), np.std(slice_p)

        if sigma < 1e-9:
            # NaN laten staan → downstream flag.
            continue

        z_p = (price[i] - mu) / sigma
        slice_diff = slice_p[1:] - slice_p[:-1]
        mu_d, sig_d = np.mean(slice_diff), np.std(slice_diff)

        if sig_d < 1e-9:
            continue

        z_d = (slice_diff[-1] - mu_d) / sig_d
        out[i] = math.atan2(z_d, z_p) / math.pi

    return out


@njit(cache=True)
def calculate_shannon_entropy(
    prices: np.ndarray,
    window: int = 100,
    num_bins: int = 15,
) -> np.ndarray:
    """Berekent genormaliseerde Shannon entropy op log-returns met ADAPTIEVE
    (equiprobable) binning.

    FIX — Fat-tail regime shift:
      De vorige implementatie binde ``z = (x − μ)/σ`` op een vaste ±3σ range.
      Tijdens een regime-shift (news, liquidatie) vallen 5σ+ samples buiten de
      bins → clip(2.99) → alles op de rand gepakt → entropy collapses naar een
      kunstmatig lage waarde die niet correspondeert met échte disorder.

      Oplossing: **equiprobable bins** — bereken de bin-grenzen als
      kwantielen van het window zelf. Per constructie valt in een uniforme
      verdeling 1/num_bins in elke bin → H_max = log2(num_bins) exact.
      Afwijkingen van die gelijke verdeling (concentratie rond 0 of zware
      staarten) verlagen de entropy op een manier die invariant is onder
      regime-schaalwijzigingen.

    Implementatie-detail (numba):
      ``np.partition`` is niet beschikbaar in numba — we sorteren het hele
      window (O(w log w)). Voor typisch window=150 is dit verwaarloosbaar
      t.o.v. de overige feature-berekeningen.
    """
    n = len(prices)
    entropy_arr = np.full(n, np.nan)
    log_ret = np.zeros(n)

    for i in range(1, n):
        if prices[i - 1] > 1e-9:
            log_ret[i] = math.log(prices[i] / prices[i - 1])

    max_entropy = math.log2(float(num_bins))

    for i in range(window - 1, n):
        win = log_ret[i - window + 1 : i + 1].copy()
        sorted_win = np.sort(win)

        # Degenerate case: vrijwel constante returns → 0 disorder.
        if sorted_win[-1] - sorted_win[0] < 1e-12:
            entropy_arr[i] = 0.0
            continue

        # Bin-grenzen = (num_bins − 1) interne kwantielen.
        # edges[k] = kwantiel (k+1)/num_bins; k = 0 .. num_bins − 2.
        edges = np.empty(num_bins - 1, dtype=np.float64)
        for k in range(num_bins - 1):
            q = (k + 1) / num_bins
            pos = q * (window - 1)
            lo = math.floor(pos)
            hi = math.ceil(pos)
            if hi >= window:
                hi = window - 1
            frac = pos - float(lo)
            edges[k] = sorted_win[lo] + frac * (sorted_win[hi] - sorted_win[lo])

        counts = np.zeros(num_bins)
        for val in win:
            # Binary search: kleinste edge-index > val.
            lo = 0
            hi = num_bins - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if val <= edges[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            bin_idx = lo if val <= edges[lo] else num_bins - 1
            # Rand-tolerantie voor de laatste bin.
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            counts[bin_idx] += 1

        entropy = 0.0
        for c in counts:
            if c > 0.0:
                p = c / float(window)
                entropy -= p * math.log2(p)

        entropy_arr[i] = entropy / max_entropy if max_entropy > 0 else 0.0

    return entropy_arr


@njit(cache=True)
def get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
    """Genereert gewichten voor fractional differencing."""
    weights = np.zeros(lim, dtype=np.float64)
    weights[0] = 1.0

    for k in range(1, lim):
        w_k = -weights[k - 1] / k * (d - k + 1)
        if abs(w_k) < thres:
            return np.ascontiguousarray(weights[:k][::-1])
        weights[k] = w_k

    return np.ascontiguousarray(weights[::-1])


@njit(cache=True)
def apply_frac_diff_mask_fast(
    series_v: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Past fractional differencing toe met een specifiek masker."""
    width = len(weights)
    n = len(series_v)
    out = np.full(n, np.nan)

    for i in range(width - 1, n):
        if mask[i]:
            out[i] = np.dot(weights, series_v[i - width + 1 : i + 1])

    return out


# =========================================================
# 1. FRACTIONAL DIFFERENCING LOGIC
# =========================================================


def _kpss_pvalue(series: np.ndarray) -> float:
    """KPSS p-value, robuust voor kleine samples en missing statsmodels.

    Returns NaN als de KPSS niet beschikbaar of numeriek faalt. Callers
    moeten op NaN testen (→ beschouw als niet-bevestigd stationair).

    KPSS null-hypothese: **serie is stationair**. Dus we willen
    p > α (NIET rejecteren) om stationariteit te bevestigen. Dit is
    complementair aan ADF (die stationariteit als alternatief heeft).
    """
    if not _KPSS_AVAILABLE or kpss_func is None:
        return float("nan")
    if series.size < 30:
        return float("nan")
    try:
        # ``regression='c'`` en ``nlags='auto'`` matchen de ADF-call zodat
        # beide tests dezelfde assumpties over de trend maken.
        with warnings.catch_warnings():
            # KPSS logt een InterpolationWarning als p-value buiten de
            # lookup-tabel valt. Dat is verwachte info, niet een fout.
            warnings.simplefilter("ignore")
            stat_result = kpss_func(series, regression="c", nlags="auto")
        p_val: float = float(stat_result[1])
        return p_val
    except Exception:
        return float("nan")


def get_optimal_d(
    series: pd.Series,
    max_d: float = 1.0,
    step: float = 0.05,
    p_thres: float = 0.01,
    ffd_thres: float = DEFAULT_FFD_THRESHOLD,
    require_kpss: bool = True,
    kpss_alpha: float = 0.05,
) -> float:
    """Vindt de minimale d voor stationariteit via ADF + KPSS **beide**.

    Stationariteit bewijzen (i.p.v. alleen non-stationariteit verwerpen)
    vraagt twee tests met tegenovergestelde null-hypotheses:

      * **ADF**  : H₀ = unit root (non-stationair).   → verwerp  (p < α)
      * **KPSS** : H₀ = stationariteit.               → accepteer (p > α)

    Als beide tests samenvallen (ADF-p < α **én** KPSS-p > α) zijn we
    statistisch zeker; conflicten (alleen ADF aanvaardt of alleen KPSS)
    duiden op niet-conclusieve stationariteit.

    Args:
        series      : Positieve prijs-serie (wordt log-getransformeerd).
        max_d       : Maximaal te proberen d (default 1.0).
        step        : Stap-grootte in d-raster (default 0.05).
        p_thres     : α voor ADF (default 0.01 strikt).
        ffd_thres   : FFD-weight truncation threshold (kan dynamisch zijn).
        require_kpss: True → KPSS moet ook bevestigen (gebruikt als
                      statsmodels' kpss beschikbaar is). False → legacy
                      ADF-only gedrag (backward compatible).
        kpss_alpha  : α voor KPSS-acceptatie (default 0.05).

    Returns:
        Minimale d waarvoor beide tests slagen (of enkel ADF als KPSS niet
        beschikbaar is). NaN als geen d in [0, max_d] stationariteit geeft.
    """
    series_log_np: np.ndarray = np.log(
        series.clip(lower=1e-9).to_numpy(dtype=np.float64)
    )

    kpss_usable = require_kpss and _KPSS_AVAILABLE

    # Eerste kandidaat die ADF-slaagt — fallback als KPSS nooit bevestigt.
    adf_only_candidate: float = float("nan")

    for d in np.arange(0.0, max_d + step, step):
        weights = get_weights_ffd(
            float(d), ffd_thres, len(series_log_np)
        )
        diff_vals = apply_frac_diff_mask_fast(
            series_log_np,
            weights,
            np.ones(len(series_log_np), dtype=bool),
        )
        clean_vals = diff_vals[~np.isnan(diff_vals)]

        if len(clean_vals) < 30:
            continue

        try:
            # FIX (issue #8 — ADF lag-selectie):
            # maxlag=1 was te restrictief: ADF kon dan alléén kiezen tussen 0 of
            # 1 lag, terwijl bar-data autocorrelatie tot 10+ lags kan dragen.
            # Daardoor werd de unit-root test te conservatief en werd een serie
            # ten onrechte als non-stationair geclassificeerd. Schwert (1989)
            # adviseert maxlag = ⌈12 · (n/100)^{1/4}⌉ als bovengrens.
            schwert_max = int(np.ceil(12.0 * (len(clean_vals) / 100.0) ** 0.25))
            adf_max_lag = max(1, min(schwert_max, len(clean_vals) // 4))
            adf_p: float = float(adfuller(
                clean_vals,
                maxlag=adf_max_lag,
                regression="c",
                autolag="AIC",
            )[1])
        except Exception:
            continue

        if adf_p >= p_thres:
            # ADF verwerpt nog geen unit root → d te laag.
            continue

        if np.isnan(adf_only_candidate):
            adf_only_candidate = float(d)

        if not kpss_usable:
            return float(d)

        kpss_p = _kpss_pvalue(clean_vals)
        if np.isnan(kpss_p):
            # KPSS gefaald (klein sample, numeriek) → val terug op ADF-beslissing.
            return float(d)

        if kpss_p > kpss_alpha:
            # ✓ ADF verwerpt unit root  EN  KPSS accepteert stationariteit.
            return float(d)
        # Anders: ADF zegt stationair, KPSS zegt non-stationair. Probeer
        # hogere d — misschien slaat KPSS om.

    # Geen d waar beide tests samenvallen. Gebruik ADF-only als vangnet zodat
    # downstream code een redelijk d-waarde heeft.
    if not np.isnan(adf_only_candidate):
        logger.info(
            "get_optimal_d: KPSS heeft stationariteit op geen enkele d "
            "bevestigd — fallback op ADF-only d=%.2f.",
            adf_only_candidate,
        )
        return adf_only_candidate

    return float("nan")


# =========================================================
# 2. FEATURE ENGINEER CLASS
# =========================================================

