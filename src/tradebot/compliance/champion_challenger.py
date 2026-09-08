# src/tradebot/compliance/champion_challenger.py
"""Champion/Challenger framework for model promotion governance.

Protocol (§9.1):
  1. New model → stage "challenger" in registry
  2. Shadow-trade challenger for ≥ T_min=14 days / N_min=200 bars
  3. Diebold-Mariano test: if pvalue < 0.05 AND challenger_sharpe > champion_sharpe → promote
  4. Manual override always possible (audit trail required)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..validation.inference import newey_west_lags

logger = logging.getLogger(__name__)

__all__ = ["DMTestResult", "ChampionChallengerConfig", "ChampionChallenger"]

_ANNUALISE = np.sqrt(8760)  # 1h bars → annual (sqrt of hours per year)

# CHIEF AUDIT 2026-05-23 (H6): expliciete bars-per-year voor zero-padded Sharpe.
# 1h-bars: 24 × 365 = 8760. Identiek aan execution/spread.compute_annualised_sharpe
# voor daily resample (sqrt(365.25)) ÷ verschil = bar-granularity convention.
_BARS_PER_YEAR_CC = 8760.0


@dataclass(frozen=True)
class DMTestResult:
    """Result of a Diebold-Mariano test.

    Attributes
    ----------
    dm_statistic :
        DM test statistic (positive = challenger superior).
    pvalue :
        One-sided p-value.
    champion_sharpe :
        Annualised Sharpe of the champion model.
    challenger_sharpe :
        Annualised Sharpe of the challenger model.
    sufficient_data :
        True if the minimum observation threshold was met.
    recommendation :
        "promote" | "retain_champion" | "insufficient_data".
    """

    dm_statistic: float
    pvalue: float
    champion_sharpe: float
    challenger_sharpe: float
    sufficient_data: bool
    recommendation: str


@dataclass
class ChampionChallengerConfig:
    """Parameters for the champion/challenger evaluation.

    Attributes
    ----------
    t_min_days :
        Minimum shadow-trade days before evaluation.
    n_min_bars :
        Minimum number of bars before evaluation.
    significance_level :
        p-value threshold for the DM test.
    sharpe_hurdle :
        Challenger must exceed champion Sharpe by at least this margin.
    """

    t_min_days: int = 14
    n_min_bars: int = 200
    significance_level: float = 0.05
    sharpe_hurdle: float = 0.0


class ChampionChallenger:
    """Evaluates whether a challenger should replace the champion.

    Parameters
    ----------
    config :
        Evaluation configuration.
    """

    def __init__(self, config: ChampionChallengerConfig | None = None) -> None:
        self._cfg = config or ChampionChallengerConfig()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        champion_pnl: pd.Series,
        challenger_pnl: pd.Series,
    ) -> DMTestResult:
        """Run Diebold-Mariano test and return a promotion recommendation.

        Parameters
        ----------
        champion_pnl :
            Per-bar PnL series for the champion (UTC-indexed).
        challenger_pnl :
            Per-bar PnL series for the challenger (same index).

        Returns
        -------
        DMTestResult with recommendation.
        """
        # Align series
        idx = champion_pnl.index.intersection(challenger_pnl.index)
        champ = champion_pnl.loc[idx].fillna(0.0)
        chal = challenger_pnl.loc[idx].fillna(0.0)
        n = len(idx)

        if n < self._cfg.n_min_bars:
            return DMTestResult(
                dm_statistic=float("nan"),
                pvalue=1.0,
                champion_sharpe=self._sharpe(champ),
                challenger_sharpe=self._sharpe(chal),
                sufficient_data=False,
                recommendation="insufficient_data",
            )

        # DM test: test whether challenger outperforms champion
        # H0: E[loss_champion - loss_challenger] = 0
        # Using squared error loss differential: d_t = e_champ^2 - e_chal^2
        # (treating PnL as negative loss: higher PnL = lower loss)
        d = champ.values - chal.values  # positive = champion superior
        dm_stat, pvalue = self._dm_test(d)

        champ_sharpe = self._sharpe(champ)
        chal_sharpe = self._sharpe(chal)

        # Promote if: DM significant in favour of challenger AND sharpe hurdle met
        challenger_superior = (
            dm_stat < 0  # negative DM = challenger has lower loss = higher PnL
            and pvalue < self._cfg.significance_level
            and (chal_sharpe - champ_sharpe) > self._cfg.sharpe_hurdle
        )
        recommendation = "promote" if challenger_superior else "retain_champion"

        logger.info(
            "CC evaluation: dm=%.4f p=%.4f champ_sr=%.3f chal_sr=%.3f → %s",
            dm_stat, pvalue, champ_sharpe, chal_sharpe, recommendation,
        )

        return DMTestResult(
            dm_statistic=dm_stat,
            pvalue=pvalue,
            champion_sharpe=champ_sharpe,
            challenger_sharpe=chal_sharpe,
            sufficient_data=True,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dm_test(d: np.ndarray) -> tuple[float, float]:
        """Harvey-Leybourne-Newbold corrected DM test (HLN 1997).

        CHIEF AUDIT 2026-05-23 (M14): Andrews-rule bandwidth voor Newey-West
        variantieschatter. De oude bandwidth k=1 is te kort voor crypto
        autocorrelatie (typisch 20-40 bar vol-clustering) → onderschat var
        en blaast de DM-statistic op, met als gevolg te veel false promotes.
        Andrews-rule: k = 4 × (n/100)^(2/9), clip naar [1, n-2].

        FASE 10, STAP 4A FIXRONDE 1 (RULING T4A-D): de bandbreedte zelf komt nu
        uit `validation.inference.newey_west_lags` in plaats van een tweede,
        letterlijk identieke uitdrukking hier. R-3 verbiedt een tweede
        implementatie van dezelfde bandbreedteregel, ook wanneer zij hetzelfde
        getal geeft; het importeren verplaatst geen enkel gemeten getal. De
        clip naar `[1, n-2]` erna is EIGEN aan deze HLN-toets (df-verlies) en
        blijft hier staan; de Bartlett-HAC-variantieschatter eronder is een
        ANDERE grootheid dan `inference.hac_variance_ratio` (een DM-teststatistiek
        tegenover een Sharpe-SE-opslag) en wordt niet geconsolideerd.
        """
        n = len(d)
        if n < 4:
            return float("nan"), 1.0
        mean_d = np.mean(d)
        k = newey_west_lags(n)
        k = min(k, max(1, n - 2))
        # Newey-West variance estimate met Andrews-bandbreedte.
        gamma_0 = np.var(d, ddof=1)
        x = d - mean_d
        nw_sum = 0.0
        for lag in range(1, k + 1):
            if n - lag < 1:
                break
            gamma_lag = float(np.dot(x[:-lag], x[lag:])) / (n - lag)
            # Bartlett-kernel weight
            w = 1.0 - lag / (k + 1.0)
            nw_sum += 2.0 * w * gamma_lag
        nw_var = (gamma_0 + nw_sum) / n
        if nw_var <= 0:
            nw_var = gamma_0 / n
        dm = mean_d / np.sqrt(max(nw_var, 1e-12))
        # HLN correction factor (zelfde k voor consistent verlies van df)
        hln_factor = np.sqrt((n + 1 - 2 * k + k * (k - 1) / n) / n)
        dm_hln = dm * hln_factor
        # One-sided p-value (testing challenger > champion, so left tail)
        pvalue = float(stats.t.cdf(dm_hln, df=n - 1))
        return float(dm_hln), pvalue

    @staticmethod
    def _sharpe(pnl: pd.Series) -> float:
        """Annualised Sharpe from per-bar PnL (1h bars assumed).

        CHIEF AUDIT 2026-05-23 (H6): zero-padding consistent met
        execution/spread.compute_annualised_sharpe.  Non-trade bars = 0%
        equity return — dropping them inflateert Sharpe met factor
        1/sqrt(occupancy) (≈ 1.58× bij 40% occupancy) en maakt de DM-test
        anti-conservatief.  Sparse-signal challengers worden anders
        ten onrechte gepromoveerd.  Geen .replace(0, np.nan).dropna() meer.
        """
        # GEEN dropna: behoud zero-bars zodat de calendar-time Sharpe matcht
        # met de live equity-curve.
        returns = pnl.copy().fillna(0.0)
        if len(returns) < 2:
            return 0.0
        sigma = float(returns.std(ddof=1))
        if sigma < 1e-9:
            return 0.0
        return float(returns.mean() / sigma * np.sqrt(_BARS_PER_YEAR_CC))
