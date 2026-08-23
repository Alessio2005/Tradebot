"""hmm_regime.py - 3-state Gaussian HMM regime detector (v3 T1.4).

Bear(0) / Flat(1) / Bull(2) op BTC+ETH macro bars.
In Bear: LONG-posities automatisch 50% reduceren.

Referentie: CHIEF_MASTER_PLAN v3 par. 3 T1.4; Hamilton (1989).

PHASE 0 - VERWIJDERDE STILLE DEGRADATIE (D-10)
----------------------------------------------
Dit bestand bevatte vijf codepaden die het 3-state Gaussian HMM stilzwijgend
vervingen door een 20/100 EMA-crossover:

  1. `try: from hmmlearn.hmm import GaussianHMM / except ImportError` zette
     _HMM_AVAILABLE=False en logde een warning.
  2. `fit()` met minder dan 50 observaties logde een warning en gaf een
     ONGEFITTE detector terug.
  3. `except Exception` in `fit()` logde de fout en liet _fitted=False staan.
  4. `except Exception` in `predict()` logde de fout en viel door.
  5. `predict()` viel onvoorwaardelijk door naar `_ema_regime_labels()`.

Het netto-effect: op elke machine zonder hmmlearn - en `hmmlearn` stond niet
in pyproject.toml - draaide de "HMM regime detector" in werkelijkheid een
EMA-crossover, terwijl elk rapport en elke ledger-entry hem als HMM
registreerde. De EMA-tak is volledig gesloopt; er bestaat geen niet-HMM-pad
meer. Ontbreekt hmmlearn, dan crasht de import met DependencyMissingError.

Deze fase herontwerpt het model NIET naar M2 Filtered HMM - dat is expliciet
Phase 6 (audit sectie 24: REDESIGN). Zie docs/DEFERRED_ISSUES.md voor de twee
causaliteitskwesties die daarbij horen (Viterbi-smoothing en de
full-sample std in de vol-feature).
"""
from __future__ import annotations

import logging
from enum import IntEnum

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from ..features.volatility import causal_expanding_std
from ..utils.failfast import DataContractError, require

logger = logging.getLogger(__name__)

# Minimum aantal bruikbare observaties voor een 3-state Gaussian HMM met
# full covariance. Onder deze grens is de EM-schatting niet identificeerbaar.
_MIN_OBS_FOR_FIT = 50

# Venster van de vol-feature. Blijft hier als module-constante in plaats van in
# `conf/`: `hmm_regime.py` is een REDESIGN-item voor Phase 6 (audit sectie 24) en
# wordt daar geparametriseerd. Phase 2 raakt uitsluitend de CAUSALITEIT van deze
# feature (DI-2), niet de parametrisatie ervan (DI-10/DI-11).
_VOL_WINDOW_BARS = 5
# Minimaal aantal returns voordat de expanding std tijdens de opstartfase een
# waarde vrijgeeft. Daaronder bestaat er geen schatting en blijft het NaN.
_VOL_MIN_PERIODS = 2


class Regime(IntEnum):
    BEAR = 0
    FLAT = 1
    BULL = 2


_LONG_CAP_BEAR = 0.5


class HMMRegimeDetector:
    """3-state Gaussisch HMM regime detector (Bear/Flat/Bull).

    Traint op dagelijkse returns + rolling volatiliteit.
    Geeft LONG-cap factor: 0.5 in Bear, 1.0 anders.

    Er bestaat geen gedegradeerd pad: elke conditie die het HMM onbruikbaar
    maakt, crasht.
    """

    def __init__(self, n_iter: int = 100, random_state: int = 42) -> None:
        self._model: GaussianHMM | None = None
        self._state_map: dict[int, int] = {}
        self._fitted = False
        self._n_iter = n_iter
        self._random_state = random_state

    def _features(self, returns: pd.Series) -> np.ndarray:
        """Return- en volatiliteitskolom. CAUSAAL - zie DI-2 hieronder.

        PHASE 2 - DI-2 GESLOTEN
        -----------------------
        Deze regel luidde:

            vol = returns.rolling(5, min_periods=2).std().fillna(returns.std())

        `returns.std()` is de standaarddeviatie over de VOLLEDIGE sample. Op
        bar 1 kreeg de vol-feature daarmee een waarde die pas aan het EINDE van
        de reeks bekend kon zijn, en het HMM werd dus deels gefit op informatie
        die het op dat moment niet had. De fout is bovendien systematisch:
        de opvulwaarde is de gemiddelde volatiliteit van het hele venster, wat
        een rustige beginperiode te hoog en een crisisperiode te laag schat.

        De vervanging gebruikt uitsluitend `[0, t]`:
          * zodra er een volledig venster is, telt de rolling std;
          * daarvoor de EXPANDING std over de tot dan toe bekende returns;
          * daarvoor is er geen schatting, en blijft de waarde NaN. Die rijen
            vallen weg in de finite-filter hieronder in plaats van te worden
            opgevuld.

        DI-1 (Viterbi-smoothing in `predict`) staat hier LOS van en blijft
        toegewezen aan Phase 6; deze wijziging raakt uitsluitend de feature.
        """
        vol_rolling = returns.rolling(
            _VOL_WINDOW_BARS, min_periods=_VOL_WINDOW_BARS
        ).std()
        vol_warmup = causal_expanding_std(returns, min_periods=_VOL_MIN_PERIODS)
        vol = vol_rolling.where(vol_rolling.notna(), vol_warmup)
        X = np.column_stack([returns.values, vol.values])
        return X[np.isfinite(X).all(axis=1)]

    def fit(self, returns: pd.Series) -> HMMRegimeDetector:
        """Fit het 3-state HMM. Crasht bij onvoldoende data of niet-convergentie."""
        X = self._features(returns)
        require(
            len(X) >= _MIN_OBS_FOR_FIT,
            "Onvoldoende observaties voor een 3-state Gaussian HMM met full "
            "covariance; de EM-schatting is niet identificeerbaar. Eerder werd "
            "hier een ongefitte detector teruggegeven, die vervolgens "
            "stilzwijgend een EMA-crossover draaide.",
            DataContractError,
            n_obs=len(X),
            required=_MIN_OBS_FOR_FIT,
        )

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=self._n_iter,
            random_state=self._random_state,
            tol=1e-4,
        )
        # Geen try/except: een mislukte EM-fit is een contractschending, geen
        # aanleiding om op een naiever model over te stappen.
        model.fit(X)

        means = model.means_[:, 0]
        order = np.argsort(means)
        self._state_map = {
            int(order[0]): int(Regime.BEAR),
            int(order[1]): int(Regime.FLAT),
            int(order[2]): int(Regime.BULL),
        }
        self._model = model
        self._fitted = True
        logger.info(
            "HMM getraind op %d obs. State-map: %s",
            len(X),
            {k: Regime(v).name for k, v in self._state_map.items()},
        )
        return self

    def predict(self, returns: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Geef (labels, state-probabilities). Crasht wanneer het HMM niet bruikbaar is.

        DEFERRED (Phase 6, DI-1): `GaussianHMM.predict` levert het Viterbi-pad,
        dat de VOLLEDIGE reeks gebruikt - inclusief observaties na t. Dat is
        smoothed, niet filtered, en schendt de regel uit audit sectie 10.2.
        Phase 0 wijzigt geen modelgedrag; dit wordt in Phase 6 vervangen door
        filtered forward-probabilities.
        """
        require(
            self._fitted and self._model is not None,
            "predict() aangeroepen op een ongefitte detector. Roep eerst fit() "
            "aan. Eerder viel dit pad stilzwijgend terug op een EMA-crossover.",
            DataContractError,
        )
        X = self._features(returns)
        require(
            X.size > 0,
            "Geen enkele eindige observatie na feature-constructie; er valt "
            "geen regime te bepalen. Eerder gaf dit pad FLAT met uniforme "
            "kansen terug alsof dat een schatting was.",
            DataContractError,
            n_input=len(returns),
        )

        assert self._model is not None  # door require() gegarandeerd
        raw = self._model.predict(X)
        labels = np.array([self._state_map[int(s)] for s in raw])
        probs = self._model.predict_proba(X)
        return labels, probs

    def get_long_cap(self, regime: int) -> float:
        return _LONG_CAP_BEAR if Regime(regime) == Regime.BEAR else 1.0

    def current_regime(self, recent_returns: pd.Series) -> Regime:
        labels, _ = self.predict(recent_returns)
        return Regime(int(labels[-1]))
