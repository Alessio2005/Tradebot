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

PHASE 7/8 STAGE C-1 - DI-1 GESLOTEN
-----------------------------------
`predict()` gebruikte `GaussianHMM.predict`, en dat is het VITERBI-pad: de
meest waarschijnlijke toestandsREEKS over het hele venster. Elke rij daarvan is
bepaald met kennis van bars die na die rij komen. Datzelfde gold voor
`predict_proba`, dat de smoothed posterior `P(S_t | F_T)` geeft. Audit sectie
10.2 staat in een backtest uitsluitend `P(S_t | F_t)` toe.

De inferentie loopt nu door `regime.markov.forward_filter`, het causale
forward-algoritme met schaling dat Phase 6 voor het M2-contract heeft gebouwd.
`hmmlearn` doet nog uitsluitend de EM-schatting van de parameters; het doet geen
enkele inferentie meer. Die scheiding is precies wat `regime/markov.py::fit_hmm`
ook aanhoudt, en om dezelfde reden.

Het verschil is niet cosmetisch. Viterbi optimaliseert het GEZAMENLIJKE pad, dus
zelfs de laatste bar van een venster kan een andere toestand krijgen dan de
argmax van de filtered posterior op diezelfde bar - de enige waarde die
`legacy_sizing.py` gebruikt.

De EMA-crossover-fallback is en blijft gesloopt (no-go 5).
"""
from __future__ import annotations

import logging
from enum import IntEnum

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from ..features.volatility import causal_expanding_std
from ..regime.markov import (
    FilteredProbabilities,
    HmmParameters,
    HmmSpec,
    forward_filter,
)
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
        # Sinds Stage C-1 wordt het hmmlearn-model NIET vastgehouden: het doet
        # alleen de EM-stap. Wat blijft, zijn de bevroren parameters, en die
        # zijn het enige dat het causale filter nodig heeft.
        self._params: HmmParameters | None = None
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
        # De parameters worden BEVROREN in het M2-contract uit regime/markov.py.
        # Het hmmlearn-model zelf wordt losgelaten: zou het blijven hangen, dan
        # is `model.predict(...)` een aanroep verderop, en dat is precies de
        # Viterbi-route die dit herontwerp sluit.
        self._params = HmmParameters(
            spec=HmmSpec(n_states=3, covariance_type="full"),
            symbol="portfolio",
            fold_id=0,
            start_prob=np.asarray(model.startprob_, dtype=np.float64),
            trans_mat=np.asarray(model.transmat_, dtype=np.float64),
            means=np.asarray(model.means_, dtype=np.float64),
            covars=np.asarray(model.covars_, dtype=np.float64),
            converged=bool(model.monitor_.converged),
            n_train_obs=int(len(X)),
            loglikelihood=float(model.score(X)),
        )
        self._fitted = True
        logger.info(
            "HMM getraind op %d obs. State-map: %s",
            len(X),
            {k: Regime(v).name for k, v in self._state_map.items()},
        )
        return self

    def filtered_regimes(
        self, returns: pd.Series
    ) -> tuple[np.ndarray, FilteredProbabilities]:
        """Geef (labels, FILTERED probabilities). Crasht op een ongefit model.

        HEET BEWUST NIET `predict`. Die naam was hier de directe aanleiding voor
        DI-1: `detector.predict(...)` las als "voorspel", terwijl de aanroep
        eronder `GaussianHMM.predict` was - Viterbi, en dus smoothed. Een naam
        die je moet uitleggen om te weten wat hij doet, is de verkeerde naam.

        `filtered_regimes` zegt in de aanroep zelf welk contract geldt.

        STAGE C-1 - DI-1 GESLOTEN. Hier stond:

            raw = self._model.predict(X)        # Viterbi over het HELE venster
            probs = self._model.predict_proba(X)  # smoothed P(S_t | F_T)

        Beide gebruiken observaties NA `t` om de toestand op `t` te bepalen. De
        inferentie loopt nu door `regime.markov.forward_filter`, dat per bar
        uitsluitend `[0, t]` gebruikt.

        `labels` is de argmax van de filtered posterior, niet het Viterbi-pad.
        Dat is een ANDER getal en niet slechts een andere berekening: Viterbi
        optimaliseert de gezamenlijke reeks, waardoor ook de laatste bar kan
        afwijken van de marginale argmax op diezelfde bar.

        Returns
        -------
        tuple
            `labels` (n,) met `Regime`-waarden, en de `FilteredProbabilities`
            zelf - een type dat per constructie geen smoothed variant kan zijn.
        """
        require(
            self._fitted and self._params is not None,
            "filtered_regimes() aangeroepen op een ongefitte detector. Roep "
            "eerst fit() aan. Eerder viel dit pad stilzwijgend terug op een "
            "EMA-crossover.",
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

        assert self._params is not None  # door require() gegarandeerd
        filtered = forward_filter(X, self._params)
        labels = np.array(
            [self._state_map[int(s)] for s in filtered.most_likely_state],
            dtype=np.int64,
        )
        return labels, filtered

    def get_long_cap(self, regime: int) -> float:
        return _LONG_CAP_BEAR if Regime(regime) == Regime.BEAR else 1.0

    def current_regime(self, recent_returns: pd.Series) -> Regime:
        """Het regime op de LAATSTE bar, uit de filtered posterior.

        Dit is de enige waarde die `portfolio/legacy_sizing.py` gebruikt, en
        precies de waarde die onder Viterbi kon afwijken: dat pad wordt
        gezamenlijk geoptimaliseerd, dus ook de laatste bar hangt af van de
        volledige reeks.
        """
        labels, _ = self.filtered_regimes(recent_returns)
        return Regime(int(labels[-1]))
