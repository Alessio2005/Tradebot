# src/tradebot/features/volatility.py
"""L3 - causale volatiliteitsfeatures. Phase 2, deliverable 3.

**Deze module sluit DI-2.**

Het lek zag er zo uit:

    vol = returns.rolling(5, min_periods=2).std().fillna(returns.std())

`returns.std()` is de standaarddeviatie over de VOLLEDIGE sample. Op bar 3 van
een reeks van 2.342 bars kreeg de feature daarmee een waarde die pas in 2026
bekend kon zijn. Het effect is bovendien systematisch en niet willekeurig: de
opvulwaarde is per constructie de gemiddelde volatiliteit van het hele
onderzoeksvenster, waardoor de rustige beginperiode van een reeks te hoog en een
crisisperiode te laag wordt geschat. Elk risicomodel dat erop steunt, is
gekalibreerd op informatie die het niet had.

Het causale alternatief staat in `causal_expanding_std`: een EXPANDING
standaarddeviatie die uitsluitend `r_0..r_t` gebruikt en die vóór
`min_periods` observaties eerlijk NaN blijft. Er wordt niets ingevuld, want er
is niets om mee te vullen.

Alle vier de estimators hier delen dezelfde eigenschap: de waarde op `t` is een
functie van uitsluitend `[0, t]`. Truncatie-invariantie is daarmee een
constructie-eigenschap, en `tests/lookahead/test_feature_causality.py` toetst
hem op de 18 gecertificeerde reeksen uit Phase 1.

Garman-Klass ontbreekt bewust. De GK-schatter kan per bar een NEGATIEVE
variantie opleveren; die realisaties vragen een expliciet gedocumenteerde
behandeling (clippen, of overstappen op een variantie-domein), en dat is een
beslissing van de volatiliteitsengine (L2, Phase 3/6), niet van de feature-laag.
Parkinson kent dat probleem niet: `ln(H/L)^2` is per constructie niet-negatief.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 8, 9, 9.1, 19 (L3), 26.
"""
from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from .base import BaseFeature, InputSpec

__all__ = [
    "EwmaVolatility",
    "ExpandingVolatility",
    "ParkinsonVolatility",
    "RealizedVolatility",
    "causal_expanding_std",
    "log_returns",
]

#: Parkinson-schaalfactor: 1 / (4 ln 2). Een wiskundig feit, geen beleidsknop.
_PARKINSON_SCALE = 1.0 / (4.0 * math.log(2.0))


def log_returns(close: pd.Series) -> pd.Series:
    """Causale log-returns. De eerste waarde is NaN en blijft NaN.

    Er wordt niet ge-`fillna`-d en niet ge-`bfill`-d: op de eerste bar bestaat er
    geen return, en dat is informatie in plaats van een probleem.
    """
    require(
        bool((close.dropna() > 0.0).all()),
        "Niet-positieve prijs in de closereeks; ln(P) is dan niet gedefinieerd "
        "en levert -inf of NaN op die stilzwijgend door elke vol-schatting "
        "propageert.",
        DataContractError,
        n_non_positive=int((close.dropna() <= 0.0).sum()),
    )
    # Expliciet via numpy op een float64-array: identieke numeriek, maar het
    # resultaat blijft aantoonbaar een Series met dezelfde index.
    log_price = pd.Series(
        np.log(close.to_numpy(dtype="float64")), index=close.index
    )
    diffed: pd.Series = log_price.diff()
    return diffed


def causal_expanding_std(returns: pd.Series, *, min_periods: int) -> pd.Series:
    """EXPANDING standaarddeviatie - de causale vervanger van het DI-2-lek.

    Op `t` gebruikt deze schatting uitsluitend `r_0..r_t`. Voor de eerste
    `min_periods` bruikbare observaties bestaat er geen schatting en is de
    uitkomst NaN.

    Dit is bewust GEEN `fillna`-variant. Het verschil met
    `rolling(w).std().fillna(returns.std())` is niet cosmetisch: die laatste
    injecteert de standaarddeviatie van de volledige sample - inclusief elke
    toekomstige crash - in de eerste bars van de reeks.

    Parameters
    ----------
    returns : reeks returns; leidende NaN's tellen niet mee als observatie.
    min_periods : minimaal aantal observaties voordat een waarde wordt
        vrijgegeven. Komt uit `conf/features/default.yaml`; er is geen default.
    """
    require(
        min_periods > 1,
        "Een standaarddeviatie over minder dan twee observaties is niet "
        "gedefinieerd; min_periods moet groter dan 1 zijn.",
        DataContractError,
        min_periods=min_periods,
    )
    return returns.astype("float64").expanding(min_periods=min_periods).std()


class RealizedVolatility(BaseFeature):
    """Rolling realized volatility over `window` bars, geannualiseerd.

        sigma_t = std(r_{t-w+1..t}) * sqrt(annualisation_factor)

    `min_periods == window`: tijdens de burn-in blijft de waarde NaN. Dit is
    exact de regel waar DI-2 tegen zondigde.
    """

    name: ClassVar[str] = "realized_volatility"
    input_spec: ClassVar[InputSpec] = InputSpec(
        datasets=("ohlcv",), columns=("close",)
    )

    def __init__(self, *, window: int, annualisation_factor: float) -> None:
        require(
            window > 1,
            "Een realized volatility over minder dan twee returns is niet "
            "gedefinieerd.",
            DataContractError,
            window=window,
        )
        require(
            annualisation_factor > 0.0,
            "annualisation_factor moet positief zijn.",
            DataContractError,
            annualisation_factor=annualisation_factor,
        )
        super().__init__(
            params={"window": int(window),
                    "annualisation_factor": float(annualisation_factor)}
        )

    @property
    def burn_in_period(self) -> int:
        # De eerste log-return is NaN, dus het eerste volledige venster van
        # `window` returns sluit pas op positie `window`.
        return int(self.params["window"])

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (f"vol_realized_{self.params['window']}",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        window = int(self.params["window"])
        scale = math.sqrt(float(self.params["annualisation_factor"]))
        r = log_returns(frame["close"])
        vol = r.rolling(window, min_periods=window).std() * scale
        return pd.DataFrame(
            {self.output_columns[0]: vol.to_numpy(dtype="float64")},
            index=frame.index,
        )


class ExpandingVolatility(BaseFeature):
    """Expanding realized volatility vanaf het begin van de reeks.

    De directe, causale tegenhanger van het DI-2-lek: waar dat de sample-brede
    std injecteerde, groeit deze schatting mee met de beschikbare historie en
    bestaat hij domweg nog niet tijdens de eerste `min_periods` bars.
    """

    name: ClassVar[str] = "expanding_volatility"
    input_spec: ClassVar[InputSpec] = InputSpec(
        datasets=("ohlcv",), columns=("close",)
    )

    def __init__(self, *, min_periods: int, annualisation_factor: float) -> None:
        require(
            min_periods > 1,
            "min_periods moet groter dan 1 zijn voor een standaarddeviatie.",
            DataContractError,
            min_periods=min_periods,
        )
        require(
            annualisation_factor > 0.0,
            "annualisation_factor moet positief zijn.",
            DataContractError,
            annualisation_factor=annualisation_factor,
        )
        super().__init__(
            params={"min_periods": int(min_periods),
                    "annualisation_factor": float(annualisation_factor)}
        )

    @property
    def burn_in_period(self) -> int:
        # Positie 0 draagt geen return; de m-de bruikbare return valt op m.
        return int(self.params["min_periods"])

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("vol_expanding",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        scale = math.sqrt(float(self.params["annualisation_factor"]))
        r = log_returns(frame["close"])
        vol = causal_expanding_std(
            r, min_periods=int(self.params["min_periods"])
        ) * scale
        return pd.DataFrame(
            {self.output_columns[0]: vol.to_numpy(dtype="float64")},
            index=frame.index,
        )


class EwmaVolatility(BaseFeature):
    """RiskMetrics EWMA-volatiliteit met een CAUSALE opstartfase.

        sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r_t^2

    De recursie is per constructie causaal. Het enige gevoelige punt is de
    SEED: een EWMA moet ergens beginnen, en de verleiding is om hem te seeden
    met de variantie over de volledige sample - dezelfde fout als DI-2, alleen
    beter verstopt.

    Hier wordt geseed met het EXPANDING gemiddelde van `r^2` over uitsluitend de
    eerste `burn_in_bars` returns, en pas vanaf dat punt wordt een waarde
    vrijgegeven. Alles daarvoor is NaN. De waarde op `t` hangt daarmee af van
    `r_1..r_t` en van niets anders.
    """

    name: ClassVar[str] = "ewma_volatility"
    input_spec: ClassVar[InputSpec] = InputSpec(
        datasets=("ohlcv",), columns=("close",)
    )

    def __init__(
        self, *, lam: float, burn_in_bars: int, annualisation_factor: float
    ) -> None:
        require(
            0.0 < lam < 1.0,
            "De EWMA-decay lambda moet strikt tussen 0 en 1 liggen "
            "(RiskMetrics-conventie: 0.94 voor daily).",
            DataContractError,
            lam=lam,
        )
        require(
            burn_in_bars > 1,
            "De seed van de EWMA-variantie vereist minstens twee returns.",
            DataContractError,
            burn_in_bars=burn_in_bars,
        )
        require(
            annualisation_factor > 0.0,
            "annualisation_factor moet positief zijn.",
            DataContractError,
            annualisation_factor=annualisation_factor,
        )
        super().__init__(
            params={
                "lam": float(lam),
                "burn_in_bars": int(burn_in_bars),
                "annualisation_factor": float(annualisation_factor),
            }
        )

    @property
    def burn_in_period(self) -> int:
        return int(self.params["burn_in_bars"])

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("vol_ewma",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        lam = float(self.params["lam"])
        seed_bars = int(self.params["burn_in_bars"])
        scale = math.sqrt(float(self.params["annualisation_factor"]))

        r2 = log_returns(frame["close"]).pow(2.0)
        n = len(r2)
        var = np.full(n, np.nan, dtype="float64")
        if n > seed_bars:
            arr = r2.to_numpy(dtype="float64")
            # Seed: gemiddelde van r^2 over r_1..r_seed. Positie 0 draagt geen
            # return en telt dus niet mee.
            seed_slice = arr[1 : seed_bars + 1]
            var[seed_bars] = float(np.mean(seed_slice))
            # Voorwaartse recursie. `ewm(adjust=False)` op de staart levert exact
            # sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r_t^2.
            tail = arr[seed_bars:].copy()
            tail[0] = var[seed_bars]
            var[seed_bars:] = (
                pd.Series(tail)
                .ewm(alpha=1.0 - lam, adjust=False)
                .mean()
                .to_numpy(dtype="float64")
            )
        vol = np.sqrt(var) * scale
        return pd.DataFrame({self.output_columns[0]: vol}, index=frame.index)


class ParkinsonVolatility(BaseFeature):
    """Parkinson high/low range-volatiliteit over `window` bars.

        sigma_t = sqrt( mean( ln(H/L)^2 ) / (4 ln 2) ) * sqrt(annualisation)

    Gebruikt uitsluitend bar-lokale informatie plus een achterwaarts venster;
    er is geen return-differentie nodig, dus de burn-in is `window - 1`.
    De schatter is per constructie niet-negatief.
    """

    name: ClassVar[str] = "parkinson_volatility"
    input_spec: ClassVar[InputSpec] = InputSpec(
        datasets=("ohlcv",), columns=("high", "low")
    )

    def __init__(self, *, window: int, annualisation_factor: float) -> None:
        require(
            window > 0,
            "Parkinson vereist een positief venster.",
            DataContractError,
            window=window,
        )
        require(
            annualisation_factor > 0.0,
            "annualisation_factor moet positief zijn.",
            DataContractError,
            annualisation_factor=annualisation_factor,
        )
        super().__init__(
            params={"window": int(window),
                    "annualisation_factor": float(annualisation_factor)}
        )

    @property
    def burn_in_period(self) -> int:
        return int(self.params["window"]) - 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (f"vol_parkinson_{self.params['window']}",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        window = int(self.params["window"])
        scale = math.sqrt(float(self.params["annualisation_factor"]))
        high = frame["high"].astype("float64")
        low = frame["low"].astype("float64")
        require(
            bool((high.dropna() > 0.0).all()) and bool((low.dropna() > 0.0).all()),
            "Niet-positieve high of low; ln(H/L) is dan niet gedefinieerd.",
            DataContractError,
        )
        require(
            bool((high >= low).dropna().all()),
            "high < low in de OHLCV-reeks. Dat is geen marktgebeurtenis maar "
            "een databreuk en hoort in Phase 1 te zijn afgevangen.",
            DataContractError,
            n_violations=int((high < low).sum()),
        )
        log_hl_sq = np.log(high / low).pow(2.0)
        var = log_hl_sq.rolling(window, min_periods=window).mean() * _PARKINSON_SCALE
        vol = np.sqrt(var) * scale
        return pd.DataFrame(
            {self.output_columns[0]: vol.to_numpy(dtype="float64")},
            index=frame.index,
        )
