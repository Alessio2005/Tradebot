"""Causale carry- en positioneringsfeatures op de gecertificeerde dagbar.

Fase 10, stap 15.3. Dit bestand HEETTE `features/microstructure.py` en bevatte
twee modules onder een naam. De ene helft leest `funding_rate` en
`open_interest` uit de PIT-store; de andere leest taker-volumes en quotes die in
het meetdomein (AD-23) niet bestaan. De splitsing voert het verdict uit
`docs/MEASUREMENT_DOMAIN.md` uit.

Wat hier staat is de behouden helft:

* `build_certified_micro_frame` -- het gecertificeerde inputframe;
* `FundingRateMean`, `FundingRateZScore`, `OpenInterestLogChange`;
* `amihud_illiquidity` -- verhuisd uit de andere helft. Hij stond tussen de
  order-flow-functies maar leest uitsluitend `close` en `volume` en is daarmee
  domein-conform. Zijn AFML-hoofdstuk is geen argument; wat hij LEEST wel.

De naam `positioning` komt uit de banner die hieronder al stond: *"PHASE 2 - L3
CAUSAL CARRY & POSITIONING FEATURES"*. Funding en open interest meten
POSITIONERING -- wat de markt aanhoudt -- en niet microstructuur.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def amihud_illiquidity(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Amihud (2002) illiquidity ratio: |r_t| / volume_t.

    High ratio = price moves a lot per unit of volume = illiquid.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1)).abs()
    vol_col = "real_volume" if "real_volume" in df.columns else (
        "tick_volume" if "tick_volume" in df.columns else "volume"
    )
    vol = df[vol_col].fillna(0.0)
    raw = log_ret / vol.replace(0, 1e-9)
    return (
        raw.rolling(window, min_periods=1).median()
        .clip(upper=raw.quantile(0.99))
        .rename("feat_amihud")
    )


# =========================================================================== #
# PHASE 2 - L3 CAUSAL CARRY & POSITIONING FEATURES
# =========================================================================== #
# Alles BOVEN deze regel is legacy (order-flow op intraday data, DI-12, Phase 3):
# vrije functies zonder provenance-contract, met `fillna(0.0)`-idiomen en
# hardcoded vensters. Die code wordt in Phase 3 herzien en is hier bewust
# ONGEWIJZIGD gelaten - hem nu aanraken zou de Phase 1-baseline onvergelijkbaar
# maken en `alpha/microstructure.py` breken.
#
# Alles HIERONDER is Phase 2-materiaal: `BaseFeature`-subklassen op de
# gecertificeerde `funding`- en `open_interest`-reeksen uit de PIT-store, met
# nul hardcoded parameters en een verplichte burn-in in NaN.
#
# De multi-granulaire koppeling (8h funding -> 1d bars, 1d open interest -> 1d
# bars) loopt UITSLUITEND via `utils.time.asof_join(direction="backward")` met
# een expliciete tolerance uit `conf/features/default.yaml`. Dat is exit
# criterium 4 van Phase 2.
#
# LET OP de keuze van het koppelmoment. De join gebruikt de `asof_ts` van de
# OHLCV-bar (het sluitingsmoment), niet zijn `event_ts` (het openingsmoment).
# Dat is het moment waarop de feature daadwerkelijk beschikbaar is, en het is
# tevens de index van de feature-matrix. Koppelen op `event_ts` zou eveneens
# causaal zijn maar gooit tot een volledige bar aan bekende funding-informatie
# weg; koppelen op een LATER moment zou een lek zijn.
# --------------------------------------------------------------------------- #
from typing import Any as _Any
from typing import ClassVar as _ClassVar

from ..utils.failfast import DataContractError as _DataContractError
from ..utils.failfast import require as _require
from ..utils.time import asof_join as _asof_join
from .base import BaseFeature as _BaseFeature
from .base import CertifiedFrame as _CertifiedFrame
from .base import DataRegister as _DataRegister
from .base import InputSpec as _InputSpec
from .base import load_certified_series as _load_certified_series

#: Naam van de kolom met de laatst BEKENDE funding rate op het beslismoment.
FUNDING_COLUMN = "funding_rate"
#: Naam van de kolom met de laatst BEKENDE open interest op het beslismoment.
OPEN_INTEREST_COLUMN = "open_interest"


def build_certified_micro_frame(
    store: _Any,
    register: _DataRegister,
    *,
    symbol: str,
    granularity: str,
    funding_granularity: str,
    open_interest_granularity: str,
    funding_tolerance: pd.Timedelta,
    open_interest_tolerance: pd.Timedelta,
    asset_class: str,
) -> _CertifiedFrame:
    """Bouw het gecertificeerde inputframe voor de microstructuurfeatures.

    OHLCV vormt het bar-raster; funding en open interest worden er backward
    op geprojecteerd via `asof_join`. Alle drie de reeksen worden eerst
    geverifieerd tegen het Phase 1 data-register: hun `data_hash` gaat mee in
    het resultaat en daarmee in elke `feature_hash` die eruit volgt.

    Elke tolerance is verplicht en komt uit `conf/features/default.yaml`. Zonder
    bovengrens draagt `merge_asof` een waarde onbeperkt vooruit, waardoor een
    funding rate uit 2021 nog aan een bar uit 2026 wordt gekoppeld.
    """
    ohlcv, ohlcv_hash = _load_certified_series(
        store, register, asset_class=asset_class, dataset="ohlcv",
        symbol=symbol, granularity=granularity,
    )
    funding, funding_hash = _load_certified_series(
        store, register, asset_class=asset_class, dataset="funding",
        symbol=symbol, granularity=funding_granularity,
    )
    oi, oi_hash = _load_certified_series(
        store, register, asset_class=asset_class, dataset="open_interest",
        symbol=symbol, granularity=open_interest_granularity,
    )

    # Het beslismoment: wanneer is de OHLCV-bar compleet en dus bruikbaar?
    decision = pd.DatetimeIndex(
        pd.to_datetime(ohlcv["asof_ts_ns"].to_numpy(), unit="ns", utc=True),
        name="decision_ts",
    )
    _require(
        bool(decision.is_monotonic_increasing),
        "De asof-reeks van de OHLCV-bars is niet oplopend; merge_asof levert "
        "dan stille onzin in plaats van een fout.",
        _DataContractError,
        symbol=symbol,
    )
    left = pd.DataFrame(index=decision)

    right_funding = pd.DataFrame(
        {
            "asof_ts": pd.to_datetime(
                funding["asof_ts_ns"].to_numpy(), unit="ns", utc=True
            ),
            FUNDING_COLUMN: funding[FUNDING_COLUMN].to_numpy(dtype="float64"),
        }
    ).sort_values("asof_ts", kind="stable")
    right_oi = pd.DataFrame(
        {
            "asof_ts": pd.to_datetime(
                oi["asof_ts_ns"].to_numpy(), unit="ns", utc=True
            ),
            OPEN_INTEREST_COLUMN: oi[OPEN_INTEREST_COLUMN].to_numpy(dtype="float64"),
        }
    ).sort_values("asof_ts", kind="stable")

    joined_funding = _asof_join(
        left, right_funding, asof_col="asof_ts", tolerance=funding_tolerance
    )
    joined_oi = _asof_join(
        left, right_oi, asof_col="asof_ts", tolerance=open_interest_tolerance
    )

    frame = ohlcv.copy()
    frame[FUNDING_COLUMN] = joined_funding[FUNDING_COLUMN].to_numpy(dtype="float64")
    frame[OPEN_INTEREST_COLUMN] = joined_oi[OPEN_INTEREST_COLUMN].to_numpy(
        dtype="float64"
    )
    return _CertifiedFrame(
        frame=frame,
        symbol=symbol,
        data_hashes=(
            (_DataRegister.key(asset_class, "ohlcv", symbol, granularity), ohlcv_hash),
            (
                _DataRegister.key(asset_class, "funding", symbol, funding_granularity),
                funding_hash,
            ),
            (
                _DataRegister.key(
                    asset_class, "open_interest", symbol, open_interest_granularity
                ),
                oi_hash,
            ),
        ),
    )


class FundingRateMean(_BaseFeature):
    """Rolling gemiddelde van de laatst bekende funding rate over `window` bars.

    De carry-component van een perpetual: een aanhoudend positieve funding rate
    betekent dat longs betalen en is een directe kostenpost voor een longpositie.
    Het gemiddelde dempt de ruis van een enkele settlement.
    """

    name: _ClassVar[str] = "funding_rate_mean"
    input_spec: _ClassVar[_InputSpec] = _InputSpec(
        datasets=("ohlcv", "funding"), columns=(FUNDING_COLUMN,)
    )

    def __init__(self, *, window: int) -> None:
        _require(
            window > 0,
            "Een gemiddelde over een niet-positief venster bestaat niet.",
            _DataContractError,
            window=window,
        )
        super().__init__(params={"window": int(window)})

    @property
    def burn_in_period(self) -> int:
        return int(self.params["window"]) - 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (f"micro_funding_mean_{self.params['window']}",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        window = int(self.params["window"])
        rate = frame[FUNDING_COLUMN].astype("float64")
        out = rate.rolling(window, min_periods=window).mean()
        return pd.DataFrame(
            {self.output_columns[0]: out.to_numpy(dtype="float64")}, index=frame.index
        )


class FundingRateZScore(_BaseFeature):
    """EXPANDING z-score van de funding rate.

    Bewust expanding en niet rolling-met-sample-statistiek: de normalisatie mag
    uitsluitend informatie tot en met `t` gebruiken. Een z-score op basis van het
    gemiddelde en de spreiding over de VOLLEDIGE reeks is dezelfde klasse fout
    als DI-2 - alleen zichtbaarder, omdat hij per constructie rond nul
    gecentreerd is over precies het venster dat wordt getoetst.
    """

    name: _ClassVar[str] = "funding_rate_zscore"
    input_spec: _ClassVar[_InputSpec] = _InputSpec(
        datasets=("ohlcv", "funding"), columns=(FUNDING_COLUMN,)
    )

    def __init__(self, *, min_periods: int) -> None:
        _require(
            min_periods > 1,
            "Een z-score vereist een standaarddeviatie, en die is niet "
            "gedefinieerd over minder dan twee observaties.",
            _DataContractError,
            min_periods=min_periods,
        )
        super().__init__(params={"min_periods": int(min_periods)})

    @property
    def burn_in_period(self) -> int:
        return int(self.params["min_periods"]) - 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("micro_funding_zscore",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        min_periods = int(self.params["min_periods"])
        rate = frame[FUNDING_COLUMN].astype("float64")
        exp = rate.expanding(min_periods=min_periods)
        mean = exp.mean()
        std = exp.std()
        released = std.notna()
        _require(
            bool((std[released] > 0.0).all()),
            "De expanding standaarddeviatie van de funding rate is exact nul. "
            "Een volstrekt constante funding rate over het volledige venster is "
            "geen marktuitkomst maar een databreuk; delen zou +/-inf opleveren "
            "en dat propageert stilzwijgend door elk later gewicht.",
            _DataContractError,
            n_zero=int((std[released] <= 0.0).sum()),
        )
        z = (rate - mean) / std
        return pd.DataFrame(
            {self.output_columns[0]: z.to_numpy(dtype="float64")}, index=frame.index
        )


class OpenInterestLogChange(_BaseFeature):
    """Log-verandering in open interest over `window` bars.

        d_t = ln(OI_t) - ln(OI_{t-window})

    Positioneringsmaat: stijgende open interest bij stijgende prijs duidt op
    nieuwe instroom, stijgende open interest bij dalende prijs op opbouw van
    shorts. De log-vorm maakt de maat schaalvrij tussen instrumenten met sterk
    verschillende contractgroottes.

    Begint een symbool met open interest later dan met OHLCV - wat op deze
    dataset voorkomt, bijvoorbeeld BTCUSDT met OHLCV vanaf 2020-03-25 en open
    interest vanaf 2020-08-05 - dan zijn de vroege waarden NaN. Dat is geen gat
    om te vullen maar het feit dat de reeks nog niet bestond.
    """

    name: _ClassVar[str] = "open_interest_log_change"
    input_spec: _ClassVar[_InputSpec] = _InputSpec(
        datasets=("ohlcv", "open_interest"), columns=(OPEN_INTEREST_COLUMN,)
    )

    def __init__(self, *, window: int) -> None:
        _require(
            window > 0,
            "Een verandering over een niet-positief venster bestaat niet.",
            _DataContractError,
            window=window,
        )
        super().__init__(params={"window": int(window)})

    @property
    def burn_in_period(self) -> int:
        return int(self.params["window"])

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (f"micro_oi_logchg_{self.params['window']}",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        window = int(self.params["window"])
        oi = frame[OPEN_INTEREST_COLUMN].astype("float64")
        _require(
            bool((oi.dropna() > 0.0).all()),
            "Niet-positieve open interest; ln(OI) is dan niet gedefinieerd en "
            "levert -inf op.",
            _DataContractError,
            n_non_positive=int((oi.dropna() <= 0.0).sum()),
        )
        log_oi = pd.Series(np.log(oi.to_numpy(dtype="float64")), index=oi.index)
        out = log_oi - log_oi.shift(window)
        return pd.DataFrame(
            {self.output_columns[0]: out.to_numpy(dtype="float64")}, index=frame.index
        )
