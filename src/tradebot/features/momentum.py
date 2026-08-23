# src/tradebot/features/momentum.py
"""L3 - causale trend- en momentumfeatures. Phase 2, deliverable 2.

Twee families, beide strikt achterwaarts kijkend:

* `RollingLogReturn` - het cumulatieve log-rendement over een venster dat
  `skip_bars` vóór `t` eindigt. De skip is geen kosmetiek: het rendement van de
  laatste bar is op `t` weliswaar bekend, maar de bekende korte-termijn reversal
  in crypto maakt hem als trendsignaal misleidend. De skip staat in
  `conf/features/default.yaml` en gaat mee in de `feature_hash`.
* `EwmaReturnSpread` - het verschil tussen een snelle en een trage EWMA van de
  LOG-prijs. Omdat het een verschil van logaritmen is, is de uitkomst
  schaalvrij: hij is vergelijkbaar tussen BTC op 60.000 en DOT op 6.

Beide gebruiken uitsluitend `shift(+k)` en `ewm(adjust=False)`. Er komt geen
`shift(-k)`, geen `center=True` en geen `fillna` aan te pas; tijdens de burn-in
is de waarde NaN en dat blijft hij.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 8, 11, 19 (L3), 26.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from .base import BaseFeature, InputSpec

__all__ = ["EwmaReturnSpread", "RollingLogReturn"]


class RollingLogReturn(BaseFeature):
    """Cumulatief log-rendement over `window` bars, eindigend `skip_bars` vóór t.

        m_t = ln(P_{t-skip}) - ln(P_{t-skip-window})

    Bij `skip_bars = 0` is dit het gewone rendement over het venster. De waarde
    op `t` hangt uitsluitend af van prijzen op of vóór `t`, dus truncatie na `t`
    verandert er per constructie niets aan.
    """

    name: ClassVar[str] = "rolling_log_return"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self, *, window: int, skip_bars: int) -> None:
        require(
            window > 0,
            "Een rendement over een niet-positief venster bestaat niet.",
            DataContractError,
            window=window,
        )
        require(
            skip_bars >= 0,
            "Een NEGATIEVE skip verschuift het venster naar de TOEKOMST. Dat is "
            "geen parameterkeuze maar een lookahead-lek.",
            DataContractError,
            skip_bars=skip_bars,
        )
        super().__init__(params={"window": int(window), "skip_bars": int(skip_bars)})

    @property
    def burn_in_period(self) -> int:
        return int(self.params["window"]) + int(self.params["skip_bars"])

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (f"mom_logret_{self.params['window']}",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        window = int(self.params["window"])
        skip = int(self.params["skip_bars"])
        close = frame["close"].astype("float64")
        require(
            bool((close.dropna() > 0.0).all()),
            "Niet-positieve prijs; ln(P) is niet gedefinieerd.",
            DataContractError,
            n_non_positive=int((close.dropna() <= 0.0).sum()),
        )
        log_price = pd.Series(
            np.log(close.to_numpy(dtype="float64")), index=close.index
        )
        mom = log_price.shift(skip) - log_price.shift(skip + window)
        return pd.DataFrame(
            {self.output_columns[0]: mom.to_numpy(dtype="float64")}, index=frame.index
        )


class EwmaReturnSpread(BaseFeature):
    """Schaalvrije trendfilter: snelle minus trage EWMA van de log-prijs.

        s_t = EWMA_fast(ln P)_t - EWMA_slow(ln P)_t

    `adjust=False` levert de zuivere recursie `y_t = (1-a) y_{t-1} + a x_t`, die
    per constructie causaal is: `y_t` hangt af van `x_0..x_t` en van niets
    daarna. `min_periods` maskeert de opstartfase met NaN in plaats van hem op
    te vullen - een EWMA die op bar 1 al een 'trend' rapporteert, rapporteert
    zijn eigen seed.
    """

    name: ClassVar[str] = "ewma_return_spread"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self, *, fast_span: int, slow_span: int) -> None:
        require(
            fast_span > 0 and slow_span > 0,
            "EWMA-spans moeten positief zijn.",
            DataContractError,
            fast_span=fast_span,
            slow_span=slow_span,
        )
        require(
            slow_span > fast_span,
            "slow_span moet groter zijn dan fast_span; anders is de spread per "
            "constructie omgekeerd van teken en meet hij het tegenovergestelde "
            "van wat de naam belooft.",
            DataContractError,
            fast_span=fast_span,
            slow_span=slow_span,
        )
        super().__init__(
            params={"fast_span": int(fast_span), "slow_span": int(slow_span)}
        )

    @property
    def burn_in_period(self) -> int:
        # min_periods=slow_span geeft de eerste waarde op positie slow_span - 1.
        return int(self.params["slow_span"]) - 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("mom_ewma_spread",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        fast = int(self.params["fast_span"])
        slow = int(self.params["slow_span"])
        close = frame["close"].astype("float64")
        require(
            bool((close.dropna() > 0.0).all()),
            "Niet-positieve prijs; ln(P) is niet gedefinieerd.",
            DataContractError,
            n_non_positive=int((close.dropna() <= 0.0).sum()),
        )
        log_price = pd.Series(
            np.log(close.to_numpy(dtype="float64")), index=close.index
        )
        ewma_fast = log_price.ewm(span=fast, adjust=False, min_periods=slow).mean()
        ewma_slow = log_price.ewm(span=slow, adjust=False, min_periods=slow).mean()
        spread = ewma_fast - ewma_slow
        return pd.DataFrame(
            {self.output_columns[0]: spread.to_numpy(dtype="float64")},
            index=frame.index,
        )
