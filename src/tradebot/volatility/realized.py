"""De variantieproxy waartegen de QLIKE-competitie wordt beslecht — deliverable 11.

WAAROM DIT MODULE BESTAAT NAAST `parkinson.py`, `garman_klass.py` EN CONSORTEN
==============================================================================
Die vier modules leveren FEATURES: rollende volatiliteit over een venster van 14
of 20 bars, met `.ffill().fillna(0.0)` aan het eind en een `return 0.0` wanneer
de reeks korter is dan het venster. Voor een feature in een ML-pijplijn is dat
verdedigbaar.

Voor een QLIKE-proxy is het fataal, op twee manieren:

1. **Rollend is het verkeerde object.** QLIKE vergelijkt de forecast VOOR BAR t
   met de gerealiseerde variantie OP BAR t. Een 14-bars gemiddelde is geen
   realisatie van bar t; het is een gladgestreken versie waarin de forecast
   grotendeels al besloten ligt. Elke estimator zou daar goed op scoren.
2. **`fillna(0.0)` is een QLIKE-bom.** QLIKE is
   ``RV/sigma^2 - ln(RV/sigma^2) - 1``; bij ``RV = 0`` is ``ln(0) = -inf``. Een
   ontbrekende bar die stilzwijgend nul wordt, maakt de loss oneindig — en wie
   dat vervolgens met ``nanmean`` wegmiddelt, heeft een competitie waarin de
   winnaar wordt bepaald door welk model toevallig op de gevulde bars stond.

Dit module levert daarom PER-BAR estimators, zonder venster en zonder invulling,
met een hard invoercontract.

WAT DE PROXY OP DEZE STORE WEL EN NIET IS
==========================================
`docs/DATA_REGISTER.md` en `reports/phase1_data_gap.md` leggen vast dat de
gecertificeerde store **0 rijen 1m/5m** bevat — een expliciet scope-besluit uit
Phase 1. Realized Variance uit intraday returns en HAR-RV daarop zijn dus niet te
schatten. :func:`realized_variance_from_intraday` bestaat en crasht; hij valt
NIET terug op een dagproxy, want dat zou de belangrijkste beperking van deze
fase onzichtbaar maken.

De proxy is daarom een **daily range-estimator**. Dat is legitiem en niet een
noodgreep: QLIKE is robuust tegen proxy-ruis zolang de proxy CONDITIONEEL ZUIVER
is (Patton 2011), en Parkinson, Garman-Klass en Rogers-Satchell zijn alle drie
zuiver voor de geintegreerde variantie onder een driftloze GBM binnen de dag.

De prijs is EFFICIENTIE, niet zuiverheid. Relatieve efficientie ten opzichte van
de gekwadrateerde dagreturn (Garman & Klass 1980):

    gekwadrateerde return   1,0   (var = 2 sigma^4)
    Parkinson               ~5,2
    Garman-Klass            ~7,4
    Rogers-Satchell         ~8,0
    5-minuts realized var   ~50+  (niet beschikbaar op deze store)

Een ruizige proxy vergroot de variantie van het QLIKE-VERSCHIL en verkleint dus
de power van de DM-HLN-toets. Dat staat vooraf in de H1-pre-registratie en het
hoort in het competitierapport, niet in een voetnoot.

DE NON-NEGATIVITEITSGARANTIE
=============================
Alle drie de per-bar estimators zijn niet-negatief GEGEVEN een geldige OHLC-bar,
en dat is een bewijs en geen aanname:

    Parkinson        (ln(H/L))^2 / (4 ln 2)         -- een kwadraat, dus >= 0
    Garman-Klass     0,5 x^2 - (2 ln2 - 1) y^2      met x = ln(H/L), y = ln(C/O).
                     Uit L <= min(O,C) <= max(O,C) <= H volgt x >= |y|, dus
                     0,5 x^2 - 0,383 y^2 >= 0,117 y^2 >= 0.
    Rogers-Satchell  ln(H/C)ln(H/O) + ln(L/C)ln(L/O)
                     Beide factoren van de eerste term >= 0, beide van de tweede
                     <= 0; twee producten van gelijkgetekende factoren, dus >= 0.

Elk van de drie bewijzen steunt op ``L <= min(O,C) <= max(O,C) <= H``. Dat is
een eigenschap van de INVOER en wordt daarom afgedwongen door
:func:`require_valid_ohlc` in plaats van aangenomen.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md 9.1, 9.2; Parkinson (1980);
Garman & Klass (1980); Rogers & Satchell (1991); Yang & Zhang (2000);
Patton (2011).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, DependencyMissingError, require

__all__ = [
    "PROXY_EFFICIENCY",
    "RangeProxy",
    "build_range_proxies",
    "garman_klass_variance",
    "parkinson_variance",
    "realized_variance_from_intraday",
    "require_valid_ohlc",
    "rogers_satchell_variance",
    "squared_return_variance",
    "yang_zhang_variance",
]

#: Relatieve efficientie ten opzichte van de gekwadrateerde dagreturn, uit de
#: oorspronkelijke artikelen. Hoger = minder ruis in de proxy = meer power in
#: de QLIKE-competitie. Deze getallen staan hier zodat het rapport ze kan tonen
#: naast de gemeten uitkomst, in plaats van de proxykeuze onbesproken te laten.
PROXY_EFFICIENCY: Mapping[str, float] = {
    "squared_return": 1.0,
    "parkinson": 5.2,
    "garman_klass": 7.4,
    "rogers_satchell": 8.0,
    "yang_zhang": 7.3,
}

_FOUR_LN2 = 4.0 * np.log(2.0)
_GK_COEF = 2.0 * np.log(2.0) - 1.0


def require_valid_ohlc(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    *, name: str = "",
) -> None:
    """Dwing ``L <= min(O,C) <= max(O,C) <= H`` en strikt positieve prijzen af.

    Dit is niet defensief programmeren maar de premisse waarop de
    non-negativiteitsbewijzen in de moduledocstring rusten. Een bar waarvan de
    low boven de close ligt, is een datafout; hem doorlaten levert een negatieve
    Garman-Klass-variantie op, en die maakt QLIKE ongedefinieerd.
    """
    arrays = {"open": open_, "high": high, "low": low, "close": close}
    for field_name, arr in arrays.items():
        require(
            bool(np.all(np.isfinite(arr))),
            f"OHLC-contract: `{field_name}` bevat niet-eindige waarden. Er "
            "wordt NIET geimputeerd; een gat is een databevinding.",
            DataContractError,
            series=name, field=field_name,
            n_non_finite=int(np.count_nonzero(~np.isfinite(arr))),
        )
        require(
            bool(np.all(arr > 0.0)),
            f"OHLC-contract: `{field_name}` bevat niet-positieve prijzen; de "
            "logaritmen in elke range-estimator zijn daar niet gedefinieerd.",
            DataContractError,
            series=name, field=field_name, min_value=float(np.min(arr)),
        )
    upper = np.maximum(open_, close)
    lower = np.minimum(open_, close)
    require(
        bool(np.all(high >= upper - 1e-12)),
        "OHLC-contract geschonden: high ligt onder open of close. De "
        "non-negativiteit van Garman-Klass en Rogers-Satchell steunt hierop.",
        DataContractError,
        series=name,
        n_violations=int(np.count_nonzero(high < upper - 1e-12)),
        worst=float(np.min(high - upper)),
    )
    require(
        bool(np.all(low <= lower + 1e-12)),
        "OHLC-contract geschonden: low ligt boven open of close.",
        DataContractError,
        series=name,
        n_violations=int(np.count_nonzero(low > lower + 1e-12)),
        worst=float(np.max(low - lower)),
    )


def _assert_non_negative(values: np.ndarray, estimator: str) -> np.ndarray:
    """De bewijzen zijn redeneringen; dit is de meting.

    Faalt zij, dan is een aanname onder het bewijs gebroken en mag er geen
    proxy naar de QLIKE-competitie ontsnappen.
    """
    require(
        bool(np.all(values >= 0.0)),
        f"{estimator} produceerde een negatieve variantie ondanks een geldig "
        "OHLC-contract. Dit hoort onmogelijk te zijn; QLIKE zou hierop "
        "ongedefinieerd worden.",
        DataContractError,
        estimator=estimator, min_value=float(np.min(values)),
        n_negative=int(np.count_nonzero(values < 0.0)),
    )
    return values


def squared_return_variance(close: np.ndarray) -> np.ndarray:
    """De naieve proxy: ``(ln(C_t / C_{t-1}))^2``. Bar 0 is NaN.

    Zuiver maar zeer ruizig (var = 2 sigma^4). Staat hier omdat de competitie
    ook tegen de meest naieve proxy moet kunnen worden herhaald: verandert de
    RANGSCHIKKING van de modellen niet tussen proxies, dan is de conclusie
    robuust tegen de proxykeuze, en dat is een sterker resultaat dan een
    competitie op alleen de efficientste proxy.
    """
    close = np.asarray(close, dtype=np.float64)
    out = np.full(close.size, np.nan, dtype=np.float64)
    out[1:] = np.log(close[1:] / close[:-1]) ** 2
    return out


def parkinson_variance(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Parkinson (1980), PER BAR: ``(ln(H/L))^2 / (4 ln 2)``."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return _assert_non_negative(
        np.log(high / low) ** 2 / _FOUR_LN2, "parkinson")


def garman_klass_variance(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
) -> np.ndarray:
    """Garman-Klass (1980), PER BAR: ``0,5 (ln H/L)^2 - (2 ln2 - 1)(ln C/O)^2``."""
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    hl = np.log(high / low)
    co = np.log(close / open_)
    return _assert_non_negative(
        0.5 * hl**2 - _GK_COEF * co**2, "garman_klass")


def rogers_satchell_variance(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
) -> np.ndarray:
    """Rogers-Satchell (1991), PER BAR. Zuiver ook bij een niet-nul drift.

    Dat is het onderscheidende kenmerk: Parkinson en Garman-Klass gaan uit van
    een driftloze GBM binnen de bar en overschatten de variantie systematisch
    wanneer de dag een sterke richting had. Op crypto-dagbars met bewegingen van
    tientallen procenten is dat geen theoretisch detail.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return _assert_non_negative(
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_),
        "rogers_satchell",
    )


def yang_zhang_variance(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    *, window: int = 20,
) -> np.ndarray:
    """Yang-Zhang (2000). **Inherent een VENSTER-estimator, geen per-bar proxy.**

    YZ combineert drie componenten waarvan er twee een VARIANTIE OVER EEN VENSTER
    zijn: de overnight-gapvariantie en de open-close-variantie. Er bestaat geen
    per-bar YZ, want de variantie van een enkele observatie is niet gedefinieerd.

    Daarom is YZ hier expliciet NIET de QLIKE-proxy. Hij staat er omdat
    deliverable 11 hem vraagt en als vergelijkingsmateriaal voor de
    vol-forecasts zelf, maar wie hem als realisatie gebruikt, vergelijkt een
    forecast met een 20-bars gemiddelde waarin die forecast al grotendeels
    besloten ligt.

    De eerste ``window`` bars zijn NaN. Er wordt niet vooruit gevuld.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    require(
        window >= 2,
        "Yang-Zhang met een venster onder 2 bars: de componentvarianties zijn "
        "dan niet gedefinieerd.",
        DataContractError, window=window,
    )
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return out

    overnight = np.full(n, np.nan)
    overnight[1:] = np.log(open_[1:] / close[:-1])
    open_close = np.log(close / open_)
    rs = rogers_satchell_variance(open_, high, low, close)

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    var_on = pd.Series(overnight).rolling(window).var(ddof=1).to_numpy()
    var_oc = pd.Series(open_close).rolling(window).var(ddof=1).to_numpy()
    mean_rs = pd.Series(rs).rolling(window).mean().to_numpy()

    values = var_on + k * var_oc + (1.0 - k) * mean_rs
    valid = np.isfinite(values)
    out[valid] = values[valid]
    _assert_non_negative(out[valid], "yang_zhang")
    return out


def realized_variance_from_intraday(
    intraday_closes: Mapping[Any, np.ndarray] | None,
    *,
    granularity: str,
) -> np.ndarray:
    """Realized Variance uit intraday returns. **Crasht op deze store.**

    Er is bewust geen fallback naar een dagproxy. Zo'n fallback zou de
    belangrijkste beperking van deze fase onzichtbaar maken: de gecertificeerde
    store bevat 0 rijen 1m/5m (`reports/phase1_data_gap.md`), en elk resultaat
    dat "realized variance" heet terwijl het een range-estimator is, is
    verkeerd gelabeld bewijs.

    De functie bestaat zodat de dag waarop de intraday-data er wel is, een
    aanroepsite hoeft te worden aangesloten in plaats van een spoor te moeten
    worden teruggevonden.
    """
    require(
        intraday_closes is not None and len(intraday_closes) > 0,
        "Realized Variance vereist intraday bars, en de gecertificeerde store "
        "bevat er geen. Er wordt NIET teruggevallen op een dagelijkse "
        "range-estimator: die is een ANDERE grootheid met een andere "
        "signaal-ruisverhouding, en hem zo noemen zou het bewijs verkeerd "
        "labelen. Zie docs/DATA_REGISTER.md en reports/phase1_data_gap.md.",
        DependencyMissingError,
        required_granularity=granularity,
        available="geen",
    )
    daily: list[float] = []
    for _day, closes in sorted(intraday_closes.items()):  # pragma: no cover
        arr = np.asarray(closes, dtype=np.float64)
        returns = np.log(arr[1:] / arr[:-1])
        daily.append(float(np.sum(returns**2)))
    return np.asarray(daily, dtype=np.float64)


@dataclass(frozen=True)
class RangeProxy:
    """Een variantieproxy, met alles wat het rapport over haar moet zeggen."""

    name: str
    values: pd.Series
    #: Bars waarop de proxy exact nul is. Die zijn NIET bruikbaar voor QLIKE
    #: (``ln(0)``) en worden geteld in plaats van weggegooid, zodat het rapport
    #: kan laten zien hoeveel van de steekproef eraan opgaat.
    n_zero_bars: int
    n_valid_bars: int
    relative_efficiency: float

    def as_record(self) -> dict[str, Any]:
        finite = self.values.to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite) & (finite > 0.0)]
        return {
            "name": self.name,
            "n_valid_bars": self.n_valid_bars,
            "n_zero_bars": self.n_zero_bars,
            "relative_efficiency_vs_squared_return": self.relative_efficiency,
            "mean_annualised_vol": (
                float(np.sqrt(finite.mean() * 365.0)) if finite.size
                else float("nan")),
            "median_variance": (
                float(np.median(finite)) if finite.size else float("nan")),
        }


def build_range_proxies(
    ohlc: pd.DataFrame, *, name: str = "", yang_zhang_window: int = 20,
) -> dict[str, RangeProxy]:
    """Bouw alle proxies op een OHLC-frame, met het contract vooraf afgedwongen."""
    o = ohlc["open"].to_numpy(dtype=np.float64)
    h = ohlc["high"].to_numpy(dtype=np.float64)
    lo = ohlc["low"].to_numpy(dtype=np.float64)
    c = ohlc["close"].to_numpy(dtype=np.float64)
    require_valid_ohlc(o, h, lo, c, name=name)

    raw = {
        "squared_return": squared_return_variance(c),
        "parkinson": parkinson_variance(h, lo),
        "garman_klass": garman_klass_variance(o, h, lo, c),
        "rogers_satchell": rogers_satchell_variance(o, h, lo, c),
        "yang_zhang": yang_zhang_variance(o, h, lo, c, window=yang_zhang_window),
    }
    out: dict[str, RangeProxy] = {}
    for proxy_name, values in raw.items():
        finite = np.isfinite(values)
        out[proxy_name] = RangeProxy(
            name=proxy_name,
            values=pd.Series(values, index=ohlc.index, name=proxy_name),
            n_zero_bars=int(np.count_nonzero(finite & (values == 0.0))),
            n_valid_bars=int(np.count_nonzero(finite & (values > 0.0))),
            relative_efficiency=PROXY_EFFICIENCY[proxy_name],
        )
    return out
