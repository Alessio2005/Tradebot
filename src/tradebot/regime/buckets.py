"""M0 Causal Vol-Buckets — de baseline die het HMM moet verslaan. Deliverable 14.

Nul latente toestanden. Nul geschatte parameters. Nul lekrisico.

Dat is geen bescheidenheid maar de reden dat M0 de juiste tegenstander is. Een
HMM schat overgangskansen, gemiddelden en covarianties uit dezelfde data waarop
hij vervolgens wordt beoordeeld; op 1.743 bars met enkele honderden observaties
per fold is de onzekerheid daarin aanzienlijk. M0 heeft die onzekerheid per
constructie NIET. Elk verschil dat het HMM laat zien, moet dus groot genoeg zijn
om een schattingsvoordeel te overtreffen dat M0 niet nodig heeft. Dat maakt hem
een oneerlijk sterke tegenstander -- en precies daarom de eerlijke baseline.

DE TWEE ASSEN, EN WAAROM ZE MOETEN SAMENVALLEN
===============================================
M0 kijkt naar twee dingen die verschillende vragen beantwoorden:

    z-score van log-EWMA-vol   WAAR staat de volatiliteit? (niveau)
    ATR(5) / ATR(20)           WAARHEEN beweegt zij? (richting)

Een bar verlaat NORMAAL alleen wanneer BEIDE assen dezelfde kant op wijzen:

    HOOG     z >= zscore_high  EN  atr_ratio >= atr_ratio_high
    LAAG     z <= zscore_low   EN  atr_ratio <= atr_ratio_low
    NORMAAL  al het overige

Waarom EN en niet OF: met OF zou elke tijdelijke uitschieting in een van beide
maten een regimewisseling opleveren, en regimewisselingen zijn TURNOVER (§0.4).
Op dit boek domineren fees -- de `xs_momentum_risk_parity`-track betaalde
€ 2.106 aan fees over 9.655 orders -- dus een baseline die vaker wisselt dan
nodig, verliest van zichzelf. De conjunctie is de conservatieve keuze en staat
als architectuurbesluit vastgelegd, niet als implementatiedetail.

WAAROM DE LOG VAN DE VOLATILITEIT
==================================
De volatiliteitsverdeling is sterk rechtsscheef. Op het NIVEAU zou een enkele
crashbar -- LUNA, mei 2022 -- de expanding standaardafwijking jarenlang
domineren, waarna de HOOG-drempel praktisch onbereikbaar wordt en M0 de rest van
het venster stilzwijgend in NORMAAL blijft hangen. In logs is de verdeling bij
benadering symmetrisch en blijft de z-score interpreteerbaar.

CAUSALITEIT IS GEMETEN, NIET BEWEERD
=====================================
Elke ingang gebruikt uitsluitend bars tot en met `t`: de EWMA-variantie is de
bestaande causale recursie, en het gemiddelde en de standaardafwijking van de
z-score zijn EXPANDING, niet over de volledige sample. Dat laatste is DI-2 en de
verleiding is groot, want `(x - x.mean()) / x.std()` is een regel code.

`tests/lookahead/test_m0_is_causal.py` bewijst het door de reeks op bar `t` af te
KAPPEN en opnieuw te classificeren; de uitkomst op `t` moet identiek zijn. Dat is
een sterker bewijs dan een code-inspectie, want het faalt ook op lekken die je
niet had bedacht.

GEEN OORDEEL IS EEN GELDIGE UITKOMST
=====================================
Vóór `zscore_min_periods` bestaat het regime niet en is de uitkomst
:data:`UNDEFINED_BUCKET` (NaN). Er wordt NIET teruggevallen op NORMAAL. Het
verschil is niet cosmetisch: "normaal" is een BEWERING over de markt, en die
bewering zou dan op de 250 bars staan waar de schatter het minst weet. Een
consument die een regime nodig heeft en NaN krijgt, hoort te crashen.

DE LATENTIECONVENTIE
=====================
Dit module classificeert bar `t` met informatie tot en met de close van `t`. Dat
is legitiem. Een POSITIE die daarop wordt ingenomen, mag pas op `t+2` bestaan
(§0.2, `shift(2)`). Die verschuiving hoort in de conditioneringslaag en staat
bewust NIET hier: wie hem hier zou inbouwen, verschuift ook de diagnostiek en
maakt de regime-occupancy onvergelijkbaar met die van het HMM.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md 10.1, 10.2, 19 (L3); fase-opdracht 0.2,
0.4, stap 8.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd

from ..schemas.config import M0BucketConfig
from ..utils.failfast import DataContractError, require
from ..volatility.ewma import ewma_variance_causal

__all__ = [
    "UNDEFINED_BUCKET",
    "BucketAssignment",
    "VolBucket",
    "causal_atr",
    "causal_vol_zscore",
    "classify_vol_buckets",
]

#: Wat er staat waar M0 geen oordeel heeft. NaN en niet een vierde toestand:
#: een vierde toestand zou meegeteld worden in de occupancy en zou suggereren
#: dat "onbekend" een marktregime is.
UNDEFINED_BUCKET = float("nan")


class VolBucket(IntEnum):
    """De drie toestanden. Ordinaal: LAAG < NORMAAL < HOOG."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


def causal_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, *, window: int,
) -> pd.Series:
    """Average True Range over `window` bars, uitsluitend op het verleden.

    ``TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)``, gemiddeld over
    een rolling venster met ``min_periods == window``. Bar 0 heeft geen vorige
    close en draagt dus geen True Range; de eerste `window` bars blijven NaN.

    Er wordt NIET vooruit gevuld. Een ATR die tijdens de opstartfase een waarde
    toont, is een ATR over minder bars dan hij beweert.
    """
    require(
        window >= 1,
        "Een ATR over minder dan één bar bestaat niet.",
        DataContractError, window=window,
    )
    require(
        len(high) == len(low) == len(close),
        "ATR op OHLC-reeksen van verschillende lengte.",
        DataContractError,
        n_high=len(high), n_low=len(low), n_close=len(close),
    )
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Bar 0 heeft geen vorige close; `max` over de drie kolommen zou daar op
    # `high - low` terugvallen en dus een TR tonen die op andere informatie
    # rust dan elke volgende. Expliciet uitsluiten in plaats van meenemen.
    true_range.iloc[0] = np.nan
    return true_range.rolling(window=window, min_periods=window).mean()


def causal_vol_zscore(
    returns: pd.Series, *, lam: float, burn_in_bars: int, min_periods: int,
) -> pd.Series:
    """Z-score van de LOG causale EWMA-volatiliteit, expanding gestandaardiseerd.

    Op bar `t` gebruiken zowel het gemiddelde als de standaardafwijking
    uitsluitend bars ``0..t``. Dat is het verschil tussen een causale z-score en
    DI-2; ``(x - x.mean()) / x.std()`` op de volledige reeks zou hier de
    volatiliteit van 2026 in de classificatie van 2021 stoppen.
    """
    require(
        min_periods > 1,
        "Een expanding standaardafwijking over minder dan twee observaties is "
        "niet gedefinieerd.",
        DataContractError, min_periods=min_periods,
    )
    variance = ewma_variance_causal(
        returns, lam=lam, burn_in_bars=burn_in_bars)
    log_vol = np.log(np.sqrt(variance))
    mean = log_vol.expanding(min_periods=min_periods).mean()
    std = log_vol.expanding(min_periods=min_periods).std()
    # Een standaardafwijking van nul betekent een constante vol over het hele
    # verleden. De z-score is daar niet gedefinieerd; NaN en geen deling.
    z = (log_vol - mean) / std.where(std > 0.0)
    z.name = "vol_zscore"
    return z


@dataclass(frozen=True)
class BucketAssignment:
    """De classificatie met de diagnostiek die het M0-vs-HMM-rapport vraagt."""

    symbol: str
    #: `VolBucket`-waarden als float, NaN waar M0 geen oordeel heeft.
    buckets: pd.Series
    zscore: pd.Series
    atr_ratio: pd.Series
    n_total: int
    n_defined: int
    #: Bars per toestand. Voedt de occupancy-eis van de Data Adequacy Gate: een
    #: toestand met tientallen bars draagt geen meetbare Sharpe.
    occupancy: Mapping[str, int]
    #: Aantal wisselingen tussen twee OPEENVOLGENDE GEDEFINIEERDE bars. Dit is
    #: de turnover-generator uit §0.4 en hoort naast elke Sharpe-claim.
    n_transitions: int

    @property
    def mean_duration(self) -> float:
        """Gemiddeld aantal bars dat een regime aanhoudt."""
        if self.n_transitions == 0:
            return float(self.n_defined)
        return self.n_defined / (self.n_transitions + 1)

    def as_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_total": self.n_total,
            "n_defined": self.n_defined,
            "coverage": self.n_defined / self.n_total if self.n_total else 0.0,
            "occupancy": dict(self.occupancy),
            "n_transitions": self.n_transitions,
            "mean_duration_bars": self.mean_duration,
        }


def classify_vol_buckets(
    ohlc: pd.DataFrame,
    cfg: M0BucketConfig,
    *,
    ewma_lambda: float,
    ewma_burn_in_bars: int,
    symbol: str = "",
) -> BucketAssignment:
    """Wijs elke bar een vol-bucket toe. Nul schattingen, nul lekrisico.

    De volgorde van de drempeltoetsing doet er niet toe: de HOOG- en
    LAAG-voorwaarden sluiten elkaar per constructie uit, want
    `zscore_low < zscore_high` en `atr_ratio_low < atr_ratio_high` worden door
    :class:`~tradebot.schemas.config.M0BucketConfig` afgedwongen. Zonder die
    validatie zou een omgedraaide config een bar tegelijk HOOG en LAAG maken en
    zou de uitkomst afhangen van de volgorde van twee `if`-takken.
    """
    for column in ("open", "high", "low", "close"):
        require(
            column in ohlc.columns,
            f"M0 vereist een OHLC-frame; kolom '{column}' ontbreekt.",
            DataContractError, symbol=symbol, columns=list(ohlc.columns),
        )
    close = ohlc["close"].astype("float64")
    returns = np.log(close).diff()

    zscore = causal_vol_zscore(
        returns, lam=ewma_lambda, burn_in_bars=ewma_burn_in_bars,
        min_periods=cfg.zscore_min_periods)
    fast = causal_atr(
        ohlc["high"], ohlc["low"], close, window=cfg.atr_fast_window)
    slow = causal_atr(
        ohlc["high"], ohlc["low"], close, window=cfg.atr_slow_window)
    atr_ratio = fast / slow.where(slow > 0.0)
    atr_ratio.name = "atr_ratio"

    defined = zscore.notna() & atr_ratio.notna()
    buckets = pd.Series(
        UNDEFINED_BUCKET, index=ohlc.index, dtype="float64", name="m0_bucket")
    buckets[defined] = float(VolBucket.NORMAL)
    high = defined & (zscore >= cfg.zscore_high) & (
        atr_ratio >= cfg.atr_ratio_high)
    low = defined & (zscore <= cfg.zscore_low) & (
        atr_ratio <= cfg.atr_ratio_low)
    require(
        not bool((high & low).any()),
        "Een bar is tegelijk HOOG en LAAG geclassificeerd. Dat kan alleen bij "
        "omgedraaide drempels, en die hoort de configvalidatie te hebben "
        "geweigerd.",
        DataContractError, symbol=symbol,
    )
    buckets[high] = float(VolBucket.HIGH)
    buckets[low] = float(VolBucket.LOW)

    defined_values = buckets[defined].to_numpy(dtype=np.float64)
    occupancy = {
        bucket.name: int(np.count_nonzero(defined_values == float(bucket)))
        for bucket in VolBucket
    }
    transitions = (
        int(np.count_nonzero(defined_values[1:] != defined_values[:-1]))
        if defined_values.size > 1 else 0
    )
    return BucketAssignment(
        symbol=symbol, buckets=buckets, zscore=zscore, atr_ratio=atr_ratio,
        n_total=int(len(ohlc)), n_defined=int(defined_values.size),
        occupancy=occupancy, n_transitions=transitions,
    )
