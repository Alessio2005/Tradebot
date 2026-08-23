# src/tradebot/features/transforms.py
"""L1 - toestandsloze, causale transforms op een cross-sectioneel panel.

Phase 3, deliverable 2.

Elke functie hier is een **pure functie**: dezelfde input levert dezelfde
output, er is geen interne toestand tussen aanroepen, en er wordt niets
onthouden van een eerdere reeks. Dat is geen stijlvoorkeur maar de reden dat
truncatie-invariantie hier een constructie-eigenschap is en geen belofte.

Het panel is breed: rijen zijn `asof_ts` (het BESCHIKBAARHEIDSmoment), kolommen
zijn symbolen. Dat onderscheid doet ertoe voor de cross-sectionele transforms:
een rangschikking over assets op tijdstip `t` mag uitsluitend waarden gebruiken
die op `t` ook echt bekend waren.

DE TWEE SOORTEN, EN WAAROM ZE VERSCHILLEND FALEN
------------------------------------------------
* **Over de tijd** (`rolling_log_return`, `expanding_zscore`,
  `rolling_quantile_winsorise`): causaal zolang elk venster ACHTERWAARTS is en
  de burn-in NaN blijft. De klassieke fout is normaliseren met een
  sample-brede statistiek - zie DI-2 en `features/volatility.py`.
* **Over de assets** (`cross_sectional_rank`): causaal zolang de rangschikking
  PER TIJDSTIP gebeurt. Een rangschikking die over de tijd normaliseert met
  statistieken van de hele reeks is dat niet, ook al oogt hij cross-sectioneel.

Een dunne cross-sectie levert GEEN positie op. `cross_sectional_rank` geeft een
rij met minder dan `min_assets` waarnemingen volledig als NaN terug, in plaats
van een rangschikking over twee namen die als een gediversifieerde weddenschap
wordt gelezen.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 8, 11, 19 (L1), 26.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from ..utils.time import assert_utc_index

__all__ = [
    "TRANSFORMS",
    "cross_sectional_rank",
    "expanding_zscore",
    "resolve_transform",
    "rolling_log_return",
    "rolling_quantile_winsorise",
]


def _validate_panel(panel: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Gemeenschappelijke ingangscontrole voor elke transform."""
    require(
        isinstance(panel, pd.DataFrame),
        "Een transform werkt op een breed panel (rijen = asof_ts, kolommen = "
        "symbolen), niet op een losse reeks.",
        DataContractError,
        transform=name,
        got=type(panel).__name__,
    )
    assert_utc_index(panel, name=f"transform({name})")
    require(
        bool(panel.index.is_unique),
        "Dubbele tijdstempels in het panel.",
        DataContractError,
        transform=name,
    )
    require(
        len(panel.columns) > 0,
        "Panel zonder kolommen.",
        DataContractError,
        transform=name,
    )
    return panel.astype("float64")


def rolling_log_return(
    panel: pd.DataFrame, *, window: int, skip_bars: int
) -> pd.DataFrame:
    """Cumulatief log-rendement over `window` bars, eindigend `skip_bars` voor t.

        r_t = ln(P_{t-skip}) - ln(P_{t-skip-window})

    De eerste `window + skip_bars` rijen zijn NaN en blijven dat.
    """
    p = _validate_panel(panel, name="rolling_log_return")
    require(
        window > 0,
        "Een rendement over een niet-positief venster bestaat niet.",
        DataContractError,
        window=window,
    )
    require(
        skip_bars >= 0,
        "Een NEGATIEVE skip verschuift het venster naar de TOEKOMST; dat is geen "
        "parameterkeuze maar een lookahead-lek.",
        DataContractError,
        skip_bars=skip_bars,
    )
    require(
        bool((p.to_numpy()[np.isfinite(p.to_numpy())] > 0.0).all()),
        "Niet-positieve prijs in het panel; ln(P) is dan niet gedefinieerd.",
        DataContractError,
    )
    log_price = pd.DataFrame(
        np.log(p.to_numpy(dtype="float64")), index=p.index, columns=p.columns
    )
    return log_price.shift(skip_bars) - log_price.shift(skip_bars + window)


def cross_sectional_rank(panel: pd.DataFrame, *, min_assets: int) -> pd.DataFrame:
    """Rank-gedemeaned score per tijdstip, geschaald naar [-1, +1].

    Per rij worden de niet-NaN waarden gerangschikt; de laagste krijgt -1, de
    hoogste +1, en de rest ligt er lineair tussenin. Het resultaat is per rij
    som-nul zolang de bezetting symmetrisch is, en het is per constructie
    schaalvrij: het hangt uitsluitend af van de ORDENING, niet van de niveaus.

    Causaal omdat de bewerking strikt binnen een tijdstip blijft. Er is geen
    enkele verwijzing naar een andere rij, dus truncatie kan de uitkomst niet
    raken.

    Rijen met minder dan `min_assets` waarnemingen zijn volledig NaN. Een
    rangschikking over twee namen is geen cross-sectie maar een muntworp, en die
    hoort als "geen positie" te worden gelezen en niet als een signaal.
    """
    p = _validate_panel(panel, name="cross_sectional_rank")
    require(
        min_assets > 1,
        "Een cross-sectionele rangschikking over minder dan twee assets is "
        "betekenisloos.",
        DataContractError,
        min_assets=min_assets,
    )
    counts = p.notna().sum(axis=1)
    ranks = p.rank(axis=1, method="average", na_option="keep")
    n = counts.to_numpy(dtype="float64")
    # rank in {1..n} -> [-1, +1]. Bij n == 1 is de schaal niet gedefinieerd, maar
    # die rijen vallen sowieso weg op de min_assets-drempel.
    denom = np.where(n > 1.0, n - 1.0, np.nan)
    scaled = (ranks.to_numpy(dtype="float64") - 1.0) / denom[:, None] * 2.0 - 1.0
    out = pd.DataFrame(scaled, index=p.index, columns=p.columns)
    enough = counts >= min_assets
    return out.where(enough, other=np.nan, axis=0)


def expanding_zscore(panel: pd.DataFrame, *, min_periods: int) -> pd.DataFrame:
    """EXPANDING z-score per kolom: uitsluitend informatie tot en met t.

    Bewust geen rolling-met-sample-statistiek en al helemaal geen
    `(x - x.mean()) / x.std()` over de volledige reeks: die laatste is per
    constructie rond nul gecentreerd over precies het venster dat wordt
    getoetst, en dat is dezelfde klasse fout als DI-2.
    """
    p = _validate_panel(panel, name="expanding_zscore")
    require(
        min_periods > 1,
        "Een z-score vereist een standaarddeviatie, en die is niet gedefinieerd "
        "over minder dan twee observaties.",
        DataContractError,
        min_periods=min_periods,
    )
    exp = p.expanding(min_periods=min_periods)
    mean, std = exp.mean(), exp.std()
    released = std.notna()
    require(
        bool((std.to_numpy()[released.to_numpy()] > 0.0).all()),
        "De expanding standaarddeviatie is exact nul. Een volstrekt constante "
        "reeks over het hele venster is geen marktuitkomst maar een databreuk; "
        "delen zou plus/min oneindig opleveren en dat propageert stilzwijgend "
        "door elk later gewicht.",
        DataContractError,
    )
    return (p - mean) / std


def rolling_quantile_winsorise(
    panel: pd.DataFrame, *, window: int, quantile: float
) -> pd.DataFrame:
    """Winsoriseer op ROLLING quantielen, nooit op de volledige sample.

    Elke waarde wordt geknipt op het `quantile`- en het `1 - quantile`-punt van
    zijn eigen achterwaartse venster. Winsoriseren op sample-brede quantielen
    zou betekenen dat de grens waarop bar 10 wordt geknipt mede wordt bepaald
    door een crash in jaar vijf.

    Tijdens de burn-in (`min_periods = window`) is er geen grens en blijft de
    waarde NaN - er wordt niet stilzwijgend ongeknipt doorgelaten.
    """
    p = _validate_panel(panel, name="rolling_quantile_winsorise")
    require(
        window > 1,
        "Winsorisatie over een venster van een bar knipt elke waarde op "
        "zichzelf en doet dus niets.",
        DataContractError,
        window=window,
    )
    require(
        0.0 < quantile < 0.5,
        "Het winsorisatie-quantiel moet strikt tussen 0 en 0.5 liggen.",
        DataContractError,
        quantile=quantile,
    )
    roll = p.rolling(window, min_periods=window)
    lower = roll.quantile(quantile)
    upper = roll.quantile(1.0 - quantile)
    clipped = pd.DataFrame(
        np.clip(
            p.to_numpy(dtype="float64"),
            lower.to_numpy(dtype="float64"),
            upper.to_numpy(dtype="float64"),
        ),
        index=p.index,
        columns=p.columns,
    )
    # Waar de grens nog niet bestaat, bestaat de geknipte waarde ook niet.
    out: pd.DataFrame = clipped.where(lower.notna() & upper.notna(), other=np.nan)
    return out


#: Naam -> transform. De pipeline verwijst er via de naam naar, zodat een
#: transformketen volledig uit `conf/` kan worden beschreven en gehasht.
TRANSFORMS: Mapping[str, Callable[..., pd.DataFrame]] = {
    "rolling_log_return": rolling_log_return,
    "cross_sectional_rank": cross_sectional_rank,
    "expanding_zscore": expanding_zscore,
    "rolling_quantile_winsorise": rolling_quantile_winsorise,
}


def resolve_transform(name: str) -> Callable[..., pd.DataFrame]:
    """Zoek een transform op naam. Een onbekende naam crasht."""
    require(
        name in TRANSFORMS,
        "Onbekende transform. Een typo in een transformketen mag nooit "
        "stilzwijgend betekenen dat er een stap wordt overgeslagen.",
        DataContractError,
        name=name,
        available=sorted(TRANSFORMS),
    )
    return TRANSFORMS[name]


def burn_in_of(name: str, params: Mapping[str, Any]) -> int:
    """De burn-in die een transform per constructie introduceert, in bars.

    Expliciet in plaats van gemeten: de pipeline moet vooraf kunnen zeggen
    vanaf welke bar zijn output bruikbaar is, zonder eerst te draaien.
    """
    if name == "rolling_log_return":
        return int(params["window"]) + int(params["skip_bars"])
    if name == "expanding_zscore":
        return int(params["min_periods"]) - 1
    if name == "rolling_quantile_winsorise":
        return int(params["window"]) - 1
    if name == "cross_sectional_rank":
        return 0
    require(
        False,
        "Onbekende transform in burn_in_of.",
        DataContractError,
        name=name,
        available=sorted(TRANSFORMS),
    )
    raise AssertionError("unreachable")  # pragma: no cover
