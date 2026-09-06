# src/tradebot/validation/inference.py
# LOC-EXCEPTION: R-3 is hier de reden dat R-4 het aflegt. Het meetcontract eist ÉÉN plek voor de
# Sharpe-SE (§3), het Sharpe-verschil (§4), de clustering/N_eff (§5) en de serialisatiepoort (§10);
# die vier delen één HAC-kern, één bloklengte en één bootstrapgenerator, en ze over drie bestanden
# verdelen zou precies de tweede implementatie maken die R-3 een defect noemt. Ruim de helft van de
# regels is de afleiding en de valkuil bij elke formule — dat is de duurste vorm van kennis om
# kwijt te raken. Cap gelijk aan de gemeten omvang bij oplevering (fase 10, stap 4A): dit bestand
# mag niet groeien.
"""De inferentiekern van fase 10 — één standaardfout, één verschiltoets, één bootstrap.

WAT HIER WOONT EN WAAROM PRECIES HIER
=====================================
`docs/MEASUREMENT_CONTRACT.md` §3, §4, §5 en §10 schrijven vier dingen voor die
tot fase 10 nergens één plek hadden:

    §3   de Lo (2002)-standaardfout van een Sharpe, met een HAC-opslag over
         `q = floor(4 (T/100)^(2/9))` Newey-West-lags;
    §4   de Ledoit-Wolf (2008)-toets op het VERSCHIL tussen twee Sharpes, met
         een gestudentiseerde CIRCULAIRE blokbootstrap (Politis-Romano);
    §5   clustering op datum en de N_eff-deflatie van een gepoolde t;
    §10  de weigering om een Sharpe te serialiseren zonder zijn drietal
         `(n_obs, bars_per_year, t_years)`.

R-3 zegt: één implementatie per statistische grootheid; een tweede is een
defect, ook wanneer zij hetzelfde uitrekent. Dit bestand IS die ene plek.

WAAROM EEN VIJFDE BOOTSTRAP -- HET ARGUMENT, NIET DE AANNAME
============================================================
De repository had bij aanvang van deze stap vijf resampling-routines. Zij zijn
geteld en niet geschat:

  1. `backtest/evaluation.py::stationary_block_bootstrap_indices`  STATIONAIR
  2. `backtest/spa.py::_stationary_bootstrap_indices`              STATIONAIR
  3. `validation/sharpe_difference.py::stationary_bootstrap_indices` STATIONAIR
  4. `backtest/metrics.py::bootstrap_ci`                  niet-overlappende blokken
  5. `cv/bootstrap.py::get_sequential_bootstrap_indices`  SEQUENTIEEL (LdP §4.5)

Een zesde toevoegen is geen vanzelfsprekendheid en het argument hoort hier te
staan in plaats van in iemands hoofd:

  * (1) tot (3) zijn STATIONAIR (Politis-Romano 1994): blokken van GEOMETRISCH
    verdeelde lengte. §4 van het meetcontract schrijft een CIRCULAIRE
    blokbootstrap voor: blokken van VASTE lengte, circulair doorlopend, elke
    observatie met gelijke kans. Dat is een andere resampling-verdeling, niet
    dezelfde met een andere naam. De gestudentiseerde variant waarop
    Ledoit-Wolf (2008) steunt is voor die vaste bloklengte afgeleid; de
    stationaire variant heeft een extra bron van variabiliteit — de bloklengte
    zelf — die in de studentisering niet zit.
  * (1) tot (3) dragen bovendien een AFGESLOTEN meting: `sharpe_difference.py`
    is de bevroren H2-methode, `spa.py` is Hansen's SPA en `evaluation.py`
    voedt de Optuna-objective. Ze herschrijven naar circulair zou de methode van
    een gepubliceerd resultaat achteraf veranderen.
  * (4) is in deze stap VERWIJDERD: nul aanroepplekken, vier numerieke literals
    in zijn handtekening, en een derde resampling-schema dat niemand had
    gekozen. Zijn functie — een blokbootstrap-interval rond één Sharpe — woont
    nu hieronder in `block_bootstrap_ci`, met dezelfde circulaire kern als de
    verschiltoets.
  * (5) is GEEN blokbootstrap over een tijdreeks maar de sequentiële bootstrap
    van Lopez de Prado (2018) §4.5: hij trekt SAMPLES op grond van hun
    overlap-uniciteit, voor CV-weging. Andere grootheid, andere invoer, geen
    consolidatiekandidaat.

Daarom: één nieuwe kern, expliciet circulair, en de bestaande blijven staan waar
zij een afgesloten meting of een andere grootheid dragen. De KALIBRATIE van de
bloklengte wordt WEL hergebruikt (`backtest/spa.py::_autocorr_block_size`) —
dat is dezelfde statistische grootheid en die mag niet twee keer bestaan.

WAT HIER NIET WOONT
===================
  * `alpha/factor_alpha.py` draagt een HAC/Newey-West REGRESSIE. Dat is de
    residuele-alfatoets van §8 van het meetcontract, een andere statistiek dan
    de standaardfout van een Sharpe. Niet consolideren.
  * `ledoit_wolf_shrunk_cov` en verwanten in `portfolio/`, `execution/` en
    `train/` zijn Ledoit-Wolf COVARIANTIE-KRIMP (2003/2004) — een ander artikel
    dan de Ledoit-Wolf (2008)-Sharpe-verschiltoets hier. De naam is dezelfde en
    de statistiek is dat niet.

Ref: Lo (2002); Ledoit & Wolf (2008); Newey & West (1987); Politis & Romano
(1992, circulaire blokbootstrap); MEASUREMENT_CONTRACT.md §1, §3, §4, §5, §10.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from ..schemas.config import inference_config
from ..utils.failfast import DataContractError, require

__all__ = [
    "BootstrapInterval",
    "ClusteredMean",
    "SharpeDifference",
    "SharpeEstimate",
    "block_bootstrap_ci",
    "calibrate_block_length",
    "circular_block_indices",
    "clustered_mean",
    "effective_breadth",
    "hac_variance_ratio",
    "neff_deflation",
    "newey_west_lags",
    "require_sharpe_triple",
    "sharpe_difference",
    "sharpe_difference_test",
    "sharpe_with_se",
]

#: Drempels en keuzes uit `conf/validation/inference.yaml`. Geen enkele daarvan
#: staat in een handtekening: `src/tradebot/validation/` heeft ratchet-budget 0.
_CFG = inference_config()

#: De drie velden die een Sharpe onlosmakelijk meedraagt (meetcontract §10).
_TRIPLE = ("n_obs", "bars_per_year", "t_years")

#: Sentinel voor "niet meegegeven". Een object en geen getal, zodat een
#: ontbrekend argument niet per ongeluk als waarde kan doorrekenen.
_UNSET = object()

AlignMode = Literal["common_valid", "common_active"]


# =========================================================================== #
# §10 — de serialisatieweigering
# =========================================================================== #
def require_sharpe_triple(record: Mapping[str, Any], *, where: str) -> None:
    """Weiger een record met een Sharpe erin zonder `(n_obs, bars_per_year, t_years)`.

    MEASUREMENT_CONTRACT.md §10, letterlijk: *"Een Sharpe zonder dat drietal is
    geen getal maar een gerucht, en de serializer MOET hem weigeren."* §10 stelt
    ook vast dat die weigering tot deze stap niet bestond. Dit is zij.

    De poort is bewust NAAM-gebaseerd: elke sleutel waarin `sharpe` voorkomt,
    trekt de eis aan. Dat vangt `net_sharpe`, `delta_sharpe` en `sharpe_a` in één
    regel, en het kan niet worden omzeild door het getal anders te noemen zonder
    dat de sleutel ophoudt een Sharpe te heten.
    """
    keys = [k for k in record if "sharpe" in str(k).lower()]
    if not keys:
        return
    missing = [f for f in _TRIPLE if record.get(f) is None]
    require(
        not missing,
        f"{where}: een Sharpe ({', '.join(sorted(keys))}) mag niet worden "
        f"geserialiseerd zonder {list(_TRIPLE)}; ontbreekt: {missing}. Zonder het "
        f"drietal is niet te bepalen tegen welk venster de waarde is gemeten, en "
        f"dus ook niet welke t-drempel erbij hoort. Zie "
        f"docs/MEASUREMENT_CONTRACT.md §10.",
        DataContractError,
        missing=missing,
    )
    n_obs = float(record["n_obs"])
    bars_per_year = float(record["bars_per_year"])
    t_years = float(record["t_years"])
    require(
        bars_per_year > 0.0 and n_obs > 0.0,
        f"{where}: n_obs={n_obs} en bars_per_year={bars_per_year} moeten positief "
        f"zijn; een venster van nul bars draagt geen Sharpe.",
        DataContractError,
    )
    expected = n_obs / bars_per_year
    require(
        abs(t_years - expected) <= 1e-9 * max(expected, 1.0),
        f"{where}: t_years={t_years} is niet n_obs/bars_per_year={expected}. Twee "
        f"vensterdefinities in één record is precies de slordigheid die "
        f"MEASUREMENT_CONTRACT.md §2 sluit (Q10).",
        DataContractError,
    )


# =========================================================================== #
# §3 — Newey-West bandbreedte en HAC-opslag
# =========================================================================== #
def newey_west_lags(n_obs: int) -> int:
    """De automatische bandbreedte van Newey & West (1987): ``4 (n/100)^(2/9)``.

    Verplaatst uit `validation/vol_metrics.py`, die hem nu hiervandaan
    importeert. Twee identieke bandbreedteregels naast elkaar zijn een
    R-3-defect, ook wanneer zij hetzelfde getal geven.

    Voor `T = 1743` (`W_FULL`) en voor `T = 1615` levert dit q = 7.
    """
    require(
        n_obs > 0,
        "Een Newey-West bandbreedte over nul observaties bestaat niet.",
        DataContractError,
        n_obs=int(n_obs),
    )
    return max(1, int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def hac_variance_ratio(series: np.ndarray, lags: int) -> float:
    """``eta_q`` — de Bartlett-gewogen lange-termijnvariantie gedeeld door gamma_0.

    Dit is de factor waarmee §3 van het meetcontract de Lo-standaardfout
    vermenigvuldigt. Hij is groter dan 1 bij positieve autocorrelatie (een
    autogecorreleerde reeks draagt minder informatie per bar) en kleiner dan 1
    bij negatieve. Beide richtingen zijn juist; een correctie die alleen omhoog
    kan, is geen correctie maar een opslag.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if lags <= 0 or n < 2:
        return 1.0
    d = x - x.mean()
    gamma_0 = float(np.dot(d, d)) / n
    if gamma_0 <= 0.0:
        return 1.0
    total = gamma_0
    for k in range(1, int(lags) + 1):
        if k >= n:
            break
        gamma_k = float(np.dot(d[:-k], d[k:])) / n
        total += 2.0 * (1.0 - k / (float(lags) + 1.0)) * gamma_k
    ratio = total / gamma_0
    # De Bartlett-kern is positief semidefiniet, dus `ratio >= 0`. Exact nul kan
    # alleen bij een gedegenereerde reeks; dan is de HAC-opslag niet
    # identificeerbaar en is 1,0 de eerlijke keuze (geen correctie), niet 0
    # (een standaardfout van nul).
    return float(ratio) if ratio > 0.0 else 1.0


# =========================================================================== #
# §3 — de Sharpe met zijn standaardfout
# =========================================================================== #
@dataclass(frozen=True)
class SharpeEstimate:
    """Een Sharpe met alles wat hem beoordeelbaar maakt (§3 + §10)."""

    #: De GEANNUALISEERDE Sharpe.
    sharpe: float
    #: De GEANNUALISEERDE Lo(2002)-standaardfout met HAC-opslag.
    se: float
    t_stat: float
    n_obs: int
    bars_per_year: float
    t_years: float
    skew: float
    kurtosis: float
    #: Het aantal Newey-West-lags dat is gebruikt; 0 = HAC uitgeschakeld.
    nw_lags: int
    #: `eta_q`. Boven 1 = autocorrelatie vergroot de onzekerheid.
    hac_inflation: float
    #: De naïeve `1/sqrt(T)`-variant, GEANNUALISEERD. Referentiekolom, nooit de
    #: toets (§3).
    se_naive: float
    #: De niet-geannualiseerde Sharpe. De DSR is hierop gedefinieerd.
    sharpe_per_bar: float

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "sharpe": self.sharpe,
            "sharpe_per_bar": self.sharpe_per_bar,
            "se": self.se,
            "se_naive": self.se_naive,
            "t_stat": self.t_stat,
            "n_obs": int(self.n_obs),
            "bars_per_year": self.bars_per_year,
            "t_years": self.t_years,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "nw_lags": int(self.nw_lags),
            "hac_inflation": self.hac_inflation,
        }
        require_sharpe_triple(record, where="SharpeEstimate.to_dict")
        return record


def _finite(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def sharpe_with_se(
    returns: pd.Series | np.ndarray,
    *,
    bars_per_year: float,
    nw_lags: int | None = None,
) -> SharpeEstimate:
    """De Sharpe met zijn Lo (2002)-standaardfout — MEASUREMENT_CONTRACT.md §3.

    ``SE(SR) = sqrt( (1 + SR^2/2 - g3 SR + ((g4-3)/4) SR^2) / T ) * sqrt(eta_q)``

    Drie effecten, drie richtingen, en zij heffen elkaar NIET op: scheefheid
    naar links (`g3 < 0`) vergroot de onzekerheid, dikke staarten (`g4 > 3`)
    vergroten haar, en positieve autocorrelatie vergroot haar via `eta_q`.
    `1/sqrt(T)` veronderstelt alle drie afwezig.

    Parameters
    ----------
    returns
        Bar-rendementen. NaN's worden verwijderd; `n_obs` telt wat overblijft.
    bars_per_year
        VERPLICHT en zonder default. Tot fase 10 stap 4A stond hier in
        `backtest/metrics.py` de waarde 8760, en elke aanroeper die de kwarg
        wegliet, blies zijn Sharpe met sqrt(24) op. Een verkeerde default is wat
        dat veroorzaakte; een tweede default zou het herhalen.
    nw_lags
        `None` = de automatische keuze `floor(4 (T/100)^(2/9))`. `0` schakelt de
        HAC-opslag uit en bestaat UITSLUITEND voor de referentiekolom van §3.
    """
    r = _finite(returns)
    n = int(r.size)
    require(
        n >= _CFG.min_obs,
        f"Een Sharpe-standaardfout vereist minimaal {_CFG.min_obs} eindige "
        f"observaties, kreeg {n}. Onder die grens steunt de Lo-benadering "
        f"nergens op en levert zij een getal dat precisie SUGGEREERT.",
        DataContractError,
        n_obs=n,
    )
    require(
        float(bars_per_year) > 0.0,
        "bars_per_year moet positief zijn; annualiseren met nul bestaat niet.",
        DataContractError,
        bars_per_year=float(bars_per_year),
    )
    sd = float(np.std(r, ddof=1))
    require(
        sd > 0.0,
        "Nul-variantie in de rendementsreeks; de Sharpe is niet gedefinieerd. Dit "
        "is een datadefect, geen resultaat van nul.",
        DataContractError,
        n_obs=n,
    )
    sr_bar = float(np.mean(r)) / sd
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))

    lags = newey_west_lags(n) if nw_lags is None else int(nw_lags)
    require(
        lags >= 0,
        "Een negatief aantal Newey-West-lags bestaat niet.",
        DataContractError,
        nw_lags=lags,
    )
    eta = hac_variance_ratio(r, lags)

    numerator = (
        1.0 + 0.5 * sr_bar**2 - g3 * sr_bar + ((g4 - 3.0) / 4.0) * sr_bar**2
    )
    require(
        numerator > 0.0,
        f"De Lo-variantie is niet positief (teller={numerator:.6g}, SR={sr_bar:.6g}, "
        f"skew={g3:.4g}, kurtosis={g4:.4g}). Dat gebeurt bij een extreem scheve "
        f"reeks met een hoge Sharpe; de asymptotiek van Lo geldt daar niet en een "
        f"standaardfout die er tóch uitrolt, is een verzinsel.",
        DataContractError,
        n_obs=n,
    )
    se_bar = math.sqrt(numerator / n * eta)
    scale = math.sqrt(float(bars_per_year))
    return SharpeEstimate(
        sharpe=sr_bar * scale,
        se=se_bar * scale,
        t_stat=sr_bar / se_bar,
        n_obs=n,
        bars_per_year=float(bars_per_year),
        t_years=n / float(bars_per_year),
        skew=g3,
        kurtosis=g4,
        nw_lags=lags,
        hac_inflation=eta,
        se_naive=math.sqrt(1.0 / n) * scale,
        sharpe_per_bar=sr_bar,
    )


# =========================================================================== #
# §4 — de circulaire blokbootstrap (Politis-Romano 1992)
# =========================================================================== #
def calibrate_block_length(series: np.ndarray) -> int:
    """De bloklengte uit het autocorrelatieverval van de reeks zelf.

    HERGEBRUIKT `backtest/spa.py::_autocorr_block_size` in plaats van hem na te
    bouwen: de bloklengte is één statistische grootheid en R-3 laat er één
    implementatie van bestaan. Wat hier NIEUW is, is de manier waarop de blokken
    worden getrokken (circulair in plaats van stationair), niet hoe lang ze zijn.

    De import staat in de functie om `validation/` -> `backtest/` op importtijd
    te vermijden; de bereikbaarheidsscanner ziet hem wel (hij loopt de hele AST).
    """
    from ..backtest.spa import _autocorr_block_size

    return int(_autocorr_block_size(np.asarray(series, dtype=np.float64)))


def circular_block_indices(
    n_obs: int, block_length: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """`(n_boot, n_obs)` indexmatrix van een CIRCULAIRE blokbootstrap.

    Blokken van VASTE lengte, startpunten uniform over `[0, n)`, circulair
    doorlopend zodat elke observatie exact dezelfde kans heeft om in een
    replicatie te belanden. Dat laatste is het verschil met een niet-circulaire
    blokbootstrap, die de randen van de reeks onderbemonstert -- en de randen
    zijn hier het begin en het einde van het meetvenster.
    """
    require(
        n_obs > 0 and n_boot > 0,
        "Een bootstrap over nul observaties of nul replicaties bestaat niet.",
        DataContractError,
        n_obs=int(n_obs), n_boot=int(n_boot),
    )
    length = max(1, min(int(block_length), int(n_obs)))
    n_blocks = int(math.ceil(n_obs / length))
    starts = rng.integers(0, n_obs, size=(int(n_boot), n_blocks))
    offsets = np.arange(length, dtype=np.int64)
    idx = (starts[:, :, None] + offsets[None, None, :]) % int(n_obs)
    return idx.reshape(int(n_boot), -1)[:, : int(n_obs)]


def _resolve(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


# =========================================================================== #
# §4 — Ledoit-Wolf (2008): het verschil tussen twee Sharpes
# =========================================================================== #
@dataclass(frozen=True)
class SharpeDifference:
    """Het gepaarde Sharpe-verschil met zijn onzekerheid (§4 + §10).

    De veldnamen zijn die van het meetcontract; `delta` en `n_paired_obs` zijn
    LEESVENSTERS op dezelfde opgeslagen waarden en geen tweede opslag. Twee
    namen voor één getal is verwarrend; twee getallen achter twee namen is een
    defect, en dat laatste is hier per constructie uitgesloten.
    """

    #: `SR(a) - SR(b)`, GEANNUALISEERD.
    delta_sharpe: float
    #: De Ledoit-Wolf HAC-standaardfout van dat verschil, GEANNUALISEERD.
    se: float
    t_stat: float
    #: Tweezijdig, uit de gestudentiseerde circulaire blokbootstrap.
    p_value: float
    ci_low: float
    ci_high: float
    ci_95_block_bootstrap: tuple[float, float]
    #: Het aantal bars waarop BEIDE ketens actief zijn. Zie `align`.
    n_effective: int
    sharpe_a: float
    sharpe_b: float
    bars_per_year: float
    t_years: float
    nw_lags: int
    block_length: int
    n_boot: int
    seed: int
    align: str
    #: Aantal bars dat de uitlijning heeft laten vallen. Hoort in de tabel:
    #: `long_only_equal_weight` verliest er 1559 van 1743 (§4 van het contract).
    n_dropped: int
    ci_level: float

    @property
    def delta(self) -> float:
        """Leesvenster op `delta_sharpe` (de naam die stap 4A's tests gebruiken)."""
        return self.delta_sharpe

    @property
    def n_paired_obs(self) -> int:
        """Leesvenster op `n_effective` (de naam die stap 4A's tests gebruiken)."""
        return self.n_effective

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "delta_sharpe": self.delta_sharpe,
            "sharpe_a": self.sharpe_a,
            "sharpe_b": self.sharpe_b,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "n_effective": int(self.n_effective),
            "n_dropped": int(self.n_dropped),
            "n_obs": int(self.n_effective),
            "bars_per_year": self.bars_per_year,
            "t_years": self.t_years,
            "nw_lags": int(self.nw_lags),
            "block_length": int(self.block_length),
            "n_boot": int(self.n_boot),
            "seed": int(self.seed),
            "align": self.align,
        }
        require_sharpe_triple(record, where="SharpeDifference.to_dict")
        return record


def _as_series(values: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype("float64")
    return pd.Series(np.asarray(values, dtype=np.float64))


def _align_pair(
    a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, align: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Snijd twee tracks op hun GEMEENSCHAPPELIJKE ACTIEVE index.

    MEASUREMENT_CONTRACT.md §4: *"Een gepaard verschil over verschillende actieve
    verzamelingen is geen gepaard verschil."* `common_valid` snijdt op eindige
    waarden; `common_active` laat daarnaast de bars vallen waarop een van beide
    exact nul staat -- de haltketen vlakt `long_only_equal_weight` op 1559 van de
    1743 bars af, en een nul-rendement op een gehalteerde bar is geen waarneming
    van die keten maar de afwezigheid ervan.
    """
    require(
        align in ("common_valid", "common_active"),
        f"Onbekende uitlijning {align!r}; toegestaan zijn 'common_valid' en "
        f"'common_active'. Zie MEASUREMENT_CONTRACT.md §4.",
        DataContractError,
        align=align,
    )
    sa, sb = _as_series(a), _as_series(b)
    index = sa.index.intersection(sb.index)
    require(
        len(index) > 0,
        "De twee reeksen hebben geen enkele index gemeen; er valt niets te paren.",
        DataContractError,
        n_a=int(sa.size), n_b=int(sb.size),
    )
    x = sa.reindex(index).to_numpy(dtype=np.float64)
    y = sb.reindex(index).to_numpy(dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if align == "common_active":
        keep &= (x != 0.0) & (y != 0.0)
    return x[keep], y[keep], int(len(index) - int(keep.sum()))


def _lw_delta_and_se(
    x: np.ndarray, y: np.ndarray, lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Ledoit-Wolf (2008) op batches: `(B, n)` in, `(B,)` verschil en SE uit.

    De vier momenten zijn `v_t = (a_t, b_t, a_t^2, b_t^2)`. Het Sharpe-verschil is
    een gladde functie van hun gemiddelden, dus de deltamethode geeft

        Var(dSR) = grad' Psi grad / n

    met `Psi` de HAC-covariantie van `v_t` (Bartlett, `lags` lags). De gradiënt
    is analytisch en niet numeriek: een numerieke gradiënt zou hier een tweede
    bron van ruis toevoegen aan een grootheid die juist de ruis moet meten.

    Eén functie voor de puntschatting (B = 1) én voor elke bootstrap-replicatie:
    een aparte replicatiekern zou stilzwijgend van de puntschatting kunnen
    afwijken, en dan meet de bootstrap iets anders dan hij studentiseert.
    """
    n = x.shape[1]
    v = np.stack([x, y, x * x, y * y], axis=-1)
    mu = v.mean(axis=1)
    d = v - mu[:, None, :]
    psi = np.einsum("bni,bnj->bij", d, d) / n
    for k in range(1, int(lags) + 1):
        if k >= n:
            break
        gamma_k = np.einsum("bni,bnj->bij", d[:, :-k, :], d[:, k:, :]) / n
        weight = 1.0 - k / (float(lags) + 1.0)
        psi = psi + weight * (gamma_k + np.transpose(gamma_k, (0, 2, 1)))

    m1, m2, q1, q2 = mu[:, 0], mu[:, 1], mu[:, 2], mu[:, 3]
    var1, var2 = q1 - m1 * m1, q2 - m2 * m2
    ok = (var1 > 0.0) & (var2 > 0.0)
    safe1 = np.where(ok, var1, 1.0)
    safe2 = np.where(ok, var2, 1.0)
    cube1, cube2 = safe1**1.5, safe2**1.5
    grad = np.stack(
        [q1 / cube1, -q2 / cube2, -m1 / (2.0 * cube1), m2 / (2.0 * cube2)], axis=-1
    )
    variance = np.einsum("bi,bij,bj->b", grad, psi, grad) / n
    delta = np.where(ok, m1 / np.sqrt(safe1) - m2 / np.sqrt(safe2), np.nan)
    se = np.where(ok & (variance > 0.0), np.sqrt(np.maximum(variance, 0.0)), np.nan)
    return delta, se


def sharpe_difference_test(
    a: pd.Series | np.ndarray,
    b: pd.Series | np.ndarray,
    *,
    bars_per_year: float,
    n_boot: int | None = None,
    seed: int | None = None,
    align: AlignMode = "common_valid",
    block_length: int | None = None,
    nw_lags: int | None = None,
) -> SharpeDifference:
    """Ledoit-Wolf (2008): toetst het VERSCHIL tussen twee Sharpes.

    Een Sharpe-verschil is geen gemiddeld rendementsverschil. Een gepaarde
    t-toets op `a - b` toetst of de gemiddelden verschillen; deze toets vraagt of
    de mean-variance-verhoudingen verschillen, en dat is de vraag die stap 10
    stelt.

    De p-waarde en het interval komen uit een GESTUDENTISEERDE circulaire
    blokbootstrap: per replicatie wordt niet alleen het verschil maar ook zijn
    eigen standaardfout herberekend, zodat de verdeling die wordt afgetapt een
    t-verdeling is en geen verschilverdeling. Dat is precies waar Ledoit-Wolf
    (2008) om draait; een percentielbootstrap op het verschil zelf is een andere,
    zwakkere toets.

    Parameters
    ----------
    bars_per_year
        VERPLICHT, zonder default, uit `conf/backtest/default.yaml`.
    n_boot, seed, block_length
        `None` = de geregistreerde waarde uit `conf/validation/inference.yaml`.
        R-5: gelijke cfg + seed geeft bit-identieke output.
    align
        `common_valid` (beide eindig) of `common_active` (beide eindig én geen
        van beide exact nul -- de haltuitlijning van §4 van het meetcontract).
    """
    x, y, n_dropped = _align_pair(a, b, align)
    n = int(x.size)
    require(
        n >= _CFG.min_obs,
        f"De gepaarde Sharpe-verschiltoets vereist minimaal {_CFG.min_obs} "
        f"gemeenschappelijk actieve bars, kreeg {n} (uitlijning {align!r}, "
        f"{n_dropped} bars gevallen). Onder die grens is de Ledoit-Wolf "
        f"-asymptotiek niet te verdedigen.",
        DataContractError,
        n_effective=n, n_dropped=n_dropped, align=align,
    )
    require(
        float(bars_per_year) > 0.0,
        "bars_per_year moet positief zijn; annualiseren met nul bestaat niet.",
        DataContractError,
    )

    boots = int(_resolve(n_boot, _CFG.n_boot))
    rng_seed = int(_resolve(seed, _CFG.seed))
    lags = newey_west_lags(n) if nw_lags is None else int(nw_lags)
    length = int(
        _resolve(block_length, _CFG.block_length)
        if _resolve(block_length, _CFG.block_length) is not None
        else calibrate_block_length(x - y)
    )
    scale = math.sqrt(float(bars_per_year))
    level = float(_CFG.ci_level)

    sr_a = float(np.mean(x) / np.std(x, ddof=1)) * scale if np.std(x, ddof=1) > 0 else 0.0
    sr_b = float(np.mean(y) / np.std(y, ddof=1)) * scale if np.std(y, ddof=1) > 0 else 0.0

    def _build(
        delta: float, se: float, t_stat: float, p_value: float,
        low: float, high: float,
    ) -> SharpeDifference:
        return SharpeDifference(
            delta_sharpe=delta, se=se, t_stat=t_stat, p_value=p_value,
            ci_low=low, ci_high=high, ci_95_block_bootstrap=(low, high),
            n_effective=n, sharpe_a=sr_a, sharpe_b=sr_b,
            bars_per_year=float(bars_per_year), t_years=n / float(bars_per_year),
            nw_lags=lags, block_length=length, n_boot=boots, seed=rng_seed,
            align=str(align), n_dropped=n_dropped, ci_level=level,
        )

    # TWEE IDENTIEKE TRACKS. Het verschil is per constructie exact nul en zijn
    # standaardfout ook: de vier momenten van `a` en `b` zijn dezelfde reeks, dus
    # `grad' Psi grad` is de variantie van een identiek nulle reeks. Elke
    # deling zou hier 0/0 geven. Nul verschil met zekerheid is het juiste
    # antwoord, en dat expliciet opschrijven is beter dan een NaN die door een
    # rapport wandelt.
    if np.array_equal(x, y):
        return _build(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    delta_hat, se_hat = _lw_delta_and_se(x[None, :], y[None, :], lags)
    require(
        bool(np.isfinite(delta_hat[0]) and np.isfinite(se_hat[0]) and se_hat[0] > 0.0),
        "De Ledoit-Wolf-covariantie is gedegenereerd: het verschil of zijn "
        "standaardfout is niet eindig. Dat wijst op een reeks zonder variatie of "
        "op een HAC-schatting die niet positief is; beide zijn een datadefect en "
        "geen resultaat.",
        DataContractError,
        n_effective=n, nw_lags=lags,
    )
    delta_bar = float(delta_hat[0])
    se_bar = float(se_hat[0])
    t_hat = delta_bar / se_bar

    # Gestudentiseerde circulaire blokbootstrap, in batches zodat het geheugen
    # niet lineair in `n_boot` groeit. De batchgrootte komt uit `conf/` omdat zij
    # de trekvolgorde uit de generator bepaalt (R-5).
    rng = np.random.default_rng(rng_seed)
    batch = int(_CFG.bootstrap_batch_size)
    studentised: list[np.ndarray] = []
    drawn = 0
    while drawn < boots:
        size = min(batch, boots - drawn)
        idx = circular_block_indices(n, length, size, rng)
        d_boot, se_boot = _lw_delta_and_se(x[idx], y[idx], lags)
        with np.errstate(divide="ignore", invalid="ignore"):
            studentised.append((d_boot - delta_bar) / se_boot)
        drawn += size
    t_boot = np.concatenate(studentised)
    t_boot = t_boot[np.isfinite(t_boot)]
    require(
        t_boot.size > 0,
        f"Geen enkele van de {boots} bootstrap-replicaties leverde een eindige "
        f"gestudentiseerde statistiek. De toets kan geen p-waarde geven en geeft "
        f"er dus geen.",
        DataContractError,
        n_boot=boots, n_effective=n,
    )

    p_value = float(np.mean(np.abs(t_boot) >= abs(t_hat)))
    alpha = 1.0 - level
    q_hi = float(np.quantile(t_boot, 1.0 - alpha / 2.0))
    q_lo = float(np.quantile(t_boot, alpha / 2.0))
    # Bootstrap-t-interval: de HOGE quantiel van de gestudentiseerde verdeling
    # bepaalt de ONDERGRENS. Dat draait tegenintuïtief om en is de reden dat een
    # percentielinterval hier het verkeerde antwoord geeft bij scheve verdelingen.
    return _build(
        delta_bar * scale, se_bar * scale, t_hat, p_value,
        (delta_bar - q_hi * se_bar) * scale,
        (delta_bar - q_lo * se_bar) * scale,
    )


#: Naam waaronder stap 4A's testbestand de toets aanroept. GEEN tweede
#: implementatie -- letterlijk dezelfde functie onder een tweede naam, omdat twee
#: door het plan vastgelegde testbestanden verschillende namen binden en R-3 twee
#: implementaties verbiedt.
sharpe_difference = sharpe_difference_test


# =========================================================================== #
# §4 — het betrouwbaarheidsinterval van één Sharpe
# =========================================================================== #
@dataclass(frozen=True)
class BootstrapInterval:
    """Een blokbootstrap-interval rond één statistiek."""

    low: float
    point: float
    high: float
    statistic: str
    block_length: int
    n_boot: int
    seed: int
    ci_level: float
    n_obs: int
    bars_per_year: float
    t_years: float

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "statistic": self.statistic,
            "low": self.low,
            "point": self.point,
            "high": self.high,
            "block_length": int(self.block_length),
            "n_boot": int(self.n_boot),
            "seed": int(self.seed),
            "ci_level": self.ci_level,
            "n_obs": int(self.n_obs),
            "bars_per_year": self.bars_per_year,
            "t_years": self.t_years,
        }
        if self.statistic == "sharpe":
            record["sharpe"] = self.point
        require_sharpe_triple(record, where="BootstrapInterval.to_dict")
        return record


def _statistic_on_batch(sample: np.ndarray, statistic: str, scale: float) -> np.ndarray:
    if statistic == "sharpe":
        sd = np.std(sample, axis=1, ddof=1)
        safe = np.where(sd > 0.0, sd, np.nan)
        return sample.mean(axis=1) / safe * scale
    return sample.mean(axis=1)


def block_bootstrap_ci(
    returns: pd.Series | np.ndarray,
    *,
    statistic: str = "sharpe",
    block_length: int | None = None,
    n_boot: int | None = None,
    bars_per_year: float,
    seed: int | None = None,
) -> BootstrapInterval:
    """Percentiel-interval uit dezelfde CIRCULAIRE blokkern als de verschiltoets.

    Vervangt `backtest/metrics.py::bootstrap_ci`, die in fase 10 stap 4A is
    verwijderd: die routine trok NIET-OVERLAPPENDE blokken van vaste lengte —
    een derde resampling-schema naast de stationaire en de circulaire — had nul
    aanroepplekken, en droeg vier numerieke literals in zijn handtekening. Er is
    geen gemeten getal dat onder die routine tot stand is gekomen.

    `statistic` is `"sharpe"` (geannualiseerd) of `"mean"` (per bar).
    """
    require(
        statistic in ("sharpe", "mean"),
        f"Onbekende statistiek {statistic!r}; toegestaan zijn 'sharpe' en 'mean'.",
        DataContractError,
        statistic=statistic,
    )
    r = _finite(returns)
    n = int(r.size)
    require(
        n >= _CFG.min_obs,
        f"Een blokbootstrap-interval vereist minimaal {_CFG.min_obs} eindige "
        f"observaties, kreeg {n}.",
        DataContractError,
        n_obs=n,
    )
    require(
        float(bars_per_year) > 0.0,
        "bars_per_year moet positief zijn; annualiseren met nul bestaat niet.",
        DataContractError,
    )
    boots = int(_resolve(n_boot, _CFG.n_boot))
    rng_seed = int(_resolve(seed, _CFG.seed))
    resolved = _resolve(block_length, _CFG.block_length)
    length = int(resolved) if resolved is not None else calibrate_block_length(r)
    scale = math.sqrt(float(bars_per_year))

    point = float(_statistic_on_batch(r[None, :], statistic, scale)[0])
    rng = np.random.default_rng(rng_seed)
    batch = int(_CFG.bootstrap_batch_size)
    draws: list[np.ndarray] = []
    drawn = 0
    while drawn < boots:
        size = min(batch, boots - drawn)
        idx = circular_block_indices(n, length, size, rng)
        draws.append(_statistic_on_batch(r[idx], statistic, scale))
        drawn += size
    values = np.concatenate(draws)
    values = values[np.isfinite(values)]
    require(
        values.size > 0,
        f"Geen enkele van de {boots} replicaties leverde een eindige statistiek.",
        DataContractError,
        n_boot=boots, n_obs=n,
    )
    alpha = 1.0 - float(_CFG.ci_level)
    return BootstrapInterval(
        low=float(np.quantile(values, alpha / 2.0)),
        point=point,
        high=float(np.quantile(values, 1.0 - alpha / 2.0)),
        statistic=statistic,
        block_length=length,
        n_boot=boots,
        seed=rng_seed,
        ci_level=float(_CFG.ci_level),
        n_obs=n,
        bars_per_year=float(bars_per_year),
        t_years=n / float(bars_per_year),
    )


# =========================================================================== #
# §5 — paneelafhankelijkheid: clustering op datum en N_eff-deflatie
# =========================================================================== #
def effective_breadth(correlation_matrix: np.ndarray) -> float:
    """`N_eff = N / (1 + (N-1) rho_bar)` — de effectieve breedte van een paneel.

    Gemeten op `data/pit_store/`: zes namen met `rho_bar = 0,7442` geven
    `N_eff = 1,271`. Zes sterk gecorreleerde namen zijn geen cross-sectie maar
    één gerichte weddenschap met zes tickers erop.
    """
    c = np.asarray(correlation_matrix, dtype=np.float64)
    require(
        c.ndim == 2 and c.shape[0] == c.shape[1] and c.shape[0] >= 2,
        "N_eff vereist een vierkante correlatiematrix met minstens twee namen.",
        DataContractError,
        shape=tuple(c.shape),
    )
    n_names = int(c.shape[0])
    off_diagonal = c[~np.eye(n_names, dtype=bool)]
    rho_bar = float(np.mean(off_diagonal))
    denominator = 1.0 + (n_names - 1) * rho_bar
    require(
        denominator > 0.0,
        f"N_eff is niet gedefinieerd bij rho_bar={rho_bar:.4f} en N={n_names}: de "
        f"noemer 1 + (N-1) rho_bar is niet positief. Zo'n correlatiestructuur kan "
        f"niet uit een geldige correlatiematrix komen.",
        DataContractError,
        rho_bar=rho_bar, n_names=n_names,
    )
    return float(n_names / denominator)


def neff_deflation(correlation_matrix: np.ndarray) -> float:
    """`sqrt(N_eff / N)` — de factor waarmee een gepoolde t wordt gedefleerd (§5).

    Voor het gemeten paneel (zes namen, `rho_bar = 0,7442`) is dit **0,460**.

    De factor wordt AFGEKAPT op 1,0. Bij een negatief gemiddelde correlatie zou
    `N_eff > N` gelden en zou de "deflatie" een t VERGROTEN. Een correctie die de
    verkeerde kant op kan werken, is erger dan geen correctie: zij ziet eruit als
    zorgvuldigheid. De deflatie corrigeert voor te weinig onafhankelijke
    informatie; zij is geen mechanisme om er meer van te claimen.
    """
    n_names = int(np.asarray(correlation_matrix).shape[0])
    return float(min(math.sqrt(effective_breadth(correlation_matrix) / n_names), 1.0))


@dataclass(frozen=True)
class ClusteredMean:
    """Een gepoold gemiddelde met alle drie de t-lezingen die §5 eist."""

    mean: float
    #: De op de tijdsindex GECLUSTERDE standaardfout.
    se: float
    #: De t bij die geclusterde standaardfout. Dit is de lezing die telt.
    t_stat: float
    n_clusters: int
    n_obs: int
    n_units: int
    #: De naïeve standaardfout die alle symbool-bars als onafhankelijk behandelt.
    se_pooled: float
    #: De naïeve t. Referentiekolom, nooit de toets.
    t_stat_pooled: float
    rho_bar: float
    n_effective: float
    #: `sqrt(N_eff / N)`, afgekapt op 1,0.
    neff_factor: float
    #: `t_pooled * neff_factor` — de derde lezing die §5 naast de andere twee eist.
    t_stat_neff_deflated: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "se": self.se,
            "t_stat": self.t_stat,
            "n_clusters": int(self.n_clusters),
            "n_obs": int(self.n_obs),
            "n_units": int(self.n_units),
            "se_pooled": self.se_pooled,
            "t_stat_pooled": self.t_stat_pooled,
            "rho_bar": self.rho_bar,
            "n_effective": self.n_effective,
            "neff_factor": self.neff_factor,
            "t_stat_neff_deflated": self.t_stat_neff_deflated,
        }


def clustered_mean(
    frame: pd.DataFrame, *, cluster_axis: str = "index"
) -> ClusteredMean:
    """Het gepoolde gemiddelde met een op datum GECLUSTERDE standaardfout (§5).

    Een gepoolde toets over symbool-bars behandelt ~10.400 observaties als
    onafhankelijk, terwijl er bij `rho_bar = 0,7442` effectief ongeveer 1733
    dagen x 1,271 namen aan informatie in zit. De geclusterde standaardfout laat
    de afhankelijkheid BINNEN een datum vrij en telt alleen de variatie TUSSEN
    data als informatie.

    Er komen drie lezingen uit, en §5 eist ze alle drie naast elkaar: de
    gepoolde (naïef), de geclusterde (de toets) en de N_eff-gedefleerde. Een
    gepoolde t zonder zijn gedefleerde tegenhanger is geen bevinding (R-8).
    """
    require(
        cluster_axis in ("index", "columns"),
        f"Onbekende clusteras {cluster_axis!r}; toegestaan zijn 'index' (datum) en "
        f"'columns' (naam). §5 van het meetcontract schrijft 'index' voor.",
        DataContractError,
        cluster_axis=cluster_axis,
    )
    require(
        isinstance(frame, pd.DataFrame) and frame.shape[0] > 0 and frame.shape[1] > 0,
        "clustered_mean verwacht een niet-leeg paneel (rijen = data, kolommen = namen).",
        DataContractError,
    )
    values = frame.to_numpy(dtype=np.float64)
    if cluster_axis == "columns":
        values = values.T
    mask = np.isfinite(values)
    n_obs = int(mask.sum())
    require(
        n_obs >= _CFG.min_obs,
        f"Een geclusterde standaardfout vereist minimaal {_CFG.min_obs} eindige "
        f"observaties, kreeg {n_obs}.",
        DataContractError,
        n_obs=n_obs,
    )
    filled = np.where(mask, values, 0.0)
    grand_mean = float(filled.sum() / n_obs)

    centred = np.where(mask, values - grand_mean, 0.0)
    cluster_sums = centred.sum(axis=1)
    non_empty = mask.any(axis=1)
    n_clusters = int(non_empty.sum())
    require(
        n_clusters >= 2,
        f"Clusteren op {cluster_axis} levert {n_clusters} cluster(s); onder twee is "
        f"er geen variatie TUSSEN clusters en dus geen standaardfout.",
        DataContractError,
        n_clusters=n_clusters,
    )
    # Cluster-robuuste meat-matrix met de gebruikelijke eindige-steekproefcorrectie
    # G/(G-1). Op 1615 dagen is die correctie verwaarloosbaar; op zes folds niet.
    meat = float(np.dot(cluster_sums, cluster_sums))
    correction = n_clusters / (n_clusters - 1.0)
    se = math.sqrt(max(meat * correction, 0.0)) / n_obs

    flat = values[mask]
    sd = float(np.std(flat, ddof=1))
    se_pooled = sd / math.sqrt(n_obs)
    require(
        se > 0.0 and se_pooled > 0.0,
        "De standaardfout van het gepoolde gemiddelde is nul; het paneel heeft geen "
        "variatie en draagt dus geen toets.",
        DataContractError,
        n_obs=n_obs,
    )

    units = frame if cluster_axis == "index" else frame.T
    n_units = int(units.shape[1])
    if n_units >= 2:
        corr = units.corr().to_numpy(dtype=np.float64)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        np.fill_diagonal(corr, 1.0)
        n_eff = effective_breadth(corr)
        factor = neff_deflation(corr)
        off = corr[~np.eye(n_units, dtype=bool)]
        rho_bar = float(np.mean(off))
    else:
        # Eén naam is geen paneel: er is niets om voor te defleren, en een factor
        # van 1,0 zegt dat eerlijk in plaats van een deflatie te suggereren.
        n_eff, factor, rho_bar = 1.0, 1.0, 0.0

    t_pooled = grand_mean / se_pooled
    return ClusteredMean(
        mean=grand_mean,
        se=se,
        t_stat=grand_mean / se,
        n_clusters=n_clusters,
        n_obs=n_obs,
        n_units=n_units,
        se_pooled=se_pooled,
        t_stat_pooled=t_pooled,
        rho_bar=rho_bar,
        n_effective=n_eff,
        neff_factor=factor,
        t_stat_neff_deflated=t_pooled * factor,
    )
