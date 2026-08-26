"""De econometrische diagnose van elke reeks die de vol-pijplijn binnenkomt.

De wetenschap achter `apps/run_econometric_diagnostics.py`. Phase 6 stappen 3
en 4, deliverables 7-9.

WAAROP WORDT GETOETST — EN WAAROM DAT DRIE VERSCHILLENDE REEKSEN ZIJN
======================================================================
Een toets op de verkeerde reeks geeft een keurige p-waarde die niets betekent.
Deze module toetst daarom expliciet drie transformaties per symbool:

``log_price``
    De niveaus. Hier hoort een eenheidswortel te zitten; is die er niet, dan is
    er iets mis met de data. ADF en KPSS zijn hier informatief, ARCH niet — op
    een niet-stationaire reeks vindt de ARCH-toets vrijwel altijd "effecten",
    omdat de gekwadrateerde residuen van een random walk vanzelf autocorreleren.
    Die uitkomst wordt gerapporteerd maar telt NIET als poortoordeel.

``log_return``
    De reeks waar de vol-pijplijn feitelijk op werkt. **Dit is de reeks waarop
    het ARCH-poortoordeel geldt.** Stationair, geen of weinig autocorrelatie in
    de niveaus, en — als de literatuur klopt — sterke autocorrelatie in de
    kwadraten.

``fracdiff``
    De fractioneel gedifferentieerde log-prijs met de minimale `d` die ADF
    p < 0,05 haalt. Het punt van FFD is dat deze reeks stationair is ZONDER het
    geheugen weg te gooien dat een gewone eerste-orde differentie vernietigt.
    Het behouden geheugen wordt gemeten als de correlatie met de oorspronkelijke
    log-prijs (AFML §5.5, figuur 5.5).

WAAROM `d` PER REEKS EN NIET GLOBAAL
=====================================
De optimale `d` is een eigenschap van de reeks, niet van het platform. BTCUSDT
en AVAXUSDT hebben verschillende persistentie; één gedeelde `d` zou de ene
onderdifferentiëren en de andere overdifferentiëren. De zoekruimte staat in
`conf/model/fracdiff.yaml`, het RESULTAAT wordt per reeks geregistreerd.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..features.fracdiff import _ffd_weights, frac_diff_ffd, min_frac_diff
from ..schemas.config import FracDiffConfig
from ..utils.failfast import DataContractError, require
from .econometrics import SeriesDiagnostics, arch_gate_verdict, diagnose_series

__all__ = ["FracDiffOutcome", "SymbolDiagnostics", "diagnose_universe"]


@dataclass(frozen=True)
class FracDiffOutcome:
    """De gekozen `d` en wat hij aan geheugen overliet."""

    d: float
    adf_p_value: float
    converged: bool
    n_used_obs: int
    #: Correlatie tussen de gedifferentieerde reeks en de oorspronkelijke
    #: log-prijs (AFML §5.5, figuur 5.5). LET OP: op dit venster is deze maat
    #: NIET monotoon in `d` en kan hij negatief zijn — zie `memory_curve` en de
    #: bespreking in `reports/ECONOMETRIC_DIAGNOSTICS.md` §4.3. Hij wordt
    #: gerapporteerd als DIAGNOSTIEK, niet als criterium; het criterium van
    #: §8.1 is `d` zelf, want een lagere `d` IS per constructie meer geheugen.
    memory_retained: float
    #: Dezelfde maat voor de gewone eerste-orde differentie, als referentie.
    memory_retained_at_d_one: float
    #: Lengte van het fixed-width gewichtenvenster. Dit is de grootheid die
    #: bepaalt hoeveel bars de FFD-transformatie KOST.
    ffd_window_bars: int
    #: Het gewicht dat door de afkapping verloren gaat. De afruil tegen
    #: `ffd_window_bars`: een kleinere drempel geeft minder truncatiebias en
    #: een langer venster.
    residual_weight_mass: float
    #: Aantal bars dat de transformatie kost. `frac_diff_ffd` LAAT DIE RIJEN
    #: VALLEN in plaats van ze op NaN te zetten, dus dit is een lengteverschil
    #: en geen NaN-telling. (Defect in mijn eigen werk: de eerste versie van
    #: deze meting telde NaN's en rapporteerde daardoor consequent 0.)
    n_bars_consumed: int
    n_obs_after_ffd: int
    #: `corr(FFD_d(x), x)` op een raster van `d`, zodat het rapport kan LATEN
    #: ZIEN dat de maat op dit venster niet monotoon is in plaats van het te
    #: beweren. Zonder deze curve zou één negatief getal in de tabel als een
    #: rekenfout worden gelezen.
    memory_curve: Mapping[str, float] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "adf_p_value": self.adf_p_value,
            "converged": self.converged,
            "n_used_obs": self.n_used_obs,
            "memory_retained": self.memory_retained,
            "memory_retained_at_d_one": self.memory_retained_at_d_one,
            "ffd_window_bars": self.ffd_window_bars,
            "residual_weight_mass": self.residual_weight_mass,
            "n_bars_consumed": self.n_bars_consumed,
            "n_obs_after_ffd": self.n_obs_after_ffd,
            "memory_curve": dict(self.memory_curve),
        }


@dataclass(frozen=True)
class SymbolDiagnostics:
    """Alles wat er over één symbool is gemeten voordat er iets is gefit."""

    symbol: str
    log_price: SeriesDiagnostics
    log_return: SeriesDiagnostics
    fracdiff: SeriesDiagnostics
    frac: FracDiffOutcome

    @property
    def arch_gate_open(self) -> bool:
        """Het poortoordeel geldt op de LOG-RETURNS, niet op de niveaus."""
        return self.log_return.arch_gate_open

    def as_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "arch_gate_open": self.arch_gate_open,
            "arch_gate_verdict": arch_gate_verdict(self.log_return),
            "fracdiff": self.frac.as_record(),
            "log_price": self.log_price.as_record(),
            "log_return": self.log_return.as_record(),
            "fracdiff_series": self.fracdiff.as_record(),
        }


def _memory_retained(original: pd.Series, differenced: pd.Series) -> float:
    """Correlatie tussen de gedifferentieerde reeks en het oorspronkelijke niveau."""
    joined = pd.concat([original, differenced], axis=1).dropna()
    require(
        len(joined) >= 50,
        "Te weinig overlappende observaties om het behouden geheugen te meten.",
        DataContractError,
        n_overlap=int(len(joined)),
    )
    return float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])


def _fracdiff_for(series: pd.Series, cfg: FracDiffConfig) -> tuple[FracDiffOutcome, pd.Series]:
    result = min_frac_diff(
        series, d_lo=cfg.d_lo, d_hi=cfg.d_hi, threshold=cfg.weight_threshold,
        p_target=cfg.adf_p_target, max_iter=cfg.max_iter,
    )
    require(
        result.converged,
        "Fractionele differentiëring bereikte geen stationariteit binnen "
        "[d_lo, d_hi]. Er wordt NIET stilzwijgend teruggevallen op d = 1: een "
        "reeks die zelfs bij d_hi een eenheidswortel houdt, is een bevinding.",
        DataContractError,
        d_hi=cfg.d_hi, adf_p=result.pvalue, n_used_obs=result.n_used_obs,
    )
    differenced = frac_diff_ffd(series, d=result.d,
                                threshold=cfg.weight_threshold)
    # `frac_diff_ffd` is gedefinieerd op het OPEN interval (0, 1); d = 1 IS de
    # gewone eerste-orde differentie en wordt daarom direct genomen in plaats
    # van via een gewichtenreeks die daar niet bestaat.
    first_order = series.diff()
    weights, residual = _ffd_weights(result.d, cfg.weight_threshold,
                                     return_residual=True)
    outcome = FracDiffOutcome(
        d=float(result.d),
        adf_p_value=float(result.pvalue),
        converged=bool(result.converged),
        n_used_obs=int(result.n_used_obs),
        memory_retained=_memory_retained(series, differenced),
        memory_retained_at_d_one=_memory_retained(series, first_order),
        ffd_window_bars=int(len(weights)),
        residual_weight_mass=float(residual),
        n_bars_consumed=int(len(series) - len(differenced)),
        n_obs_after_ffd=int(len(differenced)),
        memory_curve={
            f"{grid_d:.2f}": _memory_retained(
                series, frac_diff_ffd(series, d=grid_d,
                                      threshold=cfg.weight_threshold))
            for grid_d in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
        },
    )
    return outcome, differenced


def diagnose_universe(
    prices: pd.DataFrame,
    cfg: FracDiffConfig,
    *,
    symbols: Sequence[str] | None = None,
    alpha: float = 0.05,
    ljung_box_lags: int = 20,
    arch_lags: int = 12,
) -> Mapping[str, SymbolDiagnostics]:
    """Draai de volledige keten op elk symbool van het universum."""
    out: dict[str, SymbolDiagnostics] = {}
    for symbol in (symbols or list(prices.columns)):
        price = prices[symbol].dropna()
        log_price = np.log(price)
        log_return = log_price.diff().dropna()

        frac, differenced = _fracdiff_for(log_price, cfg)
        kwargs = {"alpha": alpha, "ljung_box_lags": ljung_box_lags,
                  "arch_lags": arch_lags}
        out[symbol] = SymbolDiagnostics(
            symbol=symbol,
            log_price=diagnose_series(
                log_price.to_numpy(), name=f"{symbol}:log_price", **kwargs),
            log_return=diagnose_series(
                log_return.to_numpy(), name=f"{symbol}:log_return", **kwargs),
            fracdiff=diagnose_series(
                differenced.dropna().to_numpy(),
                name=f"{symbol}:fracdiff(d={frac.d:.4f})", **kwargs),
            frac=frac,
        )
    return out
