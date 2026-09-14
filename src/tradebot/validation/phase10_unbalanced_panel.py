# src/tradebot/validation/phase10_unbalanced_panel.py
"""H-10.2 — wat een onevenwichtig paneel aan MEETBAARHEID oplevert, en wat niet.

Deze module rekent één ding uit: hoeveel bars een breedtepoort toevoegt en welke
t = 2-drempel daaruit volgt. Dat laatste getal is het resultaat van stap 12.

WAAROM DE BREEDTEVERDELING HIER NAAST DE DREMPEL STAAT, en niet in het rapport.
`2/sqrt(T)` veronderstelt bars van gelijke informatie-inhoud. Een breedtepoort
voegt bars toe uit de periode waarin het universum nog niet vol was, dus bars met
een SMALLERE cross-sectie. De drempel daalt dan rekenkundig harder dan de
bewijskracht stijgt. Wie alleen de drempel rapporteert, rapporteert de gunstige
helft van de meting; daarom geeft `PanelGrowth` de breedteverdeling mee en kan een
aanroeper de twee niet uit elkaar halen.

WAT HIER NIET WORDT GEDAAN. Er wordt geen Sharpe vergeleken tussen de twee
vensters. Dat is opzettelijk: zij beslaan verschillende marktperiodes, en een
verschil is dan een uitspraak over WELKE bars zijn toegevoegd in plaats van over
het systeem. De stapopdracht noemt die valkuil zelf. Een aanroeper die beide
Sharpes wil rapporteren, moet ze met hun venster erbij rapporteren (de
Sharpe-triple van MEASUREMENT_CONTRACT §10) en niet als een verbetering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = ["PanelGrowth", "measure_panel_growth"]


@dataclass(frozen=True)
class PanelGrowth:
    """De groei van één venster onder één breedte-eis, met haar prijs."""

    min_symbols_per_bar: int
    reference_min_symbols_per_bar: int
    n_bars: int
    n_bars_reference: int
    n_bars_added: int
    growth_fraction: float
    t_years: float
    t_years_reference: float
    t2_hurdle: float
    t2_hurdle_reference: float
    t2_hurdle_delta: float
    se_sharpe: float
    se_sharpe_reference: float
    first_bar: str
    first_bar_reference: str
    #: Aantal bars per cross-sectionele breedte in het NIEUWE venster.
    breadth_histogram: dict[int, int]
    #: Idem, maar uitsluitend over de bars die deze poort TOEVOEGT.
    breadth_histogram_added: dict[int, int]
    mean_breadth: float
    mean_breadth_reference: float
    mean_breadth_added: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_symbols_per_bar": self.min_symbols_per_bar,
            "reference_min_symbols_per_bar": self.reference_min_symbols_per_bar,
            "n_bars": self.n_bars,
            "n_bars_reference": self.n_bars_reference,
            "n_bars_added": self.n_bars_added,
            "growth_fraction": self.growth_fraction,
            "t_years": self.t_years,
            "t_years_reference": self.t_years_reference,
            "t2_hurdle": self.t2_hurdle,
            "t2_hurdle_reference": self.t2_hurdle_reference,
            "t2_hurdle_delta": self.t2_hurdle_delta,
            "se_sharpe": self.se_sharpe,
            "se_sharpe_reference": self.se_sharpe_reference,
            "first_bar": self.first_bar,
            "first_bar_reference": self.first_bar_reference,
            "breadth_histogram": {str(k): v for k, v in sorted(self.breadth_histogram.items())},
            "breadth_histogram_added": {
                str(k): v for k, v in sorted(self.breadth_histogram_added.items())
            },
            "mean_breadth": self.mean_breadth,
            "mean_breadth_reference": self.mean_breadth_reference,
            "mean_breadth_added": self.mean_breadth_added,
            "hurdle_formula": "2/sqrt(t_years)",
            "hurdle_caveat": (
                "2/sqrt(T) assumes bars of equal information content. The added "
                "bars carry a NARROWER cross-section (see breadth_histogram_added), "
                "so the measured hurdle reduction overstates the gain in evidence. "
                "Report the breadth distribution alongside the hurdle, never the "
                "hurdle alone."
            ),
        }


def measure_panel_growth(
    availability: pd.DataFrame,
    *,
    min_symbols_per_bar: int,
    reference_min_symbols_per_bar: int,
    bars_per_year: float,
) -> PanelGrowth:
    """Meet wat een breedte-eis aan bars toevoegt, en wat die bars aan breedte dragen.

    Parameters
    ----------
    availability
        Booleaans paneel, bars x symbolen: is dit symbool op deze bar BRUIKBAAR?
        "Bruikbaar" is de conjunctie die de aanroeper hanteert — in stap 12 is dat
        `sigma.notna() & adv.notna()`, want de risicolaag weigert een bar zonder
        ex-ante volatiliteit en zonder causale ADV.
    min_symbols_per_bar, reference_min_symbols_per_bar
        De nieuwe en de bestaande breedte-eis. De referentie is de huidige
        gebalanceerde eis (alle symbolen), zodat de groei tegen de stand van zaken
        wordt gemeten en niet tegen een gekozen vergelijkingspunt.
    bars_per_year
        VERPLICHT en zonder default, om dezelfde reden als in
        `inference.py::sharpe_with_se`.
    """
    require(
        isinstance(availability, pd.DataFrame),
        "measure_panel_growth verwacht een booleaans DataFrame (bars x symbolen). "
        "Zonder kolomidentiteit is de BREEDTE van een bar niet te tellen, en dat "
        "is de grootheid waar deze meting om draait.",
        DataContractError,
        availability_type=type(availability).__name__,
    )
    require(
        bool(availability.dtypes.eq(bool).all()),
        "availability moet volledig booleaans zijn. Een float-paneel zou hier "
        "stilzwijgend op waarheid worden getest en dan telt een 0.0-koers als "
        "afwezig terwijl hij een geldige waarneming is.",
        DataContractError,
        dtypes=sorted({str(d) for d in availability.dtypes}),
    )
    require(
        reference_min_symbols_per_bar > min_symbols_per_bar,
        "De referentie-eis moet STRENGER zijn dan de nieuwe eis; anders voegt de "
        "nieuwe eis per constructie niets toe en meet deze functie een groei die "
        "niet kan bestaan.",
        DataContractError,
        min_symbols_per_bar=int(min_symbols_per_bar),
        reference_min_symbols_per_bar=int(reference_min_symbols_per_bar),
    )
    require(
        bars_per_year > 0.0,
        "bars_per_year moet positief zijn; hij is de brug tussen bars en de "
        "t = 2-drempel, en zonder hem is die drempel niet uit te drukken.",
        DataContractError,
        bars_per_year=bars_per_year,
    )

    breadth = availability.sum(axis=1)
    new = breadth >= int(min_symbols_per_bar)
    ref = breadth >= int(reference_min_symbols_per_bar)
    require(
        bool(ref.any()),
        "Het referentievenster is leeg: geen enkele bar haalt de bestaande "
        "breedte-eis. Dan is er geen stand van zaken om de groei tegen te meten.",
        DataContractError,
        reference_min_symbols_per_bar=int(reference_min_symbols_per_bar),
    )
    added = new & ~ref

    n_new, n_ref = int(new.sum()), int(ref.sum())
    y_new, y_ref = n_new / float(bars_per_year), n_ref / float(bars_per_year)
    hurdle_new, hurdle_ref = 2.0 / np.sqrt(y_new), 2.0 / np.sqrt(y_ref)

    def _hist(mask: pd.Series) -> dict[int, int]:
        counts = breadth[mask].value_counts()
        return {int(k): int(v) for k, v in counts.items()}

    return PanelGrowth(
        min_symbols_per_bar=int(min_symbols_per_bar),
        reference_min_symbols_per_bar=int(reference_min_symbols_per_bar),
        n_bars=n_new,
        n_bars_reference=n_ref,
        n_bars_added=int(added.sum()),
        growth_fraction=float(n_new / n_ref - 1.0),
        t_years=float(y_new),
        t_years_reference=float(y_ref),
        t2_hurdle=float(hurdle_new),
        t2_hurdle_reference=float(hurdle_ref),
        t2_hurdle_delta=float(hurdle_new - hurdle_ref),
        se_sharpe=float(1.0 / np.sqrt(y_new)),
        se_sharpe_reference=float(1.0 / np.sqrt(y_ref)),
        first_bar=str(availability.index[new][0].date()),
        first_bar_reference=str(availability.index[ref][0].date()),
        breadth_histogram=_hist(new),
        breadth_histogram_added=_hist(added),
        mean_breadth=float(breadth[new].mean()),
        mean_breadth_reference=float(breadth[ref].mean()),
        mean_breadth_added=(
            float(breadth[added].mean()) if bool(added.any()) else float("nan")
        ),
    )
