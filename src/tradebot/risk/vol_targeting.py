# src/tradebot/risk/vol_targeting.py
"""L7 Unconditional Volatility Targeting — standalone en stateless.

Phase 4, deliverable 1 / stap 3. De bindende formule uit audit sectie 14.1:

    w_t = min( max_leverage , sigma_target / sigma_hat_{t+1|t} )

`sigma_target` en `max_leverage` komen uit `conf/risk/default.yaml`. De
`sigma_hat` komt uit L2 (`volatility/ewma.py`, lambda=0.94 uit
`conf/model/volatility.yaml`) en is EX-ANTE: hij gebruikt uitsluitend
informatie tot en met `t`.

WAT DEZE MODULE NIET KENT
-------------------------
Geen alpha, geen modelnaam, geen symbool-specifieke uitzondering, geen
historische performance, geen toestand tussen bars. De voorganger
(`risk/portfolio.py::vol_target_multiplier`) kende dat alles wel: hij schaalde
op een INTERNE rolling realized vol, vermenigvuldigde die met een
crisis-multiplier die aan de skewness van precies `"BTCUSDT"` hing, en gaf
`1.0` terug zodra die realized vol nul was - een stille pass-through naar "geen
targeting" op exact het moment dat de schatter niets wist.

FAIL-FAST, EN WAAROM DAT HIER HET ZWAARST WEEGT
-----------------------------------------------
Bij een ontbrekende of niet-eindige `sigma_hat` CRASHT deze module. Er is geen
laatste-bekende-waarde, geen cross-sectionele mediaan en geen constante
volatiliteit. De reden is asymmetrisch: een vol-schatting ontbreekt precies
wanneer de markt iets doet wat de schatter niet kent, en dat is het slechtst
denkbare moment om terug te vallen op een aanname. De EWMA-estimator propageert
tijdens de burn-in bewust `NaN`; wie de risicolaag aanroept vóór die estimator
warm is, hoort te crashen en niet te handelen.

Dat geldt ook voor een symbool met `a_t = 0`. Een ontbrekende `sigma_hat`
wordt niet acceptabel doordat alpha toevallig vlak staat - dan zou de
geldigheid van de risicolaag afhangen van de output van de alpha-laag, en dat
is de verstrengeling die deze fase opheft.

DE BOEKVOLATILITEIT: DE COMONOTONE BOVENGRENS
----------------------------------------------
Sectie 14.1 geeft een SCALAR `w_t`, dus is er één boekbrede `sigma_hat` nodig.
Die volgt hier uit de per-asset schattingen onder de aanname rho = 1:

    sigma_book = sum_i |a_i| * sigma_hat_i

Dat is de comonotone bovengrens: voor elke werkelijke correlatiestructuur geldt
`sigma_book_werkelijk <= sigma_book`. Twee redenen om die te kiezen boven een
geschatte covariantiematrix:

1. **Conservatief bij twijfel.** De targeting kan hierdoor te veel de-grossen,
   nooit te weinig. Een correlatieschatting die de diversificatie overschat,
   doet precies het omgekeerde - en doet dat het hardst in een crisis.
2. **Laagzuiverheid.** Een covariantiematrix is een schattingsobject met een
   eigen venster, eigen shrinkage en eigen faalmodi. Dat hoort in L8
   (allocatie), niet in de soevereine limietlaag. Scenario S2 (correlaties naar
   1) is onder deze aanname per constructie al ingeprijsd.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 9.1, 14, 14.1, 19 (L2/L7), 22.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..utils.failfast import ConfigContractError, DataContractError, require
from .contract import BOOK_SCOPE, BindingConstraint, ConstraintKind

__all__ = [
    "SIGMA_TARGET_KEY",
    "apply_volatility_target",
    "book_sigma_hat",
    "require_sigma_hat",
    "volatility_scalar",
]

#: De configuratiesleutel die de drempel zet. Verschijnt in het auditspoor,
#: zodat elk besluit herleidbaar is tot conf/risk/ en niet alleen tot een getal.
SIGMA_TARGET_KEY = "risk.sigma_target"


def require_sigma_hat(sigma_hat: object, *, symbol: str) -> float:
    """Valideer één ex-ante vol-schatting, of crash.

    Geen enkele van de afgewezen gevallen krijgt een vervangwaarde: dit is het
    punt waar `risk/portfolio.py` en `alpha/adaptive_wf.py` allebei stilzwijgend
    doorgingen.
    """
    value = float(sigma_hat) if sigma_hat is not None else float("nan")
    require(
        np.isfinite(value),
        "Ontbrekende of niet-eindige ex-ante volatiliteitsschatting. De "
        "risicolaag valt NIET terug op een laatste bekende waarde en NIET op "
        "een constante vol: een vol-schatting ontbreekt precies wanneer de "
        "markt iets doet wat de schatter niet kent, en dat is het slechtste "
        "moment voor een aanname. Draai de EWMA-estimator warm "
        "(conf/model/volatility.yaml::burn_in_bars) voordat de risicolaag "
        "wordt aangeroepen.",
        DataContractError,
        symbol=symbol,
        sigma_hat=value,
    )
    require(
        value > 0.0,
        "Niet-positieve ex-ante volatiliteit. Een vol van nul levert een "
        "oneindige vol-targeting scalar op, oftewel een positiegrootte die "
        "door niets meer wordt begrensd.",
        DataContractError,
        symbol=symbol,
        sigma_hat=value,
    )
    return value


def volatility_scalar(*, sigma_hat: float, sigma_target: float, max_leverage: float) -> float:
    """De bindende formule uit sectie 14.1: `min(max_leverage, sigma_target/sigma_hat)`.

    Puur, stateless, symbool-agnostisch. `sigma_hat` is al gevalideerd door
    `require_sigma_hat`; `sigma_target` en `max_leverage` komen uit
    `conf/risk/`.
    """
    require(
        sigma_target > 0.0,
        "sigma_target moet strikt positief zijn; een target van nul betekent "
        "'nooit handelen' en hoort een expliciete HALT te zijn, geen limiet.",
        ConfigContractError,
        sigma_target=float(sigma_target),
    )
    require(
        max_leverage > 0.0,
        "max_leverage moet strikt positief zijn.",
        ConfigContractError,
        max_leverage=float(max_leverage),
    )
    return float(min(float(max_leverage), float(sigma_target) / float(sigma_hat)))


def book_sigma_hat(
    exposures: Mapping[str, float], sigma_hat: Mapping[str, float]
) -> float:
    """Ex-ante boekvolatiliteit als comonotone bovengrens: `sum |a_i| * sigma_i`.

    Elk symbool in `exposures` MOET een geldige `sigma_hat` hebben, ook wanneer
    zijn gewenste exposure nul is - zie de moduledocstring.
    """
    total = 0.0
    for symbol, weight in exposures.items():
        sigma = require_sigma_hat(sigma_hat.get(symbol), symbol=str(symbol))
        total += abs(float(weight)) * sigma
    return float(total)


def apply_volatility_target(
    desired_exposure: Mapping[str, float],
    sigma_hat: Mapping[str, float],
    *,
    sigma_target: float,
    max_leverage: float,
) -> tuple[dict[str, float], BindingConstraint | None]:
    """Schaal het hele boek met één `w_t` en registreer of de limiet bond.

    Returns
    -------
    (permitted, binding)
        `binding` is `None` wanneer `w_t >= 1` - dan heeft de targeting de
        gevraagde exposure niet verkleind en mag er per contract (sectie 5.2
        van docs/RISK_CONTRACT.md) niets in het auditspoor komen.

    Notes
    -----
    `w_t` wordt bij 1.0 afgekapt: de risicolaag VERKLEINT uitsluitend. Bij een
    zeer rustige markt geeft de formule `sigma_target/sigma_hat > 1`, en die
    ruimte OPVULLEN zou betekenen dat L7 exposure toevoegt die L4 niet heeft
    gevraagd - dat is geen risicobeheer maar een tweede alpha-unit
    (docs/RISK_CONTRACT.md sectie 5.3). Het opschalen naar een vol-target hoort
    in L8, met de gewichten die daar worden geconstrueerd.
    """
    sigma_book = book_sigma_hat(desired_exposure, sigma_hat)

    if sigma_book <= 0.0:
        # Het boek is volledig vlak (elke a_t == 0). Er is niets te schalen en
        # er bindt niets; sigma_book kan hier niet nul zijn door een ongeldige
        # sigma_hat, want book_sigma_hat() heeft die al afgewezen.
        return {str(k): float(v) for k, v in desired_exposure.items()}, None

    raw = volatility_scalar(
        sigma_hat=sigma_book, sigma_target=sigma_target, max_leverage=max_leverage
    )
    w_t = min(raw, 1.0)

    permitted = {str(k): float(v) * w_t for k, v in desired_exposure.items()}
    if w_t >= 1.0:
        return permitted, None

    gross_before = float(sum(abs(float(v)) for v in desired_exposure.values()))
    binding = BindingConstraint(
        kind=ConstraintKind.VOL_TARGET,
        scope=BOOK_SCOPE,
        measured=sigma_book,
        threshold=float(sigma_target),
        exposure_before=gross_before,
        exposure_after=gross_before * w_t,
        config_key=SIGMA_TARGET_KEY,
    )
    return permitted, binding
