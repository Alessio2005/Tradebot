"""Greeks per aandeel (detailontwerp 5.5), in de eenheden van `tradebot.options.units`.

| Greek          | Hoe                                                               |
|----------------|-------------------------------------------------------------------|
| delta, gamma   | uit de engine (analytisch, binomiaal, FD); bij BAW met spotbumps    |
| vega           | centraal verschil in σ, per volpunt                               |
| rho            | centraal verschil in r, per procentpunt; eenzijdig vanaf r = 0     |
|                | als r < de bump (BAW weigert een call bij een negatieve rente)     |
| theta          | herwaardering één kalenderdag later, per kalenderdag              |

Vega, rho en theta gaan voor elke engine met bump-and-reprice. Zo betekenen ze bij een
Europese en een Amerikaanse optie precies hetzelfde, ook met de schaling van de looptijd
uit `ql_env` (de analytische QuantLib-vega is een afgeleide naar de geschaalde σ).
Theta is de echte verandering over één kalenderdag (of de halve resterende looptijd als die
korter is, teruggerekend naar een dag), niet een momentane afgeleide.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np

from ..calendar import NY, expiry_moment, year_fraction
from ..units import GreeksPerShare
from .engines import EngineSettings, Method
from .vanilla import OptionInputs, price, value

__all__ = ["Bumps", "greeks", "spot_bump_fraction"]


@dataclass(frozen=True, slots=True)
class Bumps:
    vol: float = 0.01          # 1 volpunt
    rate: float = 0.0001       # 1 bp
    spot_frac: float = 0.005   # bovengrens van de spotbump (fractie van spot), alleen BAW
    spot_frac_min: float = 0.0005    # ondergrens: onder dit niveau domineert afrondingsruis
    spot_sigma_frac: float = 0.25    # de bump is deze fractie van σ√T (de breedte van de dichtheid)


def spot_bump_fraction(vol: Any, t_real: Any, bumps: Bumps = Bumps()) -> Any:
    """De spotbump als fractie van spot voor delta/gamma van BAW (scalar of array).

    Schaalt met σ√T, geklemd op `[spot_frac_min, spot_frac]`. Een vaste bump van 0,5 % is op
    de expiratiedag (σ√T ≈ 1-2 %) een kwart tot de helft van de dichtheidsbreedte en geeft dan
    een gamma dat de piek rond de strike uitsmeert; bij langere looptijden (σ√T ≥ 2 %) is de
    bump ongewijzigd 0,5 %.
    """
    raw = bumps.spot_sigma_frac * vol * np.sqrt(t_real)
    return np.clip(raw, bumps.spot_frac_min, bumps.spot_frac)


def greeks(inp: OptionInputs, vol: float, method: Method,
           engines: EngineSettings = EngineSettings(), bumps: Bumps = Bumps()) -> GreeksPerShare:
    base = value(inp, vol, method, engines)

    def npv(i: OptionInputs, sigma: float = vol) -> float:
        return price(i, sigma, method, engines)

    if base.delta is not None and base.gamma is not None:
        delta, gamma = base.delta, base.gamma
    else:
        t_real = year_fraction(inp.now, expiry_moment(inp.expiry))
        h = float(spot_bump_fraction(vol, t_real, bumps)) * inp.spot
        up, down = npv(inp.with_spot(inp.spot + h)), npv(inp.with_spot(inp.spot - h))
        delta = (up - down) / (2 * h)
        gamma = (up - 2 * base.npv + down) / (h * h)

    b = min(bumps.vol, 0.5 * vol)
    vega = (npv(inp, vol + b) - npv(inp, vol - b)) / (2 * b) * 0.01

    # Nooit een rente onder nul: bij r < bump gaat de bump omlaag maar tot 0 (eenzijdiger
    # verschil), want BAW weigert een call bij een negatieve rente.
    r = inp.carry.rate
    lo = max(r - bumps.rate, 0.0)
    width = 2 * bumps.rate if lo == r - bumps.rate else r + bumps.rate - lo
    rho = (npv(inp.with_rate(r + bumps.rate)) - npv(inp.with_rate(lo))) / width * 0.01

    remaining = expiry_moment(inp.expiry) - inp.now
    step = min(timedelta(days=1), remaining / 2)
    later = inp.with_now(inp.now + step)
    # Een dividend dat in de overgeslagen dag ex gaat, valt anders uit het venster van `later`
    # en laat de optie "gratis" waarde winnen (theta van een call springt dan positief). Het
    # aandeel zakt over die dag met het dividend, dus dat gaat van de spot af.
    gone = inp.carry.dividends_in(inp.now.astimezone(NY).date(),
                                  later.now.astimezone(NY).date())
    for d in gone:
        later = later.with_spot(max(later.spot - d.amount, 1e-6))
    theta = (npv(later) - base.npv) / (step / timedelta(days=1))

    return GreeksPerShare(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
