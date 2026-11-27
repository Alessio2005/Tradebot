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

from ..calendar import expiry_moment
from ..units import GreeksPerShare
from .engines import EngineSettings, Method
from .vanilla import OptionInputs, price, value

__all__ = ["Bumps", "greeks"]


@dataclass(frozen=True, slots=True)
class Bumps:
    vol: float = 0.01          # 1 volpunt
    rate: float = 0.0001       # 1 bp
    spot_frac: float = 0.005   # 0,5 % van spot, alleen voor BAW


def greeks(inp: OptionInputs, vol: float, method: Method,
           engines: EngineSettings = EngineSettings(), bumps: Bumps = Bumps()) -> GreeksPerShare:
    base = value(inp, vol, method, engines)

    def npv(i: OptionInputs, sigma: float = vol) -> float:
        return price(i, sigma, method, engines)

    if base.delta is not None and base.gamma is not None:
        delta, gamma = base.delta, base.gamma
    else:
        h = bumps.spot_frac * inp.spot
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
    theta = (npv(inp.with_now(inp.now + step)) - base.npv) / (step / timedelta(days=1))

    return GreeksPerShare(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
