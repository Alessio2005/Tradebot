# src/tradebot/risk/limits.py
"""L7 harde limieten — concentratie, exposure caps en liquiditeit.

Phase 4, deliverable 2 / stap 4. Elke limiet is een ZELFSTANDIGE, testbare
functie met dezelfde vorm:

    limiet(exposures, ...) -> (nieuwe_exposures, [BindingConstraint, ...])

Ze kennen elkaar niet en roepen elkaar niet aan. De volgorde waarin ze worden
toegepast staat in `conf/risk/default.yaml::constraint_order` en wordt door
`risk/engine.py` uitgevoerd - nooit impliciet in de code-volgorde van dit
bestand. Dat is de stap 4-eis, en de reden is concreet: in
`portfolio/constraints.py` bepaalde de regelvolgorde of de leverage-cap vóór of
ná de concentratieclip viel, en dat verschil was nergens vastgelegd.

DE INVARIANT DIE ELKE FUNCTIE HIER RESPECTEERT
----------------------------------------------
`|nieuw_i| <= |oud_i|` en het teken blijft gelijk of wordt nul. De risicolaag
verkleint uitsluitend (docs/RISK_CONTRACT.md sectie 5.3). `BindingConstraint`
weigert te bestaan wanneer dat niet klopt, dus de invariant wordt bij elke
registratie opnieuw gecontroleerd en niet alleen hier aangenomen.

WAT HIER SAMENKOMT
------------------
Vier bestaande, onderling onbekende implementaties (sectie 4.3 van
`reports/phase4_entanglement_map.md`): `risk/position_limits.py` (compleet,
getest, door geen enkele productiecode aangeroepen), de inline caps in
`risk/portfolio.py::size_position`, `portfolio/constraints.py`, en
`live/execution_controller.py` - die laatste met `max_gross_notional`
geinitialiseerd op `float("inf")`, oftewel: wie de setter vergat, handelde
zonder gross-limiet.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 14, 14.1, 19 (L7).
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..utils.failfast import ConfigContractError, DataContractError, require
from .contract import BOOK_SCOPE, BindingConstraint, ConstraintKind

__all__ = [
    "ADV_CAP_KEY",
    "CLUSTER_CAP_KEY",
    "CONCENTRATION_KEY",
    "GROSS_CAP_KEY",
    "NET_CAP_KEY",
    "PER_ASSET_KEY",
    "apply_adv_cap",
    "apply_cluster_cap",
    "apply_concentration_cap",
    "apply_gross_cap",
    "apply_net_cap",
    "apply_per_asset_cap",
    "effective_relative_cap",
    "gross_exposure",
    "net_exposure",
]

PER_ASSET_KEY = "risk.max_position_pct"
ADV_CAP_KEY = "risk.adv_participation_cap"
CONCENTRATION_KEY = "risk.max_concentration"
CLUSTER_CAP_KEY = "risk.max_cluster_concentration"
GROSS_CAP_KEY = "risk.gross_cap"
NET_CAP_KEY = "risk.net_cap"

#: Afrondingstolerantie voor de invarianten. Geen beleidsdrempel.
_TOL = 1e-12


def gross_exposure(exposures: Mapping[str, float]) -> float:
    """`sum |w_i|`."""
    return float(sum(abs(float(v)) for v in exposures.values()))


def net_exposure(exposures: Mapping[str, float]) -> float:
    """`sum w_i` (getekend)."""
    return float(sum(float(v) for v in exposures.values()))


def _clean(exposures: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol, value in exposures.items():
        w = float(value)
        require(
            np.isfinite(w),
            "Niet-eindige exposure aangeboden aan de limietlaag.",
            DataContractError,
            symbol=str(symbol),
            exposure=w,
        )
        out[str(symbol)] = w
    return out


def _require_fraction(value: float, *, key: str) -> float:
    v = float(value)
    require(
        np.isfinite(v) and 0.0 < v <= 1.0,
        f"{key} moet een fractie in (0, 1] zijn. Een ontbrekende of ongeldige "
        "limietconfiguratie is een crash, geen 'geen limiet'.",
        ConfigContractError,
        key=key,
        value=v,
    )
    return v


def _require_positive(value: float, *, key: str) -> float:
    v = float(value)
    require(
        np.isfinite(v) and v > 0.0,
        f"{key} moet eindig en strikt positief zijn. `float('inf')` als default "
        "is precies hoe live/execution_controller.py zonder gross-limiet kon "
        "handelen.",
        ConfigContractError,
        key=key,
        value=v,
    )
    return v


# --------------------------------------------------------------------------- #
# Absolute per-asset limieten
# --------------------------------------------------------------------------- #
def apply_per_asset_cap(
    exposures: Mapping[str, float], *, max_position_pct: float
) -> tuple[dict[str, float], list[BindingConstraint]]:
    """Absolute exposure-cap per asset: `|w_i| <= max_position_pct`."""
    cap = _require_fraction(max_position_pct, key=PER_ASSET_KEY)
    out = _clean(exposures)
    bound: list[BindingConstraint] = []
    for symbol, w in list(out.items()):
        if abs(w) > cap + _TOL:
            capped = float(np.sign(w) * cap)
            bound.append(
                BindingConstraint(
                    kind=ConstraintKind.PER_ASSET_CAP,
                    scope=symbol,
                    measured=abs(w),
                    threshold=cap,
                    exposure_before=w,
                    exposure_after=capped,
                    config_key=PER_ASSET_KEY,
                )
            )
            out[symbol] = capped
    return out, bound


def apply_adv_cap(
    exposures: Mapping[str, float],
    adv_usd: Mapping[str, float],
    *,
    participation_cap: float,
    equity: float,
) -> tuple[dict[str, float], list[BindingConstraint]]:
    """Positielimiet uit liquiditeit: `|w_i| * equity <= participation_cap * ADV_i`.

    De cap wordt uitgedrukt in exposure-eenheden zodat hij in dezelfde valuta
    staat als de andere limieten:

        |w_i| <= participation_cap * ADV_i / equity

    Een symbool zonder ADV-waarde is een CRASH, geen vrijstelling. `risk/
    portfolio.py` had deze limiet wel gedocumenteerd (`max_adv_fraction`) maar
    nooit afgedwongen: de waarde stond op `None` en de aanroeper werd geacht
    zelf ADV aan te leveren, wat nergens gebeurde.
    """
    cap_frac = _require_fraction(participation_cap, key=ADV_CAP_KEY)
    eq = _require_positive(equity, key="risk_state.equity")
    out = _clean(exposures)
    bound: list[BindingConstraint] = []
    for symbol, w in list(out.items()):
        raw_adv = adv_usd.get(symbol)
        adv = float(raw_adv) if raw_adv is not None else float("nan")
        require(
            np.isfinite(adv) and adv > 0.0,
            "Ontbrekend of ongeldig ADV voor een symbool met een actieve "
            "liquiditeitslimiet. De risicolaag neemt geen liquiditeit aan die "
            "zij niet heeft gemeten.",
            DataContractError,
            symbol=symbol,
            adv_usd=adv,
        )
        limit = cap_frac * adv / eq
        if abs(w) > limit + _TOL:
            capped = float(np.sign(w) * limit)
            bound.append(
                BindingConstraint(
                    kind=ConstraintKind.ADV_CAP,
                    scope=symbol,
                    measured=abs(w),
                    threshold=limit,
                    exposure_before=w,
                    exposure_after=capped,
                    config_key=ADV_CAP_KEY,
                )
            )
            out[symbol] = capped
    return out, bound


# --------------------------------------------------------------------------- #
# Relatieve limieten (vast punt: clippen verlaagt de gross)
# --------------------------------------------------------------------------- #
def effective_relative_cap(n_active: int, cap: float) -> float:
    """De afdwingbare vorm van een relatieve cap: `max(cap, 1/n_active)`.

    Een cap onder `1/n` is wiskundig onhaalbaar - sommeren over de groepen
    geeft `gross <= n * cap * gross`, dus `n * cap >= 1` is noodzakelijk. Bij
    `cap = 0.40` en twee actieve posities bestaat er geen andere oplossing dan
    het hele boek naar nul.

    Daar hoort geen crash bij. Het aantal actieve posities is een eigenschap
    van het BOEK op deze bar, niet van de configuratie: een bar waarop alpha
    toevallig twee namen aanwijst, mag het platform niet stilleggen. En het
    hoort al helemaal geen stille nul te worden - een limiet die alles
    blokkeert ziet er in de logs uit als een limiet die werkt.

    De juiste lezing is dat de cap de dominantie van één groep begrenst. Bij
    `n` actieve groepen is de minst dominante verdeling gelijk verdeeld, dus
    `1/n`. Die vloer is de strengst AFDWINGBARE vorm van de limiet: hij staat
    nooit meer concentratie toe dan een gelijke verdeling, en hij is per
    constructie haalbaar.
    """
    require(
        n_active > 0,
        "Een relatieve limiet op een leeg boek.",
        ConfigContractError,
        cap=cap,
    )
    return float(max(float(cap), 1.0 / float(n_active)))


def _water_filling_limit(magnitudes: list[float], cap: float, *, key: str) -> float:
    """Het exacte vaste punt `L` van `|w_i| <= cap * sum_j |w_j|`, gesloten opgelost.

    Naief itereren (clip, hertel de gross, clip opnieuw) convergeert LINEAIR met
    factor `k * cap`, en dat is voor realistische configuraties tergend traag:
    bij `cap=0.40` en twee bindende assets duurt het ~120 iteraties om 1e-12 te
    halen. Een iteratieplafond zou dan een correcte configuratie laten crashen,
    en zonder plafond zou het boek stilzwijgend naar nul kunnen lopen. Dus:
    geen iteratie.

    Constructie. Zij `S` de verzameling geclipte assets (elk op `L = cap * G`)
    en `U` de rest. Dan geldt `G = |S| * L + W_U` met `W_U = sum_{i in U} |w_i|`,
    en dus

        G = W_U / (1 - |S| * cap)      en      L = cap * G.

    Sorteer aflopend op `|w|` en probeer `k = 0, 1, 2, ...` als grootte van `S`.
    De juiste `k` is de kleinste waarvoor de eerste ONgeclipte magnitude onder
    `L` valt: dan is de aanname consistent met de uitkomst.
    """
    ordered = sorted(magnitudes, reverse=True)
    n = len(ordered)
    suffix = 0.0
    tail: list[float] = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix += ordered[i]
        tail[i] = suffix

    for k in range(n):
        denom = 1.0 - k * cap
        if denom <= _TOL:
            break
        limit = cap * tail[k] / denom
        # `ordered[k]` is de grootste ONgeclipte magnitude. Past hij onder de
        # limiet, dan is de aanname "de eerste k zijn geclipt" zelfconsistent.
        if ordered[k] <= limit + _TOL:
            return float(limit)

    require(
        False,
        f"{key}={cap} levert geen consistent vast punt op voor dit boek; elke "
        "exposure zou naar nul lopen.",
        ConfigContractError,
        key=key,
        cap=cap,
        n=n,
    )
    raise AssertionError  # pragma: no cover - require() heeft al geraised


def apply_concentration_cap(
    exposures: Mapping[str, float], *, max_concentration: float
) -> tuple[dict[str, float], list[BindingConstraint]]:
    """Max fractie van de gross exposure in één asset: `|w_i| <= c * gross`."""
    cap = _require_fraction(max_concentration, key=CONCENTRATION_KEY)
    out = _clean(exposures)
    before = dict(out)
    gross_before = gross_exposure(out)
    if gross_before <= _TOL:
        return out, []

    active = [s for s, w in out.items() if abs(w) > _TOL]
    cap = effective_relative_cap(len(active), cap)

    limit = _water_filling_limit(
        [abs(w) for w in out.values()], cap, key=CONCENTRATION_KEY
    )
    for symbol, w in list(out.items()):
        if abs(w) > limit + _TOL:
            out[symbol] = float(np.sign(w) * limit)

    final_gross = gross_exposure(out)
    # `measured` is het aandeel dat het symbool bij de UITEINDELIJKE gross zou
    # hebben gehad als hij niet was geclipt. Dat is de grootheid die de clip
    # verklaart: hij ligt per constructie boven `cap` voor precies de geclipte
    # symbolen. Het aandeel VOORAF zou misleiden - een symbool kan onder de cap
    # beginnen en er alsnog boven uitkomen zodra een groter symbool de gross
    # omlaag trekt.
    bound = [
        BindingConstraint(
            kind=ConstraintKind.CONCENTRATION_CAP,
            scope=symbol,
            measured=abs(before[symbol]) / final_gross if final_gross > _TOL else 0.0,
            threshold=cap,
            exposure_before=before[symbol],
            exposure_after=out[symbol],
            config_key=CONCENTRATION_KEY,
        )
        for symbol in out
        if abs(out[symbol]) < abs(before[symbol]) - _TOL
    ]
    require(
        final_gross <= _TOL
        or max(abs(w) for w in out.values()) <= cap * final_gross + 1e-9,
        "De concentratielimiet is na oplossen nog steeds geschonden.",
        ConfigContractError,
        key=CONCENTRATION_KEY,
    )
    return out, bound


def apply_cluster_cap(
    exposures: Mapping[str, float],
    clusters: Mapping[str, str],
    *,
    max_cluster_concentration: float,
) -> tuple[dict[str, float], list[BindingConstraint]]:
    """Max fractie van de gross exposure in één sector/cluster.

    Een symbool zonder clusterlabel is een CRASH en komt niet in een
    "overige"-emmer terecht: een ongelabeld symbool zou de limiet ongemerkt
    kunnen ontlopen, en dat is precies de klasse gat die deze fase sluit.
    """
    cap = _require_fraction(max_cluster_concentration, key=CLUSTER_CAP_KEY)
    out = _clean(exposures)
    before = dict(out)
    if gross_exposure(out) <= _TOL:
        return out, []

    labels: dict[str, str] = {}
    for symbol in out:
        label = clusters.get(symbol)
        require(
            label is not None and str(label) != "",
            "Symbool zonder cluster-label terwijl de clusterlimiet actief is. "
            "Een ongelabeld symbool ontloopt de limiet stilzwijgend; vul "
            "conf/risk/default.yaml::clusters aan.",
            ConfigContractError,
            symbol=symbol,
            key=CLUSTER_CAP_KEY,
        )
        labels[symbol] = str(label)

    per_cluster: dict[str, float] = {}
    for symbol, w in out.items():
        per_cluster[labels[symbol]] = per_cluster.get(labels[symbol], 0.0) + abs(w)

    active_clusters = {c for c, g in per_cluster.items() if g > _TOL}
    cap = effective_relative_cap(len(active_clusters), cap)

    limit = _water_filling_limit(
        list(per_cluster.values()), cap, key=CLUSTER_CAP_KEY
    )
    measured: dict[str, float] = {}
    for cluster, exposure in per_cluster.items():
        if exposure > limit + _TOL:
            shrink = limit / exposure
            measured[cluster] = exposure
            for symbol in out:
                if labels[symbol] == cluster:
                    out[symbol] = out[symbol] * shrink

    # Zie de toelichting bij de concentratielimiet: het aandeel wordt gemeten
    # tegen de UITEINDELIJKE gross, want dat is wat de clip verklaart.
    final_gross = gross_exposure(out)
    measured = {
        c: (v / final_gross if final_gross > _TOL else 0.0)
        for c, v in measured.items()
    }
    bound = [
        BindingConstraint(
            kind=ConstraintKind.CLUSTER_CAP,
            scope=labels[symbol],
            measured=measured.get(labels[symbol], cap),
            threshold=cap,
            exposure_before=before[symbol],
            exposure_after=out[symbol],
            config_key=CLUSTER_CAP_KEY,
        )
        for symbol in out
        if abs(out[symbol]) < abs(before[symbol]) - _TOL
    ]
    return out, bound


# --------------------------------------------------------------------------- #
# Boekbrede caps — per contract als LAATSTE toegepast
# --------------------------------------------------------------------------- #
def apply_gross_cap(
    exposures: Mapping[str, float], *, cap: float
) -> tuple[dict[str, float], list[BindingConstraint]]:
    """Gross Exposure Cap: `sum |w_i| <= cap`, proportioneel teruggeschaald."""
    limit = _require_positive(cap, key=GROSS_CAP_KEY)
    out = _clean(exposures)
    gross = gross_exposure(out)
    if gross <= limit + _TOL:
        return out, []

    shrink = limit / gross
    bound = [
        BindingConstraint(
            kind=ConstraintKind.GROSS_CAP,
            scope=BOOK_SCOPE,
            measured=gross,
            threshold=limit,
            exposure_before=gross,
            exposure_after=gross * shrink,
            config_key=GROSS_CAP_KEY,
        )
    ]
    return {s: w * shrink for s, w in out.items()}, bound


def apply_net_cap(
    exposures: Mapping[str, float], *, cap: float
) -> tuple[dict[str, float], list[BindingConstraint]]:
    """Net Exposure Cap: `|sum w_i| <= cap`.

    Uitsluitend de DOMINANTE zijde wordt teruggeschaald. De tegenzijde
    ophogen zou de net-limiet ook halen, maar dat is exposure toevoegen die
    niemand heeft gevraagd - verboden onder sectie 5.3.
    """
    limit = _require_positive(cap, key=NET_CAP_KEY)
    out = _clean(exposures)
    net = net_exposure(out)
    if abs(net) <= limit + _TOL:
        return out, []

    longs = sum(w for w in out.values() if w > 0.0)
    shorts = -sum(w for w in out.values() if w < 0.0)

    if net > 0.0:
        # f * longs - shorts = limit
        require(
            longs > 0.0,
            "Netto exposure positief zonder long-posities; de risicostaat is "
            "inconsistent.",
            DataContractError,
            net=net,
        )
        shrink = (limit + shorts) / longs
        side = 1.0
    else:
        require(
            shorts > 0.0,
            "Netto exposure negatief zonder short-posities; de risicostaat is "
            "inconsistent.",
            DataContractError,
            net=net,
        )
        shrink = (limit + longs) / shorts
        side = -1.0

    shrink = float(np.clip(shrink, 0.0, 1.0))
    scaled = {
        s: (w * shrink if np.sign(w) == side else w) for s, w in out.items()
    }
    bound = [
        BindingConstraint(
            kind=ConstraintKind.NET_CAP,
            scope=BOOK_SCOPE,
            measured=abs(net),
            threshold=limit,
            exposure_before=abs(net),
            exposure_after=abs(net_exposure(scaled)),
            config_key=NET_CAP_KEY,
        )
    ]
    return scaled, bound
