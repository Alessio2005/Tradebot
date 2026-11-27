"""Toelating van entry-kandidaten: het volatiliteitsregime en de theta-harvest-poort.

Twee vaste regels tussen `Strategy.scan` en de risicopoort, alleen voor NIEUWE posities:

1. **Regime.** Een strategie mag alleen openen in de regimes waarvoor ze in
   `vol_regime.allowed` staat (`vol_regime`-module). UNKNOWN en STRESS staan standaard leeg.
   Een kandidaat met undefined risk (`max_loss` oneindig) komt in STRESS nooit binnen, ook niet
   met `vol_regime.enabled: false` of met de strategie in `allowed.stress`
   (`regime:stress:undefined_risk`): de engine zou hem anders meteen weer sluiten
   (`regime_stress`). Beide regels lezen hetzelfde effectieve regime.
2. **Theta harvesting.** Een kandidaat die premie ontvangt (`net_credit > 0`) moet aan twee
   voorwaarden voldoen:

   * de **theta-dekking**: het tijdswaardeverval per jaar tegenover het verwachte gamma-verlies
     per jaar bij de VERWACHTE RV (niet de IV):

         dekking = (θ_dag × 365) / (½ · |Γ| · S² · σ_RV²)

     Voor één gehedgede short optie is dat ongeveer σ_IV² / σ_RV²: boven 1 betaalt de premie
     het verwachte bewegingsverlies, en dat is precies de volatility risk premium op het
     niveau van de positie. Een positie zonder short gamma (Γ ≥ 0) heeft geen gamma-verlies
     en haalt de dekking altijd;
   * de **theta-efficiëntie**: θ per dag als fractie van het maximale verlies, minimaal
     `min_theta_per_max_loss`.

   Kandidaten die premie betalen (debit) gaan zonder deze poort door: hun theta is per
   constructie negatief, daar zit het rendement in de beweging.

Eén correctie op het regime (N14): een earnings-variant van de long straddle (`variant ==
"earnings"`) kan door zijn EIGEN event in STRESS komen. De IV30 bevat dan de event-variantie, de
IV60 minder, en de termijnhelling wordt negatief (backwardation). Is het regime STRESS met
reden `backwardation`, dan toetst `admit` voor die variant het regime opnieuw zonder
termijnhelling (`classify(..., term_slope=None)`) en gebruikt dat. Een vol-schok (`rv_expansion`)
blijft STRESS. Beperking: dit onderscheidt niet of STRESS van de marktonderliggende (SPY) komt
terwijl de eigen onderliggende toevallig ook in backwardation staat; de variant is long vol met
gedefinieerd risico, dus dat is geaccepteerd.

De uitkomst komt als spoorregels in `Candidate.rule_trace`, zodat het journal per entry laat
zien welk regime en welke dekking er golden.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from ..schemas.options import OptionsConfig
from .chain.builder import Chain
from .positions import Candidate, position_greek
from .vol_regime import RegimeSettings, VolRegime, VolState, classify, regime_settings

__all__ = ["Admission", "SelectionRules", "admit", "selection_rules", "theta_cover"]


@dataclass(frozen=True, slots=True)
class SelectionRules:
    regime_enabled: bool = True
    allowed: Mapping[VolRegime, frozenset[str]] = dataclasses.field(default_factory=dict)
    theta_enabled: bool = True
    min_theta_gamma_cover: float = 1.0
    min_theta_per_max_loss: float = 0.0
    regime_settings: RegimeSettings | None = None   # voor de herberekening bij earnings (N14)


def selection_rules(cfg: OptionsConfig) -> SelectionRules:
    vr, th = cfg.vol_regime, cfg.theta_harvest
    allowed = {VolRegime(name): frozenset(names) for name, names in vr.allowed.model_dump().items()}
    return SelectionRules(vr.enabled, allowed, th.enabled, th.min_theta_gamma_cover,
                          th.min_theta_per_max_loss,
                          regime_settings(vr, cfg.iv_metrics.atm_tenor_days))


def theta_cover(theta_day: float, gamma: float, spot: float, rv: float) -> float:
    """(θ_dag × 365) / (½ · |Γ| · S² · σ²); `inf` zonder short gamma.

    σ is de verwachte RV voor de looptijd: `RVForecast.at_horizon` van de vroegste expiratie.
    """
    if gamma >= 0:
        return math.inf
    return theta_day * 365.0 / (0.5 * -gamma * spot * spot * rv * rv)


@dataclass(frozen=True, slots=True)
class Admission:
    candidate: Candidate
    admitted: bool
    blocker: str = ""          # "regime" of "theta"; leeg als toegelaten


def admit(c: Candidate, chain: Chain, vol: VolState | None, regime: VolRegime,
          rules: SelectionRules, today: date) -> Admission:
    """Laat `c` toe of niet, met de spoorregels erbij."""
    trace: list[str] = []
    if (c.variant == "earnings" and regime is VolRegime.STRESS and vol is not None
            and vol.regime is VolRegime.STRESS and vol.reason == "backwardation"
            and rules.regime_settings is not None):
        regime = classify(vol.atm_iv, vol.rv, None, rules.regime_settings).regime
        trace.append("regime:earnings_term_ignored")
    if math.isinf(c.max_loss) and regime is VolRegime.STRESS:
        return Admission(_traced(c, ["regime:stress:undefined_risk"]), False, "regime")
    if rules.regime_enabled:
        ok = c.strategy in rules.allowed.get(regime, frozenset())
        trace.append(f"regime:{regime.value}:{'ok' if ok else 'no'}")
        if not ok:
            return Admission(_traced(c, trace), False, "regime")
    if rules.theta_enabled and c.net_credit > 0:
        ok, note = _theta_ok(c, chain, vol, rules, today)
        trace.append(note)
        if not ok:
            return Admission(_traced(c, trace), False, "theta")
    return Admission(_traced(c, trace), True)


def _theta_ok(c: Candidate, chain: Chain, vol: VolState | None,
              rules: SelectionRules, today: date) -> tuple[bool, str]:
    theta = position_greek(chain, c.legs, "theta")
    gamma = position_greek(chain, c.legs, "gamma")
    if theta is None or gamma is None or vol is None or vol.rv is None:
        return False, "theta:no_data"
    if theta <= 0:
        return False, f"theta:negative:{theta:.2f}"
    days = (min(leg.contract.expiry for leg in c.legs) - today).days
    cover = theta_cover(theta, gamma, chain.spot, vol.rv.at_horizon(days))
    max_loss_unit = c.max_loss / c.quantity
    eff = theta / max_loss_unit
    ok = cover >= rules.min_theta_gamma_cover and eff >= rules.min_theta_per_max_loss
    cover_txt = "inf" if math.isinf(cover) else f"{cover:.2f}"
    return ok, f"theta:cover={cover_txt}:eff={eff:.4f}:{'ok' if ok else 'no'}"


def _traced(c: Candidate, trace: list[str]) -> Candidate:
    return dataclasses.replace(c, rule_trace=(*c.rule_trace, *trace)) if trace else c
