# src/tradebot/validation/regime_benchmark.py
"""H2 — de campagne: elke conditioneerder door de authoritative engine. Stap 11.

WAT EEN TRIAL IS
=================
Eén trial is één conditioneerder: M1 met k = 2 of 3, M2 gaussian met k = 2 of 3,
M2 student_t met k = 2 of 3. Zes, exact het `planned_trials` van de bevroren
pre-registratie `3d3af28730a6c7f9da48d13139522a05`, en die zes zijn bij het
BEVRIEZEN al in `M` geboekt.

M0 telt NIET mee. Er wordt geen ruimte doorzocht: zijn drempels staan in
`conf/model/regime.yaml`, zijn vastgelegd vóór er is gemeten, en er valt niets
aan te overfitten. Dat is precies waarom hij een oneerlijk sterke tegenstander
is en dus de juiste baseline. De ONGECONDITIONEERDE arm telt evenmin mee: die is
de Phase 5-baseline zelf en staat er als referentiepunt, zodat een overlay die
het slechter doet dan géén overlay als zodanig zichtbaar is in plaats van
alleen "slechter dan M0".

DE VOLGORDE VAN DE STOP-CRITERIA IS GEEN IMPLEMENTATIEDETAIL
=============================================================
1. **Bezetting** gaat eerst en is absorberend. Zakt de zeldzaamste FILTERED
   toestand onder 100 observaties per fold, dan is er niets geldigs geschat en
   valt er dus ook niets te falsifiëren. `UNPROVEN -- insufficient data`.
2. **Turnover** komt daarna, en dit criterium KAN falsifiëren. De
   pre-registratie zegt waarom: het hangt niet van de Sharpe-power af. Een
   overlay die de turnover verdubbelt zonder dat de netto-uitkomst verbetert,
   heeft de extra fees niet terugverdiend -- en dat is een economische
   vaststelling, geen statistische.
3. **Geen netto verbetering** descopeert en falsificeert NIET, omdat de
   power-analyse vóór de run vaststelt dat deze opzet het verwachte effect van
   0,08 niet kan zien. Afwezigheid van bewijs is hier geen bewijs van
   afwezigheid (no-go 8).
4. **Spread-gevoeligheid** kan alleen binden wanneer er een verbetering IS.
   Zonder gemeten winst is er niets dat bij 3 bp kan verdwijnen, en het
   criterium is dan niet van toepassing in plaats van geschonden.
5. **Promotie** vereist dat geen enkel criterium bindt, plus de volledige
   Phase 2-poort.

WAT DE ARCHITECTUUR AL VOOR DE RUN BEPERKT
===========================================
De soevereine laag schaalt het boek met `w_t = min(max_leverage,
sigma_target / sigma_boek)`. Een UNIFORME factor deelt daar weer uit: een
regime-overlay kan dit boek niet de-grossen. Wat overblijft is de asymmetrie
TUSSEN symbolen -- elk symbool heeft zijn eigen regime -- en dat maakt van H2 een
cross-sectionele tilt: haal het risicobudget weg bij wat nu onrustig is en geef
het aan de rest. Zie `backtest/regime_overlay.py`; het rapport zegt het, en
`overlay_is_a_tilt` meet het per run in plaats van het aan te nemen.

Ref: pre-registratie `3d3af28730a6c7f9da48d13139522a05`; fase-opdracht stap 11,
deliverable 18; §0.3 (kostenlabels), §0.5 (het haltprobleem).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..backtest.regime_overlay import ArmResult, MarketPanels, run_overlay_arm
from ..cv.walk_forward import WalkForwardCV
from ..regime.conditioning import (
    CONDITIONER_SPECS,
    ConditionerResult,
    build_conditioner,
    m0_multiplier,
    oos_mask,
)
from ..schemas.config import AdequacyConfig, M2HmmConfig
from ..utils.failfast import DataContractError, require
from .sharpe_difference import (
    SharpeDifferenceResult,
    SharpeTestControls,
    jobson_korkie_memmel,
    run_sharpe_controls,
)

__all__ = [
    "CampaignInputs",
    "H2CampaignResult",
    "H2Verdict",
    "TrialOutcome",
    "judge_conditioner",
    "run_regime_campaign",
]

VerdictStatus = Literal["PROMOTED", "UNPROVEN", "DESCOPED", "FALSIFIED"]

_OCCUPANCY = "state_occupancy_below_adequacy"
_TURNOVER = "turnover_doubles"
_NO_IMPROVEMENT = "no_sharpe_improvement_after_costs"
_SPREAD = "improvement_disappears_at_three_bp_spread"

#: De baselinearm. Nul geschatte parameters, dus geen trial en geen bijdrage
#: aan `M`.
M0_LABEL = "m0-buckets"
#: De Phase 5-baseline zonder enige overlay. Referentie, geen deelnemer.
PLAIN_LABEL = "unconditioned"


@dataclass(frozen=True)
class H2Verdict:
    """Het oordeel over één conditioneerder, met de criteria die eraan bonden."""

    status: VerdictStatus
    binding: tuple[str, ...]
    rationale: str

    def as_record(self) -> dict[str, Any]:
        return {"status": self.status, "binding": list(self.binding),
                "rationale": self.rationale}


def judge_conditioner(
    *,
    rarest_state_obs_per_fold: float,
    min_state_obs_required: float,
    net_sharpe_delta: float,
    turnover_ratio: float,
    turnover_threshold: float,
    spread_at_which_gain_vanishes: float | None,
    spread_threshold: float,
    dm: SharpeDifferenceResult,
    controls: SharpeTestControls,
) -> H2Verdict:
    """Pas de vijf stop-criteria toe, in de volgorde van de pre-registratie."""
    if rarest_state_obs_per_fold < min_state_obs_required:
        return H2Verdict(
            status="DESCOPED", binding=(_OCCUPANCY,),
            rationale=(
                f"De zeldzaamste toestand beslaat {rarest_state_obs_per_fold:.1f} "
                f"observaties per fold tegen een eis van "
                f"{min_state_obs_required:.0f}. Op zo'n bezetting schat het "
                f"model een gemiddelde, een schaal en k-1 overgangskansen op "
                f"enkele tientallen punten; dat is een uitschieterdetector en "
                f"geen regime. `UNPROVEN -- insufficient data`, en de trial "
                f"meet niets dat te falsifiëren valt."),
        )

    binding: list[str] = []
    if turnover_ratio >= turnover_threshold and net_sharpe_delta <= 0.0:
        binding.append(_TURNOVER)
    if net_sharpe_delta <= 0.0:
        binding.append(_NO_IMPROVEMENT)
    elif (spread_at_which_gain_vanishes is not None
            and spread_at_which_gain_vanishes <= spread_threshold):
        binding.append(_SPREAD)

    if not binding:
        return H2Verdict(
            status="PROMOTED", binding=(),
            rationale=(
                f"Netto Sharpe-delta {net_sharpe_delta:+.4f} boven M0 na "
                f"kosten, turnover {turnover_ratio:.2f}x, en de gemeten winst "
                f"overleeft een half-spread van "
                f"{spread_at_which_gain_vanishes if spread_at_which_gain_vanishes is not None else float('inf'):.1f} bp. "
                f"Geen enkel stop-criterium bindt; de Phase 2-poort beslist."),
        )

    if _TURNOVER in binding:
        return H2Verdict(
            status="FALSIFIED", binding=tuple(binding),
            rationale=(
                f"De overlay handelt {turnover_ratio:.2f}x de notional van M0 "
                f"en levert daarvoor een netto Sharpe-delta van "
                f"{net_sharpe_delta:+.4f} op. De extra fees zijn niet "
                f"terugverdiend. Dit criterium hangt niet van de Sharpe-power "
                f"af -- het is een economische vaststelling en dus wél een "
                f"falsificatie."),
        )

    if _SPREAD in binding:
        return H2Verdict(
            status="FALSIFIED", binding=tuple(binding),
            rationale=(
                f"De gemeten verbetering van {net_sharpe_delta:+.4f} verdwijnt "
                f"al bij een aangenomen half-spread van "
                f"{spread_at_which_gain_vanishes:.1f} bp, onder de grens van "
                f"{spread_threshold:.1f} bp. Dan is de promotie een "
                f"spread-aanname en geen modelresultaat."),
        )

    return H2Verdict(
        status="UNPROVEN", binding=tuple(binding),
        rationale=(
            f"Netto Sharpe-delta {net_sharpe_delta:+.4f} tegen M0 "
            f"(gepaard: z = {dm.z_statistic:.3f}, p = {dm.p_value:.4g}, "
            f"rho = {dm.correlation:.3f}). Geen verbetering, en deze opzet kon "
            f"het verwachte effect van {controls.expected_effect:.2f} "
            f"Sharpe-eenheden ook niet zien: de gemeten power daar is "
            f"{controls.power_at_expected_effect:.1%} tegen een doel van "
            f"{controls.target_power:.0%}. M0 blijft productie-baseline; "
            f"afwezigheid van bewijs is geen bewijs van afwezigheid."),
    )


@dataclass(frozen=True)
class TrialOutcome:
    """Eén conditioneerder: wat hij deed, wat hij kostte, wat hij opleverde."""

    label: str
    conditioner: ConditionerResult
    arm: ArmResult
    net_sharpe_delta: float
    turnover_ratio: float
    fees_delta: float
    #: De kleinste aangenomen half-spread waarbij de winst verdwijnt. `None`
    #: wanneer er bij de basis-spread geen winst was -- dan is er niets dat kan
    #: verdwijnen en is het criterium niet van toepassing.
    spread_at_which_gain_vanishes: float | None
    spread_sweep: Mapping[float, float]
    dm: SharpeDifferenceResult
    verdict: H2Verdict

    def as_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "conditioner": self.conditioner.as_record(),
            "arm": self.arm.as_record(),
            "net_sharpe_delta": self.net_sharpe_delta,
            "turnover_ratio": self.turnover_ratio,
            "fees_delta": self.fees_delta,
            "spread_at_which_gain_vanishes": self.spread_at_which_gain_vanishes,
            "spread_sweep": {str(k): v for k, v in self.spread_sweep.items()},
            "paired_sharpe_test": self.dm.as_record(),
            "verdict": self.verdict.as_record(),
        }


@dataclass(frozen=True)
class CampaignInputs:
    """Alles wat de campagne van buiten krijgt. Eén plaats, één contract."""

    returns: pd.DataFrame
    buckets: pd.DataFrame
    base_exposures: pd.DataFrame
    panels: MarketPanels
    cv: WalkForwardCV
    adequacy: AdequacyConfig
    m2_cfg: M2HmmConfig
    risk_cfg: Any
    impact: Any
    venue: Any
    spread_source: str
    initial_equity: float
    bars_per_year: float
    base_half_spread_bps: float
    spread_sweep_bps: Sequence[float]
    alpha: float
    expected_effect: float
    minimum_detectable_effect: float
    target_power: float
    n_control_replicates: int
    block_length: float
    seed: int


@dataclass(frozen=True)
class H2CampaignResult:
    """De volledige campagne, klaar voor het rapport en de ledger."""

    plain: ArmResult
    m0: ArmResult
    #: Netto OOS Sharpe van de M0-arm bij elke aangenomen half-spread. De
    #: baseline van elke delta in `TrialOutcome.spread_sweep`.
    m0_spread_sweep: Mapping[float, float]
    m0_conditioner_transitions: Mapping[str, int]
    trials: tuple[TrialOutcome, ...]
    controls: SharpeTestControls
    m0_vs_plain: SharpeDifferenceResult
    overlay_is_a_tilt: Mapping[str, float]
    n_planned_trials: int

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for trial in self.trials:
            out[trial.verdict.status] = out.get(trial.verdict.status, 0) + 1
        return out

    def as_record(self) -> dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "n_planned_trials": self.n_planned_trials,
            "by_status": self.by_status(),
            "plain": self.plain.as_record(),
            "m0": self.m0.as_record(),
            "m0_spread_sweep": {str(k): v for k, v in self.m0_spread_sweep.items()},
            "m0_conditioner_transitions": dict(self.m0_conditioner_transitions),
            "m0_vs_unconditioned": self.m0_vs_plain.as_record(),
            "overlay_is_a_tilt": dict(self.overlay_is_a_tilt),
            "controls": self.controls.as_record(),
            "trials": [t.as_record() for t in self.trials],
        }


def _sharpe_by_spread(
    label: str, exposures: pd.DataFrame, inputs: CampaignInputs,
    mask: np.ndarray,
) -> dict[float, float]:
    """Netto OOS Sharpe van één arm bij elke aangenomen half-spread."""
    return {
        float(bps): run_overlay_arm(
            f"{label}@{bps}", exposures, inputs.panels, mask=mask,
            risk_cfg=inputs.risk_cfg, impact=inputs.impact, venue=inputs.venue,
            half_spread_bps=bps, spread_source=inputs.spread_source,
            initial_equity=inputs.initial_equity,
            bars_per_year=inputs.bars_per_year).net_sharpe_oos
        for bps in inputs.spread_sweep_bps
    }


def _spread_delta(
    challenger: Mapping[float, float], baseline: Mapping[float, float],
    base_bps: float,
) -> tuple[dict[float, float], float | None]:
    """Sharpe-delta per spread, en waar een gemeten winst verdwijnt.

    BEIDE armen draaien bij ELKE spread; de baseline-sweep wordt één keer
    berekend en hergebruikt. De verleiding is om alleen de uitdager opnieuw te
    draaien, maar de spread raakt beide armen en niet even hard: de arm die meer
    handelt, betaalt meer. Wie dat negeert, meet de spread-gevoeligheid van één
    arm en noemt het een delta.

    Is er bij de basis-spread geen winst, dan is er niets dat kan verdwijnen en
    is het antwoord `None` -- niet-van-toepassing, niet-geschonden.
    """
    sweep = {bps: challenger[bps] - baseline[bps] for bps in challenger}
    if sweep[float(base_bps)] <= 0.0:
        return sweep, None
    vanish = [bps for bps in sorted(sweep) if sweep[bps] <= 0.0]
    return sweep, (float(vanish[0]) if vanish else None)


def _tilt_diagnostics(
    plain: ArmResult, m0: ArmResult,
) -> dict[str, float]:
    """Meet dat de overlay een TILT is en geen de-grossing.

    De soevereine laag herschaalt het boek naar zijn vol-target. Blijkt de
    bruto notional van de geconditioneerde arm vrijwel gelijk aan die van de
    ongeconditioneerde, dan is het risicobudget niet gekrompen maar verplaatst,
    en elke zin in het rapport die "risicoreductie" suggereert is onjuist.
    """
    return {
        "gross_notional_ratio": (
            m0.mean_gross_notional_oos / plain.mean_gross_notional_oos
            if plain.mean_gross_notional_oos else float("nan")),
        "turnover_ratio": (
            m0.turnover_notional_oos / plain.turnover_notional_oos
            if plain.turnover_notional_oos else float("nan")),
        "exposed_bars_ratio": (
            m0.n_exposed_bars_oos / plain.n_exposed_bars_oos
            if plain.n_exposed_bars_oos else float("nan")),
    }


def run_regime_campaign(inputs: CampaignInputs) -> H2CampaignResult:
    """Draai M0, de ongeconditioneerde referentie en elke gepre-registreerde trial."""
    require(
        bool(inputs.base_exposures.index.equals(inputs.panels.prices.index)),
        "De basisexposures staan niet op de tijdas van de markprijzen.",
        DataContractError, n_exposures=len(inputs.base_exposures),
        n_prices=len(inputs.panels.prices),
    )
    bar_mask, _ = oos_mask(len(inputs.returns), inputs.cv)
    # De returns beginnen één bar later dan de prijzen: de eerste bar draagt
    # geen return. Het masker wordt daarom op de PRIJS-as gezet, met False op
    # die eerste bar -- die valt per constructie buiten elk testvenster.
    mask = np.zeros(len(inputs.panels.prices), dtype=bool)
    offset = len(inputs.panels.prices) - len(inputs.returns)
    mask[offset:] = bar_mask

    def arm(label: str, multiplier: pd.DataFrame | None) -> ArmResult:
        exposures = (inputs.base_exposures if multiplier is None
                     else inputs.base_exposures * multiplier)
        return run_overlay_arm(
            label, exposures, inputs.panels, mask=mask,
            risk_cfg=inputs.risk_cfg, impact=inputs.impact, venue=inputs.venue,
            half_spread_bps=inputs.base_half_spread_bps,
            spread_source=inputs.spread_source,
            initial_equity=inputs.initial_equity,
            bars_per_year=inputs.bars_per_year)

    plain = arm(PLAIN_LABEL, None)
    m0_values = m0_multiplier(inputs.buckets, bar_mask).reindex(
        inputs.panels.prices.index).fillna(1.0)
    m0 = arm(M0_LABEL, m0_values)

    m0_transitions = {
        symbol: int(np.count_nonzero(
            np.diff(inputs.buckets[symbol].to_numpy()[bar_mask]) != 0.0))
        for symbol in inputs.buckets.columns}

    # De M0-sweep hangt niet van de uitdager af en wordt één keer gedraaid; hem
    # per trial herhalen is dertig identieke engine-runs.
    m0_sweep = _sharpe_by_spread(
        M0_LABEL, inputs.base_exposures * m0_values, inputs, mask)

    trials: list[TrialOutcome] = []
    controls: SharpeTestControls | None = None
    for spec in CONDITIONER_SPECS:
        conditioner = build_conditioner(
            spec, returns=inputs.returns, buckets=inputs.buckets,
            cv=inputs.cv, adequacy=inputs.adequacy, m2_cfg=inputs.m2_cfg)
        values = conditioner.values.reindex(
            inputs.panels.prices.index).fillna(1.0)
        challenger = arm(spec.label, values)
        dm = jobson_korkie_memmel(
            challenger.oos_returns, m0.oos_returns,
            bars_per_year=inputs.bars_per_year, alpha=inputs.alpha,
            alternative="a-better")
        if controls is None:
            controls = run_sharpe_controls(
                m0.oos_returns, challenger.oos_returns,
                bars_per_year=inputs.bars_per_year, alpha=inputs.alpha,
                expected_effect=inputs.expected_effect,
                minimum_detectable_effect=inputs.minimum_detectable_effect,
                target_power=inputs.target_power,
                n_replicates=inputs.n_control_replicates,
                mean_block_length=inputs.block_length, seed=inputs.seed)
        sweep, vanish = _spread_delta(
            _sharpe_by_spread(
                spec.label, inputs.base_exposures * values, inputs, mask),
            m0_sweep, inputs.base_half_spread_bps)
        delta = challenger.net_sharpe_oos - m0.net_sharpe_oos
        ratio = (challenger.turnover_notional_oos / m0.turnover_notional_oos
                 if m0.turnover_notional_oos else float("inf"))
        trials.append(TrialOutcome(
            label=spec.label, conditioner=conditioner, arm=challenger,
            net_sharpe_delta=delta, turnover_ratio=ratio,
            fees_delta=challenger.fees_oos - m0.fees_oos,
            spread_at_which_gain_vanishes=vanish, spread_sweep=sweep, dm=dm,
            verdict=judge_conditioner(
                rarest_state_obs_per_fold=conditioner.rarest_state_obs_per_fold,
                min_state_obs_required=float(
                    inputs.adequacy.hmm.min_obs_per_state_per_fold),
                net_sharpe_delta=delta, turnover_ratio=ratio,
                turnover_threshold=2.0,
                spread_at_which_gain_vanishes=vanish, spread_threshold=3.0,
                dm=dm, controls=controls)))

    require(
        controls is not None,
        "De campagne heeft geen enkele trial gedraaid; er is dan ook geen "
        "controle op de toets en dus geen oordeel.",
        DataContractError,
    )
    return H2CampaignResult(
        plain=plain, m0=m0, m0_spread_sweep=m0_sweep,
        m0_conditioner_transitions=m0_transitions,
        trials=tuple(trials), controls=controls,
        m0_vs_plain=jobson_korkie_memmel(
            m0.oos_returns, plain.oos_returns,
            bars_per_year=inputs.bars_per_year, alpha=inputs.alpha,
            alternative="a-better"),
        overlay_is_a_tilt=_tilt_diagnostics(plain, m0),
        n_planned_trials=len(CONDITIONER_SPECS),
    )
