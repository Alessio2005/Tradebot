# src/tradebot/validation/meta_label_campaign.py
"""H3 — de campagne: zes specs, één oordeel. Stappen 12 en 13.

DE VOLGORDE VAN DE STOP-CRITERIA IS HIER HET HELE VERHAAL
==========================================================
1. **De negatieve controle gaat eerst en falsificeert de RUN, niet de
   hypothese.** Hetzelfde model op gerandomiseerde labels hoort een AUC rond
   0,50 te geven. Ligt hij op 0,55 of hoger, dan lekt er informatie — een
   purging die niet purget, een feature die de toekomst raakt — en is elk ander
   getal uit deze run waardeloos. Dat is de enige reden waarom dit criterium
   vóór de adequaatheid staat: er valt niets te descopen aan een meting die
   niets meet.
2. **Adequaatheid.** Zakt het EFFECTIEVE aantal events per fold onder 100, dan
   is het nominale aantal een illusie van steekproefgrootte.
3. **De AUC-drempel, op de CONSERVATIEVE ondergrens.** §24 legt 0,58 vast. De
   pre-registratie eist dat de ONDERGRENS van het conservatieve interval daar
   boven ligt — niet de puntschatting, en niet de ondergrens van het
   optimistische interval.
4. **De economische toets.** Een AUC boven 0,58 is een statistisch resultaat.
   Het gefilterde signaal gaat door de authoritative engine met volledige
   kosten; voegt het filter daar netto niets toe, dan wordt het niet
   gepromoveerd hoe goed de AUC ook is.

TWEE EFFECTIEVE STEEKPROEFGROOTTES, EN WAAROM ALLEBEI
======================================================
Het nominale aantal labels overschat de informatie op twee manieren tegelijk.
Met een horizon van tien bars delen buren negen van hun tien toekomstige bars
(AFML hoofdstuk 4), en zes perpetuals met een gemeten gemiddelde correlatie van
0,7485 zijn 1,27 onafhankelijke reeksen waard en geen zes.

De pre-registratie legde vóór de run vast dat het interval op BEIDE wordt
gerapporteerd en dat het oordeel op de conservatieve valt. Het nominale interval
staat er uitsluitend bij om te laten zien hoeveel te smal het is — dat is het
interval dat een pipeline zonder uniqueness-correctie zou produceren, en het
verschil is de reden dat meta-labeling-resultaten zo vaak niet repliceren.

Ref: Hanley & McNeil (1982); AFML hoofdstuk 3, 4 en 7; pre-registratie
`56395fa2013768014c0c915edf346770`; fase-opdracht stappen 12 en 13.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from ..backtest.regime_overlay import ArmResult, MarketPanels, run_overlay_arm
from ..cv.walk_forward import WalkForwardCV
from ..regime.conditioning import oos_mask
from ..schemas.config import MetaLabelConfig
from ..train.meta_label import (
    META_LABEL_GRID,
    FoldPredictions,
    MetaLabelDataset,
    shuffled_targets,
    walk_forward_predictions,
)
from ..utils.failfast import DataContractError, require
from .feature_importance import (
    ImportanceResult,
    mdi_importance,
    pooled_predictions,
    sfi_importance,
    weighted_auc,
)

__all__ = [
    "H3CampaignResult",
    "H3Verdict",
    "MetaLabelInputs",
    "MetaTrialOutcome",
    "auc_confidence_interval",
    "judge_meta_label",
    "operating_point",
    "run_meta_label_campaign",
]

VerdictStatus = Literal["PROMOTED", "ARCHIVED", "DESCOPED", "FALSIFIED"]

_LEAK = "negative_control_leaks"
_EVENTS = "events_below_adequacy"
_AUC = "auc_below_threshold"
_ENGINE = "filter_adds_no_net_value_through_the_engine"

#: De arm zonder filter. Referentie, geen deelnemer: het primaire signaal is
#: Phase 3 en wordt hier niet opnieuw beoordeeld.
UNFILTERED_LABEL = "unfiltered"


def auc_confidence_interval(
    auc: float, n_effective: float, alpha: float,
) -> tuple[float, float, float]:
    """Hanley-McNeil-interval om een AUC, bij een EFFECTIEVE steekproefgrootte.

    De standaardfout hangt van de AUC zelf af::

        Q1 = A / (2 - A)          Q2 = 2 A^2 / (1 + A)
        SE = sqrt([A(1-A) + (n1-1)(Q1 - A^2) + (n0-1)(Q2 - A^2)] / (n1 n0))

    `n_effective` is het TOTAAL; positieven en negatieven worden gelijk verdeeld.
    Dat is de conservatieve keuze wanneer de klassebalans dicht bij 50/50 ligt,
    en zij is dezelfde die de power-analyse in de pre-registratie maakte — een
    interval dat op een andere verdeling staat dan de MDE, is niet met die MDE
    te vergelijken.
    """
    require(
        0.0 < auc < 1.0,
        "Een AUC van exact 0 of 1 heeft geen Hanley-McNeil-interval; de "
        "formule deelt dan door een variantie van nul.",
        DataContractError, auc=auc,
    )
    require(
        n_effective >= 4.0,
        "Een betrouwbaarheidsinterval op minder dan vier effectieve "
        "waarnemingen is geen interval.",
        DataContractError, n_effective=n_effective,
    )
    n_positive = n_negative = n_effective / 2.0
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (n_positive - 1.0) * (q1 - auc**2)
        + (n_negative - 1.0) * (q2 - auc**2)
    ) / (n_positive * n_negative)
    se = math.sqrt(max(variance, 0.0))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return auc - z * se, auc + z * se, se


def operating_point(
    target: np.ndarray, probability: np.ndarray, weight: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Precision, recall en hoeveel er wordt weggefilterd op dit werkpunt.

    `kept_fraction` staat er omdat het de grootheid is die de economische toets
    bepaalt: een filter dat niets wegneemt kan niets toevoegen, en een filter
    dat bijna alles wegneemt handelt te weinig om een Sharpe op te meten.
    """
    keep = probability >= threshold
    positive = target == 1
    kept_weight = float(weight[keep].sum())
    total_weight = float(weight.sum())
    true_positive = float(weight[keep & positive].sum())
    return {
        "threshold": float(threshold),
        "precision": true_positive / kept_weight if kept_weight > 0.0
        else float("nan"),
        "recall": true_positive / float(weight[positive].sum())
        if positive.any() else float("nan"),
        "kept_fraction": kept_weight / total_weight if total_weight > 0.0
        else float("nan"),
        "base_rate": float(weight[positive].sum()) / total_weight
        if total_weight > 0.0 else float("nan"),
        "n_kept": int(keep.sum()),
        "n_total": int(keep.size),
    }


@dataclass(frozen=True)
class H3Verdict:
    status: VerdictStatus
    binding: tuple[str, ...]
    rationale: str

    def as_record(self) -> dict[str, Any]:
        return {"status": self.status, "binding": list(self.binding),
                "rationale": self.rationale}


#: Plaatshouder tot de negatieve controle is gedraaid: het oordeel over ELKE
#: spec hangt ervan af, en een verdict dat al bestaat voordat die controle
#: binnen is, zou de volgorde van de pre-registratie omdraaien.
_PENDING = H3Verdict("ARCHIVED", (), "oordeel nog niet geveld")


def judge_meta_label(
    *,
    shuffled_auc: float,
    leak_threshold: float,
    min_effective_events: float,
    events_threshold: float,
    auc: float,
    auc_lower_conservative: float,
    auc_lower_nominal: float,
    auc_threshold: float,
    net_sharpe_delta: float,
) -> H3Verdict:
    """De vier stop-criteria, in de volgorde van de pre-registratie."""
    if shuffled_auc >= leak_threshold:
        return H3Verdict(
            status="FALSIFIED", binding=(_LEAK,),
            rationale=(
                f"De negatieve controle geeft een AUC van {shuffled_auc:.4f} op "
                f"GERANDOMISEERDE labels, tegen een grens van "
                f"{leak_threshold:.2f}. Er lekt informatie in de pipeline, en "
                f"dan is elk ander getal uit deze run waardeloos. Dit "
                f"falsificeert de RUN en niet de hypothese."),
        )
    if min_effective_events < events_threshold:
        return H3Verdict(
            status="DESCOPED", binding=(_EVENTS,),
            rationale=(
                f"Het effectieve aantal events per fold zakt naar "
                f"{min_effective_events:.1f} tegen een eis van "
                f"{events_threshold:.0f}. Het nominale aantal is dan een "
                f"illusie van steekproefgrootte; `UNPROVEN -- insufficient "
                f"data`, en er valt niets te falsifiëren."),
        )
    binding: list[str] = []
    if auc_lower_conservative <= auc_threshold:
        binding.append(_AUC)
    if net_sharpe_delta <= 0.0:
        binding.append(_ENGINE)
    if not binding:
        return H3Verdict(
            status="PROMOTED", binding=(),
            rationale=(
                f"OOS-AUC {auc:.4f} met een conservatieve ondergrens van "
                f"{auc_lower_conservative:.4f} boven {auc_threshold:.2f}, en "
                f"het gefilterde signaal levert door de engine een netto "
                f"Sharpe-delta van {net_sharpe_delta:+.4f}. Geen enkel "
                f"stop-criterium bindt; de Phase 2-poort beslist."),
        )
    return H3Verdict(
        status="ARCHIVED", binding=tuple(binding),
        rationale=(
            f"OOS-AUC {auc:.4f}; conservatieve ondergrens "
            f"{auc_lower_conservative:.4f} en nominale ondergrens "
            f"{auc_lower_nominal:.4f}, tegen de drempel {auc_threshold:.2f}. "
            f"Netto Sharpe-delta door de engine {net_sharpe_delta:+.4f}. "
            f"CatBoost blijft `ARCHIVED` zoals §24 vastlegt; de drempel wordt "
            f"niet verlaagd en er wordt geen ander werkpunt gezocht waarop hij "
            f"wél wordt gehaald."),
    )


@dataclass(frozen=True)
class MetaTrialOutcome:
    label: str
    auc: float
    folds: tuple[Mapping[str, Any], ...]
    interval_nominal: tuple[float, float, float]
    interval_conservative: tuple[float, float, float]
    operating: Mapping[str, float]
    curve: tuple[Mapping[str, float], ...]
    arm: ArmResult
    net_sharpe_delta: float
    verdict: H3Verdict

    def as_record(self) -> dict[str, Any]:
        return {
            "label": self.label, "oos_auc": self.auc,
            "interval_nominal": list(self.interval_nominal),
            "interval_conservative": list(self.interval_conservative),
            "operating_point": dict(self.operating),
            "operating_curve": [dict(p) for p in self.curve],
            "arm": self.arm.as_record(),
            "net_sharpe_delta": self.net_sharpe_delta,
            "folds": [dict(f) for f in self.folds],
            "verdict": self.verdict.as_record(),
        }


@dataclass(frozen=True)
class MetaLabelInputs:
    dataset: MetaLabelDataset
    base_exposures: pd.DataFrame
    panels: MarketPanels
    cv: WalkForwardCV
    cfg: MetaLabelConfig
    risk_cfg: Any
    impact: Any
    venue: Any
    spread_source: str
    initial_equity: float
    bars_per_year: float
    half_spread_bps: float
    embargo_bars: int
    alpha: float
    auc_threshold: float
    leak_threshold: float
    events_threshold: float
    #: Aantal onafhankelijke reeksen dat zes gecorreleerde perpetuals waard
    #: zijn. Komt uit de bevroren pre-registratie, niet uit deze run.
    effective_independent_series: float
    n_symbols: int
    curve_thresholds: Sequence[float]
    #: `min_effective_events_per_fold` ZOALS DE DATA ADEQUACY GATE HEM MEET:
    #: gepurgede TRAINevents per fold, uniqueness-gecorrigeerd, uit
    #: `artefacts/governance/phase6_data_adequacy.json`.
    #:
    #: Stop-criterium 2 van de pre-registratie zegt letterlijk *"De Data
    #: Adequacy Gate meet het effectieve aantal labels per fold"*, en die poort
    #: heeft een implementatie en een artefact. Hem hier opnieuw berekenen zou
    #: een tweede definitie opleveren van een grootheid die er al een heeft, en
    #: dat is precies hoe een poort stilzwijgend van mening verandert.
    #:
    #: Het TESTfold-aantal is een andere grootheid met een andere rol: het
    #: bepaalt de precisie van de AUC en niet of er gefit mag worden. Het wordt
    #: gemeten en gerapporteerd, maar het bindt dit criterium niet.
    min_effective_events_per_fold: float


@dataclass(frozen=True)
class H3CampaignResult:
    unfiltered: ArmResult
    trials: tuple[MetaTrialOutcome, ...]
    shuffled_auc: float
    shuffled_auc_replicates: tuple[float, ...]
    mdi: ImportanceResult
    sfi: ImportanceResult
    best_label: str
    n_events: int
    positive_ratio: float
    uniqueness_ratio: float
    #: Zoals de Data Adequacy Gate hem meet: gepurgede TRAINevents per fold.
    min_effective_events_per_fold: float
    #: Wat deze run zelf meet op de TESTfolds. Bepaalt de precisie van de AUC,
    #: niet of er gefit mag worden.
    min_effective_test_events_per_fold: float
    effective_n_nominal: float
    effective_n_conservative: float
    n_planned_trials: int

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for trial in self.trials:
            out[trial.verdict.status] = out.get(trial.verdict.status, 0) + 1
        return out

    def as_record(self) -> dict[str, Any]:
        return {
            "n_trials": len(self.trials),
            "n_planned_trials": self.n_planned_trials,
            "by_status": self.by_status(),
            "unfiltered": self.unfiltered.as_record(),
            "shuffled_auc": self.shuffled_auc,
            "shuffled_auc_replicates": list(self.shuffled_auc_replicates),
            "n_events": self.n_events,
            "positive_ratio": self.positive_ratio,
            "uniqueness_ratio": self.uniqueness_ratio,
            "min_effective_events_per_fold": self.min_effective_events_per_fold,
            "min_effective_test_events_per_fold":
                self.min_effective_test_events_per_fold,
            "effective_n_nominal": self.effective_n_nominal,
            "effective_n_conservative": self.effective_n_conservative,
            "best_label": self.best_label,
            "mdi": self.mdi.as_record(),
            "sfi": self.sfi.as_record(),
            "trials": [t.as_record() for t in self.trials],
        }


def _filter_multiplier(
    dataset: MetaLabelDataset,
    folds: Sequence[FoldPredictions],
    index: pd.Index,
    columns: Sequence[str],
    threshold: float,
) -> pd.DataFrame:
    """1 waar het filter de trade doorlaat, 0 waar het hem tegenhoudt.

    Buiten de gescoorde events staat 1: daar heeft het filter geen oordeel, en
    een 0 zou een BEWERING zijn in plaats van een onthouding. Dat is dezelfde
    regel als bij de regime-overlay van H2.
    """
    values = pd.DataFrame(1.0, index=index, columns=list(columns),
                          dtype="float64")
    for fold in folds:
        blocked = fold.row_index[fold.probability < threshold]
        for row in blocked:
            values.iloc[int(dataset.event_bar[row]),
                        values.columns.get_loc(str(dataset.symbol[row]))] = 0.0
    return values


def run_meta_label_campaign(inputs: MetaLabelInputs) -> H3CampaignResult:
    """Zes specs door de purged walk-forward en door de engine."""
    dataset, cfg = inputs.dataset, inputs.cfg
    n_bars = len(inputs.panels.prices)
    bar_mask, folds_geometry = oos_mask(n_bars, inputs.cv)
    require(
        len(folds_geometry) > 0,
        "Geen enkele walk-forward fold; er valt niets out-of-sample te scoren.",
        DataContractError,
    )

    def arm(label: str, multiplier: pd.DataFrame | None) -> ArmResult:
        exposures = (inputs.base_exposures if multiplier is None
                     else inputs.base_exposures * multiplier)
        return run_overlay_arm(
            label, exposures, inputs.panels, mask=bar_mask,
            risk_cfg=inputs.risk_cfg, impact=inputs.impact, venue=inputs.venue,
            half_spread_bps=inputs.half_spread_bps,
            spread_source=inputs.spread_source,
            initial_equity=inputs.initial_equity,
            bars_per_year=inputs.bars_per_year)

    unfiltered = arm(UNFILTERED_LABEL, None)

    uniqueness_ratio = float(dataset.uniqueness.mean())
    cross_section = inputs.effective_independent_series / float(inputs.n_symbols)

    results: list[MetaTrialOutcome] = []
    per_spec_folds: dict[str, list[FoldPredictions]] = {}
    for spec in META_LABEL_GRID:
        fold_runs = walk_forward_predictions(
            dataset, inputs.cv, spec, cfg, n_bars=n_bars,
            embargo_bars=inputs.embargo_bars)
        per_spec_folds[spec.label] = fold_runs
        pooled = pooled_predictions(fold_runs)
        auc = weighted_auc(
            pooled["target"], pooled["probability"], pooled["uniqueness"])
        nominal = float(pooled["target"].size)
        conservative = nominal * uniqueness_ratio * cross_section
        multiplier = _filter_multiplier(
            dataset, fold_runs, inputs.panels.prices.index,
            list(inputs.panels.prices.columns), cfg.probability_threshold)
        filtered = arm(spec.label, multiplier)
        delta = filtered.net_sharpe_oos - unfiltered.net_sharpe_oos
        results.append(MetaTrialOutcome(
            label=spec.label, auc=auc,
            folds=tuple(f.as_record() for f in fold_runs),
            interval_nominal=auc_confidence_interval(
                auc, nominal, inputs.alpha),
            interval_conservative=auc_confidence_interval(
                auc, conservative, inputs.alpha),
            operating=operating_point(
                pooled["target"], pooled["probability"], pooled["uniqueness"],
                cfg.probability_threshold),
            curve=tuple(
                operating_point(pooled["target"], pooled["probability"],
                                pooled["uniqueness"], t)
                for t in inputs.curve_thresholds),
            arm=filtered, net_sharpe_delta=delta,
            verdict=_PENDING))

    # De negatieve controle draait op de BESTE spec: lekt de pipeline, dan lekt
    # zij daar het hardst, en een controle op de zwakste spec zou een lek
    # kunnen verbergen achter een model dat toch al niets vond.
    best = max(results, key=lambda r: r.auc)
    replicates: list[float] = []
    for shuffled in shuffled_targets(
            dataset, seed=cfg.seed, n_replicates=cfg.n_shuffle_replicates):
        control_folds = walk_forward_predictions(
            dataset, inputs.cv,
            next(s for s in META_LABEL_GRID if s.label == best.label), cfg,
            n_bars=n_bars, embargo_bars=inputs.embargo_bars, target=shuffled)
        control = pooled_predictions(control_folds)
        replicates.append(weighted_auc(
            control["target"], control["probability"], control["uniqueness"]))
    shuffled_auc = float(np.max(replicates))

    # Wat de TESTfolds dragen. Dit is de grootheid achter de precisie van de
    # AUC; de poort van stop-criterium 2 staat op de TRAINfolds en komt uit de
    # Data Adequacy Gate.
    min_effective_test = min(
        float(f.uniqueness.sum())
        for f in per_spec_folds[best.label])

    judged: list[MetaTrialOutcome] = []
    for trial in results:
        judged.append(MetaTrialOutcome(
            label=trial.label, auc=trial.auc, folds=trial.folds,
            interval_nominal=trial.interval_nominal,
            interval_conservative=trial.interval_conservative,
            operating=trial.operating, curve=trial.curve, arm=trial.arm,
            net_sharpe_delta=trial.net_sharpe_delta,
            verdict=judge_meta_label(
                shuffled_auc=shuffled_auc,
                leak_threshold=inputs.leak_threshold,
                min_effective_events=inputs.min_effective_events_per_fold,
                events_threshold=inputs.events_threshold,
                auc=trial.auc,
                auc_lower_conservative=trial.interval_conservative[0],
                auc_lower_nominal=trial.interval_nominal[0],
                auc_threshold=inputs.auc_threshold,
                net_sharpe_delta=trial.net_sharpe_delta)))

    best_spec = next(s for s in META_LABEL_GRID if s.label == best.label)
    pooled_n = float(sum(f["n_test"] for f in best.folds))
    return H3CampaignResult(
        unfiltered=unfiltered, trials=tuple(judged),
        shuffled_auc=shuffled_auc,
        shuffled_auc_replicates=tuple(replicates),
        mdi=mdi_importance(per_spec_folds[best.label],
                           list(dataset.features.columns)),
        sfi=sfi_importance(dataset, inputs.cv, best_spec, cfg, n_bars=n_bars,
                           embargo_bars=inputs.embargo_bars),
        best_label=best.label, n_events=len(dataset),
        positive_ratio=dataset.positive_ratio,
        uniqueness_ratio=uniqueness_ratio,
        min_effective_events_per_fold=inputs.min_effective_events_per_fold,
        min_effective_test_events_per_fold=min_effective_test,
        effective_n_nominal=pooled_n,
        effective_n_conservative=pooled_n * uniqueness_ratio * cross_section,
        n_planned_trials=len(META_LABEL_GRID),
    )
