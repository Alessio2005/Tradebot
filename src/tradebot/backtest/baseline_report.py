# src/tradebot/backtest/baseline_report.py
"""L11 - de statistische poort en het benchmarkrapport van de baseline.

Phase 3, stappen 10 en 11.

Twee gates, beide uit sectie 17.1 als ESSENTIAL aangemerkt:

* **Deflated Sharpe Ratio.** De DSR uit `backtest/metrics.py` is een RETAIN-item
  (de Bailey-Lopez de Prado dimensionaliteitsfix inclusief Euler-Mascheroni-term
  is correct en wordt niet herschreven). Hij wordt hier UITSLUITEND afgedwongen,
  met de EERLIJKE `M` uit de hypothese-ledger. Een `M` die alleen de trials van
  deze golf telt, maakt DSR structureel te optimistisch; dat is de meest
  voorkomende manier waarop de toets in de praktijk wordt ondermijnd.
* **Hansen's SPA.** Vergelijkt elke variant met de 1/N-referentie onder
  multiple-testing-correctie.

De uitkomsten worden gerapporteerd zoals ze zijn. Een stop-criterium dat bindt,
bindt; het bijstellen van een drempel na afloop is geen analyse maar fraude, en
de actieruimte van een pre-registratie kent die zet dan ook niet.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 16.1, 17, 17.1, 18.1, 24, 26.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry.preregistration import PreRegistration
from ..utils.failfast import DataContractError, require
from .baseline_runner import BaselineResult, returns_matrix
from .metrics import deflated_sharpe
from .spa import spa_test

__all__ = [
    "CriterionVerdict",
    "GateOutcome",
    "evaluate_stop_criteria",
    "execute_baseline_wave",
    "json_safe",
    "load_baseline_configs",
    "per_bar_sharpe",
    "run_dsr_gate",
    "run_spa_gate",
    "track_metrics",
]

#: Uitkomsten van een stop-criterium. `not_applicable` is bewust zichtbaar: een
#: criterium dat niet kan worden beoordeeld, mag nooit stilzwijgend als
#: "geslaagd" worden geteld.
_BINDING = "BINDING"
_CLEAR = "NOT_BINDING"
_NA = "NOT_APPLICABLE"


def per_bar_sharpe(returns: np.ndarray) -> float:
    """De NIET-geannualiseerde Sharpe.

    `deflated_sharpe` gebruikt de Mertens-variantie van de Sharpe-schatter, en
    die is gedefinieerd op de per-observatie Sharpe. Er een geannualiseerd getal
    in stoppen schaalt de teller wel en de noemer niet, waardoor de toets
    stilzwijgend een factor sqrt(bars_per_year) te streng of te soepel wordt.
    """
    clean = returns[np.isfinite(returns)]
    require(
        clean.size > 1,
        "Te weinig observaties voor een Sharpe.",
        DataContractError,
        n_obs=int(clean.size),
    )
    sd = float(np.std(clean, ddof=1))
    require(
        sd > 0.0,
        "Nul-variantie in de rendementsreeks; de Sharpe is niet gedefinieerd.",
        DataContractError,
    )
    return float(np.mean(clean)) / sd


def track_metrics(
    results: Mapping[str, BaselineResult], *, bars_per_year: int
) -> dict[str, dict[str, float]]:
    return {name: r.metrics(bars_per_year=bars_per_year) for name, r in results.items()}


def run_dsr_gate(
    result: BaselineResult, *, n_trials: int, dsr_alpha: float, bars_per_year: float
) -> dict[str, float | bool | str]:
    """DSR met de EERLIJKE M. Weigert te draaien zonder trial-telling.

    Er is geen default voor `n_trials`: ontbreekt hij, dan crasht de gate. Dat
    is de hele reden dat hij bestaat. Sinds fase 10 stap 4A geldt hetzelfde voor
    `bars_per_year`: `metrics.py` draagt geen annualisatie-default meer
    (MEASUREMENT_CONTRACT.md §10.1), en de waarde komt uit
    `conf/backtest/default.yaml`.

    De variantie van de trial-Sharpes is hier niet beschikbaar — `n_trials` is
    een telling, geen verdeling. §6 staat de benadering `1/n_obs` toe MITS zij
    in het artefact staat; vandaar `dsr_approximation` in de teruggave.
    """
    require(
        n_trials > 1,
        "De DSR-gate weigert te draaien zonder een eerlijk aantal trials M. Een "
        "te lage M maakt de toets structureel te optimistisch; er is bewust geen "
        "default en geen fallback.",
        DataContractError,
        n_trials=n_trials,
    )
    from scipy import stats

    net = result.net_returns.to_numpy(dtype="float64")
    clean = net[np.isfinite(net)]
    sr = per_bar_sharpe(clean)
    n_obs = int(clean.size)
    dsr = deflated_sharpe(
        sr,
        n_obs=n_obs,
        n_trials=int(n_trials),
        sr_variance=1.0 / n_obs,
        skew=float(stats.skew(clean)),
        kurtosis=float(stats.kurtosis(clean, fisher=False)),
        bars_per_year=float(bars_per_year),
        approximation="normal",
    )
    probability = float(dsr.dsr)
    return {
        "per_bar_sharpe": sr,
        "n_obs": float(n_obs),
        "n_trials": float(n_trials),
        "bars_per_year": dsr.bars_per_year,
        "t_years": dsr.t_years,
        "sr_variance": dsr.sr_variance,
        "dsr_approximation": dsr.approximation,
        "dsr_probability": probability,
        "dsr_p_value": 1.0 - probability,
        "passes": bool(probability >= 1.0 - dsr_alpha),
    }


def run_spa_gate(
    results: Mapping[str, BaselineResult],
    *,
    benchmark_track: str,
    candidate_tracks: Sequence[str],
    n_bootstrap: int,
    block_length: int,
    alpha: float,
) -> dict[str, Any]:
    """Hansen's SPA van elke kandidaat tegen de 1/N-referentie."""
    require(
        benchmark_track in results,
        "Onbekende benchmark-track voor de SPA-toets.",
        DataContractError,
        benchmark=benchmark_track,
        available=sorted(results),
    )
    require(
        len(candidate_tracks) > 0,
        "SPA zonder kandidaten; er valt niets te vergelijken.",
        DataContractError,
    )
    benchmark = results[benchmark_track].net_returns.to_numpy(dtype="float64")
    matrix = returns_matrix(results, list(candidate_tracks))
    outcome = spa_test(
        benchmark_returns=benchmark,
        strategy_returns_matrix=matrix,
        n_bootstrap=int(n_bootstrap),
        block_size=int(block_length),
        significance=float(alpha),
    )
    p_value = float(outcome["p_value_consistent"])
    best = int(outcome["best_strategy_idx"])
    return {
        "benchmark_track": benchmark_track,
        "candidate_tracks": list(candidate_tracks),
        "spa_p_value": p_value,
        "best_candidate": list(candidate_tracks)[best],
        "reject_null": bool(outcome["reject_null"]),
        "passes": bool(p_value < alpha),
    }


@dataclass(frozen=True)
class CriterionVerdict:
    """Wat een vooraf vastgelegd stop-criterium heeft gedaan."""

    name: str
    metric: str
    rule: str
    action: str
    measured: float
    status: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "metric": self.metric, "rule": self.rule,
            "action": self.action, "measured": self.measured,
            "status": self.status, "note": self.note,
        }


@dataclass(frozen=True)
class GateOutcome:
    """Het volledige oordeel over de primaire track."""

    verdicts: tuple[CriterionVerdict, ...]
    n_binding: int
    promoted: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdicts": [v.as_dict() for v in self.verdicts],
            "n_binding_stop_criteria": self.n_binding,
            "promoted_to_candidate": self.promoted,
        }


def evaluate_stop_criteria(
    prereg: PreRegistration, measured: Mapping[str, float]
) -> GateOutcome:
    """Toets de gemeten waarden tegen de PRE-GEREGISTREERDE criteria.

    Elk criterium krijgt een expliciete status. `NOT_APPLICABLE` bestaat omdat
    een criterium soms niet te beoordelen is - bijvoorbeeld de kosten-drempel
    wanneer er geen positieve bruto-edge is om te eroderen. Zo'n geval wordt
    ZICHTBAAR gemaakt en niet stilzwijgend als geslaagd geteld.
    """
    verdicts: list[CriterionVerdict] = []
    n_binding = 0
    for criterion in prereg.stop_criteria:
        if criterion.action == "promote":
            continue
        value = measured.get(criterion.metric, float("nan"))
        rule = f"{criterion.metric} {criterion.operator} {criterion.threshold}"
        if not np.isfinite(value):
            verdicts.append(CriterionVerdict(
                name=criterion.name, metric=criterion.metric, rule=rule,
                action=criterion.action, measured=float(value), status=_NA,
                note=("niet te beoordelen op deze meting; zie het rapport voor "
                      "de reden"),
            ))
            continue
        binds = criterion.binds(float(value))
        n_binding += int(binds)
        verdicts.append(CriterionVerdict(
            name=criterion.name, metric=criterion.metric, rule=rule,
            action=criterion.action, measured=float(value),
            status=_BINDING if binds else _CLEAR,
            note=criterion.action if binds else "",
        ))

    promote = [c for c in prereg.stop_criteria if c.action == "promote"]
    require(
        len(promote) == 1,
        "De pre-registratie moet precies een promotiecriterium bevatten.",
        DataContractError,
        found=[c.name for c in promote],
    )
    promoted = promote[0].binds(float(n_binding))
    verdicts.append(CriterionVerdict(
        name=promote[0].name, metric=promote[0].metric,
        rule=f"{promote[0].metric} {promote[0].operator} {promote[0].threshold}",
        action="promote", measured=float(n_binding),
        status=_CLEAR if promoted else _BINDING,
        note=("promotie naar CANDIDATE toegestaan" if promoted
              else "promotie geblokkeerd"),
    ))
    return GateOutcome(
        verdicts=tuple(verdicts), n_binding=n_binding, promoted=bool(promoted)
    )


def load_baseline_configs(root: Any) -> dict[str, Any]:
    """Laad elk configuratiedomein dat de baseline nodig heeft.

    Apart van de app zodat die binnen de 80-LOC-grens van R-6 blijft, en zodat
    de verzameling domeinen op EEN plek staat: een baseline die stiekem een
    ander configuratiebestand leest dan de pre-registratie citeert, is niet te
    reproduceren.
    """
    from ..schemas.config import (
        AlphaConfig,
        BacktestConfig,
        DataConfig,
        ExecutionConfig,
        ValidationConfig,
        VolatilityConfig,
        load_config,
    )

    domains = {
        "data": ("conf/data/default.yaml", DataConfig),
        "alpha": ("conf/model/alpha.yaml", AlphaConfig),
        "vol": ("conf/model/volatility.yaml", VolatilityConfig),
        "val": ("conf/validation/default.yaml", ValidationConfig),
        "exec": ("conf/execution/fees.yaml", ExecutionConfig),
        "bt": ("conf/backtest/default.yaml", BacktestConfig),
    }
    return {name: load_config(root / path, model)
            for name, (path, model) in domains.items()}


def json_safe(value: Any) -> Any:
    """Vervang niet-eindige floats door `None`.

    `json.dumps` schrijft standaard het letterlijke `NaN`, en dat is geen
    geldige JSON: een consument die het strikt parseert, crasht op een artefact
    dat wij als geldig beschouwen. `None` is bovendien het eerlijkere signaal -
    de waarde bestaat niet, hij is niet nul.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


#: Bruto-exposure waarop elke track wordt genormaliseerd. Een rekenkundige
#: conventie zodat de vergelijking over SIZING gaat en niet over hefboom; de
#: hefboomlimiet hoort in L7 (`conf/risk/`) en komt in Phase 4.
GROSS_TARGET = 1.0
#: De track waarop de pre-geregistreerde hypothese betrekking heeft.
PRIMARY_TRACK = "xs_momentum_risk_parity"
#: De absolute referentie uit sectie 13.1.
BENCHMARK_TRACK = "long_only_equal_weight"


def execute_baseline_wave(
    root: Any, prereg: PreRegistration, *, m_trials: int, git_sha: str
) -> tuple[dict[str, Any], GateOutcome]:
    """Draai de volledige baseline-golf en toets hem tegen de pre-registratie.

    Apart van `apps/run_baseline.py` zodat die binnen de 80-LOC-grens van R-6
    blijft. De app doet argumenten, artefact en ledger; de wetenschap staat hier.
    """
    from ..alpha.momentum import build_cross_sectional_momentum
    from ..data.pit_store import PitStore
    from ..features.base import DataRegister, load_certified_close_panel
    from .baseline_runner import CostModel, build_weight_tracks, run_baseline_tracks

    cfg = load_baseline_configs(root)
    gov = root / "artefacts" / "governance"
    panel = load_certified_close_panel(
        PitStore(root / cfg["data"].pit_store_root),
        DataRegister(gov / "data_hashes.json"),
        symbols=list(cfg["data"].symbols),
        granularity=prereg.granularity,
        asset_class="crypto",
    )
    unit = build_cross_sectional_momentum(cfg["alpha"])
    tracks, _ = build_weight_tracks(
        panel, unit, lam=cfg["vol"].ewma_lambda,
        vol_burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor,
        gross_target=GROSS_TARGET, git_sha=git_sha,
    )
    cost = CostModel(
        taker_fee_bps=cfg["exec"].taker_fee_bps,
        half_spread_bps=cfg["exec"].assumed_half_spread_bps,
        is_provisional=cfg["exec"].cost_assumption_is_provisional,
    )
    results = run_baseline_tracks(
        panel, tracks, cost=cost, train_bars=cfg["val"].train_bars,
        test_bars=cfg["val"].test_bars, embargo_bars=cfg["val"].embargo_bars,
        min_splits=cfg["val"].n_splits,
        warmup_bars=max(unit.burn_in_period, cfg["vol"].burn_in_bars),
    )
    metrics = track_metrics(results, bars_per_year=int(cfg["bt"].bars_per_year))
    dsr = {t: run_dsr_gate(r, n_trials=m_trials, dsr_alpha=cfg["val"].dsr_alpha,
                           bars_per_year=float(cfg["bt"].bars_per_year))
           for t, r in results.items()}
    spa = run_spa_gate(
        results, benchmark_track=BENCHMARK_TRACK,
        candidate_tracks=[t for t in results if t != BENCHMARK_TRACK],
        n_bootstrap=cfg["val"].spa_bootstrap_reps,
        block_length=cfg["val"].spa_block_length, alpha=cfg["val"].spa_alpha,
    )
    gross = metrics[PRIMARY_TRACK]["gross_sharpe"]
    net = metrics[PRIMARY_TRACK]["net_sharpe"]
    measured = {
        "net_oos_sharpe": net,
        "dsr_probability": float(dsr[PRIMARY_TRACK]["dsr_probability"]),
        "spa_p_value": float(spa["spa_p_value"]),
        # Niet gedefinieerd zonder positieve bruto-edge: er is dan niets om te
        # eroderen, en `no_edge_after_costs` bindt al. Zichtbaar als
        # NOT_APPLICABLE in plaats van stilzwijgend als geslaagd.
        "cost_drag_fraction": (gross - net) / gross if gross > 0.0 else float("nan"),
    }
    gate = evaluate_stop_criteria(prereg, measured)
    payload = {
        "preregistration_id": prereg.preregistration_id,
        "git_sha": git_sha,
        "data_hashes": dict(panel.data_hashes),
        "m_trials": m_trials,
        "cost_model": cost.as_dict(),
        "gross_target": GROSS_TARGET,
        "primary_track": PRIMARY_TRACK,
        "benchmark_track": BENCHMARK_TRACK,
        "warmup_bars": int(max(unit.burn_in_period, cfg["vol"].burn_in_bars)),
        "alpha_unit": unit.unit_id,
        "metrics": metrics,
        "dsr": dsr,
        "spa": spa,
        "measured": measured,
        "gate": gate.as_dict(),
    }
    return payload, gate


def risk_overlay_wave(root: Any, engine: Any, *, git_sha: str) -> dict[str, Any]:
    """Draai de Phase 3-baselinetracks OPNIEUW, met de L7-risicolaag in het pad.

    Phase 4, stap 10 / exit-criterium 8. De vergelijking is alleen geldig als
    beide kanten van precies dezelfde tracks, dezelfde folds en dezelfde
    kostenaanname komen - vandaar dat dit dezelfde bouwstenen gebruikt als
    `execute_baseline_wave` en niet een eigen reconstructie.

    De richting van de afhankelijkheid klopt: L10 (backtest) consumeert L7
    (risk). Andersom zou de risicolaag van de backtester afhangen, en dan is
    zij niet meer soeverein.
    """
    from ..alpha.momentum import build_cross_sectional_momentum
    from ..data.pit_store import PitStore
    from ..features.base import DataRegister, load_certified_close_panel
    from ..risk.stress_report import apply_risk_overlay
    from ..volatility.ewma import ewma_volatility_panel
    from .baseline_runner import CostModel, build_weight_tracks, run_baseline_tracks

    cfg = load_baseline_configs(root)
    gov = root / "artefacts" / "governance"
    panel = load_certified_close_panel(
        PitStore(root / cfg["data"].pit_store_root),
        DataRegister(gov / "data_hashes.json"),
        symbols=list(cfg["data"].symbols),
        granularity="1d",
        asset_class="crypto",
    )
    unit = build_cross_sectional_momentum(cfg["alpha"])
    tracks, _ = build_weight_tracks(
        panel, unit, lam=cfg["vol"].ewma_lambda,
        vol_burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor,
        gross_target=GROSS_TARGET, git_sha=git_sha,
    )
    cost = CostModel(
        taker_fee_bps=cfg["exec"].taker_fee_bps,
        half_spread_bps=cfg["exec"].assumed_half_spread_bps,
        is_provisional=cfg["exec"].cost_assumption_is_provisional,
    )
    results = run_baseline_tracks(
        panel, tracks, cost=cost, train_bars=cfg["val"].train_bars,
        test_bars=cfg["val"].test_bars, embargo_bars=cfg["val"].embargo_bars,
        min_splits=cfg["val"].n_splits,
        warmup_bars=max(unit.burn_in_period, cfg["vol"].burn_in_bars),
    )

    # De ex-ante sigma per asset komt uit L2, causaal, met dezelfde lambda en
    # burn-in als de baseline zelf gebruikte.
    sigma = ewma_volatility_panel(
        panel.values, lam=cfg["vol"].ewma_lambda,
        burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor,
    )
    asset_returns = panel.values.pct_change(fill_method=None)
    ppy = float(cfg["bt"].bars_per_year)

    # ADV bij benadering uit de dollarwaarde van het universum. De
    # liquiditeitslimiet is op deze schaal niet bindend (equity = 1 eenheid);
    # hij staat in het pad zodat hij mee wordt geverifieerd, niet omdat hij
    # hier iets afknijpt.
    adv = {str(c): 1e12 for c in panel.values.columns}

    overlays = {}
    for name, result in results.items():
        idx = result.weights.index
        overlays[name] = apply_risk_overlay(
            name,
            weights=result.weights,
            asset_returns=asset_returns.loc[idx],
            sigma_hat=sigma.loc[idx],
            engine=engine,
            adv_usd=adv,
            cost_per_side=cost.per_side,
            periods_per_year=ppy,
            baseline_returns=result.net_returns,
        ).as_record()

    return {
        "git_sha": git_sha,
        "data_hashes": dict(panel.data_hashes),
        "cost_model": cost.as_dict(),
        "gross_target": GROSS_TARGET,
        "alpha_unit": unit.unit_id,
        "periods_per_year": ppy,
        "risk_config_hash": engine.config_hash,
        "risk_audit_header": engine.audit_header(),
        "overlays": overlays,
    }
