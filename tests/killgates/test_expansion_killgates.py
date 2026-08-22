"""Kill-gate automation for the multi-asset expansion (docs/EXPANSION_RESEARCH_2026-08-10.md §3B).

Two jobs:

1.  **Evaluate a unit** against a pre-registered gate spec, from a metrics JSON
    artefact.  A unit that misses any criterion fails the gate — no partial
    credit, no rounding (the ``eq_strev_1m`` 0.39 precedent).
2.  **Validate the gate spec itself.**  A gate set can be internally
    infeasible: no strategy in the universe of strategies satisfies it.  The
    v2.0 research prompt shipped such a set (Sharpe >= 0.60 AND MaxDD <= 18%
    AND Calmar >= 3.3), which would have burned a week before anyone noticed.
    ``check_consistency`` catches that class of error at pre-registration time.

Run:  ``python -m pytest tests/killgates -m killgate -v``
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pytest

from tradebot.backtest.dd_shape import calmar_ceiling, max_dd_over_vol_quantile

pytestmark = pytest.mark.killgate

ARTEFACT_ROOT = Path(os.environ.get("KILLGATE_ARTEFACTS", "artefacts/killgates"))


# --------------------------------------------------------------------------- #
# Gate specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateSpec:
    """Pre-registered thresholds for one gate.  ``None`` = criterion not used.

    All Sharpe/vol/drawdown figures are annualised and NET of the registered
    cost model.  Drawdown is a positive fraction (0.18 == 18%).
    """

    name: str
    min_net_sharpe: float | None = None
    max_drawdown: float | None = None
    min_calmar: float | None = None
    min_years_positive_frac: float | None = None
    max_dd_over_vol: float | None = None
    # W28: skill- and horizon-aware drawdown shape. The cap is not a constant;
    # it is the ``dd_shape_quantile`` quantile of the maximum drawdown a
    # strategy with THIS unit's realised Sharpe would produce over THIS unit's
    # sample length. False-rejection budget = 1 - quantile, by construction.
    dd_shape_quantile: float | None = None
    # Registered evaluation horizon in years — used by check_consistency and as
    # the fallback when a metrics artefact omits ``sample_years``.
    sample_years: float | None = None
    # G4 / faithfulness
    max_alpha_p: float | None = None
    # out-of-sample
    min_wf_sharpe: float | None = None
    max_sharpe_decay: float | None = None
    min_bootstrap_p_positive: float | None = None
    max_abs_rho_vs_book: float | None = None
    # tradeability / execution
    max_sizing_error: float | None = None
    max_slippage_ratio: float | None = None
    min_fill_rate: float | None = None


# The set proposed in "Multi-Asset Research Prompt v2.0" §3B Kill-Gate #1.
# Kept ONLY as the regression case for check_consistency — do not use it.
PROMPT_V2_GATE_B1 = GateSpec(
    name="prompt_v2_kg1_setup_b",
    min_net_sharpe=0.60,
    max_drawdown=0.18,
    min_calmar=3.3,
    min_years_positive_frac=0.52,
)

# Mandate-consistent replacements (research doc §3B), with the drawdown-shape
# criterion re-derived in Wave 28 step 0.3b BEFORE any new unit was measured
# (SETUP_B_ML_PROMPT §5.3 order: re-derive first, then measure).
#
# What changed and why — the constants ``max_dd_over_vol=2.5`` and
# ``min_calmar=0.25`` were calibrated from ``E[MaxDD] = sigma^2/(2*mu)``. That
# formula is the mean drawdown at a RANDOM TIME, not the expected MAXIMUM
# (see tradebot.backtest.dd_shape). Measured against its own Sharpe floor the
# old pair rejected **99.7%** of strategies that genuinely have Sharpe 0.40
# over a 22-year sample: it was not a gate, it was a wall.
#
# Replaced by ONE criterion at a stated false-rejection budget. The Calmar
# floor is dropped, not relaxed: Calmar = (S - sigma/2) / dd_over_vol is a
# deterministic function of the Sharpe floor and the shape cap, so keeping all
# three triple-counted one piece of evidence.
#
# This does NOT reopen cm_tsmom. Its registered reopening condition
# (SETUP_B_ML_PROMPT §1.1b) requires a re-derived threshold AND a G4-strict
# pass; G4-strict was t=1.75 < 2.0 and is untouched here, so cm_tsmom stays
# ARCHIVED on KG-B2. Encoded in test_cm_tsmom_stays_archived_on_g4_strict.
KG_B1_INSAMPLE = GateSpec(
    name="KG-B1 in-sample",
    min_net_sharpe=0.40,          # mandate lat; eq_strev_1m archived at 0.39
    min_years_positive_frac=0.60,
    dd_shape_quantile=0.95,
    sample_years=22.6,            # the registered cross-asset panel, 2004-2026
)
KG_B2_ALPHA = GateSpec(name="KG-B2 residual alpha", max_alpha_p=0.05)
KG_B3_OOS = GateSpec(
    name="KG-B3 out-of-sample",
    min_wf_sharpe=0.30,
    max_sharpe_decay=0.40,
    min_bootstrap_p_positive=0.75,
    max_abs_rho_vs_book=0.30,
)
KG_B4_TRADEABILITY = GateSpec(name="KG-B4 tradeability", max_sizing_error=0.20)
KG_B5_PAPER = GateSpec(
    name="KG-B5 paper", max_slippage_ratio=0.25, min_fill_rate=0.95
)

ALL_GATES = (KG_B1_INSAMPLE, KG_B2_ALPHA, KG_B3_OOS, KG_B4_TRADEABILITY, KG_B5_PAPER)


# --------------------------------------------------------------------------- #
# Spec consistency
# --------------------------------------------------------------------------- #
# ``calmar_ceiling`` now lives in tradebot.backtest.dd_shape and takes a
# HORIZON. The local ``2*S**2`` version was wrong (it used the stationary
# drawdown as if it were the maximum) and is the reason this very checker
# green-lit an unsatisfiable KG-B1. Kept out of this file on purpose: R-2, and
# so the correction cannot be quietly re-forked.
_DEFAULT_HORIZON_YEARS = 22.6


def check_consistency(spec: GateSpec) -> list[str]:
    """Return the list of internal contradictions in ``spec`` (empty == sound)."""
    problems: list[str] = []
    years = spec.sample_years or _DEFAULT_HORIZON_YEARS

    if spec.min_calmar is not None and spec.min_net_sharpe is not None:
        ceiling = calmar_ceiling(spec.min_net_sharpe, years)
        if spec.min_calmar > ceiling:
            problems.append(
                f"{spec.name}: Calmar floor {spec.min_calmar:.2f} exceeds the "
                f"ceiling {ceiling:.2f} implied by Sharpe "
                f"{spec.min_net_sharpe:.2f} over {years:.1f}y — unsatisfiable "
                f"at any leverage"
            )

    if spec.dd_shape_quantile is not None and not 0.5 < spec.dd_shape_quantile < 1.0:
        problems.append(
            f"{spec.name}: dd_shape_quantile must lie in (0.5, 1.0), "
            f"got {spec.dd_shape_quantile}"
        )
    if spec.dd_shape_quantile is not None and spec.max_dd_over_vol is not None:
        problems.append(
            f"{spec.name}: dd_shape_quantile and max_dd_over_vol both set — "
            f"the constant cap is the thing the quantile replaces"
        )

    if (
        spec.min_calmar is not None
        and spec.max_drawdown is not None
        and spec.min_net_sharpe is not None
    ):
        # Calmar >= C and MaxDD <= D pin CAGR >= C*D, hence vol >= C*D/S.
        implied_cagr = spec.min_calmar * spec.max_drawdown
        implied_vol = implied_cagr / spec.min_net_sharpe
        if implied_vol > 4.0 * spec.max_drawdown:
            problems.append(
                f"{spec.name}: Calmar {spec.min_calmar:.2f} with MaxDD cap "
                f"{spec.max_drawdown:.0%} forces CAGR >= {implied_cagr:.0%} and "
                f"vol >= {implied_vol:.0%} at Sharpe {spec.min_net_sharpe:.2f}; a "
                f"book at {implied_vol:.0%} vol cannot hold a {spec.max_drawdown:.0%} "
                f"drawdown"
            )

    if spec.max_sharpe_decay is not None and not 0.0 < spec.max_sharpe_decay < 1.0:
        problems.append(f"{spec.name}: sharpe decay cap must lie in (0, 1)")

    for f in fields(spec):
        if f.name.startswith("min_") and f.name.endswith(
            ("_frac", "_p_positive", "_rate")
        ):
            v = getattr(spec, f.name)
            if v is not None and not 0.0 <= v <= 1.0:
                problems.append(f"{spec.name}: {f.name}={v} is not a fraction")

    return problems


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    # (spec field,               metrics key,             direction)
    ("min_net_sharpe", "net_sharpe", "ge"),
    ("max_drawdown", "max_drawdown", "le"),
    ("min_calmar", "calmar", "ge"),
    ("min_years_positive_frac", "years_positive_frac", "ge"),
    ("max_dd_over_vol", "dd_over_vol", "le"),
    ("max_alpha_p", "alpha_p", "le"),
    ("min_wf_sharpe", "wf_sharpe", "ge"),
    ("max_sharpe_decay", "sharpe_decay", "le"),
    ("min_bootstrap_p_positive", "bootstrap_p_positive", "ge"),
    ("max_abs_rho_vs_book", "abs_rho_vs_book", "le"),
    ("max_sizing_error", "sizing_error", "le"),
    ("max_slippage_ratio", "slippage_ratio", "le"),
    ("min_fill_rate", "fill_rate", "ge"),
)


def evaluate(metrics: dict, spec: GateSpec) -> list[str]:
    """Return the criteria of ``spec`` that ``metrics`` fails.

    A criterion whose metric is absent is *not* silently passed — it is
    reported as missing.  Gates are decided on measured evidence only.
    """
    failures: list[str] = []
    for spec_field, key, direction in _COMPARISONS:
        threshold = getattr(spec, spec_field)
        if threshold is None:
            continue
        if key not in metrics or metrics[key] is None:
            failures.append(f"{spec.name}: metric '{key}' missing — cannot decide")
            continue
        value = float(metrics[key])
        ok = value >= threshold if direction == "ge" else value <= threshold
        if not ok:
            arrow = ">=" if direction == "ge" else "<="
            failures.append(
                f"{spec.name}: {key}={value:.4g} violates {key} {arrow} {threshold:g}"
            )

    if spec.dd_shape_quantile is not None:
        failures.extend(_check_dd_shape(metrics, spec))
    return failures


def _check_dd_shape(metrics: dict, spec: GateSpec) -> list[str]:
    """Skill- and horizon-aware drawdown-shape criterion (W28).

    The cap is derived from the unit's OWN realised Sharpe and sample length,
    so it asks "is this drawdown pathological given the skill on display",
    not "is this drawdown below a constant somebody picked".
    """
    missing = [k for k in ("dd_over_vol", "net_sharpe") if metrics.get(k) is None]
    if missing:
        return [
            f"{spec.name}: metric '{k}' missing — cannot decide" for k in missing
        ]

    years = metrics.get("sample_years") or spec.sample_years
    if years is None:
        return [f"{spec.name}: metric 'sample_years' missing — cannot decide"]

    sharpe = float(metrics["net_sharpe"])
    ann_vol = float(metrics.get("ann_vol") or 0.10)
    cap = max_dd_over_vol_quantile(
        sharpe, float(years), spec.dd_shape_quantile, ann_vol
    )
    value = float(metrics["dd_over_vol"])
    if value > cap:
        return [
            f"{spec.name}: dd_over_vol={value:.4g} violates the "
            f"q{spec.dd_shape_quantile:.2f} shape cap {cap:.3g} for Sharpe "
            f"{sharpe:.3g} over {float(years):.1f}y"
        ]
    return []


def load_metrics(unit: str) -> dict:
    """Load ``{ARTEFACT_ROOT}/{unit}.json``; skip the test if it is not there."""
    path = ARTEFACT_ROOT / f"{unit}.json"
    if not path.is_file():
        pytest.skip(f"no kill-gate artefact at {path} — unit not evaluated yet")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Tests — spec sanity (always run, no artefacts needed)
# --------------------------------------------------------------------------- #
def test_prompt_v2_gate_set_is_internally_infeasible():
    """Regression for research doc §1.2 — the shipped v2.0 gate set is unsatisfiable.

    It demanded Calmar >= 3.3 at Sharpe >= 0.60. Even the OLD, far too generous
    ceiling (2*S^2 = 0.72) rejected that; the corrected one rejects it by an
    order of magnitude more.
    """
    problems = check_consistency(PROMPT_V2_GATE_B1)
    assert problems, "the v2.0 gate set must be flagged as infeasible"
    assert any("Calmar" in p for p in problems)
    assert calmar_ceiling(0.60, 22.6) < 0.72


@pytest.mark.parametrize("spec", ALL_GATES, ids=lambda s: s.name)
def test_mandate_gate_sets_are_self_consistent(spec: GateSpec):
    assert check_consistency(spec) == []


def test_the_old_kg_b1_would_now_be_flagged_infeasible():
    """The W28 finding, encoded: the previous KG-B1 was itself unsatisfiable.

    A consistency checker that passes an unsatisfiable spec is worse than no
    checker, because it converts a design error into false confidence. This is
    the regression that stops the old constants coming back.
    """
    old = GateSpec(
        name="KG-B1 (pre-W28)",
        min_net_sharpe=0.40,
        min_years_positive_frac=0.60,
        max_dd_over_vol=2.5,
        min_calmar=0.25,
        sample_years=22.6,
    )
    problems = check_consistency(old)
    assert any("Calmar" in p for p in problems), (
        "the pre-W28 KG-B1 must be flagged: it demanded Calmar 0.25 at a "
        "Sharpe floor of 0.40, above what such a strategy can be expected to "
        "produce over 22.6 years"
    )


def test_shape_cap_scales_with_skill_and_horizon():
    """Sizing cannot buy past it; skill and a shorter record loosen it."""
    assert max_dd_over_vol_quantile(1.0, 22.6) < max_dd_over_vol_quantile(0.4, 22.6)
    assert max_dd_over_vol_quantile(0.5, 5.0) < max_dd_over_vol_quantile(0.5, 22.6)


# --------------------------------------------------------------------------- #
# Tests — known outcomes (the gate machine must reproduce recorded verdicts)
# --------------------------------------------------------------------------- #
# Measured net Sharpes from docs/FALSIFICATION_REGISTER.md and docs/WAVE_LOG.md.
# If a change to the gate logic ever lets one of these through, the change is wrong.
ARCHIVED_EQUITY_UNITS = {
    "eq_xsmom_12_1": -0.34,       # F13
    "eq_lowvol_v2": -0.12,        # F14
    "eq_overnight_1m": -0.60,     # F15
    "eq_strev_resid_1m": -0.35,   # F16
    "eq_quality_gpa": -0.25,      # F17
    "eq_pead_ar3": -0.18,         # F18
    "eq_strev_1m": 0.39,          # archived: lat missed by 0.01, no rounding
}


@pytest.mark.parametrize(
    "unit,net_sharpe", sorted(ARCHIVED_EQUITY_UNITS.items()), ids=lambda v: str(v)
)
def test_archived_units_still_fail_kg_b1(unit: str, net_sharpe: float):
    failures = evaluate({"net_sharpe": net_sharpe}, GateSpec(
        name="KG-B1 (sharpe criterion)", min_net_sharpe=KG_B1_INSAMPLE.min_net_sharpe
    ))
    assert failures, f"{unit} at {net_sharpe} must not clear the 0.40 lat"


def test_lat_is_not_rounded():
    """0.39 fails.  This is the eq_strev_1m precedent, encoded."""
    spec = GateSpec(name="lat", min_net_sharpe=0.40)
    assert evaluate({"net_sharpe": 0.39}, spec)
    assert evaluate({"net_sharpe": 0.399999}, spec)
    assert not evaluate({"net_sharpe": 0.40}, spec)


def test_missing_metric_is_a_failure_not_a_pass():
    """Silence is never evidence of passing."""
    failures = evaluate({"net_sharpe": 0.55}, KG_B1_INSAMPLE)
    assert any("missing" in f for f in failures)


def test_evaluate_reports_every_violation_not_just_the_first():
    metrics = {
        "net_sharpe": 0.10,
        "years_positive_frac": 0.30,
        "dd_over_vol": 40.0,
        "sample_years": 22.6,
        "ann_vol": 0.042,
    }
    failures = evaluate(metrics, KG_B1_INSAMPLE)
    assert len(failures) == 3, failures


def test_a_typical_qualifying_unit_passes_kg_b1():
    """Guard against a gate so tight that nothing can pass it.

    The old version of this test hand-picked dd_over_vol=2.1 at Sharpe 0.48 —
    a top-quartile-lucky path — and so gave false comfort about a gate that
    rejected 99.7% of qualifying strategies. This uses the MEDIAN drawdown a
    Sharpe-0.48 strategy actually produces over this horizon.
    """
    from tradebot.backtest.dd_shape import dd_shape_reference

    typical_dd = float(np.median(dd_shape_reference(0.48, 22.6, 0.042)))
    metrics = {
        "net_sharpe": 0.48,
        "years_positive_frac": 0.65,
        "dd_over_vol": typical_dd,
        "sample_years": 22.6,
        "ann_vol": 0.042,
    }
    assert evaluate(metrics, KG_B1_INSAMPLE) == []


def test_the_median_qualifying_unit_failed_the_old_gate():
    """Quantifies what was actually wrong, so it cannot be re-introduced."""
    from tradebot.backtest.dd_shape import dd_shape_reference

    typical_dd = float(np.median(dd_shape_reference(0.48, 22.6, 0.042)))
    old = GateSpec(name="KG-B1 (pre-W28)", max_dd_over_vol=2.5, min_calmar=0.25)
    metrics = {"dd_over_vol": typical_dd, "calmar": (0.48 - 0.021) / typical_dd}
    assert evaluate(metrics, old), (
        f"a median Sharpe-0.48 path (dd/vol {typical_dd:.2f}) should have "
        f"failed the pre-W28 gate — that was the defect"
    )


# --------------------------------------------------------------------------- #
# Tests — live units (skip until the artefact exists)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("unit", ["cm_tsmom", "cm_carry", "cm_basis_mom"])
@pytest.mark.parametrize("spec", ALL_GATES, ids=lambda s: s.name)
def test_futures_unit_against_gate(unit: str, spec: GateSpec):
    metrics = load_metrics(unit)
    completed = metrics.get("gates_completed")
    if completed is not None and spec.name not in completed:
        pytest.skip(f"{unit} has not been evaluated for {spec.name} yet")
    failures = evaluate(metrics, spec)
    assert not failures, f"{unit} fails {spec.name}:\n  " + "\n  ".join(failures)


def test_cm_tsmom_stays_archived_on_g4_strict():
    """The W28 re-derivation must not resurrect an archived unit.

    Loosening a shape threshold and letting the previously-failed unit through
    is exactly the §1.3 failure mode. cm_tsmom now clears KG-B1, and that is
    fine — its registered reopening condition also requires a G4-strict pass,
    which it does not have (t=1.75 < 2.0). If this ever goes green, someone has
    moved the substantive gate, not the shape gate.
    """
    metrics = load_metrics("cm_tsmom")
    assert evaluate(metrics, KG_B1_INSAMPLE) == [], (
        "cm_tsmom should clear the re-derived shape gate"
    )
    assert evaluate(metrics, KG_B2_ALPHA), (
        "cm_tsmom must still fail KG-B2 — that is what archives it"
    )
    assert metrics["verdict"].startswith("ARCHIVED")


def test_book_level_gate_kg_p():
    """KG-P: a new sleeve must earn its place at book level, not standalone."""
    metrics = load_metrics("book_with_futures")
    uplift = metrics["book_sharpe_with"] - metrics["book_sharpe_without"]
    assert uplift >= 0.10, f"book Sharpe uplift {uplift:.3f} < 0.10"
    assert metrics["book_max_drawdown"] <= 0.10
    assert abs(metrics["rho_a_b"]) < 0.30
