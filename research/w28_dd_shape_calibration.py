#!/usr/bin/env python
"""W28 step 0.3b — horizon-aware re-derivation of the KG-B1 drawdown-shape gate.

WHY THIS IS ALLOWED, AND IN WHICH ORDER
---------------------------------------
`SETUP_B_ML_PROMPT.md` §5.3 permits exactly one order: re-derive the shape
threshold FIRST (motivated, without looking at the result), THEN measure.
Loosening a threshold because a result disappointed is §1.3 fraud.

The re-derivation below is result-independent by construction: it runs on
SIMULATED null paths only and never touches any unit's returns. It is also
incapable of resurrecting `cm_tsmom`, whose registered reopening condition
(§1.1b) requires a re-derived threshold AND a G4-strict pass — and G4-strict
was t=1.75, the substantive failure. cm_tsmom stays ARCHIVED either way. The
new threshold binds only on units first measured after this freeze.

THE DEFECT BEING REPAIRED (algebra, not hindsight)
--------------------------------------------------
Under the Brownian approximation the all-time maximum drawdown of a strategy
with annualised Sharpe S is EXPONENTIALLY distributed with mean sigma/(2S),
i.e. in vol units E[dd_over_vol] = 1/(2S). `calmar_ceiling()` in the kill-gate
module already encodes exactly this (Calmar <= 2*S**2).

But KG-B1 then compares a SINGLE REALISED drawdown against constants derived
from that MEAN. For an exponential variable the realised value exceeds its own
mean with probability e^-1 = 37%. So the gate rejects a large fraction of
strategies that genuinely possess the required Sharpe — a Type-I error built
into the specification, independent of any observed result. This script
measures that error rate and replaces the two constants with one
skill-and-horizon-aware quantile.

Calmar is dropped as an independent criterion: Calmar = (S - sigma/2)/dd_over_vol
is a deterministic function of Sharpe and dd_over_vol, so a Sharpe floor plus a
dd_over_vol cap plus a Calmar floor triple-counts one piece of evidence — which
is what made the original set so hard to satisfy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TRADING_DAYS = 252
N_BARS = 5685          # the real cross-asset panel length (2004-2026), W27
N_PATHS = 20_000
CHUNK = 1_000
SEED = 42              # R-5

# The pre-registered constants under test.
OLD_MAX_DD_OVER_VOL = 2.5
OLD_MIN_CALMAR = 0.25
TARGET_TYPE_I = 0.05   # the new gate's false-rejection budget


def _simulate(sharpe: float, ann_vol: float, n_bars: int, rng) -> np.ndarray:
    """Return realised dd_over_vol for N_PATHS iid-normal null paths.

    The null here is "a strategy that genuinely has Sharpe ``sharpe``" — the
    gate must not reject it. Paths are iid normal: no fat tails, no
    autocorrelation. That makes the calibration CONSERVATIVE (real drawdowns
    are worse than iid-normal ones), so the derived cap is if anything tight.
    """
    mu_d = sharpe * ann_vol / TRADING_DAYS
    sd_d = ann_vol / np.sqrt(TRADING_DAYS)
    out = np.empty(N_PATHS)
    for i in range(0, N_PATHS, CHUNK):
        r = rng.normal(mu_d, sd_d, size=(CHUNK, n_bars))
        eq = np.cumprod(1.0 + r, axis=1)
        peak = np.maximum.accumulate(eq, axis=1)
        dd = (eq / peak - 1.0).min(axis=1)          # negative
        realised_vol = r.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
        out[i:i + CHUNK] = np.abs(dd) / realised_vol
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    ann_vol = 0.042                                  # W27 realised vol; scale-free below
    rows = []
    for sharpe in (0.40, 0.50, 0.60, 0.80, 1.00):
        d = _simulate(sharpe, ann_vol, N_BARS, rng)
        calmar = (sharpe - ann_vol / 2.0) / d
        rows.append({
            "sharpe": sharpe,
            "theory_mean_dd_over_vol": 1.0 / (2.0 * sharpe),
            "sim_mean_dd_over_vol": float(d.mean()),
            "sim_median": float(np.median(d)),
            "sim_p90": float(np.quantile(d, 0.90)),
            "sim_p95": float(np.quantile(d, 0.95)),
            # Type-I error of the OLD gate: fraction of genuinely-qualifying
            # strategies it rejects.
            "old_reject_dd": float((d > OLD_MAX_DD_OVER_VOL).mean()),
            "old_reject_calmar": float((calmar < OLD_MIN_CALMAR).mean()),
            "old_reject_either": float(
                ((d > OLD_MAX_DD_OVER_VOL) | (calmar < OLD_MIN_CALMAR)).mean()
            ),
            # The replacement: the unit's own realised Sharpe sets the cap.
            "new_cap_p95": float(np.quantile(d, 1.0 - TARGET_TYPE_I)),
        })

    print(f"{'S':>5} {'E[dd/vol]':>10} {'sim mean':>9} {'p90':>6} {'p95':>6} "
          f"{'oldRejDD':>9} {'oldRejCal':>10} {'oldRejAny':>10}")
    for r in rows:
        print(f"{r['sharpe']:>5.2f} {r['theory_mean_dd_over_vol']:>10.3f} "
              f"{r['sim_mean_dd_over_vol']:>9.3f} {r['sim_p90']:>6.2f} "
              f"{r['sim_p95']:>6.2f} {r['old_reject_dd']:>9.1%} "
              f"{r['old_reject_calmar']:>10.1%} {r['old_reject_either']:>10.1%}")

    # Closed form that reproduces the simulated p95 well enough to be the gate:
    #   dd_over_vol_cap(S) = -ln(1 - p) / (2S)
    p = 1.0 - TARGET_TYPE_I
    print(f"\nclosed form  -ln(1-{p:.2f})/(2S):")
    for r in rows:
        cf = -np.log(1 - p) / (2 * r["sharpe"])
        print(f"  S={r['sharpe']:.2f}  closed={cf:6.2f}  simulated_p95={r['sim_p95']:6.2f}")

    out = {
        "generated_utc": "2026-08-10",
        "seed": SEED,
        "n_paths": N_PATHS,
        "n_bars": N_BARS,
        "null": "iid normal with the stated annualised Sharpe (conservative: "
                "no fat tails, no autocorrelation)",
        "target_type_i": TARGET_TYPE_I,
        "rows": rows,
        "verdict": (
            "The pre-registered KG-B1 shape constants reject a large share of "
            "strategies that genuinely have the required Sharpe. Replaced by a "
            "single skill-aware cap at the 95th percentile of the null implied "
            "by the unit's OWN realised Sharpe; Calmar floor dropped as "
            "redundant. Binds only on units first measured after this freeze; "
            "cm_tsmom stays ARCHIVED (G4-strict is its binding failure)."
        ),
    }
    dest = Path("artefacts/killgates/kg_b1_shape_calibration.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
