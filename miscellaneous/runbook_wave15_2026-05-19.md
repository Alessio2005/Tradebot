# Tradebot Live Operations Runbook
*Last updated: 2026-05-19 — Chief Audit Wave 15 + Sim-to-Reality Closure*

---

## 1. Engine Latency Monitoring (`latency_report`)

**Why it matters:** P95 alone misses OS-pause / network-stall pathologies that
double effective latency. A single 3-second stall behind a fill confirmation
can queue three follow-up orders and produce a phantom double-entry.

**API:**
```python
report = engine.latency_report()
# {'p50': ..., 'p95': ..., 'p99': ..., 'p999': ..., 'max': ...}
```

**Alert thresholds (crypto MFT, 1h bars):**

| Metric | Green | Amber | Red |
|--------|-------|-------|-----|
| p50    | < 50 ms | 50-150 ms | > 150 ms |
| p95    | < 200 ms | 200-400 ms | > 400 ms |
| p99    | < 400 ms | 400-800 ms | > 800 ms |
| p999   | < 4× p95 | 4-8× p95 | > 8× p95 |

**Rule of thumb (Sim-to-Reality #13):** if `p999 > 4 × p95`, the
strategy needs throttling — the queue-of-orders-behind-confirmation pattern
is dominant.

**Prometheus metric:** `tradebot_bar_latency_seconds` (histogram) —
exported by the `/metrics` endpoint at `TRADEBOT_METRICS_PORT`.

---

## 2. Intraday Drawdown Circuit Breaker (`max_intraday_drawdown_pct`)

**Why it matters (Sim-to-Reality #14):** Lifetime-peak drawdown tracking
triggers the CB only after a catastrophic equity curve erosion. A single
bad session that recovers and then relapses could never trip the lifetime CB,
yet represents real operational risk. The intraday tracker resets at UTC
midnight and treats each session independently.

**Configuration (`CircuitBreakerConfig`):**
```python
CircuitBreakerConfig(
    max_drawdown_pct=0.08,            # lifetime peak-to-trough (catastrophic)
    max_intraday_drawdown_pct=0.05,   # per-session drawdown from session peak
    max_daily_loss_pct=0.03,          # daily open-to-close loss
)
```

**On trip:**
1. `OMS.close_all()` fires immediately.
2. Trip is appended to `artefacts/circuit_breaker.log` (append-only JSONL).
3. Engine refuses to restart until the entry is acknowledged:
   edit `circuit_breaker.log` and set `"acknowledged": true`.

**To acknowledge a trip (operator procedure):**
```bash
# Open artefacts/circuit_breaker.log, find the unacknowledged entry, change:
#   "acknowledged": false  →  "acknowledged": true
# Save the file, then restart the engine.
```

---

## 3. Reference Window Stability Gate (`assess_reference_stability`)

**Why it matters (Sim-to-Reality #16):** If the reference/training window
spans a regime shift (e.g. LUNA + FTX both inside a 6-month window), the
baseline distribution is bimodal and PSI against it will never flag drift —
anything looks "stable" relative to a wide, noisy envelope.

**Pre-monitoring check:**
```python
from tradebot.monitoring.drift import assess_reference_stability, PSI_CRITICAL

reference_data = {"btc_vol": btc_vol_array, "eth_skew": eth_skew_array}
stability = assess_reference_stability(reference_data, limit=PSI_CRITICAL)

for result in stability:
    if not result.is_stable:
        print(f"UNSTABLE REFERENCE: {result.feature} "
              f"intra_PSI={result.intra_psi:.3f} n={result.n}")
        # DO NOT proceed with drift monitoring for this feature.
        # Select a homogeneous sub-window (single regime) as reference.
```

**Intra-window PSI interpretation:**
- `< 0.10` : reference is stable — drift monitoring reliable.
- `0.10–0.20` : marginal — narrow the window or use regime labels.
- `> 0.20` : UNSTABLE — drift detection unreliable, do not proceed.

**`check_feature_drift` runs this automatically** when `assess_reference=True`
(the default). Look for `WARNING: check_feature_drift: N feature(s) have
unstable reference windows` in the logs.

---

## 4. Crisis Regime Multiplier (Sim-to-Reality #5)

When BTC's rolling 60-bar skewness drops below −1.5, the engine enters
"crisis mode":
- Market impact η is multiplied by 1.5× (`negative_skew_crisis_multiplier`).
- Slippage floor is increased via `crisis_multiplier` in `compute_slippage`.

**Monitor in logs:**
```
WARNING PortfolioController: BTC negative-skew crisis active (crisis_multiplier=1.5).
```

Crisis mode disengages when BTC skew recovers above −0.5 (hysteresis band).

---

## 5. Fee Schedule Configuration (Sim-to-Reality #19)

Set `TRADEBOT_BINANCE_TIER` in `.env` to match your actual Binance Futures
account VIP tier:

```bash
TRADEBOT_BINANCE_TIER=VIP3      # e.g. VIP3: maker 1.2 bps, taker 3.0 bps
TRADEBOT_BNB_DISCOUNT=true      # add -10% if BNB-pay-fee is enabled
```

**Impact:** VIP0 vs VIP3 taker fee = 5.0 bps vs 3.0 bps — a 40% reduction
in per-trade cost. Using the wrong tier overstates live cost in backtests or
understates it in deployment, both of which bias the Sharpe estimate.

---

## 6. Strict Causal Gate (Sim-to-Reality #11)

```bash
TRADEBOT_STRICT_CAUSAL=1    # production (default)
TRADEBOT_STRICT_CAUSAL=0    # offline exploration only
```

When `=1`, calling `MetaLabeler.build_dataset()` without a `scout_for_audit`
argument raises `ValueError`. This prevents any production model from being
trained on an unlabeled (potentially lookahead-contaminated) dataset.

---

## 7. Monte Carlo Stress Test Parameters (Sim-to-Reality #18)

For production quarterly stress tests, use crisis-calibrated parameters:

```python
from tradebot.risk.stress_test import StressTestSuite

suite = StressTestSuite(price_data)
result = suite.run_monte_carlo(
    portfolio_weights,
    n_paths=10_000,
    tail_correlation=0.95,   # crisis corr floor (all assets → 1.0 in cascade)
    t_copula_df=5.0,         # fat-tail Student-t (crypto kurtosis ≈ 10-20)
)
```

Gaussian MC without these flags underestimates the 99th-percentile loss by
30–50% during liquidation cascades.

---

## 8. Reconciler Two-Strikes (Sim-to-Reality #15)

For live deployments, enable two-strikes to suppress Binance REST lag
false-positives:

```python
from tradebot.oms.reconciler import Reconciler, ReconcilerConfig

recon = Reconciler(
    tracker,
    ReconcilerConfig(
        paper_mode=False,
        fill_grace_period_sec=5.0,
        require_two_strikes=True,   # production hardening
    )
)
# Call recon.note_fill(symbol) in the order router on every fill.
```
