# TCA methodology — Tradebot v1.0

Reference for the `tradebot.tca` package. Notation follows Almgren-Chriss
(2000) and Perold (1988) Implementation-Shortfall.

## 1. Pre-trade cost model

Total expected cost for sending notional `Q` over expected horizon `T` bars:

```
E[Cost] = spread_cost + impact_cost + delay_cost
```

| Component       | Formula                                            | Source                |
|-----------------|----------------------------------------------------|-----------------------|
| `spread_cost`   | `0.5 × bid_ask_spread_t × Q`                       | per-bar mid quote     |
| `impact_cost`   | `η × σ_t × sqrt(Q / ADV_t)`                        | Almgren square-root   |
| `delay_cost`    | `0` for IOC market orders; `T × σ_t × Q × κ_d` o.w | Almgren temporary     |

Default coefficients (BTC/ETH/SOL, 1h bars, year > 2023):
- `η = 0.142` (calibrated from `apps/calibrate_impact.py`, audit-N17)
- `κ_d = 0.001` (turnover-fraction penalty, BTC; doubled for SOL)

## 2. Post-trade decomposition (Implementation Shortfall)

For each filled order, decompose against the `arrival_price`:

```
IS_total = (arrival_price − execution_vwap) × signed_qty
         = delay_component
         + market_impact
         + timing_component
```

```
delay_component   = (decision_price − arrival_price) × qty
market_impact     = (execution_vwap − arrival_price) × qty
timing_component  = (post_trade_vwap − arrival_price) × qty  (limit orders only)
```

`decision_price` = mid at signal-emit time;
`arrival_price`  = mid at first send-to-exchange timestamp;
`execution_vwap` = volume-weighted fill prices;
`post_trade_vwap` = market VWAP over the bar following the order.

## 3. Roundtrip integration test (CI)

`tests/integration/test_tca_roundtrip.py` runs the following on each PR:

1. Generate 200 synthetic orders with known `Q, σ, ADV, half_spread`.
2. `pre_trade.estimate_cost(...)` produces a forecast in bps.
3. The PaperOMS simulates a fill; `post_trade.implementation_shortfall(...)`
   computes the realised cost.
4. **Acceptance**: `|forecast − realised| / forecast < 0.20` for ≥ 95 % of
   the orders.

This caps the simulation-to-reality gap to ±20 % in expectation. Larger
deviations flag a model recalibration ticket.

## 4. Daily TCA report

`scripts/paper_trade_report.py` + `tradebot.tca.report.generate_tca_summary`
produce a markdown report each night with:

- per-symbol pre-trade vs post-trade scatter
- worst-five orders by `|IS_total|`
- per-strategy aggregate IS (bps of notional)
- timing-component vs market-impact breakdown

The report is uploaded as a workflow artefact by `nightly-regression.yml`.

## 5. Calibration cadence

| Coefficient   | Frequency | Source                                           |
|---------------|-----------|--------------------------------------------------|
| `η`           | monthly   | rolling regression of realised IS vs `sqrt(Q/ADV)` |
| `κ_d`         | quarterly | turnover-fraction → realised slippage regression |
| `spread_t`    | per bar   | live exchange book / fallback to OHLC half-range |
| `ADV_t`       | daily     | trailing 30-day median quote volume              |

`apps/calibrate_impact.py` produces the new coefficients and writes them
under `artefacts/tca/coefficients.yaml`. The values are consumed by
`pre_trade.estimate_cost` via Hydra (`conf/tca/default.yaml`).
