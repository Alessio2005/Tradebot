# Data dictionary

> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-3).
> GEMETEN: hoofdstuk 7 beschreef `artefacts/portfolio/equity_curve.parquet`,
> een artefact dat Phase 5 heeft laten vervallen. Het staat er nu als
> historisch schema, met de reden erbij.

Every parquet column the pipeline emits, with dtype, source, and causal lag.
Mandatory reading before any feature edit.

## 1. Raw bars — `market_data_parquet/{symbol}_{timeframe}.parquet`

Source: `apps/data_sync.py` → `BinanceVisionClient` → `ParquetStorage`.

| Column              | Dtype      | Unit       | Lag        | Description                                |
|---------------------|------------|------------|------------|--------------------------------------------|
| `timestamp`         | datetime64 | UTC ns     | 0          | Bar close timestamp (Binance kline-close). |
| `open`              | float64    | quote ccy  | 0          | First trade price in interval.             |
| `high`              | float64    | quote ccy  | 0          | Max trade price.                           |
| `low`               | float64    | quote ccy  | 0          | Min trade price.                           |
| `close`             | float64    | quote ccy  | 0          | Last trade price.                          |
| `volume`            | float64    | base ccy   | 0          | Σ traded base-asset quantity.              |
| `quote_volume`      | float64    | quote ccy  | 0          | Σ traded notional.                         |
| `taker_buy_volume`  | float64    | base ccy   | 0          | Buyer-initiated base volume.               |
| `taker_sell_volume` | float64    | base ccy   | 0          | Seller-initiated base volume.              |
| `n_trades`          | int64      | count      | 0          | Trade count in interval.                   |

## 2. Imbalance / dollar bars — `artefacts/bars/{symbol}.parquet`

Source: `tradebot.bars.imbalance.generate_imbalance_bars`
or `tradebot.bars.dollar.generate_dollar_bars` (Wave 10 / AUDIT A-2).

| Column         | Dtype      | Description                                                |
|----------------|------------|------------------------------------------------------------|
| `timestamp`    | datetime64 | Close-time of the synthetic bar.                           |
| `open/high/low/close` | float64 | OHLC over the bar.                                    |
| `tick_volume`  | float64    | Σ base volume during bar.                                  |
| `bar_notional` | float64    | Σ (price × tick_volume) — required for dollar bars / TCA.  |
| `bid_*` / `ask_*` | float64 | Best-bid/best-ask OHLC during bar (filled with `*` if absent). |
| `bar_type`     | category   | `tick` / `volume` / `imbalance` / `runs` / `dollar`.       |

## 3. Features — `artefacts/features/{symbol}.parquet`

Source: `tradebot.features.pipeline.FeaturePipeline.transform`.

Naming convention: `feat_{layer}_{family}_{detail}` where layer ∈ `{micro, meso, macro}`.

| Family             | Examples                                | Layer        |
|--------------------|-----------------------------------------|--------------|
| `vol_*`            | Garman-Klass, Parkinson, realised vol   | micro        |
| `ret_*`            | log-return, fwd-return (NEVER passed)   | micro/meso   |
| `mom_*`            | 12-1 momentum, RSI-z, MACD-z            | meso         |
| `ms_*`             | OFI, Kyle's λ, VPIN (AUDIT E-1)         | micro        |
| `regime_*`         | Hurst, stationarity-z, structural-break | meso         |
| `macro_*`          | DXY-z, VIX, US10Y, BTC-funding-z        | macro        |
| `funding_*`        | per-bar funding rate + 8h-cumulated     | meso (NEW; AUDIT E-4) |
| `ffd_*`            | min-frac-diff(close, d*), d* per asset  | micro (NEW; AUDIT A-1) |
| `pc_*`             | PCA component (per-fold fit; AUDIT C-3) | per layer    |

All features are **stationary** (`adfuller p < 0.05`); the build pipeline drops
any column failing the ADF gate (Wave 13 / AUDIT A-3).

## 4. Labels — `artefacts/labels/{symbol}.parquet`

Source: `tradebot.labeling.triple_barrier.TripleBarrierLabeler.label`.

| Column         | Dtype       | Description                                              |
|----------------|-------------|----------------------------------------------------------|
| `t0`           | datetime64  | Event-time (CUSUM filter trigger).                       |
| `t1`           | datetime64  | Barrier-hit timestamp (or t0 + horizon if timeout).      |
| `bin`          | int8        | -1, 0, +1 — directional outcome.                         |
| `ret`          | float64     | Realised return between t0 and t1 (entry-spread aware).  |
| `pt_hit`       | bool        | True if profit-take barrier crossed first.               |
| `sl_hit`       | bool        | True if stop-loss barrier crossed first.                 |
| `timeout`      | bool        | True if t1 = t0 + max_horizon.                           |
| `t1_idx`       | int64       | Bar-index of t1 (for purge/uniqueness math).             |

## 5. Hparams — `artefacts/hparams/{symbol}_{side}_best.json`

Schema: `tradebot.schemas.hparams.HparamsSchema`.

```json
{
  "study_name":        "BTCUSDT_long",
  "best_value":        1.234,
  "n_trials":          200,
  "n_complete":        198,
  "best_params":       { "learning_rate": 0.041, "depth": 6, ... },
  "deflated_sharpe":   0.612,
  "n_prior_experiments": 412,
  "git_sha":           "a3f8b2c"
}
```

## 6. OOS predictions — `artefacts/oos_probs/{symbol}.parquet`

Schema: `tradebot.schemas.oos_predictions.OOSPredictionSchema`.

| Column     | Dtype      | Description                                       |
|------------|------------|---------------------------------------------------|
| `ts`       | datetime64 | Event timestamp (matches labels.t0).              |
| `prob`     | float64    | Calibrated probability (Platt per CPCV path).     |
| `sigma`    | float64    | Predicted std-dev for sizing.                     |
| `fold_id`  | int32      | CPCV fold-pair index that produced this row.      |
| `side`     | category   | `long` or `short`.                                |

## 7. Portfolio output — VERVALLEN sinds Phase 5

> **Gecorrigeerd 2026-09-01 (Phase 7/8, Stage E-3).** Dit hoofdstuk beschreef
> het schema van `artefacts/portfolio/equity_curve.parquet`. Dat bestand bestaat
> niet en wordt door niets meer geproduceerd: `PortfolioBacktester` en
> `bidirectional_backtest` zijn in Phase 5 verwijderd na het pariteitsbewijs
> (`tests/integration/test_engine_parity.py`, `reports/phase5_engine_diff.md`),
> en hun DVC-stages `backtest_portfolio` en `make_tearsheet` met hen.
>
> De opvolger is `backtest/engine.py` (`EventDrivenEngine`) met
> `backtest/accounting.py` als boekhouding; het resultaat van een run staat in
> `artefacts/baseline/phase5_revaluation.json`. De kolomnamen hieronder staan er
> als HISTORISCHE beschrijving, zodat oude parquet-bestanden nog leesbaar zijn —
> niet als contract voor iets dat vandaag wordt geschreven.

### 7.1 Historisch schema (niet meer geproduceerd)

| Column            | Dtype       | Description                                  |
|-------------------|-------------|----------------------------------------------|
| `ts`              | datetime64  | Bar timestamp.                               |
| `equity`          | float64     | NAV after MTM, fees, funding.                |
| `gross_leverage`  | float64     | Σ \|weights\|.                                 |
| `net_exposure`    | float64     | Σ weights (signed).                          |
| `daily_pnl_pct`   | float64     | Day-rolled PnL fraction.                     |
| `drawdown`        | float64     | (equity_peak − equity) / equity_peak.        |

## 8. Audit log — `state/audit_log.jsonl`

Schema: `tradebot.schemas.orders.OrderAuditSchema`. Required fields (R-8):

`event_ts, order_id, symbol, side, qty_base, notional_usdt, order_type,
signal_prob, kelly_fraction, model_version, git_sha, feature_hash,
portfolio_weight, pre_trade_cost_bps`

Optional post-trade: `fill_price, fill_ts, post_trade_cost_bps,
circuit_breaker_state`.

## 9. Causality contract

For each feature column produced by the build pipeline, the contract is:

> *Row at bar t uses information available **strictly before or at** the
> close of bar t-1.*

This is enforced by:

1. `tests/lookahead/*` — explicit guard tests per family.
2. `FeatureStore.append()` — refuses to write a feature with a t value lower
   than `latest_timestamp(symbol)`.
3. `RegimeRouter` — uses `df.shift(1)` on any router-input column.

Any new feature that does not pass `tests/lookahead/` is forbidden by CI.
