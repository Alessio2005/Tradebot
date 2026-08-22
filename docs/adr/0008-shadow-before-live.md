# ADR-0008 — 14-day shadow trading before live

**Status:** Accepted · 2026-05-12

## Context

A model that passes CPCV with strong DSR-corrected Sharpe still has three
unknowns the backtest cannot reveal:

1. **Implementation shortfall in live conditions** — actual fill prices vs
   model-assumed `bar.close × (1 + slip)`.
2. **Latency-induced opportunity loss** — TCP RTT to Binance, GIL pauses,
   asyncio scheduler jitter.
3. **Regime fragility** — the live window may differ from the latest CPCV
   fold.

## Decision

Every challenger model runs in **shadow mode** for at least 14 calendar days
before being eligible for the production tier. Shadow mode means:

- `ShadowTrader` instantiated in the live engine.
- Per bar, it computes the challenger's signal alongside the champion.
- A `PaperOMS` simulates the challenger's fill price.
- All challenger outputs (signal, hypothetical fill, hypothetical PnL) are
  logged per bar to `artefacts/shadow/{challenger}_vs_{champion}.parquet`.
- **No orders are placed for the challenger.**

After 14 days, `ChampionChallenger.evaluate(...)` decides promotion (see
ADR conventions in `docs/model_risk_policy.md`).

## Reason

- 14 days at 1h-bar cadence ≈ 240 bars per asset → enough for the
  Diebold-Mariano test to converge.
- Implementation-shortfall and latency effects are observable only in
  production-shape conditions.
- Failure modes that surface in shadow (e.g. signal flipping during the
  funding-rate roll) are caught without capital at risk.

## Trade-off

- Deployment cadence slowed: at most ~2 model swaps per month per asset.
  Mitigation: shadow-evaluate multiple challengers in parallel (PaperOMS
  is cheap).
- Stale data risk: a 14-day shadow that started in regime A may end in
  regime B. Mitigation: regenerate the shadow window on regime detection
  events (see `tradebot.features.regime.StructuralBreakRegime`).

## References

- AFML chapter 18 (Backtesting and Live Trading).
- MRM Policy `docs/model_risk_policy.md` §4.
