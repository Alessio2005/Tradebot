# ADR-0006 — Hierarchical Risk Parity over Markowitz

**Status:** Accepted · 2026-05-12

## Context

Portfolio allocation for 3–10 crypto-assets with notoriously unstable
correlations (BTC-ETH ρ can swing from 0.4 to 0.95 within a quarter).
Standard Mean-Variance Optimisation (MVO) requires a well-conditioned
covariance matrix and reliable expected-return vectors. Both assumptions
break under crypto regime changes — MVO concentrates aggressively into
whichever asset has the highest in-sample Sharpe.

## Decision

`tradebot.portfolio.hrp.HierarchicalRiskParity` is the **primary** allocation
method. `markowitz.py` remains as a fallback for explicit Sharpe-maximisation
queries (mostly diagnostics).

Algorithm:
1. Covariance via Ledoit-Wolf shrinkage (existing in `execution/market_impact`).
2. Distance matrix `D_ij = sqrt(0.5 × (1 − ρ_ij))`.
3. Ward-linkage hierarchical clustering on `D`.
4. Quasi-diagonalisation of the covariance matrix (leaf order preserved).
5. Recursive bisection: at each split, allocate inversely to within-cluster
   variance.

Result: positive weights summing to 1, no matrix inversion required.

## Reason

- Matrix inversion is the main numerical instability in MVO for crypto cov
  matrices (condition number ≫ 10^4 in observation).
- HRP is robust to non-stationarity: weights change smoothly as the
  dendrogram restructures, MVO weights jump discontinuously.
- Without explicit expected-return inputs (we plug those in via
  Black-Litterman at Stage 4.5, see ADR-0008), HRP cannot over-fit on
  in-sample alpha.

## Trade-off

- Theoretically sub-optimal under perfectly known cov + μ — accepted.
- No closed-form solution → recursion costs ~30 ms for 10 assets — fine.
- Less mature literature than MVO → covered by López de Prado (AFML §16),
  with property-based tests in `tests/property/test_portfolio_properties`.

## References

- López de Prado, M. (2016). *Building Diversified Portfolios that Outperform
  Out-of-Sample*. Journal of Portfolio Management.
- AFML chapter 16.
