# src/tradebot/live/portfolio_controller.py
"""Portfolio optimizer → target weights per bar-close.

Takes the combined signal map (symbol → SignalResult) and a returns
DataFrame and emits target portfolio weights using the configured
optimisation method.

PHASE 7/8 STAGE C-1 — `method` had hier `"hrp"` als DEFAULT, en deze docstring
zei dat ook met zoveel woorden. HRP is Research Track (audit §13.1) en per
fase-6 no-go 15 technisch geblokkeerd voor productie tot turnover-gecorrigeerde
OOS-superioriteit boven Inverse Volatility is aangetoond.

Die no-go luidde "HRP is productie-toegankelijk zonder bewijs". De meting wees
uit dat het erger was: HRP was de STANDAARD-allocator van de live-controller,
en van `portfolio/optimizer.py::optimize`. Wie beide argumenten oversloeg, kreeg
een research-only allocator in het live-pad.

`method` heeft daarom geen default meer - dezelfde behandeling die `constraints`
in Phase 5 kreeg, en om dezelfde reden: een default die stilzwijgend beleid
vaststelt, is geen default maar een ongeschreven besluit.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..alpha.base import SignalResult
from ..execution.market_impact import negative_skew_crisis_multiplier
from ..portfolio.constraints import PortfolioConstraints
from ..portfolio.optimizer import OptimisationMethod, optimize
from ..utils.failfast import ConfigContractError, require

logger = logging.getLogger(__name__)

__all__ = ["PortfolioControllerConfig", "PortfolioController"]


class PortfolioControllerConfig:
    """Configuration for the portfolio controller.

    Parameters
    ----------
    method :
        Optimisation method passed to ``portfolio.optimizer.optimize()``.
    min_history_bars :
        Minimum number of bars of return history before optimisation fires.
        Falls back to equal-weight below this threshold.
    constraints :
        Portfolio constraints (leverage, concentration, turnover).
    returns_window :
        Rolling window (bars) of returns used for covariance estimation.
    signal_tilt_strength :
        CHIEF-3 (2026-05-28) — λ ∈ [0, 1] blending the HRP risk-parity
        weights with a signal-conditional posterior tilt::

            w_final = (1 - λ) · w_HRP + λ · w_tilt

        where ``w_tilt`` is the L1-normalised softmax of
        ``signal · (2·confidence - 1)`` across active assets.  HRP itself
        ignores ``expected_returns`` (see hrp.py — it is pure inverse-
        variance after clustering), so without this overlay a 0.75
        cal_prob asset gets the *same* dollar allocation as a neutral one
        and the model's edge is thrown away.

        Default 0.30 = mild tilt.  Preserves HRP's risk balance while
        letting a +0.25 confidence delta shift weights ≈ 8–12 pp.
        Set to 0.0 to recover pure HRP.
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        method: OptimisationMethod,
        min_history_bars: int = 60,
        returns_window: int = 120,
        signal_tilt_strength: float = 0.30,
    ) -> None:
        self.method = method
        self.min_history_bars = min_history_bars
        # CHIEF-5/6 (2026-05-28):
        #   max_weight=0.40 — see CHIEF-5 (0.25 was too tight for 5 assets).
        #   min_weight=0.0  — CHIEF-6.  The default PortfolioConstraints.min_weight
        #   was 0.02, which forces apply_constraints() to zero out any asset
        #   whose post-tilt weight lands below 2%.  In Run 4d this dropped
        #   DOT after a SHORT additive tilt brought its HRP baseline (~7-10%)
        #   down to ~1-2%.  Dust prevention is already handled at the order
        #   layer by CHIEF-1's inertia filter
        #   (min_notional_per_trade=$1 000, min_weight_change=2 %), so the
        #   portfolio-level floor is redundant and actively destructive
        #   under signal-tilt overlays.
        #
        # PHASE 5: de fallback-constructie `PortfolioConstraints(max_weight=0.40,
        # min_weight=0.0)` is verwijderd. Zij was een LOKALE concentratielimiet
        # in L13 (wiring audit C5) die toevallig gelijk stond aan
        # `risk.max_concentration` maar er niet uit kwam - en dus bij elke
        # wijziging van `conf/risk/` stil kon gaan afwijken.
        #
        # De caller levert nu een set uit `from_risk_config()`. Er is bewust
        # GEEN default meer: fase-opdracht §23 verbiedt de automatische
        # permissieve modus, en een L13-component die zijn eigen risicodrempel
        # kiest is precies dat. Phase 7 sluit `live/` volledig aan; tot dan
        # dwingt deze crash af dat de vraag niet per omissie wordt beantwoord.
        require(
            isinstance(constraints, PortfolioConstraints),
            "PortfolioControllerConfig vereist een expliciete "
            "PortfolioConstraints uit PortfolioConstraints.from_risk_config(). "
            "L13 kiest geen risicodrempels (fase-opdracht §4.3, §23).",
            ConfigContractError,
            got=type(constraints).__name__,
        )
        self.constraints = constraints
        self.returns_window = returns_window
        self.signal_tilt_strength = float(signal_tilt_strength)


class PortfolioController:
    """Converts signals to target portfolio weights on each bar-close.

    Parameters
    ----------
    config :
        Controller configuration.
    symbols :
        List of traded symbols.
    """

    def __init__(
        self,
        config: PortfolioControllerConfig,
        symbols: list[str],
    ) -> None:
        self._cfg = config
        self._symbols = symbols
        # Rolling price history for returns computation
        self._price_history: dict[str, list[float]] = {s: [] for s in symbols}
        self._last_weights: pd.Series | None = None
        # Rec 3 (Sim-to-Reality #5): crisis regime detection via BTC skew
        self._btc_symbol: str | None = next(
            (s for s in symbols if "BTC" in s.upper()), None
        )
        self._crisis_state_active: bool = False
        self._crisis_multiplier: float = 1.0
        self._BTC_BUFFER_LEN: int = 60  # bars for rolling skew estimation

    # ------------------------------------------------------------------
    # CHIEF-2 (2026-05-28) — Startup seed for HRP-from-day-1
    # ------------------------------------------------------------------

    def seed_price_history(self, closes: dict[str, list[float]]) -> None:
        """Pre-populate per-symbol price history from a historical buffer.

        Without this, the engine boots with empty ``_price_history`` and
        ``optimise()`` falls back to equal-weight until ``min_history_bars``
        accumulate live (≈ 3 days at the current CUSUM rate).  The
        observed Run-4b symptom: $40 040 × 5 (exact 20/20/20/20/20) on the
        first bar, ignoring the fact that ETH historically has < ⅓ the
        realised vol of AVAX → HRP would prefer a heavier ETH leg.

        Inputs are L1-truncated to ``returns_window + 1`` so a 1.5 M-bar
        live buffer cannot blow up RAM here.
        """
        cap = self._cfg.returns_window + 1
        seeded = 0
        for sym in self._symbols:
            series = closes.get(sym, [])
            if not series:
                continue
            if len(series) > cap:
                series = series[-cap:]
            self._price_history[sym] = [float(p) for p in series if p and p > 0]
            seeded += 1
        logger.info(
            "PortfolioController: seeded price_history for %d/%d symbols "
            "(cap=%d bars per symbol) — HRP active from bar 1.",
            seeded, len(self._symbols), cap,
        )

    # ------------------------------------------------------------------
    # Per-bar optimise
    # ------------------------------------------------------------------

    def optimise(
        self,
        signals: dict[str, SignalResult | None],
        prices: dict[str, float],
    ) -> pd.Series:
        """Compute target weights given current signals and prices.

        Parameters
        ----------
        signals :
            symbol → SignalResult (or None if no signal this bar).
        prices :
            symbol → current close price.

        Returns
        -------
        pd.Series of target weights, indexed by symbol.
        """
        # P0-J FIX: optimaliseer EERST op de bestaande history (verleden bars),
        # voeg DAARNA de huidige bar toe. Oude code appende de huidige bar vóór
        # de optimalisatie → covariantie-matrix bevat bar-t retour die ook de
        # fill-prijs bepaalt → drie lagen gestapelde lookahead.
        # Juiste volgorde: compute weights on history[0..t-1], then append close[t].

        # Stap 1: optimaliseer op bestaande history (strikt verleden)
        min_len = min(len(v) for v in self._price_history.values()) if self._price_history else 0
        if min_len < max(5, self._cfg.min_history_bars):
            n = len(self._symbols)
            weights = pd.Series(1.0 / n, index=self._symbols)
        else:
            # Build returns DataFrame from history BEFORE current bar
            returns_df = pd.DataFrame(
                {sym: pd.Series(self._price_history[sym]).pct_change().dropna()
                 for sym in self._symbols}
            ).tail(self._cfg.returns_window)

            # Apply signal-based expected returns if available
            signal_returns: pd.Series | None = None
            active_signals = {
                sym: r for sym, r in signals.items() if r is not None
            }
            if active_signals:
                signal_returns = pd.Series(
                    {sym: r.signal * 0.01 for sym, r in active_signals.items()}
                ).reindex(self._symbols).fillna(0.0)

            # Geen `hrp_research_gate` meegegeven: `optimize()` CRAST daardoor
            # wanneer iemand `method="hrp"` op dit live-pad configureert. Dat is
            # opzet - het live-pad is precies de plek waar een research-only
            # allocator niet mag komen.
            weights = optimize(
                returns_df,
                method=self._cfg.method,
                constraints=self._cfg.constraints,
                current_weights=self._last_weights,
                expected_returns=signal_returns,
            )

            # ── CHIEF-3/5 (2026-05-28) — Additive Grinold-Kahn tilt ─────
            # HRP itself ignores expected_returns (pure inverse-variance —
            # see hrp.py), so without this overlay a 0.75 cal_prob asset
            # gets the same allocation as a neutral one.
            #
            # Original CHIEF-3 used a softmax-blend
            #     w = (1-λ)·w_HRP + λ·softmax(max(0, signal·(2·conf-1)))
            # which has TWO failure modes (observed live in Run 4c initial
            # fills):
            #   (i)  long-only clip max(0, ·) ZEROS all SHORT signals'
            #        informational content — they can't down-tilt their
            #        own weight, only HRP decides their share.
            #   (ii) When only ONE asset has positive edge, the softmax
            #        renormalises that single asset to 1.0 → λ fraction
            #        of total NAV gets dumped on one name, the rest
            #        proportionally squeezed (with min_weight=0.02
            #        dropping the smallest to 0 — DOT got zero'd).
            #
            # Fix: additive boost (Grinold-Kahn alpha overlay), symmetric:
            #     w_i = w_HRP_i + λ · signal_i · (conf_i - 0.5)
            # then clip ≥ 0 and renormalise.  This:
            #   - LONG strong  → boost  (+λ · 0.25 ≈ +7.5 pp at λ=0.30)
            #   - SHORT strong → cut    (-λ · 0.25 ≈ -7.5 pp)
            #   - FLAT (0.5)   → no tilt (0)
            #   - Mass preserved by renormalisation; no single asset can
            #     absorb 30 % of NAV from a single weak signal.
            lam = float(self._cfg.signal_tilt_strength)
            if lam > 0.0 and active_signals:
                deltas: dict[str, float] = {}
                for sym in self._symbols:
                    r = active_signals.get(sym)
                    if r is None:
                        deltas[sym] = 0.0
                        continue
                    # signed_edge ∈ [-0.5, +0.5] from (conf-0.5) ∈ [-0.5, +0.5]
                    # × signal ∈ {-1, +1}.
                    signed_edge = float(r.signal) * (float(r.confidence) - 0.5)
                    deltas[sym] = lam * signed_edge
                delta_series = pd.Series(deltas).reindex(self._symbols).fillna(0.0)
                blended = (weights + delta_series).clip(lower=0.0)
                s_blend = float(blended.sum())
                if s_blend > 1e-9:
                    blended = blended / s_blend
                    from ..portfolio.constraints import apply_constraints
                    weights = apply_constraints(
                        blended, self._cfg.constraints, self._last_weights
                    )
                    logger.debug(
                        "PortfolioController: additive-tilt applied lam=%.2f "
                        "deltas=%s",
                        lam, {k: round(v, 4) for k, v in deltas.items()},
                    )

        # Stap 2: voeg nu pas de huidige bar's prijs toe aan history
        for sym in self._symbols:
            price = prices.get(sym, 0.0)
            if price > 0:
                self._price_history[sym].append(price)
                if len(self._price_history[sym]) > self._cfg.returns_window + 1:
                    self._price_history[sym].pop(0)

        # Rec 3 (Sim-to-Reality #5): update crisis multiplier from BTC rolling skew.
        # Uses the price history AFTER appending the current bar so the skew
        # estimate incorporates the most recent close.
        if self._btc_symbol is not None:
            btc_prices = self._price_history.get(self._btc_symbol, [])
            if len(btc_prices) >= 32:
                recent = btc_prices[-self._BTC_BUFFER_LEN:]
                btc_rets = np.diff(np.log(np.array(recent, dtype=np.float64)))
                self._crisis_multiplier, self._crisis_state_active = (
                    negative_skew_crisis_multiplier(btc_rets, state_active=self._crisis_state_active)
                )
                if self._crisis_state_active:
                    logger.warning(
                        "PortfolioController: BTC negative-skew crisis active "
                        "(crisis_multiplier=%.1f).", self._crisis_multiplier
                    )

        self._last_weights = weights
        logger.debug(
            "PortfolioController: weights %s crisis_mult=%.1f",
            {sym: round(w, 4) for sym, w in weights.items()},
            self._crisis_multiplier,
        )
        return weights

    @property
    def crisis_multiplier(self) -> float:
        """Current crisis regime multiplier (1.0 = calm, 1.5 = crisis)."""
        return self._crisis_multiplier
