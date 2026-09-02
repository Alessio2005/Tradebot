# src/tradebot/live/execution_controller.py
"""Zet doelgewichten om in orders, binnen de SOEVEREINE risicolimieten.

WAT DEZE MODULE WEL EN NIET DOET
=================================
Hij vergelijkt de huidige posities met de doelgewichten, filtert op inertie, en
levert `Order`-objecten voor de OMS-router. Wat hij NIET doet is een eigen
limiet stellen: `max_position_pct` en `gross_cap` komen uit
`conf/risk/default.yaml` en zijn een VERPLICHT constructorargument.

De oude docstring beweerde hier *"applies Kelly sizing and position limits"*.
Gemeten op 2026-09-01 klopte geen van beide:

* de Kelly-fractie werd berekend en op de `Order` gezet, maar de ordergrootte
  bleef `|delta_w| * equity / price`; niets stroomafwaarts leest
  `Order.kelly_fraction` behalve het auditlogboek;
* `_check_position_limits` vergeleek met `float("inf")`, en **niets in de boom
  zette die velden ooit** -- de poort kon per constructie niet vuren.

Beide staan in `reports/phase7_divergence_map.md` §4. De Kelly-fractie blijft nu
staan als wat zij feitelijk is: een auditveld. De limiet is vervangen door een
VERIFICATIE tegen de soevereine policy.

WAAROM VERIFIEREN EN NIET CAPPEN
=================================
AD-1: de soevereine laag is de enige die een limiet stelt. Zou deze controller
het doelboek terugschalen, dan bestond er een tweede sizing-implementatie
(no-go 6) en zou een fout stroomopwaarts onzichtbaar worden opgelost. Een
doelboek buiten de policy is een defect in de laag die het boek maakte; hier
crasht het.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..oms.order import Order, OrderSide, OrderType
from ..risk.limits import GROSS_CAP_KEY, PER_ASSET_KEY
from ..schemas.config import RiskConfig
from ..utils.failfast import ConfigContractError, require

logger = logging.getLogger(__name__)

__all__ = ["ExecutionControllerConfig", "ExecutionController"]

# Legacy dust-trade floor (USDT).  Kept only as an absolute minimum below which
# nothing trades regardless of inertia config — config.min_notional_per_trade
# is the operational gate (default $1 000).
_MIN_ORDER_NOTIONAL = 10.0  # USDT

#: Zelfde tolerantie als de VERIFICATIE in `risk/engine.py` (regels 342 en 356
#: gebruiken `+ 1e-9`). `risk/limits.py::_TOL` staat op 1e-12, maar dat is de
#: tolerantie waarmee daar wordt teruggeschaald, niet waarmee wordt getoetst.
#: Hier wordt getoetst, dus geldt de toetsingstolerantie -- anders kan deze
#: controller een boek weigeren dat de soevereine laag zojuist heeft goedgekeurd.
_LIMIT_TOL = 1e-9


class ExecutionControllerConfig:
    """Configuration for the execution controller.

    Parameters
    ----------
    risk :
        De soevereine risicopolicy uit `conf/risk/default.yaml`. VERPLICHT en
        zonder default: een ontbrekende limietconfiguratie is een crash, geen
        "geen limiet" (no-go 7, audit §23).
    kelly_divisor :
        Fractional Kelly divisor. Wordt op de `Order` vastgelegd voor het
        auditspoor en beinvloedt de ordergrootte NIET; zie de moduledocstring.
    max_kelly_fraction :
        Absolute cap op de vastgelegde Kelly-fractie.
    max_weight_change :
        Maximum allowed single-bar weight change (turnover limiter).
    git_sha :
        Current git commit SHA for audit trail.
    model_version :
        Current model version string.
    min_notional_per_trade :
        CHIEF-1 (2026-05-28) — INERTIA FILTER, absolute USDT floor.  Any
        rebalance order below this notional is dropped so the portfolio
        does not churn $200-$300 trades on $40 k positions.  With
        post-trade cost ≈ 10 bps round-trip + maker-taker imbalance, a
        $300 trade costs ≈ $0.30 → over 8 760 bars/year that compounds
        away alpha.  Default $1 000 (≈ 0.5 % of starting NAV).
    min_weight_change :
        CHIEF-1 — INERTIA FILTER, relative gate.  Drop rebalances where
        |target_w - current_w| < this threshold (default 2 %).  This is
        the "no trade band" used by every serious portfolio manager.
        Stacked with ``min_notional_per_trade`` (AND-gate): both must
        be exceeded for an order to fire.
    """

    def __init__(
        self,
        risk: RiskConfig,
        kelly_divisor: float = 4.0,
        max_kelly_fraction: float = 0.25,
        max_weight_change: float = 0.10,
        git_sha: str = "",
        model_version: str = "v1.0.0",
        min_notional_per_trade: float = 1_000.0,
        min_weight_change: float = 0.02,
    ) -> None:
        require(
            isinstance(risk, RiskConfig),
            "ExecutionControllerConfig vereist de soevereine RiskConfig. Er is "
            "geen pad waarlangs deze controller zonder limietpolicy draait.",
            ConfigContractError, received=type(risk).__name__,
        )
        self.risk = risk
        self.kelly_divisor = kelly_divisor
        self.max_kelly_fraction = max_kelly_fraction
        self.max_weight_change = max_weight_change
        self.git_sha = git_sha
        self.model_version = model_version
        # CHIEF-1 inertia parameters
        self.min_notional_per_trade = float(min_notional_per_trade)
        self.min_weight_change = float(min_weight_change)


class ExecutionController:
    """Converts target weights to Order objects.

    Parameters
    ----------
    config :
        Controller configuration.
    """

    def __init__(self, config: ExecutionControllerConfig) -> None:
        self._cfg = config
        self._seq: int = 0

        # De limieten komen uit de soevereine policy en staan hier als
        # AFLEZING, niet als eigen instelbare toestand. Er is geen setter: een
        # limiet die na constructie te wijzigen is, is een limiet die iemand
        # vergeet te zetten.
        self.max_position_pct: float = float(config.risk.max_position_pct)
        self.gross_cap: float = float(config.risk.gross_cap)

        # Wave 15 P0-5.3 — fat-finger state
        self._last_order_qty: dict[str, float] = {}
        self._daily_volume_notional: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Per-bar order generation
    # ------------------------------------------------------------------

    def size_orders(
        self,
        target_weights: pd.Series,
        current_weights: dict[str, float],
        prices: dict[str, float],
        equity: float,
        signal_probs: dict[str, float] | None = None,
        feature_hashes: dict[str, str] | None = None,
    ) -> list[Order]:
        """Generate orders to move from ``current_weights`` to ``target_weights``.

        Parameters
        ----------
        target_weights :
            Target portfolio weights (indexed by symbol, sum ≤ 1).
        current_weights :
            Current portfolio weights (symbol → fraction of equity).
        prices :
            Current close prices per symbol.
        equity :
            Current portfolio equity in USDT.
        signal_probs :
            symbol → calibrated signal probability (for audit trail).
        feature_hashes :
            symbol → feature hash (for audit trail).

        Returns
        -------
        List of Order objects.  Empty list if no rebalancing is needed.
        """
        self._verify_target_book(target_weights)

        orders: list[Order] = []
        now = pd.Timestamp.now(tz="UTC")
        date_str = now.strftime("%Y%m%d")

        for symbol in target_weights.index:
            target_w = float(target_weights[symbol])
            current_w = float(current_weights.get(symbol, 0.0))
            delta_w = target_w - current_w

            # ── CHIEF-1 (2026-05-28) — INERTIA FILTER (anti-churn) ─────────
            # Empirical evidence from Run 4b:
            #   16:02 SELL DOTUSDT $304   (delta_w ≈ 0.0015 on $40 k pos)
            #   16:57 BUY  DOTUSDT $234   (delta_w ≈ 0.0012 on $40 k pos)
            # These were costlier than informative — pure drift correction
            # spending half-spread + maker-taker delta + slippage every time.
            #
            # New gate: drop the order unless BOTH
            #   |delta_w|        ≥ min_weight_change       (relative)
            #   |delta_w|·equity ≥ min_notional_per_trade  (absolute)
            # are exceeded.  An old, looser tier (>1 bp & >$10) still applies
            # as a hard dust floor for sanity.
            inertia_w   = abs(delta_w) < self._cfg.min_weight_change
            inertia_usd = abs(delta_w) * equity < self._cfg.min_notional_per_trade
            if inertia_w or inertia_usd:
                logger.debug(
                    "ExecutionController: INERTIA skip [%s] delta_w=%.4f "
                    "notional=$%.0f (gates: w>=%.4f, $>=%.0f)",
                    symbol, delta_w, abs(delta_w) * equity,
                    self._cfg.min_weight_change, self._cfg.min_notional_per_trade,
                )
                continue
            # Hard dust floor (legacy).  Should never trigger after the
            # inertia gates above unless someone explicitly drops them to 0.
            if abs(delta_w) < 0.001:
                continue
            if abs(delta_w) > self._cfg.max_weight_change:
                delta_w = self._cfg.max_weight_change * (1 if delta_w > 0 else -1)

            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue

            notional = abs(delta_w) * equity
            if notional < _MIN_ORDER_NOTIONAL:
                continue

            qty_base = notional / price
            side = OrderSide.BUY if delta_w > 0 else OrderSide.SELL

            # Kelly fraction (proportional to |delta_w|, capped)
            kelly = min(abs(delta_w) / self._cfg.kelly_divisor, self._cfg.max_kelly_fraction)

            self._seq += 1
            order_id = f"ord_{date_str}_{symbol}_{side.value}_{self._seq:04d}"

            signal_prob = (signal_probs or {}).get(symbol, 0.5)
            feature_hash = (feature_hashes or {}).get(symbol, "")

            # Wave 15 P0-5.3 — fat-finger check
            price_for_ff = prices.get(symbol, 0.0)
            try:
                self._fat_finger_check(symbol, qty_base, price_for_ff)
            except ValueError as exc:
                logger.error("ExecutionController: %s", exc)
                continue

            orders.append(
                Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    qty_base=round(qty_base, 8),
                    signal_prob=signal_prob,
                    kelly_fraction=kelly,
                    model_version=self._cfg.model_version,
                    git_sha=self._cfg.git_sha,
                    feature_hash=feature_hash,
                    portfolio_weight=target_w,
                    pre_trade_cost_bps=0.0,
                    created_at=now,
                )
            )
            # Wave 15 P0-5.3 — update last order qty after successful order creation
            self._last_order_qty[symbol] = abs(qty_base)

            logger.debug(
                "ExecutionController: %s %s qty=%.6f delta_w=%.4f",
                symbol, side.value, qty_base, delta_w,
            )

        return orders

    # ------------------------------------------------------------------
    # Verificatie tegen de soevereine policy (Stage D, C3/C4)
    # ------------------------------------------------------------------

    def _verify_target_book(self, target_weights: pd.Series) -> None:
        """Toets het DOELBOEK aan `risk.max_position_pct` en `risk.gross_cap`.

        Dezelfde grootheden en dezelfde sleutels als `risk/limits.py`:
        `|w_i| <= max_position_pct` per symbool en `sum |w_i| <= gross_cap` over
        het boek. Gewichten, geen notionals -- dat is de definitie waarop de
        soevereine laag zijn besluit neemt, en een tweede definitie hier zou
        precies het probleem zijn dat §4 van de divergence map beschrijft.

        Crasht bij overschrijding. Cappen zou het defect stroomopwaarts
        verbergen; overslaan-en-loggen (wat de oude poort deed) laat het boek in
        een toestand achter die niemand heeft besloten.
        """
        weights = target_weights.astype(float)
        gross = float(weights.abs().sum())
        for symbol, weight in weights.items():
            require(
                abs(float(weight)) <= self.max_position_pct + _LIMIT_TOL,
                f"{PER_ASSET_KEY} overschreden door het doelboek: "
                f"|w[{symbol}]| = {abs(float(weight)):.4f} > "
                f"{self.max_position_pct:.4f}. De soevereine laag hoort dit "
                f"boek al te hebben teruggeschaald; dat het hier aankomt, is "
                f"een defect stroomopwaarts.",
                ConfigContractError, key=PER_ASSET_KEY, symbol=str(symbol),
                measured=abs(float(weight)), threshold=self.max_position_pct,
            )
        require(
            gross <= self.gross_cap + _LIMIT_TOL,
            f"{GROSS_CAP_KEY} overschreden door het doelboek: "
            f"sum |w| = {gross:.4f} > {self.gross_cap:.4f}.",
            ConfigContractError, key=GROSS_CAP_KEY, measured=gross,
            threshold=self.gross_cap,
        )

    # ------------------------------------------------------------------
    # Wave 15 P0-5.3 — Fat-finger validator
    # ------------------------------------------------------------------

    def _fat_finger_check(
        self,
        symbol: str,
        qty: float,
        price: float,
    ) -> None:
        """Fat-finger validator: reject abnormally large orders (Wave 15 P0-5.3)."""
        notional = abs(qty * price)

        # 2× jump detector: reject als qty > 2× laatste qty voor dit symbol
        last_qty = self._last_order_qty.get(symbol, 0.0)
        if last_qty > 0.0 and abs(qty) > 2.0 * last_qty:
            raise ValueError(
                f"FAT FINGER: {symbol} qty={qty:.4f} is >2× last qty={last_qty:.4f}. "
                f"Reject to prevent runaway order. (Wave 15 P0-5.3)"
            )

        # % daily volume check: reject als notional > 0.5% daily volume
        daily_vol_notional = self._daily_volume_notional.get(symbol, float("inf"))
        if notional > 0.005 * daily_vol_notional:
            raise ValueError(
                f"FAT FINGER: {symbol} notional={notional:.2f} > 0.5% daily volume. "
                f"Reject. (Wave 15 P0-5.3)"
            )
