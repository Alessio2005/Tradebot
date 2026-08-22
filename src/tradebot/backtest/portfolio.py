# src/tradebot/backtest/portfolio.py
"""Multi-asset portfolio backtester with PortfolioRiskManager integration.

Migrated from portfolio_backtest.py.
"""
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from ..risk.portfolio import PortfolioRiskManager, RiskState

# Phase 0 stap 5: het pakket `quant_architect` bestaat nergens. SchemaMismatchError
# is ooit geextraheerd naar tradebot.train.schema_guard zonder dat deze importsite
# is bijgewerkt. De except-tak definieerde een lokale RuntimeError-stub met dezelfde
# naam, waardoor een schema-mismatch niet meer als zodanig herkenbaar was.
from ..train.schema_guard import SchemaMismatchError
from .tracks import AssetTrack, PortfolioBacktestResult

logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """Multi-asset portfolio backtester met PortfolioRiskManager-integratie."""

    # AUDIT-FIX (Round 3 — Funding Δt correctie):
    # Bybit perp funding interval = 8h = 28800 seconden.
    # Bar-duration gebruikt als default: 15-min bars = 900 seconden.
    # Δt = bar_seconds / FUNDING_INTERVAL_SECONDS, zodat:
    #   funding_cost_bar = signed * funding_rate_8h * Δt
    # Bij sparse data (rate alleen ≠ 0 op de 8h-tick) is dit gelijkwaardig
    # aan de huidige aanpak zodra bar_seconds de juiste barlengte heeft.
    # Bij dense data (rate ffill'd over alle bars) voorkomt Δt de 32×
    # over-aanrekening bij 15-min bars (28800/900 = 32 bars per 8h-periode).
    _FUNDING_INTERVAL_SECONDS: float = 28800.0  # Bybit: 8h

    def __init__(
        self,
        risk_manager: PortfolioRiskManager,
        bars_per_year: float = 365.0 * 24 * 12,
        bar_seconds: float = 900.0,  # default: 15-min bars (60*15)
        # ── BLUEPRINT-FIX (IV.7 / V.8) — chief-architect stack ───────────
        # Eén gedeelde ``QuantArchitectStack`` instantie voor de hele live-loop.
        # ``None`` ⇒ legacy gedrag (geen η-update, geen schema-guard). Anders:
        #   * eta_observer.update(slip,σ,Q,V) na elke non-zero fill.
        #   * schema_guard.check(live_feature_names) per bar wanneer
        #     ``check_schema_per_bar`` is geactiveerd; bij SchemaMismatchError
        #     pauzeert de backtester (raises door naar caller).
        stack: Any | None = None,
        check_schema_per_bar: bool = False,
        live_feature_names_provider: Any | None = None,
        # CHIEF AUDIT 2026-05-28 (M-1): true multiple-testing burden for the
        # Deflated Sharpe Ratio.  Must equal the TOTAL number of hypotheses
        # explored during search/selection — i.e.
        #   n_assets × n_sides × n_optuna_trials (× n_prior_experiments).
        # The previous hard-coded ``len(tracks) × 2`` (=10) understated N by
        # ~200× and made the DSR almost equal to the raw Sharpe.  ``None`` keeps
        # the conservative ``len(tracks) × 2`` floor for legacy callers.
        total_n_hypotheses: int | None = None,
    ) -> None:
        self.risk_manager: PortfolioRiskManager = risk_manager
        self.bars_per_year: float = float(bars_per_year)
        self.total_n_hypotheses: int | None = (
            int(total_n_hypotheses) if total_n_hypotheses is not None else None
        )
        # Δt = fractie van de Bybit 8h-funding-periode per bar.
        # 15-min bars → Δt = 900/28800 = 1/32.
        self._funding_dt: float = float(bar_seconds) / self._FUNDING_INTERVAL_SECONDS
        # ── BLUEPRINT-FIX: stack-injectie ───────────────────────────────
        self.stack: Any | None = stack
        self.check_schema_per_bar: bool = bool(check_schema_per_bar)
        # Callable die per bar de live feature_names kan opleveren. None ⇒
        # schema-guard wordt overgeslagen ongeacht ``check_schema_per_bar``.
        self.live_feature_names_provider: Any | None = live_feature_names_provider

    # -----------------------------------------------------------------------
    # ALIGN ASSETS OP GEMEENSCHAPPELIJKE GRID
    # -----------------------------------------------------------------------
    def _align(self, tracks: list[AssetTrack]) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
        """Reindex elk asset op de gemeenschappelijke unie-timestamp.

        Bars zonder data krijgen 0 voor signed_returns/requested_lev/side.
        """
        if not tracks:
            # Gebruik cast en pd.to_datetime om Pylance 'Index[int]' errors te vermijden
            return cast(pd.DatetimeIndex, pd.to_datetime([])), {}

        union: pd.DatetimeIndex = cast(pd.DatetimeIndex, pd.to_datetime([]))
        for t in tracks:
            union = cast(pd.DatetimeIndex, union.union(t.timestamps))
        union = cast(pd.DatetimeIndex, union.sort_values())

        per_asset: dict[str, pd.DataFrame] = {}
        for t in tracks:
            # FUNDING-FIX: funding_rate optioneel per asset; 0 wanneer ontbrekend.
            n_t = len(t.timestamps)
            funding_arr: np.ndarray = (
                np.asarray(t.funding_rate, dtype=np.float64)
                if t.funding_rate is not None and len(t.funding_rate) == n_t
                else np.zeros(n_t, dtype=np.float64)
            )
            df = pd.DataFrame(
                {
                    "signed_return": t.signed_returns,
                    "requested_leverage": t.requested_leverage,
                    "side": t.side,
                    "funding_rate": funding_arr,
                },
                index=t.timestamps,
            )
            # ALIGNMENT-GAP-FIX (sim-to-reality gap — Silent Killer):
            #   reindex(union, fill_value=0.0) zet side=0 op elke gap-bar
            #   (event-bars van ANDERE assets).  Een BTC-trade van 3.9 eigen
            #   bars verschijnt dan als 3.9 afzonderlijke 1-bar trades op de
            #   union grid → 2-3× meer exit/re-entry events → 2-3× hogere
            #   round-trip kosten vs live trading (waar de positie gewoon
            #   OPEN blijft).
            #
            #   Fix: reindex zonder fill_value (→ NaN voor gap-bars), dan
            #   forward-fill POSITIE-kolommen (side, requested_leverage) zodat
            #   een open trade open blijft.  Returns en funding blijven 0 op
            #   gap-bars — P&L wordt alleen gerealiseerd op eigen event-bars.
            df = df.reindex(union)  # NaN voor gap-bars, GEEN fill_value=0
            # Positie forward-fillen: trade blijft open tot expliciete afsluiting
            df["side"] = df["side"].ffill().fillna(0.0).astype(int)
            df["requested_leverage"] = df["requested_leverage"].ffill().fillna(0.0)
            # Returns & funding: 0 op gap-bars (geen P&L-update tussen events)
            df["signed_return"] = df["signed_return"].fillna(0.0)
            df["funding_rate"] = df["funding_rate"].fillna(0.0)
            per_asset[t.symbol] = df

        return union, per_asset

    # -----------------------------------------------------------------------
    # MAIN BACKTEST LOOP
    # -----------------------------------------------------------------------
    def run(self, tracks: list[AssetTrack]) -> PortfolioBacktestResult:
        """Run de portfolio-backtest over de aangeleverde per-asset tracks."""
        if not tracks:
            return PortfolioBacktestResult(
                equity_curve=pd.Series(dtype=float),
                portfolio_returns=pd.Series(dtype=float),
                sharpe=0.0, max_dd=0.0, calmar=0.0, total_return=0.0,
                deflated_sharpe=0.0,
                n_bars=0,
            )

        timestamps, aligned = self._align(tracks)
        if len(timestamps) == 0:
            return PortfolioBacktestResult(
                equity_curve=pd.Series(dtype=float),
                portfolio_returns=pd.Series(dtype=float),
                sharpe=0.0, max_dd=0.0, calmar=0.0, total_return=0.0,
                deflated_sharpe=0.0,
                n_bars=0,
            )

        symbols: list[str] = [t.symbol for t in tracks]

        # CHIEF AUDIT 2026-05-23 (P0.2): Validate that all tracks supply funding
        # rates.  On Bybit perpetuals, funding is a constant cost for long-
        # biased strategies (~1-4 bps per 8h → ~2.7% CAGR at 2.5 bps avg).
        # Silently absent funding_rate data means this cost is structurally zero,
        # creating a systematic sim-to-reality gap of ~2-5% CAGR in live trading.
        _missing_funding = [t.symbol for t in tracks if t.funding_rate is None]
        if _missing_funding:
            logger.warning(
                "FUNDING GAP (sim-to-reality): tracks for %s have funding_rate=None. "
                "Funding cost will be zero in this backtest.  "
                "Provide per-bar funding rates (AssetTrack.funding_rate) to close "
                "this gap — expected live cost ~1-4 bps per 8h per long position.",
                _missing_funding,
            )

        # BARS-PER-YEAR-FIX (sim-to-reality gap):
        #   De portfolio-timestamps zijn CUSUM-event bars (onregelmatig,
        #   gem. ~4× dichter dan 1h-bars).  bars_per_year=8760 (1h-aanname)
        #   onderschat de annualisatiefactor met sqrt(4.27)≈2×, waardoor:
        #     (a) realized_vol 2× te laag gerapporteerd → vol_mult te hoog,
        #     (b) Sharpe 2× te hoog gerapporteerd (scheef beeld van performance).
        #   Fix: bereken de echte bars-per-year uit de portfolio-grid zelf.
        if len(timestamps) >= 2:
            ts0 = pd.Timestamp(timestamps[0])
            ts1 = pd.Timestamp(timestamps[-1])
            elapsed_years = max((ts1 - ts0).total_seconds() / (365.25 * 86400), 1e-9)
            actual_bpy = float(len(timestamps)) / elapsed_years
            if actual_bpy > 1.0:
                old_bpy = self.bars_per_year
                self.risk_manager.bars_per_year = actual_bpy
                self.bars_per_year = actual_bpy
                logger.info(
                    "BARS-PER-YEAR-FIX: portfolio grid = %d bars over %.2f yr "
                    "-> actual bpy=%.0f (was %.0f).",
                    len(timestamps), elapsed_years, actual_bpy, old_bpy,
                )

        # AUDIT-FIX (Round 3 — Funding Δt): lookup per asset voor funding_rate_is_dense vlag.
        tracks_by_sym: dict[str, AssetTrack] = {t.symbol: t for t in tracks}
        per_asset_caps: dict[str, float] = {t.symbol: float(t.per_asset_cap) for t in tracks}
        # REBALANCE-COST-FIX: per-asset éénzijdige fee in decimal (1bp = 0.0001).
        per_asset_cost: dict[str, float] = {
            t.symbol: max(float(t.cost_bps), 0.0) * 1e-4 for t in tracks
        }

        equity_curve: list[float] = []
        portfolio_returns: list[float] = []
        gross_history: list[float] = []
        net_history: list[float] = []
        corr_history: list[float] = []
        dd_breaker_count: int = 0

        # REBALANCE-COST + FUNDING running totals
        total_rebalance_cost: float = 0.0
        total_funding_cost: float = 0.0
        turnover_history: list[float] = []

        # Per-asset cumulative contribution to portfolio PnL (voor decomposition)
        contribution: dict[str, float] = {s: 0.0 for s in symbols}

        # REBALANCE-COST-FIX: track previous bar signed exposure per asset.
        prev_signed: dict[str, float] = {s: 0.0 for s in symbols}

        # LEVERAGE-LOCK-FIX (sim-to-reality gap):
        #   Vol_target_multiplier oscilleert elke bar (0.4×–2.5×) want de
        #   realized-vol wordt herberekend op elke CUSUM-event-bar.  Gevolg:
        #   binnen een trade wijzigt final_leverage elke bar → delta_signed ≠ 0
        #   → 5 bps cost elke bar → ~30–100 % jaarlijkse "rebalance"-kosten.
        #
        #   In live trading wordt vol-target NIET op elke bar geherbalanceerd;
        #   een positie heeft een vaste leverage vanaf entry tot exit.
        #   Fix: size_position wordt ENKEL aangeroepen bij side-change (nieuwe
        #   entry of exit).  Binnen een lopende trade wordt de leverage gelockt.
        #   DD-breaker wordt ook bij elke bar gecheckt zodat circuit-breaker
        #   bestaande posities sluit.
        prev_side_by_sym:       dict[str, int]   = {s: 0   for s in symbols}
        entry_locked_signed:    dict[str, float] = {s: 0.0 for s in symbols}

        # ── BLUEPRINT-FIX (IV.7): per-asset fill-metadata buffers, vooraf
        # gealigneerd op de gemeenschappelijke timestamp-grid. Wij doen dit
        # éénmalig vóór de hot loop zodat de inner-loop alleen indexering
        # hoeft te doen. Asset zonder ``fill_metadata`` krijgt None →
        # eta_observer.update wordt voor die asset overgeslagen.
        fill_meta_aligned: dict[str, pd.DataFrame | None] = {}
        for t in tracks:
            fm = t.fill_metadata
            if fm is None:
                fill_meta_aligned[t.symbol] = None
                continue
            try:
                fm_df = pd.DataFrame(
                    {
                        "realised_slippage": np.asarray(
                            fm.get("realised_slippage", np.zeros(len(t.timestamps))),
                            dtype=np.float64,
                        ),
                        "sigma_per_bar": np.asarray(
                            fm.get("sigma_per_bar", np.zeros(len(t.timestamps))),
                            dtype=np.float64,
                        ),
                        "order_size": np.asarray(
                            fm.get("order_size", np.zeros(len(t.timestamps))),
                            dtype=np.float64,
                        ),
                        "bar_volume": np.asarray(
                            fm.get("bar_volume", np.zeros(len(t.timestamps))),
                            dtype=np.float64,
                        ),
                    },
                    index=t.timestamps,
                ).reindex(timestamps, fill_value=0.0)
                fill_meta_aligned[t.symbol] = fm_df
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "fill_metadata voor %s niet bruikbaar (%s) — η-update uit.",
                    t.symbol, exc,
                )
                fill_meta_aligned[t.symbol] = None

        # ── BLUEPRINT-FIX (V.8): schema-guard kill-switch state.
        _schema_guard = (
            self.stack.schema_guard
            if self.stack is not None and getattr(self.stack, "schema_guard", None) is not None
            else None
        )
        _eta_observer = (
            self.stack.eta_observer
            if self.stack is not None and getattr(self.stack, "eta_observer", None) is not None
            else None
        )
        _net_alpha = (
            self.stack.net_alpha
            if self.stack is not None and getattr(self.stack, "net_alpha", None) is not None
            else None
        )

        for ts in timestamps:
            # ── BLUEPRINT-FIX (V.8): schema-guard per bar; mismatch ⇒
            # raises SchemaMismatchError direct door naar de caller, die
            # alle trading moet pauzeren. We loggen op ERROR vóór raise
            # zodat de orchestrator-monitor het oppikt.
            if (
                self.check_schema_per_bar
                and _schema_guard is not None
                and self.live_feature_names_provider is not None
            ):
                try:
                    live_names = self.live_feature_names_provider(ts)
                except Exception as exc:
                    logger.error(
                        "live_feature_names_provider raised (%s) at ts=%s — "
                        "trading wordt gepauzeerd.",
                        exc, ts,
                    )
                    raise SchemaMismatchError(
                        f"feature_names provider faalde op {ts}: {exc}"
                    ) from exc
                _schema_guard.check(live_names)
            # 1. Voor elke asset: bepaal scaled leverage via PortfolioRiskManager.
            #    LEVERAGE-LOCK: size_position wordt ALLEEN aangeroepen bij een
            #    side-change (nieuwe entry of exit). Binnen een lopende trade
            #    blijft de leverage gelockt op de entry-leverage.  Zo vermijden
            #    we fake intra-trade rebalance-kosten door vol_mult-oscillatie.
            scaled_signed: dict[str, float] = {}
            for s in symbols:
                req_lev  = float(aligned[s].at[ts, "requested_leverage"])
                side     = int(aligned[s].at[ts, "side"])
                prev_side = prev_side_by_sym.get(s, 0)

                if side == 0:
                    # Flat: sluit eventueel open locked positie.
                    scaled_signed[s] = 0.0
                    entry_locked_signed[s] = 0.0
                elif side == prev_side and entry_locked_signed.get(s, 0.0) != 0.0:
                    # CONTINUATION — zelfde richting als vorige bar.
                    # Check DD-breaker zodat circuit-breaker bestaande positie
                    # alsnog kan sluiten (maar geen nieuwe size_position call).
                    if self.risk_manager.dd_state():
                        scaled_signed[s] = 0.0
                        entry_locked_signed[s] = 0.0
                    else:
                        scaled_signed[s] = entry_locked_signed[s]
                else:
                    # NIEUWE ENTRY (of side-flip): bereken leverage éénmalig.
                    if req_lev <= 0.0:
                        scaled_signed[s] = 0.0
                        entry_locked_signed[s] = 0.0
                    else:
                        decision = self.risk_manager.size_position(
                            symbol=s,
                            raw_leverage=req_lev,
                            side=side,
                            per_asset_cap_override=per_asset_caps.get(s),
                        )
                        if decision.blocked:
                            scaled_signed[s] = 0.0
                            entry_locked_signed[s] = 0.0
                        else:
                            scaled_signed[s] = decision.final_leverage * side
                            entry_locked_signed[s] = scaled_signed[s]

                prev_side_by_sym[s] = side

            # 2. Bereken per-asset bar-return na portfolio scaling.
            # Belangrijke insight: signed_returns zijn al per-bar PnL bij
            # leverage=1.0. Dus de echte portfolio-bijdrage = scaled_signed *
            # signed_returns / sign(signed_returns) — maar de "side" is al
            # in signed_return ingebakken. We schalen dus puur met |scaled|.
            #
            # REBALANCE-COST-FIX: het verschil in signed exposure t.o.v. de
            # vorige bar moet betaald worden tegen ``cost_bps`` (vol-target en
            # correlatie-scaler herbalanceren niet "gratis"). FUNDING-FIX:
            # signed * funding_rate wordt ook hier afgerekend.
            returns_per_asset_for_update: dict[str, float] = {}
            bar_rebalance_cost: float = 0.0
            bar_funding_cost: float = 0.0
            bar_turnover: float = 0.0
            for s in symbols:
                base_r = float(aligned[s].at[ts, "signed_return"])
                signed_now = float(scaled_signed.get(s, 0.0))
                lev = abs(signed_now)
                applied_r = base_r * lev
                contribution[s] += applied_r

                # ---- REBALANCE COST ----
                delta_signed = abs(signed_now - prev_signed.get(s, 0.0))
                rebalance_cost_s = delta_signed * per_asset_cost.get(s, 0.0)
                bar_rebalance_cost += rebalance_cost_s
                bar_turnover += delta_signed

                # ---- FUNDING ----
                # AUDIT-FIX (Round 3 — Funding Δt correctie):
                # Twee conventies voor funding_rate zijn mogelijk:
                #   (a) SPARSE (default):  rate ≠ 0 slechts op de 8h-tick,
                #       andere bars = 0. Aangemaakt door _load_per_bar_funding_rate.
                #       Correct zonder Δt: signed × F geeft de volledige 8h-fee.
                #   (b) DENSE (ffill'd):  volledige 8h-rate herhaald op élke bar.
                #       Vereist Δt-schaling: signed × F × Δt, want anders wordt
                #       de 8h-fee 32× geboekt bij 15-min bars (32 bars/periode).
                # De vlag ``funding_rate_is_dense`` (nieuw veld in AssetTrack)
                # schakelt de Δt-schaling in voor dense data.
                # Long betaalt funding bij positieve rate; short ontvangt het.
                fr = float(aligned[s].at[ts, "funding_rate"])
                # CHIEF AUDIT 2026-05-23: funding cap voorkomt over-modellering
                # tijdens dislocaties (USTC 2022 piek -2%/8h). Zonder cap kan
                # een enkele exotische 8h-tick (bv. -8%/8h voor delisting) de
                # backtest 5-10× erger laten lijken dan een realistisch
                # gerundde executie zou opleveren — de exchange limiteert in
                # de praktijk via funding-cap + insurance-fund.
                fr_capped = float(np.clip(fr, -0.02, 0.02))
                _dense = getattr(tracks_by_sym.get(s), "funding_rate_is_dense", False)
                funding_cost_s = signed_now * fr_capped * (self._funding_dt if _dense else 1.0)
                bar_funding_cost += funding_cost_s

                # Voor de risk manager: feed de PRICE-direction return zodat
                # vol-target en correlatie op zuivere asset-vol worden berekend.
                # signed_returns in de track zijn PnL-gesigneerd (positief = winst
                # ongeacht richting); de risk manager verwacht prijs-richting-returns
                # zodat port_r = sum(signed_exposure × price_return) correct is.
                # Conversie: price_return = pnl_return × sign(side).
                side_val = int(aligned[s].at[ts, "side"])
                if side_val != 0:
                    raw_asset_r = base_r * side_val
                else:
                    raw_asset_r = 0.0
                returns_per_asset_for_update[s] = raw_asset_r

                # ── BLUEPRINT-FIX (IV.7): η-update na elke non-zero fill.
                # Een "fill" = bar waar de exposure ≠ 0 was OF veranderde.
                # Wanneer fill_metadata ontbreekt of de stack niet is gezet,
                # blijft dit pad een no-op (legacy gedrag).
                if _eta_observer is not None and delta_signed > 1e-12:
                    fm_df = fill_meta_aligned.get(s)
                    if fm_df is not None:
                        try:
                            row = fm_df.loc[ts]
                            slip = float(row["realised_slippage"])
                            sigma_bar = float(row["sigma_per_bar"])
                            q_size = float(row["order_size"])
                            v_bar = float(row["bar_volume"])
                            if q_size > 0.0 and v_bar > 0.0 and sigma_bar > 0.0:
                                _eta_observer.update(
                                    realised_slippage=slip,
                                    sigma_per_bar=sigma_bar,
                                    order_size=q_size,
                                    bar_volume=v_bar,
                                )
                                # Sync de NetAlphaReward η zodat de volgende
                                # reward-berekening met de actuele observatie
                                # werkt — exact wat de orchestrator-spec
                                # ("lees stack.eta_observer.eta als parameter
                                # voor de volgende NetAlphaReward.compute()")
                                # voorschrijft.
                                if _net_alpha is not None:
                                    _net_alpha.eta = float(_eta_observer.eta)
                        except Exception as exc:  # pragma: no cover
                            logger.debug(
                                "eta_observer.update faalde op %s/%s (%s).",
                                s, ts, exc,
                            )

                prev_signed[s] = signed_now

            total_rebalance_cost += bar_rebalance_cost
            total_funding_cost += bar_funding_cost
            turnover_history.append(bar_turnover)

            port_gross_ret_this_bar = sum(
                float(aligned[s].at[ts, "signed_return"]) * abs(scaled_signed.get(s, 0.0))
                for s in symbols
            )
            # NETTE bar-return = bruto-PnL − herbalanceer-kosten − funding.
            port_ret_this_bar = (
                port_gross_ret_this_bar - bar_rebalance_cost - bar_funding_cost
            )

            # 3. Update PortfolioRiskManager met dezelfde signed exposures + raw returns.
            # AUDIT-FIX (N24 — equity mutation anti-pattern):
            #   bar_costs doorgegeven aan update() zodat equity ATOMAIR wordt bijgewerkt:
            #   equity *= (1 + port_r - bar_costs).  Externe mutatie na update() verwijderd;
            #   peak_equity en equity_curve[-1] zijn altijd consistent met de netto-PnL.
            state: RiskState = self.risk_manager.update(
                timestamp=ts,
                returns_per_asset=returns_per_asset_for_update,
                active_signed_exposure=scaled_signed,
                bar_costs=bar_rebalance_cost + bar_funding_cost,
            )

            equity_curve.append(self.risk_manager.equity)
            portfolio_returns.append(port_ret_this_bar)
            gross_history.append(state.gross_leverage)
            net_history.append(state.net_leverage)
            corr_history.append(state.avg_pairwise_corr)
            if state.dd_breaker_active:
                dd_breaker_count += 1

        # ---------------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------------
        eq_series = pd.Series(equity_curve, index=timestamps, name="equity")
        ret_series = pd.Series(portfolio_returns, index=timestamps, name="portfolio_return")

        m = self.risk_manager.equity_metrics()

        # CHIEF AUDIT 2026-05-28 (M-1): Deflated Sharpe must use the TRUE
        # multiple-testing burden.  Prefer the explicitly-supplied
        # ``total_n_hypotheses`` (n_assets × n_sides × n_optuna_trials …).
        # Only fall back to the conservative ``N_assets × 2`` floor when the
        # caller did not plumb the real count — and log loudly so the
        # under-counted DSR is never silently trusted.
        _floor_n = max(1, len(tracks)) * 2
        if self.total_n_hypotheses is not None and self.total_n_hypotheses > _floor_n:
            total_n_hypotheses = self.total_n_hypotheses
        else:
            total_n_hypotheses = _floor_n
            logger.warning(
                "Deflated Sharpe using FLOOR n_hypotheses=%d (n_assets×2). "
                "Pass total_n_hypotheses=n_assets×n_sides×n_optuna_trials to "
                "PortfolioBacktester for a correctly-deflated DSR.",
                total_n_hypotheses,
            )
        logger.info("Deflated Sharpe: total_n_hypotheses=%d", total_n_hypotheses)
        deflated = _deflated_sharpe(
            sharpe=m["sharpe"],
            n_obs=eq_series.size,
            total_n_hypotheses=total_n_hypotheses,
        )

        bootstrap_ci = _bootstrap_sharpe(
            ret_series, n_boot=500, avg_block_size=20.0,
            bars_per_year=self.bars_per_year,
        )

        return PortfolioBacktestResult(
            equity_curve=eq_series,
            portfolio_returns=ret_series,
            sharpe=m["sharpe"],
            max_dd=m["max_dd"],
            calmar=m["calmar"],
            total_return=m["total_return"],
            deflated_sharpe=deflated,
            bootstrap=bootstrap_ci,
            per_asset_contribution=contribution,
            avg_gross_leverage=float(np.mean(gross_history)) if gross_history else 0.0,
            avg_net_leverage=float(np.mean(net_history)) if net_history else 0.0,
            avg_corr=float(np.mean(corr_history)) if corr_history else 0.0,
            realized_vol=self.risk_manager._realized_portfolio_vol(),
            n_dd_breaker_bars=dd_breaker_count,
            n_bars=eq_series.size,
            total_rebalance_cost=float(total_rebalance_cost),
            total_funding_cost=float(total_funding_cost),
            avg_turnover=float(np.mean(turnover_history)) if turnover_history else 0.0,
        )


# =============================================================================
# UTILITIES
# =============================================================================
def _deflated_sharpe(
    sharpe: float,
    n_obs: int,
    total_n_hypotheses: int | None = None,
    *,
    n_trials: int | None = None,  # legacy alias, prefer total_n_hypotheses
) -> float:
    """Bailey & Lopez de Prado 2014 deflated Sharpe ratio (eenvoudige variant).

    Penaliseert Sharpe voor multi-trial bias. Volledige formule vereist
    skewness/kurtosis van de strategie; we gebruiken de Gaussian-approximatie
    die in evaluation_realism.deflated_sharpe_penalty zit voor consistentie.

    Parameters
    ----------
    sharpe :
        Observed annualised Sharpe.
    n_obs :
        Number of out-of-sample observations.
    total_n_hypotheses :
        CHIEF AUDIT 2026-05-23 (M13): TOTAAL aantal hypotheses dat is
        getest tijdens search/selection. De caller MOET dit doorgeven als
        ``n_assets × n_sides × n_optuna_trials × n_prior_experiments``.
        De interne ×2 hack uit de oude implementatie is verwijderd:
        callers die alleen ``len(tracks)`` doorgeven onderschatten N
        systematisch (factor 100-1000× voor full grid search).
    n_trials :
        DEPRECATED — legacy alias voor ``total_n_hypotheses``. Wanneer
        ``total_n_hypotheses`` ontbreekt valt de functie hierop terug.
    """
    # CHIEF AUDIT 2026-05-23 (M13): aliasing voor backward-compat.
    if total_n_hypotheses is None:
        if n_trials is None:
            raise TypeError(
                "_deflated_sharpe: pass total_n_hypotheses (preferred) or "
                "the legacy n_trials= alias."
            )
        total_n_hypotheses = int(n_trials)
    n_trials_adj = max(int(total_n_hypotheses), 2)
    if n_obs < 2 or n_trials_adj <= 1:
        return float(sharpe)
    # Verwachte maximum Sharpe over N onafhankelijke trials onder H0 (no edge).
    # Approx: E[max] ≈ sqrt(2 ln N) — Gaussian extreme value.
    e_max = float(np.sqrt(2.0 * np.log(max(n_trials_adj, 2))))
    # P2.1-FIX (CHIEF AUDIT 2026-05-23): The Gaussian SE formula
    #   SE = sqrt((1 + 0.5*S²) / N)
    # assumes normally distributed returns.  Crypto returns have negative
    # skewness (~-0.5) and excess kurtosis (~6), which inflates the true
    # variance of the Sharpe estimator by ~1.6-1.8× relative to Gaussian
    # (Mertens 2002).  Applying a conservative 1.5× SE correction prevents
    # inflated strategies from passing the DSR gate.
    #
    # Full formula (Bailey-LdP with skew/kurt): see metrics.deflated_sharpe.
    # That function requires the return series; here we only have scalar S,
    # so we apply the correction factor as a safe approximation.
    _CRYPTO_SR_SE_CORRECTION: float = 1.5
    se = float(
        np.sqrt(max(1.0 + 0.5 * sharpe * sharpe, 1e-9) / n_obs)
    ) * _CRYPTO_SR_SE_CORRECTION
    deflated = (sharpe - e_max * se)
    return float(deflated)


def _bootstrap_sharpe(
    returns: pd.Series,
    n_boot: int = 500,
    avg_block_size: float = 20.0,
    bars_per_year: float = 365.0 * 24 * 12,
    seed: int = 42,
) -> dict[str, float]:
    """Stationary block bootstrap CI op portfolio Sharpe.

    Hergebruikt de logica uit evaluation_realism.block_bootstrap_path_sharpes
    maar dan op het 1-pads portfolio-resultaat.
    """
    arr = returns.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 30:
        return {"sharpe_mean": 0.0, "sharpe_std": 0.0, "sharpe_lower_5": 0.0, "sharpe_upper_95": 0.0}

    rng = np.random.default_rng(seed)
    p = 1.0 / max(avg_block_size, 1.0)

    sharpes: list[float] = []
    for _ in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            blk = max(int(rng.geometric(p)), 1)
            end = min(i + blk, n)
            for k in range(end - i):
                idx[i + k] = (start + k) % n
            i = end
        sample = arr[idx]
        mu = sample.mean()
        sigma = sample.std(ddof=1) if sample.size > 1 else 0.0
        if sigma > 1e-12:
            sharpes.append((mu / sigma) * np.sqrt(bars_per_year))

    if not sharpes:
        return {"sharpe_mean": 0.0, "sharpe_std": 0.0, "sharpe_lower_5": 0.0, "sharpe_upper_95": 0.0}
    return {
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "sharpe_lower_5": float(np.quantile(sharpes, 0.05)),
        "sharpe_upper_95": float(np.quantile(sharpes, 0.95)),
    }


# =============================================================================
# HELPER: BUILD AssetTrack VANUIT train_regime backtest output
# =============================================================================
