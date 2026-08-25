"""Causaliteit van de authoritative engine — fase-opdracht §9, exit-criterium 8.

De eis: *"Geen lookahead. Geen toekomstige bar-close gebruiken voor een
beslissing die vóór die close werd genomen. Een order geplaatst op `t` kan niet
vullen vóór `t + execution latency`."*

WAT DEZE SUITE ANDERS DOET DAN EEN GEWONE LOOKAHEAD-TEST
--------------------------------------------------------
De gebruikelijke vorm is truncatie-invariantie: knip de reeks af en controleer
dat eerdere waarden niet veranderen. Die test staat hieronder ook, maar hij is
niet voldoende. Hij bewijst dat de engine geen toekomst LEEST; hij bewijst niet
dat een order niet in het VERLEDEN vult.

Daarom is er een tweede soort: negatieve controles die aantonen dat de engine
rood wordt zodra je hem lookahead PROBEERT te geven. Een causaliteitstest die
alleen groen kan zijn, meet niets.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from tradebot.execution.context import MarketSlice, build_context
from tradebot.execution.order_router import Order
from tradebot.utils.failfast import DataContractError

from ..integration._engine_fixtures import make_engine, make_replay

pytestmark = pytest.mark.lookahead


@pytest.fixture(scope="module")
def replay():
    return make_replay(n_bars=45)


# =========================================================================== #
# 1. Truncatie-invariantie: de engine leest geen toekomst
# =========================================================================== #
class TestTruncationInvariance:
    @pytest.mark.parametrize("keep", [20, 30, 40])
    def test_truncating_the_future_does_not_change_the_past(
        self, replay, keep: int
    ) -> None:
        """Een run over `keep` bars is een prefix van de volledige run.

        Als één beslissing een latere bar zou gebruiken, zou het afknippen van
        die bar de eerdere equity-curve veranderen.
        """
        full = make_engine().run(replay.slices, replay.exposures)
        cut = make_engine().run(replay.slices[:keep], replay.exposures)
        # De laatste bar van de korte run beslist niet meer (er is geen bar om
        # op te vullen), dus vergelijk tot en met `keep - 1`.
        pd.testing.assert_series_equal(
            cut.equity_curve.iloc[: keep - 1],
            full.equity_curve.iloc[: keep - 1],
            check_exact=False, rtol=0.0, atol=0.0,
        )

    @pytest.mark.parametrize("keep", [25, 35])
    def test_truncating_does_not_change_earlier_decisions(
        self, replay, keep: int
    ) -> None:
        full = make_engine().run(replay.slices, replay.exposures)
        cut = make_engine().run(replay.slices[:keep], replay.exposures)
        n = len(cut.decisions)
        assert n > 0
        assert [d.as_record() for d in cut.decisions] == \
               [d.as_record() for d in full.decisions[:n]]

    @pytest.mark.parametrize("keep", [25, 35])
    def test_truncating_does_not_change_earlier_fills(
        self, replay, keep: int
    ) -> None:
        full = make_engine().run(replay.slices, replay.exposures)
        cut = make_engine().run(replay.slices[:keep], replay.exposures)
        n = len(cut.reports)
        assert n > 0
        assert [r.as_record() for r in cut.reports] == \
               [r.as_record() for r in full.reports[:n]]


# =========================================================================== #
# 2. Latency: een order vult nooit op zijn eigen beslisbar
# =========================================================================== #
class TestLatency:
    def test_every_fill_is_strictly_after_its_decision(self, replay) -> None:
        result = make_engine().run(replay.slices, replay.exposures)
        assert len(result.reports) > 0
        for report in result.reports:
            assert report.order.ts_earliest_fill > report.order.ts_decision
            if report.fill is not None:
                assert report.fill.ts > report.order.ts_decision

    def test_the_fill_lands_on_the_next_bar_exactly(self, replay) -> None:
        """Latency is 1 bar op een daily grid, niet 0 en niet 2."""
        result = make_engine().run(replay.slices, replay.exposures)
        by_ts = {s.ts: i for i, s in enumerate(replay.slices)}
        for report in result.reports:
            i_decision = by_ts[report.order.ts_decision]
            i_fill = by_ts[report.order.ts_earliest_fill]
            assert i_fill == i_decision + 1

    def test_an_order_that_fills_on_its_own_bar_cannot_be_constructed(
        self,
    ) -> None:
        ts = pd.Timestamp("2024-06-01", tz="UTC")
        with pytest.raises(DataContractError):
            Order(symbol="BTCUSDT", ts_decision=ts, ts_earliest_fill=ts,
                  target_qty_delta=1.0, risk_config_hash="h")

    def test_an_order_that_fills_before_its_bar_cannot_be_constructed(
        self,
    ) -> None:
        with pytest.raises(DataContractError):
            Order(symbol="BTCUSDT",
                  ts_decision=pd.Timestamp("2024-06-02", tz="UTC"),
                  ts_earliest_fill=pd.Timestamp("2024-06-01", tz="UTC"),
                  target_qty_delta=1.0, risk_config_hash="h")

    def test_a_context_cannot_promise_a_fill_on_its_own_bar(self) -> None:
        replay = make_replay(n_bars=3)
        market = replay.slices[0]
        with pytest.raises(DataContractError):
            build_context("backtest", market, equity=100_000.0,
                          high_water_mark=100_000.0, day_start_equity=100_000.0,
                          positions={}, next_fill_ts=market.ts)


# =========================================================================== #
# 3. Negatieve controles: lookahead MOET rood worden
# =========================================================================== #
class TestLookaheadIsDetected:
    def test_poisoning_a_future_bar_changes_nothing_before_it(
        self, replay
    ) -> None:
        """De canonieke negatieve controle.

        Vervang de prijzen van de LAATSTE bar door iets extreems. Als een
        eerdere beslissing die bar zou kunnen zien, verandert de equity-curve
        vóór dat punt. Zij hoort identiek te blijven.
        """
        poisoned = list(replay.slices)
        last = poisoned[-1]
        poisoned[-1] = dataclasses.replace(
            last, marks={s: p * 10.0 for s, p in last.marks.items()})

        clean = make_engine().run(replay.slices, replay.exposures)
        dirty = make_engine().run(poisoned, replay.exposures)

        pd.testing.assert_series_equal(
            dirty.equity_curve.iloc[:-1], clean.equity_curve.iloc[:-1],
            check_exact=True,
        )
        # ... en de laatste bar MOET wel verschillen, anders toetst de
        # vergiftiging niets.
        assert dirty.equity_curve.iloc[-1] != clean.equity_curve.iloc[-1]

    def test_poisoning_a_mid_series_bar_only_affects_from_that_bar_on(
        self, replay
    ) -> None:
        k = len(replay.slices) // 2
        poisoned = list(replay.slices)
        target = poisoned[k]
        poisoned[k] = dataclasses.replace(
            target, marks={s: p * 3.0 for s, p in target.marks.items()})

        clean = make_engine().run(replay.slices, replay.exposures)
        dirty = make_engine().run(poisoned, replay.exposures)
        pd.testing.assert_series_equal(
            dirty.equity_curve.iloc[:k], clean.equity_curve.iloc[:k],
            check_exact=True,
        )
        assert dirty.equity_curve.iloc[k] != clean.equity_curve.iloc[k]

    def test_a_replay_that_goes_backwards_in_time_is_rejected(
        self, replay
    ) -> None:
        shuffled = list(replay.slices)
        shuffled[2], shuffled[5] = shuffled[5], shuffled[2]
        with pytest.raises(DataContractError):
            make_engine().run(shuffled, replay.exposures)

    def test_duplicate_timestamps_are_rejected(self, replay) -> None:
        doubled = [replay.slices[0], replay.slices[0], *replay.slices[1:]]
        with pytest.raises(DataContractError):
            make_engine().run(doubled, replay.exposures)


# =========================================================================== #
# 4. De beslissing gebruikt uitsluitend informatie van haar eigen bar
# =========================================================================== #
class TestDecisionInputsAreCausal:
    def test_the_market_state_timestamp_equals_the_decision_bar(
        self, replay
    ) -> None:
        for market in replay.slices[:-1]:
            context = build_context(
                "backtest", market, equity=100_000.0,
                high_water_mark=100_000.0, day_start_equity=100_000.0,
                positions={}, next_fill_ts=market.ts + pd.Timedelta(days=1),
            )
            assert context.market_state().asof_ts == market.ts
            assert context.asof() == market.ts

    def test_a_market_slice_requires_a_timezone_aware_timestamp(self) -> None:
        with pytest.raises(DataContractError):
            MarketSlice(
                ts=pd.Timestamp("2024-01-01"), marks={"BTCUSDT": 100.0},
                sigma_hat={"BTCUSDT": 0.5}, sigma_daily={"BTCUSDT": 0.03},
                adv_notional={"BTCUSDT": 1e9},
                bar_volume_notional={"BTCUSDT": 1e9},
                clusters={"BTCUSDT": "crypto_perp"},
                funding_rate={"BTCUSDT": 0.0},
            )

    def test_the_engine_never_marks_against_a_price_it_does_not_have(
        self, replay
    ) -> None:
        """Een symbool dat op de fillbar geen prijs heeft, crasht de run."""
        broken = list(replay.slices)
        target = broken[3]
        dropped = dict(target.marks)
        dropped.pop(next(iter(dropped)))
        broken[3] = dataclasses.replace(target, marks=dropped)
        with pytest.raises(DataContractError):
            make_engine().run(broken, replay.exposures)


# =========================================================================== #
# 5. Het boek eindigt zonder orders die nooit konden vullen
# =========================================================================== #
def test_the_last_bar_does_not_decide(replay) -> None:
    """Er is geen bar meer om op te vullen, dus er ontstaat geen order.

    Zonder deze regel zou de run eindigen met een openstaande order, en de
    engine crasht daar expliciet op in plaats van hem stil te laten vallen.
    """
    result = make_engine().run(replay.slices, replay.exposures)
    last_ts = replay.slices[-1].ts
    assert all(r.order.ts_decision < last_ts for r in result.reports)
    assert len(result.decisions) == len(replay.slices) - 1
