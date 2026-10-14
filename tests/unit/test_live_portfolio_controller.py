# tests/unit/test_live_portfolio_controller.py
"""`live/portfolio_controller.py`: seed, gelijke weging, optimalisatie, Grinold-Kahn-tilt en BTC-crisismultiplier."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha.base import SignalResult
from tradebot.live import portfolio_controller as pc
from tradebot.live.portfolio_controller import PortfolioController, PortfolioControllerConfig
from tradebot.portfolio.constraints import PortfolioConstraints
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import ConfigContractError

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TS = pd.Timestamp("2024-01-01", tz="UTC")


@pytest.fixture(scope="module")
def constraints() -> PortfolioConstraints:
    risk = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
    return PortfolioConstraints.from_risk_config(risk, min_weight=0.0, max_turnover=1.0)


def _controller(constraints: PortfolioConstraints, symbols=None, **cfg) -> PortfolioController:
    cfg = {"method": "erc", "min_history_bars": 20, "returns_window": 60, **cfg}
    return PortfolioController(PortfolioControllerConfig(constraints, **cfg), symbols or SYMBOLS)


def _walks(n: int = 80, seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    vols = {"BTCUSDT": 0.01, "ETHUSDT": 0.02, "SOLUSDT": 0.04}
    return {s: list(100 * np.exp(np.cumsum(rng.normal(0, v, n)))) for s, v in vols.items()}


def _signal(symbol: str, signal: float, confidence: float) -> SignalResult:
    return SignalResult(symbol, TS, signal, confidence, 10, f"{symbol}_x")


def _prices(walks: dict[str, list[float]]) -> dict[str, float]:
    return {s: w[-1] for s, w in walks.items()}


class TestConfig:
    def test_the_constraints_must_be_the_sovereign_object(self) -> None:
        with pytest.raises(ConfigContractError, match="expliciete PortfolioConstraints"):
            PortfolioControllerConfig({"max_weight": 0.5}, "erc")             # type: ignore[arg-type]

    def test_values_are_stored(self, constraints: PortfolioConstraints) -> None:
        cfg = PortfolioControllerConfig(constraints, "minvar", min_history_bars=7, returns_window=33,
                                        signal_tilt_strength=1)
        assert (cfg.method, cfg.min_history_bars, cfg.returns_window) == ("minvar", 7, 33)
        assert cfg.signal_tilt_strength == 1.0 and cfg.constraints is constraints


class TestSeed:
    def test_history_is_capped_cleaned_and_symbols_without_data_are_skipped(
        self, constraints: PortfolioConstraints, caplog: pytest.LogCaptureFixture
    ) -> None:
        ctrl = _controller(constraints, returns_window=10)
        closes = {"BTCUSDT": list(range(1, 101)), "ETHUSDT": [5.0, 0.0, -1.0, 7.0, None]}
        with caplog.at_level(logging.INFO, logger=pc.logger.name):
            ctrl.seed_price_history(closes)                                    # type: ignore[arg-type]
        assert ctrl._price_history["BTCUSDT"] == [float(p) for p in range(90, 101)]        # laatste returns_window + 1
        assert ctrl._price_history["ETHUSDT"] == [5.0, 7.0] and ctrl._price_history["SOLUSDT"] == []
        assert "seeded price_history for 2/3 symbols" in caplog.text


class TestOptimise:
    def test_with_too_little_history_the_weights_are_equal(self, constraints: PortfolioConstraints) -> None:
        ctrl = _controller(constraints)
        w = ctrl.optimise({}, {"BTCUSDT": 100.0, "ETHUSDT": 10.0, "SOLUSDT": 0.0})
        assert w.tolist() == [pytest.approx(1 / 3)] * 3 and list(w.index) == SYMBOLS
        assert [len(ctrl._price_history[s]) for s in SYMBOLS] == [1, 1, 0]       # prijs 0 wordt niet bewaard

    def test_the_weights_come_from_the_history_before_the_current_bar(self, constraints: PortfolioConstraints) -> None:
        walks = _walks()
        ctrl = _controller(constraints, signal_tilt_strength=0.0)
        ctrl.seed_price_history(walks)
        calm = ctrl.optimise({}, _prices(walks))
        shocked = _controller(constraints, signal_tilt_strength=0.0)
        shocked.seed_price_history(walks)
        crash = {**_prices(walks), "SOLUSDT": walks["SOLUSDT"][-1] * 0.1}          # de huidige bar mag de gewichten niet raken
        pd.testing.assert_series_equal(calm, shocked.optimise({}, crash))
        assert calm.sum() == pytest.approx(1.0, abs=1e-6) and (calm >= -1e-12).all()
        assert calm["BTCUSDT"] > calm["SOLUSDT"]                                   # laagste vol krijgt meer

    def test_the_history_window_slides(self, constraints: PortfolioConstraints) -> None:
        walks = _walks(70)
        ctrl = _controller(constraints, returns_window=60)
        ctrl.seed_price_history(walks)
        ctrl.optimise({}, {s: 123.0 for s in SYMBOLS})
        assert all(len(h) == 61 and h[-1] == 123.0 for h in ctrl._price_history.values())

    def test_signals_tilt_the_weights_symmetrically_around_the_base_allocation(
        self, constraints: PortfolioConstraints
    ) -> None:
        walks = _walks()
        base_ctrl = _controller(constraints, signal_tilt_strength=0.0)
        tilted = _controller(constraints, signal_tilt_strength=0.5)
        for ctrl in (base_ctrl, tilted):
            ctrl.seed_price_history(walks)
        signals = {"BTCUSDT": _signal("BTCUSDT", 1.0, 0.95), "ETHUSDT": _signal("ETHUSDT", -1.0, 0.95),
                   "SOLUSDT": None}
        base = base_ctrl.optimise(signals, _prices(walks))
        out = tilted.optimise(signals, _prices(walks))
        assert out["BTCUSDT"] > base["BTCUSDT"] and out["ETHUSDT"] < base["ETHUSDT"]
        assert out.sum() == pytest.approx(1.0, abs=1e-6) and (out >= -1e-12).all()

    def test_a_neutral_confidence_leaves_the_allocation_unchanged(self, constraints: PortfolioConstraints) -> None:
        walks = _walks()
        a, b = (_controller(constraints, signal_tilt_strength=s) for s in (0.0, 0.5))
        for ctrl in (a, b):
            ctrl.seed_price_history(walks)
        neutral = {s: _signal(s, 1.0, 0.5) for s in SYMBOLS}
        np.testing.assert_allclose(a.optimise(neutral, _prices(walks)), b.optimise(neutral, _prices(walks)), atol=1e-9)

    def test_a_tilt_that_wipes_out_every_weight_is_skipped(
        self, constraints: PortfolioConstraints, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        walks = _walks()
        ctrl = _controller(constraints, signal_tilt_strength=1.0)
        ctrl.seed_price_history(walks)
        monkeypatch.setattr(pc, "optimize", lambda *a, **kw: pd.Series(0.0, index=SYMBOLS))
        signals = {s: _signal(s, -1.0, 0.99) for s in SYMBOLS}                    # alles omlaag getild
        out = ctrl.optimise(signals, _prices(walks))
        assert (out == 0.0).all()

    def test_the_previous_weights_are_passed_on_to_the_optimiser(
        self, constraints: PortfolioConstraints, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        walks = _walks()
        ctrl = _controller(constraints, signal_tilt_strength=0.0)
        ctrl.seed_price_history(walks)
        seen = []

        def spy(returns, **kw):
            seen.append(kw["current_weights"])
            return pd.Series(1 / 3, index=SYMBOLS)
        monkeypatch.setattr(pc, "optimize", spy)
        ctrl.optimise({}, _prices(walks))
        ctrl.optimise({}, _prices(walks))
        assert seen[0] is None and isinstance(seen[1], pd.Series)

    def test_hrp_is_refused_on_the_live_path(self, constraints: PortfolioConstraints) -> None:
        walks = _walks()
        ctrl = _controller(constraints, method="hrp")
        ctrl.seed_price_history(walks)
        with pytest.raises(ConfigContractError, match="hrp_research_gate"):
            ctrl.optimise({}, _prices(walks))


class TestCrisisMultiplier:
    def _crashy(self) -> dict[str, list[float]]:
        rng = np.random.default_rng(1)
        rets = rng.normal(0.0005, 0.002, 59)
        rets[[10, 30, 50]] = [-0.15, -0.12, -0.2]                                  # zware linker staart
        btc = list(100 * np.exp(np.concatenate([[0.0], np.cumsum(rets)])))
        return {**_walks(60), "BTCUSDT": btc}

    def test_calm_markets_keep_the_multiplier_at_one(self, constraints: PortfolioConstraints) -> None:
        walks = {**_walks(60), "BTCUSDT": list(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.005, 60))))}
        ctrl = _controller(constraints)
        ctrl.seed_price_history(walks)
        ctrl.optimise({}, _prices(walks))
        assert ctrl.crisis_multiplier == 1.0 and ctrl._crisis_state_active is False

    def test_a_negatively_skewed_btc_tape_raises_the_multiplier_and_warns(
        self, constraints: PortfolioConstraints, caplog: pytest.LogCaptureFixture
    ) -> None:
        walks = self._crashy()
        ctrl = _controller(constraints)
        ctrl.seed_price_history(walks)
        with caplog.at_level(logging.WARNING, logger=pc.logger.name):
            ctrl.optimise({}, _prices(walks))
        assert ctrl.crisis_multiplier == 1.5 and ctrl._crisis_state_active and "crisis active" in caplog.text

    def test_a_short_btc_history_or_no_btc_symbol_never_triggers_the_overlay(
        self, constraints: PortfolioConstraints
    ) -> None:
        short = _controller(constraints)
        short.optimise({}, {"BTCUSDT": 100.0, "ETHUSDT": 10.0, "SOLUSDT": 1.0})
        assert short._crisis_multiplier == 1.0
        no_btc = _controller(constraints, symbols=["ETHUSDT", "SOLUSDT"])
        assert no_btc._btc_symbol is None
        walks = _walks()
        no_btc.seed_price_history(walks)
        no_btc.optimise({}, {s: walks[s][-1] for s in ("ETHUSDT", "SOLUSDT")})
        assert no_btc.crisis_multiplier == 1.0
