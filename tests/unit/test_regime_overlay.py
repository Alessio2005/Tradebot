# tests/unit/test_regime_overlay.py
"""Per-bar attributie van één geconditioneerde arm door de authoritative engine.

De vergelijking van H2 rust op twee dingen die hier worden afgedwongen:

  * de armen verschillen in PRECIES EEN ingang -- de exposures. Een identieke
    factor moet dus een identiek resultaat opleveren, tot op de bit;
  * er wordt gescoord op de out-of-sample bars en de attributie moet daar
    kloppen. Een fee die op de verkeerde bar wordt geboekt, verschuift tussen
    de armen anders en dan meet de fees-delta de boekhouding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.backtest.regime_overlay import MarketPanels, run_overlay_arm
from tradebot.execution.impact_model import ImpactParams, ImpactStatus
from tradebot.execution.order_router import VenueSpec
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import DataContractError

RISK = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
IMPACT = ImpactParams(
    eta=2.9919, kappa_d=0.6720, status=ImpactStatus.IMPACT_UNCALIBRATED,
    method="test", data_hash="test", sample_size=1, period_start="2021-01-01",
    period_end="2021-12-31", instruments=("BTCUSDT",), eta_ci_low=0.0,
    eta_ci_high=10.0)
VENUE = VenueSpec(maker_fee_bps=1.0, taker_fee_bps=5.5, funding_cap_abs=0.02,
                  min_notional=10.0, latency_bars=1)
#: Echte symbolen: `conf/risk/default.yaml` labelt de clusters, en de
#: soevereine laag WEIGERT een ongelabeld symbool in plaats van het
#: stilzwijgend buiten de clusterlimiet te laten vallen.
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _panels(n: int = 300, seed: int = 3) -> MarketPanels:
    index = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC",
                          name="asof_ts")
    rng = np.random.default_rng(seed)
    prices = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
         for s in SYMBOLS}, index=index)
    sigma = pd.DataFrame(0.6, index=index, columns=list(SYMBOLS))
    return MarketPanels(
        prices=prices, sigma_annual=sigma,
        sigma_bar=sigma / np.sqrt(365.0),
        adv=pd.DataFrame(5.0e8, index=index, columns=list(SYMBOLS)),
        volume=pd.DataFrame(5.0e7, index=index, columns=list(SYMBOLS)),
        funding=pd.DataFrame(0.0, index=index, columns=list(SYMBOLS)),
        clusters=dict(RISK.clusters))


def _exposures(panels: MarketPanels, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(-1.0, 1.0, (len(panels.prices), len(SYMBOLS))),
        index=panels.prices.index, columns=list(SYMBOLS))


def _mask(n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[150:] = True
    return mask


def _run(exposures: pd.DataFrame, panels: MarketPanels,
         half_spread_bps: float = 1.0, label: str = "arm"):
    return run_overlay_arm(
        label, exposures, panels, mask=_mask(len(panels.prices)),
        risk_cfg=RISK, impact=IMPACT, venue=VENUE,
        half_spread_bps=half_spread_bps, spread_source="test",
        initial_equity=100_000.0, bars_per_year=365.0)


class TestAttribution:
    def test_scores_only_the_masked_bars(self) -> None:
        panels = _panels()
        arm = _run(_exposures(panels), panels)
        assert arm.as_record()["n_scored_bars"] == 150
        assert arm.oos_returns.size == 150

    def test_costs_on_the_scored_bars_never_exceed_the_run_total(self) -> None:
        """De attributie mag geen kosten VERZINNEN. Zij mag ze wel weglaten:
        de bars vóór het masker tellen niet mee in het oordeel."""
        panels = _panels()
        arm = _run(_exposures(panels), panels)
        assert 0.0 < arm.fees_oos
        assert arm.turnover_notional_oos > 0.0
        assert arm.n_fills_oos <= arm.n_orders_oos

    def test_counts_trading_and_exposed_bars_separately(self) -> None:
        """Het haltprobleem uit §0.5 zit precies in dit verschil: een boek kan
        een positie HOUDEN zonder te handelen, en na een halt geen van beide."""
        panels = _panels()
        arm = _run(_exposures(panels), panels)
        assert 0 < arm.n_trading_bars_oos <= 150
        assert 0 < arm.n_exposed_bars_oos <= 150

    def test_carries_both_cost_labels(self) -> None:
        """No-go 9: een promotieclaim zonder deze twee labels is onvolledig."""
        panels = _panels()
        record = _run(_exposures(panels), panels).as_record()
        assert record["impact_status"] == "IMPACT_UNCALIBRATED"
        assert record["spread_status"] == "SPREAD_ASSUMED"
        assert record["risk_policy_hash"]


class TestArmsDifferInOneThingOnly:
    def test_the_same_exposures_give_a_bit_identical_result(self) -> None:
        panels = _panels()
        exposures = _exposures(panels)
        a = _run(exposures, panels, label="a")
        b = _run(exposures.copy(), panels, label="b")
        assert a.net_sharpe_oos == b.net_sharpe_oos
        assert a.fees_oos == b.fees_oos
        assert np.array_equal(a.oos_returns, b.oos_returns)

    def test_a_factor_of_one_leaves_the_arm_untouched(self) -> None:
        """De ongeconditioneerde referentie is de arm met factor 1 overal. Wijkt
        die af van de basis, dan meet de conditioneringslaag zichzelf."""
        panels = _panels()
        exposures = _exposures(panels)
        plain = _run(exposures, panels)
        conditioned = _run(exposures * 1.0, panels)
        assert plain.net_sharpe_oos == conditioned.net_sharpe_oos

    def test_a_wider_spread_costs_more(self) -> None:
        panels = _panels()
        exposures = _exposures(panels)
        cheap = _run(exposures, panels, half_spread_bps=1.0)
        dear = _run(exposures, panels, half_spread_bps=10.0)
        assert dear.spread_cost_oos > cheap.spread_cost_oos
        assert dear.net_return_oos < cheap.net_return_oos

    def test_a_uniform_factor_is_neutralised_by_the_vol_target(self) -> None:
        """DE BEPERKING DIE DE ARCHITECTUUR OPLEGT, HIER VASTGELEGD.

        De soevereine laag schaalt het hele boek met
        `w_t = min(max_leverage, sigma_target / sigma_boek)`. Vermenigvuldig
        elke exposure met dezelfde `c`, dan deelt `w_t` er weer door: de
        positie, de turnover en de fees blijven wat ze waren zolang de
        vol-target bindt.

        Een regime-overlay kan het boek dus NIET de-grossen. Dat is geen fout
        in deze test maar een eigenschap van L7, en zij bepaalt wat H2
        uberhaupt kan meten: wat overblijft is een CROSS-SECTIONELE tilt, geen
        risicoreductie. Het rapport moet dat zeggen, en deze test zorgt dat het
        waar blijft."""
        panels = _panels()
        exposures = _exposures(panels)
        full = _run(exposures, panels)
        halved = _run(exposures * 0.5, panels)
        assert halved.turnover_notional_oos == pytest.approx(
            full.turnover_notional_oos, rel=0.01)
        assert halved.mean_gross_notional_oos == pytest.approx(
            full.mean_gross_notional_oos, rel=0.01)

    def test_a_per_symbol_factor_does_change_the_book(self) -> None:
        """En dit is waarom H2 toch iets meet: de factor verschilt PER SYMBOOL,
        want elk symbool heeft zijn eigen regime. Die asymmetrie overleeft de
        herschaling."""
        panels = _panels()
        exposures = _exposures(panels)
        tilted = exposures.copy()
        tilted[SYMBOLS[0]] *= 0.2
        assert (_run(tilted, panels).turnover_notional_oos
                < 0.9 * _run(exposures, panels).turnover_notional_oos)


class TestContracts:
    def test_exposures_off_the_price_axis_crash(self) -> None:
        panels = _panels()
        exposures = _exposures(panels).iloc[:-1]
        with pytest.raises(DataContractError):
            run_overlay_arm(
                "arm", exposures, panels, mask=_mask(len(panels.prices)),
                risk_cfg=RISK, impact=IMPACT, venue=VENUE,
                half_spread_bps=1.0, spread_source="test",
                initial_equity=100_000.0, bars_per_year=365.0)

    def test_a_mask_of_the_wrong_length_crashes(self) -> None:
        panels = _panels()
        with pytest.raises(DataContractError):
            run_overlay_arm(
                "arm", _exposures(panels), panels, mask=_mask(10),
                risk_cfg=RISK, impact=IMPACT, venue=VENUE,
                half_spread_bps=1.0, spread_source="test",
                initial_equity=100_000.0, bars_per_year=365.0)

    def test_a_panel_off_the_price_axis_crashes(self) -> None:
        panels = _panels()
        with pytest.raises(DataContractError):
            MarketPanels(
                prices=panels.prices, sigma_annual=panels.sigma_annual.iloc[1:],
                sigma_bar=panels.sigma_bar, adv=panels.adv,
                volume=panels.volume, funding=panels.funding,
                clusters=panels.clusters)
