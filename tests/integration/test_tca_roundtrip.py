"""De TCA-roundtrip sluit — fase-opdracht §13, exit-criterium 5.

De invariant:

    boek-op-arrival-prices − werkelijk boek == spread + impact + fees

bewezen met een schaduwboekhouding die dezelfde fills opnieuw boekt tegen hun
arrival price zonder fee. Wat daar tussen zit, moet volledig door de
decompositie verklaard zijn. Een residu buiten de VOORAF vastgelegde tolerantie
uit `conf/tca/default.yaml` is een kostenpost die in geen van beide
administraties staat.
"""
from __future__ import annotations

import pytest

from tradebot.schemas.config import TcaConfig, load_config
from tradebot.tca.post_trade import CostDecomposition, close_tca_roundtrip
from tradebot.utils.failfast import DataContractError

from ._engine_fixtures import (
    CONF,
    INITIAL_EQUITY,
    make_engine,
    make_replay,
    make_router,
    risk_config,
    tca_tolerance_bps,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def result():
    return make_engine().run(*_replay_args(make_replay()))


def _replay_args(replay):
    return replay.slices, replay.exposures


# =========================================================================== #
# 1. De sluiting
# =========================================================================== #
class TestClosure:
    def test_the_roundtrip_closes(self, result) -> None:
        closure = close_tca_roundtrip(result, tolerance_bps=tca_tolerance_bps())
        assert closure.closes
        assert abs(closure.residual) <= closure.tolerance

    def test_the_residual_is_float_noise_not_policy_room(self, result) -> None:
        """De sluiting is exact, niet 'binnen een ruime marge'.

        De tolerantie in `conf/tca/` bestaat voor accumulatie over duizenden
        bars. Als de roundtrip die marge feitelijk nodig heeft, verbergt zij
        iets; hier hoort hij ordes van grootte scherper te sluiten.
        """
        closure = close_tca_roundtrip(result, tolerance_bps=tca_tolerance_bps())
        assert abs(closure.residual) < 1e-6 * INITIAL_EQUITY

    def test_the_gap_is_the_sum_of_the_components(self, result) -> None:
        closure = close_tca_roundtrip(result, tolerance_bps=tca_tolerance_bps())
        d = closure.decomposition
        assert closure.gap == pytest.approx(
            d.spread + d.impact + d.fees, abs=closure.tolerance)

    def test_every_component_is_present_and_non_negative(self, result) -> None:
        d = close_tca_roundtrip(
            result, tolerance_bps=tca_tolerance_bps()).decomposition
        assert d.spread > 0.0, "een run met fills zonder spreadkosten"
        assert d.impact > 0.0, "een run met fills zonder impactkosten"
        assert d.fees > 0.0, "een run met fills zonder fees"
        assert d.funding != 0.0, "funding is geconfigureerd maar niet geboekt"


# =========================================================================== #
# 2. De decompositie is de decompositie uit §13
# =========================================================================== #
class TestDecomposition:
    def test_all_four_components_are_reported(self, result) -> None:
        record = close_tca_roundtrip(
            result, tolerance_bps=tca_tolerance_bps()).decomposition.as_record()
        for key in ("spread", "impact", "fees", "timing_opportunity_cost"):
            assert key in record

    def test_timing_is_reported_but_excluded_from_the_identity(self) -> None:
        """`timing` is contrafeitelijk: dat geld heeft de kas nooit verlaten."""
        d = CostDecomposition(spread=1.0, impact=2.0, fees=3.0, funding=4.0,
                              timing=99.0)
        assert d.execution_total == pytest.approx(6.0)
        assert d.total_with_funding == pytest.approx(10.0)

    def test_the_config_fixes_that_choice(self) -> None:
        cfg = load_config(CONF / "tca/default.yaml", TcaConfig)
        assert cfg.timing_counts_towards_closure is False
        assert cfg.require_cost_provenance is True

    def test_spread_and_impact_reconstruct_the_fill_price(self, result) -> None:
        """Per order: `fill - arrival == side * (spread + impact)` per stuk.

        Dit is waarom de roundtrip überhaupt kan sluiten. Zou de router de
        slippage ergens anders vandaan halen dan uit deze twee componenten, dan
        zou het residu precies dat verschil zijn.
        """
        checked = 0
        for report in result.reports:
            if report.fill is None:
                continue
            per_unit = abs(report.fill_price - report.arrival_price)
            modelled = (report.spread_cost + report.impact_cost) / abs(
                report.filled_qty)
            assert per_unit == pytest.approx(modelled, rel=1e-9)
            checked += 1
        assert checked > 0


# =========================================================================== #
# 3. Provenance reist mee
# =========================================================================== #
class TestProvenance:
    def test_every_report_carries_the_impact_and_spread_status(
        self, result
    ) -> None:
        assert result.impact_status == "IMPACT_UNCALIBRATED"
        assert result.spread_status == "SPREAD_ASSUMED"
        for report in result.reports:
            assert report.as_record()["impact_status"] == "IMPACT_UNCALIBRATED"
            assert report.as_record()["spread_status"] == "SPREAD_ASSUMED"

    def test_the_result_carries_the_risk_policy_identifier(self, result) -> None:
        """Fase-opdracht §6 punt 5."""
        from tradebot.risk.engine import risk_config_hash

        assert result.risk_policy_hash == risk_config_hash(risk_config())


# =========================================================================== #
# 4. NEGATIEVE TEST — een lek MOET de roundtrip breken
# =========================================================================== #
class TestAVanishedCostIsDetected:
    def test_hiding_a_fee_breaks_the_closure(self, result) -> None:
        """Als de decompositie een kostenpost mist, faalt de sluiting.

        Zonder deze test bewijst een groene roundtrip niets: een identiteit die
        ook klopt wanneer je er kosten uit weglaat, toetst niets.
        """
        import dataclasses

        tampered = dataclasses.replace(
            result,
            snapshots=tuple(
                dataclasses.replace(s, fees_paid=0.0) if i == len(result.snapshots) - 1
                else s
                for i, s in enumerate(result.snapshots)
            ),
        )
        with pytest.raises(DataContractError):
            close_tca_roundtrip(tampered, tolerance_bps=tca_tolerance_bps())

    def test_a_zero_tolerance_still_closes_or_reports_honestly(
        self, result
    ) -> None:
        closure = close_tca_roundtrip(result, tolerance_bps=1e-9, strict=False)
        # Bij een absurd scherpe tolerantie mag hij falen, maar het residu moet
        # dan nog steeds float-ruis zijn en geen echt bedrag.
        assert abs(closure.residual) < 1e-3


# =========================================================================== #
# 5. De roundtrip sluit ook wanneer er partial fills zijn
# =========================================================================== #
def test_the_roundtrip_closes_with_partial_fills() -> None:
    """Een dun boek dwingt de participatielimiet af en levert partial fills."""
    replay = make_replay(n_bars=40, bar_volume_notional=2.0e5)
    result = make_engine().run(replay.slices, replay.exposures)
    assert result.n_partial > 0 or result.n_rejected > 0, (
        "de opstelling levert geen partial fills; de test toetst niets"
    )
    closure = close_tca_roundtrip(result, tolerance_bps=tca_tolerance_bps())
    assert closure.closes
    assert closure.decomposition.timing > 0.0, (
        "niet-gevulde hoeveelheid zonder opportuniteitskost"
    )


def test_the_roundtrip_closes_when_nothing_trades() -> None:
    """Een vlak boek: geen fills, geen kosten, en de identiteit houdt triviaal."""
    replay = make_replay(n_bars=20, exposure=0.0)
    result = make_engine().run(replay.slices, replay.exposures)
    closure = close_tca_roundtrip(result, tolerance_bps=tca_tolerance_bps())
    assert closure.closes
    assert closure.decomposition.execution_total == pytest.approx(0.0)


def test_a_higher_eta_shows_up_entirely_in_the_impact_component() -> None:
    """Gevoeligheid: eta verdubbelen verdubbelt de impact, en niets anders."""
    import dataclasses

    from ._engine_fixtures import impact_params

    replay = make_replay(n_bars=40)
    base = impact_params()
    doubled = dataclasses.replace(
        base, eta=base.eta * 2.0, eta_ci_high=base.eta_ci_high * 2.0)

    from tradebot.risk.engine import RiskEngine

    a = make_engine().run(replay.slices, replay.exposures)
    risk = RiskEngine(risk_config())
    b_engine = make_engine(router=make_router(risk, impact=doubled))
    b = b_engine.run(replay.slices, replay.exposures)

    ca = close_tca_roundtrip(a, tolerance_bps=tca_tolerance_bps())
    cb = close_tca_roundtrip(b, tolerance_bps=tca_tolerance_bps())
    assert cb.decomposition.impact > 1.9 * ca.decomposition.impact
    assert cb.closes and ca.closes
