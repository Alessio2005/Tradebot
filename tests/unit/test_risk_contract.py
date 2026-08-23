"""Het L7 risicocontract — de invarianten uit `docs/RISK_CONTRACT.md`.

Phase 4, stap 2. Deze suite bewijst niet dat de engine goede besluiten neemt;
hij bewijst dat een SLECHT besluit niet uitdrukbaar is. Dat onderscheid is de
reden dat de invarianten in de dataclasses zitten en niet in de engine: een
latere refactor van `engine.py` kan ze niet per ongeluk kwijtraken.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from tradebot.risk.contract import (
    BOOK_SCOPE,
    BindingConstraint,
    ConstraintKind,
    MarketState,
    RiskDecision,
    RiskState,
    validate_desired_exposure,
)
from tradebot.utils.failfast import DataContractError

UTC_TS = pd.Timestamp("2026-01-01", tz="UTC")


def _state(equity: float = 1.0, hwm: float = 1.0, day: float = 1.0) -> RiskState:
    return RiskState(equity=equity, high_water_mark=hwm, day_start_equity=day)


def _constraint(before: float, after: float) -> BindingConstraint:
    return BindingConstraint(
        kind=ConstraintKind.GROSS_CAP,
        scope=BOOK_SCOPE,
        measured=before,
        threshold=after,
        exposure_before=before,
        exposure_after=after,
        config_key="risk.gross_cap",
    )


class TestDesiredExposureIsTheL4Contract:
    """`a_t in [-1, +1]`, en een schending crasht in plaats van te clippen."""

    def test_valid_range_passes_through_unchanged(self) -> None:
        assert validate_desired_exposure({"A": 1.0, "B": -1.0, "C": 0.0}) == {
            "A": 1.0, "B": -1.0, "C": 0.0,
        }

    @pytest.mark.parametrize("bad", [1.4, -1.4, float("nan"), float("inf")])
    def test_out_of_contract_crashes_and_is_not_clipped(self, bad: float) -> None:
        with pytest.raises(DataContractError):
            validate_desired_exposure({"A": bad})

    def test_empty_input_crashes(self) -> None:
        with pytest.raises(DataContractError):
            validate_desired_exposure({})


class TestRiskStateIsCausalAndFailFast:
    def test_drawdown_and_daily_loss_are_positive_fractions(self) -> None:
        st = _state(equity=0.8, hwm=1.0, day=0.9)
        assert st.drawdown == pytest.approx(0.20)
        assert st.daily_loss == pytest.approx(1.0 - 0.8 / 0.9)

    def test_high_water_mark_below_equity_crashes(self) -> None:
        """De HWM is monotoon niet-dalend; het omgekeerde duidt op externe mutatie."""
        with pytest.raises(DataContractError):
            RiskState(equity=1.5, high_water_mark=1.0, day_start_equity=1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
    def test_non_finite_or_non_positive_equity_crashes(self, bad: float) -> None:
        with pytest.raises(DataContractError):
            RiskState(equity=bad, high_water_mark=1.0, day_start_equity=1.0)

    def test_halted_without_reason_crashes(self) -> None:
        """Een HALTED-staat zonder reden is niet auditbaar."""
        with pytest.raises(DataContractError):
            RiskState(
                equity=1.0, high_water_mark=1.0, day_start_equity=1.0,
                halted=True, halt_reason="",
            )

    def test_state_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _state().equity = 2.0  # type: ignore[misc]


class TestMarketStateCarriesNoStrategyInformation:
    def test_naive_timestamp_crashes(self) -> None:
        with pytest.raises(DataContractError):
            MarketState(asof_ts=pd.Timestamp("2026-01-01"), sigma_hat={"A": 0.5})

    def test_mappings_are_read_only_after_construction(self) -> None:
        """Puurheid: een consument mag de marktstaat niet na constructie bijstellen."""
        ms = MarketState(asof_ts=UTC_TS, sigma_hat={"A": 0.5})
        with pytest.raises(TypeError):
            ms.sigma_hat["A"] = 9.9  # type: ignore[index]

    def test_mutating_the_source_dict_does_not_leak_into_the_state(self) -> None:
        src = {"A": 0.5}
        ms = MarketState(asof_ts=UTC_TS, sigma_hat=src)
        src["A"] = 9.9
        assert ms.sigma_hat["A"] == 0.5

    def test_carries_no_alpha_fields(self) -> None:
        """Sectie 14: L7 kent geen strategie-identiteit en geen verwachte alpha.

        Deze test is de vroege waarschuwing voor stap 9. Zodra iemand een veld
        met alpha-betekenis aan de marktstaat toevoegt, faalt hij hier al -
        voordat de volledige ontkoppelingstest er is.
        """
        fields = {f.name for f in dataclasses.fields(MarketState)}
        forbidden = {
            "alpha", "expected_alpha", "edge", "mu", "signal", "confidence",
            "model", "model_name", "unit", "unit_name", "strategy", "conviction",
            "prediction", "forecast", "score", "sharpe", "hitrate",
        }
        assert not (fields & forbidden), f"alpha-veld in MarketState: {fields & forbidden}"


class TestBindingConstraintOnlyShrinks:
    def test_shrinking_is_allowed(self) -> None:
        assert _constraint(1.0, 0.5).exposure_after == 0.5

    def test_growing_exposure_crashes(self) -> None:
        """Sectie 5.3: de risicolaag verkleint uitsluitend."""
        with pytest.raises(DataContractError):
            _constraint(0.5, 0.9)

    def test_record_is_json_serialisable_and_names_its_config_key(self) -> None:
        rec = _constraint(1.0, 0.5).as_record()
        assert rec["kind"] == "gross_cap"
        assert rec["config_key"] == "risk.gross_cap"
        assert set(rec) == {
            "kind", "scope", "measured", "threshold",
            "exposure_before", "exposure_after", "config_key",
        }


class TestRiskDecisionRegistersEverything:
    def test_unconstrained_requires_an_empty_audit_trail(self) -> None:
        with pytest.raises(DataContractError):
            RiskDecision(
                permitted_exposure={"A": 1.0}, binding_constraints=(),
                unconstrained=False, config_hash="h", risk_state_out=_state(),
            )

    def test_a_bound_constraint_forbids_claiming_unconstrained(self) -> None:
        """Deliverable 4: nooit `a_t` ongewijzigd doorgeven zonder registratie."""
        with pytest.raises(DataContractError):
            RiskDecision(
                permitted_exposure={"A": 0.5},
                binding_constraints=(_constraint(1.0, 0.5),),
                unconstrained=True, config_hash="h", risk_state_out=_state(),
            )

    def test_missing_config_hash_crashes(self) -> None:
        with pytest.raises(DataContractError):
            RiskDecision(
                permitted_exposure={"A": 1.0}, binding_constraints=(),
                unconstrained=True, config_hash="", risk_state_out=_state(),
            )

    def test_gross_and_net_are_computed_from_the_permitted_book(self) -> None:
        d = RiskDecision(
            permitted_exposure={"A": 0.6, "B": -0.4}, binding_constraints=(),
            unconstrained=True, config_hash="h", risk_state_out=_state(),
        )
        assert d.gross() == pytest.approx(1.0)
        assert d.net() == pytest.approx(0.2)

    def test_bound_kinds_preserves_order_and_deduplicates(self) -> None:
        vol = BindingConstraint(
            kind=ConstraintKind.VOL_TARGET, scope=BOOK_SCOPE, measured=0.72,
            threshold=0.12, exposure_before=1.0, exposure_after=0.167,
            config_key="risk.sigma_target",
        )
        d = RiskDecision(
            permitted_exposure={"A": 0.1},
            binding_constraints=(vol, _constraint(0.167, 0.1), _constraint(0.1, 0.1)),
            unconstrained=False, config_hash="h", risk_state_out=_state(),
        )
        assert d.bound_kinds == (ConstraintKind.VOL_TARGET, ConstraintKind.GROSS_CAP)

    def test_record_is_a_complete_machine_readable_audit_trail(self) -> None:
        import json

        d = RiskDecision(
            permitted_exposure={"A": 0.5},
            binding_constraints=(_constraint(1.0, 0.5),),
            unconstrained=False, config_hash="deadbeef", risk_state_out=_state(),
        )
        rec = d.as_record()
        json.dumps(rec)  # crasht wanneer er iets niet-serialiseerbaars in zit
        assert rec["config_hash"] == "deadbeef"
        assert rec["binding_constraints"][0]["kind"] == "gross_cap"
        assert rec["risk_state_out"]["halted"] is False
