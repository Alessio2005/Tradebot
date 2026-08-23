"""De HALTED-toestand herstelt niet vanzelf — ook niet als de markt herstelt.

Phase 4, deliverable 10 / exit-criterium 5. Dit is de scherpste test van de
kill switches: niet "vuurt hij?", maar "blijft hij vuren?".

Vóór deze fase haalde geen van de vier bestaande breakers dit. De scherpste
onder hen, `live/circuit_breaker.py`, schreef zijn trip weg naar een
append-only log maar hield de TOESTAND in `self._halt_reason` - weg na een
herstart. Een kill switch die een crash niet overleeft, beschermt niet tegen
het scenario waarin hij het hardst nodig is.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tradebot.risk.contract import ConstraintKind, RiskState
from tradebot.risk.kill_switches import (
    HaltRecord,
    HaltStore,
    advance_high_water_mark,
    apply_daily_loss_governor,
    apply_drawdown_breaker,
    apply_halt,
)
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import DataContractError

CONF = Path(__file__).resolve().parents[2] / "conf"
BOOK = {"A": 1.0, "B": -0.5}
TS = pd.Timestamp("2026-01-01T00:00:00Z")


@pytest.fixture
def store(tmp_path: Path) -> HaltStore:
    return HaltStore(tmp_path / "risk_halt.json")


@pytest.fixture
def cfg() -> RiskConfig:
    return load_config(CONF / "risk/default.yaml", RiskConfig)


def _record(kind: str = "drawdown_breaker") -> HaltRecord:
    return HaltRecord(
        kind=kind, reason="test halt", measured=0.09, threshold=0.08,
        config_key="risk.max_drawdown_pct", halted_at=TS.isoformat(),
    )


class TestTheHaltSurvivesARestart:
    def test_a_fresh_store_object_still_sees_the_halt(self, tmp_path: Path) -> None:
        """Dit IS de procesherstart: een nieuw object op hetzelfde pad."""
        path = tmp_path / "risk_halt.json"
        HaltStore(path).engage(_record())

        after_restart = HaltStore(path)
        assert after_restart.is_halted()
        loaded = after_restart.load()
        assert loaded is not None
        assert loaded.reason == "test halt"
        assert loaded.config_key == "risk.max_drawdown_pct"

    def test_the_halt_is_written_to_disk_not_kept_in_memory(self, store: HaltStore) -> None:
        store.engage(_record())
        assert store.path.is_file()
        raw = json.loads(store.path.read_text(encoding="utf-8"))
        assert raw["kind"] == "drawdown_breaker"
        assert raw["threshold"] == pytest.approx(0.08)


class TestTheHaltDoesNotRecoverOnItsOwn:
    def test_a_full_equity_recovery_does_not_lift_the_halt(
        self, store: HaltStore, cfg: RiskConfig
    ) -> None:
        """De markt herstelt volledig; het boek blijft gehalt."""
        crashed = RiskState(equity=0.90, high_water_mark=1.0, day_start_equity=1.0)
        _, bound, halted = apply_drawdown_breaker(
            BOOK, crashed, levels=cfg.drawdown_breaker_levels,
            max_drawdown_pct=cfg.max_drawdown_pct, store=store, asof_ts=TS,
        )
        assert halted.halted
        assert bound[0].kind is ConstraintKind.DRAWDOWN_BREAKER

        # Equity klimt terug naar een NIEUWE High-Water Mark.
        recovered = advance_high_water_mark(halted, 1.20)
        assert recovered.drawdown == pytest.approx(0.0)
        assert recovered.halted, "de halt verdween toen de markt herstelde"
        assert store.is_halted(), "de duurzame halt verdween toen de markt herstelde"

        permitted, halt_bound = apply_halt(BOOK, recovered)
        assert permitted == {"A": 0.0, "B": 0.0}
        assert halt_bound[0].kind is ConstraintKind.HALTED

    def test_a_new_trading_day_does_not_lift_the_halt(self, store: HaltStore) -> None:
        """De Daily Loss Governor is geen dagelijkse reset-knop."""
        losing = RiskState(equity=0.96, high_water_mark=1.0, day_start_equity=1.0)
        _, _, halted = apply_daily_loss_governor(
            BOOK, losing, daily_loss_limit=0.03, store=store, asof_ts=TS,
        )
        assert halted.halted and store.is_halted()

        next_day = RiskState(
            equity=0.96, high_water_mark=1.0, day_start_equity=0.96,
            halted=halted.halted, halt_reason=halted.halt_reason,
            halted_at=halted.halted_at,
        )
        assert next_day.daily_loss == pytest.approx(0.0)
        assert next_day.halted
        assert apply_halt(BOOK, next_day)[0] == {"A": 0.0, "B": 0.0}

    def test_the_first_cause_is_never_overwritten_by_a_later_one(
        self, store: HaltStore
    ) -> None:
        """De post-mortem moet zien wat het boek stopte, niet wat er daarna raakte."""
        store.engage(_record(kind="daily_loss_governor"))
        store.engage(
            HaltRecord(
                kind="drawdown_breaker", reason="later cause", measured=0.5,
                threshold=0.08, config_key="risk.max_drawdown_pct",
                halted_at=TS.isoformat(),
            )
        )
        active = store.load()
        assert active is not None
        assert active.kind == "daily_loss_governor"
        assert active.reason == "test halt"


class TestReleaseIsManualAndAudited:
    def test_release_requires_an_operator_and_a_justification(
        self, store: HaltStore
    ) -> None:
        store.engage(_record())
        with pytest.raises(DataContractError):
            store.release(operator="", justification="looks fine now")
        with pytest.raises(DataContractError):
            store.release(operator="risk-officer", justification="   ")
        assert store.is_halted(), "een afgewezen reset mag de halt niet aantasten"

    def test_a_valid_release_lifts_the_halt_and_is_journalled(
        self, store: HaltStore
    ) -> None:
        store.engage(_record())
        released = store.release(
            operator="risk-officer", justification="post-mortem afgerond, oorzaak verholpen"
        )
        assert released.kind == "drawdown_breaker"
        assert not store.is_halted()

        entries = [
            json.loads(line)
            for line in store.journal_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [e["event"] for e in entries] == ["engage", "release"]
        assert entries[1]["operator"] == "risk-officer"
        assert entries[1]["released_halt"]["reason"] == "test halt"

    def test_releasing_when_nothing_is_halted_crashes(self, store: HaltStore) -> None:
        with pytest.raises(DataContractError):
            store.release(operator="risk-officer", justification="niets te doen")

    def test_there_is_no_automatic_unlock_path_in_the_module(self) -> None:
        """Statisch: geen enkele functie zet `halted` terug behalve `release`."""
        src = (
            Path(__file__).resolve().parents[2]
            / "src" / "tradebot" / "risk" / "kill_switches.py"
        ).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert code.count("halted=False") == 0, "een codepad zet halted expliciet op False"
        assert "except" not in code, "try/except in de kill-switch-laag"
        # `unlink` mag uitsluitend binnen release() staan.
        assert code.count(".unlink()") == 1


class TestCorruptStateIsNotReadAsPermissionToTrade:
    def test_an_empty_state_file_crashes(self, tmp_path: Path) -> None:
        path = tmp_path / "risk_halt.json"
        path.write_text("", encoding="utf-8")
        with pytest.raises(DataContractError):
            HaltStore(path).load()

    def test_a_truncated_state_file_crashes(self, tmp_path: Path) -> None:
        path = tmp_path / "risk_halt.json"
        path.write_text('{"kind": "drawdown_breaker"}', encoding="utf-8")
        with pytest.raises(DataContractError):
            HaltStore(path).load()

    def test_a_non_object_state_file_crashes(self, tmp_path: Path) -> None:
        path = tmp_path / "risk_halt.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(DataContractError):
            HaltStore(path).load()

    def test_no_file_at_all_means_not_halted(self, tmp_path: Path) -> None:
        """Alleen de afwezigheid van het bestand is toestemming."""
        assert HaltStore(tmp_path / "absent.json").load() is None


class TestTieredDegrossingBelowTheHardLimit:
    def test_the_deepest_reached_tier_wins(self, cfg: RiskConfig) -> None:
        state = RiskState(equity=0.935, high_water_mark=1.0, day_start_equity=1.0)
        assert state.drawdown == pytest.approx(0.065)
        out, bound, new_state = apply_drawdown_breaker(
            BOOK, state, levels=cfg.drawdown_breaker_levels,
            max_drawdown_pct=cfg.max_drawdown_pct,
        )
        assert out["A"] == pytest.approx(0.25)  # 0.25x, de diepste geraakte trap
        assert bound[0].threshold == pytest.approx(0.06)
        assert not new_state.halted, "een trap is de-grossing, geen halt"

    def test_the_shallow_tier_only_halves(self, cfg: RiskConfig) -> None:
        state = RiskState(equity=0.95, high_water_mark=1.0, day_start_equity=1.0)
        out, bound, _ = apply_drawdown_breaker(
            BOOK, state, levels=cfg.drawdown_breaker_levels,
            max_drawdown_pct=cfg.max_drawdown_pct,
        )
        assert out["A"] == pytest.approx(0.5)
        assert bound[0].config_key == "risk.drawdown_breaker_levels"

    def test_a_shallow_drawdown_does_not_bind(self, cfg: RiskConfig) -> None:
        state = RiskState(equity=0.99, high_water_mark=1.0, day_start_equity=1.0)
        out, bound, new_state = apply_drawdown_breaker(
            BOOK, state, levels=cfg.drawdown_breaker_levels,
            max_drawdown_pct=cfg.max_drawdown_pct,
        )
        assert out == BOOK
        assert bound == []
        assert not new_state.halted

    def test_de_grossing_precedes_the_halt_so_it_can_still_act(
        self, cfg: RiskConfig
    ) -> None:
        """De trappen liggen onder de harde grens; anders zijn ze dode code."""
        assert cfg.drawdown_breaker_levels
        assert all(t.drawdown < cfg.max_drawdown_pct for t in cfg.drawdown_breaker_levels)


class TestTheHighWaterMarkIsCausal:
    def test_it_only_ever_rises(self) -> None:
        state = RiskState(equity=1.0, high_water_mark=1.0, day_start_equity=1.0)
        state = advance_high_water_mark(state, 1.20)
        assert state.high_water_mark == pytest.approx(1.20)
        state = advance_high_water_mark(state, 0.80)
        assert state.high_water_mark == pytest.approx(1.20)
        assert state.drawdown == pytest.approx(1.0 - 0.80 / 1.20)

    def test_it_carries_the_halt_forward(self) -> None:
        halted = RiskState(
            equity=1.0, high_water_mark=1.0, day_start_equity=1.0,
            halted=True, halt_reason="eerder gehalt",
        )
        assert advance_high_water_mark(halted, 2.0).halted

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_an_invalid_equity_crashes(self, bad: float) -> None:
        state = RiskState(equity=1.0, high_water_mark=1.0, day_start_equity=1.0)
        with pytest.raises(DataContractError):
            advance_high_water_mark(state, bad)
