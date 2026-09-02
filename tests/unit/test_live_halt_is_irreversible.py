# tests/unit/test_live_halt_is_irreversible.py
"""De live-HALT vervalt niet met de klok. Stage D, C1 en C2.

WAT ER MIS WAS, EN WAAROM HET MOEILIJK TE ZIEN WAS
====================================================
`live/circuit_breaker.py` schreef zijn trips wel degelijk naar schijf en
weigerde bij het opstarten te starten met een niet-geaccordeerde trip. Dat oogt
als een correcte kill switch. Het probleem zat in één regel:

    cutoff_ts = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)

De weigering gold alleen voor trips van de laatste 24 uur. Een systeem dat
halteerde en 25 uur later opnieuw werd gestart, startte gewoon op — met een trip
die nog altijd `acknowledged: false` droeg. De halt werd niet opgeheven door een
mens; hij verliep door de klok.

`risk/kill_switches.py::HaltStore.release` zegt letterlijk waarom dat niet mag:

    "Er is geen tijdgebaseerde, geen herstel-gebaseerde en geen automatische
     ontgrendeling. Audit sectie 14 laat geen bypass toe, en een kill switch die
     vanzelf opheft is een bypass met een klok eraan."

Deze tests dwingen af dat de live-breaker diezelfde store gebruikt in plaats van
een eigen, verlopende variant.

Ref: `reports/phase7_divergence_map.md` §5; fase-opdracht Stage D-2 (C1, C2);
exit-criterium D9; no-go 9.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from tradebot.live.state import SystemState
from tradebot.risk.kill_switches import HaltStore
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import DataContractError

RISK = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)


def _config(tmp_path: Path, **overrides: object) -> CircuitBreakerConfig:
    return CircuitBreakerConfig.from_risk_config(
        RISK, halt_store_path=tmp_path / "risk_halt.json", **overrides)


def _tripped(tmp_path: Path) -> tuple[CircuitBreaker, SystemState]:
    """Een breaker die zojuist op de levenslange drawdown is afgegaan."""
    state = SystemState(equity=88_000.0, equity_peak=100_000.0)
    state.daily_pnl_open = 100_000.0
    breaker = CircuitBreaker(_config(tmp_path), state)
    now = pd.Timestamp.now(tz="UTC")
    state.last_feed_ts = now
    assert breaker.check(now) is not None, "opzet: deze breaker hoort te trippen"
    return breaker, state


class TestTheThresholdsComeFromTheSovereignPolicy:
    """C1 — geen tweede, onafhankelijke kopie van de risicodrempels."""

    def test_the_risk_thresholds_are_the_ones_from_conf(
            self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        assert cfg.max_drawdown_pct == RISK.max_drawdown_pct
        assert cfg.max_daily_loss_pct == RISK.daily_loss_limit
        assert cfg.max_position_age_h == RISK.max_position_age_h

    def test_a_stricter_policy_moves_the_breaker(self, tmp_path: Path) -> None:
        """De eigenschap die twee losse kopieën juist niet hebben."""
        strict = RISK.model_copy(update={"max_drawdown_pct": 0.02})
        cfg = CircuitBreakerConfig.from_risk_config(
            strict, halt_store_path=tmp_path / "h.json")
        assert cfg.max_drawdown_pct == 0.02


class TestTheHaltSurvivesAndDoesNotExpire:
    """C2 — de halt staat in de soevereine store, niet in een verlopend log."""

    def test_a_trip_is_written_to_the_halt_store(self, tmp_path: Path) -> None:
        _tripped(tmp_path)
        assert HaltStore(tmp_path / "risk_halt.json").is_halted()

    def test_a_fresh_breaker_refuses_to_start_after_a_trip(
            self, tmp_path: Path) -> None:
        _tripped(tmp_path)
        with pytest.raises(RuntimeError, match="HALT"):
            CircuitBreaker(_config(tmp_path),
                           SystemState(equity=100_000.0, equity_peak=100_000.0))

    def test_the_halt_does_not_expire_after_twenty_five_hours(
            self, tmp_path: Path) -> None:
        """DE REGRESSIE. Onder het oude venster van 24 uur startte dit gewoon op.

        De halt wordt hier kunstmatig 25 uur oud gemaakt; hij hoort nog steeds
        te blokkeren, want alleen een operator kan hem opheffen.
        """
        _tripped(tmp_path)
        store = HaltStore(tmp_path / "risk_halt.json")
        record = store.load()
        assert record is not None
        old = pd.Timestamp.utcnow() - pd.Timedelta(hours=25)
        store.path.write_text(
            store.path.read_text(encoding="utf-8").replace(
                record.halted_at, old.isoformat()),
            encoding="utf-8")

        with pytest.raises(RuntimeError, match="HALT"):
            CircuitBreaker(_config(tmp_path),
                           SystemState(equity=100_000.0, equity_peak=100_000.0))

    def test_without_a_trip_a_breaker_starts_normally(
            self, tmp_path: Path) -> None:
        """Negatieve controle: de poort moet ook groen kunnen zijn."""
        breaker = CircuitBreaker(
            _config(tmp_path),
            SystemState(equity=100_000.0, equity_peak=100_000.0))
        assert not breaker.is_active


class TestOnlyAnOperatorCanLift:
    def test_reset_without_an_operator_is_refused(self, tmp_path: Path) -> None:
        breaker, _ = _tripped(tmp_path)
        with pytest.raises(DataContractError):
            breaker.reset(operator="", justification="")

    def test_reset_with_an_operator_and_a_reason_lifts_the_halt(
            self, tmp_path: Path) -> None:
        breaker, _ = _tripped(tmp_path)
        breaker.reset(operator="quant-oncall",
                      justification="feed hersteld, boek handmatig afgebouwd")
        assert not HaltStore(tmp_path / "risk_halt.json").is_halted()
        assert not breaker.is_active
        # En daarna start een verse breaker weer.
        CircuitBreaker(_config(tmp_path),
                       SystemState(equity=100_000.0, equity_peak=100_000.0))

    def test_the_release_is_journalled(self, tmp_path: Path) -> None:
        breaker, _ = _tripped(tmp_path)
        breaker.reset(operator="quant-oncall", justification="post-mortem 214")
        journal = HaltStore(tmp_path / "risk_halt.json").journal_path
        text = journal.read_text(encoding="utf-8")
        assert "engage" in text and "release" in text
        assert "quant-oncall" in text

    def test_the_first_cause_is_not_overwritten_by_a_second_trip(
            self, tmp_path: Path) -> None:
        """`HaltStore.engage` beschermt de eerste oorzaak; de breaker erft dat."""
        breaker, state = _tripped(tmp_path)
        first = HaltStore(tmp_path / "risk_halt.json").load()
        assert first is not None
        state.circuit_breaker_active = False      # forceer een tweede evaluatie
        state.equity = 1.0                        # een veel ergere drawdown
        breaker.check(pd.Timestamp.now(tz="UTC"))
        second = HaltStore(tmp_path / "risk_halt.json").load()
        assert second is not None
        assert second.reason == first.reason


class TestTheHaltStateIsNotFoundRelativeToTheWorkingDirectory:
    """De noodrem mag niet afhangen van `os.getcwd()`.

    `_HALT_STORE_PATH` en `_CB_LOG_PATH` waren `Path("artefacts/...")`, dus
    relatief aan de werkdirectory. Sinds de 24-uursvensters weg zijn (zie de
    klasse hierboven) is dat bestand de ENIGE bron voor "staan we stil". Start
    het live-proces dan vanuit een andere map -- een systemd-unit zonder
    `WorkingDirectory`, een cron-job, een submap -- dan kijkt
    `_refuse_start_when_halted` in een pad dat niet bestaat, vindt geen halt, en
    begint te handelen terwijl het boek gesloten hoort te zijn. De trip die
    daarop volgt belandt vervolgens in een tweede, ongerelateerd bestand.

    Deze tests draaien in een SUBPROCES. De autouse-fixture in `conftest.py`
    wijst beide paden naar `tmp_path` -- terecht, anders laat een trippende test
    een blijvende halt achter -- maar daardoor is de echte waarde binnen de
    suite niet te zien. Alleen een verse interpreter, gestart vanuit een andere
    map, meet wat een operator werkelijk zou krijgen.
    """

    @staticmethod
    def _paths_from_a_process_started_in(cwd: Path) -> tuple[str, str]:
        import subprocess

        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, r'%s'); "
             "from tradebot.live import circuit_breaker as cb; "
             # `.resolve()` is wat de bug zichtbaar maakt. Een RELATIEF pad
             # drukt zichzelf vanuit elke map identiek af; pas opgelost tegen
             # de werkdirectory wijst het naar een ander bestand.
             "print(cb._HALT_STORE_PATH.resolve()); "
             "print(cb._CB_LOG_PATH.resolve())"
             % (ROOT / "src")],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        return out[0].strip(), out[1].strip()

    def test_a_process_started_elsewhere_finds_the_same_halt_state(
            self, tmp_path: Path) -> None:
        halt_here, log_here = self._paths_from_a_process_started_in(ROOT)
        halt_there, log_there = self._paths_from_a_process_started_in(tmp_path)
        assert halt_there == halt_here
        assert log_there == log_here

    def test_the_paths_are_absolute_and_anchored_on_the_repo(
            self, tmp_path: Path) -> None:
        halt, log = self._paths_from_a_process_started_in(tmp_path)
        assert Path(halt).is_absolute() and Path(log).is_absolute()
        assert str(tmp_path) not in halt
        assert Path(halt) == ROOT / "artefacts" / "risk" / "halt_state.json"
        assert Path(log) == ROOT / "artefacts" / "circuit_breaker.log"
