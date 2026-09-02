# tests/unit/test_alert_halt_severity.py
"""`HALT` is een niveau dat handelt, niet een niveau dat harder roept.

Stage D-3 / exit-criterium D4: *"`AlertSeverity` kent `HALT`; een `HALT`
schakelt uit zonder te vragen."*

WAAROM EEN VIERDE NIVEAU EN NIET GEWOON `CRITICAL`
====================================================
`CRITICAL` betekent in deze codebase *"een mens moet hiernaar kijken"*. Het is
een bericht. De drie niveaus die er stonden — INFO, WARNING, CRITICAL —
verschillen alleen in de logfunctie en de emoji; geen van drieen verandert iets
aan wat het systeem doet.

`HALT` is geen bericht maar een HANDELING: het boek gaat dicht en er wordt niet
om toestemming gevraagd. Zou hij als `CRITICAL` worden verstuurd, dan hangt het
stilleggen af van wie er wakker is. Daarom engageert dit niveau de soevereine
`HaltStore` — dezelfde store die `risk/engine.py` en de live-breaker gebruiken —
en is een `HALT`-alert zonder store een CRASH: een alarm dat belooft te
halteren en dat niet kan, is erger dan geen alarm.

Ref: fase-opdracht Stage D-3; exit-criterium D4; `risk/kill_switches.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.monitoring.alerts import Alert, AlertRouter, AlertSeverity
from tradebot.risk.kill_switches import HaltStore
from tradebot.utils.failfast import ConfigContractError


def _halt(title: str = "vol forecast onbruikbaar") -> Alert:
    return Alert(title=title, message="QLIKE ontspoord",
                 severity=AlertSeverity.HALT,
                 metadata={"measured": 9.9, "threshold": 2.0})


class TestTheLevelExists:
    def test_halt_is_a_severity(self) -> None:
        assert AlertSeverity.HALT.value == "HALT"

    def test_there_are_four_levels(self) -> None:
        assert {s.value for s in AlertSeverity} == {
            "INFO", "WARNING", "CRITICAL", "HALT"}

    def test_halt_renders_in_the_slack_payload(self) -> None:
        payload = _halt().to_slack_payload()
        assert "[HALT]" in payload["text"]


class TestAHaltActuallyHalts:
    def test_sending_a_halt_engages_the_store(self, tmp_path: Path) -> None:
        store = HaltStore(tmp_path / "halt.json")
        AlertRouter(halt_store=store).send(_halt())
        assert store.is_halted()

    def test_the_halt_record_carries_the_alert(self, tmp_path: Path) -> None:
        store = HaltStore(tmp_path / "halt.json")
        AlertRouter(halt_store=store).send(_halt("feed weg"))
        record = store.load()
        assert record is not None
        assert "feed weg" in record.reason
        assert record.kind == "alert"

    def test_a_halt_without_a_store_crashes(self) -> None:
        """Een alarm dat belooft te halteren en dat niet kan, is een leugen."""
        with pytest.raises(ConfigContractError, match="HaltStore"):
            AlertRouter().send(_halt())

    def test_the_first_halt_cause_is_kept(self, tmp_path: Path) -> None:
        store = HaltStore(tmp_path / "halt.json")
        router = AlertRouter(halt_store=store)
        router.send(_halt("eerste oorzaak"))
        router.send(_halt("tweede oorzaak"))
        record = store.load()
        assert record is not None
        assert "eerste oorzaak" in record.reason


class TestTheOtherLevelsStillDoNothing:
    """Negatieve controle: alleen HALT halteert, de rest blijft een bericht."""

    @pytest.mark.parametrize(
        "severity",
        [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL],
    )
    def test_a_non_halt_alert_leaves_the_book_open(
            self, severity: AlertSeverity, tmp_path: Path) -> None:
        store = HaltStore(tmp_path / "halt.json")
        AlertRouter(halt_store=store).send(
            Alert(title="t", message="m", severity=severity))
        assert not store.is_halted()

    def test_a_non_halt_alert_needs_no_store(self) -> None:
        AlertRouter().send(Alert(title="t", message="m",
                                 severity=AlertSeverity.CRITICAL))
