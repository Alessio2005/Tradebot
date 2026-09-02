# tests/unit/test_external_monitors_can_actually_halt.py
"""De monitors buiten de breaker sluiten het boek ECHT. Stage D, D3/D4/D9.

WAAROM DEZE TESTS ER NIET WAREN, EN WAT DAT VERBORG
====================================================
Stage D leverde drie groene criteria op het haltpad:

* **D3** — `tests/unit/test_background_tasks_are_held.py`: elke
  `asyncio.create_task` houdt zijn referentie vast;
* **D4** — `tests/unit/test_alert_halt_severity.py`: `AlertSeverity.HALT`
  schakelt uit zonder te vragen;
* **D9** — `tests/unit/test_live_halt_is_irreversible.py`: `HALTED` overleeft
  een procesherstart.

Alle drie waren groen terwijl `LiveSharpeMonitor.check` bij een gemeten
degradatie een `AttributeError` wierp in plaats van het boek te sluiten. Dat kon
omdat elk van die tests de VORM toetst en niet het GEDRAG: D3 is een
AST-wandeling die bewijst dat de referentie wordt vastgehouden -- niet dat de
taak kan draaien; D4 bouwt zijn router met een `HaltStore` erin en raakt de
module-level `send_alert` niet; D9 laat de breaker zichzelf trippen via
`check()` en komt langs geen enkele externe aanroeper.

Niets in de suite reed een degradatie door een ECHTE monitor een ECHTE breaker
in en keek daarna of het boek dicht was. Precies in dat gat pasten drie
defecten:

1. `sharpe_monitor` riep `circuit_breaker.trip(...)` aan, en die methode bestond
   niet -- alleen het private `_trip`. Al bij `df96805` zo, en Stage D hardde
   deze aanroep nog (DI-7, de referentie wordt vastgehouden) zonder te merken
   dat er niets was om vast te houden.
2. `exchange_status` riep `_trip("...", ts=...)` aan met een STRING. Bij
   `df96805` was dat correct -- `_trip(reason: str, ts)` -- maar Stage D gaf
   `_trip` een `_Breach` en brak daarmee de aanroeper. Een regressie in het
   exchange-haltpad, aangebracht door de wijziging die trips getypeerd maakte.
3. `_HALT_STORE_PATH` stond relatief aan de werkdirectory.

De les is dezelfde als in §4.5 van het exit-rapport: een poort die de vorm meet,
laat het geval door waarin de vorm klopt en de werking niet. Deze tests toetsen
daarom uitsluitend de uitkomst -- staat het boek dicht, en staat de reden erin.

Ref: `reports/phase7_8_exit_report.md` §4.9; fase-opdracht Stage D-2/D-3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from tradebot.live.state import SystemState
from tradebot.monitoring.sharpe_monitor import LiveSharpeMonitor
from tradebot.risk.kill_switches import HaltStore
from tradebot.schemas.config import RiskConfig, load_config

RISK = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)


def _breaker(tmp_path: Path) -> tuple[CircuitBreaker, HaltStore]:
    """Een verse breaker met een eigen, geisoleerde halt-store."""
    store = tmp_path / "risk_halt.json"
    cfg = CircuitBreakerConfig.from_risk_config(RISK, halt_store_path=store)
    breaker = CircuitBreaker(
        cfg, SystemState(equity=100_000.0, equity_peak=100_000.0))
    return breaker, HaltStore(store)


class TestTheSharpeMonitorClosesTheBook:
    """Een gemeten degradatie moet het boek sluiten, niet crashen."""

    @staticmethod
    def _degrade(breaker: CircuitBreaker) -> dict | None:
        """Voer verliezen in tot de monitor ver onder zijn baseline zit.

        Met ECHTE spreiding. Een reeks identieke verliezen zou hier ook
        halteren, maar om de verkeerde reden -- zie
        `TestAnUndefinedSharpeIsNoJudgement`. Dan zou deze test groen staan op
        een pad dat met de gemeten degradatie niets te maken heeft.
        """
        import numpy as np

        monitor = LiveSharpeMonitor(
            backtest_sharpe=1.5, backtest_sharpe_std=0.25,
            circuit_breaker=breaker, min_trades_before_halt=20,
        )
        rng = np.random.default_rng(5)
        result = None
        for r in rng.normal(-0.02, 0.05, 25):
            result = monitor.record_trade_return(float(r))
        return result

    def test_a_measured_degradation_engages_the_halt(self, tmp_path: Path) -> None:
        """DE test die ontbrak. Hij was rood op de toestand van voor de reparatie.

        Niet met een assertion maar met een `AttributeError`: `trip` bestond
        niet. Een monitor die stierf op het moment dat hij nodig was.
        """
        breaker, store = _breaker(tmp_path)
        assert not store.is_halted(), "opzet: het boek staat open"

        result = self._degrade(breaker)

        assert result is not None
        assert result["halt_triggered"] is True
        assert breaker.is_active, "de breaker meldt zichzelf niet als actief"
        assert store.is_halted(), "het boek staat nog OPEN na een halt-signaal"

    def test_the_reason_survives_into_the_store(self, tmp_path: Path) -> None:
        """Een halt zonder leesbare oorzaak maakt de post-mortem tot giswerk."""
        breaker, store = _breaker(tmp_path)
        self._degrade(breaker)

        record = store.load()
        assert record is not None
        assert "live_sharpe_degradation" in record.reason
        # De cijfers horen erbij: hoever, en tegen welke drempel.
        assert record.threshold == pytest.approx(2.0)
        assert record.measured > record.threshold

    def test_a_halt_survives_a_process_restart(self, tmp_path: Path) -> None:
        """D9, maar via de EXTERNE weg in plaats van via `check()`."""
        breaker, _ = _breaker(tmp_path)
        self._degrade(breaker)

        cfg = CircuitBreakerConfig.from_risk_config(
            RISK, halt_store_path=tmp_path / "risk_halt.json")
        with pytest.raises(Exception) as excinfo:
            CircuitBreaker(
                cfg, SystemState(equity=100_000.0, equity_peak=100_000.0))
        assert "halt" in str(excinfo.value).lower()

    def test_a_healthy_book_is_left_alone(self, tmp_path: Path) -> None:
        """Negatieve controle. Zonder deze zou 'altijd halteren' ook slagen."""
        import numpy as np

        breaker, store = _breaker(tmp_path)
        monitor = LiveSharpeMonitor(
            backtest_sharpe=1.5, backtest_sharpe_std=0.25,
            circuit_breaker=breaker, min_trades_before_halt=20,
        )
        # Een boek dat het rond zijn baseline doet: positieve drift, echte
        # spreiding. Een reeks IDENTIEKE rendementen zou hier niet deugen --
        # zie `TestAnUndefinedSharpeIsNoJudgement`.
        rng = np.random.default_rng(11)
        result = None
        for r in rng.normal(0.0095, 0.10, 25):
            result = monitor.record_trade_return(float(r))

        assert result is not None
        assert result["conclusive"] is True
        assert not result.get("halt_triggered", False)
        assert not store.is_halted()
        assert not breaker.is_active


class TestAnUndefinedSharpeIsNoJudgement:
    """Nulvariantie mag geen halt opleveren. GEMETEN, 2026-09-02.

    Deze klasse bestaat door een negatieve controle die faalde. Ik voerde 25
    winstgevende trades van +1 % in en verwachtte dat het boek open bleef; de
    monitor haltteerde. `_compute_sharpe` gaf `0.0` terug zodra de standaard-
    deviatie nul was, en 0,0 is hier niet neutraal maar het slechtste getal dat
    er bestaat: tegen een baseline van 1,5 met sigma 0,25 leest het als 6,0
    sigma degradatie, ruim over de drempel van 2,0.

    Een perfect winstgevend boek werd dus gesloten omdat het te CONSTANT was, en
    een stilstaand boek -- louter nulrendementen -- net zo goed. De poort vuurde
    precies in de gevallen waarin hij niets had gemeten.

    Dit is het spiegelbeeld van de fout in `vol_forecast_monitor`, waar "geen
    oordeel" als GROEN werd gelezen. Hier werd het als ROOD gelezen, en van de
    twee is dit de duurdere: groen laat een systeem doorlopen dat al liep, rood
    grijpt in.
    """

    @staticmethod
    def _feed(breaker: CircuitBreaker, value: float) -> dict | None:
        monitor = LiveSharpeMonitor(
            backtest_sharpe=1.5, backtest_sharpe_std=0.25,
            circuit_breaker=breaker, min_trades_before_halt=20,
        )
        result = None
        for _ in range(25):
            result = monitor.record_trade_return(value)
        return result

    @pytest.mark.parametrize(
        ("value", "what"),
        [(0.01, "een perfect winstgevend boek"),
         (0.0, "een boek dat stilstaat"),
         (-0.01, "een gestaag verliesgevend boek")],
    )
    def test_zero_variance_does_not_halt(
            self, tmp_path: Path, value: float, what: str) -> None:
        breaker, store = _breaker(tmp_path)
        result = self._feed(breaker, value)

        assert result is not None
        assert result["conclusive"] is False, what
        assert result["halt_triggered"] is False, what
        assert not store.is_halted(), f"{what} is gehalteerd op nulvariantie"
        assert not breaker.is_active, what

    def test_the_sharpe_itself_is_nan_and_not_zero(self) -> None:
        """0,0 zou een MEETRESULTAAT zijn; dit is er geen."""
        import math

        import numpy as np

        undefined = LiveSharpeMonitor._compute_sharpe(np.full(25, 0.01))
        assert math.isnan(undefined)
        # En een gewone reeks levert nog steeds een gewoon getal.
        rng = np.random.default_rng(3)
        ordinary = LiveSharpeMonitor._compute_sharpe(rng.normal(0.01, 0.1, 25))
        assert math.isfinite(ordinary)

    def test_a_real_degradation_still_halts(self, tmp_path: Path) -> None:
        """De reparatie mag de poort niet doof maken."""
        import numpy as np

        breaker, store = _breaker(tmp_path)
        monitor = LiveSharpeMonitor(
            backtest_sharpe=1.5, backtest_sharpe_std=0.25,
            circuit_breaker=breaker, min_trades_before_halt=20,
        )
        rng = np.random.default_rng(5)
        for r in rng.normal(-0.02, 0.05, 25):
            monitor.record_trade_return(float(r))

        assert store.is_halted()


class TestTheExternalTripEntryPointIsTyped:
    """`trip()` is de publieke ingang; `_Breach` blijft prive."""

    def test_trip_engages_the_store_with_its_numbers(self, tmp_path: Path) -> None:
        breaker, store = _breaker(tmp_path)
        breaker.trip("exchange_status:bybit:maintenance",
                     measured=1.0, threshold=0.0,
                     config_key="live.exchange_status")

        record = store.load()
        assert record is not None
        assert record.reason == "exchange_status:bybit:maintenance"
        assert record.config_key == "live.exchange_status"
        assert breaker.is_active

    def test_the_exchange_status_monitor_calls_it_the_way_it_exists(self) -> None:
        """De aanroeper en de handtekening horen bij elkaar te passen.

        `exchange_status.py` gaf `_trip` een string terwijl die sinds Stage D
        een `_Breach` verwacht. Statisch te zien, en niemand keek.
        """
        import ast
        import inspect

        from tradebot.live import exchange_status

        tree = ast.parse(inspect.getsource(exchange_status))
        private = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_trip"
        ]
        assert not private, (
            f"`exchange_status.py` roept het prive `_trip` aan op regel "
            f"{private}. De publieke, getypeerde ingang is `trip()`.")
