"""`carry/service.py`: de 24/7-dienst, ronde voor ronde, met elke storing die hij moet overleven.

Een klok die de test zet, een nep-Kraken voor de data (`kraken_get`), een nep-beurs die zijn
stand tussen rondes houdt (zoals een echte), en alarmen die worden vastgelegd in plaats van
verstuurd.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests
import yaml

from tests.unit.test_carry_u1 import FakeVenue, kraken_get
from tradebot.carry import service as sv
from tradebot.carry.alerts import Alerter
from tradebot.carry.safety import SingleInstance
from tradebot.data import kraken_feed as kf
from tradebot.utils.failfast import ConfigContractError

ROOT = Path(__file__).resolve().parents[2]
DATA_NOW = pd.Timestamp("2026-10-27 06:00", tz="UTC")
GET = kraken_get(DATA_NOW)


def _config(tmp_path: Path, **change) -> Path:
    raw = yaml.safe_load((ROOT / "conf/live/u1.example.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {k: str(tmp_path / Path(v).name) for k, v in raw["paths"].items()}
    raw["capital_usd"] = 100_000.0
    for key, value in change.items():
        section, _, name = key.partition("__")
        if name:
            raw[section][name] = value
        else:
            raw[section] = value
    path = tmp_path / "u1.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


class _Clock:
    def __init__(self, t: str) -> None:
        self.t = pd.Timestamp(t, tz="UTC")

    def __call__(self) -> pd.Timestamp:
        return self.t


def _closes(now: pd.Timestamp) -> tuple[dict[str, float], dict[str, float]]:
    p = kf.fetch_panels(GET, now=now)
    last = p.perp["close"].index[-1]
    return p.spot["close"].loc[last].to_dict(), p.perp["close"].loc[last].to_dict()


def _service(tmp_path: Path, *, at: str = "2026-10-27 00:30", venue: FakeVenue | None = None,
             env: dict[str, str] | None = None, **change):
    cfg = sv.load_service_config(_config(tmp_path, **change))
    clock = _Clock(at)
    spot, perp = _closes(clock.t)
    v = venue or FakeVenue(equity_usd=100_000.0)
    v.spot_px, v.perp_px = v.spot_px or spot, v.perp_px or perp
    v.time = v.time or clock.t.timestamp()
    alerts: list[tuple[str, str, str]] = []
    alerter = Alerter(log=lambda s: None)
    alerter.send = lambda level, key, text: alerts.append((level, key, text)) or True
    pings: list[str] = []
    svc = sv.Service(cfg, root=ROOT, env=env or {}, get=GET, alerter=alerter, clock=clock,
                     sleep=lambda s: None, ping=pings.append, log=lambda s: None)
    svc._venue = lambda state: v
    return svc, clock, v, alerts, pings


def _keys(alerts) -> list[str]:
    return [k for _, k, _ in alerts]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_the_example_config_loads(tmp_path: Path):
    cfg = sv.load_service_config(ROOT / "conf/live/u1.example.yaml")
    assert cfg.mode == "dry_run" and not cfg.live and cfg.schedule.poll_minutes == 15
    assert cfg.paths.state.name == "u1-state.json"


@pytest.mark.parametrize(("change", "match"), [
    ({"mode": "paper"}, "dry_run of live"),
    ({"capital_usd": 1e9}, "plafond"),
    ({"schedule__poll_minutes": 0.5}, "tussen 1"),
    ({"schedule__step_deadline_utc": "00:05"}, "deadline"),
    ({"schedule__step_after_utc": "25:00"}, "HH:MM"),
    ({"watchdog__max_failures_before_alert": 0}, "Minstens"),
    ({"alerts": {"daily_summary": True}}, "Sleutels wijken af"),
    ({"extra": 1}, "Onbekende of ontbrekende"),
])
def test_a_broken_config_is_refused(tmp_path: Path, change, match):
    with pytest.raises(ConfigContractError, match=match):
        sv.load_service_config(_config(tmp_path, **change))


# --------------------------------------------------------------------------- #
# De dagelijkse stap
# --------------------------------------------------------------------------- #
def test_before_the_step_window_only_the_heartbeat_runs(tmp_path: Path):
    svc, clock, venue, alerts, pings = _service(
        tmp_path, at="2026-10-27 00:05", env={"HEARTBEAT_URL": "https://hc.example/x"})
    out = svc.run_once()
    assert out["events"] == [] and "error" not in out
    assert not svc.cfg.paths.state.exists() and pings == ["https://hc.example/x"]
    assert sv.healthcheck(svc.cfg, clock.t)


def test_the_daily_step_runs_once_and_reports(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path)
    out = svc.run_once()
    assert out["events"] == ["stap 2026-10-27: gehandeld"]
    st = sv.load_u1_state(svc.cfg.paths.state)
    assert st.last_bar.startswith("2026-10-27") and st.halted is None and st.expected_spot
    assert venue.spot == pytest.approx(st.expected_spot)
    assert _keys(alerts) == ["summary-2026-10-27"]
    clock.t += pd.Timedelta(minutes=15)
    out = svc.run_once()                       # dezelfde dag: alleen de watchdog
    assert out["events"] == [] and out["margin_headroom"] > svc.cfg.limits.margin_warn
    lines = svc.cfg.paths.journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and '"event": "stap"' in lines[0]


def test_data_that_is_not_fresh_is_retried_and_alerted_after_three_rounds(tmp_path: Path):
    # De nep-data loopt tot de bar van 28 oktober; op de 29e is die van de 29e er niet.
    svc, clock, venue, alerts, _ = _service(tmp_path, at="2026-10-29 00:30")
    for _ in range(3):
        out = svc.run_once()
        assert out["error"].startswith("Deferred") and "nog niet" in out["error"]
    assert _keys(alerts) == ["failing"] and svc.failures == 3
    clock.t = pd.Timestamp("2026-10-29 06:30", tz="UTC")
    svc.run_once()
    assert "late-2026-10-29" in _keys(alerts)


def test_a_refused_step_halts_the_book_with_a_critical_alert(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path, limits__max_order_usd=1000.0)
    out = svc.run_once()
    assert out["events"][0].startswith("stap geweigerd: SafetyBreach")
    st = sv.load_u1_state(svc.cfg.paths.state)
    assert st.halted.startswith("SafetyBreach") and venue.fills == []
    assert ("critical", "step-refused") in [(lvl, k) for lvl, k, _ in alerts]
    clock.t += pd.Timedelta(days=1)
    out = svc.run_once()                       # stil: geen stap, wel een herinnering
    assert out["events"][0].startswith("stil:") and "halted" in _keys(alerts)


def test_a_failed_execution_halts_and_alerts(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path, venue=FakeVenue(equity_usd=100_000.0,
                                                                         fail_at=2))
    svc.run_once()
    st = sv.load_u1_state(svc.cfg.paths.state)
    assert st.halted.startswith("uitvoering mislukt") and venue.spot == {}
    assert "step-halted" in _keys(alerts)


def test_money_needed_on_the_exchange_is_asked_for(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path, venue=FakeVenue(equity_usd=1_000.0))
    svc.run_once()
    funding = [t for _, k, t in alerts if k == "funding"]
    assert funding and "spotwallet van Kraken" in funding[0] and venue.fills == []


def test_a_week_in_cash_proposes_a_sweep_to_the_fund(tmp_path: Path, monkeypatch):
    svc, clock, venue, alerts, _ = _service(tmp_path, alerts__daily_summary=False)
    real = sv.u1.build_context

    def flat(root, market):
        ctx = real(root, market)
        return sv.u1.U1Context(**{**ctx.__dict__, "target": ctx.target * 0.0})
    monkeypatch.setattr(sv.u1, "build_context", flat)
    sv.save_u1_state(sv.U1State(config_hash=flat(ROOT, kf.kraken_market(
        kf.fetch_panels(GET, now=clock.t), sv.robust_book_v5_config(
            ROOT / sv.v5.CONFIG))).config_hash, mode="dry_run", capital_usd=100_000.0,
        committed=(0.0,) * 7), svc.cfg.paths.state)
    svc.run_once()
    assert [k for k in _keys(alerts)] == ["sweep"]


# --------------------------------------------------------------------------- #
# Stoppen en sluiten
# --------------------------------------------------------------------------- #
def test_the_halt_file_stops_new_risk_but_not_the_watchdog(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path)
    svc.cfg.paths.halt_file.write_text("", encoding="utf-8")
    out = svc.run_once()
    assert out["events"] == ["HALT-bestand: geen nieuw risico"] and venue.fills == []
    assert _keys(alerts) == ["halt-file"]


def test_the_unwind_file_closes_everything_once(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path)
    svc.run_once()                             # instappen
    assert venue.spot
    svc.cfg.paths.unwind_file.write_text("handmatig, test\n", encoding="utf-8")
    clock.t += pd.Timedelta(minutes=15)
    out = svc.run_once()
    assert out["events"] == ["unwind: handmatig, test"] and venue.spot == {}
    assert sv.load_u1_state(svc.cfg.paths.state).halted == "unwind: handmatig, test"
    assert not svc.cfg.paths.unwind_file.exists()
    assert len(list(tmp_path.glob("UNWIND.done-*"))) == 1 and "unwind" in _keys(alerts)


def test_an_unwind_file_without_a_book_only_acknowledges(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path, at="2026-10-27 00:05")
    svc.cfg.paths.unwind_file.write_text("", encoding="utf-8")
    assert svc.run_once()["events"] == ["unwind: UNWIND-bestand"] and venue.fills == []


# --------------------------------------------------------------------------- #
# De watchdog
# --------------------------------------------------------------------------- #
def _entered(tmp_path: Path, **kw):
    svc, clock, venue, alerts, pings = _service(tmp_path, **kw)
    svc.run_once()
    alerts.clear()
    clock.t += pd.Timedelta(minutes=15)
    return svc, clock, venue, alerts


def test_a_position_that_changed_on_the_exchange_halts_the_book(tmp_path: Path):
    svc, clock, venue, alerts = _entered(tmp_path)
    venue.perp_short["BTCUSDT"] *= 0.5        # ADL of een liquidatie van de helft
    venue.spot["BTCUSDT"] *= 0.5               # (gehedged, dus geen hedge-gat)
    svc.run_once()
    st = sv.load_u1_state(svc.cfg.paths.state)
    assert st.halted.startswith("de beurs wijkt af") and "drift" in _keys(alerts)


def test_a_hedge_gap_is_closed_reduce_only_and_halts(tmp_path: Path):
    svc, clock, venue, alerts = _entered(tmp_path)
    venue.perp_short.pop("ETHUSDT")            # de ETH-short is weg
    out = svc.run_once()
    assert any(e.startswith("hedge-gat") for e in out["events"])
    assert "ETHUSDT" not in venue.spot         # de ongehedgede spot is verkocht
    st = sv.load_u1_state(svc.cfg.paths.state)
    assert st.halted is not None and "hedge" in _keys(alerts)
    assert st.expected_spot == venue.spot


def test_a_hedge_gap_is_only_reported_when_repair_is_off(tmp_path: Path):
    svc, clock, venue, alerts = _entered(tmp_path, watchdog__repair_hedge=False)
    venue.perp_short.pop("ETHUSDT")
    svc.run_once()
    assert "ETHUSDT" in venue.spot and any("NIET" in t for _, k, t in alerts if k == "hedge")


def test_low_margin_warns_and_critical_margin_unwinds(tmp_path: Path):
    svc, clock, venue, alerts = _entered(tmp_path)
    venue.equity_usd = 50_000.0               # headroom 50k / (70k x 0,5) = 1,43
    svc.run_once()
    assert [lvl for lvl, k, _ in alerts if k == "margin"] == ["warning"]
    venue.equity_usd = 30_000.0               # 0,86: kritiek
    clock.t += pd.Timedelta(minutes=15)
    svc.run_once()
    assert venue.spot == {} and venue.perp_short == {}
    assert sv.load_u1_state(svc.cfg.paths.state).halted.startswith("unwind: marge")


def test_a_clock_that_drifts_from_kraken_is_alerted(tmp_path: Path):
    svc, clock, venue, alerts = _entered(tmp_path)
    venue.time = clock.t.timestamp() - 30.0
    svc.run_once()
    assert "clock" in _keys(alerts)


# --------------------------------------------------------------------------- #
# Wat de dienst moet overleven
# --------------------------------------------------------------------------- #
def test_a_network_failure_is_a_failed_round_not_a_crash(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path, watchdog__max_failures_before_alert=1)

    def down(url, params):
        raise requests.ConnectionError("Kraken onbereikbaar")
    svc.get = down
    out = svc.run_once()
    assert out["error"].startswith("ConnectionError") and _keys(alerts) == ["failing"]
    svc.get = GET
    assert "error" not in svc.run_once() and svc.failures == 0


def test_an_unexpected_error_is_alerted_and_the_service_continues(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path)

    def bug(now, out):
        raise ZeroDivisionError("oeps")
    svc._round = bug
    out = svc.run_once()
    assert out["error"] == "ZeroDivisionError: oeps" and "unexpected" in _keys(alerts)


def test_a_second_process_on_the_same_state_is_skipped(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path)
    with SingleInstance(svc.cfg.paths.state.with_suffix(".lock")):
        out = svc.run_once()
    assert out["error"].startswith("Deferred") and venue.fills == []


def test_a_dry_run_state_never_runs_under_a_live_config(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path)
    svc.run_once()
    live = sv.load_service_config(_config(tmp_path, mode="live"))
    svc.cfg = live
    out = svc.run_once()
    assert "andere modus" in out["error"]


def test_a_heartbeat_that_cannot_be_written_is_a_failed_round(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path, watchdog__max_failures_before_alert=1)
    svc.cfg.paths.heartbeat.mkdir()          # een map waar een bestand moet: schrijven faalt
    out = svc.run_once()
    assert out["error"].startswith("IsADirectoryError") and venue.fills == []
    assert _keys(alerts) == ["failing"]


def test_a_failing_heartbeat_ping_does_not_stop_the_round(tmp_path: Path):
    logs: list[str] = []
    svc, clock, venue, alerts, _ = _service(tmp_path, env={"HEARTBEAT_URL": "https://hc/x"})

    def down(url):
        raise requests.Timeout("weg")
    svc.ping, svc.log = down, logs.append
    assert svc.run_once()["events"] and any("heartbeat-ping faalde" in x for x in logs)


def test_the_real_venue_and_ping_are_wired(tmp_path: Path, monkeypatch):
    cfg = sv.load_service_config(_config(tmp_path))
    svc = sv.Service(cfg, root=ROOT, env={}, log=lambda s: None)
    st = sv.U1State(config_hash="h", mode="dry_run", capital_usd=1.0)
    assert svc._venue(st).dry_run
    seen = []

    class R:
        def raise_for_status(self) -> None:
            seen.append("ok")
    monkeypatch.setattr(requests, "get", lambda url, timeout: seen.append(url) or R())
    sv._ping("https://hc.example/y")
    assert seen == ["https://hc.example/y", "ok"]


# --------------------------------------------------------------------------- #
# De lus, de healthcheck, de CLI
# --------------------------------------------------------------------------- #
def test_serve_runs_rounds_until_stopped(tmp_path: Path):
    svc, clock, venue, alerts, _ = _service(tmp_path, at="2026-10-27 00:05")
    assert svc.serve(max_rounds=2) == 2
    assert _keys(alerts)[0] == "start"
    naps: list[float] = []

    def nap(s: float) -> None:
        naps.append(s)
        if len(naps) == 3:
            svc.stop()
    svc.sleep = nap
    assert svc.serve() == 1 and naps == [5.0, 5.0, 5.0]


def test_healthcheck_reads_the_age_of_the_heartbeat(tmp_path: Path):
    cfg = sv.load_service_config(_config(tmp_path))
    assert not sv.healthcheck(cfg)
    cfg.paths.heartbeat.write_text("2026-10-27T00:00:00+00:00\n", encoding="utf-8")
    assert sv.healthcheck(cfg, pd.Timestamp("2026-10-27 00:44", tz="UTC"))
    assert not sv.healthcheck(cfg, pd.Timestamp("2026-10-27 00:46", tz="UTC"))


def test_main_healthcheck_once_and_serve(tmp_path: Path, monkeypatch):
    path = _config(tmp_path)
    assert sv.main(["--config", str(path), "--healthcheck"]) == 1
    calls = []
    monkeypatch.setattr(sv.Service, "run_once", lambda self: calls.append("once") or {})
    assert sv.main(["--config", str(path), "--once"]) == 0
    monkeypatch.setattr(sv.Service, "run_once", lambda self: {"error": "x"})
    assert sv.main(["--config", str(path), "--once"]) == 1
    monkeypatch.setattr(sv.Service, "serve", lambda self: calls.append("serve") or 0)
    assert sv.main(["--config", str(path)]) == 0 and calls == ["once", "serve"]



@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_runs_as_a_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["service", "--config", str(_config(tmp_path)),
                                      "--healthcheck"])
    with pytest.raises(SystemExit) as done:
        runpy.run_module("tradebot.carry.service", run_name="__main__")
    assert done.value.code == 1           # nog geen heartbeat
