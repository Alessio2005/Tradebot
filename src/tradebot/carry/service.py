"""Het Kraken-boek 24/7: de dagelijkse stap, een watchdog, alarmen en een heartbeat.

    python -I -m tradebot.carry.service --config /etc/tradebot/u1.yaml           # de dienst
    python -I -m tradebot.carry.service --config /etc/tradebot/u1.yaml --once    # één ronde
    python -I -m tradebot.carry.service --config /etc/tradebot/u1.yaml --healthcheck

ELKE RONDE (standaard elk kwartier), in deze volgorde:
1. **Heartbeat**: het bestand `paths.heartbeat` krijgt de tijd (de healthcheck van docker
   leest het), en `HEARTBEAT_URL` krijgt een GET (een dead man's switch van buiten: valt de
   server zelf weg, dan alarmeert die dienst).
2. **Lock** op de staat: draait er al een proces, dan slaat deze ronde over.
3. **UNWIND-bestand**: bestaat het, dan sluit het boek alles gehedged en staat stil; het
   bestand wordt hernoemd (`.done-<tijd>`), zodat het niet elke ronde opnieuw sluit.
4. **Watchdog** (als er een staat is): de beurs tegen de verwachte stand (een verschil:
   liquidatie, auto-deleveraging, een handmatige ingreep -> stilstand); een hedge-gat
   (reduce-only dichten en stilzetten); de marge (waarschuwen, kritiek: alles sluiten); de
   klok tegen die van Kraken.
5. **De dagelijkse stap**, na `schedule.step_after_utc` en eenmaal per afgesloten dagbar,
   niet als het boek stilstaat of het HALT-bestand er is. Data die nog niet vers is, of een
   live koers die te ver van de close ligt: de volgende ronde opnieuw. Een harde grens of
   een breuk: stilstand en een kritiek alarm. Is de stap om `step_deadline_utc` nog niet
   gelukt: een alarm.

WAT DE DIENST NOOIT DOET: zelf weer gaan handelen na een stilstand (dat doet een mens met
`python -I -m tradebot.carry.u1 resume`), een order herhalen die faalde (een tweede poging
kan een dubbele order zijn), of een fout inslikken zonder het log en (na een paar rondes)
een alarm. Een ronde die faalt, laat de dienst draaien: de volgende ronde probeert opnieuw.
SIGTERM en SIGINT ronden de lopende ronde af en stoppen dan.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from ..data import kraken_feed as kf
from ..schemas.robust_book_v2 import robust_book_v5_config
from ..systematic import programme_v5 as v5
from ..utils.failfast import ConfigContractError, DataContractError, require
from . import u1
from .alerts import Alerter
from .kraken import KrakenError
from .safety import (
    Deferred,
    Limits,
    SafetyBreach,
    SingleInstance,
    hedge_gaps,
    margin_headroom,
    repair_orders,
)
from .u1 import U1State, append_journal, execute_each, load_u1_state, save_u1_state
from .venue import ReconciliationBreak, reconcile_book

__all__ = ["ServiceConfig", "Service", "healthcheck", "load_service_config", "main"]

#: Fouten waarna de volgende ronde het gewoon opnieuw probeert (netwerk, beurs, een feed of
#: staat die niet klopt). Ze tellen; na `max_failures_before_alert` rondes volgt een alarm.
#: Binnen de stap is een `DataContractError` wel fataal (`FATAL`): daar is het een weigering.
TRANSIENT = (requests.RequestException, KrakenError, Deferred, DataContractError, OSError,
             TimeoutError)
#: Fouten in de stap waarna het boek stilgaat tot een mens kijkt.
FATAL = (SafetyBreach, ReconciliationBreak, DataContractError)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    state: Path
    journal: Path
    heartbeat: Path
    halt_file: Path
    unwind_file: Path


@dataclass(frozen=True)
class Schedule:
    poll_minutes: float
    step_after_utc: str
    step_deadline_utc: str


@dataclass(frozen=True)
class WatchdogCfg:
    repair_hedge: bool
    unwind_on_margin_critical: bool
    max_failures_before_alert: int


@dataclass(frozen=True)
class AlertCfg:
    min_interval_minutes: float
    daily_summary: bool


@dataclass(frozen=True)
class ServiceConfig:
    mode: str
    capital_usd: float
    paths: Paths
    schedule: Schedule
    limits: Limits
    watchdog: WatchdogCfg
    alerts: AlertCfg

    @property
    def live(self) -> bool:
        return self.mode == "live"


def _section(cls: type, raw: Any, name: str) -> Any:
    keys = {f.name for f in fields(cls)}
    require(isinstance(raw, Mapping) and set(raw) == keys, "Sleutels wijken af van het "
            "schema.", ConfigContractError, section=name, expected=sorted(keys),
            got=sorted(raw) if isinstance(raw, Mapping) else None)
    return cls(**{k: raw[k] for k in keys})


def _minutes(hhmm: str) -> int:
    h, _, m = str(hhmm).partition(":")
    require(h.isdigit() and m.isdigit() and int(h) < 24 and int(m) < 60,
            "Een tijd is HH:MM (UTC).", ConfigContractError, value=hhmm)
    return int(h) * 60 + int(m)


def load_service_config(path: Path) -> ServiceConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    want = {"mode", "capital_usd", "paths", "schedule", "limits", "watchdog", "alerts"}
    require(set(raw) == want, "Onbekende of ontbrekende sleutels in de serverconfig.",
            ConfigContractError, got=sorted(raw), want=sorted(want))
    paths = _section(Paths, raw["paths"], "paths")
    cfg = ServiceConfig(
        mode=str(raw["mode"]), capital_usd=float(raw["capital_usd"]),
        paths=Paths(**{f.name: Path(getattr(paths, f.name)) for f in fields(Paths)}),
        schedule=_section(Schedule, raw["schedule"], "schedule"),
        limits=Limits.from_mapping(raw["limits"]),
        watchdog=_section(WatchdogCfg, raw["watchdog"], "watchdog"),
        alerts=_section(AlertCfg, raw["alerts"], "alerts"))
    require(cfg.mode in ("dry_run", "live"), "mode is dry_run of live.", ConfigContractError,
            mode=cfg.mode)
    require(0.0 < cfg.capital_usd <= cfg.limits.max_capital_usd, "Het kapitaal ligt niet "
            "tussen nul en het plafond.", ConfigContractError, capital=cfg.capital_usd,
            cap=cfg.limits.max_capital_usd)
    require(1.0 <= float(cfg.schedule.poll_minutes) <= 240.0, "poll_minutes ligt tussen 1 "
            "en 240.", ConfigContractError)
    require(_minutes(cfg.schedule.step_after_utc) < _minutes(cfg.schedule.step_deadline_utc),
            "De deadline ligt na het begin van de stap.", ConfigContractError)
    require(int(cfg.watchdog.max_failures_before_alert) >= 1, "Minstens één mislukte ronde "
            "voor een alarm.", ConfigContractError)
    return cfg


# --------------------------------------------------------------------------- #
# De dienst
# --------------------------------------------------------------------------- #
def _ping(url: str) -> None:
    requests.get(url, timeout=10).raise_for_status()


class Service:
    """Eén ronde is `run_once`; `serve` herhaalt die tot SIGTERM (zie de moduledocstring)."""

    def __init__(self, cfg: ServiceConfig, *, root: Path = u1.ROOT,
                 env: Mapping[str, str] | None = None, get: kf.Get | None = None,
                 http: Any | None = None, alerter: Alerter | None = None,
                 clock: Callable[[], pd.Timestamp] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 ping: Callable[[str], None] = _ping,
                 log: Callable[[str], None] = print) -> None:
        self.cfg, self.root = cfg, root
        self.env = dict(os.environ if env is None else env)
        self.get = kf._get if get is None else get
        self.http = http
        self.alerter = alerter or Alerter.from_env(
            self.env, min_interval_s=cfg.alerts.min_interval_minutes * 60.0, log=log)
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.sleep, self.ping, self.log = sleep, ping, log
        self.failures = 0
        self.stopping = False

    # ------------------------------------------------------------------ ronde
    def run_once(self) -> dict[str, Any]:
        now = self.clock()
        out: dict[str, Any] = {"at": now.isoformat(), "events": []}
        try:
            # Ook de heartbeat zit in de ronde: een volle schijf is een mislukte ronde (en
            # dan ook geen stap: de staat zou niet op te slaan zijn), geen dode dienst.
            self._heartbeat(now)
            with SingleInstance(self.cfg.paths.state.with_suffix(".lock")):
                self._round(now, out)
            self.failures = 0
        except TRANSIENT as e:
            self._failed(out, e)
        except Exception as e:  # onverwacht: loggen en alarmeren, de dienst blijft draaien
            self._failed(out, e)
            self.alerter.send("critical", "unexpected", f"onverwachte fout: "
                              f"{type(e).__name__}: {str(e)[:300]}")
        self.log(json.dumps({"round": out}, sort_keys=True, default=str))
        return out

    def _failed(self, out: dict[str, Any], e: BaseException) -> None:
        self.failures += 1
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        if self.failures >= int(self.cfg.watchdog.max_failures_before_alert):
            self.alerter.send("warning", "failing", f"{self.failures} rondes op rij mislukt; "
                              f"laatste: {out['error']}")

    def _heartbeat(self, now: pd.Timestamp) -> None:
        p = self.cfg.paths.heartbeat
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(now.isoformat() + "\n", encoding="utf-8")
        url = self.env.get("HEARTBEAT_URL")
        if url:
            try:
                self.ping(url)
            except Exception as e:  # een dead man's switch die faalt, stopt de dienst niet
                self.log(f"heartbeat-ping faalde: {type(e).__name__}")

    # ------------------------------------------------------------------ stappen
    def _state(self) -> U1State | None:
        p = self.cfg.paths.state
        if not p.is_file():
            return None
        st = load_u1_state(p)
        require(st.mode == self.cfg.mode, "De staat en de config hebben een andere modus; "
                "begin een nieuwe staat voor live.", DataContractError, state=st.mode,
                config=self.cfg.mode)
        return st

    def _save(self, state: U1State, record: Mapping[str, Any], now: pd.Timestamp) -> None:
        append_journal(self.cfg.paths.journal, {"at": now.isoformat(), **record})
        save_u1_state(state, self.cfg.paths.state, now=now)

    def _venue(self, state: U1State) -> Any:
        return u1.make_venue(sorted(kf.COINS), state, live=self.cfg.live, http=self.http,
                             env=self.env)

    def _context(self, now: pd.Timestamp) -> u1.U1Context:
        panels = kf.fetch_panels(self.get, now=now)
        if panels.last < now.floor("D"):
            raise Deferred(f"De Kraken-data loopt tot {panels.last.date()}; de bar van "
                           f"{now.floor('D').date()} is er nog niet.")
        cfg5 = robust_book_v5_config(self.root / v5.CONFIG)
        return u1.build_context(self.root, kf.kraken_market(panels, cfg5))

    def _round(self, now: pd.Timestamp, out: dict[str, Any]) -> None:
        state = self._state()
        if self.cfg.paths.unwind_file.is_file():
            self._unwind_file(state, now, out)
            return
        if state is not None:
            state = self._watchdog(state, now, out)
        halt_file = self.cfg.paths.halt_file.is_file()
        if halt_file:
            out["events"].append("HALT-bestand: geen nieuw risico")
            self.alerter.send("warning", "halt-file", "HALT-bestand aanwezig: het boek bouwt "
                              "geen risico op")
        if state is not None and state.halted is not None:
            out["events"].append(f"stil: {state.halted}")
            self.alerter.send("warning", "halted", f"het boek staat stil: {state.halted}. "
                              "Hervatten na onderzoek: python -I -m tradebot.carry.u1 resume")
            return
        today = now.floor("D")
        after = today + pd.Timedelta(minutes=_minutes(self.cfg.schedule.step_after_utc))
        deadline = today + pd.Timedelta(minutes=_minutes(self.cfg.schedule.step_deadline_utc))
        done = state is not None and state.last_bar is not None and \
            pd.Timestamp(state.last_bar) >= today
        if halt_file or done or now < after:
            return
        try:
            self._daily_step(state, now, out)
        except TRANSIENT as e:
            # Alleen een stap die na de deadline MISLUKT, is te laat; een stap die in deze
            # ronde slaagt (bv. na een herstart in de middag) niet.
            if now >= deadline:
                self.alerter.send("warning", f"late-{today.date()}", f"de stap van "
                                  f"{today.date()} is om {now:%H:%M} UTC nog niet gelukt: "
                                  f"{type(e).__name__}")
            raise

    def _daily_step(self, state: U1State | None, now: pd.Timestamp,
                    out: dict[str, Any]) -> None:
        ctx = self._context(now)
        if state is None:
            state = U1State(config_hash=ctx.config_hash, mode=self.cfg.mode,
                            capital_usd=self.cfg.capital_usd)
        venue = self._venue(state)
        try:
            record, new = u1.step(ctx, state, venue, limits=self.cfg.limits,
                                  live_prices=venue.prices())
        except FATAL as e:
            reason = f"{type(e).__name__}: {str(e)[:400]}"
            self._save(replace(state, halted=reason), {"event": "stap geweigerd",
                                                       "reason": reason}, now)
            out["events"].append(f"stap geweigerd: {reason}")
            self.alerter.send("critical", "step-refused", f"stap geweigerd, het boek staat "
                              f"stil: {reason}")
            return
        self._save(new, {"event": "stap", **record}, now)
        out["events"].append(f"stap {record['bar'][:10]}: {record['action']}")
        if new.halted is not None:
            self.alerter.send("critical", "step-halted", f"na de stap staat het boek stil: "
                              f"{new.halted}")
        if record.get("funding_request_usd", 0.0) > 0.0:
            self.alerter.send("warning", "funding", f"zet {record['funding_request_usd']:,.0f} "
                              "USD op de spotwallet van Kraken: het boek wil instappen")
        if record.get("sweep_out_usd", 0.0) > 0.0:
            self.alerter.send("info", "sweep", f"{record['sweep_out_usd']:,.0f} USD kan naar "
                              "het geldmarktfonds: een week geen carry")
        if self.cfg.alerts.daily_summary:
            tgt = ", ".join(f"{s} {v:.2f}" for s, v in record["target"].items())
            self.alerter.send("info", f"summary-{record['bar'][:10]}",
                              f"stap {record['bar'][:10]} ({self.cfg.mode}): "
                              f"{record['action']}; doel {tgt}; beurs "
                              f"{record['exchange_equity_usd']:,.0f} USD")

    # ------------------------------------------------------------------ watchdog
    def _watchdog(self, state: U1State, now: pd.Timestamp, out: dict[str, Any]) -> U1State:
        venue = self._venue(state)
        vb = venue.book()
        spot_px, perp_px = venue.prices()
        lim = self.cfg.limits
        skew = abs(venue.server_time() - now.timestamp())
        if skew > lim.max_clock_skew_s:
            self.alerter.send("warning", "clock", f"de klok wijkt {skew:.1f} s af van Kraken "
                              "(nonce en dagwissel leunen erop)")
        if state.halted is None:
            breaks = reconcile_book(state.expected_spot, state.expected_perp, vb,
                                    prices=spot_px)
            if breaks:
                reason = "de beurs wijkt af van het boek: " + "; ".join(breaks)
                state = replace(state, halted=reason)
                self._save(state, {"event": "watchdog", "reason": reason}, now)
                self.alerter.send("critical", "drift", reason + " (liquidatie, ADL of een "
                                  "handmatige ingreep?)")
        gaps = hedge_gaps(vb, spot_px, lim.hedge_tolerance_usd)
        if gaps:
            out["events"].append(f"hedge-gat {gaps}")
            if self.cfg.watchdog.repair_hedge:
                fills, err = execute_each(venue, repair_orders(gaps, spot_px, perp_px))
                fresh = venue.book()
                state = replace(state, expected_spot=dict(fresh.spot),
                                expected_perp=dict(fresh.perp_short),
                                halted=state.halted or f"hedge-gat {gaps} gedicht")
                self._save(state, {"event": "hedge-reparatie", "gaps": gaps,
                                   "fills": len(fills), "error": err}, now)
            self.alerter.send("critical", "hedge", f"hedge-gat {gaps}; "
                              + ("reduce-only gedicht, boek stil" if
                                 self.cfg.watchdog.repair_hedge else "NIET gerepareerd"))
        head = margin_headroom(vb, perp_px, lim.initial_margin)
        out["margin_headroom"] = None if math.isinf(head) else head
        if head < lim.margin_critical:
            self.alerter.send("critical", "margin", f"marge-headroom {head:.2f} onder "
                              f"{lim.margin_critical}")
            if self.cfg.watchdog.unwind_on_margin_critical:
                record, state = u1.unwind_book(state, venue, reason=f"marge {head:.2f}",
                                               limits=lim, spot_px=spot_px, perp_px=perp_px)
                self._save(state, {"event": "watchdog-unwind", **record}, now)
        elif head < lim.margin_warn:
            self.alerter.send("warning", "margin", f"marge-headroom {head:.2f} onder "
                              f"{lim.margin_warn}")
        return state

    def _unwind_file(self, state: U1State | None, now: pd.Timestamp,
                     out: dict[str, Any]) -> None:
        f = self.cfg.paths.unwind_file
        reason = f.read_text(encoding="utf-8").strip() or "UNWIND-bestand"
        if state is not None:
            venue = self._venue(state)
            spot_px, perp_px = venue.prices()
            record, state = u1.unwind_book(state, venue, reason=reason,
                                           limits=self.cfg.limits, spot_px=spot_px,
                                           perp_px=perp_px)
            self._save(state, {"event": "unwind", **record}, now)
        f.rename(f.with_name(f"{f.name}.done-{now:%Y%m%dT%H%M%S}"))
        out["events"].append(f"unwind: {reason}")
        self.alerter.send("critical", "unwind", f"UNWIND uitgevoerd ({reason}); het boek "
                          "staat stil")

    # ------------------------------------------------------------------ lus
    def stop(self, *_: Any) -> None:
        self.stopping = True

    def serve(self, *, max_rounds: int | None = None) -> int:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self.alerter.send("info", "start", f"dienst gestart ({self.cfg.mode}, kapitaal "
                          f"{self.cfg.capital_usd:,.0f} USD, kanalen: "
                          f"{', '.join(self.alerter.channels) or 'alleen log'})")
        rounds = 0
        while not self.stopping and (max_rounds is None or rounds < max_rounds):
            self.run_once()
            rounds += 1
            waited = 0.0
            while not self.stopping and waited < self.cfg.schedule.poll_minutes * 60.0 and (
                    max_rounds is None or rounds < max_rounds):
                self.sleep(5.0)
                waited += 5.0
        self.log("dienst gestopt")
        return rounds


def healthcheck(cfg: ServiceConfig, now: pd.Timestamp | None = None) -> bool:
    """Gezond als de laatste heartbeat jonger is dan drie rondes."""
    p = cfg.paths.heartbeat
    if not p.is_file():
        return False
    beat = pd.Timestamp(p.read_text(encoding="utf-8").strip())
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    return bool(now - beat <= pd.Timedelta(minutes=3 * float(cfg.schedule.poll_minutes)))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -m tradebot.carry.service")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--once", action="store_true", help="één ronde en stoppen")
    ap.add_argument("--healthcheck", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_service_config(args.config)
    if args.healthcheck:
        return 0 if healthcheck(cfg) else 1
    svc = Service(cfg)
    if args.once:
        out = svc.run_once()
        return 1 if "error" in out else 0
    svc.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
