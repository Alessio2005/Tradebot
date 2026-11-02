"""Robuust boek v10: U1 (carry + kasrente) plus een flow/trend-sleeve op het stilstaande geld.

    python -I -m tradebot.data.positioning_series     # op Actions: spot-close en CFTC-TFF
    python -I -m tradebot.systematic.programme_v10 freeze   # ledger (M += aantal trials)
    python -I -m tradebot.systematic.programme_v10 run      # hele cyclus, per jaar, criteria

WAAROM. Eigenaarsvraag 2026-11-02: niet alleen van funding-carry afhangen, en inspelen op
instituties die netto long gaan. U1 stond in carryloze jaren 80-95 % in het fonds. v10 laat
de carry ongemoeid (die heeft voorrang op het kapitaal) en zet een deel van het geld dat
in het fonds zou staan long in BTC/ETH-spot, alleen als:

* **flow:** de netto long van asset managers op het CME-contract (CFTC Traders in Financial
  Futures; (long - short) / open interest) hoger is dan `flow_lookback_reports` rapporten
  eerder. Een rapport telt pas vanaf dinsdag + `release_lag_days` (verschijnt vrijdag), en
  is het ouder dan `flow_stale_days`, dan staat de flow uit;
* **trend:** de close boven zijn gemiddelde van `trend_days` dagen staat.

Spot, geen hefboom: kan op Kraken EU en telt daar als onderpand. Ontwerp en criteria staan
in `conf/model/robust_book_v10.yaml`, vastgelegd vóór de data er was.

DE BEREKENING, per dag t (besluit op de close van t, rendement van t naar t+1):
1. `on_i,t` = flow en/of trend (per trial), per munt; zonder voldoende historie uit.
2. `w_i,t = on_i,t · (τ / σ_i,t,jaar) / 2`, σ uit de EWMA van dagrendementen.
3. Schaal de rij terug tot Σ w ≤ min(max_weight, 1 - buffer - committed_t), met
   `committed_t` het kapitaal dat de carry die dag vastlegt (U1).
4. Handel alleen bij aan/uit of een afwijking > `rebalance_band` van het doel; kosten
   `spot_bps` per eenheid omzet. Tussen handel driften de gewichten met de prijs.
5. `net_t = net_U1_t + Σ held_i,t-1 · r_i,t - min(Σ held, cash_fraction_t) · kasrente_t - kosten_t`.
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..data import positioning_series as ps
from ..features.registry import current_git_sha
from ..registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from ..utils.failfast import ConfigContractError, DataContractError, require
from ..utils.hashing import hash_config
from . import programme_v2 as v2

__all__ = ["V10Config", "book", "evaluate", "flow_signal", "freeze", "load_config", "main",
           "run_programme", "sleeve_targets", "trend_signal"]

ROOT = v2.ROOT
CONFIG = Path("conf/model/robust_book_v10.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v10")
WAVE = 10
COINS = ("BTC", "ETH")
#: Het standaardcontract per munt op de CME (niet de micro's): de naam begint zo.
CME_MARKET = {"BTC": r"^BITCOIN - CHICAGO MERCANTILE",
              "ETH": r"^ETHER(?: CASH SETTLED)? - CHICAGO MERCANTILE"}
DAYS = 365.0


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Base:
    programme: str
    candidate: str
    operating_buffer: float
    haircut_apr: float


@dataclass(frozen=True)
class Signal:
    trend_days: int
    flow_lookback_reports: int
    release_lag_days: int
    flow_stale_days: int


@dataclass(frozen=True)
class Sizing:
    vol_span: int
    min_history_days: int
    rebalance_band: float


@dataclass(frozen=True)
class Costs:
    spot_bps: float


@dataclass(frozen=True)
class Trial:
    use_flow: bool
    use_trend: bool
    vol_target: float
    max_weight: float


@dataclass(frozen=True)
class Criteria:
    min_cagr_uplift: float
    min_monthly_excess_sharpe: float
    max_drawdown: float
    holdout_start: str
    holdout_min_excess_uplift: float
    max_worst_year_giveup: float


@dataclass(frozen=True)
class V10Config:
    base: Base
    signal: Signal
    sizing: Sizing
    costs: Costs
    trials: dict[str, Trial]
    primary: str
    criteria: Criteria


def _section(cls: type, raw: Any, name: str) -> Any:
    keys = {f.name for f in fields(cls)}
    require(isinstance(raw, Mapping) and set(raw) == keys, "Sleutels wijken af van het schema.",
            ConfigContractError, section=name, expected=sorted(keys),
            got=sorted(raw) if isinstance(raw, Mapping) else None)
    return cls(**{k: raw[k] for k in keys})


def load_config(path: Path) -> V10Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    top = {"base", "signal", "sizing", "costs", "trials", "primary", "criteria"}
    require(isinstance(raw, Mapping) and set(raw) == top,
            "Onbekende of ontbrekende sleutels in de v10-config.", ConfigContractError,
            got=sorted(raw) if isinstance(raw, Mapping) else None)
    require(isinstance(raw["trials"], Mapping) and bool(raw["trials"]), "Geen trials.",
            ConfigContractError)
    cfg = V10Config(
        base=_section(Base, raw["base"], "base"), signal=_section(Signal, raw["signal"], "signal"),
        sizing=_section(Sizing, raw["sizing"], "sizing"),
        costs=_section(Costs, raw["costs"], "costs"),
        trials={str(k): _section(Trial, v, f"trials.{k}") for k, v in raw["trials"].items()},
        primary=str(raw["primary"]), criteria=_section(Criteria, raw["criteria"], "criteria"))
    require(cfg.primary in cfg.trials, "De hoofdkandidaat is geen trial.", ConfigContractError,
            primary=cfg.primary)
    for name, t in cfg.trials.items():
        require((t.use_flow or t.use_trend) and 0.0 < t.vol_target <= 0.5
                and 0.0 < t.max_weight <= 1.0, "Een trial buiten zijn bereik.",
                ConfigContractError, trial=name, values=asdict(t))
    s, z = cfg.signal, cfg.sizing
    require(s.trend_days >= 2 and s.flow_lookback_reports >= 1 and s.release_lag_days >= 3
            and s.flow_stale_days > 7 and z.vol_span >= 2 and z.min_history_days >= 1
            and 0.0 <= z.rebalance_band < 1.0 and cfg.costs.spot_bps >= 0.0
            and 0.0 <= cfg.base.operating_buffer < 1.0 and 0.0 <= cfg.base.haircut_apr < 0.05,
            "Een parameter buiten zijn bereik.", ConfigContractError)
    return cfg


# --------------------------------------------------------------------------- #
# Signalen
# --------------------------------------------------------------------------- #
def trend_signal(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """Close boven het gemiddelde van `days` dagen; zonder volle historie: uit."""
    sma = close.rolling(days, min_periods=days).mean()
    return (close > sma) & sma.notna()


def _am_net(tff: pd.DataFrame, coin: str) -> pd.Series:
    rows = tff[tff["market"].astype(str).str.contains(CME_MARKET[coin], regex=True)]
    g = rows.groupby("report_date")[["asset_mgr_long", "asset_mgr_short", "open_interest"]].sum()
    g = g[g["open_interest"] > 0]
    return ((g["asset_mgr_long"] - g["asset_mgr_short"]) / g["open_interest"]).rename(coin)


def flow_signal(tff: pd.DataFrame, index: pd.DatetimeIndex, sig: Signal) -> pd.DataFrame:
    """Per dag en munt: is de netto long van asset managers gestegen (point-in-time)?

    Een rapport geldt vanaf rapportdatum + `release_lag_days` tot het volgende, en vervalt
    na `flow_stale_days`. Zonder rapport (of zonder vergelijkbaar eerder rapport): uit."""
    out = {}
    for coin in COINS:
        net = _am_net(tff, coin).sort_index()
        rising = (net.diff(sig.flow_lookback_reports) > 0) & net.shift(
            sig.flow_lookback_reports).notna()
        eff = pd.Series(rising.to_numpy(), index=net.index + pd.Timedelta(
            days=sig.release_lag_days))
        when = pd.Series(eff.index, index=eff.index)
        val = eff.reindex(index, method="ffill")
        age = index.to_series() - when.reindex(index, method="ffill")
        fresh = age <= pd.Timedelta(days=sig.flow_stale_days)
        out[coin] = (val.fillna(False).astype(bool) & fresh).to_numpy()
    return pd.DataFrame(out, index=index)


# --------------------------------------------------------------------------- #
# De sleeve en het boek
# --------------------------------------------------------------------------- #
def sleeve_targets(close: pd.DataFrame, tff: pd.DataFrame, committed: pd.Series,
                   cfg: V10Config, trial: Trial) -> pd.DataFrame:
    """Doelgewicht per munt op de close van elke dag (stap 1-3 van de docstring)."""
    s, z = cfg.signal, cfg.sizing
    on = pd.DataFrame(True, index=close.index, columns=list(COINS))
    if trial.use_trend:
        on &= trend_signal(close, s.trend_days)
    if trial.use_flow:
        on &= flow_signal(tff, close.index, s)
    ret = close.pct_change()
    sigma = np.sqrt((ret ** 2).ewm(span=z.vol_span, min_periods=z.vol_span).mean() * DAYS)
    history = close.notna().cumsum() >= z.min_history_days
    on &= history & sigma.gt(0.0) & close.notna()
    raw = (trial.vol_target / sigma).where(on, 0.0).fillna(0.0) / len(COINS)
    room = (1.0 - cfg.base.operating_buffer - committed.reindex(close.index).fillna(1.0))
    cap = np.minimum(trial.max_weight, room.clip(lower=0.0))
    total = raw.sum(axis=1)
    factor = (cap / total.where(total > 0)).clip(upper=1.0).fillna(0.0)
    return raw.mul(factor, axis=0)


def book(u1: pd.DataFrame, close: pd.DataFrame, target: pd.DataFrame,
         cfg: V10Config) -> pd.DataFrame:
    """De dagreeks van U1 plus de sleeve (stap 4-5 van de docstring).

    `u1` en `close` delen de dag-index (naïef, het sluitmoment)."""
    ret = close.pct_change().fillna(0.0).to_numpy()
    tgt = target.to_numpy()
    rate = ((u1["rf"] * DAYS - cfg.base.haircut_apr).clip(lower=0.0) / DAYS).to_numpy()
    band, fee = cfg.sizing.rebalance_band, cfg.costs.spot_bps / 1e4
    n = len(u1)
    held = np.zeros(tgt.shape[1])
    sleeve, lost, cost, expo = (np.zeros(n) for _ in range(4))
    for t in range(n):
        pnl = float(held @ ret[t])
        lost[t] = min(held.sum(), float(u1["cash_fraction"].iloc[t])) * rate[t]
        sleeve[t] = pnl
        day = float(u1["net"].iloc[t]) + pnl - lost[t]
        held = held * (1.0 + ret[t]) / (1.0 + day)
        want = tgt[t]
        flip = (want > 0) != (held > 0)
        off = np.abs(held - want) > band * np.where(want > 0, want, np.inf)
        trade = flip | off
        new = np.where(trade, want, held)
        cost[t] = float(np.abs(new - held).sum()) * fee
        held = new
        expo[t] = held.sum()
    out = u1[["net", "rf", "cash_fraction"]].rename(columns={"net": "u1"}).copy()
    out["sleeve"], out["lost_interest"], out["costs"] = sleeve, lost, cost
    out["exposure"] = expo
    out["net"] = out["u1"] + out["sleeve"] - out["lost_interest"] - out["costs"]
    out["excess"] = out["net"] - out["rf"]
    return out


# --------------------------------------------------------------------------- #
# Meten
# --------------------------------------------------------------------------- #
def _cagr(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (DAYS / len(r)) - 1.0)


def _monthly_sharpe(excess: pd.Series) -> float:
    m = (1.0 + excess).groupby(excess.index.to_period("M")).prod() - 1.0
    sd = float(m.std(ddof=1))
    return float(m.mean() / sd * math.sqrt(12.0)) if sd > 0 else float("nan")


def _max_dd(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float(-(eq / eq.cummax() - 1.0).min())


def _years(r: pd.Series) -> dict[str, float]:
    return {str(y): float((1.0 + c).prod() - 1.0) for y, c in r.groupby(r.index.year)}


def summary(frame: pd.DataFrame, holdout_start: str) -> dict[str, Any]:
    ho = frame.loc[pd.Timestamp(holdout_start):]
    return {"cagr": _cagr(frame["net"]), "excess_cagr": _cagr(frame["excess"]),
            "monthly_excess_sharpe": _monthly_sharpe(frame["excess"]),
            "max_drawdown": _max_dd(frame["net"]), "years": _years(frame["net"]),
            "holdout_excess": float((1.0 + ho["excess"]).prod() - 1.0),
            "mean_exposure": float(frame["exposure"].mean()) if "exposure" in frame else 0.0,
            "costs": float(frame["costs"].sum()) if "costs" in frame else 0.0}


def evaluate(trial: dict[str, Any], base: dict[str, Any], c: Criteria) -> dict[str, Any]:
    """De vooraf vastgelegde criteria: alle moeten gelden."""
    worst = min(trial["years"].values())
    worst_base = min(base["years"].values())
    checks = {
        "cagr_uplift": (trial["cagr"] - base["cagr"], c.min_cagr_uplift, ">="),
        "monthly_excess_sharpe": (trial["monthly_excess_sharpe"],
                                  c.min_monthly_excess_sharpe, ">="),
        "max_drawdown": (trial["max_drawdown"], c.max_drawdown, "<="),
        "holdout_excess_uplift": (trial["holdout_excess"] - base["holdout_excess"],
                                  c.holdout_min_excess_uplift, ">"),
        "worst_year_giveup": (worst_base - worst, c.max_worst_year_giveup, "<="),
    }
    out: dict[str, Any] = {}
    for k, (v, thr, op) in checks.items():
        ok = (v >= thr if op == ">=" else v <= thr if op == "<=" else v > thr) \
            if math.isfinite(v) else False
        out[k] = {"value": v, "threshold": thr, "op": op, "pass": bool(ok)}
    out["verdict"] = "pass" if all(x["pass"] for x in out.values()) else "rejected"
    return out


# --------------------------------------------------------------------------- #
# Invoer, ledger, run
# --------------------------------------------------------------------------- #
def load_u1(root: Path, cfg: V10Config) -> pd.DataFrame:
    d = root / "artefacts/research" / cfg.base.programme
    parts = [pd.read_csv(d / f"{cfg.base.candidate}_daily_{w}.csv", index_col=0,
                         parse_dates=True) for w in ("backcast", "w_dev", "holdout")]
    f = pd.concat(parts).sort_index()
    f.index = f.index.tz_convert(None)
    require(not f.index.has_duplicates and f.index.is_monotonic_increasing,
            "De U1-vensters overlappen.", DataContractError)
    return f


def _inputs(root: Path, cfg: V10Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    u1 = load_u1(root, cfg)
    spot = ps.load_spot_close(root)
    require(list(spot.columns) == list(COINS), "De spotreeks mist een munt.",
            DataContractError, columns=list(spot.columns))
    close = spot.reindex(pd.date_range(spot.index.min(), u1.index.max(), freq="D")).ffill()
    return u1, close, ps.load_tff(root)


def _data_hash(root: Path) -> str:
    manifest = json.loads((root / ps.MANIFEST).read_text(encoding="utf-8"))
    return hash_config({k: v["sha256"] for k, v in sorted(manifest["files"].items())})


def freeze(root: Path = ROOT, *, git_sha: str | None = None) -> int:
    """Boek alle trials van v10 in de ledger, vóór de eerste run. Geeft de nieuwe M."""
    cfg = load_config(root / CONFIG)
    ledger = HypothesisLedger(root / v2.LEDGER_PATH)
    require(not [e for e in ledger.entries() if e.get("unit") == "robust_book_v10_programme"],
            "v10 is al geboekt.", DataContractError)
    return ledger.append(LedgerEntry.from_config(
        wave=WAVE, unit="robust_book_v10_programme", market="crypto",
        config=asdict(cfg), git_sha=git_sha or current_git_sha(),
        data_hash=_data_hash(root),
        preregistration_id="v10-" + hash_config(asdict(cfg))[:32],
        n_trials=len(cfg.trials), result="interim",
        notes="v10: U1 plus een long-flat BTC/ETH-spot-sleeve op het stilstaande geld, aan bij "
              "stijgende netto long van asset managers op de CME (CFTC TFF) en/of trend. "
              "Regels en criteria vastgelegd vóór de data (conf/model/robust_book_v10.yaml). "
              "De prijsfamilie heeft alle vensters al gelezen (X4/Y4/Y5): de backtest kan "
              "alleen verwerpen; het schone bewijs is het vooruit-sample."))


def run_programme(root: Path = ROOT) -> dict[str, Any]:
    cfg = load_config(root / CONFIG)
    require(any(e.get("unit") == "robust_book_v10_programme"
                for e in HypothesisLedger(root / v2.LEDGER_PATH).entries()),
            "Eerst `freeze`: de trials staan niet in de ledger.", DataContractError)
    u1, close, tff = _inputs(root, cfg)
    close = close.reindex(u1.index)
    hs = cfg.criteria.holdout_start
    u1_frame = u1[["net", "rf", "cash_fraction"]].assign(excess=u1["net"] - u1["rf"])
    base = summary(u1_frame, hs)
    out_dir = root / ARTEFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    trials: dict[str, Any] = {}
    for name, trial in cfg.trials.items():
        target = sleeve_targets(close, tff, u1["committed"], cfg, trial)
        frame = book(u1, close, target, cfg)
        res = summary(frame, hs)
        res["sleeve_by_year"] = _years(frame["sleeve"] - frame["lost_interest"]
                                       - frame["costs"])
        res["criteria"] = evaluate(res, base, cfg.criteria)
        trials[name] = res
        frame.to_csv(out_dir / f"{name}_daily.csv", float_format="%.10g",
                     lineterminator="\n")
    bh = close["BTC"].pct_change().dropna()
    record = {"programme": "robust_book_v10", "primary": cfg.primary,
              "verdict": trials[cfg.primary]["criteria"]["verdict"],
              "ledger_total": HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses(),
              "window": [str(u1.index.min().date()), str(u1.index.max().date())],
              "u1": base, "trials": trials,
              "btc_buy_and_hold": {"cagr": _cagr(bh), "max_drawdown": _max_dd(bh),
                                   "years": _years(bh)}}
    (out_dir / "programme.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                            encoding="utf-8")
    return record


def main(argv: Sequence[str]) -> Any:
    require(len(argv) == 1 and argv[0] in ("freeze", "run"),
            "Gebruik: python -I -m tradebot.systematic.programme_v10 freeze|run",
            ConfigContractError)
    out = freeze(ROOT) if argv[0] == "freeze" else run_programme(ROOT)
    print(json.dumps(out, indent=1, default=str) if isinstance(out, dict) else out)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
