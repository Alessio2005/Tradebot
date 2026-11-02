"""Robuust boek v12: U1 plus een long/short-trend-sleeve (long spot, short perp).

    python -I -m tradebot.systematic.programme_v12 freeze   # ledger (M += aantal trials)
    python -I -m tradebot.systematic.programme_v12 run      # hele cyclus, per jaar, criteria

Vervolg op v10/v11 (`conf/model/robust_book_v12.yaml`). Het verschil: het signaal heeft een
TEKEN. Boven de trend long spot, eronder short perp; de short ontvangt de funding.

* `sma`: +1 als de close boven zijn SMA (v10, `trend_days`), anders -1; dagelijks, met de
  band van v10;
* `ensemble`: `2 · frac - 1` uit het ensemble van v11; alleen op de herbalanceerdag
  bijkopen of herschalen, sluiten of omdraaien elke dag;
* gewicht `s · (τ / σ) / 2`; Σ|w| ≤ min(plafond, vrij geld van de carry).

Per dag: `net = U1 + Σ w·r + Σ max(-w, 0)·funding - min(Σ|w|, kas)·kasrente - kosten`.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..data import external_series as ex
from ..features.registry import current_git_sha
from ..registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from ..utils.failfast import ConfigContractError, DataContractError, require
from ..utils.hashing import hash_config
from . import programme_v2 as v2
from . import programme_v10 as v10
from . import programme_v11 as v11

__all__ = ["V12Config", "book", "freeze", "load_config", "main", "run_programme",
           "signed_targets"]

ROOT = v2.ROOT
CONFIG = Path("conf/model/robust_book_v12.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v12")
WAVE = 12
UNIT = "robust_book_v12_programme"
SIGNALS = ("sma", "ensemble")


@dataclass(frozen=True)
class Trial:
    signal: str
    vol_target: float
    max_weight: float


@dataclass(frozen=True)
class V12Config:
    v11: v11.V11Config
    v10_config: str
    v11_config: str
    funding_columns: dict[str, str]
    trials: dict[str, Trial]
    primary: str


def load_config(path: Path, root: Path) -> V12Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    top = {"v10_config", "v11_config", "funding_columns", "trials", "primary"}
    require(isinstance(raw, Mapping) and set(raw) == top,
            "Onbekende of ontbrekende sleutels in de v12-config.", ConfigContractError,
            got=sorted(raw) if isinstance(raw, Mapping) else None)
    require(isinstance(raw["trials"], Mapping) and bool(raw["trials"]), "Geen trials.",
            ConfigContractError)
    keys = {f.name for f in fields(Trial)}
    trials = {}
    for k, v in raw["trials"].items():
        require(isinstance(v, Mapping) and set(v) == keys and v["signal"] in SIGNALS
                and 0.0 < v["vol_target"] <= 0.5 and 0.0 < v["max_weight"] <= 1.0,
                "Een trial buiten zijn schema of bereik.", ConfigContractError, trial=k)
        trials[str(k)] = Trial(**v)
    fc = dict(raw["funding_columns"])
    require(set(fc) == set(v10.COINS), "Funding per munt ontbreekt.", ConfigContractError)
    cfg = V12Config(v11=v11.load_config(root / str(raw["v11_config"]), root),
                    v10_config=str(raw["v10_config"]), v11_config=str(raw["v11_config"]),
                    funding_columns=fc, trials=trials, primary=str(raw["primary"]))
    require(cfg.v11.v10_config == cfg.v10_config, "v10 en v11 wijzen naar andere configs.",
            ConfigContractError)
    require(cfg.primary in cfg.trials, "De hoofdkandidaat is geen trial.", ConfigContractError)
    return cfg


def signed_targets(close: pd.DataFrame, committed: pd.Series, cfg: V12Config,
                   trial: Trial) -> pd.DataFrame:
    c10 = cfg.v11.v10
    if trial.signal == "sma":
        sma = close.rolling(c10.signal.trend_days, min_periods=c10.signal.trend_days).mean()
        score = np.sign(close - sma).where(sma.notna(), 0.0).fillna(0.0)
    else:
        frac = v11.ensemble_score(close, cfg.v11.signal.lookbacks)
        full = pd.concat([close.shift(L).notna() for L in cfg.v11.signal.lookbacks]).groupby(
            level=0).all()
        score = (2.0 * frac - 1.0).where(full, 0.0)
    ret = close.pct_change()
    z = c10.sizing
    sigma = np.sqrt((ret ** 2).ewm(span=z.vol_span, min_periods=z.vol_span).mean() * v10.DAYS)
    ok = sigma.gt(0.0) & close.notna()
    raw = (score * trial.vol_target / sigma).where(ok, 0.0).fillna(0.0) / len(v10.COINS)
    room = 1.0 - c10.base.operating_buffer - committed.reindex(close.index).fillna(1.0)
    cap = np.minimum(trial.max_weight, room.clip(lower=0.0))
    gross = raw.abs().sum(axis=1)
    factor = (cap / gross.where(gross > 0)).clip(upper=1.0).fillna(0.0)
    return raw.mul(factor, axis=0)


def book(u1: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame, target: pd.DataFrame,
         cfg: v10.V10Config, rebalance: np.ndarray | None = None) -> pd.DataFrame:
    """Als `programme_v10.book`, maar met een teken: short ontvangt de funding."""
    ret = close.pct_change().fillna(0.0).to_numpy()
    fund = funding.reindex(close.index).fillna(0.0).to_numpy()
    tgt = target.to_numpy()
    rate = ((u1["rf"] * v10.DAYS - cfg.base.haircut_apr).clip(lower=0.0) / v10.DAYS).to_numpy()
    band, fee = cfg.sizing.rebalance_band, cfg.costs.spot_bps / 1e4
    n = len(u1)
    held = np.zeros(tgt.shape[1])
    sleeve, carry, lost, cost, expo = (np.zeros(n) for _ in range(5))
    for t in range(n):
        pnl = float(held @ ret[t])
        carry[t] = float(np.maximum(-held, 0.0) @ fund[t])
        lost[t] = min(np.abs(held).sum(), float(u1["cash_fraction"].iloc[t])) * rate[t]
        sleeve[t] = pnl
        day = float(u1["net"].iloc[t]) + pnl + carry[t] - lost[t]
        held = held * (1.0 + ret[t]) / (1.0 + day)
        want = tgt[t]
        flip = np.sign(want) != np.sign(held)
        off = np.abs(held - want) > band * np.where(want != 0, np.abs(want), np.inf)
        trade = flip | off
        if rebalance is not None and not rebalance[t]:
            trade &= flip
        new = np.where(trade, want, held)
        cost[t] = float(np.abs(new - held).sum()) * fee
        held = new
        expo[t] = float(np.abs(held).sum())
    out = u1[["net", "rf", "cash_fraction"]].rename(columns={"net": "u1"}).copy()
    out["sleeve"], out["funding"], out["lost_interest"] = sleeve, carry, lost
    out["costs"], out["exposure"] = cost, expo
    out["net"] = (out["u1"] + out["sleeve"] + out["funding"] - out["lost_interest"]
                  - out["costs"])
    out["excess"] = out["net"] - out["rf"]
    return out


def freeze(root: Path = ROOT, *, git_sha: str | None = None) -> int:
    cfg = load_config(root / CONFIG, root)
    ledger = HypothesisLedger(root / v2.LEDGER_PATH)
    require(not [e for e in ledger.entries() if e.get("unit") == UNIT],
            "v12 is al geboekt.", DataContractError)
    params = asdict(cfg)
    return ledger.append(LedgerEntry.from_config(
        wave=WAVE, unit=UNIT, market="crypto", config=params,
        git_sha=git_sha or current_git_sha(), data_hash=v10._data_hash(root),
        preregistration_id="v12-" + hash_config(params)[:32],
        n_trials=len(cfg.trials), result="interim",
        notes="v12: U1 plus een long/short-trend-sleeve (long spot, short perp met funding). "
              "Post-hoc na v10 en v11 (2022 bindt elke long-only sleeve); criteria van v10 "
              "ongewijzigd. Schoon bewijs: vooruit-sample."))


def run_programme(root: Path = ROOT) -> dict[str, Any]:
    cfg = load_config(root / CONFIG, root)
    require(any(e.get("unit") == UNIT for e in HypothesisLedger(root / v2.LEDGER_PATH).entries()),
            "Eerst `freeze`: de trials staan niet in de ledger.", DataContractError)
    c10 = cfg.v11.v10
    u1, close, _ = v10._inputs(root, c10)
    close = close.reindex(u1.index)
    raw_f = ex.load_funding(root)
    funding = pd.DataFrame({c: raw_f[col] for c, col in cfg.funding_columns.items()})
    funding.index = funding.index + pd.Timedelta(days=1)  # dagsom van t telt op de close t+1
    crit = c10.criteria
    base = v10.summary(u1[["net", "rf", "cash_fraction"]].assign(excess=u1["net"] - u1["rf"]),
                       crit.holdout_start)
    weekly = np.asarray(u1.index.dayofweek == cfg.v11.rebalance_weekday)
    out_dir = root / ARTEFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    trials: dict[str, Any] = {}
    for name, trial in cfg.trials.items():
        target = signed_targets(close, u1["committed"], cfg, trial)
        frame = book(u1, close, funding, target, c10,
                     rebalance=weekly if trial.signal == "ensemble" else None)
        res = v10.summary(frame, crit.holdout_start)
        res["sleeve_by_year"] = v10._years(frame["sleeve"] + frame["funding"]
                                           - frame["lost_interest"] - frame["costs"])
        res["short_share"] = float((target < 0).any(axis=1).mean())
        res["criteria"] = v10.evaluate(res, base, crit)
        trials[name] = res
        frame.to_csv(out_dir / f"{name}_daily.csv", float_format="%.10g", lineterminator="\n")
    record = {"programme": "robust_book_v12", "primary": cfg.primary,
              "verdict": trials[cfg.primary]["criteria"]["verdict"],
              "ledger_total": HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses(),
              "window": [str(u1.index.min().date()), str(u1.index.max().date())],
              "u1": base, "trials": trials}
    (out_dir / "programme.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                            encoding="utf-8")
    return record


def main(argv: Sequence[str]) -> Any:
    require(len(argv) == 1 and argv[0] in ("freeze", "run"),
            "Gebruik: python -I -m tradebot.systematic.programme_v12 freeze|run",
            ConfigContractError)
    out = freeze(ROOT) if argv[0] == "freeze" else run_programme(ROOT)
    print(json.dumps(out, indent=1, default=str) if isinstance(out, dict) else out)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
