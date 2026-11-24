"""Robuust boek v11: U1 plus een trend-ensemble-sleeve op het stilstaande geld.

    python -I -m tradebot.systematic.programme_v11 freeze   # ledger (M += aantal trials)
    python -I -m tradebot.systematic.programme_v11 run      # hele cyclus, per jaar, criteria

Vervolg op v10 (`programme_v10`), met dezelfde invoer, boekhouding, kosten en criteria.
Anders is alleen het signaal en het ritme (`conf/model/robust_book_v11.yaml`):

* **score** per munt = het deel van de terugkijkperiodes L met `close_t > close_{t-L}`
  (time-series momentum over een ensemble; 0 .. 1). Zonder volle historie: 0;
* **vol-filter** (per trial): score 0 als de vol van 30 dagen boven zijn
  `vol_filter_quantile`-percentiel over `vol_filter_window` dagen staat;
* **gewicht** `score · (τ / σ) / 2`, teruggeschaald tot het plafond en tot het geld dat de
  carry niet vastlegt, zoals v10;
* **ritme:** bijkopen en herschalen alleen op `rebalance_weekday`; uitstappen elke dag.
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

from ..features.registry import current_git_sha
from ..registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from ..utils.failfast import ConfigContractError, DataContractError, require
from ..utils.hashing import hash_config
from . import programme_v2 as v2
from . import programme_v10 as v10

__all__ = ["V11Config", "ensemble_score", "freeze", "load_config", "main", "run_programme",
           "sleeve_targets", "vol_filter"]

ROOT = v2.ROOT
CONFIG = Path("conf/model/robust_book_v11.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v11")
WAVE = 11
UNIT = "robust_book_v11_programme"


@dataclass(frozen=True)
class Signal:
    lookbacks: tuple[int, ...]
    vol_filter_window: int
    vol_filter_quantile: float


@dataclass(frozen=True)
class Trial:
    vol_filter: bool
    vol_target: float
    max_weight: float


@dataclass(frozen=True)
class V11Config:
    v10: v10.V10Config
    v10_config: str
    signal: Signal
    rebalance_weekday: int
    trials: dict[str, Trial]
    primary: str


def _section(cls: type, raw: Any, name: str) -> Any:
    keys = {f.name for f in fields(cls)}
    require(isinstance(raw, Mapping) and set(raw) == keys, "Sleutels wijken af van het schema.",
            ConfigContractError, section=name, expected=sorted(keys),
            got=sorted(raw) if isinstance(raw, Mapping) else None)
    return cls(**{k: raw[k] for k in keys})


def load_config(path: Path, root: Path) -> V11Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    top = {"v10_config", "signal", "rebalance_weekday", "trials", "primary"}
    require(isinstance(raw, Mapping) and set(raw) == top,
            "Onbekende of ontbrekende sleutels in de v11-config.", ConfigContractError,
            got=sorted(raw) if isinstance(raw, Mapping) else None)
    require(isinstance(raw["trials"], Mapping) and bool(raw["trials"]), "Geen trials.",
            ConfigContractError)
    sig = _section(Signal, raw["signal"], "signal")
    sig = Signal(lookbacks=tuple(int(x) for x in sig.lookbacks),
                 vol_filter_window=int(sig.vol_filter_window),
                 vol_filter_quantile=float(sig.vol_filter_quantile))
    cfg = V11Config(v10=v10.load_config(root / str(raw["v10_config"])),
                    v10_config=str(raw["v10_config"]), signal=sig,
                    rebalance_weekday=int(raw["rebalance_weekday"]),
                    trials={str(k): _section(Trial, v, f"trials.{k}")
                            for k, v in raw["trials"].items()},
                    primary=str(raw["primary"]))
    require(cfg.primary in cfg.trials, "De hoofdkandidaat is geen trial.", ConfigContractError)
    require(bool(sig.lookbacks) and min(sig.lookbacks) >= 1 and sig.vol_filter_window >= 30
            and 0.0 < sig.vol_filter_quantile < 1.0 and 0 <= cfg.rebalance_weekday <= 6,
            "Een parameter buiten zijn bereik.", ConfigContractError)
    for name, t in cfg.trials.items():
        require(0.0 < t.vol_target <= 0.5 and 0.0 < t.max_weight <= 1.0,
                "Een trial buiten zijn bereik.", ConfigContractError, trial=name)
    return cfg


def ensemble_score(close: pd.DataFrame, lookbacks: Sequence[int]) -> pd.DataFrame:
    """Het deel van de periodes met `close_t > close_{t-L}`; zonder volle historie 0."""
    votes = [(close > close.shift(L)).astype(float).where(close.shift(L).notna(), np.nan)
             for L in lookbacks]
    total = sum(votes[1:], votes[0])
    return (total / len(lookbacks)).fillna(0.0)


def vol_filter(sigma: pd.DataFrame, window: int, quantile: float) -> pd.DataFrame:
    """True = de vol is NIET extreem (onder zijn percentiel over het venster, causaal)."""
    thr = sigma.rolling(window, min_periods=window).quantile(quantile)
    return ~(sigma > thr)


def sleeve_targets(close: pd.DataFrame, committed: pd.Series, cfg: V11Config,
                   trial: Trial) -> pd.DataFrame:
    base, z = cfg.v10.base, cfg.v10.sizing
    score = ensemble_score(close, cfg.signal.lookbacks)
    ret = close.pct_change()
    sigma = np.sqrt((ret ** 2).ewm(span=z.vol_span, min_periods=z.vol_span).mean()
                    * v10.DAYS)
    if trial.vol_filter:
        score = score.where(vol_filter(sigma, cfg.signal.vol_filter_window,
                                       cfg.signal.vol_filter_quantile), 0.0)
    ok = sigma.gt(0.0) & close.notna()
    raw = (score * trial.vol_target / sigma).where(ok, 0.0).fillna(0.0) / len(v10.COINS)
    room = 1.0 - base.operating_buffer - committed.reindex(close.index).fillna(1.0)
    cap = np.minimum(trial.max_weight, room.clip(lower=0.0))
    total = raw.sum(axis=1)
    factor = (cap / total.where(total > 0)).clip(upper=1.0).fillna(0.0)
    return raw.mul(factor, axis=0)


def freeze(root: Path = ROOT, *, git_sha: str | None = None) -> int:
    cfg = load_config(root / CONFIG, root)
    ledger = HypothesisLedger(root / v2.LEDGER_PATH)
    require(not [e for e in ledger.entries() if e.get("unit") == UNIT],
            "v11 is al geboekt.", DataContractError)
    params = asdict(cfg)
    return ledger.append(LedgerEntry.from_config(
        wave=WAVE, unit=UNIT, market="crypto", config=params,
        git_sha=git_sha or current_git_sha(), data_hash=v10._data_hash(root),
        preregistration_id="v11-" + hash_config(params)[:32],
        n_trials=len(cfg.trials), result="interim",
        notes="v11: U1 plus een trend-ensemble-sleeve (TSMOM over 20-320 d), wekelijks "
              "herbalanceren, optioneel vol-filter. Post-hoc na v10 (die de prijzen en 2022 "
              "liet zien); criteria van v10 ongewijzigd. Schoon bewijs: vooruit-sample."))


def run_programme(root: Path = ROOT) -> dict[str, Any]:
    cfg = load_config(root / CONFIG, root)
    require(any(e.get("unit") == UNIT for e in HypothesisLedger(root / v2.LEDGER_PATH).entries()),
            "Eerst `freeze`: de trials staan niet in de ledger.", DataContractError)
    u1, close, _ = v10._inputs(root, cfg.v10)
    close = close.reindex(u1.index)
    crit = cfg.v10.criteria
    base = v10.summary(u1[["net", "rf", "cash_fraction"]].assign(excess=u1["net"] - u1["rf"]),
                       crit.holdout_start)
    rebalance = (u1.index.dayofweek == cfg.rebalance_weekday)
    out_dir = root / ARTEFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    trials: dict[str, Any] = {}
    for name, trial in cfg.trials.items():
        target = sleeve_targets(close, u1["committed"], cfg, trial)
        frame = v10.book(u1, close, target, cfg.v10, rebalance=np.asarray(rebalance))
        res = v10.summary(frame, crit.holdout_start)
        res["sleeve_by_year"] = v10._years(frame["sleeve"] - frame["lost_interest"]
                                           - frame["costs"])
        res["criteria"] = v10.evaluate(res, base, crit)
        trials[name] = res
        frame.to_csv(out_dir / f"{name}_daily.csv", float_format="%.10g", lineterminator="\n")
    record = {"programme": "robust_book_v11", "primary": cfg.primary,
              "verdict": trials[cfg.primary]["criteria"]["verdict"],
              "ledger_total": HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses(),
              "window": [str(u1.index.min().date()), str(u1.index.max().date())],
              "u1": base, "trials": trials}
    (out_dir / "programme.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                            encoding="utf-8")
    return record


def main(argv: Sequence[str]) -> Any:
    require(len(argv) == 1 and argv[0] in ("freeze", "run"),
            "Gebruik: python -I -m tradebot.systematic.programme_v11 freeze|run",
            ConfigContractError)
    out = freeze(ROOT) if argv[0] == "freeze" else run_programme(ROOT)
    print(json.dumps(out, indent=1, default=str) if isinstance(out, dict) else out)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
