"""Robuust boek v9: v5-H3 op Kraken EU, met de kasrente op wat het boek niet gebruikt.

    python -I -m tradebot.data.external_series             # op Actions: T-bill, €STR, funding
    python -I -m tradebot.systematic.programme_v9 freeze   # ledger + prereg (M = ledger)
    python -I -m tradebot.systematic.programme_v9 run      # de hele cyclus, poorten, jaren

WAAROM v9 BESTAAT. Eigenaarsbesluit 2026-10-25: weg van Binance (niet in Nederland), en elk
jaar winst. Op Kraken EU kan van de bevroren boeken alleen de regelset van v5-H3 (BTC en
ETH) draaien, met de spot als onderpand in de futureswallet (`carry/venues.py`). H3 verloor
nooit veel, maar stond in 2022, 2023, 2025 en 2026 85-100 % van de tijd in dollars die in
de backtest NIETS opbrachten. In werkelijkheid staan die in een geldmarktfonds. v9 telt
dat mee, en rekent de fees van Kraken.

WAT v9 NIET IS. Geen nieuwe edge: de regels, het been en de drempels zijn die van H3, bit
voor bit (de bevroren dagreeks van v5 is de invoer). De kasrente is geen alfa. Daarom meten
de Sharpe-poorten het rendement BOVEN de T-bill; zonder die aftrek telt rente als edge en
is elke kasstrategie een Sharpe-wonder. Alle vensters zijn verbruikt (v5 las W_DEV, de
backcast en de holdout, en v9 is ontworpen met de jaarcijfers van H3 in de hand); de enige
schone toets is het vooruit-sample. Ontwerp: `docs/research/2026-10-25-robust-book-v9-kraken-cash.md`.

DE BEREKENING, per dag t (alles uit bevroren of gehashte invoer):
1. `net_K` = de dagreeks van H3 met de fees van Kraken (`carry.venues.fee_overlay`, exact).
2. `committed_t` = spot / 0,70, hoogstens 1: het deel van het kapitaal dat H3 vastlegt
   (spot plus zijn USD-buffer).
3. Geld gaat pas `release_after_days` na het laatste gebruik naar het fonds, en komt
   `lead_days` vóór het volgende gebruik terug: `need_t = max(committed[t-R .. t+L])`. Het
   besluit van t+1 valt op t (lag 1); de tweede dag is de overboeking. Deze regel kan
   alleen rente wegnemen.
4. `cash_t = max(0, 1 − buffer − need_t) · max(0, DTB3_{t-1} − 50 bp) / 365`.
5. `net_t = net_K + cash_t`; `excess_t = net_t − DTB3_{t-1} / 365`.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from ..carry import venues as vn
from ..data import external_series as ex
from ..features.registry import current_git_sha
from ..registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from ..registry.preregistration import (
    freeze_preregistration,
    load_preregistration_spec,
    require_preregistration,
)
from ..utils.failfast import ConfigContractError, DataContractError, require
from ..utils.hashing import hash_config
from . import programme_v2 as v2
from .book import BookResult
from .evaluate import dsr_record, gate_z, summarize

__all__ = ["CANDIDATE", "V9Config", "build", "cash_fraction", "freeze", "load_config", "main",
           "run_programme"]

ROOT = v2.ROOT
CONFIG = Path("conf/model/robust_book_v9.yaml")
SPEC_PATH = Path("conf/research/preregistration_robust_book_v9.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v9")
FORWARD_LOCK = Path("artefacts/governance/holdout_lock_forward_2026_10.json")
CANDIDATE = "U1_KRAKEN_MAJORS_CASH"
WAVE = 9
#: Stress voor de sluitgrens: de alts-haircut van v6-v8, op BTC/ETH-onderpand.
STRESS_HAIRCUT = 0.35
#: Kraken retail: initiële marge 1 / 2:1 = 50 % van de notional, sluiting op de helft.
CLOSE_OUT_SHARE = 0.5
#: Wat de handel van H3 betreft, zijn deze poortmaten die van de batterij van v5.
INHERITED = ("sharpe_ratio_lag2_over_base", "plateau_fraction", "min_perturbation_sharpe", "pbo")
COSTS = ("fees", "spread", "slippage", "impact")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Base:
    programme: str
    candidate: str
    spot_leg: float
    margin_leg: float


@dataclass(frozen=True)
class Venue:
    name: str
    capital_usd: float
    maker: bool


@dataclass(frozen=True)
class Cash:
    haircut_apr: float
    day_count: int
    release_after_days: int
    lead_days: int
    operating_buffer: float


@dataclass(frozen=True)
class Windows:
    backcast_start: str
    backcast_end: str
    train_start: str
    train_end: str
    validate_start: str
    w_dev_end: str
    holdout_start: str
    holdout_end: str


@dataclass(frozen=True)
class V9Config:
    base: Base
    venue: Venue
    cash: Cash
    windows: Windows
    planned_trials: int


def _section(cls: type, raw: Mapping[str, Any], name: str) -> Any:
    keys = {f.name for f in fields(cls)}
    require(isinstance(raw, Mapping) and set(raw) == keys, "Sleutels wijken af van het schema.",
            ConfigContractError, section=name, expected=sorted(keys),
            got=sorted(raw) if isinstance(raw, Mapping) else None)
    return cls(**{k: raw[k] for k in keys})


def load_config(path: Path) -> V9Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(set(raw) == {"base", "venue", "cash", "windows", "planned_trials"},
            "Onbekende of ontbrekende sleutels in de v9-config.", ConfigContractError,
            got=sorted(raw))
    cfg = V9Config(base=_section(Base, raw["base"], "base"),
                   venue=_section(Venue, raw["venue"], "venue"),
                   cash=_section(Cash, raw["cash"], "cash"),
                   windows=_section(Windows, raw["windows"], "windows"),
                   planned_trials=int(raw["planned_trials"]))
    c, b = cfg.cash, cfg.base
    require(0.0 <= c.haircut_apr < 0.05 and c.day_count in (360, 365)
            and c.release_after_days >= 0 and c.lead_days >= 1
            and 0.0 <= c.operating_buffer < 1.0, "Een kasparameter buiten zijn bereik.",
            ConfigContractError, cash=asdict(c))
    require(0.0 < b.spot_leg < 1.0 and abs(b.spot_leg + b.margin_leg - 1.0) < 1e-9,
            "Spotbeen en buffer vormen samen het kapitaal.", ConfigContractError,
            base=asdict(b))
    w = cfg.windows
    order = [w.backcast_start, w.backcast_end, w.train_start, w.train_end, w.validate_start,
             w.w_dev_end, w.holdout_start, w.holdout_end]
    require(order == sorted(order), "De vensters lopen niet op.", ConfigContractError)
    require(cfg.planned_trials == 1, "v9 heeft één trial.", ConfigContractError)
    return cfg


def _params(cfg: V9Config) -> dict[str, Any]:
    return asdict(cfg)


# --------------------------------------------------------------------------- #
# Invoer en hashes
# --------------------------------------------------------------------------- #
def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _base_paths(root: Path, cfg: V9Config) -> dict[str, Path]:
    d = root / "artefacts/research" / cfg.base.programme
    out = {f"{cfg.base.candidate}_daily_{w}": d / f"{cfg.base.candidate}_daily_{w}.csv"
           for w in ("backcast", "w_dev", "holdout")}
    out[f"{cfg.base.candidate}_record"] = d / f"{cfg.base.candidate}.json"
    return out


def _data_hashes(root: Path, cfg: V9Config) -> tuple[tuple[str, str], ...]:
    """De bevroren H3-reeksen, de beursprofielen en de externe reeksen (via hun manifest)."""
    out = [(f"{cfg.base.programme}/{k}", _sha16(p)) for k, p in _base_paths(root, cfg).items()]
    out.append(("conf/execution/venues.yaml", _sha16(root / vn.VENUES_PATH)))
    manifest = json.loads((root / ex.MANIFEST).read_text(encoding="utf-8"))
    require(str(ex.DTB3_CSV) in manifest["files"], "Geen T-bill in het manifest.",
            DataContractError)
    out += [(f"external/{Path(k).stem}", rec["sha256"][:16])
            for k, rec in sorted(manifest["files"].items())]
    return tuple(sorted(out))


def _prereg(root: Path, cfg: V9Config):
    hashes = _data_hashes(root, cfg)
    return load_preregistration_spec(root / SPEC_PATH, data_hashes=hashes,
                                     parameters=_params(cfg)), hashes


def freeze(root: Path = ROOT, *, git_sha: str | None = None) -> Path:
    """Boek de trial in de ledger en bevries de pre-registratie, vóór de eerste run."""
    cfg = load_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    require(prereg.planned_trials == cfg.planned_trials, "planned_trials wijkt af.",
            DataContractError)
    require((root / FORWARD_LOCK).is_file(), "Het vooruit-slot van de carryfamilie ontbreekt.",
            DataContractError, path=str(FORWARD_LOCK))
    sha = git_sha or current_git_sha()
    ledger = HypothesisLedger(root / v2.LEDGER_PATH)
    require(not [e for e in ledger.entries()
                 if e.get("preregistration_id") == prereg.preregistration_id],
            "Dit programma is al geboekt.", DataContractError)
    ledger.append(LedgerEntry.from_config(
        wave=WAVE, unit="robust_book_v9_programme", market="crypto", config=_params(cfg),
        git_sha=sha, data_hash=hash_config(dict(hashes)),
        preregistration_id=prereg.preregistration_id, n_trials=cfg.planned_trials,
        result="interim",
        notes="Eén geplande trial: U1 = v5-H3 op Kraken EU (spot als onderpand, Kraken-fees) "
              "met kasrente op het ongebruikte kapitaal. Post-hoc: ontworpen met de "
              "jaarcijfers van H3 in de hand; alle vensters zijn verbruikt. Het schone "
              "bewijs is het vooruit-sample."))
    return freeze_preregistration(prereg, git_sha=sha,
                                  ledger_total_at_freeze=ledger.total_n_hypotheses(),
                                  directory=root / v2.PREREG_DIR)


# --------------------------------------------------------------------------- #
# De reeks
# --------------------------------------------------------------------------- #
def pair_fee_bps(root: Path, cfg: V9Config, *, maker: bool | None = None,
                 capital_usd: float | None = None) -> float:
    venue = vn.load_venues(root / vn.VENUES_PATH)[cfg.venue.name]
    m = cfg.venue.maker if maker is None else maker
    cap = cfg.venue.capital_usd if capital_usd is None else capital_usd
    return venue.spot_fee_bps(cap, maker=m) + venue.perp_fee_bps(maker=m)


def committed(frame: pd.DataFrame, cfg: V9Config) -> pd.Series:
    """Het deel van het kapitaal dat H3 vastlegt: spot plus zijn buffer, hoogstens 1."""
    return (frame["spot_notional"] / cfg.base.spot_leg).clip(lower=0.0, upper=1.0)


def cash_fraction(used: pd.Series, cfg: V9Config) -> pd.Series:
    """Wat in het fonds staat: niet gebruikt in [t − R, t + L], min de werkvoorraad."""
    c = cfg.cash
    ahead = used[::-1].rolling(c.lead_days + 1, min_periods=1).max()[::-1]
    need = ahead.rolling(c.release_after_days + 1, min_periods=1).max()
    return (1.0 - c.operating_buffer - need).clip(lower=0.0)


def _on_index(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Een dagreeks (naïeve datum) als de waarde van de VORIGE dag, op het UTC-raster."""
    prev = series.copy()
    prev.index = prev.index + pd.Timedelta(days=1)
    days = index.tz_convert(None).normalize()
    out = prev.reindex(days)
    require(not out.isna().any(), "Een rente ontbreekt op een dag van het boek.",
            DataContractError, name=series.name, missing=int(out.isna().sum()))
    return pd.Series(out.to_numpy(), index=index, name=series.name)


def build(frame: pd.DataFrame, dtb3: pd.Series, cfg: V9Config, pair_fee: float) -> pd.DataFrame:
    """De dagreeks van U1 uit de bevroren H3-reeks: zie de moduledocstring."""
    out = frame.copy()
    k = float(pair_fee) / vn.FROZEN_PAIR_FEE_BPS
    out["fees"] = frame["fees"] * k
    rf = _on_index(dtb3, frame.index)
    rate = (rf - cfg.cash.haircut_apr).clip(lower=0.0) / cfg.cash.day_count
    out["committed"] = committed(frame, cfg)
    out["cash_fraction"] = cash_fraction(out["committed"], cfg)
    out["cash"] = out["cash_fraction"] * rate
    out["rf"] = rf / cfg.cash.day_count
    out["net"] = vn.fee_overlay(frame, pair_fee) + out["cash"]
    out["excess"] = out["net"] - out["rf"]
    return out


# --------------------------------------------------------------------------- #
# Meten
# --------------------------------------------------------------------------- #
def _result(frame: pd.DataFrame, col: str = "net") -> BookResult:
    f = frame.copy()
    f["net"] = frame[col]
    empty = pd.DataFrame(index=frame.index)
    return BookResult(frame=f, held=empty, trades=empty)


def _slice(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return frame.loc[pd.Timestamp(a, tz="UTC"):pd.Timestamp(b, tz="UTC")]


def _finite(x: float | None) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _cagr(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (365.0 / len(r)) - 1.0)


def years(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per kalenderjaar: totaal, boven de T-bill, en waar het vandaan kwam."""
    out = {}
    for y, f in frame.groupby(frame.index.year):
        out[str(y)] = {"return": float((1.0 + f["net"]).prod() - 1.0),
                       "excess": float((1.0 + f["excess"]).prod() - 1.0),
                       "cash": float(f["cash"].sum()), "book": float((f["net"] - f["cash"]).sum()),
                       "t_bill": float(f["rf"].sum()), "n_days": len(f),
                       "cash_fraction": float(f["cash_fraction"].mean())}
    return out


def _cycle_summary(r: pd.Series) -> dict[str, Any]:
    yearly = {str(y): float((1.0 + c).prod() - 1.0) for y, c in r.groupby(r.index.year)}
    return {"cagr": _cagr(r), "min_year": min(yearly.values()), "yearly": yearly,
            "years_negative": sorted(y for y, v in yearly.items() if v < 0.0)}


def close_out_multiple(cfg: V9Config, *, haircut: float = STRESS_HAIRCUT,
                       retail_leverage: float = 2.0) -> float:
    """Hoeveel BTC/ETH mag stijgen zonder herbalancering voordat Kraken het boek sluit.

    Per eenheid short s = spot: wallet = s(1−h)x + s·m − s(x − 1) = s(1 + m − h·x); sluiting
    onder `CLOSE_OUT_SHARE` van de initiële marge s·x / lev. Grens: x = (1 + m) / (h + c/lev)."""
    m = cfg.base.margin_leg / cfg.base.spot_leg
    return (1.0 + m) / (haircut + CLOSE_OUT_SHARE / retail_leverage)


def _summary(f: pd.DataFrame, col: str, *, bootstrap: bool) -> dict[str, Any]:
    """`summarize`, of — zonder variantie (een venster zonder positie, een vaste rente) —
    een ongedefinieerde Sharpe. Die bindt elke Sharpe-poort, zoals in v5."""
    r = f[col]
    if float(r.std(ddof=1)) > 0.0:
        return summarize(_result(f, col), bootstrap=bootstrap)
    eq = (1.0 + r).cumprod()
    nan = float("nan")
    return {"sharpe": nan, "sharpe_se": nan, "sharpe_monthly": None, "cagr": _cagr(r),
            "max_drawdown": float((1.0 - eq / eq.cummax().clip(lower=1.0)).max()),
            "n_obs": len(r), "note": "geen variantie: de Sharpe is niet gedefinieerd"}


def _window_read(f: pd.DataFrame, dev_sr: float) -> dict[str, Any]:
    ex_s = _summary(f, "excess", bootstrap=True)
    net_s = _summary(f, "net", bootstrap=False)
    return {"excess": ex_s, "net": net_s,
            "z_vs_dev": gate_z(ex_s["sharpe"], ex_s["sharpe_se"], dev_sr),
            "n_liquidations": int((f["liquidation"] != 0).sum()) if "liquidation" in f else 0}


def _funding_check(root: Path, frame: pd.DataFrame) -> dict[str, Any] | None:
    """Kraken tegen Binance op BTC/ETH, waar beide noteren; en wat het verschil met U1 doet."""
    path = root / ex.FUNDING_CSV
    if not path.is_file():
        return None
    fund = ex.load_funding(root)
    coins = [c for c in ex.KRAKEN_PERPS if {f"kraken_{c}", f"binance_{c}"} <= set(fund)]
    if not coins:
        return None
    out: dict[str, Any] = {"per_coin": {}}
    diffs = []
    for c in coins:
        both = fund[[f"kraken_{c}", f"binance_{c}"]].dropna()
        d = both[f"kraken_{c}"] - both[f"binance_{c}"]
        diffs.append(d.rename(c))
        out["per_coin"][c] = {
            "first": str(both.index.min().date()), "last": str(both.index.max().date()),
            "n_days": len(both), "kraken_apr": float(both[f"kraken_{c}"].mean() * 365),
            "binance_apr": float(both[f"binance_{c}"].mean() * 365),
            "diff_apr": float(d.mean() * 365),
            "correlation": float(both.corr().iloc[0, 1])}
    # De split BTC/ETH per dag zit niet in de dagreeks: het gemiddelde van de munten.
    delta = pd.concat(diffs, axis=1).mean(axis=1)
    days = frame.index.tz_convert(None).normalize()
    adj = pd.Series(delta.reindex(days).fillna(0.0).to_numpy(), index=frame.index)
    out["u1_with_kraken_funding"] = _cycle_summary(frame["net"] + frame["spot_notional"] * adj)
    out["note"] = ("Benadering: de short per munt is niet in de dagreeks; het verschil is het "
                   "gemiddelde van BTC en ETH maal de spotnotional. De in- en uitstap van H3 "
                   "blijven die op de Binance-funding.")
    return out


def run_programme(root: Path = ROOT, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = load_config(root / CONFIG)
    prereg, hashes = _prereg(root, cfg)
    require_preregistration(prereg.preregistration_id, directory=root / v2.PREREG_DIR)
    m_total = v2.frozen_m(root, prereg.preregistration_id)
    w = cfg.windows
    base = vn.cycle(root, cfg.base.programme, cfg.base.candidate)
    fee = pair_fee_bps(root, cfg)
    u1 = build(base, ex.load_dtb3(root), cfg, fee)

    dev = _slice(u1, w.train_start, w.w_dev_end)
    dev_ex, dev_net = summarize(_result(dev, "excess")), summarize(_result(dev), bootstrap=False)
    train = summarize(_result(_slice(u1, w.train_start, w.train_end), "excess"), bootstrap=False)
    valid = summarize(_result(_slice(u1, w.validate_start, w.w_dev_end), "excess"),
                      bootstrap=False)
    doubled = dev["excess"] + dev[list(COSTS)].sum(axis=1)
    sr_2x = summarize(_result(dev.assign(excess=doubled), "excess"), bootstrap=False)["sharpe"]
    record = json.loads(_base_paths(root, cfg)[f"{cfg.base.candidate}_record"]
                        .read_text(encoding="utf-8"))
    inherited = {k: float(record["gate_metrics"][k]) for k in INHERITED}
    dsr = dsr_record(dev["excess"], n_trials=m_total, trial_sharpes_per_bar=())
    reads = {"backcast": _window_read(_slice(u1, w.backcast_start, w.backcast_end),
                                      dev_ex["sharpe"]),
             "holdout": _window_read(_slice(u1, w.holdout_start, w.holdout_end),
                                     dev_ex["sharpe"])}
    full = _cycle_summary(u1["net"])
    metrics: dict[str, float] = {
        "excess_sharpe_w_dev": dev_ex["sharpe"],
        "excess_sharpe_monthly_w_dev": dev_ex["sharpe_monthly"]
        if dev_ex["sharpe_monthly"] is not None else float("nan"),
        "net_cagr_w_dev": dev_net["cagr"],
        "excess_sharpe_train": train["sharpe"], "excess_sharpe_validate": valid["sharpe"],
        "max_drawdown_w_dev": dev_net["max_drawdown"],
        "excess_sharpe_ratio_2x_cost_over_base": sr_2x / dev_ex["sharpe"],
        "excess_sharpe_ci_low_w_dev": dev_ex["sharpe_ci_low"],
        "dsr_w_dev": float(dsr["dsr"]),
        "n_liquidations_w_dev": float((dev["liquidation"] != 0).sum()),
        "close_out_price_multiple_stress": close_out_multiple(cfg),
        "min_calendar_year_return": full["min_year"],
        "excess_cagr_full_cycle": _cagr(u1["excess"]),
        **inherited}
    for label, r in reads.items():
        metrics[f"{label}_sharpe_z_vs_dev"] = r["z_vs_dev"]
        metrics[f"{label}_max_drawdown"] = r["net"]["max_drawdown"]
        metrics[f"{label}_n_liquidations"] = float(r["n_liquidations"])
    metrics["backcast_excess_sharpe"] = reads["backcast"]["excess"]["sharpe"]
    gates = v2._gates(metrics, prereg)
    falsify = {c.name for c in prereg.stop_criteria if c.action == "falsify"}
    n_binding = int(sum(g["binds"] for g in gates.values()))
    verdict = "promote_to_paper_trading" if n_binding == 0 else (
        "falsified_out_of_sample" if any(g["binds"] for k, g in gates.items() if k in falsify)
        else "archived")

    sens: dict[str, Any] = {
        "h3_binance_frozen": _cycle_summary(base["net"]),
        "h3_kraken_without_cash": _cycle_summary(vn.fee_overlay(base, fee)),
        "u1_maker_1m": _cycle_summary(build(base, ex.load_dtb3(root), cfg, pair_fee_bps(
            root, cfg, maker=True))["net"]),
        "u1_new_account_taker": _cycle_summary(build(base, ex.load_dtb3(root), cfg, pair_fee_bps(
            root, cfg, capital_usd=0.0))["net"]),
        "t_bill_only": _cycle_summary(u1["rf"])}
    if (root / ex.ESTR_CSV).is_file():
        estr = _on_index(ex.load_estr(root), u1.index) / cfg.cash.day_count
        sens["u1_eur_hedged"] = _cycle_summary(u1["net"] - (u1["rf"] - estr))
        sens["u1_eur_hedged"]["note"] = ("Belegger in euro's die de dollar afdekt: de afdekking "
                                         "kost ongeveer DTB3 − €STR over het hele kapitaal.")
    funding = _funding_check(root, u1)

    out: dict[str, Any] = {
        "programme": "robust_book_v9", "candidate": CANDIDATE,
        "preregistration_id": prereg.preregistration_id, "m_total": m_total,
        "pair_fee_bps": fee, "gate_metrics": metrics, "gates": gates,
        "n_binding": n_binding, "verdict": verdict,
        "inherited_from": f"artefacts/research/{cfg.base.programme}/{cfg.base.candidate}.json",
        "w_dev": {"excess": dev_ex, "net": dev_net}, "train_excess": train,
        "validate_excess": valid, "dsr": dsr,
        "reads": {k: {"excess": v["excess"], "net": v["net"], "z_vs_dev": v["z_vs_dev"],
                      "n_liquidations": v["n_liquidations"]} for k, v in reads.items()},
        "full_cycle": full | {"excess_cagr": _cagr(u1["excess"]),
                              "years": years(u1)},
        "sensitivity": sens, "funding_kraken_vs_binance": funding,
        "caveat": "Post-hoc: v9 is ontworpen met de jaarcijfers van H3 in de hand, en W_DEV, "
                  "de backcast en de holdout zijn door v5 gelezen. Spread, impact en funding "
                  "zijn die van Binance. Het schone bewijs is het vooruit-sample.",
        EVIDENCE_KEY: NOT_ADMISSIBLE}
    art = root / ARTEFACT_DIR
    v2._dump(art / f"{CANDIDATE}.json", out)
    for label, (a, b) in {"backcast": (w.backcast_start, w.backcast_end),
                          "w_dev": (w.train_start, w.w_dev_end),
                          "holdout": (w.holdout_start, w.holdout_end)}.items():
        _slice(u1, a, b).to_csv(art / f"{CANDIDATE}_daily_{label}.csv", float_format="%.10g",
                                lineterminator="\n")
    result = {"promote_to_paper_trading": "accepted",
              "falsified_out_of_sample": "falsified"}.get(verdict, "archived")
    HypothesisLedger(root / v2.LEDGER_PATH).append(LedgerEntry.from_config(
        wave=WAVE, unit="robust_book_v9_verdict", market="crypto", config=_params(cfg),
        git_sha=current_git_sha(), data_hash=hash_config(dict(hashes)),
        preregistration_id=prereg.preregistration_id, n_trials=0, result=result,
        amends=hash_config(_params(cfg)),
        metrics={"candidate": CANDIDATE, "verdict": verdict,
                 "excess_sharpe_w_dev": _finite(dev_ex["sharpe"]),
                 "net_cagr_w_dev": float(dev_net["cagr"]), "full_cagr": full["cagr"],
                 "min_year": full["min_year"], "years_negative": full["years_negative"]},
        notes="Oordeel v9 over W_DEV, de backcast, de holdout en de hele cyclus (alle "
              "verbruikt). Het vooruit-sample volgt."))
    for y, rec in out["full_cycle"]["years"].items():
        log(f"{y}: {rec['return']:+.2%} (boven T-bill {rec['excess']:+.2%}, "
            f"kas {rec['cash']:+.2%}, in fonds {rec['cash_fraction']:.0%})")
    log(f"binding: {sorted(k for k, g in gates.items() if g['binds'])}")
    log(f"oordeel: {verdict}")
    return out


def main(argv: Sequence[str]) -> None:
    require(len(argv) == 1 and argv[0] in ("freeze", "run"),
            "Gebruik: python -I -m tradebot.systematic.programme_v9 {freeze|run}",
            DataContractError)
    if argv[0] == "freeze":
        print(freeze(ROOT))
    else:
        run_programme(ROOT)


if __name__ == "__main__":
    main(sys.argv[1:])
