"""De bevroren uitkomsten van het robuuste boek, opnieuw uitgedrukt met een Sharpe die de
autocorrelatie van de dagrendementen meeneemt (audit 2026-10-24, SK-1).

    python -I -m tradebot.systematic.restate     # -> artefacts/research/restatement.json

Leest uitsluitend de bevroren dagreeksen (`artefacts/research/robust_book_v*/*_daily_*.csv`),
draait niets opnieuw, verandert geen poort, geen oordeel en geen ledger-entry. Per
kandidaat en venster: de bevroren iid-Sharpe naast de maand-, kwartaal- en Lo-Sharpe; voor
de vensters die er zijn ook de hele cyclus. Voor elke kandidaat met een W_DEV-reeks de DSR
op MAANDrendementen bij oplopende M, en voor het beste boek (v8-T2) het scenario van een
beursfaillissement met de vol die bij die Sharpe hoort.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..backtest.metrics import deflated_sharpe, max_drawdown
from ..utils.failfast import DataContractError, require
from ..validation.serial_sharpe import period_returns, serial_sharpe
from .market import BARS_PER_YEAR
from .programme_v2 import ROOT

__all__ = ["OUT", "dsr_monthly", "exchange_default_table", "restate", "window_record"]

RESEARCH = Path("artefacts/research")
OUT = RESEARCH / "restatement.json"
_FILE = re.compile(r"^(?P<cand>.+)_daily_(?P<window>w_dev|backcast|holdout|gate)\.csv$")
WINDOW_ORDER = ("backcast", "w_dev", "gate", "holdout")
M_GRID = (27, 79, 500, 2000)
FLAGSHIP = ("robust_book_v8", "T2_CARRY_TRANCHE_JUMPCAP")


def _f(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _longest_underwater(r: pd.Series) -> int:
    eq = (1.0 + r).cumprod()
    under = (eq < eq.cummax()).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def window_record(r: pd.Series) -> dict[str, Any]:
    """Alle maten van één reeks netto dagrendementen."""
    r = r.dropna()
    ser = serial_sharpe(r, bars_per_year=int(BARS_PER_YEAR), lags=(10, 30, 60, 120))
    eq = np.concatenate(([1.0], np.cumprod(1.0 + r.to_numpy())))
    years = len(r) / BARS_PER_YEAR
    rec: dict[str, Any] = {k: (_f(v) if isinstance(v, float) else v)
                           for k, v in ser.as_record().items()}
    rec.update({"start": str(r.index[0].date()), "end": str(r.index[-1].date()),
                "n_obs": int(len(r)), "cagr": _f(eq[-1] ** (1.0 / years) - 1.0),
                "max_drawdown": float(max_drawdown(eq)[0]),
                "longest_underwater_days": _longest_underwater(r),
                "yearly_return": {str(y): float((1.0 + c).prod() - 1.0)
                                  for y, c in r.groupby(r.index.year)}})
    if len(r) > BARS_PER_YEAR:
        r12 = (1.0 + r).rolling(int(BARS_PER_YEAR)).apply(np.prod, raw=True) - 1.0
        rec["rolling_12m_return"] = {"min": float(r12.min()), "p10": float(r12.quantile(0.1)),
                                     "median": float(r12.median())}
    return rec


def dsr_monthly(r: pd.Series, m_grid: Sequence[int] = M_GRID) -> dict[str, Any]:
    """De DSR op maandrendementen, met de normale benadering V = 1/T (geen trialvariantie)."""
    m = period_returns(r.dropna(), "ME").to_numpy()
    sd = float(m.std(ddof=1))
    require(m.size >= 6 and sd > 0.0, "Te weinig maanden voor een DSR.", DataContractError,
            n_months=int(m.size))
    sr = float(m.mean() / sd)
    g3, g4 = float(stats.skew(m)), float(stats.kurtosis(m, fisher=False))
    out: dict[str, Any] = {"n_months": int(m.size), "sharpe_per_month": sr,
                           "skew": g3, "kurtosis": g4}
    for M in m_grid:
        res = deflated_sharpe(sr, n_obs=int(m.size), n_trials=int(M), sr_variance=1.0 / m.size,
                              skew=g3, kurtosis=g4, bars_per_year=12)
        out[f"M_{M}"] = {"dsr": float(res.dsr),
                         "required_annual_sharpe": float(res.sr_zero * math.sqrt(12))}
    return out


def exchange_default_table(mu: float, sigma: float) -> dict[str, dict[str, float]]:
    """Sharpe met een sprongverlies: kans p per jaar, verlies LGD = 1/aantal beurzen.

    `(mu - p·LGD) / sqrt(sigma² + p(1-p)·LGD²)`: het verwachte verlies gaat van het
    rendement af, de Bernoulli-variantie komt bij de vol."""
    out: dict[str, dict[str, float]] = {}
    for p in (0.01, 0.03, 0.05):
        row = {}
        for venues in (1, 2, 3):
            lgd = 1.0 / venues
            row[f"{venues}_venue"] = float((mu - p * lgd) / math.sqrt(
                sigma ** 2 + p * (1.0 - p) * lgd ** 2))
        out[f"p_{p:.2f}"] = row
    return out


def _series(root: Path) -> dict[tuple[str, str], dict[str, pd.Series]]:
    out: dict[tuple[str, str], dict[str, pd.Series]] = {}
    for path in sorted((root / RESEARCH).glob("robust_book_v*/*_daily_*.csv")):
        hit = _FILE.match(path.name)
        require(hit is not None, "Onverwachte bestandsnaam.", DataContractError,
                path=str(path))
        assert hit is not None
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        out.setdefault((path.parent.name, hit["cand"]), {})[hit["window"]] = frame["net"]
    return out


def restate(root: Path = ROOT) -> dict[str, Any]:
    series = _series(root)
    require(bool(series), "Geen bevroren dagreeksen gevonden.", DataContractError,
            root=str(root))
    books: dict[str, Any] = {}
    for (prog, cand), windows in series.items():
        rec: dict[str, Any] = {w: window_record(windows[w])
                               for w in WINDOW_ORDER if w in windows}
        if {"backcast", "w_dev", "holdout"} <= set(windows):
            full = pd.concat([windows[w] for w in ("backcast", "w_dev", "holdout")]).sort_index()
            rec["full_cycle"] = window_record(full)
            logp = np.log1p(full)
            rec["full_cycle"]["share_of_log_pnl_by_year"] = {
                str(y): float(v / logp.sum()) for y, v in logp.groupby(full.index.year).sum().items()}
        if "w_dev" in windows:
            rec["dsr_monthly_w_dev"] = dsr_monthly(windows["w_dev"])
        books.setdefault(prog, {})[cand] = rec
    out: dict[str, Any] = {
        "what": "Bevroren dagreeksen opnieuw uitgedrukt; geen poort, oordeel of ledger-entry "
                "veranderd. sharpe_iid = de bevroren definitie (mean/sd*sqrt(365)).",
        "method": {"sharpe_monthly": "samengestelde maandrendementen, mean/sd*sqrt(12)",
                   "sharpe_quarterly": "samengestelde kwartaalrendementen, mean/sd*sqrt(4)",
                   "sharpe_loL": "Lo (2002) eta(365) met Bartlett-gewichten tot L lags",
                   "dsr_monthly_w_dev": "Bailey-Lopez de Prado op maandrendementen, V=1/T"},
        "books": books}
    prog, cand = FLAGSHIP
    flag = books.get(prog, {}).get(cand, {}).get("full_cycle")
    if flag is not None:
        mu, sigma = flag["cagr"], flag["ann_vol_monthly"]
        out["exchange_default_" + cand] = {
            "mu_cagr_full_cycle": mu, "sigma_monthly_implied": sigma,
            "sigma_reported_iid": flag["ann_vol_iid"],
            "honest_vol": exchange_default_table(mu, sigma),
            "as_reported_vol": exchange_default_table(mu, flag["ann_vol_iid"])}
    return out


def main(argv: Sequence[str]) -> dict[str, Any]:
    require(len(argv) == 0, "Gebruik: python -I -m tradebot.systematic.restate",
            DataContractError)
    out = restate(ROOT)
    (ROOT / OUT).write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
    print(f"{OUT}: {sum(len(v) for v in out['books'].values())} kandidaten")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
