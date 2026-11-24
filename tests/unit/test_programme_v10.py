"""`systematic/programme_v10.py`: de flow/trend-sleeve op het stilstaande geld van U1.

Synthetisch: een tijdelijke root met de echte config, een U1-reeks, spotprijzen, een
TFF-rapport en een lege ledger. Geen netwerk en geen bevroren artefact wordt gelezen."""
from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tradebot.data import positioning_series as ps
from tradebot.systematic import programme_v2 as v2
from tradebot.systematic import programme_v10 as p
from tradebot.utils.failfast import ConfigContractError, DataContractError

REPO = Path(__file__).resolve().parents[2]
DAYS = pd.date_range("2020-04-01", "2026-09-30", freq="D")


def _cfg() -> p.V10Config:
    return p.load_config(REPO / p.CONFIG)


def _prices(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(len(DAYS))
    drift = 0.0015 * np.sign(np.sin(t / 180.0))
    out = {}
    for i, coin in enumerate(p.COINS):
        r = drift + rng.normal(0.0, 0.03, len(DAYS)) * (1 + 0.3 * i)
        out[coin] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=DAYS)


def _tff(index: pd.DatetimeIndex) -> pd.DataFrame:
    tues = pd.date_range(index.min(), index.max(), freq="W-TUE")
    rows = []
    for k, d in enumerate(tues):
        for code, market in (("133741", "BITCOIN - CHICAGO MERCANTILE EXCHANGE"),
                             ("133742", "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE"),
                             ("146021", "ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE")):
            rows.append({"report_date": d, "market": market, "code": code,
                         "open_interest": 1000.0 if k != 5 else 0.0,
                         "asset_mgr_long": 300.0 + 50.0 * np.sin(k / 6.0),
                         "asset_mgr_short": 100.0, "lev_money_long": 0.0,
                         "lev_money_short": 0.0, "dealer_long": 0.0, "dealer_short": 0.0})
    return pd.DataFrame(rows)


def _u1(index: pd.DatetimeIndex) -> pd.DataFrame:
    carry = (index.year == 2021) | (index.year == 2024)
    committed = np.where(carry, 1.0, 0.0)
    rf = np.full(len(index), 0.04 / 365)
    cash_fraction = np.where(carry, 0.0, 0.95)
    net = np.where(carry, 0.0006, 0.0) + cash_fraction * (0.035 / 365)
    return pd.DataFrame({"net": net, "rf": rf, "cash_fraction": cash_fraction,
                         "committed": committed}, index=index)


# --------------------------------------------------------------------------- #
def test_the_real_config_loads_and_bad_ones_are_refused(tmp_path: Path):
    cfg = _cfg()
    assert cfg.primary == "F1_FLOW_TREND_10" and len(cfg.trials) == 4
    raw = yaml.safe_load((REPO / p.CONFIG).read_text(encoding="utf-8"))

    def bad(mutate, match: str) -> None:
        r = json.loads(json.dumps(raw))
        mutate(r)
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(r), encoding="utf-8")
        with pytest.raises(ConfigContractError, match=match):
            p.load_config(path)

    bad(lambda r: r.pop("costs"), "v10-config")
    bad(lambda r: r.__setitem__("trials", {}), "Geen trials")
    bad(lambda r: r["signal"].pop("trend_days"), "schema")
    bad(lambda r: r.__setitem__("primary", "X"), "hoofdkandidaat")
    bad(lambda r: r["trials"]["F1_FLOW_TREND_10"].__setitem__("use_trend", False)
        or r["trials"]["F1_FLOW_TREND_10"].__setitem__("use_flow", False), "trial buiten")
    bad(lambda r: r["sizing"].__setitem__("rebalance_band", 1.5), "parameter buiten")
    path = tmp_path / "list.yaml"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ConfigContractError, match="v10-config"):
        p.load_config(path)
    with pytest.raises(ConfigContractError, match="schema"):
        p._section(p.Costs, [1], "costs")


def test_trend_needs_full_history():
    close = pd.DataFrame({"BTC": [1.0, 2.0, 3.0, 1.0], "ETH": [3.0, 2.0, 1.0, 5.0]})
    on = p.trend_signal(close, 2)
    assert on["BTC"].tolist() == [False, True, True, False]
    assert on["ETH"].tolist() == [False, False, False, True]


def test_flow_is_point_in_time_uses_the_standard_contract_and_goes_stale():
    sig = _cfg().signal
    tues = pd.date_range("2021-01-05", periods=6, freq="W-TUE")
    longs = [100, 100, 100, 100, 150, 150]
    rows = [{"report_date": d, "market": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
             "open_interest": 1000.0, "asset_mgr_long": float(v), "asset_mgr_short": 0.0}
            for d, v in zip(tues, longs, strict=True)]
    rows.append({"report_date": tues[4], "market": "MICRO BITCOIN - CHICAGO MERCANTILE",
                 "open_interest": 1.0, "asset_mgr_long": 0.0, "asset_mgr_short": 1e9})
    idx = pd.date_range("2021-01-01", "2021-03-31", freq="D")
    on = p.flow_signal(pd.DataFrame(rows), idx, sig)
    first = tues[4] + pd.Timedelta(days=sig.release_lag_days)
    assert not on.loc[: first - pd.Timedelta(days=1), "BTC"].any()      # nog niet bekend
    assert on.loc[first, "BTC"]                                           # vanaf zondag
    stale = tues[5] + pd.Timedelta(days=sig.release_lag_days + sig.flow_stale_days + 1)
    assert not on.loc[stale:, "BTC"].any()                                # vervallen
    assert not on["ETH"].any()                                            # geen rapport


def test_sleeve_respects_the_carry_the_cap_and_the_trial():
    cfg = _cfg()
    close = _prices()
    tff = _tff(DAYS)
    u1 = _u1(DAYS)
    f1 = p.sleeve_targets(close, tff, u1["committed"], cfg, cfg.trials["F1_FLOW_TREND_10"])
    total = f1.sum(axis=1)
    assert (total <= 0.40 + 1e-12).all() and (f1 >= 0).all().all()
    assert (total[u1["committed"] >= 0.95] == 0).all()          # carry heeft voorrang
    assert total.max() > 0
    f3 = p.sleeve_targets(close, tff, u1["committed"], cfg, cfg.trials["F3_TREND_10"])
    f4 = p.sleeve_targets(close, tff, u1["committed"], cfg, cfg.trials["F4_FLOW_10"])
    assert (f3.sum(axis=1) >= 0).all() and not f3.equals(f4)
    assert (f1.gt(0) <= f3.gt(0)).all().all()                   # F1 = F3 én flow


def test_book_charges_costs_and_lost_interest_and_drifts_between_trades():
    cfg = _cfg()
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    u1 = pd.DataFrame({"net": [0.0] * 4, "rf": [0.05 / 365] * 4,
                       "cash_fraction": [0.9] * 4}, index=idx)
    close = pd.DataFrame({"BTC": [100.0, 110.0, 110.0, 110.0],
                          "ETH": [100.0, 100.0, 100.0, 100.0]}, index=idx)
    tgt = pd.DataFrame({"BTC": [0.2, 0.2, 0.0, 0.0], "ETH": [0.0] * 4}, index=idx)
    f = p.book(u1, close, tgt, cfg)
    fee = cfg.costs.spot_bps / 1e4
    assert f["costs"].iloc[0] == pytest.approx(0.2 * fee)
    assert f["sleeve"].iloc[1] == pytest.approx(0.02)
    assert f["costs"].iloc[1] == 0.0                     # drift binnen de band
    assert f["exposure"].iloc[1] > 0.2
    assert f["costs"].iloc[2] == pytest.approx(f["exposure"].iloc[1] * fee, rel=1e-3)
    assert f["lost_interest"].iloc[1] == pytest.approx(0.2 * 0.045 / 365)
    assert (f["net"] - f["rf"]).equals(f["excess"])


def test_summary_and_criteria():
    idx = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"net": 0.0003 + rng.normal(0, 0.002, len(idx)),
                          "rf": 0.0001}, index=idx)
    frame["excess"] = frame["net"] - frame["rf"]
    s = p.summary(frame, "2025-07-01")
    assert s["mean_exposure"] == 0.0 and s["costs"] == 0.0 and set(s["years"]) == {"2024", "2025"}
    flat = frame.assign(excess=0.0)
    assert np.isnan(p._monthly_sharpe(flat["excess"]))
    c = _cfg().criteria
    base = s | {"cagr": s["cagr"] - 0.05, "holdout_excess": s["holdout_excess"] - 0.01,
                "monthly_excess_sharpe": 9.0}
    ok = p.evaluate(s | {"monthly_excess_sharpe": 2.0, "max_drawdown": 0.05}, base, c)
    assert ok["verdict"] == "pass"
    bad = p.evaluate(s | {"monthly_excess_sharpe": float("nan"), "max_drawdown": 0.5}, base, c)
    assert bad["verdict"] == "rejected"
    assert not bad["monthly_excess_sharpe"]["pass"] and not bad["max_drawdown"]["pass"]


# --------------------------------------------------------------------------- #
def _root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / p.CONFIG).parent.mkdir(parents=True)
    shutil.copy(REPO / p.CONFIG, root / p.CONFIG)
    (root / v2.LEDGER_PATH).parent.mkdir(parents=True)
    (root / v2.LEDGER_PATH).write_text(json.dumps(
        {"version": 1, "seed_total": 10, "seed_note": "test", "entries": []}), encoding="utf-8")
    cfg = _cfg()
    d = root / "artefacts/research" / cfg.base.programme
    d.mkdir(parents=True)
    u1 = _u1(DAYS)
    u1.index = u1.index.tz_localize("UTC").rename("asof_ts")
    cuts = {"backcast": ("2020-04-01", "2020-12-31"), "w_dev": ("2021-01-01", "2025-06-30"),
            "holdout": ("2025-07-01", "2026-09-30")}
    for w, (a, b) in cuts.items():
        u1.loc[pd.Timestamp(a, tz="UTC"):pd.Timestamp(b, tz="UTC")].to_csv(
            d / f"{cfg.base.candidate}_daily_{w}.csv")
    prices = _prices()
    tff = _tff(DAYS)

    def spot(coin: str) -> pd.Series:
        return prices[coin].rename(coin).rename_axis("date")

    class R:
        def json(self) -> list:
            out = []
            for r in tff.to_dict("records"):
                row = {k: "0" for k in ps.TFF_COLUMNS}
                row.update({"report_date_as_yyyy_mm_dd": r["report_date"].isoformat(),
                            "market_and_exchange_names": r["market"],
                            "cftc_contract_market_code": r["code"],
                            "open_interest_all": str(r["open_interest"]),
                            "asset_mgr_positions_long": str(r["asset_mgr_long"]),
                            "asset_mgr_positions_short": str(r["asset_mgr_short"])})
                out.append(row)
            return out
    ps.fetch_all(root, get=lambda url, params: R(), spot=spot)
    return root


def test_freeze_then_run_writes_the_record(tmp_path: Path):
    root = _root(tmp_path)
    with pytest.raises(DataContractError, match="freeze"):
        p.run_programme(root)
    assert p.freeze(root, git_sha="abc1234") == 14
    with pytest.raises(DataContractError, match="al geboekt"):
        p.freeze(root, git_sha="abc1234")
    rec = p.run_programme(root)
    assert rec["ledger_total"] == 14 and rec["window"] == ["2020-04-01", "2026-09-30"]
    assert set(rec["trials"]) == set(_cfg().trials)
    assert rec["verdict"] in ("pass", "rejected")
    for name, t in rec["trials"].items():
        assert (root / p.ARTEFACT_DIR / f"{name}_daily.csv").is_file()
        assert set(t["criteria"]) >= {"cagr_uplift", "verdict"}
    on_disk = json.loads((root / p.ARTEFACT_DIR / "programme.json").read_text(encoding="utf-8"))
    assert on_disk["primary"] == "F1_FLOW_TREND_10"


def test_inputs_refuse_overlap_and_missing_coins(tmp_path: Path):
    root = _root(tmp_path)
    cfg = _cfg()
    d = root / "artefacts/research" / cfg.base.programme
    shutil.copy(d / f"{cfg.base.candidate}_daily_holdout.csv",
                d / f"{cfg.base.candidate}_daily_backcast.csv")
    with pytest.raises(DataContractError, match="overlappen"):
        p.load_u1(root, cfg)
    root2 = _root(tmp_path / "b")
    ps.fetch_all(root2, get=lambda url, params: (_ for _ in ()).throw(ValueError("x")),
                 spot=lambda coin: _prices()["BTC"].rename("BTC").rename_axis("date"))
    with pytest.raises(DataContractError, match="mist een munt"):
        p._inputs(root2, cfg)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_main_and_the_script_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    root = _root(tmp_path)
    monkeypatch.setattr(p, "ROOT", root)
    monkeypatch.setattr(p, "current_git_sha", lambda: "abc1234")
    assert p.main(["freeze"]) == 14
    assert capsys.readouterr().out.strip() == "14"
    rec = p.main(["run"])
    assert json.loads(capsys.readouterr().out)["primary"] == rec["primary"]
    with pytest.raises(ConfigContractError, match="Gebruik"):
        p.main([])
    monkeypatch.setattr(sys, "argv", ["programme_v10"])
    with pytest.raises(ConfigContractError, match="Gebruik"):
        runpy.run_module("tradebot.systematic.programme_v10", run_name="__main__")
