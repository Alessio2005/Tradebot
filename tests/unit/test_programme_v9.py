"""Robuust boek v9 (`systematic/programme_v9.py`): config, kasrente, freeze en run.

De run draait op de ECHTE bevroren H3-reeksen van v5, met een synthetische T-bill (4 %),
€STR (2 %) en funding, in een tijdelijke root met een kopie van de ledger.
"""
from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tradebot.carry import venues as vn
from tradebot.data import external_series as ex
from tradebot.systematic import programme_v2 as v2
from tradebot.systematic import programme_v9 as v9
from tradebot.utils.failfast import ConfigContractError, DataContractError
from tradebot.validation.holdout import HoldoutAlreadyUsed

ROOT = Path(__file__).resolve().parents[2]
DAYS = pd.date_range("2019-09-01", "2026-09-30", freq="D")


def _fred(rate: float = 4.0) -> str:
    """Rond `rate`, met een weekpatroon: een vaste rente heeft geen variantie."""
    return "observation_date,DTB3\n" + "".join(
        f"{d.date()},{rate + 0.01 * (i % 7)}\n" for i, d in enumerate(DAYS))


def _ecb(rate: float = 2.0) -> str:
    return "TIME_PERIOD,OBS_VALUE\n" + "".join(f"{d.date()},{rate}\n" for d in DAYS)


class _Resp:
    def __init__(self, *, text: str = "", payload: dict | None = None) -> None:
        self.text, self._payload = text, payload

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload


def _kraken(sym: str) -> dict:
    hours = pd.date_range("2023-01-01", "2023-03-31 23:00", freq="h")
    rate = 2e-6 if sym == "PF_XBTUSD" else 3e-6
    return {"result": "success",
            "rates": [{"timestamp": h.isoformat() + "Z", "relativeFundingRate": rate}
                      for h in hours]}


def _get(*, ecb: bool = True, kraken: bool = True):
    def get(url: str, params: dict[str, str]) -> _Resp:
        if url == ex.FRED_URL:
            return _Resp(text=_fred())
        if url == ex.ECB_ESTR_URL:
            if not ecb:
                raise ex.requests.ConnectionError("geen ECB")
            return _Resp(text=_ecb())
        if not kraken:
            raise ex.requests.HTTPError("403")
        return _Resp(payload=_kraken(params["symbol"]))
    return get


def _binance(coin: str) -> pd.Series:
    idx = pd.date_range("2023-01-01", "2023-03-31", freq="D", name="date")
    return pd.Series(1e-4, index=idx, name=f"binance_{coin}")


def _root(tmp_path: Path, *, ecb: bool = True, kraken: bool = True) -> Path:
    for rel in (v9.CONFIG, v9.SPEC_PATH, vn.VENUES_PATH, v9.FORWARD_LOCK, v2.LEDGER_PATH):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    shutil.copytree(ROOT / "artefacts/research/robust_book_v5",
                    tmp_path / "artefacts/research/robust_book_v5")
    ex.fetch_all(tmp_path, get=_get(ecb=ecb, kraken=kraken), binance=_binance)
    return tmp_path


@pytest.fixture(scope="module")
def cfg() -> v9.V9Config:
    return v9.load_config(ROOT / v9.CONFIG)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _bad(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load((ROOT / v9.CONFIG).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "v9.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda r: r.update(extra=1), "Onbekende of ontbrekende"),
    (lambda r: r["cash"].update(typo=1), "Sleutels wijken af"),
    (lambda r: r.update(cash=[1]), "Sleutels wijken af"),
    (lambda r: r["cash"].update(lead_days=0), "kasparameter"),
    (lambda r: r["cash"].update(day_count=252), "kasparameter"),
    (lambda r: r["base"].update(margin_leg=0.5), "Spotbeen en buffer"),
    (lambda r: r["windows"].update(holdout_start="2021-01-01"), "lopen niet op"),
    (lambda r: r.update(planned_trials=2), "één trial"),
])
def test_the_config_refuses_what_it_does_not_know(tmp_path: Path, mutate, match):
    with pytest.raises(ConfigContractError, match=match):
        v9.load_config(_bad(tmp_path, mutate))


def test_the_committed_config_is_the_one_the_prereg_names(cfg):
    assert cfg.base.candidate == "H3_BASIS_MAJORS" and cfg.venue.name == "kraken_eu"
    assert v9.pair_fee_bps(ROOT, cfg) == 27.0
    assert v9.pair_fee_bps(ROOT, cfg, maker=True) == 12.0
    assert v9.pair_fee_bps(ROOT, cfg, capital_usd=0.0) == 85.0


# --------------------------------------------------------------------------- #
# De kasrente
# --------------------------------------------------------------------------- #
def test_cash_waits_a_week_after_use_and_leaves_two_days_before(cfg):
    used = pd.Series([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0], dtype=float)
    frac = v9.cash_fraction(used, cfg).round(10).tolist()
    # twee dagen vóór gebruik weg uit het fonds; zeven dagen na het laatste gebruik terug.
    assert frac[:2] == [0.95, 0.95]
    assert frac[2:13] == [0.0] * 11
    assert frac[13:] == [0.45] * 4


def test_build_adds_cash_and_kraken_fees_and_measures_over_the_t_bill(cfg):
    idx = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC", name="asof_ts")
    spot = [0.0] * 4 + [0.7] * 4 + [0.0] * 4
    frame = pd.DataFrame({"net": 0.001, "fees": -1.55e-4, "spot_notional": spot}, index=idx)
    dtb3 = pd.Series(0.04, index=pd.date_range("2023-12-01", "2024-01-31", name="date"),
                     name="dtb3")
    u = v9.build(frame, dtb3, cfg, pair_fee=31.0)
    assert u["fees"].tolist() == pytest.approx([-3.1e-4] * 12)
    day = 0.035 / 365  # DTB3 4 % min 50 bp
    assert u["cash"].iloc[0] == pytest.approx(0.95 * day)
    assert u["cash"].iloc[2:].max() == 0.0          # tot een week na de laatste dag in gebruik
    assert u["net"].iloc[0] == pytest.approx(0.001 - 1.55e-4 + 0.95 * day)
    assert u["excess"].iloc[0] == pytest.approx(u["net"].iloc[0] - 0.04 / 365)
    assert u["committed"].tolist() == [0.0] * 4 + [1.0] * 4 + [0.0] * 4
    with pytest.raises(DataContractError, match="rente ontbreekt"):
        v9.build(frame, dtb3.loc["2024-01-05":], cfg, pair_fee=31.0)


def test_negative_rates_do_not_cost_the_fund_anything(cfg):
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame({"net": 0.0, "fees": 0.0, "spot_notional": 0.0}, index=idx)
    dtb3 = pd.Series(0.002, index=pd.date_range("2023-12-30", "2024-01-05", name="date"))
    assert v9.build(frame, dtb3, cfg, pair_fee=15.5)["cash"].tolist() == [0.0] * 3


def test_a_window_without_variance_has_no_sharpe():
    idx = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    s = v9._summary(pd.DataFrame({"excess": -1e-5}, index=idx), "excess", bootstrap=True)
    assert s["sharpe"] != s["sharpe"] and s["sharpe_monthly"] is None
    assert s["cagr"] < 0.0 and s["max_drawdown"] == pytest.approx(1 - (1 - 1e-5) ** 40)


def test_the_close_out_bound(cfg):
    assert v9.close_out_multiple(cfg) == pytest.approx((1 + 0.3 / 0.7) / 0.60)
    assert v9.close_out_multiple(cfg, haircut=0.10) == pytest.approx((1 + 0.3 / 0.7) / 0.35)


# --------------------------------------------------------------------------- #
# Freeze en run
# --------------------------------------------------------------------------- #
def test_freeze_then_run_books_the_trial_and_its_verdict(tmp_path: Path):
    root = _root(tmp_path)
    before = len(json.loads((root / v2.LEDGER_PATH).read_text(encoding="utf-8"))["entries"])
    path = v9.freeze(root, git_sha="abc1234")
    assert path.is_file()
    with pytest.raises(DataContractError, match="al geboekt"):
        v9.freeze(root, git_sha="abc1234")
    logs: list[str] = []
    out = v9.run_programme(root, log=logs.append)
    entries = json.loads((root / v2.LEDGER_PATH).read_text(encoding="utf-8"))["entries"]
    assert [e["unit"] for e in entries[before:]] == ["robust_book_v9_programme",
                                                     "robust_book_v9_verdict"]
    assert out["m_total"] == entries[before]["n_trials"] + sum(
        e["n_trials"] for e in entries[:before])
    assert out["verdict"] in {"promote_to_paper_trading", "falsified_out_of_sample", "archived"}
    assert out["gates"]["g2_cagr_below_target"]["binds"]       # het rendementsmandaat
    assert set(out["full_cycle"]["years"]) == {str(y) for y in range(2020, 2027)}
    assert {"u1_eur_hedged", "u1_maker_1m", "h3_kraken_without_cash"} <= set(out["sensitivity"])
    f = out["funding_kraken_vs_binance"]
    assert f["per_coin"]["BTC"]["diff_apr"] == pytest.approx((24 * 2e-6 - 1e-4) * 365)
    for g in ("g13_losing_calendar_year", "g14_does_not_beat_cash", "g11_close_out_too_close"):
        assert g in out["gates"]
    for w in ("backcast", "w_dev", "holdout"):
        assert (root / v9.ARTEFACT_DIR / f"{v9.CANDIDATE}_daily_{w}.csv").is_file()
    assert any(line.startswith("oordeel:") for line in logs)
    # De kas gaat niet in de Sharpe-poort: boven de T-bill van 4 % is H3 in 2023 negatief.
    assert out["full_cycle"]["years"]["2023"]["excess"] < 0.0 < out["full_cycle"]["years"][
        "2023"]["return"]


def test_without_estr_and_kraken_the_run_skips_those_checks(tmp_path: Path):
    root = _root(tmp_path, ecb=False, kraken=False)
    v9.freeze(root, git_sha="abc1234")
    out = v9.run_programme(root, log=lambda s: None)
    assert "u1_eur_hedged" not in out["sensitivity"]
    assert out["funding_kraken_vs_binance"] is None
    (root / ex.FUNDING_CSV).unlink()
    assert v9._funding_check(root, pd.DataFrame()) is None


def test_freeze_refuses_without_the_forward_lock_or_the_t_bill(tmp_path: Path):
    root = _root(tmp_path)
    (root / v9.FORWARD_LOCK).unlink()
    with pytest.raises(DataContractError, match="vooruit-slot"):
        v9.freeze(root, git_sha="abc1234")
    manifest = json.loads((root / ex.MANIFEST).read_text(encoding="utf-8"))
    del manifest["files"][str(ex.DTB3_CSV)]
    (root / ex.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataContractError, match="Geen T-bill"):
        v9.freeze(root, git_sha="abc1234")


def test_run_refuses_without_a_frozen_preregistration(tmp_path: Path):
    with pytest.raises(DataContractError, match="pre-registratie"):
        v9.run_programme(_root(tmp_path), log=lambda s: None)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_main_dispatches_and_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    root = _root(tmp_path)
    monkeypatch.setattr(v9, "ROOT", root)
    monkeypatch.setattr(v9, "current_git_sha", lambda: "abc1234")
    v9.main(["freeze"])
    assert "preregistration_" in capsys.readouterr().out
    monkeypatch.setattr(v9, "run_programme", lambda r: capsys.readouterr())
    v9.main(["run"])
    with pytest.raises(DataContractError, match="Gebruik"):
        v9.main([])
    monkeypatch.setattr(sys, "argv", ["programme_v9"])
    with pytest.raises(DataContractError, match="Gebruik"):
        runpy.run_module("tradebot.systematic.programme_v9", run_name="__main__")


def test_the_committed_v9_run_follows_from_the_committed_inputs(cfg):
    """De dagreeksen en jaarcijfers in `artefacts/research/robust_book_v9` zijn wat `build`
    geeft op de bevroren H3-reeks en de gecommitte T-bill: niets met de hand."""
    art = ROOT / v9.ARTEFACT_DIR
    out = json.loads((art / f"{v9.CANDIDATE}.json").read_text(encoding="utf-8"))
    base = vn.cycle(ROOT, cfg.base.programme, cfg.base.candidate)
    u1 = v9.build(base, ex.load_dtb3(ROOT), cfg, v9.pair_fee_bps(ROOT, cfg))
    w = cfg.windows
    for label, (a, b) in {"backcast": (w.backcast_start, w.backcast_end),
                          "w_dev": (w.train_start, w.w_dev_end),
                          "holdout": (w.holdout_start, w.holdout_end)}.items():
        saved = pd.read_csv(art / f"{v9.CANDIDATE}_daily_{label}.csv", index_col=0,
                            parse_dates=True)
        fresh = v9._slice(u1, a, b)
        assert saved.index.equals(fresh.index) and list(saved.columns) == list(fresh.columns)
        pd.testing.assert_frame_equal(saved, fresh, check_dtype=False, rtol=1e-9, atol=1e-15)
    fresh_years = v9.years(u1)
    for y, rec in out["full_cycle"]["years"].items():
        assert rec == pytest.approx(fresh_years[y], rel=1e-12)
    assert out["preregistration_id"] == v9._prereg(ROOT, cfg)[0].preregistration_id
    assert out["full_cycle"]["years_negative"] == [] and out["verdict"] == \
        "falsified_out_of_sample"


# --------------------------------------------------------------------------- #
# De vooruit-lezing (op een kopie van de governance: het slot telt de lezing)
# --------------------------------------------------------------------------- #
def _forward_root(tmp_path: Path) -> Path:
    for rel in (v9.CONFIG, v9.SPEC_PATH, vn.VENUES_PATH, v9.FORWARD_LOCK, v2.LEDGER_PATH,
                Path("conf/model/robust_book_v5.yaml"), Path("conf/execution/impact.yaml"),
                v9.ARTEFACT_DIR / f"{v9.CANDIDATE}.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    prereg = json.loads((ROOT / v9.ARTEFACT_DIR / f"{v9.CANDIDATE}.json").read_text(
        encoding="utf-8"))["preregistration_id"]
    name = f"preregistration_{prereg}.json"
    shutil.copy(ROOT / v2.PREREG_DIR / name, tmp_path / v2.PREREG_DIR / name)
    return tmp_path


def _reads(root: Path) -> list[str]:
    lock = json.loads((root / v9.FORWARD_LOCK).read_text(encoding="utf-8"))
    return [r["hypothesis_id"] for r in lock["reads"]]


def test_the_forward_read_refuses_before_six_months_and_registers_nothing(tmp_path: Path):
    from tests.unit.test_carry_u1 import kraken_get
    root = _forward_root(tmp_path)
    before = _reads(root)
    now = pd.Timestamp("2026-10-27 06:00", tz="UTC")
    with pytest.raises(DataContractError, match="zes volle maanden"):
        v9.read_forward(root, get=kraken_get(now), now=now, log=lambda s: None)
    assert _reads(root) == before and not (root / v9.FORWARD_DIR).exists()


def test_the_forward_read_runs_once_on_kraken_data(tmp_path: Path):
    from tests.unit.test_carry_u1 import kraken_get
    root = _forward_root(tmp_path)
    now = pd.Timestamp("2027-04-03 06:00", tz="UTC")
    logs: list[str] = []
    out = v9.read_forward(root, get=kraken_get(now), now=now, log=logs.append)
    assert out["hypothesis_id"] in _reads(root)
    assert out["window"][0].startswith("2026-10-01")
    assert set(out["gate_metrics"]) == {"forward_sharpe_z_vs_dev", "forward_max_drawdown",
                                        "forward_n_liquidations", "forward_return"}
    assert out["verdict"] in {"forward_clear", "falsified_forward", "archived_forward"}
    assert out["prior_verdict"] == "falsified_out_of_sample"
    assert (root / v9.FORWARD_DIR / "manifest.json").is_file()
    assert (root / v9.FORWARD_DIR / "dtb3.csv").is_file()
    entries = json.loads((root / v2.LEDGER_PATH).read_text(encoding="utf-8"))["entries"]
    assert entries[-1]["unit"] == "robust_book_v9_forward"
    assert logs and logs[-1].endswith(out["verdict"])
    with pytest.raises(HoldoutAlreadyUsed):
        v9.read_forward(root, get=kraken_get(now), now=now, log=lambda s: None)


def test_main_dispatches_the_forward_read(monkeypatch: pytest.MonkeyPatch):
    seen = []
    monkeypatch.setattr(v9, "read_forward", seen.append)
    v9.main(["forward"])
    assert seen == [v9.ROOT]
