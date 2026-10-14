"""De v2-orkestratie van begin tot eind: bevriezen, meten op W_DEV, één holdout-lezing, oordeel.

Alles draait op een tmp-wortel met een klein synthetisch Binance-paneel (zeven munten, 225
dagen), een eigen config met korte vensters en lookbacks, en een ledger met alleen een
seed. De echte `data/` en `artefacts/` worden niet gelezen of beschreven; alleen
`conf/execution/impact.yaml` en de echte preregistratiespec worden naar `tmp_path` gekopieerd.

De batterij zelf en de PBO zijn getoetst in `test_programme_v2_smoke.py`. Hier draait de
echte batterij één keer (met één ruisreeks en een PBO op vier deelsets, zodat de run in
seconden klaar is); de oordeel-scenario's gebruiken een lichte batterij met dezelfde
handtekening, zodat de gate-drempels uit de spec het oordeel bepalen en niet het toeval.
"""
from __future__ import annotations

import functools
import inspect
import json
import runpy
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tradebot.backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from tradebot.registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from tradebot.registry.preregistration import load_preregistration_spec
from tradebot.schemas.robust_book_v2 import robust_book_v2_config
from tradebot.systematic import programme_v2 as v2
from tradebot.systematic.evaluate import pbo_record
from tradebot.utils.failfast import DataContractError
from tradebot.validation.holdout import HoldoutAlreadyUsed, freeze_holdout

REPO = Path(__file__).resolve().parents[2]
SHA = "abc1234"
LEDGER_SEED = 10
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT")
N_BARS = 225                      # 2021-01-01 .. 2021-08-13
PANELS = ("close", "funding", "high", "low", "quote_volume")
HOLDOUT_START, HOLDOUT_END = "2021-07-01", "2021-08-10"
HOLDOUT_GATES = ("gate_sharpe_collapse", "gate_drawdown")
PROMOTION = "promotion_requires_all_clear"


# --------------------------------------------------------------------------- #
# De wortel: config, spec, ledger, manifest, impactparameters en synthetische panelen
# --------------------------------------------------------------------------- #
def make_panels(symbols: tuple[str, ...] = SYMBOLS, n_bars: int = N_BARS,
                seed: int = 3) -> dict[str, pd.DataFrame]:
    """Close/high/low/quote_volume/funding zoals `data/binance_vision.py::build_panels` ze levert."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="D", tz="UTC", name="asof_ts")
    common = rng.standard_t(4, size=n_bars) * 0.02
    drift = np.zeros((n_bars, len(symbols)))                    # hardnekkige trends per munt
    for t in range(1, n_bars):
        drift[t] = 0.97 * drift[t - 1] + 0.003 * rng.normal(size=len(symbols))
    rets = (0.7 * common[:, None] + rng.standard_t(4, size=(n_bars, len(symbols))) * 0.03
            + drift).clip(-0.5, 0.5)
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=list(symbols))
    if "DOGEUSDT" in close:
        close.loc[close.index[:30], "DOGEUSDT"] = np.nan        # late notering
    if "LTCUSDT" in close:
        close.loc[close.index[150:], "LTCUSDT"] = np.nan        # delisting binnen W_DEV
    spread = np.abs(rng.normal(0.0, 0.01, size=close.shape)) * close
    quote_volume = pd.DataFrame(np.exp(rng.normal(17.0, 0.6, size=close.shape)), index=idx,
                                columns=close.columns)
    funding = pd.DataFrame(rng.normal(2e-4, 3e-4, size=close.shape), index=idx,
                           columns=close.columns)
    listed = close.notna()
    return {"close": close, "high": (close + spread).where(listed),
            "low": (close - spread).where(listed), "quote_volume": quote_volume.where(listed),
            "funding": funding.where(listed)}


def small_config(version: int) -> dict[str, Any]:
    """De echte config van die versie, met een klein universum, korte vensters en lookbacks."""
    raw = yaml.safe_load((REPO / f"conf/model/robust_book_v{version}.yaml").read_text("utf-8"))
    raw["universe"] = {"top_n": 6, "min_history_bars": 20, "adv_window": 10,
                       "min_adv_usd": 1.0e6, "vol_window": 20, "min_ann_vol": 0.15}
    raw["windows"] = {"train_start": "2021-01-01", "train_end": "2021-04-30",
                      "validate_start": "2021-05-01", "w_dev_end": "2021-06-30",
                      "holdout_start": HOLDOUT_START, "holdout_end": HOLDOUT_END}
    raw["costs"]["spread_window"] = 20
    raw["sizing"]["vol_span"] = 20
    raw["xsmom"].update(lookbacks=[5, 10], rebalance_bars=5)
    raw["xscarry"].update(window_bars=5, rebalance_bars=5)
    raw["trend"]["lookbacks"] = [5, 10, 20]
    return raw


def gate_spec(*, w_dev_binds: bool = False, holdout_gate: str | None = None) -> Callable:
    """Vervang de poorten door vijf waarvan de drempels het oordeel vastleggen.

    `g1` bindt op W_DEV als `w_dev_binds`; de holdoutpoort `holdout_gate` bindt altijd; de
    rest bindt nooit. Zo bepaalt de spec het oordeel en niet het toeval van de synthetische data.
    """
    def crit(name, metric, operator, threshold):
        return {"name": name, "metric": metric, "operator": operator, "threshold": threshold,
                "action": "falsify" if name in HOLDOUT_GATES else "archive",
                "rationale": "Drempel van de toets, niet van het onderzoek."}

    def edit(spec: dict[str, Any]) -> None:
        spec["stop_criteria"] = [
            crit("g1_sharpe_below_target", "net_sharpe_w_dev", "<", 1e9 if w_dev_binds else -1e9),
            crit("g4_drawdown", "max_drawdown_w_dev", ">", 10.0),
            crit("gate_sharpe_collapse", "gate_sharpe_z_vs_dev", "<",
                 1e9 if holdout_gate == "gate_sharpe_collapse" else -1e9),
            crit("gate_drawdown", "max_drawdown_gate", ">",
                 -1.0 if holdout_gate == "gate_drawdown" else 10.0),
            {**crit(PROMOTION, "n_binding_stop_criteria", "<=", 0), "action": "promote"},
        ]
    return edit


def build_root(root: Path, *, version: int = 2, panels: dict[str, pd.DataFrame] | None = None,
               cfg_edit: Callable | None = None, spec_edit: Callable | None = None,
               ledger_seed: int = LEDGER_SEED) -> Path:
    """Een complete wortel: wat `freeze` en `run_programme` relatief aan `root` lezen."""
    for rel in ("conf/model", "conf/research", "conf/execution", "artefacts/data",
                "artefacts/governance", "data/binance_vision/panels"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    cfg = small_config(version)
    if cfg_edit:
        cfg_edit(cfg)
    (root / f"conf/model/robust_book_v{version}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    specs = ["preregistration_robust_book_v2.yaml"] if version == 2 else [
        "preregistration_robust_book_v3.yaml", "preregistration_robust_book_v4_confirm.yaml"]
    for name in specs:
        spec = yaml.safe_load((REPO / "conf/research" / name).read_text(encoding="utf-8"))
        if spec_edit and name != "preregistration_robust_book_v4_confirm.yaml":
            spec_edit(spec)
        (root / "conf/research" / name).write_text(
            yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copy(REPO / "conf/execution/impact.yaml", root / "conf/execution/impact.yaml")
    (root / "artefacts/data/binance_um_manifest.json").write_text(
        json.dumps({"objects_sha256": "ab" * 32, "n_symbols": len(SYMBOLS)}), encoding="utf-8")
    (root / "artefacts/governance/hypothesis_ledger.json").write_text(json.dumps(
        {"version": 1, "seed_total": ledger_seed, "seed_note": "testledger", "entries": []}),
        encoding="utf-8")
    for name, frame in (panels or make_panels()).items():
        frame.to_parquet(root / f"data/binance_vision/panels/{name}.parquet")
    return root


def copy_root(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ledger_entries(root: Path) -> list[dict[str, Any]]:
    return HypothesisLedger(root / v2.LEDGER_PATH).entries()


def lock_reads(root: Path) -> list[str]:
    return [e["hypothesis_id"] for e in read_json(root / v2.LOCK_PATH)["reads"]]


def light_battery(real: Callable, calls: list[tuple[str, dict[str, Any]]]) -> Callable:
    """Een battery met de handtekening van de echte, die alleen de uitkomsten levert die de
    orkestratie leest. De aanroep wordt tegen de echte handtekening gebonden en bewaard."""
    signature = inspect.signature(real)

    def light(name, m, cfg, costs, **kw):
        signature.bind(name, m, cfg, costs, **kw)
        calls.append((name, kw))
        a, b = kw["w_dev"]
        net = kw["base"].window(a, b).net
        sharpe = v2._sr(kw["base"].window(a, b))
        bat = {"base_sharpe": sharpe, "plateau_fraction": 1.0, "min_perturbation_sharpe": sharpe,
               "universe_positive_fraction": 1.0, "costs": {"x2": 0.9 * sharpe},
               "delay": {"lag_2": 0.9 * sharpe}}
        return bat, {"base": net, "damped": 0.9 * net, "lagged": net.shift(1).fillna(0.0)}
    return light


def fake_pbo(variants, *, n_subsets: int = 4):
    return {"pbo": 0.25, "logit_pbo": -1.0, "n_combinations": 6, "n_variants": len(variants),
            "n_subsets": n_subsets}


def freeze_and_run(root: Path, mp: pytest.MonkeyPatch, *, light: bool,
                   calls: list | None = None) -> SimpleNamespace:
    """freeze + run_programme, met de echte (verkleinde) of de lichte batterij."""
    mp.setattr(v2, "current_git_sha", lambda: SHA)
    if light:
        mp.setattr(v2, "battery", light_battery(v2.battery, [] if calls is None else calls))
        mp.setattr(v2, "pbo_record", fake_pbo)
    else:
        mp.setattr(v2, "pbo_record", functools.partial(pbo_record, n_subsets=4))
        mp.setattr(v2, "N_SEEDS", 1)
    logs: list[str] = []
    frozen = v2.freeze(root, git_sha=SHA)
    summary = v2.run_programme(root, log=logs.append)
    return SimpleNamespace(root=root, frozen=frozen, summary=summary, logs=logs)


@pytest.fixture(autouse=True)
def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Geen enkele test roept `git` aan; de sha is een vaste waarde."""
    monkeypatch.setattr(v2, "current_git_sha", lambda: SHA)


@pytest.fixture
def frozen_root(tmp_path: Path) -> Path:
    """Een wortel waarvan het programma is geboekt en bevroren, nog niet gemeten."""
    root = build_root(tmp_path / "root")
    v2.freeze(root, git_sha=SHA)
    return root


@pytest.fixture(scope="module")
def real_flow(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Eén volledige run met de ECHTE batterij en de echte spec (alleen verkleind)."""
    root = build_root(tmp_path_factory.mktemp("v2_real") / "root")
    with pytest.MonkeyPatch.context() as mp:
        return freeze_and_run(root, mp, light=False)


SCENARIOS = {
    "promote": {"w_dev_binds": False, "holdout_gate": None},
    "falsified": {"w_dev_binds": False, "holdout_gate": "gate_sharpe_collapse"},
    # Dezelfde run laat ook X5 buiten beeld: geen enkele sleeve haalt de inclusiedrempel.
    "archived": {"w_dev_binds": True, "holdout_gate": None},
}


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory: pytest.TempPathFactory) -> dict[str, SimpleNamespace]:
    """Per oordeel een eigen bevroren spec, een W_DEV-run en één holdout-lezing."""
    out = {}
    for name, knobs in SCENARIOS.items():
        def no_x5(cfg: dict[str, Any], name: str = name) -> None:
            if name == "archived":
                cfg["combo"]["inclusion_min_train_sharpe"] = 1e9
        root = build_root(tmp_path_factory.mktemp(f"v2_{name}") / "root", cfg_edit=no_x5,
                          spec_edit=gate_spec(**knobs))
        calls: list = []
        with pytest.MonkeyPatch.context() as mp:
            flow = freeze_and_run(root, mp, light=True, calls=calls)
            flow.holdout = v2.read_holdout(root, log=flow.logs.append)
        flow.calls = calls
        out[name] = flow
    return out


# --------------------------------------------------------------------------- #
# Bevriezen
# --------------------------------------------------------------------------- #
def test_freeze_books_the_programme_and_freezes_the_preregistration_and_the_lock(tmp_path):
    root = build_root(tmp_path / "root")
    path = v2.freeze(root, git_sha=SHA)

    (entry,) = ledger_entries(root)
    frozen = read_json(path)
    cfg = robust_book_v2_config(root / v2.CONFIG)
    assert path == root / v2.PREREG_DIR / f"preregistration_{entry['preregistration_id']}.json"
    assert (entry["wave"], entry["unit"], entry["market"], entry["n_trials"], entry["result"],
            entry["git_sha"]) == (2, v2.PROGRAMME_UNIT, "crypto", 7, "interim", SHA)
    assert frozen["ledger_total_at_freeze"] == LEDGER_SEED + 7
    assert frozen["git_sha"] == SHA and frozen["content"]["planned_trials"] == 7
    # De parameters komen uit de config, niet uit de spec.
    assert frozen["content"]["parameters"] == json.loads(cfg.model_dump_json())
    assert [name for name, _ in frozen["content"]["data_hashes"]] == sorted(
        ["binance_um/objects", *(f"binance_um/panel/{p}" for p in PANELS)])
    lock = read_json(root / v2.LOCK_PATH)
    assert (lock["split_utc"], lock["git_sha"], lock["reads"]) == (
        f"{HOLDOUT_START}T00:00:00+00:00", SHA, [])


def test_freeze_takes_the_sha_from_the_working_tree_when_none_is_given(
        tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "current_git_sha", lambda: "fedcba9")
    root = build_root(tmp_path / "root")
    v2.freeze(root)
    assert ledger_entries(root)[0]["git_sha"] == "fedcba9"
    assert read_json(root / v2.LOCK_PATH)["git_sha"] == "fedcba9"


def test_freezing_the_same_programme_twice_is_refused_and_books_nothing_new(frozen_root):
    with pytest.raises(DataContractError, match="Dit programma is al geboekt"):
        v2.freeze(frozen_root, git_sha=SHA)
    assert len(ledger_entries(frozen_root)) == 1


def test_freeze_refuses_when_the_preregistration_plans_another_number_of_trials(tmp_path):
    root = build_root(tmp_path / "root", spec_edit=lambda spec: spec.update(planned_trials=8))
    with pytest.raises(DataContractError, match="planned_trials wijkt af"):
        v2.freeze(root, git_sha=SHA)
    assert ledger_entries(root) == []
    assert not (root / v2.LOCK_PATH).exists()
    assert list((root / v2.PREREG_DIR).glob("preregistration_*.json")) == []


def test_freeze_does_not_create_a_missing_ledger(tmp_path):
    root = build_root(tmp_path / "root")
    (root / v2.LEDGER_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="Ledger not found"):
        v2.freeze(root, git_sha=SHA)
    assert not (root / v2.LEDGER_PATH).exists() and not (root / v2.LOCK_PATH).exists()


def test_freeze_keeps_a_holdout_lock_that_already_exists(tmp_path):
    root = build_root(tmp_path / "root")
    lock = root / v2.LOCK_PATH
    freeze_holdout(split_utc=f"{HOLDOUT_START}T00:00:00+00:00", out=lock, git_sha="oldsha1")
    before = lock.read_bytes()
    v2.freeze(root, git_sha=SHA)
    assert lock.read_bytes() == before and read_json(lock)["git_sha"] == "oldsha1"


# --------------------------------------------------------------------------- #
# De bouwstenen: data-hashes, kosten, gates, dump, alternatieve universa
# --------------------------------------------------------------------------- #
def test_the_data_hashes_cover_the_manifest_and_every_panel_and_follow_the_bytes(tmp_path):
    root = build_root(tmp_path / "root")
    hashes = dict(v2._data_hashes(root))
    assert hashes["binance_um/objects"] == "ab" * 8             # eerste 16 tekens van de manifest-hash
    assert list(hashes) == sorted(hashes) and len(hashes) == 1 + len(PANELS)
    assert all(len(h) == 16 for h in hashes.values())
    funding = root / "data/binance_vision/panels/funding.parquet"
    funding.write_bytes(funding.read_bytes() + b"x")
    changed = dict(v2._data_hashes(root))
    assert changed["binance_um/panel/funding"] != hashes["binance_um/panel/funding"]
    assert {k: v for k, v in changed.items() if k != "binance_um/panel/funding"} == {
        k: v for k, v in hashes.items() if k != "binance_um/panel/funding"}


def test_timestamps_are_utc():
    assert v2._ts("2021-07-01") == pd.Timestamp("2021-07-01T00:00:00+00:00")
    assert str(v2._ts("2021-07-01").tz) == "UTC"


def test_the_cost_spec_follows_the_config_and_the_repo_impact_parameters(tmp_path):
    root = build_root(tmp_path / "root")
    cfg = robust_book_v2_config(root / v2.CONFIG)
    costs = v2.cost_spec(root, cfg)
    impact = yaml.safe_load((REPO / "conf/execution/impact.yaml").read_text("utf-8"))["impact"]
    assert costs.taker_fee == pytest.approx(5.5e-4)
    assert costs.half_spread == pytest.approx(1.0e-4)          # min_half_spread_bps: de vloer
    assert costs.aum_usd == cfg.execution.aum_usd
    assert costs.impact is not None and costs.impact.eta == pytest.approx(impact["eta"])
    assert costs.eta == cfg.costs.impact_y                     # literatuur-Y overschrijft de repo-eta


def test_gates_bind_on_the_preregistered_operator_and_on_a_non_finite_value():
    prereg = load_preregistration_spec(REPO / v2.SPEC_PATH, data_hashes=(("x", "0"),),
                                       parameters={"p": 1})
    gates = v2._gates({"net_sharpe_w_dev": 0.99, "max_drawdown_w_dev": 0.25,
                       "dsr_w_dev": float("nan"), "pbo": 0.5, "plateau_fraction": 0.9,
                       "gate_sharpe_z_vs_dev": -2.0, "n_binding_stop_criteria": 0.0,
                       "not_a_gate": 1.0}, prereg)
    assert set(gates) == {"g1_sharpe_below_target", "g4_drawdown", "g8_dsr_below_threshold",
                          "g9_overfit_probability", "g6_no_plateau", "gate_sharpe_collapse"}
    assert gates["g1_sharpe_below_target"] == {
        "metric": "net_sharpe_w_dev", "value": 0.99, "operator": "<", "threshold": 1.0,
        "binds": True}
    assert gates["g4_drawdown"]["binds"] is False              # 0,25 > 0,25 is onwaar
    assert gates["g9_overfit_probability"]["binds"] is True    # 0,5 >= 0,5
    assert gates["g6_no_plateau"]["binds"] is False            # 0,9 < 0,8 is onwaar
    assert gates["g8_dsr_below_threshold"]["binds"] is True    # NaN telt als bindend
    assert gates["gate_sharpe_collapse"]["binds"] is True      # -2,0 < -1,645


def test_dump_creates_the_folder_and_writes_sorted_utf8_json_with_numpy_numbers(tmp_path):
    target = tmp_path / "a" / "b" / "out.json"
    v2._dump(target, {"z": np.float64(1.5), "a": "é", "m": {"k": np.float32(0.25)}})
    assert target.read_text(encoding="utf-8") == json.dumps(
        {"a": "é", "m": {"k": 0.25}, "z": 1.5}, indent=2, ensure_ascii=False)


def test_the_alternative_universes_are_the_same_panels_with_another_top_n_cut_off_at_the_end(
        tmp_path, monkeypatch):
    root = build_root(tmp_path / "root")
    cfg = robust_book_v2_config(root / v2.CONFIG)
    seen = []
    real = v2.load_breadth_market

    def spy(panel_dir, alt):
        seen.append((panel_dir, alt.universe.top_n))
        return real(panel_dir, alt)
    monkeypatch.setattr(v2, "load_breadth_market", spy)
    end = pd.Timestamp("2021-06-01", tz="UTC")
    alt = v2._alt_universes(root, cfg, end)
    assert list(alt) == ["top_30", "top_100"]
    assert seen == [(root / v2.PANEL_DIR, 30), (root / v2.PANEL_DIR, 100)]
    assert all(m.book.index.max() < end for m in alt.values())
    assert cfg.universe.top_n == 6                              # de basisconfig blijft ongemoeid


# --------------------------------------------------------------------------- #
# run_programme: weigeringen
# --------------------------------------------------------------------------- #
def test_run_refuses_without_a_frozen_preregistration_and_writes_nothing(tmp_path):
    root = build_root(tmp_path / "root")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v2.run_programme(root, log=lambda _: None)
    assert not (root / v2.ARTEFACT_DIR).exists()


def test_run_refuses_when_the_panels_changed_after_freezing(frozen_root):
    panels = make_panels()
    panels["close"] = panels["close"] * 1.0000001
    panels["close"].to_parquet(frozen_root / "data/binance_vision/panels/close.parquet")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v2.run_programme(frozen_root, log=lambda _: None)
    assert not (frozen_root / v2.ARTEFACT_DIR).exists()


def test_run_refuses_when_the_config_changed_after_freezing(frozen_root):
    path = frozen_root / v2.CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["known_prior_trials"] += 1
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v2.run_programme(frozen_root, log=lambda _: None)
    assert not (frozen_root / v2.ARTEFACT_DIR).exists()


def test_run_refuses_a_universe_with_too_few_coins_and_writes_nothing(tmp_path):
    root = build_root(tmp_path / "root", panels=make_panels(SYMBOLS[:3]))
    v2.freeze(root, git_sha=SHA)
    with pytest.raises(DataContractError, match="Te weinig munten ooit in het universum"):
        v2.run_programme(root, log=lambda _: None)
    assert not (root / v2.ARTEFACT_DIR).exists()


# --------------------------------------------------------------------------- #
# run_programme: de echte run
# --------------------------------------------------------------------------- #
def _real_spec() -> dict[str, Any]:
    return yaml.safe_load((REPO / v2.SPEC_PATH).read_text(encoding="utf-8"))


def test_run_writes_a_record_and_a_daily_frame_per_trial_and_one_summary(real_flow):
    out = real_flow.root / v2.ARTEFACT_DIR
    assert {p.name for p in out.iterdir()} == (
        {f"{n}.json" for n in v2.TRIALS} | {f"{n}_daily_w_dev.csv" for n in v2.TRIALS}
        | {"programme_w_dev.json"})
    frame = pd.read_csv(out / "X1_XSMOM_daily_w_dev.csv", index_col=0, parse_dates=True)
    assert str(frame.index[0].date()) == "2021-01-01" and str(frame.index[-1].date()) == "2021-06-30"
    assert {"gross", "funding", "fees", "spread", "slippage", "impact", "net"} <= set(frame)


def test_every_record_carries_its_provenance_and_stays_unadmissible_as_evidence(real_flow):
    cfg = robust_book_v2_config(real_flow.root / v2.CONFIG)
    prereg_id = real_flow.summary["preregistration_id"]
    for name in v2.TRIALS:
        rec = read_json(real_flow.root / v2.ARTEFACT_DIR / f"{name}.json")
        assert rec["trial"] == name and rec["hypothesis"] == v2.HYPOTHESES[name]
        assert rec["role"] == ("reference" if name in v2.REFERENCES else "candidate")
        assert (rec["preregistration_id"], rec["git_sha"]) == (prereg_id, SHA)
        assert rec[EVIDENCE_KEY] == NOT_ADMISSIBLE
        assert rec["parameters"] == json.loads(cfg.model_dump_json())
        assert rec["windows"] == {"train": ["2021-01-01", "2021-04-30"],
                                  "validate": ["2021-05-01", "2021-06-30"],
                                  "holdout": [HOLDOUT_START, "unread"]}
        assert set(rec["data_hashes"]) == {"binance_um/objects",
                                           *(f"binance_um/panel/{p}" for p in PANELS)}
        assert rec["execution"]["lag_bars"] == 1
        assert "Abdi-Ranaldo" in rec["execution"]["half_spread"]


def test_the_delisted_coin_is_closed_by_the_book_and_the_exit_is_recorded(real_flow):
    rec = read_json(real_flow.root / v2.ARTEFACT_DIR / "R2_EW50_HOLD.json")
    assert rec["n_forced_exits"] >= 1                          # LTCUSDT verdwijnt op bar 150


def test_the_holdout_stays_unread_by_the_w_dev_run(real_flow):
    assert read_json(real_flow.root / v2.LOCK_PATH)["reads"] == []
    assert not (real_flow.root / v2.ARTEFACT_DIR / "holdout_read.json").exists()
    for name in v2.TRIALS:
        csv = pd.read_csv(real_flow.root / v2.ARTEFACT_DIR / f"{name}_daily_w_dev.csv",
                          index_col=0, parse_dates=True)
        assert csv.index.max() < pd.Timestamp(HOLDOUT_START, tz="UTC")


def test_the_trial_counts_come_from_the_ledger_and_the_known_prior(real_flow):
    s = real_flow.summary
    assert s["m_programme"] == LEDGER_SEED + 7                 # seed plus de zeven geboekte trials
    assert s["m_including_known_prior"] == LEDGER_SEED + 7 + 52
    dsr = read_json(real_flow.root / v2.ARTEFACT_DIR / "X1_XSMOM.json")["dsr"]
    assert set(dsr) == {f"m_{LEDGER_SEED + 7}", f"m_{LEDGER_SEED + 7 + 52}"}
    assert dsr[f"m_{LEDGER_SEED + 7 + 52}"]["dsr"] <= dsr[f"m_{LEDGER_SEED + 7}"]["dsr"]


def test_x5_takes_exactly_the_sleeves_whose_train_sharpe_clears_the_threshold(real_flow):
    x5 = read_json(real_flow.root / v2.ARTEFACT_DIR / "X5_COMBO.json")
    cfg = robust_book_v2_config(real_flow.root / v2.CONFIG)
    sharpes = x5["train_sharpe_of_sleeves"]
    assert set(sharpes) == set(v2.SLEEVES)
    expected = [s for s in cfg.combo.candidates if sharpes[s] > cfg.combo.inclusion_min_train_sharpe]
    # De synthetische markt geeft sleeves met een positieve én een negatieve TRAIN-Sharpe, zodat de
    # regel naar beide kanten wordt toegepast.
    assert 0 < len(expected) < len(cfg.combo.candidates)
    assert x5["included"] == expected == real_flow.summary["x5_included"]
    assert f"X5 neemt op: {tuple(expected)}" in real_flow.logs


def test_the_batteries_the_gates_and_the_score_are_attached_to_every_candidate_only(real_flow):
    spec_gates = {c["name"] for c in _real_spec()["stop_criteria"]} - set(HOLDOUT_GATES) - {PROMOTION}
    for name in v2.TRIALS:
        rec = read_json(real_flow.root / v2.ARTEFACT_DIR / f"{name}.json")
        if name in v2.REFERENCES:
            assert not {"battery", "gates", "pbo", "robustness_score"} & set(rec)
            continue
        assert {"perturbation", "costs", "capacity", "delay", "signal_noise", "missing_data",
                "rebalance", "universe", "start_dates", "end_dates", "monte_carlo_drawdown",
                "plateau_fraction", "min_perturbation_sharpe"} <= set(rec["battery"])
        assert rec["battery"]["signal_noise"]["n_seeds"] == 1
        assert set(rec["gates"]) == spec_gates                  # elke poort heeft een gemeten metriek
        assert rec["n_binding_w_dev"] == sum(g["binds"] for g in rec["gates"].values())
        assert 0.0 <= rec["pbo"]["pbo"] <= 1.0 and rec["pbo"]["n_subsets"] == 4
        assert 0.0 <= rec["robustness_score"]["total"] <= 100.0
        m = rec["gate_metrics"]
        assert m["net_sharpe_w_dev"] == rec["w_dev"]["sharpe"]
        assert m["dsr_w_dev"] == rec["dsr"][f"m_{LEDGER_SEED + 7}"]["dsr"]


def test_the_candidate_with_the_highest_robustness_score_is_selected(real_flow):
    s = real_flow.summary
    scores = {n: read_json(real_flow.root / v2.ARTEFACT_DIR / f"{n}.json")["robustness_score"]["total"]
              for n in v2.CANDIDATES}
    assert s["robustness_scores"] == scores
    assert s["selected"] == max(scores, key=scores.get)
    assert s["passes_all_w_dev_gates"] == {
        n: read_json(real_flow.root / v2.ARTEFACT_DIR / f"{n}.json")["n_binding_w_dev"] == 0
        for n in v2.CANDIDATES}
    assert set(s["table"]) == set(v2.TRIALS)
    assert s[EVIDENCE_KEY] == NOT_ADMISSIBLE and s["programme"] == "robust_book_v2"
    assert s == read_json(real_flow.root / v2.ARTEFACT_DIR / "programme_w_dev.json")
    assert [x for x in real_flow.logs if x.startswith("batterij ")] == [
        f"batterij {n}" for n in v2.CANDIDATES]


def test_the_shipped_spec_plans_as_many_trials_as_the_shipped_config():
    cfg = robust_book_v2_config(REPO / v2.CONFIG)
    spec = load_preregistration_spec(REPO / v2.SPEC_PATH, data_hashes=(("x", "0"),),
                                     parameters={"p": 1})
    assert spec.planned_trials == cfg.planned_trials == len(v2.TRIALS)


# --------------------------------------------------------------------------- #
# De scenario's: de gate-drempels bepalen het oordeel
# --------------------------------------------------------------------------- #
def test_the_light_battery_is_called_per_candidate_with_the_recorded_included_sleeves(scenarios):
    calls = scenarios["promote"].calls
    assert [name for name, _ in calls] == list(v2.CANDIDATES) and len(calls) == 5
    included = tuple(scenarios["promote"].summary["x5_included"])
    for _, kw in calls:
        assert kw["included"] == included
        assert set(kw["alt_universes"]) == {"top_30", "top_100"}
        assert kw["w_dev"] == (pd.Timestamp("2021-01-01", tz="UTC"),
                               pd.Timestamp("2021-06-30", tz="UTC"))


@pytest.mark.parametrize(("name", "verdict", "n_binding"), [
    ("promote", "promote_to_paper_trading", 0),
    ("falsified", "falsified_on_holdout", 1),
    ("archived", "archived_w_dev_gates", 1)])
def test_the_verdict_follows_the_binding_gates_of_the_frozen_preregistration(
        scenarios, name, verdict, n_binding):
    out = scenarios[name].holdout
    assert (out["verdict"], out["n_binding_total"]) == (verdict, n_binding)
    assert set(out["gates"]) == set(HOLDOUT_GATES)
    assert sum(g["binds"] for g in out["gates"].values()) == (1 if name == "falsified" else 0)


def test_without_a_sleeve_above_the_threshold_x5_is_not_constructible_and_never_selected(scenarios):
    flow = scenarios["archived"]
    out = flow.root / v2.ARTEFACT_DIR
    rec = read_json(out / "X5_COMBO.json")
    assert rec["status"] == "not_constructible" and rec["included"] == []
    assert not {"w_dev", "battery", "gates", "robustness_score"} & set(rec)
    assert not (out / "X5_COMBO_daily_w_dev.csv").exists()
    assert flow.summary["x5_included"] == [] and "X5_COMBO" not in flow.summary["robustness_scores"]
    assert flow.summary["selected"] != "X5_COMBO"
    assert flow.summary["passes_all_w_dev_gates"]["X5_COMBO"] is False
    assert "X5_COMBO" not in flow.summary["table"]
    assert [name for name, _ in flow.calls] == list(v2.CANDIDATES[:-1])
    assert flow.holdout["included"] == []


# --------------------------------------------------------------------------- #
# read_holdout
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_holdout(real_flow, tmp_path_factory) -> SimpleNamespace:
    """De ene holdout-lezing van de echte run, op een kopie van de wortel."""
    root = copy_root(real_flow.root, tmp_path_factory.mktemp("v2_holdout") / "root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(v2, "current_git_sha", lambda: SHA)
        logs: list[str] = []
        out = v2.read_holdout(root, log=logs.append)
    return SimpleNamespace(root=root, out=out, logs=logs)


def _hid(flow: SimpleNamespace) -> str:
    return f"robust_book_v2/{flow.summary['preregistration_id']}/{flow.summary['selected']}"


def test_the_holdout_read_is_registered_before_the_first_holdout_number_exists(
        real_flow, tmp_path, monkeypatch):
    root = copy_root(real_flow.root, tmp_path / "root")
    registered_when_running = []
    real_run = v2.run

    def spy(*args, **kwargs):
        registered_when_running.append(lock_reads(root))
        return real_run(*args, **kwargs)
    monkeypatch.setattr(v2, "run", spy)
    out = v2.read_holdout(root, log=lambda _: None)
    assert out["hypothesis_id"] == _hid(real_flow)
    assert registered_when_running[0] == [_hid(real_flow)]     # geregistreerd vóór de eerste berekening
    assert len(registered_when_running) == 3                    # kandidaat plus twee referenties
    assert lock_reads(root) == [_hid(real_flow)]                # één lezing, niet drie


def test_the_holdout_read_reports_the_selected_candidate_and_both_references_only(
        real_flow, real_holdout):
    out, root = real_holdout.out, real_holdout.root
    name = real_flow.summary["selected"]
    assert out["candidate"] == name and out["hypothesis_id"] == _hid(real_flow)
    assert out["included"] == real_flow.summary["x5_included"]
    assert set(out["references_on_holdout"]) == set(v2.REFERENCES)
    assert read_json(root / v2.ARTEFACT_DIR / "holdout_read.json") == out
    assert sorted(p.name for p in (root / v2.ARTEFACT_DIR).glob("*_daily_holdout.csv")) == [
        f"{name}_daily_holdout.csv"]
    assert out[EVIDENCE_KEY] == NOT_ADMISSIBLE
    dev = read_json(root / v2.ARTEFACT_DIR / f"{name}.json")["w_dev"]["sharpe"]
    assert out["w_dev_sharpe"] == dev
    assert real_holdout.logs[-1].startswith(f"holdout {name}: Sharpe ")
    assert real_holdout.logs[-1].endswith(out["verdict"])


def test_the_holdout_window_starts_at_the_frozen_split_and_ends_at_holdout_end(real_holdout):
    out, root = real_holdout.out, real_holdout.root
    # Het paneel loopt door tot 13 augustus; de config sluit de holdout op 10 augustus.
    assert out["holdout_window"] == [HOLDOUT_START, HOLDOUT_END]
    csv = pd.read_csv(root / v2.ARTEFACT_DIR / f"{out['candidate']}_daily_holdout.csv",
                      index_col=0, parse_dates=True)
    assert [str(csv.index[0].date()), str(csv.index[-1].date())] == [HOLDOUT_START, HOLDOUT_END]
    assert out["holdout"]["n_obs"] == len(csv) == 41
    assert out["holdout"]["start"] == HOLDOUT_START and out["holdout"]["end"] == HOLDOUT_END
    assert set(out["yearly"]) == {"2021"}


def test_the_holdout_gates_are_judged_on_the_z_score_and_the_holdout_drawdown(real_holdout):
    out, root = real_holdout.out, real_holdout.root
    z, mdd = out["gate_metrics"]["gate_sharpe_z_vs_dev"], out["gate_metrics"]["max_drawdown_gate"]
    expected_z = (out["holdout"]["sharpe"] - out["w_dev_sharpe"]) / out["holdout"]["sharpe_se"]
    assert z == pytest.approx(expected_z) and mdd == out["holdout"]["max_drawdown"]
    assert set(out["gates"]) == set(HOLDOUT_GATES)
    assert out["gates"]["gate_sharpe_collapse"]["binds"] == (z < -1.645)
    assert out["gates"]["gate_drawdown"]["binds"] == (mdd > 0.25)
    n_w_dev = read_json(root / v2.ARTEFACT_DIR / f"{out['candidate']}.json")["n_binding_w_dev"]
    assert out["n_binding_total"] == sum(g["binds"] for g in out["gates"].values()) + n_w_dev
    expected = ("promote_to_paper_trading" if out["n_binding_total"] == 0 else
                "falsified_on_holdout" if any(g["binds"] for g in out["gates"].values())
                else "archived_w_dev_gates")
    assert out["verdict"] == expected


def test_a_second_read_of_the_same_hypothesis_is_refused_and_the_first_result_stays(
        real_holdout, tmp_path):
    root = copy_root(real_holdout.root, tmp_path / "root")
    before = (root / v2.ARTEFACT_DIR / "holdout_read.json").read_bytes()
    with pytest.raises(HoldoutAlreadyUsed, match="heeft het poortsample al gelezen"):
        v2.read_holdout(root, log=lambda _: None)
    assert (root / v2.ARTEFACT_DIR / "holdout_read.json").read_bytes() == before
    assert len(lock_reads(root)) == 1


def test_the_holdout_read_refuses_a_w_dev_run_of_another_preregistration_and_reads_nothing(
        real_flow, tmp_path):
    root = copy_root(real_flow.root, tmp_path / "root")
    path = root / v2.ARTEFACT_DIR / "programme_w_dev.json"
    path.write_text(json.dumps({**read_json(path), "preregistration_id": "f" * 32}), encoding="utf-8")
    with pytest.raises(DataContractError, match="hoort bij een andere preregistratie"):
        v2.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == []                               # de lezing is niet verbruikt
    assert not (root / v2.ARTEFACT_DIR / "holdout_read.json").exists()


def test_the_holdout_read_refuses_when_the_data_changed_since_the_freeze(real_flow, tmp_path):
    root = copy_root(real_flow.root, tmp_path / "root")
    funding = root / "data/binance_vision/panels/funding.parquet"
    funding.write_bytes(funding.read_bytes() + b"x")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v2.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == []


def test_the_holdout_read_without_a_frozen_lock_is_refused(real_flow, tmp_path):
    root = copy_root(real_flow.root, tmp_path / "root")
    (root / v2.LOCK_PATH).unlink()
    with pytest.raises(DataContractError, match="ontbreekt"):
        v2.read_holdout(root, log=lambda _: None)
    assert not (root / v2.ARTEFACT_DIR / "holdout_read.json").exists()


def test_the_holdout_read_stops_when_the_full_run_does_not_reproduce_the_w_dev_sharpe(
        real_flow, tmp_path):
    root = copy_root(real_flow.root, tmp_path / "root")
    rec_path = root / v2.ARTEFACT_DIR / f"{real_flow.summary['selected']}.json"
    rec = read_json(rec_path)
    rec["w_dev"]["sharpe"] += 1e-6
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(DataContractError, match="de holdout lekte"):
        v2.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == [_hid(real_flow)]                # de lezing staat er wel
    assert not (root / v2.ARTEFACT_DIR / "holdout_read.json").exists()


def test_a_holdout_of_fewer_than_thirty_bars_is_refused_but_the_read_stays_registered(tmp_path):
    root = build_root(tmp_path / "root", panels=make_panels(n_bars=200))     # t/m 19 juli: 19 bars
    v2.freeze(root, git_sha=SHA)
    prereg_id = next((root / v2.PREREG_DIR).glob("preregistration_*.json")).stem.split("_")[1]
    (root / v2.ARTEFACT_DIR).mkdir(parents=True)
    (root / v2.ARTEFACT_DIR / "programme_w_dev.json").write_text(json.dumps(
        {"preregistration_id": prereg_id, "selected": "X4_TREND_LF", "x5_included": []}),
        encoding="utf-8")
    with pytest.raises(DataContractError, match="minimaal 30 eindige observaties, kreeg 19"):
        v2.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == [f"robust_book_v2/{prereg_id}/X4_TREND_LF"]


def test_the_combination_is_rebuilt_from_the_sleeves_the_w_dev_run_recorded(real_flow, tmp_path):
    root = copy_root(real_flow.root, tmp_path / "root")
    path = root / v2.ARTEFACT_DIR / "programme_w_dev.json"
    path.write_text(json.dumps({**read_json(path), "selected": "X5_COMBO"}), encoding="utf-8")
    out = v2.read_holdout(root, log=lambda _: None)             # reproduceert de W_DEV-Sharpe van X5
    assert out["candidate"] == "X5_COMBO" and out["included"] == real_flow.summary["x5_included"]
    assert out["hypothesis_id"].endswith("/X5_COMBO")
    assert out["w_dev_sharpe"] == read_json(root / v2.ARTEFACT_DIR / "X5_COMBO.json")["w_dev"]["sharpe"]


def test_a_combination_without_recorded_sleeves_cannot_be_read(real_flow, tmp_path):
    root = copy_root(real_flow.root, tmp_path / "root")
    path = root / v2.ARTEFACT_DIR / "programme_w_dev.json"
    path.write_text(json.dumps({**read_json(path), "selected": "X5_COMBO", "x5_included": []}),
                    encoding="utf-8")
    with pytest.raises(DataContractError, match="X5 zonder opgenomen sleeves"):
        v2.read_holdout(root, log=lambda _: None)


def test_build_refuses_a_combination_without_sleeves():
    with pytest.raises(DataContractError, match="X5 zonder opgenomen sleeves"):
        v2.build("X5_COMBO", None, None, included=())


# --------------------------------------------------------------------------- #
# book_verdict
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "ledger_result"), [
    ("promote", "accepted"), ("falsified", "falsified"), ("archived", "archived")])
def test_the_verdict_is_booked_as_a_zero_trial_amendment_of_the_programme_entry(
        scenarios, tmp_path, name, ledger_result):
    flow = scenarios[name]
    root = copy_root(flow.root, tmp_path / "root")
    before = HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses()
    v2.book_verdict(root)

    programme, verdict = ledger_entries(root)
    hold = flow.holdout
    assert HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses() == before   # telt nul trials
    assert (verdict["wave"], verdict["unit"], verdict["n_trials"], verdict["result"]) == (
        2, "robust_book_v2_verdict", 0, ledger_result)
    assert verdict["amends"] == programme["config_hash"]
    assert verdict["preregistration_id"] == programme["preregistration_id"] == (
        flow.summary["preregistration_id"])
    assert verdict["git_sha"] == SHA and verdict["data_hash"] == programme["data_hash"]
    assert verdict["metrics"] == {
        "selected": hold["candidate"], "w_dev_sharpe": hold["w_dev_sharpe"],
        "holdout_sharpe": hold["holdout"]["sharpe"], "verdict": hold["verdict"],
        "robustness_scores": flow.summary["robustness_scores"]}


@pytest.mark.parametrize(("verdict", "ledger_result"), [
    ("promote_to_paper_trading", "accepted"), ("falsified_on_holdout", "falsified"),
    ("archived_w_dev_gates", "archived"), ("some_future_verdict", "archived")])
def test_every_verdict_string_maps_to_a_ledger_result_and_the_unknown_one_is_archived(
        frozen_root, verdict, ledger_result):
    out = frozen_root / v2.ARTEFACT_DIR
    out.mkdir(parents=True)
    (out / "holdout_read.json").write_text(json.dumps(
        {"verdict": verdict, "candidate": "X1_XSMOM", "w_dev_sharpe": 1.5,
         "holdout": {"sharpe": 0.5}}), encoding="utf-8")
    (out / "programme_w_dev.json").write_text(json.dumps(
        {"robustness_scores": {"X1_XSMOM": 61.0}}), encoding="utf-8")
    v2.book_verdict(frozen_root)
    assert ledger_entries(frozen_root)[-1]["result"] == ledger_result


def test_a_verdict_cannot_be_booked_before_the_holdout_was_read(frozen_root):
    with pytest.raises(FileNotFoundError, match=r"holdout_read\.json"):
        v2.book_verdict(frozen_root)
    assert len(ledger_entries(frozen_root)) == 1


def test_a_verdict_on_a_programme_that_was_never_booked_is_refused(tmp_path):
    root = build_root(tmp_path / "root")
    out = root / v2.ARTEFACT_DIR
    out.mkdir(parents=True)
    (out / "holdout_read.json").write_text(json.dumps(
        {"verdict": "archived_w_dev_gates", "candidate": "X1_XSMOM", "w_dev_sharpe": 1.0,
         "holdout": {"sharpe": 0.0}}), encoding="utf-8")
    (out / "programme_w_dev.json").write_text(json.dumps({"robustness_scores": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Amendement wijst naar config_hash"):
        v2.book_verdict(root)
    assert ledger_entries(root) == []


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Vervang de vier opdrachten, zodat `main` nooit de echte wortel van de repository raakt."""
    called: list[str] = []
    monkeypatch.setattr(v2, "freeze", lambda: called.append("freeze") or Path("pre_x.json"))
    monkeypatch.setattr(v2, "run_programme",
                        lambda: called.append("run") or {"selected": "X1", "score": np.float32(12.5)})
    monkeypatch.setattr(v2, "read_holdout",
                        lambda: called.append("holdout") or {"verdict": "archived_w_dev_gates"})
    monkeypatch.setattr(v2, "book_verdict", lambda: called.append("verdict"))
    return called


@pytest.mark.parametrize(("command", "printed"), [
    ("freeze", "pre_x.json\n"),
    ("run", json.dumps({"selected": "X1", "score": 12.5}, indent=2) + "\n"),
    ("holdout", json.dumps({"verdict": "archived_w_dev_gates"}, indent=2) + "\n"),
    ("verdict", "")])
def test_main_runs_exactly_the_requested_command_and_prints_its_result(
        commands, capsys, command, printed):
    v2.main([command])
    assert commands == [command]
    assert capsys.readouterr().out == printed


@pytest.mark.parametrize("argv", [[], ["thaw"], ["FREEZE"], ["freeze", "run"]])
def test_main_refuses_a_missing_unknown_or_double_command_before_running_anything(
        commands, argv):
    with pytest.raises(DataContractError, match=r"Gebruik: python -I -m tradebot\.systematic\.programme_v2"):
        v2.main(argv)
    assert commands == []


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_runs_as_a_script_and_refuses_a_call_without_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["programme_v2"])
    with pytest.raises(DataContractError, match="Gebruik"):
        runpy.run_module("tradebot.systematic.programme_v2", run_name="__main__")


# --------------------------------------------------------------------------- #
# Bevindingen: gepind, niet aangepast (het onderzoeksgedrag is vooraf geregistreerd)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "BEVINDING, niet aangepast: read_holdout vergelijkt de split van het slot niet met "
    "windows.holdout_start. Staat het slot eerder, dan overlapt de 'holdout' W_DEV. v1 "
    "(programme.py: 'gate_start wijkt af van de bevroren holdout-split') weigert dat wel."))
def test_a_lock_with_another_split_than_the_configured_holdout_start_is_refused(
        real_flow, tmp_path):
    root = copy_root(real_flow.root, tmp_path / "root")
    lock = root / v2.LOCK_PATH
    lock.write_text(json.dumps({**read_json(lock), "split_utc": "2021-06-01T00:00:00+00:00"}),
                    encoding="utf-8")
    with pytest.raises(DataContractError, match="split"):
        v2.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == []                               # geweigerd vóór de lezing wordt verbruikt


@pytest.mark.xfail(strict=True, reason=(
    "BEVINDING, niet aangepast: run_programme leest de DSR-M live uit de ledger. Is er na het "
    "bevriezen iets bijgeboekt, dan verschuift M en is dezelfde run niet reproduceerbaar; de "
    "preregistratie noemt M als vast getal ('cumulatieve ledger-M (14)')."))
def test_the_dsr_trial_count_stays_the_one_of_the_freeze_when_the_ledger_grows(
        scenarios, tmp_path, monkeypatch):
    root = copy_root(scenarios["promote"].root, tmp_path / "root")
    frozen_total = read_json(scenarios["promote"].frozen)["ledger_total_at_freeze"]
    HypothesisLedger(root / v2.LEDGER_PATH).append(LedgerEntry.from_config(
        wave=3, unit="robust_book_v3_programme", market="crypto", config={"p": 1}, git_sha=SHA,
        data_hash="0" * 16, preregistration_id="f" * 32, n_trials=3))
    monkeypatch.setattr(v2, "battery", light_battery(v2.battery, []))
    monkeypatch.setattr(v2, "pbo_record", fake_pbo)
    summary = v2.run_programme(root, log=lambda _: None)
    assert summary["m_programme"] == frozen_total
