"""De orkestratie van het robuuste boek v1 van begin tot eind: bevriezen, W_DEV meten, één
poortlezing en het oordeel in de ledger -- op een wegwerpmap met een synthetische markt.

`test_systematic_programme.py` bewijst dat elke trial te bouwen en te draaien is; hier staat wat
het PROGRAMMA ermee doet (`systematic/programme.py`): in welke volgorde hij boekt en weigert, wat hij
naar schijf schrijft en dat hij het poortsample pas leest nadat de lezing in het slot staat. Alles
leeft in `tmp_path`: de config, de preregistratie, de ledger en het holdout-slot zijn kleine
kopieën met dezelfde vorm als de echte, de markt wordt via `programme.load_market` ingespoten,
en de git-SHA komt uit de test. Er wordt niets uit `data/` gelezen en niets in de repo geschreven.

Twee dure binnenstappen worden vervangen, omdat een flowtoets ze niet toetst en ze elders hun
eigen toetsen hebben: `programme.battery` (tientallen boekruns per kandidaat; zie
`TestBatteryBranches` hieronder en `test_systematic_programme.py`) en de volledige CSCV-enumeratie
van `pbo_record` (12.870 combinaties; hier 6, met dezelfde functie). De dummy geeft vaste, bekende
uitkomsten, zodat de poorten deterministisch bepaald zijn door wat `run_programme` ermee doet.
"""
from __future__ import annotations

import json
import math
import re
import runpy
import sys
import zlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.systematic_fixtures import SYMBOLS, perturb_after, synthetic_market
from tests.weekly_fixtures import synthetic_market as weekly_market
from tradebot.backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from tradebot.data.weekly_market import WeeklyMarket
from tradebot.execution.impact_model import ImpactStatus
from tradebot.features.base import DataRegister
from tradebot.registry.hypothesis_ledger import HypothesisLedger
from tradebot.registry.preregistration import (
    StopCriterion,
    load_preregistration_spec,
    require_preregistration,
)
from tradebot.registry.weekly_programme import certified_data_hashes
from tradebot.schemas.robust_book import RobustBookConfig, robust_book_config
from tradebot.systematic import evaluate, programme
from tradebot.systematic.book import BookResult, CostSpec
from tradebot.systematic.evaluate import summarize
from tradebot.systematic.market import BookMarket, from_weekly_market
from tradebot.systematic.sleeves import SizingParams, SleeveTargets
from tradebot.utils.failfast import ConfigContractError, DataContractError
from tradebot.utils.hashing import hash_config
from tradebot.validation.holdout import HoldoutAlreadyUsed, freeze_holdout

REPO = Path(__file__).resolve().parents[2]
REAL_SPEC = REPO / "conf" / "research" / "preregistration_robust_book.yaml"
CONFIG_PATH = Path("conf/model/robust_book.yaml")
DATA_HASHES_PATH = Path("artefacts/governance/data_hashes.json")

#: De commit waarop bevroren wordt, en een andere waarop W_DEV en de poort draaien: zo zijn
#: `frozen_git_sha` en `git_sha` in de records te onderscheiden.
GIT = "abc1234"
RUN_GIT = "run5678"

N_BARS = 435  # 2021-01-01 .. 2022-03-11: W_DEV is 229 bars (twee kalenderjaren), het poortsample 55
WINDOWS = {"w_dev_start": "2021-06-01", "train_end": "2021-09-30",
           "validate_start": "2021-10-01", "w_dev_end": "2022-01-15", "gate_start": "2022-01-16"}
GATE_START = WINDOWS["gate_start"]

#: Kosten die zich laten narekenen: 6 bp taker, 2,5 bp halve spread, eta 1,5.
FEES = """\
execution:
  maker_fee_bps: 2.0
  taker_fee_bps: 6.0
  assumed_half_spread_bps: 2.5
  slippage_model: half_spread
  cost_assumption_is_provisional: true
"""
IMPACT = """\
impact:
  eta: 1.5
  kappa_d: 0.5
  status: IMPACT_UNCALIBRATED
  method: test
  data_hash: 0123456789abcdef
  sample_size: 100
  period_start: "2021-01-01"
  period_end: "2022-12-31"
  eta_ci_low: 1.0
  eta_ci_high: 2.0
  instruments: [BTCUSDT, ETHUSDT]
"""
PROMOTION = "promotion_requires_all_clear"
GATE_CRITERIA = {"gate_sharpe_collapse", "gate_drawdown"}


# --------------------------------------------------------------------------- #
# Een wegwerpmap in de vorm van de repo
# --------------------------------------------------------------------------- #
def small_config(planned_trials: int) -> dict[str, Any]:
    """Dezelfde sleutels als `conf/model/robust_book.yaml`, met vensters die in 435 bars passen.

    `inclusion_min_train_sharpe = -100` laat alle vier de sleeves in C5 komen, wat de markt ook doet.
    """
    return {
        "symbols": list(SYMBOLS),
        "windows": dict(WINDOWS),
        "execution": {"lag_bars": 1, "aum_usd": 100_000.0},
        "sizing": {"vol_span": 20, "min_history_bars": 20, "asset_vol_target": 0.20,
                   "portfolio_vol_target": 0.20, "max_scale": 2.0, "per_asset_cap": 1.0,
                   "gross_cap": 3.0},
        "trend": {"lookbacks": [5, 10, 20, 40], "z_clip": 1.0},
        "core": {"assets": ["BTCUSDT", "ETHUSDT"]},
        "carry": {"window_bars": 7, "signal_lag_bars": 1, "rebalance_bars": 7},
        "combo": {"inclusion_min_train_sharpe": -100.0},
        "planned_trials": planned_trials,
        "known_prior_trials": 52,
    }


def no_sleeve_qualifies(cfg: dict[str, Any]) -> None:
    cfg["combo"]["inclusion_min_train_sharpe"] = 100.0


def set_thresholds(**by_name: float) -> Callable[[dict[str, Any]], None]:
    """Een wijziging van de preregistratie-spec: nieuwe drempels voor benoemde criteria."""
    def edit(spec: dict[str, Any]) -> None:
        criteria = {c["name"]: c for c in spec["stop_criteria"]}
        assert set(by_name) <= set(criteria), set(by_name) - set(criteria)
        for name, value in by_name.items():
            criteria[name]["threshold"] = value
    return edit


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_root(
    root: Path,
    *,
    cfg_edit: Callable[[dict[str, Any]], None] | None = None,
    spec_edit: Callable[[dict[str, Any]], None] | None = None,
    ledger_seed: int = 0,
    lock_split: str | None = GATE_START,
    impact: str = IMPACT,
) -> Path:
    """Config, preregistratie-spec, kosten, data-register, lege ledger en holdout-slot onder `root`.

    De spec is die van de repo (dezelfde vijftien stop-criteria), de rest is klein en eigen.
    """
    spec = yaml.safe_load(REAL_SPEC.read_text(encoding="utf-8"))
    cfg = small_config(int(spec["planned_trials"]))
    if cfg_edit is not None:
        cfg_edit(cfg)
    if spec_edit is not None:
        spec_edit(spec)
    _write(root / CONFIG_PATH, yaml.safe_dump(cfg, sort_keys=False))
    _write(root / programme.SPEC_PATH,
           yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=10_000))
    _write(root / "conf/execution/fees.yaml", FEES)
    _write(root / "conf/execution/impact.yaml", impact)
    hashes = {f"crypto/{dataset}/{sym}/{gran}": f"{zlib.crc32(f'{dataset}{sym}'.encode()):08x}" * 4
              for sym in SYMBOLS
              for dataset, gran in (("ohlcv", "1d"), ("funding", "8h"), ("open_interest", "1d"))}
    _write(root / DATA_HASHES_PATH, json.dumps(hashes))
    _write(root / programme.LEDGER_PATH, json.dumps(
        {"version": 1, "seed_total": ledger_seed, "seed_note": "test", "entries": []}))
    if lock_split is not None:
        freeze_holdout(split_utc=f"{lock_split}T00:00:00+00:00", out=root / programme.LOCK_PATH,
                       git_sha=GIT)
    return root


def flow_market() -> BookMarket:
    return synthetic_market(N_BARS)


def utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_id(frozen: Path) -> str:
    return str(read_json(frozen)["preregistration_id"])


def load_cfg(root: Path) -> RobustBookConfig:
    return robust_book_config(root / CONFIG_PATH)


class FakeBattery:
    """Vervangt `programme.battery` en onthoudt hoe hij geroepen werd (zie de moduledocstring)."""

    X2, LAG2, PLATEAU, MIN_PERTURBATION, LEAVE_ONE_OUT = 0.4, 0.25, 0.5, -0.2, 0.5

    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []

    def __call__(
        self, name: str, market: BookMarket, cfg: RobustBookConfig, costs: CostSpec, *,
        included: tuple[str, ...], base: BookResult, w_dev: tuple[pd.Timestamp, pd.Timestamp],
    ) -> tuple[dict[str, Any], dict[str, pd.Series]]:
        self.calls.append(SimpleNamespace(name=name, included=tuple(included), w_dev=w_dev,
                                          costs=costs, base=base))
        a, b = w_dev
        net = base.window(a, b).net
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        family = {"base": net}
        for i in range(1, 7):
            family[f"variant_{i}"] = net * (1.0 - 0.1 * i) + rng.normal(0.0, 1e-4, len(net))
        bat = {"base_sharpe": 0.0, "costs": {"x2": self.X2}, "delay": {"lag_2": self.LAG2},
               "plateau_fraction": self.PLATEAU, "min_perturbation_sharpe": self.MIN_PERTURBATION,
               "leave_one_out_positive_fraction": self.LEAVE_ONE_OUT}
        return bat, family


def install(
    mp: pytest.MonkeyPatch,
    market: BookMarket,
    *,
    battery: FakeBattery | None = None,
    loads: list[Path] | None = None,
) -> FakeBattery:
    """Spuit de markt, de git-SHA en de twee goedkope binnenstappen in `programme`."""
    battery = battery if battery is not None else FakeBattery()

    def load_market(root: Path, cfg: RobustBookConfig) -> BookMarket:
        if loads is not None:
            loads.append(Path(root))
        return market

    mp.setattr(programme, "load_market", load_market)
    mp.setattr(programme, "current_git_sha", lambda: RUN_GIT)
    mp.setattr(programme, "battery", battery)
    mp.setattr(programme, "pbo_record", lambda variants: evaluate.pbo_record(variants, n_subsets=4))
    return battery


def independent_run(root: Path, market: BookMarket, name: str, *, included: tuple[str, ...] = (),
                    until: pd.Timestamp | None = None) -> BookResult:
    """Dezelfde trial, rechtstreeks uit de bouwstenen (zonder `run_programme`/`read_gate`)."""
    cfg = load_cfg(root)
    mk = market if until is None else market.truncate(until)
    return programme.run_targets(programme.build_trial(name, mk, cfg, included=included), mk,
                                 programme.cost_spec(root, cfg), lag=cfg.execution.lag_bars)


def stage_gate(
    tmp_path: Path,
    market: BookMarket,
    *,
    name: str = "C3_VOLMAN_CORE",
    included: tuple[str, ...] = (),
    dev_binding: int = 0,
    sharpe_shift: float = 0.0,
    spec_edit: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, str]:
    """Een bevroren programma met de W_DEV-artefacten die `read_gate` leest, zonder de run zelf.

    De W_DEV-Sharpe in het artefact is die van de afgekapte run op het W_DEV-venster, zoals
    `run_programme` hem zou schrijven; `sharpe_shift` verstoort hem om de consistentiecontrole te raken.
    """
    root = write_root(tmp_path / "root", spec_edit=spec_edit)
    pid = frozen_id(programme.freeze(root, git_sha=GIT))
    res = independent_run(root, market, name, included=included, until=utc(GATE_START))
    w_dev = summarize(res.window(utc(WINDOWS["w_dev_start"]), utc(WINDOWS["w_dev_end"])),
                      bootstrap=False)["sharpe"]
    out = root / programme.ARTEFACT_DIR
    _write(out / "programme_w_dev.json", json.dumps({
        "preregistration_id": pid, "selected": name, "c5_included": list(included),
        "robustness_scores": {name: 41.5}}))
    _write(out / f"{name}.json", json.dumps({
        "w_dev": {"sharpe": w_dev + sharpe_shift}, "n_binding_w_dev": dev_binding}))
    return root, pid


# --------------------------------------------------------------------------- #
# De hele keten, één keer gedraaid
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """freeze -> run -> gate -> verdict op één wegwerpmap. De toetsen lezen hem, schrijven niet."""
    root = tmp_path_factory.mktemp("robust_book_chain") / "root"
    market = flow_market()
    logs = SimpleNamespace(run=[], gate=[])
    with pytest.MonkeyPatch.context() as mp:
        battery = install(mp, market)
        write_root(root)
        frozen = programme.freeze(root, git_sha=GIT)
        summary = programme.run_programme(root, log=logs.run.append)
        gate = programme.read_gate(root, log=logs.gate.append)
        programme.book_verdict(root)
    return SimpleNamespace(root=root, market=market, battery=battery, logs=logs, frozen=frozen,
                           summary=summary, gate=gate, cfg=load_cfg(root),
                           pid=frozen_id(frozen), art=root / programme.ARTEFACT_DIR)


@pytest.fixture(scope="module")
def no_combo(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """De run waarin geen sleeve in C5 komt, op een markt waarvan het poortsample vervangen is door ruis."""
    root = tmp_path_factory.mktemp("robust_book_no_combo") / "root"
    market = flow_market()
    last_dev_bar = int(market.index.get_loc(utc(GATE_START))) - 1
    scrambled = perturb_after(market, last_dev_bar)
    logs: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        battery = install(mp, scrambled)
        write_root(root, cfg_edit=no_sleeve_qualifies)
        programme.freeze(root, git_sha=GIT)
        summary = programme.run_programme(root, log=logs.append)
    return SimpleNamespace(root=root, battery=battery, logs=logs, summary=summary,
                           art=root / programme.ARTEFACT_DIR)


# --------------------------------------------------------------------------- #
# Kleine functies
# --------------------------------------------------------------------------- #
class TestSmallHelpers:
    def test_a_window_date_is_midnight_utc(self) -> None:
        ts = programme._ts("2022-03-01")
        assert ts == pd.Timestamp("2022-03-01T00:00:00+00:00") and str(ts.tz) == "UTC"

    def test_the_parameters_are_the_config_as_plain_json_types(self) -> None:
        cfg = robust_book_config()
        params = programme.programme_parameters(cfg)
        assert params["trend"]["lookbacks"] == list(cfg.trend.lookbacks)
        assert params["symbols"] == list(cfg.symbols) and params["windows"]["gate_start"] == cfg.windows.gate_start
        assert json.loads(json.dumps(params)) == params

    def test_the_planned_trials_are_exactly_the_trials_the_programme_runs(self) -> None:
        spec = load_preregistration_spec(REAL_SPEC, data_hashes=(("x", "0"),), parameters={"p": 1})
        assert robust_book_config().planned_trials == spec.planned_trials == len(programme.TRIALS) == 7
        assert programme.TRIALS == programme.REFERENCES + programme.CANDIDATES

    def test_costs_are_read_from_the_root_and_converted_from_basis_points(self, tmp_path: Path) -> None:
        root = write_root(tmp_path)
        cfg = load_cfg(root)
        costs = programme.cost_spec(root, cfg)
        assert costs.taker_fee == pytest.approx(6.0e-4) and costs.half_spread == pytest.approx(2.5e-4)
        assert costs.aum_usd == cfg.execution.aum_usd == 100_000.0
        assert (costs.extra_slippage, costs.multiplier, costs.impact_eta) == (0.0, 1.0, None)
        assert costs.impact is not None and costs.impact.eta == 1.5
        assert costs.impact.status is ImpactStatus.IMPACT_UNCALIBRATED and costs.eta == 1.5

    def test_a_missing_cost_file_stops_the_costs(self, tmp_path: Path) -> None:
        root = write_root(tmp_path)
        (root / "conf/execution/impact.yaml").unlink()
        with pytest.raises(ConfigContractError, match=r"impact\.yaml"):
            programme.cost_spec(root, load_cfg(root))

    def test_the_market_comes_from_the_weekly_loader_for_exactly_the_configured_symbols(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wm: WeeklyMarket = weekly_market(n=120, symbols=("BTCUSDT", "ETHUSDT"))
        asked: list[tuple[Path, tuple[str, ...]]] = []

        def fake(root: Path, symbols: tuple[str, ...]) -> WeeklyMarket:
            asked.append((root, tuple(symbols)))
            return wm

        monkeypatch.setattr(programme, "load_weekly_market", fake)
        cfg = robust_book_config()
        market = programme.load_market(tmp_path, cfg)
        assert asked == [(tmp_path, cfg.symbols)]
        expected = from_weekly_market(wm)
        pd.testing.assert_frame_equal(market.close, expected.close)
        pd.testing.assert_frame_equal(market.sigma_daily, expected.sigma_daily)
        assert market.symbols == ("BTCUSDT", "ETHUSDT")

    def test_dump_creates_the_folders_sorts_the_keys_and_keeps_text_and_numpy_numbers(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "x.json"
        programme._dump(target, {"b": np.float32(1.5), "a": "é", "n": float("nan")})
        assert target.read_bytes().decode("utf-8") == '{\n  "a": "é",\n  "b": 1.5,\n  "n": NaN\n}'

    def test_rebalancing_every_bar_or_less_often_than_never_leaves_the_targets_alone(self) -> None:
        idx = pd.date_range("2021-01-01", periods=6, freq="D", tz="UTC")
        t = SleeveTargets(name="T", weights=pd.DataFrame({"A": np.arange(6.0)}, index=idx),
                          rebalance=pd.Series(True, index=idx))
        assert programme._every(t, 1) is t and programme._every(t, 0) is t

    def test_every_k_keeps_the_first_decision_and_then_each_kth_decision_only(self) -> None:
        idx = pd.date_range("2021-01-01", periods=8, freq="D", tz="UTC")
        decisions = [False, True, True, False, True, True, True, True]  # zes besluiten, het eerste op bar 1
        t = SleeveTargets(name="T", weights=pd.DataFrame({"A": np.arange(8.0)}, index=idx),
                          rebalance=pd.Series(decisions, index=idx))
        thinned = programme._every(t, 3)
        # Besluit 0 en besluit 3 (bar 1 en bar 5): geteld over besluiten, niet over kalenderbars.
        assert thinned.rebalance.tolist() == [False, True, False, False, False, True, False, False]
        assert thinned.weights["A"].notna().tolist() == thinned.rebalance.tolist()
        assert thinned.weights["A"][thinned.rebalance].tolist() == [1.0, 5.0]
        assert thinned.name == "T"


# --------------------------------------------------------------------------- #
# De poorten van de preregistratie
# --------------------------------------------------------------------------- #
class TestGates:
    @staticmethod
    def _prereg() -> SimpleNamespace:
        def crit(name: str, metric: str, op: str, threshold: float) -> StopCriterion:
            return StopCriterion(name=name, metric=metric, operator=op, threshold=threshold,
                                 action="archive", rationale="test")
        return SimpleNamespace(stop_criteria=(
            crit("low", "m_low", "<", 1.0), crit("high", "m_high", ">", 0.25),
            crit("at_most", "m_at_most", "<=", 0.0), crit("at_least", "m_at_least", ">=", 0.5),
            crit(PROMOTION, "n_binding_stop_criteria", "<=", 0.0)))

    def test_each_operator_binds_on_its_own_side_of_the_threshold(self) -> None:
        gates = programme._gates({"m_low": 0.99, "m_high": 0.26, "m_at_most": 0.0, "m_at_least": 0.5},
                                 self._prereg())
        assert {k: g["binds"] for k, g in gates.items()} == {
            "low": True, "high": True, "at_most": True, "at_least": True}
        gates = programme._gates({"m_low": 1.0, "m_high": 0.25, "m_at_most": 0.01, "m_at_least": 0.49},
                                 self._prereg())
        assert {k: g["binds"] for k, g in gates.items()} == {
            "low": False, "high": False, "at_most": False, "at_least": False}

    def test_a_gate_reports_its_metric_value_operator_and_threshold(self) -> None:
        gates = programme._gates({"m_low": 0.5}, self._prereg())
        assert gates == {"low": {"metric": "m_low", "value": 0.5, "operator": "<", "threshold": 1.0,
                                 "binds": True}}

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_value_that_is_not_finite_binds_whatever_side_it_is_on(self, value: float) -> None:
        gates = programme._gates({"m_low": value, "m_high": value, "m_at_most": value,
                                  "m_at_least": value}, self._prereg())
        assert all(g["binds"] is True for g in gates.values()) and len(gates) == 4

    def test_only_the_measured_criteria_are_evaluated_and_the_promotion_rule_never(self) -> None:
        gates = programme._gates({"m_high": 0.1, "n_binding_stop_criteria": 0.0, "unrelated": 5.0},
                                 self._prereg())
        assert set(gates) == {"high"}

    def test_a_metric_given_as_numpy_scalar_becomes_a_plain_float(self) -> None:
        gates = programme._gates({"m_low": np.float32(0.5)}, self._prereg())
        assert type(gates["low"]["value"]) is float and type(gates["low"]["binds"]) is bool


# --------------------------------------------------------------------------- #
# Bevriezen
# --------------------------------------------------------------------------- #
class TestFreeze:
    def test_it_books_the_planned_trials_first_and_then_freezes_the_preregistration(self, tmp_path: Path) -> None:
        root = write_root(tmp_path)
        path = programme.freeze(root, git_sha=GIT)
        cfg = load_cfg(root)
        frozen = read_json(path)
        (entry,) = HypothesisLedger(root / programme.LEDGER_PATH).entries()
        hashes = certified_data_hashes(DataRegister(root / DATA_HASHES_PATH), cfg.symbols)

        assert path == root / programme.PREREG_DIR / f"preregistration_{frozen['preregistration_id']}.json"
        assert (frozen["git_sha"], frozen["ledger_total_at_freeze"]) == (GIT, cfg.planned_trials)
        assert frozen["content"]["parameters"] == programme.programme_parameters(cfg)
        assert frozen["content"]["data_hashes"] == [list(pair) for pair in hashes]
        assert len(hashes) == 3 * len(cfg.symbols)

        assert (entry["wave"], entry["unit"], entry["market"], entry["result"]) == (
            1, "robust_book_v1_programme", "crypto", "interim")
        assert entry["n_trials"] == cfg.planned_trials and entry["amends"] == ""
        assert entry["preregistration_id"] == frozen["preregistration_id"] and entry["git_sha"] == GIT
        assert entry["config_hash"] == hash_config(programme.programme_parameters(cfg))
        assert entry["data_hash"] == hash_config(dict(hashes))
        assert "B1" in entry["notes"] and "C5" in entry["notes"]

    def test_the_ledger_total_at_freezing_counts_what_was_there_before(self, tmp_path: Path) -> None:
        root = write_root(tmp_path, ledger_seed=5)
        path = programme.freeze(root, git_sha=GIT)
        assert read_json(path)["ledger_total_at_freeze"] == 5 + load_cfg(root).planned_trials

    def test_without_an_explicit_sha_it_uses_the_current_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(programme, "current_git_sha", lambda: "feed123")
        root = write_root(tmp_path)
        path = programme.freeze(root)
        assert read_json(path)["git_sha"] == "feed123"
        assert HypothesisLedger(root / programme.LEDGER_PATH).entries()[0]["git_sha"] == "feed123"

    def test_a_programme_can_be_booked_only_once_even_from_another_commit(self, tmp_path: Path) -> None:
        root = write_root(tmp_path)
        programme.freeze(root, git_sha=GIT)
        with pytest.raises(DataContractError, match="Dit programma is al geboekt"):
            programme.freeze(root, git_sha="other99")
        assert len(HypothesisLedger(root / programme.LEDGER_PATH).entries()) == 1

    def test_a_spec_that_plans_another_number_of_trials_than_the_config_books_nothing(self, tmp_path: Path) -> None:
        root = write_root(tmp_path, cfg_edit=lambda c: c.update(planned_trials=c["planned_trials"] + 1))
        with pytest.raises(DataContractError, match="planned_trials in de preregistratie wijkt af van de config"):
            programme.freeze(root, git_sha=GIT)
        assert HypothesisLedger(root / programme.LEDGER_PATH).entries() == []
        assert list((root / programme.PREREG_DIR).glob("preregistration_*.json")) == []

    def test_without_a_ledger_nothing_is_frozen_and_no_ledger_is_invented(self, tmp_path: Path) -> None:
        root = write_root(tmp_path)
        (root / programme.LEDGER_PATH).unlink()
        with pytest.raises(FileNotFoundError, match="Ledger not found"):
            programme.freeze(root, git_sha=GIT)
        assert not (root / programme.LEDGER_PATH).exists()
        assert list((root / programme.PREREG_DIR).glob("preregistration_*.json")) == []

    def test_a_data_series_missing_from_the_register_stops_the_freeze(self, tmp_path: Path) -> None:
        root = write_root(tmp_path)
        register = read_json(root / DATA_HASHES_PATH)
        del register["crypto/funding/SOLUSDT/8h"]
        _write(root / DATA_HASHES_PATH, json.dumps(register))
        with pytest.raises(DataContractError, match="NIET in het gecertificeerde data-register"):
            programme.freeze(root, git_sha=GIT)
        assert HypothesisLedger(root / programme.LEDGER_PATH).entries() == []


# --------------------------------------------------------------------------- #
# W_DEV
# --------------------------------------------------------------------------- #
class TestRunProgrammeRefusals:
    def test_it_refuses_without_a_frozen_preregistration_before_loading_any_market(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loads: list[Path] = []
        install(monkeypatch, flow_market(), loads=loads)
        root = write_root(tmp_path)
        with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
            programme.run_programme(root, log=lambda _m: None)
        assert loads == [] and not (root / programme.ARTEFACT_DIR).exists()

    @pytest.mark.parametrize("what", ["config", "spec", "data register"])
    def test_a_change_after_freezing_is_an_amendment_and_the_run_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, what: str,
    ) -> None:
        loads: list[Path] = []
        install(monkeypatch, flow_market(), loads=loads)
        root = write_root(tmp_path)
        programme.freeze(root, git_sha=GIT)
        if what == "config":
            cfg = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
            cfg["known_prior_trials"] += 1
            _write(root / CONFIG_PATH, yaml.safe_dump(cfg, sort_keys=False))
        elif what == "spec":
            spec = yaml.safe_load((root / programme.SPEC_PATH).read_text(encoding="utf-8"))
            spec["stop_criteria"][0]["threshold"] = 0.5
            _write(root / programme.SPEC_PATH, yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
        else:
            register = read_json(root / DATA_HASHES_PATH)
            register["crypto/ohlcv/BTCUSDT/1d"] = "f" * 32
            _write(root / DATA_HASHES_PATH, json.dumps(register))
        with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
            programme.run_programme(root, log=lambda _m: None)
        assert loads == []

    def test_a_gate_start_that_differs_from_the_frozen_split_is_refused_before_loading_the_market(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loads: list[Path] = []
        install(monkeypatch, flow_market(), loads=loads)
        root = write_root(tmp_path, lock_split="2022-04-01")
        programme.freeze(root, git_sha=GIT)
        with pytest.raises(DataContractError, match="gate_start wijkt af van de bevroren holdout-split"):
            programme.run_programme(root, log=lambda _m: None)
        assert loads == [] and not (root / programme.ARTEFACT_DIR).exists()

    def test_without_a_holdout_lock_there_is_no_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        loads: list[Path] = []
        install(monkeypatch, flow_market(), loads=loads)
        root = write_root(tmp_path, lock_split=None)
        programme.freeze(root, git_sha=GIT)
        with pytest.raises(FileNotFoundError, match="holdout_lock_robust_book_v1"):
            programme.run_programme(root, log=lambda _m: None)
        assert loads == []

    def test_a_ledger_with_other_content_at_freezing_is_refused_before_anything_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install(monkeypatch, flow_market())
        root = write_root(tmp_path, ledger_seed=3)
        programme.freeze(root, git_sha=GIT)
        with pytest.raises(DataContractError, match="Bij het bevriezen stond er iets anders in de ledger") as err:
            programme.run_programme(root, log=lambda _m: None)
        assert "ledger_total=10" in str(err.value) and "planned=7" in str(err.value)
        assert not (root / programme.ARTEFACT_DIR).exists()


class TestRunProgramme:
    def test_the_summary_scores_every_candidate_and_tabulates_every_trial(self, chain: SimpleNamespace) -> None:
        s, cfg = chain.summary, chain.cfg
        assert s["programme"] == "robust_book_v1" and s["preregistration_id"] == chain.pid
        assert s["git_sha"] == RUN_GIT and s[EVIDENCE_KEY] == NOT_ADMISSIBLE
        assert (s["m_programme"], s["m_including_known_prior"]) == (7, 7 + cfg.known_prior_trials)
        assert s["c5_included"] == list(programme.SLEEVES)
        assert set(s["robustness_scores"]) == set(programme.CANDIDATES)
        assert all(0.0 <= v <= 100.0 for v in s["robustness_scores"].values())
        assert s["selected"] == max(s["robustness_scores"], key=lambda k: s["robustness_scores"][k])
        assert set(s["table"]) == set(programme.TRIALS)
        assert set(s["table"]["C1_TREND_LS"]) == {
            "sharpe", "sharpe_se", "cagr", "ann_vol", "max_drawdown", "sortino", "calmar",
            "ann_turnover", "avg_gross_leverage", "sharpe_ci_low", "sharpe_ci_high"}

    def test_the_gates_passed_agree_with_the_binding_count_of_each_candidate(self, chain: SimpleNamespace) -> None:
        passes = chain.summary["passes_all_w_dev_gates"]
        assert set(passes) == set(programme.CANDIDATES)
        for name in programme.CANDIDATES:
            rec = read_json(chain.art / f"{name}.json")
            assert passes[name] is (rec["n_binding_w_dev"] == 0)

    def test_the_summary_is_written_next_to_the_trial_records(self, chain: SimpleNamespace) -> None:
        written = json.dumps(read_json(chain.art / "programme_w_dev.json"), sort_keys=True)
        assert written == json.dumps(chain.summary, sort_keys=True, default=float)

    def test_the_log_runs_references_then_sleeves_then_the_combination_then_the_batteries(
        self, chain: SimpleNamespace,
    ) -> None:
        assert chain.logs.run == (
            [f"run {n}" for n in programme.REFERENCES + programme.SLEEVES]
            + [f"C5 neemt op (TRAIN-Sharpe > -100.0): {programme.SLEEVES}"]
            + [f"batterij {n}" for n in programme.CANDIDATES])

    def test_each_candidate_goes_through_the_battery_once_on_the_dev_window_with_its_own_result(
        self, chain: SimpleNamespace,
    ) -> None:
        calls = chain.battery.calls
        win = chain.cfg.windows
        assert [c.name for c in calls] == list(programme.CANDIDATES)
        assert all(c.included == programme.SLEEVES for c in calls)
        assert all(c.w_dev == (utc(win.w_dev_start), utc(win.w_dev_end)) for c in calls)
        assert all(c.costs == programme.cost_spec(chain.root, chain.cfg) for c in calls)
        record = read_json(chain.art / "C2_TREND_LF.json")
        assert record["battery"] == {
            "base_sharpe": 0.0, "costs": {"x2": FakeBattery.X2}, "delay": {"lag_2": FakeBattery.LAG2},
            "plateau_fraction": FakeBattery.PLATEAU,
            "min_perturbation_sharpe": FakeBattery.MIN_PERTURBATION,
            "leave_one_out_positive_fraction": FakeBattery.LEAVE_ONE_OUT}
        assert record["pbo"]["n_variants"] == 7  # de familie van de dummy: basis + zes varianten

    def test_the_trials_are_run_on_a_market_that_stops_before_the_gate_sample(
        self, chain: SimpleNamespace,
    ) -> None:
        """Niet alleen de cijfers maar ook de invoer: het poortsample zit niet in de runs, ook al
        levert de lader het wel (de markt loopt door tot na `gate_start`)."""
        assert chain.market.index.max() >= utc(GATE_START)
        assert chain.battery.calls
        for call in chain.battery.calls:
            assert call.base.frame.index.max() < utc(GATE_START), call.name

    def test_every_trial_leaves_a_record_and_a_daily_series_of_exactly_the_dev_window(
        self, chain: SimpleNamespace,
    ) -> None:
        win = chain.cfg.windows
        days = pd.date_range(win.w_dev_start, win.w_dev_end, freq="D", tz="UTC")
        for name in programme.TRIALS:
            assert (chain.art / f"{name}.json").is_file()
            daily = pd.read_csv(chain.art / f"{name}_daily_w_dev.csv", index_col=0)
            assert len(daily) == len(days) and daily.index[0][:10] == win.w_dev_start
            assert daily.index[-1][:10] == win.w_dev_end
            assert {"net", "gross", "funding", "fees", "turnover", "gross_leverage"} <= set(daily.columns)

    def test_the_daily_series_is_the_net_return_of_the_independent_run(self, chain: SimpleNamespace) -> None:
        win = chain.cfg.windows
        res = independent_run(chain.root, chain.market, "C3_VOLMAN_CORE", until=utc(GATE_START))
        expected = res.window(utc(win.w_dev_start), utc(win.w_dev_end)).net.to_numpy()
        daily = pd.read_csv(chain.art / "C3_VOLMAN_CORE_daily_w_dev.csv", index_col=0)
        np.testing.assert_allclose(daily["net"].to_numpy(), expected, rtol=1e-8, atol=1e-12)

    def test_a_candidate_record_carries_provenance_windows_costs_and_the_unread_gate(
        self, chain: SimpleNamespace,
    ) -> None:
        rec = read_json(chain.art / "C1_TREND_LS.json")
        win = chain.cfg.windows
        register = DataRegister(chain.root / DATA_HASHES_PATH)
        assert (rec["trial"], rec["role"]) == ("C1_TREND_LS", "candidate")
        assert rec["hypothesis"] == programme.HYPOTHESES["C1_TREND_LS"]
        assert rec["preregistration_id"] == chain.pid
        assert (rec["git_sha"], rec["frozen_git_sha"]) == (RUN_GIT, GIT)
        assert rec["data_hashes"] == dict(certified_data_hashes(register, chain.cfg.symbols))
        assert rec["parameters"] == programme.programme_parameters(chain.cfg)
        assert rec[EVIDENCE_KEY] == NOT_ADMISSIBLE
        assert rec["windows"] == {
            "train": [win.w_dev_start, win.train_end], "validate": [win.validate_start, win.w_dev_end],
            "w_dev": [win.w_dev_start, win.w_dev_end], "gate": [win.gate_start, "unread"]}
        costs = programme.cost_spec(chain.root, chain.cfg)
        assert rec["execution"] == {"lag_bars": 1, "costs": json.loads(json.dumps(costs.as_record()))}
        assert rec["execution"]["costs"]["taker_fee_bps"] == pytest.approx(6.0)

    def test_the_windows_of_a_record_end_before_the_gate_and_do_not_overlap(self, chain: SimpleNamespace) -> None:
        rec = read_json(chain.art / "C1_TREND_LS.json")
        win = chain.cfg.windows
        assert (rec["w_dev"]["start"], rec["w_dev"]["end"]) == (win.w_dev_start, win.w_dev_end)
        assert (rec["train"]["start"], rec["train"]["end"]) == (win.w_dev_start, win.train_end)
        assert (rec["validate"]["start"], rec["validate"]["end"]) == (win.validate_start, win.w_dev_end)
        assert rec["train"]["n_obs"] + rec["validate"]["n_obs"] == rec["w_dev"]["n_obs"]
        assert rec["w_dev"]["end"] < win.gate_start
        assert sum(y["n_obs"] for y in rec["yearly"].values()) == rec["w_dev"]["n_obs"]
        assert set(rec["yearly"]) == {"2021", "2022"}
        assert set(rec["regimes"]) == {"bull_btc_above_ma200", "bear_btc_below_ma200", "high_vol",
                                       "low_vol", "high_correlation", "low_correlation"}
        assert all(0.0 <= r["share_of_bars"] <= 1.0 for r in rec["regimes"].values())

    def test_the_gate_metrics_are_the_measured_figures_in_the_units_of_the_preregistration(
        self, chain: SimpleNamespace,
    ) -> None:
        rec = read_json(chain.art / "C1_TREND_LS.json")
        gm, dev = rec["gate_metrics"], rec["w_dev"]
        assert gm["net_sharpe_w_dev"] == dev["sharpe"] and gm["net_cagr_w_dev"] == dev["cagr"]
        assert gm["net_sharpe_train"] == rec["train"]["sharpe"]
        assert gm["net_sharpe_validate"] == rec["validate"]["sharpe"]
        assert gm["max_drawdown_w_dev"] == dev["max_drawdown"]
        assert gm["sharpe_ci_low_w_dev"] == dev["sharpe_ci_low"] and gm["pbo"] == rec["pbo"]["pbo"]
        assert gm["plateau_fraction"] == FakeBattery.PLATEAU
        assert gm["min_perturbation_sharpe"] == FakeBattery.MIN_PERTURBATION
        if dev["sharpe"] > 0:
            assert gm["sharpe_ratio_2x_cost_over_base"] == pytest.approx(FakeBattery.X2 / dev["sharpe"])
            assert gm["sharpe_ratio_lag2_over_base"] == pytest.approx(FakeBattery.LAG2 / dev["sharpe"])
        else:
            assert math.isnan(gm["sharpe_ratio_2x_cost_over_base"])
            assert math.isnan(gm["sharpe_ratio_lag2_over_base"])
        assert set(gm) == {
            "net_sharpe_w_dev", "net_cagr_w_dev", "net_sharpe_train", "net_sharpe_validate",
            "max_drawdown_w_dev", "sharpe_ratio_2x_cost_over_base", "sharpe_ratio_lag2_over_base",
            "plateau_fraction", "min_perturbation_sharpe", "sharpe_ci_low_w_dev", "dsr_w_dev", "pbo"}

    def test_the_dsr_gate_is_deflated_by_the_programme_trials_not_by_the_double_counted_total(
        self, chain: SimpleNamespace,
    ) -> None:
        rec = read_json(chain.art / "C1_TREND_LS.json")
        m_prog, prior = chain.cfg.planned_trials, chain.cfg.known_prior_trials
        # De ledger stond bij het bevriezen al op `m_prog` (de eigen boeking): 7 + 7 telt dezelfde trials twee keer.
        assert set(rec["dsr"]) == {f"m_{m_prog}", f"m_{2 * m_prog}", f"m_{m_prog + prior}"}
        assert rec["gate_metrics"]["dsr_w_dev"] == rec["dsr"][f"m_{m_prog}"]["dsr"]
        assert rec["dsr"][f"m_{m_prog}"]["dsr"] >= rec["dsr"][f"m_{m_prog + prior}"]["dsr"]

    def test_every_dev_gate_is_evaluated_in_the_run_and_the_two_gate_criteria_at_the_gate(
        self, chain: SimpleNamespace,
    ) -> None:
        prereg = require_preregistration(chain.pid, directory=chain.root / programme.PREREG_DIR)
        decisive = {c.name for c in prereg.stop_criteria if c.name != PROMOTION}
        for name in programme.CANDIDATES:
            rec = read_json(chain.art / f"{name}.json")
            assert set(rec["gates"]) == decisive - GATE_CRITERIA
            assert rec["n_binding_w_dev"] == sum(g["binds"] for g in rec["gates"].values())
        assert set(chain.gate["gate_gates"]) == GATE_CRITERIA

    def test_a_reference_is_recorded_without_battery_gates_or_score(self, chain: SimpleNamespace) -> None:
        for name in programme.REFERENCES:
            rec = read_json(chain.art / f"{name}.json")
            assert rec["role"] == "reference" and rec["trial"] == name
            assert not {"battery", "pbo", "gates", "gate_metrics", "n_binding_w_dev",
                        "robustness_score"} & set(rec)
            assert {"w_dev", "train", "validate", "yearly", "regimes", "dsr"} <= set(rec)

    def test_the_combination_names_its_members_and_their_train_sharpe(self, chain: SimpleNamespace) -> None:
        rec = read_json(chain.art / "C5_COMBO.json")
        assert rec["included"] == list(programme.SLEEVES)
        assert set(rec["train_sharpe_of_sleeves"]) == set(programme.SLEEVES)
        for name, value in rec["train_sharpe_of_sleeves"].items():
            assert value == pytest.approx(read_json(chain.art / f"{name}.json")["train"]["sharpe"])

    def test_the_reference_figures_are_those_of_the_dev_window_with_the_configured_costs(
        self, chain: SimpleNamespace,
    ) -> None:
        win = chain.cfg.windows
        for name in ("B1_BTC_HOLD", "C2_TREND_LF"):
            res = independent_run(chain.root, chain.market, name, until=utc(GATE_START))
            expected = summarize(res.window(utc(win.w_dev_start), utc(win.w_dev_end)))
            table = chain.summary["table"][name]
            for key, value in table.items():
                assert value == pytest.approx(expected[key], nan_ok=True), (name, key)

    def test_the_run_is_blind_to_the_gate_sample_and_reports_it_as_unread(
        self, chain: SimpleNamespace, no_combo: SimpleNamespace,
    ) -> None:
        """Dezelfde W_DEV-cijfers op een markt waarvan alles vanaf `gate_start` door ruis is vervangen."""
        for name in (*programme.REFERENCES, "C1_TREND_LS", "C2_TREND_LF", "C3_VOLMAN_CORE", "C4_CARRY_XS"):
            assert json.dumps(no_combo.summary["table"][name], sort_keys=True) == json.dumps(
                chain.summary["table"][name], sort_keys=True), name
        for name in programme.SLEEVES:
            assert no_combo.summary["robustness_scores"][name] == chain.summary["robustness_scores"][name]
        assert read_json(no_combo.art / "C1_TREND_LS.json")["windows"]["gate"] == [GATE_START, "unread"]


class TestWhenNoSleeveQualifiesForTheCombination:
    def test_the_combination_is_not_built_and_left_out_of_score_and_table(self, no_combo: SimpleNamespace) -> None:
        s = no_combo.summary
        assert s["c5_included"] == []
        assert set(s["robustness_scores"]) == set(programme.SLEEVES) and s["selected"] in programme.SLEEVES
        assert set(s["table"]) == set(programme.REFERENCES + programme.SLEEVES)
        assert s["passes_all_w_dev_gates"]["C5_COMBO"] is False

    def test_it_does_not_run_a_battery_or_leave_a_file_for_the_combination(self, no_combo: SimpleNamespace) -> None:
        assert [c.name for c in no_combo.battery.calls] == list(programme.SLEEVES)
        assert sorted(p.name for p in no_combo.art.glob("C5_COMBO*")) == []
        assert (no_combo.art / "C4_CARRY_XS.json").is_file()

    def test_the_log_says_that_nothing_was_included(self, no_combo: SimpleNamespace) -> None:
        assert "C5 neemt op (TRAIN-Sharpe > 100.0): ()" in no_combo.logs
        assert "batterij C5_COMBO" not in no_combo.logs


# --------------------------------------------------------------------------- #
# Het poortsample
# --------------------------------------------------------------------------- #
class TestReadGate:
    def test_the_read_is_registered_once_under_the_candidate_and_the_preregistration(
        self, chain: SimpleNamespace,
    ) -> None:
        g = chain.gate
        expected_id = f"robust_book_v1/{chain.pid}/{chain.summary['selected']}"
        lock = read_json(chain.root / programme.LOCK_PATH)
        assert g["hypothesis_id"] == expected_id and g["candidate"] == chain.summary["selected"]
        assert [r["hypothesis_id"] for r in lock["reads"]] == [expected_id]
        assert g["included"] == chain.summary["c5_included"]
        assert (lock["split_utc"], lock["git_sha"]) == (f"{GATE_START}T00:00:00+00:00", GIT)

    def test_the_gate_window_runs_from_the_split_to_the_last_bar(self, chain: SimpleNamespace) -> None:
        last = chain.market.index[-1]
        days = pd.date_range(utc(GATE_START), last, freq="D")
        assert chain.gate["gate_window"] == [GATE_START, str(last.date())]
        assert chain.gate["gate"]["n_obs"] == len(days)
        daily = pd.read_csv(chain.art / f"{chain.gate['candidate']}_daily_gate.csv", index_col=0)
        assert len(daily) == len(days) and daily.index[0][:10] == GATE_START

    def test_the_gate_figures_are_those_of_the_candidate_and_both_references_on_the_gate_window(
        self, chain: SimpleNamespace,
    ) -> None:
        g, last = chain.gate, chain.market.index[-1]
        window = (utc(GATE_START), last)

        def sharpe(name: str, included: tuple[str, ...] = ()) -> float:
            res = independent_run(chain.root, chain.market, name, included=included)
            return float(summarize(res.window(*window), bootstrap=False)["sharpe"])

        assert g["gate"]["sharpe"] == pytest.approx(sharpe(g["candidate"], tuple(g["included"])))
        for ref in programme.REFERENCES:
            assert g["references_on_gate"][ref]["sharpe"] == pytest.approx(sharpe(ref))

    def test_the_collapse_statistic_compares_the_gate_sharpe_with_the_dev_sharpe_in_standard_errors(
        self, chain: SimpleNamespace,
    ) -> None:
        g = chain.gate
        dev = read_json(chain.art / f"{g['candidate']}.json")["w_dev"]["sharpe"]
        assert g["w_dev_sharpe"] == dev
        z = (g["gate"]["sharpe"] - dev) / g["gate"]["sharpe_se"]
        assert g["gate_metrics"]["gate_sharpe_z_vs_dev"] == pytest.approx(z)
        assert g["gate_metrics"]["max_drawdown_gate"] == g["gate"]["max_drawdown"]
        prereg = require_preregistration(chain.pid, directory=chain.root / programme.PREREG_DIR)
        for name, gate in g["gate_gates"].items():
            crit = prereg.stop_criterion(name)
            assert gate["binds"] is (crit.binds(gate["value"]) or not math.isfinite(gate["value"]))
        assert g["gate_gates"]["gate_sharpe_collapse"]["value"] == g["gate_metrics"]["gate_sharpe_z_vs_dev"]
        assert g["gate_gates"]["gate_drawdown"]["value"] == g["gate"]["max_drawdown"]

    def test_the_verdict_follows_from_the_binding_criteria_and_the_record_is_written(
        self, chain: SimpleNamespace,
    ) -> None:
        g = chain.gate
        dev_binding = read_json(chain.art / f"{g['candidate']}.json")["n_binding_w_dev"]
        gate_binding = sum(x["binds"] for x in g["gate_gates"].values())
        assert g["n_binding_total"] == dev_binding + gate_binding
        if g["n_binding_total"] == 0:
            assert g["verdict"] == "promote_to_paper_trading"
        elif gate_binding:
            assert g["verdict"] == "falsified_on_gate"
        else:
            assert g["verdict"] == "archived_w_dev_gates"
        assert g[EVIDENCE_KEY] == NOT_ADMISSIBLE and "niet maagdelijk" in g["caveat"]
        assert json.dumps(read_json(chain.art / "gate_read.json"), sort_keys=True) == json.dumps(
            g, sort_keys=True, default=float)

    def test_the_log_line_names_the_candidate_its_gate_sharpe_and_the_verdict(self, chain: SimpleNamespace) -> None:
        g = chain.gate
        (line,) = chain.logs.gate
        assert line.startswith(f"poort {g['candidate']}: Sharpe {g['gate']['sharpe']:.3f} (SE ")
        assert line.endswith(f", {g['verdict']}")

    def test_a_second_reading_is_refused_and_leaves_lock_and_record_alone(
        self, chain: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install(monkeypatch, chain.market)
        lock_before = (chain.root / programme.LOCK_PATH).read_bytes()
        record_before = (chain.art / "gate_read.json").read_bytes()
        with pytest.raises(HoldoutAlreadyUsed, match="al gelezen"):
            programme.read_gate(chain.root, log=lambda _m: None)
        assert (chain.root / programme.LOCK_PATH).read_bytes() == lock_before
        assert (chain.art / "gate_read.json").read_bytes() == record_before

    def test_without_a_frozen_preregistration_the_gate_stays_unread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loads: list[Path] = []
        install(monkeypatch, flow_market(), loads=loads)
        root = write_root(tmp_path)
        with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
            programme.read_gate(root, log=lambda _m: None)
        assert loads == [] and read_json(root / programme.LOCK_PATH)["reads"] == []

    def test_without_a_w_dev_run_the_gate_stays_unread(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        loads: list[Path] = []
        install(monkeypatch, flow_market(), loads=loads)
        root = write_root(tmp_path)
        programme.freeze(root, git_sha=GIT)
        with pytest.raises(FileNotFoundError, match=r"programme_w_dev\.json"):
            programme.read_gate(root, log=lambda _m: None)
        assert loads == [] and read_json(root / programme.LOCK_PATH)["reads"] == []

    def test_a_w_dev_run_of_another_preregistration_is_refused_and_the_gate_stays_unread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loads: list[Path] = []
        install(monkeypatch, flow_market(), loads=loads)
        root, pid = stage_gate(tmp_path, flow_market())
        summary = read_json(root / programme.ARTEFACT_DIR / "programme_w_dev.json")
        summary["preregistration_id"] = "0" * len(pid)
        _write(root / programme.ARTEFACT_DIR / "programme_w_dev.json", json.dumps(summary))
        with pytest.raises(DataContractError, match="hoort bij een andere preregistratie"):
            programme.read_gate(root, log=lambda _m: None)
        assert loads == [] and read_json(root / programme.LOCK_PATH)["reads"] == []

    def test_a_full_run_that_does_not_reproduce_the_dev_sharpe_is_refused_but_the_read_stays_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        market = flow_market()
        install(monkeypatch, market)
        root, pid = stage_gate(tmp_path, market, sharpe_shift=1e-6)
        with pytest.raises(DataContractError, match="reproduceert de W_DEV-Sharpe niet: het poortsample lekte"):
            programme.read_gate(root, log=lambda _m: None)
        reads = read_json(root / programme.LOCK_PATH)["reads"]
        assert [r["hypothesis_id"] for r in reads] == [f"robust_book_v1/{pid}/C3_VOLMAN_CORE"]
        assert not (root / programme.ARTEFACT_DIR / "gate_read.json").exists()
        assert not (root / programme.ARTEFACT_DIR / "C3_VOLMAN_CORE_daily_gate.csv").exists()

    def test_a_difference_of_floating_point_noise_is_not_a_leak(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        market = flow_market()
        install(monkeypatch, market)
        root, _ = stage_gate(tmp_path, market, sharpe_shift=1e-12)
        out = programme.read_gate(root, log=lambda _m: None)
        assert out["candidate"] == "C3_VOLMAN_CORE" and (root / programme.ARTEFACT_DIR / "gate_read.json").is_file()


class TestTheVerdict:
    @pytest.mark.parametrize(
        ("dev_binding", "collapse_binds", "drawdown_binds", "verdict", "booked"),
        [
            pytest.param(0, False, False, "promote_to_paper_trading", "accepted", id="nothing-binds"),
            pytest.param(3, False, False, "archived_w_dev_gates", "archived", id="only-the-dev-gates-bind"),
            pytest.param(0, True, False, "falsified_on_gate", "falsified", id="one-gate-binds-alone"),
            pytest.param(2, False, True, "falsified_on_gate", "falsified", id="a-gate-beats-the-dev-gates"),
        ])
    def test_the_gate_reading_and_the_booking_follow_the_binding_criteria(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_binding: int, collapse_binds: bool,
        drawdown_binds: bool, verdict: str, booked: str,
    ) -> None:
        market = flow_market()
        install(monkeypatch, market)
        edit = set_thresholds(gate_sharpe_collapse=1e9 if collapse_binds else -1e9,
                              gate_drawdown=-1.0 if drawdown_binds else 1e9)
        root, pid = stage_gate(tmp_path, market, dev_binding=dev_binding, spec_edit=edit)

        out = programme.read_gate(root, log=lambda _m: None)
        assert math.isfinite(out["gate_metrics"]["gate_sharpe_z_vs_dev"])
        assert {k: g["binds"] for k, g in out["gate_gates"].items()} == {
            "gate_sharpe_collapse": collapse_binds, "gate_drawdown": drawdown_binds}
        assert out["n_binding_total"] == dev_binding + collapse_binds + drawdown_binds
        assert out["verdict"] == verdict

        programme.book_verdict(root, git_sha=GIT)
        entry = HypothesisLedger(root / programme.LEDGER_PATH).entries()[-1]
        assert entry["result"] == booked and entry["metrics"]["verdict"] == verdict

    def test_a_combination_is_rebuilt_from_its_members_on_the_full_market(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        market = flow_market()
        install(monkeypatch, market)
        members = ("C3_VOLMAN_CORE", "C4_CARRY_XS")
        edit = set_thresholds(gate_sharpe_collapse=-1e9, gate_drawdown=1e9)
        root, _ = stage_gate(tmp_path, market, name="C5_COMBO", included=members, spec_edit=edit)
        out = programme.read_gate(root, log=lambda _m: None)
        window = (utc(GATE_START), market.index[-1])

        def gate_sharpe(included: tuple[str, ...]) -> float:
            res = independent_run(root, market, "C5_COMBO", included=included)
            return float(summarize(res.window(*window), bootstrap=False)["sharpe"])

        assert out["candidate"] == "C5_COMBO" and out["included"] == list(members)
        assert out["gate"]["sharpe"] == pytest.approx(gate_sharpe(members))
        assert out["gate"]["sharpe"] != pytest.approx(gate_sharpe(("C4_CARRY_XS",)))
        assert out["verdict"] == "promote_to_paper_trading"

    def test_the_booking_is_a_zero_trial_amendment_of_the_programme_entry(self, chain: SimpleNamespace) -> None:
        ledger = HypothesisLedger(chain.root / programme.LEDGER_PATH)
        booking, verdict = ledger.entries()
        g = chain.gate
        assert ledger.total_n_hypotheses() == chain.cfg.planned_trials
        assert (verdict["unit"], verdict["n_trials"], verdict["wave"], verdict["market"]) == (
            "robust_book_v1_verdict", 0, 1, "crypto")
        assert verdict["amends"] == booking["config_hash"] == hash_config(
            programme.programme_parameters(chain.cfg))
        assert verdict["config_hash"] == booking["config_hash"]
        assert verdict["preregistration_id"] == chain.pid and verdict["git_sha"] == RUN_GIT
        assert verdict["data_hash"] == booking["data_hash"]
        assert verdict["result"] == {"promote_to_paper_trading": "accepted",
                                     "falsified_on_gate": "falsified"}.get(g["verdict"], "archived")
        assert verdict["metrics"] == {
            "selected": g["candidate"], "w_dev_sharpe": g["w_dev_sharpe"],
            "gate_sharpe": g["gate"]["sharpe"], "verdict": g["verdict"],
            "robustness_scores": chain.summary["robustness_scores"]}
        assert "Oordeel robuust boek v1" in verdict["notes"]

    def test_an_unknown_verdict_is_archived_not_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        market = flow_market()
        install(monkeypatch, market)
        root, pid = stage_gate(tmp_path, market)
        _write(root / programme.ARTEFACT_DIR / "gate_read.json", json.dumps({
            "verdict": "pending_review", "candidate": "C3_VOLMAN_CORE", "w_dev_sharpe": 0.5,
            "gate": {"sharpe": 0.25}}))
        programme.book_verdict(root)
        entry = HypothesisLedger(root / programme.LEDGER_PATH).entries()[-1]
        assert entry["result"] == "archived" and entry["git_sha"] == RUN_GIT

    def test_a_verdict_cannot_be_booked_for_a_programme_that_was_never_booked(self, tmp_path: Path) -> None:
        root = write_root(tmp_path)
        out = root / programme.ARTEFACT_DIR
        _write(out / "gate_read.json", json.dumps({
            "verdict": "promote_to_paper_trading", "candidate": "C3_VOLMAN_CORE", "w_dev_sharpe": 0.5,
            "gate": {"sharpe": 0.25}}))
        _write(out / "programme_w_dev.json", json.dumps({"robustness_scores": {}}))
        with pytest.raises(ValueError, match="Amendement wijst naar config_hash"):
            programme.book_verdict(root, git_sha=GIT)
        assert HypothesisLedger(root / programme.LEDGER_PATH).entries() == []


# --------------------------------------------------------------------------- #
# De batterij: de takken die de flowtoets niet raakt
# --------------------------------------------------------------------------- #
COSTS = CostSpec(taker_fee=5.5e-4, half_spread=1e-4, impact=None, aum_usd=1e5)
ONE_VARIANT = "max_scale_1.0"


def run_battery(name: str, included: tuple[str, ...]) -> SimpleNamespace:
    """De echte `programme.battery` op een markt van drie munten en een config met een kern van één munt.

    Een kern van één munt (BTC) is volgens het schema geldig en maakt de twee overslaan-takken
    bereikbaar: met de standaardkern (BTC en ETH) laat elk universum minstens één kernmunt over.
    Twee binnenstappen zijn verkleind omdat ze niet het onderwerp zijn: één trekking per ruis- en
    gatenreeks (`N_SEEDS`) en een parameterfamilie van één lid (de familie heeft zijn eigen toets,
    `test_systematic_programme.py::test_the_perturbation_family_matches_the_preregistration`).
    """
    cfg = RobustBookConfig.model_validate(small_config(len(programme.TRIALS)))
    cfg = cfg.model_copy(update={"core": cfg.core.model_copy(update={"assets": ("BTCUSDT",)})})
    market = synthetic_market(n_bars=220).subset(("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    window = (market.index[120], market.index[200])

    def one_variant(_name: str, config: RobustBookConfig, _included: tuple[str, ...]) -> dict[str, dict]:
        return {ONE_VARIANT: {"sizing": replace(SizingParams.from_config(config), max_scale=1.0)}}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(programme, "N_SEEDS", 1)
        mp.setattr(programme, "perturbation_family", one_variant)
        base = programme.run_targets(programme.build_trial(name, market, cfg, included=included),
                                     market, COSTS, lag=1)
        bat, family = programme.battery(name, market, cfg, COSTS, included=included, base=base, w_dev=window)
    return SimpleNamespace(bat=bat, family=family, base=base, market=market, cfg=cfg, window=window,
                           name=name, included=included)


def window_sharpe(res: BookResult, window: tuple[pd.Timestamp, pd.Timestamp]) -> float:
    return float(summarize(res.window(*window), bootstrap=False)["sharpe"])


ALL_FIVE = {"without_BTCUSDT", "without_ETHUSDT", "without_SOLUSDT", "btc_eth_only", "without_btc_alts_only"}
KEEPS_BTC = {"without_ETHUSDT", "without_SOLUSDT", "btc_eth_only"}


@pytest.fixture(scope="module")
def carry() -> SimpleNamespace:
    return run_battery("C4_CARRY_XS", ())


@pytest.fixture(scope="module")
def core() -> SimpleNamespace:
    return run_battery("C3_VOLMAN_CORE", ())


@pytest.fixture(scope="module")
def combo_with_core() -> SimpleNamespace:
    return run_battery("C5_COMBO", ("C3_VOLMAN_CORE", "C4_CARRY_XS"))


@pytest.fixture(scope="module")
def combo_without_core() -> SimpleNamespace:
    return run_battery("C5_COMBO", ("C4_CARRY_XS",))


class TestBatteryBranches:
    def test_the_carry_candidate_varies_its_own_rebalance_interval_not_the_thinned_decisions(
        self, carry: SimpleNamespace,
    ) -> None:
        m, cfg = carry.market, carry.cfg
        assert set(carry.bat["rebalance"]) == {"every_3", "every_14"}
        t_base = programme.build_trial(carry.name, m, cfg)
        for k in (3, 14):
            own = programme.build_trial(carry.name, m, cfg, carry_rebalance=k)
            expected = window_sharpe(programme.run_targets(own, m, COSTS, lag=1), carry.window)
            thinned = window_sharpe(programme.run_targets(programme._every(t_base, k), m, COSTS, lag=1),
                                    carry.window)
            assert carry.bat["rebalance"][f"every_{k}"] == pytest.approx(expected)
            assert carry.bat["rebalance"][f"every_{k}"] != pytest.approx(thinned)

    def test_a_candidate_without_a_core_keeps_every_universe_even_when_the_core_is_one_coin(
        self, carry: SimpleNamespace,
    ) -> None:
        assert set(carry.bat["universe"]) == ALL_FIVE

    def test_the_battery_delivers_every_figure_run_programme_reads(self, carry: SimpleNamespace) -> None:
        bat = carry.bat
        assert bat["base_sharpe"] == pytest.approx(window_sharpe(carry.base, carry.window))
        assert set(bat["costs"]) == {"x0", "x2", "x3", "plus_10bp_slippage"}
        assert set(bat["capacity"]) == {"aum_1000000", "aum_10000000", "aum_100000000"}
        assert set(bat["delay"]) == {"lag_2", "lag_3"}
        assert set(bat["perturbation"]) == {ONE_VARIANT} and set(carry.family) == {"base", ONE_VARIANT}
        assert bat["min_perturbation_sharpe"] == bat["median_perturbation_sharpe"] == bat["perturbation"][ONE_VARIANT]
        assert bat["plateau_fraction"] in (0.0, 1.0)
        assert 0.0 <= bat["leave_one_out_positive_fraction"] <= 1.0
        assert set(bat["monte_carlo_drawdown"]) == {"one_year", "full_window"}

    def test_the_noise_and_the_gaps_run_as_many_draws_as_the_seed_count_says(self, carry: SimpleNamespace) -> None:
        for key in ("signal_noise", "missing_data"):
            draw = carry.bat[key]
            assert draw["n_seeds"] == 1
            assert draw["min"] == draw["median"] == draw["max"]

    def test_a_core_candidate_skips_the_universes_that_leave_it_without_a_core_asset(
        self, core: SimpleNamespace,
    ) -> None:
        assert set(core.bat["universe"]) == KEEPS_BTC
        assert set(core.bat["rebalance"]) == {"every_3", "every_7"}

    def test_a_universe_runs_the_candidate_on_that_universe_only(self, core: SimpleNamespace) -> None:
        sub = core.market.subset(("BTCUSDT", "SOLUSDT"))
        run = programme.run_targets(programme.build_trial(core.name, sub, core.cfg), sub, COSTS, lag=1)
        assert core.bat["universe"]["without_ETHUSDT"] == pytest.approx(window_sharpe(run, core.window))

    def test_the_leave_one_out_fraction_counts_only_the_universes_that_were_run(self, core: SimpleNamespace) -> None:
        loo = [core.bat["universe"][k] > 0 for k in ("without_ETHUSDT", "without_SOLUSDT")]
        assert core.bat["leave_one_out_positive_fraction"] == pytest.approx(float(np.mean(loo)))

    def test_a_combination_with_the_core_sleeve_skips_the_same_universes(
        self, combo_with_core: SimpleNamespace,
    ) -> None:
        assert set(combo_with_core.bat["universe"]) == KEEPS_BTC

    def test_a_combination_without_the_core_sleeve_keeps_every_universe(
        self, combo_without_core: SimpleNamespace,
    ) -> None:
        assert set(combo_without_core.bat["universe"]) == ALL_FIVE


# --------------------------------------------------------------------------- #
# De opdrachtregel
# --------------------------------------------------------------------------- #
def _record_commands(mp: pytest.MonkeyPatch) -> list[str]:
    """Vervang de vier opdrachten door opnemers, zodat `main` nooit de echte repo-boom raakt."""
    calls: list[str] = []

    def freeze() -> Path:
        calls.append("freeze")
        return Path("artefacts/governance/preregistration_x.json")

    def run() -> dict[str, Any]:
        calls.append("run")
        return {"selected": "C2_TREND_LF", "score": np.float32(12.5), "included": ["C1_TREND_LS"]}

    def gate() -> dict[str, Any]:
        calls.append("gate")
        return {"verdict": "archived_w_dev_gates", "z": np.float32(-0.5)}

    def verdict() -> None:
        calls.append("verdict")

    mp.setattr(programme, "freeze", freeze)
    mp.setattr(programme, "run_programme", run)
    mp.setattr(programme, "read_gate", gate)
    mp.setattr(programme, "book_verdict", verdict)
    return calls


class TestMain:
    USAGE = "Gebruik: python -m tradebot.systematic.programme {freeze|run|gate|verdict}"

    @pytest.mark.parametrize("argv", [[], ["freeze", "now"], ["thaw"], ["RUN"], ["gate", "--force"], [""]])
    def test_a_wrong_command_line_is_refused_before_anything_runs(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str],
    ) -> None:
        calls = _record_commands(monkeypatch)
        with pytest.raises(DataContractError, match=re.escape(self.USAGE)):
            programme.main(argv)
        assert calls == []

    def test_freeze_prints_the_path_of_the_frozen_preregistration(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        calls = _record_commands(monkeypatch)
        programme.main(["freeze"])
        assert calls == ["freeze"]
        assert capsys.readouterr().out == "artefacts/governance/preregistration_x.json\n"

    def test_run_prints_the_summary_as_json_including_numpy_numbers(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        calls = _record_commands(monkeypatch)
        programme.main(["run"])
        assert calls == ["run"]
        assert json.loads(capsys.readouterr().out) == {
            "selected": "C2_TREND_LF", "score": 12.5, "included": ["C1_TREND_LS"]}

    def test_gate_prints_the_reading_as_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        calls = _record_commands(monkeypatch)
        programme.main(["gate"])
        assert calls == ["gate"]
        assert json.loads(capsys.readouterr().out) == {"verdict": "archived_w_dev_gates", "z": -0.5}

    def test_verdict_books_and_prints_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        calls = _record_commands(monkeypatch)
        programme.main(["verdict"])
        assert calls == ["verdict"] and capsys.readouterr().out == ""

    @pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
    def test_the_module_runs_as_a_script(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["programme", "thaw"])
        with pytest.raises(DataContractError, match="Gebruik"):
            runpy.run_module("tradebot.systematic.programme", run_name="__main__")
