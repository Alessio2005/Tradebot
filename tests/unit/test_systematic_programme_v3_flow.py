"""De v3-orkestratie van begin tot eind, met de v4-bevestigingslezing die in dezelfde module staat.

Net als `test_systematic_programme_v2_flow.py` draait alles op een tmp-wortel met een klein
synthetisch Binance-paneel (zeven munten, 225 dagen), een eigen config met korte vensters, en
een ledger met alleen een seed; de wortel en de hulpmiddelen komen uit dat bestand. De echte
`data/` en `artefacts/` worden niet gelezen of beschreven.

v3 verschilt van v2 in precies twee dingen en dat is wat hier wordt bewezen: de run draait met
een vaste halve spread en partiële aanpassing (kappa), en de batterij krijgt de v3-bouwer, de
v3-runner, de kappa-familie en de spreadstress mee. Per oordeel (promote, falsified, archived)
staat een eigen spec met drempels die het oordeel vastleggen, zodat de poorten van de
preregistratie het oordeel bepalen en niet het toeval van de data. Dezelfde drie oordelen
(bevestigd, weerlegd, niet bevestigd) staan voor de v4-bevestiging, die dezelfde holdout-slot
gebruikt als de v3-lezing maar onder een eigen hypothese-id.
"""
from __future__ import annotations

import functools
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.unit.test_systematic_programme_v2_flow import (
    HOLDOUT_END,
    HOLDOUT_START,
    LEDGER_SEED,
    PANELS,
    REPO,
    SHA,
    build_root,
    copy_root,
    gate_spec,
    hybrid_battery,
    ledger_entries,
    lock_reads,
    make_panels,
    read_json,
)
from tradebot.backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from tradebot.registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from tradebot.registry.preregistration import load_preregistration_spec
from tradebot.schemas.robust_book_v2 import robust_book_v3_config
from tradebot.systematic import programme_v2 as v2
from tradebot.systematic import programme_v3 as v3
from tradebot.systematic.evaluate import pbo_record
from tradebot.utils.failfast import DataContractError
from tradebot.validation.holdout import HoldoutAlreadyUsed, freeze_holdout

PROGRAMME_TRIALS = 3
KAPPAS = {"trade_rate_0.1": {"trade_rate": 0.1}, "trade_rate_0.33": {"trade_rate": 0.33},
          "trade_rate_1.0": {"trade_rate": 1.0}}
REAL_BATTERY_FOR = ("Y2_CARRY",)
HOLDOUT_GATES = ("gate_sharpe_collapse", "gate_drawdown")
PROMOTION = "promotion_requires_all_clear"
LOCK_SPLIT = f"{HOLDOUT_START}T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Geen enkele test roept `git` aan; de sha is een vaste waarde."""
    monkeypatch.setattr(v3, "current_git_sha", lambda: SHA)


def root_v3(path: Path, **kw: Any) -> Path:
    return build_root(path, version=3, **kw)


def freeze_lock(root: Path) -> Path:
    """v3 bevriest zelf geen slot: het slot is dat van v2 en bestaat al als v3 begint."""
    lock = root / v2.LOCK_PATH
    freeze_holdout(split_utc=LOCK_SPLIT, out=lock, git_sha=SHA)
    return lock


def freeze_and_run(root: Path, mp: pytest.MonkeyPatch, *, real_for: tuple[str, ...] = (),
                   calls: list | None = None) -> SimpleNamespace:
    """freeze + run_programme, met een verkleinde batterij en PBO."""
    mp.setattr(v3, "current_git_sha", lambda: SHA)
    mp.setattr(v2, "battery", hybrid_battery(v2.battery, [] if calls is None else calls, real_for))
    mp.setattr(v3, "pbo_record", functools.partial(pbo_record, n_subsets=4))
    mp.setattr(v2, "N_SEEDS", 1)
    frozen = v3.freeze(root, git_sha=SHA)
    freeze_lock(root)
    logs: list[str] = []
    summary = v3.run_programme(root, log=logs.append)
    return SimpleNamespace(root=root, frozen=frozen, summary=summary, logs=logs)


def light_battery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v2, "battery", hybrid_battery(v2.battery, []))
    monkeypatch.setattr(v3, "pbo_record", functools.partial(pbo_record, n_subsets=4))
    monkeypatch.setattr(v2, "N_SEEDS", 1)


@pytest.fixture
def frozen_root(tmp_path: Path) -> Path:
    """Een wortel waarvan het programma is geboekt en bevroren, nog niet gemeten."""
    root = root_v3(tmp_path / "root")
    v3.freeze(root, git_sha=SHA)
    freeze_lock(root)
    return root


SCENARIOS = {
    "promote": {"w_dev_binds": False, "holdout_gate": None},
    "falsified": {"w_dev_binds": False, "holdout_gate": "gate_sharpe_collapse"},
    "archived": {"w_dev_binds": True, "holdout_gate": None},
}


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory: pytest.TempPathFactory) -> dict[str, SimpleNamespace]:
    """Per oordeel een eigen bevroren spec, een W_DEV-run en één holdout-lezing.

    `pre_holdout` is een kopie van de wortel vóór die lezing, voor de tests die zelf lezen.
    """
    out = {}
    for name, knobs in SCENARIOS.items():
        base = tmp_path_factory.mktemp(f"v3_{name}")
        root = root_v3(base / "root", spec_edit=gate_spec(**knobs))
        calls: list = []
        with pytest.MonkeyPatch.context() as mp:
            flow = freeze_and_run(root, mp, real_for=REAL_BATTERY_FOR if name == "promote" else (),
                                  calls=calls)
            flow.pre_holdout = copy_root(root, base / "pre_holdout")
            flow.holdout_logs = []
            flow.holdout = v3.read_holdout(root, log=flow.holdout_logs.append)
        flow.calls = calls
        out[name] = flow
    return out


# --------------------------------------------------------------------------- #
# Bevriezen
# --------------------------------------------------------------------------- #
def test_freeze_books_the_three_trials_and_freezes_the_preregistration(tmp_path):
    root = root_v3(tmp_path / "root")
    path = v3.freeze(root, git_sha=SHA)

    (entry,) = ledger_entries(root)
    frozen = read_json(path)
    cfg = robust_book_v3_config(root / v3.CONFIG)
    assert path == root / v2.PREREG_DIR / f"preregistration_{entry['preregistration_id']}.json"
    assert (entry["wave"], entry["unit"], entry["market"], entry["n_trials"], entry["result"],
            entry["git_sha"]) == (3, v3.PROGRAMME_UNIT, "crypto", PROGRAMME_TRIALS, "interim", SHA)
    assert frozen["ledger_total_at_freeze"] == LEDGER_SEED + PROGRAMME_TRIALS
    assert frozen["git_sha"] == SHA and frozen["content"]["planned_trials"] == PROGRAMME_TRIALS
    assert frozen["content"]["parameters"] == json.loads(cfg.model_dump_json())
    assert [name for name, _ in frozen["content"]["data_hashes"]] == sorted(
        ["binance_um/objects", *(f"binance_um/panel/{p}" for p in PANELS)])


def test_freeze_leaves_the_holdout_lock_to_the_programme_that_owns_it(tmp_path):
    root = root_v3(tmp_path / "root")
    v3.freeze(root, git_sha=SHA)
    assert not (root / v2.LOCK_PATH).exists()


def test_freeze_takes_the_sha_from_the_working_tree_when_none_is_given(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "current_git_sha", lambda: "fedcba9")
    root = root_v3(tmp_path / "root")
    v3.freeze(root)
    assert ledger_entries(root)[0]["git_sha"] == "fedcba9"


def test_freezing_the_same_programme_twice_is_refused_and_books_nothing_new(frozen_root):
    with pytest.raises(DataContractError, match="Dit programma is al geboekt"):
        v3.freeze(frozen_root, git_sha=SHA)
    assert len(ledger_entries(frozen_root)) == 1


def test_freeze_refuses_when_the_preregistration_plans_another_number_of_trials(tmp_path):
    root = root_v3(tmp_path / "root", spec_edit=lambda spec: spec.update(planned_trials=4))
    with pytest.raises(DataContractError, match="planned_trials wijkt af"):
        v3.freeze(root, git_sha=SHA)
    assert ledger_entries(root) == []
    assert list((root / v2.PREREG_DIR).glob("preregistration_*.json")) == []


def test_freeze_does_not_create_a_missing_ledger(tmp_path):
    root = root_v3(tmp_path / "root")
    (root / v2.LEDGER_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="Ledger not found"):
        v3.freeze(root, git_sha=SHA)
    assert not (root / v2.LEDGER_PATH).exists()


# --------------------------------------------------------------------------- #
# De bouwstenen: kosten, parameters, kandidaten
# --------------------------------------------------------------------------- #
def test_the_candidates_are_the_v2_sleeves_with_the_combination_of_carry_and_trend():
    assert v3.CANDIDATES == ("Y2_CARRY", "Y4_TREND_LF", "Y5_COMBO")
    assert v3.SLEEVE_OF == {"Y2_CARRY": ("X2_XSCARRY", ()), "Y4_TREND_LF": ("X4_TREND_LF", ()),
                            "Y5_COMBO": ("X5_COMBO", ("X2_XSCARRY", "X4_TREND_LF"))}
    assert set(v3.HYPOTHESES) == set(v3.CANDIDATES)


def test_the_parameters_are_the_config_as_plain_json_types(tmp_path):
    root = root_v3(tmp_path / "root")
    cfg = robust_book_v3_config(root / v3.CONFIG)
    params = v3._params(cfg)
    assert params == json.loads(json.dumps(params))
    assert params["v3"]["trade_rate"] == 0.2 and params["planned_trials"] == PROGRAMME_TRIALS


def test_the_cost_spec_uses_the_fixed_spread_of_the_config_and_can_be_stressed(tmp_path):
    root = root_v3(tmp_path / "root")
    cfg = robust_book_v3_config(root / v3.CONFIG)
    base = v3.cost_spec(root, cfg)
    stress = v3.cost_spec(root, cfg, half_spread_bps=cfg.v3.half_spread_stress_bps)
    assert base.half_spread == pytest.approx(3.0e-4) and stress.half_spread == pytest.approx(6.0e-4)
    for costs in (base, stress):
        assert costs.taker_fee == pytest.approx(5.5e-4)
        assert costs.aum_usd == cfg.execution.aum_usd and costs.eta == cfg.costs.impact_y
        assert costs.impact is not None


# --------------------------------------------------------------------------- #
# run_programme: weigeringen
# --------------------------------------------------------------------------- #
def test_run_refuses_without_a_frozen_preregistration_and_writes_nothing(tmp_path):
    root = root_v3(tmp_path / "root")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v3.run_programme(root, log=lambda _: None)
    assert not (root / v3.ARTEFACT_DIR).exists()


def test_run_refuses_when_the_panels_changed_after_freezing(frozen_root):
    panels = make_panels()
    panels["close"] = panels["close"] * 1.0000001
    panels["close"].to_parquet(frozen_root / "data/binance_vision/panels/close.parquet")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v3.run_programme(frozen_root, log=lambda _: None)
    assert not (frozen_root / v3.ARTEFACT_DIR).exists()


def test_run_refuses_when_the_config_changed_after_freezing(frozen_root):
    path = frozen_root / v3.CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["v3"]["trade_rate"] = 0.25
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v3.run_programme(frozen_root, log=lambda _: None)
    assert not (frozen_root / v3.ARTEFACT_DIR).exists()


# --------------------------------------------------------------------------- #
# run_programme: de uitkomsten (scenario promote: geen enkele poort bindt)
# --------------------------------------------------------------------------- #
def _record(flow: SimpleNamespace, name: str) -> dict[str, Any]:
    return read_json(flow.root / v3.ARTEFACT_DIR / f"{name}.json")


def test_run_writes_a_record_and_a_daily_frame_per_candidate_and_one_summary(scenarios):
    flow = scenarios["promote"]
    out = flow.pre_holdout / v3.ARTEFACT_DIR
    assert {p.name for p in out.iterdir()} == (
        {f"{n}.json" for n in v3.CANDIDATES} | {f"{n}_daily_w_dev.csv" for n in v3.CANDIDATES}
        | {"programme_w_dev.json"})
    frame = pd.read_csv(out / "Y5_COMBO_daily_w_dev.csv", index_col=0, parse_dates=True)
    assert [str(frame.index[0].date()), str(frame.index[-1].date())] == ["2021-01-01", "2021-06-30"]
    assert {"gross", "funding", "fees", "spread", "slippage", "impact", "net"} <= set(frame)


def test_every_record_carries_its_provenance_and_stays_unadmissible_as_evidence(scenarios):
    flow = scenarios["promote"]
    cfg = robust_book_v3_config(flow.root / v3.CONFIG)
    for name in v3.CANDIDATES:
        rec = _record(flow, name)
        assert rec["trial"] == name and rec["hypothesis"] == v3.HYPOTHESES[name]
        assert rec["role"] == "candidate" and rec[EVIDENCE_KEY] == NOT_ADMISSIBLE
        assert (rec["preregistration_id"], rec["git_sha"]) == (
            flow.summary["preregistration_id"], SHA)
        assert rec["parameters"] == json.loads(cfg.model_dump_json())
        assert rec["windows"] == {"train": ["2021-01-01", "2021-04-30"],
                                  "validate": ["2021-05-01", "2021-06-30"],
                                  "holdout": [HOLDOUT_START, "unread"]}
        assert set(rec["data_hashes"]) == {"binance_um/objects",
                                           *(f"binance_um/panel/{p}" for p in PANELS)}
        assert {"w_dev", "train", "validate", "yearly", "regimes", "dsr", "battery", "pbo",
                "gates", "gate_metrics", "robustness_score", "n_forced_exits"} <= set(rec)
        assert set(rec["regimes"]) == {"bull", "bear", "high_vol", "low_vol", "high_corr",
                                       "low_corr"}


def test_the_costs_are_the_fixed_three_basis_points_and_the_trade_rate_is_the_configured_one(
        scenarios):
    for name in v3.CANDIDATES:
        execution = _record(scenarios["promote"], name)["execution"]
        assert execution["lag_bars"] == 1 and execution["trade_rate"] == 0.2
        assert execution["costs"]["half_spread_bps"] == pytest.approx(3.0)
        assert execution["costs"]["taker_fee_bps"] == pytest.approx(5.5)


def test_the_w_dev_run_leaves_the_holdout_unread(scenarios):
    pre = scenarios["promote"].pre_holdout
    assert read_json(pre / v2.LOCK_PATH)["reads"] == []
    assert not (pre / v3.ARTEFACT_DIR / "holdout_read.json").exists()
    for name in v3.CANDIDATES:
        csv = pd.read_csv(pre / v3.ARTEFACT_DIR / f"{name}_daily_w_dev.csv", index_col=0,
                          parse_dates=True)
        assert csv.index.max() < pd.Timestamp(HOLDOUT_START, tz="UTC")


def test_the_trial_counts_come_from_the_ledger_and_the_known_prior(scenarios):
    flow = scenarios["promote"]
    s = flow.summary
    m, wide = LEDGER_SEED + PROGRAMME_TRIALS, LEDGER_SEED + PROGRAMME_TRIALS + 52
    assert (s["m_programme"], s["m_including_known_prior"]) == (m, wide)
    dsr = _record(flow, "Y2_CARRY")["dsr"]
    assert set(dsr) == {f"m_{m}", f"m_{wide}"}
    assert dsr[f"m_{wide}"]["dsr"] <= dsr[f"m_{m}"]["dsr"]      # meer trials: strengere deflatie


def test_the_battery_gets_the_v3_builder_runner_family_kappa_and_the_spread_stress(scenarios):
    flow = scenarios["promote"]
    assert [name for name, _ in flow.calls] == list(v3.CANDIDATES)
    for name, kw in flow.calls:
        sleeve, included = v3.SLEEVE_OF[name]
        assert kw["builder"] is v3.build
        assert (kw["family_name"], kw["included"]) == (sleeve, included)
        assert kw["run_family"] == KAPPAS
        assert set(kw["extra_costs"]) == {"half_spread_stress"}
        assert kw["extra_costs"]["half_spread_stress"].half_spread == pytest.approx(6.0e-4)
        assert set(kw["alt_universes"]) == {"top_30", "top_100"}
        assert kw["w_dev"] == (pd.Timestamp("2021-01-01", tz="UTC"),
                               pd.Timestamp("2021-06-30", tz="UTC"))


def test_the_trials_and_the_alternative_universes_are_run_on_markets_that_stop_before_the_holdout(
        scenarios):
    """Niet alleen de cijfers maar ook de invoer: de holdout zit niet in de runs, ook al levert het
    paneel hem wel (het loopt door tot 13 augustus)."""
    cut = pd.Timestamp(HOLDOUT_START, tz="UTC")
    flow = scenarios["promote"]
    assert make_panels()["close"].index.max() >= cut and flow.calls
    for name, kw in flow.calls:
        assert kw["base"].frame.index.max() < cut, name
        for label, alt in kw["alt_universes"].items():
            assert alt.book.close.index.max() < cut, (name, label)


def test_the_runner_the_battery_gets_trades_at_the_configured_rate_unless_told_otherwise(
        scenarios):
    cfg = robust_book_v3_config(scenarios["promote"].root / v3.CONFIG)
    runner = dict(scenarios["promote"].calls)["Y2_CARRY"]["runner"]
    m = v3.load_breadth_market(scenarios["promote"].root / v2.PANEL_DIR, cfg).truncate(
        v2._ts(cfg.windows.holdout_start))
    costs = v3.cost_spec(scenarios["promote"].root, cfg)
    target = v3.build("Y2_CARRY", m, cfg)
    slow = runner(target, m, costs, lag=1)
    fast = runner(target, m, costs, lag=1, trade_rate=1.0)
    assert slow.frame["turnover"].sum() < fast.frame["turnover"].sum()
    assert float(slow.net.sum()) == float(
        v2.run(target, m, costs, lag=1, spread_panel=False, trade_rate=0.2).net.sum())


def test_the_real_battery_output_is_attached_to_the_candidate_it_ran_for(scenarios):
    rec = _record(scenarios["promote"], "Y2_CARRY")
    assert {"perturbation", "costs", "capacity", "delay", "signal_noise", "missing_data",
            "rebalance", "universe", "start_dates", "end_dates", "monte_carlo_drawdown",
            "plateau_fraction", "min_perturbation_sharpe",
            "universe_positive_fraction"} <= set(rec["battery"])
    assert set(KAPPAS) <= set(rec["battery"]["perturbation"])
    assert "half_spread_stress" in rec["battery"]["costs"]
    assert {k for k in rec["battery"]["perturbation"] if k.startswith("carry_window")}
    assert rec["pbo"]["n_subsets"] == 4 and 0.0 <= rec["pbo"]["pbo"] <= 1.0


def test_gates_and_score_follow_the_preregistered_metrics_of_the_candidate(scenarios):
    flow = scenarios["promote"]
    m_prog = LEDGER_SEED + PROGRAMME_TRIALS
    for name in v3.CANDIDATES:
        rec = _record(flow, name)
        assert set(rec["gates"]) == {"g1_sharpe_below_target", "g4_drawdown"}
        assert rec["gates"]["g1_sharpe_below_target"]["value"] == rec["w_dev"]["sharpe"]
        assert rec["gates"]["g4_drawdown"]["value"] == rec["w_dev"]["max_drawdown"]
        assert rec["n_binding_w_dev"] == sum(g["binds"] for g in rec["gates"].values()) == 0
        assert 0.0 <= rec["robustness_score"]["total"] <= 100.0
        m = rec["gate_metrics"]
        assert m["net_sharpe_train"] == rec["train"]["sharpe"]
        assert m["net_sharpe_validate"] == rec["validate"]["sharpe"]
        assert m["sharpe_ci_low_w_dev"] == rec["w_dev"]["sharpe_ci_low"]
        assert m["dsr_w_dev"] == rec["dsr"][f"m_{m_prog}"]["dsr"] and m["pbo"] == rec["pbo"]["pbo"]
        assert m["plateau_fraction"] == rec["battery"]["plateau_fraction"]
        assert m["min_perturbation_sharpe"] == rec["battery"]["min_perturbation_sharpe"]


def test_the_cost_and_delay_ratios_are_nan_when_the_sharpe_is_not_positive(scenarios):
    """Een verhouding tot een Sharpe van nul of lager zegt niets; ze moet dan niet-eindig zijn,
    zodat de poort bindt (zie `v2._gates`)."""
    positive = {}
    for name in v3.CANDIDATES:
        rec = _record(scenarios["promote"], name)
        m, bat, sharpe = rec["gate_metrics"], rec["battery"], rec["w_dev"]["sharpe"]
        positive[name] = sharpe > 0
        if sharpe > 0:
            assert m["sharpe_ratio_2x_cost_over_base"] == pytest.approx(bat["costs"]["x2"] / sharpe)
            assert m["sharpe_ratio_lag2_over_base"] == pytest.approx(bat["delay"]["lag_2"] / sharpe)
        else:
            assert np.isnan(m["sharpe_ratio_2x_cost_over_base"])
            assert np.isnan(m["sharpe_ratio_lag2_over_base"])
    # De synthetische markt geeft carry een negatieve en trend een positieve Sharpe: beide takken.
    assert positive == {"Y2_CARRY": False, "Y4_TREND_LF": True, "Y5_COMBO": True}


def test_the_candidate_with_the_highest_robustness_score_is_selected(scenarios):
    flow = scenarios["promote"]
    scores = {n: _record(flow, n)["robustness_score"]["total"] for n in v3.CANDIDATES}
    s = flow.summary
    assert s["robustness_scores"] == scores and s["selected"] == max(scores, key=scores.get)
    assert s["passes_all_w_dev_gates"] == {n: True for n in v3.CANDIDATES}
    assert set(s["table"]) == set(v3.CANDIDATES)
    assert s[EVIDENCE_KEY] == NOT_ADMISSIBLE and s["programme"] == "robust_book_v3"
    assert s == read_json(flow.root / v3.ARTEFACT_DIR / "programme_w_dev.json")
    assert flow.logs == ["run Y2_CARRY", "run Y4_TREND_LF", "run Y5_COMBO",
                         "batterij Y2_CARRY", "batterij Y4_TREND_LF", "batterij Y5_COMBO"]


def test_a_failing_w_dev_gate_is_reported_per_candidate_in_the_summary(scenarios):
    s = scenarios["archived"].summary
    assert s["passes_all_w_dev_gates"] == {n: False for n in v3.CANDIDATES}
    assert all(_record(scenarios["archived"], n)["n_binding_w_dev"] == 1 for n in v3.CANDIDATES)


def test_every_w_dev_gate_of_the_shipped_spec_reads_a_metric_the_programme_computes(scenarios):
    """Een typefout in een metriek van de echte spec zou die poort stilzwijgend laten vervallen."""
    flow = scenarios["promote"]
    w_dev_metrics = set(_record(flow, "Y2_CARRY")["gate_metrics"])
    holdout_metrics = set(flow.holdout["gate_metrics"])
    spec = yaml.safe_load((REPO / v3.SPEC_PATH).read_text(encoding="utf-8"))
    names = {c["name"] for c in spec["stop_criteria"]}
    assert set(HOLDOUT_GATES) | {PROMOTION} <= names
    for c in spec["stop_criteria"]:
        if c["name"] != PROMOTION:
            assert c["metric"] in (holdout_metrics if c["name"] in HOLDOUT_GATES
                                   else w_dev_metrics), c["name"]


def test_the_shipped_specs_plan_as_many_trials_as_the_shipped_programme_runs():
    cfg = robust_book_v3_config(REPO / v3.CONFIG)
    spec = load_preregistration_spec(REPO / v3.SPEC_PATH, data_hashes=(("x", "0"),),
                                     parameters={"p": 1})
    confirm = load_preregistration_spec(REPO / v3.CONFIRM_SPEC, data_hashes=(("x", "0"),),
                                        parameters={"p": 1})
    assert spec.planned_trials == cfg.planned_trials == len(v3.CANDIDATES)
    assert confirm.planned_trials == 1


# --------------------------------------------------------------------------- #
# De scenario's: de drempels van de preregistratie bepalen het oordeel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "verdict", "n_binding", "holdout_binds"), [
    ("promote", "promote_to_paper_trading", 0, 0),
    ("falsified", "falsified_on_holdout", 1, 1),
    ("archived", "archived_w_dev_gates", 1, 0)])
def test_the_verdict_follows_the_binding_gates_of_the_frozen_preregistration(
        scenarios, name, verdict, n_binding, holdout_binds):
    out = scenarios[name].holdout
    assert (out["verdict"], out["n_binding_total"]) == (verdict, n_binding)
    assert set(out["gates"]) == set(HOLDOUT_GATES)
    assert sum(g["binds"] for g in out["gates"].values()) == holdout_binds
    assert scenarios[name].holdout_logs[-1].endswith(verdict)


# --------------------------------------------------------------------------- #
# read_holdout
# --------------------------------------------------------------------------- #
def _hid(flow: SimpleNamespace, name: str | None = None) -> str:
    return f"robust_book_v3/{flow.summary['preregistration_id']}/{name or flow.summary['selected']}"


def test_the_holdout_read_is_registered_before_the_first_holdout_number_exists(
        scenarios, tmp_path, monkeypatch):
    flow = scenarios["promote"]
    root = copy_root(flow.pre_holdout, tmp_path / "root")
    registered_when_running = []
    real_run = v2.run

    def spy(*args, **kwargs):
        registered_when_running.append(lock_reads(root))
        return real_run(*args, **kwargs)
    monkeypatch.setattr(v2, "run", spy)
    out = v3.read_holdout(root, log=lambda _: None)
    assert out["hypothesis_id"] == _hid(flow)
    assert registered_when_running == [[_hid(flow)]]            # vóór de enige berekening, en één keer
    assert lock_reads(root) == [_hid(flow)]


def test_the_holdout_read_reports_the_selected_candidate_only(scenarios):
    flow = scenarios["promote"]
    out = flow.holdout
    name = flow.summary["selected"]
    assert out["candidate"] == name and out["hypothesis_id"] == _hid(flow)
    assert read_json(flow.root / v3.ARTEFACT_DIR / "holdout_read.json") == out
    assert sorted(p.name for p in (flow.root / v3.ARTEFACT_DIR).glob("*_daily_holdout.csv")) == [
        f"{name}_daily_holdout.csv"]
    assert out[EVIDENCE_KEY] == NOT_ADMISSIBLE and "trendfamilie" in out["caveat"]
    assert out["w_dev_sharpe"] == _record(flow, name)["w_dev"]["sharpe"]
    assert lock_reads(flow.root) == [_hid(flow)]
    assert flow.holdout_logs[-1].startswith(f"holdout {name}: Sharpe ")


def test_the_holdout_window_starts_at_the_frozen_split_and_ends_at_holdout_end(scenarios):
    flow = scenarios["promote"]
    out = flow.holdout
    # Het paneel loopt door tot 13 augustus; de config sluit de holdout op 10 augustus.
    assert out["holdout_window"] == [HOLDOUT_START, HOLDOUT_END]
    csv = pd.read_csv(flow.root / v3.ARTEFACT_DIR / f"{out['candidate']}_daily_holdout.csv",
                      index_col=0, parse_dates=True)
    assert [str(csv.index[0].date()), str(csv.index[-1].date())] == [HOLDOUT_START, HOLDOUT_END]
    assert out["holdout"]["n_obs"] == len(csv) == 41
    assert set(out["yearly"]) == {"2021"}


def test_the_holdout_gates_are_judged_on_the_z_score_and_the_holdout_drawdown(scenarios):
    flow = scenarios["promote"]
    out = flow.holdout
    z, mdd = out["gate_metrics"]["gate_sharpe_z_vs_dev"], out["gate_metrics"]["max_drawdown_gate"]
    expected_z = (out["holdout"]["sharpe"] - out["w_dev_sharpe"]) / out["holdout"]["sharpe_se"]
    assert z == pytest.approx(expected_z) and mdd == out["holdout"]["max_drawdown"]
    assert out["gates"]["gate_sharpe_collapse"]["value"] == z
    assert out["gates"]["gate_drawdown"]["value"] == mdd
    assert out["n_binding_total"] == _record(flow, out["candidate"])["n_binding_w_dev"] == 0


def test_a_second_read_of_the_same_hypothesis_is_refused_and_the_first_result_stays(
        scenarios, tmp_path):
    root = copy_root(scenarios["promote"].root, tmp_path / "root")
    before = (root / v3.ARTEFACT_DIR / "holdout_read.json").read_bytes()
    with pytest.raises(HoldoutAlreadyUsed, match="heeft het poortsample al gelezen"):
        v3.read_holdout(root, log=lambda _: None)
    assert (root / v3.ARTEFACT_DIR / "holdout_read.json").read_bytes() == before
    assert len(lock_reads(root)) == 1


def test_the_holdout_read_refuses_a_w_dev_run_of_another_preregistration_and_reads_nothing(
        scenarios, tmp_path):
    root = copy_root(scenarios["promote"].pre_holdout, tmp_path / "root")
    path = root / v3.ARTEFACT_DIR / "programme_w_dev.json"
    path.write_text(json.dumps({**read_json(path), "preregistration_id": "f" * 32}),
                    encoding="utf-8")
    with pytest.raises(DataContractError, match="hoort bij een andere preregistratie"):
        v3.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == []                               # de lezing is niet verbruikt
    assert not (root / v3.ARTEFACT_DIR / "holdout_read.json").exists()


def test_the_holdout_read_refuses_when_the_data_changed_since_the_freeze(scenarios, tmp_path):
    root = copy_root(scenarios["promote"].pre_holdout, tmp_path / "root")
    funding = root / "data/binance_vision/panels/funding.parquet"
    funding.write_bytes(funding.read_bytes() + b"x")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v3.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == []


def test_the_holdout_read_without_a_frozen_lock_is_refused(scenarios, tmp_path):
    root = copy_root(scenarios["promote"].pre_holdout, tmp_path / "root")
    (root / v2.LOCK_PATH).unlink()
    with pytest.raises(DataContractError, match="ontbreekt"):
        v3.read_holdout(root, log=lambda _: None)
    assert not (root / v3.ARTEFACT_DIR / "holdout_read.json").exists()


def test_the_holdout_read_stops_when_the_full_run_does_not_reproduce_the_w_dev_sharpe(
        scenarios, tmp_path):
    flow = scenarios["promote"]
    root = copy_root(flow.pre_holdout, tmp_path / "root")
    rec_path = root / v3.ARTEFACT_DIR / f"{flow.summary['selected']}.json"
    rec = read_json(rec_path)
    rec["w_dev"]["sharpe"] += 1e-6
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(DataContractError, match="de holdout lekte"):
        v3.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == [_hid(flow)]                     # de lezing staat er wel
    assert not (root / v3.ARTEFACT_DIR / "holdout_read.json").exists()


def test_the_combination_is_read_with_its_fixed_sleeves(scenarios, tmp_path):
    """Y5 heeft geen inclusieregel: de sleeves staan vast in `SLEEVE_OF`, niet in het W_DEV-record."""
    flow = scenarios["promote"]
    root = copy_root(flow.pre_holdout, tmp_path / "root")
    path = root / v3.ARTEFACT_DIR / "programme_w_dev.json"
    path.write_text(json.dumps({**read_json(path), "selected": "Y5_COMBO"}), encoding="utf-8")
    out = v3.read_holdout(root, log=lambda _: None)             # reproduceert de W_DEV-Sharpe van Y5
    assert out["candidate"] == "Y5_COMBO" and out["hypothesis_id"] == _hid(flow, "Y5_COMBO")
    assert out["w_dev_sharpe"] == _record(flow, "Y5_COMBO")["w_dev"]["sharpe"]


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
    v3.book_verdict(root)

    programme, verdict = ledger_entries(root)
    hold = flow.holdout
    assert HypothesisLedger(root / v2.LEDGER_PATH).total_n_hypotheses() == before  # telt nul trials
    assert (verdict["wave"], verdict["unit"], verdict["n_trials"], verdict["result"]) == (
        3, "robust_book_v3_verdict", 0, ledger_result)
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
    out = frozen_root / v3.ARTEFACT_DIR
    out.mkdir(parents=True)
    (out / "holdout_read.json").write_text(json.dumps(
        {"verdict": verdict, "candidate": "Y2_CARRY", "w_dev_sharpe": 1.5,
         "holdout": {"sharpe": 0.5}}), encoding="utf-8")
    (out / "programme_w_dev.json").write_text(json.dumps(
        {"robustness_scores": {"Y2_CARRY": 61.0}}), encoding="utf-8")
    v3.book_verdict(frozen_root)
    assert ledger_entries(frozen_root)[-1]["result"] == ledger_result


def test_a_verdict_cannot_be_booked_before_the_holdout_was_read(frozen_root):
    with pytest.raises(FileNotFoundError, match=r"holdout_read\.json"):
        v3.book_verdict(frozen_root)
    assert len(ledger_entries(frozen_root)) == 1


def test_a_verdict_on_a_programme_that_was_never_booked_is_refused(tmp_path):
    root = root_v3(tmp_path / "root")
    out = root / v3.ARTEFACT_DIR
    out.mkdir(parents=True)
    (out / "holdout_read.json").write_text(json.dumps(
        {"verdict": "archived_w_dev_gates", "candidate": "Y2_CARRY", "w_dev_sharpe": 1.0,
         "holdout": {"sharpe": 0.0}}), encoding="utf-8")
    (out / "programme_w_dev.json").write_text(json.dumps({"robustness_scores": {}}),
                                              encoding="utf-8")
    with pytest.raises(ValueError, match="Amendement wijst naar config_hash"):
        v3.book_verdict(root)
    assert ledger_entries(root) == []


# --------------------------------------------------------------------------- #
# v4: één bevestigingslezing van Y5_COMBO, ongewijzigd
# --------------------------------------------------------------------------- #
CONFIRM_SCENARIOS = {
    "confirmed": {"collapse": -1e9, "drawdown": 10.0, "no_positive": -1e9},
    "falsified": {"collapse": -1e9, "drawdown": -1.0, "no_positive": -1e9},
    "not_confirmed": {"collapse": -1e9, "drawdown": 10.0, "no_positive": 1e9},
}
CONFIRM_OUTCOME = {
    "confirmed": ("confirmed_paper_trade_candidate", 0, "accepted"),
    "falsified": ("falsified_on_holdout", 1, "falsified"),
    "not_confirmed": ("not_confirmed", 1, "archived"),
}


def edit_confirm_spec(root: Path, *, collapse: float, drawdown: float, no_positive: float) -> None:
    """Zet de drempels van de bevestigingsspec, zodat zij het oordeel vastleggen."""
    path = root / v3.CONFIRM_SPEC
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    thresholds = {"confirm_sharpe_collapse": collapse, "confirm_drawdown": drawdown,
                  "confirm_no_positive_sharpe": no_positive}
    for criterion in spec["stop_criteria"]:
        if criterion["name"] in thresholds:
            criterion["threshold"] = thresholds[criterion["name"]]
    path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")


@pytest.fixture(scope="module")
def confirms(tmp_path_factory: pytest.TempPathFactory,
             scenarios: dict[str, SimpleNamespace]) -> dict[str, SimpleNamespace]:
    """Per oordeel een kopie van de v3-wortel vóór de v3-lezing, met een eigen bevroren spec."""
    out = {}
    for name, knobs in CONFIRM_SCENARIOS.items():
        root = copy_root(scenarios["promote"].pre_holdout,
                         tmp_path_factory.mktemp(f"v4_{name}") / "root")
        edit_confirm_spec(root, **knobs)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(v3, "current_git_sha", lambda: SHA)
            frozen = v3.freeze_confirm(root, git_sha=SHA)
            logs: list[str] = []
            read = v3.read_confirm(root, log=logs.append)
        out[name] = SimpleNamespace(root=root, frozen=frozen, read=read, logs=logs)
    return out


def test_freeze_confirm_books_one_trial_and_freezes_the_candidate_with_the_parameters(tmp_path):
    root = root_v3(tmp_path / "root")
    path = v3.freeze_confirm(root, git_sha=SHA)

    (entry,) = ledger_entries(root)
    frozen = read_json(path)
    cfg = robust_book_v3_config(root / v3.CONFIG)
    assert (entry["wave"], entry["unit"], entry["n_trials"], entry["result"], entry["git_sha"]) == (
        4, "robust_book_v4_confirm", 1, "interim", SHA)
    assert frozen["ledger_total_at_freeze"] == LEDGER_SEED + 1
    assert frozen["content"]["planned_trials"] == 1
    assert frozen["content"]["parameters"] == {**json.loads(cfg.model_dump_json()),
                                               "confirm_candidate": "Y5_COMBO"}


def test_freeze_confirm_takes_the_sha_from_the_working_tree_when_none_is_given(tmp_path):
    root = root_v3(tmp_path / "root")
    v3.freeze_confirm(root)
    assert ledger_entries(root)[0]["git_sha"] == SHA


def test_the_same_confirmation_cannot_be_booked_twice(tmp_path):
    root = root_v3(tmp_path / "root")
    v3.freeze_confirm(root, git_sha=SHA)
    with pytest.raises(DataContractError, match="Deze bevestiging is al geboekt"):
        v3.freeze_confirm(root, git_sha=SHA)
    assert len(ledger_entries(root)) == 1


def test_the_confirmation_and_the_v3_programme_are_booked_independently(frozen_root):
    v3.freeze_confirm(frozen_root, git_sha=SHA)
    programme, confirm = ledger_entries(frozen_root)
    assert programme["preregistration_id"] != confirm["preregistration_id"]
    assert HypothesisLedger(frozen_root / v2.LEDGER_PATH).total_n_hypotheses() == (
        LEDGER_SEED + PROGRAMME_TRIALS + 1)


def test_the_confirmation_is_refused_before_it_is_frozen_and_reads_nothing(scenarios, tmp_path):
    root = copy_root(scenarios["promote"].pre_holdout, tmp_path / "root")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v3.read_confirm(root, log=lambda _: None)
    assert lock_reads(root) == [] and not (root / v3.CONFIRM_DIR).exists()


@pytest.mark.parametrize(("name", "outcome"), list(CONFIRM_OUTCOME.items()))
def test_the_confirmation_verdict_follows_the_binding_gates_of_its_frozen_preregistration(
        confirms, name, outcome):
    verdict, n_binding, _ = outcome
    out = confirms[name].read
    assert (out["verdict"], out["n_binding"]) == (verdict, n_binding)
    assert set(out["gates"]) == {"confirm_sharpe_collapse", "confirm_drawdown",
                                 "confirm_no_positive_sharpe"}
    assert sum(g["binds"] for g in out["gates"].values()) == n_binding
    assert confirms[name].logs[-1].endswith(verdict)


@pytest.mark.parametrize(("name", "outcome"), list(CONFIRM_OUTCOME.items()))
def test_the_confirmation_verdict_is_booked_as_a_zero_trial_amendment(confirms, name, outcome):
    _, _, ledger_result = outcome
    root = confirms[name].root
    confirm, verdict = ledger_entries(root)[-2:]
    assert (confirm["unit"], confirm["n_trials"], confirm["result"]) == (
        "robust_book_v4_confirm", 1, "interim")
    assert (verdict["wave"], verdict["unit"], verdict["n_trials"], verdict["result"]) == (
        4, "robust_book_v4_confirm_verdict", 0, ledger_result)
    assert verdict["amends"] == confirm["config_hash"]
    assert verdict["preregistration_id"] == confirm["preregistration_id"]
    assert verdict["git_sha"] == SHA
    assert verdict["metrics"] == {"candidate": "Y5_COMBO", "verdict": confirms[name].read["verdict"],
                                  "holdout_sharpe": confirms[name].read["holdout"]["sharpe"]}


def test_the_confirmation_reports_the_candidate_its_gates_and_its_two_parts(confirms):
    flow = confirms["confirmed"]
    out = flow.read
    assert out["candidate"] == "Y5_COMBO" and out["holdout_window"] == [HOLDOUT_START, HOLDOUT_END]
    assert out[EVIDENCE_KEY] == NOT_ADMISSIBLE and "carrydeel" in out["caveat"]
    assert set(out["gate_metrics"]) == {"gate_sharpe_z_vs_dev", "max_drawdown_gate",
                                        "holdout_net_sharpe"}
    assert out["gate_metrics"]["holdout_net_sharpe"] == out["holdout"]["sharpe"]
    assert set(out["decomposition_of_the_candidate"]) == {"carry", "trend"}
    for part in out["decomposition_of_the_candidate"].values():
        assert set(part) == {"sharpe", "cagr", "max_drawdown"}
    assert read_json(flow.root / v3.CONFIRM_DIR / "confirm_read.json") == out
    csv = pd.read_csv(flow.root / v3.CONFIRM_DIR / "Y5_COMBO_daily_holdout.csv", index_col=0,
                      parse_dates=True)
    assert len(csv) == out["holdout"]["n_obs"] == 41
    assert flow.logs[-1].startswith("bevestiging Y5_COMBO: Sharpe ")


def test_the_confirmation_registers_its_read_under_its_own_hypothesis(confirms):
    flow = confirms["confirmed"]
    prereg_id = flow.frozen.stem.removeprefix("preregistration_")
    assert flow.read["hypothesis_id"] == f"robust_book_v4/{prereg_id}/Y5_COMBO"
    assert lock_reads(flow.root) == [flow.read["hypothesis_id"]]


def test_the_confirmation_does_not_use_up_the_v3_read(confirms, tmp_path):
    """Het slot is gedeeld, de hypothese-id's niet: na de bevestiging kan v3 nog lezen."""
    root = copy_root(confirms["confirmed"].root, tmp_path / "root")
    out = v3.read_holdout(root, log=lambda _: None)
    assert out["hypothesis_id"].startswith("robust_book_v3/")
    assert [r.split("/")[0] for r in lock_reads(root)] == ["robust_book_v4", "robust_book_v3"]


def test_a_second_confirmation_read_is_refused_and_the_first_result_stays(confirms, tmp_path):
    root = copy_root(confirms["confirmed"].root, tmp_path / "root")
    before = (root / v3.CONFIRM_DIR / "confirm_read.json").read_bytes()
    with pytest.raises(HoldoutAlreadyUsed, match="heeft het poortsample al gelezen"):
        v3.read_confirm(root, log=lambda _: None)
    assert (root / v3.CONFIRM_DIR / "confirm_read.json").read_bytes() == before
    assert len(lock_reads(root)) == 1


def test_the_confirmation_stops_when_the_full_run_does_not_reproduce_the_w_dev_sharpe(
        scenarios, tmp_path):
    root = copy_root(scenarios["promote"].pre_holdout, tmp_path / "root")
    edit_confirm_spec(root, **CONFIRM_SCENARIOS["confirmed"])
    v3.freeze_confirm(root, git_sha=SHA)
    rec_path = root / v3.ARTEFACT_DIR / "Y5_COMBO.json"
    rec = read_json(rec_path)
    rec["w_dev"]["sharpe"] += 1e-6
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(DataContractError, match="de holdout lekte"):
        v3.read_confirm(root, log=lambda _: None)
    assert len(lock_reads(root)) == 1                            # de lezing staat er wel
    assert not (root / v3.CONFIRM_DIR / "confirm_read.json").exists()
    assert [e["unit"] for e in ledger_entries(root)] == [v3.PROGRAMME_UNIT, "robust_book_v4_confirm"]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Vervang de zes opdrachten, zodat `main` nooit de echte wortel van de repository raakt."""
    called: list[str] = []
    monkeypatch.setattr(v3, "freeze", lambda: called.append("freeze") or Path("pre_x.json"))
    monkeypatch.setattr(v3, "run_programme", lambda: called.append("run") or {
        "selected": "Y2", "score": np.float32(12.5)})
    monkeypatch.setattr(v3, "read_holdout", lambda: called.append("holdout") or {
        "verdict": "archived_w_dev_gates"})
    monkeypatch.setattr(v3, "book_verdict", lambda: called.append("verdict"))
    monkeypatch.setattr(v3, "freeze_confirm",
                        lambda: called.append("freeze-confirm") or Path("pre_y.json"))
    monkeypatch.setattr(v3, "read_confirm", lambda: called.append("confirm") or {
        "verdict": "not_confirmed"})
    return called


@pytest.mark.parametrize(("command", "printed"), [
    ("freeze", "pre_x.json\n"),
    ("run", json.dumps({"selected": "Y2", "score": 12.5}, indent=2) + "\n"),
    ("holdout", json.dumps({"verdict": "archived_w_dev_gates"}, indent=2) + "\n"),
    ("verdict", ""),
    ("freeze-confirm", "pre_y.json\n"),
    ("confirm", json.dumps({"verdict": "not_confirmed"}, indent=2) + "\n")])
def test_main_runs_exactly_the_requested_command_and_prints_its_result(
        commands, capsys, command, printed):
    v3.main([command])
    assert commands == [command]
    assert capsys.readouterr().out == printed


@pytest.mark.parametrize("argv", [[], ["thaw"], ["FREEZE"], ["freeze", "run"]])
def test_main_refuses_a_missing_unknown_or_double_command_before_running_anything(
        commands, argv):
    with pytest.raises(DataContractError,
                       match=r"Gebruik: python -I -m tradebot\.systematic\.programme_v3"):
        v3.main(argv)
    assert commands == []


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_runs_as_a_script_and_refuses_a_call_without_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["programme_v3"])
    with pytest.raises(DataContractError, match="Gebruik"):
        runpy.run_module("tradebot.systematic.programme_v3", run_name="__main__")


# --------------------------------------------------------------------------- #
# Bevindingen: gepind, niet aangepast (het onderzoeksgedrag is vooraf geregistreerd)
# --------------------------------------------------------------------------- #
def test_a_lock_with_another_split_than_the_configured_holdout_start_is_refused(
        scenarios, tmp_path):
    root = copy_root(scenarios["promote"].pre_holdout, tmp_path / "root")
    lock = root / v2.LOCK_PATH
    lock.write_text(json.dumps({**read_json(lock), "split_utc": "2021-06-01T00:00:00+00:00"}),
                    encoding="utf-8")
    with pytest.raises(DataContractError, match="split"):
        v3.read_holdout(root, log=lambda _: None)
    assert lock_reads(root) == []                               # geweigerd vóór de lezing wordt verbruikt


def test_the_dsr_trial_count_stays_the_one_of_the_freeze_when_the_ledger_grows(
        scenarios, tmp_path, monkeypatch):
    flow = scenarios["archived"]                                # de goedkoopste run
    root = copy_root(flow.pre_holdout, tmp_path / "root")
    frozen_total = read_json(flow.frozen)["ledger_total_at_freeze"]
    HypothesisLedger(root / v2.LEDGER_PATH).append(LedgerEntry.from_config(
        wave=4, unit="robust_book_v4_confirm", market="crypto", config={"p": 1}, git_sha=SHA,
        data_hash="0" * 16, preregistration_id="f" * 32, n_trials=1))
    light_battery(monkeypatch)
    summary = v3.run_programme(root, log=lambda _: None)
    assert summary["m_programme"] == frozen_total
