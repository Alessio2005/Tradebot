"""De v6-orkestratie op de synthetische markt: elke kandidaat bouwt, draait en scoort; de
selectie kiest de laagste hefboom die alles haalt; de preregistratie past bij de config;
het vooruit-sample weigert zonder genoeg data, VOORDAT er een lezing wordt geregistreerd."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.lookahead.test_breadth_causality import _cfg
from tests.lookahead.test_levered_carry_causality import _mm
from tests.unit.test_programme_v2_smoke import IMPACT
from tradebot.registry.preregistration import load_preregistration_spec
from tradebot.schemas.robust_book_v2 import robust_book_v6_config
from tradebot.systematic import programme_v2 as v2
from tradebot.systematic.book import CostSpec
from tradebot.systematic.evaluate import pbo_record
from tradebot.systematic.harvest import BasisCosts
from tradebot.systematic.programme_v6 import (
    CANDIDATES,
    CONFIG,
    ROOT,
    SPEC_PATH,
    _family,
    _read,
    battery,
    financing,
    forward_window_ready,
    margin_spec,
    select,
    simulate,
    summary,
)
from tradebot.utils.failfast import DataContractError

PERP = CostSpec(taker_fee=5.5e-4, half_spread=3e-4, impact=IMPACT, aum_usd=1e6, impact_eta=1.0)
SPOT = CostSpec(taker_fee=10e-4, half_spread=5e-4, impact=IMPACT, aum_usd=1e6, impact_eta=1.0)
COSTS = BasisCosts(perp=PERP, spot=SPOT)


def _v6():
    cfg = robust_book_v6_config()
    # Synthetische funding is ~7 %/jaar: lagere drempels, zodat er iets gehouden wordt.
    return cfg.model_copy(update={
        "universe": _cfg().universe,
        "basis": cfg.basis.model_copy(update={"enter_apr": 0.05, "exit_apr": 0.0,
                                              "min_spot_adv_usd": 1e5}),
        "leverage": cfg.leverage.model_copy(update={"majors": ("BTCUSDT",)}),
    })


@pytest.fixture(scope="module")
def setup():
    m = _mm()
    return m, _v6(), (m.index[200], m.index[590])


@pytest.mark.parametrize("name", CANDIDATES)
def test_every_v6_candidate_builds_runs_and_stays_inside_the_mandate(setup, name):
    m, cfg, (a, b) = setup
    res = simulate(name, m, cfg, COSTS).window(a, b)
    s = summary(res, bootstrap=False)
    assert np.isfinite(s["sharpe"]) and s["avg_gross_leverage"] > 0.0
    assert abs(s["avg_net_leverage"]) < 0.01, "long spot, short perp: netto nul"
    traded = res.frame["turnover"] > 0
    assert (res.frame.loc[traded, "gross_leverage"] <= cfg.leverage.gross_cap + 1e-9).all()
    for key in ("ann_financing", "ann_liquidation", "n_liquidations", "min_stress_headroom",
                "avg_loan", "n_governed"):
        assert key in s, key


def test_more_leverage_means_more_loan_and_more_financing(setup):
    m, cfg, (a, b) = setup
    lo = summary(simulate("K1_CARRY_PM_1X", m, cfg, COSTS).window(a, b), bootstrap=False)
    hi = summary(simulate("K3_CARRY_PM_2X", m, cfg, COSTS).window(a, b), bootstrap=False)
    assert hi["avg_loan"] > lo["avg_loan"]
    assert hi["ann_financing"] < lo["ann_financing"] <= 0.0


def test_overrides_change_the_book(setup):
    m, cfg, (a, b) = setup
    base = simulate("K2_CARRY_PM_1P5X", m, cfg, COSTS).window(a, b).net
    for ov in ({"slots": 2}, {"every": 7}, {"missing_seed": 1}, {"noise_seed": 1},
               {"financing": "stress"}):
        other = simulate("K2_CARRY_PM_1P5X", m, cfg, COSTS, ov=ov).window(a, b).net
        assert not np.allclose(base, other), ov
    # De margestress verandert de P&L niet zolang niets liquideert of de governor bindt;
    # wel hoe dicht het boek bij liquidatie staat.
    loose = summary(simulate("K2_CARRY_PM_1P5X", m, cfg, COSTS).window(a, b), bootstrap=False)
    tight = summary(simulate("K2_CARRY_PM_1P5X", m, cfg, COSTS, ov={"margin_stress": True})
                    .window(a, b), bootstrap=False)
    assert tight["min_stress_headroom"] < loose["min_stress_headroom"]
    assert tight["min_uni_mmr"] < loose["min_uni_mmr"]


def test_financing_modes(setup):
    m, cfg, _ = setup
    base, stress = financing(m.basis, cfg, "base"), financing(m.basis, cfg, "stress")
    assert (stress >= base).all() and (base >= cfg.leverage.financing_floor_apr).all()
    assert (financing(m.basis, cfg, "optimistic") == cfg.leverage.financing_optimistic_apr).all()
    with pytest.raises(DataContractError):
        financing(m.basis, cfg, "free")
    assert margin_spec(cfg, stress=True).haircut_alt > margin_spec(cfg).haircut_alt


def test_unknown_candidate_is_refused(setup):
    m, cfg, _ = setup
    with pytest.raises(DataContractError):
        simulate("K9_NOPE", m, cfg, COSTS)


@pytest.mark.parametrize("name", ["K3_CARRY_PM_2X"])
def test_v6_battery_covers_every_check(setup, name):
    m, cfg, (a, b) = setup
    base = simulate(name, m, cfg, COSTS)
    stress = BasisCosts(perp=PERP, spot=SPOT.scaled(multiplier=2.0))
    bat, family = battery(name, m, cfg, COSTS, stress, base=base, w_dev=(a, b),
                          alt={"same": m})
    for key in ("perturbation", "costs", "costs_x2_cagr", "capacity", "delay", "signal_noise",
                "missing_data", "rebalance", "universe", "start_dates", "end_dates",
                "monte_carlo_drawdown", "financing_stress", "financing_optimistic",
                "margin_stress", "governor_off"):
        assert key in bat, key
    assert set(bat["perturbation"]) == set(_family())
    assert bat["financing_stress"]["cagr"] <= bat["financing_optimistic"]["cagr"] + 1e-12 or \
        bat["financing_stress"]["avg_loan"] == 0.0
    assert 0.0 <= pbo_record(family)["pbo"] <= 1.0


def _rec(n_binding: int, score: float) -> dict:
    return {"n_binding_w_dev": n_binding, "robustness_score": {"total": score}}


def test_selection_takes_the_lowest_leverage_that_passes_everything():
    lev = {"K1_CARRY_PM_1X": 1.0, "K2_CARRY_PM_1P5X": 1.5, "K3_CARRY_PM_2X": 2.0}
    recs = {"K1_CARRY_PM_1X": _rec(1, 99.0), "K2_CARRY_PM_1P5X": _rec(0, 80.0),
            "K3_CARRY_PM_2X": _rec(0, 90.0)}
    assert select(recs, lev) == ("K2_CARRY_PM_1P5X", "lowest_leverage_all_clear")
    none = {k: _rec(1, s) for k, s in (("K1_CARRY_PM_1X", 70.0), ("K2_CARRY_PM_1P5X", 95.0),
                                       ("K3_CARRY_PM_2X", 90.0))}
    assert select(none, lev) == ("K2_CARRY_PM_1P5X", "robustness_score_none_clear")


def test_the_preregistration_matches_the_config_and_the_programme_metrics():
    cfg = robust_book_v6_config(ROOT / CONFIG)
    spec = load_preregistration_spec(ROOT / SPEC_PATH, data_hashes=(("x", "0"),),
                                     parameters={"planned": cfg.planned_trials})
    assert spec.planned_trials == cfg.planned_trials == len(CANDIDATES)
    assert tuple(cfg.leverage.candidates) == CANDIDATES
    metrics = {c.metric for c in spec.stop_criteria}
    # Elke metriek die het programma uitrekent, heeft een criterium -- en omgekeerd.
    w_dev = {"net_sharpe_w_dev", "net_cagr_w_dev", "net_sharpe_train", "net_sharpe_validate",
             "max_drawdown_w_dev", "sharpe_ratio_2x_cost_over_base",
             "sharpe_ratio_lag2_over_base", "plateau_fraction", "min_perturbation_sharpe",
             "sharpe_ci_low_w_dev", "dsr_w_dev", "pbo", "n_liquidations_w_dev",
             "n_liquidations_margin_stress_w_dev", "net_sharpe_financing_stress_w_dev",
             "net_cagr_financing_stress_w_dev"}
    oos = {"backcast_sharpe_z_vs_dev", "backcast_max_drawdown", "backcast_n_liquidations",
           "backcast_net_sharpe", "holdout_sharpe_z_vs_dev", "holdout_max_drawdown",
           "holdout_n_liquidations", "forward_sharpe_z_vs_dev", "forward_max_drawdown",
           "forward_n_liquidations", "forward_net_sharpe"}
    assert metrics == w_dev | oos | {"n_binding_stop_criteria"}
    # G1-G9 zijn letterlijk die van v5: geen drempel versoepeld.
    v5 = load_preregistration_spec(ROOT / "conf/research/preregistration_robust_book_v5.yaml",
                                   data_hashes=(("x", "0"),), parameters={"p": 1})
    old = {c.name: (c.metric, c.operator, c.threshold) for c in v5.stop_criteria
           if c.name.startswith("g")}
    new = {c.name: (c.metric, c.operator, c.threshold) for c in spec.stop_criteria}
    assert all(new[k] == v for k, v in old.items())


def test_a_flat_window_reads_as_flat(setup):
    m, cfg, (a, b) = setup
    from tradebot.systematic.leverage import run_levered
    flat = pd.DataFrame(0.0, index=m.index, columns=list(m.symbols))
    res = run_levered(flat, m, COSTS, lag=1, band=0.5, hedge_tolerance=0.02,
                      margin=margin_spec(cfg), financing=financing(m.basis, cfg))
    r = _read(res, a, b, dev_sr=2.0)
    assert r["invested"] is False and np.isnan(r["z_vs_dev"]) and r["n_liquidations"] == 0
    spec = load_preregistration_spec(ROOT / SPEC_PATH, data_hashes=(("x", "0"),),
                                     parameters={"p": 1})
    gates = v2._gates({"holdout_sharpe_z_vs_dev": r["z_vs_dev"]}, spec)
    assert gates["holdout_sharpe_collapse"]["binds"] is True


def test_the_forward_window_needs_six_full_months():
    idx = pd.date_range("2020-01-02", "2027-03-31", freq="D", tz="UTC")
    ok, need = forward_window_ready(idx, "2026-10-01", 6)
    assert not ok and need == pd.Timestamp("2027-04-01", tz="UTC")
    ok, _ = forward_window_ready(idx.append(pd.DatetimeIndex(
        [pd.Timestamp("2027-04-01", tz="UTC")])), "2026-10-01", 6)
    assert ok


def test_the_forward_read_refuses_without_panels_and_registers_nothing(tmp_path):
    from tradebot.systematic.programme_v6 import read_forward

    lock = ROOT / "artefacts/governance/holdout_lock_forward_2026_10.json"
    before = lock.read_text(encoding="utf-8") if lock.exists() else None
    summary_w = ROOT / "artefacts/research/robust_book_v6/programme_w_dev.json"
    if not summary_w.exists():
        pytest.skip("de W_DEV-run van v6 bestaat nog niet")
    with pytest.raises(DataContractError):
        read_forward(ROOT, panel_root=tmp_path, log=lambda _: None)
    after = lock.read_text(encoding="utf-8") if lock.exists() else None
    assert before == after
    if after is not None:
        assert json.loads(after)["reads"] == json.loads(before)["reads"]
