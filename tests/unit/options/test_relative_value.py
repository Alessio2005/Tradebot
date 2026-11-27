"""Relative value op de VRP: z-scores, herweging en de engine-koppeling."""
from __future__ import annotations

import math
from datetime import date

import pytest

from tests.unit.options.test_engine_service import Rig
from tradebot.options.positions import Candidate
from tradebot.options.relative_value import rv_adjust, vrp_zscores
from tradebot.options.vol_regime import RVForecast, VolRegime, VolState


def vs(iv: float | None, rv: float | None = 0.20) -> VolState:
    fc = None if rv is None else RVForecast(rv, rv, rv, rv)
    return VolState(VolRegime.NORMAL, iv, fc, None, "t")


def cand(credit: float = 0.5, rank: float = 2.0, und: str = "A") -> Candidate:
    return Candidate("credit_spread", und, (), 1, credit, 100.0, rank, ("t",))


def test_zscores_are_population_scores_over_known_names():
    z = vrp_zscores({"A": vs(0.20), "B": vs(0.30), "C": vs(0.40)})
    assert z["B"] == pytest.approx(0.0)
    assert z["C"] == pytest.approx(math.sqrt(1.5)) and z["A"] == pytest.approx(-z["C"])


def test_names_without_iv_or_rv_get_zero_and_stay_out_of_the_distribution():
    z = vrp_zscores({"A": vs(0.20), "B": vs(0.30), "C": vs(0.40), "D": vs(None), "E": vs(0.5, None)})
    assert z["D"] == 0.0 and z["E"] == 0.0
    assert z["C"] == pytest.approx(math.sqrt(1.5))


def test_fewer_than_three_names_or_zero_spread_gives_zero():
    assert vrp_zscores({"A": vs(0.2), "B": vs(0.4)}) == {"A": 0.0, "B": 0.0}
    assert set(vrp_zscores({"A": vs(0.3), "B": vs(0.3), "C": vs(0.3)}).values()) == {0.0}


def test_credit_is_boosted_by_rich_vrp_and_debit_is_cut():
    up = rv_adjust(cand(0.5), 1.0, 0.2)
    assert up.rank_score == pytest.approx(2.4) and up.rule_trace == ("t", "rv_z:1.00")
    assert rv_adjust(cand(-0.5), 1.0, 0.2).rank_score == pytest.approx(1.6)


def test_the_factor_is_clamped():
    assert rv_adjust(cand(), 10.0, 1.0).rank_score == pytest.approx(3.0)
    assert rv_adjust(cand(), -10.0, 1.0).rank_score == pytest.approx(1.0)


def test_engine_reranks_only_with_a_weight(tmp_path, monkeypatch):
    from tests.unit.options.test_engine_service import cfg_with
    from tradebot.options.engine import service
    from tradebot.options.selection import Admission

    monkeypatch.setattr(service, "admit", lambda c, *a: Admission(c, True))
    on = Rig(tmp_path / "on", cfg_with("relative_value", weight=0.5)).e
    off = Rig(tmp_path / "off").e
    for e in (on, off):
        e.chains = {"A": None, "B": None}
    cands = [cand(0.5, 2.0, "A"), cand(0.5, 1.8, "B")]
    today = date(2026, 12, 14)
    same = off._admitted(cands, {}, {}, today)
    assert same == cands and not any("rv_z" in t for c in same for t in c.rule_trace)
    out = on._admitted(cands, {}, {}, today, {"A": -1.0, "B": 1.0})
    assert [c.underlying for c in out] == ["B", "A"]
    assert out[0].rule_trace[-1] == "rv_z:1.00"


def test_the_scan_adds_the_trace_only_with_a_weight(tmp_path):
    from tests.unit.options.test_engine_service import D0, at, cfg_with
    on = Rig(tmp_path / "on", cfg_with("relative_value", weight=0.5))
    off = Rig(tmp_path / "off")
    for r in (on, off):
        r.run(at(D0, 10, 1))
    (c_on,), (c_off,) = on.events("candidate"), off.events("candidate")
    assert c_on["trace"][-1] == "rv_z:0.00"
    assert not any(t.startswith("rv_z") for t in c_off["trace"])
