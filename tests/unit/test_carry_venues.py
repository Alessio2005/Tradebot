"""Beurzen voor het carryboek (`carry/venues.py`): profielen, toegang, kosten, overlay."""
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

from tests.unit.test_programme_v6_smoke import COSTS
from tradebot.carry import venues as vn
from tradebot.carry.venues import BookNeeds, basis_costs, fee_overlay, load_venues
from tradebot.utils.failfast import ConfigContractError, DataContractError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def venues():
    return load_venues(ROOT / vn.VENUES_PATH)


def test_the_profiles_say_who_can_host_a_carry_book(venues):
    kraken, bybit, binance = venues["kraken_eu"], venues["bybit_eu"], venues["binance"]
    assert kraken.available_nl and kraken.perps_available_nl
    assert bybit.available_nl and not bybit.perps_available_nl
    assert not binance.available_nl
    assert kraken.retail_max_perp_leverage == 2.0 and not kraken.unified_margin
    assert kraken.collateral_assets is not None and {"BTC", "ETH"} <= kraken.collateral_assets
    assert binance.collateral_assets is None
    assert all(s for v in venues.values() for s in v.sources)


def test_fees_follow_the_tier_a_book_reaches(venues):
    k = venues["kraken_eu"]
    assert k.spot_fee_bps(0.0, maker=False) == 80.0
    assert k.spot_fee_bps(2_499.0, maker=True) == 40.0
    assert k.spot_fee_bps(1e6, maker=False) == 22.0 and k.spot_fee_bps(1e6, maker=True) == 10.0
    assert k.perp_fee_bps(maker=False) == 5.0 and k.perp_fee_bps(maker=True) == 2.0
    with pytest.raises(DataContractError, match="geen perps"):
        venues["bybit_eu"].perp_fee_bps(maker=False)


def test_the_blockers_name_every_reason(venues):
    k = venues["kraken_eu"]
    unified_alts = BookNeeds(1.25, True, ("BTC", "ETH", "SOL"), 2.0)
    msgs = k.blockers(unified_alts)
    assert any("unified margin" in m for m in msgs)
    assert any("SOL" in m and "BTC" not in m for m in msgs)
    two_wallet = BookNeeds(0.70 / 0.30, False, (), 0.70)
    assert any("0.67x" in m for m in k.blockers(two_wallet))
    assert k.blockers(BookNeeds(1.25, False, ("BTC", "ETH"), 1.0)) == []
    assert venues["binance"].blockers(BookNeeds(1.0, True, ("BTC",), 1.0))[0].startswith(
        "niet beschikbaar")
    assert BookNeeds(1.0, False, (), 1.0).max_leg_at(2.0) == pytest.approx(2.0 / 3.0)


def test_basis_costs_take_the_venue_fees_and_keep_the_rest(venues):
    c = basis_costs(COSTS, venues["kraken_eu"], capital_usd=1e6, maker=False)
    assert c.spot.taker_fee == pytest.approx(22e-4) and c.perp.taker_fee == pytest.approx(5e-4)
    assert c.spot.half_spread == COSTS.spot.half_spread and c.perp.aum_usd == COSTS.perp.aum_usd


def test_the_fee_overlay_is_exact_on_the_frame_identity():
    frame = pd.DataFrame({"net": [0.001, -0.002], "fees": [-1.55e-4, 0.0]})
    assert fee_overlay(frame, vn.FROZEN_PAIR_FEE_BPS).tolist() == frame["net"].tolist()
    doubled = fee_overlay(frame, 2 * vn.FROZEN_PAIR_FEE_BPS)
    assert doubled.tolist() == pytest.approx([0.001 - 1.55e-4, -0.002])
    with pytest.raises(DataContractError, match="net en fees"):
        fee_overlay(frame[["net"]], 10.0)


def test_the_comparison_on_the_frozen_books(venues):
    out = vn.compare(ROOT, venues)
    assert set(out["books"]) == set(vn.BOOKS)
    t2 = out["books"]["T2_CARRY_TRANCHE_JUMPCAP"]
    assert t2["binance_model"]["cagr"] == pytest.approx(0.1336, abs=5e-4)
    assert t2["kraken_entry_taker"]["cagr"] < t2["kraken_1m_taker"]["cagr"] < t2[
        "binance_model"]["cagr"] < t2["kraken_1m_maker"]["cagr"]
    assert t2["blockers"]["kraken_eu"] and t2["blockers"]["binance"]
    assert out["scenarios"]["kraken_entry_taker"]["pair_fee_bps"] == 85.0
    assert set(t2["binance_model"]["yearly_return"]) == {str(y) for y in range(2020, 2027)}
    # De enige bevroren regelset die op Kraken EU kan: H3 met de spot als onderpand.
    hosted = [k for k, rec in out["books"].items() if not rec["blockers"]["kraken_eu"]]
    assert hosted == ["H3_BASIS_MAJORS_COLLATERAL"]
    h3, h3c = out["books"]["H3_BASIS_MAJORS"], out["books"]["H3_BASIS_MAJORS_COLLATERAL"]
    assert h3["kraken_1m_taker"] == h3c["kraken_1m_taker"]


def test_the_committed_comparison_is_current(venues):
    committed = json.loads((ROOT / vn.OUT).read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(vn.compare(ROOT, venues), sort_keys=True))
    flat = lambda d: pd.json_normalize(d, sep="/").iloc[0].to_dict()  # noqa: E731
    a, b = flat(committed), flat(fresh)
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float):
            assert np.isclose(a[k], b[k], rtol=1e-12, atol=1e-15), k
        else:
            assert a[k] == b[k], k


def _bad(tmp_path: Path, mutate) -> Path:
    doc = yaml.safe_load((ROOT / vn.VENUES_PATH).read_text(encoding="utf-8"))
    mutate(doc["venues"]["kraken_eu"])
    path = tmp_path / "venues.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_a_broken_profile_is_refused(tmp_path: Path):
    with pytest.raises(ConfigContractError, match="spot-tiers"):
        load_venues(_bad(tmp_path, lambda v: v["spot_tiers"].reverse()))
    with pytest.raises(ConfigContractError, match="perp-fees"):
        load_venues(_bad(tmp_path, lambda v: v.update(perp_taker_bps=None)))


def test_main_writes_the_artefact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    (tmp_path / "conf/execution").mkdir(parents=True)
    shutil.copy(ROOT / vn.VENUES_PATH, tmp_path / vn.VENUES_PATH)
    for prog, _, _ in vn.BOOKS.values():
        shutil.copytree(ROOT / "artefacts/research" / prog, tmp_path / "artefacts/research" / prog,
                        dirs_exist_ok=True)
    monkeypatch.setattr(vn, "ROOT", tmp_path)
    out = vn.main([])
    assert json.loads((tmp_path / vn.OUT).read_text(encoding="utf-8"))["scenarios"] == \
        json.loads(json.dumps(out["scenarios"]))
    assert "boeken" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_refuses_arguments_as_a_script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["venues", "x"])
    with pytest.raises(DataContractError, match="Gebruik"):
        runpy.run_module("tradebot.carry.venues", run_name="__main__")
