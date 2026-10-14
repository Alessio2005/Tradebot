"""De v6-orkestratie (het generieke `Programme`) van begin tot eind, op de kleine synthetische
wereld van `test_systematic_programme_v5_flow.py`, in een tijdelijke root.

Bewezen wordt wat v6 aan de orkestratie toevoegt aan v5 en wat v7 en v8 erven: de margemarkt
en de hash van de mark-panelen, het ene vooruit-slot voor de hele carryfamilie, de selectie van
de laagste hefboom, de liquidatie-poorten en de VOORUIT-lezing. Die lezing registreert zich in
haar slot voordat er over de vooruit-data gerekend wordt; te weinig data en ontbrekende
panelen weigeren zonder iets te registreren; een veranderde historie weigert nadat de lezing
al geteld is. De rekenkern (boek, batterij) wordt elders getoetst.
"""
from __future__ import annotations

import json
import math
import runpy
import shutil
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tests.unit.test_systematic_programme_v5_flow import (
    FORWARD_BASE,
    FORWARD_START,
    GIT_SHA,
    HOLDOUT,
    MANIFESTS,
    N_BASE,
    N_WORLD,
    PANEL_BASE,
    PANEL_SETS,
    SEED_TOTAL,
    SYMBOLS,
    T0,
    canonical,
    day,
    edit_yaml,
    ledger_entries,
    lock_state,
    quiet,
    synthetic_world,
    write_panels,
    write_root,
    write_w_dev_artefacts,
)
from tradebot.backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from tradebot.registry.preregistration import PreRegistration
from tradebot.systematic import programme_v2 as v2
from tradebot.systematic import programme_v5 as v5
from tradebot.systematic import programme_v6 as v6
from tradebot.systematic.book import BookResult
from tradebot.utils.failfast import DataContractError
from tradebot.utils.hashing import hash_config
from tradebot.validation.holdout import HoldoutAlreadyUsed, freeze_holdout


@pytest.fixture(autouse=True)
def _pin_git_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v6, "current_git_sha", lambda: GIT_SHA)


# --------------------------------------------------------------------------- #
# Hulpmiddelen, ook voor de v7/v8-toetsen: alles gaat via een `Programme`
# --------------------------------------------------------------------------- #
def version_of(prog: v6.Programme) -> str:
    return prog.name.removeprefix("robust_book_")


def programme_cfg(root: Path, prog: v6.Programme = v6.V6) -> Any:
    return prog.load_config(root / prog.config)


def programme_dev_sharpe(root: Path, prog: v6.Programme, name: str) -> float:
    """De W_DEV-Sharpe die een echte run voor `name` zou opslaan."""
    cfg = programme_cfg(root, prog)
    w = cfg.windows
    market = v6.load_market(root, cfg, end=v2._ts(w.holdout_start))
    res = prog.simulate(name, market, cfg, v6.costs_for(root, cfg))
    return v2._sr(res.window(v2._ts(w.train_start), v2._ts(w.w_dev_end)))


def seed_programme(root: Path, prog: v6.Programme = v6.V6, *, name: str | None = None,
                   n_binding: int = 0, dev_sharpe: float | None = None) -> SimpleNamespace:
    """Bevries `prog` in een bestaande root en laat de W_DEV-artefacten van `name` achter."""
    v6.freeze(root, git_sha=GIT_SHA, prog=prog)
    name = name or prog.candidates[0]
    prereg_id = v6._prereg(root, programme_cfg(root, prog), prog)[0].preregistration_id
    sharpe = programme_dev_sharpe(root, prog, name) if dev_sharpe is None else dev_sharpe
    write_w_dev_artefacts(root / prog.artefact_dir, prereg_id=prereg_id, name=name,
                          dev_sharpe=sharpe, n_binding=n_binding,
                          selection_rule="lowest_leverage_all_clear")
    return SimpleNamespace(root=root, prog=prog, prereg_id=prereg_id, name=name,
                           dev_sharpe=sharpe, hid=f"{prog.name}/{prereg_id}/{name}")


def frozen_programme(root: Path, prog: v6.Programme = v6.V6, *, name: str | None = None,
                     n_binding: int = 0, dev_sharpe: float | None = None,
                     **kw: Any) -> SimpleNamespace:
    """Een nieuwe root waarin `prog` is bevroren en de W_DEV-artefacten van `name` staan."""
    write_root(root, version_of(prog), **kw)
    return seed_programme(root, prog, name=name, n_binding=n_binding, dev_sharpe=dev_sharpe)


def reads_in(root: Path, lock: Path) -> list[str]:
    return [r["hypothesis_id"] for r in lock_state(root, lock)["reads"]]


def all_locks(root: Path) -> dict[str, list[str]]:
    return {str(lock): reads_in(root, lock)
            for lock in (v2.LOCK_PATH, v5.BACKCAST_LOCK, v6.FORWARD_LOCK)}


@pytest.fixture(scope="module")
def dev_sharpes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    """De W_DEV-Sharpe van elke v6-kandidaat in de standaardwereld."""
    root = write_root(tmp_path_factory.mktemp("v6_dev"), "v6")
    return {name: programme_dev_sharpe(root, v6.V6, name) for name in v6.CANDIDATES}


@pytest.fixture
def frozen(tmp_path: Path, dev_sharpes: dict[str, float]) -> Callable[..., SimpleNamespace]:
    def make(sub: str | None = None, *, name: str = "K1_CARRY_PM_1X", **kw: Any) -> SimpleNamespace:
        if "world" not in kw and "dev_sharpe" not in kw:
            kw["dev_sharpe"] = dev_sharpes[name]
        return frozen_programme(tmp_path if sub is None else tmp_path / sub, name=name, **kw)
    return make


def add_forward_panels(root: Path, world: Any = None, n_bars: int | None = None,
                       base: str = FORWARD_BASE) -> Path:
    """Schrijf de vooruit-panelen (alle bars van de wereld, of `n_bars`)."""
    write_panels(root, synthetic_world() if world is None else world, base=base, n_bars=n_bars)
    return root / base


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Eén echte keten `freeze` -> `run`, met de echte batterij; de rest leest hieruit."""
    root = write_root(tmp_path_factory.mktemp("v6_chain"), "v6")
    seen: list[pd.Timestamp] = []
    real_simulate, real_pbo = v6.V6.simulate, v6.pbo_record

    def spy(name: str, m: Any, *args: Any, **kw: Any) -> BookResult:
        seen.append(m.index.max())
        return real_simulate(name, m, *args, **kw)

    prog = replace(v6.V6, simulate=spy)
    logs: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(v6, "current_git_sha", lambda: GIT_SHA)
        mp.setattr(v6, "N_SEEDS", 2)
        mp.setattr(v6, "pbo_record", lambda variants: real_pbo(variants, n_subsets=4))
        v6.freeze(root, git_sha=GIT_SHA, prog=prog)
        locks = {k: (root / k).read_bytes()
                 for k in (v2.LOCK_PATH, v5.BACKCAST_LOCK, v6.FORWARD_LOCK)}
        summary = v6.run_programme(root, log=logs.append, prog=prog)
    directory = root / v6.ARTEFACT_DIR
    records = {n: json.loads((directory / f"{n}.json").read_text("utf-8")) for n in v6.CANDIDATES}
    return SimpleNamespace(root=root, summary=summary, records=records, logs=logs, seen=seen,
                           locks=locks, cfg=programme_cfg(root))


# --------------------------------------------------------------------------- #
# Markt, hashes, kosten
# --------------------------------------------------------------------------- #
def test_data_hashes_add_the_mark_manifest_and_mark_panels_to_those_of_v5(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    hashes = dict(v6._data_hashes(root))
    assert {k: v for k, v in hashes.items() if not k.startswith("binance_mark")} == dict(
        v5._data_hashes(root))
    assert hashes["binance_mark/objects"] == MANIFESTS["mark"] * 8
    mark = {k for k in hashes if k.startswith("binance_mark/panel/")}
    assert mark == {f"binance_mark/panel/{s}" for s in PANEL_SETS["mark_panels"]}
    assert list(v6._data_hashes(root)) == sorted(v6._data_hashes(root))
    # De panelen kunnen uit een andere map komen (zo leest het vooruit-sample ze).
    other = root / "elders"
    shutil.copytree(root / v6.MARK_DIR, other)
    (other / "high.parquet").unlink()
    moved = dict(v6._data_hashes(root, mark_dir=other.relative_to(root)))
    assert {k for k in moved if k.startswith("binance_mark/panel/")} == {"binance_mark/panel/low"}
    # Een markpaneel dat verandert, verandert precies zijn eigen hash.
    path = root / v6.MARK_DIR / "low.parquet"
    (pd.read_parquet(path) * 0.99).to_parquet(path)
    after = dict(v6._data_hashes(root))
    assert {k for k in after if after[k] != hashes[k]} == {"binance_mark/panel/low"}


def test_the_preregistration_reads_the_spec_of_its_programme(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    cfg = programme_cfg(root)
    prereg, hashes = v6._prereg(root, cfg)
    assert isinstance(prereg, PreRegistration) and prereg.planned_trials == 3
    assert hashes == v6._data_hashes(root) and prereg.data_hashes == hashes
    assert prereg.parameters == v6._params(cfg) == json.loads(cfg.model_dump_json())
    assert v6._prereg(root, cfg, v6.V6)[0].preregistration_id == prereg.preregistration_id
    other = root / "conf/research/andere.yaml"
    shutil.copyfile(root / v6.SPEC_PATH, other)
    edit_yaml(other, lambda d: d.update(planned_trials=2))
    elsewhere = replace(v6.V6, spec=other.relative_to(root))
    assert v6._prereg(root, cfg, elsewhere)[0].planned_trials == 2
    # De mark-panelen horen bij het bewijs: verandert er een, dan is het een andere registratie.
    path = root / v6.MARK_DIR / "high.parquet"
    (pd.read_parquet(path) * 1.01).to_parquet(path)
    assert v6._prereg(root, cfg)[0].preregistration_id != prereg.preregistration_id


def test_the_candidates_of_the_config_must_be_those_of_the_programme(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    v6._require_candidates(programme_cfg(root))
    v6._require_candidates(programme_cfg(root), v6.V6)
    def drop_k3(d: dict[str, Any]) -> None:
        del d["leverage"]["candidates"]["K3_CARRY_PM_2X"]
        d["planned_trials"] = 2

    changed = write_root(tmp_path / "changed", "v6", edit_config=drop_k3)
    cfg = programme_cfg(changed)
    with pytest.raises(DataContractError, match="kandidaten in de config wijken af"):
        v6._require_candidates(cfg)
    with pytest.raises(DataContractError, match="kandidaten in de config wijken af"):
        v6.freeze(changed, git_sha=GIT_SHA)
    assert ledger_entries(changed) == [] and not (changed / v6.FORWARD_LOCK).exists()
    with pytest.raises(DataContractError, match="kandidaten in de config wijken af"):
        v6.run_programme(changed, log=quiet)
    assert not (changed / v6.ARTEFACT_DIR).exists()


def test_costs_for_hands_its_spread_overrides_to_the_v5_cost_model(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    cfg = programme_cfg(root)
    base = v6.costs_for(root, cfg)
    assert base == v5.costs_for(root, cfg)
    stress = v6.costs_for(root, cfg, perp_half_spread_bps=6.0, spot_half_spread_bps=10.0)
    assert stress.perp.half_spread == pytest.approx(6e-4)
    assert stress.spot.half_spread == pytest.approx(10e-4)
    assert stress.perp.taker_fee == base.perp.taker_fee


def test_load_market_lays_the_mark_and_spot_extremes_on_the_perp_grid(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    cfg = programme_cfg(root)
    world = synthetic_world()
    m = v6.load_market(root, cfg)
    cols = list(m.symbols)
    assert len(m.index) == N_BASE and set(cols) == set(SYMBOLS)
    assert m.mark_high.to_numpy().tolist() == (
        world["mark_panels"]["high"].iloc[:N_BASE][cols].to_numpy().tolist())
    assert m.mark_low.to_numpy().tolist() == (
        world["mark_panels"]["low"].iloc[:N_BASE][cols].to_numpy().tolist())
    assert m.spot_high.to_numpy().tolist() == (
        world["spot_panels"]["high"].iloc[:N_BASE][cols].to_numpy().tolist())
    cut = v2._ts(day(HOLDOUT[0]))
    part = v6.load_market(root, cfg, end=cut)
    assert len(part.index) == HOLDOUT[0] == len(part.mark_high) == len(part.spot_low)
    assert part.index.max() < cut
    narrow = v6.load_market(root, cfg, top_n=4)
    assert int(narrow.basis.perp.universe.sum(axis=1).max()) == 4
    assert int(m.basis.perp.universe.sum(axis=1).max()) == cfg.universe.top_n == 6


def test_the_forward_market_reads_its_own_panels_and_leaves_the_frozen_ones_alone(
        tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    cfg = programme_cfg(root)
    pr = add_forward_panels(root)
    before = {p: p.read_bytes() for p in (root / PANEL_BASE).glob("*/*.parquet")}
    frozen_market = v6.load_market(root, cfg)
    fwd = v6.load_market(root, cfg, panel_root=pr)
    assert len(fwd.index) == N_WORLD and fwd.index[:N_BASE].equals(frozen_market.index)
    assert fwd.mark_high.iloc[:N_BASE].equals(frozen_market.mark_high)
    cut = v2._ts(day(HOLDOUT[0]))
    narrow = v6.load_market(root, cfg, panel_root=pr, end=cut, top_n=4)
    assert narrow.index.max() < cut
    assert int(narrow.basis.perp.universe.sum(axis=1).max()) == 4
    assert {p: p.read_bytes() for p in (root / PANEL_BASE).glob("*/*.parquet")} == before


def test_the_panel_hashes_of_a_forward_root_cover_every_parquet_file(tmp_path: Path) -> None:
    pr = add_forward_panels(write_root(tmp_path, "v6"))
    hashes = v6._panel_hashes(pr)
    files = sorted(f"{sub}/{name}.parquet" for sub, names in PANEL_SETS.items() for name in names)
    assert list(hashes) == files and len(set(hashes.values())) == len(files)
    assert all(len(h) == 16 for h in hashes.values())
    path = pr / "spot_panels" / "close.parquet"
    (pd.read_parquet(path) * 1.01).to_parquet(path)
    after = v6._panel_hashes(pr)
    assert {k for k in after if after[k] != hashes[k]} == {"spot_panels/close.parquet"}


def test_the_financing_modes_and_the_margin_spec_follow_the_leverage_block(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    cfg = programme_cfg(root)
    market = v6.load_market(root, cfg)
    base = v6.financing(market.basis, cfg)
    assert (base >= cfg.leverage.financing_floor_apr).all()
    assert (v6.financing(market.basis, cfg, "stress") >= base).all()
    assert (v6.financing(market.basis, cfg, "optimistic") == cfg.leverage.financing_optimistic_apr).all()
    assert v6.margin_spec(cfg).as_record()["majors"] == list(cfg.leverage.majors)


# --------------------------------------------------------------------------- #
# Bevriezen
# --------------------------------------------------------------------------- #
def test_freeze_books_the_programme_and_freezes_the_forward_slot(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    holdout_lock, backcast_lock = (root / v2.LOCK_PATH).read_bytes(), (root / v5.BACKCAST_LOCK).read_bytes()
    path = v6.freeze(root, git_sha=GIT_SHA)
    cfg = programme_cfg(root)
    prereg, hashes = v6._prereg(root, cfg)
    assert path == root / v2.PREREG_DIR / f"preregistration_{prereg.preregistration_id}.json"
    frozen = json.loads(path.read_text("utf-8"))
    assert frozen["git_sha"] == GIT_SHA and frozen["ledger_total_at_freeze"] == SEED_TOTAL + 3
    (entry,) = ledger_entries(root)
    assert (entry["unit"], entry["wave"], entry["market"]) == ("robust_book_v6_programme", 6, "crypto")
    assert (entry["n_trials"], entry["result"]) == (3, "interim")
    assert entry["notes"] == v6.V6.notes
    assert entry["preregistration_id"] == prereg.preregistration_id and entry["git_sha"] == GIT_SHA
    assert entry["data_hash"] == hash_config(dict(hashes))
    assert entry["config_hash"] == hash_config(v6._params(cfg))
    forward = lock_state(root, v6.FORWARD_LOCK)
    assert forward["split_utc"] == f"{cfg.forward.start}T00:00:00+00:00"
    assert forward["git_sha"] == GIT_SHA and forward["reads"] == []
    assert (root / v2.LOCK_PATH).read_bytes() == holdout_lock, "het holdout-slot is van v2"
    assert (root / v5.BACKCAST_LOCK).read_bytes() == backcast_lock, "het backcast-slot is van v5"


def test_an_existing_forward_slot_is_shared_by_the_whole_carry_family(tmp_path: Path) -> None:
    """Het vooruit-slot is er één; lezingen zijn per hypothese-id. Een tweede programma
    bevriest het niet opnieuw (en crasht dus ook niet op `HoldoutAlreadyFrozen`)."""
    root = write_root(tmp_path, "v6")
    freeze_holdout(split_utc="2021-08-01T00:00:00+00:00", out=root / v6.FORWARD_LOCK,
                   git_sha="earlier")
    before = (root / v6.FORWARD_LOCK).read_bytes()
    v6.freeze(root, git_sha=GIT_SHA)
    assert (root / v6.FORWARD_LOCK).read_bytes() == before
    assert len(ledger_entries(root)) == 1


def test_freeze_asks_git_for_the_sha_when_none_is_given(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    root = write_root(tmp_path, "v6")
    monkeypatch.setattr(v6, "current_git_sha", lambda: "feed456")
    v6.freeze(root)
    assert ledger_entries(root)[0]["git_sha"] == "feed456"
    assert lock_state(root, v6.FORWARD_LOCK)["git_sha"] == "feed456"


def test_freeze_refuses_a_second_booking_and_leaves_ledger_and_slots_alone(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    v6.freeze(root, git_sha=GIT_SHA)
    snapshot = {p: (root / p).read_bytes() for p in (v2.LEDGER_PATH, v6.FORWARD_LOCK)}
    with pytest.raises(DataContractError, match="al geboekt"):
        v6.freeze(root, git_sha=GIT_SHA)
    assert {p: (root / p).read_bytes() for p in snapshot} == snapshot


@pytest.mark.parametrize(("config_trials", "spec_trials"), [(3, 4), (4, 3), (4, 4)])
def test_freeze_refuses_when_the_planned_trials_differ_from_the_candidates(
        tmp_path: Path, config_trials: int, spec_trials: int) -> None:
    root = write_root(tmp_path, "v6", edit_config=lambda d: d.update(planned_trials=config_trials),
                      edit_spec=lambda d: d.update(planned_trials=spec_trials))
    with pytest.raises(DataContractError, match="planned_trials wijkt af"):
        v6.freeze(root, git_sha=GIT_SHA)
    assert ledger_entries(root) == [] and not (root / v6.FORWARD_LOCK).exists()
    assert not list((root / v2.PREREG_DIR).glob("preregistration_*.json"))


# --------------------------------------------------------------------------- #
# De W_DEV-run
# --------------------------------------------------------------------------- #
def test_the_w_dev_run_refuses_without_a_frozen_preregistration_and_writes_nothing(
        tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v6.run_programme(root, log=quiet)
    assert not (root / v6.ARTEFACT_DIR).exists()


def test_the_w_dev_run_selects_the_lowest_leverage_that_clears_every_gate(
        chain: SimpleNamespace) -> None:
    s, cfg, recs = chain.summary, chain.cfg, chain.records
    prereg = v6._prereg(chain.root, cfg)[0]
    assert s["programme"] == "robust_book_v6" and s["preregistration_id"] == prereg.preregistration_id
    assert s["git_sha"] == GIT_SHA and s[EVIDENCE_KEY] == NOT_ADMISSIBLE
    assert s["m_programme"] == SEED_TOTAL + 3
    assert s["m_including_known_prior"] == s["m_programme"] + cfg.known_prior_trials
    assert set(s["robustness_scores"]) == set(v6.CANDIDATES)
    assert s["robustness_scores"] == {n: recs[n]["robustness_score"]["total"] for n in v6.CANDIDATES}
    clean = [n for n in v6.CANDIDATES if recs[n]["n_binding_w_dev"] == 0]
    assert clean, "de lenige preregistratie laat alles door"
    leverage = cfg.leverage.candidates
    assert s["selected"] == min(clean, key=lambda n: leverage[n])
    assert s["selection_rule"] == "lowest_leverage_all_clear"
    assert s["passes_all_w_dev_gates"] == {n: recs[n]["n_binding_w_dev"] == 0 for n in v6.CANDIDATES}
    assert s["binding_w_dev"] == {n: sorted(k for k, g in recs[n]["gates"].items() if g["binds"])
                                  for n in v6.CANDIDATES}
    for n in v6.CANDIDATES:
        assert set(s["table"][n]) >= {"sharpe", "ann_financing", "avg_loan", "n_liquidations",
                                      "min_stress_headroom", "sharpe_ci_low", "sharpe_ci_high"}
        assert s["table"][n]["sharpe"] == recs[n]["w_dev"]["sharpe"]
    stored = json.loads((chain.root / v6.ARTEFACT_DIR / "programme_w_dev.json").read_text("utf-8"))
    assert canonical(stored) == canonical(s)
    assert chain.logs[:3] == [f"run {n}" for n in v6.CANDIDATES]
    assert [m for m in chain.logs if m.startswith("batterij")] == [
        f"batterij {n}" for n in v6.CANDIDATES]


def test_every_w_dev_record_carries_its_provenance_gates_and_leverage_stress(
        chain: SimpleNamespace) -> None:
    cfg = chain.cfg
    prereg, hashes = v6._prereg(chain.root, cfg)
    w = cfg.windows
    for name, rec in chain.records.items():
        assert rec["trial"] == name and rec["hypothesis"] == v6.HYPOTHESES[name]
        assert rec["role"] == "candidate" and rec[EVIDENCE_KEY] == NOT_ADMISSIBLE
        assert rec["leverage_per_leg"] == cfg.leverage.candidates[name]
        assert rec["preregistration_id"] == prereg.preregistration_id and rec["git_sha"] == GIT_SHA
        assert rec["data_hashes"] == dict(hashes) and rec["parameters"] == v6._params(cfg)
        assert rec["windows"] == {"train": [w.train_start, w.train_end],
                                  "validate": [w.validate_start, w.w_dev_end],
                                  "backcast": [cfg.backcast.start, "unread"],
                                  "holdout": [w.holdout_start, "unread"],
                                  "forward": [cfg.forward.start, "unread"]}
        assert rec["w_dev"]["start"] == w.train_start and rec["w_dev"]["end"] == w.w_dev_end
        assert rec["execution"]["margin"] == v6.margin_spec(cfg).as_record()
        assert rec["execution"]["lag_bars"] == cfg.execution.lag_bars
        assert set(rec["dsr"]) == {f"m_{chain.summary['m_programme']}",
                                   f"m_{chain.summary['m_including_known_prior']}"}
        bat = rec["battery"]
        assert bat["signal_noise"]["n_seeds"] == bat["missing_data"]["n_seeds"] == 2
        assert set(bat["universe"]) == {"top_30", "top_100"}
        assert set(bat["perturbation"]) == set(v6._family())
        assert {"financing_stress", "financing_optimistic", "margin_stress", "governor_off"} <= set(bat)
        assert rec["pbo"]["n_subsets"] == 4 and rec["pbo"]["n_combinations"] == 6
        assert set(rec["gates"]) <= {c.name for c in prereg.stop_criteria}
        assert rec["n_binding_w_dev"] == sum(g["binds"] for g in rec["gates"].values())
        metrics = rec["gate_metrics"]
        assert metrics["net_sharpe_w_dev"] == rec["w_dev"]["sharpe"]
        assert metrics["n_liquidations_w_dev"] == float(rec["w_dev"]["n_liquidations"]) == 0.0
        assert metrics["n_liquidations_margin_stress_w_dev"] == float(
            bat["margin_stress"]["n_liquidations"])
        assert metrics["net_sharpe_financing_stress_w_dev"] == bat["financing_stress"]["sharpe"]
        assert metrics["net_cagr_financing_stress_w_dev"] == bat["financing_stress"]["cagr"]
        assert metrics["pbo"] == rec["pbo"]["pbo"]
        assert 0.0 <= rec["robustness_score"]["total"] <= 100.0


def test_the_w_dev_run_never_looks_at_the_holdout_or_the_forward_sample(
        chain: SimpleNamespace) -> None:
    w = chain.cfg.windows
    assert chain.seen and max(chain.seen) < v2._ts(w.holdout_start)
    for name in v6.CANDIDATES:
        daily = pd.read_csv(chain.root / v6.ARTEFACT_DIR / f"{name}_daily_w_dev.csv",
                            index_col=0, parse_dates=True)
        assert str(daily.index[0].date()) == w.train_start
        assert str(daily.index[-1].date()) == w.w_dev_end
        assert {"net", "financing", "liquidation", "loan"} <= set(daily.columns)
    assert not list((chain.root / v6.ARTEFACT_DIR).glob("*backcast*"))
    assert not list((chain.root / v6.ARTEFACT_DIR).glob("*holdout*"))
    for lock, before in chain.locks.items():
        assert (chain.root / lock).read_bytes() == before
        assert json.loads(before)["reads"] == []
    assert len(ledger_entries(chain.root)) == 1, "de run boekt niets bij: dat doet het oordeel"


def test_the_oos_read_continues_straight_from_the_w_dev_run(chain: SimpleNamespace,
                                                            tmp_path: Path) -> None:
    root = tmp_path / "root"
    shutil.copytree(chain.root, root)
    out = v6.read_oos(root, log=quiet)
    s, rec = chain.summary, chain.records[chain.summary["selected"]]
    assert out["candidate"] == s["selected"] and out["selection_rule"] == s["selection_rule"]
    assert out["w_dev_sharpe"] == rec["w_dev"]["sharpe"] and out["n_binding_w_dev"] == 0
    assert out["verdict"] == "promote_to_paper_trading" and out["n_binding_total"] == 0
    assert out["hypothesis_id"] == f"robust_book_v6/{s['preregistration_id']}/{s['selected']}"
    assert out["reads"]["backcast"]["invested"] and out["reads"]["holdout"]["invested"]
    (_, verdict) = ledger_entries(root)
    assert verdict["metrics"]["w_dev_cagr"] == rec["w_dev"]["cagr"]


def test_the_forward_read_continues_straight_from_the_w_dev_run(chain: SimpleNamespace,
                                                                tmp_path: Path) -> None:
    root = tmp_path / "root"
    shutil.copytree(chain.root, root)
    add_forward_panels(root)
    out = v6.read_forward(root, log=quiet)
    s = chain.summary
    assert out["candidate"] == s["selected"] and out["verdict"] == "eligible_for_capital"
    assert out["w_dev_sharpe"] == chain.records[s["selected"]]["w_dev"]["sharpe"]
    assert out["hypothesis_id"] == f"robust_book_v6/{s['preregistration_id']}/{s['selected']}"


# --------------------------------------------------------------------------- #
# De OOS-lezing
# --------------------------------------------------------------------------- #
def test_the_oos_read_registers_both_reads_writes_the_artefacts_and_books_the_verdict(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen(name="K2_CARRY_PM_1P5X")
    root = ctx.root
    logs: list[str] = []
    out = v6.read_oos(root, log=logs.append)
    cfg = programme_cfg(root)
    assert out["hypothesis_id"] == ctx.hid and out["candidate"] == ctx.name
    assert out["verdict"] == "promote_to_paper_trading" and out[EVIDENCE_KEY] == NOT_ADMISSIBLE
    assert out["selection_rule"] == "lowest_leverage_all_clear"
    assert out["n_binding_total"] == 0 and out["w_dev_sharpe"] == ctx.dev_sharpe
    assert reads_in(root, v5.BACKCAST_LOCK) == [ctx.hid] == reads_in(root, v2.LOCK_PATH)
    assert reads_in(root, v6.FORWARD_LOCK) == [], "het vooruit-sample blijft dicht"
    assert out["reads"]["backcast"]["window"] == [cfg.backcast.start, cfg.backcast.end]
    assert out["reads"]["holdout"]["window"] == [cfg.windows.holdout_start, cfg.windows.holdout_end]
    for label in ("backcast", "holdout"):
        r = out["reads"][label]
        assert r["invested"] is True and r["pct_bars_invested"] > 0.0 and r["n_liquidations"] == 0
        assert math.isfinite(r["z_vs_dev"]) and r["yearly"] is not None
        assert {"ann_financing", "avg_loan", "min_uni_mmr"} <= set(r["summary"])
        daily = pd.read_csv(root / v6.ARTEFACT_DIR / f"{ctx.name}_daily_{label}.csv",
                            index_col=0, parse_dates=True)
        assert [str(daily.index[0].date()), str(daily.index[-1].date())] == r["window"]
        assert len(daily) == r["summary"]["n_obs"]
    metrics = out["gate_metrics"]
    assert metrics["backcast_n_liquidations"] == 0.0 == metrics["holdout_n_liquidations"]
    assert metrics["holdout_max_drawdown"] == out["reads"]["holdout"]["summary"]["max_drawdown"]
    assert set(out["gates"]) == {
        "backcast_sharpe_collapse", "backcast_drawdown", "backcast_liquidated",
        "backcast_no_positive_sharpe", "holdout_sharpe_collapse", "holdout_drawdown",
        "holdout_liquidated"}
    stored = json.loads((root / v6.ARTEFACT_DIR / "oos_read.json").read_text("utf-8"))
    assert canonical(stored) == canonical(out)
    programme, verdict = ledger_entries(root)
    assert verdict["unit"] == "robust_book_v6_verdict" and verdict["wave"] == 6
    assert verdict["result"] == "accepted" and verdict["n_trials"] == 0
    assert verdict["amends"] == programme["config_hash"] != ""
    assert verdict["preregistration_id"] == ctx.prereg_id
    assert verdict["notes"].startswith("Oordeel robust_book_v6 na W_DEV")
    assert verdict["metrics"]["selected"] == ctx.name and verdict["metrics"]["w_dev_cagr"] == 0.15
    assert verdict["metrics"]["holdout_cagr"] == out["reads"]["holdout"]["summary"]["cagr"]
    assert verdict["metrics"]["holdout_invested"] is True
    assert logs[-1] == "oordeel: promote_to_paper_trading"
    assert logs[0].startswith(f"backcast {ctx.name}: Sharpe ") and "liquidaties 0" in logs[0]


OORDELEN = [
    pytest.param((), 0, "promote_to_paper_trading", "accepted", id="alles-groen"),
    pytest.param(("holdout_drawdown",), 0, "falsified_out_of_sample", "falsified",
                 id="een-falsificatiepoort-bindt"),
    pytest.param(("backcast_no_positive_sharpe",), 0, "archived", "archived",
                 id="alleen-een-archiefpoort-bindt"),
    pytest.param(("backcast_no_positive_sharpe", "holdout_liquidated"), 0,
                 "falsified_out_of_sample", "falsified", id="falsificatie-wint-van-archief"),
    pytest.param((), 2, "archived", "archived", id="w-dev-poorten-gaan-mee"),
    pytest.param(("backcast_liquidated",), 3, "falsified_out_of_sample", "falsified",
                 id="falsificatie-wint-van-w-dev"),
]


@pytest.mark.parametrize(("bind", "n_dev", "verdict", "result"), OORDELEN)
def test_the_verdict_follows_the_binding_gates(frozen: Callable[..., SimpleNamespace],
                                               bind: tuple[str, ...], n_dev: int, verdict: str,
                                               result: str) -> None:
    ctx = frozen(bind=bind, n_binding=n_dev)
    out = v6.read_oos(ctx.root, log=quiet)
    assert out["verdict"] == verdict
    assert out["n_binding_w_dev"] == n_dev and out["n_binding_total"] == len(bind) + n_dev
    assert {k for k, g in out["gates"].items() if g["binds"]} == set(bind)
    entry = ledger_entries(ctx.root)[-1]
    assert entry["result"] == result and entry["metrics"]["verdict"] == verdict
    assert entry["unit"] == "robust_book_v6_verdict" and entry["n_trials"] == 0


def test_a_liquidation_in_the_holdout_falsifies_through_its_frozen_gate(
        frozen: Callable[..., SimpleNamespace]) -> None:
    """Met de ECHTE drempel van `holdout_liquidated` (> 0): een squeeze op de mark price in de
    holdout liquideert het hefboomboek en falsificeert; dezelfde wereld zonder squeeze niet."""
    calm = frozen("calm", name="K3_CARRY_PM_2X", keep=("holdout_liquidated",))
    out = v6.read_oos(calm.root, log=quiet)
    assert out["reads"]["holdout"]["n_liquidations"] == 0
    assert out["gates"]["holdout_liquidated"]["binds"] is False
    assert out["verdict"] == "promote_to_paper_trading"

    squeezed = frozen("squeezed", name="K3_CARRY_PM_2X", keep=("holdout_liquidated",),
                      world=synthetic_world(spike=(HOLDOUT[0] + 15, 2.0)))
    out = v6.read_oos(squeezed.root, log=quiet)
    assert out["reads"]["holdout"]["n_liquidations"] == 1
    assert out["reads"]["backcast"]["n_liquidations"] == 0
    gate = out["gates"]["holdout_liquidated"]
    assert gate["value"] == 1.0 and gate["threshold"] == 0.0 and gate["binds"] is True
    assert out["verdict"] == "falsified_out_of_sample"
    assert ledger_entries(squeezed.root)[-1]["result"] == "falsified"


def test_a_holdout_without_any_position_is_undefined_and_binds_without_a_nan_in_the_ledger(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen(name="K1_CARRY_PM_1X", world=synthetic_world(flat_from=HOLDOUT[0] - 25))
    out = v6.read_oos(ctx.root, log=quiet)
    hold = out["reads"]["holdout"]
    assert hold["invested"] is False and hold["note"] == v5.FLAT_WINDOW
    assert hold["pct_bars_invested"] == 0.0 and hold["n_liquidations"] == 0
    assert math.isnan(hold["z_vs_dev"]) and math.isnan(hold["summary"]["sharpe"])
    assert hold["summary"]["cagr"] == 0.0 and hold["summary"]["max_drawdown"] == 0.0
    assert out["reads"]["backcast"]["invested"] is True
    assert {k for k, g in out["gates"].items() if g["binds"]} == {"holdout_sharpe_collapse"}
    assert out["verdict"] == "falsified_out_of_sample"
    assert "NaN" not in (ctx.root / v2.LEDGER_PATH).read_text("utf-8")
    metrics = ledger_entries(ctx.root)[-1]["metrics"]
    assert metrics["holdout_sharpe"] is None and metrics["holdout_invested"] is False
    assert metrics["holdout_cagr"] == 0.0 and metrics["backcast_sharpe"] is not None


def test_each_read_is_in_its_lock_before_the_book_is_simulated(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    at_call: list[dict[str, list[str]]] = []

    def spy(*args: Any, **kw: Any) -> BookResult:
        at_call.append(all_locks(ctx.root))
        return v6.V6.simulate(*args, **kw)

    v6.read_oos(ctx.root, log=quiet, prog=replace(v6.V6, simulate=spy))
    assert at_call == [{str(v2.LOCK_PATH): [ctx.hid], str(v5.BACKCAST_LOCK): [ctx.hid],
                        str(v6.FORWARD_LOCK): []}]


def test_a_second_oos_read_is_refused_and_changes_nothing(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    v6.read_oos(ctx.root, log=quiet)
    snapshot = {p: (ctx.root / p).read_bytes()
                for p in (v2.LEDGER_PATH, v2.LOCK_PATH, v5.BACKCAST_LOCK,
                          v6.ARTEFACT_DIR / "oos_read.json")}
    with pytest.raises(HoldoutAlreadyUsed, match="al gelezen"):
        v6.read_oos(ctx.root, log=quiet)
    assert {p: (ctx.root / p).read_bytes() for p in snapshot} == snapshot


def test_the_oos_read_refuses_a_w_dev_run_that_belongs_to_another_preregistration(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    write_w_dev_artefacts(ctx.root / v6.ARTEFACT_DIR, prereg_id="0" * 32, name=ctx.name,
                          dev_sharpe=ctx.dev_sharpe, selection_rule="x")
    with pytest.raises(DataContractError, match="andere preregistratie"):
        v6.read_oos(ctx.root, log=quiet)
    assert all(not reads for reads in all_locks(ctx.root).values())
    assert len(ledger_entries(ctx.root)) == 1


def test_the_oos_read_refuses_a_preregistration_that_was_never_frozen(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v6")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v6.read_oos(root, log=quiet)
    assert reads_in(root, v5.BACKCAST_LOCK) == [] == reads_in(root, v2.LOCK_PATH)


def test_a_w_dev_sharpe_that_the_full_run_cannot_reproduce_is_refused_after_registration(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    write_w_dev_artefacts(ctx.root / v6.ARTEFACT_DIR, prereg_id=ctx.prereg_id, name=ctx.name,
                          dev_sharpe=ctx.dev_sharpe + 0.5, selection_rule="x")
    with pytest.raises(DataContractError, match="reproduceert de W_DEV-Sharpe niet"):
        v6.read_oos(ctx.root, log=quiet)
    assert reads_in(ctx.root, v5.BACKCAST_LOCK) == [ctx.hid] == reads_in(ctx.root, v2.LOCK_PATH)
    assert not (ctx.root / v6.ARTEFACT_DIR / "oos_read.json").exists()
    assert len(ledger_entries(ctx.root)) == 1


# --------------------------------------------------------------------------- #
# De vooruit-lezing
# --------------------------------------------------------------------------- #
def test_the_forward_read_registers_before_it_simulates_and_books_the_verdict(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    pr = add_forward_panels(ctx.root)
    at_call: list[dict[str, list[str]]] = []

    def spy(*args: Any, **kw: Any) -> BookResult:
        at_call.append(all_locks(ctx.root))
        return v6.V6.simulate(*args, **kw)

    logs: list[str] = []
    out = v6.read_forward(ctx.root, log=logs.append, prog=replace(v6.V6, simulate=spy))
    assert at_call == [{str(v2.LOCK_PATH): [], str(v5.BACKCAST_LOCK): [],
                        str(v6.FORWARD_LOCK): [ctx.hid]}]
    cfg = programme_cfg(ctx.root)
    assert out["hypothesis_id"] == ctx.hid and out["candidate"] == ctx.name
    assert out["verdict"] == "eligible_for_capital" and out[EVIDENCE_KEY] == NOT_ADMISSIBLE
    assert out["w_dev_sharpe"] == ctx.dev_sharpe
    read = out["read"]
    assert read["window"] == [cfg.forward.start, day(N_WORLD - 1)]
    assert read["invested"] is True and read["n_liquidations"] == 0
    assert read["summary"]["n_obs"] == N_WORLD - FORWARD_START
    assert set(out["gates"]) == {"forward_sharpe_collapse", "forward_drawdown",
                                 "forward_liquidated", "forward_no_positive_sharpe"}
    assert out["gate_metrics"]["forward_net_sharpe"] == read["summary"]["sharpe"]
    assert out["gate_metrics"]["forward_n_liquidations"] == 0.0
    stored = json.loads((ctx.root / v6.ARTEFACT_DIR / "forward_read.json").read_text("utf-8"))
    assert canonical(stored) == canonical(out)
    programme, entry = ledger_entries(ctx.root)
    assert entry["unit"] == "robust_book_v6_forward" and entry["wave"] == 6
    assert entry["result"] == "accepted" and entry["n_trials"] == 0
    assert entry["amends"] == programme["config_hash"] != ""
    assert entry["preregistration_id"] == ctx.prereg_id
    assert entry["data_hash"] == hash_config(v6._panel_hashes(pr))
    assert entry["metrics"] == {"selected": ctx.name, "forward_sharpe": read["summary"]["sharpe"],
                                "forward_cagr": read["summary"]["cagr"],
                                "verdict": "eligible_for_capital"}
    assert logs[0].startswith(f"vooruit {ctx.name}: ") and logs[0].endswith("-> eligible_for_capital")
    assert all_locks(ctx.root) == {str(v2.LOCK_PATH): [], str(v5.BACKCAST_LOCK): [],
                                   str(v6.FORWARD_LOCK): [ctx.hid]}


FORWARD_OORDELEN = [
    pytest.param((), "eligible_for_capital", "accepted", id="alles-groen"),
    pytest.param(("forward_drawdown",), "falsified_forward", "falsified", id="falsificatie"),
    pytest.param(("forward_no_positive_sharpe",), "archived_forward", "archived", id="archief"),
    pytest.param(("forward_no_positive_sharpe", "forward_liquidated"), "falsified_forward",
                 "falsified", id="falsificatie-wint-van-archief"),
]


@pytest.mark.parametrize(("bind", "verdict", "result"), FORWARD_OORDELEN)
def test_the_forward_verdict_follows_the_binding_gates(
        frozen: Callable[..., SimpleNamespace], bind: tuple[str, ...], verdict: str,
        result: str) -> None:
    ctx = frozen(bind=bind)
    add_forward_panels(ctx.root)
    out = v6.read_forward(ctx.root, log=quiet)
    assert out["verdict"] == verdict
    assert {k for k, g in out["gates"].items() if g["binds"]} == set(bind)
    entry = ledger_entries(ctx.root)[-1]
    assert entry["result"] == result and entry["metrics"]["verdict"] == verdict


def test_the_forward_read_judges_with_the_frozen_preregistration_not_a_recomputed_one(
        frozen: Callable[..., SimpleNamespace]) -> None:
    """Nieuwe data verandert de data-hashes, dus een herberekende preregistratie zou een andere
    ID hebben: de lezing gebruikt de bevroren ID uit de W_DEV-run."""
    ctx = frozen()
    add_forward_panels(ctx.root)
    assert v6._prereg(ctx.root, programme_cfg(ctx.root))[0].preregistration_id == ctx.prereg_id
    new_close = ctx.root / PANEL_BASE / "spot_panels/close.parquet"
    (pd.read_parquet(new_close) * 1.0001).to_parquet(new_close)
    assert v6._prereg(ctx.root, programme_cfg(ctx.root))[0].preregistration_id != ctx.prereg_id
    out = v6.read_forward(ctx.root, log=quiet)
    assert out["verdict"] == "eligible_for_capital"
    assert ledger_entries(ctx.root)[-1]["preregistration_id"] == ctx.prereg_id


def test_a_forward_sample_without_any_position_is_undefined_and_falsifies(
        frozen: Callable[..., SimpleNamespace]) -> None:
    world = synthetic_world(flat_from=HOLDOUT[1] - 15)
    ctx = frozen(world=world)
    add_forward_panels(ctx.root, world)
    out = v6.read_forward(ctx.root, log=quiet)
    assert out["read"]["invested"] is False and out["read"]["note"] == v5.FLAT_WINDOW
    assert math.isnan(out["gate_metrics"]["forward_sharpe_z_vs_dev"])
    assert {k for k, g in out["gates"].items() if g["binds"]} == {
        "forward_sharpe_collapse", "forward_no_positive_sharpe"}
    assert out["verdict"] == "falsified_forward"
    entry = ledger_entries(ctx.root)[-1]
    assert entry["metrics"]["forward_sharpe"] is None and "NaN" not in json.dumps(entry)


def test_the_forward_read_refuses_without_panels_and_registers_nothing(
        frozen: Callable[..., SimpleNamespace], tmp_path: Path) -> None:
    ctx = frozen()
    before = (ctx.root / v6.FORWARD_LOCK).read_bytes()
    with pytest.raises(DataContractError, match="Geen vooruit-panelen"):
        v6.read_forward(ctx.root, log=quiet)
    with pytest.raises(DataContractError, match="Geen vooruit-panelen"):
        v6.read_forward(ctx.root, panel_root=tmp_path / "elders", log=quiet)
    assert (ctx.root / v6.FORWARD_LOCK).read_bytes() == before
    assert len(ledger_entries(ctx.root)) == 1


def test_too_little_forward_data_is_refused_before_any_registration(
        frozen: Callable[..., SimpleNamespace], tmp_path: Path) -> None:
    """Eén bar te weinig voor de volle maand: de weigering kijkt alleen naar het raster, en het
    vooruit-slot blijft onaangeroerd. Een absolute `panel_root` wordt zoals hij is gebruikt."""
    ctx = frozen()
    cfg = programme_cfg(ctx.root)
    end = pd.Timestamp(cfg.forward.start, tz="UTC") + pd.DateOffset(months=cfg.forward.min_months)
    n_short = (end - T0).days            # n bars: de laatste is dag n - 1, dus de maand is nog niet vol
    other = add_forward_panels(tmp_path / "andere_root", n_bars=n_short, base="vooruit")
    before = (ctx.root / v6.FORWARD_LOCK).read_bytes()
    with pytest.raises(DataContractError, match="zes volle maanden") as exc:
        v6.read_forward(ctx.root, panel_root=other, log=quiet)
    assert str(end) in str(exc.value)
    assert (ctx.root / v6.FORWARD_LOCK).read_bytes() == before
    assert reads_in(ctx.root, v6.FORWARD_LOCK) == []
    assert not (ctx.root / v6.ARTEFACT_DIR / "forward_read.json").exists()
    assert len(ledger_entries(ctx.root)) == 1
    # Met die ene bar erbij is de maand vol en wordt de lezing wel gedaan.
    add_forward_panels(tmp_path / "andere_root", n_bars=n_short + 1, base="vooruit")
    out = v6.read_forward(ctx.root, panel_root=other, log=quiet)
    assert out["verdict"] == "eligible_for_capital"
    assert reads_in(ctx.root, v6.FORWARD_LOCK) == [ctx.hid]


def test_the_forward_read_needs_the_frozen_forward_slot(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    add_forward_panels(ctx.root)
    (ctx.root / v6.FORWARD_LOCK).unlink()
    with pytest.raises(DataContractError, match="ontbreekt"):
        v6.read_forward(ctx.root, log=quiet)
    assert len(ledger_entries(ctx.root)) == 1


def test_the_forward_read_refuses_a_frozen_preregistration_that_does_not_exist(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    add_forward_panels(ctx.root)
    write_w_dev_artefacts(ctx.root / v6.ARTEFACT_DIR, prereg_id="f" * 32, name=ctx.name,
                          dev_sharpe=ctx.dev_sharpe, selection_rule="x")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v6.read_forward(ctx.root, log=quiet)
    assert reads_in(ctx.root, v6.FORWARD_LOCK) == []


def test_a_changed_history_refuses_the_forward_read_after_it_has_been_counted(
        frozen: Callable[..., SimpleNamespace]) -> None:
    """Panelen waarin de W_DEV-historie anders is, reproduceren de bevroren Sharpe niet. De
    lezing staat dan al in het slot (afmaken kan alleen eenmaal en zichtbaar), er is geen
    uitkomst opgeslagen of geboekt."""
    ctx = frozen()
    add_forward_panels(ctx.root, synthetic_world(seed=6))
    with pytest.raises(DataContractError, match="reproduceren de W_DEV-Sharpe niet"):
        v6.read_forward(ctx.root, log=quiet)
    assert reads_in(ctx.root, v6.FORWARD_LOCK) == [ctx.hid]
    assert not (ctx.root / v6.ARTEFACT_DIR / "forward_read.json").exists()
    assert len(ledger_entries(ctx.root)) == 1
    with pytest.raises(HoldoutAlreadyUsed):
        v6.read_forward(ctx.root, log=quiet)


def test_a_crash_while_computing_leaves_the_forward_read_registered(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    add_forward_panels(ctx.root)

    def boom(*_: Any, **__: Any) -> BookResult:
        raise RuntimeError("de berekening crasht")

    with pytest.raises(RuntimeError, match="crasht"):
        v6.read_forward(ctx.root, log=quiet, prog=replace(v6.V6, simulate=boom))
    assert reads_in(ctx.root, v6.FORWARD_LOCK) == [ctx.hid]
    assert len(ledger_entries(ctx.root)) == 1


def test_a_second_forward_read_is_refused_and_changes_nothing(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    add_forward_panels(ctx.root)
    v6.read_forward(ctx.root, log=quiet)
    snapshot = {p: (ctx.root / p).read_bytes() for p in (
        v2.LEDGER_PATH, v6.FORWARD_LOCK, v6.ARTEFACT_DIR / "forward_read.json")}
    with pytest.raises(HoldoutAlreadyUsed, match="al gelezen"):
        v6.read_forward(ctx.root, log=quiet)
    assert {p: (ctx.root / p).read_bytes() for p in snapshot} == snapshot


# --------------------------------------------------------------------------- #
# Het CLI
# --------------------------------------------------------------------------- #
def _fake_commands(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake(name: str, result: Any) -> Callable[..., Any]:
        def call(*args: Any, **kw: Any) -> Any:
            calls.append((name, {"args": args, **kw}))
            return result
        return call

    monkeypatch.setattr(v6, "freeze", fake("freeze", Path("artefacts/prereg.json")))
    monkeypatch.setattr(v6, "run_programme", fake("run", {"selected": "K1_CARRY_PM_1X"}))
    monkeypatch.setattr(v6, "read_oos", fake("oos", {"verdict": "archived"}))
    monkeypatch.setattr(v6, "read_forward", fake("forward", {"verdict": "eligible_for_capital"}))
    return calls


@pytest.mark.parametrize(("command", "printed"), [
    ("freeze", "artefacts/prereg.json\n"),
    ("run", json.dumps({"selected": "K1_CARRY_PM_1X"}, indent=2) + "\n"),
    ("oos", json.dumps({"verdict": "archived"}, indent=2) + "\n"),
    ("forward", json.dumps({"verdict": "eligible_for_capital"}, indent=2) + "\n"),
])
def test_main_runs_exactly_the_requested_step_and_prints_its_result(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str,
        printed: str) -> None:
    calls = _fake_commands(monkeypatch)
    v6.main([command])
    assert calls == [({"freeze": "freeze", "run": "run", "oos": "oos", "forward": "forward"}[command],
                      {"args": ()})]
    assert capsys.readouterr().out == printed


@pytest.mark.parametrize("argv", [[], ["nonsense"], ["freeze", "run"], ["oos-resume"]])
def test_main_refuses_bad_usage_before_running_anything(
        monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    calls = _fake_commands(monkeypatch)
    usage = r"Gebruik: python -I -m tradebot\.systematic\.programme_v6 \{freeze\|run\|oos\|forward\}"
    with pytest.raises(DataContractError, match=usage):
        v6.main(argv)
    assert calls == []


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_runs_as_a_script_and_hands_its_arguments_to_main(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["programme_v6"])
    with pytest.raises(DataContractError, match="Gebruik: python -I -m"):
        runpy.run_module("tradebot.systematic.programme_v6", run_name="__main__")
