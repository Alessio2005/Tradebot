"""De v5-orkestratie van begin tot eind, op een kleine synthetische wereld in een tijdelijke root.

Alles wat het programma leest of schrijft staat onder `tmp_path`: de config en de
preregistratie zijn afgeleid van de echte bestanden onder `conf/` (zelfde sleutels, een
klein universum en korte vensters), de panelen zijn synthetisch parquet, ledger en sloten
beginnen leeg. Bewezen wordt de ORKESTRATIE: bevriezen (ledger, preregistratie, backcast-slot),
de W_DEV-run (artefacten, selectie, niets van de holdout), de twee OOS-lezingen (elke lezing
in haar slot VOOR er gerekend wordt, elk oordeel, elke weigering, de hervatting na een crash)
en het CLI. De rekenkern zelf (boek, batterij) wordt elders getoetst.

Dit bestand levert ook de gereedschapskist (`write_root`, `synthetic_world`, ...) waarmee
`test_systematic_programme_v6_flow.py` en `test_systematic_programme_v7_v8_flow.py` dezelfde
wereld bouwen voor v6, v7 en v8.

De volledige PBO (CSCV, 16 deelsets, 12 870 combinaties) kost ~10 s per kandidaat en is een
eigen toets (`pbo_record` in de batterijtoetsen); de runs hier gebruiken dezelfde functie met
4 deelsets en 2 zaden per verstoring, zodat de keten in seconden loopt.
"""
from __future__ import annotations

import hashlib
import json
import math
import runpy
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tradebot.backtest.vectorized import EVIDENCE_KEY, NOT_ADMISSIBLE
from tradebot.execution.trade_costs import load_impact_params
from tradebot.registry.hypothesis_ledger import HypothesisLedger
from tradebot.schemas.robust_book_v2 import robust_book_v5_config
from tradebot.systematic import programme_v2 as v2
from tradebot.systematic import programme_v5 as v5
from tradebot.systematic.book import BookResult
from tradebot.utils.failfast import DataContractError
from tradebot.utils.hashing import hash_config
from tradebot.validation.holdout import HoldoutAlreadyUsed, freeze_holdout

REPO = Path(__file__).resolve().parents[2]
GIT_SHA = "abc1234"
SEED_TOTAL = 18

# --------------------------------------------------------------------------- #
# De wereld: 6 munten, 205 dagbars vanaf 2021-01-01. De bevroren panelen hebben de eerste
# 164 bars (tot en met het einde van de holdout), de vooruit-panelen alle 205. Het universum
# vraagt 30 bars opwarming; elk venster heeft de 30 waarnemingen die een Sharpe-standaardfout
# minimaal vraagt, zodat een run in seconden loopt.
# --------------------------------------------------------------------------- #
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT")
T0 = pd.Timestamp("2021-01-01", tz="UTC")
N_BASE, N_WORLD = 164, 205
BACKCAST, TRAIN, VALIDATE, HOLDOUT = (32, 61), (62, 95), (96, 127), (128, 163)
FORWARD_START = 170
PANEL_BASE = "data/binance_vision"
FORWARD_BASE = "data/binance_vision/forward"
PANEL_SETS = {"panels": ("close", "high", "low", "quote_volume", "funding"),
              "spot_panels": ("close", "high", "low", "quote_volume"),
              "mark_panels": ("high", "low")}
MANIFESTS = {"um": "11", "spot": "22", "mark": "33"}


def day(bar: int) -> str:
    """De datum (ISO) van bar `bar`."""
    return str((T0 + pd.Timedelta(days=bar)).date())


def synthetic_world(*, seed: int = 5, funding_mean: float = 6e-4, flat_from: int | None = None,
                    spike: tuple[int, float] | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    """Perp-, spot- en markpanelen met een rustige, positieve funding.

    `flat_from`: vanaf die bar is de funding sterk negatief, dus sluit het boek en komt er
    geen positie meer in. `spike=(bar, factor)`: op die bar staan de perp-high en de
    mark-high van elke munt `factor` keer boven de close (een squeeze)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(T0, periods=N_WORLD, freq="D", name="asof_ts")
    cols = list(SYMBOLS)
    common = rng.standard_t(4, size=N_WORLD) * 0.015
    idio = rng.standard_t(4, size=(N_WORLD, len(cols))) * 0.02
    rets = (0.7 * common[:, None] + idio).clip(-0.3, 0.3)
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=cols)
    wick = pd.DataFrame(np.abs(rng.normal(0.0, 0.01, size=close.shape)), index=idx, columns=cols)
    # BTC en ETH het meest liquide: zij staan altijd in het universum.
    qv = pd.DataFrame(np.exp(rng.normal(17.0, 0.4, size=close.shape)), index=idx,
                      columns=cols).mul(np.linspace(3.0, 1.0, len(cols)), axis=1)
    funding = pd.DataFrame(rng.normal(funding_mean, 2e-4, size=close.shape), index=idx,
                           columns=cols)
    if flat_from is not None:
        funding.iloc[flat_from:] = -1e-3
    basis = pd.DataFrame(rng.normal(0.0, 5e-4, size=close.shape), index=idx, columns=cols)
    spot = close * (1.0 - basis)
    high, low = close * (1.0 + wick), close * (1.0 - wick)
    mark_high, mark_low = close * (1.0 + 0.8 * wick), close * (1.0 - 0.8 * wick)
    if spike is not None:
        bar, factor = spike
        high.iloc[bar] = close.iloc[bar] * factor
        mark_high.iloc[bar] = close.iloc[bar] * factor
    return {
        "panels": {"close": close, "high": high, "low": low, "quote_volume": qv,
                   "funding": funding},
        "spot_panels": {"close": spot, "high": spot * 1.001, "low": spot * 0.999,
                        "quote_volume": qv * 0.5},
        "mark_panels": {"high": mark_high, "low": mark_low},
    }


def write_panels(root: Path, world: Mapping[str, Mapping[str, pd.DataFrame]], *,
                 base: str = PANEL_BASE, n_bars: int | None = None) -> None:
    for sub, frames in world.items():
        directory = root / base / sub
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            (frame if n_bars is None else frame.iloc[:n_bars]).to_parquet(
                directory / f"{name}.parquet")


def write_manifests(root: Path) -> None:
    directory = root / "artefacts/data"
    directory.mkdir(parents=True, exist_ok=True)
    for key, digit in MANIFESTS.items():
        (directory / f"binance_{key}_manifest.json").write_text(
            json.dumps({"objects_sha256": digit * 32}), encoding="utf-8")


def model_config(version: str) -> dict[str, Any]:
    """De echte config van `version` ("v5".."v8"), met een klein universum en korte vensters."""
    d = yaml.safe_load((REPO / f"conf/model/robust_book_{version}.yaml").read_text("utf-8"))
    d["universe"] = {"top_n": 6, "min_history_bars": 20, "adv_window": 10, "min_adv_usd": 1e5,
                     "vol_window": 20, "min_ann_vol": 0.15}
    d["windows"] = {"train_start": day(TRAIN[0]), "train_end": day(TRAIN[1]),
                    "validate_start": day(VALIDATE[0]), "w_dev_end": day(VALIDATE[1]),
                    "holdout_start": day(HOLDOUT[0]), "holdout_end": day(HOLDOUT[1])}
    d["backcast"] = {"start": day(BACKCAST[0]), "end": day(BACKCAST[1])}
    d["trend"]["lookbacks"] = [5, 10, 20]
    d["sizing"]["vol_span"] = 30
    # Synthetische funding is ~22 %/jaar; lagere drempels laten het boek iets houden.
    d["basis"].update(funding_span=10, enter_apr=0.05, exit_apr=0.0, min_spot_adv_usd=1e5)
    if "forward" in d:
        d["forward"] = {"start": day(FORWARD_START), "min_months": 1}
    return d


def preregistration_spec(version: str, *, bind: Sequence[str] = (),
                         keep: Sequence[str] = ()) -> dict[str, Any]:
    """De echte preregistratie van `version`, met elke drempel onbindbaar -- behalve de
    criteria in `bind`, die altijd binden, en die in `keep`, die hun echte drempel houden.
    Zo bepaalt de test welk oordeel valt."""
    d = yaml.safe_load((REPO / f"conf/research/preregistration_robust_book_{version}.yaml")
                       .read_text("utf-8"))
    names = {c["name"] for c in d["stop_criteria"]}
    assert set(bind) | set(keep) <= names, sorted((set(bind) | set(keep)) - names)
    for c in d["stop_criteria"]:
        if c["name"] in keep:
            continue
        below = c["operator"] in ("<", "<=")
        never = -1e9 if below else 1e9
        c["threshold"] = -never if c["name"] in bind else never
    return d


def edit_yaml(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    d = yaml.safe_load(path.read_text("utf-8"))
    edit(d)
    path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_root(root: Path, version: str, *, world: Mapping[str, Mapping[str, pd.DataFrame]]
               | None = None, bind: Sequence[str] = (), keep: Sequence[str] = (),
               backcast_lock: bool | None = None,
               edit_config: Callable[[dict[str, Any]], None] | None = None,
               edit_spec: Callable[[dict[str, Any]], None] | None = None) -> Path:
    """Een complete tijdelijke repo-root: config, preregistratie, impactparameters, panelen,
    manifesten, een lege ledger en het holdout-slot (zoals v2 het bevroor). Het
    backcast-slot bestaat alleen vanaf v6: v5 bevriest het zelf."""
    cfg = model_config(version)
    spec = preregistration_spec(version, bind=bind, keep=keep)
    if edit_config is not None:
        edit_config(cfg)
    if edit_spec is not None:
        edit_spec(spec)
    for rel, payload in ((f"conf/model/robust_book_{version}.yaml", cfg),
                         (f"conf/research/preregistration_robust_book_{version}.yaml", spec)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                                encoding="utf-8")
    (root / "conf/execution").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "conf/execution/impact.yaml", root / "conf/execution/impact.yaml")
    write_panels(root, synthetic_world() if world is None else world, n_bars=N_BASE)
    write_manifests(root)
    ledger = root / v2.LEDGER_PATH
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"version": 1, "seed_total": SEED_TOTAL, "entries": []}),
                      encoding="utf-8")
    freeze_holdout(split_utc=f"{day(HOLDOUT[0])}T00:00:00+00:00", out=root / v2.LOCK_PATH,
                   git_sha=GIT_SHA)
    if (version != "v5") if backcast_lock is None else backcast_lock:
        freeze_holdout(split_utc=f"{day(TRAIN[0])}T00:00:00+00:00", out=root / v5.BACKCAST_LOCK,
                       git_sha=GIT_SHA)
    return root


def canonical(payload: Any) -> str:
    """Een vergelijkbare vorm van een resultaat: NaN is gelijk aan NaN, tuples zijn lijsten."""
    return json.dumps(payload, sort_keys=True, default=float)


def ledger_entries(root: Path) -> list[dict[str, Any]]:
    return HypothesisLedger(root / v2.LEDGER_PATH).entries()


def lock_state(root: Path, lock: Path) -> dict[str, Any]:
    return json.loads((root / lock).read_text(encoding="utf-8"))


def write_w_dev_artefacts(directory: Path, *, prereg_id: str, name: str, dev_sharpe: float,
                          n_binding: int = 0, **summary_extra: Any) -> None:
    """De twee bestanden die een W_DEV-run achterlaat en de OOS-lezing leest -- alleen de
    velden die de lezing gebruikt. De echte keten (run -> lezing) staat in de ketentoetsen."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "programme_w_dev.json").write_text(json.dumps({
        "preregistration_id": prereg_id, "selected": name,
        "robustness_scores": {name: 77.5}, **summary_extra}), encoding="utf-8")
    (directory / f"{name}.json").write_text(json.dumps({
        "trial": name, "w_dev": {"sharpe": dev_sharpe, "cagr": 0.15},
        "n_binding_w_dev": n_binding}), encoding="utf-8")


# --------------------------------------------------------------------------- #
# v5-specifiek
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _pin_git_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v5, "current_git_sha", lambda: GIT_SHA)


def v5_cfg(root: Path):
    return robust_book_v5_config(root / v5.CONFIG)


def v5_dev_sharpe(root: Path, name: str) -> float:
    """De W_DEV-Sharpe die een echte run voor `name` zou opslaan."""
    cfg = v5_cfg(root)
    w = cfg.windows
    market = v5.load_market(root, cfg, end=v2._ts(w.holdout_start))
    res = v5.simulate(name, market, cfg, v5.costs_for(root, cfg))
    return v2._sr(res.window(v2._ts(w.train_start), v2._ts(w.w_dev_end)))


def v5_frozen_root(root: Path, *, name: str = "H1_BASIS", n_binding: int = 0,
                   dev_sharpe: float | None = None, **kw: Any) -> SimpleNamespace:
    """Een root waarin v5 is bevroren en de W_DEV-artefacten van `name` staan."""
    write_root(root, "v5", **kw)
    v5.freeze(root, git_sha=GIT_SHA)
    prereg_id = v5._prereg(root, v5_cfg(root))[0].preregistration_id
    sharpe = v5_dev_sharpe(root, name) if dev_sharpe is None else dev_sharpe
    write_w_dev_artefacts(root / v5.ARTEFACT_DIR, prereg_id=prereg_id, name=name,
                          dev_sharpe=sharpe, n_binding=n_binding)
    return SimpleNamespace(root=root, prereg_id=prereg_id, name=name, dev_sharpe=sharpe,
                           hid=f"robust_book_v5/{prereg_id}/{name}")


def quiet(_: str) -> None:
    return None


@pytest.fixture(scope="module")
def dev_sharpes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    """De W_DEV-Sharpe van elke kandidaat in de standaardwereld: wat een echte run opslaat."""
    root = write_root(tmp_path_factory.mktemp("v5_dev"), "v5")
    return {name: v5_dev_sharpe(root, name) for name in v5.CANDIDATES}


@pytest.fixture
def frozen(tmp_path: Path, dev_sharpes: dict[str, float]) -> Callable[..., SimpleNamespace]:
    """Fabriek van bevroren roots onder `tmp_path`. In de standaardwereld ligt de W_DEV-Sharpe al
    vast; een eigen wereld rekent hem zelf uit."""
    def make(sub: str | None = None, *, name: str = "H1_BASIS", **kw: Any) -> SimpleNamespace:
        if "world" not in kw and "dev_sharpe" not in kw:
            kw["dev_sharpe"] = dev_sharpes[name]
        return v5_frozen_root(tmp_path if sub is None else tmp_path / sub, name=name, **kw)
    return make


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Eén echte keten `freeze` -> `run`, met de echte batterij: de rest van het bestand
    leest hieruit (nooit schrijven: kopieer de root eerst)."""
    root = write_root(tmp_path_factory.mktemp("v5_chain"), "v5")
    seen: list[pd.Timestamp] = []
    real_simulate, real_pbo = v5.simulate, v5.pbo_record

    def spy(name: str, m: Any, *args: Any, **kw: Any) -> BookResult:
        seen.append(m.index.max())
        return real_simulate(name, m, *args, **kw)

    logs: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(v5, "current_git_sha", lambda: GIT_SHA)
        mp.setattr(v5, "N_SEEDS", 2)
        mp.setattr(v5, "simulate", spy)
        mp.setattr(v5, "pbo_record", lambda variants: real_pbo(variants, n_subsets=4))
        v5.freeze(root, git_sha=GIT_SHA)
        locks = {k: (root / k).read_bytes() for k in (v2.LOCK_PATH, v5.BACKCAST_LOCK)}
        summary = v5.run_programme(root, log=logs.append)
    directory = root / v5.ARTEFACT_DIR
    records = {n: json.loads((directory / f"{n}.json").read_text("utf-8")) for n in v5.CANDIDATES}
    return SimpleNamespace(root=root, summary=summary, records=records, logs=logs, seen=seen,
                           locks=locks, cfg=v5_cfg(root))


# --------------------------------------------------------------------------- #
# Hulpfuncties
# --------------------------------------------------------------------------- #
def test_data_hashes_pin_both_manifests_and_every_panel_file(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5")
    hashes = dict(v5._data_hashes(root))
    assert hashes["binance_um/objects"] == "11" * 8
    assert hashes["binance_spot/objects"] == "22" * 8
    assert not any(k.startswith("binance_mark") for k in hashes), "mark hoort bij v6"
    for set_name, key in (("panels", "binance_um"), ("spot_panels", "binance_spot")):
        for stem in PANEL_SETS[set_name]:
            raw = (root / PANEL_BASE / set_name / f"{stem}.parquet").read_bytes()
            assert hashes[f"{key}/panel/{stem}"] == hashlib.sha256(raw).hexdigest()[:16]
    assert len(hashes) == 2 + len(PANEL_SETS["panels"]) + len(PANEL_SETS["spot_panels"])
    assert list(v5._data_hashes(root)) == sorted(v5._data_hashes(root))
    # Een spotpaneel dat verandert, verandert precies zijn eigen hash.
    path = root / v5.SPOT_DIR / "close.parquet"
    (pd.read_parquet(path) * 1.01).to_parquet(path)
    after = dict(v5._data_hashes(root))
    assert {k for k in after if after[k] != hashes[k]} == {"binance_spot/panel/close"}


def test_the_preregistration_id_follows_the_data_and_the_parameters(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5")
    cfg = v5_cfg(root)
    prereg, hashes = v5._prereg(root, cfg)
    assert hashes == v5._data_hashes(root) and prereg.data_hashes == hashes
    assert prereg.parameters == v5._params(cfg) == json.loads(cfg.model_dump_json())
    assert prereg.planned_trials == cfg.planned_trials == len(v5.CANDIDATES)
    before = prereg.preregistration_id
    assert v5._prereg(root, cfg)[0].preregistration_id == before
    path = root / v5.SPOT_DIR / "quote_volume.parquet"
    (pd.read_parquet(path) * 2.0).to_parquet(path)
    assert v5._prereg(root, cfg)[0].preregistration_id != before


def test_costs_for_builds_both_legs_and_honours_the_spread_overrides(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5")
    cfg = v5_cfg(root)
    costs = v5.costs_for(root, cfg)
    assert costs.perp.taker_fee == pytest.approx(cfg.costs.taker_fee_bps * 1e-4)
    assert costs.perp.half_spread == pytest.approx(cfg.v3.half_spread_bps * 1e-4)
    assert costs.spot.taker_fee == pytest.approx(cfg.basis.spot_taker_fee_bps * 1e-4)
    assert costs.spot.half_spread == pytest.approx(cfg.basis.spot_half_spread_bps * 1e-4)
    # Beide benen delen het impactmodel, de boekgrootte en de literatuur-Y.
    eta = load_impact_params(REPO / "conf/execution/impact.yaml").eta
    assert costs.perp.impact.eta == pytest.approx(eta)
    assert costs.spot.impact is costs.perp.impact
    assert costs.spot.aum_usd == costs.perp.aum_usd == cfg.execution.aum_usd
    assert costs.spot.impact_eta == costs.perp.impact_eta == cfg.costs.impact_y
    stress = v5.costs_for(root, cfg, perp_half_spread_bps=cfg.v3.half_spread_stress_bps,
                          spot_half_spread_bps=cfg.basis.spot_half_spread_stress_bps)
    assert stress.perp.half_spread == pytest.approx(cfg.v3.half_spread_stress_bps * 1e-4)
    assert stress.spot.half_spread == pytest.approx(cfg.basis.spot_half_spread_stress_bps * 1e-4)
    assert (stress.perp.taker_fee, stress.spot.taker_fee) == (costs.perp.taker_fee,
                                                              costs.spot.taker_fee)


def test_load_market_cuts_at_the_end_and_top_n_overrides_the_universe(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5")
    cfg = v5_cfg(root)
    full = v5.load_market(root, cfg)
    assert len(full.index) == N_BASE and full.index[0] == T0
    assert set(full.symbols) == set(SYMBOLS)
    cut = v2._ts(day(HOLDOUT[0]))
    part = v5.load_market(root, cfg, end=cut)
    assert part.index.max() == cut - pd.Timedelta(days=1) and len(part.index) == HOLDOUT[0]
    assert part.perp.book.close.equals(full.perp.book.close.loc[part.index])
    assert int(full.perp.universe.sum(axis=1).max()) == cfg.universe.top_n == 6
    narrow = v5.load_market(root, cfg, top_n=4)
    assert int(narrow.perp.universe.sum(axis=1).max()) == 4
    assert cfg.universe.top_n == 6, "de config is bevroren en blijft zoals hij was"


def test_liquidations_are_read_from_the_basis_account_of_every_combination(tmp_path: Path) -> None:
    """Een squeeze op één dag boven de futureswallet: precies één liquidatie, in elke vorm van
    de audit (los basisboek, of de subrekeningen van de combinatie met trend)."""
    root = write_root(tmp_path, "v5", world=synthetic_world(spike=(TRAIN[0] + 20, 3.0)))
    cfg = v5_cfg(root)
    market, costs = v5.load_market(root, cfg), v5.costs_for(root, cfg)
    alone = v5.simulate("H1_BASIS", market, cfg, costs)
    combo = v5.simulate("H2_BASIS_TREND", market, cfg, costs)
    assert "accounts" not in alone.audit and "accounts" in combo.audit
    assert alone.audit["n_liquidations"] == 1 and v5._liquidations(alone) == 1
    assert v5._liquidations(combo) == 1
    empty = pd.DataFrame()
    assert v5._liquidations(BookResult(frame=empty, held=empty, trades=empty, audit={})) == 0
    nested = {"accounts": {"basis": {"audit": {"n_liquidations": 2}},
                           "trend": {"audit": {}}, "other": {"audit": {"n_liquidations": 3}}}}
    assert v5._liquidations(BookResult(frame=empty, held=empty, trades=empty, audit=nested)) == 5


def test_an_undefined_sharpe_is_null_in_the_ledger_and_numbers_stay_numbers() -> None:
    assert v5._finite(float("nan")) is None and v5._finite(float("inf")) is None
    assert v5._finite(-float("inf")) is None
    assert v5._finite(1.5) == 1.5 and v5._finite(np.float64(-2.0)) == -2.0


# --------------------------------------------------------------------------- #
# Bevriezen
# --------------------------------------------------------------------------- #
def test_freeze_books_the_programme_freezes_the_preregistration_and_the_backcast_slot(
        tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5")
    holdout_lock_before = (root / v2.LOCK_PATH).read_bytes()
    path = v5.freeze(root, git_sha=GIT_SHA)
    cfg = v5_cfg(root)
    prereg, hashes = v5._prereg(root, cfg)
    assert path == root / v2.PREREG_DIR / f"preregistration_{prereg.preregistration_id}.json"
    frozen = json.loads(path.read_text("utf-8"))
    assert frozen["git_sha"] == GIT_SHA
    assert frozen["ledger_total_at_freeze"] == SEED_TOTAL + cfg.planned_trials
    (entry,) = ledger_entries(root)
    assert (entry["unit"], entry["wave"], entry["market"]) == ("robust_book_v5_programme", 5, "crypto")
    assert (entry["n_trials"], entry["result"]) == (cfg.planned_trials, "interim")
    assert entry["preregistration_id"] == prereg.preregistration_id
    assert entry["git_sha"] == GIT_SHA
    assert entry["data_hash"] == hash_config(dict(hashes))
    assert entry["config_hash"] == hash_config(v5._params(cfg))
    back = lock_state(root, v5.BACKCAST_LOCK)
    assert back["split_utc"] == f"{cfg.windows.train_start}T00:00:00+00:00"
    assert back["git_sha"] == GIT_SHA and back["reads"] == []
    assert (root / v2.LOCK_PATH).read_bytes() == holdout_lock_before, "het holdout-slot is van v2"


def test_freeze_asks_git_for_the_sha_when_none_is_given(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    root = write_root(tmp_path, "v5")
    monkeypatch.setattr(v5, "current_git_sha", lambda: "feed456")
    v5.freeze(root)
    assert ledger_entries(root)[0]["git_sha"] == "feed456"
    assert lock_state(root, v5.BACKCAST_LOCK)["git_sha"] == "feed456"


def test_freeze_refuses_a_second_booking_and_leaves_ledger_and_lock_alone(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5")
    v5.freeze(root, git_sha=GIT_SHA)
    ledger_before = (root / v2.LEDGER_PATH).read_bytes()
    lock_before = (root / v5.BACKCAST_LOCK).read_bytes()
    with pytest.raises(DataContractError, match="al geboekt"):
        v5.freeze(root, git_sha=GIT_SHA)
    assert (root / v2.LEDGER_PATH).read_bytes() == ledger_before
    assert (root / v5.BACKCAST_LOCK).read_bytes() == lock_before


def test_freeze_refuses_when_the_preregistration_plans_another_number_of_trials(
        tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5", edit_spec=lambda d: d.update(planned_trials=4))
    with pytest.raises(DataContractError, match="planned_trials wijkt af"):
        v5.freeze(root, git_sha=GIT_SHA)
    assert ledger_entries(root) == []
    assert not (root / v5.BACKCAST_LOCK).exists()
    assert not list((root / v2.PREREG_DIR).glob("preregistration_*.json"))


# --------------------------------------------------------------------------- #
# De W_DEV-run
# --------------------------------------------------------------------------- #
def test_the_w_dev_run_refuses_without_a_frozen_preregistration_and_writes_nothing(
        tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5")
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v5.run_programme(root, log=quiet)
    assert not (root / v5.ARTEFACT_DIR).exists()


def test_the_w_dev_run_writes_a_summary_that_agrees_with_every_record(
        chain: SimpleNamespace) -> None:
    s, cfg = chain.summary, chain.cfg
    prereg = v5._prereg(chain.root, cfg)[0]
    assert s["programme"] == "robust_book_v5" and s["preregistration_id"] == prereg.preregistration_id
    assert s["git_sha"] == GIT_SHA and s[EVIDENCE_KEY] == NOT_ADMISSIBLE
    # M telt de ledgerstand na het boeken van dit programma, met en zonder bekende eerdere trials.
    assert s["m_programme"] == SEED_TOTAL + cfg.planned_trials
    assert s["m_including_known_prior"] == s["m_programme"] + cfg.known_prior_trials
    scores = s["robustness_scores"]
    assert set(scores) == set(v5.CANDIDATES)
    assert scores == {n: chain.records[n]["robustness_score"]["total"] for n in v5.CANDIDATES}
    assert s["selected"] == max(scores, key=lambda k: scores[k])
    assert s["passes_all_w_dev_gates"] == {n: chain.records[n]["n_binding_w_dev"] == 0
                                           for n in v5.CANDIDATES}
    for n in v5.CANDIDATES:
        table = s["table"][n]
        assert table["sharpe"] == chain.records[n]["w_dev"]["sharpe"]
        assert set(table) >= {"sharpe", "cagr", "ann_funding", "avg_gross_leverage", "sharpe_ci_low"}
    stored = json.loads((chain.root / v5.ARTEFACT_DIR / "programme_w_dev.json").read_text("utf-8"))
    assert canonical(stored) == canonical(s)
    assert chain.logs[:3] == [f"run {n}" for n in v5.CANDIDATES]
    assert [m for m in chain.logs if m.startswith("batterij")] == [
        f"batterij {n}" for n in v5.CANDIDATES]


def test_every_w_dev_record_carries_its_provenance_gates_and_battery(chain: SimpleNamespace) -> None:
    cfg = chain.cfg
    prereg, hashes = v5._prereg(chain.root, cfg)
    w = cfg.windows
    for name, rec in chain.records.items():
        assert rec["trial"] == name and rec["hypothesis"] == v5.HYPOTHESES[name]
        assert rec["role"] == "candidate" and rec[EVIDENCE_KEY] == NOT_ADMISSIBLE
        assert rec["preregistration_id"] == prereg.preregistration_id and rec["git_sha"] == GIT_SHA
        assert rec["data_hashes"] == dict(hashes) and rec["parameters"] == v5._params(cfg)
        assert rec["windows"] == {"train": [w.train_start, w.train_end],
                                  "validate": [w.validate_start, w.w_dev_end],
                                  "backcast": [cfg.backcast.start, "unread"],
                                  "holdout": [w.holdout_start, "unread"]}
        assert rec["w_dev"]["start"] == w.train_start and rec["w_dev"]["end"] == w.w_dev_end
        assert rec["train"]["end"] == w.train_end and rec["validate"]["start"] == w.validate_start
        assert rec["execution"]["lag_bars"] == cfg.execution.lag_bars
        assert set(rec["execution"]["costs"]) == {"perp", "spot"}
        m_prog = chain.summary["m_programme"]
        assert set(rec["dsr"]) == {f"m_{m_prog}", f"m_{chain.summary['m_including_known_prior']}"}
        assert rec["n_liquidations"] == 0 and rec["n_forced_exits"] == 0
        assert set(rec["regimes"]) == {"bull", "bear", "high_vol", "low_vol", "high_corr", "low_corr"}
        bat = rec["battery"]
        assert bat["signal_noise"]["n_seeds"] == bat["missing_data"]["n_seeds"] == 2
        assert set(bat["universe"]) == {"top_30", "top_100"}
        assert set(bat["perturbation"]) == set(v5._family(name, cfg))
        assert rec["pbo"]["n_subsets"] == 4 and rec["pbo"]["n_combinations"] == 6
        assert rec["pbo"]["n_variants"] == 1 + len(bat["perturbation"])
        # De poorten komen uit de preregistratie; wat bindt wordt geteld.
        assert set(rec["gates"]) <= {c.name for c in prereg.stop_criteria}
        assert rec["n_binding_w_dev"] == sum(g["binds"] for g in rec["gates"].values())
        metrics = rec["gate_metrics"]
        assert metrics["net_sharpe_w_dev"] == rec["w_dev"]["sharpe"]
        assert metrics["net_sharpe_train"] == rec["train"]["sharpe"]
        assert metrics["pbo"] == rec["pbo"]["pbo"]
        assert metrics["dsr_w_dev"] == rec["dsr"][f"m_{m_prog}"]["dsr"]
        if rec["w_dev"]["sharpe"] > 0:
            ratio = bat["costs"]["x2"] / rec["w_dev"]["sharpe"]
            assert metrics["sharpe_ratio_2x_cost_over_base"] == pytest.approx(ratio)
        else:     # zonder positieve Sharpe is de kostenverhouding ongedefinieerd, en dat bindt
            assert math.isnan(metrics["sharpe_ratio_2x_cost_over_base"])
            assert rec["gates"]["g5_cost_fragile"]["binds"] is True
            assert rec["n_binding_w_dev"] >= 1
        assert 0.0 <= rec["robustness_score"]["total"] <= 100.0


def test_the_w_dev_run_never_looks_at_the_holdout(chain: SimpleNamespace) -> None:
    """Elke markt die gesimuleerd wordt eindigt vóór `holdout_start` (ook de alternatieve
    universa van de batterij); de dagreeksen op schijf stoppen op `w_dev_end`; en de
    sloten zijn onaangeroerd."""
    w = chain.cfg.windows
    assert chain.seen and max(chain.seen) < v2._ts(w.holdout_start)
    for name in v5.CANDIDATES:
        daily = pd.read_csv(chain.root / v5.ARTEFACT_DIR / f"{name}_daily_w_dev.csv",
                            index_col=0, parse_dates=True)
        assert str(daily.index[0].date()) == w.train_start
        assert str(daily.index[-1].date()) == w.w_dev_end
        assert {"net", "gross", "funding"} <= set(daily.columns)
    assert not list((chain.root / v5.ARTEFACT_DIR).glob("*backcast*"))
    assert not list((chain.root / v5.ARTEFACT_DIR).glob("*holdout*"))
    for lock, before in chain.locks.items():
        assert (chain.root / lock).read_bytes() == before
        assert json.loads(before)["reads"] == []
    assert len(ledger_entries(chain.root)) == 1, "de run boekt niets bij: dat doet het oordeel"


def test_the_oos_read_continues_straight_from_the_w_dev_run(chain: SimpleNamespace,
                                                            tmp_path: Path) -> None:
    root = tmp_path / "root"
    shutil.copytree(chain.root, root)
    out = v5.read_oos(root, log=quiet)
    s = chain.summary
    assert out["candidate"] == s["selected"]
    assert out["w_dev_sharpe"] == chain.records[s["selected"]]["w_dev"]["sharpe"]
    assert out["n_binding_w_dev"] == chain.records[s["selected"]]["n_binding_w_dev"] == 0
    assert out["verdict"] == "promote_to_paper_trading" and out["n_binding_total"] == 0
    assert out["hypothesis_id"] == f"robust_book_v5/{s['preregistration_id']}/{s['selected']}"
    assert out["reads"]["backcast"]["invested"] and out["reads"]["holdout"]["invested"]
    assert out["resumed_after_crash"] is None


# --------------------------------------------------------------------------- #
# De OOS-lezing
# --------------------------------------------------------------------------- #
def _reads(root: Path, lock: Path) -> list[str]:
    return [r["hypothesis_id"] for r in lock_state(root, lock)["reads"]]


def test_the_oos_read_registers_both_reads_writes_the_artefacts_and_books_the_verdict(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    root = ctx.root
    logs: list[str] = []
    out = v5.read_oos(root, log=logs.append)
    cfg = v5_cfg(root)
    assert out["hypothesis_id"] == ctx.hid and out["candidate"] == ctx.name
    assert out["verdict"] == "promote_to_paper_trading" and out[EVIDENCE_KEY] == NOT_ADMISSIBLE
    assert out["n_binding_total"] == 0 and out["w_dev_sharpe"] == ctx.dev_sharpe
    assert _reads(root, v5.BACKCAST_LOCK) == [ctx.hid] == _reads(root, v2.LOCK_PATH)
    # De vensters: de backcast van `backcast.start` tot de bar voor het slot, de holdout vanaf het slot.
    assert out["reads"]["backcast"]["window"] == [cfg.backcast.start, cfg.backcast.end]
    assert out["reads"]["holdout"]["window"] == [cfg.windows.holdout_start, cfg.windows.holdout_end]
    for label in ("backcast", "holdout"):
        r = out["reads"][label]
        assert r["invested"] is True and r["pct_bars_invested"] > 0.0
        assert math.isfinite(r["z_vs_dev"]) and r["yearly"] is not None
        daily = pd.read_csv(root / v5.ARTEFACT_DIR / f"{ctx.name}_daily_{label}.csv",
                            index_col=0, parse_dates=True)
        assert [str(daily.index[0].date()), str(daily.index[-1].date())] == r["window"]
        assert len(daily) == r["summary"]["n_obs"]
    metrics = out["gate_metrics"]
    assert metrics["backcast_net_sharpe"] == out["reads"]["backcast"]["summary"]["sharpe"]
    assert metrics["holdout_sharpe_z_vs_dev"] == out["reads"]["holdout"]["z_vs_dev"]
    assert set(out["gates"]) == {"backcast_sharpe_collapse", "backcast_drawdown",
                                 "backcast_no_positive_sharpe", "holdout_sharpe_collapse",
                                 "holdout_drawdown"}
    stored = json.loads((root / v5.ARTEFACT_DIR / "oos_read.json").read_text("utf-8"))
    assert canonical(stored) == canonical(out)
    programme, verdict = ledger_entries(root)
    assert verdict["unit"] == "robust_book_v5_verdict" and verdict["wave"] == 5
    assert verdict["result"] == "accepted" and verdict["n_trials"] == 0
    assert verdict["amends"] == programme["config_hash"] != ""
    assert verdict["preregistration_id"] == ctx.prereg_id
    assert verdict["metrics"]["selected"] == ctx.name
    assert verdict["metrics"]["verdict"] == "promote_to_paper_trading"
    assert verdict["metrics"]["w_dev_sharpe"] == ctx.dev_sharpe
    assert verdict["metrics"]["holdout_invested"] is True
    assert verdict["metrics"]["holdout_sharpe"] == out["reads"]["holdout"]["summary"]["sharpe"]
    assert logs[-1] == "oordeel: promote_to_paper_trading"
    assert logs[0].startswith(f"backcast {ctx.name}: Sharpe ") and logs[1].startswith("holdout ")


OORDELEN = [
    pytest.param((), 0, "promote_to_paper_trading", "accepted", id="alles-groen"),
    pytest.param(("holdout_drawdown",), 0, "falsified_out_of_sample", "falsified",
                 id="een-falsificatiepoort-bindt"),
    pytest.param(("backcast_no_positive_sharpe",), 0, "archived", "archived",
                 id="alleen-een-archiefpoort-bindt"),
    pytest.param(("backcast_no_positive_sharpe", "backcast_drawdown"), 0,
                 "falsified_out_of_sample", "falsified", id="falsificatie-wint-van-archief"),
    pytest.param((), 2, "archived", "archived", id="w-dev-poorten-gaan-mee"),
    pytest.param(("holdout_sharpe_collapse",), 3, "falsified_out_of_sample", "falsified",
                 id="falsificatie-wint-van-w-dev"),
]


@pytest.mark.parametrize(("bind", "n_dev", "verdict", "result"), OORDELEN)
def test_the_verdict_follows_the_binding_gates(frozen: Callable[..., SimpleNamespace],
                                               bind: tuple[str, ...], n_dev: int, verdict: str,
                                               result: str) -> None:
    ctx = frozen(bind=bind, n_binding=n_dev)
    out = v5.read_oos(ctx.root, log=quiet)
    assert out["verdict"] == verdict
    assert out["n_binding_w_dev"] == n_dev and out["n_binding_total"] == len(bind) + n_dev
    assert {k for k, g in out["gates"].items() if g["binds"]} == set(bind)
    entry = ledger_entries(ctx.root)[-1]
    assert entry["result"] == result and entry["metrics"]["verdict"] == verdict
    assert entry["unit"] == "robust_book_v5_verdict" and entry["n_trials"] == 0


def test_a_holdout_without_any_position_is_undefined_and_binds_without_a_nan_in_the_ledger(
        frozen: Callable[..., SimpleNamespace]) -> None:
    """De funding keert om vóór de holdout: het boek sluit en komt er niet meer in. De Sharpe
    is dan ONGEDEFINIEERD en bindt (ook zonder dat een drempel dat zegt); de ledger krijgt
    `null`, geen NaN."""
    ctx = frozen(name="H3_BASIS_MAJORS", world=synthetic_world(flat_from=VALIDATE[1] - 25))
    out = v5.read_oos(ctx.root, log=quiet)
    hold = out["reads"]["holdout"]
    assert hold["invested"] is False and hold["note"] == v5.FLAT_WINDOW
    assert hold["pct_bars_invested"] == 0.0 and hold["max_gross_leverage"] == 0.0
    assert math.isnan(hold["z_vs_dev"]) and math.isnan(hold["summary"]["sharpe"])
    assert hold["summary"]["cagr"] == 0.0 and hold["summary"]["max_drawdown"] == 0.0
    assert hold["summary"]["n_obs"] == HOLDOUT[1] - HOLDOUT[0] + 1
    assert out["reads"]["backcast"]["invested"] is True
    assert {k for k, g in out["gates"].items() if g["binds"]} == {"holdout_sharpe_collapse"}
    assert out["verdict"] == "falsified_out_of_sample"
    ledger_text = (ctx.root / v2.LEDGER_PATH).read_text("utf-8")
    assert "NaN" not in ledger_text
    metrics = ledger_entries(ctx.root)[-1]["metrics"]
    assert metrics["holdout_sharpe"] is None and metrics["holdout_invested"] is False
    assert metrics["backcast_sharpe"] is not None


def test_each_read_is_in_its_lock_before_the_book_is_simulated(
        frozen: Callable[..., SimpleNamespace], monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = frozen()
    real = v5.simulate
    at_call: list[tuple[list[str], list[str]]] = []

    def spy(*args: Any, **kw: Any) -> BookResult:
        at_call.append((_reads(ctx.root, v5.BACKCAST_LOCK), _reads(ctx.root, v2.LOCK_PATH)))
        return real(*args, **kw)

    monkeypatch.setattr(v5, "simulate", spy)
    v5.read_oos(ctx.root, log=quiet)
    assert at_call == [([ctx.hid], [ctx.hid])]


def test_a_second_oos_read_is_refused_and_changes_nothing(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    v5.read_oos(ctx.root, log=quiet)
    snapshot = {p: (ctx.root / p).read_bytes()
                for p in (v2.LEDGER_PATH, v2.LOCK_PATH, v5.BACKCAST_LOCK,
                          v5.ARTEFACT_DIR / "oos_read.json")}
    with pytest.raises(HoldoutAlreadyUsed, match="al gelezen"):
        v5.read_oos(ctx.root, log=quiet)
    assert {p: (ctx.root / p).read_bytes() for p in snapshot} == snapshot


def test_the_oos_read_refuses_a_w_dev_run_that_belongs_to_another_preregistration(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    write_w_dev_artefacts(ctx.root / v5.ARTEFACT_DIR, prereg_id="0" * 32, name=ctx.name,
                          dev_sharpe=ctx.dev_sharpe)
    with pytest.raises(DataContractError, match="andere preregistratie"):
        v5.read_oos(ctx.root, log=quiet)
    assert _reads(ctx.root, v5.BACKCAST_LOCK) == [] == _reads(ctx.root, v2.LOCK_PATH)
    assert len(ledger_entries(ctx.root)) == 1


def test_the_oos_read_refuses_a_preregistration_that_was_never_frozen(tmp_path: Path) -> None:
    root = write_root(tmp_path, "v5", backcast_lock=True)
    with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
        v5.read_oos(root, log=quiet)
    assert _reads(root, v5.BACKCAST_LOCK) == [] == _reads(root, v2.LOCK_PATH)


def test_a_w_dev_sharpe_that_the_full_run_cannot_reproduce_is_refused_after_registration(
        frozen: Callable[..., SimpleNamespace]) -> None:
    """Lekt er iets (de volledige run wijkt af van de afgekapte), dan weigert de lezing. De
    lezingen staan dan al in hun slot: dat is de goede kant van de race."""
    ctx = frozen()
    write_w_dev_artefacts(ctx.root / v5.ARTEFACT_DIR, prereg_id=ctx.prereg_id, name=ctx.name,
                          dev_sharpe=ctx.dev_sharpe + 0.5)
    with pytest.raises(DataContractError, match="reproduceert de W_DEV-Sharpe niet"):
        v5.read_oos(ctx.root, log=quiet)
    assert _reads(ctx.root, v5.BACKCAST_LOCK) == [ctx.hid] == _reads(ctx.root, v2.LOCK_PATH)
    assert not (ctx.root / v5.ARTEFACT_DIR / "oos_read.json").exists()
    assert len(ledger_entries(ctx.root)) == 1


# --------------------------------------------------------------------------- #
# Hervatting na een crash
# --------------------------------------------------------------------------- #
def _crash_after_registration(ctx: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> BookResult:
        raise RuntimeError("de berekening crasht")

    with monkeypatch.context() as mp:
        mp.setattr(v5, "simulate", boom)
        with pytest.raises(RuntimeError, match="crasht"):
            v5.read_oos(ctx.root, log=quiet)


def test_a_crash_after_registration_is_finished_once_visibly_and_gives_the_same_reading(
        frozen: Callable[..., SimpleNamespace], monkeypatch: pytest.MonkeyPatch) -> None:
    plain = frozen("plain")
    expected = v5.read_oos(plain.root, log=quiet)

    ctx = frozen("crashed")
    assert ctx.hid == plain.hid, "dezelfde wereld, dezelfde preregistratie"
    _crash_after_registration(ctx, monkeypatch)
    assert _reads(ctx.root, v5.BACKCAST_LOCK) == [ctx.hid] == _reads(ctx.root, v2.LOCK_PATH)
    assert not (ctx.root / v5.ARTEFACT_DIR / "oos_read.json").exists()
    with pytest.raises(HoldoutAlreadyUsed):
        v5.read_oos(ctx.root, log=quiet)            # opnieuw lezen is een tweede lezing

    out = v5.read_oos(ctx.root, log=quiet, resume_reason=v5.RESUME_REASON)
    assert out["resumed_after_crash"] == v5.RESUME_REASON and expected["resumed_after_crash"] is None
    assert canonical({**out, "resumed_after_crash": None}) == canonical(expected)
    for lock in (v5.BACKCAST_LOCK, v2.LOCK_PATH):
        state = lock_state(ctx.root, lock)
        assert [r["hypothesis_id"] for r in state["reads"]] == [ctx.hid], "geen tweede lezing"
        (resume,) = state["resumes"]
        assert resume["hypothesis_id"] == ctx.hid and resume["reason"] == v5.RESUME_REASON
    assert ledger_entries(ctx.root)[-1]["unit"] == "robust_book_v5_verdict"
    with pytest.raises(HoldoutAlreadyUsed, match="al eenmaal hervat"):
        v5.read_oos(ctx.root, log=quiet, resume_reason=v5.RESUME_REASON)


def test_only_a_read_that_was_registered_can_be_resumed(
        frozen: Callable[..., SimpleNamespace]) -> None:
    ctx = frozen()
    with pytest.raises(DataContractError, match="al geregistreerde lezing"):
        v5.read_oos(ctx.root, log=quiet, resume_reason=v5.RESUME_REASON)
    assert _reads(ctx.root, v5.BACKCAST_LOCK) == []
    assert "resumes" not in lock_state(ctx.root, v5.BACKCAST_LOCK)
    assert len(ledger_entries(ctx.root)) == 1


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

    monkeypatch.setattr(v5, "freeze", fake("freeze", Path("artefacts/prereg.json")))
    monkeypatch.setattr(v5, "run_programme", fake("run", {"selected": "H1_BASIS", "x": 1.5}))
    monkeypatch.setattr(v5, "read_oos", fake("oos", {"verdict": "archived"}))
    return calls


@pytest.mark.parametrize(("command", "called", "kwargs", "printed"), [
    ("freeze", "freeze", {"args": ()}, "artefacts/prereg.json\n"),
    ("run", "run", {"args": ()}, json.dumps({"selected": "H1_BASIS", "x": 1.5}, indent=2) + "\n"),
    ("oos", "oos", {"args": ()}, json.dumps({"verdict": "archived"}, indent=2) + "\n"),
    ("oos-resume", "oos", {"args": (), "resume_reason": v5.RESUME_REASON},
     json.dumps({"verdict": "archived"}, indent=2) + "\n"),
])
def test_main_runs_exactly_the_requested_step_and_prints_its_result(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str,
        called: str, kwargs: dict[str, Any], printed: str) -> None:
    calls = _fake_commands(monkeypatch)
    v5.main([command])
    assert calls == [(called, kwargs)]
    assert capsys.readouterr().out == printed


@pytest.mark.parametrize("argv", [[], ["nonsense"], ["freeze", "run"], ["oos", "--resume"]])
def test_main_refuses_bad_usage_before_running_anything(
        monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    calls = _fake_commands(monkeypatch)
    usage = r"Gebruik: python -I -m tradebot\.systematic\.programme_v5 \{freeze\|run\|oos\|oos-resume\}"
    with pytest.raises(DataContractError, match=usage):
        v5.main(argv)
    assert calls == []


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_runs_as_a_script_and_hands_its_arguments_to_main(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Zonder argumenten weigert hij: bewijst dat het scriptpad `main(sys.argv[1:])` aanroept,
    zonder iets te bevriezen of te lezen."""
    monkeypatch.setattr(sys, "argv", ["programme_v5"])
    with pytest.raises(DataContractError, match="Gebruik: python -I -m"):
        runpy.run_module("tradebot.systematic.programme_v5", run_name="__main__")
